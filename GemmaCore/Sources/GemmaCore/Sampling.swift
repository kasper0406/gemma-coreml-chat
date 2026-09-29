/// Temperature + top-p (nucleus) sampling for next-token prediction.
///
/// The vocabulary is 262144 entries, so a sampler that sorts the whole
/// distribution costs ~25 ms per token. This one does a few linear passes
/// over the vocabulary and orders only as much of it as the nucleus walk
/// consumes:
///
///   * Softcap: the logits are `head`'s raw output. The final softcap,
///     `cap * tanh(x / cap)` (``GemmaConfig/finalLogitSoftcap``), runs here in
///     fp32 — vDSP scale, vvtanhf, vDSP scale folded with the temperature —
///     not in the fp16 graph. It maps ±inf to ±cap; greedy skips it (it is
///     monotone).
///   * Probabilities: softcap, temperature, max-shift, exp and sum run through
///     Accelerate on one Float buffer that is reused across calls. fp32 logits
///     are scaled straight into it; fp16 logits are converted once into it.
///     Normalizing is a single multiply per token, done only for the tokens
///     the nucleus walk looks at (the same rounding a vector multiply gives).
///     The exp over the whole vocabulary is the bulk of the remaining cost.
///   * Nucleus: the exact top-p set is the shortest prefix of the descending
///     order (probability desc, then token id asc) whose running Float sum
///     reaches `topP`. That order is produced band by band. Band k holds the
///     tokens whose probability lies in [pmax·e^-edge[k], pmax·e^-edge[k-1]):
///     one SIMD scan over the vocabulary gathers it. Only as much of the band
///     is put in order as the walk consumes: partial sorts select its largest
///     64, then 256, 1024, ... keys until the running sum reaches `topP`, so
///     even a band holding the whole vocabulary (a nearly flat distribution)
///     costs O(log nucleus) scans of it plus ordering the nucleus, never a
///     sort of the whole band. The
///     bands are disjoint value ranges in descending order, so the walked
///     sequence is exactly the prefix a full sort yields. The result is bit
///     for bit what sorting the whole vocabulary gives (see `SamplingTests`).
///     For this model the first band (within 6 nats of the top token, rarely
///     more than a hundred tokens) almost always holds the whole nucleus.
///   * Degenerate input: if the exp sum is not finite (1/temperature
///     overflows or a logit is NaN), the
///     distribution is taken as a point mass on the first largest non-NaN
///     logit, and that token is returned.
///   * Greedy decoding (temperature <= 0) is a vDSP argmax over the logits
///     with no copy at all for fp32 input, plus a vDSP sum that detects NaN
///     and infinities (vDSP_maxvi mishandles NaN); those resolve like the
///     degenerate case above.

import Accelerate
import CoreML
import Foundation
import Synchronization

public enum Sampling {
    /// Band edges in nats below the most probable token. The final band takes
    /// everything below the last edge.
    private static let bandEdges: [Float] = [6, 12, 24]

    /// Buffers reused across calls. The lock serializes sampling across
    /// concurrent generations; a call holds it for ~0.15 ms on real logits.
    private static let scratch = Mutex(Scratch())

    /// Sample next token from logits with temperature and top-p filtering.
    ///
    /// - Parameters:
    ///   - logits: MLMultiArray of shape (vocabSize,) or (1, vocabSize), fp16 or fp32
    ///   - temperature: Sampling temperature (0 = greedy)
    ///   - topP: Nucleus sampling probability threshold
    /// - Returns: Sampled token ID
    public static func sampleNextToken(
        logits: MLMultiArray,
        temperature: Float = 1.0,
        topP: Float = 0.9
    ) -> Int32 {
        // Greedy draws no random number, matching the historical RNG contract.
        if temperature <= 0 {
            return sampleNextToken(logits: logits, temperature: temperature, topP: topP, uniform: 0)
        }
        return sampleNextToken(
            logits: logits,
            temperature: temperature,
            topP: topP,
            uniform: Float.random(in: 0..<1)
        )
    }

    /// Deterministic core: same as `sampleNextToken` but with the uniform draw
    /// supplied by the caller. Exposed so sampling can be tested reproducibly.
    public static func sampleNextToken(
        logits: MLMultiArray,
        temperature: Float,
        topP: Float,
        uniform: Float
    ) -> Int32 {
        scratch.withLock { s in
            // Greedy deliberately takes the raw logits' argmax: the softcap is monotonic and only ties values for |raw| > ~300 (measured max 47).
            if temperature <= 0 { return s.argmax(logits) }
            s.nucleus(logits, temperature: temperature, topP: topP)
            var accum: Float = 0
            for i in 0..<s.nucleusCount {
                accum += s.probs[i]
                if accum >= uniform { return Scratch.token(s.keys[i]) }
            }
            return Scratch.token(s.keys[0])
        }
    }

    /// The distribution `sampleNextToken` draws from: the nucleus in walk
    /// order with its renormalized probabilities.
    static func nucleus(
        logits: MLMultiArray,
        temperature: Float,
        topP: Float
    ) -> [(id: Int32, prob: Float)] {
        scratch.withLock { s in
            s.nucleus(logits, temperature: temperature, topP: topP)
            return (0..<s.nucleusCount).map { (Scratch.token(s.keys[$0]), s.probs[$0]) }
        }
    }

    /// Only ever touched under `scratch`'s lock.
    private final class Scratch: @unchecked Sendable {
        /// One Float per vocabulary entry: scaled logits, then unnormalized
        /// probabilities. After `nucleus(_:temperature:topP:)` its first
        /// `nucleusCount` entries are the nucleus's renormalized probabilities.
        private(set) var probs = UnsafeMutableBufferPointer<Float>(start: nil, count: 0)
        /// Gathered tokens as `(probability bits << 32) | ~token`. Probabilities
        /// are non-negative, so descending integer order *is* the walk order:
        /// probability desc, then token id asc. After `nucleus(_:temperature:topP:)`
        /// the first `nucleusCount` keys are the nucleus in walk order.
        private(set) var keys = UnsafeMutableBufferPointer<UInt64>(start: nil, count: 0)
        private(set) var nucleusCount = 0

        static func token(_ key: UInt64) -> Int32 {
            Int32(bitPattern: ~UInt32(truncatingIfNeeded: key))
        }

        private static func probability(_ key: UInt64) -> Float {
            Float(bitPattern: UInt32(truncatingIfNeeded: key >> 32))
        }

        /// Size the buffers for a `count`-entry vocabulary.
        private func reserve(_ count: Int) {
            guard probs.count < count else { return }
            probs.deallocate()
            keys.deallocate()
            probs = .allocate(capacity: count)
            keys = .allocate(capacity: count)
        }

        func argmax(_ logits: MLMultiArray) -> Int32 {
            if logits.dataType == .float32 {
                return logits.withUnsafeBufferPointer(ofType: Float.self) { buf in
                    Scratch.argmax(buf.baseAddress!, count: logits.count)
                }
            }
            reserve(logits.count)
            Scratch.convertFloat16(logits, into: probs.baseAddress!)
            return Scratch.argmax(probs.baseAddress!, count: logits.count)
        }

        /// First index of the largest value, ignoring NaN (+inf is largest);
        /// 0 if every value is NaN.
        private static func argmax(_ x: UnsafePointer<Float>, count: Int) -> Int32 {
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(x, 1, &maxVal, &maxIdx, vDSP_Length(count))
            // vDSP_maxvi's answer is unreliable once a NaN is present. A
            // finite sum rules out NaN and infinities; otherwise (rare) scan.
            var sum: Float = 0
            vDSP_sve(x, 1, &sum, vDSP_Length(count))
            if sum.isFinite { return Int32(maxIdx) }
            var best = -1
            for i in 0..<count where !x[i].isNaN && (best < 0 || x[i] > x[best]) { best = i }
            return Int32(max(best, 0))
        }

        /// Fill the first `nucleusCount` entries of `keys` with the nucleus and
        /// of `probs` with its renormalized probabilities.
        func nucleus(_ logits: MLMultiArray, temperature: Float, topP: Float) {
            let count = logits.count
            reserve(count)
            let p = probs.baseAddress!
            let n = vDSP_Length(count)

            // Softcap and temperature, then numerical-stability shift by the max.
            Scratch.softcap(logits, temperature: temperature, into: p)
            var maxVal: Float = 0
            vDSP_maxv(p, 1, &maxVal, n)
            var negMax = -maxVal
            vDSP_vsadd(p, 1, &negMax, p, 1, n)

            // Unnormalized probabilities. A token's probability is
            // `p[i] * invSum`, computed only for the tokens a band gathers.
            // The top token is exp(0) = 1, so its probability is `invSum`.
            var count32 = Int32(count)
            vvexpf(p, p, &count32)
            var sum: Float = 0
            vDSP_sve(p, 1, &sum, n)
            // A NaN logit or an overflowing 1/temperature ends up as a NaN
            // here (inf - inf, 0 * inf, NaN - max).
            guard sum.isFinite else { return pointMass(argmax(logits)) }
            let invSum = 1.0 / sum

            // Walk the descending order band by band until the mass reaches topP.
            var gathered = 0
            var cutoff: Int?
            var cumulative: Float = 0
            var hi = Float.infinity
            for band in 0...Sampling.bandEdges.count where cutoff == nil {
                let lo = band < Sampling.bandEdges.count ? invSum * exp(-Sampling.bandEdges[band]) : 0
                let start = gathered
                gathered = gather(p, count: count, scale: invSum, lo: lo, hi: hi, from: start)
                let band = UnsafeMutableBufferPointer(rebasing: keys[start..<gathered])
                if let taken = Scratch.walk(band, cumulative: &cumulative, topP: topP) {
                    cutoff = start + taken
                }
                hi = lo
            }
            nucleusCount = cutoff ?? gathered

            // Renormalize the nucleus, reusing the front of `probs`.
            for i in 0..<nucleusCount { p[i] = Scratch.probability(keys[i]) }
            var topSum: Float = 0
            vDSP_sve(p, 1, &topSum, vDSP_Length(nucleusCount))
            var invTopSum = 1.0 / topSum
            vDSP_vsmul(p, 1, &invTopSum, p, 1, vDSP_Length(nucleusCount))
        }

        /// The nucleus of a degenerate distribution: all mass on `token`.
        private func pointMass(_ token: Int32) {
            keys[0] = UInt64(Float(1).bitPattern) << 32 | UInt64(~UInt32(bitPattern: token))
            probs[0] = 1
            nucleusCount = 1
        }

        /// Continue the walk into `band`: order it descending only as far as
        /// the running sum needs to reach `topP`, in growing chunks (64, 256,
        /// 1024, ...) each selected by a partial sort of what is left. When
        /// the walk takes c of the band's b keys, that is O(log c) chunks, and
        /// a chunk of k keys costs one pass over the rest of the band — a
        /// comparison per key, plus an O(log k) heap replacement for each key
        /// that beats the chunk's smallest so far — and a heap sort of its k
        /// keys. The chunks order fewer than 4c + 64 keys (the last one
        /// overshoots c), so the worst case, every key a replacement (a band
        /// in ascending order), is O(b·log²c + c·log c); a whole-band sort
        /// happens only when the walk needs about a quarter of it. Returns how
        /// many keys that took, or nil if the whole band did not reach it; the
        /// walked keys end up first in `band`, in descending order.
        private static func walk(
            _ band: UnsafeMutableBufferPointer<UInt64>,
            cumulative: inout Float,
            topP: Float
        ) -> Int? {
            var done = 0
            var chunk = 64
            while done < band.count {
                let end = min(done + chunk, band.count)
                selectLargest(band, from: done, to: end)
                for i in done..<end {
                    cumulative += probability(band[i])
                    if cumulative >= topP { return i + 1 }
                }
                done = end
                chunk *= 4
            }
            return nil
        }

        /// Partial sort: move the largest `end - start` keys of `keys[start...]`
        /// into `keys[start..<end]`, in descending order.
        private static func selectLargest(_ keys: UnsafeMutableBufferPointer<UInt64>, from start: Int, to end: Int) {
            // Min-heap of the largest keys seen so far; a key that beats its
            // root replaces it. Most keys of a large band are rejected by the
            // one comparison.
            let heap = UnsafeMutableBufferPointer(rebasing: keys[start..<end])
            let k = heap.count
            for i in stride(from: k / 2 - 1, through: 0, by: -1) { siftDown(heap, from: i, heapSize: k) }
            for i in end..<keys.count where keys[i] > heap[0] {
                let evicted = heap[0]
                heap[0] = keys[i]
                keys[i] = evicted
                siftDown(heap, from: 0, heapSize: k)
            }
            // Move the minimum to the back until the heap reads descending.
            for size in stride(from: k - 1, to: 0, by: -1) {
                heap.swapAt(0, size)
                siftDown(heap, from: 0, heapSize: size)
            }
        }

        /// Restore the min-heap property below `i` in `heap[0..<heapSize]`.
        private static func siftDown(_ heap: UnsafeMutableBufferPointer<UInt64>, from i: Int, heapSize: Int) {
            let value = heap[i]
            var i = i
            while true {
                var child = 2 * i + 1
                if child >= heapSize { break }
                if child + 1 < heapSize && heap[child + 1] < heap[child] { child += 1 }
                if heap[child] >= value { break }
                heap[i] = heap[child]
                i = child
            }
            heap[i] = value
        }

        /// Write the key of every token with probability `lo <= e[i] * scale < hi`
        /// to `keys`, starting at index `start`, in ascending token order.
        /// Returns the index past the last key written. NaN falls in no band.
        private func gather(
            _ e: UnsafePointer<Float>, count: Int, scale: Float, lo: Float, hi: Float, from start: Int
        ) -> Int {
            let out = keys.baseAddress!
            var written = start
            typealias Lanes = SIMD16<Float>
            let scaleV = Lanes(repeating: scale)
            let loV = Lanes(repeating: lo)
            let hiV = Lanes(repeating: hi)
            let raw = UnsafeRawPointer(e)
            func take(_ i: Int) {
                let p = e[i] * scale
                if p >= lo && p < hi {
                    out[written] = UInt64(p.bitPattern) << 32 | UInt64(~UInt32(i))
                    written += 1
                }
            }
            var i = 0
            while i + Lanes.scalarCount <= count {
                let p = raw.loadUnaligned(fromByteOffset: i * MemoryLayout<Float>.stride, as: Lanes.self) * scaleV
                if any((p .>= loV) .& (p .< hiV)) {
                    for j in i..<(i + Lanes.scalarCount) { take(j) }
                }
                i += Lanes.scalarCount
            }
            while i < count {
                take(i)
                i += 1
            }
            return written
        }

        /// `cap * tanh(x / cap) / temperature` of every logit into
        /// `destination`: a scale by 1/cap, vvtanhf, and a scale by
        /// cap · (1/temperature).
        static func softcap(_ logits: MLMultiArray, temperature: Float, into destination: UnsafeMutablePointer<Float>) {
            let n = vDSP_Length(logits.count)
            let cap = Float(GemmaConfig.finalLogitSoftcap)
            var invCap = 1 / cap
            if logits.dataType == .float32 {
                logits.withUnsafeBufferPointer(ofType: Float.self) { buf in
                    vDSP_vsmul(buf.baseAddress!, 1, &invCap, destination, 1, n)
                }
            } else {
                convertFloat16(logits, into: destination)
                vDSP_vsmul(destination, 1, &invCap, destination, 1, n)
            }
            var count = Int32(logits.count)
            vvtanhf(destination, destination, &count)
            var scale = cap * (1 / temperature)
            vDSP_vsmul(destination, 1, &scale, destination, 1, n)
        }

        /// Materialize fp16 logits as Float32 in `destination`.
        private static func convertFloat16(_ logits: MLMultiArray, into destination: UnsafeMutablePointer<Float>) {
            precondition(logits.dataType == .float16, "Sampling: unsupported logits dtype \(logits.dataType)")
            let count = logits.count
            logits.withUnsafeBufferPointer(ofType: Float16.self) { buf in
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: buf.baseAddress!),
                    height: 1, width: vImagePixelCount(count), rowBytes: count * 2
                )
                var dst = vImage_Buffer(
                    data: destination,
                    height: 1, width: vImagePixelCount(count), rowBytes: count * 4
                )
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            }
        }
    }
}
