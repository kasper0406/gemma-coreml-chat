/// Temperature + top-p (nucleus) sampling for next-token prediction.
///
/// The vocabulary is 262144 entries, so a sampler that sorts the whole
/// distribution costs ~25 ms per token. This one never orders more of the
/// vocabulary than the nucleus walk actually reaches:
///
///   * Probabilities: temperature, max-shift, exp and sum run through
///     Accelerate on one Float buffer that is reused across calls. fp32 logits
///     are scaled straight into it; fp16 logits are converted once into it.
///     Normalizing is a single multiply per token, done only for the tokens
///     the nucleus walk looks at (the same rounding a vector multiply gives).
///     The exp over the whole vocabulary is the bulk of the remaining cost.
///   * Nucleus: the exact top-p set is the shortest prefix of the descending
///     order (probability desc, then token id asc) whose running Float sum
///     reaches `topP`. That order is produced band by band. Band k holds the
///     tokens whose probability lies in [pmax·e^-edge[k], pmax·e^-edge[k-1]):
///     one SIMD scan over the vocabulary gathers it, and only it is sorted.
///     The bands are disjoint value ranges in descending order, so their
///     concatenation is exactly the prefix a full sort yields, and the walk
///     stops in the band where the mass reaches `topP`. The result is bit for
///     bit what sorting the whole vocabulary gives (see `SamplingTests`). For
///     this model the first band (within 6 nats of the top token, rarely more
///     than a hundred tokens) almost always holds the whole nucleus; only a
///     very flat distribution or `topP` near 1 reaches further bands.
///   * Greedy decoding (temperature <= 0) is a pure vDSP argmax over the logits
///     with no copy at all for fp32 input.

import Accelerate
import CoreML
import Foundation
import Synchronization

public enum Sampling {
    /// Band edges in nats below the most probable token. The final band takes
    /// everything below the last edge.
    private static let bandEdges: [Float] = [6, 12, 24]

    /// Buffers reused across calls. Sampling is sequential per generation, so
    /// the lock is uncontended; it only keeps concurrent engines correct.
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
            if temperature <= 0 { return s.argmax(logits) }
            s.nucleus(logits, temperature: temperature, topP: topP)
            var accum: Float = 0
            for (key, prob) in zip(s.order, s.nucleusProbs) {
                accum += prob
                if accum >= uniform { return Scratch.token(key) }
            }
            return s.order.first.map(Scratch.token) ?? 0
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
            return zip(s.order, s.nucleusProbs).map { (Scratch.token($0), $1) }
        }
    }

    /// Only ever touched under `scratch`'s lock.
    private final class Scratch: @unchecked Sendable {
        /// Scaled logits, then unnormalized probabilities, one per vocabulary entry.
        private var probs = UnsafeMutableBufferPointer<Float>(start: nil, count: 0)
        /// Walk order as `(probability bits << 32) | ~token`. Probabilities
        /// are non-negative, so descending integer order *is* the walk order:
        /// probability desc, then token id asc. Holds the nucleus after
        /// `nucleus(_:temperature:topP:)`.
        private(set) var order: [UInt64] = []
        /// Renormalized probabilities of `order`, in the same order.
        private(set) var nucleusProbs: [Float] = []

        static func token(_ key: UInt64) -> Int32 {
            Int32(bitPattern: ~UInt32(truncatingIfNeeded: key))
        }

        private static func probability(_ key: UInt64) -> Float {
            Float(bitPattern: UInt32(truncatingIfNeeded: key >> 32))
        }

        /// A buffer of at least `count` Floats.
        private func buffer(_ count: Int) -> UnsafeMutablePointer<Float> {
            if probs.count < count {
                probs.deallocate()
                probs = .allocate(capacity: count)
            }
            return probs.baseAddress!
        }

        func argmax(_ logits: MLMultiArray) -> Int32 {
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            let n = vDSP_Length(logits.count)
            if logits.dataType == .float32 {
                logits.withUnsafeBufferPointer(ofType: Float.self) { buf in
                    vDSP_maxvi(buf.baseAddress!, 1, &maxVal, &maxIdx, n)
                }
            } else {
                let p = buffer(logits.count)
                Scratch.convertFloat16(logits, into: p)
                vDSP_maxvi(p, 1, &maxVal, &maxIdx, n)
            }
            return Int32(maxIdx)
        }

        /// Fill `order` with the nucleus and `nucleusProbs` with its
        /// renormalized probabilities.
        func nucleus(_ logits: MLMultiArray, temperature: Float, topP: Float) {
            let count = logits.count
            let p = buffer(count)
            let n = vDSP_Length(count)

            // Temperature, then numerical-stability shift by the max.
            var invTemp = 1.0 / temperature
            if logits.dataType == .float32 {
                logits.withUnsafeBufferPointer(ofType: Float.self) { buf in
                    vDSP_vsmul(buf.baseAddress!, 1, &invTemp, p, 1, n)
                }
            } else {
                Scratch.convertFloat16(logits, into: p)
                vDSP_vsmul(p, 1, &invTemp, p, 1, n)
            }
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
            let invSum = 1.0 / sum

            // Walk the descending order band by band until the mass reaches topP.
            order.removeAll(keepingCapacity: true)
            var cutoff: Int?
            var cumulative: Float = 0
            var hi = Float.infinity
            for band in 0...Sampling.bandEdges.count where cutoff == nil {
                let lo = band < Sampling.bandEdges.count ? invSum * exp(-Sampling.bandEdges[band]) : 0
                let start = order.count
                gather(p, count: count, scale: invSum, lo: lo, hi: hi)
                order[start...].sort(by: >)
                for i in start..<order.count {
                    cumulative += Scratch.probability(order[i])
                    if cumulative >= topP {
                        cutoff = i + 1
                        break
                    }
                }
                hi = lo
            }
            order.removeSubrange((cutoff ?? order.count)...)

            // Renormalize the nucleus.
            nucleusProbs.removeAll(keepingCapacity: true)
            nucleusProbs.append(contentsOf: order.lazy.map(Scratch.probability))
            var topSum: Float = 0
            vDSP_sve(nucleusProbs, 1, &topSum, vDSP_Length(nucleusProbs.count))
            var invTopSum = 1.0 / topSum
            vDSP_vsmul(nucleusProbs, 1, &invTopSum, &nucleusProbs, 1, vDSP_Length(nucleusProbs.count))
        }

        /// Append the key of every token with probability `lo <= e[i] * scale < hi`
        /// to `order`, in ascending token order. NaN falls in no band.
        private func gather(_ e: UnsafePointer<Float>, count: Int, scale: Float, lo: Float, hi: Float) {
            typealias Lanes = SIMD16<Float>
            let scaleV = Lanes(repeating: scale)
            let loV = Lanes(repeating: lo)
            let hiV = Lanes(repeating: hi)
            let raw = UnsafeRawPointer(e)
            func take(_ i: Int) {
                let p = e[i] * scale
                if p >= lo && p < hi {
                    order.append(UInt64(p.bitPattern) << 32 | UInt64(~UInt32(i)))
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
