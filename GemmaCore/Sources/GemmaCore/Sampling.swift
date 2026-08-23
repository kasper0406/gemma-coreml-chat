/// Temperature + top-p (nucleus) sampling for next-token prediction.
///
/// The vocabulary is 262144 entries, so a naive sampler that sorts the whole
/// distribution costs ~20 ms per token — a third of the budget at 60-100 tok/s.
/// This implementation never sorts the full vocabulary:
///
///   * All elementwise work (temperature, max-shift, exp, sum) runs through
///     Accelerate on a single Float scratch buffer. fp16 logits are converted
///     once, straight into that buffer; there is no intermediate Double array.
///   * Top-p operates on a bounded top-k preselect (`preselectK` candidates)
///     found in a single streaming pass with a running threshold. Scaling by
///     1/temperature and subtracting the max are monotone, and exp is monotone,
///     so the k largest *logits* are exactly the k most probable tokens: the
///     preselect is an exact prefix of the descending order the full sort would
///     have produced. If the nucleus is not contained in those k candidates
///     (their probability mass never reaches `topP`), the code falls back to the
///     full descending sort, so the result is exact for every input rather than
///     an approximation. With p <= 0.95 a 256-token nucleus essentially never
///     occurs for this model, so the fallback is a safety net, not a hot path.
///   * Greedy decoding (temperature <= 0) is a pure vDSP argmax over the logits
///     with no copy at all for fp32 input.

import Accelerate
import CoreML
import Foundation

public enum Sampling {
    /// Number of candidates preselected before the top-p walk.
    private static let preselectK = 256

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
        let count = logits.count

        if temperature <= 0 {
            return withFloatLogits(logits) { ptr, n in
                var maxVal: Float = 0
                var maxIdx: vDSP_Length = 0
                vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(n))
                return Int32(maxIdx)
            }
        }

        let probs = UnsafeMutablePointer<Float>.allocate(capacity: count)
        defer { probs.deallocate() }
        copyLogits(logits, into: probs, count: count)

        // Temperature, then numerical-stability shift by the max.
        var invTemp = 1.0 / temperature
        vDSP_vsmul(probs, 1, &invTemp, probs, 1, vDSP_Length(count))
        var maxVal: Float = 0
        vDSP_maxv(probs, 1, &maxVal, vDSP_Length(count))
        var negMax = -maxVal
        vDSP_vsadd(probs, 1, &negMax, probs, 1, vDSP_Length(count))

        // Preselect the top-k while the buffer still holds (monotone) logits.
        let k = min(preselectK, count)
        var candidates = topKDescending(probs, count: count, k: k)

        // Probabilities.
        var n = Int32(count)
        vvexpf(probs, probs, &n)
        var sum: Float = 0
        vDSP_sve(probs, 1, &sum, vDSP_Length(count))
        var invSum = 1.0 / sum
        vDSP_vsmul(probs, 1, &invSum, probs, 1, vDSP_Length(count))

        // Does the nucleus fit inside the preselect?
        let cutoff: Int
        if let fits = nucleusCutoff(probs, candidates: candidates, topP: topP) {
            cutoff = fits
        } else {
            // Rare: the nucleus is wider than the preselect. Take the exact path.
            candidates = fullDescendingOrder(probs, count: count)
            cutoff = nucleusCutoff(probs, candidates: candidates, topP: topP) ?? count
        }

        // Renormalize the nucleus and draw.
        var topProbs = [Float](repeating: 0, count: cutoff)
        for i in 0..<cutoff { topProbs[i] = probs[Int(candidates[i])] }
        var topSum: Float = 0
        vDSP_sve(topProbs, 1, &topSum, vDSP_Length(topProbs.count))
        var invTopSum = 1.0 / topSum
        vDSP_vsmul(topProbs, 1, &invTopSum, &topProbs, 1, vDSP_Length(topProbs.count))

        var accum: Float = 0
        for (i, prob) in topProbs.enumerated() {
            accum += prob
            if accum >= uniform { return candidates[i] }
        }
        return candidates[0]
    }

    /// Number of leading `candidates` whose probability mass reaches `topP`,
    /// or nil if the candidate list never gets there.
    private static func nucleusCutoff(
        _ probs: UnsafePointer<Float>,
        candidates: [Int32],
        topP: Float
    ) -> Int? {
        var cumulative: Float = 0
        for (i, idx) in candidates.enumerated() {
            cumulative += probs[Int(idx)]
            if cumulative >= topP { return i + 1 }
        }
        return nil
    }

    // MARK: - Logit access

    /// Run `body` over the logits as contiguous Float32. fp32 input is used in
    /// place; fp16 input is converted once into a temporary buffer.
    private static func withFloatLogits<R>(
        _ logits: MLMultiArray,
        _ body: (UnsafePointer<Float>, Int) -> R
    ) -> R {
        let count = logits.count
        switch logits.dataType {
        case .float32:
            return logits.withUnsafeBufferPointer(ofType: Float.self) { buf in
                body(buf.baseAddress!, count)
            }
        case .float16:
            let scratch = UnsafeMutablePointer<Float>.allocate(capacity: count)
            defer { scratch.deallocate() }
            copyLogits(logits, into: scratch, count: count)
            return body(scratch, count)
        default:
            preconditionFailure("Sampling: unsupported logits dtype \(logits.dataType)")
        }
    }

    /// Materialize the logits as Float32 in `destination`.
    private static func copyLogits(
        _ logits: MLMultiArray,
        into destination: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        switch logits.dataType {
        case .float32:
            logits.withUnsafeBufferPointer(ofType: Float.self) { buf in
                destination.update(from: buf.baseAddress!, count: count)
            }
        case .float16:
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
        default:
            preconditionFailure("Sampling: unsupported logits dtype \(logits.dataType)")
        }
    }

    // MARK: - Selection

    /// Indices of the `k` largest values, ordered descending.
    ///
    /// One streaming pass with a running threshold `tau`: a value only enters
    /// the candidate buffer if it beats the current k-th best. The buffer holds
    /// 2k entries and is trimmed back to k (raising `tau`) whenever it fills, so
    /// the expected number of trims is O(log(count/k)) — a handful for a 262K
    /// vocabulary. Total cost is one linear scan, no full sort.
    private static func topKDescending(
        _ values: UnsafePointer<Float>,
        count: Int,
        k: Int
    ) -> [Int32] {
        let capacity = 2 * k
        let vals = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        let idxs = UnsafeMutablePointer<Int32>.allocate(capacity: capacity)
        let order = UnsafeMutablePointer<vDSP_Length>.allocate(capacity: capacity)
        defer {
            vals.deallocate()
            idxs.deallocate()
            order.deallocate()
        }

        // Sort the first `n` entries descending in place, keeping pairs together.
        func sortPrefix(_ n: Int) {
            // vDSP_vsorti permutes an existing index vector; it must start as identity.
            for i in 0..<n { order[i] = vDSP_Length(i) }
            vDSP_vsorti(vals, order, nil, vDSP_Length(n), -1)
            let tmpV = UnsafeMutablePointer<Float>.allocate(capacity: n)
            let tmpI = UnsafeMutablePointer<Int32>.allocate(capacity: n)
            defer {
                tmpV.deallocate()
                tmpI.deallocate()
            }
            for i in 0..<n {
                let p = Int(order[i])
                tmpV[i] = vals[p]
                tmpI[i] = idxs[p]
            }
            vals.update(from: tmpV, count: n)
            idxs.update(from: tmpI, count: n)
        }

        var n = 0
        var tau = -Float.infinity
        for i in 0..<count {
            let v = values[i]
            if v > tau {
                vals[n] = v
                idxs[n] = Int32(i)
                n += 1
                if n == capacity {
                    sortPrefix(n)
                    n = k
                    tau = vals[k - 1]
                }
            }
        }
        sortPrefix(n)
        return Array(UnsafeBufferPointer(start: idxs, count: min(n, k)))
    }

    /// Exact fallback: full descending order over the whole vocabulary.
    private static func fullDescendingOrder(
        _ values: UnsafePointer<Float>,
        count: Int
    ) -> [Int32] {
        var order = Array(0..<vDSP_Length(count))
        order.withUnsafeMutableBufferPointer { buf in
            vDSP_vsorti(values, buf.baseAddress!, nil, vDSP_Length(count), -1)
        }
        return order.map { Int32($0) }
    }
}
