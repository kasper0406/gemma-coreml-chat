/// Temperature + top-p (nucleus) sampling for next-token prediction.
///
/// Mirrors `sample_next_token()` from Python's generate.py.
/// Uses Accelerate for efficient operations on the 262K vocab logits.

import Accelerate
import CoreML
import Foundation

enum Sampling {
    /// Sample next token from logits with temperature and top-p filtering.
    ///
    /// - Parameters:
    ///   - logits: MLMultiArray of shape (vocabSize,) or (1, vocabSize)
    ///   - temperature: Sampling temperature (0 = greedy)
    ///   - topP: Nucleus sampling probability threshold
    /// - Returns: Sampled token ID
    static func sampleNextToken(
        logits: MLMultiArray,
        temperature: Float = 1.0,
        topP: Float = 0.9
    ) -> Int32 {
        let vocabSize = logits.count

        // Fast path: greedy decoding
        if temperature <= 0 {
            return greedyArgmax(logits: logits, count: vocabSize)
        }

        // Copy logits to a contiguous Float array
        var floats = [Float](repeating: 0, count: vocabSize)
        let ptr = logits.dataPointer.bindMemory(to: Float32.self, capacity: vocabSize)
        floats.withUnsafeMutableBufferPointer { dest in
            dest.baseAddress!.update(from: ptr, count: vocabSize)
        }

        // Apply temperature
        var invTemp = 1.0 / temperature
        vDSP_vsmul(floats, 1, &invTemp, &floats, 1, vDSP_Length(vocabSize))

        // Numerical stability: subtract max
        var maxVal: Float = 0
        vDSP_maxv(floats, 1, &maxVal, vDSP_Length(vocabSize))
        var negMax = -maxVal
        vDSP_vsadd(floats, 1, &negMax, &floats, 1, vDSP_Length(vocabSize))

        // Exp
        var count = Int32(vocabSize)
        vvexpf(&floats, floats, &count)

        // Normalize to probabilities
        var sum: Float = 0
        vDSP_sve(floats, 1, &sum, vDSP_Length(vocabSize))
        var invSum = 1.0 / sum
        vDSP_vsmul(floats, 1, &invSum, &floats, 1, vDSP_Length(vocabSize))

        // Top-p filtering: sort by probability descending
        var indices = [vDSP_Length](0..<vDSP_Length(vocabSize))
        // Sort indices by descending probability
        indices.sort { floats[Int($0)] > floats[Int($1)] }

        // Find cutoff where cumulative probability exceeds topP
        var cumulative: Float = 0
        var cutoff = vocabSize
        for (i, idx) in indices.enumerated() {
            cumulative += floats[Int(idx)]
            if cumulative >= topP {
                cutoff = i + 1
                break
            }
        }

        // Collect top-p probabilities and re-normalize
        let topIndices = Array(indices.prefix(cutoff))
        var topProbs = topIndices.map { floats[Int($0)] }
        var topSum: Float = 0
        vDSP_sve(topProbs, 1, &topSum, vDSP_Length(topProbs.count))
        var invTopSum = 1.0 / topSum
        vDSP_vsmul(topProbs, 1, &invTopSum, &topProbs, 1, vDSP_Length(topProbs.count))

        // Weighted random sample
        let r = Float.random(in: 0..<1)
        var accum: Float = 0
        for (i, prob) in topProbs.enumerated() {
            accum += prob
            if accum >= r {
                return Int32(topIndices[i])
            }
        }

        // Fallback: return the most probable token
        return Int32(topIndices[0])
    }

    /// Greedy argmax over logits.
    private static func greedyArgmax(logits: MLMultiArray, count: Int) -> Int32 {
        let ptr = logits.dataPointer.bindMemory(to: Float32.self, capacity: count)
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(count))
        return Int32(maxIdx)
    }
}
