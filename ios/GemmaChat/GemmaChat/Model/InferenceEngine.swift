/// Chunked prefill + single-token decode inference engine.
///
/// Mirrors `generate_kvcached()` from Python's generate.py.
/// Returns an `AsyncStream<Int32>` of generated token IDs.

import CoreML
import Foundation

struct InferenceEngine: Sendable {
    let model: CoreMLModel
    let temperature: Float
    let topP: Float

    /// Run full generation: chunked prefill of prompt, then decode loop.
    ///
    /// - Parameters:
    ///   - promptIDs: Token IDs for the full prompt (from chat template)
    ///   - maxNewTokens: Maximum tokens to generate
    ///   - existingKVState: Optional pre-populated KV state (from eager prefill)
    ///   - prefillOffset: If using existingKVState, how many tokens were already prefilled
    /// - Returns: AsyncStream yielding generated token IDs (including EOS)
    func generate(
        promptIDs: [Int32],
        maxNewTokens: Int = 256,
        existingKVState: KVCacheState? = nil,
        prefillOffset: Int = 0
    ) -> AsyncStream<Int32> {
        AsyncStream { continuation in
            // Run inference on a background queue to avoid blocking
            Task.detached { [self] in
                do {
                    let ids = truncatePromptIDs(
                        promptIDs,
                        maxSeqLen: GemmaConfig.maxSeqLen,
                        reserveForGeneration: maxNewTokens
                    )
                    let nReal = ids.count
                    // --- Chunked Prefill ---
                    var kvState: KVCacheState
                    var logits: MLMultiArray

                    if let existing = existingKVState, prefillOffset > 0 {
                        // Resume prefill from where eager prefill left off
                        let (prefillLogits, prefillKV) = try self.continuePrefill(
                            ids: ids,
                            fromOffset: prefillOffset,
                            kvState: existing
                        )
                        kvState = prefillKV

                        if let prefillLogits {
                            logits = prefillLogits
                        } else {
                            // All chunks were already prefilled by eager prefill.
                            // Run a single decode step with the last token to get logits.
                            let lastToken = ids[nReal - 1]
                            let (decLogits, decKV) = try model.decode(
                                token: lastToken,
                                position: Int32(nReal - 1),
                                kvState: prefillKV
                            )
                            logits = decLogits
                            kvState = decKV
                        }
                    } else {
                        // Full prefill from scratch
                        (logits, kvState) = try self.fullPrefill(ids: ids)
                    }

                    // Extract the logits for the last real token.
                    // Prefill logits are 2D (chunkSize, vocabSize) — pick the last real position.
                    // Decode logits are already 1D (vocabSize) — use directly.
                    let vocabSize = GemmaConfig.vocabSize
                    let lastLogits: MLMultiArray
                    if logits.shape.count > 1 && logits.shape[0].intValue > 1 {
                        let nChunks = (nReal + GemmaConfig.chunkSize - 1) / GemmaConfig.chunkSize
                        let lastChunkLen = nReal - (nChunks - 1) * GemmaConfig.chunkSize
                        let lastTokenPosInChunk = lastChunkLen - 1
                        lastLogits = try extractLogitsAt(
                            position: lastTokenPosInChunk,
                            from: logits,
                            vocabSize: vocabSize
                        )
                    } else {
                        lastLogits = logits
                    }

                    // --- Decode Loop ---
                    let maxSteps = min(maxNewTokens, GemmaConfig.maxSeqLen - nReal)
                    var currentLogits = lastLogits
                    var currentKV = kvState

                    for step in 0..<maxSteps {
                        let nextID = Sampling.sampleNextToken(
                            logits: currentLogits,
                            temperature: temperature,
                            topP: topP
                        )
                        continuation.yield(nextID)

                        if GemmaConfig.stopTokenIDs.contains(nextID) { break }
                        if Task.isCancelled { break }

                        let position = Int32(nReal + step)
                        let (decLogits, decKV) = try model.decode(
                            token: nextID,
                            position: position,
                            kvState: currentKV
                        )
                        currentLogits = decLogits
                        currentKV = decKV
                    }
                } catch {
                    // Stream ends on error; the ChatViewModel handles the error
                    print("Inference error: \(error)")
                }
                continuation.finish()
            }
        }
    }

    /// Run full prefill from scratch.
    func fullPrefill(ids: [Int32]) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        let emptyKV = KVCacheState.empty(
            kvInputNames: model.prefillKVInputNames,
            inputDescriptions: model.prefillModel.modelDescription.inputDescriptionsByName
        )
        let (logits, kv) = try continuePrefill(ids: ids, fromOffset: 0, kvState: emptyKV)
        guard let logits else {
            fatalError("fullPrefill produced no logits — prompt is empty")
        }
        return (logits, kv)
    }

    /// Continue prefill from a given offset with existing KV state.
    ///
    /// If `fromOffset` already covers all chunks, returns nil logits (caller
    /// must use the last logits from wherever the prior prefill came from).
    func continuePrefill(
        ids: [Int32],
        fromOffset: Int,
        kvState: KVCacheState
    ) throws -> (logits: MLMultiArray?, kvState: KVCacheState) {
        let nReal = ids.count
        let nChunks = (nReal + GemmaConfig.chunkSize - 1) / GemmaConfig.chunkSize
        let paddedLen = nChunks * GemmaConfig.chunkSize
        let padded = ids + [Int32](repeating: GemmaConfig.padTokenID,
                                   count: paddedLen - nReal)

        let startChunk = fromOffset / GemmaConfig.chunkSize
        var currentKV = kvState
        var lastLogits: MLMultiArray? = nil

        for chunkIdx in startChunk..<nChunks {
            let start = chunkIdx * GemmaConfig.chunkSize
            let chunkTokens = Array(padded[start..<(start + GemmaConfig.chunkSize)])

            let tokens = MLMultiArray.int32Row(chunkTokens)
            let (logits, newKV) = try model.prefill(
                tokens: tokens,
                seqLen: Int32(start),
                kvState: currentKV
            )
            currentKV = newKV.withProcessedTokens(start + GemmaConfig.chunkSize)
            lastLogits = logits
        }

        return (lastLogits, currentKV)
    }

    /// Run prefill for a single chunk. Used by eager prefill.
    func prefillSingleChunk(
        chunkTokens: [Int32],
        startPosition: Int,
        kvState: KVCacheState
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        precondition(chunkTokens.count == GemmaConfig.chunkSize)
        let tokens = MLMultiArray.int32Row(chunkTokens)
        let (logits, newKV) = try model.prefill(
            tokens: tokens,
            seqLen: Int32(startPosition),
            kvState: kvState
        )
        return (logits, newKV.withProcessedTokens(startPosition + GemmaConfig.chunkSize))
    }

    // MARK: - Helpers

    /// Extract a single position's logits from a (CHUNK_SIZE, vocabSize) array.
    private func extractLogitsAt(
        position: Int,
        from logits: MLMultiArray,
        vocabSize: Int
    ) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [NSNumber(value: vocabSize)], dataType: .float32)
        let srcPtr = logits.dataPointer.bindMemory(to: Float32.self, capacity: logits.count)
        let dstPtr = result.dataPointer.bindMemory(to: Float32.self, capacity: vocabSize)
        let offset = position * vocabSize
        dstPtr.update(from: srcPtr.advanced(by: offset), count: vocabSize)
        return result
    }

    /// Keep the last tokens so the prompt fits within maxSeqLen.
    private func truncatePromptIDs(
        _ ids: [Int32],
        maxSeqLen: Int,
        reserveForGeneration: Int
    ) -> [Int32] {
        let cap = max(maxSeqLen - reserveForGeneration, 1)
        if ids.count > cap {
            return Array(ids.suffix(cap))
        }
        return ids
    }
}
