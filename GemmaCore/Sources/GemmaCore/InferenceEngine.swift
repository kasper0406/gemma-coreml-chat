/// Chunked prefill + single-token decode inference engine.
///
/// Returns an `AsyncStream<Int32>` of generated token IDs.

import CoreML
import Foundation

public struct InferenceEngine: Sendable {
    public let model: CoreMLModel
    public let temperature: Float
    public let topP: Float

    public init(model: CoreMLModel, temperature: Float = 1.0, topP: Float = 0.9) {
        self.model = model
        self.temperature = temperature
        self.topP = topP
    }

    /// Run full generation: chunked prefill of prompt, then decode loop.
    ///
    /// - Parameters:
    ///   - promptIDs: Token IDs for the full prompt (from chat template)
    ///   - maxNewTokens: Maximum tokens to generate
    ///   - existingKVState: Optional pre-populated KV state (from eager prefill)
    ///   - prefillOffset: If using existingKVState, how many tokens were already prefilled
    /// - Returns: AsyncStream yielding generated token IDs (including EOS)
    public func generate(
        promptIDs: [Int32],
        maxNewTokens: Int = 256,
        existingKVState: KVCacheState? = nil,
        prefillOffset: Int = 0
    ) -> AsyncStream<Int32> {
        AsyncStream { continuation in
            Task.detached { [self] in
                do {
                    let genStart = CFAbsoluteTimeGetCurrent()
                    let ids = truncatePromptIDs(
                        promptIDs,
                        maxSeqLen: GemmaConfig.maxSeqLen,
                        reserveForGeneration: maxNewTokens
                    )
                    let nReal = ids.count
                    let nChunks = (nReal + GemmaConfig.chunkSize - 1) / GemmaConfig.chunkSize
                    Log.info("[Perf] Prompt: \(nReal) tokens, \(nChunks) chunks, prefillOffset=\(prefillOffset)")

                    try await self.model.loadPrefill()

                    // --- Chunked Prefill ---
                    let prefillStart = CFAbsoluteTimeGetCurrent()
                    var kvState: KVCacheState
                    var logits: MLMultiArray

                    if let existing = existingKVState, prefillOffset > 0 {
                        let (prefillLogits, prefillKV) = try self.continuePrefill(
                            ids: ids,
                            fromOffset: prefillOffset,
                            kvState: existing
                        )
                        kvState = prefillKV

                        if let prefillLogits {
                            logits = prefillLogits
                        } else {
                            // All chunks were already prefilled.
                            // Run a single decode step with the last token to get logits.
                            let lastToken = ids[nReal - 1]
                            let gcSize = kvState.currentGlobalCacheSize.map { Int32($0) }
                            let (decLogits, decKV) = try model.decode(
                                token: lastToken,
                                position: Int32(nReal - 1),
                                kvState: prefillKV,
                                globalCacheSize: gcSize
                            )
                            logits = decLogits
                            kvState = decKV
                        }
                    } else {
                        (logits, kvState) = try self.fullPrefill(ids: ids)
                    }
                    let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
                    Log.info("[Perf] Prefill done: \(String(format: "%.2f", prefillTime))s")

                    self.model.releasePrefill()

                    // Extract the logits for the last real token.
                    let vocabSize = GemmaConfig.vocabSize
                    let lastLogits: MLMultiArray
                    if logits.shape.count > 1 && logits.shape[0].intValue > 1 {
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
                    var totalSampleTime = 0.0
                    var totalDecodeTime = 0.0
                    var decodeSteps = 0

                    for step in 0..<maxSteps {
                        let sampleStart = CFAbsoluteTimeGetCurrent()
                        let nextID = Sampling.sampleNextToken(
                            logits: currentLogits,
                            temperature: temperature,
                            topP: topP
                        )
                        let sampleTime = CFAbsoluteTimeGetCurrent() - sampleStart
                        totalSampleTime += sampleTime

                        continuation.yield(nextID)

                        if GemmaConfig.stopTokenIDs.contains(nextID) { break }
                        if Task.isCancelled { break }

                        let position = Int32(nReal + step)

                        currentKV = currentKV.grownToFit(
                            needed: Int(position) + 1,
                            maxLen: GemmaConfig.maxSeqLen
                        )

                        let decStart = CFAbsoluteTimeGetCurrent()
                        let gcSize = currentKV.currentGlobalCacheSize.map { Int32($0) }
                        let (decLogits, decKV) = try model.decode(
                            token: nextID,
                            position: position,
                            kvState: currentKV,
                            globalCacheSize: gcSize
                        )
                        let decTime = CFAbsoluteTimeGetCurrent() - decStart
                        totalDecodeTime += decTime
                        decodeSteps += 1

                        if step < 3 {
                            Log.info("[Perf] Step \(step): sample=\(String(format: "%.3f", sampleTime))s, decode=\(String(format: "%.3f", decTime))s")
                        } else if (step + 1) % 10 == 0 {
                            let avgSample = totalSampleTime / Double(step + 1)
                            let avgDecode = totalDecodeTime / Double(decodeSteps)
                            Log.info("[Perf] Step \(step): avg sample=\(String(format: "%.3f", avgSample))s, avg decode=\(String(format: "%.3f", avgDecode))s")
                        }

                        currentLogits = decLogits
                        currentKV = decKV
                    }

                    let totalTime = CFAbsoluteTimeGetCurrent() - genStart
                    let tokPerSec = decodeSteps > 0 ? Double(decodeSteps) / totalDecodeTime : 0
                    Log.info("[Perf] Done: \(decodeSteps) tokens in \(String(format: "%.1f", totalTime))s (prefill=\(String(format: "%.1f", prefillTime))s, decode=\(String(format: "%.1f", totalDecodeTime))s, sample=\(String(format: "%.2f", totalSampleTime))s) \(String(format: "%.2f", tokPerSec)) tok/s")
                } catch {
                    Log.info("Inference error: \(error)")
                }
                continuation.finish()
            }
        }
    }

    /// Run full prefill from scratch.
    public func fullPrefill(ids: [Int32]) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        let nChunks = (ids.count + GemmaConfig.chunkSize - 1) / GemmaConfig.chunkSize
        let paddedLen = nChunks * GemmaConfig.chunkSize

        let emptyKV = KVCacheState.empty(
            kvInputNames: model.prefillKVInputNames,
            inputDescriptions: model.prefillInputDescriptions,
            globalNames: model.globalKVInputNames,
            initialGlobalSize: model.globalKVInputNames.isEmpty ? nil : paddedLen
        )
        let (logits, kv) = try continuePrefill(ids: ids, fromOffset: 0, kvState: emptyKV)
        guard let logits else {
            fatalError("fullPrefill produced no logits — prompt is empty")
        }
        return (logits, kv)
    }

    /// Continue prefill from a given offset with existing KV state.
    public func continuePrefill(
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
        var currentKV = kvState.grownToFit(needed: paddedLen, maxLen: GemmaConfig.maxSeqLen)
        var lastLogits: MLMultiArray? = nil
        let chunksToProcess = nChunks - startChunk

        for chunkIdx in startChunk..<nChunks {
            let chunkStart = CFAbsoluteTimeGetCurrent()
            let start = chunkIdx * GemmaConfig.chunkSize
            let chunkTokens = Array(padded[start..<(start + GemmaConfig.chunkSize)])

            let tokens = MLMultiArray.int32Row(chunkTokens)
            let gcSize = currentKV.currentGlobalCacheSize.map { Int32($0) }
            let (logits, newKV) = try model.prefill(
                tokens: tokens,
                seqLen: Int32(start),
                kvState: currentKV,
                globalCacheSize: gcSize
            )
            currentKV = newKV.withProcessedTokens(start + GemmaConfig.chunkSize)
            lastLogits = logits

            let chunkTime = CFAbsoluteTimeGetCurrent() - chunkStart
            let chunkNum = chunkIdx - startChunk + 1
            Log.info("[Perf] Prefill chunk \(chunkNum)/\(chunksToProcess) (pos=\(start)): \(String(format: "%.2f", chunkTime))s")
        }

        return (lastLogits, currentKV)
    }

    /// Run prefill for a single chunk. Used by eager prefill.
    public func prefillSingleChunk(
        chunkTokens: [Int32],
        startPosition: Int,
        kvState: KVCacheState
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        precondition(chunkTokens.count == GemmaConfig.chunkSize)
        let tokens = MLMultiArray.int32Row(chunkTokens)
        let gcSize = kvState.currentGlobalCacheSize.map { Int32($0) }
        let (logits, newKV) = try model.prefill(
            tokens: tokens,
            seqLen: Int32(startPosition),
            kvState: kvState,
            globalCacheSize: gcSize
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
