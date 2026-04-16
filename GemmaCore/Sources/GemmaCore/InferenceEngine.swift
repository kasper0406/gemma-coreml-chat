/// Chunked prefill + single-token decode inference engine.
///
/// Returns an `AsyncThrowingStream<Int32, Error>` of generated token IDs.

import CoreML
import Foundation

/// Errors raised during inference.
public enum InferenceError: Error, LocalizedError {
    case emptyPrompt

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt:
            "Cannot run inference on an empty prompt"
        }
    }
}

/// Captures post-generation KV state for reuse across turns.
///
/// Pass the same instance to successive ``InferenceEngine/generate`` calls
/// to skip re-prefilling tokens that are already in the KV cache.
public final class GenerationContext: @unchecked Sendable {
    /// Token sequence currently represented in the KV cache (prompt + generated).
    public internal(set) var cachedTokens: [Int32] = []

    /// KV cache state after the last generation.
    public internal(set) var kvState: KVCacheState?

    public init() {}

    /// Discard cached state (e.g., on conversation reset).
    public func reset() {
        cachedTokens = []
        kvState = nil
    }
}

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
    ///   - existingKVState: Optional pre-populated KV state (from eager prefill or prior turn)
    ///   - prefillOffset: If using existingKVState, how many tokens were already prefilled
    ///   - context: Optional context for capturing post-generation KV state (enables cross-turn reuse)
    /// - Returns: AsyncStream yielding generated token IDs (including EOS)
    public func generate(
        promptIDs: [Int32],
        maxNewTokens: Int = 256,
        existingKVState: KVCacheState? = nil,
        prefillOffset: Int = 0,
        context: GenerationContext? = nil
    ) -> AsyncThrowingStream<Int32, Error> {
        AsyncThrowingStream { continuation in
            Task.detached { [self] in
                do {
                    let genStart = CFAbsoluteTimeGetCurrent()
                    let ids = truncatePromptIDs(
                        promptIDs,
                        maxSeqLen: GemmaConfig.maxSeqLen,
                        reserveForGeneration: maxNewTokens
                    )

                    // Invalidate KV reuse if truncation changed the prompt —
                    // the cached prefix no longer matches the truncated suffix.
                    var effectiveKVState = existingKVState
                    var effectivePrefillOffset = prefillOffset
                    if ids.count < promptIDs.count && prefillOffset > 0 {
                        Log.info("[KV] Prompt was truncated (\(promptIDs.count)→\(ids.count)) — invalidating KV reuse")
                        effectiveKVState = nil
                        effectivePrefillOffset = 0
                    }

                    let nReal = ids.count
                    let nChunks = (nReal + GemmaConfig.chunkSize - 1) / GemmaConfig.chunkSize
                    Log.info("[Perf] Prompt: \(nReal) tokens, \(nChunks) chunks, prefillOffset=\(effectivePrefillOffset)")

                    try await self.model.loadPrefill()

                    // --- Chunked Prefill ---
                    let prefillStart = CFAbsoluteTimeGetCurrent()
                    var kvState: KVCacheState
                    var logits: MLMultiArray

                    if let existing = effectiveKVState, effectivePrefillOffset > 0 {
                        let (prefillLogits, prefillKV) = try self.continuePrefill(
                            ids: ids,
                            fromOffset: effectivePrefillOffset,
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

                    // Pre-allocate the global KV cache to its final decode size so we don't
                    // re-allocate every step. Each grow returns a fresh MLMultiArray; once it
                    // is passed to a prediction, CoreML backs it with an IOSurface that isn't
                    // released promptly. Repeated growth exhausts the IOSurface pool after
                    // enough steps ("Failed to allocate E5 buffer object").
                    let targetCacheSize = min(nReal + maxSteps, GemmaConfig.maxSeqLen)
                    var currentKV = try kvState.grownToFit(
                        needed: targetCacheSize, maxLen: GemmaConfig.maxSeqLen
                    )
                    var currentLogits = lastLogits
                    var totalSampleTime = 0.0
                    var totalDecodeTime = 0.0
                    var decodeSteps = 0
                    var generatedIDs: [Int32] = []

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

                        currentKV = try currentKV.grownToFit(
                            needed: Int(position) + 1,
                            maxLen: GemmaConfig.maxSeqLen
                        )

                        // Safety: verify position fits in the global cache before decode.
                        if let gcSize = currentKV.currentGlobalCacheSize, Int(position) >= gcSize {
                            Log.info("[Safety] Position \(position) >= global cache size \(gcSize) — stopping generation")
                            break
                        }

                        let decStart = CFAbsoluteTimeGetCurrent()
                        let gcSize = currentKV.currentGlobalCacheSize.map { Int32($0) }
                        // autoreleasepool: force prompt release of CoreML prediction
                        // temporaries (MLFeatureProvider, internal IOSurface-backed
                        // buffers) that are otherwise held until the task yields.
                        let (decLogits, decKV) = try autoreleasepool {
                            try model.decode(
                                token: nextID,
                                position: position,
                                kvState: currentKV,
                                globalCacheSize: gcSize
                            )
                        }
                        let decTime = CFAbsoluteTimeGetCurrent() - decStart
                        totalDecodeTime += decTime
                        decodeSteps += 1
                        generatedIDs.append(nextID)

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

                    // Save state for cross-turn KV reuse
                    context?.cachedTokens = Array(ids.prefix(nReal)) + generatedIDs
                    context?.kvState = currentKV
                    continuation.finish()
                } catch {
                    Log.info("Inference error: \(error)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Run full prefill from scratch.
    public func fullPrefill(ids: [Int32]) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        let nChunks = (ids.count + GemmaConfig.chunkSize - 1) / GemmaConfig.chunkSize
        let paddedLen = nChunks * GemmaConfig.chunkSize

        let emptyKV = try KVCacheState.empty(
            kvInputNames: model.prefillKVInputNames,
            shapes: model.prefillKVShapes,
            dtypes: model.prefillKVDtypes,
            globalNames: model.globalKVInputNames,
            initialGlobalSize: model.globalKVInputNames.isEmpty ? nil : paddedLen
        )
        let (logits, kv) = try continuePrefill(ids: ids, fromOffset: 0, kvState: emptyKV)
        guard let logits else {
            throw InferenceError.emptyPrompt
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
        var currentKV = try kvState.grownToFit(needed: paddedLen, maxLen: GemmaConfig.maxSeqLen)
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
            currentKV = newKV
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
        return (logits, newKV)
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
