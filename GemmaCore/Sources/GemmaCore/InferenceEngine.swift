/// Chunked prefill + single-token decode inference engine.
///
/// Returns an `AsyncThrowingStream<Int32, Error>` of generated token IDs.
///
/// Chunk size is never assumed: it comes from ``CoreMLModel/chunkSize``, which
/// the model reads off its own prefill signature (and pins to 1 for
/// decode-only artifacts, which prefill by looping decode).

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
/// Pass the same instance to successive ``InferenceEngine/generate`` calls to
/// skip re-prefilling tokens that are already in the KV cache. The captured
/// ``KVCacheState`` is the live cache — the same object the generation decoded
/// into — so `cachedTokens` and it must be dropped together; that is what
/// ``reset()`` is for.
public final class GenerationContext: @unchecked Sendable {
    /// Token sequence currently represented in the KV cache (prompt + generated).
    public internal(set) var cachedTokens: [Int32] = []

    /// KV cache state after the last generation.
    public internal(set) var kvState: KVCacheState?

    public init() {}

    /// Discard cached state (e.g., on conversation reset). The next generation
    /// allocates a fresh cache.
    public func reset() {
        cachedTokens = []
        kvState = nil
    }
}

public struct InferenceEngine: Sendable {
    public let model: CoreMLModel
    public let temperature: Float
    public let topP: Float

    /// Cache rows reserved beyond the prompt when the decode loop sizes its
    /// cache up front.
    ///
    /// Reserving the full `maxNewTokens` (1024 in the CLI) pushes a
    /// 1100-token conversation from the 2048-row pair into the 4096-row one
    /// before a single token is generated: an extra multi-GB function load, and
    /// every decode step then attends over a cache twice as long as the
    /// conversation. A couple of hundred rows of slack covers a typical reply
    /// without crossing a boundary; a longer generation grows mid-loop instead,
    /// which costs one state migration (``CoreMLModel/grownToFit(_:needed:)``)
    /// and is the right trade for the reply that actually needs the room.
    private static let decodeCacheHeadroom = 256

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
    ///   - respectStopTokens: When false, generation runs to `maxNewTokens` even if a stop token
    ///     is sampled (benchmarks need a fixed number of decode steps)
    /// - Returns: AsyncStream yielding generated token IDs (including EOS)
    public func generate(
        promptIDs: [Int32],
        maxNewTokens: Int = 256,
        existingKVState: KVCacheState? = nil,
        prefillOffset: Int = 0,
        context: GenerationContext? = nil,
        respectStopTokens: Bool = true
    ) -> AsyncThrowingStream<Int32, Error> {
        AsyncThrowingStream { continuation in
            Task.detached { [self] in
                do {
                    let genStart = CFAbsoluteTimeGetCurrent()
                    let ids = truncatePromptIDs(
                        promptIDs,
                        maxSeqLen: model.effectiveMaxSeqLen,
                        reserveForGeneration: maxNewTokens
                    )

                    // Invalidate KV reuse if truncation changed the prompt —
                    // the cached prefix no longer matches the truncated suffix.
                    // Clearing the state here routes us through `fullPrefill`,
                    // which allocates a fresh cache.
                    var effectiveKVState = existingKVState
                    var effectivePrefillOffset = prefillOffset
                    if ids.count < promptIDs.count && prefillOffset > 0 {
                        Log.info("[KV] Prompt was truncated (\(promptIDs.count)→\(ids.count)) — invalidating KV reuse")
                        effectiveKVState = nil
                        effectivePrefillOffset = 0
                    }

                    let nReal = ids.count
                    let chunkSize = model.chunkSize
                    let nChunks = (nReal + chunkSize - 1) / chunkSize
                    Log.info("[Perf] Prompt: \(nReal) tokens, \(nChunks) chunks of \(chunkSize), prefillOffset=\(effectivePrefillOffset)")

                    // --- Chunked Prefill ---
                    let prefillStart = CFAbsoluteTimeGetCurrent()
                    var currentKV: KVCacheState
                    var currentLogits: MLMultiArray

                    if let existing = effectiveKVState, effectivePrefillOffset > 0 {
                        let (prefillLogits, prefillKV) = try await self.continuePrefill(
                            ids: ids,
                            fromOffset: effectivePrefillOffset,
                            kvState: existing
                        )
                        currentKV = prefillKV

                        if let prefillLogits {
                            currentLogits = prefillLogits
                        } else {
                            // All chunks were already prefilled. Run a single
                            // decode step with the last token to get logits.
                            currentLogits = try model.decode(
                                token: ids[nReal - 1],
                                position: Int32(nReal - 1),
                                kvState: currentKV
                            )
                        }
                    } else {
                        (currentLogits, currentKV) = try await self.fullPrefill(ids: ids)
                    }
                    let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
                    Log.info("[Perf] Prefill done: \(String(format: "%.2f", prefillTime))s")

                    // --- Decode Loop ---
                    let maxSteps = min(maxNewTokens, model.effectiveMaxSeqLen - nReal)

                    // Size the cache once for the prompt plus a modest amount
                    // of generation (see `decodeCacheHeadroom`) instead of
                    // per step: every grow migrates the whole cache into a
                    // fresh MLState, so the steady state wants to be a no-op.
                    let targetCacheSize = min(
                        nReal + min(maxSteps, Self.decodeCacheHeadroom),
                        model.effectiveMaxSeqLen
                    )
                    currentKV = try await model.grownToFit(currentKV, needed: targetCacheSize)

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

                        if respectStopTokens && GemmaConfig.stopTokenIDs.contains(nextID) { break }
                        if Task.isCancelled { break }

                        let position = Int32(nReal + step)
                        currentKV = try await model.grownToFit(
                            currentKV, needed: Int(position) + 1
                        )

                        // Safety: the cache size is clamped to what the model
                        // actually loaded, so a long conversation can reach
                        // past it rather than growing again.
                        if Int(position) >= currentKV.size {
                            Log.info("[Safety] Position \(position) >= cache size \(currentKV.size) — stopping generation")
                            break
                        }

                        let decStart = CFAbsoluteTimeGetCurrent()
                        // autoreleasepool: force prompt release of CoreML prediction
                        // temporaries (MLFeatureProvider, internal IOSurface-backed
                        // buffers) that are otherwise held until the task yields.
                        currentLogits = try autoreleasepool {
                            try model.decode(
                                token: nextID,
                                position: position,
                                kvState: currentKV
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
                    }

                    let totalTime = CFAbsoluteTimeGetCurrent() - genStart
                    let tokPerSec = decodeSteps > 0 ? Double(decodeSteps) / totalDecodeTime : 0
                    Log.info("[Perf] Done: \(decodeSteps) tokens in \(String(format: "%.1f", totalTime))s (prefill=\(String(format: "%.1f", prefillTime))s, decode=\(String(format: "%.1f", totalDecodeTime))s, sample=\(String(format: "%.2f", totalSampleTime))s) \(String(format: "%.2f", tokPerSec)) tok/s")

                    // Save state for cross-turn KV reuse
                    context?.cachedTokens = Array(ids.prefix(nReal)) + generatedIDs
                    context?.kvState = currentKV
                    continuation.finish()
                } catch {
                    // Predictions mutate the KV caches in place, so a run that
                    // died part-way leaves them ahead of `cachedTokens`. Drop
                    // the context so the next turn re-prefills against a fresh
                    // cache instead of trusting a half-written one.
                    context?.reset()
                    Log.info("Inference error: \(error)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Run full prefill from scratch into a fresh KV cache.
    public func fullPrefill(ids: [Int32]) async throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        let chunkSize = model.chunkSize
        let paddedLen = ((ids.count + chunkSize - 1) / chunkSize) * chunkSize

        // One bucketing policy for everyone — see `KVCacheSizePolicy`.
        let size = model.cacheSizePolicy.size(forNeeded: paddedLen)
        try await model.ensureLoaded(forGlobalCacheSize: size)

        // A full prefill starts from position 0, so it needs a *fresh* cache:
        // any surviving sliding K/V would be read back as valid once
        // `sliding_pos_ring` is repopulated.
        let emptyKV = try model.makeEmptyKVState(size: size)
        let (logits, kv) = try await continuePrefill(ids: ids, fromOffset: 0, kvState: emptyKV)
        guard let logits else {
            throw InferenceError.emptyPrompt
        }
        return (logits, kv)
    }

    /// Continue prefill from a given offset with an existing KV cache.
    ///
    /// Returns the logits of the last *real* prompt token (nil when `fromOffset`
    /// already covers the whole prompt) and the cache to keep decoding with —
    /// a different object from `kvState` when the prompt forced a grow.
    public func continuePrefill(
        ids: [Int32],
        fromOffset: Int,
        kvState: KVCacheState
    ) async throws -> (logits: MLMultiArray?, kvState: KVCacheState) {
        let chunkSize = model.chunkSize
        let nReal = ids.count
        let nChunks = (nReal + chunkSize - 1) / chunkSize
        let paddedLen = nChunks * chunkSize
        let padded = ids + [Int32](repeating: GemmaConfig.padTokenID,
                                   count: paddedLen - nReal)

        let startChunk = fromOffset / chunkSize
        let currentKV = try await model.grownToFit(kvState, needed: paddedLen)
        // The cache only guarantees its decode function is loaded; prefill for
        // the same size may still be cold when nothing had to grow.
        try await model.ensureLoaded(forGlobalCacheSize: currentKV.size)

        var lastLogits: MLMultiArray? = nil
        let chunksToProcess = nChunks - startChunk

        for chunkIdx in startChunk..<nChunks {
            let chunkStart = CFAbsoluteTimeGetCurrent()
            let start = chunkIdx * chunkSize
            let chunkTokens = Array(padded[start..<(start + chunkSize)])
            // Only the last real token's row is ever wanted, and only the final
            // chunk is padded, so every earlier chunk wants its last row.
            let realInChunk = min(nReal - start, chunkSize)

            lastLogits = try model.prefill(
                tokens: chunkTokens,
                startPosition: Int32(start),
                logitsRow: realInChunk - 1,
                kvState: currentKV
            )

            let chunkTime = CFAbsoluteTimeGetCurrent() - chunkStart
            let chunkNum = chunkIdx - startChunk + 1
            Log.info("[Perf] Prefill chunk \(chunkNum)/\(chunksToProcess) (pos=\(start)): \(String(format: "%.2f", chunkTime))s")
        }

        return (lastLogits, currentKV)
    }

    /// Run prefill for a single full chunk. Used by eager prefill.
    ///
    /// Returns the logits of the chunk's last token. The caller is responsible
    /// for sizing `kvState` to fit `startPosition + chunkTokens.count`;
    /// ``CoreMLModel/prefill(tokens:startPosition:logitsRow:kvState:)`` rejects
    /// a chunk that would run past the end.
    public func prefillSingleChunk(
        chunkTokens: [Int32],
        startPosition: Int,
        kvState: KVCacheState
    ) async throws -> MLMultiArray {
        precondition(chunkTokens.count == model.chunkSize)
        try await model.ensureLoaded(forGlobalCacheSize: kvState.size)
        return try model.prefill(
            tokens: chunkTokens,
            startPosition: Int32(startPosition),
            logitsRow: chunkTokens.count - 1,
            kvState: kvState
        )
    }

    // MARK: - Helpers

    /// Keep the last tokens so the prompt fits within maxSeqLen.
    ///
    /// The cap is rounded down to a chunk boundary: prefill pads the prompt up
    /// to a multiple of the model's chunk size, so an unrounded cap can pad
    /// past the largest cache the model loaded.
    private func truncatePromptIDs(
        _ ids: [Int32],
        maxSeqLen: Int,
        reserveForGeneration: Int
    ) -> [Int32] {
        let raw = max(maxSeqLen - reserveForGeneration, 1)
        let chunk = model.chunkSize
        let cap = max((raw / chunk) * chunk, min(chunk, maxSeqLen))
        if ids.count > cap {
            return Array(ids.suffix(cap))
        }
        return ids
    }
}
