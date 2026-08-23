/// Eager prefill manager: prefills prompt chunks in the background as the user types.
///
/// When the tokenized input crosses a chunk boundary, the newly-complete chunk
/// is prefilled immediately, so by the time the user taps Send, most or all
/// prefill work is already done. The chunk width is whatever the loaded model
/// declares (``CoreMLModel/chunkSize``), never a constant of our own.
///
/// Thread safety: this is a Swift actor, so all state mutation is serialized.

import CoreML
import Foundation
import GemmaCore

/// Observable state for the UI to display prefill progress.
enum PrefillStatus: Sendable, Equatable {
    case idle
    case prefilling(completed: Int, total: Int)
    case ready(chunks: Int)
    case error(String)
}

actor EagerPrefillManager {
    private let engine: InferenceEngine
    private let tokenizer: GemmaTokenizer
    private let model: CoreMLModel

    /// Tokens that have been prefilled so far.
    private var prefillTokens: [Int] = []
    /// How many complete chunks have been prefilled.
    private var completedChunks: Int = 0
    /// Current KV cache state after the last completed chunk.
    private var kvState: KVCacheState
    /// Logits from the last prefilled chunk (needed for decode start).
    private var lastLogits: MLMultiArray?
    /// Whether a prefill operation is currently running.
    private var isPrefilling: Bool = false

    /// Current status for UI display.
    private(set) var status: PrefillStatus = .idle

    init(engine: InferenceEngine, tokenizer: GemmaTokenizer, model: CoreMLModel) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.model = model
        // Smallest materialized pair, which `CoreMLModel.load` always brings
        // up, so the state can be made without awaiting a load — safe to force.
        self.kvState = try! model.makeEmptyKVState()
    }

    /// Tokens per prefill call, as the loaded artifact declares it. Never
    /// assume a value here: a decode-only build prefills one token at a time,
    /// a full build a whole chunk.
    private var chunkSize: Int { model.chunkSize }

    /// Most tokens we may eagerly prefill: whole chunks only, and never past
    /// what the loaded model can hold.
    ///
    /// `KVCacheSizePolicy` clamps to the largest size actually loaded, so
    /// without this cap a long prompt would size the cache to that clamped
    /// value and then chunk right past the end of it.
    private var maxEagerTokens: Int {
        (model.effectiveMaxSeqLen / chunkSize) * chunkSize
    }

    /// Called when the user's input text changes.
    /// Tokenizes the full prompt and prefills any newly-complete chunks.
    ///
    /// - Parameters:
    ///   - currentText: The user's current input text
    ///   - history: Conversation history (for chat template)
    ///   - systemPrompt: Optional system prompt
    func textChanged(
        currentText: String,
        history: [ChatMessage],
        systemPrompt: String? = nil
    ) async {
        guard !isPrefilling else { return }

        // Build the full prompt with current text
        var fullHistory = history
        if !currentText.isEmpty {
            fullHistory.append(ChatMessage(role: .user, content: currentText))
        }

        let newTokens = tokenizer.encodeChatPrompt(
            history: fullHistory,
            systemPrompt: systemPrompt
        )

        // Check if existing prefill is still valid
        let prefillBoundary = completedChunks * chunkSize
        if prefillBoundary > 0 {
            let isValid = newTokens.count >= prefillBoundary
                && newTokens.prefix(prefillBoundary).elementsEqual(prefillTokens.prefix(prefillBoundary))
            if !isValid {
                Log.info("[Perf] Prefix changed — resetting eager prefill")
                reset()
            }
        }

        // Check if there are new complete chunks to prefill, capped to what
        // the model can actually hold.
        let eagerTokens = min(newTokens.count, maxEagerTokens)
        if eagerTokens < newTokens.count {
            Log.info("[Perf] Eager prefill capped at \(eagerTokens)/\(newTokens.count) tokens (model max \(model.effectiveMaxSeqLen))")
        }
        let totalChunks = eagerTokens / chunkSize
        if totalChunks > completedChunks {
            await prefillNewChunks(tokens: newTokens, upToChunk: totalChunks)
        }
    }

    /// Complete any remaining partial chunk and return state for decode.
    ///
    /// Called when the user taps Send. Prefills the last (possibly partial) chunk
    /// and returns everything the inference engine needs to start decoding.
    ///
    /// - Parameters:
    ///   - finalText: The submitted message text
    ///   - history: Full conversation history including this message
    ///   - systemPrompt: Optional system prompt
    /// - Returns: Tuple of (promptIDs, kvState, prefillOffset) for the inference engine
    func finishPrefill(
        finalText: String,
        history: [ChatMessage],
        systemPrompt: String? = nil
    ) async throws -> (promptIDs: [Int32], kvState: KVCacheState, prefillOffset: Int) {
        // Wait for any in-flight prefill to complete
        if isPrefilling {
            Log.info("[Perf] finishPrefill: waiting for in-flight eager prefill...")
        }
        while isPrefilling {
            try await Task.sleep(for: .milliseconds(10))
        }

        // Tokenize the final prompt
        var fullHistory = history
        fullHistory.append(ChatMessage(role: .user, content: finalText))
        let finalTokens = tokenizer.encodeChatPrompt(
            history: fullHistory,
            systemPrompt: systemPrompt
        )
        let promptIDs = finalTokens.map { Int32($0) }

        // Check if our prefill is still valid for the final tokens
        let prefillBoundary = completedChunks * chunkSize
        let isValid = prefillBoundary > 0
            && finalTokens.count >= prefillBoundary
            && finalTokens.prefix(prefillBoundary).elementsEqual(prefillTokens.prefix(prefillBoundary))

        if isValid && completedChunks > 0 {
            // Prefill is still valid — engine only needs to process remaining chunks
            Log.info("[Perf] finishPrefill: reusing \(completedChunks) eager chunks (\(prefillBoundary)/\(finalTokens.count) tokens)")
            status = .ready(chunks: completedChunks)
            let result = (promptIDs, kvState, prefillBoundary)
            // Release internal state — the engine now owns the KV cache
            clearInternalState()
            return result
        } else {
            // Prefill invalidated — engine does full prefill
            Log.info("[Perf] finishPrefill: eager prefill invalid, full re-prefill (\(finalTokens.count) tokens)")
            status = .idle
            clearInternalState()
            return (
                promptIDs,
                try model.makeEmptyKVState(),
                0
            )
        }
    }

    /// Reset all prefill state (e.g., on /reset or new conversation).
    func reset() {
        clearInternalState()
        status = .idle
    }

    /// Seed the manager with KV state from a completed generation.
    ///
    /// After `generate()` finishes, call this instead of `reset()` so that the
    /// next turn's eager prefill (and `finishPrefill`) can skip tokens that are
    /// already in the KV cache.
    func seedFromGeneration(_ context: GenerationContext) {
        let cached = context.cachedTokens
        guard let kv = context.kvState, !cached.isEmpty else {
            reset()
            return
        }
        prefillTokens = cached.map { Int($0) }
        completedChunks = cached.count / chunkSize
        kvState = kv
        lastLogits = nil
        isPrefilling = false
        status = completedChunks > 0 ? .ready(chunks: completedChunks) : .idle
        Log.info("[KV] Seeded eager prefill with \(cached.count) tokens (\(completedChunks) complete chunks)")
    }

    /// Release KV cache and logits memory without changing status.
    private func clearInternalState() {
        prefillTokens = []
        completedChunks = 0
        // Tiny allocation (no global size override) — safe to force.
        kvState = try! model.makeEmptyKVState()
        lastLogits = nil
        isPrefilling = false
    }

    // MARK: - Private

    /// Prefill chunks from completedChunks to upToChunk.
    private func prefillNewChunks(tokens: [Int], upToChunk: Int) async {
        guard upToChunk > completedChunks else { return }
        isPrefilling = true
        let startChunk = completedChunks
        let totalToProcess = upToChunk - startChunk
        Log.info("[Perf] Eager prefill: \(totalToProcess) chunks (\(startChunk)..<\(upToChunk))")
        let batchStart = CFAbsoluteTimeGetCurrent()

        do {
            // Size/grow the caches to fit all chunks we're about to process,
            // via the model's own bucketing policy so the cache shape and the
            // resolved `prefill_<N>` function agree. The pair has to be loaded
            // first: a cache's state buffers are made from its own handle.
            let roundedSize = model.cacheSizePolicy.size(forNeeded: upToChunk * chunkSize)
            try await model.ensureLoaded(forGlobalCacheSize: roundedSize)
            if startChunk == 0 {
                kvState = try model.makeEmptyKVState(size: roundedSize)
            } else {
                kvState = try await model.grownToFit(kvState, needed: roundedSize)
            }
            for chunkIdx in startChunk..<upToChunk {
                let chunkStart = CFAbsoluteTimeGetCurrent()
                let start = chunkIdx * chunkSize
                let chunkTokens = Array(tokens[start..<(start + chunkSize)])
                    .map { Int32($0) }

                status = .prefilling(completed: chunkIdx, total: upToChunk)

                lastLogits = try await engine.prefillSingleChunk(
                    chunkTokens: chunkTokens,
                    startPosition: start,
                    kvState: kvState
                )
                completedChunks = chunkIdx + 1
                prefillTokens = tokens

                let chunkTime = CFAbsoluteTimeGetCurrent() - chunkStart
                let chunkNum = chunkIdx - startChunk + 1
                Log.info("[Perf] Eager chunk \(chunkNum)/\(totalToProcess) (pos=\(start)): \(String(format: "%.2f", chunkTime))s")
            }
            let totalTime = CFAbsoluteTimeGetCurrent() - batchStart
            Log.info("[Perf] Eager prefill done: \(totalToProcess) chunks in \(String(format: "%.1f", totalTime))s")
            status = .ready(chunks: completedChunks)
        } catch {
            status = .error(error.localizedDescription)
        }

        isPrefilling = false
    }
}
