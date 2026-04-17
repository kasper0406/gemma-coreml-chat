/// Eager prefill manager: prefills prompt chunks in the background as the user types.
///
/// When the tokenized input crosses a CHUNK_SIZE boundary, the newly-complete
/// chunk is prefilled immediately, so by the time the user taps Send, most or
/// all prefill work is already done.
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
    /// How many complete CHUNK_SIZE chunks have been prefilled.
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
        // Initial empty state is a tiny allocation (no global size override) — safe to force.
        self.kvState = try! Self.emptyKV(model: model, initialGlobalSize: nil)
    }

    private static func emptyKV(model: CoreMLModel, initialGlobalSize: Int?) throws -> KVCacheState {
        try KVCacheState.empty(
            kvInputNames: model.prefillKVInputNames,
            shapes: model.prefillKVShapes,
            dtypes: model.prefillKVDtypes,
            globalNames: model.globalKVInputNames,
            initialGlobalSize: initialGlobalSize
        )
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
        let prefillBoundary = completedChunks * GemmaConfig.chunkSize
        if prefillBoundary > 0 {
            let isValid = newTokens.count >= prefillBoundary
                && newTokens.prefix(prefillBoundary).elementsEqual(prefillTokens.prefix(prefillBoundary))
            if !isValid {
                Log.info("[Perf] Prefix changed — resetting eager prefill")
                reset()
            }
        }

        // Check if there are new complete chunks to prefill
        let totalChunks = newTokens.count / GemmaConfig.chunkSize
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
        let prefillBoundary = completedChunks * GemmaConfig.chunkSize
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
                try Self.emptyKV(model: model, initialGlobalSize: nil),
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
        completedChunks = cached.count / GemmaConfig.chunkSize
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
        kvState = try! Self.emptyKV(model: model, initialGlobalSize: nil)
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
            // Size/grow global caches to fit all chunks we're about to process
            if !model.globalKVInputNames.isEmpty {
                let neededSize = upToChunk * GemmaConfig.chunkSize
                // Round to materialized size if applicable.
                let roundedSize = model.materializedSize(forCacheSize: neededSize) ?? neededSize
                if startChunk == 0 {
                    kvState = try Self.emptyKV(model: model, initialGlobalSize: roundedSize)
                } else {
                    kvState = try kvState.grownToFit(needed: roundedSize, maxLen: GemmaConfig.maxSeqLen)
                }
                try await model.ensureLoaded(forGlobalCacheSize: roundedSize)
            }
            for chunkIdx in startChunk..<upToChunk {
                let chunkStart = CFAbsoluteTimeGetCurrent()
                let start = chunkIdx * GemmaConfig.chunkSize
                let chunkTokens = Array(tokens[start..<(start + GemmaConfig.chunkSize)])
                    .map { Int32($0) }

                status = .prefilling(completed: chunkIdx, total: upToChunk)

                let (logits, newKV) = try await engine.prefillSingleChunk(
                    chunkTokens: chunkTokens,
                    startPosition: start,
                    kvState: kvState
                )

                kvState = newKV
                lastLogits = logits
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
