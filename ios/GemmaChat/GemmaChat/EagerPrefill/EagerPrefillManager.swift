/// Eager prefill manager: prefills prompt chunks in the background as the user types.
///
/// When the tokenized input crosses a CHUNK_SIZE boundary, the newly-complete
/// chunk is prefilled immediately, so by the time the user taps Send, most or
/// all prefill work is already done.
///
/// Thread safety: this is a Swift actor, so all state mutation is serialized.

import CoreML
import Foundation

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
    /// Ordered state input names, their shapes, and their dtypes.
    private let kvInputNames: [String]
    private let kvShapes: [String: [NSNumber]]
    private let kvDtypes: [String: MLMultiArrayDataType]

    /// Tokens that have been prefilled so far.
    private var prefillTokens: [Int] = []
    /// How many complete CHUNK_SIZE chunks have been prefilled.
    private var completedChunks: Int = 0
    /// Current KV cache state after the last completed chunk.
    private var kvState: KVCacheState
    /// KV state checkpoint after each completed chunk.
    /// checkpoints[i] = KV state after chunk i was prefilled.
    /// Invariant: checkpoints.count == completedChunks.
    private var checkpoints: [KVCacheState] = []
    /// Logits from the last prefilled chunk (needed for decode start).
    private var lastLogits: MLMultiArray?
    /// Whether a prefill operation is currently running.
    private var isPrefilling: Bool = false

    /// Current status for UI display.
    private(set) var status: PrefillStatus = .idle

    init(engine: InferenceEngine, tokenizer: GemmaTokenizer,
         kvInputNames: [String],
         inputDescriptions: [String: MLFeatureDescription])
    {
        self.engine = engine
        self.tokenizer = tokenizer
        self.kvInputNames = kvInputNames
        // Pre-extract shapes and dtypes so we don't store non-Sendable MLFeatureDescription
        var shapes: [String: [NSNumber]] = [:]
        var dtypes: [String: MLMultiArrayDataType] = [:]
        for name in kvInputNames {
            if let desc = inputDescriptions[name],
               let constraint = desc.multiArrayConstraint
            {
                shapes[name] = constraint.shape
                dtypes[name] = constraint.dataType
            }
        }
        self.kvShapes = shapes
        self.kvDtypes = dtypes
        self.kvState = Self.makeEmptyKV(names: kvInputNames, shapes: shapes, dtypes: dtypes)
    }

    private static func makeEmptyKV(
        names: [String], shapes: [String: [NSNumber]], dtypes: [String: MLMultiArrayDataType]
    ) -> KVCacheState {
        var dict: [String: MLMultiArray] = [:]
        for name in names {
            let shape = shapes[name] ?? [1]
            let dtype = dtypes[name] ?? .float16
            let array = try! MLMultiArray(shape: shape, dataType: dtype)
            if dtype == .int32 {
                // Ring position tracker: -1 means "empty slot"
                let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
                for i in 0..<array.count { ptr[i] = -1 }
            }
            dict[name] = array
        }
        return KVCacheState(arraysByName: dict, inputNames: names, processedTokens: 0)
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
                && Array(newTokens.prefix(prefillBoundary)) == Array(prefillTokens.prefix(prefillBoundary))
            if !isValid {
                // Prefill invalidated — find last valid chunk
                let lastValid = findLastValidChunk(newTokens: newTokens)
                if lastValid < completedChunks {
                    print("[Perf] Prefix changed at chunk \(lastValid)/\(completedChunks) — restoring checkpoint (saved \(lastValid) re-prefills)")
                    resetTo(chunk: lastValid, tokens: newTokens)
                }
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
            print("[Perf] finishPrefill: waiting for in-flight eager prefill...")
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
            && Array(finalTokens.prefix(prefillBoundary)) == Array(prefillTokens.prefix(prefillBoundary))

        if isValid && completedChunks > 0 {
            // Prefill is still valid — engine only needs to process remaining chunks
            print("[Perf] finishPrefill: reusing \(completedChunks) eager chunks (\(prefillBoundary)/\(finalTokens.count) tokens)")
            status = .ready(chunks: completedChunks)
            let result = (promptIDs, kvState, prefillBoundary)
            // Release internal state — the engine now owns the KV cache
            clearInternalState()
            return result
        } else if !checkpoints.isEmpty {
            // Full state invalid, but we may be able to salvage a prefix via checkpoints
            let lastValid = findLastValidChunk(newTokens: finalTokens)
            if lastValid > 0 {
                let validBoundary = lastValid * GemmaConfig.chunkSize
                print("[Perf] finishPrefill: partial reuse — \(lastValid)/\(completedChunks) chunks (\(validBoundary)/\(finalTokens.count) tokens)")
                let result = (promptIDs, checkpoints[lastValid - 1], validBoundary)
                clearInternalState()
                return result
            } else {
                print("[Perf] finishPrefill: eager prefill invalid, full re-prefill (\(finalTokens.count) tokens)")
                status = .idle
                clearInternalState()
                return (
                    promptIDs,
                    Self.makeEmptyKV(names: kvInputNames, shapes: kvShapes, dtypes: kvDtypes),
                    0
                )
            }
        } else {
            // Prefill invalidated — engine does full prefill
            print("[Perf] finishPrefill: eager prefill invalid, full re-prefill (\(finalTokens.count) tokens)")
            status = .idle
            clearInternalState()
            return (
                promptIDs,
                Self.makeEmptyKV(names: kvInputNames, shapes: kvShapes, dtypes: kvDtypes),
                0
            )
        }
    }

    /// Reset all prefill state (e.g., on /reset or new conversation).
    func reset() {
        clearInternalState()
        status = .idle
    }

    /// Release KV cache and logits memory without changing status.
    private func clearInternalState() {
        prefillTokens = []
        completedChunks = 0
        kvState = Self.makeEmptyKV(names: kvInputNames, shapes: kvShapes, dtypes: kvDtypes)
        checkpoints = []
        lastLogits = nil
        isPrefilling = false
    }

    // MARK: - Private

    /// Find the last chunk whose tokens still match.
    private func findLastValidChunk(newTokens: [Int]) -> Int {
        let chunkSize = GemmaConfig.chunkSize
        for chunk in stride(from: completedChunks, through: 0, by: -1) {
            let boundary = chunk * chunkSize
            if boundary == 0 { return 0 }
            if newTokens.count >= boundary
                && Array(newTokens.prefix(boundary)) == Array(prefillTokens.prefix(boundary))
            {
                return chunk
            }
        }
        return 0
    }

    /// Reset prefill state back to a given chunk boundary.
    /// Uses saved checkpoints to restore KV state without re-prefilling.
    private func resetTo(chunk: Int, tokens: [Int]) {
        if chunk == 0 {
            reset()
            return
        }
        // Restore from checkpoint — O(1) instead of re-prefilling from scratch.
        kvState = checkpoints[chunk - 1]
        completedChunks = chunk
        checkpoints = Array(checkpoints.prefix(chunk))
        lastLogits = nil
        prefillTokens = tokens
    }

    /// Prefill chunks from completedChunks to upToChunk.
    private func prefillNewChunks(tokens: [Int], upToChunk: Int) async {
        guard upToChunk > completedChunks else { return }
        isPrefilling = true
        let startChunk = completedChunks
        let totalToProcess = upToChunk - startChunk
        print("[Perf] Eager prefill: \(totalToProcess) chunks (\(startChunk)..<\(upToChunk))")
        let batchStart = CFAbsoluteTimeGetCurrent()

        do {
            for chunkIdx in startChunk..<upToChunk {
                let chunkStart = CFAbsoluteTimeGetCurrent()
                let start = chunkIdx * GemmaConfig.chunkSize
                let chunkTokens = Array(tokens[start..<(start + GemmaConfig.chunkSize)])
                    .map { Int32($0) }

                status = .prefilling(completed: chunkIdx, total: upToChunk)

                let (logits, newKV) = try engine.prefillSingleChunk(
                    chunkTokens: chunkTokens,
                    startPosition: start,
                    kvState: kvState
                )

                kvState = newKV
                lastLogits = logits
                completedChunks = chunkIdx + 1
                checkpoints.append(kvState.deepCopy())
                prefillTokens = tokens

                let chunkTime = CFAbsoluteTimeGetCurrent() - chunkStart
                let chunkNum = chunkIdx - startChunk + 1
                print("[Perf] Eager chunk \(chunkNum)/\(totalToProcess) (pos=\(start)): \(String(format: "%.2f", chunkTime))s")
            }
            let totalTime = CFAbsoluteTimeGetCurrent() - batchStart
            print("[Perf] Eager prefill done: \(totalToProcess) chunks in \(String(format: "%.1f", totalTime))s")
            status = .ready(chunks: completedChunks)
        } catch {
            status = .error(error.localizedDescription)
        }

        isPrefilling = false
    }
}
