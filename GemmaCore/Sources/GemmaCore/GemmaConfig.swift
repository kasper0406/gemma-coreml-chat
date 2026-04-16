/// Constants mirroring gemma_chat/config.py for Gemma4-E2B.
///
/// KV cache layout: 15 non-shared layers (0–14) for the 35-layer model.
/// Layers 15–34 are KV-shared and read from layers 13 (sliding) and 14 (global).

import Foundation

public enum GemmaConfig {
    /// Upper bound for global KV caches.  Models exported with RangeDim
    /// can dynamically grow up to this limit.
    public static let maxSeqLen = 65_536

    /// Sliding window size for LOCAL_SLIDING layers (ring-buffer length).
    public static let slidingWindowSize = 512

    /// Tokens per chunked-prefill call.
    public static let chunkSize = 8

    /// Vocabulary size (Gemma4-E2B).
    public static let vocabSize = 262_144

    /// Number of KV arrays: 15 layers × (k + v) + 1 sliding_pos_ring.
    public static let numStateArrays = 31

    /// EOS token ID for Gemma4.
    public static let eosTokenID: Int32 = 1

    /// End-of-turn token ID (``<turn|>``).
    public static let eotTokenID: Int32 = 106

    /// All token IDs that should stop generation.
    public static let stopTokenIDs: Set<Int32> = [eosTokenID, eotTokenID]

    /// BOS (beginning-of-sequence) token ID.
    public static let bosTokenID: Int = 2

    /// Pad token ID.
    public static let padTokenID: Int32 = 0
}
