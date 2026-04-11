/// Constants mirroring gemma_chat/config.py for Gemma4-E2B.
///
/// KV cache layout: 15 non-shared layers (0–14) for the 35-layer model.
/// Layers 15–34 are KV-shared and read from layers 13 (sliding) and 14 (global).

import Foundation

enum GemmaConfig {
    /// Maximum sequence length the CoreML model was exported with.
    static let maxSeqLen = 10_000

    /// Sliding window size for LOCAL_SLIDING layers (ring-buffer length).
    static let slidingWindowSize = 512

    /// Tokens per chunked-prefill call.
    static let chunkSize = 8

    /// Vocabulary size (Gemma4-E2B).
    static let vocabSize = 262_144

    /// Number of KV arrays: 15 layers × (k + v) + 1 sliding_pos_ring.
    static let numStateArrays = 31

    /// EOS token ID for Gemma4.
    static let eosTokenID: Int32 = 1

    /// Pad token ID.
    static let padTokenID: Int32 = 0
}
