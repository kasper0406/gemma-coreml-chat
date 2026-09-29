/// Constants mirroring gemma_chat/config.py for Gemma4-E2B.
///
/// Deliberately small: anything the exported artifact already states — chunk
/// size, cache lengths, vocabulary — is read from the model description at load
/// (see ``CoreMLModel``) so a re-export can change it without a matching edit
/// here. What is left is tokenizer-level vocabulary that no CoreML feature
/// describes.

import Foundation

public enum GemmaConfig {
    /// Upper bound for the exported context length. Only used to clamp
    /// user-supplied context limits before the model reports what it actually
    /// materialized (`CoreMLModel.effectiveMaxSeqLen`).
    public static let maxSeqLen = 65_536

    /// EOS token ID for Gemma4.
    public static let eosTokenID: Int32 = 1

    /// End-of-turn token ID (``<turn|>``).
    public static let eotTokenID: Int32 = 106

    /// All token IDs that should stop generation.
    public static let stopTokenIDs: Set<Int32> = [eosTokenID, eotTokenID]

    /// BOS (beginning-of-sequence) token ID.
    public static let bosTokenID: Int = 2

    /// Pad token ID, used to fill the tail of a short prefill chunk.
    public static let padTokenID: Int32 = 0

    // The architecture constants the host needs to build each step's RoPE rows
    // and sliding mask (``HostInputs``), which no CoreML feature states. Gemma 4
    // E2B and E4B share them; `tests/test_gemma_config_swift.py` checks them
    // against every variant in gemma_chat/config.py.

    /// Positions a sliding-window layer attends to, the query's own included.
    public static let slidingWindow = 512
    /// RoPE base frequency and head dimension of the sliding-window layers.
    public static let slidingRopeBase = 10_000.0
    public static let slidingHeadDim = 256
    /// RoPE base frequency and head dimension of the global layers.
    public static let globalRopeBase = 1_000_000.0
    public static let globalHeadDim = 512
}
