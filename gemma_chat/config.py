"""Gemma4 configuration (E2B / E4B) and shared constants."""

from gemma_chat.model import AttentionType, Gemma4Config

# Maximum sequence length for the CoreML export.
# Global KV caches use RangeDim(1, MAX_SEQ_LEN) and grow dynamically;
# sliding caches are fixed at sliding_window_size (512).
MAX_SEQ_LEN = 65536

# Number of tokens processed per chunked-prefill call.
# Chosen to roughly balance compute and memory-bandwidth on A-series chips,
# and to allow eager prefill as the user types.
# Every materialized cache size must be >= CHUNK_SIZE (a chunk has to fit in
# the cache); `gemma-export` enforces that on --materialize-sizes.
CHUNK_SIZE = 128

# First layer of each layer chunk: every prefill / decode step runs as one
# function per chunk (``decode_coreml.layer_chunks``), then the logit head.
# The Neural Engine compiles no function past a size limit, so the chunks must
# stay under it at every exported size.  Measured on an M4 Pro (macOS 27): with
# the head in a function of its own, all 35 layers fit in one function for
# prefill and decode at every size up to 16384, and each extra chunk costs
# ~0.7 ms per GPU decode step while saving nothing on the ANE — so one chunk.
# (0, 18) is the measured-eligible two-chunk layout for a device with a lower
# limit.  See README ("Layer chunks").
LAYER_CHUNK_STARTS = (0,)

# Full Gemma4-E2B architecture (35 layers: 7 × [SLIDING×4, GLOBAL])
_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)

E2B_CONFIG = Gemma4Config(
    num_embed=262144,
    embed_dim=1536,
    # Sliding layers: hidden_dim=6144, head_dim=256
    hidden_dim=6144,
    num_heads=8,
    head_dim=256,
    num_kv_heads=1,
    # Global attention layers use head_dim=512
    global_head_dim=512,
    global_hidden_dim=0,        # MLP width is NOT tied to attn type
    # MLP widens from layer 15 onward (independent of attention type)
    wide_mlp_from_layer=15,
    wide_hidden_dim=12288,
    final_logit_softcap=30.0,
    attention_types=_ATTENTION_PATTERN * 7,
    sliding_window_size=512,
    rope_base_frequency=10_000.0,
    global_rope_base_frequency=1_000_000.0,
    rope_fraction_sliding=1.0,
    rope_fraction_global=0.25,
    per_layer_input_dim=256,
    num_kv_shared_layers=20,
)

# HuggingFace model ID for Gemma4-E2B instruction-tuned
# Full Gemma4-E4B architecture (42 layers: 7 × [SLIDING×5, GLOBAL]).
# Differences from E2B that matter downstream:
#   * num_kv_heads=2 (E2B has 1), so GQA groups 4 query heads per KV head
#   * no double-wide MLP -- every layer uses hidden_dim
#   * 18 of the 42 layers share KV from an earlier layer
_E4B_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)

E4B_CONFIG = Gemma4Config(
    num_embed=262144,
    embed_dim=2560,
    hidden_dim=10240,
    num_heads=8,
    head_dim=256,
    num_kv_heads=2,
    global_head_dim=512,
    global_hidden_dim=0,
    wide_mlp_from_layer=-1,   # E4B does NOT use a double-wide MLP
    wide_hidden_dim=0,
    final_logit_softcap=30.0,
    attention_types=_E4B_ATTENTION_PATTERN * 7,
    sliding_window_size=512,
    rope_base_frequency=10_000.0,
    global_rope_base_frequency=1_000_000.0,
    rope_fraction_sliding=1.0,
    rope_fraction_global=0.25,
    per_layer_input_dim=256,
    num_kv_shared_layers=18,
)

# name -> (config, HuggingFace id, default .mlpackage path)
VARIANTS = {
    "e2b": (E2B_CONFIG, "google/gemma-4-E2B-it", "gemma4-e2b.mlpackage"),
    "e4b": (E4B_CONFIG, "google/gemma-4-E4B-it", "gemma4-e4b.mlpackage"),
}


HF_MODEL_ID = "google/gemma-4-E2B-it"

# Path to the exported CoreML multifunction .mlpackage containing both
# prefill and decode functions with shared (int4-quantized) weights.
# Global KV caches use RangeDim(1, MAX_SEQ_LEN) for dynamic context sizing.
MLPACKAGE_PATH = "gemma4-e2b.mlpackage"
