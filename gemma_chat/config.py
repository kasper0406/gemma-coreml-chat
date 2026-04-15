"""Gemma4-E2B configuration and shared constants."""

from gemma_chat.model import AttentionType, Gemma4Config

# Maximum sequence length for the CoreML export.
# Global KV caches use RangeDim(1, MAX_SEQ_LEN) and grow dynamically;
# sliding caches are fixed at sliding_window_size (512).
MAX_SEQ_LEN = 65536

# Number of tokens processed per chunked-prefill call.
# Chosen to roughly balance compute and memory-bandwidth on A-series chips,
# and to allow eager prefill as the user types.
CHUNK_SIZE = 8

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
HF_MODEL_ID = "google/gemma-4-E2B-it"

# Path where the exported multifunction CoreML model is saved
MLPACKAGE_PATH = "gemma4-e2b.mlpackage"
