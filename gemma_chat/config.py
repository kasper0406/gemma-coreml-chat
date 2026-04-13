"""Gemma4 configuration and shared constants."""

from gemma_chat.model import AttentionType, Gemma4Config

# Maximum sequence length used for the CoreML export.
# The model is exported with a fixed shape (1, MAX_SEQ_LEN).
# Prompts and generated text are padded/truncated to fit.
MAX_SEQ_LEN = 4096

# Number of tokens processed per chunked-prefill call.
# Chosen to roughly balance compute and memory-bandwidth on A-series chips,
# and to allow eager prefill as the user types.
CHUNK_SIZE = 8

# ── E2B (35 layers: 7 × [SLIDING×4, GLOBAL]) ──────────────────────────────

_E2B_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)

E2B_CONFIG = Gemma4Config(
    num_embed=262144,
    embed_dim=1536,
    hidden_dim=6144,
    num_heads=8,
    head_dim=256,
    num_kv_heads=1,
    global_head_dim=512,
    global_hidden_dim=0,
    wide_mlp_from_layer=15,
    wide_hidden_dim=12288,
    final_logit_softcap=30.0,
    attention_types=_E2B_ATTENTION_PATTERN * 7,
    sliding_window_size=512,
    rope_base_frequency=10_000.0,
    global_rope_base_frequency=1_000_000.0,
    rope_fraction_sliding=1.0,
    rope_fraction_global=0.25,
    per_layer_input_dim=256,
    num_kv_shared_layers=20,
)

# ── E4B (42 layers: 7 × [SLIDING×5, GLOBAL]) ──────────────────────────────

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
    wide_mlp_from_layer=-1,   # E4B does NOT use double-wide MLP
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

# ── Variant registry ───────────────────────────────────────────────────────

VARIANTS = {
    "e2b": {
        "config": E2B_CONFIG,
        "hf_model_id": "google/gemma-4-E2B-it",
        "mlpackage_path": "gemma4-e2b.mlpackage",
    },
    "e4b": {
        "config": E4B_CONFIG,
        "hf_model_id": "google/gemma-4-E4B-it",
        "mlpackage_path": "gemma4-e4b.mlpackage",
    },
}

# Backward-compat aliases (E2B defaults, used by many modules)
HF_MODEL_ID = VARIANTS["e2b"]["hf_model_id"]
MLPACKAGE_PATH = VARIANTS["e2b"]["mlpackage_path"]
