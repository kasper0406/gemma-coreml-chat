# Copilot Instructions

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Export Gemma4-E2B to CoreML — one-time, takes 10–30 min
uv run gemma-export

# Run the chat TUI (requires .mlpackage to exist)
uv run gemma-chat

# Diagnostics
uv run gemma-compare-inference --prompt "Hi"   # A/B: full model vs KV decode
uv run gemma-parity-decode --max-tokens 8       # JAX vs CoreML logit parity
```

There is no test suite.

## Architecture

This is a two-phase pipeline for running `google/gemma-4-E2B-it` locally on Apple Silicon via CoreML:

**Phase 1 — Export (`gemma-export`, run once):**
1. `weight_mapper.py` downloads HuggingFace safetensors and maps them to a nested Flax param dict (transposing all Linear kernels from `[out, in]` to `[in, out]`).
2. `model.py` defines the full Gemma4 transformer in JAX/Flax (`Gemma4Transformer`).
3. `decode_coreml.py` provides JAX-traceable `chunk_prefill_step` and `decode_step` functions that operate on a flat KV cache (30 arrays for 15 non-shared layers × k + v). These are designed for lowering to StableHLO.
4. `cache_spec.py` defines the KV-cache layout: 15 caches (12 sliding ring-buffers + 3 global) for the 35-layer model. Layers 15–34 are KV-shared and read from layers 13 (sliding) and 14 (global).
5. `export.py` traces both functions with `jax.jit(...).lower(...).compiler_ir('stablehlo')`, converts each to CoreML MIL via `stablehlo-coreml`, then merges them into a single multifunction `.mlpackage` using `MultiFunctionDescriptor` with shared (int8-quantized) weights.

**Phase 2 — Chat (`gemma-chat`, runtime):**
- `generate.py` loads the multifunction `.mlpackage` (prefill + decode functions) and runs KV-cached autoregressive inference. Chunked prefill processes the prompt in `CHUNK_SIZE=8` token chunks, then single-token decode generates new tokens with temperature + top-p sampling. Also provides a legacy full-sequence `generate()` path.
- `chat.py` parses CLI args (backend, model path, decode-only, etc.), loads models, and delegates to `tui_app.py`.
- `tui_app.py` provides the Textual-based terminal UI with streaming tokens, conversation history, token counts, and `/reset`/`/quit`/`/help` commands.
- `jax_generate.py` provides a JAX-only token stream using `decode_jax.py` (for `--backend jax`).
- `decode_jax.py` provides a pure-JAX KV-cached decode path with `prefill()` and `decode_step()` for reference and parity testing.

**Key dependency:** `stablehlo-coreml-experimental` from PyPI (see `pyproject.toml`). Gemma does not depend on a local stablehlo-coreml fork.

## Key Conventions

**Dtype handling in `model.py`:**
- `RMSNorm` and `softmax` explicitly cast to `float32` before computing to prevent fp16 underflow/overflow (e.g., `exp(30) > fp16 max`). Return values stay `float32` so JAX inserts explicit casts in StableHLO — this is intentional and required.
- `bfloat16` params are converted to `float16` before tracing because the MLIR Python bindings cannot extract raw bytes from `bfloat16 DenseFPElementsAttr`.

**Export pipeline gotchas in `export.py`:**
- Three passes are explicitly removed from the pipeline: `add_fp16_cast` (would override fp32 paths and cause NaN), `fuse_layernorm_or_instancenorm`, and `fuse_elementwise_to_batchnorm` (produce incorrect fusions for this model).
- **On-the-fly weight compression:** PyPI stablehlo-coreml does not ship a constant-lowering hook. `gemma_chat/stablehlo_streaming_patch.py` monkey-patches `StableHloConverter.op_constant` so large weights become int8 `constexpr_blockwise_shift_scale` during StableHLO→MIL, capping how much fp16 MIL is resident at once (critical on ~16 GB hosts). Gemma-specific rules (threshold 2048, fp32→fp16+cast for XLA-folded constants) stay in `export.py`.
- **MIL pass pipeline:** `gemma_chat/mil_passes/ct_convert_pipeline.py` builds `deepcopy(PassPipeline.DEFAULT)`, inserts `quantize_const_weights` at index 0, then appends: `remove_noop_slice_update`, `replace_erf_gelu`, `collapse_reshape_chains`, `remove_redundant_maximum`, `remove_broadcast_tiles`. Additionally, `replace_scalar_broadcasts` is appended to the **backend** `_BACKEND_MIL_PASSES` pipeline (which runs after the main pipeline and contains `const_elimination` passes that re-fold `fill` ops). Export then removes the three problematic passes listed above.
- The model must be saved **before** any `del`/`gc.collect()`. coremltools uses multiprocessing internally; GC can trigger `sys.exit()` on macOS/Python 3.12 before the save completes.
- Always use a **fresh** pipeline object from `build_ct_convert_pass_pipeline()` before `remove_passes()` — do not mutate stablehlo-coreml’s module-level `DEFAULT_HLO_PIPELINE` in place.
- **Multifunction export:** Both chunk_prefill and decode are exported as separate `.mlpackage` files, then merged via `MultiFunctionDescriptor` with `const_deduplication` so int8 weight blobs are shared. The default function is `"decode"`.

**CoreML model loading in `generate.py`:**
- `load_coreml_model()` defaults to `ComputeUnit.ALL` (CPU + GPU + ANE). ANE compilation is slow (~10–30 min on first load) but gives the best runtime performance. For faster compilation during development, pass `ct.ComputeUnit.CPU_ONLY`.
- Multifunction models require passing `function_name="decode"` or `function_name="prefill"` when loading.

**PLE (Per-Layer Input Embeddings):**
- PLE is always enabled (`per_layer_input_dim=256` in `config.py`). The full model with PLE is ~7.8 GB params.

**Gemma4 vs Gemma2 prompt format:**
- Gemma4 uses `<|turn>` / `<turn|>` markers, not `<start_of_turn>`. Always use `tokenizer.apply_chat_template()` — never hand-roll the prompt.

**Attention types:**
- The 35-layer model repeats the pattern `[LOCAL_SLIDING × 4, GLOBAL] × 7`.
- Global layers use `head_dim=512` and `rope_fraction=0.25` at 1M base frequency; sliding layers use `head_dim=256` and full RoPE at 10K base frequency.
- MLP width widens from layer 15 onward (`wide_mlp_from_layer=15`, `wide_hidden_dim=12288` vs `hidden_dim=6144`), independent of attention type.

**KV sharing:**
- The last 20 layers (15–34) reuse K/V from the last non-shared sliding layer (13) and global layer (14). Only 15 layers have their own KV cache slots. Controlled by `num_kv_shared_layers=20` in `Gemma4Config`.

## AI assistants (Copilot / Cursor / Claude / Gemini)

**Solo-iteration project — keep the API small:**

- **Minimal surface area** — Fewer flags, env vars, and config knobs; prefer one straightforward path.
- **Avoid compatibility / legacy toggles** unless the user explicitly requests them (no dual Python+export paths, "old package" modes, etc.).
- **Breaking changes are fine** — Rename/remove APIs; do not keep obsolete options for hypothetical callers.

Document required setup (e.g. re-export models) instead of adding switches. Root `AGENTS.md` and `.cursor/rules/minimal-api-surface.mdc` mirror this.
