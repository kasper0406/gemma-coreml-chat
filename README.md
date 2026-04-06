# Gemma4 inference using Apple CoreML

![Demo](demo.gif)

Run **Google Gemma 4 E2B** locally on **Apple Silicon** via **CoreML**.

This project re-implements the Gemma 4 transformer in **JAX/Flax**, exports it to a CoreML `.mlpackage` through **StableHLO**, and provides a **terminal chat UI** for interactive inference — no cloud APIs, everything runs on-device.

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Export the model to CoreML (one-time, ~10-30 min)
uv run gemma-export-decode

# 3. Chat!
uv run gemma-chat
```

> **Prerequisites:** macOS on Apple Silicon and access to `google/gemma-4-E2B-it` on [Hugging Face](https://huggingface.co/google/gemma-4-E2B-it) (accept the model license first).

## How it works

**Step 1 — Export** (`gemma-export-decode`, run once):

`weight_mapper.py` downloads the HF checkpoint, `model.py` defines the full transformer in JAX/Flax, and `export_decode.py` traces it via `jax.jit` → StableHLO → CoreML MIL, producing a single multifunction `.mlpackage` with both **chunked prefill** and **KV-cached decode** functions.

**Step 2 — Chat** (`gemma-chat`):

Loads the `.mlpackage`, runs autoregressive inference with KV caching, and provides a Textual-based terminal UI with streaming tokens, conversation history, and token counts.

## Usage details

**Export options:**

```bash
uv run gemma-export-decode                    # int8 weights → gemma4-e2b.mlpackage
```

**Chat options:**

```bash
uv run gemma-chat                             # uses gemma4-e2b.mlpackage (default)
uv run gemma-chat --model path/to/other.mlpackage
uv run gemma-chat --backend jax               # use JAX/Flax weights directly (for comparison)
```

In the TUI: `/quit` or `/exit` to leave, `/reset` to clear history, `/help` for commands.

**Compute units:**

By default, the CoreML model uses all available compute units (CPU, GPU, and ANE). This provides the best runtime performance but compilation is significantly slower (~10–30 min on first load). To speed up compilation, you can modify `load_coreml_model()` in [gemma_chat/generate.py](gemma_chat/generate.py#L138) to use `ct.ComputeUnit.CPU_ONLY`, which compiles in seconds but runtime will be slower in production.

**Diagnostics:**

```bash
# A/B: full model vs KV decode (greedy)
uv run gemma-compare-inference --prompt "Hi"

# JAX vs CoreML logit parity after prefill
uv run gemma-parity-decode --max-tokens 8
```

## License

This code is released under the [MIT License](LICENSE).

The **Gemma model weights** are subject to [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms). You must accept the model license on the Hugging Face Hub before downloading weights.
