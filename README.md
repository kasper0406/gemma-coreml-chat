# Gemma4 inference using Apple CoreML

![Demo](demo.gif)

Run **Google Gemma 4 E2B** locally on **Apple Silicon** via **CoreML**.

This project re-implements the Gemma 4 transformer in **JAX/Flax**, exports it to a CoreML `.mlpackage` through **StableHLO**, and provides both a **Swift CLI chat** and an **iOS app** for interactive inference — no cloud APIs, everything runs on-device.

## Quickstart

```bash
# 1. Install Python dependencies (export only)
uv sync

# 2. Export the model to CoreML (one-time, ~10-30 min)
uv run gemma-export

# 3. Build and run the Swift CLI chat
cd cli && swift build -c release
.build/release/GemmaChatCLI --model ../gemma4-e2b.mlpackage
```

> **Prerequisites:** macOS on Apple Silicon and access to `google/gemma-4-E2B-it` on [Hugging Face](https://huggingface.co/google/gemma-4-E2B-it) (accept the model license first).

## Architecture

### Phase 1 — Export (Python, run once)

`uv run gemma-export` downloads HF weights, defines the full transformer in JAX/Flax, and traces it via `jax.jit` → StableHLO → CoreML MIL, producing a single multifunction `.mlpackage` with both **chunked prefill** and **KV-cached decode** functions.

### Phase 2 — Inference (Swift)

All inference runs through native Swift for ~20x faster model loading vs Python coremltools:

- **GemmaCore/** — Shared Swift Package (SPM library) with model loading, KV cache management, tokenization, and the inference engine
- **cli/** — Swift CLI chat app using GemmaCore
- **ios/GemmaChat/** — iOS SwiftUI chat app using GemmaCore

## Usage

### Export options

```bash
uv run gemma-export                            # multifunction .mlpackage with shared int8 weights
uv run gemma-export --decode-only              # decode only (no prefill)
uv run gemma-export --separate                 # separate decode + prefill .mlpackage files
```

### Swift CLI chat

```bash
cd cli && swift build -c release
.build/release/GemmaChatCLI --model ../gemma4-e2b.mlpackage
```

In the CLI: `/quit` to exit, `/reset` to clear history, `/help` for commands. The tokenizer is auto-downloaded from HuggingFace on first use.

### iOS app

The iOS app lives in `ios/GemmaChat/` and uses the same `GemmaCore` Swift package. The Xcode build phase automatically downloads `tokenizer.json` (~31 MB) from HuggingFace on first build.

### Diagnostics

```bash
# A/B: full model vs KV decode (greedy)
uv run gemma-compare-inference --prompt "Hi"

# JAX vs CoreML logit parity after prefill
uv run gemma-parity-decode --max-tokens 8
```

## Project structure

```
GemmaCore/          Swift Package — shared inference library (model loading, KV cache, tokenizer, engine)
cli/                Swift CLI chat app (readline-based, streaming output)
ios/GemmaChat/      iOS SwiftUI chat app
gemma_chat/         Python export pipeline (JAX → StableHLO → CoreML)
benchmarks/         Swift model loading benchmarks
```

## License

This code is released under the [MIT License](LICENSE).

The **Gemma model weights** are subject to [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms). You must accept the model license on the Hugging Face Hub before downloading weights.
