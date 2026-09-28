# Gemma 4 inference using Apple CoreML

![Demo](demo.gif)

Run **Google Gemma 4 E2B** locally on **Apple Silicon** via **CoreML**.

This project re-implements the Gemma 4 transformer in **JAX/Flax**, exports it to a CoreML `.mlpackage` through **StableHLO**, and provides both a **Swift CLI chat** and an **iOS app** for interactive inference — no cloud APIs, everything runs on-device.

## Prerequisites

- **macOS on Apple Silicon** (M1 or newer)
- **Xcode 16+** with the Command Line Tools installed (provides `swift` and `xcodebuild`)
- **Python 3.12+** with [`uv`](https://github.com/astral-sh/uv) installed
- For the iOS app: a device running **iOS 18+**
- A Hugging Face account with access to [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) — accept the model license before first export, then `huggingface-cli login` (or set `HF_TOKEN`)

## Quickstart

```bash
# 1. Install Python dependencies (export only)
uv sync

# 2. Export the model to CoreML (one-time, ~10-30 min, ~8 GB disk)
uv run gemma-export

# 3. Build and run the Swift CLI chat
cd cli && swift build -c release
.build/release/GemmaChatCLI --model ../gemma4-e2b.mlpackage
```

The first CLI launch compiles `.mlpackage` → `.mlmodelc` next to the source (cached for subsequent runs).

## Performance

Measured on an M4 Pro MacBook Pro (48 GB), macOS 27.0: the full model at a 512-token context, with CPU + GPU + ANE energy taken from `powermetrics`. That energy covers those three blocks only, not the whole machine. The figures are medians after warm-up, so they leave out loading. Decode is greedy: 256 tokens generated after a 128-token prompt.

| `--compute-units` | Prefill | Prefill energy | Decode | Decode energy |
|---|---|---|---|---|
| `cpu-and-gpu` | 1,420 tok/s | 17 mJ/token | 83 tok/s | 233 mJ/token |
| `cpu-and-ne` | 2,040 tok/s | 2.6 mJ/token | 49 tok/s | 52 mJ/token |
| `all` | 1,950 tok/s | 2.7 mJ/token | 65 tok/s | 141 mJ/token |

- **The Neural Engine is the efficient choice.** It generates tokens at about 60% of the GPU's speed while using about 4.5× less energy per token. It processes the prompt faster than the GPU, at about 6× less energy.
- **`all` lands in between.** Core ML splits decode across the GPU and the Neural Engine.
- **With a 2048-token cache, decode takes 12.8 ms/token on the GPU and 23.0 ms/token on the Neural Engine.**
- **The Neural Engine path has limits.** It has only been verified on macOS 27, and it supports contexts up to 16,384 tokens.

## Running the Swift CLI chat

```bash
cd cli
swift build -c release
.build/release/GemmaChatCLI --model ../gemma4-e2b.mlpackage
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model <path>` | `./gemma4-e2b.mlpackage` | Path to a `.mlpackage` or pre-compiled `.mlmodelc` |
| `--compute-units <units>` | `cpu-and-gpu` | `cpu-and-gpu`, `cpu-and-ne` (the Neural Engine, see [Performance](#performance)), `all`, `cpu-only` |
| `--verbose` | off | Show diagnostic logs on stderr |
| `--log-file <path>` | — | Redirect diagnostic logs to a file |

### CLI chat commands

- `/reset` — clear conversation history and KV cache
- `/quit` — exit
- `/help` — list commands

## Running the iOS chat app

The iOS app lives in `ios/GemmaChat/` and uses `GemmaCore` as a local SPM dependency. The exported `gemma4-e2b.mlpackage` at the repo root is bundled into the app automatically (see `ios/GemmaChat/project.yml`).

1. Make sure `gemma4-e2b.mlpackage` exists at the repo root (run `uv run gemma-export` first if it doesn't).
2. Open `ios/GemmaChat/GemmaChat.xcodeproj` in Xcode.
3. Select a signing team under **Signing & Capabilities** (required for on-device runs).
4. Pick a physical iPhone/iPad destination and **Run**. The simulator does not have enough memory for Gemma 4 E2B.

On first build, Xcode downloads `tokenizer.json` (~31 MB) from Hugging Face via the `Download Tokenizer` build phase.

To regenerate the `.xcodeproj` after editing `project.yml`, install [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`) and run `cd ios/GemmaChat && xcodegen`.

> **Note:** the app loads a ~4 GB model into memory — we recommend a device with 8 GB+ RAM (iPhone 15 Pro or newer).

## Project structure

```
GemmaCore/      Swift Package — shared inference library (model, KV cache, tokenizer, engine)
cli/            Swift CLI chat app
ios/GemmaChat/  iOS SwiftUI chat app
gemma_chat/     Python export pipeline (JAX → StableHLO → CoreML)
tests/          Python tests for MIL passes, stateful KV export, and multifunction export
benchmarks/     Standalone Swift benchmark for model loading / first prediction
```

## Troubleshooting

- **`Error: model not found`** — pass `--model <path>`, or run from the repo root where `gemma4-e2b.mlpackage` lives.
- **Incoherent replies from an older export** — re-run `uv run gemma-export`. The fixes to the attention mask and the sliding-window cache are in the export, not the runtime.
- **Loading a `.mlmodelc` directly** — copy the package's `Embeddings/` directory into it (the runtime reads the embedding tables from there), and `Tokenizer/` too unless you want the tokenizer downloaded from Hugging Face.
- **The first `cpu-and-ne` / `all` launch is slow.** Core ML compiles the model for the Neural Engine once and caches the result. That takes about a minute at `--max-context 1024`, and longer for larger contexts.

## License

This code is released under the [MIT License](LICENSE).

The **Gemma model weights** are subject to [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms). You must accept the model license on the Hugging Face Hub before downloading weights.
