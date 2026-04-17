# Benchmarks

Measures CoreML inference performance for the Gemma4-E2B model on Apple Silicon.

Unlike Python's coremltools — which recompiles on every `MLModel` load — the
benchmark runner drives a **Swift** executable (`benchmarks/swift/bench/`) that
links the shared `GemmaCore` library, reuses the same inference path as the
CLI and iOS app, and caches compiled `.mlmodelc` artifacts between runs. The
Python layer only orchestrates: it loops over the (backend × context × run)
matrix, spawns the Swift binary per measurement, and optionally records a
`powermetrics` trace alongside each run.

## What we benchmark

- **Prefill latency** — wall time from `engine.generate` start until the first
  yielded token (i.e. the sample drawn from the final prefill chunk's logits),
  divided by the synthetic prompt length to get tokens/sec.
- **Decode throughput** — tokens/sec measured over the time between the first
  and last yielded tokens, which brackets pure decode-model predictions.
- **Power consumption** — CPU, GPU, and ANE milliwatts via `powermetrics`
  (requires passwordless `sudo`, else silently skipped).

## Usage

```bash
# Full benchmark suite (builds the Swift binary on first invocation)
uv run gemma-bench

# Specific backends and context lengths
uv run gemma-bench --backends cpu,ane --context-lengths 128,512,2048

# Skip power monitoring
uv run gemma-bench --no-power

# More decode tokens per run, fewer repetitions
uv run gemma-bench --decode-tokens 128 --runs 3
```

Results are saved as JSON to `benchmarks/results/` and plots are generated
automatically.

## Prerequisites

- **Xcode** (not just the Command Line Tools). The Swift bench targets
  macOS 15, which needs the Xcode SDK. Select it before the first run:

  ```bash
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  ```

- **Passwordless `powermetrics`** (optional, for power data). Add to
  `/etc/sudoers`:

  ```
  %admin ALL = (root) NOPASSWD: /usr/bin/powermetrics
  ```

  Without this, power monitoring is silently skipped.

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemma4-e2b.mlpackage` | Path to the exported `.mlpackage` |
| `--backends` | `cpu,gpu,ane,all` | Comma-separated compute backends |
| `--context-lengths` | 128–16384 | Comma-separated prompt lengths |
| `--runs` | `3` | Repetitions per configuration |
| `--decode-tokens` | `32` | Tokens to decode per run |
| `--timeout` | `300` | Per-run timeout in seconds |
| `--output-dir` | `benchmarks/results` | Where to write JSON + plots |
| `--no-power` | off | Disable power monitoring |
| `--no-warmup` | off | Skip the primer prefill+decode |

## Standalone Swift bench (legacy)

`benchmarks/swift/model_bench.swift` is a zero-dependency sanity check that
loads a `.mlpackage`, compiles, and runs a short prefill+decode without going
through `GemmaCore`. Build and run directly with `swiftc`; see the header in
that file.

## Multifunction + RangeDim diagnostic

`benchmarks/multifunction_rangedim_bug.{py,swift}` are the minimum repro for
the E5RT multifunction/RangeDim loading bug that prompted the
`remove_broadcast_tiles` MIL pass fix. Kept for regression-watching.
