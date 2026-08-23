# Core ML: a single weight blob file ≥ 2 GiB silently disables the ANE for the entire model

**Component:** Core ML / MLComputePlan / ANE eligibility
**Severity:** Silent performance degradation (no error, no log)

## Environment

- Mac16,8 (MacBook Pro, M4 Pro, 48 GB), macOS 26.6.2 (25G83); also reproduced on 26.5.2 (25F84)
- Xcode 26.6, coremltools 9.0, mlprogram / iOS18 opset

## Summary

If an `.mlpackage`'s weight blob file (`Data/com.apple.CoreML/weights/weight.bin`) reaches 2³¹ bytes, **every op in the model becomes ANE-ineligible** — `MLComputePlan` with `.cpuAndNeuralEngine` reports the ANE in the *supported* device set for exactly zero ops, including trivially ANE-capable ops (`matmul`, `mul`, `reshape`). No error or diagnostic is produced anywhere; the model silently runs on CPU. The same weights split across multiple blob files < 2 GiB each are fully ANE-eligible.

The cliff is the **size of a single blob file**, not total model size.

## Reproduction

Synthetic generator: `synth.py` / `synth_shard.py` (in the artifact folder). Each builds an mlprogram of N identical blocks — `constexpr_blockwise_shift_scale` (int4, per-channel scales, `[8192, 8192]`) → `matmul` → `mul` — sized to hit a target blob size, then loads `MLComputePlan` with `.cpuAndNeuralEngine` and counts ops whose supported-device set contains the ANE.

Measured sweep (ANE-supported ops / total):

| weight.bin size | result |
|---|---|
| 0.54 GB | 32/32 supported, ANE-scheduled |
| 1.61 GB | 96/96 |
| **2.15 GB** | **0/128** |
| 2.50 GB | 0/160 |
| 3.22 GB | 0/192 |
| 3.22 GB, sharded into 4 files ≤ 1.07 GB | **192/192** |

## Expected

Either the model remains ANE-eligible regardless of blob file size, or Core ML reports a diagnosable error/log message when blob size disqualifies a model from a requested compute unit.

## Actual

Total, silent ANE ineligibility of the whole program at exactly 2³¹ bytes in one blob file.

## Impact

Any quantized LLM-class model above ~2 GB (e.g. a 2B-parameter int4 model with fp16 embeddings) is silently locked out of the ANE unless the author discovers this limit and shards blobs manually. coremltools always writes a single blob, so every such model produced by the standard toolchain is affected.

## Workaround

Override `MILProtoExporter.get_weight_path` to roll over to `weight_1.bin`, `weight_2.bin`, … before 2 GiB is reached (implemented in this repo as `gemma_chat/weight_shards.py`). Confirmed to restore full ANE eligibility with byte-identical weights.

## Artifacts

`/Volumes/git/ane-radar-artifacts/`: `synth.py` (unsharded generator), `synth_shard.py` (sharded generator). Both are self-contained (~50 lines, coremltools only) and print the eligibility counts.
