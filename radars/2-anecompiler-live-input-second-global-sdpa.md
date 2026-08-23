# ANECompiler: "live input tensor not used in network" — ANECCompile fails when a transformer has ≥ 2 large-head SDPA ops deep in the graph

**Component:** ANECompiler / MILCompilerForANE (via Core ML `.cpuAndNeuralEngine`)
**Severity:** Compile failure at 10–15 layers; silent all-CPU fallback with **no error at all** at 35 layers

## Environment

- Mac16,8 (M4 Pro, 48 GB), macOS 26.6.2 (25G83); also reproduced on 26.5.2 (25F84)
- Xcode 26.6, coremltools 9.0, mlprogram / iOS18 opset, stateful KV-cache model (fp16 states), weight blobs sharded < 2 GiB (see report 1 — sharding is a prerequisite, otherwise report 1's bug masks this one)

## Summary

A Gemma-architecture transformer export (repeating blocks of sliding-window attention with fused `scaled_dot_product_attention`, plus periodic global-attention blocks) fails ANE compilation once the graph contains **two or more global-attention SDPA ops** (head dim 512). The unified log shows, after the first N procedures compile cleanly:

```
(ANECompiler) Error: live input tensor <private> not used in network
(ANECompiler) Function call to BuildLayerGraph() failed in ZinCompilerCoreClassic.cpp:302
... MILCompilerForANE error: ... Error=_ANECompiler : ANECCompile() FAILED
```

At the full 35-layer scale, `ANECompilerService` dies mid-compile instead (three PID restarts in the log, no "End of compilation"), and Core ML then reports the model as all-CPU **with no error surfaced to the client at all**.

## Bisection evidence (minimal pair in artifacts)

| model | layers | global SDPAs | `MLComputePlan.load` `.cpuAndNeuralEngine` |
|---|---|---|---|
| `L9.mlpackage` | 9 | 1 | OK — 19/19 ANE procedures, 795/1228 ops ANE-supported |
| `L10.mlpackage` | 10 | 2 | **FAIL** at procedure 11/11 with the log above |

Ablations on the failing L10 program (rewriting the shipped MIL) isolate the trigger to *the deepest global SDPA specifically* — not any resource budget:

- Remove the state-write wrapper (`fill_like`+`add`) → still fails
- Drop the global cache writes entirely → still fails
- Reduce state-feature count 20 → 18 (graph shape preserved) → still fails
- Remove 4 sliding SDPAs (−96 ops) → still fails
- Bypass the **first** global SDPA (layer 4) → still fails
- Bypass the **second** global SDPA (layer 9) only → **passes**, 20/20 procedures
- **Decompose** that same SDPA into `matmul → add(mask) → softmax → matmul` → **passes**, 21/21 procedures (`L10decG.mlpackage`)

So an equivalent computation expressed without the fused SDPA op compiles fine; the fused op past a depth threshold produces a partitioner procedure whose declared input has no consumer.

## Expected

Either the SDPA compiles (it does at depth ≤ 9), or a diagnosable error naming the offending tensor is surfaced through the Core ML API. The 35-layer behavior — compiler service crash + silent CPU fallback with zero client-visible error — makes the failure undiagnosable without unified-log spelunking.

## Artifacts

`/Volumes/git/ane-radar-artifacts/`: `L9.mlpackage` (passes) vs `L10.mlpackage` (fails) — minimal pair differing only in layer count; `L10decG.mlpackage` (identical to L10 with the two global SDPAs decomposed — passes); `*.anelog` (captured ANECompilerService unified-log output), `*.planlog`/`*.jsonl` (MLComputePlan results per function). Note: the offending tensor name is `<private>` in the logs; we could not enable `log config --mode private_data:on` to capture it.

## Workaround

Export the global-attention sites decomposed (`matmul → add(mask) → softmax → matmul`) instead of fused SDPA. Measured cost on the GPU path: none.
