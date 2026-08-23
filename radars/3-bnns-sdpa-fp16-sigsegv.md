# BNNS: SIGSEGV in BNNSGraphContextExecute_v2 executing fp16 scaled_dot_product_attention with query length ≥ 2

**Component:** BNNS (BasicNeuralNetworkSubroutines) via Core ML / Espresso `BnnsCpuInferenceOperation` — affects `.cpuOnly` and `.cpuAndNeuralEngine`
**Severity:** Hard crash (EXC_BAD_ACCESS / KERN_INVALID_ADDRESS at wild addresses)

## Environment

- Mac16,8 (M4 Pro, 48 GB), macOS 26.6.2 (25G83); also reproduced on 26.5.2 (25F84) — on 26.5.2 the same site sometimes failed "politely" with `Error(s) occurred executing a BNNS Op` instead of crashing
- Xcode 26.6, coremltools 9.0, mlprogram / iOS18 opset

## Summary

`scaled_dot_product_attention` executed by BNNS crashes with a segmentation fault when **all** of the following hold (each verified necessary by ablation):

1. **fp16** tensors (fp32 passes),
2. **query length ≥ 2** (Lq = 1 passes — single-token decode is fine, multi-token prefill crashes),
3. the SDPA is **inside a BNNS graph segment with at least one other op** (any producer or consumer op — e.g. a `mul` scaling the query; the SDPA alone as the entire program passes),
4. **large head dim** — D = 512 and 256 crash, D = 64 passes; the (Lk, D) size dependence is alignment-flavored, not monotonic: (512,8), (8,512), (64,64), (112,112) pass; (128,64), (64,128), (96,96), (512,32), (32,512), (128,128), (256,256), (512,512) crash.

`attn_mask` presence is irrelevant (crashes with and without).

Crash signature (identical across the standalone repro, 5/9/10-layer models, and the full 35-layer model):

```
EXC_BAD_ACCESS KERN_INVALID_ADDRESS (wild address, e.g. 0x5dfa98000)
libBNNS.dylib (5 unsymbolicated frames)
BNNSGraphContextExecute_v2 + 548
E5RT::Ops::BnnsCpuInferenceOperation::Impl::ExecuteSync()
-[MLE5Engine predictionFromFeatures:usingState:options:]
```

## Reproduction

Standalone ~20-line script: `/Volumes/git/ane-radar-artifacts/synth_sdpa3.py` (builds one mlprogram with a single SDPA plus a `mul` on the query, converts, predicts on CPU_ONLY):

```
python3 synth_sdpa3.py 8 128 512 512 mq        # heads=8 Lq=128 Lk=512 D=512, mul on q → exit 139 (SIGSEGV)
python3 synth_sdpa3.py 8 1   512 512 mq        # Lq=1  → passes
python3 synth_sdpa3.py 8 128 512 512 mq+f32    # fp32  → passes
python3 synth_sdpa3.py 8 128 512 512 decomp+mq # decomposed attention → passes
```

Model-scale repro: `L5.mlpackage` in the artifact folder — a 5-layer model whose `prefill_512` function (chunk of 128 query tokens) crashes on `.cpuOnly` at the layer-4 global-attention SDPA, while its `decode_512` (Lq = 1) runs correctly. Graph-cut bisection localized the crash to that single SDPA op; bypassing or decomposing only it makes the same package run.

Crash reports included: `GemmaChatCLI-2026-08-22-2253*.ips`, `GemmaChatCLI-2026-08-23-*.ips`, `python3.12-2026-08-23-*.ips`.

## Expected

The op executes (as it does at fp32, or at Lq = 1, or decomposed), or fails with a structured Core ML error.

## Actual

Process crashes inside libBNNS with a wild-address write.

## Related concern — silent wrong results (possibly the same defect pre-crash)

See report 4: with the crashing configuration removed (global SDPAs decomposed), the remaining **fused fp16 sliding-window SDPAs (D = 256, Lq = 128)** execute on BNNS without crashing but appear to produce **numerically wrong prefill results**. If the underlying defect is an out-of-bounds access, sizes that stay within mapped memory would corrupt silently rather than crash — matching what we observe.

## Workaround

Decompose SDPA into `matmul → add(mask) → softmax → matmul` for the affected sites (this repo does so for global-attention layers), or avoid BNNS execution entirely (`.cpuAndGPU`).
