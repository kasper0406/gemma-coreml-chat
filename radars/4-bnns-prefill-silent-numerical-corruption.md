# BNNS: silent numerically-wrong prefill results for fp16 transformer graphs (`.cpuOnly` / `.cpuAndNeuralEngine`), same model correct on `.cpuAndGPU`

**Component:** BNNS via Core ML / Espresso `BnnsCpuInferenceOperation`
**Severity:** Silent incorrect results — the worst failure mode: no crash, no error, plausible-looking but wrong output

## Environment

- Mac16,8 (M4 Pro, 48 GB), macOS 26.6.2 (25G83)
- Xcode 26.6, coremltools 9.0, mlprogram / iOS18 opset

## Summary

A 35-layer Gemma-architecture fp16 model (stateful KV caches, chunk-128 prefill, global-attention sites decomposed to avoid the crash in report 3) produces **correct conversational output on `.cpuAndGPU`** and **incoherent, prompt-oblivious output on `.cpuOnly` and `.cpuAndNeuralEngine`** — the generated text reads fluently but shows the model never received the prompt content (e.g. told "My name is Kasper and I live in London", the reply discusses receiving "some text"; asked to recall, it cannot).

Decode-path numerics are verified **bit-identical** between backends at the single-token level, and the crash-class ops (report 3) were removed — pointing at the **prefill path** (query length 128), most plausibly the fused fp16 sliding-window SDPA kernels (head dim 256, Lq 128), which sit adjacent to the crashing size classes of report 3. If report 3's root cause is an out-of-bounds access, in-bounds-but-wrong variants of the same defect would corrupt silently — matching this behavior exactly.

## Reproduction

1. Export the model at this repo's branch `kn/metal-performance` (commit 4d27279): `uv run gemma-export` → `gemma4-e2b.mlpackage`.
2. Run the identical scripted conversation twice:

```
printf 'Hi! My name is Kasper and I live in London. Please remember this.\nWhat is my name and where do I live?\n/quit\n' \
  | cli/.build/release/GemmaChatCLI --model gemma4-e2b.mlpackage --compute-units cpu-and-gpu   # correct: recalls Kasper/London
printf '...same...' \
  | cli/.build/release/GemmaChatCLI --model gemma4-e2b.mlpackage --compute-units cpu-only      # wrong: never registers the prompt
```

Observed `.cpuOnly` replies to the script above (verbatim): "Hello! I see you're sending some text to me, perhaps you have a question…", then "thought", then an identity ramble with no recall — versus perfect recall on `.cpuAndGPU` in the same binary with the same package.

A smaller deterministic repro should be extractable by comparing a single `prefill_512` call's logits between `.cpuOnly` and `.cpuAndGPU` on the 5-layer mini (`/Volumes/git/ane-radar-artifacts/L5decomposed` equivalent — see repo `tests/` for the coremltools prediction scaffolding); we have not yet reduced it below the model level.

## Expected

Numerically equivalent (within fp16 tolerance) results across compute-unit configurations.

## Actual

Prefill on the BNNS path yields corrupted prompt encodings; generation proceeds from garbage state with no error.

## Impact

`.cpuOnly` is unusable for this model class, and — more importantly — `.cpuAndNeuralEngine` is also affected because its CPU segments run on BNNS, blocking correct ANE deployment even where ANE compilation succeeds.

## Notes for triage

- Related crash with hard evidence and a 20-line standalone repro: report 3 (fp16 SDPA, Lq ≥ 2, size-dependent). This report is likely the same defect at sizes that stay within mapped memory.
- On macOS 26.5.2 the same model class crashed outright in prefill (report 3's signature), so this silent variant became reachable only after removing the crashing sites.
