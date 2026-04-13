# Dynamic Context Window — Implementation Plan

## Problem

Global KV caches (3 out of 15) are fixed at MAX_SEQ_LEN=4096, wasting memory at
short contexts. We want CoreML models with RangeDim on these caches so they can
be sized dynamically at runtime.

## Architecture Recap

- 15 KV cache slots: 12 sliding ring-buffers (fixed at 512) + 3 global (layers 4, 9, 14)
- Flat KV array: 30 entries. Global cache indices: 8,9 / 18,19 / 28,29
- Layers 15–34 are KV-shared (read from layers 13 sliding, 14 global)

## Key Findings

### stablehlo-coreml `symbolic-shapes` branch (commit 83de18f)

You can find and modify the code at `/Volumes/git/stablehlo-coreml` in case you need to make changes. Notice that in order for changes to go in this repository, they have to be general in nature.

Already has full converter support for dynamic shapes:

| Op | Handler | Status |
|---|---|---|
| `DynamicBroadcastInDimOp` | `op_dynamic_broadcast_in_dim` — uses `expand_dims` (no reshape with symbolic dims) | ✅ |
| `DynamicIotaOp` | `op_dynamic_iota` — `range_1d(0, N, 1)` for 1D case | ✅ |
| `GetDimensionSizeOp` | `op_get_dimension_size` — `mb.shape` + `slice_by_index` | ✅ |
| `CustomCallOp (shape_assertion)` | Skipped (constraints checked at export time) | ✅ |
| `dot_general` with dynamic result | `_dot_general_dynamic` fast-path via direct matmul | ✅ |
| `build_func` | Replaces `?` dims with `Symbol(f'dim_{N}')` | ✅ |

Tests passing: symbolic add/mul/broadcast, matmul (lhs/rhs dynamic), batched einsum,
dynamic iota (arange), reductions, scaled dot-product attention, multi-dim.

### Three export paths evaluated

| Path | OOM? | Phantom arg? | Notes |
|---|---|---|---|
| `jax.export.export()` → `mlir_module()` text | **YES** (~30GB+ text for 8GB model) | No | Works for small models only |
| `jax.export.export()` → `mlir_module_serialized` bytes | No (compact VHLO bytecode) | No | Produces VHLO ops, needs dialect conversion |
| `jax.jit(fn).trace(*specs).lower().compiler_ir('stablehlo')` | **No** (returns ir.Module directly) | Yes: `%arg0: tensor<i32> {jax.global_constant = "N"}` | **Best option** |

### The phantom `%arg0` is NOT a problem

- It's just an int32 scalar input representing the dynamic dimension N
- The converter creates a regular `mb.placeholder(shape=[1], dtype=int32)` for it
- `%arg0` IS used in the function body (e.g., `stablehlo.minimum %arg0, %c_0` for computing gather slice sizes)
- At runtime, pass the current cache length as an int32 input
- All other inputs shift by 1 index — manageable in export code

### Spike test reference

`spike_flexible/test_symbolic_e2e.py` demonstrates the full E2E flow for a small model:
- JAX symbolic → stablehlo-coreml → CoreML with RangeDim → runtime at N=1,4,8,16,32,64
- Uses `export()` path (fine for small models)
- Conversion succeeds, but runtime fails with `select` op shape validation error
- This error may have been fixed by subsequent commits (4a55d6d) — needs re-testing

## Approach: `trace()` path with symbolic shapes

```python
from jax import export as jax_export

(N,) = jax_export.symbolic_shape("N")

# Global cache inputs use symbolic N
global_kv_shape = jax.ShapeDtypeStruct((1, N, nkv, hd), jnp.float16)

# Sliding cache inputs stay fixed
sliding_kv_shape = jax.ShapeDtypeStruct((1, 512, nkv, hd), jnp.float16)

# Trace with symbolic specs
traced = jax.jit(decode_fn).trace(token_id_spec, position_spec, *kv_specs, ring_spec)
hlo_module = traced.lower().compiler_ir('stablehlo')
# hlo_module is ir.Module with tensor<1x?xNxHD> for global caches
# plus %arg0: tensor<i32> {jax.global_constant = "N"}

# Convert — stablehlo-coreml handles all dynamic ops
mil_prog = stablehlo_coreml.convert(hlo_module, ...)

# ct.convert with RangeDim for global cache inputs
ct_inputs = [
    ct.TensorType(name="N", shape=(1,)),  # phantom dim input
    ct.TensorType(name="token_id", shape=(1,)),
    ct.TensorType(name="position", shape=(1,)),
    # ... sliding KV: fixed shapes
    # ... global KV: ct.RangeDim(1, max_seq_len)
    ct.TensorType(name="k_4", shape=ct.Shape(
        shape=(1, ct.RangeDim(1, max_seq_len, default=128), nkv, hd_global)
    )),
    # ...
]
```

## Implementation Todos

### 1. test-select-dynamic
**Where**: `stablehlo-coreml/tests/test_symbolic_shapes.py`

Add a test for `jnp.where(mask, w, fill)` with dynamic broadcast — the exact pattern
Gemma's attention masking uses:
```python
def masked_attention(q, k_cache, position):
    w = jnp.matmul(q, jnp.swapaxes(k_cache, -2, -1))  # (1, 1, N)
    mask = jnp.arange(k_cache.shape[1]) <= position      # (N,) — dynamic_iota
    return jnp.where(mask, w, -1e4)                       # select with broadcast
```

If CoreML runtime rejects this (like the spike test saw), fix options:
- Fix `op_select` in converter to explicitly broadcast inputs
- Or restructure Gemma code to use additive masking: `w + mask_additive`

### 2. export-symbolic-trace
**Where**: `gemma_chat/export.py`

Changes to `export_decode_step` and `export_chunk_prefill`:
1. `(N,) = jax_export.symbolic_shape("N")`
2. Build `ShapeDtypeStruct((1, N, nkv, hd), fp16)` for 6 global cache inputs
3. Use `trace(*specs).lower().compiler_ir('stablehlo')` instead of current `lower()`
4. Account for phantom `%arg0` — prepend "N" to input_names list
5. Build `ct_inputs` list with `ct.RangeDim(1, max_seq_len)` for global cache dims
6. Pass `ct_inputs` to `_hlo_to_mlpackage`

### 3. generate-dynamic-runtime
**Where**: `gemma_chat/generate.py`

1. Pass dim N (current global cache length) as int32 input to CoreML predictions
2. Use existing `_flexible_global_names()` and `_ensure_global_cache_capacity()` helpers
3. Start with small global caches (e.g., 128) and grow as needed

### 4. verify-export
Full export and verification:
- Model converts without errors
- RangeDim appears in CoreML spec for global cache inputs
- Optimization counts: 35 SDPA, 70 GELU, ~558 int4 quant
- Parity test at multiple cache sizes (16, 128, 512, 1024)

### 5. ios-dynamic
**Where**: `ios/GemmaChat/`

Update Swift iOS app:
- Pass dynamic-sized global caches to CoreML model
- Update KV state management for variable-length global arrays
- Pass dim N as int32 input

## Risks & Mitigations

### 1. `select` + dynamic broadcast (HIGH)
`jnp.where(mask, w, fill)` where mask shape derives from `dynamic_iota` might fail
CoreML's runtime shape validator ("Incompatible Dimension"). The spike test hit this.

**Mitigation**: Test first in stablehlo-coreml. If it fails:
- Option A: Fix converter's `op_select` to explicitly broadcast all inputs to same shape
- Option B: Switch Gemma to additive masking: `w = w + mask_additive` (0 for valid, -1e4 for invalid)

### 2. SDPA fusion with symbolic shapes (MEDIUM)
The SDPA MIL pass pattern-matches Q@K^T → scale → mask → softmax → @V. With dynamic dims,
the intermediate shapes are symbolic. The pattern matcher might not handle symbolic shapes.

**Mitigation**: Check if SDPA fusion count drops. If so, update the pass to handle symbolic dims.

### 3. `dynamic_gather` (LOW)
If `trace()` produces `dynamic_gather` for cache operations, the converter would need a new handler.
Not expected for Gemma's patterns (we access full caches, not symbolic slices).

## File Reference

| File | Role |
|---|---|
| `gemma_chat/export.py` | Export pipeline — main changes here |
| `gemma_chat/generate.py` | Runtime inference — dynamic cache sizing |
| `gemma_chat/decode_coreml.py` | JAX-traceable functions — already shape-polymorphic |
| `gemma_chat/cache_spec.py` | KV cache layout (12 sliding + 3 global) |
| `gemma_chat/config.py` | MAX_SEQ_LEN=4096, CHUNK_SIZE=8 |
| `/Volumes/git/stablehlo-coreml/stablehlo_coreml/converter.py` | Dynamic op handlers |
| `spike_flexible/test_symbolic_e2e.py` | E2E reference (small model) |
| `stablehlo-coreml/tests/test_symbolic_shapes.py` | Existing symbolic shape tests |
