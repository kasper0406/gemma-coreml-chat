#!/usr/bin/env python3
"""Incremental multifunction + dynamic dim test.

Builds up MIL programs step-by-step to find which op pattern causes
E5RT failures in multifunction mode with flexible (symbolic) dimensions.

Approach: Use Symbol objects in MIL builder TensorSpec to create programs
with true symbolic shapes (matching what stablehlo-coreml produces), then
fix the shapeRange bounds after ct.convert().

Usage:
    .venv/bin/python tests/test_multifunction_dynamic.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
import traceback
import warnings

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types, get_new_symbol


# ── Config ─────────────────────────────────────────────────────────────────

# Symbolic dim for dynamic cache length
N_SYM = get_new_symbol("N")

# Flex bounds: (dim_index, lower_bound, upper_bound)
FLEX_LO = 1
FLEX_HI = 4096

# Runtime test size (within bounds)
TEST_N = 32


# ── Helpers ────────────────────────────────────────────────────────────────

def _fix_flex_bounds(model, flex_inputs: dict[str, tuple[int, int, int]]):
    """Fix shapeRange bounds on inputs after ct.convert().

    ct.convert() sets default bounds of [1, 2] for symbolic dims.
    We need to set proper bounds for our use case.
    """
    spec = model._spec

    for inp in spec.description.input:
        if inp.name not in flex_inputs:
            continue
        dim_idx, lo, hi = flex_inputs[inp.name]
        arr = inp.type.multiArrayType
        shape = list(arr.shape)
        if len(shape) > dim_idx:
            shape[dim_idx] = lo
            arr.ClearField("shape")
            for s in shape:
                arr.shape.append(s)
        arr.ClearField("shapeRange")
        for i, dim_val in enumerate(arr.shape):
            sr = arr.shapeRange.sizeRanges.add()
            if i == dim_idx:
                sr.lowerBound = lo
                sr.upperBound = hi
            else:
                sr.lowerBound = dim_val
                sr.upperBound = dim_val


def _save_and_test_multifunction(
    build_fn_a,
    build_fn_b,
    test_name: str,
    flex_inputs: dict[str, tuple[int, int, int]] | None = None,
    test_inputs_a: dict | None = None,
) -> tuple[bool, list[str], bool]:
    """Build, convert, merge as multifunction, and test loading + predict."""
    tmp = tempfile.mkdtemp(prefix=f"mf-{test_name}-")
    try:
        path_a = os.path.join(tmp, "a.mlpackage")
        path_b = os.path.join(tmp, "b.mlpackage")
        mf_path = os.path.join(tmp, "mf.mlpackage")

        for build_fn, path in [(build_fn_a, path_a), (build_fn_b, path_b)]:
            prog = build_fn()
            model = ct.convert(
                prog,
                source="milinternal",
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                skip_model_load=True,
            )
            if flex_inputs:
                _fix_flex_bounds(model, flex_inputs)
            model.save(path)
            del model, prog

        desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(path_a, "main", "func_a")
        desc.add_function(path_b, "main", "func_b")
        desc.default_function_name = "func_a"
        ct.utils.save_multifunction(desc, mf_path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = ct.models.MLModel(mf_path, function_name="func_a")

        ok = loaded.__proxy__ is not None
        e5rt_msgs = [str(x.message) for x in w
                     if "E5RT" in str(x.message) or
                     "will not be able to run predict" in str(x.message)]

        predict_ok = False
        if ok and test_inputs_a is not None:
            try:
                result = loaded.predict(test_inputs_a)
                predict_ok = result is not None
            except Exception as e:
                e5rt_msgs.append(f"predict error: {e}")

        return ok, e5rt_msgs, predict_ok
    except Exception as e:
        traceback.print_exc()
        return False, [f"exception: {e}"], False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_test(test_name, build_a, build_b,
              flex_inputs=None, test_inputs_a=None):
    """Run a single test and print results."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    ok, e5rt, predict_ok = _save_and_test_multifunction(
        build_a, build_b, test_name, flex_inputs, test_inputs_a
    )
    status = "✓ PASS" if ok else "✗ FAIL"
    predict_status = ""
    if test_inputs_a is not None:
        predict_status = f"  predict={'✓' if predict_ok else '✗'}"
    print(f"  {status}{predict_status}")
    if e5rt:
        for msg in e5rt:
            print(f"  E5RT: {msg[:300]}")
    return ok and (predict_ok if test_inputs_a else True)


# ── Tests ──────────────────────────────────────────────────────────────────


def test_01_baseline_concrete():
    """Two identical functions, all concrete shapes. Sanity check."""
    def build():
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 64, 1, 4), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(cache):
            return mb.add(x=cache, y=np.float16(0.0), name="cache_out")
        return prog

    return _run_test("01_baseline_concrete", build, build,
                     test_inputs_a={"cache": np.zeros((1, 64, 1, 4), dtype=np.float16)})


def test_02_identity_flex():
    """Simplest flex: symbolic dim, identity-like passthrough."""
    def build():
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(cache):
            return mb.add(x=cache, y=np.float16(0.0), name="cache_out")
        return prog

    return _run_test("02_identity_flex", build, build,
                     flex_inputs={"cache": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={"cache": np.zeros((1, TEST_N, 1, 4), dtype=np.float16)})


def test_03_slice_update_flex():
    """KV cache write: slice_update on symbolic dim."""
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 1, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1,), dtype=types.int32),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(cache, new_val, position):
            pos = mb.gather(x=position, indices=0)
            begin = mb.concat(values=[
                np.array([0], dtype=np.int32),
                mb.expand_dims(x=pos, axes=[0]),
                np.array([0, 0], dtype=np.int32),
            ], axis=0)
            updated = mb.slice_update(
                x=cache, update=new_val,
                begin=begin,
                end=[1, 0, 1, 4],
                begin_mask=[True, False, True, True],
                end_mask=[True, True, True, True],
                name="cache_out",
            )
            return updated
        return prog

    return _run_test("03_slice_update_flex", build, build,
                     flex_inputs={"cache": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "cache": np.zeros((1, TEST_N, 1, 4), dtype=np.float16),
                         "new_val": np.ones((1, 1, 1, 4), dtype=np.float16),
                         "position": np.array([0], dtype=np.int32),
                     })


def test_04_transpose_flex():
    """Transpose on symbolic dim (SDPA K/V layout)."""
    def build():
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, N_SYM, 8, 4), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(kv):
            t = mb.transpose(x=kv, perm=[0, 2, 1, 3])  # (1, 8, N, 4)
            return mb.add(x=t, y=np.float16(0.0), name="out")
        return prog

    return _run_test("04_transpose_flex", build, build,
                     flex_inputs={"kv": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={"kv": np.zeros((1, TEST_N, 8, 4), dtype=np.float16)})


def test_05_matmul_flex():
    """Q @ K^T where K has symbolic seq len."""
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 8, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 8, N_SYM, 4), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k):
            kt = mb.transpose(x=k, perm=[0, 1, 3, 2])  # (1,8,4,N)
            return mb.matmul(x=q, y=kt, name="scores")  # (1,8,1,N)
        return prog

    return _run_test("05_matmul_flex", build, build,
                     flex_inputs={"k": (2, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "q": np.random.randn(1, 8, 1, 4).astype(np.float16),
                         "k": np.random.randn(1, 8, TEST_N, 4).astype(np.float16),
                     })


def test_06_sdpa_flex():
    """Scaled dot-product attention with symbolic KV length."""
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 8, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 8, N_SYM, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 8, N_SYM, 4), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v):
            return mb.scaled_dot_product_attention(
                query=q, key=k, value=v, name="out")
        return prog

    return _run_test("06_sdpa_flex", build, build,
                     flex_inputs={"k": (2, FLEX_LO, FLEX_HI),
                                  "v": (2, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "q": np.random.randn(1, 8, 1, 4).astype(np.float16),
                         "k": np.random.randn(1, 8, TEST_N, 4).astype(np.float16),
                         "v": np.random.randn(1, 8, TEST_N, 4).astype(np.float16),
                     })


def test_07_select_flex():
    """select/where with symbolic dim — the E5RT error mentions 'select'."""
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, N_SYM, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, N_SYM, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, N_SYM), dtype=types.bool),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(a, b, cond):
            cond_3d = mb.expand_dims(x=cond, axes=[2])
            return mb.select(cond=cond_3d, a=a, b=b, name="out")
        return prog

    return _run_test("07_select_flex", build, build,
                     flex_inputs={"a": (1, FLEX_LO, FLEX_HI),
                                  "b": (1, FLEX_LO, FLEX_HI),
                                  "cond": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "a": np.ones((1, TEST_N, 4), dtype=np.float16),
                         "b": np.zeros((1, TEST_N, 4), dtype=np.float16),
                         "cond": np.ones((1, TEST_N), dtype=bool),
                     })


def test_08_select_mixed_flex():
    """select where 'a' comes from tile (concrete) and 'b' is flex input.

    This is the minimal repro from the debug guide.
    """
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, N_SYM, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 1, 4), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(cache, fill_val):
            # Tile fill_val to match cache shape — produces concrete-shaped output
            tiled = mb.tile(x=fill_val, reps=[1, 512, 1])  # (1, 512, 4)
            # But cache has symbolic dim — select has mixed shapes
            # Need a bool mask with symbolic dim
            cond = mb.fill(shape=[1, N_SYM, 1], value=True)
            out = mb.select(cond=cond, a=cache, b=cache, name="out")
            return out
        return prog

    return _run_test("08_select_mixed_flex", build, build,
                     flex_inputs={"cache": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "cache": np.ones((1, TEST_N, 4), dtype=np.float16),
                         "fill_val": np.zeros((1, 1, 4), dtype=np.float16),
                     })


def test_09_cache_update_and_matmul():
    """Cache update (slice_update) + matmul with symbolic dim."""
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 1, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 1, 8, 4), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(k_cache, k_new, q):
            updated = mb.slice_update(
                x=k_cache, update=k_new,
                begin=[0, 0, 0, 0], end=[1, 1, 1, 4],
                name="k_cache_out",
            )
            kt = mb.transpose(x=updated, perm=[0, 2, 1, 3])  # (1, 1, N, 4)
            kt2 = mb.transpose(x=kt, perm=[0, 1, 3, 2])      # (1, 1, 4, N)
            scores = mb.matmul(x=q, y=kt2, name="scores")     # (1, 1, 8, N)
            return scores, updated
        return prog

    return _run_test("09_cache_update_and_matmul", build, build,
                     flex_inputs={"k_cache": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "k_cache": np.random.randn(1, TEST_N, 1, 4).astype(np.float16),
                         "k_new": np.random.randn(1, 1, 1, 4).astype(np.float16),
                         "q": np.random.randn(1, 1, 8, 4).astype(np.float16),
                     })


def test_10_gqa_repeat_flex():
    """GQA repeat: expand_dims → tile → reshape on symbolic dim.

    This is the pattern that broadcasts nkv=1 → nH=8.
    """
    def build():
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(k):
            # (1, N, 1, 4) → expand → (1, N, 1, 1, 4) → tile → (1, N, 1, 8, 4)
            k_exp = mb.expand_dims(x=k, axes=[3])
            k_rep = mb.tile(x=k_exp, reps=[1, 1, 1, 8, 1])
            # reshape back: need dynamic shape
            # Use shape_of to get N dynamically
            k_shape = mb.shape(x=k)
            n_dim = mb.gather(x=k_shape, indices=1)
            new_shape = mb.concat(values=[
                np.array([1], dtype=np.int32),
                mb.expand_dims(x=n_dim, axes=[0]),
                np.array([8, 4], dtype=np.int32),
            ], axis=0)
            k_full = mb.reshape(x=k_rep, shape=new_shape, name="out")
            return k_full
        return prog

    return _run_test("10_gqa_repeat_flex", build, build,
                     flex_inputs={"k": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={"k": np.random.randn(1, TEST_N, 1, 4).astype(np.float16)})


def test_11_two_different_functions():
    """Decode (seq=1) + prefill (seq=8) with shared flex cache."""
    def build_decode():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, N_SYM, 4), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(token, cache):
            cache_t = mb.transpose(x=cache, perm=[0, 2, 1])
            scores = mb.matmul(x=token, y=cache_t, name="scores")
            cache_p = mb.add(x=cache, y=np.float16(0.0), name="cache_out")
            return scores, cache_p
        return prog

    def build_prefill():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 8, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, N_SYM, 4), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(tokens, cache):
            cache_t = mb.transpose(x=cache, perm=[0, 2, 1])
            scores = mb.matmul(x=tokens, y=cache_t, name="scores")
            cache_p = mb.add(x=cache, y=np.float16(0.0), name="cache_out")
            return scores, cache_p
        return prog

    return _run_test("11_two_different_functions", build_decode, build_prefill,
                     flex_inputs={"cache": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "token": np.random.randn(1, 1, 4).astype(np.float16),
                         "cache": np.random.randn(1, TEST_N, 4).astype(np.float16),
                     })


def test_12_mixed_fixed_and_flex():
    """Mix of fixed-shape (sliding) and flex-shape (global) caches."""
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 512, 1, 4), dtype=types.fp16),   # sliding (fixed)
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16), # global (flex)
                mb.TensorSpec(shape=(1, 1, 8, 4), dtype=types.fp16),     # query
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(k_sliding, k_global, q):
            ks = mb.transpose(x=k_sliding, perm=[0, 2, 1, 3])
            kg = mb.transpose(x=k_global, perm=[0, 2, 1, 3])
            ks_t = mb.transpose(x=ks, perm=[0, 1, 3, 2])
            kg_t = mb.transpose(x=kg, perm=[0, 1, 3, 2])
            scores_s = mb.matmul(x=q, y=ks_t, name="scores_sliding")
            scores_g = mb.matmul(x=q, y=kg_t, name="scores_global")
            k_s_out = mb.add(x=k_sliding, y=np.float16(0.0), name="k_sliding_out")
            k_g_out = mb.add(x=k_global, y=np.float16(0.0), name="k_global_out")
            return scores_s, scores_g, k_s_out, k_g_out
        return prog

    return _run_test("12_mixed_fixed_and_flex", build, build,
                     flex_inputs={"k_global": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "k_sliding": np.random.randn(1, 512, 1, 4).astype(np.float16),
                         "k_global": np.random.randn(1, TEST_N, 1, 4).astype(np.float16),
                         "q": np.random.randn(1, 1, 8, 4).astype(np.float16),
                     })


def test_13_full_attention_flex():
    """Full attention: cache update → GQA → Q@K^T → softmax → @V."""
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16),  # k_cache
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16),  # v_cache
                mb.TensorSpec(shape=(1, 1, 8, 4), dtype=types.fp16),      # q
                mb.TensorSpec(shape=(1, 1, 1, 4), dtype=types.fp16),      # k_new
                mb.TensorSpec(shape=(1, 1, 1, 4), dtype=types.fp16),      # v_new
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(k_cache, v_cache, q, k_new, v_new):
            # Cache update
            k_upd = mb.slice_update(
                x=k_cache, update=k_new,
                begin=[0, 0, 0, 0], end=[1, 1, 1, 4],
                name="k_cache_out",
            )
            v_upd = mb.slice_update(
                x=v_cache, update=v_new,
                begin=[0, 0, 0, 0], end=[1, 1, 1, 4],
                name="v_cache_out",
            )

            # GQA: (1,N,1,4) → tile → (1,N,8,4) — use dynamic reshape
            k_exp = mb.expand_dims(x=k_upd, axes=[3])
            k_rep = mb.tile(x=k_exp, reps=[1, 1, 1, 8, 1])
            k_shape = mb.shape(x=k_upd)
            n_dim = mb.gather(x=k_shape, indices=1)
            new_shape = mb.concat(values=[
                np.array([1], dtype=np.int32),
                mb.expand_dims(x=n_dim, axes=[0]),
                np.array([8, 4], dtype=np.int32),
            ], axis=0)
            k_full = mb.reshape(x=k_rep, shape=new_shape)

            v_exp = mb.expand_dims(x=v_upd, axes=[3])
            v_rep = mb.tile(x=v_exp, reps=[1, 1, 1, 8, 1])
            v_full = mb.reshape(x=v_rep, shape=new_shape)

            # Layout for attention
            q_t = mb.transpose(x=q, perm=[0, 2, 1, 3])       # (1,8,1,4)
            k_t = mb.transpose(x=k_full, perm=[0, 2, 1, 3])  # (1,8,N,4)
            v_t = mb.transpose(x=v_full, perm=[0, 2, 1, 3])  # (1,8,N,4)

            # Q @ K^T → softmax → @ V
            kt2 = mb.transpose(x=k_t, perm=[0, 1, 3, 2])
            scores = mb.matmul(x=q_t, y=kt2)
            s_f32 = mb.cast(x=scores, dtype="fp32")
            weights = mb.softmax(x=s_f32, axis=-1)
            w_f16 = mb.cast(x=weights, dtype="fp16")
            out = mb.matmul(x=w_f16, y=v_t, name="attn_out")

            return out, k_upd, v_upd
        return prog

    return _run_test("13_full_attention_flex", build, build,
                     flex_inputs={"k_cache": (1, FLEX_LO, FLEX_HI),
                                  "v_cache": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "k_cache": np.random.randn(1, TEST_N, 1, 4).astype(np.float16),
                         "v_cache": np.random.randn(1, TEST_N, 1, 4).astype(np.float16),
                         "q": np.random.randn(1, 1, 8, 4).astype(np.float16),
                         "k_new": np.random.randn(1, 1, 1, 4).astype(np.float16),
                         "v_new": np.random.randn(1, 1, 1, 4).astype(np.float16),
                     })


def test_14_attention_with_mask():
    """Attention with dynamic mask (select/where on symbolic dim)."""
    def build():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 8, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 8, N_SYM, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 8, N_SYM, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 1, 1, N_SYM), dtype=types.bool),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            kt = mb.transpose(x=k, perm=[0, 1, 3, 2])
            scores = mb.matmul(x=q, y=kt)
            masked = mb.select(
                cond=mask, a=scores, b=np.float16(-10000.0),
            )
            s_f32 = mb.cast(x=masked, dtype="fp32")
            weights = mb.softmax(x=s_f32, axis=-1)
            w_f16 = mb.cast(x=weights, dtype="fp16")
            out = mb.matmul(x=w_f16, y=v, name="out")
            return out
        return prog

    return _run_test("14_attention_with_mask", build, build,
                     flex_inputs={"k": (2, FLEX_LO, FLEX_HI),
                                  "v": (2, FLEX_LO, FLEX_HI),
                                  "mask": (3, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "q": np.random.randn(1, 8, 1, 4).astype(np.float16),
                         "k": np.random.randn(1, 8, TEST_N, 4).astype(np.float16),
                         "v": np.random.randn(1, 8, TEST_N, 4).astype(np.float16),
                         "mask": np.ones((1, 1, 1, TEST_N), dtype=bool),
                     })


def test_15_decode_vs_prefill_attention():
    """Decode + prefill with different seq dims, flex cache, GQA, SDPA."""
    def build_decode():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 8, 1, 4), dtype=types.fp16),       # q
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16),   # k_cache
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16),   # v_cache
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k_cache, v_cache):
            k_t = mb.transpose(x=k_cache, perm=[0, 2, 1, 3])
            v_t = mb.transpose(x=v_cache, perm=[0, 2, 1, 3])
            k_8 = mb.tile(x=k_t, reps=[1, 8, 1, 1])
            v_8 = mb.tile(x=v_t, reps=[1, 8, 1, 1])
            kt = mb.transpose(x=k_8, perm=[0, 1, 3, 2])
            scores = mb.matmul(x=q, y=kt)
            s_f32 = mb.cast(x=scores, dtype="fp32")
            weights = mb.softmax(x=s_f32, axis=-1)
            w_f16 = mb.cast(x=weights, dtype="fp16")
            out = mb.matmul(x=w_f16, y=v_8, name="out")
            k_out = mb.add(x=k_cache, y=np.float16(0.0), name="k_cache_out")
            v_out = mb.add(x=v_cache, y=np.float16(0.0), name="v_cache_out")
            return out, k_out, v_out
        return prog

    def build_prefill():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 8, 8, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16),
                mb.TensorSpec(shape=(1, N_SYM, 1, 4), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k_cache, v_cache):
            k_t = mb.transpose(x=k_cache, perm=[0, 2, 1, 3])
            v_t = mb.transpose(x=v_cache, perm=[0, 2, 1, 3])
            k_8 = mb.tile(x=k_t, reps=[1, 8, 1, 1])
            v_8 = mb.tile(x=v_t, reps=[1, 8, 1, 1])
            kt = mb.transpose(x=k_8, perm=[0, 1, 3, 2])
            scores = mb.matmul(x=q, y=kt)
            s_f32 = mb.cast(x=scores, dtype="fp32")
            weights = mb.softmax(x=s_f32, axis=-1)
            w_f16 = mb.cast(x=weights, dtype="fp16")
            out = mb.matmul(x=w_f16, y=v_8, name="out")
            k_out = mb.add(x=k_cache, y=np.float16(0.0), name="k_cache_out")
            v_out = mb.add(x=v_cache, y=np.float16(0.0), name="v_cache_out")
            return out, k_out, v_out
        return prog

    return _run_test("15_decode_vs_prefill_attention", build_decode, build_prefill,
                     flex_inputs={"k_cache": (1, FLEX_LO, FLEX_HI),
                                  "v_cache": (1, FLEX_LO, FLEX_HI)},
                     test_inputs_a={
                         "q": np.random.randn(1, 8, 1, 4).astype(np.float16),
                         "k_cache": np.random.randn(1, TEST_N, 1, 4).astype(np.float16),
                         "v_cache": np.random.randn(1, TEST_N, 1, 4).astype(np.float16),
                     })


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import platform
    print("CoreML Multifunction + Dynamic Dimensions — Incremental Tests")
    print(f"coremltools: {ct.__version__}  macOS: {platform.mac_ver()[0]}")
    print()

    tests = [
        test_01_baseline_concrete,
        test_02_identity_flex,
        test_03_slice_update_flex,
        test_04_transpose_flex,
        test_05_matmul_flex,
        test_06_sdpa_flex,
        test_07_select_flex,
        test_08_select_mixed_flex,
        test_09_cache_update_and_matmul,
        test_10_gqa_repeat_flex,
        test_11_two_different_functions,
        test_12_mixed_fixed_and_flex,
        test_13_full_attention_flex,
        test_14_attention_with_mask,
        test_15_decode_vs_prefill_attention,
    ]

    results = {}
    for test_fn in tests:
        try:
            ok = test_fn()
            results[test_fn.__name__] = ok
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            traceback.print_exc()
            results[test_fn.__name__] = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    n_pass = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    print(f"\n  {n_pass} passed, {n_fail} failed out of {len(results)} tests")


if __name__ == "__main__":
    main()
