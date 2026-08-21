#!/usr/bin/env python3
"""Test whether EnumeratedShapes helps with the ANE dynamic-shape ops limitation.

The root cause of ANE failures for dynamic-context exports is the trio of ops
produced when a symbolic dim flows through `shape → range_1d → expand_dims`
(see commit 54a0b47).  ANE supports these ops only with concrete shapes.

This test builds a minimal MIL program that includes exactly that pattern
(mirroring what the global-attention mask derivation produces in the real
model) and tries to load it under three flexibility regimes:

  A) RangeDim (shapeRange)      — baseline "dynamic" case
  B) EnumeratedShapes           — does protobuf-level enumeration rescue ANE?
  C) Multifunction w/ concrete  — one function per concrete size

For each, we try CPU_ONLY, CPU_AND_NE, and ALL, and report load + predict.

Usage:
    .venv/bin/python tests/test_enumerated_vs_range.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
import traceback
import warnings
from dataclasses import dataclass

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types, get_new_symbol


SIZES = [32, 64, 128]          # enumerated / concrete values under test
HEADS = 4
HDIM = 8
RUN_SIZE = 64                  # a value from SIZES for predict sanity check


# ── The problematic pattern ────────────────────────────────────────────────
#
# shape(k_cache) → gather dim-1 → range_1d(0, N) → expand_dims → compare
# against `position` → mask(where) on attention scores.
#
# This replicates the global-attention "pos_k <= pos_q" mask in
# decode_coreml._attn_decode / _attn_chunk (is_global branch).
#
# The helper builds a program with that pattern using a dim `n` which can be
# symbolic or concrete.


def _build_attn_masked(n) -> mb.program:
    """Build Q·K^T with a mask derived via shape→range_1d→expand_dims.

    n: symbolic Symbol, or concrete int.  When symbolic, the program uses the
       "shape of k → range_1d" path; when concrete, it still uses it (same
       graph) so both regimes test the exact same op sequence, only the
       flexibility metadata differs.
    """
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, HEADS, 1, HDIM), dtype=types.fp16),   # q
            mb.TensorSpec(shape=(1, n, HEADS, HDIM), dtype=types.fp16),   # k_cache
            mb.TensorSpec(shape=(1, n, HEADS, HDIM), dtype=types.fp16),   # v_cache
            mb.TensorSpec(shape=(1,), dtype=types.int32),                 # position
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(q, k_cache, v_cache, position):
        # K/V layout for attention: (1, N, H, D) → (1, H, N, D)
        kt = mb.transpose(x=k_cache, perm=[0, 2, 1, 3])
        vt = mb.transpose(x=v_cache, perm=[0, 2, 1, 3])
        # scores = q · kᵀ
        ktT = mb.transpose(x=kt, perm=[0, 1, 3, 2])
        scores = mb.matmul(x=q, y=ktT)                     # (1, H, 1, N)

        # --- THE ANE-BREAKING TRIO (shape → range_1d → expand_dims) --------
        # Derive pos_k = [0 .. N-1] dynamically from cache shape.
        k_shape = mb.shape(x=k_cache)                      # (4,) int32
        n_dim = mb.gather(x=k_shape, indices=1)            # () scalar int32
        # range_1d(end=N) yields (N,) of ints; symbolic when N is symbolic.
        pos_k = mb.range_1d(start=0, end=n_dim, step=1)    # (N,) int32
        pos_k_b = mb.expand_dims(x=pos_k, axes=[0, 1, 2])  # (1,1,1,N)
        # pos_q scalar via gather on position vector.
        pos_q = mb.gather(x=position, indices=0)           # () int32
        pos_q_b = mb.expand_dims(x=pos_q, axes=[0, 1, 2, 3])  # (1,1,1,1)
        # mask = pos_k <= pos_q   → (1,1,1,N) bool
        mask = mb.less_equal(x=pos_k_b, y=pos_q_b)

        masked = mb.select(cond=mask, a=scores, b=np.float16(-10000.0))
        s32 = mb.cast(x=masked, dtype="fp32")
        w = mb.softmax(x=s32, axis=-1)
        wf16 = mb.cast(x=w, dtype="fp16")
        out = mb.matmul(x=wf16, y=vt, name="attn_out")      # (1, H, 1, D)
        return out
    return prog


# ── Protobuf-level shape-flexibility helpers ───────────────────────────────


def _apply_range_dim(model, input_name: str, dim_idx: int, lo: int, hi: int):
    """Set shapeRange on a flexible dim (equivalent to RangeDim)."""
    spec = model._spec
    for inp in spec.description.input:
        if inp.name != input_name:
            continue
        arr = inp.type.multiArrayType
        shape = list(arr.shape)
        # Default shape uses `lo`
        if len(shape) > dim_idx:
            shape[dim_idx] = lo
            arr.ClearField("shape")
            for s in shape:
                arr.shape.append(s)
        arr.ClearField("ShapeFlexibility")
        for i, d in enumerate(arr.shape):
            sr = arr.shapeRange.sizeRanges.add()
            if i == dim_idx:
                sr.lowerBound = lo
                sr.upperBound = hi
            else:
                sr.lowerBound = d
                sr.upperBound = d


def _apply_enumerated(model, input_name: str, shapes: list[tuple[int, ...]],
                      default_shape: tuple[int, ...]):
    """Set enumeratedShapes on a flexible dim."""
    spec = model._spec
    for inp in spec.description.input:
        if inp.name != input_name:
            continue
        arr = inp.type.multiArrayType
        arr.ClearField("ShapeFlexibility")
        arr.ClearField("shape")
        for d in default_shape:
            arr.shape.append(d)
        for s in shapes:
            nds = arr.enumeratedShapes.shapes.add()
            for d in s:
                nds.shape.append(d)


# ── Variant builders ───────────────────────────────────────────────────────


@dataclass
class Variant:
    name: str
    build_model: callable   # () -> ct.models.MLModel  (not yet shape-finalized)
    finalize: callable      # (model) -> None  (applies flexibility)


def build_rangedim() -> ct.models.MLModel:
    n = get_new_symbol("N")
    prog = _build_attn_masked(n)
    model = ct.convert(
        prog, source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )
    _apply_range_dim(model, "k_cache", 1, min(SIZES), max(SIZES))
    _apply_range_dim(model, "v_cache", 1, min(SIZES), max(SIZES))
    return model


def build_enumerated() -> ct.models.MLModel:
    n = get_new_symbol("N")
    prog = _build_attn_masked(n)
    model = ct.convert(
        prog, source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )
    k_shapes = [(1, s, HEADS, HDIM) for s in SIZES]
    default = (1, SIZES[0], HEADS, HDIM)
    _apply_enumerated(model, "k_cache", k_shapes, default)
    _apply_enumerated(model, "v_cache", k_shapes, default)
    return model


def build_concrete_multifunction(tmp: str) -> str:
    """Build one MIL program per concrete size, merge via save_multifunction."""
    fn_paths = []
    for s in SIZES:
        prog = _build_attn_masked(s)
        m = ct.convert(
            prog, source="milinternal",
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT32,
            skip_model_load=True,
        )
        p = os.path.join(tmp, f"func_{s}.mlpackage")
        m.save(p)
        fn_paths.append((s, p))

    mf_path = os.path.join(tmp, "mf.mlpackage")
    desc = ct.utils.MultiFunctionDescriptor()
    for s, p in fn_paths:
        desc.add_function(p, "main", f"attn_{s}")
    desc.default_function_name = f"attn_{SIZES[0]}"
    ct.utils.save_multifunction(desc, mf_path)
    return mf_path


def build_materialized(tmp: str) -> str:
    """Use coremltools' materialize_dynamic_shape_mlmodel.

    Builds one dynamic-shape (RangeDim) model, then materializes concrete
    per-size functions via `common::materialize_symbolic_shape_program`.
    The materialized multifunction shares weights across functions
    (const_deduplication) and has NO dynamic shape ops in any function.
    """
    from coremltools.models.utils import materialize_dynamic_shape_mlmodel

    # Build a symbolic-shape MLModel (reuse the RangeDim builder).
    dynamic_model = build_rangedim()
    dyn_path = os.path.join(tmp, "dyn.mlpackage")
    dynamic_model.save(dyn_path)

    # Reload to get an MLModel that `materialize_dynamic_shape_mlmodel` accepts.
    dynamic_model = ct.models.MLModel(dyn_path, skip_model_load=True)

    mat_map: dict[str, dict[str, tuple[int, ...]]] = {}
    for s in SIZES:
        mat_map[f"attn_{s}"] = {
            "k_cache": (1, s, HEADS, HDIM),
            "v_cache": (1, s, HEADS, HDIM),
        }
    mat_path = os.path.join(tmp, "mat.mlpackage")
    materialize_dynamic_shape_mlmodel(dynamic_model, mat_map, mat_path)
    return mat_path


# ── Runner ─────────────────────────────────────────────────────────────────


def _predict_inputs(n: int) -> dict:
    return {
        "q": np.random.randn(1, HEADS, 1, HDIM).astype(np.float16),
        "k_cache": np.random.randn(1, n, HEADS, HDIM).astype(np.float16),
        "v_cache": np.random.randn(1, n, HEADS, HDIM).astype(np.float16),
        "position": np.array([n - 1], dtype=np.int32),
    }


def _try_compute_unit(mlpackage_path: str, cu: ct.ComputeUnit,
                      function_name: str | None = None) -> tuple[bool, bool, str]:
    """Returns (loaded, predict_ok, diagnostic)."""
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            kwargs = {}
            if function_name is not None:
                kwargs["function_name"] = function_name
            m = ct.models.MLModel(mlpackage_path, compute_units=cu, **kwargs)
        # ct.models.MLModel may set __proxy__ to None if loading failed at
        # E5RT level (non-fatal).
        loaded = m.__proxy__ is not None
        note = ""
        if not loaded:
            e5 = [str(x.message) for x in ws
                  if "E5RT" in str(x.message) or "not be able to" in str(x.message)]
            note = "; ".join(s[:180] for s in e5[:3])
        predict_ok = False
        if loaded:
            try:
                inp = _predict_inputs(RUN_SIZE)
                _ = m.predict(inp)
                predict_ok = True
            except Exception as e:
                note = f"predict: {type(e).__name__}: {str(e)[:160]}"
        return loaded, predict_ok, note
    except Exception as e:
        return False, False, f"{type(e).__name__}: {str(e)[:200]}"


def run_matrix():
    import platform
    print("CoreML EnumeratedShapes vs RangeDim vs Multifunction-Concrete")
    print(f"coremltools: {ct.__version__}  macOS: {platform.mac_ver()[0]}\n")

    tmp = tempfile.mkdtemp(prefix="enum-vs-range-")
    try:
        # --- Variant A: RangeDim ---
        print("=" * 60)
        print(f"A) RangeDim  (N in [{min(SIZES)}, {max(SIZES)}])")
        print("=" * 60)
        m_range = build_rangedim()
        p_range = os.path.join(tmp, "a_rangedim.mlpackage")
        m_range.save(p_range)
        del m_range

        # --- Variant B: EnumeratedShapes ---
        print("\n" + "=" * 60)
        print(f"B) EnumeratedShapes  (N ∈ {SIZES})")
        print("=" * 60)
        m_enum = build_enumerated()
        p_enum = os.path.join(tmp, "b_enumerated.mlpackage")
        m_enum.save(p_enum)
        del m_enum

        # --- Variant C: Multifunction w/ concrete sizes (hand-built) ---
        print("\n" + "=" * 60)
        print(f"C) Multifunction concrete  (functions: {['attn_'+str(s) for s in SIZES]})")
        print("=" * 60)
        p_conc = build_concrete_multifunction(tmp)

        # --- Variant D: Materialized symbolic-shape program ---
        print("\n" + "=" * 60)
        print(f"D) Materialized   (via common::materialize_symbolic_shape_program)")
        print("=" * 60)
        p_mat = build_materialized(tmp)

        # --- Test matrix ---
        print("\n" + "=" * 60)
        print("TEST MATRIX  (load + predict on each compute unit)")
        print("=" * 60)

        variants = [
            ("A_range",   p_range, None),
            ("B_enum",    p_enum,  None),
            ("C_conc",    p_conc,  f"attn_{RUN_SIZE}"),
            ("D_matmap",  p_mat,   f"attn_{RUN_SIZE}"),
        ]
        cus = [
            ("cpu", ct.ComputeUnit.CPU_ONLY),
            ("ane", ct.ComputeUnit.CPU_AND_NE),
            ("all", ct.ComputeUnit.ALL),
        ]

        header = f"  {'variant':<10} " + "  ".join(f"{c:<20}" for c, _ in cus)
        print(header)
        for vname, ppath, fn in variants:
            row = [f"  {vname:<10}"]
            for _, cu in cus:
                loaded, pred, note = _try_compute_unit(ppath, cu, fn)
                mark = (
                    "✓ load+pred" if (loaded and pred)
                    else "✓ load ✗pred" if loaded
                    else "✗ load"
                )
                cell = mark + (f" [{note[:40]}]" if note else "")
                row.append(f"{cell:<20}")
            print("  ".join(row))

    except Exception as e:
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_matrix()
