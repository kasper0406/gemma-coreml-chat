#!/usr/bin/env python3
"""Verify that loading every materialized function has little memory overhead.

Expectation: because `materialize_symbolic_shape_program` dedupes constants
across functions and CoreML mmap's the .mlpackage weight blob, loading all
N functions should cost roughly the same resident memory as loading one.

The test:
1. Build a synthetic dynamic-shape mlpackage with a ~sizeable constant weight
   (large enough that duplicated copies would show up clearly in RSS).
2. Materialize it into a multifunction with M sizes × {prefill, decode} = 2M
   functions.
3. Baseline: load one function on CPU, measure RSS.
4. Load each remaining function and measure the RSS increase per load.
5. Assert: total growth loading all 2M functions stays well under M×
   (something like <2× the baseline cost is a reasonable upper bound — the
   real cost is just per-function compiled metadata, not weight bytes).

Usage:
    .venv/bin/python tests/test_materialize_memory.py
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types, get_new_symbol

from gemma_chat.materialize import materialize_mlpackage


# Pick sizes + weight sizing so the constant is large enough for duplicated
# loads to be obvious in RSS, yet small enough that the test stays fast.
SIZES = [64, 128, 256, 512, 1024]          # 5 sizes × 2 phases = 10 functions
CONST_BYTES = 64 * 1024 * 1024              # 64 MB per const (fp16)


def _rss_mb() -> float:
    """Current RSS in MB (macOS `ps`-based, avoids the ru_maxrss stickiness)."""
    try:
        import subprocess as _sp
        out = _sp.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip()) / 1024
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _build_prog(n, weight: np.ndarray, name: str):
    """Tiny dynamic-shape program with a chunky (D_in, D_out) matmul weight
    plus the ANE-breaking ``shape → range_1d → expand_dims`` op trio.
    The 2D weight is hard for the optimizer to fold away."""
    D_in, D_out = weight.shape
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, n, D_in), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, D_in), dtype=types.fp16),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(k, q, position):
        w = mb.const(val=weight, name=f"{name}_w")
        scores = mb.matmul(x=k, y=w)          # (1, n, D_out)
        # ANE-breaking trio mirroring the real model's mask derivation.
        k_shape = mb.shape(x=k)
        n_dim = mb.gather(x=k_shape, indices=1)
        pos_k = mb.range_1d(start=0, end=n_dim, step=1)
        pos_k_b = mb.expand_dims(x=pos_k, axes=[0, 2])     # (1, n, 1)
        pos_q = mb.gather(x=position, indices=0)
        pos_q_b = mb.expand_dims(x=pos_q, axes=[0, 1, 2])
        mask = mb.less_equal(x=pos_k_b, y=pos_q_b)         # (1, n, 1)
        mask_f = mb.cast(x=mask, dtype="fp16")
        out = mb.mul(x=scores, y=mask_f, name="out")       # (1, n, D_out)
        return out
    return prog


def _apply_rangedim(model, names, lo, hi, dim_idx=1):
    spec = model._spec
    target = set(names)
    for inp in spec.description.input:
        if inp.name not in target:
            continue
        arr = inp.type.multiArrayType
        shape = list(arr.shape)
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


def _build_dynamic_multifunction(tmp: Path, weight: np.ndarray) -> Path:
    """{prefill, decode} multifunction with RangeDim k input."""
    n = get_new_symbol("N")
    prog = _build_prog(n, weight, "func")
    m = ct.convert(
        prog, source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )
    _apply_rangedim(m, ["k"], min(SIZES), max(SIZES))

    pref_pkg = tmp / "prefill.mlpackage"
    dec_pkg = tmp / "decode.mlpackage"
    m.save(str(pref_pkg))
    m.save(str(dec_pkg))
    del m

    mf = tmp / "dynamic.mlpackage"
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(str(pref_pkg), "main", "prefill")
    desc.add_function(str(dec_pkg), "main", "decode")
    desc.default_function_name = "decode"
    ct.utils.save_multifunction(desc, str(mf))
    return mf


def _on_disk_size_mb(pkg: Path) -> float:
    return sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    # Square-ish weight so the compiler can't easily constant-fold it.
    n_elements = CONST_BYTES // 2  # fp16 → 2 bytes/elem
    D_in = 512
    D_out = n_elements // D_in
    print(f"materialize memory overhead  ct={ct.__version__}  "
          f"baseline RSS={_rss_mb():.0f} MB")
    print(f"sizes={SIZES}  weight=({D_in}×{D_out}) fp16 "
          f"(~{D_in * D_out * 2 / 1e6:.0f} MB)\n")

    tmp = Path(tempfile.mkdtemp(prefix="mat-mem-"))
    try:
        np.random.seed(0)
        weight = (np.random.randn(D_in, D_out) * 1e-4).astype(np.float16)

        t0 = time.time()
        dyn = _build_dynamic_multifunction(tmp, weight)
        print(f"built dynamic  ({_on_disk_size_mb(dyn):.1f} MB on disk) "
              f"in {time.time() - t0:.1f}s")

        mat = tmp / "materialized.mlpackage"
        t0 = time.time()
        materialize_mlpackage(dyn, mat, SIZES)
        print(f"materialized   ({_on_disk_size_mb(mat):.1f} MB on disk) "
              f"in {time.time() - t0:.1f}s")
        print()

        # --- Measure RSS as we load each function one by one. ---
        peek = ct.models.MLModel(str(mat), skip_model_load=True)
        fn_names = sorted(fd.name for fd in peek._spec.description.functions)
        del peek

        print(f"loading {len(fn_names)} functions on CPU_ONLY\n")
        print(f"  {'step':<5} {'function':<22} {'RSS (MB)':>10} {'Δ (MB)':>10}")

        rss_before = _rss_mb()
        baseline_rss = None
        baseline_fn = None
        loaded: list = []   # keep refs so the kernels aren't GC'd between measurements
        peak_delta = 0.0
        first_delta = None

        for i, fn in enumerate(fn_names):
            m = ct.models.MLModel(
                str(mat),
                compute_units=ct.ComputeUnit.CPU_ONLY,
                function_name=fn,
            )
            loaded.append(m)
            rss = _rss_mb()
            delta = rss - rss_before
            if baseline_rss is None:
                # The first load brings the weights into RSS. Baseline off that.
                baseline_rss = rss
                baseline_fn = fn
                print(f"  {i+1:<5} {fn:<22} {rss:>10.0f} {delta:>10.0f}  ← baseline")
            else:
                incr = rss - baseline_rss
                if first_delta is None:
                    first_delta = incr
                peak_delta = max(peak_delta, incr)
                print(f"  {i+1:<5} {fn:<22} {rss:>10.0f} {incr:>+10.0f}")

        print()
        print(f"baseline (after 1st load of {baseline_fn!r}): {baseline_rss:.0f} MB")
        print(f"peak extra RSS for remaining {len(fn_names) - 1} loads:  "
              f"{peak_delta:.0f} MB")
        per_fn = peak_delta / max(1, len(fn_names) - 1)
        print(f"avg per-function overhead:                {per_fn:.1f} MB "
              f"(weights ≈ {CONST_BYTES / 1e6:.0f} MB)")
        print()

        # --- Verdict ---
        weight_mb = CONST_BYTES / 1e6
        # If constants weren't shared, loading N-1 extra functions would add
        # ~(N-1) × weight_mb.  Allow up to 1× weight_mb total extra as slack
        # (per-function compiled metadata + CoreML scratch).
        budget = max(weight_mb, 50.0)
        ok = peak_delta < budget
        verdict = "✓" if ok else "✗"
        print(f"{verdict} total extra RSS {peak_delta:.0f} MB < budget {budget:.0f} MB "
              f"(weight size)  — constants deduplicated across functions")
        return 0 if ok else 1

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
