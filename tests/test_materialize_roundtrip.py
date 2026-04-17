#!/usr/bin/env python3
"""End-to-end check for `gemma-materialize`.

1. Build a tiny dynamic-shape multifunction mlpackage (prefill+decode, one
   RangeDim'd input each) that exercises the ANE-breaking op trio
   (shape → range_1d → expand_dims flowing into an attention mask).
2. Run `gemma_chat.materialize.materialize_mlpackage` to produce a concrete
   multifunction with one function per target size.
3. Load each materialized function on CPU, ANE, ALL and call predict.

Usage:
    .venv/bin/python tests/test_materialize_roundtrip.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types, get_new_symbol

from gemma_chat.materialize import materialize_mlpackage


SIZES = [32, 64, 128]
H, D = 4, 8


def _attn_prog(n):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, H, 1, D), dtype=types.fp16),
            mb.TensorSpec(shape=(1, n, H, D), dtype=types.fp16),
            mb.TensorSpec(shape=(1, n, H, D), dtype=types.fp16),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(q, k, v, position):
        kt = mb.transpose(x=k, perm=[0, 2, 1, 3])
        vt = mb.transpose(x=v, perm=[0, 2, 1, 3])
        ktT = mb.transpose(x=kt, perm=[0, 1, 3, 2])
        scores = mb.matmul(x=q, y=ktT)
        # ANE-breaking trio:
        k_shape = mb.shape(x=k)
        n_dim = mb.gather(x=k_shape, indices=1)
        pos_k = mb.range_1d(start=0, end=n_dim, step=1)
        pos_k_b = mb.expand_dims(x=pos_k, axes=[0, 1, 2])
        pos_q = mb.gather(x=position, indices=0)
        pos_q_b = mb.expand_dims(x=pos_q, axes=[0, 1, 2, 3])
        mask = mb.less_equal(x=pos_k_b, y=pos_q_b)
        masked = mb.select(cond=mask, a=scores, b=np.float16(-10000.0))
        w = mb.softmax(x=mb.cast(x=masked, dtype="fp32"), axis=-1)
        out = mb.matmul(x=mb.cast(x=w, dtype="fp16"), y=vt, name="out")
        return out
    return prog


def _apply_rangedim(model, names, lo, hi):
    spec = model._spec
    target = set(names)
    for inp in spec.description.input:
        if inp.name not in target:
            continue
        arr = inp.type.multiArrayType
        shape = list(arr.shape)
        shape[1] = lo
        arr.ClearField("shape")
        for s in shape:
            arr.shape.append(s)
        arr.ClearField("ShapeFlexibility")
        for i, d in enumerate(arr.shape):
            sr = arr.shapeRange.sizeRanges.add()
            if i == 1:
                sr.lowerBound = lo
                sr.upperBound = hi
            else:
                sr.lowerBound = d
                sr.upperBound = d


def _build_dynamic_mlpackage(tmp: Path) -> Path:
    """Build a multifunction {prefill, decode} with one RangeDim input each."""
    n = get_new_symbol("N")
    prog = _attn_prog(n)
    m = ct.convert(
        prog, source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )
    _apply_rangedim(m, ["k", "v"], min(SIZES), max(SIZES))
    pref_pkg = tmp / "prefill.mlpackage"
    dec_pkg = tmp / "decode.mlpackage"
    m.save(str(pref_pkg))
    m.save(str(dec_pkg))
    del m

    mf_path = tmp / "dynamic.mlpackage"
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(str(pref_pkg), "main", "prefill")
    desc.add_function(str(dec_pkg), "main", "decode")
    desc.default_function_name = "decode"
    ct.utils.save_multifunction(desc, str(mf_path))
    return mf_path


def _predict(model, n: int):
    return model.predict({
        "q": np.random.randn(1, H, 1, D).astype(np.float16),
        "k": np.random.randn(1, n, H, D).astype(np.float16),
        "v": np.random.randn(1, n, H, D).astype(np.float16),
        "position": np.array([n - 1], dtype=np.int32),
    })


def main():
    import platform
    print(f"gemma-materialize roundtrip  ct={ct.__version__}  "
          f"macOS={platform.mac_ver()[0]}\n")

    tmp = Path(tempfile.mkdtemp(prefix="gemma-mat-rt-"))
    try:
        dyn = _build_dynamic_mlpackage(tmp)
        print(f"built dynamic mlpackage: {dyn}\n")

        out = tmp / "materialized.mlpackage"
        materialize_mlpackage(dyn, out, SIZES)
        print(f"\nmaterialized: {out}\n")

        # List functions in the materialized pkg.
        peek = ct.models.MLModel(str(out), skip_model_load=True)
        fns = [fd.name for fd in peek._spec.description.functions]
        print(f"materialized functions: {fns}\n")
        expected = {f"{p}_{s}" for p in ("prefill", "decode") for s in SIZES}
        missing = expected - set(fns)
        if missing:
            print(f"  MISSING: {missing}")
            return 1

        # Test matrix: each function × each compute unit.
        print("TEST MATRIX")
        print(f"  {'function':<20} " + "  ".join(f"{c:<14}" for c in ("cpu", "ane", "all")))

        pass_count = 0
        fail_count = 0
        for fn in sorted(fns):
            size = int(fn.rsplit("_", 1)[-1])
            cells = [f"  {fn:<20}"]
            for cu_name, cu in [
                ("cpu", ct.ComputeUnit.CPU_ONLY),
                ("ane", ct.ComputeUnit.CPU_AND_NE),
                ("all", ct.ComputeUnit.ALL),
            ]:
                try:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        m = ct.models.MLModel(
                            str(out), compute_units=cu, function_name=fn,
                        )
                    loaded = m.__proxy__ is not None
                    if not loaded:
                        cells.append(f"  {'✗ load':<14}")
                        fail_count += 1
                        continue
                    _ = _predict(m, size)
                    cells.append(f"  {'✓':<14}")
                    pass_count += 1
                except Exception as e:
                    cells.append(f"  {'✗ ' + type(e).__name__:<14}")
                    fail_count += 1
            print("  ".join(cells))

        print(f"\nsummary: {pass_count} pass / {fail_count} fail")
        return 0 if fail_count == 0 else 1

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
