#!/usr/bin/env python3
"""Minimal reproducer: multifunction CoreML model with RangeDim fails to load.

Single-function models with RangeDim (flexible shapes) load and predict fine.
When two such models are merged via MultiFunctionDescriptor, the resulting
multifunction .mlpackage fails to load on macOS (tested on macOS 15/26).

To run:
    pip install coremltools numpy
    python multifunction_rangedim_bug.py

Expected: both single-function and multifunction models load and predict.
Actual:   multifunction model fails to load (model.__proxy__ is None).

Filed as Apple Feedback: FB_NUMBER
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def _apply_rangedim(spec, input_name: str, dim: int, lower: int, upper: int) -> None:
    """Set RangeDim on a specific input dimension at the protobuf level."""
    for inp in spec.description.input:
        if inp.name == input_name:
            ma = inp.type.multiArrayType
            # Ensure sizeRanges has enough entries
            while len(ma.shapeRange.sizeRanges) < len(ma.shape):
                r = ma.shapeRange.sizeRanges.add()
                r.lowerBound = ma.shape[len(ma.shapeRange.sizeRanges) - 1]
                r.upperBound = ma.shape[len(ma.shapeRange.sizeRanges) - 1]
            ma.shapeRange.sizeRanges[dim].lowerBound = lower
            ma.shapeRange.sizeRanges[dim].upperBound = upper
            return
    raise ValueError(f"Input {input_name!r} not found")


def _apply_rangedim_output(spec, output_name: str, dim: int, lower: int, upper: int) -> None:
    """Set RangeDim on a specific output dimension at the protobuf level."""
    for out in spec.description.output:
        if out.name == output_name:
            ma = out.type.multiArrayType
            while len(ma.shapeRange.sizeRanges) < len(ma.shape):
                r = ma.shapeRange.sizeRanges.add()
                r.lowerBound = ma.shape[len(ma.shapeRange.sizeRanges) - 1]
                r.upperBound = ma.shape[len(ma.shapeRange.sizeRanges) - 1]
            ma.shapeRange.sizeRanges[dim].lowerBound = lower
            ma.shapeRange.sizeRanges[dim].upperBound = upper
            return
    raise ValueError(f"Output {output_name!r} not found")


def _build_simple_model(name: str, output_path: Path) -> None:
    """Build a tiny single-function model with a symbolic dim on cache."""
    from coremltools.converters.mil.mil import get_new_symbol

    seq = get_new_symbol()

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1,), dtype=types.int32),
            mb.TensorSpec(shape=(1, seq, 4), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(position, cache):
        return mb.reduce_sum(x=cache, axes=[1], name="output")

    model = ct.convert(
        prog,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )

    # Apply RangeDim at protobuf level (same technique as real Gemma export)
    spec = model.get_spec()
    _apply_rangedim(spec, "cache", dim=1, lower=1, upper=1024)
    weights_dir = model.weights_dir
    model.save(str(output_path))

    # Overwrite with RangeDim-patched spec
    import shutil
    shutil.rmtree(str(output_path))
    ct.utils.save_spec(spec, str(output_path), weights_dir=weights_dir)
    print(f"  Saved {name} → {output_path}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mf-rangedim-bug-"))
    print(f"Working dir: {tmp}\n")

    model_a_path = tmp / "func_a.mlpackage"
    model_b_path = tmp / "func_b.mlpackage"
    merged_path = tmp / "merged.mlpackage"

    # ── Step 1: Build two single-function models with RangeDim ──
    print("Step 1: Building two single-function models with RangeDim…")
    _build_simple_model("func_a", model_a_path)
    _build_simple_model("func_b", model_b_path)

    # ── Step 2: Verify single-function models load fine ──
    print("\nStep 2: Loading single-function models (should work)…")
    for label, path in [("func_a", model_a_path), ("func_b", model_b_path)]:
        m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
        if m.__proxy__ is None:
            print(f"  ❌ {label}: FAILED to load (proxy is None)")
        else:
            result = m.predict({
                "position": np.array([0], dtype=np.int32),
                "cache": np.ones((1, 8, 4), dtype=np.float16),
            })
            print(f"  ✅ {label}: loaded OK, output shape = {list(result.values())[0].shape}")
        del m

    # ── Step 3: Merge into multifunction model ──
    print("\nStep 3: Merging into multifunction model…")
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

    desc = MultiFunctionDescriptor()
    desc.add_function(
        str(model_a_path),
        src_function_name="main",
        target_function_name="prefill",
    )
    desc.add_function(
        str(model_b_path),
        src_function_name="main",
        target_function_name="decode",
    )
    desc.default_function_name = "decode"
    save_multifunction(desc, str(merged_path))
    print(f"  Saved merged → {merged_path}")

    # ── Step 4: Try to load the multifunction model ──
    print("\nStep 4: Loading multifunction model (this is where it fails)…")
    for fn_name in ["decode", "prefill"]:
        try:
            m = ct.models.MLModel(
                str(merged_path),
                compute_units=ct.ComputeUnit.CPU_ONLY,
                function_name=fn_name,
            )
            if m.__proxy__ is None:
                print(f"  ❌ {fn_name}: FAILED — model loaded but proxy is None")
                print(f"     This means CoreML could not compile the model.")
            else:
                result = m.predict({
                    "position": np.array([0], dtype=np.int32),
                    "cache": np.ones((1, 8, 4), dtype=np.float16),
                })
                print(
                    f"  ✅ {fn_name}: loaded OK, "
                    f"output shape = {list(result.values())[0].shape}"
                )
            del m
        except Exception as e:
            print(f"  ❌ {fn_name}: EXCEPTION — {type(e).__name__}: {e}")

    # ── Step 5: Show the spec for debugging ──
    print("\nStep 5: Inspecting merged spec…")
    spec_model = ct.models.MLModel(str(merged_path), skip_model_load=True)
    spec = spec_model.get_spec()
    for fd in spec.description.functions:
        print(f"\n  Function: {fd.name}")
        for inp in fd.input:
            ma = inp.type.multiArrayType
            shape = list(ma.shape)
            ranges = [(r.lowerBound, r.upperBound) for r in ma.shapeRange.sizeRanges]
            print(f"    Input  {inp.name}: shape={shape}, ranges={ranges or 'none'}")
        for out in fd.output:
            ma = out.type.multiArrayType
            shape = list(ma.shape)
            ranges = [(r.lowerBound, r.upperBound) for r in ma.shapeRange.sizeRanges]
            print(f"    Output {out.name}: shape={shape}, ranges={ranges or 'none'}")

    print(f"\nTemp dir (not cleaned): {tmp}")
    print("You can inspect the models manually or attach them to a bug report.")


if __name__ == "__main__":
    main()
