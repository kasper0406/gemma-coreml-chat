"""Convert a dynamic-shape `.mlpackage` into a multifunction one with concrete
power-of-2 KV-cache sizes.

Motivation
----------
The default export uses `RangeDim` on global KV caches for a 1..65k dynamic
context.  That produces MIL ops (`shape`, `range_1d`, `expand_dims`) whose
outputs flow into attention masking — and those ops are **only supported by
the GPU backend**.  CPU and ANE refuse to load the model.

At runtime the KV cache is grown by a factor of 2 on exhaustion, so in
practice only a handful of concrete sizes are ever observed: 8, 16, 32, …,
65536.  This utility takes a dynamic-shape model and materializes one
concrete-shape function per power of 2, via CoreML's built-in
`materialize_symbolic_shape_program` MIL pass.  The resulting multifunction
model has:

- **No** dynamic shape ops in any function (each is specialized to a concrete
  cache length, so the shape → range → mask chain folds to a constant).
- Deduplicated constants — all sizes share the same int4 weights.

The runtime picks the function whose size matches the current cache length.

Usage
-----
```
uv run gemma-materialize --input gemma4-e2b.mlpackage --output gemma4-e2b-mat.mlpackage
```

Defaults to sizes [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
16384, 32768, 65536], matching the runtime doubling growth strategy.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import coremltools as ct


# Default: powers of 2 from CHUNK_SIZE (8) to MAX_SEQ_LEN (65536).
DEFAULT_SIZES: tuple[int, ...] = tuple(
    2 ** k for k in range(3, 17)   # 8, 16, 32, ..., 65536
)


def _function_inputs(spec, function_name: str):
    """Return the FeatureDescription list for a function in a (multi)function
    model spec.  Handles both single-function and multifunction layouts."""
    if spec.description.functions:
        for fd in spec.description.functions:
            if fd.name == function_name:
                return list(fd.input)
        raise ValueError(
            f"function {function_name!r} not found in multifunction spec; "
            f"have {[fd.name for fd in spec.description.functions]}"
        )
    return list(spec.description.input)


def _flexible_dim_inputs(spec, function_name: str) -> list[tuple[str, list, int]]:
    """Return (input_name, default_shape, symbolic_dim_index) for inputs
    whose shape has a range on exactly one dim."""
    out = []
    for inp in _function_inputs(spec, function_name):
        arr = inp.type.multiArrayType
        if not (arr.shapeRange and arr.shapeRange.sizeRanges):
            continue
        shape = list(arr.shape)
        ranges = list(arr.shapeRange.sizeRanges)
        sym_idxs = [
            i for i, sr in enumerate(ranges)
            if sr.lowerBound != sr.upperBound
        ]
        if len(sym_idxs) != 1:
            raise NotImplementedError(
                f"Input {inp.name!r} has {len(sym_idxs)} symbolic dims; "
                "materialize currently supports exactly one."
            )
        out.append((inp.name, shape, sym_idxs[0]))
    return out


def _materialize_single_function(
    source_path: Path,
    dest_path: Path,
    sizes: Sequence[int],
    source_function_name: str,
    target_prefix: str,
) -> None:
    """Materialize one function of a (multi)function mlpackage into N concrete
    functions, saved as a new multifunction mlpackage at `dest_path`.

    The N target functions are named ``f"{target_prefix}_{size}"`` for each
    size in `sizes``.

    Inlines the core of ``coremltools.models.utils.materialize_dynamic_shape_mlmodel``
    because that helper hard-codes ``default_function_name = "main"`` after
    materialization, which breaks when the source is a non-"main" function
    in a multifunction mlpackage.
    """
    from coremltools.converters.mil.converter import mil_convert as _mil_convert
    from coremltools.converters.mil.frontend.milproto import load as _milproto_to_pymil
    from coremltools.converters.mil.mil.passes.pass_pipeline import (
        PassPipelineManager as _PassPipelineManager,
    )

    src_model = ct.models.MLModel(
        str(source_path),
        skip_model_load=True,
        function_name=source_function_name,
    )
    flexibles = _flexible_dim_inputs(src_model._spec, source_function_name)
    if not flexibles:
        raise RuntimeError(
            f"Source {source_path} function {source_function_name!r} has no "
            "flexible-dim inputs; nothing to materialize."
        )

    mat_map: dict[str, dict[str, tuple[int, ...]]] = {}
    for size in sizes:
        per_fn: dict[str, tuple[int, ...]] = {}
        for name, shape_tmpl, sym_dim in flexibles:
            concrete = list(shape_tmpl)
            concrete[sym_dim] = size
            per_fn[name] = tuple(concrete)
        mat_map[f"{target_prefix}_{size}"] = per_fn

    # Load the full pymil program (reuses _mil_program if already loaded).
    prog = (
        src_model._mil_program
        if src_model._mil_program is not None
        else _milproto_to_pymil.load(
            src_model._spec,
            src_model._spec.specificationVersion,
            src_model.weights_dir,
        )
    )

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
    pipeline.set_options(
        "common::materialize_symbolic_shape_program",
        {
            "function_name_to_materialization_map": mat_map,
            "source_function_name": source_function_name,
        },
    )
    _PassPipelineManager.apply_pipeline(prog, pipeline)

    # After materialization, point the default at one of the new functions.
    # (The upstream helper hard-codes "main", which breaks for non-"main"
    # source functions.)
    new_default = f"{target_prefix}_{max(sizes)}"
    prog.default_function_name = new_default
    prog.export_as_multifunction = len(mat_map) > 1 or new_default != source_function_name
    prog.skip_all_passes = True

    specification_version = src_model._spec.specificationVersion
    if prog.export_as_multifunction:
        specification_version = max(ct.target.iOS18, specification_version)

    out = _mil_convert(
        prog,
        convert_from="milinternal",
        convert_to="mlprogram",
        specification_version=specification_version,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    out.save(str(dest_path))


def _source_function_names(spec) -> list[str]:
    """Return the source function names in a (multi)function mlpackage."""
    names = [fd.name for fd in spec.description.functions]
    return names if names else ["main"]


def materialize_mlpackage(
    source_path: Path,
    dest_path: Path,
    sizes: Sequence[int] = DEFAULT_SIZES,
) -> None:
    """Materialize a dynamic-shape .mlpackage into a concrete multifunction.

    If the source has one function (e.g. from ``gemma-export --separate``),
    the output has ``size_{N}`` functions.  If the source is multifunction
    (prefill + decode), the output has ``{fname}_{N}`` functions for each
    source function ``fname`` and each ``N`` in `sizes`.
    """
    # Peek at spec to learn which functions exist.
    peek = ct.models.MLModel(str(source_path), skip_model_load=True)
    fn_names = _source_function_names(peek._spec)
    del peek

    with tempfile.TemporaryDirectory(prefix="gemma-mat-") as tmp:
        partials: list[tuple[str, Path]] = []
        for fname in fn_names:
            part = Path(tmp) / f"{fname}_mat.mlpackage"
            _materialize_single_function(
                source_path, part, sizes,
                source_function_name=fname,
                target_prefix=fname,
            )
            partials.append((fname, part))

        if len(partials) == 1 and len(sizes) == 1:
            # Trivial: one materialized package, just move it.
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.move(str(partials[0][1]), str(dest_path))
            return

        # Merge: one combined multifunction with ``{fname}_{size}`` functions.
        desc = ct.utils.MultiFunctionDescriptor()
        for fname, part in partials:
            for size in sizes:
                desc.add_function(
                    str(part),
                    src_function_name=f"{fname}_{size}",
                    target_function_name=f"{fname}_{size}",
                )
        # Pick the largest size of the first source function as default.
        first_fname = partials[0][0]
        desc.default_function_name = f"{first_fname}_{max(sizes)}"

        if dest_path.exists():
            shutil.rmtree(dest_path)
        ct.utils.save_multifunction(desc, str(dest_path))


def _parse_sizes(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"size {v} must be positive")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("no sizes given")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Materialize a dynamic-shape .mlpackage into a multifunction "
            "model with one concrete-shape function per power-of-2 size. "
            "Produces an ANE-compatible artifact (no dynamic shape ops)."
        )
    )
    p.add_argument(
        "--input", required=True,
        help="Source .mlpackage (must have RangeDim on KV inputs)",
    )
    p.add_argument(
        "--output", required=True,
        help="Destination .mlpackage",
    )
    p.add_argument(
        "--sizes", type=_parse_sizes, default=list(DEFAULT_SIZES),
        help=(
            "Comma-separated list of concrete cache sizes (default: "
            "powers of 2 from 8 to 65536)"
        ),
    )
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        print(f"error: input does not exist: {src}", file=sys.stderr)
        sys.exit(2)

    print(f"Materializing {src} → {dst}")
    print(f"  sizes: {args.sizes}")
    materialize_mlpackage(src, dst, args.sizes)
    final_size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"  done — {dst} ({final_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
