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

Defaults to sizes [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
matching the runtime doubling growth strategy.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

import coremltools as ct


# Default: powers of 2 from 512 to MAX_SEQ_LEN (65536).
# Starting at 512 avoids compiling tiny functions that would never be used
# in practice (model compile time dominates at small sizes).
DEFAULT_SIZES: tuple[int, ...] = tuple(
    2 ** k for k in range(9, 17)   # 512, 1024, 2048, ..., 65536
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


def _clear_weight_ids(prog) -> None:
    """Drop any ``weight_id`` attrs on const ops so the dedup pass can reassign
    them. ``const_deduplication._deduplicate_const_across_functions`` hard-errors
    if any const already has one set, and the milproto loader may partially
    rehydrate them depending on coremltools version."""
    for fn in prog.functions.values():
        for op in fn.operations:
            if op.op_type == "const" and getattr(op, "weight_id", None) is not None:
                op.weight_id = None


def _materialize_multifunction_source(
    source_path: Path,
    dest_path: Path,
    sizes: Sequence[int],
    source_function_names: Sequence[str],
) -> None:
    """Materialize every function of a (multi)function source into concrete
    per-size clones, loading the source pymil program **once**.

    For each ``src_fn`` in ``source_function_names``, produces
    ``{src_fn}_{size}`` target functions by running the
    ``materialize_symbolic_shape_program`` pass in-place on the same program.
    The original dynamic-shape source functions are dropped before save (the
    whole point of materializing is to shed RangeDim ops for ANE).

    Avoids the memory blow-up of the old "materialize each phase → then
    ``save_multifunction`` merge" flow: that merge loads each per-phase
    multifunction package back into pymil, and rehydration undoes on-disk
    weight sharing — each function gets freshly materialized numpy arrays,
    scaling RAM with ``phases × sizes_per_phase`` instead of the weight set.
    """
    from coremltools.converters.mil.converter import mil_convert as _mil_convert
    from coremltools.converters.mil.frontend.milproto import load as _milproto_to_pymil
    from coremltools.converters.mil.mil.passes.defs.cleanup.const_deduplication import (
        const_deduplication,
    )
    from coremltools.converters.mil.mil.passes.defs.symbol_transform import (
        materialize_symbolic_shape_program,
    )

    src_model = ct.models.MLModel(str(source_path), skip_model_load=True)
    spec = src_model._spec

    prog = (
        src_model._mil_program
        if src_model._mil_program is not None
        else _milproto_to_pymil.load(
            spec, spec.specificationVersion, src_model.weights_dir,
        )
    )

    # Re-establish cross-function const dedup on the loaded program. The
    # milproto loader doesn't preserve on-disk ``weight_id`` sharing (those
    # are a save-time construct), so prefill and decode come back with
    # independent const ops even where the bytes are identical. Running the
    # dedup pass NOW — before materialize clones those ops — assigns matching
    # weight_ids by content hash, and the materialize pass propagates them to
    # every concrete-shape clone. The final save then blob-shares across all
    # {prefill,decode}_{size} functions, keeping the on-device artifact the
    # size of one weight set instead of two.
    _clear_weight_ids(prog)
    const_deduplication()._deduplicate_const_across_functions(prog)

    src_specs: list[tuple[str, list]] = []
    for src_fn in source_function_names:
        flexibles = _flexible_dim_inputs(spec, src_fn)
        if not flexibles:
            raise RuntimeError(
                f"Source function {src_fn!r} has no flexible-dim inputs; "
                "nothing to materialize."
            )
        src_specs.append((src_fn, flexibles))

    for src_fn, flexibles in src_specs:
        mat_map: dict[str, dict[str, tuple[int, ...]]] = {}
        for size in sizes:
            per_fn: dict[str, tuple[int, ...]] = {}
            for name, shape_tmpl, sym_dim in flexibles:
                concrete = list(shape_tmpl)
                concrete[sym_dim] = size
                per_fn[name] = tuple(concrete)
            mat_map[f"{src_fn}_{size}"] = per_fn

        pass_obj = materialize_symbolic_shape_program()
        pass_obj.source_function_name = src_fn
        pass_obj.function_name_to_materialization_map = mat_map
        pass_obj.apply(prog)

    for src_fn, _ in src_specs:
        if src_fn in prog.functions:
            del prog.functions[src_fn]

    # Smallest decode (if present) as default — least work on load; matches
    # gemma-export's convention.
    if f"decode_{min(sizes)}" in prog.functions:
        prog.default_function_name = f"decode_{min(sizes)}"
    else:
        first_src = source_function_names[0]
        prog.default_function_name = f"{first_src}_{min(sizes)}"
    prog.export_as_multifunction = True
    prog.skip_all_passes = True

    specification_version = max(spec.specificationVersion, ct.target.iOS18)
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


def materialize_mlpackage(
    source_path: Path,
    dest_path: Path,
    sizes: Sequence[int] = DEFAULT_SIZES,
) -> None:
    """Materialize a dynamic-shape .mlpackage into a concrete multifunction.

    For a single-function (``main``-only) source, the output contains
    ``main_{N}`` functions. For a named-function or multifunction source
    (e.g. prefill + decode), the output contains ``{fname}_{N}`` functions
    for each source function ``fname`` and each ``N`` in ``sizes``.
    """
    peek = ct.models.MLModel(str(source_path), skip_model_load=True)
    has_named_functions = len(peek._spec.description.functions) > 0
    fn_names = _source_function_names(peek._spec)
    del peek

    if not has_named_functions:
        # Old-style single-function source.
        _materialize_single_function(
            source_path, dest_path, sizes,
            source_function_name="main",
            target_prefix=fn_names[0],
        )
    else:
        _materialize_multifunction_source(
            source_path, dest_path, sizes, fn_names,
        )


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
            "powers of 2 from 512 to 65536)"
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
