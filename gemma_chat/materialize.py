"""Convert a dynamic-shape `.mlpackage` into a multifunction one with concrete
power-of-2 KV-cache sizes.

Motivation
----------
The default export uses `RangeDim` on global KV caches for a 1..65k dynamic
context.  That produces MIL ops (`shape`, `range_1d`, `expand_dims`) whose
outputs flow into attention masking — and those ops are **only supported by
the GPU backend**.  CPU and ANE refuse to load the model.

At runtime the KV cache is grown by a factor of 2 on exhaustion, so in
practice only a handful of concrete sizes are ever observed: 512, 1024, …,
65536.  This utility takes a dynamic-shape model and materializes one
concrete-shape function per power of 2, via CoreML's built-in
`materialize_symbolic_shape_program` MIL pass.  The resulting multifunction
model has:

- **No** dynamic shape ops in any function (each is specialized to a concrete
  cache length, so the shape → range → mask chain folds to a constant).
- Deduplicated constants — all sizes share the same int4 weights.
- The global KV caches as Core ML **state** instead of I/O: concrete shapes are
  exactly what state features were missing, so `global_kv_caches_to_states`
  runs here, right after materialization.  Note this makes the state layout
  size-dependent — a state made from `prefill_512` fits only the `*_512`
  pair, and growing the cache means migrating contents into a new state.
- The cache *length* folded in as a constant: JAX's dimension-variable argument
  `N` is a value, not a shape, so materialization leaves it a runtime input and
  the global attention mask symbolic.  `concretize_cache_length` replaces it,
  which makes every shape concrete and drops `N` from the signature.  The
  attention fusion is deliberately *not* re-run afterwards — see
  `_concretize_cache_lengths` for the two Apple defects that keeps us clear of.

The runtime picks the function whose size matches the current cache length.

Note: as of macOS 26 / current coremltools (2026) the CPU/ANE load failure no
longer reproduces on macOS — the dynamic-shape model loads on `CPU_ONLY` there.
Materialization still matters for iOS/ANE and older OS versions.

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

import gemma_chat.weight_shards  # noqa: F401  — caps blob files below 2 GiB
from gemma_chat.mil_passes.concretize_cache_length import concretize_cache_length
from gemma_chat.mil_passes.global_cache_states import global_kv_caches_to_states


# Default: powers of 2 from 512 to MAX_SEQ_LEN (65536).
# Starting at 512 avoids compiling tiny functions that would never be used
# in practice (model compile time dominates at small sizes).
DEFAULT_SIZES: tuple[int, ...] = tuple(
    2 ** k for k in range(9, 17)   # 512, 1024, 2048, ..., 65536
)


def _concretize_cache_lengths(prog, function_name_to_length: dict[str, int]) -> None:
    """Fold each function's cache length in, then clean up.

    Folding ``N`` in is load-bearing regardless of attention fusion: it is what
    makes the global attention mask (``range_1d(end=N)``,
    ``fill(shape=[1, H, 1, N])``) concrete, so no function keeps a symbolic
    shape, and it is what removes ``N`` from the signature the Swift runtime
    binds against.  ``dead_code_elimination`` then drops what folding it in made
    unreachable (it keeps ``coreml_update_state`` explicitly, so the cache
    writes survive).

    **Why the attention fusion is deliberately NOT re-run here.**  Concrete
    shapes would let ``common::fuse_attention_to_sdpa`` finally collect the
    *global* attention sites it had to skip during export (it bails on symbolic
    dimensions; the *sliding* sites were always concrete, are fused at convert
    time, and stay fused — they are not affected by either defect below).
    Fusing the global sites trips two Apple bugs, both on macOS 26.5:

    1. **ANE partitioner.** Once a function holds two or more global SDPAs,
       ANECCompile() fails on the segment containing the *deepest* one with
       "live input tensor not used in network", and the whole model silently
       drops off the Neural Engine.  Bisected on truncated exports: a 9-layer
       model (one global layer) compiles every procedure; the 10-layer model
       (two global layers) stops one procedure short and logs that message —
       scratchpad artifacts ``bisect/L9.*`` vs ``bisect/L10.*``.
    2. **BNNS.** A fused global SDPA SIGSEGVs when a CPU segment executes it
       with query length >= 2 — i.e. every chunk-128 prefill, on ``cpu-only``
       and on the CPU segments of ``cpu-and-ne``.  Minimal repro:
       ``scratchpad/bisect/synth_sdpa3.py 8 128 512 512 mq`` (one fp16 SDPA at
       the global site's shape, with the attention-scale ``mul`` feeding the
       query) exits 139.  ``Lq=1`` passes and ``Lq=2`` crashes; the ``+f32``
       and ``+decomp`` variants pass, and so does ``plain`` — the producer op
       on the query is part of the trigger.

    Leaving the global sites as the ``matmul → add(mask) → softmax → matmul``
    they already are avoids both.  When Apple fixes either defect, re-running
    ``common::fuse_attention_to_sdpa`` here (plus DCE) is all it takes to get
    the fused form back — benchmark it against the decomposed form first.
    """
    from coremltools.converters.mil.mil.passes.pass_pipeline import (
        PassPipelineManager as _PassPipelineManager,
    )

    pass_obj = concretize_cache_length()
    pass_obj.function_name_to_length = function_name_to_length
    pass_obj.apply(prog)

    pipeline = ct.PassPipeline.EMPTY
    pipeline.append_pass("common::dead_code_elimination")
    _PassPipelineManager.apply_pipeline(prog, pipeline)


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
    kept_weight_ids = _weight_ids(prog)
    _PassPipelineManager.apply_pipeline(prog, pipeline)
    _scope_auto_weight_ids(prog, source_function_name, target_prefix, kept_weight_ids)

    # Now that every function has concrete shapes, the global KV caches can
    # become Core ML state like the sliding ones already are.
    global_kv_caches_to_states().apply(prog)

    # The cache *length* is still a runtime input; fold it in.
    _concretize_cache_lengths(
        prog, {f"{target_prefix}_{size}": size for size in sizes}
    )

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


def _weight_ids(prog) -> set[str]:
    """Return the ``weight_id``s currently set on any const in the program."""
    return {
        op.weight_id
        for fn in prog.functions.values()
        for op in fn.operations
        if op.op_type == "const" and getattr(op, "weight_id", None) is not None
    }


def _scope_auto_weight_ids(
    prog, source_function_name: str, target_prefix: str, kept: set[str],
) -> None:
    """Make the materialize pass's invented ``weight_id``s unique per source fn.

    ``materialize_symbolic_shape_program`` gives every const it clones that has
    no ``weight_id`` one derived from the const's *name* alone
    (``const_{name}_weight_id``).  Const names are only unique within a
    function, so two source functions that each hold a differently-shaped const
    of the same name — ``prefill``'s ``range_1d_0`` is ``int32[CHUNK_SIZE]``,
    ``decode``'s is ``int32[sliding_window]`` — end up claiming one weight-blob
    entry.  The blob then holds one of them and the other function declares a
    shape the blob cannot supply, so the model fails to load with
    *"Attribute val has incompatible type with operation output"*.

    (Nothing caught this before CHUNK_SIZE grew: consts of fewer than 10
    elements are stored inline in the proto and never reach the blob at all.)

    Rewriting those ids to include the source function name keeps every clone of
    one source const sharing a blob entry — all sizes of a phase are clones —
    while separating the phases.  Ids in ``kept`` (assigned by
    ``const_deduplication``, which groups by dtype + shape + value) are left
    alone; those are the ones that must keep sharing across phases, and they are
    the reason the artifact is the size of one weight set rather than two.
    """
    for fname, fn in prog.functions.items():
        # The clones, plus the source itself when it survives to the save.
        if fname != source_function_name and not fname.startswith(f"{target_prefix}_"):
            continue
        for op in fn.operations:
            if op.op_type != "const":
                continue
            weight_id = getattr(op, "weight_id", None)
            if weight_id is None or weight_id in kept:
                continue
            # The public setter is write-once (it asserts the id is unset), so
            # clear the backing field before re-scoping.
            op._weight_id = None
            op.weight_id = f"{source_function_name}::{op.name}"


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

    kept_weight_ids = _weight_ids(prog)
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
        _scope_auto_weight_ids(prog, src_fn, src_fn, kept_weight_ids)

    for src_fn, _ in src_specs:
        if src_fn in prog.functions:
            del prog.functions[src_fn]

    # Concrete shapes everywhere now, so the global KV caches can become Core ML
    # state like the sliding ones already are.  Must run after the source
    # functions are dropped — those still carry symbolic cache lengths.
    global_kv_caches_to_states().apply(prog)

    # The cache *length* is still a runtime input; fold it in.
    _concretize_cache_lengths(
        prog,
        {f"{src_fn}_{size}": size for src_fn, _ in src_specs for size in sizes},
    )

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
