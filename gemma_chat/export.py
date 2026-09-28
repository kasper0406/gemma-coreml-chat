"""Export Gemma4-E2B as a CoreML multifunction .mlpackage of layer chunks.

Every prefill / decode step runs as a sequence of **layer-chunk functions**
(``decode_coreml.layer_chunks``) followed by one shared ``head``: the Neural
Engine takes no function past a size limit, and one too-large function takes
the whole package off it.  The package holds, per cache size ``N``:

* ``prefill_c<k>_<N>`` / ``decode_c<k>_<N>`` — layer chunk ``k``: the hidden
  state in (``token_embed`` itself for chunk 0), the hidden state out
  (final-normed after the last chunk);
* ``state_<N>`` — declares every KV cache and reads a sliver of each; never
  predicted in the hot path, it only exists so the runtime can make the one
  ``MLState`` all chunks of that size share (Core ML shares states across a
  package's functions by name, and a function may declare a subset of them);

and, size-independent:

* ``head`` — final-normed hidden ``[1, 1, D]`` → fp32 logits, used by decode
  and (on the row the runtime picks) prefill.

Matmul weights are quantized with per-channel scales (blockwise ones are not
ANE-eligible): int4 in the layer chunks, int8 in ``head``, whose vocab is
split into slices that keep the Neural Engine's weight DMA fast
(``decode_coreml.head_slices``) — see ``mil_passes/quantize_const_weights`` —
and deduplicated across functions.

The embedding lookups are not in the graph: the functions take the embedding
rows (``token_embed``, ``ple_rows``) and the runtime looks them up in the
block-32 int4 tables this exporter ships in the package's ``Embeddings/``
directory, next to ``Tokenizer/`` — see ``gemma_chat.host_embeddings``.

All 15 KV caches are Core ML **state**.  The sliding ones are static-shaped, so
the StableHLO→MIL converter binds them directly (:func:`_chunk_io_plan`); the
global ones carry a symbolic length through conversion — a state cannot have a
flexible shape — and become state once materialization has given every
function a concrete cache length (``mil_passes.global_cache_states``).  The
int32 ``sliding_pos_ring`` stays an input (states must be floating point) and
the runtime updates it itself.

``--no-materialize`` keeps the dynamic-shape functions, which do not load (a
RangeDim program that declares states fails with E5RT/BNNS errors); it is only
useful for inspecting the converted program.

Usage:
    uv run gemma-export
    uv run gemma-export --output gemma4-e2b.mlpackage
    uv run gemma-export --materialize-sizes 512,1024,2048,4096
    uv run gemma-export --skip-warmup     # save RAM on constrained machines
"""

from __future__ import annotations

import argparse
import gc
import shutil
import subprocess
import sys
import os as _os
import signal as _signal

import dataclasses
from pathlib import Path
import numpy as np

import gemma_chat.weight_shards  # noqa: F401  — caps blob files below 2 GiB


def _inplace_bf16_to_f16(d: dict) -> None:
    """Recursively convert bfloat16 numpy leaves to float16 in-place."""
    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            _inplace_bf16_to_f16(v)
            del v
        elif hasattr(v, 'dtype') and v.dtype.name == 'bfloat16':
            d[k] = v.astype(np.float16)   # allocate new f16
            del v                          # free old bf16 immediately
        else:
            del v


def _rss_mb() -> float:
    """Current (not peak) RSS in MB."""
    try:
        # macOS: read from ps for current RSS (ru_maxrss is peak only).
        import subprocess as _sp
        out = _sp.check_output(
            ["ps", "-o", "rss=", "-p", str(_os.getpid())],
            text=True,
        ).strip()
        return int(out) / 1024  # ps gives KB on macOS
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        except Exception:
            return 0.0


def _release_malloc() -> None:
    """Force macOS malloc to return freed memory to the OS."""
    gc.collect()
    gc.collect()
    try:
        import ctypes as _c, ctypes.util as _cu
        _c.CDLL(_cu.find_library("c")).malloc_zone_pressure_relief(
            _c.c_void_p(0), _c.c_size_t(0),
        )
    except Exception:
        pass


def _signal_handler(signum, frame):
    print(f"  [SIGNAL] Received signal {signum}  RSS={_rss_mb():.0f} MB", flush=True)
    import traceback
    traceback.print_stack(frame)
    _os._exit(signum)


_signal.signal(_signal.SIGTERM, _signal_handler)

import jax
import jax.numpy as jnp

import coremltools as ct
from stablehlo_coreml import StateSpec
from stablehlo_coreml.converter import convert as hlo_to_mil

from gemma_chat.config import CHUNK_SIZE, HF_MODEL_ID, MAX_SEQ_LEN, VARIANTS
from gemma_chat.model import Gemma4Transformer, Gemma4Config, AttentionType
from gemma_chat.weight_mapper import load_params
from gemma_chat.decode_coreml import (
    LayerChunk, decode_chunk, layer_chunks, logits_head, prefill_chunk,
)
from gemma_chat.cache_spec import build_cache_specs, sliding_ring_length
from gemma_chat import host_embeddings


# ── Truncated config / params for --num-layers ────────────────────────────


def _truncated_config(cfg: Gemma4Config, num_layers: int) -> Gemma4Config:
    """Return a copy of *cfg* truncated to *num_layers*.

    KV sharing is disabled (the truncated model is too short for the shared
    tail).  ``wide_mlp_from_layer`` is clamped so layers beyond the cutoff
    don't widen.
    """
    if num_layers >= cfg.num_layers:
        return cfg
    import dataclasses
    return dataclasses.replace(
        cfg,
        attention_types=cfg.attention_types[:num_layers],
        num_kv_shared_layers=0,
        wide_mlp_from_layer=(
            cfg.wide_mlp_from_layer
            if cfg.wide_mlp_from_layer >= 0 and cfg.wide_mlp_from_layer < num_layers
            else -1
        ),
    )


def _truncate_params(params: dict, num_layers: int, ple_dim: int) -> dict:
    """Drop layers beyond *num_layers* and slice PLE embedding in-place."""
    # Remove per-layer dicts for layers we don't need
    for i in list(params.keys()):
        if i.startswith("layers."):
            idx = int(i.split(".")[1])
            if idx >= num_layers:
                del params[i]
    # Slice PLE embedding: (vocab, full_layers*d) → (vocab, num_layers*d)
    full = params["embed_tokens_per_layer"]
    params["embed_tokens_per_layer"] = full[:, :num_layers * ple_dim]
    # Slice per_layer_model_projection: (D, full_layers*d) → (D, num_layers*d)
    proj = params["per_layer_model_projection"]["kernel"]
    params["per_layer_model_projection"]["kernel"] = proj[:, :num_layers * ple_dim]
    return params


# ── Function I/O plans ─────────────────────────────────────────────────────


@dataclasses.dataclass
class _IOPlan:
    """How one traced function's arguments and results map onto Core ML.

    ``arg_specs`` are the trace specs in argument order, **excluding** JAX's
    leading dimension-variable argument ``N`` (present iff ``has_global``).
    ``states`` maps traced-argument indices (counting ``N``) to
    :class:`StateSpec`; ``input_names`` / ``output_names`` name the remaining
    inputs and outputs, in order.  ``flexible`` are the input names whose dim 1
    is the symbolic global cache length (their ``_out`` outputs too).
    """
    arg_specs: list
    has_global: bool
    states: dict[int, StateSpec]
    input_names: list[str]
    output_names: list[str]
    flexible: list[str]


def _cache_spec_for(config: Gemma4Config, slot: int, N):
    """Trace spec of cache slot ``slot``: sliding ones are ``(1, R, nkv, hd)``
    (``R = sliding_ring_length``), global ones ``(1, N, nkv, hd)`` with ``N``
    symbolic."""
    spec = build_cache_specs(config, 1)[slot]
    length = N if spec.attn_type == AttentionType.GLOBAL else sliding_ring_length(config)
    return jax.ShapeDtypeStruct((1, length, spec.num_kv_heads, spec.head_dim), jnp.float16)


def _is_global_slot(config: Gemma4Config, slot: int) -> bool:
    return build_cache_specs(config, 1)[slot].attn_type == AttentionType.GLOBAL


def _chunk_io_plan(
    config: Gemma4Config, chunk: LayerChunk, first: bool, tokens: int, N,
) -> _IOPlan:
    """The signature of one layer-chunk function.

    Arguments, in order: ``[N] + leading + [k_s, v_s for s in chunk.slots] +
    [sliding_pos_ring]``, where ``leading`` is ``[hidden]`` (all but the first
    chunk) ``+ [token_embed, ple_rows, position]``.  Results:
    ``[hidden_out] + [k_s_out, v_s_out for s in chunk.writes]``.

    Sliding caches become state here — written back from their result if the
    chunk owns them, read-only otherwise.  Global caches stay I/O with a
    symbolic length (``k_s`` in, ``k_s_out`` out when written) until
    ``mil_passes.global_cache_states`` converts them after materialization.
    Names are ``k_<slot>`` / ``v_<slot>`` throughout; that pass relies on it.
    """
    d = config.per_layer_input_dim
    D = config.embed_dim
    leading = (["hidden"] if not first else []) + ["token_embed", "ple_rows", "position"]
    arg_specs = (
        ([jax.ShapeDtypeStruct((1, tokens, D), jnp.float16)] if not first else [])
        + [
            jax.ShapeDtypeStruct((1, tokens, D), jnp.float16),
            jax.ShapeDtypeStruct((1, tokens, len(chunk.layers) * d), jnp.float16),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ]
    )
    has_global = any(_is_global_slot(config, s) for s in chunk.slots)
    uses_ring = any(
        config.attention_types[i] == AttentionType.LOCAL_SLIDING for i in chunk.layers
    )
    base = (1 if has_global else 0) + len(leading)
    written = {s: 1 + 2 * j for j, s in enumerate(chunk.writes)}  # k result index

    states: dict[int, StateSpec] = {}
    input_names = (["N"] if has_global else []) + leading
    output_names = ["hidden_out"]
    flexible: list[str] = []
    for j, slot in enumerate(chunk.slots):
        for half, prefix in enumerate(("k", "v")):
            name = f"{prefix}_{slot}"
            arg_specs.append(_cache_spec_for(config, slot, N))
            out = written[slot] + half if slot in written else None
            if _is_global_slot(config, slot):
                input_names.append(name)
                flexible.append(name)
                if out is not None:
                    output_names.append(name + "_out")
            else:
                states[base + 2 * j + half] = StateSpec(output=out, name=name)
    if uses_ring:
        arg_specs.append(jax.ShapeDtypeStruct((1, sliding_ring_length(config)), jnp.int32))
        input_names.append("sliding_pos_ring")
    return _IOPlan(arg_specs, has_global, states, input_names, output_names, flexible)


def _state_io_plan(config: Gemma4Config, N) -> _IOPlan:
    """The ``state`` function: every cache as an argument, all read-only."""
    slots = range(len(build_cache_specs(config, 1)))
    has_global = any(_is_global_slot(config, s) for s in slots)
    base = 1 if has_global else 0
    arg_specs, states, flexible = [], {}, []
    input_names = ["N"] if has_global else []
    for slot in slots:
        for half, prefix in enumerate(("k", "v")):
            name = f"{prefix}_{slot}"
            arg_specs.append(_cache_spec_for(config, slot, N))
            if _is_global_slot(config, slot):
                input_names.append(name)
                flexible.append(name)
            else:
                states[base + 2 * slot + half] = StateSpec(output=None, name=name)
    return _IOPlan(arg_specs, has_global, states, input_names, ["probe"], flexible)


def _host_tables_dir(phase_output: Path) -> Path:
    """Where the decode phase leaves the host embedding tables for the parent."""
    return Path(phase_output).with_suffix(".embeddings")


def _rename_model_io(
    cml_model,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Rename inputs and outputs on a single-function CoreML model (in-place).

    Uses ``ct.utils.rename_feature`` which updates both the spec-level
    ``FeatureDescription`` names **and** the MLProgram function input/output
    names.  This is required for ``save_multifunction`` — its validator
    checks that spec names and MLProgram names match.

    State features are *not* covered here: they live in ``description.state``,
    not ``description.input``/``.output``, and already carry their final names
    from the ``StateSpec(name=...)`` handed to the converter.  So the two name
    lists must enumerate only the non-state inputs/outputs.
    """
    from coremltools.models.utils import rename_feature

    spec = cml_model._spec
    desc = spec.description
    if len(input_names) != len(desc.input):
        raise ValueError(
            f"input_names length {len(input_names)} != spec inputs {len(desc.input)}"
        )
    if len(output_names) != len(desc.output):
        raise ValueError(
            f"output_names length {len(output_names)} != spec outputs {len(desc.output)}"
        )
    # Rename inputs (old positional _argN → meaningful name).
    for feat, new_name in zip(list(desc.input), input_names):
        if feat.name != new_name:
            rename_feature(spec, feat.name, new_name, rename_outputs=False)
    # Rename outputs (old MIL op names → meaningful name).
    for feat, new_name in zip(list(desc.output), output_names):
        if feat.name != new_name:
            rename_feature(spec, feat.name, new_name, rename_inputs=False)


def _hlo_to_mil_streaming(hlo_module, states: dict[int, StateSpec], weight_bits: int):
    """Convert StableHLO to MIL, quantizing weights as they stream past.

    ``states`` maps traced-argument indices to :class:`StateSpec` — see
    :func:`_chunk_io_plan`.  Those arguments become Core ML state features and
    disappear from the model's inputs/outputs.  Every weight is quantized
    per-channel, ``weight_bits`` wide.

    Returns the MIL program.  The caller should ``del hlo_module`` after
    this returns to free the MLIR IR (~3.5 GB) before the heavier
    ct.convert + save steps.
    """
    import ctypes as _ctypes, ctypes.util as _ctypes_util

    import numpy as np
    from coremltools.converters.mil import Builder as mb

    from gemma_chat.mil_passes.quantize_const_weights import _quantize_weight
    from gemma_chat.stablehlo_streaming_patch import (
        install_stablehlo_streaming_patch,
        set_streaming_quantizer,
    )
    try:
        _libc = _ctypes.CDLL(_ctypes_util.find_library("c"))
        _libc.malloc_zone_pressure_relief(_ctypes.c_void_p(0), _ctypes.c_size_t(0))
        print(
            f"  [convert] malloc_zone_pressure_relief done  RSS={_rss_mb():.0f} MB",
            flush=True,
        )
    except Exception as _e:
        print(f"  [convert] malloc_zone_pressure_relief skipped: {_e}", flush=True)

    # ── Streaming quantization during HLO→MIL ──
    # Returning None hands the constant back to the converter untouched, which
    # emits it as a plain ``mb.const`` — that is how a weight opts out.
    _WEIGHT_THRESHOLD = 2048
    _stream_counter = [0, 0]  # [count, total_bytes]

    def _stream_quantize(arr: np.ndarray, name: str):
        if arr.ndim < 2 or arr.size <= _WEIGHT_THRESHOLD:
            return None
        if arr.dtype not in (np.float16, np.float32):
            return None
        # Weights reach here as fp16 (``_inplace_bf16_to_f16`` runs before the
        # trace); the downcast is just a guard.  There used to be a matching
        # ``mb.cast(..., dtype="fp32")`` on the way out for weights whose
        # consumer was fp32 — that is gone: no fp32 weight survives the trace,
        # and the 35 o_proj weights that *were* re-materialized as fp32 at
        # runtime (528 MB) got there from JAX's own dot-promotion, not from
        # here.  With the graph fp16 end to end they are consumed as fp16.
        arr = arr.astype(np.float16, copy=False)
        q_data, scale = _quantize_weight(arr, weight_bits)
        _stream_counter[0] += 1
        _stream_counter[1] += arr.nbytes
        del arr
        if _stream_counter[0] % 20 == 0:
            print(
                f"    streaming-quantized {_stream_counter[0]} int{weight_bits}  "
                f"({_stream_counter[1] / 1e9:.2f} GB)  RSS={_rss_mb():.0f} MB",
                flush=True,
            )
        return mb.constexpr_blockwise_shift_scale(
            data=q_data, scale=scale, name=f"{name}_int{weight_bits}",
        )

    install_stablehlo_streaming_patch()
    set_streaming_quantizer(_stream_quantize)
    print(f"  [convert] streaming int{weight_bits} quantization enabled", flush=True)

    print(
        f"  [convert {_os.getpid()}] hlo_to_mil ({len(states)} state args) …",
        flush=True,
    )
    try:
        mil_program = hlo_to_mil(
            hlo_module, minimum_deployment_target=ct.target.iOS18, states=states,
        )
    finally:
        set_streaming_quantizer(None)

    if _stream_counter[0]:
        print(
            f"  StableHLO→MIL done — streaming-quantized "
            f"{_stream_counter[0]} tensors to int{weight_bits} "
            f"({_stream_counter[1] / 1e9:.2f} GB fp16).",
            flush=True,
        )
    else:
        print(f"  StableHLO→MIL conversion done.", flush=True)

    return mil_program


def _mil_to_mlpackage(
    mil_program,
    output_path: Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    flexible_shapes: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Run ct.convert → rename → flex shapes → save.

    flexible_shapes: maps input names (after renaming) to ``(lower, upper)``
        bounds for dimension 1. Applied directly to the protobuf spec after
        rename. This bypasses ``ct.convert(inputs=...)`` which cannot match
        names generated by the stablehlo converter.
    """
    import threading
    import traceback as _tb

    from gemma_chat.mil_passes.ct_convert_pipeline import build_ct_convert_pass_pipeline

    # ── MIL pass pipeline (quantize_const_weights as belt-and-suspenders) ──
    pipeline = build_ct_convert_pass_pipeline()
    print(f"  [convert] ct.convert …", flush=True)

    _stop = threading.Event()

    def _monitor():
        while not _stop.wait(timeout=2.0):
            print(f"  [convert-rss] {_rss_mb():.0f} MB", flush=True)

    threading.Thread(target=_monitor, daemon=True).start()
    try:
        try:
            convert_kwargs = dict(
                source="milinternal",
                minimum_deployment_target=ct.target.iOS18,
                # FLOAT32 here means "leave the dtypes alone", not "compute in
                # fp32": the traced graph already places fp16 and fp32 by hand
                # (see the precision note in ``decode_coreml``), and this is the
                # only setting that preserves that placement — see the
                # ``add_fp16_cast`` note in ``mil_passes/ct_convert_pipeline.py``.
                compute_precision=ct.precision.FLOAT32,
                pass_pipeline=pipeline,
                skip_model_load=True,
            )
            cml_model = ct.convert(mil_program, **convert_kwargs)
        except BaseException as _e:
            print(f"\n!!! [convert] ct.convert raised {type(_e).__name__}: {_e}", flush=True)
            _tb.print_exc()
            raise
    finally:
        _stop.set()

    del mil_program, pipeline
    _release_malloc()
    print(f"  [convert] ct.convert done  RSS={_rss_mb():.0f} MB", flush=True)

    if input_names or output_names:
        _rename_model_io(cml_model, input_names or [], output_names or [])

    # Apply flexible shape ranges directly to the protobuf spec.
    if flexible_shapes:
        spec = cml_model._spec
        # Build output lookup: "k_4_out" → ("k_4" range)
        out_flex = {}
        for name, bounds in flexible_shapes.items():
            out_flex[name + "_out"] = bounds

        def _apply_flex(feature_desc, lookup):
            if feature_desc.name not in lookup:
                return
            lo, hi = lookup[feature_desc.name]
            arr = feature_desc.type.multiArrayType
            # Fix empty output shapes: set shape to [1, lo, ...] default
            if len(arr.shape) == 0:
                # Need a concrete shape for the spec. Find matching input.
                for inp in spec.description.input:
                    if inp.name in flexible_shapes and inp.name == feature_desc.name.removesuffix("_out"):
                        for d in inp.type.multiArrayType.shape:
                            arr.shape.append(d)
                        break
            arr.ClearField("shapeRange")
            for dim_idx, dim_size in enumerate(arr.shape):
                sr = arr.shapeRange.sizeRanges.add()
                if dim_idx == 1:
                    sr.lowerBound = lo
                    sr.upperBound = hi
                else:
                    sr.lowerBound = dim_size
                    sr.upperBound = dim_size

        for inp in spec.description.input:
            _apply_flex(inp, flexible_shapes)
        for out in spec.description.output:
            _apply_flex(out, out_flex)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [convert {_os.getpid()}] saving to {output_path} …", flush=True)
    try:
        cml_model.save(str(output_path))
    except BaseException as _e:
        print(f"\n!!! [convert] save raised {type(_e).__name__}: {_e}", flush=True)
        _tb.print_exc()
        raise
    print(f"  [convert {_os.getpid()}] saved → {output_path.resolve()}", flush=True)
    del cml_model


# ── Phase export ───────────────────────────────────────────────────────────


def _export_function(fn, plan: _IOPlan, output_path: Path, weight_bits: int = 4) -> None:
    """Trace ``fn`` against ``plan``, convert it and save one .mlpackage, its
    weights quantized per-channel to ``weight_bits``."""
    print(f"  Tracing {output_path.stem} …", flush=True)
    hlo_module = jax.jit(fn).trace(*plan.arg_specs).lower().compiler_ir("stablehlo")
    mil_program = _hlo_to_mil_streaming(hlo_module, plan.states, weight_bits)
    del hlo_module
    _release_malloc()
    _mil_to_mlpackage(
        mil_program, output_path,
        input_names=plan.input_names,
        output_names=plan.output_names,
        flexible_shapes={name: (1, MAX_SEQ_LEN) for name in plan.flexible},
    )
    jax.clear_caches()
    _release_malloc()


def export_phase(
    phase: str,
    output_dir: str | Path,
    model_id: str = HF_MODEL_ID,
    variant: str = "e2b",
    skip_warmup: bool = False,
    num_layers: int | None = None,
) -> None:
    """Export one phase's functions as single-function packages in ``output_dir``.

    ``prefill``: ``prefill_c<k>.mlpackage`` per layer chunk, ``CHUNK_SIZE``
    tokens per call.  ``decode``: ``decode_c<k>.mlpackage`` per layer chunk,
    ``head.mlpackage`` and ``state.mlpackage``, plus the host embedding tables
    in :func:`_host_tables_dir`.  The global caches keep a symbolic length
    here; the parent merges everything and materializes one function per size.
    """
    from jax import export as jax_export

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = VARIANTS[variant][0]
    full_num_layers = config.num_layers
    if num_layers is not None and num_layers < config.num_layers:
        config = _truncated_config(config, num_layers)
        print(f"  Truncated config to {config.num_layers} layers: "
              f"{[a[:3] for a in config.attention_types]}")

    print("=" * 60)
    print(f"{phase} export — loading weights")
    print("=" * 60)
    params = load_params(model_id=model_id, config=config)
    if num_layers is not None and num_layers < full_num_layers:
        _truncate_params(params, num_layers, config.per_layer_input_dim)

    if not skip_warmup:
        from flax import nnx
        from gemma_chat.weight_mapper import load_params_into_model
        model_tmp = Gemma4Transformer(config=config, rngs=nnx.Rngs(params=0))
        load_params_into_model(model_tmp, params, config)
        _ = model_tmp(jnp.ones((1, 8), dtype=jnp.int32))
        del model_tmp
        print("  Eager warmup OK", flush=True)

    print("  Converting params to float16 …", flush=True)
    _inplace_bf16_to_f16(params)
    # The host looks both embeddings up, from tables quantized exactly as the
    # graph used to quantize them.  The per-layer table leaves the graph
    # entirely; the token table stays, as the (tied) logit head.
    ple_table = params.pop("embed_tokens_per_layer")
    if phase == "decode":
        print("  Writing host embedding tables …", flush=True)
        host_embeddings.write_tables(
            {"token_embed": params["embed_tokens"], "ple_rows": ple_table},
            config.embed_dim,
            _host_tables_dir(output_dir),
        )
    del ple_table
    _release_malloc()

    (N,) = jax_export.symbolic_shape("N", constraints=[f"N >= {CHUNK_SIZE}"])
    tokens = CHUNK_SIZE if phase == "prefill" else 1
    step = prefill_chunk if phase == "prefill" else decode_chunk

    gc.disable()
    try:
        chunks = layer_chunks(config)
        for k, chunk in enumerate(chunks):
            print("=" * 60)
            print(f"{phase} chunk {k}/{len(chunks)}: layers {chunk.layers.start}–"
                  f"{chunk.layers.stop - 1}, writes {list(chunk.writes)}, "
                  f"reads {list(chunk.reads)}")
            print("=" * 60)
            plan = _chunk_io_plan(config, chunk, first=k == 0, tokens=tokens, N=N)

            # JAX adds the dimension variable ``N`` itself; the Python
            # function only sees the arguments in ``plan.arg_specs``.
            def chunk_fn(*args, chunk=chunk, first=k == 0):
                if first:
                    token_embed, ple_rows, position, *rest = args
                    hidden = token_embed
                else:
                    hidden, token_embed, ple_rows, position, *rest = args
                caches = {s: (rest[2 * j], rest[2 * j + 1]) for j, s in enumerate(chunk.slots)}
                ring = rest[2 * len(chunk.slots)] if len(rest) > 2 * len(chunk.slots) else None
                hidden, written = step(
                    params, chunk, hidden, token_embed, ple_rows, position[0],
                    caches, ring, cfg=config,
                )
                return (hidden,) + tuple(c for s in chunk.writes for c in written[s])

            _export_function(chunk_fn, plan, output_dir / f"{phase}_c{k}.mlpackage")

        if phase == "decode":
            head_plan = _IOPlan(
                [jax.ShapeDtypeStruct((1, 1, config.embed_dim), jnp.float16)],
                False, {}, ["hidden"], ["logits"], [],
            )
            # int8: int4 is too lossy for the logits (see ``logits_head``).
            _export_function(
                lambda hidden: logits_head(params, hidden, config),
                head_plan, output_dir / "head.mlpackage", weight_bits=8,
            )

            state_plan = _state_io_plan(config, N)

            def state_fn(*caches):
                return jnp.stack([c[0, 0, 0, 0] for c in caches])

            _export_function(state_fn, state_plan, output_dir / "state.mlpackage")
    finally:
        gc.enable()


def _embed_tokenizer(model_id: str, mlpackage_path: Path) -> None:
    """Download tokenizer files from HuggingFace and embed them into the .mlpackage."""
    import json
    import shutil as _shutil
    from huggingface_hub import hf_hub_download

    tok_dir = mlpackage_path / "Tokenizer"
    tok_dir.mkdir(exist_ok=True)

    for fname in ("tokenizer.json", "tokenizer_config.json"):
        local = hf_hub_download(repo_id=model_id, filename=fname)
        dst = tok_dir / fname
        _shutil.copy2(local, dst)
        sz = dst.stat().st_size
        print(f"  Embedded {fname} ({sz / 1e6:.1f} MB)")

    # swift-transformers requires config.json with model_type to load the tokenizer
    config_path = tok_dir / "config.json"
    config_path.write_text(json.dumps({"model_type": "gemma"}, indent=2))
    print(f"  Wrote config.json (model_type=gemma)")

    print(f"  Tokenizer stored in {tok_dir}/")


def _embed_host_tables(tables_dir: Path, mlpackage_path: Path) -> None:
    """Move the decode phase's host embedding tables into the .mlpackage."""
    dst = mlpackage_path / host_embeddings.DIR_NAME
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(tables_dir), str(dst))
    print(f"  Host embedding tables stored in {dst}/")


# ── Entry point ─────────────────────────────────────────────────────────────


def _parse_materialize_sizes(s: str | None) -> list[int]:
    """Parse a comma-separated list of sizes, or default to powers of 2."""
    from gemma_chat.materialize import DEFAULT_SIZES
    if s is None:
        return list(DEFAULT_SIZES)
    out = [int(tok) for tok in s.split(",") if tok.strip()]
    if not out:
        raise argparse.ArgumentTypeError("--materialize-sizes: empty list")
    if any(v <= 0 for v in out):
        raise argparse.ArgumentTypeError("--materialize-sizes: values must be positive")
    return sorted(set(out))


def main() -> None:
    import tempfile

    parser = argparse.ArgumentParser(
        description=(
            "Export Gemma4-E2B as a CoreML .mlpackage of layer-chunk functions "
            "(prefill + decode, one set per cache size) plus a shared logit head."
        )
    )
    parser.add_argument(
        "--variant",
        default="e2b",
        choices=sorted(VARIANTS),
        help="Model variant to export (default: e2b)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: the variant's, e.g. gemma4-e2b.mlpackage)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HuggingFace model ID (default: the variant's)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Largest default materialized size (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip eager XLA warmup to save ~2 GB RAM (use on memory-constrained machines)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Truncate model to this many layers (for fast iteration / testing)",
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Export only the decode functions (skip prefill)",
    )
    parser.add_argument(
        "--materialize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Specialize every function to concrete cache sizes (default).  "
            "--no-materialize keeps the dynamic-shape functions, which do not "
            "load (E5RT/BNNS errors on a RangeDim program that declares "
            "states); only useful for inspecting the converted program."
        ),
    )
    parser.add_argument(
        "--materialize-sizes",
        default=None,
        help=(
            "Comma-separated concrete cache sizes "
            f"(default: powers of 2 from 512 up to --max-seq-len; every size "
            f"must be >= CHUNK_SIZE = {CHUNK_SIZE})"
        ),
    )
    # Internal: run a single phase (used by subprocess isolation).
    parser.add_argument("--_phase", help=argparse.SUPPRESS)
    parser.add_argument("--_phase-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # --variant supplies the defaults for --model-id and --output.
    _cfg, _hf, _path = VARIANTS[args.variant]
    if args.model_id is None:
        args.model_id = _hf
    if args.output is None:
        args.output = _path

    # If invoked as a subprocess for a single phase, run it and exit.
    if args._phase:
        export_phase(
            args._phase, args._phase_output, model_id=args.model_id,
            variant=args.variant, skip_warmup=args.skip_warmup,
            num_layers=args.num_layers,
        )
        return

    materialize_sizes: list[int] = []
    if args.materialize:
        materialize_sizes = _parse_materialize_sizes(args.materialize_sizes)
        over = [s for s in materialize_sizes if s > args.max_seq_len]
        if over:
            print(
                f"  [materialize] dropping sizes > --max-seq-len ({args.max_seq_len}): {over}",
                flush=True,
            )
            materialize_sizes = [s for s in materialize_sizes if s <= args.max_seq_len]
        if not materialize_sizes:
            parser.error("--materialize: no valid sizes after clamping to --max-seq-len")
        # A prefill chunk is written into the cache in one go, so a cache
        # shorter than a chunk cannot be filled.
        too_small = [s for s in materialize_sizes if s < CHUNK_SIZE]
        if too_small:
            parser.error(
                f"--materialize-sizes: {too_small} are smaller than "
                f"CHUNK_SIZE ({CHUNK_SIZE}); a prefill chunk must fit in the cache"
            )
        print(f"\nMaterialize plan: one function set per size in {materialize_sizes}",
              flush=True)

    output = Path(args.output)

    def _subprocess_phase(phase: str, phase_output: Path) -> None:
        """Re-invoke ourselves in a subprocess for memory isolation."""
        cmd = [sys.executable, "-m", "gemma_chat.export"]
        cmd += ["--model-id", args.model_id, "--variant", args.variant]
        if args.skip_warmup:
            cmd += ["--skip-warmup"]
        if args.num_layers is not None:
            cmd += ["--num-layers", str(args.num_layers)]
        cmd += ["--_phase", phase, "--_phase-output", str(phase_output)]
        result = subprocess.run(cmd, env={**_os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if result.returncode != 0:
            print(f"\n!!! {phase} export failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)

    def _subprocess_materialize(src: Path, dst: Path, sizes: list[int]) -> None:
        """Materialize `src` → `dst` in a lean subprocess.

        Target is ``gemma_chat.materialize`` (not ``gemma_chat.export``) so the
        child doesn't import JAX/flax at startup — the pymil load + materialize
        pass + final save need every spare GB.
        """
        cmd = [sys.executable, "-m", "gemma_chat.materialize"]
        cmd += ["--input", str(src), "--output", str(dst)]
        cmd += ["--sizes", ",".join(str(s) for s in sizes)]
        print(f"\n  [materialize] {src.name} → {len(sizes)} sizes …", flush=True)
        result = subprocess.run(cmd, env={**_os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if result.returncode != 0:
            print(f"\n!!! materialize failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)

    tmp_dir = Path(tempfile.mkdtemp(prefix="gemma-export-"))
    phases = ["decode"] if args.decode_only else ["prefill", "decode"]
    print(f"\nExport plan: {' + '.join(phases)} -> {output}\n  Temp dir: {tmp_dir}\n",
          flush=True)

    try:
        for phase in phases:
            _subprocess_phase(phase, tmp_dir / phase)
            print(f"\n  {phase} functions exported to {tmp_dir / phase}\n", flush=True)

        from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

        # ── Merge every function into one package (weight deduplication) ──
        # Runs in the parent: the sources hold one weight set between them,
        # so the merge's peak memory is bounded by it.
        print("=" * 60)
        print("Merging into multifunction .mlpackage (weight deduplication) ...")
        print("=" * 60)
        desc = MultiFunctionDescriptor()
        for phase in phases:
            for pkg in sorted((tmp_dir / phase).glob("*.mlpackage")):
                desc.add_function(str(pkg), src_function_name="main",
                                  target_function_name=pkg.stem)
        desc.default_function_name = "decode_c0"

        combined = tmp_dir / "combined.mlpackage" if materialize_sizes else output
        if combined.exists():
            shutil.rmtree(combined)
        save_multifunction(desc, str(combined))
        _embed_tokenizer(args.model_id, combined)
        _embed_host_tables(_host_tables_dir(tmp_dir / "decode"), combined)

        if materialize_sizes:
            print("=" * 60)
            print("Materializing dynamic shapes to per-size concrete functions …")
            print("=" * 60)
            _subprocess_materialize(combined, output, materialize_sizes)

        final_size = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
        print(f"\n  Final model: {output} ({final_size / 1e9:.2f} GB)\n")
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
