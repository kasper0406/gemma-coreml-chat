"""Export Gemma4-E2B chunk_prefill + decode_step to a single multifunction CoreML .mlpackage.

Both functions are merged into one model using CoreML's MultiFunctionDescriptor
(iOS 18+), sharing weights for automatic deduplication.  The chunked prefill
processes CHUNK_SIZE tokens per call instead of the full sequence, enabling
arbitrarily long contexts without quadratic memory growth.

Usage:
    uv run gemma-export
    uv run gemma-export --output gemma4-e2b.mlpackage
    uv run gemma-export --skip-warmup   # save RAM on constrained machines
"""

from __future__ import annotations

import argparse
import sys
import os as _os
import signal as _signal

from pathlib import Path
import numpy as np


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
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    except Exception:
        return 0.0


def _signal_handler(signum, frame):
    print(f"  [SIGNAL] Received signal {signum}  RSS={_rss_mb():.0f} MB", flush=True)
    import traceback
    traceback.print_stack(frame)
    _os._exit(signum)


_signal.signal(_signal.SIGTERM, _signal_handler)

import jax
import jax.numpy as jnp
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax import export as jax_export

import coremltools as ct
from stablehlo_coreml.converter import convert as hlo_to_mil

from gemma_chat.config import CHUNK_SIZE, E2B_CONFIG, HF_MODEL_ID, MAX_SEQ_LEN
from gemma_chat.model import Gemma4Transformer, Gemma4Config, AttentionType
from gemma_chat.weight_mapper import load_params
from gemma_chat.decode_coreml import (
    chunk_prefill_step, decode_step, empty_pos_ring,
)
from gemma_chat.cache_spec import build_cache_specs


# ── Shared helpers ─────────────────────────────────────────────────────────


def _build_kv_names(n_layers: int) -> list[str]:
    """Return 30 KV cache names: ['k_0', 'v_0', 'k_1', 'v_1', ..., 'k_14', 'v_14']."""
    names: list[str] = []
    for i in range(n_layers):
        names.append(f"k_{i}")
        names.append(f"v_{i}")
    return names


def _build_ct_inputs(
    config: Gemma4Config,
    cache_specs,
    max_seq_len: int,
    prefix_inputs: list,
) -> list:
    """Build ct.TensorType list for ct.convert with RangeDim on global KV caches.

    MIL inputs from stablehlo-coreml are named _arg0, _arg1, ... (sanitized from %argN).
    Global KV caches get RangeDim [1, max_seq_len] on dim 1.
    """
    ct_inputs = list(prefix_inputs)
    arg_idx = len(prefix_inputs)
    for spec in cache_specs:
        shape = (1, spec.cache_len, spec.num_kv_heads, spec.head_dim)
        if spec.attn_type == AttentionType.GLOBAL:
            flex_shape = ct.Shape(shape=(
                1,
                ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=max_seq_len),
                spec.num_kv_heads,
                spec.head_dim,
            ))
            ct_inputs.append(ct.TensorType(name=f"%arg{arg_idx}", shape=flex_shape))
            ct_inputs.append(ct.TensorType(name=f"%arg{arg_idx + 1}", shape=flex_shape))
        else:
            ct_inputs.append(ct.TensorType(name=f"%arg{arg_idx}", shape=shape))
            ct_inputs.append(ct.TensorType(name=f"%arg{arg_idx + 1}", shape=shape))
        arg_idx += 2
    # sliding_pos_ring
    ct_inputs.append(ct.TensorType(
        name=f"%arg{arg_idx}",
        shape=(1, config.sliding_window_size),
    ))
    return ct_inputs


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


def _hlo_to_mlpackage(
    hlo_module,
    output_path: Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    ct_inputs: list | None = None,
) -> None:
    """Run hlo_to_mil → ct.convert → save in the current process.

    Streaming int8 quantization runs during HLO→MIL conversion and
    ``quantize_const_weights`` catches any remaining large float constants
    in the MIL pipeline. The resulting ``.mlpackage`` uses
    ``constexpr_blockwise_shift_scale`` ops whose ``const`` inputs (int8
    data, fp16 scale) are visible to ``save_multifunction``'s cross-function
    ``const_deduplication`` pass — so weight sharing still works when
    merging quantized functions.

    ct_inputs: optional list of ``ct.TensorType`` for ``ct.convert(inputs=...)``.
        Used to specify RangeDim flexible shape bounds when the MIL program
        has symbolic dimensions.
    """
    import ctypes as _ctypes, ctypes.util as _ctypes_util
    import threading
    import traceback as _tb

    import numpy as np
    from coremltools.converters.mil import Builder as mb

    from gemma_chat.mil_passes.ct_convert_pipeline import build_ct_convert_pass_pipeline
    from gemma_chat.mil_passes.quantize_const_weights import (
        _quantize_symmetric_blockwise,
        _GROUP_SIZE,
    )
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

    # ── Streaming mixed-precision quantization during HLO→MIL ──
    _WEIGHT_THRESHOLD = 2048
    _VOCAB_SIZE = 262144
    _stream_counter = [0, 0, 0]  # [count_int4, count_int8, total_bytes]

    def _stream_quantize(arr: np.ndarray, name: str):
        if arr.ndim < 2 or arr.size <= _WEIGHT_THRESHOLD:
            return None
        if arr.dtype not in (np.float16, np.float32):
            return None
        orig_dtype = arr.dtype
        if arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        nbits = 4
        q_data, scale = _quantize_symmetric_blockwise(
            arr, axis=0, group_size=_GROUP_SIZE, nbits=nbits,
        )
        del arr
        _stream_counter[0] += 1
        _stream_counter[2] += q_data.nbytes * 2
        total = _stream_counter[0] + _stream_counter[1]
        if total % 20 == 0:
            print(
                f"    streaming-quantized {_stream_counter[0]} int4 + "
                f"{_stream_counter[1]} int8  "
                f"({_stream_counter[2] / 1e9:.2f} GB)  RSS={_rss_mb():.0f} MB",
                flush=True,
            )
        suffix = "_int4"
        result = mb.constexpr_blockwise_shift_scale(
            data=q_data, scale=scale, name=name + suffix,
        )
        if orig_dtype == np.float32:
            result = mb.cast(x=result, dtype="fp32", name=name + "_fp32")
        return result

    install_stablehlo_streaming_patch()
    set_streaming_quantizer(_stream_quantize)
    print("  [convert] streaming mixed-precision quantization enabled", flush=True)

    print(f"  [convert {_os.getpid()}] hlo_to_mil …", flush=True)
    try:
        mil_program = hlo_to_mil(hlo_module, minimum_deployment_target=ct.target.iOS18)
    finally:
        set_streaming_quantizer(None)

    if _stream_counter[0]:
        print(
            f"  StableHLO→MIL done — streaming-quantized {_stream_counter[0]} tensors "
            f"({_stream_counter[1] / 1e9:.2f} GB fp16 → int4).",
            flush=True,
        )
    else:
        print(f"  StableHLO→MIL conversion done.", flush=True)

    # ── MIL pass pipeline (quantize_const_weights as belt-and-suspenders) ──
    pipeline = build_ct_convert_pass_pipeline()
    pipeline.remove_passes([
        "common::add_fp16_cast",
        "common::fuse_layernorm_or_instancenorm",
        "common::fuse_elementwise_to_batchnorm",
    ])
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
                compute_precision=ct.precision.FLOAT16,
                pass_pipeline=pipeline,
                skip_model_load=True,
            )
            if ct_inputs is not None:
                convert_kwargs["inputs"] = ct_inputs
            cml_model = ct.convert(mil_program, **convert_kwargs)
        except BaseException as _e:
            print(f"\n!!! [convert] ct.convert raised {type(_e).__name__}: {_e}", flush=True)
            _tb.print_exc()
            raise
    finally:
        _stop.set()

    del mil_program, pipeline

    if input_names or output_names:
        _rename_model_io(cml_model, input_names or [], output_names or [])

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


# ── Chunked-prefill export ─────────────────────────────────────────────────


def export_chunk_prefill(
    output_path: str | Path,
    model_id: str = HF_MODEL_ID,
    max_seq_len: int = MAX_SEQ_LEN,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Export the chunked-prefill model (process CHUNK_SIZE tokens per call).

    Global KV caches use symbolic dim 1 (flexible shapes via RangeDim),
    so a single model works at any cache length up to ``max_seq_len``.

    Inputs:  tokens (1, chunk_size) int32
             start_position (1,) int32  — absolute position of first token in chunk
             k_0, v_0, ..., k_14, v_14 — current KV cache arrays float16
             sliding_pos_ring (1, sliding_window_size) int32 — ring position tracker
    Outputs: logits (chunk_size, vocab_size) float32
             k_0_out, v_0_out, ..., k_14_out, v_14_out — updated KV cache float16
             sliding_pos_ring_out (1, sliding_window_size) int32 — updated ring
    """
    import gc
    import numpy as np

    output_path = Path(output_path)
    config = E2B_CONFIG
    context = jax_mlir.make_ir_context()

    print("=" * 60)
    print("Chunk-prefill export — Step 1/3  Loading weights")
    print("=" * 60)
    params = load_params(model_id=model_id, config=config)

    print("=" * 60)
    print(f"Chunk-prefill export — Step 2/3  Tracing (chunk_size={chunk_size})")
    print("=" * 60)

    # Eager warmup to prime XLA PJRT.
    from flax import nnx
    from gemma_chat.weight_mapper import load_params_into_model
    model_tmp = Gemma4Transformer(config=config, rngs=nnx.Rngs(params=0))
    load_params_into_model(model_tmp, params, config)
    tiny_tokens = jnp.ones((1, 8), dtype=jnp.int32)
    _ = model_tmp(tiny_tokens)
    del model_tmp, tiny_tokens
    print(f"  Eager warmup OK", flush=True)

    print("  Converting params to float16 …", flush=True)
    _inplace_bf16_to_f16(params)

    gc.disable()
    try:
        cache_specs = build_cache_specs(config, max_seq_len)
        kv_shapes = [(1, s.cache_len, s.num_kv_heads, s.head_dim) for s in cache_specs]
        pos_ring_shape = (1, config.sliding_window_size)

        def chunk_prefill_fn(tokens, start_pos_1d, *kv_and_ring):
            """
            tokens: (1, chunk_size) int32
            start_pos_1d: (1,) int32 — absolute position of first token
            kv_and_ring: 30 KV arrays + 1 sliding_pos_ring
            """
            kv_flat = list(kv_and_ring[:-1])
            sliding_pos_ring = kv_and_ring[-1]
            start_pos = start_pos_1d[0]
            logits, kv_new, ring_new = chunk_prefill_step(
                params, tokens, start_pos, kv_flat, sliding_pos_ring,
                cfg=config, chunk_size=chunk_size,
            )
            return (logits,) + tuple(kv_new) + (ring_new,)

        # Build example arrays and symbolic shape specs
        example_args = [
            np.zeros((1, chunk_size), dtype=np.int32),   # tokens
            np.zeros((1,), dtype=np.int32),              # start_position
        ]
        shape_specs = [None, None]  # concrete shapes for tokens, start_pos

        for spec in cache_specs:
            shape = (1, spec.cache_len, spec.num_kv_heads, spec.head_dim)
            example = np.zeros(shape, dtype=np.float16)
            if spec.attn_type == AttentionType.GLOBAL:
                sym_shape = f"1, N, {spec.num_kv_heads}, {spec.head_dim}"
                example_args.extend([example, example])  # k, v
                shape_specs.extend([sym_shape, sym_shape])
            else:
                example_args.extend([example, example])  # k, v
                shape_specs.extend([None, None])  # concrete

        example_args.append(np.full(pos_ring_shape, -1, dtype=np.int32))
        shape_specs.append(None)

        args_specs = jax_export.symbolic_args_specs(
            tuple(example_args), shape_specs, constraints=["N >= 1"],
        )

        print("  Symbolic export of chunk_prefill_step to StableHLO …", flush=True)
        exported = jax_export.export(jax.jit(chunk_prefill_fn))(*args_specs)
        mlir_module = exported.mlir_module()
        hlo_module = ir.Module.parse(str(mlir_module), context=context)
        print("  Symbolic export OK.", flush=True)

        del exported, mlir_module, params

        mlir_cache = output_path.with_suffix('.mlirbc')
        print(f"  Saving MLIR cache → {mlir_cache} …", flush=True)
        try:
            with open(mlir_cache, 'wb') as _f:
                hlo_module.operation.write_bytecode(_f)
            sz = mlir_cache.stat().st_size
            print(f"  MLIR cache saved ({sz/1e9:.2f} GB).", flush=True)
        except Exception as _we:
            print(f"  WARNING: write_bytecode failed: {_we}", flush=True)

        print("=" * 60)
        print("Chunk-prefill export — Step 3/3  ct.convert + save")
        print("=" * 60)
        n_kv = len(cache_specs)
        kv_names = _build_kv_names(n_kv)
        kv_out_names = [n + "_out" for n in kv_names]

        ct_inputs = _build_ct_inputs(
            config, cache_specs, max_seq_len,
            prefix_inputs=[
                ct.TensorType(name="%arg0", shape=(1, chunk_size)),
                ct.TensorType(name="%arg1", shape=(1,)),
            ],
        )
        _hlo_to_mlpackage(
            hlo_module, output_path,
            input_names=["tokens", "start_position"] + kv_names + ["sliding_pos_ring"],
            output_names=["logits"] + kv_out_names + ["sliding_pos_ring_out"],
            ct_inputs=ct_inputs,
        )
    finally:
        gc.enable()


# ── Decode-step export ─────────────────────────────────────────────────────


def export_decode_step(
    output_path: str | Path,
    model_id: str = HF_MODEL_ID,
    max_seq_len: int = MAX_SEQ_LEN,
    skip_warmup: bool = False,
) -> None:
    """Export the single-token decode-step model.

    Global KV caches use symbolic dim 1 (flexible shapes via RangeDim),
    so a single model works at any cache length up to ``max_seq_len``.

    Inputs:  token_id (1,) int32
             position (1,) int32  — absolute position of this token
             k_0, v_0, ..., k_14, v_14 — current KV cache arrays float16
             sliding_pos_ring (1, sliding_window_size) int32 — ring position tracker
    Outputs: logits (vocab_size,) float32
             k_0_out, v_0_out, ..., k_14_out, v_14_out — updated KV cache float16
             sliding_pos_ring_out (1, sliding_window_size) int32 — updated ring
    """
    import numpy as np
    import gc

    output_path = Path(output_path)
    config = E2B_CONFIG
    context = jax_mlir.make_ir_context()

    print("=" * 60)
    print("Decode export — Step 1/3  Loading weights")
    print("=" * 60)
    params = load_params(model_id=model_id, config=config)

    print("=" * 60)
    print("Decode export — Step 2/3  Tracing to StableHLO")
    print("=" * 60)

    # Eager warmup to prime XLA PJRT (skippable to save memory on RAM-constrained machines).
    if not skip_warmup:
        from flax import nnx
        from gemma_chat.weight_mapper import load_params_into_model
        model_tmp = Gemma4Transformer(config=config, rngs=nnx.Rngs(params=0))
        load_params_into_model(model_tmp, params, config)
        tiny_tokens = jnp.ones((1, 8), dtype=jnp.int32)
        _ = model_tmp(tiny_tokens)
        del model_tmp, tiny_tokens
        print(f"  Eager warmup OK", flush=True)
    else:
        print("  Skipping eager warmup (--skip-warmup).", flush=True)

    print("  Converting params to float16 …", flush=True)
    _inplace_bf16_to_f16(params)

    gc.disable()
    try:
        cache_specs = build_cache_specs(config, max_seq_len)
        kv_shapes = [(1, s.cache_len, s.num_kv_heads, s.head_dim) for s in cache_specs]
        pos_ring_shape = (1, config.sliding_window_size)

        def decode_fn(token_id_1d, position_1d, *kv_and_ring):
            """
            token_id_1d: (1,) int32
            position_1d: (1,) int32 — absolute decode position
            kv_and_ring: 30 KV arrays + 1 sliding_pos_ring
            """
            kv_flat = list(kv_and_ring[:-1])
            sliding_pos_ring = kv_and_ring[-1]
            token_id = token_id_1d[0]
            position = position_1d[0]
            logits, kv_new, ring_new = decode_step(
                params, token_id, position, kv_flat, sliding_pos_ring,
                cfg=config,
            )
            return (logits,) + tuple(kv_new) + (ring_new,)

        # Build example arrays and symbolic shape specs
        example_args = [
            np.zeros((1,), dtype=np.int32),   # token_id
            np.zeros((1,), dtype=np.int32),   # position
        ]
        shape_specs = [None, None]  # concrete shapes

        for spec in cache_specs:
            shape = (1, spec.cache_len, spec.num_kv_heads, spec.head_dim)
            example = np.zeros(shape, dtype=np.float16)
            if spec.attn_type == AttentionType.GLOBAL:
                sym_shape = f"1, N, {spec.num_kv_heads}, {spec.head_dim}"
                example_args.extend([example, example])  # k, v
                shape_specs.extend([sym_shape, sym_shape])
            else:
                example_args.extend([example, example])  # k, v
                shape_specs.extend([None, None])  # concrete

        example_args.append(np.full(pos_ring_shape, -1, dtype=np.int32))
        shape_specs.append(None)

        args_specs = jax_export.symbolic_args_specs(
            tuple(example_args), shape_specs, constraints=["N >= 1"],
        )

        print("  Symbolic export of decode_step to StableHLO …", flush=True)
        exported = jax_export.export(jax.jit(decode_fn))(*args_specs)
        mlir_module = exported.mlir_module()
        hlo_module = ir.Module.parse(str(mlir_module), context=context)
        print("  Symbolic export OK.", flush=True)

        del exported, mlir_module, params

        mlir_cache = output_path.with_suffix('.mlirbc')
        print(f"  Saving MLIR cache → {mlir_cache} …", flush=True)
        try:
            with open(mlir_cache, 'wb') as _f:
                hlo_module.operation.write_bytecode(_f)
            sz = mlir_cache.stat().st_size
            print(f"  MLIR cache saved ({sz/1e9:.2f} GB).", flush=True)
        except Exception as _we:
            print(f"  WARNING: write_bytecode failed: {_we}", flush=True)

        print("=" * 60)
        print("Decode export — Step 3/3  ct.convert + save")
        print("=" * 60)
        n_kv = len(cache_specs)
        kv_names = _build_kv_names(n_kv)
        kv_out_names = [n + "_out" for n in kv_names]

        ct_inputs = _build_ct_inputs(
            config, cache_specs, max_seq_len,
            prefix_inputs=[
                ct.TensorType(name="%arg0", shape=(1,)),
                ct.TensorType(name="%arg1", shape=(1,)),
            ],
        )
        _hlo_to_mlpackage(
            hlo_module, output_path,
            input_names=["token_id", "position"] + kv_names + ["sliding_pos_ring"],
            output_names=["logits"] + kv_out_names + ["sliding_pos_ring_out"],
            ct_inputs=ct_inputs,
        )
    finally:
        gc.enable()


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    import shutil
    import tempfile

    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

    parser = argparse.ArgumentParser(
        description=(
            "Export Gemma4-E2B prefill + decode into a single multifunction "
            "CoreML .mlpackage with shared (int8-quantized) weights."
        )
    )
    parser.add_argument(
        "--output",
        default="gemma4-e2b.mlpackage",
        help="Output path for the multifunction .mlpackage (default: gemma4-e2b.mlpackage)",
    )
    parser.add_argument(
        "--model-id",
        default=HF_MODEL_ID,
        help=f"HuggingFace model ID (default: {HF_MODEL_ID})",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Max sequence length (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip eager XLA warmup to save ~2 GB RAM (use on memory-constrained machines)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    tmp_dir = Path(tempfile.mkdtemp(prefix="gemma-export-"))
    tmp_prefill = tmp_dir / "prefill.mlpackage"
    tmp_decode = tmp_dir / "decode.mlpackage"

    print(
        f"\nExport plan: chunk_prefill (chunk_size={CHUNK_SIZE}) + decode -> {output}\n"
        f"  Temp dir: {tmp_dir}\n",
        flush=True,
    )

    try:
        # -- Phase 1: Export chunk prefill (int8 streaming quantization) --
        export_chunk_prefill(
            output_path=tmp_prefill,
            model_id=args.model_id,
            max_seq_len=args.max_seq_len,
            chunk_size=CHUNK_SIZE,
        )
        print(f"\n  Chunk-prefill exported to {tmp_prefill}\n")

        # -- Phase 2: Export decode (int8 streaming quantization) --
        print(
            "=" * 60,
            "\nDecode-step export (second artifact).\n",
            "=" * 60,
            "\n",
            sep="",
            flush=True,
        )
        export_decode_step(
            output_path=tmp_decode,
            model_id=args.model_id,
            max_seq_len=args.max_seq_len,
            skip_warmup=args.skip_warmup,
        )
        print(f"\n  Decode exported to {tmp_decode}\n")

        # -- Phase 3: Merge int8 models (const dedup shares weight blobs) --
        print("=" * 60)
        print("Merging chunk_prefill + decode (int8 weight deduplication) ...")
        print("=" * 60)

        desc = MultiFunctionDescriptor()
        desc.add_function(str(tmp_prefill), src_function_name="main", target_function_name="prefill")
        desc.add_function(str(tmp_decode), src_function_name="main", target_function_name="decode")
        desc.default_function_name = "decode"

        if output.exists():
            shutil.rmtree(output)
        save_multifunction(desc, str(output))

        shutil.rmtree(tmp_prefill, ignore_errors=True)
        shutil.rmtree(tmp_decode, ignore_errors=True)

        final_size = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
        print(f"\n  Final model: {output} ({final_size / 1e9:.2f} GB)\n")

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
