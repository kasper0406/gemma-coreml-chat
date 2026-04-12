"""JAX vs CoreML parity check for decode (and optional prefill).

Loads HF weights, runs ``decode_coreml.decode_step`` for each prompt token,
then runs the same token sequence through the CoreML model. Reports
max abs diff and cosine similarity on the final logits vector.

Optional ``--greedy-steps N`` compares greedy argmax token IDs for N decode
steps after the same prefill (catches drift in the autoregressive loop).

Requires network/HF cache for weights unless already downloaded.

Example::

    uv run gemma-parity-decode --model gemma4-e2b.mlpackage \\
        --prompt "Hello" --max-tokens 8

    uv run gemma-parity-decode --model gemma4-e2b.mlpackage \\
        --prompt "Hello" --greedy-steps 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gemma_chat.config import E2B_CONFIG, HF_MODEL_ID, MAX_SEQ_LEN
from gemma_chat.decode_coreml import decode_step, empty_pos_ring, kv_cache_shapes
from gemma_chat.generate import (
    _coreml_kv_state_after_decode,
    load_coreml_model,
    load_tokenizer,
)
from gemma_chat.weight_mapper import load_params

import jax.numpy as jnp


def _jax_slow_prefill_logits(
    prompt_ids: list[int],
    *,
    model_id: str,
    max_seq_len: int,
):
    cfg = E2B_CONFIG
    params = load_params(model_id=model_id, config=cfg)
    shapes = kv_cache_shapes(cfg, max_seq_len)
    kv_flat = []
    for s in shapes:
        kv_flat.append(jnp.zeros(s, dtype=jnp.float16))
        kv_flat.append(jnp.zeros(s, dtype=jnp.float16))
    pos_ring = empty_pos_ring(cfg)

    logits = None
    for pos, tid in enumerate(prompt_ids):
        logits, kv_flat, pos_ring = decode_step(
            params,
            jnp.int32(tid),
            jnp.int32(pos),
            kv_flat,
            pos_ring,
            cfg=cfg,
        )
    return np.asarray(logits, dtype=np.float32), params, cfg


def _coreml_slow_prefill_logits(
    model_path: Path,
    prompt_ids: list[int],
    *,
    max_seq_len: int,
):
    import coremltools as ct

    dm = load_coreml_model(model_path, function_name="decode")
    dec_in = list(dm.input_description)
    dec_out = list(dm.output_description)
    kv_names_in = dec_in[2:]
    kv_names_out = dec_out[1:]

    # Initialize KV caches from function description (multifunction-aware).
    spec_path = Path(model_path).resolve()
    spec_m = ct.models.MLModel(str(spec_path), function_name="decode", skip_model_load=True)
    kv_inputs = []
    for fd in spec_m._spec.description.functions:
        if fd.name == "decode":
            kv_inputs = list(fd.input)[2:]
            break
    if not kv_inputs:
        kv_inputs = list(spec_m._spec.description.input)[2:]
    kv_state = {}
    for inp in kv_inputs:
        shape = tuple(inp.type.multiArrayType.shape)
        if len(shape) == 2:  # (1, W) — sliding_pos_ring
            kv_state[inp.name] = np.full(shape, -1, dtype=np.int32)
        else:  # (1, cache_len, nkv, hd) — KV cache
            kv_state[inp.name] = np.zeros(shape, dtype=np.float16)

    logits = None
    for pos, tid in enumerate(prompt_ids):
        step_input = {
            dec_in[0]: np.array([tid], dtype=np.int32),
            dec_in[1]: np.array([pos], dtype=np.int32),
        }
        step_input.update(kv_state)
        result = dm.predict(step_input)
        logits_key = dec_out[0] if dec_out[0] in result else next(k for k in result if k not in kv_names_out)
        logits = np.asarray(result[logits_key], dtype=np.float32).reshape(-1)
        kv_state = _coreml_kv_state_after_decode(kv_names_in, kv_names_out, result)
    return logits


def _jax_greedy_tokens_after_prefill(
    prompt_ids: list[int],
    *,
    model_id: str,
    max_seq_len: int,
    n_extra: int,
) -> list[int]:
    """Greedy argmax for ``n_extra`` tokens after slow prefill (JAX ``decode_coreml``)."""
    cfg = E2B_CONFIG
    params = load_params(model_id=model_id, config=cfg)
    shapes = kv_cache_shapes(cfg, max_seq_len)
    kv_flat = []
    for s in shapes:
        kv_flat.append(jnp.zeros(s, dtype=jnp.float16))
        kv_flat.append(jnp.zeros(s, dtype=jnp.float16))
    pos_ring = empty_pos_ring(cfg)

    logits = None
    for pos, tid in enumerate(prompt_ids):
        logits, kv_flat, pos_ring = decode_step(
            params,
            jnp.int32(tid),
            jnp.int32(pos),
            kv_flat,
            pos_ring,
            cfg=cfg,
        )
    t = len(prompt_ids)
    out: list[int] = []
    logits_np = np.asarray(logits, dtype=np.float32)
    for i in range(n_extra):
        tid = int(np.argmax(logits_np))
        out.append(tid)
        logits, kv_flat, pos_ring = decode_step(
            params,
            jnp.int32(tid),
            jnp.int32(t + i),
            kv_flat,
            pos_ring,
            cfg=cfg,
        )
        logits_np = np.asarray(logits, dtype=np.float32)
    return out


def _coreml_greedy_tokens_after_prefill(
    model_path: Path,
    prompt_ids: list[int],
    *,
    max_seq_len: int,
    n_extra: int,
) -> list[int]:
    """Greedy argmax for ``n_extra`` tokens after slow prefill (CoreML decode function)."""
    import coremltools as ct

    dm = load_coreml_model(model_path, function_name="decode")
    dec_in = list(dm.input_description)
    dec_out = list(dm.output_description)
    kv_names_in = dec_in[2:]
    kv_names_out = dec_out[1:]

    # Initialize KV caches from function description (multifunction-aware).
    spec_path = Path(model_path).resolve()
    spec_m = ct.models.MLModel(str(spec_path), function_name="decode", skip_model_load=True)
    kv_inputs = []
    for fd in spec_m._spec.description.functions:
        if fd.name == "decode":
            kv_inputs = list(fd.input)[2:]
            break
    if not kv_inputs:
        kv_inputs = list(spec_m._spec.description.input)[2:]
    kv_state = {}
    for inp in kv_inputs:
        shape = tuple(inp.type.multiArrayType.shape)
        if len(shape) == 2:
            kv_state[inp.name] = np.full(shape, -1, dtype=np.int32)
        else:
            kv_state[inp.name] = np.zeros(shape, dtype=np.float16)

    logits = None
    for pos, tid in enumerate(prompt_ids):
        step_input = {
            dec_in[0]: np.array([tid], dtype=np.int32),
            dec_in[1]: np.array([pos], dtype=np.int32),
        }
        step_input.update(kv_state)
        result = dm.predict(step_input)
        logits_key = dec_out[0] if dec_out[0] in result else next(k for k in result if k not in kv_names_out)
        logits = np.asarray(result[logits_key], dtype=np.float32).reshape(-1)
        kv_state = _coreml_kv_state_after_decode(kv_names_in, kv_names_out, result)

    t = len(prompt_ids)
    out: list[int] = []
    for i in range(n_extra):
        tid = int(np.argmax(logits))
        out.append(tid)
        decode_input = {
            dec_in[0]: np.array([tid], dtype=np.int32),
            dec_in[1]: np.array([t + i], dtype=np.int32),
        }
        decode_input.update(kv_state)
        decode_result = dm.predict(decode_input)
        logits_key = dec_out[0] if dec_out[0] in decode_result else next(k for k in decode_result if k not in kv_names_out)
        logits = np.asarray(decode_result[logits_key], dtype=np.float32).reshape(-1)
        kv_state = _coreml_kv_state_after_decode(
            kv_names_in, kv_names_out, decode_result
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prefill logits: JAX decode_step vs CoreML.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("gemma4-e2b.mlpackage"),
        help="Multifunction .mlpackage (default: gemma4-e2b.mlpackage)",
    )
    parser.add_argument(
        "--prompt",
        default="Hello",
        help="Test string (encoded with tokenizer)",
    )
    parser.add_argument(
        "--model-id",
        default=HF_MODEL_ID,
        help=f"HF model id for weights/tokenizer (default: {HF_MODEL_ID})",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Must match export (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Only use the first N prompt tokens (default: 16) for a quicker check",
    )
    parser.add_argument(
        "--skip-coreml",
        action="store_true",
        help="Only run JAX (no CoreML package required)",
    )
    parser.add_argument(
        "--greedy-steps",
        type=int,
        default=0,
        help=(
            "After prefill, compare this many greedy decode steps (token IDs) "
            "between JAX decode_coreml and CoreML (0 = skip, default)"
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Encode prompt as raw text (skip chat template). "
             "Without this flag the prompt is wrapped with the Gemma4 chat "
             "template so greedy decode produces meaningful output.",
    )
    args = parser.parse_args()

    if not args.skip_coreml and not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    tokenizer = load_tokenizer(args.model_id)
    if args.raw:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        messages = [{"role": "user", "content": args.prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    prompt_ids = prompt_ids[: args.max_tokens]
    if not prompt_ids:
        print("Empty prompt after tokenization.", file=sys.stderr)
        sys.exit(1)

    jax_logits, _params, _cfg = _jax_slow_prefill_logits(
        prompt_ids,
        model_id=args.model_id,
        max_seq_len=args.max_seq_len,
    )
    jax_logits = jax_logits.reshape(-1)

    print(
        f"JAX logits: shape={jax_logits.shape}, "
        f"min={jax_logits.min():.4f} max={jax_logits.max():.4f}",
        flush=True,
    )

    if args.skip_coreml:
        if args.greedy_steps > 0:
            jax_ids = _jax_greedy_tokens_after_prefill(
                prompt_ids,
                model_id=args.model_id,
                max_seq_len=args.max_seq_len,
                n_extra=args.greedy_steps,
            )
            print(f"\nJAX greedy ({args.greedy_steps} steps):", flush=True)
            print(f"  IDs:  {jax_ids}", flush=True)
            print(f"  Text: {tokenizer.decode(jax_ids)!r}", flush=True)
        return

    cm_logits = _coreml_slow_prefill_logits(
        args.model,
        prompt_ids,
        max_seq_len=args.max_seq_len,
    )

    if cm_logits.shape != jax_logits.shape:
        print(
            f"Shape mismatch: JAX {jax_logits.shape} vs CoreML {cm_logits.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    diff = np.abs(jax_logits - cm_logits)
    print(
        f"CoreML logits: min={cm_logits.min():.4f} max={cm_logits.max():.4f}",
        flush=True,
    )
    print(f"Max abs diff: {diff.max():.6f}", flush=True)
    print(f"Mean abs diff: {diff.mean():.6f}", flush=True)
    jn = np.linalg.norm(jax_logits)
    cn = np.linalg.norm(cm_logits)
    if jn > 0 and cn > 0:
        cos = float(np.dot(jax_logits, cm_logits) / (jn * cn))
        print(f"Cosine similarity: {cos:.6f}", flush=True)

    if args.greedy_steps > 0:
        jax_ids = _jax_greedy_tokens_after_prefill(
            prompt_ids,
            model_id=args.model_id,
            max_seq_len=args.max_seq_len,
            n_extra=args.greedy_steps,
        )
        cm_ids = _coreml_greedy_tokens_after_prefill(
            args.model,
            prompt_ids,
            max_seq_len=args.max_seq_len,
            n_extra=args.greedy_steps,
        )
        print(f"\nGreedy tokens ({args.greedy_steps} steps):", flush=True)
        print(f"  JAX:    {jax_ids}", flush=True)
        print(f"  JAX text: {tokenizer.decode(jax_ids)!r}", flush=True)
        print(f"  CoreML: {cm_ids}", flush=True)
        print(f"  CoreML text: {tokenizer.decode(cm_ids)!r}", flush=True)
        mismatches = [i for i, (a, b) in enumerate(zip(jax_ids, cm_ids)) if a != b]
        if mismatches:
            print(
                f"  Mismatch at steps {mismatches[:16]}"
                f"{'…' if len(mismatches) > 16 else ''} ({len(mismatches)} total)",
                flush=True,
            )
        else:
            print("  All greedy steps match.", flush=True)


if __name__ == "__main__":
    main()
