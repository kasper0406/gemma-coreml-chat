"""Autoregressive text generation using the exported CoreML model.

Global attention KV caches use dynamic sizing — they start small and grow
as the conversation extends, avoiding the cost of ``Q @ K^T`` over thousands
of empty slots.  Sliding attention caches are fixed-size ring buffers.
"""

from __future__ import annotations

import gc
import numpy as np
from pathlib import Path
from typing import Iterator

import coremltools as ct

from gemma_chat.cache_spec import build_cache_specs
from gemma_chat.config import CHUNK_SIZE, E2B_CONFIG, MAX_SEQ_LEN, MLPACKAGE_PATH, HF_MODEL_ID
from gemma_chat.model import AttentionType


def _validate_decode_step_inputs(_spec_m: ct.models.MLModel, package_path: str) -> None:
    """Decode-step models use rank-1 token and position inputs; prefill uses rank-2 tokens."""
    desc = _spec_m._spec.description
    if len(desc.input) < 2:
        return
    in0 = desc.input[0]
    shape0 = tuple(in0.type.multiArrayType.shape)
    if len(shape0) == 2:
        raise ValueError(
            f"The package {package_path!r} is a prefill model (first input shape {shape0}, rank 2). "
            "Use the decode-step .mlpackage with --decode-model "
            "(e.g. gemma4-e2b-decode.mlpackage from `uv run gemma-export`), "
            "not *_prefill.mlpackage."
        )


def _validate_prefill_inputs(_spec_m: ct.models.MLModel, package_path: str) -> None:
    """Prefill models expect a rank-2 token grid and rank-1 seq_len."""
    desc = _spec_m._spec.description
    if len(desc.input) < 2:
        raise ValueError(f"Prefill model {package_path!r} needs at least 2 inputs.")
    in0 = desc.input[0]
    shape0 = tuple(in0.type.multiArrayType.shape)
    if len(shape0) != 2:
        raise ValueError(
            f"The package {package_path!r} does not look like a prefill model "
            f"(first input shape {shape0}, expected rank 2). "
            "Use *_prefill.mlpackage from `uv run gemma-export --prefill-only` "
            "or the full export with prefill."
        )


def _coreml_kv_state_after_decode(
    kv_names_in: list[str],
    kv_names_out: list[str],
    result: dict,
) -> dict[str, np.ndarray]:
    """Map next-step KV feed dict from decode ``predict`` output.

    Prefer matching by tensor name (input name == output name) so result order
    in the dict cannot permute caches. Fall back to positional (input_i → output_i)
    when names differ, as in some converted graphs.
    """
    if len(kv_names_in) != len(kv_names_out):
        raise ValueError(
            f"KV input count {len(kv_names_in)} != output count {len(kv_names_out)}"
        )
    if all(kin in result for kin in kv_names_in):
        return {kin: result[kin] for kin in kv_names_in}
    state: dict[str, np.ndarray] = {}
    for kin, kout in zip(kv_names_in, kv_names_out):
        if kout not in result:
            raise KeyError(
                f"Decode output missing KV tensor {kout!r} (for input {kin!r}). "
                f"Available keys: {list(result.keys())}"
            )
        state[kin] = result[kout]
    return state


# ── Dynamic global KV cache helpers ────────────────────────────────────────

def _global_kv_input_names(kv_names: list[str]) -> set[str]:
    """Return the KV input names that correspond to global attention caches.

    ``kv_names`` is the full list of state input names (30 KV + 1 ring),
    ordered ``[k_0, v_0, k_1, v_1, ..., k_14, v_14, sliding_pos_ring]``.
    """
    specs = build_cache_specs()
    names: set[str] = set()
    for slot, spec in enumerate(specs):
        if spec.attn_type == AttentionType.GLOBAL:
            names.add(kv_names[slot * 2])      # k
            names.add(kv_names[slot * 2 + 1])  # v
    return names


def _flexible_global_names(kv_names: list[str], state_inputs) -> set[str]:
    """Return global KV input names that have RangeDim (flexible shapes).

    Models exported with symbolic shapes declare ``shapeRange`` on global KV
    inputs.  For fixed-shape models this returns an empty set, disabling
    dynamic cache sizing (caches use the spec shape instead).
    """
    arch_global = _global_kv_input_names(kv_names)
    flex: set[str] = set()
    for inp in state_inputs:
        if inp.name in arch_global:
            sr = inp.type.multiArrayType.shapeRange
            if sr and sr.sizeRanges:
                flex.add(inp.name)
    return flex


def _ensure_global_cache_capacity(
    kv_state: dict[str, np.ndarray],
    global_names: set[str],
    needed_len: int,
    max_len: int,
) -> None:
    """Grow global KV caches (doubling strategy) if dim-1 < ``needed_len``."""
    for name in global_names:
        arr = kv_state[name]
        cur = arr.shape[1]
        if cur >= needed_len:
            continue
        new_len = min(max(cur * 2, needed_len), max_len)
        grown = np.zeros(
            (arr.shape[0], new_len, arr.shape[2], arr.shape[3]),
            dtype=arr.dtype,
        )
        grown[:, :cur] = arr
        kv_state[name] = grown


def _init_kv_state(
    state_inputs,
    global_names: set[str],
    global_cache_len: int,
) -> dict[str, np.ndarray]:
    """Build initial KV state dict with dynamic global cache sizing.

    ``state_inputs`` are the protobuf input descriptors (after tokens + position).
    Global caches use ``global_cache_len``; sliding caches use the spec shape.
    """
    kv: dict[str, np.ndarray] = {}
    for inp in state_inputs:
        shape = tuple(inp.type.multiArrayType.shape)
        if len(shape) == 2:  # (1, W) — sliding_pos_ring
            kv[inp.name] = np.full(shape, -1, dtype=np.int32)
        elif inp.name in global_names:
            shape = (shape[0], global_cache_len, shape[2], shape[3])
            kv[inp.name] = np.zeros(shape, dtype=np.float16)
        else:
            kv[inp.name] = np.zeros(shape, dtype=np.float16)
    return kv


# ── Tokenizer ──────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str = HF_MODEL_ID):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    return tok


def stop_token_ids(tokenizer) -> set[int]:
    """Return the set of token IDs that should terminate generation.

    Includes the EOS token and the end-of-turn token (``<turn|>`` for Gemma4).
    """
    ids = {tokenizer.eos_token_id}
    eot = tokenizer.special_tokens_map.get("eot_token")
    if eot is not None:
        eot_id = tokenizer.convert_tokens_to_ids(eot)
        if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
            ids.add(eot_id)
    return ids


# ── CoreML model ───────────────────────────────────────────────────────────

import re as _re


class _CompiledModelWrapper:
    """Thin wrapper around MLModel that exposes input/output name lists.

    For multifunction models, ``input_description`` and ``output_description``
    on the top-level spec are empty.  This wrapper reads them from the
    per-function description stored in ``spec.description.functions`` instead.
    """

    def __init__(
        self,
        model: ct.models.MLModel,
        function_name: str | None = None,
        package_path: Path | None = None,
    ):
        self._model = model
        self._package_path = package_path

        # Try top-level description first (single-function models).
        in_desc = list(model.input_description)
        out_desc = list(model.output_description)

        if not in_desc and function_name is not None:
            # Multifunction model — read from per-function description.
            for fd in model._spec.description.functions:
                if fd.name == function_name:
                    in_desc = [inp.name for inp in fd.input]
                    out_desc = [out.name for out in fd.output]
                    break

        self.input_names: list[str] = in_desc
        self.output_names: list[str] = out_desc
        # Keep legacy aliases used by parity_decode.py
        self.input_description = in_desc
        self.output_description = out_desc

    def predict(self, data):
        return self._model.predict(data)


def load_coreml_model(
    mlpackage_path: str | Path = MLPACKAGE_PATH,
    compute_units: "ct.ComputeUnit" = None,
    function_name: str | None = None,
) -> _CompiledModelWrapper:
    if compute_units is None:
        compute_units = ct.ComputeUnit.ALL

    path = Path(mlpackage_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"CoreML model not found at {path}.\n"
            "Run  uv run gemma-export  first to produce the .mlpackage."
        )

    fn_label = f" (function={function_name})" if function_name else ""
    print(f"Loading CoreML model from {path}{fn_label} …")

    kwargs: dict = {"compute_units": compute_units}
    if function_name is not None:
        kwargs["function_name"] = function_name

    loaded = ct.models.MLModel(str(path), **kwargs)
    model = _CompiledModelWrapper(loaded, function_name=function_name, package_path=path)
    print("  Model loaded.")
    return model


# ── Inference helpers ──────────────────────────────────────────────────────

def truncate_prompt_ids(
    ids: list[int],
    max_seq_len: int,
    *,
    reserve_for_generation: int = 0,
) -> list[int]:
    """Keep the last tokens so chat tails (user turn + generation prompt) stay visible.

    If ``reserve_for_generation`` > 0 (KV decode path), cap length at
    ``max_seq_len - reserve_for_generation`` so prefill indices plus up to
    that many new tokens stay within ``0 .. max_seq_len - 1``.
    """
    cap = max_seq_len - reserve_for_generation if reserve_for_generation else max_seq_len
    cap = max(cap, 1)
    if len(ids) > cap:
        return ids[-cap:]
    return list(ids)


def _pad_tokens(token_ids: list[int], max_seq_len: int, pad_id: int) -> np.ndarray:
    """Right-pad token_ids to max_seq_len.  If longer, keep the *rightmost* tokens."""
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[-max_seq_len:]
    padded = token_ids + [pad_id] * (max_seq_len - len(token_ids))
    return np.array(padded, dtype=np.int32)[np.newaxis, :]   # (1, max_seq_len)


def _get_input_name(cml_model: ct.models.MLModel) -> str:
    names = list(cml_model.input_description)
    if len(names) != 1:
        raise ValueError(f"Expected 1 model input, got: {names}")
    return names[0]


def _get_output_name(cml_model: ct.models.MLModel) -> str:
    names = list(cml_model.output_description)
    if len(names) != 1:
        raise ValueError(f"Expected 1 model output, got: {names}")
    return names[0]


def load_kvcached_models(
    prefill_path: str | Path,
    decode_path: str | Path,
) -> tuple[_CompiledModelWrapper, _CompiledModelWrapper]:
    """Load prefill + decode mlpackages for KV-cached inference."""
    prefill_model = load_coreml_model(prefill_path)
    decode_model = load_coreml_model(decode_path)
    return prefill_model, decode_model


def _logits_at_last_token(
    cml_model: ct.models.MLModel,
    token_ids: list[int],
    pad_id: int,
    max_seq_len: int,
) -> np.ndarray:
    """Run the model and return logits for the last *real* token. Shape: (vocab,)"""
    n_real = min(len(token_ids), max_seq_len)
    last_pos = n_real - 1  # right-padded: real tokens at 0..n_real-1

    padded = _pad_tokens(token_ids, max_seq_len, pad_id)
    input_name = _get_input_name(cml_model)
    output_name = _get_output_name(cml_model)

    result = cml_model.predict({input_name: padded})
    out_key = output_name if output_name in result else list(result.keys())[0]
    logits = result[out_key]          # (1, max_seq_len, vocab)
    return logits[0, last_pos, :]         # (vocab,)


def sample_next_token(logits: np.ndarray, temperature: float = 1.0, top_p: float = 0.9) -> int:
    """Sample next token id with temperature + top-p (nucleus) sampling."""
    if temperature <= 0.0:
        return int(np.argmax(logits))

    logits = logits.astype(np.float64)
    logits /= temperature

    # Numerical stability
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # Top-p filtering
    sorted_idx = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumulative, top_p) + 1
    top_idx = sorted_idx[:cutoff]

    top_probs = probs[top_idx]
    top_probs /= top_probs.sum()

    return int(np.random.choice(top_idx, p=top_probs))


# ── Main generation function ────────────────────────────────────────────────

def generate(
    prompt_ids: list[int],
    cml_model: ct.models.MLModel,
    pad_id: int,
    stop_ids: set[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Iterator[int]:
    """Yield generated token IDs one at a time until a stop token or max_new_tokens."""
    token_ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        logits = _logits_at_last_token(cml_model, token_ids, pad_id, max_seq_len)
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        yield next_id
        if next_id in stop_ids:
            break
        token_ids.append(next_id)


def generate_text(
    prompt: str,
    cml_model: ct.models.MLModel,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = MAX_SEQ_LEN,
    prompt_ids: list[int] | None = None,
) -> str:
    """Convenience wrapper: prompt string → decoded response string.

    Pass ``prompt_ids`` to reuse a pre-tokenized prompt (avoids double encoding).
    """
    if prompt_ids is None:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id or 0
    stop_ids = stop_token_ids(tokenizer)

    generated = []
    for token_id in generate(
        prompt_ids,
        cml_model=cml_model,
        pad_id=pad_id,
        stop_ids=stop_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
    ):
        if token_id in stop_ids:
            break
        generated.append(token_id)

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── KV-cached generation ────────────────────────────────────────────────────

def generate_kvcached(
    prompt_ids: list[int],
    model_path: str | Path,
    pad_id: int = 0,
    stop_ids: set[int] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = MAX_SEQ_LEN,
    decode_only: bool = False,
    verbose: bool = True,
    decode_model: "_CompiledModelWrapper | None" = None,
    prefill_model: "_CompiledModelWrapper | None" = None,
) -> Iterator[int]:
    """Yield generated token IDs using KV-cached decode inference.

    Loads both chunked-prefill and decode functions from a single multifunction
    ``.mlpackage``.  Chunked prefill processes the prompt in CHUNK_SIZE-token
    chunks, then switches to single-token decode for generation.

    With ``decode_only=True``, only the decode function is used (token-by-token
    "slow prefill").

    Pre-loaded models can be passed via ``decode_model`` / ``prefill_model``
    to avoid reloading on every call (used by the TUI chat app).
    """
    if stop_ids is None:
        stop_ids = {1}
    prompt_ids = truncate_prompt_ids(
        list(prompt_ids),
        max_seq_len,
        reserve_for_generation=max_new_tokens,
    )
    n_real = len(prompt_ids)
    model_path = Path(model_path)

    # ── Load decode model ────────────────────────────────────────────────────
    if decode_model is None:
        if verbose:
            print("Loading decode model…", flush=True)
        decode_model = load_coreml_model(model_path, function_name="decode")

    dec_in = list(decode_model.input_description)
    dec_out = list(decode_model.output_description)
    kv_names_in = dec_in[2:]      # 30 KV input names + sliding_pos_ring
    kv_names_out = dec_out[1:]    # 30 KV output names + sliding_pos_ring_out

    # Read decode function's protobuf input descriptors for shape info.
    _dec_state_inputs = []
    for fd in decode_model._model._spec.description.functions:
        if fd.name == "decode":
            _dec_state_inputs = list(fd.input)[2:]
            break
    if not _dec_state_inputs:
        _dec_state_inputs = list(decode_model._model._spec.description.input)[2:]

    # Only enable dynamic sizing for global caches that have RangeDim.
    # Fixed-shape models → global_names is empty → all caches use spec shapes.
    global_names = _flexible_global_names(kv_names_in, _dec_state_inputs)
    if global_names and verbose:
        print(f"  Dynamic global KV caches: {len(global_names)} inputs", flush=True)

    kv_state: dict[str, np.ndarray] = {}

    logits: np.ndarray | None = None

    if not decode_only:
        # ── Chunked prefill ─────────────────────────────────────────────────
        if prefill_model is None:
            if verbose:
                print("Loading prefill model…", flush=True)
            prefill_model = load_coreml_model(model_path, function_name="prefill")

        pref_in = list(prefill_model.input_description)
        pref_out = list(prefill_model.output_description)
        pref_kv_in = pref_in[2:]     # 30 KV input names
        pref_kv_out = pref_out[1:]   # 30 KV output names

        # Compute padded length (needed for global cache sizing).
        n_chunks = (n_real + CHUNK_SIZE - 1) // CHUNK_SIZE
        padded_len = n_chunks * CHUNK_SIZE

        # Initialize KV caches — global caches sized to padded_len, not max_seq_len.
        _spec_m = ct.models.MLModel(str(model_path.resolve()), skip_model_load=True,
                                     function_name="prefill")
        _all_inputs = []
        func_descs = _spec_m._spec.description.functions
        if func_descs:
            for fd in func_descs:
                if fd.name == "prefill":
                    _all_inputs = list(fd.input)
                    break
        else:
            _all_inputs = list(_spec_m._spec.description.input)

        # Validate prefill token input shape matches CHUNK_SIZE.
        if _all_inputs:
            _token_shape = tuple(_all_inputs[0].type.multiArrayType.shape)
            _expected = (1, CHUNK_SIZE)
            if _token_shape != _expected:
                raise ValueError(
                    f"Prefill model token input shape is {_token_shape} but "
                    f"expected {_expected} (CHUNK_SIZE={CHUNK_SIZE}). "
                    f"The .mlpackage is stale — re-export with:  "
                    f"uv run gemma-export"
                )

        pref_global_names = _flexible_global_names(pref_kv_in, _all_inputs[2:])
        pref_kv_state = _init_kv_state(
            _all_inputs[2:], pref_global_names, padded_len,
        )
        del _spec_m, _all_inputs

        prompt_padded = list(prompt_ids) + [pad_id] * (padded_len - n_real)

        if verbose:
            print(
                f"Prefill: {n_real} tokens in {n_chunks} chunks "
                f"(chunk_size={CHUNK_SIZE}, global_kv={padded_len})…",
                flush=True,
            )

        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SIZE
            chunk_tokens = prompt_padded[start : start + CHUNK_SIZE]
            chunk_input = {
                pref_in[0]: np.array(chunk_tokens, dtype=np.int32)[np.newaxis, :],
                pref_in[1]: np.array([start], dtype=np.int32),
            }
            chunk_input.update(pref_kv_state)

            pr = prefill_model.predict(chunk_input)

            # Extract logits (CHUNK_SIZE, vocab_size)
            logits_key = pref_out[0] if pref_out[0] in pr else next(
                k for k in pr if k not in pref_kv_out
            )
            chunk_logits = np.asarray(pr[logits_key], dtype=np.float32)

            # Update KV state for next chunk
            pref_kv_state = {}
            for pk_out, pk_in in zip(pref_kv_out, pref_kv_in):
                pref_kv_state[pk_in] = pr[pk_out]

        # Pick logits at the last real token's position within the last chunk
        last_token_pos_in_chunk = (n_real - 1) % CHUNK_SIZE
        logits = chunk_logits[last_token_pos_in_chunk]  # (vocab_size,)

        # Map prefill KV outputs → decode KV inputs by position
        if len(pref_kv_out) != len(kv_names_in):
            raise ValueError(
                f"Prefill KV output count ({len(pref_kv_out)}) != "
                f"decode KV input count ({len(kv_names_in)})"
            )
        kv_state = {}
        for pk_out, dk_in in zip(pref_kv_out, kv_names_in):
            kv_state[dk_in] = pr[pk_out]
    else:
        # ── Token-by-token "slow prefill" ────────────────────────────────────
        kv_state = _init_kv_state(_dec_state_inputs, global_names, max(n_real, 1))

        if verbose:
            print(f"Processing {n_real} prompt tokens (decode-only)…", flush=True)
        for pos, token_id in enumerate(prompt_ids):
            step_input = {
                dec_in[0]: np.array([token_id], dtype=np.int32),
                dec_in[1]: np.array([pos], dtype=np.int32),
            }
            step_input.update(kv_state)
            result = decode_model.predict(step_input)
            logits_key = dec_out[0] if dec_out[0] in result else next(
                k for k in result if k not in kv_names_out
            )
            logits = np.asarray(result[logits_key], dtype=np.float32).reshape(-1)
            kv_state = _coreml_kv_state_after_decode(
                kv_names_in, kv_names_out, result
            )

    # ── Decode loop ───────────────────────────────────────────────────────────
    if logits is None:
        raise RuntimeError("internal error: logits unset after prefill")
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)

    max_steps = min(max_new_tokens, max_seq_len - n_real)
    for step in range(max_steps):
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        yield next_id
        if next_id in stop_ids:
            break

        position = n_real + step
        _ensure_global_cache_capacity(kv_state, global_names, position + 1, max_seq_len)

        decode_input = {
            dec_in[0]: np.array([next_id], dtype=np.int32),
            dec_in[1]: np.array([position], dtype=np.int32),
        }
        decode_input.update(kv_state)

        decode_result = decode_model.predict(decode_input)
        logits_key = dec_out[0] if dec_out[0] in decode_result else next(
            k for k in decode_result if k not in kv_names_out
        )
        logits = np.asarray(decode_result[logits_key], dtype=np.float32).reshape(-1)
        kv_state = _coreml_kv_state_after_decode(
            kv_names_in, kv_names_out, decode_result
        )

def generate_text_kvcached(
    prompt: str,
    tokenizer,
    model_path: str | Path,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = MAX_SEQ_LEN,
    decode_only: bool = False,
    prompt_ids: list[int] | None = None,
    verbose: bool = True,
) -> str:
    """Convenience wrapper for KV-cached inference: prompt → response string."""
    if prompt_ids is None:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id or 0
    stop_ids_ = stop_token_ids(tokenizer)

    generated = []
    for token_id in generate_kvcached(
        prompt_ids,
        model_path=model_path,
        pad_id=pad_id,
        stop_ids=stop_ids_,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
        decode_only=decode_only,
        verbose=verbose,
    ):
        if token_id in stop_ids_:
            break
        generated.append(token_id)

    return tokenizer.decode(generated, skip_special_tokens=True)
