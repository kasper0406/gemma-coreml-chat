"""Autoregressive text generation using the exported CoreML model.

The CoreML model expects a fixed-shape int32 tensor (1, MAX_SEQ_LEN).
We right-pad the current token sequence so the *last real* token sits at
position len(tokens)-1 and its logits drive the next-token prediction.
"""

from __future__ import annotations

import gc
import numpy as np
from pathlib import Path
from typing import Iterator

import coremltools as ct

from gemma_chat.config import CHUNK_SIZE, E2B_CONFIG, MAX_SEQ_LEN, MLPACKAGE_PATH, HF_MODEL_ID


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
            "(e.g. gemma4-e2b-decode.mlpackage from `uv run gemma-export-decode`), "
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
            "Use *_prefill.mlpackage from `uv run gemma-export-decode --prefill-only` "
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


# ── Tokenizer ──────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str = HF_MODEL_ID):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    return tok


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
            "Run  uv run gemma-export-decode  first to produce the .mlpackage."
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
    eos_id: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Iterator[int]:
    """Yield generated token IDs one at a time until EOS or max_new_tokens."""
    token_ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        logits = _logits_at_last_token(cml_model, token_ids, pad_id, max_seq_len)
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        yield next_id
        if next_id == eos_id:
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
    eos_id = tokenizer.eos_token_id

    generated = []
    for token_id in generate(
        prompt_ids,
        cml_model=cml_model,
        pad_id=pad_id,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
    ):
        if token_id == eos_id:
            break
        generated.append(token_id)

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── KV-cached generation ────────────────────────────────────────────────────

def generate_kvcached(
    prompt_ids: list[int],
    model_path: str | Path,
    pad_id: int = 0,
    eos_id: int = 1,
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
    kv_names_in = dec_in[2:]      # 30 KV input names
    kv_names_out = dec_out[1:]    # 30 KV output names

    # Read KV shapes from spec.
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

        # Initialize KV caches as zeros with shapes from the prefill function.
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
                    f"uv run gemma-export-decode"
                )

        _kv_inputs = _all_inputs[2:]
        pref_kv_state = {
            inp.name: np.zeros(tuple(inp.type.multiArrayType.shape), dtype=np.float16)
            for inp in _kv_inputs
        }
        del _spec_m, _all_inputs, _kv_inputs

        # Pad prompt to a multiple of CHUNK_SIZE
        n_chunks = (n_real + CHUNK_SIZE - 1) // CHUNK_SIZE
        padded_len = n_chunks * CHUNK_SIZE
        prompt_padded = list(prompt_ids) + [pad_id] * (padded_len - n_real)

        if verbose:
            print(
                f"Prefill: {n_real} tokens in {n_chunks} chunks "
                f"(chunk_size={CHUNK_SIZE})…",
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
        # Initialize KV caches as zeros with shapes from the function description.
        _spec_m = ct.models.MLModel(str(model_path.resolve()), skip_model_load=True,
                                     function_name="decode")
        # For multifunction models, I/O is in spec.description.functions.
        _kv_inputs = []
        func_descs = _spec_m._spec.description.functions
        if func_descs:
            for fd in func_descs:
                if fd.name == "decode":
                    _kv_inputs = list(fd.input)[2:]
                    break
        else:
            _kv_inputs = list(_spec_m._spec.description.input)[2:]
        kv_state = {
            inp.name: np.zeros(tuple(inp.type.multiArrayType.shape), dtype=np.float16)
            for inp in _kv_inputs
        }
        del _spec_m, _kv_inputs

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
        if next_id == eos_id:
            break

        position = n_real + step
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
    eos_id = tokenizer.eos_token_id

    generated = []
    for token_id in generate_kvcached(
        prompt_ids,
        model_path=model_path,
        pad_id=pad_id,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
        decode_only=decode_only,
        verbose=verbose,
    ):
        if token_id == eos_id:
            break
        generated.append(token_id)

    return tokenizer.decode(generated, skip_special_tokens=True)
