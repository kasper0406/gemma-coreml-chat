"""Core benchmark runner — times prefill and decode across backends/contexts."""

from __future__ import annotations

import json
import platform
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from benchmarks.power import PowerMonitor, PowerTrace

# Re-use inference machinery from gemma_chat.generate
from gemma_chat.config import CHUNK_SIZE, E2B_CONFIG, MAX_SEQ_LEN
from gemma_chat.generate import (
    _coreml_kv_state_after_decode,
    _ensure_global_cache_capacity,
    _flexible_global_names,
    _init_kv_state,
    load_coreml_model,
    sample_next_token,
)

# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result from a single benchmark run."""

    context_length: int
    backend: str
    run_index: int
    prefill_time_s: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_time_s: float = 0.0
    decode_tokens_per_sec: float = 0.0
    num_decode_tokens: int = 0
    power: dict = field(default_factory=dict)
    timed_out: bool = False
    error: str | None = None


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark session."""

    model_path: str
    backends: list[str]
    context_lengths: list[int]
    num_runs: int = 5
    decode_tokens: int = 64
    timeout_s: int = 300
    enable_power: bool = True
    max_seq_len: int = MAX_SEQ_LEN

    def to_dict(self) -> dict:
        return asdict(self)


BACKEND_MAP = {
    "cpu": "CPU_ONLY",
    "ane": "CPU_AND_NE",
    "all": "ALL",
}


def _compute_unit(backend: str):
    """Map backend name to coremltools ComputeUnit."""
    import coremltools as ct
    name = BACKEND_MAP.get(backend, backend.upper())
    return getattr(ct, "ComputeUnit", None) and getattr(ct.ComputeUnit, name)


# ---------------------------------------------------------------------------
# Prefill benchmark
# ---------------------------------------------------------------------------


def _bench_prefill(
    prefill_model,
    decode_model,
    prompt_ids: list[int],
    max_seq_len: int,
) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
    """Run chunked prefill and return (elapsed_s, kv_state, logits)."""
    import coremltools as ct

    n_real = len(prompt_ids)

    # Read model I/O names
    pref_in = list(prefill_model.input_description)
    pref_out = list(prefill_model.output_description)
    _pref_has_n = len(pref_in) > 0 and pref_in[0] == "N"
    _n_pref_ctrl = 3 if _pref_has_n else 2
    pref_kv_in = pref_in[_n_pref_ctrl:]
    pref_kv_out = pref_out[1:]

    # Prefill spec for shape info
    _spec = ct.models.MLModel(
        str(Path(prefill_model._model.path).parent),
        skip_model_load=True,
        function_name="prefill",
    )
    _all_inputs = []
    func_descs = _spec._spec.description.functions
    if func_descs:
        for fd in func_descs:
            if fd.name == "prefill":
                _all_inputs = list(fd.input)
                break
    if not _all_inputs:
        _all_inputs = list(_spec._spec.description.input)
    pref_state_inputs = _all_inputs[_n_pref_ctrl:]
    pref_global = _flexible_global_names(pref_kv_in, pref_state_inputs)

    # Chunks
    n_chunks = (n_real + CHUNK_SIZE - 1) // CHUNK_SIZE
    padded_len = n_chunks * CHUNK_SIZE

    # Init KV state sized to padded_len for global caches
    init_global_len = min(padded_len, CHUNK_SIZE) if pref_global else padded_len
    kv_state = _init_kv_state(pref_state_inputs, pref_global, init_global_len)

    padded_ids = prompt_ids + [0] * (padded_len - n_real)

    start = time.perf_counter()

    logits = None
    for chunk_idx in range(n_chunks):
        s = chunk_idx * CHUNK_SIZE
        chunk = np.array(padded_ids[s : s + CHUNK_SIZE], dtype=np.int32)[np.newaxis, :]
        pos = np.array([s], dtype=np.int32)

        needed = s + CHUNK_SIZE
        _ensure_global_cache_capacity(kv_state, pref_global, needed, max_seq_len)

        global_len = next(
            (kv_state[n].shape[1] for n in pref_global), 0,
        )
        if _pref_has_n:
            inp = {
                pref_in[0]: np.array([global_len], dtype=np.int32),
                pref_in[1]: chunk,
                pref_in[2]: pos,
            }
        else:
            inp = {pref_in[0]: chunk, pref_in[1]: pos}
        inp.update(kv_state)

        result = prefill_model.predict(inp)
        logits_key = pref_out[0] if pref_out[0] in result else next(
            k for k in result if k not in pref_kv_out
        )
        logits = np.asarray(result[logits_key], dtype=np.float32)

        kv_state = _coreml_kv_state_after_decode(pref_kv_in, pref_kv_out, result)

    elapsed = time.perf_counter() - start

    # Convert prefill KV state to decode model naming if needed
    dec_in = list(decode_model.input_description)
    _dec_has_n = len(dec_in) > 0 and dec_in[0] == "N"
    _n_dec_ctrl = 3 if _dec_has_n else 2
    dec_kv_in = dec_in[_n_dec_ctrl:]

    # Map prefill output names → decode input names (same order)
    mapped_kv: dict[str, np.ndarray] = {}
    for i, dk in enumerate(dec_kv_in):
        pk = pref_kv_in[i] if i < len(pref_kv_in) else dk
        if pk in kv_state:
            mapped_kv[dk] = kv_state[pk]
        elif dk in kv_state:
            mapped_kv[dk] = kv_state[dk]

    return elapsed, mapped_kv, logits


# ---------------------------------------------------------------------------
# Decode benchmark
# ---------------------------------------------------------------------------


def _bench_decode(
    decode_model,
    kv_state: dict[str, np.ndarray],
    logits: np.ndarray,
    n_real: int,
    num_tokens: int,
    max_seq_len: int,
) -> tuple[float, int]:
    """Run decode steps and return (elapsed_s, tokens_generated)."""
    dec_in = list(decode_model.input_description)
    dec_out = list(decode_model.output_description)
    _has_n = len(dec_in) > 0 and dec_in[0] == "N"
    _n_ctrl = 3 if _has_n else 2
    kv_names_in = dec_in[_n_ctrl:]
    kv_names_out = dec_out[1:]

    _dec_state_inputs = []
    for fd in decode_model._model._spec.description.functions:
        if fd.name == "decode":
            _dec_state_inputs = list(fd.input)[_n_ctrl:]
            break
    if not _dec_state_inputs:
        _dec_state_inputs = list(decode_model._model._spec.description.input)[_n_ctrl:]
    global_names = _flexible_global_names(kv_names_in, _dec_state_inputs)

    flat_logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    # Greedy first token
    next_id = int(np.argmax(flat_logits))

    start = time.perf_counter()
    generated = 0

    for step in range(num_tokens):
        position = n_real + step
        _ensure_global_cache_capacity(kv_state, global_names, position + 1, max_seq_len)

        global_len = next(
            (kv_state[n].shape[1] for n in global_names), 0,
        ) if _has_n else 0

        if _has_n:
            inp = {
                dec_in[0]: np.array([global_len], dtype=np.int32),
                dec_in[1]: np.array([next_id], dtype=np.int32),
                dec_in[2]: np.array([position], dtype=np.int32),
            }
        else:
            inp = {
                dec_in[0]: np.array([next_id], dtype=np.int32),
                dec_in[1]: np.array([position], dtype=np.int32),
            }
        inp.update(kv_state)

        result = decode_model.predict(inp)
        logits_key = dec_out[0] if dec_out[0] in result else next(
            k for k in result if k not in kv_names_out
        )
        flat_logits = np.asarray(result[logits_key], dtype=np.float32).reshape(-1)
        kv_state = _coreml_kv_state_after_decode(kv_names_in, kv_names_out, result)

        next_id = int(np.argmax(flat_logits))
        generated += 1

        # Stop on EOS
        if next_id in {1, 107}:
            break

    elapsed = time.perf_counter() - start
    return elapsed, generated


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_single(
    model_path: Path,
    prompt_ids: list[int],
    context_length: int,
    backend: str,
    run_index: int,
    decode_tokens: int = 64,
    max_seq_len: int = MAX_SEQ_LEN,
    timeout_s: int = 300,
    enable_power: bool = True,
    decode_model=None,
    prefill_model=None,
) -> RunResult:
    """Run a single prefill+decode benchmark."""
    result = RunResult(
        context_length=context_length,
        backend=backend,
        run_index=run_index,
    )

    # Set up timeout via SIGALRM
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)

    try:
        cu = _compute_unit(backend)

        if decode_model is None:
            decode_model = load_coreml_model(
                model_path, compute_units=cu, function_name="decode",
            )
        if prefill_model is None:
            prefill_model = load_coreml_model(
                model_path, compute_units=cu, function_name="prefill",
            )

        power_mon = PowerMonitor() if enable_power else None
        if power_mon:
            power_mon.start()

        try:
            # Prefill
            prefill_time, kv_state, logits = _bench_prefill(
                prefill_model, decode_model, prompt_ids, max_seq_len,
            )
            result.prefill_time_s = prefill_time
            result.prefill_tokens_per_sec = len(prompt_ids) / prefill_time if prefill_time > 0 else 0

            # Decode
            decode_time, n_generated = _bench_decode(
                decode_model, kv_state, logits, len(prompt_ids),
                decode_tokens, max_seq_len,
            )
            result.decode_time_s = decode_time
            result.num_decode_tokens = n_generated
            result.decode_tokens_per_sec = n_generated / decode_time if decode_time > 0 else 0

        finally:
            if power_mon:
                power_mon.stop()
                result.power = power_mon.trace.to_dict()

    except _Timeout:
        result.timed_out = True
        result.error = f"Timed out after {timeout_s}s"
    except Exception as e:
        result.error = str(e)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return result


# ---------------------------------------------------------------------------
# Full benchmark session
# ---------------------------------------------------------------------------


def run_benchmark(
    config: BenchmarkConfig,
    prompt_ids_by_length: dict[int, list[int]],
    verbose: bool = True,
) -> list[RunResult]:
    """Run the complete benchmark matrix.

    ``prompt_ids_by_length`` maps context_length → token IDs for that length.
    """
    results: list[RunResult] = []
    model_path = Path(config.model_path)
    total = len(config.backends) * len(config.context_lengths) * (config.num_runs + 1)
    done = 0

    for backend in config.backends:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Backend: {backend}")
            print(f"{'='*60}")

        # Load models once per backend
        cu = _compute_unit(backend)
        if verbose:
            print(f"Loading models (compute_units={backend})…", flush=True)
        decode_model = load_coreml_model(
            model_path, compute_units=cu, function_name="decode",
        )
        prefill_model = load_coreml_model(
            model_path, compute_units=cu, function_name="prefill",
        )

        timed_out_at: int | None = None  # context length where timeout occurred

        for ctx_len in config.context_lengths:
            if timed_out_at is not None:
                if verbose:
                    print(
                        f"  Skipping ctx={ctx_len} "
                        f"(timed out at ctx={timed_out_at})"
                    )
                continue

            prompt_ids = prompt_ids_by_length.get(ctx_len)
            if prompt_ids is None:
                if verbose:
                    print(f"  Skipping ctx={ctx_len} (no prompt)")
                continue

            if verbose:
                print(f"\n  Context length: {ctx_len}")

            # Warm-up run (discarded)
            if verbose:
                print(f"    Warm-up run…", flush=True)
            warmup = run_single(
                model_path, prompt_ids, ctx_len, backend,
                run_index=-1,
                decode_tokens=min(8, config.decode_tokens),
                max_seq_len=config.max_seq_len,
                timeout_s=config.timeout_s,
                enable_power=False,
                decode_model=decode_model,
                prefill_model=prefill_model,
            )
            done += 1
            if warmup.timed_out:
                timed_out_at = ctx_len
                if verbose:
                    print(f"    ⚠ Warm-up timed out — skipping remaining lengths for {backend}")
                continue

            # Measured runs
            for run_idx in range(config.num_runs):
                if verbose:
                    pct = done * 100 // total
                    print(
                        f"    Run {run_idx + 1}/{config.num_runs}  "
                        f"[{pct}% overall]",
                        flush=True,
                    )
                r = run_single(
                    model_path, prompt_ids, ctx_len, backend,
                    run_index=run_idx,
                    decode_tokens=config.decode_tokens,
                    max_seq_len=config.max_seq_len,
                    timeout_s=config.timeout_s,
                    enable_power=config.enable_power,
                    decode_model=decode_model,
                    prefill_model=prefill_model,
                )
                done += 1
                results.append(r)

                if r.timed_out:
                    timed_out_at = ctx_len
                    if verbose:
                        print(f"    ⚠ Timed out — skipping remaining for {backend}")
                    break

                if verbose and not r.error:
                    print(
                        f"      prefill: {r.prefill_tokens_per_sec:.1f} tok/s  "
                        f"decode: {r.decode_tokens_per_sec:.1f} tok/s"
                    )

    return results


def save_results(
    results: list[RunResult],
    config: BenchmarkConfig,
    output_path: Path,
) -> None:
    """Save benchmark results as JSON."""
    data = {
        "config": config.to_dict(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "node": platform.node(),
            "mac_ver": platform.mac_ver()[0],
        },
        "results": [asdict(r) for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
