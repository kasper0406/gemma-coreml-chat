# Benchmarks

Measures prefill and decode speed of the exported model on Apple Silicon, and
the energy per token of the **CPU, GPU and ANE rails** as `powermetrics`
reports them — not whole-system energy (no DRAM, display or the rest of the
machine).

`uv run gemma-bench` drives a **Swift** executable (`benchmarks/swift/bench/`)
that links `GemmaCore` and makes the same prefill / decode / head calls as the
CLI and the iOS app, with greedy sampling. The Python side (`runner.py`,
`power.py`) runs the matrix, records power and does the arithmetic.

## Usage

```bash
# Compare two exports on the GPU and the Neural Engine at 512 and 2048 tokens
uv run gemma-bench --models base.mlpackage,new.mlpackage \
    --compute-units cpu-and-gpu,cpu-and-ne --context-lengths 512,2048 --runs 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | `gemma4-e2b.mlpackage` | Comma-separated packages, compared A/B |
| `--compute-units` | `cpu-and-gpu,cpu-and-ne` | From `cpu-and-gpu`, `cpu-and-ne`, `all`, `cpu-only` |
| `--context-lengths` | `512,2048` | Cache sizes; each must be exported |
| `--runs` | `3` | Repetitions per configuration (at least 3) |
| `--timeout` | `1800` | Per-run timeout in seconds (a first ANE compile takes minutes) |
| `--output-dir` | `benchmarks/results/<timestamp>` | Where results go |

The output directory gets `results.json` (every run, the gate record and the
summary), `summary.md` (the tables printed at the end) and `power/`, one file
per run with its raw `powermetrics` samples.

## Method

**Workload** — fixed in the Swift binary, identical for every configuration.
After loading, compiling and a warm-up (none of it measured), one process runs:

| window | what |
|---|---|
| `idle_pre` | 4 s of sleep, the model loaded |
| `prefill` | 4096 prompt tokens as whole-cache prompts, each into a fresh cache: 8 × 512 at context 512, 2 × 2048 at 2048 |
| `decode` | 256 greedy tokens after a 128-token prompt, into a cache of the context length |
| `idle_post` | 4 s of sleep |

with a 1 s pause after prefill and after decode, so a phase's power tail does
not land in the next window. The binary reports every window boundary as a
`CLOCK_UPTIME_RAW` timestamp; the runner maps them to wall-clock time.

**Power** — `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 100
-f plist` runs for the whole process. Each sample is the mean power of each
rail over the `elapsed_ns` since the previous one, so samples tile time. Their
`timestamp` is wall-clock, truncated to the second, at the end of the sample:
intersecting the constraint every sample puts on the start of the first one
anchors the whole sequence to a few milliseconds (the width is recorded per
run, next to how long after its reconstructed end each sample arrived).

**Energy** — for each phase, `Σ P × overlap(sample, phase window)` per rail:
every sample counts for exactly the part of its interval inside the window.
Idle power is the mean of the two idle windows; *above idle* subtracts it
over the phase's duration. Reported per token: ms/token, mJ/token (total and
above idle) and the mean W of each rail.

**Quiet machine** — the *machine state* (power source, Low Power Mode,
display on, screen locked) and the *background CPU*: the CPU use of every
process except the harness — this process, its descendants (the bench
binary, `powermetrics`, `top`) and its parent process (whatever launched it;
not the parent's other children) — from one-second `top` samples of all
processes; nothing else is exempt, and busy processes are logged.

After priming, a calibration takes 60 samples of the idle background, polling
the machine state with each; it wants AC power, no Low Power Mode, and that
state readable and unchanged throughout. One rule, in points of one core:
the calibration is used only if its mean is at most 25 and its busiest
sample at most 55 (25 + 30), else it is retaken, for up to 15 minutes.
Against its mean *m*, a window averaging above *m* + 5 has drifted and a
sample above *m* + 30 is a spike. The 5 is the sensitivity — sustained
interference of a twentieth of a core or more is caught; the 30 tolerates
brief housekeeping (a one-second burst) but not a third of a core; the 25
refuses a machine that is not idle to begin with.

Before every run the gate wants the calibrated machine state and three
samples within the limits; otherwise it pauses 15 s and retries, and gives
the run up after 15 minutes. During every run a monitor polls the machine
state and takes a CPU sample back to back (a poll every ~2 s) from before the
bench launches until after it exits, load and warm-up included. A run is not
kept if any poll failed or differed from the calibrated state, if its span
from the start of `idle_pre` to the end of `idle_post` saw a spike or more
than 2 s unsampled, or if any one of its windows (`idle_pre`, `prefill`,
`decode`, `idle_post`) drifted. Load and warm-up are exempt from the CPU
check only: nothing is measured there, and the CoreML/ANE daemons that load
and compile for the bench (`aned`, `ANECompilerService`, …) are outside the
harness, so their CPU there is the workload's; whatever the load leaves
running shows up in `idle_pre`. A state or process command that fails,
times out or prints nothing parseable fails the calibration, the gate or
the run. The calibration and every gate record land in `results.json`, and
every run's CPU samples and state polls with it. The runner refuses to
start at all on battery.

**Anchor** — a run is not kept if its power samples' stamps admit no common
start (a dropped sample or a clock step), a `powermetrics` document was
malformed, or the anchor is wider than 1% of the run's shortest window (the
midpoint estimate then moves a boundary by at most 0.5% of any window).

**Configurations** — each gets an id, `c<index>-<package stem>-<units>-<ctx>`,
which names its raw-sample files and its summary row; the summary lists
which package path each id ran, so two packages with the same file name stay
apart.

**Order** — one unmeasured priming run per configuration (it compiles and
caches every function), then the repetitions, with the configuration order
rotated by one each repetition.

**Summary** — the median of the kept runs (gate passed, no error) with the
min–max spread; dropped runs are listed with the reason.

## Prerequisites

- **Xcode** (not just the Command Line Tools). The Swift bench targets
  macOS 15, which needs the Xcode SDK:

  ```bash
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  ```

- **Passwordless `powermetrics`**. Add to `/etc/sudoers`:

  ```
  %admin ALL = (root) NOPASSWD: /usr/bin/powermetrics
  ```

## Standalone Swift bench (legacy)

`benchmarks/swift/model_bench.swift` is a zero-dependency sanity check that
loads a `.mlpackage`, compiles, and runs a short prefill+decode without going
through `GemmaCore`. Build and run directly with `swiftc`; see the header in
that file.

## Multifunction + RangeDim diagnostic

`benchmarks/multifunction_rangedim_bug.{py,swift}` are the minimum repro for
the E5RT multifunction/RangeDim loading bug that prompted the
`remove_broadcast_tiles` MIL pass fix. Kept for regression-watching.
