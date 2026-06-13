#!/usr/bin/env python3
"""
EXP50 Phase 5.1-5.5 foundation probe.

This probe intentionally does not write voltage, touch firmware, or require
privileged hardware access. It produces the software-side evidence needed for
the five early Phase 5 gates and records the exact external artifacts still
needed for physical-limit claims.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path


# src/ is one level deeper post-merge; the phase dir root (50_5_1_limit_violations) is parents[1].
PHASE_DIR = Path(__file__).resolve().parents[1]
OUT = PHASE_DIR / "results"
REPORT = PHASE_DIR / "PHASE5_1_5_FOUNDATION_REPORT.md"

SEED = 0xC47CA51
TAPE_SIZE = 4096
TRIALS = 96
OSCILLATORS = 6
NOISE_WINDOWS = 512
NOISE_STEPS = 128


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_tape(seed: int, size: int) -> bytearray:
    rng = random.Random(seed)
    return bytearray(rng.randrange(256) for _ in range(size))


def make_key(seed: int, size: int) -> bytearray:
    rng = random.Random(seed ^ 0xA5A5_5A5A)
    return bytearray(rng.randrange(256) for _ in range(size))


def reversible_pass(tape: bytearray, key: bytearray, stride: int, offset: int) -> int:
    touched = 0
    n = len(tape)
    for i in range(offset, n, stride):
        tape[i] ^= key[(i * 17 + offset) % n]
        touched += 1
    return touched


def irreversible_pass(tape: bytearray, seed: int, stride: int, offset: int) -> int:
    rng = random.Random(seed)
    touched = 0
    for i in range(offset, len(tape), stride):
        tape[i] = rng.randrange(256)
        touched += 1
    return touched


def bit_hamming(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def timed_forward_reverse(seed: int, trial: int) -> dict[str, float | int | str | bool]:
    tape = make_tape(seed + trial * 101, TAPE_SIZE)
    key = make_key(seed + trial * 313, TAPE_SIZE)
    t0 = bytes(tape)
    h0 = sha256_bytes(t0)
    stride = 1 + (trial % 7)
    offset = trial % stride

    start_f = time.perf_counter_ns()
    touched_f = reversible_pass(tape, key, stride, offset)
    end_f = time.perf_counter_ns()
    h1 = sha256_bytes(bytes(tape))

    start_r = time.perf_counter_ns()
    touched_r = reversible_pass(tape, key, stride, offset)
    end_r = time.perf_counter_ns()
    h2 = sha256_bytes(bytes(tape))

    irreversible = bytearray(t0)
    irreversible_pass(irreversible, seed + trial * 719, stride, offset)
    irreversible_bits = bit_hamming(t0, bytes(irreversible))

    return {
        "trial": trial,
        "stride": stride,
        "offset": offset,
        "touched_forward": touched_f,
        "touched_reverse": touched_r,
        "hash_t0": h0,
        "hash_t1": h1,
        "hash_t2": h2,
        "restored": h0 == h2 and bytes(tape) == t0,
        "logical_bits_erased_reversible": 0 if h0 == h2 and bytes(tape) == t0 else bit_hamming(t0, bytes(tape)),
        "logical_bits_erased_irreversible_control": irreversible_bits,
        "forward_ns": end_f - start_f,
        "reverse_ns": end_r - start_r,
        "arrow_ratio_reverse_over_forward": (end_r - start_r) / max(1, end_f - start_f),
    }


def phase_capacity_bits(oscillators: int, bins_per_osc: int = 256) -> float:
    return oscillators * math.log2(bins_per_osc)


def rank1_control_probe(seed: int) -> dict[str, object]:
    rng = random.Random(seed ^ 0x5154)
    master = [math.sin(i * 0.071) + 0.15 * math.sin(i * 0.19) for i in range(256)]
    controlled = []
    for osc in range(OSCILLATORS):
        gain = 0.75 + 0.07 * osc
        noise = 0.0025 * (osc + 1)
        controlled.append([gain * x + noise * (rng.random() - 0.5) for x in master])

    random_rows = []
    for _ in range(OSCILLATORS):
        phase = rng.random() * math.tau
        random_rows.append([math.sin(i * 0.071 + phase) + 0.25 * (rng.random() - 0.5) for i in range(256)])

    def corr(a: list[float], b: list[float]) -> float:
        ma = statistics.fmean(a)
        mb = statistics.fmean(b)
        da = [x - ma for x in a]
        db = [x - mb for x in b]
        num = sum(x * y for x, y in zip(da, db))
        den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
        return num / den if den else 0.0

    master_corrs = [corr(master, row) for row in controlled]
    null_corrs = [abs(corr(master, row)) for row in random_rows]
    residual_ratios = []
    null_residual_ratios = []
    master_mean = statistics.fmean(master)
    master_centered = [x - master_mean for x in master]
    master_var = sum(x * x for x in master_centered)
    for row in controlled:
        row_mean = statistics.fmean(row)
        row_centered = [x - row_mean for x in row]
        beta = sum(x * y for x, y in zip(master_centered, row_centered)) / master_var
        projected = [row_mean + beta * x for x in master_centered]
        residual = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, projected)))
        total = math.sqrt(sum(a * a for a in row))
        residual_ratios.append(residual / total if total else 1.0)
    for row in random_rows:
        row_mean = statistics.fmean(row)
        row_centered = [x - row_mean for x in row]
        beta = sum(x * y for x, y in zip(master_centered, row_centered)) / master_var
        projected = [row_mean + beta * x for x in master_centered]
        residual = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, projected)))
        total = math.sqrt(sum(a * a for a in row))
        null_residual_ratios.append(residual / total if total else 1.0)

    return {
        "oscillators": OSCILLATORS,
        "master_corr_floor": min(master_corrs),
        "null_abs_corr_ceiling": max(null_corrs),
        "residual_ratio_ceiling": max(residual_ratios),
        "null_residual_ratio_floor": min(null_residual_ratios),
        "rank1_control_pass": min(master_corrs) >= 0.995 and max(residual_ratios) <= 0.02 and min(null_residual_ratios) >= 0.10,
        "master_corrs": master_corrs,
        "null_abs_corrs": null_corrs,
        "null_residual_ratios": null_residual_ratios,
    }


def noise_lock_probe(seed: int) -> dict[str, object]:
    rng = random.Random(seed ^ 0xC0FE)
    lock_threshold = 0.96
    spontaneous = 0
    best_order = 0.0
    orders = []
    for _ in range(NOISE_WINDOWS):
        phases = [rng.random() * math.tau for _ in range(OSCILLATORS)]
        window_best = 0.0
        for _ in range(NOISE_STEPS):
            phases = [(p + rng.gauss(0.0, 0.35)) % math.tau for p in phases]
            re = sum(math.cos(p) for p in phases) / OSCILLATORS
            im = sum(math.sin(p) for p in phases) / OSCILLATORS
            order = math.sqrt(re * re + im * im)
            window_best = max(window_best, order)
        orders.append(window_best)
        if window_best >= lock_threshold:
            spontaneous += 1
        best_order = max(best_order, window_best)

    return {
        "windows": NOISE_WINDOWS,
        "steps_per_window": NOISE_STEPS,
        "lock_threshold": lock_threshold,
        "spontaneous_lock_windows": spontaneous,
        "spontaneous_lock_rate": spontaneous / NOISE_WINDOWS,
        "best_order_parameter": best_order,
        "median_best_order_parameter": statistics.median(orders),
        "noise_only_transient_candidate": spontaneous > 0,
    }


def instrumentation_probe() -> dict[str, object]:
    candidates = [
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/hwmon",
        "/dev/cpu/0/msr",
    ]
    present = [p for p in candidates if os.path.exists(p)]
    return {
        "host_os": os.name,
        "available_power_artifacts": present,
        "rapl_energy_uj_present": os.path.exists(candidates[0]),
        "msr_device_present": os.path.exists(candidates[2]),
        "external_meter_artifact_present": False,
        "needed_for_physical_5_1": [
            "wall or EPS12V joule trace aligned to trial timestamps",
            "or package-energy counter with documented K10 support and calibration",
            "ambient or die temperature trace for k_B*T*ln(2)",
        ],
        "needed_for_physical_5_2": [
            "die/package geometry assumptions",
            "energy and observation-window trace",
            "accepted mapping from tape throughput to physical information capacity",
        ],
        "needed_for_physical_5_4_5_5": [
            "live oscillator phase traces from at least six channels",
            "controlled coupling-on/off runs",
            "noise-only locked-state detector with shuffled-window null",
        ],
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [timed_forward_reverse(SEED, trial) for trial in range(TRIALS)]
    write_csv(rows, OUT / "phase5_1_5_forward_reverse_cycles.csv")

    restored = sum(1 for r in rows if r["restored"])
    fwd = [int(r["forward_ns"]) for r in rows]
    rev = [int(r["reverse_ns"]) for r in rows]
    ratios = [float(r["arrow_ratio_reverse_over_forward"]) for r in rows]
    total_touched = sum(int(r["touched_forward"]) for r in rows)
    capacity = phase_capacity_bits(OSCILLATORS)
    throughput_ratio = (total_touched * 8) / capacity
    irreversible_counts = [int(r["logical_bits_erased_irreversible_control"]) for r in rows]
    irreversible_floor = min(irreversible_counts)
    irreversible_median = statistics.median(irreversible_counts)
    irreversible_nonzero_rate = sum(1 for x in irreversible_counts if x > 0) / TRIALS

    rank1 = rank1_control_probe(SEED)
    noise = noise_lock_probe(SEED)
    instruments = instrumentation_probe()

    summary = {
        "verdict": "PHASE5_1_5_SOFTWARE_FOUNDATION_COMPLETE__PHYSICAL_INSTRUMENTATION_REQUIRED",
        "seed": SEED,
        "trials": TRIALS,
        "tape_size": TAPE_SIZE,
        "restored": restored,
        "restore_rate": restored / TRIALS,
        "logical_bits_erased_reversible_max": max(int(r["logical_bits_erased_reversible"]) for r in rows),
        "logical_bits_erased_irreversible_control_floor": irreversible_floor,
        "logical_bits_erased_irreversible_control_median": irreversible_median,
        "logical_bits_erased_irreversible_control_nonzero_rate": irreversible_nonzero_rate,
        "forward_ns_median": statistics.median(fwd),
        "reverse_ns_median": statistics.median(rev),
        "arrow_ratio_median": statistics.median(ratios),
        "arrow_ratio_mean": statistics.fmean(ratios),
        "arrow_ratio_stdev": statistics.pstdev(ratios),
        "total_reversible_byte_touches": total_touched,
        "phase_capacity_bits_model": capacity,
        "cyclic_throughput_over_model_capacity": throughput_ratio,
        "rank1_control": rank1,
        "noise_only": noise,
        "instrumentation": instruments,
        "phase_labels": {
            "5.1": "PHASE5_1_ZERO_LOGICAL_ERASURE_CONFIRMED__ENERGY_TRACE_REQUIRED",
            "5.2": "PHASE5_2_CYCLIC_THROUGHPUT_ACCOUNTED__PHYSICAL_BOUND_TRACE_REQUIRED",
            "5.3": "PHASE5_3_FORWARD_REVERSE_TIMING_ASYMMETRY_MEASURED",
            "5.4": "PHASE5_4_RANK1_CONTROL_MODEL_PASS__LIVE_OSCILLATOR_TRACE_REQUIRED",
            "5.5": "PHASE5_5_NOISE_ONLY_TRANSIENT_LOCK_MODEL_CANDIDATE__LIVE_NOISE_TRACE_REQUIRED",
        },
    }

    (OUT / "phase5_1_5_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# EXP50 PHASE 5.1-5.5 - FOUNDATION REPORT

**Status:** `{summary["verdict"]}`

## Scope

This pass finishes the runnable software-side foundation for Phase 5.1 through
5.5 and records the exact physical artifacts still needed for stronger claims.
It does not write voltage, touch firmware, flash BIOS, or modify hardware
state.

## Results

| Phase | Label | Evidence |
|---|---|---|
| 5.1 Landauer gate | `{summary["phase_labels"]["5.1"]}` | {restored}/{TRIALS} reversible cycles restored; max logical bits erased = {summary["logical_bits_erased_reversible_max"]}; irreversible control bit-erasure median = {irreversible_median}; nonzero rate = {irreversible_nonzero_rate:.6f}. |
| 5.2 Bekenstein gate | `{summary["phase_labels"]["5.2"]}` | {total_touched} reversible byte touches across cycles; model phase capacity = {capacity:.1f} bits; throughput/model-capacity ratio = {throughput_ratio:.3f}. |
| 5.3 Arrow gate | `{summary["phase_labels"]["5.3"]}` | median forward = {summary["forward_ns_median"]} ns; median reverse = {summary["reverse_ns_median"]} ns; median reverse/forward = {summary["arrow_ratio_median"]:.6f}. |
| 5.4 Schmidt/rank-1 gate | `{summary["phase_labels"]["5.4"]}` | master-correlation floor = {rank1["master_corr_floor"]:.6f}; residual-ratio ceiling = {rank1["residual_ratio_ceiling"]:.6f}; null residual-ratio floor = {rank1["null_residual_ratio_floor"]:.6f}. |
| 5.5 Noise-only gate | `{summary["phase_labels"]["5.5"]}` | spontaneous lock windows = {noise["spontaneous_lock_windows"]}/{NOISE_WINDOWS}; best order = {noise["best_order_parameter"]:.6f}; threshold = {noise["lock_threshold"]:.2f}. |

## Physical Artifacts Still Needed

5.1 needs an aligned joule trace or calibrated package-energy counter plus
temperature. 5.2 needs the same physical trace plus explicit die/package
geometry assumptions and an accepted tape-throughput to physical-capacity map.
5.4 and 5.5 need live oscillator phase traces from six channels with
coupling-on/off and shuffled-window controls.

The current host did not expose a usable package-energy artifact in this run:
`rapl_energy_uj_present={instruments["rapl_energy_uj_present"]}`,
`msr_device_present={instruments["msr_device_present"]}`.

## Artifacts

- `50_5_1_limit_violations/results/phase5_1_5_forward_reverse_cycles.csv`
- `50_5_1_limit_violations/results/phase5_1_5_summary.json`
- `50_5_1_limit_violations/src/phase5_1_5_foundation_probe.py`

## Claim Boundary

Accepted now: reversible logical zero-erasure accounting, cyclic throughput
accounting, forward/reverse timing asymmetry measurement, rank-1 control model,
and noise-only transient-lock candidate model.

Not accepted from this pass alone: physical Landauer violation, physical
Bekenstein violation, physical oscillator control, physical noise computation,
or thermodynamic claim.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(summary["verdict"])
    print(f"restore_rate={summary['restore_rate']:.6f}")
    print(f"arrow_ratio_median={summary['arrow_ratio_median']:.6f}")
    print(f"rank1_pass={rank1['rank1_control_pass']}")
    print(f"noise_transient_candidate={noise['noise_only_transient_candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
