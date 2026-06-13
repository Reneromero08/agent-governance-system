#!/usr/bin/env python3
"""
EXP50 Phase 5.1-5.5 proxy hardening.

This runner extends the Phenom-side foundation without claiming physical limit
violations. It hardens the software-visible parts still worth pushing:

- 5.3: pinned forward/reverse timing by core.
- 5.4: one reference coordinate predicting multiple timing readout channels.
- 5.5: noise-only jitter windows against shuffled-window nulls.

No voltage, clock, firmware, MSR write, or board-state mutation is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "proxy_hardening"
REPORT = ROOT / "PHASE5_1_5_PROXY_HARDENING.md"

SEED = 0x50515A
CORES = [0, 1, 2, 3, 4, 5]
TRIALS_PER_CORE = 36
TAPE_SIZE = 8192
JITTER_WINDOWS = 80
JITTER_SAMPLES = 384
DRIVE_LEVELS = [1, 2, 4, 8, 16, 32]


def set_affinity(core: int) -> bool:
    try:
        os.sched_setaffinity(0, {core})
        return True
    except Exception:
        return False


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="ascii", errors="ignore").strip()
    except Exception:
        return None


def locate_k10temp() -> str | None:
    base = Path("/sys/class/hwmon")
    if not base.exists():
        return None
    for d in base.glob("hwmon*"):
        name = read_text(str(d / "name"))
        if name == "k10temp":
            p = d / "temp1_input"
            if p.exists():
                return str(p)
    return None


K10TEMP = locate_k10temp()


def temp_c() -> float | None:
    if not K10TEMP:
        return None
    raw = read_text(K10TEMP)
    if raw is None:
        return None
    try:
        return int(raw) / 1000.0
    except ValueError:
        return None


def cur_freq_khz(core: int) -> int | None:
    raw = read_text(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def tape(seed: int, size: int) -> bytearray:
    rng = random.Random(seed)
    return bytearray(rng.randrange(256) for _ in range(size))


def digest(data: bytearray) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def reversible_round(buf: bytearray, key: bytearray, step: int, offset: int) -> int:
    touched = 0
    n = len(buf)
    for i in range(offset, n, step):
        buf[i] ^= key[(i * 33 + offset * 17) % n]
        touched += 1
    return touched


def forward_reverse_trial(seed: int, trial: int, level: int) -> dict[str, object]:
    buf = tape(seed + trial * 101 + level * 1009, TAPE_SIZE)
    key = tape(seed ^ 0xA55A55AA ^ (trial * 313) ^ level, TAPE_SIZE)
    h0 = digest(buf)
    original = bytes(buf)
    step = max(1, 9 - min(level, 8))
    offset = trial % step

    t0 = time.perf_counter_ns()
    touched_f = reversible_round(buf, key, step, offset)
    t1 = time.perf_counter_ns()
    touched_r = reversible_round(buf, key, step, offset)
    t2 = time.perf_counter_ns()

    restored = h0 == digest(buf) and bytes(buf) == original
    return {
        "trial": trial,
        "level": level,
        "touched_f": touched_f,
        "touched_r": touched_r,
        "forward_ns": t1 - t0,
        "reverse_ns": t2 - t1,
        "ratio": (t2 - t1) / max(1, t1 - t0),
        "restored": restored,
    }


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def corr(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ma = statistics.fmean(a)
    mb = statistics.fmean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return sum(x * y for x, y in zip(da, db)) / den if den else 0.0


def pinned_timing() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    core_summary: dict[str, object] = {}
    for core in CORES:
        affinity_ok = set_affinity(core)
        before_t = temp_c()
        before_f = cur_freq_khz(core)
        core_rows = []
        for trial in range(TRIALS_PER_CORE):
            row = forward_reverse_trial(SEED + core * 10000, trial, 4)
            row["core"] = core
            core_rows.append(row)
            rows.append(row)
        after_t = temp_c()
        after_f = cur_freq_khz(core)
        ratios = [float(r["ratio"]) for r in core_rows]
        core_summary[str(core)] = {
            "affinity_ok": affinity_ok,
            "restore_rate": sum(1 for r in core_rows if r["restored"]) / len(core_rows),
            "forward_ns_median": median([float(r["forward_ns"]) for r in core_rows]),
            "reverse_ns_median": median([float(r["reverse_ns"]) for r in core_rows]),
            "ratio_median": median(ratios),
            "ratio_mad": median([abs(x - median(ratios)) for x in ratios]),
            "temp_c_before": before_t,
            "temp_c_after": after_t,
            "freq_khz_before": before_f,
            "freq_khz_after": after_f,
        }
    all_ratios = [float(r["ratio"]) for r in rows]
    return {
        "rows": rows,
        "core_summary": core_summary,
        "all_restore_rate": sum(1 for r in rows if r["restored"]) / len(rows),
        "ratio_median_all": median(all_ratios),
        "ratio_spread_max_minus_min": max(all_ratios) - min(all_ratios),
        "gate": "PHASE5_3_PINNED_TIMING_HARDENED_PROXY",
    }


def reference_channel_matrix() -> dict[str, object]:
    channels: dict[str, list[float]] = {str(core): [] for core in CORES}
    reference: list[float] = []
    rows: list[dict[str, object]] = []
    for level in DRIVE_LEVELS:
        reference.append(float(level))
        for core in CORES:
            set_affinity(core)
            samples = [forward_reverse_trial(SEED + core * 3333, i, level) for i in range(10)]
            metric = median([float(s["forward_ns"]) + float(s["reverse_ns"]) for s in samples])
            channels[str(core)].append(metric)
            rows.append({"core": core, "level": level, "metric_ns": metric})
    corrs = {core: corr(reference, vals) for core, vals in channels.items()}
    abs_floor = min(abs(v) for v in corrs.values())
    signs = [1 if v >= 0 else -1 for v in corrs.values()]
    sign_agreement = max(signs.count(1), signs.count(-1)) / len(signs)

    # Rank-1 proxy: center each channel over drive levels and measure how much
    # channel energy is explained by the signed reference direction.
    total_energy = 0.0
    residual_energy = 0.0
    ref_mean = statistics.fmean(reference)
    ref_c = [x - ref_mean for x in reference]
    ref_norm2 = sum(x * x for x in ref_c)
    for vals in channels.values():
        mean_v = statistics.fmean(vals)
        val_c = [x - mean_v for x in vals]
        beta = sum(x * y for x, y in zip(ref_c, val_c)) / ref_norm2 if ref_norm2 else 0.0
        pred = [beta * x for x in ref_c]
        total_energy += sum(x * x for x in val_c)
        residual_energy += sum((x - y) ** 2 for x, y in zip(val_c, pred))
    explained = 1.0 - residual_energy / total_energy if total_energy else 0.0

    return {
        "rows": rows,
        "reference_levels": reference,
        "channels": channels,
        "reference_correlations": corrs,
        "abs_corr_floor": abs_floor,
        "sign_agreement": sign_agreement,
        "rank1_explained_energy": explained,
        "gate": "PHASE5_4_REFERENCE_TO_MULTICHANNEL_PROXY_MEASURED",
    }


def jitter_window(core: int, samples: int) -> list[int]:
    set_affinity(core)
    out = []
    prev = time.perf_counter_ns()
    x = 0x12345678 ^ core
    for _ in range(samples):
        x = ((x << 13) ^ (x >> 7) ^ (x << 17)) & 0xFFFFFFFF
        now = time.perf_counter_ns()
        out.append(now - prev)
        prev = now
    return out


def order_parameter(values: list[int]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    centered = [v - med for v in values]
    scale = statistics.median([abs(v) for v in centered]) or 1.0
    phases = [(v / scale) % (2.0 * math.pi) for v in centered]
    re = sum(math.cos(p) for p in phases) / len(phases)
    im = sum(math.sin(p) for p in phases) / len(phases)
    return math.sqrt(re * re + im * im)


def noise_jitter_null() -> dict[str, object]:
    rows = []
    real_orders = []
    shuffled_orders = []
    rng = random.Random(SEED ^ 0x5155)
    for core in CORES:
        for window in range(JITTER_WINDOWS):
            vals = jitter_window(core, JITTER_SAMPLES)
            real = order_parameter(vals)
            shuffled = vals[:]
            rng.shuffle(shuffled)
            null = order_parameter(shuffled)
            real_orders.append(real)
            shuffled_orders.append(null)
            rows.append({"core": core, "window": window, "real_order": real, "shuffled_order": null})
    delta = median(real_orders) - median(shuffled_orders)
    # Since this order metric is distributional, the shuffle null is expected to
    # match if there is no temporal structure. A small delta is a hard cap.
    return {
        "rows": rows,
        "real_order_median": median(real_orders),
        "shuffled_order_median": median(shuffled_orders),
        "median_delta": delta,
        "real_p95": sorted(real_orders)[int(0.95 * (len(real_orders) - 1))],
        "shuffled_p95": sorted(shuffled_orders)[int(0.95 * (len(shuffled_orders) - 1))],
        "gate": "PHASE5_5_NOISE_JITTER_SHUFFLE_NULL_MEASURED",
    }


def verdict(timing: dict[str, object], matrix: dict[str, object], noise: dict[str, object]) -> str:
    labels = [
        "PHASE5_3_PINNED_TIMING_HARDENED_PROXY",
        "PHASE5_4_REFERENCE_TO_MULTICHANNEL_PROXY_MEASURED",
        "PHASE5_5_NOISE_JITTER_SHUFFLE_NULL_MEASURED",
    ]
    if float(timing["all_restore_rate"]) == 1.0:
        labels.append("RESTORATION_INTACT")
    if float(matrix["rank1_explained_energy"]) >= 0.80 and float(matrix["abs_corr_floor"]) >= 0.75:
        labels.append("RANK1_PROXY_STRONG")
    else:
        labels.append("RANK1_PROXY_PARTIAL")
    if abs(float(noise["median_delta"])) <= 0.02:
        labels.append("NOISE_TEMPORAL_STRUCTURE_NOT_SEPARATED_FROM_SHUFFLE")
    else:
        labels.append("NOISE_TEMPORAL_STRUCTURE_CANDIDATE_PROXY")
    return "__".join(labels)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    timing = pinned_timing()
    matrix = reference_channel_matrix()
    noise = noise_jitter_null()
    summary = {
        "verdict": verdict(timing, matrix, noise),
        "seed": SEED,
        "host": os.uname().nodename if hasattr(os, "uname") else os.name,
        "k10temp_path": K10TEMP,
        "pinned_timing": {k: v for k, v in timing.items() if k != "rows"},
        "reference_channel_matrix": {k: v for k, v in matrix.items() if k != "rows"},
        "noise_jitter_null": {k: v for k, v in noise.items() if k != "rows"},
        "claim_boundary": [
            "Proxy hardening only.",
            "No physical Landauer or Bekenstein claim.",
            "No physical oscillator-control claim.",
            "No physical noise-computation claim.",
        ],
    }
    (OUT / "phase5_1_5_proxy_hardening_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    report = f"""# EXP50 PHASE 5.1-5.5 - PROXY HARDENING

**Status:** `{summary["verdict"]}`

## Scope

This run pushes Phase 5.3-5.5 inside current Phenom software observability.
It is a proxy hardening run, not a physical-limit proof. It does not write
voltage, clocks, firmware, or MSRs.

## Results

| Gate | Result |
|---|---|
| 5.3 pinned timing | restore rate `{timing["all_restore_rate"]:.6f}`, all-core median reverse/forward `{timing["ratio_median_all"]:.6f}` |
| 5.4 reference-to-channel proxy | abs correlation floor `{matrix["abs_corr_floor"]:.6f}`, sign agreement `{matrix["sign_agreement"]:.6f}`, rank-1 explained energy `{matrix["rank1_explained_energy"]:.6f}` |
| 5.5 noise jitter shuffle null | real median `{noise["real_order_median"]:.6f}`, shuffled median `{noise["shuffled_order_median"]:.6f}`, delta `{noise["median_delta"]:.6f}` |

## Interpretation

5.3 is hardened as a pinned timing proxy across cores. 5.4 gains a measured
many-channel software proxy for one reference coordinate. 5.5 gains a
shuffle-null cap for noise-only jitter: if the median delta is near zero, the
current software-visible noise ordering is not separated from the shuffled
null.

## Claim Boundary

This does not close physical 5.1, 5.2, 5.4, or 5.5. Those still require the
external/physical artifacts listed in `PHASE5_1_5_LIVE_RUNBOOK.md`.

## Artifacts

- `50_5_1_limit_violations/results/proxy_hardening/phase5_1_5_proxy_hardening_summary.json`
- `50_5_1_limit_violations/src/phase5_1_5_proxy_hardening.py`
"""
    REPORT.write_text(report, encoding="utf-8")
    print(summary["verdict"])
    print(f"timing_restore={timing['all_restore_rate']:.6f}")
    print(f"rank1_explained={matrix['rank1_explained_energy']:.6f}")
    print(f"noise_delta={noise['median_delta']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
