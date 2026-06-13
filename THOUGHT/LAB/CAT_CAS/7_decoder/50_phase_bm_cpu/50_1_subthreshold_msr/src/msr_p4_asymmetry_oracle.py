#!/usr/bin/env python3
"""Read-only P4 definition asymmetry oracle for Exp50.

The firmware route proved the constructor-relevant P4 field is reconstructed
from runtime MSRC001_0068.  Earlier observation found cores with different P4
definition MSRs.  This probe does not write MSRs; it asks whether that existing
per-core asymmetry creates a reproducible software-visible timing/state
separator after null comparisons.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import time
from statistics import mean, pstdev

PSTATE_BASE = 0xC0010064
PSTATE_STATUS = 0xC0010063
COFVID_STATUS = 0xC0010071


def read_msr(core: int, msr: int) -> int:
    with open(f"/dev/cpu/{core}/msr", "rb", buffering=0) as handle:
        os.lseek(handle.fileno(), msr, os.SEEK_SET)
        return struct.unpack("<Q", handle.read(8))[0]


def decode_pstate(value: int) -> dict[str, object]:
    fid = value & 0x3F
    did = (value >> 6) & 0x07
    vid = (value >> 9) & 0x7F
    return {
        "raw": f"0x{value:016x}",
        "fid": fid,
        "did": did,
        "vid": vid,
        "freq_mhz": round(100.0 * (fid + 0x10) / (2 ** did), 3),
        "decoded_vcore": round(1.55 - vid * 0.0125, 6),
    }


def decode_cofvid(value: int) -> dict[str, object]:
    return {
        "raw": f"0x{value:016x}",
        "fid": value & 0x3F,
        "did": (value >> 6) & 0x07,
        "vid": (value >> 9) & 0x7F,
    }


def parse_cores(text: str) -> list[int]:
    cores: list[int] = []
    for part in text.split(","):
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            cores.extend(range(start, end + 1))
        elif part.strip():
            cores.append(int(part))
    return cores


def summarize(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "pstdev": None}
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(mean(values), 3),
        "pstdev": round(pstdev(values), 3) if len(values) > 1 else 0.0,
    }


def busy_worker(core: int, stop_event: mp.Event) -> int:
    os.sched_setaffinity(0, {core})
    value = 0x9E3779B9
    while not stop_event.is_set():
        value = ((value * 1664525) + 1013904223) & 0xFFFFFFFF
        value ^= value >> 13
    return value


def start_workers(mode: str, sample_core: int, cores: list[int]) -> tuple[mp.Event, list[mp.Process]]:
    stop_event = mp.Event()
    if mode == "baseline":
        return stop_event, []
    if mode == "self_load":
        worker_cores = [sample_core]
    elif mode == "neighbors_load":
        idx = cores.index(sample_core)
        worker_cores = [cores[(idx - 1) % len(cores)], cores[(idx + 1) % len(cores)]]
    elif mode == "other_group_load":
        worker_cores = [core for core in cores if core != sample_core]
    elif mode == "all_load":
        worker_cores = list(cores)
    else:
        raise ValueError(f"unknown mode: {mode}")

    workers = []
    for core in worker_cores:
        proc = mp.Process(target=busy_worker, args=(core, stop_event))
        proc.start()
        workers.append(proc)
    time.sleep(0.05)
    return stop_event, workers


def stop_workers(stop_event: mp.Event, workers: list[mp.Process]) -> None:
    stop_event.set()
    for proc in workers:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


def workload(iterations: int) -> tuple[int, int]:
    value = 0xC0010068
    start = time.perf_counter_ns()
    for i in range(iterations):
        value = ((value << 5) ^ (value >> 2) ^ (i * 0x45D9F3B)) & 0xFFFFFFFF
    end = time.perf_counter_ns()
    return end - start, value


def sample_core(core: int, samples: int, delay: float, workload_iters: int) -> dict[str, object]:
    os.sched_setaffinity(0, {core})
    p4_raw = read_msr(core, PSTATE_BASE + 4)
    p4 = decode_pstate(p4_raw)
    rows = []
    checksum = 0
    for _ in range(samples):
        before_pstate = read_msr(core, PSTATE_STATUS)
        before_cofvid = read_msr(core, COFVID_STATUS)
        elapsed, value = workload(workload_iters)
        checksum ^= value
        after_pstate = read_msr(core, PSTATE_STATUS)
        after_cofvid = read_msr(core, COFVID_STATUS)
        rows.append({
            "elapsed_ns": elapsed,
            "before_pstate": before_pstate,
            "after_pstate": after_pstate,
            "before_cofvid": before_cofvid,
            "after_cofvid": after_cofvid,
        })
        if delay:
            time.sleep(delay)

    timings = [row["elapsed_ns"] for row in rows]
    cofvids = [row["before_cofvid"] for row in rows] + [row["after_cofvid"] for row in rows]
    pstates = [row["before_pstate"] for row in rows] + [row["after_pstate"] for row in rows]
    decoded = [decode_cofvid(value) for value in cofvids]
    return {
        "core": core,
        "p4_msrc001_0068": p4,
        "checksum": f"0x{checksum:08x}",
        "timing_ns": summarize(timings),
        "cofvid_vid_unique": sorted({item["vid"] for item in decoded}),
        "cofvid_fid_did_vid_unique": sorted({
            f"fid={item['fid']:02x}/did={item['did']}/vid={item['vid']:02x}"
            for item in decoded
        }),
        "cofvid_raw_unique": sorted({f"0x{value:016x}" for value in cofvids}),
        "pstate_status_unique": sorted({f"0x{value:016x}" for value in pstates}),
    }


def cyclic_null_range(values: list[float], groups: list[str]) -> dict[str, object]:
    if len(set(groups)) < 2:
        return {"max_null_range_ns": None, "null_ranges_ns": []}
    offsets = [1, 3, 5, 7, 11, 13, 17]
    ranges = []
    for offset in offsets:
        shifted = groups[offset % len(groups):] + groups[:offset % len(groups)]
        bins: dict[str, list[float]] = {}
        for label, value in zip(shifted, values):
            bins.setdefault(label, []).append(value)
        means = [mean(bin_values) for bin_values in bins.values() if bin_values]
        if len(means) >= 2:
            ranges.append({"offset": offset, "range_ns": round(max(means) - min(means), 3)})
    return {
        "max_null_range_ns": max((item["range_ns"] for item in ranges), default=None),
        "null_ranges_ns": ranges,
    }


def group_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    values = []
    groups = []
    group_bins: dict[str, list[float]] = {}
    for case in cases:
        p4 = case["p4_msrc001_0068"]
        assert isinstance(p4, dict)
        label = f"p4_did={p4['did']}/vid={p4['vid']}"
        timing = case["timing_ns"]
        assert isinstance(timing, dict)
        value = float(timing["mean"])
        values.append(value)
        groups.append(label)
        group_bins.setdefault(label, []).append(value)

    group_means = {label: round(mean(items), 3) for label, items in sorted(group_bins.items())}
    if len(group_means) >= 2:
        observed_range = round(max(group_means.values()) - min(group_means.values()), 3)
    else:
        observed_range = None
    null = cyclic_null_range(values, groups)
    max_null = null["max_null_range_ns"]
    ratio = None
    if observed_range is not None and max_null not in (None, 0):
        ratio = round(observed_range / float(max_null), 6)

    return {
        "p4_group_means_ns": group_means,
        "observed_p4_group_range_ns": observed_range,
        "max_null_group_range_ns": max_null,
        "observed_over_max_null": ratio,
        "oracle_candidate": bool(ratio is not None and ratio >= 1.25 and observed_range is not None and observed_range >= 1000.0),
        "null": null,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only P4 asymmetry timing/state oracle")
    parser.add_argument("--cores", default="0-5")
    parser.add_argument("--modes", default="baseline,self_load,neighbors_load,other_group_load,all_load")
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--workload-iters", type=int, default=768)
    args = parser.parse_args()

    cores = parse_cores(args.cores)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    result = {
        "mode": "read_only_p4_asymmetry_oracle",
        "writes": False,
        "cores": cores,
        "modes": modes,
        "samples_per_case": args.samples,
        "workload_iters": args.workload_iters,
        "acceptance": [
            "No MSR writes are performed.",
            "Existing per-core MSRC001_0068 asymmetry is used as the label.",
            "Observed P4-group timing separation must beat cyclic core-label nulls by >=1.25x and >=1000 ns.",
        ],
        "cases": [],
    }

    original_affinity = os.sched_getaffinity(0)
    try:
        for mode in modes:
            mode_cases = []
            for core in cores:
                stop_event, workers = start_workers(mode, core, cores)
                try:
                    case = sample_core(core, args.samples, args.delay, args.workload_iters)
                    case["load_mode"] = mode
                    mode_cases.append(case)
                    result["cases"].append(case)
                finally:
                    stop_workers(stop_event, workers)
            result.setdefault("mode_summaries", {})[mode] = group_summary(mode_cases)
    finally:
        os.sched_setaffinity(0, original_affinity)

    candidates = [
        {"mode": mode, **summary}
        for mode, summary in result.get("mode_summaries", {}).items()
        if summary["oracle_candidate"]
    ]
    result["global_summary"] = {
        "case_count": len(result["cases"]),
        "oracle_candidate_count": len(candidates),
        "oracle_candidates": candidates,
        "p4_groups": sorted({
            f"p4_did={case['p4_msrc001_0068']['did']}/vid={case['p4_msrc001_0068']['vid']}"
            for case in result["cases"]
        }),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
