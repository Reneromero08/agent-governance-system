#!/usr/bin/env python3
"""
Read-only state-window timing oracle for Exp50.

The harness samples runtime COFVID/PSTATE state, executes a fixed local integer
workload, bins timing by observed state, and compares observed state separation
against deterministic cyclic-label nulls. No MSR writes are performed.
"""

import argparse
import json
import multiprocessing as mp
import os
import struct
import time
from statistics import mean, pstdev


PSTATE_STATUS = 0xC0010063
COFVID_STATUS = 0xC0010071


def read_msr(core, msr):
    with open(f"/dev/cpu/{core}/msr", "rb", buffering=0) as handle:
        os.lseek(handle.fileno(), msr, os.SEEK_SET)
        return struct.unpack("<Q", handle.read(8))[0]


def parse_cores(text):
    cores = []
    for part in text.split(","):
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    return cores


def decode_cofvid(value):
    return {
        "fid": value & 0x3F,
        "did": (value >> 6) & 0x07,
        "vid": (value >> 9) & 0x7F,
    }


def state_key(pstate_status, cofvid_raw):
    decoded = decode_cofvid(cofvid_raw)
    return (
        f"pstate={pstate_status:x}"
        f"/fid={decoded['fid']:02x}"
        f"/did={decoded['did']}"
        f"/vid={decoded['vid']:02x}"
    )


def fixed_workload(iterations):
    value = 0xC0DEC0DE
    start = time.perf_counter_ns()
    for index in range(iterations):
        value = ((value * 1664525) + 1013904223 + index) & 0xFFFFFFFF
        value ^= (value >> 11)
    end = time.perf_counter_ns()
    return end - start, value


def busy_worker(core, stop_event):
    os.sched_setaffinity(0, {core})
    value = 0x2468ACE0
    while not stop_event.is_set():
        value = ((value << 5) ^ (value >> 2) ^ 0xA5A5A5A5) & 0xFFFFFFFF
    return value


def start_workers(mode, sample_core, all_cores):
    stop_event = mp.Event()
    if mode == "baseline":
        return stop_event, []
    if mode == "self_load":
        worker_cores = [sample_core]
    elif mode == "neighbor_load":
        worker_cores = [all_cores[(all_cores.index(sample_core) + 1) % len(all_cores)]]
    elif mode == "all_load":
        worker_cores = list(all_cores)
    else:
        raise ValueError(f"unknown mode: {mode}")

    workers = []
    for core in worker_cores:
        proc = mp.Process(target=busy_worker, args=(core, stop_event))
        proc.start()
        workers.append(proc)
    time.sleep(0.05)
    return stop_event, workers


def stop_workers(stop_event, workers):
    stop_event.set()
    for proc in workers:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


def summarize(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "pstdev": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 3),
        "pstdev": round(pstdev(values), 3) if len(values) > 1 else 0.0,
    }


def bin_means(labels, timings, min_count):
    bins = {}
    for label, timing in zip(labels, timings):
        bins.setdefault(label, []).append(timing)
    summaries = {
        label: summarize(values)
        for label, values in sorted(bins.items())
        if len(values) >= min_count
    }
    if len(summaries) < 2:
        return summaries, None
    means = [item["mean"] for item in summaries.values()]
    return summaries, round(max(means) - min(means), 3)


def null_ranges(labels, timings, min_count):
    if len(labels) < 2:
        return []
    offsets = [1, 3, 7, 13, 31, 63, 127]
    ranges = []
    for offset in offsets:
        shift = offset % len(labels)
        if shift == 0:
            continue
        shifted = labels[shift:] + labels[:shift]
        _, spread = bin_means(shifted, timings, min_count)
        if spread is not None:
            ranges.append({"offset": offset, "range_ns": spread})
    return ranges


def sample_case(core, samples, delay, workload_iters, min_count):
    os.sched_setaffinity(0, {core})
    labels = []
    timings = []
    checksum = 0
    for _ in range(samples):
        pstate_status = read_msr(core, PSTATE_STATUS)
        cofvid_raw = read_msr(core, COFVID_STATUS)
        elapsed, value = fixed_workload(workload_iters)
        checksum ^= value
        labels.append(state_key(pstate_status, cofvid_raw))
        timings.append(elapsed)
        if delay:
            time.sleep(delay)

    state_bins, observed_range = bin_means(labels, timings, min_count)
    nulls = null_ranges(labels, timings, min_count)
    max_null = max((item["range_ns"] for item in nulls), default=None)
    if observed_range is not None and max_null is not None:
        observed_over_max_null = round(observed_range / max_null, 6) if max_null else None
    else:
        observed_over_max_null = None

    return {
        "core": core,
        "samples": samples,
        "workload_iters": workload_iters,
        "checksum": f"0x{checksum:08x}",
        "state_count_raw": len(set(labels)),
        "state_count_with_min_count": len(state_bins),
        "timing_ns": summarize(timings),
        "state_bins": state_bins,
        "observed_state_mean_range_ns": observed_range,
        "null_ranges_ns": nulls,
        "max_null_range_ns": max_null,
        "observed_over_max_null": observed_over_max_null,
        "oracle_candidate": (
            observed_over_max_null is not None
            and observed_over_max_null >= 1.25
            and observed_range is not None
            and observed_range >= 1000.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only Exp50 state-window timing oracle")
    parser.add_argument("--cores", default="0-5")
    parser.add_argument("--modes", default="baseline,self_load,neighbor_load,all_load")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--workload-iters", type=int, default=256)
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()

    cores = parse_cores(args.cores)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    result = {
        "mode": "read_only_state_window_oracle",
        "writes": False,
        "cores": cores,
        "modes": modes,
        "samples_per_case": args.samples,
        "workload_iters": args.workload_iters,
        "min_count_per_state": args.min_count,
        "acceptance": [
            "No MSR writes are performed.",
            "Timing is binned by observed COFVID/PSTATE_STATUS state.",
            "Observed state separation is compared against deterministic cyclic-label nulls.",
            "oracle_candidate is true only for state separation beyond nulls.",
        ],
        "cases": [],
    }

    original_affinity = os.sched_getaffinity(0)
    try:
        for mode in modes:
            for core in cores:
                stop_event, workers = start_workers(mode, core, cores)
                try:
                    case = sample_case(core, args.samples, args.delay, args.workload_iters, args.min_count)
                    case["load_mode"] = mode
                    result["cases"].append(case)
                finally:
                    stop_workers(stop_event, workers)
    finally:
        os.sched_setaffinity(0, original_affinity)

    candidates = [
        {
            "load_mode": case["load_mode"],
            "core": case["core"],
            "observed_state_mean_range_ns": case["observed_state_mean_range_ns"],
            "max_null_range_ns": case["max_null_range_ns"],
            "observed_over_max_null": case["observed_over_max_null"],
        }
        for case in result["cases"]
        if case["oracle_candidate"]
    ]
    result["global_summary"] = {
        "case_count": len(result["cases"]),
        "oracle_candidate_count": len(candidates),
        "oracle_candidates": candidates,
        "cases_with_two_or_more_states": sum(
            1 for case in result["cases"] if case["state_count_with_min_count"] >= 2
        ),
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
