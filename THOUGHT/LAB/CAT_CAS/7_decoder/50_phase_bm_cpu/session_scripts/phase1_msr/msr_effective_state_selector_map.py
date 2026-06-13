#!/usr/bin/env python3
"""Read-only effective-state selector map for Exp50.

This probe samples PSTATE_STATUS, COFVID_STATUS, and timing while applying
ordinary CPU load selectors. It does not change MSR definitions or platform
settings.
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
MPERF = 0xC00000E7
APERF = 0xC00000E8


def read_msr(core: int, msr: int) -> int | None:
    try:
        with open(f"/dev/cpu/{core}/msr", "rb", buffering=0) as handle:
            os.lseek(handle.fileno(), msr, os.SEEK_SET)
            return struct.unpack("<Q", handle.read(8))[0]
    except OSError:
        return None


def parse_cores(text: str) -> list[int]:
    cores: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    return cores


def decode_pstate_def(value: int | None) -> dict[str, object] | None:
    if value is None:
        return None
    fid = value & 0x3F
    did = (value >> 6) & 0x07
    vid = (value >> 9) & 0x7F
    return {
        "raw": f"0x{value:016x}",
        "fid": fid,
        "did": did,
        "vid": vid,
        "freq_mhz": round(100.0 * (fid + 0x10) / (2 ** did), 3),
        "decoded_label_v": round(1.55 - vid * 0.0125, 6),
    }


def decode_status(value: int | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "raw": f"0x{value:016x}",
        "fid": value & 0x3F,
        "did": (value >> 6) & 0x07,
        "vid": (value >> 9) & 0x7F,
        "pstate": value & 0x07,
    }


def summarize(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "pstdev": None, "cv": None}
    m = mean(values)
    sd = pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(m, 3),
        "pstdev": round(sd, 3),
        "cv": round(sd / m, 9) if m else None,
    }


def busy_worker(core: int, stop_event: mp.Event) -> int:
    os.sched_setaffinity(0, {core})
    value = 0x9E3779B9
    while not stop_event.is_set():
        value = ((value * 1664525) + 1013904223) & 0xFFFFFFFF
        value ^= value >> 11
        value = ((value << 7) ^ value) & 0xFFFFFFFF
    return value


def start_workers(mode: str, sample_core: int, cores: list[int]) -> tuple[mp.Event, list[mp.Process]]:
    stop_event = mp.Event()
    if mode == "baseline":
        worker_cores: list[int] = []
    elif mode == "self_load":
        worker_cores = [sample_core]
    elif mode == "neighbor_load":
        idx = cores.index(sample_core)
        worker_cores = [cores[(idx + 1) % len(cores)]]
    elif mode == "all_load":
        worker_cores = list(cores)
    elif mode == "other_load":
        worker_cores = [core for core in cores if core != sample_core]
    else:
        raise ValueError(f"unknown mode: {mode}")

    workers = []
    for core in worker_cores:
        proc = mp.Process(target=busy_worker, args=(core, stop_event))
        proc.start()
        workers.append(proc)
    if workers:
        time.sleep(0.05)
    return stop_event, workers


def stop_workers(stop_event: mp.Event, workers: list[mp.Process]) -> None:
    stop_event.set()
    for proc in workers:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


def timed_work(iterations: int) -> tuple[int, int]:
    value = 0xC0010068
    start = time.perf_counter_ns()
    for i in range(iterations):
        value = ((value * 1103515245) + 12345 + i) & 0xFFFFFFFF
        value ^= (value >> 16)
    return time.perf_counter_ns() - start, value


def sample(core: int, samples: int, delay: float, iterations: int) -> dict[str, object]:
    os.sched_setaffinity(0, {core})
    pdefs = [decode_pstate_def(read_msr(core, PSTATE_BASE + p)) for p in range(5)]
    rows = []
    checksum = 0
    for _ in range(samples):
        before = {
            "pstate_status": read_msr(core, PSTATE_STATUS),
            "cofvid_status": read_msr(core, COFVID_STATUS),
            "mperf": read_msr(core, MPERF),
            "aperf": read_msr(core, APERF),
        }
        elapsed, value = timed_work(iterations)
        after = {
            "pstate_status": read_msr(core, PSTATE_STATUS),
            "cofvid_status": read_msr(core, COFVID_STATUS),
            "mperf": read_msr(core, MPERF),
            "aperf": read_msr(core, APERF),
        }
        checksum ^= value
        aperf_ratio = None
        if before["aperf"] is not None and before["mperf"] is not None and after["aperf"] is not None and after["mperf"] is not None:
            dmperf = after["mperf"] - before["mperf"]
            daperf = after["aperf"] - before["aperf"]
            if dmperf > 0:
                aperf_ratio = daperf / dmperf
        rows.append({
            "elapsed_ns": elapsed,
            "before_pstate": decode_status(before["pstate_status"]),
            "after_pstate": decode_status(after["pstate_status"]),
            "before_cofvid": decode_status(before["cofvid_status"]),
            "after_cofvid": decode_status(after["cofvid_status"]),
            "aperf_mperf_ratio": aperf_ratio,
        })
        if delay:
            time.sleep(delay)

    def unique_status(key: str) -> list[str]:
        values = []
        for row in rows:
            item = row[key]
            if isinstance(item, dict):
                values.append(f"fid={item['fid']:02x}/did={item['did']}/vid={item['vid']:02x}/raw={item['raw']}")
        return sorted(set(values))

    return {
        "core": core,
        "pstate_definitions": pdefs,
        "checksum": f"0x{checksum:08x}",
        "timing_ns": summarize([float(row["elapsed_ns"]) for row in rows]),
        "aperf_mperf_ratio": summarize([float(row["aperf_mperf_ratio"]) for row in rows if row["aperf_mperf_ratio"] is not None]),
        "pstate_status_unique": unique_status("before_pstate") + unique_status("after_pstate"),
        "cofvid_status_unique": unique_status("before_cofvid") + unique_status("after_cofvid"),
    }


def mode_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    timing_means = [float(case["timing_ns"]["mean"]) for case in cases if case["timing_ns"]["mean"] is not None]
    ratios = [float(case["aperf_mperf_ratio"]["mean"]) for case in cases if case["aperf_mperf_ratio"]["mean"] is not None]
    cofvid_labels = sorted(set(label for case in cases for label in case["cofvid_status_unique"]))
    pstate_labels = sorted(set(label for case in cases for label in case["pstate_status_unique"]))
    return {
        "core_count": len(cases),
        "timing_mean_ns": summarize(timing_means),
        "aperf_mperf_mean": summarize(ratios),
        "cofvid_status_labels": cofvid_labels,
        "pstate_status_labels": pstate_labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only effective-state selector map")
    parser.add_argument("--cores", default="0-5")
    parser.add_argument("--modes", default="baseline,self_load,neighbor_load,other_load,all_load")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--workload-iters", type=int, default=768)
    args = parser.parse_args()

    cores = parse_cores(args.cores)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    result: dict[str, object] = {
        "mode": "read_only_effective_state_selector_map",
        "setting_changes": False,
        "cores": cores,
        "modes": modes,
        "samples_per_case": args.samples,
        "workload_iters": args.workload_iters,
        "acceptance": [
            "No platform settings are changed.",
            "Internal state labels are recorded per row.",
            "Timing carrier claims require state-label movement or APERF/MPERF movement.",
        ],
        "cases": [],
        "mode_summaries": {},
    }

    original_affinity = os.sched_getaffinity(0)
    try:
        for mode in modes:
            mode_cases = []
            for core in cores:
                stop_event, workers = start_workers(mode, core, cores)
                try:
                    case = sample(core, args.samples, args.delay, args.workload_iters)
                    case["load_mode"] = mode
                    mode_cases.append(case)
                    result["cases"].append(case)
                finally:
                    stop_workers(stop_event, workers)
            result["mode_summaries"][mode] = mode_summary(mode_cases)
    finally:
        os.sched_setaffinity(0, original_affinity)

    all_cofvid = sorted(set(label for case in result["cases"] for label in case["cofvid_status_unique"]))
    all_pstate = sorted(set(label for case in result["cases"] for label in case["pstate_status_unique"]))
    result["global_summary"] = {
        "case_count": len(result["cases"]),
        "cofvid_label_count": len(all_cofvid),
        "pstate_label_count": len(all_pstate),
        "cofvid_labels": all_cofvid,
        "pstate_labels": all_pstate,
        "effective_state_moved": bool(len(all_cofvid) > 1 or len(all_pstate) > 1),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
