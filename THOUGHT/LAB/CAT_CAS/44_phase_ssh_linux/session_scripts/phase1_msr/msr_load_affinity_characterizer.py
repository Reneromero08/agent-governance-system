#!/usr/bin/env python3
"""
Read-only load/affinity characterization for Exp44.

No MSR writes are performed. The script pins the sampler and optional busy
workers with sched_setaffinity, reads P-state/COFVID MSRs, and records timing
jitter so runtime VID behavior can be compared across scheduler/load states.
"""

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


def read_msr(core, msr):
    with open(f"/dev/cpu/{core}/msr", "rb", buffering=0) as handle:
        os.lseek(handle.fileno(), msr, os.SEEK_SET)
        return struct.unpack("<Q", handle.read(8))[0]


def decode_pstate(value):
    lo = value & 0xFFFFFFFF
    fid = lo & 0x3F
    did = (lo >> 6) & 0x07
    vid = (lo >> 9) & 0x7F
    return {
        "raw": f"0x{value:016x}",
        "fid": fid,
        "did": did,
        "vid": vid,
        "freq_mhz": round(100.0 * (fid + 0x10) / (2 ** did), 3),
        "voltage_v": round(1.55 - vid * 0.0125, 6),
    }


def timing_jitter(iterations):
    last = time.perf_counter_ns()
    deltas = []
    for _ in range(iterations):
        now = time.perf_counter_ns()
        deltas.append(now - last)
        last = now
    return {
        "iterations": iterations,
        "delta_ns_min": min(deltas),
        "delta_ns_max": max(deltas),
        "delta_ns_mean": round(mean(deltas), 3),
        "delta_ns_pstdev": round(pstdev(deltas), 3) if len(deltas) > 1 else 0.0,
    }


def busy_worker(core, stop_event):
    os.sched_setaffinity(0, {core})
    x = 0x12345678
    while not stop_event.is_set():
        x = ((x * 1103515245) + 12345) & 0xFFFFFFFF
    return x


def start_workers(mode, sample_core, all_cores):
    stop_event = mp.Event()
    worker_cores = []
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


def sample(core, samples, delay, jitter_iters):
    os.sched_setaffinity(0, {core})
    p4 = decode_pstate(read_msr(core, PSTATE_BASE + 4))
    pstate_status = []
    cofvid = []
    jitter = []
    for _ in range(samples):
        pstate_status.append(read_msr(core, PSTATE_STATUS))
        cofvid.append(read_msr(core, COFVID_STATUS))
        jitter.append(timing_jitter(jitter_iters))
        if delay:
            time.sleep(delay)

    vids = [(value >> 9) & 0x7F for value in cofvid]
    dids = [(value >> 6) & 0x07 for value in cofvid]
    fids = [value & 0x3F for value in cofvid]
    jitter_means = [item["delta_ns_mean"] for item in jitter]

    return {
        "core": core,
        "p4_msrc001_0068": p4,
        "cofvid_unique_raw": sorted(f"0x{value:016x}" for value in set(cofvid)),
        "pstate_status_unique_raw": sorted(f"0x{value:016x}" for value in set(pstate_status)),
        "cofvid_vid_min": min(vids),
        "cofvid_vid_max": max(vids),
        "cofvid_vid_mean": round(mean(vids), 6),
        "cofvid_vid_pstdev": round(pstdev(vids), 6) if len(vids) > 1 else 0.0,
        "cofvid_did_unique": sorted(set(dids)),
        "cofvid_fid_unique": sorted(set(fids)),
        "jitter_delta_ns_mean_min": round(min(jitter_means), 3),
        "jitter_delta_ns_mean_max": round(max(jitter_means), 3),
        "jitter_delta_ns_mean_mean": round(mean(jitter_means), 3),
    }


def parse_cores(text):
    cores = []
    for part in text.split(","):
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    return cores


def main():
    parser = argparse.ArgumentParser(description="Read-only Exp44 load/affinity characterizer")
    parser.add_argument("--cores", default="0-5")
    parser.add_argument("--modes", default="baseline,self_load,neighbor_load,all_load")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--jitter-iters", type=int, default=128)
    args = parser.parse_args()

    cores = parse_cores(args.cores)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    result = {
        "mode": "read_only_load_affinity_characterization",
        "writes": False,
        "cores": cores,
        "modes": modes,
        "samples_per_case": args.samples,
        "acceptance": [
            "No MSR writes are performed.",
            "COFVID VID distribution is captured for each core/mode.",
            "P4 MSRC001_0068 is captured for each core/mode.",
            "Timing jitter is captured for each core/mode.",
        ],
        "cases": [],
    }

    original_affinity = os.sched_getaffinity(0)
    try:
        for mode in modes:
            for core in cores:
                stop_event, workers = start_workers(mode, core, cores)
                try:
                    case = sample(core, args.samples, args.delay, args.jitter_iters)
                    case["load_mode"] = mode
                    result["cases"].append(case)
                finally:
                    stop_workers(stop_event, workers)
    finally:
        os.sched_setaffinity(0, original_affinity)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
