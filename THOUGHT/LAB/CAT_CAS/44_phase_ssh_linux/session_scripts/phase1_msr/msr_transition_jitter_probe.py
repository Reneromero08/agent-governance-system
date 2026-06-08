#!/usr/bin/env python3
"""
Read-only transition/jitter probe for Exp44.

No MSR writes are performed. The sampler pins itself to one core at a time,
optionally starts deterministic busy workers, reads PSTATE_STATUS and
COFVID_STATUS at a fixed cadence, and reports VID transition counts plus timing
delta summaries.
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
    fid = value & 0x3F
    did = (value >> 6) & 0x07
    vid = (value >> 9) & 0x7F
    return {
        "raw": f"0x{value:016x}",
        "fid": fid,
        "did": did,
        "vid": vid,
        "freq_mhz": round(100.0 * (fid + 0x10) / (2 ** did), 3),
        "voltage_v": round(1.55 - vid * 0.0125, 6),
    }


def decode_cofvid(value):
    return {
        "raw": f"0x{value:016x}",
        "fid": value & 0x3F,
        "did": (value >> 6) & 0x07,
        "vid": (value >> 9) & 0x7F,
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


def busy_worker(core, stop_event):
    os.sched_setaffinity(0, {core})
    value = 0x13579BDF
    while not stop_event.is_set():
        value = ((value << 7) ^ (value >> 3) ^ 0x9E3779B9) & 0xFFFFFFFF
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


def sample_case(core, samples, delay, transition_sample_limit):
    os.sched_setaffinity(0, {core})
    p4 = decode_pstate(read_msr(core, PSTATE_BASE + 4))
    rows = []
    last_timestamp = time.perf_counter_ns()
    for index in range(samples):
        pstate_status = read_msr(core, PSTATE_STATUS)
        cofvid_raw = read_msr(core, COFVID_STATUS)
        now = time.perf_counter_ns()
        cofvid = decode_cofvid(cofvid_raw)
        rows.append({
            "index": index,
            "delta_ns": now - last_timestamp,
            "pstate_status": pstate_status,
            "cofvid_raw": cofvid_raw,
            "vid": cofvid["vid"],
            "fid": cofvid["fid"],
            "did": cofvid["did"],
        })
        last_timestamp = now
        if delay:
            time.sleep(delay)

    vid_transitions = []
    pstate_transitions = []
    transition_deltas = []
    steady_deltas = []
    for prev, current in zip(rows, rows[1:]):
        vid_changed = current["vid"] != prev["vid"]
        pstate_changed = current["pstate_status"] != prev["pstate_status"]
        if vid_changed:
            transition_deltas.append(current["delta_ns"])
            if len(vid_transitions) < transition_sample_limit:
                vid_transitions.append({
                    "index": current["index"],
                    "from_vid": prev["vid"],
                    "to_vid": current["vid"],
                    "from_raw": f"0x{prev['cofvid_raw']:016x}",
                    "to_raw": f"0x{current['cofvid_raw']:016x}",
                    "from_pstate_status": f"0x{prev['pstate_status']:016x}",
                    "to_pstate_status": f"0x{current['pstate_status']:016x}",
                    "delta_ns": current["delta_ns"],
                })
        else:
            steady_deltas.append(current["delta_ns"])
        if pstate_changed and len(pstate_transitions) < transition_sample_limit:
            pstate_transitions.append({
                "index": current["index"],
                "from_pstate_status": f"0x{prev['pstate_status']:016x}",
                "to_pstate_status": f"0x{current['pstate_status']:016x}",
                "from_vid": prev["vid"],
                "to_vid": current["vid"],
                "delta_ns": current["delta_ns"],
            })

    vids = [row["vid"] for row in rows]
    fid_did_vid_states = sorted({
        (row["fid"], row["did"], row["vid"]) for row in rows
    })
    return {
        "core": core,
        "samples": samples,
        "p4_msrc001_0068": p4,
        "cofvid_vid_unique": sorted(set(vids)),
        "cofvid_state_unique": [
            {"fid": fid, "did": did, "vid": vid} for fid, did, vid in fid_did_vid_states
        ],
        "cofvid_raw_unique": sorted({f"0x{row['cofvid_raw']:016x}" for row in rows}),
        "pstate_status_unique": sorted({f"0x{row['pstate_status']:016x}" for row in rows}),
        "vid_transition_count": sum(
            1 for prev, current in zip(rows, rows[1:]) if current["vid"] != prev["vid"]
        ),
        "pstate_transition_count": sum(
            1 for prev, current in zip(rows, rows[1:])
            if current["pstate_status"] != prev["pstate_status"]
        ),
        "sample_delta_ns": summarize([row["delta_ns"] for row in rows[1:]]),
        "vid_transition_delta_ns": summarize(transition_deltas),
        "vid_steady_delta_ns": summarize(steady_deltas),
        "vid_transition_samples": vid_transitions,
        "pstate_transition_samples": pstate_transitions,
    }


def compact_case(case):
    transition_mean = case["vid_transition_delta_ns"]["mean"]
    steady_mean = case["vid_steady_delta_ns"]["mean"]
    if transition_mean is not None and steady_mean is not None:
        transition_minus_steady_ns = round(transition_mean - steady_mean, 3)
    else:
        transition_minus_steady_ns = None
    return {
        "core": case["core"],
        "load_mode": case["load_mode"],
        "cofvid_vid_unique": case["cofvid_vid_unique"],
        "pstate_status_unique": case["pstate_status_unique"],
        "vid_transition_count": case["vid_transition_count"],
        "pstate_transition_count": case["pstate_transition_count"],
        "sample_delta_ns_mean": case["sample_delta_ns"]["mean"],
        "sample_delta_ns_pstdev": case["sample_delta_ns"]["pstdev"],
        "vid_transition_delta_ns_mean": transition_mean,
        "vid_steady_delta_ns_mean": steady_mean,
        "transition_minus_steady_ns": transition_minus_steady_ns,
        "vid_transition_samples": case["vid_transition_samples"],
    }


def build_global_summary(cases):
    vid_transition_cases = [case for case in cases if case["vid_transition_count"]]
    pstate_transition_cases = [case for case in cases if case["pstate_transition_count"]]
    return {
        "case_count": len(cases),
        "total_vid_transitions": sum(case["vid_transition_count"] for case in cases),
        "total_pstate_transitions": sum(case["pstate_transition_count"] for case in cases),
        "vid_transition_case_count": len(vid_transition_cases),
        "pstate_transition_case_count": len(pstate_transition_cases),
        "vid_transition_cases": [
            {
                "load_mode": case["load_mode"],
                "core": case["core"],
                "count": case["vid_transition_count"],
                "vids": case["cofvid_vid_unique"],
            }
            for case in vid_transition_cases
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only Exp44 COFVID transition/jitter probe")
    parser.add_argument("--cores", default="0-5")
    parser.add_argument("--modes", default="baseline,self_load,neighbor_load,all_load")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--delay", type=float, default=0.002)
    parser.add_argument("--transition-sample-limit", type=int, default=16)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    cores = parse_cores(args.cores)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    result = {
        "mode": "read_only_transition_jitter_probe",
        "writes": False,
        "cores": cores,
        "modes": modes,
        "samples_per_case": args.samples,
        "delay_seconds": args.delay,
        "acceptance": [
            "No MSR writes are performed.",
            "VID transition counts are computed per core and mode.",
            "PSTATE_STATUS transitions are computed per core and mode.",
            "Timing deltas are summarized for VID-transition and steady samples.",
        ],
        "cases": [],
    }

    original_affinity = os.sched_getaffinity(0)
    try:
        for mode in modes:
            for core in cores:
                stop_event, workers = start_workers(mode, core, cores)
                try:
                    case = sample_case(core, args.samples, args.delay, args.transition_sample_limit)
                    case["load_mode"] = mode
                    result["cases"].append(case)
                finally:
                    stop_workers(stop_event, workers)
    finally:
        os.sched_setaffinity(0, original_affinity)

    result["global_summary"] = build_global_summary(result["cases"])
    if args.summary_only:
        result["cases"] = [compact_case(case) for case in result["cases"]]

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
