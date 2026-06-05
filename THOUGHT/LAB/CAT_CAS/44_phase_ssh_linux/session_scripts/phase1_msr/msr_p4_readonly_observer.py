#!/usr/bin/env python3
"""
Read-only P-state/MSR observer for Exp44.

This script performs no MSR writes. It samples MSRC001_0064..0068 and
COFVID_STATUS so the firmware-derived P4 path can be compared with runtime
hardware state without changing voltage or P-state definitions.
"""

import argparse
import json
import os
import struct
import time
from statistics import mean, pstdev


PSTATE_BASE = 0xC0010064
COFVID_STATUS = 0xC0010071
PSTATE_STATUS = 0xC0010063


def read_msr(core, msr):
    path = f"/dev/cpu/{core}/msr"
    with open(path, "rb", buffering=0) as handle:
        os.lseek(handle.fileno(), msr, os.SEEK_SET)
        return struct.unpack("<Q", handle.read(8))[0]


def decode_pstate_def(value):
    lo = value & 0xFFFFFFFF
    fid = lo & 0x3F
    did = (lo >> 6) & 0x07
    vid = (lo >> 9) & 0x7F
    enabled = (value >> 63) & 1
    freq_mhz = 100.0 * (fid + 0x10) / (2 ** did)
    voltage = 1.55 - (vid * 0.0125)
    return {
        "raw": f"0x{value:016x}",
        "enabled": enabled,
        "fid": fid,
        "did": did,
        "vid": vid,
        "freq_mhz": round(freq_mhz, 3),
        "voltage_v": round(voltage, 6),
    }


def decode_cofvid(value):
    fid = value & 0x3F
    did = (value >> 6) & 0x07
    vid = (value >> 9) & 0x7F
    cur_pstate = value & 0x07
    freq_mhz = 100.0 * (fid + 0x10) / (2 ** did)
    voltage = 1.55 - (vid * 0.0125)
    return {
        "raw": f"0x{value:016x}",
        "cur_pstate_low_bits": cur_pstate,
        "fid": fid,
        "did": did,
        "vid": vid,
        "freq_mhz": round(freq_mhz, 3),
        "voltage_v": round(voltage, 6),
    }


def sample_core(core, samples, delay):
    pstate_defs = {}
    for pstate in range(5):
        msr = PSTATE_BASE + pstate
        pstate_defs[f"P{pstate}"] = decode_pstate_def(read_msr(core, msr))

    cofvid_samples = []
    pstate_status_samples = []
    for _ in range(samples):
        cofvid_samples.append(read_msr(core, COFVID_STATUS))
        try:
            pstate_status_samples.append(read_msr(core, PSTATE_STATUS))
        except OSError:
            pstate_status_samples.append(None)
        if delay:
            time.sleep(delay)

    vids = [(value >> 9) & 0x7F for value in cofvid_samples]
    dids = [(value >> 6) & 0x07 for value in cofvid_samples]
    fids = [value & 0x3F for value in cofvid_samples]

    return {
        "core": core,
        "pstate_definitions": pstate_defs,
        "cofvid_first": decode_cofvid(cofvid_samples[0]),
        "cofvid_last": decode_cofvid(cofvid_samples[-1]),
        "cofvid_unique_raw": sorted(f"0x{value:016x}" for value in set(cofvid_samples)),
        "pstate_status_unique_raw": sorted(
            f"0x{value:016x}" for value in set(pstate_status_samples) if value is not None
        ),
        "sample_stats": {
            "samples": samples,
            "vid_min": min(vids),
            "vid_max": max(vids),
            "vid_mean": round(mean(vids), 6),
            "vid_pstdev": round(pstdev(vids), 6) if len(vids) > 1 else 0.0,
            "did_unique": sorted(set(dids)),
            "fid_unique": sorted(set(fids)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only Exp44 P4/MSR observer")
    parser.add_argument("--cores", default="0-5", help="Core list/range, e.g. 0-5 or 0,4")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    cores = []
    for part in args.cores.split(","):
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))

    result = {
        "mode": "read_only",
        "writes": False,
        "msr_focus": {
            "p4_definition": "MSRC001_0068",
            "cofvid_status": "MSRC001_0071",
            "pstate_status": "MSRC001_0063",
        },
        "acceptance": [
            "Script completes with no writes.",
            "P4 MSRC001_0068 raw/VID is captured for every requested core.",
            "COFVID_STATUS VID distribution is reported for every requested core.",
            "Any runtime clamp remains visible as COFVID VID not dropping below observed floor.",
        ],
        "cores": [],
    }

    for core in cores:
        try:
            result["cores"].append(sample_core(core, args.samples, args.delay))
        except Exception as exc:
            result["cores"].append({"core": core, "error": str(exc)})

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("=== Exp44 read-only P4/MSR observer ===")
    print("mode=read_only writes=False")
    for core in result["cores"]:
        if "error" in core:
            print(f"core {core['core']}: ERROR {core['error']}")
            continue
        p4 = core["pstate_definitions"]["P4"]
        stats = core["sample_stats"]
        last = core["cofvid_last"]
        print(
            f"core {core['core']}: P4={p4['raw']} P4_VID=0x{p4['vid']:02x} "
            f"COFVID_LAST={last['raw']} COFVID_VID=0x{last['vid']:02x} "
            f"VID_RANGE=0x{stats['vid_min']:02x}-0x{stats['vid_max']:02x}"
        )


if __name__ == "__main__":
    main()
