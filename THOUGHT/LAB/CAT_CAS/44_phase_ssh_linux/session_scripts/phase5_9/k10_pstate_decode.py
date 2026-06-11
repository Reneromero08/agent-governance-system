#!/usr/bin/env python3
"""Decode AMD K10 P-state MSRs from rdmsr output.

Expected input CSV:
msr,value_hex
0xC0010064,8000019e40000c14
"""

import argparse
import csv


def decode_pstate(msr, value):
    fid = value & 0x3f
    did = (value >> 6) & 0x7
    vid = (value >> 9) & 0x7f
    enabled = (value >> 63) & 0x1
    freq_mhz = 100.0 * (fid + 0x10) / (1 << did)
    vcore = 1.55 - 0.0125 * vid
    return {
        "msr": msr,
        "raw": f"{value:016x}",
        "enabled": enabled,
        "fid": fid,
        "did": did,
        "vid": vid,
        "freq_mhz": f"{freq_mhz:.1f}",
        "decoded_vcore": f"{vcore:.4f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.input, newline="") as f:
        for row in csv.DictReader(f):
            value = row.get("value_hex", "").strip()
            if not value or value.startswith("ERROR"):
                continue
            rows.append(decode_pstate(row["msr"], int(value, 16)))

    with open(args.output, "w", newline="") as f:
        fields = ["msr", "raw", "enabled", "fid", "did", "vid", "freq_mhz", "decoded_vcore"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
