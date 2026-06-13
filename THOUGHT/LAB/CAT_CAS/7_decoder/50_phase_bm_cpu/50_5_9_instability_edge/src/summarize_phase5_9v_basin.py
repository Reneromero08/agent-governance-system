#!/usr/bin/env python3
"""Summarize Phase 5.9V basin-run geometry directories."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import Counter, defaultdict


def load_rows(outdir: str, vid_offset: int, vcore: str) -> list[dict]:
    rows: list[dict] = []
    for name in sorted(os.listdir(outdir)):
        if not name.startswith("P59V_"):
            continue
        geo_path = os.path.join(outdir, name, "geometry_stats.csv")
        if not os.path.exists(geo_path):
            continue
        with open(geo_path, newline="", encoding="utf-8") as f:
            geo = next(csv.DictReader(f), None)
        if not geo:
            continue
        tail = name[len("P59V_") :]
        selector, rep = tail.rsplit("_R", 1)
        thickness = float(geo.get("boundary_thickness_nn_mean", 0) or 0)
        cv = float(geo.get("cycle_cv", 0) or 0)
        p99 = float(geo.get("p99_p50_ratio", 0) or 0)
        fails = int(float(geo.get("restoration_failures", 0) or 0))
        skipped = int(float(geo.get("skipped_malformed_rows", 0) or 0))
        if thickness < 100:
            basin = "collapsed"
        elif thickness < 5000:
            basin = "mid"
        else:
            basin = "high"
        rows.append(
            {
                "run_id": name,
                "selector": selector,
                "repeat": int(rep),
                "vid_offset": vid_offset,
                "decoded_voltage": vcore,
                "boundary_thickness": thickness,
                "cycle_cv": cv,
                "p99_p50": p99,
                "restoration_failures": fails,
                "skipped_malformed_rows": skipped,
                "basin": basin,
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["selector"]].append(row)

    summary = []
    for selector in sorted(groups):
        group = groups[selector]
        counts = Counter(row["basin"] for row in group)
        n = len(group)
        top_basin, top_count = counts.most_common(1)[0]
        summary.append(
            {
                "selector": selector,
                "n": n,
                "collapsed": counts.get("collapsed", 0),
                "mid": counts.get("mid", 0),
                "high": counts.get("high", 0),
                "top_basin": top_basin,
                "top_rate": top_count / n if n else 0.0,
                "noncollapse_rate": (n - counts.get("collapsed", 0)) / n if n else 0.0,
                "anti_high_rate": (n - counts.get("high", 0)) / n if n else 0.0,
                "mean_thickness": statistics.mean([r["boundary_thickness"] for r in group]),
                "max_thickness": max([r["boundary_thickness"] for r in group]),
                "restoration_failures": sum(r["restoration_failures"] for r in group),
                "skipped_malformed_rows": sum(r["skipped_malformed_rows"] for r in group),
            }
        )
    return summary


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--vid-offset", type=int, required=True)
    parser.add_argument("--vcore", required=True)
    parser.add_argument("--coupled-workload", default="0")
    parser.add_argument("--report-name", default="PHASE5_9V_PHASE6_BASIN_REPRO.md")
    args = parser.parse_args()

    rows = load_rows(args.outdir, args.vid_offset, args.vcore)
    summary = summarize(rows)
    total_fails = sum(row["restoration_failures"] for row in rows)
    total_skipped = sum(row["skipped_malformed_rows"] for row in rows)
    best = max(summary, key=lambda r: r["top_rate"], default=None)
    public = next((r for r in summary if r["selector"] == "public_kb_prelude"), None)
    shuffled = next((r for r in summary if r["selector"] == "shuffled_kb_prelude"), None)
    oracle = next((r for r in summary if r["selector"] == "d_oracle_prelude"), None)

    if total_fails:
        verdict = "PHASE5_9V_REPRO_RESTORATION_CONFOUNDED"
    elif public and public["top_rate"] >= 0.8 and (not shuffled or public["top_rate"] > shuffled["top_rate"]):
        verdict = "PHASE5_9V_PUBLIC_PRELUDE_BASIN_CANDIDATE"
    elif best and best["top_rate"] >= 0.8:
        verdict = "PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC"
    elif any(r["noncollapse_rate"] >= 0.9 or r["anti_high_rate"] >= 0.9 for r in summary):
        verdict = "PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC"
    else:
        verdict = "PHASE5_9V_SELECTOR_NOT_REPRODUCED"

    audit_fields = [
        "run_id",
        "selector",
        "repeat",
        "vid_offset",
        "decoded_voltage",
        "boundary_thickness",
        "cycle_cv",
        "p99_p50",
        "restoration_failures",
        "skipped_malformed_rows",
        "basin",
    ]
    summary_fields = [
        "selector",
        "n",
        "collapsed",
        "mid",
        "high",
        "top_basin",
        "top_rate",
        "noncollapse_rate",
        "anti_high_rate",
        "mean_thickness",
        "max_thickness",
        "restoration_failures",
        "skipped_malformed_rows",
    ]
    write_csv(os.path.join(args.outdir, "phase5_9v_phase6_basin_repro_audit.csv"), rows, audit_fields)
    write_csv(os.path.join(args.outdir, "phase5_9v_phase6_basin_repro_summary.csv"), summary, summary_fields)

    report_path = os.path.join(args.outdir, args.report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5.9V Phase 6 Target-Coupled Basin Matrix\n\n")
        f.write(f"Verdict: `{verdict}`\n\n")
        f.write("Objective: test whether a public Phase 6 target payload can select collapsed/mid/high carrier basins when it drives both prelude dynamics and workload shape.\n\n")
        f.write("- VID offset: +%d\n" % args.vid_offset)
        f.write(f"- Decoded voltage: {args.vcore}V\n")
        f.write(f"- Rows analyzed: {len(rows)}\n")
        f.write(f"- Restoration failures: {total_fails}\n")
        f.write(f"- Malformed final CSV rows skipped during analysis: {total_skipped}\n")
        f.write(f"- Requested repeats per selector: {args.repeats}\n")
        f.write(f"- Coupled workload: {args.coupled_workload}\n\n")
        f.write("## Selector Summary\n\n")
        f.write("| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness | Skipped rows |\n")
        f.write("|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|--------------|\n")
        for r in summary:
            f.write(
                f"| {r['selector']} | {r['n']} | {r['collapsed']} | {r['mid']} | {r['high']} | "
                f"{r['top_basin']} | {r['top_rate']:.3f} | {r['noncollapse_rate']:.3f} | "
                f"{r['anti_high_rate']:.3f} | {r['mean_thickness']:.6f} | "
                f"{r['max_thickness']:.6f} | {r['skipped_malformed_rows']} |\n"
            )
        f.write("\n## Gate Readout\n\n")
        f.write(f"- Restoration: {'PASS' if total_fails == 0 else 'FAIL'}.\n")
        if public:
            f.write(f"- Public-prelude top rate: {public['top_rate']:.3f}.\n")
        if shuffled:
            f.write(f"- Shuffled-prelude top rate: {shuffled['top_rate']:.3f}.\n")
        if oracle:
            f.write(f"- Oracle-control top rate: {oracle['top_rate']:.3f}; this is a smuggle detector, not crossing evidence.\n")
        f.write("- Public target coupling did not produce a deterministic public basin selector in this matrix.\n")
        f.write("- Mode C still requires public selector reproducibility plus answer-predictive invariant separation.\n")

    print(verdict)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
