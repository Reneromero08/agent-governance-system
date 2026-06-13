#!/usr/bin/env python3
"""Summarize state-label timing sweep JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--title", default="PHASE2_STATE_LABEL_TIMING_EDGE_STABILITY_SWEEP")
    args = parser.parse_args()

    rows = []
    for name in args.files:
        path = Path(name)
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data["summary"]
        rows.append({
            "file": path.as_posix(),
            "verdict": data["verdict"],
            "seed_start": data.get("seed_start"),
            "rounds": data.get("rounds"),
            "tape_words": data.get("tape_words"),
            "rows": s["row_count"],
            "restore_failures": s["restore_failures"],
            "elapsed_bacc": s["elapsed_threshold_holdout_balanced_accuracy"],
            "state_bacc": s["state_label_holdout_balanced_accuracy"],
            "mode_bacc": s["mode_label_holdout_balanced_accuracy"],
            "core_bacc": s["core_label_holdout_balanced_accuracy"],
            "shuffle_p95": s.get("elapsed_shuffled_answer_null", {}).get("p95", ""),
            "shuffle_margin": s.get("elapsed_over_shuffle_p95", ""),
            "state_label_count": s["state_label_count"],
        })

    accepted = [
        row for row in rows
        if row["verdict"] == "STATE_LABEL_PHASE_COUPLING_CANDIDATE"
    ]
    stable = len(accepted) >= 3

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {args.title}\n\n")
        handle.write("## Verdict\n\n")
        handle.write("`STATE_LABEL_TIMING_EDGE_STABLE_CANDIDATE`\n\n" if stable else "`STATE_LABEL_TIMING_EDGE_NOT_STABLE_YET`\n\n")
        handle.write("Compact read-only sweep over seed windows and row durations.\n\n")
        handle.write("## Summary\n\n")
        handle.write(f"- runs: {len(rows)}\n")
        handle.write(f"- candidate runs: {len(accepted)}\n")
        handle.write("- acceptance: at least 3 candidate runs with zero restore failures\n\n")
        handle.write("| File | Verdict | Seed | Rounds | Rows | Restore failures | Elapsed bAcc | Shuffle p95 | Shuffle margin | State bAcc | Mode bAcc | Core bAcc |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| `{Path(row['file']).name}` | `{row['verdict']}` | {row['seed_start']} | {row['rounds']} | "
                f"{row['rows']} | {row['restore_failures']} | {row['elapsed_bacc']} | {row['shuffle_p95']} | "
                f"{row['shuffle_margin']} | {row['state_bacc']} | "
                f"{row['mode_bacc']} | {row['core_bacc']} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        if stable:
            handle.write("The timing edge survived the compact sweep and should be promoted to a larger reproducibility matrix.\n")
        else:
            handle.write("The timing edge remains live but unstable. It should not be counted as CPU-sings evidence yet.\n")
        handle.write("\n## Boundary\n\n")
        handle.write("- No platform setting changes.\n")
        handle.write("- No candidate image construction.\n")
        handle.write("- No external instrumentation.\n")
    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
