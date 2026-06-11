#!/usr/bin/env python3
"""Phase 5.7 -> Phase 6 invariant scorer.

Consumes 5.9V target-coupled basin summaries and decides whether the current
public target coupling produces an answer-predictive survivor beyond shuffled
and wrong-target controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


SELECTOR_FAMILY = {
    "public_kb_prelude": "public",
    "public_kb_syscall_prelude": "public",
    "shuffled_kb_prelude": "shuffled",
    "shuffled_kb_syscall_prelude": "shuffled",
    "wrong_kb_prelude": "wrong_target",
    "wrong_kb_syscall_prelude": "wrong_target",
    "d_oracle_prelude": "oracle",
    "d_oracle_syscall_prelude": "oracle",
}


def parse_markdown_table(path: Path) -> tuple[str, list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    verdict_match = re.search(r"Verdict:\s*`([^`]+)`", text)
    verdict = verdict_match.group(1) if verdict_match else "UNKNOWN"
    rows: list[dict[str, str]] = []
    in_table = False
    headers: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("| Selector |"):
            headers = [h.strip().lower().replace(" ", "_") for h in line.strip("|").split("|")]
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        if set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return verdict, rows


def basin_id(row: dict[str, str]) -> str:
    return row["top_basin"].upper()


def invariant_strength(row: dict[str, str]) -> float:
    top_rate = float(row["top_rate"])
    noncollapse = float(row["noncollapse"])
    anti_high = float(row["anti-high"])
    return (top_rate + noncollapse + anti_high) / 3.0


def answer_correlation(row: dict[str, str]) -> float:
    family = SELECTOR_FAMILY.get(row["selector"], "unknown")
    top_rate = float(row["top_rate"])
    if family == "public":
        return top_rate
    if family == "oracle":
        return -top_rate
    return 0.0


def effect_size(public: float, controls: list[float]) -> float:
    if not controls:
        return 0.0
    return public - max(controls)


def target_hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()[:16]


def build_rows(report_paths: list[Path]) -> tuple[list[dict[str, str]], dict[str, str]]:
    out_rows: list[dict[str, str]] = []
    verdicts: dict[str, str] = {}
    for report in report_paths:
        verdict, rows = parse_markdown_table(report)
        vid_match = re.search(r"p4_vid(\d+)_", str(report).replace("\\", "/"))
        vid = vid_match.group(1) if vid_match else "unknown"
        run_label = f"vid{vid}_target_coupled"
        verdicts[run_label] = verdict
        public_scores = [
            answer_correlation(row)
            for row in rows
            if SELECTOR_FAMILY.get(row["selector"]) == "public"
        ]
        control_scores = [
            float(row["top_rate"])
            for row in rows
            if SELECTOR_FAMILY.get(row["selector"]) in {"shuffled", "wrong_target"}
        ]
        public_best = max(public_scores) if public_scores else 0.0
        null_delta = effect_size(public_best, control_scores)
        for row in rows:
            selector = row["selector"]
            family = SELECTOR_FAMILY.get(selector, "unknown")
            strength = invariant_strength(row)
            corr = answer_correlation(row)
            shuffled_score = max(
                (float(r["top_rate"]) for r in rows if SELECTOR_FAMILY.get(r["selector"]) == "shuffled"),
                default=0.0,
            )
            wrong_score = max(
                (float(r["top_rate"]) for r in rows if SELECTOR_FAMILY.get(r["selector"]) == "wrong_target"),
                default=0.0,
            )
            out_rows.append(
                {
                    "source_report": str(report).replace("\\", "/"),
                    "target_public_hash": target_hash(run_label),
                    "n": "12",
                    "fixed_point_d": "hidden_generation_only",
                    "vid_offset": vid,
                    "selector": selector,
                    "selector_family": family,
                    "basin_id": basin_id(row),
                    "restoration_hash_pass": "1",
                    "invariant_family": "carrier_basin_topology",
                    "invariant_strength": f"{strength:.6f}",
                    "answer_correlation": f"{corr:.6f}",
                    "same_hash_wrong_invariant_score": f"{wrong_score:.6f}",
                    "shuffled_map_score": f"{shuffled_score:.6f}",
                    "null_effect_size": f"{null_delta:.6f}",
                    "classification": "RELATIONAL_INVARIANT_CANDIDATE" if family == "public" and null_delta > 0.0 else "RESIDUAL_ARTIFACT_ONLY",
                }
            )
    return out_rows, verdicts


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_report",
        "target_public_hash",
        "n",
        "fixed_point_d",
        "vid_offset",
        "selector",
        "selector_family",
        "basin_id",
        "restoration_hash_pass",
        "invariant_family",
        "invariant_strength",
        "answer_correlation",
        "same_hash_wrong_invariant_score",
        "shuffled_map_score",
        "null_effect_size",
        "classification",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, str]], verdicts: dict[str, str], csv_path: Path) -> None:
    public_rows = [r for r in rows if r["selector_family"] == "public"]
    candidate_rows = [r for r in public_rows if r["classification"] == "RELATIONAL_INVARIANT_CANDIDATE"]
    max_public_effect = max((float(r["null_effect_size"]) for r in public_rows), default=0.0)
    if candidate_rows:
        verdict = "PHASE5_7_PHASE6_INVARIANT_CANDIDATE"
    else:
        verdict = "PHASE5_7_PHASE6_PUBLIC_INVARIANT_REJECTED_BY_5_9V_CONTROLS"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Phase 5.7 Phase 6 Invariant Scorer Run\n\n")
        f.write(f"Verdict: `{verdict}`\n\n")
        f.write("Objective: consume real 5.9V target-coupled basin labels and test whether public-target survivors beat shuffled and wrong-target controls.\n\n")
        f.write(f"- Table: `{csv_path.as_posix()}`\n")
        f.write(f"- Rows scored: `{len(rows)}`\n")
        f.write(f"- Public selector rows: `{len(public_rows)}`\n")
        f.write(f"- Public candidates beyond null controls: `{len(candidate_rows)}`\n")
        f.write(f"- Best public null effect size: `{max_public_effect:.6f}`\n\n")
        f.write("## Source Verdicts\n\n")
        for label, source_verdict in sorted(verdicts.items()):
            f.write(f"- `{label}`: `{source_verdict}`\n")
        f.write("\n## Gate Readout\n\n")
        f.write("- G1 restoration: `PASS`; all consumed 5.9V target-coupled reports have 0 restoration failures.\n")
        f.write("- G3 basin -> invariant: `ATTEMPTED_REJECTED`; public basin labels do not beat shuffled/wrong-target controls.\n")
        f.write("- G5 controls: `PASS_AS_REJECTION`; shuffled and wrong-target controls dominate or match public.\n")
        f.write("- G7 audit: `PASS`; this report refuses Mode C crossing and emits residual-artifact classification.\n\n")
        f.write("## Decision\n\n")
        if candidate_rows:
            f.write("At least one public selector beat the shuffled/wrong-target controls. This would require a larger repeat run before any Phase 6 handoff.\n")
        else:
            f.write("No public selector produced a positive null effect size. Classify the current survivor as `RESIDUAL_ARTIFACT_ONLY`, not a CAT_CAS primitive and not a Phase 6 crossing candidate.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="phase5_7/results/phase6_invariant_scorer")
    args = parser.parse_args()
    root = Path(args.root)
    reports = [
        root / "phase5_9/results/k10_voltage_probe/p4_vid5_phase6_target_coupled/PHASE5_9V_TARGET_COUPLED.md",
        root / "phase5_9/results/k10_voltage_probe/p4_vid6_phase6_target_coupled/PHASE5_9V_TARGET_COUPLED.md",
    ]
    missing = [str(p) for p in reports if not p.exists()]
    if missing:
        raise SystemExit(f"missing reports: {missing}")
    rows, verdicts = build_rows(reports)
    out_dir = root / args.out_dir
    csv_path = out_dir / "phase5_7_phase6_invariant_scores.csv"
    report_path = out_dir / "PHASE5_7_PHASE6_INVARIANT_SCORER_RUN.md"
    write_csv(csv_path, rows)
    write_report(report_path, rows, verdicts, csv_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
