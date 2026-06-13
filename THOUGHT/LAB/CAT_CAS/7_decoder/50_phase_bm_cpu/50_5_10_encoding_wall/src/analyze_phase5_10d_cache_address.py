#!/usr/bin/env python3
"""Analyze EXP50 Phase 5.10D cache/address topology probe CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


MODES = ("none", "compute", "same_address", "different_address", "random_address")


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for raw in fh:
            if raw.startswith("#"):
                continue
            rows.append(raw)
    if not rows:
        return []
    parsed = list(csv.DictReader(rows))
    out = []
    for row in parsed:
        try:
            row["family"] = int(row["family"])
            row["rep"] = int(row["rep"])
            row["cycles_per_touch"] = float(row["cycles_per_touch"])
        except (KeyError, ValueError):
            continue
        out.append(row)
    return out


def summarize(rows: list[dict]) -> dict:
    by_mode: dict[str, list[float]] = defaultdict(list)
    by_family_mode: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        mode = row.get("aggressor", "")
        if mode not in MODES:
            continue
        val = float(row["cycles_per_touch"])
        fam = int(row["family"])
        by_mode[mode].append(val)
        by_family_mode[(fam, mode)].append(val)

    mode_summary = {}
    for mode in MODES:
        vals = by_mode.get(mode, [])
        mode_summary[mode] = {
            "n": len(vals),
            "mean": mean(vals) if vals else None,
            "median": median(vals) if vals else None,
        }

    family_effects = []
    signs = []
    for fam in sorted({int(r["family"]) for r in rows}):
        same = by_family_mode.get((fam, "same_address"), [])
        controls = []
        for mode in ("none", "compute", "different_address", "random_address"):
            controls.extend(by_family_mode.get((fam, mode), []))
        if same and controls:
            eff = median(same) - median(controls)
            family_effects.append({"family": fam, "same_minus_controls_median": eff})
            if abs(eff) > 1e-9:
                signs.append(1 if eff > 0 else -1)

    return {
        "mode_summary": mode_summary,
        "family_effects": family_effects,
        "sign_agreement": max(signs.count(1), signs.count(-1)) / len(signs) if signs else 0.0,
    }


def permutation_p(rows: list[dict], rounds: int = 2000, seed: int = 51044) -> tuple[float, float]:
    values = []
    labels = []
    for row in rows:
        mode = row.get("aggressor", "")
        if mode not in MODES:
            continue
        values.append(float(row["cycles_per_touch"]))
        labels.append(1 if mode == "same_address" else 0)
    if not values or sum(labels) == 0 or sum(labels) == len(labels):
        return 0.0, 1.0

    def stat(vals: list[float], labs: list[int]) -> float:
        a = [v for v, lab in zip(vals, labs) if lab]
        b = [v for v, lab in zip(vals, labs) if not lab]
        return abs(median(a) - median(b))

    observed = stat(values, labels)
    rng = random.Random(seed)
    ge = 1
    labs = labels[:]
    for _ in range(rounds):
        rng.shuffle(labs)
        if stat(values, labs) >= observed:
            ge += 1
    return observed, ge / (rounds + 1)


def decide(summary: dict, observed: float, p_value: float) -> str:
    modes = summary["mode_summary"]
    same_n = modes["same_address"]["n"] or 0
    null_n = (modes["none"]["n"] or 0) + (modes["compute"]["n"] or 0) + (modes["random_address"]["n"] or 0)
    if same_n < 24 or null_n < 48:
        return "PHASE5_10D_UNDERPOWERED"
    if p_value >= 0.01:
        return "PHASE5_10D_NO_TOPOLOGY_BASIN"
    if summary["sign_agreement"] < 0.80:
        return "PHASE5_10D_ARTIFACT_DOMINANT"
    same_med = modes["same_address"]["median"]
    rand_med = modes["random_address"]["median"]
    comp_med = modes["compute"]["median"]
    if same_med is None or rand_med is None or comp_med is None:
        return "PHASE5_10D_UNDERPOWERED"
    if abs(same_med - rand_med) < 0.10 * max(1.0, abs(observed)):
        return "PHASE5_10D_ARTIFACT_DOMINANT"
    return "PHASE5_10D_TOPOLOGY_PREP_CANDIDATE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ns = ap.parse_args()

    rows = read_rows(ns.csv)
    summary = summarize(rows)
    observed, p_value = permutation_p(rows)
    verdict = decide(summary, observed, p_value)
    report = {
        "input": str(ns.csv),
        "rows": len(rows),
        "observed_abs_median_effect": observed,
        "permutation_p": p_value,
        "verdict": verdict,
        **summary,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
