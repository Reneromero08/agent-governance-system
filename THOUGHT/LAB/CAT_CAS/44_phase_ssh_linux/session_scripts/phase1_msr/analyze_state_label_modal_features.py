#!/usr/bin/env python3
"""Analyze modal timing/state features from state-label probe artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev


def balanced_accuracy(pred: list[int], truth: list[int]) -> float:
    parts = []
    for cls in (0, 1):
        idx = [i for i, value in enumerate(truth) if value == cls]
        if idx:
            parts.append(sum(int(pred[i] == truth[i]) for i in idx) / len(idx))
    return sum(parts) / len(parts) if parts else 0.0


def split_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [r for r in rows if int(r["seed"]) % 4 != 0]
    test = [r for r in rows if int(r["seed"]) % 4 == 0]
    return train, test


def build_norm(rows: list[dict[str, object]], key_fn) -> dict[str, tuple[float, float]]:
    bins: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bins[key_fn(row)].append(float(row["elapsed_ns"]))
    out = {}
    for key, values in bins.items():
        mu = mean(values)
        sd = pstdev(values) if len(values) > 1 else 1.0
        out[key] = (mu, sd if sd else 1.0)
    return out


def value_for(row: dict[str, object], family: str, norms: dict[str, dict[str, tuple[float, float]]]) -> float:
    elapsed = float(row["elapsed_ns"])
    if family == "elapsed":
        return elapsed
    if family == "mode_norm":
        mu, sd = norms["mode"].get(str(row["mode"]), (elapsed, 1.0))
        return (elapsed - mu) / sd
    if family == "core_norm":
        mu, sd = norms["core"].get(str(row["core"]), (elapsed, 1.0))
        return (elapsed - mu) / sd
    if family == "mode_core_norm":
        key = f"{row['mode']}|{row['core']}"
        mu, sd = norms["mode_core"].get(key, (elapsed, 1.0))
        return (elapsed - mu) / sd
    raise ValueError(f"unknown family: {family}")


def threshold_bacc(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    family: str,
    train_truth: list[int],
    test_truth: list[int],
    norms: dict[str, dict[str, tuple[float, float]]],
) -> float:
    values = sorted({value_for(row, family, norms) for row in train})
    if not values:
        return 0.0
    thresholds = values[:: max(1, len(values) // 32)]
    best = (0.0, thresholds[0], 0)
    for threshold in thresholds:
        for direction in (0, 1):
            pred = [int((value_for(row, family, norms) >= threshold) ^ bool(direction)) for row in train]
            score = balanced_accuracy(pred, train_truth)
            if score > best[0]:
                best = (score, threshold, direction)
    _, threshold, direction = best
    pred_test = [int((value_for(row, family, norms) >= threshold) ^ bool(direction)) for row in test]
    return balanced_accuracy(pred_test, test_truth)


def quantile_edges(values: list[float], bins: int) -> list[float]:
    values = sorted(values)
    if not values:
        return []
    return [values[min(len(values) - 1, max(0, int(len(values) * i / bins)))] for i in range(1, bins)]


def quantile_label(value: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def majority_bin_bacc(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    family: str,
    train_truth: list[int],
    test_truth: list[int],
    norms: dict[str, dict[str, tuple[float, float]]],
    bins: int,
    prefix: str,
) -> float:
    edges = quantile_edges([value_for(row, family, norms) for row in train], bins)
    global_majority = Counter(train_truth).most_common(1)[0][0]
    learned: dict[str, Counter[int]] = defaultdict(Counter)
    for row, truth in zip(train, train_truth):
        q = quantile_label(value_for(row, family, norms), edges)
        if prefix == "quantile":
            key = str(q)
        elif prefix == "state_quantile":
            key = f"{row['state_label']}|q={q}"
        elif prefix == "mode_quantile":
            key = f"{row['mode']}|q={q}"
        elif prefix == "core_quantile":
            key = f"{row['core']}|q={q}"
        else:
            raise ValueError(prefix)
        learned[key][truth] += 1
    pred = []
    for row in test:
        q = quantile_label(value_for(row, family, norms), edges)
        if prefix == "quantile":
            key = str(q)
        elif prefix == "state_quantile":
            key = f"{row['state_label']}|q={q}"
        elif prefix == "mode_quantile":
            key = f"{row['mode']}|q={q}"
        else:
            key = f"{row['core']}|q={q}"
        pred.append(learned[key].most_common(1)[0][0] if key in learned else global_majority)
    return balanced_accuracy(pred, test_truth)


def shuffled_p95(
    rows: list[dict[str, object]],
    scorer,
    repeats: int = 32,
) -> float:
    train, test = split_rows(rows)
    train_truth = [int(r["answer"]) for r in train]
    test_truth = [int(r["answer"]) for r in test]
    combined = train_truth + test_truth
    values = []
    for i in range(repeats):
        rng = random.Random(0x51A7E000 + i)
        shuffled = list(combined)
        rng.shuffle(shuffled)
        values.append(scorer(shuffled[: len(train)], shuffled[len(train):]))
    values = sorted(values)
    return values[int(0.95 * (len(values) - 1))] if values else 0.0


def analyze_file(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    train, test = split_rows(rows)
    if not train or not test:
        return []
    norms = {
        "mode": build_norm(train, lambda row: str(row["mode"])),
        "core": build_norm(train, lambda row: str(row["core"])),
        "mode_core": build_norm(train, lambda row: f"{row['mode']}|{row['core']}"),
    }
    train_truth = [int(r["answer"]) for r in train]
    test_truth = [int(r["answer"]) for r in test]
    out = []

    families = ["elapsed", "mode_norm", "core_norm", "mode_core_norm"]
    for family in families:
        def score_threshold(st, sv, fam=family):
            return threshold_bacc(train, test, fam, st, sv, norms)
        bacc = threshold_bacc(train, test, family, train_truth, test_truth, norms)
        null = shuffled_p95(rows, score_threshold)
        out.append({
            "file": path.name,
            "seed_start": data.get("seed_start"),
            "rounds": data.get("rounds"),
            "rows": len(rows),
            "feature": f"{family}_threshold",
            "balanced_accuracy": round(bacc, 6),
            "shuffle_p95": round(null, 6),
            "shuffle_margin": round(bacc - null, 6),
        })
        for prefix in ["quantile", "state_quantile", "mode_quantile", "core_quantile"]:
            def score_bins(st, sv, fam=family, pre=prefix):
                return majority_bin_bacc(train, test, fam, st, sv, norms, 4, pre)
            qbacc = majority_bin_bacc(train, test, family, train_truth, test_truth, norms, 4, prefix)
            qnull = shuffled_p95(rows, score_bins)
            out.append({
                "file": path.name,
                "seed_start": data.get("seed_start"),
                "rounds": data.get("rounds"),
                "rows": len(rows),
                "feature": f"{family}_{prefix}",
                "balanced_accuracy": round(qbacc, 6),
                "shuffle_p95": round(qnull, 6),
                "shuffle_margin": round(qbacc - qnull, 6),
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    rows = []
    for name in args.files:
        rows.extend(analyze_file(Path(name)))
    candidates = [
        row for row in rows
        if float(row["balanced_accuracy"]) >= 0.60 and float(row["shuffle_margin"]) >= 0.05
    ]
    by_feature: dict[str, int] = Counter(str(row["feature"]) for row in candidates)
    by_feature_seed: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        by_feature_seed[str(row["feature"])].add(str(row["seed_start"]))

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# PHASE2_STATE_LABEL_MODAL_FEATURE_SEARCH\n\n")
        handle.write("## Verdict\n\n")
        stable_features = [feature for feature, seeds in by_feature_seed.items() if len(seeds) >= 3]
        if stable_features:
            handle.write("`STATE_LABEL_MODAL_FEATURE_CANDIDATE`\n\n")
        else:
            handle.write("`STATE_LABEL_MODAL_FEATURE_NOT_CONFIRMED`\n\n")
        handle.write("Modal feature search over existing joined state/timing rows.\n\n")
        handle.write("## Summary\n\n")
        handle.write(f"- source files: {len(set(row['file'] for row in rows))}\n")
        handle.write(f"- feature rows: {len(rows)}\n")
        handle.write(f"- candidate feature rows: {len(candidates)}\n")
        handle.write(f"- stable features with >=3 distinct candidate seeds: {', '.join(stable_features) if stable_features else 'none'}\n\n")
        handle.write("## Candidate Feature Counts\n\n")
        handle.write("| Feature | Candidate rows | Distinct seeds |\n")
        handle.write("|---|---:|---:|\n")
        for feature, count in sorted(by_feature.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"| `{feature}` | {count} | {len(by_feature_seed[feature])} |\n")
        handle.write("\n")
        handle.write("## Top Rows\n\n")
        handle.write("| File | Feature | bAcc | Shuffle p95 | Margin |\n")
        handle.write("|---|---|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda r: float(r["shuffle_margin"]), reverse=True)[:20]:
            handle.write(
                f"| `{row['file']}` | `{row['feature']}` | {row['balanced_accuracy']} | "
                f"{row['shuffle_p95']} | {row['shuffle_margin']} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        if stable_features:
            handle.write("At least one modal feature family survived in three or more rows; promote to a larger reproducibility matrix.\n")
        else:
            handle.write("No modal feature family survived the shuffled-answer criterion across three or more rows. The current state-label modal route is not confirmed.\n")
        handle.write("\n## Boundary\n\n")
        handle.write("- Local artifact analysis only.\n")
        handle.write("- No platform setting changes.\n")
        handle.write("- No candidate image construction.\n")
    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
