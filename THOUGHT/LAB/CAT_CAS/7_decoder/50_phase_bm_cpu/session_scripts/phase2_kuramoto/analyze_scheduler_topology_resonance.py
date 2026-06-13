#!/usr/bin/env python3
"""Analyze scheduler topology resonance CSV output."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


FEATURES = ["elapsed_a_ns", "elapsed_b_ns", "delta_abs_ns", "sum_ns"]


def bacc(pred: list[int], truth: list[int]) -> float:
    parts = []
    for cls in (0, 1):
        idx = [i for i, v in enumerate(truth) if v == cls]
        if idx:
            parts.append(sum(1 for i in idx if pred[i] == truth[i]) / len(idx))
    return sum(parts) / len(parts) if parts else 0.0


def split(rows):
    train = [r for r in rows if int(r["seed"]) % 4 != 0]
    test = [r for r in rows if int(r["seed"]) % 4 == 0]
    return train, test


def threshold_score(train, test, feature, train_truth, test_truth):
    vals = sorted({float(r[feature]) for r in train})
    if not vals:
        return 0.0
    thresholds = vals[:: max(1, len(vals) // 32)]
    best = (0.0, thresholds[0], 0)
    for th in thresholds:
        for inv in (0, 1):
            pred = [int((float(r[feature]) >= th) ^ bool(inv)) for r in train]
            score = bacc(pred, train_truth)
            if score > best[0]:
                best = (score, th, inv)
    _, th, inv = best
    pred = [int((float(r[feature]) >= th) ^ bool(inv)) for r in test]
    return bacc(pred, test_truth)


def majority_score(train, test, key, train_truth, test_truth):
    global_major = Counter(train_truth).most_common(1)[0][0]
    bins = defaultdict(Counter)
    for r, y in zip(train, train_truth):
        bins[r[key]][y] += 1
    pred = [bins[r[key]].most_common(1)[0][0] if r[key] in bins else global_major for r in test]
    return bacc(pred, test_truth)


def shuffle_p95(rows, scorer, repeats=64):
    train, test = split(rows)
    yt = [int(r["answer"]) for r in train]
    yv = [int(r["answer"]) for r in test]
    combo = yt + yv
    vals = []
    for i in range(repeats):
        rng = random.Random(0x5CED000 + i)
        s = list(combo)
        rng.shuffle(s)
        vals.append(scorer(s[:len(yt)], s[len(yt):]))
    vals.sort()
    return vals[int(0.95 * (len(vals) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-in", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    with Path(args.csv_in).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train, test = split(rows)
    train_truth = [int(r["answer"]) for r in train]
    test_truth = [int(r["answer"]) for r in test]
    results = []
    for feature in FEATURES:
        def scorer(st, sv, f=feature):
            return threshold_score(train, test, f, st, sv)
        score = threshold_score(train, test, feature, train_truth, test_truth)
        null = shuffle_p95(rows, scorer)
        results.append({
            "feature": f"{feature}_threshold",
            "balanced_accuracy": round(score, 6),
            "shuffle_p95": round(null, 6),
            "shuffle_margin": round(score - null, 6),
        })
    for key in ["mode", "offset_iters", "carrier_low"]:
        def scorer(st, sv, k=key):
            return majority_score(train, test, k, st, sv)
        score = majority_score(train, test, key, train_truth, test_truth)
        null = shuffle_p95(rows, scorer)
        results.append({
            "feature": f"{key}_majority",
            "balanced_accuracy": round(score, 6),
            "shuffle_p95": round(null, 6),
            "shuffle_margin": round(score - null, 6),
        })

    candidates = [r for r in results if r["balanced_accuracy"] >= 0.60 and r["shuffle_margin"] >= 0.05]
    restore_failures = sum(1 for r in rows if int(r["restore_ok"]) != 1)
    verdict = "SCHEDULER_TOPOLOGY_RESONANCE_CANDIDATE" if candidates and restore_failures == 0 else "SCHEDULER_TOPOLOGY_RESONANCE_NOT_CONFIRMED"

    with Path(args.csv_out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "balanced_accuracy", "shuffle_p95", "shuffle_margin"])
        writer.writeheader()
        writer.writerows(results)
    with Path(args.md_out).open("w", encoding="utf-8") as handle:
        handle.write("# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE\n\n")
        handle.write("## Verdict\n\n")
        handle.write(f"`{verdict}`\n\n")
        handle.write("Core-pair phase-offset topology probe with shuffled-answer nulls.\n\n")
        handle.write("## Summary\n\n")
        handle.write(f"- rows: {len(rows)}\n")
        handle.write(f"- restore failures: {restore_failures}\n")
        handle.write(f"- answer balance: {dict(Counter(int(r['answer']) for r in rows))}\n")
        handle.write(f"- mean delta ns: {round(mean(float(r['delta_abs_ns']) for r in rows), 3)}\n\n")
        handle.write("| Feature | bAcc | Shuffle p95 | Margin |\n")
        handle.write("|---|---:|---:|---:|\n")
        for r in sorted(results, key=lambda item: item["shuffle_margin"], reverse=True):
            handle.write(f"| `{r['feature']}` | {r['balanced_accuracy']} | {r['shuffle_p95']} | {r['shuffle_margin']} |\n")
        handle.write("\n## Interpretation\n\n")
        if candidates:
            handle.write("At least one topology feature beat the shuffled-answer null. Rerun on fresh seed windows before promotion.\n")
        else:
            handle.write("No topology feature beat the shuffled-answer null with the required margin.\n")
        handle.write("\n## Boundary\n\n")
        handle.write("- No platform setting changes.\n")
        handle.write("- No candidate image construction.\n")
        handle.write("- No external instrumentation.\n")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
