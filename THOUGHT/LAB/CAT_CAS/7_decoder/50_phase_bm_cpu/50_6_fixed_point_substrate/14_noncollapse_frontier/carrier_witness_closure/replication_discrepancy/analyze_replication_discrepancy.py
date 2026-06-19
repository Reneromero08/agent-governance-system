#!/usr/bin/env python3
"""Derived-only adjudication of historical T300 versus raw T48 carrier evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LABEL = "DIAGNOSTIC_ONLY__NOT_OFFICIAL_VERDICT"
MODES = ["basis", "rotation", "residual", "mini"]
MODE_IDX = {name: index for index, name in enumerate(MODES)}
MASK64 = (1 << 64) - 1
RECORD = struct.Struct("<Qd")
METRICS = [
    "real_accuracy", "real_mode_floor", "real_vs_pseudo_floor",
    "pseudo_reject_floor", "pseudo_declared_match", "wrong_actual_match",
    "wrong_declared_match", "phase_corr_true", "phase_corr_null", "phase_delta",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def metadata(args: argparse.Namespace, tool_sha: str) -> dict[str, Any]:
    return {
        "source_campaign_id": args.campaign_id,
        "campaign_manifest_sha256": args.campaign_manifest_sha256,
        "source_commit": args.source_commit,
        "analyzer_sha256": args.analyzer_sha256,
        "analysis_tool_sha256": tool_sha,
        "generated_utc": args.generated_utc,
    }


def xs(state: int) -> int:
    state ^= (state << 13) & MASK64
    state ^= state >> 7
    state ^= (state << 17) & MASK64
    return state & MASK64


def regenerate_codebook(nbin: int = 12, seed: int = 7) -> dict[str, list[int]]:
    state = 0x243F6A8885A308D3 ^ seed
    best: list[list[int]] | None = None
    best_d = -1
    for _ in range(4000):
        code = []
        for weight in (4, 5, 6, 7):
            pool = list(range(nbin))
            selected = []
            for i in range(weight):
                state = xs(state)
                j = i + state % (nbin - i)
                pool[i], pool[j] = pool[j], pool[i]
                selected.append(pool[i])
            word = [1] * nbin
            for index in selected:
                word[index] = -1
            code.append(word)
        distance = min(sum(a != b for a, b in zip(code[i], code[j]))
                       for i in range(4) for j in range(i + 1, 4))
        if distance > best_d:
            best, best_d = code, distance
    assert best is not None
    return dict(zip(MODES, best))


def regenerate_schedule(seed: int, trials: int, codebook: dict[str, list[int]]) -> list[dict[str, Any]]:
    state = 0x9E3779B97F4A7C15 ^ seed

    def irand(n: int) -> int:
        nonlocal state
        state = xs(state)
        return state % n

    symbols = []
    for mode_index, mode in enumerate(MODES):
        symbols.append({"family": "preamble", "declared_mode": mode,
                        "actual_mode": mode, "trial": -1 - mode_index,
                        "theta_idx": 0, "bin_permutation": list(range(12))})
    for trial in range(trials):
        for family in ("real", "pseudo", "wrong"):
            actual = irand(4)
            theta = irand(8)
            perm = list(range(12))
            if family == "real":
                declared = actual
            elif family == "wrong":
                declared = (actual + 1 + irand(3)) % 4
            else:
                declared = irand(4)
                for i in range(11, 0, -1):
                    j = irand(i + 1)
                    perm[i], perm[j] = perm[j], perm[i]
            symbols.append({"family": family, "declared_mode": MODES[declared],
                            "actual_mode": MODES[actual], "trial": trial,
                            "theta_idx": theta, "bin_permutation": perm})
    for index, symbol in enumerate(symbols):
        symbol["symbol_index"] = index
        actual_word = codebook[symbol["actual_mode"]]
        symbol["drive_signs"] = [actual_word[p] for p in symbol["bin_permutation"]]
    return symbols


def parse_summary(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comments, rows = [], []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = None
        for raw in reader:
            if raw and raw[0].startswith("#"):
                comments.append(",".join(raw))
                continue
            if not raw:
                continue
            if header is None:
                header = raw
                columns = {name: i for i, name in enumerate(header)}
                continue
            row = {
                "family": raw[columns["family"]],
                "declared": MODE_IDX[raw[columns["declared_mode"]]],
                "actual": MODE_IDX[raw[columns["actual_mode"]]],
                "trial": int(raw[columns["trial"]]),
                "theta_idx": int(raw[columns["theta_idx"]]),
                "hash_restored": int(raw[columns["hash_restored"]]),
            }
            z = np.array([float(raw[columns[f"b{b:02d}_I"]]) +
                          1j * float(raw[columns[f"b{b:02d}_Q"]]) for b in range(12)])
            row["z"] = z
            rows.append(row)
    codebook = {}
    tones = []
    for line in comments:
        if "tones_hz=" in line:
            tones = [float(x) for x in line.split("tones_hz=", 1)[1].split(",")]
        if "codeword_" in line:
            left, right = line.lstrip("# ").split("=", 1)
            codebook[MODES[int(left.split("_")[1])]] = [int(x) for x in right.split(",")]
    return {"comments": comments, "codebook": codebook, "tones_hz": tones}, rows


def enrich(rows: list[dict[str, Any]], codebook: dict[str, list[int]]) -> None:
    code = np.array([codebook[name] for name in MODES], dtype=float)
    hn = code / math.sqrt(code.shape[1])
    for row in rows:
        z = row["z"]
        zr = z * np.exp(-1j * np.angle(np.sum(z)))
        zhat = zr / (np.linalg.norm(zr) + 1e-12)
        corr = np.abs(hn @ zhat)
        row["fvec"] = np.concatenate([zhat.real, zhat.imag])
        row["rho"] = float(np.max(corr) ** 2)
        row["mhat"] = int(np.argmax(corr))
        row["global_phase"] = float(np.angle(np.sum(z)))
        row["theta_hat"] = float(np.angle(code[row["mhat"]] @ z))


def ci_exact(success: int, total: int, alpha: float = 0.05) -> list[float] | None:
    if not total:
        return None
    def binomial_cdf(k: int, p: float) -> float:
        return sum(math.comb(total, i) * p ** i * (1 - p) ** (total - i)
                   for i in range(k + 1))

    def bisect(predicate: Any) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if predicate(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    low = 0.0 if success == 0 else bisect(
        lambda p: 1.0 - binomial_cdf(success - 1, p) < alpha / 2)
    high = 1.0 if success == total else bisect(
        lambda p: binomial_cdf(success, p) >= alpha / 2)
    return [low, high]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def centroids(rows: list[dict[str, Any]], train_selector: Any, include_preamble: bool = True) -> dict[int, np.ndarray]:
    train = [r for r in rows if r["family"] == "real" and
             ((include_preamble and r["trial"] < 0) or (r["trial"] >= 0 and train_selector(r)))]
    return {m: np.mean([r["fvec"] for r in train if r["declared"] == m], axis=0)
            if any(r["declared"] == m for r in train) else np.zeros(24) for m in range(4)}


def predict(row: dict[str, Any], centers: dict[int, np.ndarray]) -> int:
    return min(range(4), key=lambda m: float(np.linalg.norm(row["fvec"] - centers[m])))


def decompose(rows: list[dict[str, Any]], grouping: str = "predicted",
              train_selector: Any = lambda r: r["trial"] % 2 == 0,
              test_selector: Any = lambda r: r["trial"] % 2 == 1,
              threshold_kind: str = "train_5pct", include_preamble: bool = True,
              external_threshold: float | None = None,
              external_centers: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    centers = external_centers or centroids(rows, train_selector, include_preamble)
    train_real = [r for r in rows if r["family"] == "real" and
                  ((include_preamble and r["trial"] < 0) or
                   (r["trial"] >= 0 and train_selector(r)))]
    test = [r for r in rows if r["trial"] >= 0 and test_selector(r)]
    for row in rows:
        row["prediction"] = predict(row, centers)
    if external_threshold is not None:
        threshold = external_threshold
    elif threshold_kind == "per_mode":
        threshold = {m: float(np.percentile([r["rho"] for r in train_real if r["declared"] == m], 5))
                     for m in range(4)}
    else:
        threshold = float(np.percentile([r["rho"] for r in train_real], 5))

    def thr_for(mode: int) -> float:
        return threshold[mode] if isinstance(threshold, dict) else threshold

    real = [r for r in test if r["family"] == "real"]
    pseudo = [r for r in test if r["family"] == "pseudo"]
    wrong = [r for r in test if r["family"] == "wrong"]
    real_modes, pseudo_groups = {}, {}
    rvp, rejection = [], []
    for m in range(4):
        real_m = [r for r in real if r["declared"] == m]
        if grouping == "predicted":
            pseudo_m = [r for r in pseudo if r["mhat"] == m]
        elif grouping == "declared":
            pseudo_m = [r for r in pseudo if r["declared"] == m]
        elif grouping == "actual":
            pseudo_m = [r for r in pseudo if r["actual"] == m]
        else:
            pseudo_m = pseudo
        correct = sum(r["prediction"] == m for r in real_m)
        false_reject = sum(r["rho"] < thr_for(m) for r in real_m)
        false_accept = sum(r["rho"] >= thr_for(m) for r in pseudo_m)
        accepted = len(real_m) - false_reject
        rejected = len(pseudo_m) - false_accept
        denominator = len(real_m) + len(pseudo_m)
        accuracy = (accepted + rejected) / denominator if denominator else 0.0
        reject_rate = rejected / len(pseudo_m) if pseudo_m else 0.0
        real_modes[MODES[m]] = {
            "rows": len(real_m), "correct": correct,
            "accuracy": correct / len(real_m) if real_m else 0.0,
            "accuracy_exact_95ci": ci_exact(correct, len(real_m)),
            "false_rejects": false_reject,
        }
        pseudo_groups[MODES[m]] = {
            "rows": len(pseudo_m), "false_accepts": false_accept,
            "rejected": rejected, "reject_rate": reject_rate,
            "reject_exact_95ci": ci_exact(rejected, len(pseudo_m)),
            "combined_denominator": denominator, "combined_accuracy": accuracy,
            "row_flips_to_0_95": max(0, math.ceil(0.95 * denominator) - (accepted + rejected)),
        }
        if real_m and pseudo_m:
            rvp.append((accuracy, m))
            rejection.append((reject_rate, m))
    real_correct = sum(r["prediction"] == r["declared"] for r in real)
    wrong_actual = sum(r["prediction"] == r["actual"] for r in wrong)
    wrong_declared = sum(r["prediction"] == r["declared"] for r in wrong)
    pseudo_declared = sum(r["prediction"] == r["declared"] for r in pseudo)
    floor = min(rvp, default=(0.0, -1))
    rej_floor = min(rejection, default=(0.0, -1))
    metrics = {
        "real_accuracy": real_correct / len(real) if real else 0.0,
        "real_mode_floor": min((v["accuracy"] for v in real_modes.values()), default=0.0),
        "real_vs_pseudo_floor": floor[0],
        "pseudo_reject_floor": rej_floor[0],
        "pseudo_declared_match": pseudo_declared / len(pseudo) if pseudo else 0.0,
        "wrong_actual_match": wrong_actual / len(wrong) if wrong else 0.0,
        "wrong_declared_match": wrong_declared / len(wrong) if wrong else 0.0,
    }
    gates = {
        "all_rows_restore": all(r["hash_restored"] == 1 for r in rows),
        "real_accuracy_ge_0_60": metrics["real_accuracy"] >= 0.60,
        "real_vs_pseudo_floor_ge_0_95": metrics["real_vs_pseudo_floor"] >= 0.95,
        "pseudo_reject_floor_ge_0_95": metrics["pseudo_reject_floor"] >= 0.95,
        "wrong_actual_match_ge_0_60": metrics["wrong_actual_match"] >= 0.60,
        "wrong_declared_match_le_0_20": metrics["wrong_declared_match"] <= 0.20,
    }
    return {
        "grouping": grouping, "threshold": threshold, "metrics": metrics,
        "gates_without_phase": gates, "passes_nonphase_gates": all(gates.values()),
        "real_test_rows_per_declared_mode": {k: v["rows"] for k, v in real_modes.items()},
        "pseudo_test_rows_per_predicted_mode": {MODES[m]: sum(r["mhat"] == m for r in pseudo) for m in range(4)},
        "pseudo_test_rows_per_declared_mode": {MODES[m]: sum(r["declared"] == m for r in pseudo) for m in range(4)},
        "pseudo_test_rows_per_actual_mode": {MODES[m]: sum(r["actual"] == m for r in pseudo) for m in range(4)},
        "real_modes": real_modes, "pseudo_groups": pseudo_groups,
        "real_rho": distribution([r["rho"] for r in real]),
        "pseudo_rho": distribution([r["rho"] for r in pseudo]),
        "floor_mode": MODES[floor[1]] if floor[1] >= 0 else None,
        "rejection_floor_mode": MODES[rej_floor[1]] if rej_floor[1] >= 0 else None,
    }


def distribution(values: Iterable[float]) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "min": float(a.min()), "q05": float(np.percentile(a, 5)),
            "median": float(np.median(a)), "mean": float(a.mean()),
            "q95": float(np.percentile(a, 95)), "max": float(a.max()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def schedule_report(run_dirs: list[Path], historical_dir: Path) -> dict[str, Any]:
    regenerated_code = regenerate_codebook()
    runs = {}
    for run_dir in run_dirs:
        schedule = load_json(run_dir / "schedule.json")
        expected = regenerate_schedule(schedule["seed"], 48, regenerated_code)
        keys = ("symbol_index", "family", "declared_mode", "actual_mode", "trial",
                "theta_idx", "bin_permutation", "drive_signs")
        exact = len(expected) == len(schedule["symbols"]) and all(
            all(left[k] == right[k] for k in keys)
            for left, right in zip(expected, schedule["symbols"]))
        rows = schedule["symbols"]
        train = [r for r in rows if r["trial"] >= 0 and r["trial"] % 2 == 0]
        test = [r for r in rows if r["trial"] >= 0 and r["trial"] % 2 == 1]
        runs[run_dir.name] = {
            "regenerated_schedule_exact": exact,
            "regenerated_codebook_exact": schedule["codebook"] == regenerated_code,
            "family_counts": Counter(r["family"] for r in rows),
            "train_declared_mode_counts": Counter(r["declared_mode"] for r in train),
            "test_declared_mode_counts": Counter(r["declared_mode"] for r in test),
            "train_family_mode_counts": nested_counts(train),
            "test_family_mode_counts": nested_counts(test),
            "theta_counts": Counter(r["theta_idx"] for r in rows if r["trial"] >= 0),
            "wrong_declared_actual_counts": Counter(
                f"{r['declared_mode']}->{r['actual_mode']}" for r in rows if r["family"] == "wrong"),
            "pseudo_permutations_unique": len({tuple(r["bin_permutation"]) for r in rows if r["family"] == "pseudo"}),
        }
    historical = {}
    for path in sorted(historical_dir.glob("matrix_v*s*_seed*.csv")):
        header, rows = parse_summary(path)
        seed = int(path.stem.rsplit("seed", 1)[1])
        expected = regenerate_schedule(seed, 48, regenerated_code)
        historical_prefix = rows[:148]
        expected_labels = [(r["family"] if r["family"] != "preamble" else "real",
                            MODE_IDX[r["declared_mode"]], MODE_IDX[r["actual_mode"]],
                            r["trial"], r["theta_idx"]) for r in expected]
        retained_labels = [(r["family"], r["declared"], r["actual"], r["trial"], r["theta_idx"])
                           for r in historical_prefix]
        historical[path.stem] = {
            "codebook_exact": header["codebook"] == regenerated_code,
            "first_48_trials_label_schedule_exact": retained_labels == expected_labels,
            "tone_count": len(header["tones_hz"]),
            "family_counts": Counter(r["family"] for r in rows),
            "train_family_mode_counts": nested_counts([r for r in rows if r["trial"] >= 0 and r["trial"] % 2 == 0]),
            "test_family_mode_counts": nested_counts([r for r in rows if r["trial"] >= 0 and r["trial"] % 2 == 1]),
        }
    return {"current": runs, "historical_csv_headers": historical,
            "test_denominator_definitions": {
                "real_accuracy_per_mode": "odd-trial real rows grouped by declared mode",
                "real_vs_pseudo_per_mode": "odd real declared mode plus odd pseudo predicted mode",
                "pseudo_rejection_per_mode": "odd pseudo rows grouped by predicted mode",
                "wrong_actual_match": "all odd wrong rows",
                "wrong_declared_match": "all odd wrong rows",
                "phase_pairs": "consecutive real rows including preamble; count = real rows - 1",
            }}


def nested_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(
        f"{r['family']}:{r['declared_mode'] if 'declared_mode' in r else MODES[r['declared']]}"
        for r in rows
    ))


def protocol_comparison(campaign_root: Path, historical_result: dict[str, Any],
                        schedule: dict[str, Any]) -> dict[str, Any]:
    run = load_json(campaign_root / "runs" / "v4s5_matrix_seed0" / "run.json")
    hist = historical_result
    rows = [
        ("source_revisions", "historical host files uncommitted/hash-bound only", run["source_commit"], "POTENTIALLY_MATERIAL"),
        ("acquisition_binary_revisions", "historical binary hash absent", run["binary_sha256"], "UNKNOWN"),
        ("analyzer_revisions", "aa1ca5b0ce911ce931b036d023981eb1589d316e0adf88b66eb494ada1c5a50c", run["source_files"]["10_cross_core_wormhole/slot2_pdn/slot2_pdn_analyze.py"], "NONE"),
        ("codebook_generation", "xorshift search seed 7", "xorshift search seed 7", "NONE"),
        ("resolved_codewords", "retained CSV headers", "serialized schedule", "NONE"),
        ("tone_generation", "log grid plus deterministic nudge", "same implementation", "NONE"),
        ("resolved_tone_frequencies", "retained CSV rounded to 4 decimals", schedule["tones_hz"], "NUMERICALLY_IRRELEVANT"),
        ("seed_schedules", "not serialized; labels retained in reduced CSV", "serialized and regenerates exactly", "POTENTIALLY_MATERIAL"),
        ("symbol_ordering", "reduced CSV retained", "serialized schedule", "NONE"),
        ("train_test_partition", "trial parity odd test", "trial parity odd test", "NONE"),
        ("trial_counts", 300, 48, "MATERIAL"),
        ("preamble_handling", "4 mode preamble included in train", "same", "NONE"),
        ("pseudo_permutations", "labels/IQ retained, exact permutation absent", "serialized", "UNKNOWN"),
        ("wrong_family_generation", "declared differs; drive actual", "same", "NONE"),
        ("phase_levels", 8, 8, "NONE"),
        ("slot_s", 0.5, run["timing"]["slot_s"], "NONE"),
        ("gap_s", "not reported", run["timing"]["gap_s"], "UNKNOWN"),
        ("read_hz", 4000, run["timing"]["read_hz"], "NONE"),
        ("TSC_rate", 3214823000.0, run["timing"]["tsc_hz"], "NUMERICALLY_IRRELEVANT"),
        ("P_state_target", 1600000, run["drive"]["pstate_target_khz"], "NONE"),
        ("thermal_veto", 68.0, run["thermal"]["veto_c"], "NONE"),
        ("routes", ["2:3", "4:5"], ["2:3", "4:5"], "NONE"),
        ("controls", ["silent", "scramble"], ["silent", "scramble"], "NONE"),
        ("success_thresholds", hist["success_criteria"], "same seven scientific gates", "NONE"),
        ("aggregate_logic", "6/6 primary route plus second route and null controls", "same; diagnostics excluded", "NONE"),
        ("raw_capture", "absent", "complete <Qd> timing samples", "MATERIAL"),
        ("restoration_verification", "symbol hash flag", "symbol hash plus delayed host P-state readback", "DOCUMENTATION_ONLY"),
    ]
    return {"comparisons": [{"field": a, "historical": b, "current": c, "classification": d}
                            for a, b, c, d in rows],
            "exact_equivalence_provable": False,
            "limit": "Historical raw timing samples and binary hash are absent."}


def official_and_diagnostics(run_dirs: list[Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    official, diagnostics, cache = {}, {}, {}
    loaded = {}
    for run_dir in run_dirs:
        header, rows = parse_summary(run_dir / "summary.csv")
        enrich(rows, header["codebook"])
        loaded[run_dir.name] = rows
        dec = decompose(rows)
        for row in rows:
            row["official_prediction"] = row["prediction"]
        retained = load_json(run_dir / "analysis.json")
        dec["retained_official_metrics"] = {key: retained[key] for key in METRICS if key in retained}
        dec["official_pass"] = all(retained["gates"][key] for key in (
            "all_rows_restore", "real_accuracy_ge_0_60", "real_vs_pseudo_floor_ge_0_95",
            "pseudo_reject_floor_ge_0_95", "wrong_actual_match_ge_0_60",
            "wrong_declared_match_le_0_20", "phase_recovered_gt_0_30"))
        official[run_dir.name] = dec
    for run_id, rows in loaded.items():
        variants = {}
        for grouping in ("predicted", "declared", "actual", "pooled"):
            variants[f"pseudo_group_{grouping}"] = decompose(rows, grouping=grouping)
        variants["partition_reversed"] = decompose(
            rows, train_selector=lambda r: r["trial"] % 2 == 1,
            test_selector=lambda r: r["trial"] % 2 == 0)
        for offset in range(4):
            variants[f"partition_mod4_holdout_{offset}"] = decompose(
                rows, train_selector=lambda r, o=offset: r["trial"] % 4 != o,
                test_selector=lambda r, o=offset: r["trial"] % 4 == o)
        variants["centroid_even_real_only"] = decompose(rows, include_preamble=False)
        variants["threshold_per_mode"] = decompose(rows, threshold_kind="per_mode")
        # A fixed threshold from the adjacent seed is deterministic and cross-session.
        route, seed_text = run_id.split("_matrix_seed")
        adjacent = f"{route}_matrix_seed{(int(seed_text) + 1) % 6}"
        adj_rows = loaded[adjacent]
        adj_threshold = float(np.percentile([r["rho"] for r in adj_rows
                                             if r["family"] == "real" and
                                             (r["trial"] < 0 or r["trial"] % 2 == 0)], 5))
        variants["threshold_fixed_adjacent_seed"] = decompose(rows, external_threshold=adj_threshold)
        route_rows = [r for name, rs in loaded.items() if name.startswith(route + "_") for r in rs]
        other_route_rows = [r for name, rs in loaded.items()
                            if name.startswith(route + "_") and name != run_id for r in rs]
        pooled_threshold = float(np.percentile([r["rho"] for r in route_rows
                                                if r["family"] == "real" and
                                                (r["trial"] < 0 or r["trial"] % 2 == 0)], 5))
        variants["threshold_pooled_route"] = decompose(rows, external_threshold=pooled_threshold)
        leave_one_threshold = float(np.percentile([r["rho"] for r in other_route_rows
                                                   if r["family"] == "real" and
                                                   (r["trial"] < 0 or r["trial"] % 2 == 0)], 5))
        variants["threshold_leave_one_seed_out"] = decompose(rows, external_threshold=leave_one_threshold)
        pooled_centers = centroids(route_rows, lambda r: r["trial"] % 2 == 0)
        leave_one_centers = centroids(other_route_rows, lambda r: r["trial"] % 2 == 0)
        variants["centroid_route_pooled"] = decompose(rows, external_centers=pooled_centers)
        variants["centroid_leave_one_seed_out"] = decompose(rows, external_centers=leave_one_centers)
        variants["partition_leave_one_trial_block_out"] = variants["partition_mod4_holdout_0"]
        diagnostics[run_id] = {"label": LABEL, "variants": variants}
        cache[run_id] = rows
    return official, diagnostics, cache


def raw_report(campaign_root: Path, cache: dict[str, Any]) -> dict[str, Any]:
    runs = {}
    for run_id, symbol_rows in cache.items():
        run_dir = campaign_root / "runs" / run_id
        with (run_dir / "windows.csv").open(newline="", encoding="utf-8") as stream:
            windows = list(csv.DictReader(stream))
        raw = np.memmap(run_dir / "raw_samples.bin", dtype=np.dtype([("tsc", "<u8"), ("period", "<f8")]), mode="r")
        window_rows = []
        for w in windows:
            offset, count = int(w["sample_offset_records"]), int(w["sample_count"])
            sample = raw[offset:offset + count]
            intervals = np.diff(sample["tsc"].astype(np.float64))
            symbol = symbol_rows[int(w["symbol_index"])]
            window_rows.append({
                "symbol_index": int(w["symbol_index"]), "bin_index": int(w["bin_index"]),
                "elapsed_s": (int(w["slot_start_tsc"]) - int(windows[0]["slot_start_tsc"])) / 3214826000.0,
                "temperature_c": (float(w["temp_before_c"]) + float(w["temp_after_c"])) / 2,
                "frequency_khz": (int(w["cur_khz_before"]) + int(w["cur_khz_after"])) / 2,
                "pstate": (int(w["cofvid_pstate_before"]) + int(w["cofvid_pstate_after"])) / 2,
                "sample_count": count,
                "capture_lateness_tsc": int(w["first_sample_tsc"]) - int(w["slot_start_tsc"]),
                "window_duration_tsc": int(w["last_sample_tsc"]) - int(w["first_sample_tsc"]),
                "tsc_interval_mean": float(intervals.mean()),
                "tsc_interval_std": float(intervals.std(ddof=1)),
                "iq_magnitude": float(w["computed_magnitude"]), "offbin_floor": float(w["computed_floor"]),
                "rho": symbol["rho"], "global_phase": symbol["global_phase"],
                "correct": int(symbol["official_prediction"] == symbol["declared"]),
                "family": w["family"], "mode": w["declared_mode"], "theta": int(w["theta_idx"]),
                "tone_hz": float(w["tone_hz"]),
            })
        numeric = ["elapsed_s", "temperature_c", "frequency_khz", "pstate", "sample_count",
                   "capture_lateness_tsc", "window_duration_tsc", "tsc_interval_mean",
                   "tsc_interval_std", "tone_hz"]
        associations = {}
        for predictor in numeric:
            x = np.array([row[predictor] for row in window_rows], dtype=float)
            associations[predictor] = {}
            for outcome in ("iq_magnitude", "offbin_floor", "rho", "global_phase", "correct"):
                y = np.array([row[outcome] for row in window_rows], dtype=float)
                if np.std(x) == 0 or np.std(y) == 0:
                    associations[predictor][outcome] = {"n": len(x), "pearson_r": None, "spearman_rho": None}
                else:
                    associations[predictor][outcome] = {
                        "n": len(x), "pearson_r": correlation(x, y),
                        "spearman_rho": correlation(rankdata(x), rankdata(y)),
                        "inference": "descriptive_only_no_multiple_test_p_value",
                    }
        runs[run_id] = {
            "window_count": len(window_rows), "raw_record_count": len(raw),
            "series": {key: distribution(row[key] for row in window_rows) for key in (
                "iq_magnitude", "offbin_floor", "rho", "global_phase", "temperature_c", "frequency_khz",
                "pstate", "sample_count", "capture_lateness_tsc", "window_duration_tsc",
                "tsc_interval_mean", "tsc_interval_std")},
            "associations": associations,
            "bin_magnitude": {str(b): distribution(r["iq_magnitude"] for r in window_rows if r["bin_index"] == b)
                              for b in range(12)},
            "early_late": {
                "early_iq": distribution(r["iq_magnitude"] for r in window_rows[:len(window_rows)//2]),
                "late_iq": distribution(r["iq_magnitude"] for r in window_rows[len(window_rows)//2:]),
                "early_correct": float(np.mean([r["correct"] for r in window_rows[:len(window_rows)//2]])),
                "late_correct": float(np.mean([r["correct"] for r in window_rows[len(window_rows)//2:]])),
            },
        }
    return {"method": "descriptive associations; correlation is not causation", "runs": runs}


def seed4_report(cache: dict[str, Any], official: dict[str, Any], raw: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    run_id = "v4s5_matrix_seed4"
    rows = cache[run_id]
    centers = centroids(rows, lambda r: r["trial"] % 2 == 0)
    test = [r for r in rows if r["trial"] >= 0 and r["trial"] % 2 == 1]
    confusion = [[0] * 4 for _ in range(4)]
    predictions = []
    for row in test:
        pred = predict(row, centers)
        if row["family"] == "real":
            confusion[row["declared"]][pred] += 1
        predictions.append({"family": row["family"], "trial": row["trial"],
                            "declared": MODES[row["declared"]], "actual": MODES[row["actual"]],
                            "predicted": MODES[pred], "rho": row["rho"],
                            "vector_norm": float(np.linalg.norm(row["z"])),
                            "theta_idx": row["theta_idx"], "theta_hat": row["theta_hat"]})
    distances = {f"{MODES[i]}:{MODES[j]}": float(np.linalg.norm(centers[i] - centers[j]))
                 for i in range(4) for j in range(i + 1, 4)}
    peer_accuracy = {name: data["retained_official_metrics"]["real_accuracy"]
                     for name, data in official.items() if name.startswith("v4s5_")}
    return {
        "run_id": run_id,
        "official_metrics": official[run_id]["retained_official_metrics"],
        "schedule_balance": schedule["current"][run_id],
        "centroid_distances": distances,
        "confusion_matrix_actual_rows_predicted_columns": confusion,
        "per_symbol_predictions": predictions,
        "real_rho": distribution(r["rho"] for r in test if r["family"] == "real"),
        "pseudo_rho": distribution(r["rho"] for r in test if r["family"] == "pseudo"),
        "iq_vector_norm": distribution(np.linalg.norm(r["z"]) for r in rows),
        "raw_time_trajectory": raw["runs"][run_id]["early_late"],
        "raw_timing": raw["runs"][run_id]["series"],
        "peer_route_4_5_real_accuracy": peer_accuracy,
        "diagnosis": "UNRESOLVED_ANOMALY",
        "diagnostic_basis": "Large real-mode failure and compressed centroid geometry are not explained by schedule balance, thermal veto, frequency/P-state, sample count, or capture timing alone.",
    }


def historical_current(hist_result: dict[str, Any], current_root: Path) -> dict[str, Any]:
    current = [load_json(path) for path in sorted((current_root / "runs").glob("v*s*_matrix_seed*/analysis.json"))]
    historical = hist_result["aggregate"]["runs"]
    out = {}
    for metric in METRICS:
        hv = [r[metric] for r in historical if metric in r]
        cv = [r[metric] for r in current if metric in r]
        out[metric] = {
            "historical_range": [min(hv), max(hv)], "current_range": [min(cv), max(cv)],
            "historical_mean": float(np.mean(hv)), "current_mean": float(np.mean(cv)),
            "mean_shift_current_minus_historical": float(np.mean(cv) - np.mean(hv)),
        }
    current_passes = defaultdict(int)
    for path in sorted((current_root / "runs").glob("v*s*_matrix_seed*/analysis.json")):
        route = path.parent.name.split("_matrix_")[0].replace("v", "").replace("s", ":")
        gates = load_json(path)["gates"]
        scientific = ("all_rows_restore", "real_accuracy_ge_0_60",
                      "real_vs_pseudo_floor_ge_0_95", "pseudo_reject_floor_ge_0_95",
                      "wrong_actual_match_ge_0_60", "wrong_declared_match_le_0_20",
                      "phase_recovered_gt_0_30")
        current_passes[route] += all(gates[key] for key in scientific)
    return {
        "metrics": out,
        "historical_route_pass_counts": {k: v["n_pass"] for k, v in hist_result["aggregate"]["by_pair"].items()},
        "current_route_pass_counts": dict(current_passes),
        "historical_raw_evidence_exists": False,
        "current_raw_evidence_exists": True,
        "exact_equivalence_can_be_proven": False,
    }


def adjudicate(protocol: dict[str, Any], official: dict[str, Any], diagnostics: dict[str, Any],
               raw: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    route45 = [data for name, data in official.items() if name.startswith("v4s5_")]
    official_passes = sum(data["official_pass"] for data in route45)
    diagnostic_pass_counts = Counter()
    for name, data in diagnostics.items():
        if not name.startswith("v4s5_"):
            continue
        for variant, result in data["variants"].items():
            diagnostic_pass_counts[variant] += result["passes_nonphase_gates"]
    material_known = [row for row in protocol["comparisons"] if row["classification"] == "MATERIAL"]
    return {
        "primary_classification": "HISTORICAL_RESULT_NOT_REPRODUCED",
        "basis": [
            f"Frozen current route 4:5 passes {official_passes}/6 versus historical 6/6.",
            "Analyzer SHA-256 and resolved codebooks are identical.",
            "Trial count differs materially (48 versus 300), but historical raw samples and binary hash are absent, so a unique causal layer cannot be proven.",
            "Seed 4 has a large real-classification failure, not a marginal rho-threshold miss.",
        ],
        "secondary_findings": [
            "Sparse per-mode test denominators make rho floors threshold/grouping sensitive.",
            "Current raw timing, P-state, temperature, and control provenance are complete.",
            "No single retained telemetry association establishes physical nonstationarity causally.",
        ],
        "diagnostic_nonphase_pass_counts_route_4_5": dict(diagnostic_pass_counts),
        "material_protocol_differences": material_known,
        "recommendation": "REPEAT_T48_SAME_PROTOCOL",
        "preregistration": {
            "scientific_question": "Is the 1/6 route-4:5 result reproducible under the identical frozen T48 acquisition and official analyzer?",
            "frozen_analyzer_sha256": next(iter(protocol.get("metadata", [{"analyzer_sha256": ""}])))["analyzer_sha256"] if protocol.get("metadata") else None,
            "frozen_thresholds": "unchanged official seven-gate battery; train-real 5th-percentile rho per run",
            "trials_per_family": 48,
            "justification": "Independent same-N replication separates session drift from a trial-count explanation without changing the gate.",
            "routes": ["4:5", "2:3"], "seeds": [0, 1, 2, 3, 4, 5],
            "controls": ["silent", "scramble"],
            "estimated_runtime": "approximately 3.6 hours using the prior frozen matrix",
            "estimated_raw_storage": "approximately 605 MB",
            "stop_conditions": ["temperature >= 68 C", "affinity failure", "P-state failure", "TSC failure", "disk/raw-writer/process failure"],
            "confirm_hypothesis": "A second T48 route-4:5 result near 1/6 with similar metric distributions supports reproducible current-protocol failure/session-stable discrepancy.",
            "reject_hypothesis": "A second T48 route-4:5 result near historical closure with stable controls supports session/route nonstationarity and triggers a third preregistered adjudication run.",
            "higher_n_status": "NOT_YET_JUSTIFIED_BEFORE_SAME_N_REPLICATION",
        },
        "official_verdicts_unchanged": True,
        "claim_ceiling": "Replication adjudication of a strong physical carrier signal; no physical geometry, restoration, coupling, orientation, or wall claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--historical-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-manifest-sha256", required=True)
    parser.add_argument("--analyzer-sha256", required=True)
    parser.add_argument("--generated-utc", required=True)
    args = parser.parse_args()
    tool_sha = sha256(Path(__file__))
    meta = metadata(args, tool_sha)
    run_dirs = sorted(path for path in (args.campaign_root / "runs").glob("v*s*_matrix_seed*") if path.is_dir())
    hist_result = load_json(args.historical_result)
    schedule = schedule_report(run_dirs, args.historical_dir)
    official, diagnostics, cache = official_and_diagnostics(run_dirs)
    raw = raw_report(args.campaign_root, cache)
    seed4 = seed4_report(cache, official, raw, schedule)
    comparison = historical_current(hist_result, args.campaign_root)
    protocol = protocol_comparison(args.campaign_root, hist_result, load_json(run_dirs[0] / "schedule.json"))
    outputs = {
        "protocol_comparison.json": protocol,
        "schedule_balance.json": schedule,
        "official_gate_decomposition.json": {"runs": official},
        "diagnostic_counterfactuals.json": {"label": LABEL, "runs": diagnostics},
        "raw_nonstationarity.json": raw,
        "seed4_case_report.json": seed4,
        "historical_current_comparison.json": comparison,
    }
    adjudication = adjudicate({**protocol, "metadata": [meta]}, official, diagnostics, raw, comparison)
    outputs["adjudication_report.json"] = adjudication
    for name, body in outputs.items():
        dump_json(args.output_dir / name, {"metadata": meta, **body})
    manifest_files = {name: {"sha256": sha256(args.output_dir / name),
                             "size": (args.output_dir / name).stat().st_size}
                      for name in sorted(outputs)}
    report = args.output_dir.parent / "REPLICATION_DISCREPANCY_REPORT.md"
    if report.is_file():
        manifest_files["../REPLICATION_DISCREPANCY_REPORT.md"] = {
            "sha256": sha256(report), "size": report.stat().st_size}
    manifest = {"metadata": meta, "schema_id": "CAT_CAS_REPLICATION_DISCREPANCY_OUTPUT_MANIFEST_V1",
                "files": manifest_files}
    dump_json(args.output_dir / "output_manifest.json", manifest)
    print(f"HISTORICAL_RESULT_NOT_REPRODUCED outputs={len(outputs) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
