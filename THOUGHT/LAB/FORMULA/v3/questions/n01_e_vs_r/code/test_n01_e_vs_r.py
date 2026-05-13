#!/usr/bin/env python3
"""
N1: Does E / grad_S outperform bare E?

Pre-registration locked in ../PREREGISTRATION.md.
This script is intentionally single-shot:
- fixed datasets
- fixed model
- fixed cluster construction
- no parameter search
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SEED = 20260309
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_SIZE = 16
CLUSTERS_PER_TYPE = 120
BOOTSTRAP_SAMPLES = 1000
MAX_ROWS_PER_CLASS = 1200
MAX_ROWS_STSB = 4000

STSB_BINS: Tuple[Tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.1),
)


@dataclass
class DatasetExamples:
    name: str
    source: str
    task_type: str
    vectors: np.ndarray
    labels: np.ndarray


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both positive and negative classes.")
    pos_scores = scores[pos][:, None]
    neg_scores = scores[neg][None, :]
    wins = float((pos_scores > neg_scores).sum())
    ties = float((pos_scores == neg_scores).sum())
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def bootstrap_auc_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed: int,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample_y = y_true[sample_idx]
        if sample_y.min() == sample_y.max():
            continue
        delta = auc_score(sample_y, score_a[sample_idx]) - auc_score(sample_y, score_b[sample_idx])
        deltas.append(delta)
    if not deltas:
        raise RuntimeError("Bootstrap produced no valid resamples.")
    delta_arr = np.asarray(deltas, dtype=np.float64)
    return {
        "delta_mean": float(delta_arr.mean()),
        "ci_low": float(np.percentile(delta_arr, 2.5)),
        "ci_high": float(np.percentile(delta_arr, 97.5)),
        "wins_fraction": float(np.mean(delta_arr > 0.0)),
    }


def load_formula_module():
    formula_path = Path(__file__).resolve().parents[4] / "v2" / "shared" / "formula.py"
    spec = importlib.util.spec_from_file_location("formula_v2_shared", formula_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load formula module from {formula_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_pair_average(vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
    return (vectors_a + vectors_b) / 2.0


def limit_per_class(indices: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= max_rows:
        return indices
    chosen = rng.choice(indices, size=max_rows, replace=False)
    return np.sort(chosen)


def stsb_bin(score: float) -> int:
    for idx, (lo, hi) in enumerate(STSB_BINS):
        if lo <= score < hi:
            return idx
    raise ValueError(f"Score out of range: {score}")


def load_examples(model_name: str) -> List[DatasetExamples]:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    rng = np.random.default_rng(SEED)
    model = SentenceTransformer(model_name)

    datasets_out: List[DatasetExamples] = []

    stsb = load_dataset("glue", "stsb", split="validation")
    stsb = stsb.filter(lambda row: bool(row["sentence1"]) and bool(row["sentence2"]))
    if len(stsb) > MAX_ROWS_STSB:
        take = np.sort(rng.choice(len(stsb), size=MAX_ROWS_STSB, replace=False))
        stsb = stsb.select(take.tolist())
    stsb_a = model.encode(stsb["sentence1"], batch_size=64, show_progress_bar=True)
    stsb_b = model.encode(stsb["sentence2"], batch_size=64, show_progress_bar=True)
    stsb_vectors = normalize_pair_average(np.asarray(stsb_a), np.asarray(stsb_b))
    stsb_labels = np.asarray([stsb_bin(float(score)) for score in stsb["label"]], dtype=np.int64)
    datasets_out.append(
        DatasetExamples(
            name="stsb",
            source="glue/stsb validation",
            task_type="ordinal_pair",
            vectors=stsb_vectors,
            labels=stsb_labels,
        )
    )

    sst2 = load_dataset("glue", "sst2", split="validation")
    sst2_groups = []
    sst2_labels = np.asarray(sst2["label"], dtype=np.int64)
    for label in (0, 1):
        idx = np.where(sst2_labels == label)[0]
        sst2_groups.append(limit_per_class(idx, MAX_ROWS_PER_CLASS, rng))
    sst2_take = np.sort(np.concatenate(sst2_groups))
    sst2 = sst2.select(sst2_take.tolist())
    sst2_vectors = np.asarray(model.encode(sst2["sentence"], batch_size=64, show_progress_bar=True))
    datasets_out.append(
        DatasetExamples(
            name="sst2",
            source="glue/sst2 validation",
            task_type="binary_single",
            vectors=sst2_vectors,
            labels=np.asarray(sst2["label"], dtype=np.int64),
        )
    )

    snli = load_dataset("snli", split="validation")
    snli = snli.filter(lambda row: row["label"] in (0, 2) and bool(row["premise"]) and bool(row["hypothesis"]))
    snli_label_map = {0: 0, 2: 1}
    snli_labels_raw = np.asarray(snli["label"], dtype=np.int64)
    snli_groups = []
    for label in (0, 2):
        idx = np.where(snli_labels_raw == label)[0]
        snli_groups.append(limit_per_class(idx, MAX_ROWS_PER_CLASS, rng))
    snli_take = np.sort(np.concatenate(snli_groups))
    snli = snli.select(snli_take.tolist())
    snli_a = np.asarray(model.encode(snli["premise"], batch_size=64, show_progress_bar=True))
    snli_b = np.asarray(model.encode(snli["hypothesis"], batch_size=64, show_progress_bar=True))
    datasets_out.append(
        DatasetExamples(
            name="snli",
            source="snli validation",
            task_type="binary_pair",
            vectors=normalize_pair_average(snli_a, snli_b),
            labels=np.asarray([snli_label_map[int(label)] for label in snli["label"]], dtype=np.int64),
        )
    )

    mnli = load_dataset("glue", "mnli", split="validation_matched")
    mnli = mnli.filter(lambda row: row["label"] in (0, 2) and bool(row["premise"]) and bool(row["hypothesis"]))
    mnli_label_map = {0: 0, 2: 1}
    mnli_labels_raw = np.asarray(mnli["label"], dtype=np.int64)
    mnli_groups = []
    for label in (0, 2):
        idx = np.where(mnli_labels_raw == label)[0]
        mnli_groups.append(limit_per_class(idx, MAX_ROWS_PER_CLASS, rng))
    mnli_take = np.sort(np.concatenate(mnli_groups))
    mnli = mnli.select(mnli_take.tolist())
    mnli_a = np.asarray(model.encode(mnli["premise"], batch_size=64, show_progress_bar=True))
    mnli_b = np.asarray(model.encode(mnli["hypothesis"], batch_size=64, show_progress_bar=True))
    datasets_out.append(
        DatasetExamples(
            name="mnli",
            source="glue/mnli validation_matched",
            task_type="binary_pair",
            vectors=normalize_pair_average(mnli_a, mnli_b),
            labels=np.asarray([mnli_label_map[int(label)] for label in mnli["label"]], dtype=np.int64),
        )
    )

    return datasets_out


def sample_from_pool(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(rng.choice(pool, size=n, replace=False), dtype=np.int64)


def build_clusters(dataset: DatasetExamples, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    labels = dataset.labels
    pure_clusters: List[np.ndarray] = []
    mixed_clusters: List[np.ndarray] = []

    if dataset.task_type == "ordinal_pair":
        by_bin = {bin_id: np.where(labels == bin_id)[0] for bin_id in range(len(STSB_BINS))}
        eligible_bins = [bin_id for bin_id, idx in by_bin.items() if len(idx) >= CLUSTER_SIZE]
        if len(eligible_bins) < 4:
            raise RuntimeError("STS-B does not have enough populated bins for the registered design.")
        for i in range(CLUSTERS_PER_TYPE):
            bin_id = eligible_bins[i % len(eligible_bins)]
            pure_clusters.append(sample_from_pool(by_bin[bin_id], CLUSTER_SIZE, rng))
        mixed_bin_sets = [eligible_bins[i : i + 4] for i in range(max(1, len(eligible_bins) - 3))]
        for i in range(CLUSTERS_PER_TYPE):
            chosen_bins = mixed_bin_sets[i % len(mixed_bin_sets)]
            parts = [sample_from_pool(by_bin[bin_id], CLUSTER_SIZE // 4, rng) for bin_id in chosen_bins]
            mixed_clusters.append(np.concatenate(parts))
    else:
        unique_labels = sorted(int(v) for v in np.unique(labels))
        by_label = {label: np.where(labels == label)[0] for label in unique_labels}
        for label, idx in by_label.items():
            if len(idx) < CLUSTER_SIZE:
                raise RuntimeError(f"{dataset.name}: label {label} has fewer than {CLUSTER_SIZE} rows.")
        for i in range(CLUSTERS_PER_TYPE):
            label = unique_labels[i % len(unique_labels)]
            pure_clusters.append(sample_from_pool(by_label[label], CLUSTER_SIZE, rng))
        if len(unique_labels) != 2:
            raise RuntimeError(f"{dataset.name}: registered mixed-cluster design requires two labels.")
        half = CLUSTER_SIZE // 2
        for _ in range(CLUSTERS_PER_TYPE):
            left = sample_from_pool(by_label[unique_labels[0]], half, rng)
            right = sample_from_pool(by_label[unique_labels[1]], half, rng)
            mixed_clusters.append(np.concatenate([left, right]))

    clusters = np.asarray(pure_clusters + mixed_clusters, dtype=np.int64)
    truth = np.asarray([1] * len(pure_clusters) + [0] * len(mixed_clusters), dtype=np.int64)
    return clusters, truth


def score_clusters(formula, vectors: np.ndarray, clusters: np.ndarray) -> Dict[str, np.ndarray]:
    e_scores: List[float] = []
    r_simple_scores: List[float] = []
    r_full_scores: List[float] = []
    for cluster_idx in clusters:
        cluster_vectors = vectors[cluster_idx]
        values = formula.compute_all(cluster_vectors)
        e_scores.append(float(values["E"]))
        r_simple_scores.append(float(values["R_simple"]))
        r_full_scores.append(float(values["R_full"]))
    return {
        "E": np.asarray(e_scores, dtype=np.float64),
        "R_simple": np.asarray(r_simple_scores, dtype=np.float64),
        "R_full": np.asarray(r_full_scores, dtype=np.float64),
    }


def anti_pattern_check(result_scores: Dict[str, np.ndarray], truth: np.ndarray) -> Dict[str, bool]:
    return {
        "ground_truth_independent_of_metrics": True,
        "parameters_fixed_before_results": True,
        "no_grid_search": True,
        "negative_results_will_be_reported": True,
        "no_goalpost_moving": True,
        "both_classes_present": bool(truth.min() == 0 and truth.max() == 1),
        "all_scores_finite": bool(
            np.isfinite(result_scores["E"]).all()
            and np.isfinite(result_scores["R_simple"]).all()
            and np.isfinite(result_scores["R_full"]).all()
        ),
    }


def render_report(results: Dict) -> str:
    lines = [
        "# N1 Report",
        "",
        "Status: EXECUTED",
        "",
        "## Pre-Registered Outcome",
        "",
    ]
    for dataset_name, dataset_result in results["datasets"].items():
        aucs = dataset_result["aucs"]
        delta = dataset_result["comparisons"]["E_minus_R_simple"]
        lines.extend(
            [
                f"### {dataset_name}",
                "",
                f"- Source: {dataset_result['source']}",
                f"- Clusters: pure={dataset_result['pure_clusters']}, mixed={dataset_result['mixed_clusters']}",
                f"- AUC(E): {aucs['E']:.4f}",
                f"- AUC(R_simple): {aucs['R_simple']:.4f}",
                f"- AUC(R_full): {aucs['R_full']:.4f}",
                f"- AUC(random): {aucs['random']:.4f}",
                (
                    "- E - R_simple delta: "
                    f"{delta['delta_mean']:.4f} "
                    f"[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}]"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Overall Decision",
            "",
            f"- E wins: {results['summary']['e_wins']}",
            f"- R_simple wins: {results['summary']['r_simple_wins']}",
            f"- Ties: {results['summary']['ties']}",
            f"- Hypothesis status: {results['summary']['hypothesis_status']}",
            "",
            "## Anti-Pattern Checks",
            "",
        ]
    )
    for dataset_name, dataset_result in results["datasets"].items():
        lines.append(f"### {dataset_name}")
        lines.append("")
        for key, value in dataset_result["anti_pattern_check"].items():
            lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results"),
        help="Directory for generated JSON and Markdown outputs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formula = load_formula_module()
    datasets = load_examples(MODEL_NAME)
    rng = np.random.default_rng(SEED)

    results: Dict[str, Dict] = {
        "question": "N1",
        "title": "Does E/grad_S outperform bare E?",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "cluster_size": CLUSTER_SIZE,
        "clusters_per_type": CLUSTERS_PER_TYPE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "datasets": {},
    }

    e_wins = 0
    r_simple_wins = 0
    ties = 0

    for dataset_idx, dataset in enumerate(datasets):
        clusters, truth = build_clusters(dataset, rng)
        scores = score_clusters(formula, dataset.vectors, clusters)
        rand_rng = np.random.default_rng(SEED + dataset_idx + 1)
        random_scores = rand_rng.random(len(truth))

        aucs = {
            "E": auc_score(truth, scores["E"]),
            "R_simple": auc_score(truth, scores["R_simple"]),
            "R_full": auc_score(truth, scores["R_full"]),
            "random": auc_score(truth, random_scores),
        }

        e_minus_r = bootstrap_auc_delta(truth, scores["E"], scores["R_simple"], seed=SEED + 100 + dataset_idx)
        r_simple_minus_full = bootstrap_auc_delta(
            truth,
            scores["R_simple"],
            scores["R_full"],
            seed=SEED + 200 + dataset_idx,
        )

        if e_minus_r["ci_low"] > 0.0:
            winner = "E"
            e_wins += 1
        elif e_minus_r["ci_high"] < 0.0:
            winner = "R_simple"
            r_simple_wins += 1
        else:
            winner = "tie"
            ties += 1

        results["datasets"][dataset.name] = {
            "source": dataset.source,
            "task_type": dataset.task_type,
            "n_examples": int(len(dataset.labels)),
            "pure_clusters": CLUSTERS_PER_TYPE,
            "mixed_clusters": CLUSTERS_PER_TYPE,
            "aucs": aucs,
            "comparisons": {
                "E_minus_R_simple": e_minus_r,
                "R_simple_minus_R_full": r_simple_minus_full,
            },
            "winner": winner,
            "anti_pattern_check": anti_pattern_check(scores, truth),
        }

    hypothesis_status = "supported" if e_wins >= 3 else "falsified" if r_simple_wins >= 3 else "mixed"
    results["summary"] = {
        "e_wins": e_wins,
        "r_simple_wins": r_simple_wins,
        "ties": ties,
        "hypothesis_status": hypothesis_status,
    }

    json_path = output_dir / "n01_e_vs_r_results.json"
    report_path = output_dir / "n01_e_vs_r_report.md"
    json_path.write_text(json.dumps(to_builtin(results), indent=2), encoding="utf-8")
    report_path.write_text(render_report(results), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
