#!/usr/bin/env python3
"""
Exploratory follow-up for N1.

Purpose:
- use label-purity datasets where semantic coherence is directly tied to labels
- characterize where R_simple helps or hurts relative to E
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


SEED = 20260309
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_SIZE = 16
PER_CLASS_LIMIT = 800
BOOTSTRAP_SAMPLES = 1000
CLUSTERS_PER_RECIPE = 40


@dataclass
class DatasetSpec:
    name: str
    source: str
    texts: List[str]
    labels: np.ndarray


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos_scores = scores[y_true == 1][:, None]
    neg_scores = scores[y_true == 0][None, :]
    wins = float((pos_scores > neg_scores).sum())
    ties = float((pos_scores == neg_scores).sum())
    return (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores.T))


def bootstrap_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    target: np.ndarray,
    mode: str,
    seed: int,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    n = len(target)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_target = target[idx]
        sample_a = values_a[idx]
        sample_b = values_b[idx]
        if mode == "spearman":
            delta = spearmanr(sample_a, sample_target).statistic - spearmanr(sample_b, sample_target).statistic
        elif mode == "auc":
            if sample_target.min() == sample_target.max():
                continue
            delta = auc_score(sample_target, sample_a) - auc_score(sample_target, sample_b)
        else:
            raise ValueError(mode)
        deltas.append(float(delta))
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "delta_mean": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "wins_fraction": float(np.mean(arr > 0.0)),
    }


def load_formula_module():
    formula_path = Path(__file__).resolve().parents[4] / "v2" / "shared" / "formula.py"
    spec = importlib.util.spec_from_file_location("formula_v2_shared", formula_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load formula module from {formula_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_datasets() -> List[DatasetSpec]:
    from datasets import load_dataset

    rng = np.random.default_rng(SEED)
    datasets_out: List[DatasetSpec] = []

    ag = load_dataset("ag_news", split="test")
    datasets_out.append(make_dataset_spec("ag_news", "ag_news test", ag["text"], np.asarray(ag["label"]), rng))

    emotion = load_dataset("dair-ai/emotion", split="test")
    datasets_out.append(
        make_dataset_spec("emotion", "dair-ai/emotion test", emotion["text"], np.asarray(emotion["label"]), rng)
    )

    sst2 = load_dataset("glue", "sst2", split="validation")
    datasets_out.append(make_dataset_spec("sst2", "glue/sst2 validation", sst2["sentence"], np.asarray(sst2["label"]), rng))

    snli = load_dataset("snli", split="validation")
    snli = snli.filter(lambda row: row["label"] in (0, 2) and bool(row["premise"]) and bool(row["hypothesis"]))
    snli_texts = [f"{p} [SEP] {h}" for p, h in zip(snli["premise"], snli["hypothesis"])]
    snli_labels = np.asarray([0 if int(label) == 0 else 1 for label in snli["label"]], dtype=np.int64)
    datasets_out.append(make_dataset_spec("snli", "snli validation", snli_texts, snli_labels, rng))

    mnli = load_dataset("glue", "mnli", split="validation_matched")
    mnli = mnli.filter(lambda row: row["label"] in (0, 2) and bool(row["premise"]) and bool(row["hypothesis"]))
    mnli_texts = [f"{p} [SEP] {h}" for p, h in zip(mnli["premise"], mnli["hypothesis"])]
    mnli_labels = np.asarray([0 if int(label) == 0 else 1 for label in mnli["label"]], dtype=np.int64)
    datasets_out.append(make_dataset_spec("mnli", "glue/mnli validation_matched", mnli_texts, mnli_labels, rng))

    return datasets_out


def make_dataset_spec(
    name: str,
    source: str,
    texts: Sequence[str],
    labels: np.ndarray,
    rng: np.random.Generator,
) -> DatasetSpec:
    texts = list(texts)
    keep_indices: List[int] = []
    for label in sorted(int(v) for v in np.unique(labels)):
        idx = np.where(labels == label)[0]
        if len(idx) > PER_CLASS_LIMIT:
            idx = np.sort(rng.choice(idx, size=PER_CLASS_LIMIT, replace=False))
        keep_indices.extend(idx.tolist())
    keep = np.asarray(sorted(keep_indices), dtype=np.int64)
    return DatasetSpec(
        name=name,
        source=source,
        texts=[texts[i] for i in keep],
        labels=labels[keep],
    )


def embed_datasets(specs: List[DatasetSpec]) -> Dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    embedded: Dict[str, np.ndarray] = {}
    for spec in specs:
        embedded[spec.name] = np.asarray(model.encode(spec.texts, batch_size=64, show_progress_bar=True))
    return embedded


def sample(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(rng.choice(pool, size=n, replace=False), dtype=np.int64)


def cluster_recipes(unique_labels: Sequence[int]) -> List[Tuple[str, List[Tuple[int, int]]]]:
    if len(unique_labels) == 2:
        return [
            ("pure", [(unique_labels[0], 16)]),
            ("pure", [(unique_labels[1], 16)]),
            ("skewed", [(unique_labels[0], 12), (unique_labels[1], 4)]),
            ("skewed", [(unique_labels[1], 12), (unique_labels[0], 4)]),
            ("balanced", [(unique_labels[0], 8), (unique_labels[1], 8)]),
        ]
    return [
        ("pure", []),
        ("skewed", []),
        ("balanced2", []),
        ("balanced4", []),
    ]


def build_clusters(spec: DatasetSpec, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = spec.labels
    unique_labels = sorted(int(v) for v in np.unique(labels))
    by_label = {label: np.where(labels == label)[0] for label in unique_labels}
    for label, idx in by_label.items():
        if len(idx) < 16:
            raise RuntimeError(f"{spec.name}: label {label} has only {len(idx)} rows")

    clusters: List[np.ndarray] = []
    purity: List[float] = []
    pure_binary: List[int] = []

    if len(unique_labels) == 2:
        recipes = cluster_recipes(unique_labels)
        for recipe_name, parts in recipes:
            for _ in range(CLUSTERS_PER_RECIPE):
                if recipe_name == "pure":
                    label, count = parts[0]
                    cluster = sample(by_label[label], count, rng)
                    cluster_labels = np.full(count, label, dtype=np.int64)
                else:
                    pieces = [sample(by_label[label], count, rng) for label, count in parts]
                    cluster = np.concatenate(pieces)
                    cluster_labels = np.concatenate(
                        [np.full(count, label, dtype=np.int64) for label, count in parts]
                    )
                clusters.append(cluster)
                purity.append(float(np.bincount(cluster_labels).max() / CLUSTER_SIZE))
                pure_binary.append(int(recipe_name == "pure"))
    else:
        for _ in range(CLUSTERS_PER_RECIPE):
            label = unique_labels[_ % len(unique_labels)]
            cluster = sample(by_label[label], CLUSTER_SIZE, rng)
            clusters.append(cluster)
            purity.append(1.0)
            pure_binary.append(1)
        for _ in range(CLUSTERS_PER_RECIPE):
            chosen = rng.choice(unique_labels, size=2, replace=False)
            parts = [sample(by_label[int(chosen[0])], 12, rng), sample(by_label[int(chosen[1])], 4, rng)]
            clusters.append(np.concatenate(parts))
            purity.append(12 / 16)
            pure_binary.append(0)
        for _ in range(CLUSTERS_PER_RECIPE):
            chosen = rng.choice(unique_labels, size=2, replace=False)
            parts = [sample(by_label[int(chosen[0])], 8, rng), sample(by_label[int(chosen[1])], 8, rng)]
            clusters.append(np.concatenate(parts))
            purity.append(8 / 16)
            pure_binary.append(0)
        for _ in range(CLUSTERS_PER_RECIPE):
            chosen = rng.choice(unique_labels, size=4, replace=False)
            parts = [sample(by_label[int(label)], 4, rng) for label in chosen]
            clusters.append(np.concatenate(parts))
            purity.append(4 / 16)
            pure_binary.append(0)

    return (
        np.asarray(clusters, dtype=np.int64),
        np.asarray(purity, dtype=np.float64),
        np.asarray(pure_binary, dtype=np.int64),
    )


def score_clusters(formula, vectors: np.ndarray, clusters: np.ndarray) -> Dict[str, np.ndarray]:
    e_values: List[float] = []
    r_values: List[float] = []
    rf_values: List[float] = []
    gs_values: List[float] = []
    for cluster in clusters:
        result = formula.compute_all(vectors[cluster])
        e_values.append(float(result["E"]))
        r_values.append(float(result["R_simple"]))
        rf_values.append(float(result["R_full"]))
        gs_values.append(float(result["grad_S"]))
    return {
        "E": np.asarray(e_values),
        "R_simple": np.asarray(r_values),
        "R_full": np.asarray(rf_values),
        "grad_S": np.asarray(gs_values),
    }


def choose_winner(delta: Dict[str, float]) -> str:
    if delta["ci_low"] > 0.0:
        return "E"
    if delta["ci_high"] < 0.0:
        return "R_simple"
    return "tie"


def main() -> None:
    formula = load_formula_module()
    specs = load_datasets()
    embedded = embed_datasets(specs)
    rng = np.random.default_rng(SEED)

    results: Dict[str, object] = {
        "question": "N1 exploratory follow-up",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "cluster_size": CLUSTER_SIZE,
        "clusters_per_recipe": CLUSTERS_PER_RECIPE,
        "datasets": {},
    }

    for dataset_idx, spec in enumerate(specs):
        clusters, purity, pure_binary = build_clusters(spec, rng)
        scores = score_clusters(formula, embedded[spec.name], clusters)

        rho_e = float(spearmanr(scores["E"], purity).statistic)
        rho_r = float(spearmanr(scores["R_simple"], purity).statistic)
        rho_rf = float(spearmanr(scores["R_full"], purity).statistic)

        auc_e = float(auc_score(pure_binary, scores["E"]))
        auc_r = float(auc_score(pure_binary, scores["R_simple"]))
        auc_rf = float(auc_score(pure_binary, scores["R_full"]))

        delta_rho = bootstrap_delta(scores["E"], scores["R_simple"], purity, "spearman", SEED + 10 + dataset_idx)
        delta_auc = bootstrap_delta(scores["E"], scores["R_simple"], pure_binary, "auc", SEED + 20 + dataset_idx)

        results["datasets"][spec.name] = {
            "source": spec.source,
            "n_examples": int(len(spec.labels)),
            "n_labels": int(len(np.unique(spec.labels))),
            "spearman": {
                "E": rho_e,
                "R_simple": rho_r,
                "R_full": rho_rf,
            },
            "auc_pure_vs_not_pure": {
                "E": auc_e,
                "R_simple": auc_r,
                "R_full": auc_rf,
            },
            "delta": {
                "spearman_E_minus_R_simple": delta_rho,
                "auc_E_minus_R_simple": delta_auc,
            },
            "winner_spearman": choose_winner(delta_rho),
            "winner_auc": choose_winner(delta_auc),
            "grad_S_mean": float(np.mean(scores["grad_S"])),
            "purity_levels": sorted(set(float(v) for v in purity)),
        }

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "n01_label_purity_followup.json"
    json_path.write_text(json.dumps(to_builtin(results), indent=2), encoding="utf-8")

    lines = [
        "# N1 Exploratory Label-Purity Follow-Up",
        "",
        f"Model: `{MODEL_NAME}`",
        "",
    ]
    for name, payload in results["datasets"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Source: {payload['source']}",
                f"- Spearman: E={payload['spearman']['E']:.4f}, R_simple={payload['spearman']['R_simple']:.4f}, R_full={payload['spearman']['R_full']:.4f}",
                f"- AUC pure-vs-not: E={payload['auc_pure_vs_not_pure']['E']:.4f}, R_simple={payload['auc_pure_vs_not_pure']['R_simple']:.4f}, R_full={payload['auc_pure_vs_not_pure']['R_full']:.4f}",
                (
                    f"- Delta Spearman E-R_simple: {payload['delta']['spearman_E_minus_R_simple']['delta_mean']:.4f} "
                    f"[{payload['delta']['spearman_E_minus_R_simple']['ci_low']:.4f}, {payload['delta']['spearman_E_minus_R_simple']['ci_high']:.4f}]"
                ),
                (
                    f"- Delta AUC E-R_simple: {payload['delta']['auc_E_minus_R_simple']['delta_mean']:.4f} "
                    f"[{payload['delta']['auc_E_minus_R_simple']['ci_low']:.4f}, {payload['delta']['auc_E_minus_R_simple']['ci_high']:.4f}]"
                ),
                f"- Winner by Spearman: {payload['winner_spearman']}",
                f"- Winner by AUC: {payload['winner_auc']}",
                "",
            ]
        )
    (output_dir / "n01_label_purity_followup.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {output_dir / 'n01_label_purity_followup.md'}")


if __name__ == "__main__":
    main()
