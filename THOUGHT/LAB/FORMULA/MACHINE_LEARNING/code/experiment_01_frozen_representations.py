from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

from formula_ml import compute_formula_metrics


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sst2", "ag_news"], default="sst2")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument("--clusters-per-class", type=int, default=40)
    parser.add_argument("--limit-per-label", type=int, default=200)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def load_labeled_texts(dataset_name: str, limit_per_label: int) -> Dict[int, List[str]]:
    if dataset_name == "sst2":
        ds = load_dataset("glue", "sst2", split="validation")
        text_key = "sentence"
        label_key = "label"
    elif dataset_name == "ag_news":
        ds = load_dataset("ag_news", split="test")
        text_key = "text"
        label_key = "label"
    else:
        raise ValueError(dataset_name)

    grouped: Dict[int, List[str]] = defaultdict(list)
    for row in ds:
        label = int(row[label_key])
        if len(grouped[label]) < limit_per_label:
            grouped[label].append(str(row[text_key]))
        if all(len(v) >= limit_per_label for v in grouped.values()) and grouped:
            pass

    return {k: v for k, v in grouped.items() if len(v) >= limit_per_label}


def build_clusters(
    grouped_texts: Dict[int, List[str]],
    cluster_size: int,
    clusters_per_class: int,
    rng: random.Random,
) -> List[Tuple[str, int, List[str]]]:
    labels = sorted(grouped_texts)
    clusters: List[Tuple[str, int, List[str]]] = []

    for label in labels:
        texts = grouped_texts[label]
        for _ in range(clusters_per_class):
            cluster = rng.sample(texts, cluster_size)
            clusters.append(("pure", label, cluster))

    if len(labels) < 2:
        raise ValueError("need at least two labels")

    half = cluster_size // 2
    if cluster_size % 2 != 0:
        raise ValueError("cluster_size must be even for mixed clusters")

    for label in labels:
        other_labels = [x for x in labels if x != label]
        for _ in range(clusters_per_class):
            other = rng.choice(other_labels)
            cluster = rng.sample(grouped_texts[label], half) + rng.sample(grouped_texts[other], half)
            rng.shuffle(cluster)
            clusters.append(("mixed", label, cluster))

    return clusters


def embed_unique_texts(model_name: str, device: str, texts: Sequence[str]) -> Dict[str, np.ndarray]:
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    return {text: emb for text, emb in zip(texts, embeddings)}


def evaluate_clusters(
    clusters: Sequence[Tuple[str, int, List[str]]],
    embedding_lookup: Dict[str, np.ndarray],
) -> Dict[str, object]:
    rows = []
    y_true = []
    metric_names = None

    for cluster_type, label, texts in clusters:
        stack = np.vstack([embedding_lookup[t] for t in texts])
        metrics = compute_formula_metrics(stack).as_dict()
        if metric_names is None:
            metric_names = list(metrics.keys())
        row = {
            "cluster_type": cluster_type,
            "label": label,
            "size": len(texts),
            **metrics,
        }
        rows.append(row)
        y_true.append(1 if cluster_type == "pure" else 0)

    assert metric_names is not None
    aucs = {}
    summaries = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        sane = [0.0 if not np.isfinite(v) else v for v in values]
        aucs[name] = float(roc_auc_score(y_true, sane))

        pure_vals = [float(r[name]) for r in rows if r["cluster_type"] == "pure" and np.isfinite(r[name])]
        mixed_vals = [float(r[name]) for r in rows if r["cluster_type"] == "mixed" and np.isfinite(r[name])]
        summaries[name] = {
            "pure_mean": float(mean(pure_vals)) if pure_vals else float("nan"),
            "pure_std": float(pstdev(pure_vals)) if len(pure_vals) > 1 else 0.0,
            "mixed_mean": float(mean(mixed_vals)) if mixed_vals else float("nan"),
            "mixed_std": float(pstdev(mixed_vals)) if len(mixed_vals) > 1 else 0.0,
        }

    ranked = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "rows": rows,
        "aucs": aucs,
        "ranked_metrics": ranked,
        "summaries": summaries,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    grouped = load_labeled_texts(args.dataset, args.limit_per_label)
    if len(grouped) < 2:
        raise RuntimeError(f"dataset {args.dataset} did not produce enough labels")

    clusters = build_clusters(grouped, args.cluster_size, args.clusters_per_class, rng)
    unique_texts = sorted({text for _, _, texts in clusters for text in texts})
    embedding_lookup = embed_unique_texts(args.model, args.device, unique_texts)
    results = evaluate_clusters(clusters, embedding_lookup)

    payload = {
        "experiment": "EXPERIMENT_01_FROZEN_REPRESENTATIONS",
        "dataset": args.dataset,
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "cluster_size": args.cluster_size,
        "clusters_per_class": args.clusters_per_class,
        "limit_per_label": args.limit_per_label,
        "n_unique_texts": len(unique_texts),
        "n_clusters": len(clusters),
        **results,
    }

    output_path = Path(args.output) if args.output else RESULTS_DIR / f"experiment_01_{args.dataset}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved: {output_path}")
    print("metric ranking by ROC AUC:")
    for name, auc in payload["ranked_metrics"]:
        print(f"  {name}: {auc:.4f}")


if __name__ == "__main__":
    main()
