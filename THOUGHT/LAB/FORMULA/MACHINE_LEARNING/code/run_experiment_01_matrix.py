from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from experiment_01_frozen_representations import run_experiment


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2", "ag_news"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument("--clusters-per-class", type=int, default=8)
    parser.add_argument("--limit-per-label", type=int, default=80)
    parser.add_argument("--output", default=str(RESULTS_DIR / "experiment_01_matrix_summary.json"))
    return parser.parse_args()


def aggregate(runs: List[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[float]] = {}
    per_condition: Dict[str, Dict[str, object]] = {}

    for run in runs:
        key = f"{run['dataset']} | {run['model']}"
        aucs = run["aucs"]
        assert isinstance(aucs, dict)
        if key not in per_condition:
            per_condition[key] = {"runs": []}
        per_condition[key]["runs"].append(
            {
                "seed": run["seed"],
                "aucs": aucs,
            }
        )
        for metric, value in aucs.items():
            grouped.setdefault(metric, []).append(float(value))

    for key, value in per_condition.items():
        run_list = value["runs"]
        first_metrics = run_list[0]["aucs"].keys()
        metric_means = {
            metric: float(mean([float(r["aucs"][metric]) for r in run_list]))
            for metric in first_metrics
        }
        metric_ranking = sorted(metric_means.items(), key=lambda kv: kv[1], reverse=True)
        value["metric_means"] = metric_means
        value["metric_ranking"] = metric_ranking

    overall_metric_means = {
        metric: float(mean(values))
        for metric, values in grouped.items()
    }
    overall_ranking = sorted(overall_metric_means.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "overall_metric_means": overall_metric_means,
        "overall_ranking": overall_ranking,
        "per_condition": per_condition,
    }


def main() -> None:
    args = parse_args()
    runs: List[Dict[str, object]] = []

    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                print(f"running dataset={dataset} model={model} seed={seed}")
                result = run_experiment(
                    dataset=dataset,
                    model=model,
                    device=args.device,
                    seed=seed,
                    cluster_size=args.cluster_size,
                    clusters_per_class=args.clusters_per_class,
                    limit_per_label=args.limit_per_label,
                )
                runs.append(result)

    payload = {
        "experiment": "EXPERIMENT_01_MATRIX",
        "datasets": args.datasets,
        "models": args.models,
        "device": args.device,
        "seeds": args.seeds,
        "cluster_size": args.cluster_size,
        "clusters_per_class": args.clusters_per_class,
        "limit_per_label": args.limit_per_label,
        "runs": runs,
        "summary": aggregate(runs),
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {output_path}")
    print("overall ranking:")
    for metric, score in payload["summary"]["overall_ranking"]:
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()
