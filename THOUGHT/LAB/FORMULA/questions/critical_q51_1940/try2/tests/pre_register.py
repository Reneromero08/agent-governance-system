from __future__ import annotations

from pathlib import Path
import json
import math


OUTPUT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"
RUN_DATE = "2026-02-03"

DATASET_URL = "https://download.tensorflow.org/data/questions-words.txt"
DATASET_PATH = "results/dataset/questions-words.txt"

MODEL_LIST = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "all-roberta-large-v1",
    "paraphrase-MiniLM-L6-v2",
    "paraphrase-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "multi-qa-mpnet-base-dot-v1",
    "nli-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-small",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
]

CONFIG = {
    "run_date": RUN_DATE,
    "dataset_url": DATASET_URL,
    "dataset_path": DATASET_PATH,
    "split_seed": 1337,
    "train_split_ratio": 0.8,
    "pca_components": 2,
    "phase_definition": "theta = atan2(PC2, PC1)",
    "phase_error_threshold_radians": math.pi / 4,
    "batch_size": 64,
    "normalize_embeddings": False,
    "models": MODEL_LIST,
    "success_thresholds": {
        "pass_rate_gt": 0.60,
        "mean_error_lt_radians": math.pi / 4,
        "overall_models_pass_fraction": 0.60,
    },
    "dependencies": {
        "sentence-transformers": "2.7.0",
        "scikit-learn": "1.4.2",
        "numpy": "1.26.4",
    },
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = RESULTS_DIR / "config.json"
    _write_json(config_path, CONFIG)

    pre_reg_lines = [
        "# Pre-Registration: Q51 Phase Arithmetic Validity (Try2)",
        "",
        f"Date: {RUN_DATE}",
        "",
        "## Hypothesis",
        "Phase-difference consistency holds for external analogy data when phases are computed",
        "from a global PCA fit on a training split of the analogy vocabulary.",
        "",
        "## Prediction",
        "For each evaluated embedding model, mean absolute phase error < pi/4 and pass rate > 0.60.",
        "",
        "## Falsification",
        "Mean absolute phase error >= pi/4 OR pass rate <= 0.60 for a model.",
        "",
        "## Data Source",
        f"- {DATASET_URL}",
        "",
        "## Success Threshold",
        "- pass_rate > 0.60 AND mean_error < pi/4 (0.785398...) per model",
        "- overall: at least 60% of successfully evaluated models meet the threshold",
        "",
        "## Fixed Parameters",
        f"- PCA components: {CONFIG['pca_components']}",
        f"- Phase definition: {CONFIG['phase_definition']}",
        f"- Phase error threshold: {CONFIG['phase_error_threshold_radians']:.6f} radians",
        f"- Train/test split ratio: {CONFIG['train_split_ratio']}",
        f"- Split seed: {CONFIG['split_seed']}",
        f"- Batch size: {CONFIG['batch_size']}",
        f"- Normalize embeddings: {CONFIG['normalize_embeddings']}",
        "",
        "## Model List (Fixed, No Substitutions)",
    ]
    pre_reg_lines.extend([f"- {name}" for name in MODEL_LIST])
    pre_reg_lines.extend(
        [
            "",
            "## Anti-Patterns Guardrail",
            "- No synthetic data generation",
            "- No parameter search or post-hoc threshold changes",
            "- Ground truth is independent of phase metrics",
            "",
            "## Notes",
            "- PCA is fit only on the training split vocabulary to preserve train/test separation.",
            "- All results (pass and fail) will be reported.",
        ]
    )

    (REPORTS_DIR / "pre_registration.md").write_text("\n".join(pre_reg_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
