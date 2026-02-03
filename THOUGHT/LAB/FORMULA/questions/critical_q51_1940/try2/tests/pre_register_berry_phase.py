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
    "loop_word_count": 4,
    "loops_per_category_max": 3,
    "pca_components": 2,
    "rotation_angles": [0.0, math.pi / 8, math.pi / 4, math.pi / 2],
    "batch_size": 64,
    "normalize_embeddings": False,
    "epsilon": 1e-9,
    "models": MODEL_LIST,
    "metrics": [
        "gamma",
        "gamma_mod_2pi",
        "quant_frac_1_8",
        "quant_score_1_8",
        "quant_score_1",
        "invariance_diff_global",
        "invariance_diff_local",
    ],
    "success_thresholds": {
        "invariance_mean_abs_diff_lt": 0.2,
        "quant_score_gt": 0.8,
    },
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = RESULTS_DIR / "q51_berry_phase_try2_config.json"
    _write_json(config_path, CONFIG)

    pre_reg_lines = [
        "# Pre-Registration: Q51 Berry Phase / Holonomy Correctness (Try2)",
        "",
        f"Date: {RUN_DATE}",
        "",
        "## HYPOTHESIS",
        "Gamma computed as sum of angle(z_{i+1}/z_i) is not invariant across projection choices (local vs global PCA); quantization may be projection-induced.",
        "",
        "## PREDICTION",
        "Gamma values will vary across projection bases and reflections beyond a small tolerance, and quantization will not be stable.",
        "",
        "## FALSIFICATION",
        "If gamma is invariant (mean |Δγ| < 0.2 rad) across projection choices (local vs global PCA) and reflections, and quantization scores remain high,",
        "that would support a valid topological interpretation.",
        "",
        "## DATA SOURCE",
        f"- {DATASET_URL}",
        "",
        "## SUCCESS THRESHOLD (HOLONOMY EVIDENCE)",
        "- mean |Δγ| across bases < 0.2 rad",
        "- quant_score_1_8 > 0.8 across bases",
        "",
        "## FIXED PARAMETERS",
        f"- Loop size: {CONFIG['loop_word_count']} words",
        f"- Loops per category max: {CONFIG['loops_per_category_max']}",
        f"- PCA components: {CONFIG['pca_components']}",
        f"- Rotation angles: {CONFIG['rotation_angles']}",
        f"- Batch size: {CONFIG['batch_size']}",
        f"- Normalize embeddings: {CONFIG['normalize_embeddings']}",
        "",
        "## MODEL LIST (Fixed, No Substitutions)",
    ]
    pre_reg_lines.extend([f"- {name}" for name in MODEL_LIST])
    pre_reg_lines.extend(
        [
            "",
            "## Anti-Patterns Guardrail",
            "- No synthetic data generation",
            "- No parameter search or post-hoc thresholds",
            "- Loop construction is deterministic from external dataset order",
            "",
            "## Notes",
            "- Loops are formed by grouping consecutive words in each category.",
            "- All results (pass and fail) will be reported.",
        ]
    )

    (REPORTS_DIR / "q51_berry_phase_try2_prereg.md").write_text(
        "\n".join(pre_reg_lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
