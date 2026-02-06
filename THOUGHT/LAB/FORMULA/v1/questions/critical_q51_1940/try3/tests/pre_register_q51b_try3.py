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
    "random_seed": 1337,
    "num_bases": 50,
    "batch_size": 64,
    "normalize_embeddings": False,
    "train_split_ratio": 0.8,
    "epsilon": 1e-9,
    "phase_error_threshold_radians": math.pi / 4,
    "phase_pass_min_bases": 50,
    "lambda_ratio_threshold": 0.8,
    "lambda_pass_fraction_min": 0.6,
    "loop_word_count": 3,
    "loops_per_category_max": 3,
    "holonomy_diff_threshold_radians": 0.2,
    "holonomy_pass_fraction_min": 0.6,
    "distortion": {
        "epsilon": 0.01,
        "tanh": True,
    },
    "probe": {
        "ridge": 1e-6,
        "complex_better_ratio": 0.9,
        "complex_better_fraction_min": 0.6,
        "phase_baseline": "predict cos/sin then normalize",
    },
    "metrics": [
        "exp1_median_phase_error",
        "exp1_pass_bases",
        "exp2_median_lambda_diff",
        "exp2_baseline_median",
        "exp2_ratio",
        "exp3_median_gamma_diff_sign",
        "exp3_median_gamma_diff_tanh",
        "exp4_complex_probe_median_error",
        "exp4_phase_probe_median_error",
    ],
    "models": MODEL_LIST,
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = RESULTS_DIR / "q51b_try3_config.json"
    _write_json(config_path, CONFIG)

    pre_reg_lines = [
        "# Pre-Registration: Q51b Complex-Linearity Stress Test (Try3)",
        "",
        f"Date: {RUN_DATE}",
        "",
        "## HYPOTHESIS",
        "Complex-linear structure is not intrinsic; phase arithmetic and winding effects will weaken under random bases and nonlinear distortions.",
        "",
        "## PREDICTION",
        "Exp1: Median phase error will exceed π/4 in many random bases.",
        "Exp2: |λ_ab−λ_cd| will not be consistently smaller than random baselines.",
        "Exp3: Winding γ will change materially under sign/tanh distortions.",
        "Exp4: Complex probe will not consistently outperform phase-only probe.",
        "",
        "## FALSIFICATION",
        "If Exp1 passes on ≥50 bases, Exp2 beats baseline on ≥60% of bases, Exp3 survives distortions on ≥60% of bases,",
        "and Exp4 complex probe beats phase probe on ≥60% of bases, then intrinsic complex structure is supported.",
        "",
        "## DATA SOURCE",
        f"- {DATASET_URL}",
        "",
        "## SUCCESS THRESHOLD",
        f"- Exp1: median phase error < π/4 on ≥{CONFIG['phase_pass_min_bases']} bases",
        f"- Exp2: median |λ_ab−λ_cd| ratio < {CONFIG['lambda_ratio_threshold']} on ≥{int(CONFIG['lambda_pass_fraction_min'] * 100)}% of bases",
        f"- Exp3: median |Δγ| < {CONFIG['holonomy_diff_threshold_radians']} on ≥{int(CONFIG['holonomy_pass_fraction_min'] * 100)}% of bases",
        f"- Exp4: complex probe error ratio < {CONFIG['probe']['complex_better_ratio']} on ≥{int(CONFIG['probe']['complex_better_fraction_min'] * 100)}% of bases",
        "",
        "## FIXED PARAMETERS",
        f"- Random seed: {CONFIG['random_seed']}",
        f"- Random bases: {CONFIG['num_bases']}",
        f"- Batch size: {CONFIG['batch_size']}",
        f"- Normalize embeddings: {CONFIG['normalize_embeddings']}",
        f"- Train split ratio: {CONFIG['train_split_ratio']}",
        f"- Loop size: {CONFIG['loop_word_count']}",
        f"- Loops per category max: {CONFIG['loops_per_category_max']}",
        f"- Distortion epsilon: {CONFIG['distortion']['epsilon']}",
        f"- Probe ridge: {CONFIG['probe']['ridge']}",
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
            "- Random bases fixed by seed",
            "",
            "## Notes",
            "- No PCA or data-dependent bases are used.",
            "- All results (pass and fail) will be reported.",
        ]
    )

    (REPORTS_DIR / "q51b_try3_prereg.md").write_text("\n".join(pre_reg_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
