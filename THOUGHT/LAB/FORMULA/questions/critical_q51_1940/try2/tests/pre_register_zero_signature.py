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
    "pca_components": 3,
    "octant_sign_rule": "sign = +1 if component >= 0 else -1",
    "octant_index_rule": "k = (pc1>=0) + 2*(pc2>=0) + 4*(pc3>=0)",
    "phase_mapping": "theta_k = (k + 0.5) * pi/4",
    "batch_size": 64,
    "normalize_embeddings": False,
    "models": MODEL_LIST,
    "success_thresholds": {
        "s_over_n_lt": 0.05,
        "chi_square_p_gt": 0.05,
        "max_pair_diff_lt": 0.02,
    },
    "metrics": [
        "|S|/n",
        "chi_square_p_uniform",
        "max_pair_diff",
        "df_tones_m2",
        "df_tones_m3",
        "random_null_expected_s_over_n",
    ],
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = RESULTS_DIR / "q51_zero_signature_try2_config.json"
    _write_json(config_path, CONFIG)

    pre_reg_lines = [
        "# Pre-Registration: Q51 Zero Signature Interpretation (Try2)",
        "",
        f"Date: {RUN_DATE}",
        "",
        "## HYPOTHESIS",
        "|S|/n near zero does NOT uniquely imply 8th roots-of-unity structure; it is compatible with weaker cancellation.",
        "",
        "## PREDICTION",
        "Across models, |S|/n will be small (< 0.05), but uniformity tests and higher harmonics will not uniquely select roots-of-unity.",
        "",
        "## FALSIFICATION",
        "If octant distribution is both uniform (chi-square p > 0.05) AND all low-order harmonics are near zero,",
        "that would be stronger evidence consistent with roots-of-unity structure.",
        "",
        "## DATA SOURCE",
        f"- {DATASET_URL}",
        "",
        "## SUCCESS THRESHOLD (ROOTS-OF-UNITY EVIDENCE)",
        "- |S|/n < 0.05",
        "- chi-square p > 0.05 (uniform octant distribution)",
        "- max opposite-pair diff < 0.02",
        "",
        "## FIXED PARAMETERS",
        f"- PCA components: {CONFIG['pca_components']}",
        f"- Sign rule: {CONFIG['octant_sign_rule']}",
        f"- Octant index rule: {CONFIG['octant_index_rule']}",
        f"- Phase mapping: {CONFIG['phase_mapping']}",
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
            "- Ground truth independent of |S|/n",
            "",
            "## Notes",
            "- This test is adversarial and assumes the strongest null hypothesis.",
            "- All results (pass and fail) will be reported.",
        ]
    )

    (REPORTS_DIR / "q51_zero_signature_try2_prereg.md").write_text(
        "\n".join(pre_reg_lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
