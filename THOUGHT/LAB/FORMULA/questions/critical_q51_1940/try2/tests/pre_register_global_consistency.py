from __future__ import annotations

from pathlib import Path
import json


OUTPUT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"
RUN_DATE = "2026-02-03"

INPUT_FILES = {
    "phase_arithmetic_results": "results/results.json",
    "zero_signature_results": "results/q51_zero_signature_try2_results.json",
    "berry_phase_results": "results/q51_berry_phase_try2_results.json",
}

CONFIG = {
    "run_date": RUN_DATE,
    "input_files": INPUT_FILES,
    "reported_metrics": {
        "cramers_v_octant_phase": 0.27,
    },
    "consistency_thresholds": {
        "phase_fraction_passing_min": 0.6,
        "phase_mean_error_max_radians": 0.7853981633974483,
        "zero_mean_s_over_n_max": 0.05,
        "zero_uniform_p_min": 0.05,
        "zero_uniform_fraction_min": 0.5,
        "berry_mean_abs_diff_max": 0.2,
        "berry_quant_score_min": 0.8,
    },
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = RESULTS_DIR / "q51_global_consistency_try2_config.json"
    _write_json(config_path, CONFIG)

    pre_reg_lines = [
        "# Pre-Registration: Q51 Global Consistency Check (Try2)",
        "",
        f"Date: {RUN_DATE}",
        "",
        "## HYPOTHESIS",
        "No direct circularity exists between the four tests, and the four results can be simultaneously true without logical contradiction.",
        "Any stronger inference (roots-of-unity structure or topological holonomy) is not forced by their intersection.",
        "",
        "## PREDICTION",
        "A dependency scan will show shared inputs (dataset + PCA projections) but no direct data-flow between test outputs and other tests.",
        "Consistency checks will show no mutual contradictions among the reported metrics.",
        "",
        "## FALSIFICATION",
        "If any test directly consumes another test's output as input, or if computed metrics are mutually exclusive under the fixed thresholds,",
        "the hypothesis is falsified.",
        "",
        "## DATA SOURCE",
        "- Phase arithmetic results: results/results.json",
        "- Zero-signature results: results/q51_zero_signature_try2_results.json",
        "- Berry phase results: results/q51_berry_phase_try2_results.json",
        "- Reported Cramer's V (octant–phase association): 0.27 (as given in prompt)",
        "- Underlying external dataset: https://download.tensorflow.org/data/questions-words.txt",
        "",
        "## SUCCESS THRESHOLD",
        "- direct_dependency_found = False",
        "- contradictions_count = 0",
        "",
        "## FIXED PARAMETERS",
        f"- Phase fraction passing min: {CONFIG['consistency_thresholds']['phase_fraction_passing_min']}",
        f"- Phase mean error max (rad): {CONFIG['consistency_thresholds']['phase_mean_error_max_radians']}",
        f"- Zero-signature mean |S|/n max: {CONFIG['consistency_thresholds']['zero_mean_s_over_n_max']}",
        f"- Zero-signature uniform p min: {CONFIG['consistency_thresholds']['zero_uniform_p_min']}",
        f"- Zero-signature uniform fraction min: {CONFIG['consistency_thresholds']['zero_uniform_fraction_min']}",
        f"- Berry mean |Δγ| max: {CONFIG['consistency_thresholds']['berry_mean_abs_diff_max']}",
        f"- Berry quant score min: {CONFIG['consistency_thresholds']['berry_quant_score_min']}",
        "",
        "## Anti-Patterns Guardrail",
        "- No synthetic data generation",
        "- No parameter search or post-hoc threshold changes",
        "- Only existing external-data results are used",
        "",
        "## Notes",
        "- This check is adversarial: results are treated as hostile evidence against each other.",
        "- All findings (pass/fail) will be reported.",
    ]

    (REPORTS_DIR / "q51_global_consistency_try2_prereg.md").write_text(
        "\n".join(pre_reg_lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
