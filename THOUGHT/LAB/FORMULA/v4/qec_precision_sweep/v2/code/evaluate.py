"""Evaluate QEC v2 sweep results against frozen preregistration criteria.

Reads sweep output JSON, evaluates all models against frozen p_th,
applies pass/fail criteria from PREREGISTRATION_v2.md, and generates
a verdict report with bootstrap confidence intervals.

Also supports cross-model comparison between two noise model sweeps.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FROZEN_P_THRESHOLD = 0.007071067811865475


def _split(rows: list[dict[str, Any]], heldout: set[int]) -> tuple[list, list]:
    train = [r for r in rows if int(r["distance"]) not in heldout]
    test = [r for r in rows if int(r["distance"]) in heldout]
    return train, test


def feature_matrix(rows: list[dict[str, Any]], feature_set: str) -> np.ndarray:
    p = np.array([r["physical_error_rate"] for r in rows], dtype=float)
    d = np.array([r["distance"] for r in rows], dtype=float)
    basis_z = np.array([1.0 if r["basis"] == "z" else 0.0 for r in rows], dtype=float)

    eps = 1e-60

    if feature_set == "formula_score":
        E_vals = np.maximum(1.0 - p, eps)
        grad_S_vals = np.maximum(p / FROZEN_P_THRESHOLD, eps)
        sigma_vals = np.maximum(np.sqrt(FROZEN_P_THRESHOLD / np.maximum(p, eps)), eps)
        fsv = np.log(np.maximum((E_vals / grad_S_vals) * (sigma_vals**d), eps))
        return np.column_stack([fsv])

    if feature_set == "formula_components":
        E_vals = np.maximum(1.0 - p, eps)
        grad_S_vals = np.maximum(p / FROZEN_P_THRESHOLD, eps)
        sigma_vals = np.maximum(np.sqrt(FROZEN_P_THRESHOLD / np.maximum(p, eps)), eps)
        x1 = np.log(np.maximum(E_vals / grad_S_vals, eps))
        x2 = d * np.log(np.maximum(sigma_vals, eps))
        return np.column_stack([x1, x2, basis_z])

    if feature_set == "standard_qec_scaling":
        return np.column_stack([np.log(np.maximum(p, eps)), d, d * np.log(np.maximum(p, eps)), basis_z])

    if feature_set == "p_only":
        return np.column_stack([np.log(np.maximum(p, eps)), basis_z])

    if feature_set == "distance_only":
        return np.column_stack([d, basis_z])

    raise ValueError(f"Unknown feature_set: {feature_set}")


def evaluate_single(rows: list[dict[str, Any]], heldout: set[int]) -> dict[str, Any]:
    train, test = _split(rows, heldout)
    y_train = np.array([r["log_suppression"] for r in train], dtype=float)
    y_test = np.array([r["log_suppression"] for r in test], dtype=float)

    out: dict[str, Any] = {
        "target": "log_suppression = ln(physical_error_rate / logical_error_rate_laplace)",
        "heldout_distances": sorted(heldout),
        "train_conditions": len(train),
        "test_conditions": len(test),
        "models": {},
    }

    model_names = [
        "formula_score",
        "formula_components",
        "standard_qec_scaling",
        "p_only",
        "distance_only",
    ]

    for name in model_names:
        try:
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(feature_matrix(train, name), y_train)
            pred_train = model.predict(feature_matrix(train, name))
            pred_test = model.predict(feature_matrix(test, name))

            train_mae = float(mean_absolute_error(y_train, pred_train))
            test_mae = float(mean_absolute_error(y_test, pred_test))
            train_r2 = float(r2_score(y_train, pred_train))
            test_r2 = float(r2_score(y_test, pred_test))

            bootstrap_maes = []
            rng = np.random.RandomState(20260513)
            for _ in range(1000):
                idx = rng.choice(len(y_test), size=len(y_test), replace=True)
                bootstrap_maes.append(
                    float(mean_absolute_error(y_test[idx], pred_test[idx]))
                )
            bootstrap_maes_arr = np.array(bootstrap_maes)
            ci_low = float(np.percentile(bootstrap_maes_arr, 2.5))
            ci_high = float(np.percentile(bootstrap_maes_arr, 97.5))

            out["models"][name] = {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "test_mae_ci_95": [ci_low, ci_high],
                "test_predictions": [
                    {
                        "basis": r["basis"],
                        "distance": r["distance"],
                        "physical_error_rate": r["physical_error_rate"],
                        "actual_log_suppression": float(y),
                        "predicted_log_suppression": float(yhat),
                        "absolute_error": float(abs(y - yhat)),
                    }
                    for r, y, yhat in zip(test, y_test, pred_test)
                ],
            }
        except Exception as exc:
            out["models"][name] = {
                "train_mae": None,
                "test_mae": None,
                "train_r2": None,
                "test_r2": None,
                "test_mae_ci_95": None,
                "error": str(exc),
            }

    return out


def verdict_single(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Apply pass/fail criteria from Section 10.1 of PREREGISTRATION_v2.md."""
    models = evaluation["models"]
    fs = models.get("formula_score", {})
    fc = models.get("formula_components", {})
    sqec = models.get("standard_qec_scaling", {})
    po = models.get("p_only", {})
    do = models.get("distance_only", {})

    reasons = []
    status = "UNKNOWN"

    fs_mae = fs.get("test_mae")
    sqec_mae = sqec.get("test_mae")
    po_mae = po.get("test_mae")
    do_mae = do.get("test_mae")
    fc_r2 = fc.get("test_r2")

    if fs_mae is None or sqec_mae is None:
        return {"status": "ERROR", "label": "ERROR", "reasons": ["Model fitting failed"]}

    fs_sqec_ratio = fs_mae / max(sqec_mae, 1e-60)
    sqec_fs_ratio = sqec_mae / max(fs_mae, 1e-60)

    pass_conditions = []
    fail_conditions = []

    # PASS condition 1: within 5% of standard QEC
    if fs_mae <= sqec_mae * 1.05:
        pass_conditions.append(
            f"formula_score MAE ({fs_mae:.4f}) <= standard_qec_scaling MAE * 1.05 ({sqec_mae * 1.05:.4f})"
        )
    else:
        fail_conditions.append(
            f"formula_score MAE ({fs_mae:.4f}) > standard_qec_scaling MAE * 1.05 ({sqec_mae * 1.05:.4f})"
        )

    # PASS condition 2: better than p_only
    if po_mae is not None and fs_mae < po_mae:
        pass_conditions.append(
            f"formula_score MAE ({fs_mae:.4f}) < p_only MAE ({po_mae:.4f})"
        )
    elif po_mae is not None:
        fail_conditions.append(
            f"formula_score MAE ({fs_mae:.4f}) >= p_only MAE ({po_mae:.4f})"
        )

    # PASS condition 3: formula_components R2 >= 0
    if fc_r2 is not None and fc_r2 >= 0.0:
        pass_conditions.append(
            f"formula_components R2 ({fc_r2:.4f}) >= 0.0"
        )
    elif fc_r2 is not None:
        fail_conditions.append(
            f"formula_components R2 ({fc_r2:.4f}) < 0.0"
        )

    # FAIL condition 1: standard QEC > 10% better
    if sqec_fs_ratio < 0.90:
        fail_conditions.append(
            f"standard_qec_scaling MAE ({sqec_mae:.4f}) < formula_score MAE * 0.90 ({fs_mae * 0.90:.4f})"
        )

    # FAIL condition 2: not better than p_only
    if po_mae is not None and fs_mae >= po_mae:
        fail_conditions.append(
            f"formula_score MAE ({fs_mae:.4f}) >= p_only MAE ({po_mae:.4f})"
        )

    # FAIL condition 3: distance_only is as good or better
    if do_mae is not None and do_mae <= fs_mae:
        fail_conditions.append(
            f"distance_only MAE ({do_mae:.4f}) <= formula_score MAE ({fs_mae:.4f})"
        )

    if len(pass_conditions) >= 2 and len(fail_conditions) == 0:
        status = "PASS"
        reasons = pass_conditions
    elif len(fail_conditions) >= 2:
        status = "FAIL"
        reasons = fail_conditions
    else:
        status = "AMBIGUOUS"
        reasons = pass_conditions + fail_conditions

    return {
        "status": status,
        "label": status,
        "pass_conditions": pass_conditions,
        "fail_conditions": fail_conditions,
    }


def falsification_check(depol_eval: dict, meas_eval: dict) -> dict[str, Any]:
    """Apply falsification criteria from Section 10.3 of PREREGISTRATION_v2.md."""
    conditions = []
    falsified = False

    for label, ev in [("depol", depol_eval), ("meas", meas_eval)]:
        models = ev["models"]
        fs = models.get("formula_score", {})
        sqec = models.get("standard_qec_scaling", {})
        po = models.get("p_only", {})
        fc = models.get("formula_components", {})

        fs_mae = fs.get("test_mae")
        sqec_mae = sqec.get("test_mae")
        po_mae = po.get("test_mae")
        fc_r2 = fc.get("test_r2")

        if fs_mae is None or sqec_mae is None:
            continue

        c1 = fs_mae > 1.5 * sqec_mae
        c2 = fc_r2 is not None and fc_r2 < 0.0
        c3 = po_mae is not None and po_mae < fs_mae

        conditions.append(
            {
                "noise_model": label,
                "fs_mae": fs_mae,
                "sqec_mae": sqec_mae,
                "ratio": fs_mae / max(sqec_mae, 1e-60),
                "fs_worse_than_1_5_sqec": c1,
                "fc_r2_negative": c2,
                "p_only_better_than_fs": c3,
                "all_three": c1 and c2 and c3,
            }
        )

    if len(conditions) >= 2:
        both_all_three = all(c["all_three"] for c in conditions)
        if both_all_three:
            falsified = True

    return {
        "falsified": falsified,
        "conditions": conditions,
    }


def confirmation_check(depol_verdict: dict, meas_verdict: dict,
                        depol_eval: dict, meas_eval: dict) -> dict[str, Any]:
    """Apply confirmation criteria from Section 10.4 of PREREGISTRATION_v2.md."""
    if depol_verdict["status"] != "PASS" or meas_verdict["status"] != "PASS":
        return {"confirmed": False, "reason": "Both noise models must PASS"}

    conditions_met = []
    conditions_failed = []

    for label, ev in [("depol", depol_eval), ("meas", meas_eval)]:
        models = ev["models"]
        fs = models.get("formula_score", {})
        sqec = models.get("standard_qec_scaling", {})
        fc = models.get("formula_components", {})

        fs_r2 = fs.get("test_r2")
        sqec_r2 = sqec.get("test_r2")
        fc_mae = fc.get("test_mae")
        sqec_mae = sqec.get("test_mae")

        if fs_r2 is not None and sqec_r2 is not None:
            if fs_r2 >= sqec_r2 - 0.03:
                conditions_met.append(
                    f"{label}: formula_score R2 ({fs_r2:.4f}) >= standard_qec_scaling R2 - 0.03 ({sqec_r2 - 0.03:.4f})"
                )
            else:
                conditions_failed.append(
                    f"{label}: formula_score R2 ({fs_r2:.4f}) < standard_qec_scaling R2 - 0.03 ({sqec_r2 - 0.03:.4f})"
                )

        if fc_mae is not None and sqec_mae is not None:
            if fc_mae < sqec_mae:
                conditions_met.append(
                    f"{label}: formula_components MAE ({fc_mae:.4f}) < standard_qec_scaling MAE ({sqec_mae:.4f})"
                )

    fc_better_than_sqec = any(
        "formula_components MAE" in c for c in conditions_met
    )

    if len(conditions_failed) == 0 and fc_better_than_sqec:
        return {
            "confirmed": True,
            "reason": "All confirmation criteria met",
            "conditions_met": conditions_met,
        }
    else:
        return {
            "confirmed": False,
            "reason": "Not all confirmation criteria met" if conditions_failed else "formula_components did not beat standard_qec_scaling on any model",
            "conditions_met": conditions_met,
            "conditions_failed": conditions_failed,
        }


def cross_model_verdict(depol_verdict: dict, meas_verdict: dict,
                        depol_eval: dict, meas_eval: dict) -> dict[str, Any]:
    """Apply cross-model verdict from Section 10.2 of PREREGISTRATION_v2.md."""
    d_status = depol_verdict["status"]
    m_status = meas_verdict["status"]

    verdict_map = {
        ("PASS", "PASS"): "CONFIRMED",
        ("PASS", "FAIL"): "MIXED",
        ("FAIL", "PASS"): "MIXED",
        ("FAIL", "FAIL"): "NEGATIVE",
        ("PASS", "AMBIGUOUS"): "MIXED",
        ("AMBIGUOUS", "PASS"): "MIXED",
        ("FAIL", "AMBIGUOUS"): "NEGATIVE",
        ("AMBIGUOUS", "FAIL"): "NEGATIVE",
        ("AMBIGUOUS", "AMBIGUOUS"): "INCONCLUSIVE",
        ("ERROR", "ERROR"): "ERROR",
    }

    label = verdict_map.get((d_status, m_status), "INCONCLUSIVE")

    falsification = falsification_check(depol_eval, meas_eval)
    confirmation = confirmation_check(depol_verdict, meas_verdict, depol_eval, meas_eval)

    if label == "CONFIRMED" and not confirmation["confirmed"]:
        label = "MIXED"

    result = {
        "overall_verdict": label,
        "depol_verdict": d_status,
        "meas_verdict": m_status,
        "falsification": falsification,
        "confirmation": confirmation,
    }

    return result


def write_verdict_report(path: Path, payload: dict[str, Any]) -> None:
    cv = payload["cross_verdict"]
    lines = [
        "# QEC Precision Sweep v2 -- Evaluator Report",
        "",
        f"Report id: `{payload['report_id']}`",
        f"UTC time: `{payload['created_utc']}`",
        f"Frozen p_th: `{FROZEN_P_THRESHOLD}`",
        "",
        "## Overall Cross-Model Verdict",
        "",
        f"**{cv['overall_verdict']}**",
        "",
    ]

    for nm_key in ["depol", "meas"]:
        nm_data = payload.get(nm_key)
        if nm_data is None:
            continue
        eval_data = nm_data["evaluation"]
        verdict_data = nm_data["verdict"]
        models = eval_data["models"]
        noise_label = nm_data.get("noise_model", nm_key.upper())

        lines.extend([
            f"## Noise Model: {noise_label}",
            "",
            f"Run id: `{nm_data['run_id']}`",
            f"Per-model verdict: **{verdict_data['status']}**",
            "",
            "### Model Metrics",
            "",
            "| Model | Train MAE | Test MAE | 95% CI | Train R2 | Test R2 |",
            "|---|---:|---:|---|---:|---:|",
        ])

        for name, metrics in sorted(models.items(), key=lambda kv: (
            kv[1].get("test_mae") if kv[1].get("test_mae") is not None else float("inf")
        )):
            ci = metrics.get("test_mae_ci_95")
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A"
            lines.append(
                f"| `{name}` | "
                f"{metrics['train_mae']:.4f} | "
                f"{metrics['test_mae']:.4f} | "
                f"{ci_str} | "
                f"{metrics['train_r2']:.4f} | "
                f"{metrics['test_r2']:.4f} |"
            )

        lines.append("")
        lines.append("### Pass/Fail Criteria")
        lines.append("")
        if verdict_data.get("pass_conditions"):
            lines.append("**Pass conditions met:**")
            for c in verdict_data["pass_conditions"]:
                lines.append(f"- {c}")
        if verdict_data.get("fail_conditions"):
            lines.append("")
            lines.append("**Fail conditions triggered:**")
            for c in verdict_data["fail_conditions"]:
                lines.append(f"- {c}")
        if not verdict_data.get("pass_conditions") and not verdict_data.get("fail_conditions"):
            lines.append("No conditions evaluated (model fitting error).")
        lines.append("")

    fals = cv.get("falsification", {})
    lines.extend([
        "## Falsification Check",
        "",
        f"**Falsified: {fals.get('falsified', 'N/A')}**",
        "",
    ])
    for cond in fals.get("conditions", []):
        lines.extend([
            f"### {cond['noise_model']}",
            f"- formula_score / standard_qec MAE ratio: {cond['ratio']:.4f}",
            f"- fs > 1.5 * sqec: {cond['fs_worse_than_1_5_sqec']}",
            f"- fc R2 < 0: {cond['fc_r2_negative']}",
            f"- p_only better than fs: {cond['p_only_better_than_fs']}",
            f"- All three falsification conditions: {cond['all_three']}",
            "",
        ])

    conf = cv.get("confirmation", {})
    lines.extend([
        "## Confirmation Check",
        "",
        f"**Confirmed: {conf.get('confirmed', 'N/A')}**",
        f"Reason: {conf.get('reason', 'N/A')}",
        "",
    ])
    if conf.get("conditions_met"):
        lines.append("Conditions met:")
        for c in conf["conditions_met"]:
            lines.append(f"- {c}")
    if conf.get("conditions_failed"):
        lines.append("")
        lines.append("Conditions failed:")
        for c in conf["conditions_failed"]:
            lines.append(f"- {c}")

    lines.extend([
        "",
        "## Preregistration",
        "",
        "The pass/fail criteria applied here are defined in:",
        "`THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/PREREGISTRATION_v2.md`",
        "",
        "No post-hoc remapping occurred. The frozen p_th = 0.007071067811865475",
        "was fixed before any v2 sweep ran.",
        "",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_sweep_run(source_dir: str) -> dict[str, Any]:
    src = Path(source_dir)
    json_files = list(src.glob("qec_precision_sweep_v2.json"))
    if not json_files:
        json_files = list(src.glob("qec_precision_sweep.json"))
    if not json_files:
        raise FileNotFoundError(f"No sweep JSON found in {source_dir}")
    return json.loads(json_files[0].read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="Directory containing sweep results JSON")
    parser.add_argument("--source-dir-meas", default=None, help="Second noise model directory for cross-model report")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    depol_payload = load_sweep_run(args.source_dir)
    depol_rows = depol_payload["conditions"]
    heldout = set(depol_payload["config"]["heldout_distances"])
    depol_eval = evaluate_single(depol_rows, heldout)
    depol_verdict = verdict_single(depol_eval)

    noise_model_1 = depol_payload["config"].get("noise_model", "depol")

    payload: dict[str, Any] = {
        "report_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_p_threshold": FROZEN_P_THRESHOLD,
        "depol": {
            "run_id": depol_payload["run_id"],
            "noise_model": noise_model_1,
            "evaluation": depol_eval,
            "verdict": depol_verdict,
        },
    }

    if args.source_dir_meas:
        meas_payload = load_sweep_run(args.source_dir_meas)
        meas_rows = meas_payload["conditions"]
        meas_heldout = set(meas_payload["config"]["heldout_distances"])
        meas_eval = evaluate_single(meas_rows, meas_heldout)
        meas_verdict = verdict_single(meas_eval)
        noise_model_2 = meas_payload["config"].get("noise_model", "meas")
        payload["meas"] = {
            "run_id": meas_payload["run_id"],
            "noise_model": noise_model_2,
            "evaluation": meas_eval,
            "verdict": meas_verdict,
        }
        cv = cross_model_verdict(depol_verdict, meas_verdict, depol_eval, meas_eval)
        payload["cross_verdict"] = cv
    else:
        cv = {
            "overall_verdict": depol_verdict["status"],
            "depol_verdict": depol_verdict["status"],
            "meas_verdict": "NOT_RUN",
            "falsification": {"falsified": False, "conditions": []},
            "confirmation": {"confirmed": False, "reason": "Second noise model not run"},
        }
        payload["cross_verdict"] = cv

    out_dir = Path(args.output_dir) if args.output_dir else (RESULTS / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "evaluation_v2.json"
    report_path = out_dir / "EVALUATION_REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_verdict_report(report_path, payload)

    print(json.dumps(payload["cross_verdict"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
