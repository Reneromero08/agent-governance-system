#!/usr/bin/env python3
"""Test whether Gate A exposes state beyond direct input response."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RUNS = (
    "gate_a_first_light_20260711i",
    "gate_a_first_light_20260711j",
    "pilot_anchor_np_20260711b",
    "pilot_anchor_sham_20260711a",
)


class PilotAnalysisError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise PilotAnalysisError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_session(root: Path, run_id: str) -> list[dict[str, Any]]:
    run = root / run_id
    final = json.loads((run / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    require(final["status"] == "GATE_A_FIRST_LIGHT_COMPLETE", f"incomplete run: {run_id}")
    lockin = load_jsonl(run / "runtime" / "LOCKIN_IQ.jsonl")
    trace = load_jsonl(run / "runtime" / "slot_trace.jsonl")
    analysis = json.loads((run / "FIRST_LIGHT_ANALYSIS.json").read_text(encoding="utf-8"))
    ring = {int(row["slot_index"]): row for row in analysis["ring_slots"]}
    require(len(lockin) == len(trace) == len(ring) == 16, f"slot coverage mismatch: {run_id}")
    rows: list[dict[str, Any]] = []
    for index in range(16):
        measured = lockin[index]
        control = trace[index]
        require(int(measured["slot_index"]) == int(control["index"]) == index, "slot mapping mismatch")
        phase_index = control["phase_index"]
        phase = 0.0 if phase_index is None else 2.0 * math.pi * float(phase_index) / 8.0
        drive = 1.0 if control["drive_on"] else 0.0
        rows.append(
            {
                "run_id": run_id,
                "slot_index": index,
                "token": control["token"],
                "response": np.array(
                    [
                        float(measured["lockin_i"]),
                        float(measured["lockin_q"]),
                        float(ring[index]["mean_ring_period"]),
                    ],
                    dtype=float,
                ),
                "control": np.array(
                    [
                        drive,
                        drive * math.cos(phase),
                        drive * math.sin(phase),
                    ],
                    dtype=float,
                ),
            }
        )
    return rows


def transitions(sessions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for run_id, rows in sessions.items():
        for index in range(15):
            values.append(
                {
                    "run_id": run_id,
                    "target_slot": index + 1,
                    "state": rows[index]["response"],
                    "control": rows[index + 1]["control"],
                    "target": rows[index + 1]["response"],
                }
            )
    return values


def standardized_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    y_mean = train_y.mean(axis=0)
    y_scale = train_y.std(axis=0)
    y_scale[y_scale < 1e-12] = 1.0
    x_train = (train_x - x_mean) / x_scale
    x_test = (test_x - x_mean) / x_scale
    design = np.column_stack([np.ones(len(x_train)), x_train])
    test_design = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ ((train_y - y_mean) / y_scale))
    return (test_design @ coefficients) * y_scale + y_mean


def normalized_rmse(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    error = (prediction - target) / scale
    return {
        "all": float(np.sqrt(np.mean(error**2))),
        "complex": float(np.sqrt(np.mean(error[:, :2] ** 2))),
        "ring": float(np.sqrt(np.mean(error[:, 2] ** 2))),
    }


def evaluate(sessions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = transitions(sessions)
    folds: dict[str, Any] = {}
    aggregate_errors: dict[str, list[np.ndarray]] = {
        name: [] for name in ("mean", "last_value", "input_only", "time_index", "state_control")
    }
    aggregate_targets: list[np.ndarray] = []
    aggregate_scales: list[np.ndarray] = []

    for held_out in sessions:
        train = [row for row in rows if row["run_id"] != held_out]
        test = [row for row in rows if row["run_id"] == held_out]
        train_y = np.stack([row["target"] for row in train])
        test_y = np.stack([row["target"] for row in test])
        scale = train_y.std(axis=0)
        scale[scale < 1e-12] = 1.0

        train_control = np.stack([row["control"] for row in train])
        test_control = np.stack([row["control"] for row in test])
        train_state_control = np.stack([np.concatenate([row["state"], row["control"]]) for row in train])
        test_state_control = np.stack([np.concatenate([row["state"], row["control"]]) for row in test])
        train_time = np.eye(15)[[int(row["target_slot"]) - 1 for row in train]]
        test_time = np.eye(15)[[int(row["target_slot"]) - 1 for row in test]]

        predictions = {
            "mean": np.repeat(train_y.mean(axis=0, keepdims=True), len(test), axis=0),
            "last_value": np.stack([row["state"] for row in test]),
            "input_only": standardized_ridge(train_control, train_y, test_control),
            "time_index": standardized_ridge(train_time, train_y, test_time),
            "state_control": standardized_ridge(train_state_control, train_y, test_state_control),
        }
        metrics = {name: normalized_rmse(value, test_y, scale) for name, value in predictions.items()}
        folds[held_out] = metrics
        aggregate_targets.append(test_y)
        aggregate_scales.extend([scale] * len(test))
        for name, prediction in predictions.items():
            aggregate_errors[name].append(prediction - test_y)

    targets = np.concatenate(aggregate_targets)
    scales = np.stack(aggregate_scales)
    aggregate: dict[str, dict[str, float]] = {}
    for name, parts in aggregate_errors.items():
        errors = np.concatenate(parts) / scales
        aggregate[name] = {
            "all": float(np.sqrt(np.mean(errors**2))),
            "complex": float(np.sqrt(np.mean(errors[:, :2] ** 2))),
            "ring": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        }

    state_score = aggregate["state_control"]["all"]
    beats_aggregate = all(
        state_score < aggregate[name]["all"]
        for name in ("mean", "last_value", "input_only", "time_index")
    )
    beats_each_fold = all(
        fold["state_control"]["all"] < fold["input_only"]["all"]
        and fold["state_control"]["all"] < fold["last_value"]["all"]
        for fold in folds.values()
    )
    return {
        "schema_id": "CAT_CAS_OBSERVABLE_CARRIER_PILOT_ANALYSIS_V1",
        "session_ids": list(sessions),
        "response_fields": ["lockin_I", "lockin_Q", "ring_osc_period"],
        "operator": "shared affine next-state response from current measured state and next executed phase control",
        "folds": folds,
        "aggregate_normalized_rmse": aggregate,
        "state_control_to_input_only_ratio": state_score / aggregate["input_only"]["all"],
        "state_control_to_last_value_ratio": state_score / aggregate["last_value"]["all"],
        "OBSERVABLE_CARRIER_STATE_FOUND": bool(beats_aggregate and beats_each_fold),
        "claim_ceiling": "phase-sensitive physical readout; state milestone requires held-out superiority over all baselines",
        "target_row_count": int(len(targets)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_ids = tuple(args.runs or DEFAULT_RUNS)
    sessions = {run_id: load_session(args.runs_root, run_id) for run_id in run_ids}
    result = evaluate(sessions)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
