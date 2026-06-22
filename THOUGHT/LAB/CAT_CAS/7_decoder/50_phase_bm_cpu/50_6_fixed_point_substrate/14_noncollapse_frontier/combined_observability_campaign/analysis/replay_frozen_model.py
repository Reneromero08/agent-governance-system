#!/usr/bin/env python3
"""Replay the immutable V1 frozen-model contract without training or selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_SHA256 = "f1b8047ba0d80d027edcec9a841c8a4320dfe6796bbad7271fd43dc6a86dee5e"
CACHE_SHA256 = {
    "v2s3_seed5.npz": "d7cf0f9fb07fabb9fce50d45594067331021775dd88851cdaf0d4dc80fe9e628",
    "v4s5_seed5.npz": "225614b19b4323069086668e26face5947fe135332adfb3da96b1a577f8fb93c",
}
EXPECTED_SEED5 = {
    "v2s3_seed5": {
        "stage_b_joint_accuracy": 0.04131944444444444,
        "stage_b_mode_accuracy": 0.259375,
        "stage_b_theta_accuracy": 0.15833333333333333,
        "stage_c_zero_input_gain": 0.0028150917247609097,
        "operator_nrmse": 0.9600993584436434,
    },
    "v4s5_seed5": {
        "stage_b_joint_accuracy": 0.07152777777777777,
        "stage_b_mode_accuracy": 0.2611111111111111,
        "stage_b_theta_accuracy": 0.2375,
        "stage_c_zero_input_gain": 0.00018943988888986407,
        "operator_nrmse": 0.5161325831243231,
    },
}
PERMANENT_STATEMENTS = [
    "RETROSPECTIVE_FULL_DATASET_NEGATIVE_ADJUDICATION",
    "PRISTINE_FINAL_TEST_HYGIENE_NOT_PROVEN",
    "V2_RERUN_NOT_AUTHORIZED",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def recorded_seed5(adjudication: dict, session: str) -> dict[str, float]:
    ledger = adjudication["IMPLEMENTATION_RECOVERY_ANALYSIS"]
    stage_b = ledger["stage_b_by_heldout_session"][session]
    stage_c = ledger["stage_c_by_heldout_session"][session]
    operator = ledger["operator_by_heldout_session"][session]
    return {
        "stage_b_joint_accuracy": stage_b["joint_accuracy"],
        "stage_b_mode_accuracy": stage_b["mode_accuracy"],
        "stage_b_theta_accuracy": stage_b["theta_accuracy"],
        "stage_c_zero_input_gain": stage_c["zero_input_gain"],
        "operator_nrmse": operator["nrmse"],
    }


def replay(model_path: Path, evidence_root: Path) -> dict:
    digest = sha256(model_path)
    if digest != MODEL_SHA256:
        raise ValueError(f"frozen model digest mismatch: {digest}")

    # Deserialization occurs only after the byte digest has passed.
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model["seed5_used_for_selection"] or model["selection_sessions"] != [
        "v2s3_seed3", "v4s5_seed3"
    ]:
        raise ValueError("frozen model violates partition contract")
    coefficients = np.asarray(model["selected_operator_coefficients"], dtype=float)
    centroids = np.asarray(model["stage_b_centroids"], dtype=float)
    if coefficients.shape != (63, 2) or centroids.shape != (32, 14):
        raise ValueError("frozen model shape mismatch")
    if not np.isfinite(coefficients).all() or not np.isfinite(centroids).all():
        raise ValueError("frozen model contains non-finite values")

    cache_bindings = {}
    for name, expected in CACHE_SHA256.items():
        path = evidence_root / "feature_cache" / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"seed-5 cache digest mismatch: {name}")
        with np.load(path, allow_pickle=False) as cache:
            shapes = {key: list(cache[key].shape) for key in cache.files}
            nonfinite_counts = {
                key: int(np.size(cache[key]) - np.isfinite(cache[key]).sum())
                for key in ("recovery", "original", "gate")
            }
            if shapes["recovery"] != [8288, 6, 2] or shapes["original"] != [8288, 2]:
                raise ValueError(f"seed-5 cache shape mismatch: {name}")
        cache_bindings[name] = {
            "sha256": actual,
            "shapes": shapes,
            "derived_nonfinite_counts": nonfinite_counts,
        }

    adjudication_path = evidence_root / "full_adjudication.json"
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    replayed = {
        session: recorded_seed5(adjudication, session) for session in EXPECTED_SEED5
    }
    if replayed != EXPECTED_SEED5:
        raise ValueError("recorded seed-5 outputs do not match frozen replay contract")
    ledger = adjudication["IMPLEMENTATION_RECOVERY_ANALYSIS"]
    verdict = {
        "stage_b": ledger["stage_b_verdict"],
        "stage_c": ledger["stage_c_verdict"],
        "predictive_operator": ledger["predictive_operator_verdict"],
    }
    expected_verdict = {
        "stage_b": "NO_ORDER_RESOLUTION",
        "stage_c": "DRIVEN_RELATIONAL_TRANSPORT_ONLY",
        "predictive_operator": "NO_STABLE_PREDICTIVE_OPERATOR",
    }
    if verdict != expected_verdict:
        raise ValueError("final verdict mismatch")

    return {
        "schema_id": "INDEPENDENT_FROZEN_MODEL_REPLAY_AUDIT_V1",
        "audit_name": "INDEPENDENT_FROZEN_MODEL_REPLAY_AUDIT",
        "frozen_model": {"path": str(model_path), "sha256": digest},
        "model_selection_performed": False,
        "code_changes_during_replay": False,
        "seed5_retry_performed": False,
        "seed5_cache_bindings": cache_bindings,
        "recorded_seed5_outputs_confirmed": replayed,
        "final_verdict_confirmed": verdict,
        "permanent_v1_statement": PERMANENT_STATEMENTS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = replay(args.model, args.evidence_root)
    payload = canonical_bytes(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps({"manifest": str(args.output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
