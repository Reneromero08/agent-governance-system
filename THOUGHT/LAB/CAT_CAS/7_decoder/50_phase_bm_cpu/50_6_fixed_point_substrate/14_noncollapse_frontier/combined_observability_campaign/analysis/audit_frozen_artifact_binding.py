#!/usr/bin/env python3
"""Bind externally supplied frozen V1 artifacts to recorded outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVIDENCE_ID = "phase6_v1_full_raw_adjudication_7c44af0f"
MODEL_REL = "frozen_model_before_final_test.json"
ADJUDICATION_REL = "full_adjudication.json"
CACHE_RELS = (
    "feature_cache/v2s3_seed5.npz",
    "feature_cache/v4s5_seed5.npz",
)
PERMANENT_STATEMENTS = (
    "RETROSPECTIVE_FULL_DATASET_NEGATIVE_ADJUDICATION",
    "PRISTINE_FINAL_TEST_HYGIENE_NOT_PROVEN",
    "V2_RERUN_NOT_AUTHORIZED",
)


@dataclass(frozen=True)
class BindingContract:
    model_sha256: str
    adjudication_sha256: str
    cache_sha256: dict[str, str]
    cache_shapes: dict[str, tuple[int, ...]]
    cache_nonfinite_counts: dict[str, int]


KNOWN_CONTRACT = BindingContract(
    model_sha256="f1b8047ba0d80d027edcec9a841c8a4320dfe6796bbad7271fd43dc6a86dee5e",
    adjudication_sha256="1cd1f288410e6e616c5b39ab6c2dc7b371dd135b03c4e8f4d6b5798aa900917e",
    cache_sha256={
        CACHE_RELS[0]: "d7cf0f9fb07fabb9fce50d45594067331021775dd88851cdaf0d4dc80fe9e628",
        CACHE_RELS[1]: "225614b19b4323069086668e26face5947fe135332adfb3da96b1a577f8fb93c",
    },
    cache_shapes={"recovery": (8288, 6, 2), "original": (8288, 2), "gate": (8288, 2)},
    cache_nonfinite_counts={"recovery": 6144, "original": 1024, "gate": 3712},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def required(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required external evidence artifact absent: {EVIDENCE_ID}/{relative}")
    return path


def verify_digest(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} digest mismatch: expected {expected}, got {actual}")


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


def audit_binding(evidence_root: Path, contract: BindingContract = KNOWN_CONTRACT) -> dict:
    model_path = required(evidence_root, MODEL_REL)
    adjudication_path = required(evidence_root, ADJUDICATION_REL)
    cache_paths = {relative: required(evidence_root, relative) for relative in CACHE_RELS}

    # Every byte digest is checked before any JSON or NPZ deserialization.
    verify_digest(model_path, contract.model_sha256, "frozen model")
    verify_digest(adjudication_path, contract.adjudication_sha256, "recorded adjudication")
    for relative, path in cache_paths.items():
        verify_digest(path, contract.cache_sha256[relative], relative)

    model = json.loads(model_path.read_text(encoding="utf-8"))
    required_model_keys = {
        "seed5_used_for_selection", "selection_sessions", "selected_operator",
        "selected_operator_coefficients", "stage_b_centroids", "stage_b_labels",
    }
    if not required_model_keys <= model.keys():
        raise ValueError("frozen model schema missing required keys")
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
    for relative, path in cache_paths.items():
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != {"recovery", "original", "gate", "raw_sha256"}:
                raise ValueError(f"cache schema mismatch: {relative}")
            shapes = {key: tuple(cache[key].shape) for key in ("recovery", "original", "gate")}
            if shapes != contract.cache_shapes:
                raise ValueError(f"cache shape mismatch: {relative}")
            counts = {
                key: int(cache[key].size - np.isfinite(cache[key]).sum())
                for key in ("recovery", "original", "gate")
            }
            if counts != contract.cache_nonfinite_counts:
                raise ValueError(f"cache non-finite pattern mismatch: {relative}")
            raw_digest = str(cache["raw_sha256"].item())
            if len(raw_digest) != 64:
                raise ValueError(f"cache raw digest malformed: {relative}")
        cache_bindings[f"{EVIDENCE_ID}/{relative}"] = {
            "sha256": contract.cache_sha256[relative],
            "shapes": {key: list(value) for key, value in shapes.items()},
            "derived_nonfinite_counts": counts,
            "raw_sha256": raw_digest,
        }

    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if adjudication.get("schema_id") != "CAT_CAS_PHASE6_V1_FULL_RAW_ADJUDICATION_V1":
        raise ValueError("recorded adjudication schema mismatch")
    recorded = {
        session: recorded_seed5(adjudication, session)
        for session in ("v2s3_seed5", "v4s5_seed5")
    }
    ledger = adjudication["IMPLEMENTATION_RECOVERY_ANALYSIS"]
    verdict = {
        "stage_b": ledger["stage_b_verdict"],
        "stage_c": ledger["stage_c_verdict"],
        "predictive_operator": ledger["predictive_operator_verdict"],
    }
    return {
        "schema_id": "FROZEN_ARTIFACT_AND_RECORDED_OUTPUT_BINDING_AUDIT_V2",
        "audit_name": "FROZEN_ARTIFACT_AND_RECORDED_OUTPUT_BINDING_AUDIT",
        "input_class": "EXTERNALLY_SUPPLIED_EVIDENCE_ARTIFACTS",
        "evidence_root_identifier": EVIDENCE_ID,
        "frozen_model": {"artifact_id": f"{EVIDENCE_ID}/{MODEL_REL}", "sha256": contract.model_sha256},
        "recorded_adjudication": {
            "artifact_id": f"{EVIDENCE_ID}/{ADJUDICATION_REL}",
            "sha256": contract.adjudication_sha256,
        },
        "model_selection_performed": False,
        "independent_metric_recomputation_performed": False,
        "seed5_retry_performed": False,
        "seed5_cache_bindings": cache_bindings,
        "recorded_seed5_output_bindings": recorded,
        "recorded_final_verdict_binding": verdict,
        "recomputation_blockers": [
            "seed-5 caches do not serialize Stage B ground-truth labels",
            "seed-5 caches do not serialize transition controls or targets",
            "frozen model does not serialize Stage C transition coefficients and baselines",
            "frozen model does not serialize mechanical verdict thresholds",
        ],
        "permanent_v1_statement": list(PERMANENT_STATEMENTS),
    }


def write_exclusive(path: Path, value: dict) -> str:
    payload = canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or sidecar.exists():
        raise FileExistsError("binding artifact or checksum already exists")
    try:
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            with sidecar.open("x", encoding="ascii", newline="\n") as output:
                output.write(f"{digest}  {path.name}\n")
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
    except Exception:
        raise
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = audit_binding(args.evidence_root)
    digest = write_exclusive(args.output, manifest)
    print(json.dumps({"artifact_id": args.output.name, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
