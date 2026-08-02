#!/usr/bin/env python3
"""Validate the read-only H0-H7 compatibility classification.

The audit reads only committed summaries and state records inherited from the frozen
Family 10h tomography lineage. It does not reinterpret evidence, contact hardware, or
promote any physical claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
MATRIX_PATH = PACKAGE_ROOT / "I1_COMPATIBILITY_MATRIX.json"


def find_repo_root() -> Path:
    """Locate the repository root from this package path."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "THOUGHT").is_dir() and (parent / ".github").is_dir():
            return parent
    raise RuntimeError("repository root not found")


def load_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object and reject non-object roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def git_blob_sha(path: Path) -> str:
    """Compute the canonical Git SHA-1 blob identifier for a file."""

    payload = path.read_bytes()
    preimage = f"blob {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(preimage).hexdigest()  # noqa: S324 - Git object identity


def require_text(path: Path, *needles: str) -> None:
    """Require every literal marker in a committed text artifact."""

    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise AssertionError(f"missing markers in {path}: {missing}")


def validate_source_identity(repo_root: Path, matrix: dict[str, Any]) -> dict[str, str]:
    """Verify every frozen source path against its recorded Git blob SHA."""

    verified: dict[str, str] = {}
    for name, source in matrix["source_blobs"].items():
        path = repo_root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = git_blob_sha(path)
        expected = source["blob_sha"]
        if actual != expected:
            raise AssertionError(f"source blob drift for {name}: {actual} != {expected}")
        verified[name] = actual
    return verified


def validate_retained_evidence(repo_root: Path, matrix: dict[str, Any]) -> dict[str, Any]:
    """Check the exact retained statements that force the I1 classification."""

    sources = matrix["source_blobs"]
    path = lambda name: repo_root / sources[name]["path"]  # noqa: E731

    goal = load_json(path("AUTONOMOUS_SMALL_WALL_GOAL_STATE.json"))
    claim = goal["claim_state"]
    observable = goal["current_observable"]

    assert claim["scalar_q_readout_confirmed_prospectively"] is True
    assert claim["local_paired_differential_confirmed_prospectively"] is True
    assert claim["full_carrier_state_tomography_established"] is False
    assert claim["physical_relational_memory_established"] is False
    assert claim["r2_restoration_established"] is False
    assert claim["catalytic_borrowing_established"] is False
    assert claim["small_wall_crossed"] is False
    assert observable["D_local"] == "R_primary - R_sham"
    assert observable["claim_ceiling"] == (
        "PUBLIC_POST_SOURCE_LOCAL_PRIMARY_MINUS_SHAM_DIFFERENTIAL_CONFIRMED"
    )

    require_text(
        path("RELATIONAL_SUCCESSOR_DECISION.md"),
        "FAMILY10H_SECOND_OPERATOR_DIMENSION_UNRESOLVED_RETROSPECTIVE",
        "Scalar slope: `1.8612688337053558`",
        "R2: `0.9940328492833816`",
        "No full tomography",
    )
    require_text(
        path("OPERATOR_DIMENSION_AUDIT.md"),
        "second direction survives required strata: `False`",
        "No candidate survived the full retrospective law",
    )
    require_text(
        path("OPERATOR_DIMENSION_DEEP_HUNT.md"),
        "stable second axis across required strata: `False`",
        "No clean second operator dimension survived",
    )
    require_text(
        path("RELATION_COMPOSITION_LOOP_DISCOVERY_SUMMARY.md"),
        "composition-loop candidate: `True`",
        "source-off abs/alive abs: `0.063`",
        "This is exploratory evidence only",
    )
    require_text(
        path("RELATION_COMPOSITION_LOOP_RESET_SCREEN_SUMMARY.md"),
        "composition-loop reset candidate: `False`",
        "source-off abs/alive abs: `0.823`",
        "reset abs/alive abs: `0.667`",
        "one-factor strata same sign: `False`",
    )
    require_text(
        path("RELATION_BALANCED_PHASE_SOURCE_OFF_SCREEN_SUMMARY.md"),
        "balanced phase source-off/dead candidate: `False`",
        "source-off abs/alive abs: `0.985`",
    )
    require_text(
        path("LOCAL_PAIRED_DIFFERENTIAL_AUDIT.md"),
        "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_LAW_SUPPORTED_FOR_PROSPECTIVE_FREEZE",
        "Prospective law frozen: `True`",
        "Small Wall crossed: `false`",
    )

    replay = load_json(path("SCALAR_REPLAY_ADVERSARY_REPORT.json"))
    replay_summary = replay["report"]["summary"]
    if not replay_summary:
        raise AssertionError("scalar replay summary is empty")
    assert all(
        entry["adversary_predicts_candidate_on_heldout_data"] is False
        for entry in replay_summary.values()
    )
    assert replay["report"]["leakage_boundary"]["target_label_or_private_relation_used"] is False

    return {
        "scalar_q_coordinate_confirmed": True,
        "local_paired_coordinate_confirmed": True,
        "full_carrier_state_absent": True,
        "second_operator_axis_unresolved": True,
        "composition_followup_rejected": True,
        "r2_restoration_absent": True,
        "retrospective_scalar_replay_model_failed": True,
        "bounded_replay_gate_not_promoted": True,
    }


def validate_gate_logic(matrix: dict[str, Any]) -> dict[str, str]:
    """Enforce that I1 does not silently promote any H gate."""

    gates = matrix["h_gates"]
    expected = {f"H{index}" for index in range(8)}
    if set(gates) != expected:
        raise AssertionError(f"H-gate set mismatch: {set(gates)}")
    if any(gate["passed"] is not False for gate in gates.values()):
        raise AssertionError("I1 may not mark any H0-H7 gate passed")
    if gates["H0"]["status"] != "PARTIAL_CALIBRATION_ONLY":
        raise AssertionError("H0 must remain partial calibration only")
    if gates["H4"]["status"] != "ORDER_EFFECT_OBSERVED_BUT_CONNECTION_LAW_NOT_ESTABLISHED":
        raise AssertionError("H4 claim boundary drift")
    if gates["H7"]["status"] != "DIAGNOSTIC_ADVERSARY_ONLY":
        raise AssertionError("H7 claim boundary drift")

    decision = matrix["i1_decision"]
    assert decision["transport_candidate_freeze_allowed"] is False
    assert decision["synthetic_i2_design_allowed"] is True
    assert decision["live_execution_authorized"] is False

    forbidden = {
        "FAMILY10H_CATALYTIC_HOLONOMY_CANDIDATE",
        "NATIVE_CATALYTIC_FIBER_PUSHFORWARD_ESTABLISHED",
        "SMALL_WALL_CROSSED",
    }
    serialized = json.dumps(matrix, sort_keys=True)
    for token in forbidden:
        if f'"{token}"' in serialized:
            raise AssertionError(f"forbidden positive result token present: {token}")

    return {name: gate["status"] for name, gate in sorted(gates.items())}


def run_audit() -> dict[str, Any]:
    """Run the complete read-only compatibility audit."""

    repo_root = find_repo_root()
    matrix = load_json(MATRIX_PATH)
    source_identity = validate_source_identity(repo_root, matrix)
    evidence = validate_retained_evidence(repo_root, matrix)
    gates = validate_gate_logic(matrix)

    return {
        "audit_id": matrix["audit_id"],
        "source_identity_verified": source_identity,
        "retained_evidence_checks": evidence,
        "h_gate_status": gates,
        "decision": matrix["i1_decision"]["result"],
        "next_gate": matrix["i1_decision"]["next_gate"],
        "claim_ceiling": matrix["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, sort_keys=True))
