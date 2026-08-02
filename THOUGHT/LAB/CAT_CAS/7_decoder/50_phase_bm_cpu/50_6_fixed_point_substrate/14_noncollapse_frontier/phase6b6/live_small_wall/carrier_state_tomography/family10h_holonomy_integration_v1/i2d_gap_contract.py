#!/usr/bin/env python3
"""Validate the non-executing I2D physical reversibility gap contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = PACKAGE_ROOT / "I2D_REVERSIBILITY_GAP_CONTRACT.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def git_blob_sha(path: Path) -> str:
    """Return a canonical Git blob SHA-1."""

    payload = path.read_bytes()
    return hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def validate_source_identity(contract: dict[str, Any]) -> dict[str, str]:
    """Bind the gap contract to the exact I1-I2C artifacts."""

    verified: dict[str, str] = {}
    for filename, expected in contract["source_blobs"].items():
        path = PACKAGE_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = git_blob_sha(path)
        if actual != expected:
            raise AssertionError(f"source drift for {filename}: {actual} != {expected}")
        verified[filename] = actual
    return verified


def validate_inherited_status(contract: dict[str, Any]) -> dict[str, bool]:
    """Verify that current capabilities match the qualified predecessor gates."""

    i1 = load_json(PACKAGE_ROOT / "I1_COMPATIBILITY_MATRIX.json")
    i2 = load_json(PACKAGE_ROOT / "I2_GENERATOR_CATALOG.json")
    i2a = load_json(PACKAGE_ROOT / "I2A_RUNTIME_INTERFACE.json")
    i2b = load_json(PACKAGE_ROOT / "I2B_PAIR_DESIGN.json")
    i2c = load_json(PACKAGE_ROOT / "I2C_SYNTHETIC_MODEL_CONTRACT.json")
    snapshot = contract["current_capability_snapshot"]

    assert i1["h_gates"]["H0"]["status"] == "PARTIAL_CALIBRATION_ONLY"
    assert all(gate["passed"] is False for gate in i1["h_gates"].values())
    assert snapshot["scalar_q_calibration_confirmed"] is True
    assert snapshot["local_primary_minus_sham_differential_confirmed"] is True
    assert snapshot["stable_second_carrier_axis"] is False
    assert snapshot["frozen_multivariate_carrier_state"] is False

    assert i2["catalog_summary"]["h1_admissible_generator_count"] == 0
    assert i2["catalog_summary"]["h1_admissible_generator_pair_count"] == 0
    assert snapshot["h1_admissible_generator_count"] == 0
    assert snapshot["h1_admissible_generator_pair_count"] == 0

    assert i2a["acceptance"]["persistent_carrier_interface"] is True
    assert i2a["acceptance"]["physical_backend_present"] is False
    assert snapshot["persistent_post_source_runtime_interface"] is True
    assert snapshot["physical_operator_backend"] is False

    assert i2b["synthetic_admission"]["h1_pair_established"] is False
    assert i2b["synthetic_admission"]["h2_inverse_laws_established"] is False
    assert snapshot["qualified_inverse_count"] == 0

    assert i2c["scope"]["synthetic_only"] is True
    assert i2c["scope"]["physical_realization_established"] is False
    assert i2c["i2c_decision"]["synthetic_protocol_laws_qualified"] is True
    assert snapshot["reversible_reference_model"] == "synthetic_only"
    assert snapshot["independent_accumulator"] is False
    assert snapshot["r2_restoration"] is False

    return {
        "i1_h0_partial_and_h1_h7_unpassed": True,
        "i2_zero_generators_and_pairs": True,
        "i2a_interface_only_no_backend": True,
        "i2b_no_pair_or_inverse": True,
        "i2c_synthetic_only": True,
    }


def validate_observable_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Check that new observables are required and not falsely marked available."""

    observables = contract["minimum_state_observable_contract"]
    assert observables["logical_byte_digest"]["required"] is True
    assert observables["logical_byte_digest"]["dispositive_for_reversibility"] is False
    assert observables["D_single_scalar_q_coordinate"]["required_as_calibration"] is True
    assert observables["D_single_scalar_q_coordinate"]["sufficient_as_state"] is False
    assert observables["D_local_primary_minus_sham"]["required_as_local_coordinate"] is True
    assert observables["D_local_primary_minus_sham"]["sufficient_as_state"] is False

    unavailable = (
        "multi_observer_coherence_vector",
        "overlap_stratified_line_response",
        "timing_and_probe_vector",
        "public_codeword_discrimination",
    )
    for key in unavailable:
        assert observables[key]["required"] is True
        assert observables[key]["available"] is False

    discrimination = observables["public_codeword_discrimination"]
    assert discrimination["minimum_public_states"] >= 3
    assert discrimination["heldout_rank_at_least"] >= 2

    return {
        "required_observable_count": len(observables),
        "new_unavailable_observable_count": len(unavailable),
        "minimum_public_states": discrimination["minimum_public_states"],
        "minimum_heldout_rank": discrimination["heldout_rank_at_least"],
    }


def validate_test_matrix(contract: dict[str, Any]) -> dict[str, Any]:
    """Require the complete REV-01 through REV-15 fail-closed matrix."""

    tests = contract["reversibility_tests"]
    expected_ids = [f"REV-{index:02d}" for index in range(1, 16)]
    actual_ids = [test["id"] for test in tests]
    if actual_ids != expected_ids:
        raise AssertionError(f"reversibility test IDs drifted: {actual_ids}")
    if any(test["required"] is not True for test in tests):
        raise AssertionError("every reversibility test must be required")
    if any(test["current_pass"] is not False for test in tests):
        raise AssertionError("no physical reversibility test may currently pass")
    if any(not test["law"].strip() for test in tests):
        raise AssertionError("every reversibility test requires a law")

    kills = contract["hard_kills"]
    if len(kills) < 8 or any(not value for value in kills.values()):
        raise AssertionError("hard-kill matrix incomplete")

    return {
        "required_test_count": len(tests),
        "current_pass_count": sum(bool(test["current_pass"]) for test in tests),
        "hard_kill_count": len(kills),
    }


def validate_freeze_decision(contract: dict[str, Any]) -> dict[str, Any]:
    """Enforce that the physical package remains not freeze-ready."""

    freeze = contract["prospective_freeze_requirements"]
    if any(value is not False for value in freeze.values()):
        raise AssertionError("no prospective physical freeze field may be true")

    decision = contract["i2d_decision"]
    assert decision["result"] == (
        "I2D_PHYSICAL_REVERSIBILITY_GAP_CONTRACT_COMPLETE__"
        "OBSERVABLE_AND_BACKEND_PREREQUISITES_UNMET"
    )
    assert decision["gap_contract_complete"] is True
    assert decision["physical_package_freeze_ready"] is False
    assert decision["live_execution_authorized"] is False
    assert decision["next_gate"] == "I2E_PROSPECTIVE_MEASUREMENT_HARNESS_SKELETON"

    return {
        "gap_contract_complete": True,
        "physical_package_freeze_ready": False,
        "next_gate": decision["next_gate"],
    }


def validate_gap_contract() -> dict[str, Any]:
    """Run the complete I2D contract audit."""

    contract = load_json(CONTRACT_PATH)
    return {
        "contract_id": contract["contract_id"],
        "source_identity_verified": validate_source_identity(contract),
        "inherited_status_checks": validate_inherited_status(contract),
        "observable_contract": validate_observable_contract(contract),
        "test_matrix": validate_test_matrix(contract),
        "freeze_decision": validate_freeze_decision(contract),
        "decision": contract["i2d_decision"]["result"],
        "claim_ceiling": contract["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(validate_gap_contract(), indent=2, sort_keys=True))
