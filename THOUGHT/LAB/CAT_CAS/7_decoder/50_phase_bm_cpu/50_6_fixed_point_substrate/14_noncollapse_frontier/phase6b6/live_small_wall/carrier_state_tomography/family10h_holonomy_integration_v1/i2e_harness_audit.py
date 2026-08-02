#!/usr/bin/env python3
"""Validate the fixture-only I2E prospective measurement harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from measurement_harness import (
    build_valid_synthetic_transaction,
    run_harness_self_test,
    validate_transaction,
)


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "I2E_MEASUREMENT_HARNESS_CONTRACT.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    preimage = f"blob {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(preimage).hexdigest()  # noqa: S324 - Git identity


def validate_harness_contract() -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)

    blobs = {
        contract["source_blob"]["path"]: contract["source_blob"]["blob_sha"],
        contract["implementation_blob"]["path"]: contract["implementation_blob"]["blob_sha"],
    }
    verified: dict[str, str] = {}
    for filename, expected in blobs.items():
        actual = git_blob_sha(ROOT / filename)
        if actual != expected:
            raise AssertionError(f"blob drift for {filename}: {actual} != {expected}")
        verified[filename] = actual

    expected_counts = {
        "required_transaction_sections": 12,
        "required_state_checkpoints": 6,
        "required_state_components": 9,
        "required_observers": 4,
        "required_overlap_strata": 4,
        "required_operation_roles": 4,
        "required_environment_fields": 9,
        "negative_fixture_classes": 12,
    }
    actual_counts = {key: len(contract[key]) for key in expected_counts}
    if actual_counts != expected_counts:
        raise AssertionError(f"schema count drift: {actual_counts}")

    record = build_valid_synthetic_transaction()
    valid = validate_transaction(record)
    assert valid["state_checkpoint_count"] == 6
    assert valid["operation_receipt_count"] == 8
    assert valid["physical_package_freeze_ready"] is False
    assert valid["live_execution_authorized"] is False
    assert record["identity"]["physical_backend"] is False
    assert record["identity"]["live_authority"] is False
    assert record["thresholds"]["frozen"] is False
    assert all(
        value is None
        for key, value in record["thresholds"].items()
        if key != "frozen"
    )
    assert record["restoration"]["physical_state_equivalence"] is False
    assert record["restoration"]["qualified_inverse"] is False
    assert record["restoration"]["r2_restoration_claimed"] is False
    assert len({x["carrier_id"] for x in record["state_observations"].values()}) == 1
    assert len(record["carrier"]["public_codewords"]) == 3
    assert record["carrier"]["heldout_rank_claimed"] is None

    self_test = run_harness_self_test()
    negative = self_test["negative_fixture_results"]
    assert list(negative) == contract["negative_fixture_classes"]
    assert all(expected == actual for expected, actual in negative.items())
    assert self_test["passed"] is True

    acceptance = contract["harness_acceptance"]
    assert acceptance["transaction_schema_complete"] is True
    assert acceptance["valid_synthetic_fixture_available"] is True
    assert acceptance["negative_fixture_count"] == 12
    assert acceptance["all_negative_fixtures_rejected"] is True
    assert acceptance["physical_backend_present"] is False
    assert acceptance["numerical_thresholds_frozen"] is False
    assert acceptance["physical_package_freeze_ready"] is False

    decision = contract["i2e_decision"]
    assert decision["schema_freeze_allowed"] is True
    assert decision["physical_package_freeze_ready"] is False
    assert decision["live_execution_authorized"] is False
    assert decision["next_gate"] == "I2F_TOPOLOGY_BACKEND_AND_NDESTRUCTIVE_PROBE_SPEC"

    return {
        "contract_id": contract["contract_id"],
        "blob_identity_verified": verified,
        "schema_counts": actual_counts,
        "valid_fixture_class": valid["classification"],
        "negative_fixture_results": negative,
        "decision": decision["result"],
        "next_gate": decision["next_gate"],
        "claim_ceiling": contract["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(validate_harness_contract(), indent=2, sort_keys=True))
