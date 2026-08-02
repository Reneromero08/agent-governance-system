#!/usr/bin/env python3
"""Validate the I2C overwrite and reversible synthetic ownership models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from synthetic_ownership_model import (
    CARRIER_SIZE,
    OWNER_A,
    OWNER_B,
    OWNER_HOME,
    OWNERS,
    OwnershipState,
    SwapOperation,
    apply_word,
    verify_synthetic_models,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = PACKAGE_ROOT / "I2C_SYNTHETIC_MODEL_CONTRACT.json"


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


def compare_expected(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Require exact values for every frozen expected field."""

    for key, value in expected.items():
        if key == "promotion":
            continue
        if actual.get(key) != value:
            raise AssertionError(f"model mismatch for {key}: {actual.get(key)!r} != {value!r}")


def representative_state(line_index: int, owner: str) -> OwnershipState:
    """Return a baseline modified at one representative line."""

    owners = [OWNER_HOME] * CARRIER_SIZE
    owners[line_index] = owner
    return OwnershipState(tuple(owners))


def validate_reversible_algebra() -> dict[str, bool]:
    """Independently verify local two-sided inverses and disjoint commutation."""

    op_a = SwapOperation("A", frozenset(range(0, 16)), OWNER_HOME, OWNER_A)
    op_b = SwapOperation("B", frozenset(range(8, 24)), OWNER_HOME, OWNER_B)

    for operation, representative_lines in (
        (op_a, (0, 8)),
        (op_b, (8, 16)),
    ):
        for line_index in representative_lines:
            for owner in OWNERS:
                state = representative_state(line_index, owner)
                if operation.inverse.apply(operation.apply(state)) != state:
                    raise AssertionError(f"left inverse failed: {operation.name}")
                if operation.apply(operation.inverse.apply(state)) != state:
                    raise AssertionError(f"right inverse failed: {operation.name}")

    disjoint_a = SwapOperation("A_disjoint", frozenset(range(0, 8)), OWNER_HOME, OWNER_A)
    disjoint_b = SwapOperation("B_disjoint", frozenset(range(16, 24)), OWNER_HOME, OWNER_B)
    disjoint_commutator = (
        disjoint_a,
        disjoint_b,
        disjoint_a.inverse,
        disjoint_b.inverse,
    )
    baseline = OwnershipState.home_baseline()
    if apply_word(baseline, disjoint_commutator) != baseline:
        raise AssertionError("nonempty disjoint-support commutator was not identity")

    return {
        "A_local_two_sided_inverse": True,
        "B_local_two_sided_inverse": True,
        "nonempty_disjoint_support_commutes": True,
    }


def validate_contract() -> dict[str, Any]:
    """Validate model identity, exact results, and claim boundaries."""

    contract = load_json(CONTRACT_PATH)
    model_path = PACKAGE_ROOT / contract["model_blob"]["path"]
    actual_blob = git_blob_sha(model_path)
    if actual_blob != contract["model_blob"]["blob_sha"]:
        raise AssertionError(
            f"model blob drift: {actual_blob} != {contract['model_blob']['blob_sha']}"
        )

    report = verify_synthetic_models()
    compare_expected(report["overwrite_model"], contract["overwrite_model_expected"])
    compare_expected(
        report["reversible_reference_model"],
        contract["reversible_reference_expected"],
    )

    scope = contract["scope"]
    assert scope["synthetic_only"] is True
    assert scope["family10h_backend_used"] is False
    assert scope["pmu_used"] is False
    assert scope["network_used"] is False
    assert scope["physical_realization_established"] is False

    kills = contract["kill_laws"]
    if not all(value is True for value in kills.values()):
        raise AssertionError("every I2C kill law must be active")

    decision = contract["i2c_decision"]
    assert decision["result"] == report["decision"]
    assert decision["overwrite_physical_candidate_promotable"] is False
    assert decision["reversible_reference_physical_candidate_promotable"] is False
    assert decision["synthetic_protocol_laws_qualified"] is True
    assert decision["live_execution_authorized"] is False
    assert decision["next_gate"] == "I2D_PHYSICAL_REVERSIBILITY_GAP_CONTRACT"

    return {
        "contract_id": contract["contract_id"],
        "model_blob": actual_blob,
        "overwrite_model": report["overwrite_model"],
        "reversible_reference_model": report["reversible_reference_model"],
        "independent_algebra_checks": validate_reversible_algebra(),
        "decision": decision["result"],
        "next_gate": decision["next_gate"],
        "claim_ceiling": contract["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(validate_contract(), indent=2, sort_keys=True))
