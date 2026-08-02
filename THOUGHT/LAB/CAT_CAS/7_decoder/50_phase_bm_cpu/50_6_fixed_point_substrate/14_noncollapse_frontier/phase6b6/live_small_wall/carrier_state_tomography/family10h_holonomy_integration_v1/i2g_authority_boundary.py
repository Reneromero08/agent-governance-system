#!/usr/bin/env python3
"""Validate the ungranted I2G read-only target inventory authority boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "I2G_READ_ONLY_INVENTORY_AUTHORITY.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def validate_authority_boundary() -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    source = contract["source_blob"]
    actual = git_blob_sha(ROOT / source["path"])
    if actual != source["blob_sha"]:
        raise AssertionError(f"I2F source drift: {actual} != {source['blob_sha']}")

    authority = contract["current_authority"]
    if any(value is not False for value in authority.values()):
        raise AssertionError("no I2G authority field may currently be granted")

    required = contract["required_before_contact"]
    if any(value is not None for value in required.values()):
        raise AssertionError("pre-contact fields must remain unset")

    scope = contract["proposed_read_only_scope"]
    forbidden = contract["forbidden_actions"]
    outputs = contract["inventory_output_fields"]
    if len(scope) != 10 or len(set(scope)) != 10:
        raise AssertionError("read-only scope drift")
    if len(forbidden) != 10 or len(set(forbidden)) != 10:
        raise AssertionError("forbidden-action set drift")
    if len(outputs) != 19 or len(set(outputs)) != 19:
        raise AssertionError("inventory output schema drift")

    acceptance = contract["acceptance"]
    assert acceptance == {
        "authority_boundary_complete": True,
        "current_authority_granted": False,
        "target_contact_performed": False,
        "write_attempt_count": 0,
        "scientific_measurement_count": 0,
        "physical_package_freeze_ready": False,
    }

    decision = contract["i2g_decision"]
    assert decision["result"] == (
        "I2G_READ_ONLY_TARGET_INVENTORY_AUTHORITY_CONTRACT_COMPLETE__"
        "AUTHORITY_NOT_GRANTED"
    )
    assert decision["github_only_work_complete"] is True
    assert decision["read_only_target_inventory_blocked"] is True
    assert decision["physical_package_freeze_ready"] is False
    assert decision["live_execution_authorized"] is False
    assert decision["next_gate"] == "I2G_AUTHORITY_GRANT_AND_TARGET_ACCESS_REQUIRED"

    return {
        "authority_id": contract["authority_id"],
        "source_blob_verified": actual,
        "read_only_scope_count": len(scope),
        "forbidden_action_count": len(forbidden),
        "inventory_output_field_count": len(outputs),
        "authority_granted": False,
        "target_contact_performed": False,
        "write_attempt_count": 0,
        "scientific_measurement_count": 0,
        "decision": decision["result"],
        "next_gate": decision["next_gate"],
        "claim_ceiling": contract["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(validate_authority_boundary(), indent=2, sort_keys=True))
