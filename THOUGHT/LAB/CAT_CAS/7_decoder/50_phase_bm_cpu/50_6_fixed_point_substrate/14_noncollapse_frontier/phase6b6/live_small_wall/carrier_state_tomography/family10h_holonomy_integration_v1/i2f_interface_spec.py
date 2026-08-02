#!/usr/bin/env python3
"""Validate the non-executing I2F physical interface specification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "I2F_TOPOLOGY_BACKEND_PROBE_CONTRACT.json"


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


def validate_source_identity(contract: dict[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for filename, expected in contract["source_blobs"].items():
        actual = git_blob_sha(ROOT / filename)
        if actual != expected:
            raise AssertionError(f"source drift for {filename}: {actual} != {expected}")
        verified[filename] = actual
    return verified


def validate_topology(contract: dict[str, Any]) -> dict[str, Any]:
    roles = contract["topology_roles"]
    core_fields = (
        "preparation_core",
        "home_reclaim_core",
        "remote_core_A",
        "remote_core_B",
        "route_control_core",
        "assignment_source",
    )
    if any(roles[field] is not None for field in core_fields):
        raise AssertionError("I2F may not assign physical topology")
    assert roles["assignment_frozen"] is False
    assert all(value is True for value in contract["topology_laws"].values())
    return {
        "role_count": 5,
        "assigned_role_count": 0,
        "assignment_frozen": False,
    }


def validate_carrier_and_source(contract: dict[str, Any]) -> dict[str, Any]:
    carrier = contract["carrier_layout"]
    assert carrier["persistent_single_allocation_required"] is True
    assert carrier["page_identity_manifest_required"] is True
    assert carrier["line_identity_manifest_required"] is True
    assert carrier["logical_initialization_answer_blind"] is True
    assert carrier["line_set_A"] is None
    assert carrier["line_set_B"] is None
    assert carrier["line_set_cardinality"] is None
    assert carrier["intersection_cardinality"] is None
    assert carrier["partial_overlap_required"] is True
    assert carrier["disjoint_control_required"] is True
    assert carrier["identical_support_control_required"] is True
    assert carrier["layout_frozen"] is False

    isolation = contract["source_isolation"]
    assert isolation["process_model"] == "exec_based_preparation_only_source_required"
    assert isolation["preparation_payload_contains_challenge"] is False
    assert isolation["preparation_payload_contains_operator_word"] is False
    assert isolation["preparation_payload_contains_accumulator_target"] is False
    assert isolation["challenge_entropy_generated_after_waitpid"] is True
    assert isolation["source_descriptors_closed_before_challenge"] is True
    assert isolation["source_ipc_zero_before_challenge"] is True
    assert isolation["source_helpers_zero_before_challenge"] is True
    assert isolation["fork_only_secrecy_allowed"] is False
    assert isolation["implementation_frozen"] is False

    return {
        "carrier_layout_frozen": False,
        "source_isolation_implementation_frozen": False,
        "fork_only_secrecy_rejected": True,
    }


def validate_backend_specs(contract: dict[str, Any]) -> dict[str, int]:
    backend = contract["backend_abi"]
    assert backend["implementation_status"] == "SPEC_ONLY_NOT_IMPLEMENTED"
    assert backend["physical_execution_enabled"] is False
    assert len(backend["required_methods"]) == 10
    assert len(backend["directional_operation_fields"]) == 17
    assert len(backend["required_result_classes"]) == 7

    probe = contract["nondestructive_probe_spec"]
    assert probe["implementation_status"] == "SPEC_ONLY_NOT_IMPLEMENTED"
    assert len(probe["required_observers"]) == 4
    assert len(probe["required_strata"]) == 4
    assert len(probe["required_channels"]) == 7
    assert probe["measurement_only_baselines_required"] is True
    assert probe["repeated_probe_baselines_required"] is True
    assert probe["probe_order_randomization_required"] is True
    assert probe["probe_disturbance_receipt_required"] is True
    assert probe["same_carrier_before_and_after_probe_required"] is True
    assert probe["disturbance_metric"] is None
    assert probe["disturbance_ceiling"] is None
    assert probe["measurement_deadline"] is None
    assert probe["numerical_fields_frozen"] is False

    accumulator = contract["accumulator_backend_spec"]
    assert accumulator["implementation_status"] == "SPEC_ONLY_NOT_IMPLEMENTED"
    assert all(
        value is True
        for key, value in accumulator.items()
        if key != "implementation_status"
    )

    environment = contract["environment_receipt_spec"]
    assert len(environment["required_fields"]) == 9
    assert environment["unclassified_field_is_hard_fail"] is True

    return {
        "backend_method_count": len(backend["required_methods"]),
        "directional_receipt_field_count": len(backend["directional_operation_fields"]),
        "probe_channel_count": len(probe["required_channels"]),
        "environment_field_count": len(environment["required_fields"]),
    }


def validate_decision(contract: dict[str, Any]) -> dict[str, Any]:
    blockers = contract["freeze_blockers"]
    if len(blockers) != 9 or len(set(blockers)) != 9:
        raise AssertionError("freeze blocker set drift")

    decision = contract["i2f_decision"]
    assert decision["result"] == (
        "I2F_TOPOLOGY_BACKEND_AND_NONDESTRUCTIVE_PROBE_SPEC_COMPLETE__"
        "ALL_PHYSICAL_ASSIGNMENTS_UNSET"
    )
    assert decision["interface_spec_complete"] is True
    assert decision["target_inventory_required_next"] is True
    assert decision["physical_package_freeze_ready"] is False
    assert decision["live_execution_authorized"] is False
    assert decision["next_gate"] == "I2G_READ_ONLY_TARGET_INVENTORY_AUTHORITY_REQUIRED"

    return {
        "freeze_blocker_count": len(blockers),
        "physical_package_freeze_ready": False,
        "next_gate": decision["next_gate"],
    }


def validate_i2f_spec() -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    return {
        "contract_id": contract["contract_id"],
        "source_identity_verified": validate_source_identity(contract),
        "topology": validate_topology(contract),
        "carrier_and_source": validate_carrier_and_source(contract),
        "backend_specs": validate_backend_specs(contract),
        "decision": validate_decision(contract),
        "claim_ceiling": contract["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(validate_i2f_spec(), indent=2, sort_keys=True))
