#!/usr/bin/env python3
"""Validate the compile-only post-source operator runtime interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
INTERFACE_PATH = PACKAGE_ROOT / "I2A_RUNTIME_INTERFACE.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def git_blob_sha(path: Path) -> str:
    """Return the canonical Git blob SHA-1 for a file."""

    payload = path.read_bytes()
    return hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def require_markers(path: Path, *markers: str) -> None:
    """Require literal contract markers in one implementation file."""

    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"missing markers in {path}: {missing}")


def validate_interface() -> dict[str, Any]:
    """Validate source identity, lifecycle ordering, and fail-closed claim limits."""

    interface = load_json(INTERFACE_PATH)
    header = PACKAGE_ROOT / "post_source_operator_runtime.h"
    source = PACKAGE_ROOT / "post_source_operator_runtime.c"

    actual_blobs = {
        header.name: git_blob_sha(header),
        source.name: git_blob_sha(source),
    }
    if actual_blobs != interface["implementation_blobs"]:
        raise AssertionError(
            f"implementation blob drift: {actual_blobs} != {interface['implementation_blobs']}"
        )

    require_markers(
        header,
        "F10HI_PHASE_SOURCE_DEAD_SEALED",
        "F10HI_PHASE_RECEIVER_WORD_OPEN",
        "F10HI_OP_REMOTE_STORE_SAME_VALUE_EXTRACTION_TARGET",
        "F10HI_OP_QUERY_PROBE",
        "F10HI_OP_DESTRUCTIVE_RESET",
        "F10HI_OP_UNRESOLVED_SECOND_GENERATOR",
        "F10HI_OP_UNQUALIFIED_INVERSE",
        "challenge_selected_before_source_death",
        "f10hi_apply_backend_fn",
    )
    require_markers(
        source,
        "runtime->phase != F10HI_PHASE_SOURCE_PREPARED",
        "source_death->source_alive",
        "source_death->open_source_ipc != 0u",
        "source_death->source_helper_count != 0u",
        "source_death->challenge_selected_before_source_death",
        "runtime->phase != F10HI_PHASE_SOURCE_DEAD_SEALED",
        "!f10hi_operator_is_admissible_extraction_target(operator_spec)",
        "operator_spec->inverse_of_instance_id != 0u",
        "F10HI_OP_QUERY_PROBE",
        "F10HI_OP_DESTRUCTIVE_RESET",
        "F10HI_OP_UNRESOLVED_SECOND_GENERATOR",
        "physical_backend_implemented\\\":false",
        "h1_generator_pair_established\\\":false",
        "live_execution_authorized\\\":false",
    )

    expected_lifecycle = [
        "ALLOCATED",
        "SOURCE_PREPARED",
        "SOURCE_DEAD_SEALED",
        "RECEIVER_WORD_OPEN",
        "RECEIVER_WORD_CLOSED",
        "RESTORATION_RECORDED",
        "DESTROYED",
    ]
    if interface["lifecycle"] != expected_lifecycle:
        raise AssertionError("lifecycle order drift")

    acceptance = interface["acceptance"]
    required_true = {
        "persistent_carrier_interface",
        "source_death_before_word_enforced",
        "receiver_word_after_seal_enforced",
        "readout_as_generator_rejected",
        "destructive_reset_as_inverse_rejected",
        "unresolved_second_generator_rejected",
        "unqualified_inverse_rejected",
        "synthetic_backend_explicitly_labeled",
    }
    if not all(acceptance[key] is True for key in required_true):
        raise AssertionError("one or more I2A acceptance controls are not true")
    for key in (
        "physical_backend_present",
        "h1_generator_established",
        "h1_generator_pair_established",
    ):
        if acceptance[key] is not False:
            raise AssertionError(f"forbidden positive acceptance field: {key}")

    target = interface["operator_roles"]["REMOTE_STORE_SAME_VALUE_EXTRACTION_TARGET"]
    assert target["physical_generator_established"] is False
    assert target["inverse_established"] is False
    assert interface["operator_roles"]["QUERY_PROBE"]["may_enter_receiver_word"] is False
    assert interface["operator_roles"]["DESTRUCTIVE_RESET"]["may_be_inverse"] is False
    assert (
        interface["operator_roles"]["UNRESOLVED_SECOND_GENERATOR"]["may_enter_receiver_word"]
        is False
    )

    decision = interface["i2a_decision"]
    assert decision["result"] == (
        "I2A_COMPILE_ONLY_POST_SOURCE_RUNTIME_INTERFACE_ESTABLISHED__PHYSICAL_BACKEND_NOT_IMPLEMENTED"
    )
    assert decision["interface_freeze_allowed"] is True
    assert decision["physical_transport_candidate_freeze_allowed"] is False
    assert decision["live_execution_authorized"] is False

    return {
        "interface_id": interface["interface_id"],
        "implementation_blobs": actual_blobs,
        "lifecycle": expected_lifecycle,
        "compile_flags": interface["compile_contract"]["flags"],
        "decision": decision["result"],
        "next_gate": decision["next_gate"],
        "claim_ceiling": interface["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(validate_interface(), indent=2, sort_keys=True))
