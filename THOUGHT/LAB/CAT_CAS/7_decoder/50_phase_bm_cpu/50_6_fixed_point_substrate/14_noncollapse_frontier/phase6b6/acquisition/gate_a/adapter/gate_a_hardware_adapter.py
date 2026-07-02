#!/usr/bin/env python3
"""Gate A host adapter with default no-drive behavior."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import build_gate_a_execution_bundle as bundle
import gate_a_authority

BASE_MAIN = "03985a74d27e654c151cccad28c6221d91f70180"
REVIEWED_PLAN_HEAD = "65d20b4bc65ddd9260a3c90d92612d2da48763a6"
PLAN_REVIEW_ID = 4617290767
SCHEDULE_SHA256 = "418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116"
NAMESPACE_SHA256 = "5b3090f642d28492e182630e6349eccd8181704f08129d40d886c8f529dfd50e"
TARGET_IDENTITY_SHA256 = "10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4"
REMOTE_EXECUTION_ROOT = gate_a_authority.REMOTE_EXECUTION_ROOT
REMOTE_OUTPUT_ROOT = gate_a_authority.REMOTE_OUTPUT_ROOT
EXPECTED_SEQUENCE = ["I", "I", "I", "I", "C0", "D0", "S0E", "S0E", "S0E", "S0E", "O0", "O0", "A0P", "A0N", "T", "T"]

HERE = Path(__file__).resolve().parent
GATE_A = HERE.parent
SCHEDULE_PATH = GATE_A / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"
NAMESPACE_PATH = GATE_A / "GATE_A_TARGET_NAMESPACE_BINDING.json"
MANIFEST_PATH = HERE / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"
AUTHORITY_SCHEMA_PATH = HERE / "schemas" / "gate_a_execution_authority.schema.json"


class AdapterError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AdapterError(message)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_with_self_field_removed(value: dict[str, Any], field: str) -> str:
    unsigned = copy.deepcopy(value)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_bytes(unsigned))


def repo_root() -> Path:
    current = HERE
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    raise AdapterError("repository root not found")


def validate_schedule(schedule: dict[str, Any]) -> None:
    require(set(schedule) == {
        "architecture_integration",
        "base_main_commit",
        "expanded_slot_rule",
        "frequency_and_voltage",
        "preconditions",
        "schedule_sha256",
        "schema_id",
        "scientific_use",
        "session",
        "slot_definitions",
        "slot_sequence",
        "status",
        "stop_conditions",
        "target",
        "timing",
    }, "schedule top-level key set mismatch")
    require(schedule["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_SCHEDULE_V1", "schedule schema mismatch")
    require(schedule["base_main_commit"] == "9c41637992536f43d10d152ec176a3577aef1623", "schedule predecessor mismatch")
    require(schedule["schedule_sha256"] == SCHEDULE_SHA256, "schedule declared digest mismatch")
    require(digest_with_self_field_removed(schedule, "schedule_sha256") == SCHEDULE_SHA256, "schedule canonical digest mismatch")
    require(schedule["slot_sequence"] == EXPECTED_SEQUENCE, "slot sequence mismatch")
    require(schedule["target"]["ssh_target"] == "root@192.168.137.100", "target identity mismatch")
    require(schedule["target"]["target_identity_stdout_sha256"] == TARGET_IDENTITY_SHA256, "target identity digest mismatch")
    require(schedule["timing"] == {
        "automatic_retry": False,
        "maximum_execution_count": 1,
        "nominal_duration_s": 8.0,
        "nominal_samples_per_slot": 4000,
        "read_hz": 8000,
        "slot_count": 16,
        "slot_s": 0.5,
        "temperature_veto_c": 68.0,
    }, "timing mismatch")
    require(schedule["frequency_and_voltage"] == {
        "expected_observed_khz": 1600000,
        "frequency_write_authorized": False,
        "mismatch_action": "STOP_BEFORE_DRIVE",
        "msr_write_authorized": False,
        "voltage_write_authorized": False,
    }, "frequency and voltage boundary mismatch")
    require(all(value is False for value in schedule["scientific_use"].values()), "scientific use must be false")
    defs = schedule["slot_definitions"]
    require(set(defs) == {"I", "C0", "D0", "S0E", "O0", "A0P", "A0N", "T"}, "slot token set mismatch")
    for token in ("I", "C0", "D0", "O0", "T"):
        executed = defs[token]["executed"]
        require(executed["drive_on"] is False, f"{token} must not drive")
        for key in ("amplitude_level", "phase_action", "physical_tone_index", "sender_epoch_id", "sign"):
            require(executed[key] is None, f"{token} executed {key} must be null")
    require(defs["S0E"]["executed"]["sender_epoch_id"] == "gate-a:step:epoch0", "STEP sender epoch mismatch")
    require(defs["A0P"]["executed"]["sign"] == 1, "positive anchor mismatch")
    require(defs["A0N"]["executed"]["sign"] == -1, "negative anchor mismatch")


def validate_namespace(namespace: dict[str, Any]) -> None:
    require(set(namespace) == {
        "base_main_commit",
        "binding_sha256",
        "must_be_absent_before_deployment",
        "remote_execution_root",
        "remote_output_root",
        "remove_only_after_verified_copy_back",
        "schema_id",
        "status",
    }, "namespace top-level key set mismatch")
    require(namespace["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NAMESPACE_BINDING_V1", "namespace schema mismatch")
    require(namespace["binding_sha256"] == NAMESPACE_SHA256, "namespace declared digest mismatch")
    require(digest_with_self_field_removed(namespace, "binding_sha256") == NAMESPACE_SHA256, "namespace canonical digest mismatch")
    require(namespace["remote_execution_root"] == REMOTE_EXECUTION_ROOT, "remote execution root mismatch")
    require(namespace["remote_output_root"] == REMOTE_OUTPUT_ROOT, "remote output root mismatch")
    require(namespace["must_be_absent_before_deployment"] is True, "deployment absence guard missing")
    require(namespace["remove_only_after_verified_copy_back"] is True, "copy-back cleanup guard missing")
    require(namespace["status"] == "FROZEN__NO_EXECUTION_AUTHORITY", "namespace status mismatch")


def validate_bundle_manifest(manifest: dict[str, Any]) -> None:
    bundle.validate_committed_manifest_exact(manifest, "HEAD")


def validate_future_authority(
    authority: dict[str, Any],
    *,
    authority_sha256: str,
    authority_bytes: bytes,
    expected_reviewed_adapter_head: str,
    expected_independent_review_id: int,
    expected_manifest: dict[str, Any],
) -> dict[str, Any]:
    exact_manifest = bundle.validate_committed_manifest_exact(expected_manifest, "HEAD")
    return gate_a_authority.validate_execution_authority(
        authority,
        authority_sha256=authority_sha256,
        authority_bytes=authority_bytes,
        expected_reviewed_adapter_head=expected_reviewed_adapter_head,
        expected_independent_review_id=expected_independent_review_id,
        exact_manifest=exact_manifest,
    )


def load_authority(
    path: Path | None,
    digest: str | None,
    *,
    expected_reviewed_adapter_head: str | None,
    expected_independent_review_id: int | None,
    expected_manifest: dict[str, Any],
) -> dict[str, Any]:
    require(path is not None and digest is not None, "future execution authority artifact and SHA-256 are required")
    require(expected_reviewed_adapter_head is not None, "reviewed adapter head is required")
    require(expected_independent_review_id is not None, "independent review ID is required")
    exact_manifest = bundle.validate_committed_manifest_exact(expected_manifest, "HEAD")
    return gate_a_authority.load_and_validate_execution_authority(
        path,
        authority_sha256=digest,
        expected_reviewed_adapter_head=expected_reviewed_adapter_head,
        expected_independent_review_id=expected_independent_review_id,
        exact_manifest=exact_manifest,
    )


def render_operation_plan(schedule: dict[str, Any], namespace: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_OPERATION_PLAN_V1",
        "mode": "NO_DRIVE",
        "deployment_allowed": False,
        "connection_allowed": False,
        "sender_start_allowed": False,
        "capture_start_allowed": False,
        "control_write_allowed": False,
        "evidence_cleanup_allowed": False,
        "schedule_sha256": schedule["schedule_sha256"],
        "target_namespace_sha256": namespace["binding_sha256"],
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "slot_sequence": schedule["slot_sequence"],
        "remote_execution_root": namespace["remote_execution_root"],
        "remote_output_root": namespace["remote_output_root"],
        "copy_back_receipt_required_before_cleanup": True,
        "cleanup_after_verified_copy_required": True,
        "authority_artifact_created": False,
        "engineering_smoke_authorized": False,
        "hardware_ran": False,
    }


@dataclass(frozen=True)
class AdapterContext:
    schedule: dict[str, Any]
    namespace: dict[str, Any]
    manifest: dict[str, Any]


def load_context() -> AdapterContext:
    schedule = load_object(SCHEDULE_PATH)
    namespace = load_object(NAMESPACE_PATH)
    manifest = load_object(MANIFEST_PATH)
    validate_schedule(schedule)
    validate_namespace(namespace)
    validate_bundle_manifest(manifest)
    return AdapterContext(schedule=schedule, namespace=namespace, manifest=manifest)


def reject_without_authority(action: str, args: argparse.Namespace, context: AdapterContext) -> None:
    if not args.authority_artifact or not args.authority_sha256:
        raise AdapterError(f"{action} rejected: exact future authority artifact is required")
    load_authority(
        Path(args.authority_artifact),
        args.authority_sha256,
        expected_reviewed_adapter_head=args.reviewed_adapter_head,
        expected_independent_review_id=args.independent_review_id,
        expected_manifest=context.manifest,
    )
    raise AdapterError(f"{action} rejected: live transport is not executed in this qualification phase")


def qualify_no_drive(context: AdapterContext) -> dict[str, Any]:
    plan = render_operation_plan(context.schedule, context.namespace, context.manifest)
    return {
        "status": "GATE_A_ADAPTER_NO_DRIVE_QUALIFIED",
        "deployment_rejected_without_authority": True,
        "connection_rejected_without_authority": True,
        "sender_start_rejected_without_authority": True,
        "capture_start_rejected_without_authority": True,
        "control_write_rejected_without_authority": True,
        "cleanup_rejected_without_copy_back_receipt": True,
        "authority_artifact_created": False,
        "engineering_smoke_authorized": False,
        "hardware_ran": False,
        "operation_plan": plan,
        "transport": "NO_DRIVE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate A host adapter")
    parser.add_argument("--qualify-no-drive", action="store_true", help="validate and render no-drive qualification")
    parser.add_argument("--render-plan", action="store_true", help="render deterministic operation plan")
    parser.add_argument("--deploy", action="store_true", help="reject deployment unless exact future authority is supplied")
    parser.add_argument("--connect", action="store_true", help="reject connection unless exact future authority is supplied")
    parser.add_argument("--start-sender", action="store_true", help="reject sender start unless exact future authority is supplied")
    parser.add_argument("--start-capture", action="store_true", help="reject capture start unless exact future authority is supplied")
    parser.add_argument("--write-control", action="store_true", help="reject control write unless exact future authority is supplied")
    parser.add_argument("--cleanup-after-verified-copy", action="store_true", help="reject cleanup without exact receipt and authority")
    parser.add_argument("--authority-artifact", default=None)
    parser.add_argument("--authority-sha256", default=None)
    parser.add_argument("--reviewed-adapter-head", default=None)
    parser.add_argument("--independent-review-id", type=int, default=None)
    parser.add_argument("--copy-back-receipt", default=None)
    args = parser.parse_args(argv)

    context = load_context()
    if not any((args.qualify_no_drive, args.render_plan, args.deploy, args.connect, args.start_sender, args.start_capture, args.write_control, args.cleanup_after_verified_copy)):
        args.qualify_no_drive = True

    if args.render_plan:
        print(json.dumps(render_operation_plan(context.schedule, context.namespace, context.manifest), sort_keys=True, indent=2))
        return 0
    if args.qualify_no_drive:
        print(json.dumps(qualify_no_drive(context), sort_keys=True, indent=2))
        return 0
    if args.cleanup_after_verified_copy and not args.copy_back_receipt:
        raise AdapterError("evidence cleanup rejected: closed copy-back receipt is required")
    for flag, action in (
        (args.deploy, "deployment"),
        (args.connect, "connection"),
        (args.start_sender, "sender start"),
        (args.start_capture, "capture start"),
        (args.write_control, "control write"),
        (args.cleanup_after_verified_copy, "evidence cleanup"),
    ):
        if flag:
            reject_without_authority(action, args, context)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AdapterError, bundle.BundleError, gate_a_authority.AuthorityError, json.JSONDecodeError) as exc:
        print(f"gate_a_hardware_adapter: {exc}", file=sys.stderr)
        raise SystemExit(1)
