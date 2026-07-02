#!/usr/bin/env python3
"""Target-side Gate A runner interface with no-drive qualification mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import build_gate_a_execution_bundle as bundle
import gate_a_authority

HERE = Path(__file__).resolve().parent
WORKER = HERE / "gate_a_worker.c"
EXPECTED_SCHEDULE_SHA256 = "418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116"
EXPECTED_NAMESPACE_SHA256 = "5b3090f642d28492e182630e6349eccd8181704f08129d40d886c8f529dfd50e"
EXPECTED_TARGET = gate_a_authority.EXPECTED_TARGET


class TargetRunnerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetRunnerError(message)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def compile_worker(output: Path, extra_flags: list[str] | None = None) -> None:
    flags = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O2", "-pthread", str(WORKER), "-lm", "-o", str(output)]
    if extra_flags:
        flags[1:1] = extra_flags
    subprocess.run(flags, check=True)


def run_validate_only(executable: Path) -> dict[str, Any]:
    result = subprocess.run([str(executable), "--validate-only"], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)


def qualify_no_drive() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gate_a_no_drive_") as tmp:
        exe = Path(tmp) / "gate_a_worker"
        compile_worker(exe)
        payload = run_validate_only(exe)
    require(payload["status"] == "GATE_A_WORKER_VALIDATE_ONLY_OK", "worker validation failed")
    return {
        "status": "GATE_A_TARGET_RUNNER_NO_DRIVE_QUALIFIED",
        "null_baseline": "NO_DRIVE_ZERO_COUNT_BASELINE",
        "compiled": True,
        "worker_validate_only": payload,
        "network_connections_opened": 0,
        "hardware_probes": 0,
        "sender_starts": 0,
        "receiver_captures": 0,
        "control_writes": 0,
        "hardware_executions": 0,
    }


def validate_authority(path: Path, digest: str, args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_object(HERE / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json")
    require(args.source_head, "source head is required")
    require(args.independent_review_id is not None, "independent review ID is required")
    require(args.execution_bundle_sha256 == manifest["execution_bundle_sha256"], "bundle argument mismatch")
    require(args.schedule_sha256 == EXPECTED_SCHEDULE_SHA256, "schedule argument mismatch")
    require(args.target == EXPECTED_TARGET, "target argument mismatch")
    require(args.namespace_sha256 == EXPECTED_NAMESPACE_SHA256, "namespace argument mismatch")
    require(args.output_root == gate_a_authority.REMOTE_OUTPUT_ROOT, "output root argument mismatch")
    authority, authority_bytes = gate_a_authority.load_json_object_bytes(path)
    return gate_a_authority.validate_execution_authority(
        authority,
        authority_sha256=digest,
        authority_bytes=authority_bytes,
        expected_reviewed_adapter_head=args.source_head,
        expected_independent_review_id=args.independent_review_id,
        expected_manifest=manifest,
    )


def execute_authorized(args: argparse.Namespace) -> None:
    require(args.authority_artifact and args.authority_sha256, "exact authority artifact and SHA-256 are required")
    require(args.execution_bundle_sha256, "execution bundle digest is required")
    require(args.source_head, "source head is required")
    require(args.independent_review_id is not None, "independent review ID is required")
    require(args.schedule_sha256, "schedule digest is required")
    require(args.target, "target is required")
    require(args.namespace_sha256, "namespace digest is required")
    require(args.output_root, "output root is required")
    output_root = Path(args.output_root)
    require(not output_root.exists(), "existing output root rejected")
    validate_authority(Path(args.authority_artifact), args.authority_sha256, args)
    raise TargetRunnerError("authorized live execution path is intentionally unused in this qualification phase")


def cleanup_after_verified_copy(args: argparse.Namespace) -> None:
    require(args.copy_back_receipt, "closed copy-back receipt is required")
    receipt = load_object(Path(args.copy_back_receipt))
    require(receipt.get("schema_id") == "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1", "copy-back receipt schema mismatch")
    require(receipt.get("retained_evidence_custody_verified") is True, "retained-evidence custody not verified")
    raise TargetRunnerError("cleanup is unavailable in no-drive qualification")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate A target runner")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--qualify-no-drive", action="store_true")
    modes.add_argument("--probe-only", action="store_true")
    modes.add_argument("--execute-authorized", action="store_true")
    modes.add_argument("--cleanup-after-verified-copy", action="store_true")
    parser.add_argument("--authority-artifact")
    parser.add_argument("--authority-sha256")
    parser.add_argument("--execution-bundle-sha256")
    parser.add_argument("--source-head")
    parser.add_argument("--independent-review-id", type=int)
    parser.add_argument("--schedule-sha256")
    parser.add_argument("--target")
    parser.add_argument("--namespace-sha256")
    parser.add_argument("--output-root")
    parser.add_argument("--copy-back-receipt")
    args = parser.parse_args(argv)

    if args.qualify_no_drive:
        print(json.dumps(qualify_no_drive(), sort_keys=True, indent=2))
        return 0
    if args.probe_only:
        raise TargetRunnerError("probe-only source exists but is not run during this phase")
    if args.execute_authorized:
        execute_authorized(args)
    if args.cleanup_after_verified_copy:
        cleanup_after_verified_copy(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TargetRunnerError, bundle.BundleError, gate_a_authority.AuthorityError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"gate_a_target_runner: {exc}", file=sys.stderr)
        raise SystemExit(1)
