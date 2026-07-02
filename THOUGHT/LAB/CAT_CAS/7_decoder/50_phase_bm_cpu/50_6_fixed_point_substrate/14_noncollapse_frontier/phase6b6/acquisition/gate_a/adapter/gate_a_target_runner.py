#!/usr/bin/env python3
"""Target-side Gate A runner interface with no-drive qualification mode.

This runner is Git-free. It imports only gate_a_target_bundle and
gate_a_authority, both of which are packaged inside the deterministic archive.
It never imports the host-only Git-aware bundle builder, so an extracted bundle
can start it and validate itself without a .git repository.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import gate_a_authority
import gate_a_target_bundle as target_bundle

HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
WORKER = HERE / "gate_a_worker.c"
EXPECTED_SCHEDULE_SHA256 = target_bundle.SCHEDULE_SHA256
EXPECTED_NAMESPACE_SHA256 = target_bundle.NAMESPACE_SHA256
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


class RootState(enum.Enum):
    ABSENT = "absent"
    PRESENT = "present"
    UNOBSERVABLE = "unobservable"


OutputRootInspector = Callable[[Path], "RootState"]


def inspect_output_root(path: Path, *, stat_func: Callable[[Path], os.stat_result] = os.lstat) -> RootState:
    """Fail-closed inspection of a remote output-root path.

    Only a positively established missing path returns ABSENT. Any existing
    object (directory, regular file, symlink, or special file) returns PRESENT.
    Any inability to observe the path (permission denied, I/O error, not a
    directory in an ancestor, or any other OSError) returns UNOBSERVABLE.
    ``stat_func`` defaults to ``os.lstat`` so symlinks are not followed and the
    caller cannot be tricked by a link into a readable location; it is injectable
    only to exercise the OSError mapping deterministically in tests.
    """
    try:
        stat_func(path)
    except FileNotFoundError:
        return RootState.ABSENT
    except OSError:
        return RootState.UNOBSERVABLE
    return RootState.PRESENT


def require_output_root_absent(path: Path, *, inspector: OutputRootInspector = inspect_output_root) -> None:
    state = inspector(path)
    require(state is RootState.ABSENT, f"output root not provably absent (state={state.value})")


def locally_validated_manifest(bundle_root: Path, *, permitted_runtime_outputs: set[str] | None = None) -> dict[str, Any]:
    manifest = target_bundle.load_manifest(bundle_root)
    target_bundle.validate_extracted_bundle(bundle_root, manifest, strict=True, permitted_runtime_outputs=permitted_runtime_outputs)
    return manifest


def compile_worker(output: Path, extra_flags: list[str] | None = None) -> None:
    flags = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O2", "-pthread", str(WORKER), "-lm", "-o", str(output)]
    if extra_flags:
        flags[1:1] = extra_flags
    subprocess.run(flags, check=True)


def run_validate_only(executable: Path) -> dict[str, Any]:
    result = subprocess.run([str(executable), "--validate-only"], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)


def qualify_no_drive(bundle_root: Path = BUNDLE_ROOT, *, compile_c: bool = True, permitted_runtime_outputs: set[str] | None = None) -> dict[str, Any]:
    manifest = target_bundle.load_manifest(bundle_root)
    local = target_bundle.validate_extracted_bundle(bundle_root, manifest, strict=True, permitted_runtime_outputs=permitted_runtime_outputs)
    worker_payload: dict[str, Any] | None = None
    compiled = False
    if compile_c:
        with tempfile.TemporaryDirectory(prefix="gate_a_no_drive_") as tmp:
            exe = Path(tmp) / "gate_a_worker"
            compile_worker(exe)
            worker_payload = run_validate_only(exe)
        require(worker_payload["status"] == "GATE_A_WORKER_VALIDATE_ONLY_OK", "worker validation failed")
        compiled = True
    return {
        "status": "GATE_A_TARGET_RUNNER_NO_DRIVE_QUALIFIED",
        "null_baseline": "NO_DRIVE_ZERO_COUNT_BASELINE",
        "git_free": True,
        "local_bundle_validation": local,
        "compiled": compiled,
        "worker_validate_only": worker_payload,
        "network_connections_opened": 0,
        "hardware_probes": 0,
        "sender_starts": 0,
        "receiver_captures": 0,
        "control_writes": 0,
        "msr_accesses": 0,
        "hardware_executions": 0,
    }


def validate_authority(path: Path, digest: str, args: argparse.Namespace, exact_manifest: dict[str, Any]) -> dict[str, Any]:
    require(args.source_head, "source head is required")
    require(args.independent_review_id is not None, "independent review ID is required")
    require(args.execution_bundle_sha256 == exact_manifest["execution_bundle_sha256"], "bundle argument mismatch")
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
        exact_manifest=exact_manifest,
    )


def execute_authorized(
    args: argparse.Namespace,
    *,
    bundle_root: Path = BUNDLE_ROOT,
    output_root_inspector: OutputRootInspector = inspect_output_root,
    permitted_runtime_outputs: set[str] | None = None,
) -> None:
    require(args.authority_artifact and args.authority_sha256, "exact authority artifact and SHA-256 are required")
    require(args.execution_bundle_sha256, "execution bundle digest is required")
    require(args.source_head, "source head is required")
    require(args.independent_review_id is not None, "independent review ID is required")
    require(args.schedule_sha256, "schedule digest is required")
    require(args.target, "target is required")
    require(args.namespace_sha256, "namespace digest is required")
    require(args.output_root, "output root is required")
    exact_manifest = locally_validated_manifest(bundle_root, permitted_runtime_outputs=permitted_runtime_outputs)
    validate_authority(Path(args.authority_artifact), args.authority_sha256, args, exact_manifest)
    require_output_root_absent(Path(args.output_root), inspector=output_root_inspector)
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
    except (TargetRunnerError, target_bundle.TargetBundleError, gate_a_authority.AuthorityError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"gate_a_target_runner: {exc}", file=sys.stderr)
        raise SystemExit(1)
