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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import gate_a_authority
import gate_a_engineering_smoke_executor as smoke_executor
import gate_a_target_bundle as target_bundle

HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
WORKER = HERE / "gate_a_worker.c"
SCHEDULE = BUNDLE_ROOT / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"
EXPECTED_SCHEDULE_SHA256 = target_bundle.SCHEDULE_SHA256
EXPECTED_NAMESPACE_SHA256 = target_bundle.NAMESPACE_SHA256
EXPECTED_TARGET = gate_a_authority.EXPECTED_TARGET
TRANSPORT_CLAIM_SCHEMA = "CAT_CAS_PHASE6B6_GATE_A_TRANSPORT_CLAIM_V1"


class TargetRunnerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetRunnerError(message)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_transport_claim_root(authority_sha256: str) -> Path:
    require(len(authority_sha256) == 64 and all(c in "0123456789abcdef" for c in authority_sha256), "transport claim authority digest malformed")
    return Path(f"/root/.catcas_gate_a_claim_{authority_sha256}")


def expected_transport_claim(authority_sha256: str, execution_bundle_sha256: str) -> dict[str, Any]:
    return {
        "schema_id": TRANSPORT_CLAIM_SCHEMA,
        "authority_sha256": authority_sha256,
        "execution_bundle_sha256": execution_bundle_sha256,
        "maximum_execution_count": 1,
        "automatic_retry": False,
    }


def expected_execution_started(authority_sha256: str, execution_bundle_sha256: str) -> dict[str, Any]:
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_STARTED_V1",
        "authority_sha256": authority_sha256,
        "execution_bundle_sha256": execution_bundle_sha256,
        "runtime_execution_count": 1,
        "automatic_retry": False,
    }


def validate_transport_claim_and_mark_started(
    claim_root: Path,
    *,
    authority_sha256: str,
    execution_bundle_sha256: str,
) -> dict[str, Any]:
    require(claim_root == expected_transport_claim_root(authority_sha256), "transport claim root mismatch")
    require(claim_root.is_dir() and not claim_root.is_symlink(), "transport claim root is not a real directory")
    claim_path = claim_root / "CLAIM.json"
    claim = load_object(claim_path)
    expected_claim = expected_transport_claim(authority_sha256, execution_bundle_sha256)
    require(set(claim) == set(expected_claim), "transport claim key set mismatch")
    require(claim == expected_claim, "transport claim binding mismatch")
    marker_path = claim_root / "EXECUTION_STARTED.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(marker_path, flags, 0o600)
    except FileExistsError as exc:
        raise TargetRunnerError("transport claim has already started execution") from exc
    marker = expected_execution_started(authority_sha256, execution_bundle_sha256)
    payload = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        smoke_executor._write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(claim_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "claim_sha256": _sha256_file(claim_path),
        "execution_started_sha256": _sha256_file(marker_path),
    }


def validate_retained_transport_claim(
    claim_root: Path,
    *,
    authority_sha256: str,
    execution_bundle_sha256: str,
) -> dict[str, str]:
    require(claim_root == expected_transport_claim_root(authority_sha256), "retained transport claim root mismatch")
    require(claim_root.is_dir() and not claim_root.is_symlink(), "retained transport claim root is not a real directory")
    claim_path = claim_root / "CLAIM.json"
    marker_path = claim_root / "EXECUTION_STARTED.json"
    claim_sha256 = _sha256_file(claim_path)
    marker_sha256 = _sha256_file(marker_path)
    claim = load_object(claim_path)
    marker = load_object(marker_path)
    expected_claim = expected_transport_claim(authority_sha256, execution_bundle_sha256)
    expected_marker = expected_execution_started(authority_sha256, execution_bundle_sha256)
    require(set(claim) == set(expected_claim) and claim == expected_claim, "retained transport claim binding mismatch")
    require(set(marker) == set(expected_marker) and marker == expected_marker, "execution-start marker binding mismatch")
    return {
        "claim_sha256": claim_sha256,
        "execution_started_sha256": marker_sha256,
    }


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


def runtime_source_root() -> Path:
    packaged = BUNDLE_ROOT / "runtime"
    if packaged.is_dir():
        return packaged
    source = HERE.parents[3] / "holo_runtime_v2"
    require(source.is_dir(), "holo_runtime_v2 source root missing")
    return source


def compile_worker(
    output: Path,
    extra_flags: list[str] | None = None,
    *,
    authority_sha256: str | None = None,
    authorized_output_root: Path | None = None,
    temp_dir: Path | None = None,
) -> None:
    runtime = runtime_source_root()
    flags = [
        "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O2",
        "-pthread", "-I", str(runtime), str(WORKER),
        str(runtime / "gate_a_engineering_smoke_runtime.c"), str(runtime / "captured_file.c"),
        "-lm", "-o", str(output),
    ]
    if authority_sha256 is not None:
        require(len(authority_sha256) == 64 and all(c in "0123456789abcdef" for c in authority_sha256), "compiled worker authority digest malformed")
        flags.insert(1, f'-DGATE_A_COMPILED_AUTHORITY_SHA256="{authority_sha256}"')
        require(authorized_output_root is not None and authorized_output_root.is_absolute(), "compiled worker output binding missing")
        flags.insert(2, f'-DGATE_A_COMPILED_OUTPUT_ROOT="{authorized_output_root}"')
    else:
        require(authorized_output_root is None, "unbound worker cannot carry an output binding")
    if extra_flags:
        flags[1:1] = extra_flags
    env = os.environ.copy()
    if temp_dir is not None:
        require(temp_dir.is_dir() and not temp_dir.is_symlink(), "compiler temporary directory is not a real directory")
        env["TMPDIR"] = str(temp_dir)
    subprocess.run(flags, check=True, env=env)


class BoundWorkerRuntime:
    """Compile and invoke a worker bound to the exact claimed authority."""

    def __init__(self):
        self.calls = 0
        self._worker: smoke_executor.WorkerRuntime | None = None

    def execute(self, plan: smoke_executor.FrozenPlan) -> dict[str, Any]:
        require(self.calls == 0, "bound worker runtime may execute only once")
        self.calls += 1
        build_root = plan.output_root / "worker_build"
        build_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        temp_dir = build_root / "tmp"
        temp_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        executable = build_root / "gate_a_worker"
        compile_worker(
            executable,
            authority_sha256=plan.authority_sha256,
            authorized_output_root=plan.output_root / "runtime",
            temp_dir=temp_dir,
        )
        self._worker = smoke_executor.WorkerRuntime(executable)
        return self._worker.execute(plan)

    def verify_evidence(self, plan: smoke_executor.FrozenPlan, result: dict[str, Any]) -> None:
        require(self._worker is not None, "bound worker evidence verifier called before execution")
        self._worker.verify_evidence(plan, result)


def run_validate_only(executable: Path) -> dict[str, Any]:
    result = subprocess.run([str(executable), "--validate-only"], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)


def _mock_slot_record(index: int, token: str) -> dict[str, Any]:
    driven = token in {"S0E", "A0P", "A0N"}
    phase = 4 if token == "A0N" else (0 if driven else None)
    sign = -1 if token == "A0N" else (1 if driven else None)
    epoch = None
    if token == "S0E":
        epoch = "gate-a:step:epoch0"
    elif token == "A0P":
        epoch = "gate-a:anchor:positive"
    elif token == "A0N":
        epoch = "gate-a:anchor:negative"
    return {
        "index": index,
        "token": token,
        "requested_start_s": index * 0.5,
        "requested_end_s": (index + 1) * 0.5,
        "drive_on": driven,
        "amplitude_level": 2 if driven else None,
        "phase_index": phase,
        "sign": sign,
        "sender_epoch_id": epoch,
    }


def run_mock_self_test(executable: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    runtime_root = evidence_root / "runtime"
    result = subprocess.run(
        [str(executable), "--self-test-retain", str(runtime_root)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(result.stdout)
    retained = load_object(runtime_root / "runtime_result.json")
    runtime_result = {
        "status": "GATE_A_ENGINEERING_SMOKE_COMPLETE",
        "automatic_retry": False,
        "runtime_execution_count": 1,
        "slot_records": [
            _mock_slot_record(index, token)
            for index, token in enumerate(smoke_executor.SEQUENCE)
        ],
        "capture": {
            "continuous": True,
            "covers_complete_sequence": True,
            "sample_count": retained["sample_count"],
            "slot_sample_counts": [4000] * 16,
            "origin_tsc": retained["capture_origin_tsc"],
            "deadline_tsc": retained["capture_deadline_tsc"],
            "first_sample_tsc": retained["capture_first_sample_tsc"],
            "last_sample_tsc": retained["capture_last_sample_tsc"],
            "tsc_hz": retained["capture_tsc_hz"],
        },
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "step_sender_epoch_count": 1,
        "hardware_executed": False,
    }
    plan = smoke_executor.FrozenPlan(
        authority_sha256="a" * 64,
        execution_bundle_sha256="b" * 64,
        output_root=evidence_root,
    )
    smoke_executor.verify_retained_runtime_evidence(plan, runtime_result)
    payload["retained_raw_lockin_and_lifecycle_verified"] = True
    return payload


def prove_unbound_worker_rejects_live(executable: Path, output_root: Path) -> bool:
    result = subprocess.run([
        str(executable), "--execute-authorized",
        "--authority-sha256", "a" * 64,
        "--schedule-sha256", EXPECTED_SCHEDULE_SHA256,
        "--execution-bundle-sha256", "b" * 64,
        "--output-root", str(output_root),
        "--sender-core", "4",
        "--receiver-core", "5",
        "--read-hz", "8000",
        "--slot-s", "0.5",
        "--temperature-veto-c", "68.0",
        "--required-frequency-khz", "1600000",
    ], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode != 0, "ordinary worker build accepted live execution")
    require(not output_root.exists() and not output_root.is_symlink(), "unbound worker created runtime output")
    return True


def qualify_no_drive(bundle_root: Path = BUNDLE_ROOT, *, compile_c: bool = True, permitted_runtime_outputs: set[str] | None = None) -> dict[str, Any]:
    manifest = target_bundle.load_manifest(bundle_root)
    local = target_bundle.validate_extracted_bundle(bundle_root, manifest, strict=True, permitted_runtime_outputs=permitted_runtime_outputs)
    worker_payload: dict[str, Any] | None = None
    self_test_payload: dict[str, Any] | None = None
    direct_live_rejected = False
    compiled = False
    if compile_c:
        with tempfile.TemporaryDirectory(prefix="gate_a_no_drive_") as tmp:
            exe = Path(tmp) / "gate_a_worker"
            compile_worker(exe)
            worker_payload = run_validate_only(exe)
            self_test_payload = run_mock_self_test(exe, Path(tmp) / "mock_evidence")
            direct_live_rejected = prove_unbound_worker_rejects_live(exe, Path(tmp) / "unbound_runtime")
        require(worker_payload["status"] == "GATE_A_WORKER_VALIDATE_ONLY_OK", "worker validation failed")
        require(worker_payload["live_execution_bound"] is False, "ordinary qualification worker unexpectedly authority-bound")
        require(self_test_payload["status"] == "GATE_A_WORKER_MOCK_SELF_TEST_OK", "worker mock self-test failed")
        require(self_test_payload["retained_raw_lockin_and_lifecycle_verified"] is True, "worker raw lock-in/lifecycle custody verification failed")
        compiled = True
    return {
        "status": "GATE_A_TARGET_RUNNER_NO_DRIVE_QUALIFIED",
        "null_baseline": "NO_DRIVE_ZERO_COUNT_BASELINE",
        "git_free": True,
        "local_bundle_validation": local,
        "compiled": compiled,
        "worker_validate_only": worker_payload,
        "worker_mock_self_test": self_test_payload,
        "ordinary_worker_live_execution_rejected": direct_live_rejected,
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
    surfaces: smoke_executor.ExecutionSurfaces | None = None,
    compile_c: bool = True,
) -> dict[str, Any]:
    require(args.authority_artifact and args.authority_sha256, "exact authority artifact and SHA-256 are required")
    require(args.execution_bundle_sha256, "execution bundle digest is required")
    require(args.source_head, "source head is required")
    require(args.independent_review_id is not None, "independent review ID is required")
    require(args.schedule_sha256, "schedule digest is required")
    require(args.target, "target is required")
    require(args.namespace_sha256, "namespace digest is required")
    require(args.output_root, "output root is required")
    exact_manifest = locally_validated_manifest(bundle_root, permitted_runtime_outputs=permitted_runtime_outputs)
    authority_validation = validate_authority(Path(args.authority_artifact), args.authority_sha256, args, exact_manifest)
    require_output_root_absent(Path(args.output_root), inspector=output_root_inspector)
    schedule = load_object(bundle_root / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json")
    smoke_executor.validate_frozen_schedule(schedule)
    output_root = Path(args.output_root)

    if surfaces is not None:
        return smoke_executor.execute_once(
            authority_validation=authority_validation,
            authority_sha256=args.authority_sha256,
            execution_bundle_sha256=exact_manifest["execution_bundle_sha256"],
            schedule=schedule,
            output_root=output_root,
            surfaces=surfaces,
        )

    require(compile_c, "production execution requires the packaged C worker")
    require(args.transport_claim_root, "durable transport claim root is required")
    validate_transport_claim_and_mark_started(
        Path(args.transport_claim_root),
        authority_sha256=args.authority_sha256,
        execution_bundle_sha256=exact_manifest["execution_bundle_sha256"],
    )
    claim_root = Path(args.claim_root or "/root/.catcas_phase6b6_gate_a_claims")
    live_surfaces = smoke_executor.ExecutionSurfaces(
        preflight=smoke_executor.LocalPreflight(),
        claims=smoke_executor.FileClaimStore(claim_root),
        evidence=smoke_executor.JsonEvidenceStore(output_root),
        runtime=BoundWorkerRuntime(),
    )
    return smoke_executor.execute_once(
        authority_validation=authority_validation,
        authority_sha256=args.authority_sha256,
        execution_bundle_sha256=exact_manifest["execution_bundle_sha256"],
        schedule=schedule,
        output_root=output_root,
        surfaces=live_surfaces,
    )


def evidence_inventory_sha256(root: Path) -> str:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TargetRunnerError(f"evidence inventory contains symlink: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"evidence inventory contains non-file: {path}")
        data = path.read_bytes()
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    require(files, "evidence inventory is empty")
    inventory = {"schema_id": "CAT_CAS_PHASE6B6_GATE_A_EVIDENCE_INVENTORY_V1", "files": files}
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def cleanup_after_verified_copy(
    args: argparse.Namespace,
    *,
    remove_tree: Callable[[Path], None] = shutil.rmtree,
    expected_output_root: str = gate_a_authority.REMOTE_OUTPUT_ROOT,
) -> dict[str, Any]:
    require(args.copy_back_receipt, "closed copy-back receipt is required")
    require(args.output_root, "output root is required")
    require(args.output_root == expected_output_root, "cleanup output root is not the frozen namespace")
    require(args.authority_sha256, "cleanup authority digest is required")
    require(args.execution_bundle_sha256, "cleanup bundle digest is required")
    require(getattr(args, "transport_claim_root", None), "retained transport claim root is required")
    receipt = load_object(Path(args.copy_back_receipt))
    require(set(receipt) == {
        "schema_id", "remote_output_root", "retained_evidence_custody_verified",
        "evidence_inventory_sha256", "copy_back_complete", "authority_sha256",
        "execution_bundle_sha256", "target_evidence_inventory_sha256",
        "downloaded_evidence_inventory_sha256", "archive_sha256",
    }, "copy-back receipt key set mismatch")
    require(receipt["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1", "copy-back receipt schema mismatch")
    require(receipt["remote_output_root"] == args.output_root, "copy-back receipt output-root mismatch")
    require(receipt["authority_sha256"] == args.authority_sha256, "copy-back receipt authority mismatch")
    require(receipt["execution_bundle_sha256"] == args.execution_bundle_sha256, "copy-back receipt bundle mismatch")
    require(receipt["retained_evidence_custody_verified"] is True, "retained-evidence custody not verified")
    require(receipt["copy_back_complete"] is True, "copy-back is incomplete")
    require(receipt["target_evidence_inventory_sha256"] == receipt["downloaded_evidence_inventory_sha256"], "target/downloaded inventory mismatch")
    require(receipt["target_evidence_inventory_sha256"] == receipt["evidence_inventory_sha256"], "copy-back inventory binding mismatch")
    archive_sha256 = receipt["archive_sha256"]
    require(isinstance(archive_sha256, str) and len(archive_sha256) == 64 and all(c in "0123456789abcdef" for c in archive_sha256), "archive digest malformed")
    digest = receipt["evidence_inventory_sha256"]
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "evidence inventory digest malformed")
    output_root = Path(args.output_root)
    require(output_root.exists() and output_root.is_dir() and not output_root.is_symlink(), "output root is not a real directory")
    attempt_path = output_root / "ATTEMPT.json"
    require(attempt_path.is_file() and not attempt_path.is_symlink(), "retained attempt binding is missing")
    attempt = load_object(attempt_path)
    require(attempt.get("authority_sha256") == args.authority_sha256, "retained attempt authority mismatch")
    require(attempt.get("execution_bundle_sha256") == args.execution_bundle_sha256, "retained attempt bundle mismatch")
    require(evidence_inventory_sha256(output_root) == digest, "copy-back receipt inventory does not match retained evidence")
    retained_claim = validate_retained_transport_claim(
        Path(args.transport_claim_root),
        authority_sha256=args.authority_sha256,
        execution_bundle_sha256=args.execution_bundle_sha256,
    )
    remove_tree(output_root)
    require(not output_root.exists(), "output root cleanup not verified")
    retained_claim_after = validate_retained_transport_claim(
        Path(args.transport_claim_root),
        authority_sha256=args.authority_sha256,
        execution_bundle_sha256=args.execution_bundle_sha256,
    )
    require(retained_claim_after == retained_claim, "retained transport claim changed during cleanup")
    return {
        "status": "GATE_A_CLEANUP_COMPLETE_AFTER_VERIFIED_COPY_BACK",
        "remote_output_root": args.output_root,
        "claim_retained": True,
        "claim_sha256": retained_claim["claim_sha256"],
        "execution_started_sha256": retained_claim["execution_started_sha256"],
    }


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
    parser.add_argument("--claim-root")
    parser.add_argument("--copy-back-receipt")
    parser.add_argument("--transport-claim-root")
    args = parser.parse_args(argv)

    if args.qualify_no_drive:
        print(json.dumps(qualify_no_drive(), sort_keys=True, indent=2))
        return 0
    if args.probe_only:
        raise TargetRunnerError("probe-only source exists but is not run during this phase")
    if args.execute_authorized:
        print(json.dumps(execute_authorized(args), sort_keys=True, indent=2))
    if args.cleanup_after_verified_copy:
        print(json.dumps(cleanup_after_verified_copy(args), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TargetRunnerError, target_bundle.TargetBundleError, gate_a_authority.AuthorityError, smoke_executor.ExecutorError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"gate_a_target_runner: {exc}", file=sys.stderr)
        raise SystemExit(1)
