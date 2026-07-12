#!/usr/bin/env python3
"""Target-side one-shot runner for authority-gated frequency preparation.

The runner is Git-free.  It validates the extracted payload and exact authority,
consumes an authority-specific durable claim, re-observes target identity, performs
one bounded pin/observe/restore transaction, and seals a local evidence packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Callable

import gate_a_frequency_preparation_authority as authority
import gate_a_frequency_preparation_bundle as target_bundle
import gate_a_frequency_preparation_live as live

CLAIM_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_CLAIM_V1"
STARTED_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_STARTED_V1"
RESULT_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_TARGET_RESULT_V1"
INVENTORY_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_EVIDENCE_INVENTORY_V1"
MAX_EVIDENCE_FILE_BYTES = 8 * 1024 * 1024


class TargetError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_durable(path: Path, data: bytes, *, exclusive: bool = True) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            require(written > 0, f"short evidence write: {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    write_durable(path, canonical_line(value), exclusive=exclusive)


def read_regular(path: Path, *, limit: int = 1024 * 1024) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    data = path.read_bytes()
    require(len(data) <= limit, f"file exceeds closed limit: {path}")
    return data


def read_sysfs(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            total += len(chunk)
            require(total <= 4096, f"sysfs value exceeds limit: {path}")
            chunks.append(chunk)
        data = b"".join(chunks)
        require(data != b"", f"sysfs value empty: {path}")
        return data
    finally:
        os.close(fd)


def write_sysfs(path: Path, data: bytes) -> None:
    require(data.endswith(b"\n") and data[:-1].isdigit(), "noncanonical sysfs write payload")
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            require(written > 0, f"short sysfs write: {path}")
            view = view[written:]
    finally:
        os.close(fd)


def observe_target_identity(
    *,
    hostname_func: Callable[[], str] = socket.gethostname,
    machine_func: Callable[[], str] = platform.machine,
    cpuinfo_path: Path = Path("/proc/cpuinfo"),
) -> dict[str, str]:
    hostname = hostname_func().strip()
    architecture = machine_func().strip()
    cpu_model = ""
    for line in cpuinfo_path.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith("model name") and ":" in line:
            cpu_model = line.split(":", 1)[1].strip()
            break
    require(hostname != "" and architecture != "" and cpu_model != "", "target identity incomplete")
    return {"hostname": hostname, "architecture": architecture, "cpu_model": cpu_model}


def expected_claim(permit: authority.PreparationPermit) -> dict[str, Any]:
    return {
        "schema_id": CLAIM_SCHEMA_ID,
        "authority_id": permit.authority_id,
        "authority_sha256": permit.authority_sha256,
        "bundle_sha256": permit.bundle_sha256,
        "maximum_transaction_count": 1,
        "automatic_retry": False,
    }


def expected_started(permit: authority.PreparationPermit) -> dict[str, Any]:
    return {
        "schema_id": STARTED_SCHEMA_ID,
        "authority_id": permit.authority_id,
        "authority_sha256": permit.authority_sha256,
        "bundle_sha256": permit.bundle_sha256,
        "transaction_count": 1,
        "automatic_retry": False,
    }


def consume_claim(claim_root: Path, permit: authority.PreparationPermit) -> dict[str, str]:
    require(str(claim_root) == permit.remote_claim_root, "claim root mismatch")
    require(claim_root.is_dir() and not claim_root.is_symlink(), "claim root invalid")
    claim_path = claim_root / "CLAIM.json"
    claim_bytes = read_regular(claim_path, limit=64 * 1024)
    claim = json.loads(claim_bytes.decode("utf-8"))
    require(claim == expected_claim(permit), "claim binding mismatch")
    started_path = claim_root / "EXECUTION_STARTED.json"
    write_json(started_path, expected_started(permit), exclusive=True)
    return {
        "claim_sha256": sha256_bytes(claim_bytes),
        "execution_started_sha256": sha256_bytes(read_regular(started_path)),
    }


def build_inventory(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), f"invalid evidence path: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == "FINAL_INVENTORY.json":
            continue
        data = path.read_bytes()
        require(len(data) <= MAX_EVIDENCE_FILE_BYTES, f"evidence file too large: {relative}")
        files.append({"path": relative, "size": len(data), "sha256": sha256_bytes(data)})
    require(files, "target evidence inventory empty")
    return {"schema_id": INVENTORY_SCHEMA_ID, "files": files}


def _install_interrupt_handlers() -> Callable[[], None]:
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        signal.alarm(0)
        signal.signal(signum, signal.SIG_IGN)
        raise InterruptedError(f"frequency preparation interrupted by signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    previous[signal.SIGALRM] = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(45)

    def restore() -> None:
        signal.alarm(0)
        for signum, old in previous.items():
            signal.signal(signum, old)

    return restore


def run_target(
    *,
    bundle_root: Path,
    manifest_path: Path,
    authority_path: Path,
    authority_sha256: str,
    expected_reviewed_source_commit: str,
    expected_independent_review_id: int,
    claim_root: Path,
    output_root: Path,
    sysfs_root: Path = Path("/sys"),
    identity_observer: Callable[[], dict[str, str]] = observe_target_identity,
    read_bytes: Callable[[Path], bytes] = read_sysfs,
    write_bytes: Callable[[Path, bytes], None] = write_sysfs,
) -> dict[str, Any]:
    manifest_bytes = read_regular(manifest_path)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    require(isinstance(manifest, dict), "manifest must be object")
    target_bundle.validate_extracted_bundle(bundle_root, manifest)
    authority_value, permit = authority.load_and_validate_authority(
        authority_path,
        authority_sha256=authority_sha256,
        exact_manifest=manifest,
        expected_reviewed_source_commit=expected_reviewed_source_commit,
        expected_independent_review_id=expected_independent_review_id,
    )
    require(sha256_bytes(manifest_bytes) == authority_value["manifest_sha256"], "authority manifest SHA-256 mismatch")
    require(str(output_root) == permit.remote_output_root, "output root mismatch")
    require(not output_root.exists(), "output root must be absent")
    claim_receipt = consume_claim(claim_root, permit)
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)

    write_json(output_root / "AUTHORITY_RECEIPT.json", {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_AUTHORITY_RECEIPT_V1",
        "authority_id": permit.authority_id,
        "authority_sha256": permit.authority_sha256,
        "bundle_sha256": permit.bundle_sha256,
        "reviewed_source_commit": permit.reviewed_source_commit,
        "independent_review_id": permit.independent_review_id,
        "automatic_retry": False,
    })
    write_json(output_root / "CLAIM_RECEIPT.json", claim_receipt)

    result: dict[str, Any] = {
        "schema_id": RESULT_SCHEMA_ID,
        "authority_id": permit.authority_id,
        "authority_sha256": permit.authority_sha256,
        "bundle_sha256": permit.bundle_sha256,
        "target_identity_complete": False,
        "transaction_started": False,
        "transaction_status": None,
        "restoration_complete": False,
        "frequency_write_attempt_count": 0,
        "automatic_retry": False,
        "retry_count": 0,
        "engineering_smoke_executions": 0,
        "failure": None,
    }

    transaction_receipt: dict[str, Any] | None = None
    try:
        observed_identity = identity_observer()
        identity_digest = authority.target_identity_digest(observed_identity)
        write_json(output_root / "TARGET_IDENTITY.json", {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_TARGET_IDENTITY_V1",
            "identity": observed_identity,
            "identity_sha256": identity_digest,
        })
        require(observed_identity == permit.target_identity, "target identity fields mismatch")
        result["target_identity_complete"] = True
        result["transaction_started"] = True
        restore_signals = _install_interrupt_handlers()
        try:
            transaction_receipt = live.execute_authorized_preparation_restoration(
                permit,
                sysfs_root=sysfs_root,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
            )
        finally:
            restore_signals()
        write_json(output_root / "TRANSACTION_RECEIPT.json", transaction_receipt)
        result["transaction_status"] = transaction_receipt["status"]
        result["restoration_complete"] = bool(transaction_receipt["restoration_complete"])
        result["frequency_write_attempt_count"] = int(transaction_receipt["frequency_write_attempt_count"])
        if transaction_receipt["status"] != "QUALIFIED_PREPARATION_AND_RESTORATION":
            result["failure"] = transaction_receipt.get("preparation_failure") or transaction_receipt.get("restoration_failure")
    except BaseException as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        if transaction_receipt is not None:
            result["transaction_status"] = transaction_receipt.get("status")
            result["restoration_complete"] = bool(transaction_receipt.get("restoration_complete"))
            result["frequency_write_attempt_count"] = int(transaction_receipt.get("frequency_write_attempt_count", 0))
        write_json(output_root / "FAILURE.json", {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_FAILURE_V1",
            "failure": result["failure"],
        })

    if result["failure"] is None and result["transaction_status"] == "QUALIFIED_PREPARATION_AND_RESTORATION":
        result["status"] = "QUALIFIED_PREPARATION_AND_RESTORATION"
    elif result["restoration_complete"]:
        result["status"] = "FAILED_CLOSED_PREPARATION__RESTORED"
    elif result["transaction_started"]:
        result["status"] = "FAILED_CLOSED_RESTORATION_UNPROVEN"
    else:
        result["status"] = "FAILED_CLOSED_PRECONDITION_NO_WRITES"

    write_json(output_root / "RESULT.json", result)
    inventory = build_inventory(output_root)
    write_json(output_root / "FINAL_INVENTORY.json", inventory)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one authority-gated frequency preparation/restoration transaction")
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--reviewed-source-commit", required=True)
    parser.add_argument("--independent-review-id", required=True, type=int)
    parser.add_argument("--claim-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_target(
            bundle_root=args.bundle_root,
            manifest_path=args.manifest,
            authority_path=args.authority,
            authority_sha256=args.authority_sha256,
            expected_reviewed_source_commit=args.reviewed_source_commit,
            expected_independent_review_id=args.independent_review_id,
            claim_root=args.claim_root,
            output_root=args.output_root,
        )
    except BaseException as exc:
        print(json.dumps({"status": "FAILED_CLOSED_TARGET_RUNNER", "failure": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "QUALIFIED_PREPARATION_AND_RESTORATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
