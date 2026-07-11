#!/usr/bin/env python3
"""Verify one exact active Gate A authority without contacting the target."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import build_gate_a_execution_bundle as bundle
import gate_a_authority
import gate_a_hardware_adapter as hardware_adapter

HERE = Path(__file__).resolve().parent
REPO_ROOT = bundle.repo_root().resolve()
ACTIVE_AUTHORITY = HERE / "GATE_A_EXECUTION_AUTHORITY.json"
CONSUMED_AUTHORITY = HERE / "GATE_A_EXECUTION_AUTHORITY_CONSUMED_7e1e8835.json"
MANIFEST = HERE / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"
AUTHORITY_SCHEMA = HERE / "schemas" / "gate_a_execution_authority.schema.json"

EXPECTED_REVIEWED_HEAD = "040b80fbb10c6a8fa63bb590a86c6dc8d4ff4d59"
EXPECTED_REVIEW_ID = 4677123997
EXPECTED_AUTHORITY_SHA256 = "1dabfc7bbfc65e988542b0c4580f031309c5abfc53521dd849e5c0ac71e24fd4"
EXPECTED_AUTHORITY_BLOB = "b69b1fa6c7d2c710c76fac0115c2227006fd2212"
EXPECTED_CONSUMED_SHA256 = "7e1e8835bd67590e4e554ae112a2c8a6ca99dd8b9b3a9aafdb23fee31907d682"
EXPECTED_CONSUMED_BLOB = "709c799f60e30984d3c80715af480fbe5deac952"


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def git(*args: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        check=False,
    )


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_blob(path: Path) -> str:
    result = git("rev-parse", f"HEAD:{rel(path)}")
    require(result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", result.stdout.strip()) is not None, f"committed blob unavailable: {path.name}")
    return result.stdout.strip()


def verify_authority_namespace() -> list[str]:
    require(ACTIVE_AUTHORITY.is_file() and not ACTIVE_AUTHORITY.is_symlink(), "active authority missing or not regular")
    require(CONSUMED_AUTHORITY.is_file() and not CONSUMED_AUTHORITY.is_symlink(), "consumed authority archive missing or not regular")
    require(sha256(ACTIVE_AUTHORITY) == EXPECTED_AUTHORITY_SHA256, "active authority SHA-256 mismatch")
    require(committed_blob(ACTIVE_AUTHORITY) == EXPECTED_AUTHORITY_BLOB, "active authority Git blob mismatch")
    require(sha256(CONSUMED_AUTHORITY) == EXPECTED_CONSUMED_SHA256, "consumed authority SHA-256 mismatch")
    require(committed_blob(CONSUMED_AUTHORITY) == EXPECTED_CONSUMED_BLOB, "consumed authority Git blob mismatch")

    expected = sorted((rel(ACTIVE_AUTHORITY), rel(CONSUMED_AUTHORITY)))
    tracked = sorted(
        line
        for line in git("ls-files").stdout.splitlines()
        if Path(line).name.casefold().startswith("gate_a_execution_authority")
        and Path(line).name.casefold().endswith(".json")
        and line != rel(AUTHORITY_SCHEMA)
    )
    require(tracked == expected, f"execution-authority tracked set mismatch: {tracked}")
    status = git(
        "status", "--porcelain=v1", "--untracked-files=all", "--",
        ":(icase,glob)**/gate_a_execution_authority*.json",
        f":(exclude){rel(AUTHORITY_SCHEMA)}",
    )
    require(status.returncode == 0 and status.stdout == "", "execution-authority namespace differs from HEAD")
    return tracked


def main() -> int:
    head_result = git("rev-parse", "HEAD")
    require(head_result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head_result.stdout.strip()) is not None, "authority-bearing HEAD unavailable")
    head = head_result.stdout.strip()
    require(git("merge-base", "--is-ancestor", EXPECTED_REVIEWED_HEAD, head).returncode == 0, "reviewed source is not an ancestor of authority-bearing HEAD")

    tracked = verify_authority_namespace()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exact_manifest = bundle.validate_committed_manifest_exact(manifest, "HEAD")
    authority_bytes = ACTIVE_AUTHORITY.read_bytes()
    authority = json.loads(authority_bytes.decode("utf-8"))

    validation = gate_a_authority.validate_execution_authority(
        authority,
        authority_sha256=EXPECTED_AUTHORITY_SHA256,
        authority_bytes=authority_bytes,
        expected_reviewed_adapter_head=EXPECTED_REVIEWED_HEAD,
        expected_independent_review_id=EXPECTED_REVIEW_ID,
        exact_manifest=exact_manifest,
    )
    custody = hardware_adapter.validate_authority_git_custody(ACTIVE_AUTHORITY, authority)

    report = {
        "status": "GATE_A_ACTIVE_AUTHORITY_EXACT",
        "authority_bearing_head": head,
        "reviewed_source": EXPECTED_REVIEWED_HEAD,
        "independent_review_id": EXPECTED_REVIEW_ID,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "authority_git_blob_sha1": EXPECTED_AUTHORITY_BLOB,
        "consumed_authority_sha256": EXPECTED_CONSUMED_SHA256,
        "consumed_authority_git_blob_sha1": EXPECTED_CONSUMED_BLOB,
        "execution_bundle_sha256": exact_manifest["execution_bundle_sha256"],
        "tracked_authority_paths": tracked,
        "production_validation": validation,
        "git_custody": custody,
        "maximum_execution_count": 1,
        "automatic_retry": False,
        "target_contacts": 0,
        "hardware_executions": 0,
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerifyError, gate_a_authority.AuthorityError, hardware_adapter.AdapterError, bundle.BundleError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"verify_gate_a_active_authority: {exc}", file=sys.stderr)
        raise SystemExit(1)
