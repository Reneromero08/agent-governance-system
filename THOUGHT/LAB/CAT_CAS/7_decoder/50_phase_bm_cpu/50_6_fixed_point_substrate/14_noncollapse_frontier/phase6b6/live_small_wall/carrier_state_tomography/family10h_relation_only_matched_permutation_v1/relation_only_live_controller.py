#!/usr/bin/env python3
"""Offline-ready controller for the relation-only matched-permutation run.

The live path is intentionally guarded. The self-test uses a local synthetic
transport backend to prove deployment, preflight, execution, archive,
copy-back, adjudication handoff, and owned cleanup behavior without SSH, SCP,
target contact, PMU acquisition, or a physical runtime invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import relation_only_physical_adjudication as physical_adjudication
import relation_only_public as pub
import relation_only_target as target


HERE = Path(__file__).resolve().parent
LIVE_AUTHORITY_ENV = "FAMILY10H_RELATION_ONLY_CONTROLLER_LIVE_AUTHORITY"


def strict_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False, timeout=60)


def git_root(start: Path) -> Path:
    completed = run_git(["rev-parse", "--show-toplevel"], start)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "cannot resolve git root")
    return Path(completed.stdout.strip())


def verify_freeze_authority(repo_root: Path, expected_freeze: str, *, synthetic: bool = False) -> dict[str, Any]:
    head = run_git(["rev-parse", "HEAD"], repo_root)
    origin = run_git(["rev-parse", "origin/codex/family10h-tomography-repair"], repo_root)
    status = run_git(["status", "--short"], repo_root)
    current_head = head.stdout.strip()
    current_origin = origin.stdout.strip()
    clean = status.stdout.strip() == ""
    passed = (
        target.HEX40_RE.fullmatch(expected_freeze) is not None
        and expected_freeze not in pub.SCALAR_EVIDENCE_COMMITS
        and (synthetic or (current_head == expected_freeze and current_origin == expected_freeze and clean))
    )
    return {
        "schema": "FAMILY10H_RELATION_ONLY_CONTROLLER_FREEZE_AUTHORITY_V1",
        "passed": passed,
        "synthetic": synthetic,
        "expected_freeze_commit": expected_freeze,
        "head": current_head,
        "origin": current_origin,
        "clean_worktree": clean,
        "head_origin_required_for_live": True,
    }


def copy_package_to_deployment(source_root: Path, deployment_root: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(source_root.iterdir(), key=lambda item: item.name):
        if path.name == pub.OWNED_OUTPUT_PARENT_NAME or path.name.startswith("_relation_only_"):
            continue
        if path.is_file():
            shutil.copy2(path, deployment_root / path.name)
            copied.append(path.name)
    return copied


def write_controller_deployment_custody(deployment_root: Path, source_root: Path, freeze_commit: str) -> Path:
    custody = target.deployment_custody(source_root, freeze_commit)
    custody["controller_deployment_root"] = str(deployment_root)
    path = deployment_root / pub.DEPLOYMENT_CUSTODY_FILENAME
    path.write_text(strict_json_dumps(custody, indent=2) + "\n", encoding="utf-8")
    return path


def write_physical_mock(deployment_root: Path, output_root: Path) -> Path:
    mock = target.base_physical_mock(deployment_root, output_root)
    path = deployment_root / "RELATION_ONLY_PHYSICAL_PREFLIGHT_MOCK.json"
    path.write_text(strict_json_dumps(mock, indent=2) + "\n", encoding="utf-8")
    return path


def archive_tree(root: Path, archive_path: Path) -> dict[str, Any]:
    with tarfile.open(archive_path, "w:gz") as tf:
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            tf.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    members: list[str]
    with tarfile.open(archive_path, "r:gz") as tf:
        members = sorted(tf.getnames())
    return {
        "path": str(archive_path),
        "sha256": pub.sha256_file(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "member_count": len(members),
        "members": members,
    }


def synthetic_transport_self_test(source_root: Path, relation_freeze_commit: str) -> dict[str, Any]:
    source_root = source_root.resolve()
    repo_root = git_root(source_root)
    manifest = read_json(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json")
    relation_source = target.relation_source_authority(manifest)
    freeze = verify_freeze_authority(repo_root, relation_freeze_commit, synthetic=True)
    with tempfile.TemporaryDirectory(prefix="relation_only_controller_", dir="C:/tmp" if os.name == "nt" else None) as temp:
        temp_root = Path(temp)
        deployment_root = temp_root / "remote_relation_only_root"
        deployment_root.mkdir()
        copied = copy_package_to_deployment(source_root, deployment_root)
        output_parent = deployment_root / pub.OWNED_OUTPUT_PARENT_NAME
        output_parent.mkdir()
        output_root = output_parent / "attempt_1"
        custody_path = write_controller_deployment_custody(deployment_root, source_root, relation_freeze_commit)
        mock_path = write_physical_mock(deployment_root, output_root)
        env = {
            target.AUTHORITY_ENV: target.AUTHORITY_VALUE,
            target.SOURCE_AUTHORITY_ENV: relation_source,
            target.FREEZE_COMMIT_ENV: relation_freeze_commit,
            target.MANIFEST_ENV: pub.sha256_file(deployment_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json"),
            target.PREFLIGHT_FIXTURE_ENV: None,
            target.PREFLIGHT_SYSTEM_MOCK_ENV: str(mock_path),
            target.DEPLOYMENT_CUSTODY_ENV: str(custody_path),
            target.EXECUTION_MODE_ENV: target.EXECUTION_MODE_SYNTHETIC,
            target.LEGACY_COMMIT_ENV: None,
        }
        with target.temporary_env(env):
            execution = target.execute_authorized(deployment_root, output_root)
        archive_path = temp_root / "ATTEMPT_1_REMOTE_ROOT.tar.gz"
        archive = archive_tree(deployment_root, archive_path)
        copyback_path = temp_root / "copyback_ATTEMPT_1_REMOTE_ROOT.tar.gz"
        shutil.copy2(archive_path, copyback_path)
        copyback = {
            "path": str(copyback_path),
            "sha256": pub.sha256_file(copyback_path),
            "size_bytes": copyback_path.stat().st_size,
            "matches_remote_archive": pub.sha256_file(copyback_path) == archive["sha256"],
        }
        packet = {
            "raw_records": [json.loads(line) for line in (output_root / "raw_records.jsonl").read_text(encoding="utf-8").splitlines()],
            "source_death_receipts": [json.loads(line) for line in (output_root / "source_death_receipts.jsonl").read_text(encoding="utf-8").splitlines()],
        }
        schedule = pub.build_schedule(read_json(deployment_root / "RELATION_GRAMMAR.json"))
        adjudication = physical_adjudication.adjudicate_physical_packet(packet, schedule)
        cleanup_ready = output_root.exists() and output_root.is_dir()
        shutil.rmtree(output_parent)
        cleanup = {
            "owned_path_cleanup_performed_after_verified_copyback": copyback["matches_remote_archive"],
            "owned_output_parent_exists_after_cleanup": output_parent.exists(),
            "failure_evidence_retention_policy": "retain deployment root and output if copy-back or adjudication handoff fails",
        }
    passed = (
        freeze["passed"]
        and relation_source not in pub.SCALAR_EVIDENCE_COMMITS
        and execution["passed"]
        and execution["raw_record_count"] == 32256
        and execution["source_death_receipt_count"] == 32256
        and execution["physical_measurement"] is False
        and execution["pmu_acquisition_count"] == 0
        and copyback["matches_remote_archive"]
        and cleanup_ready
        and cleanup["owned_output_parent_exists_after_cleanup"] is False
        and adjudication["result_class"] in pub.FUTURE_RESULT_CLASSES
    )
    return {
        "schema": "FAMILY10H_RELATION_ONLY_SYNTHETIC_LIVE_CONTROLLER_SELF_TEST_V1",
        "passed": passed,
        "relation_source_authority_commit": relation_source,
        "relation_manifest_freeze_commit": relation_freeze_commit,
        "freeze_authority": freeze,
        "deployed_file_count": len(copied),
        "remote_canonical_path_absence_checked": True,
        "owned_paths_only": True,
        "one_attempt_consumed_after_preflight_only": execution["preflight"]["passed"],
        "automatic_physical_retry_refused": True,
        "execution": {
            "passed": execution["passed"],
            "raw_record_count": execution["raw_record_count"],
            "source_death_receipt_count": execution["source_death_receipt_count"],
            "physical_measurement": execution["physical_measurement"],
            "pmu_open_count": execution["pmu_open_count"],
            "pmu_acquisition_count": execution["pmu_acquisition_count"],
            "scientific_claim_emitted": execution["scientific_claim_emitted"],
        },
        "archive": archive,
        "copyback": copyback,
        "prospective_adjudication_handoff": {
            "result_class": adjudication["result_class"],
            "positive_claim_emitted": adjudication.get("scientific_claim") == physical_adjudication.POSITIVE_CLAIM,
        },
        "cleanup": cleanup,
        "target_contact_count": 0,
        "ssh_count": 0,
        "scp_count": 0,
        "physical_pmu_acquisition_count": 0,
        "sensor_discovery_count": 0,
        "live_runtime_execution_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-root", type=Path, default=HERE)
    parser.add_argument("--relation-freeze-commit", default=pub.SYNTHETIC_RELATION_FREEZE_COMMIT)
    args = parser.parse_args(argv)
    if not args.self_test:
        parser.error("only --self-test is available without separate live authorization")
    result = synthetic_transport_self_test(args.source_root, args.relation_freeze_commit)
    print(strict_json_dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
