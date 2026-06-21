#!/usr/bin/env python3
"""Build, verify, transfer, and seal the Phase 6 executor evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

COMMIT_RE = re.compile(r"[0-9a-f]{40}")
PACKAGE_REL = Path(
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/"
    "50_6_fixed_point_substrate/14_noncollapse_frontier"
)
CAMPAIGN_REL = PACKAGE_REL / "combined_observability_campaign"
RUNTIME_REL = PACKAGE_REL / "holo_runtime"
GATE_REL = PACKAGE_REL / "gate_r"
SOURCE_PATHS = (
    RUNTIME_REL / "combined_pdn_hardware.h",
    RUNTIME_REL / "combined_pdn_hardware.c",
    RUNTIME_REL / "combined_pdn_runner.c",
    RUNTIME_REL / "test_combined_pdn_runner.py",
    RUNTIME_REL / "Makefile",
    RUNTIME_REL / "README.md",
    RUNTIME_REL / "make_engineering_smoke_schedule.py",
    CAMPAIGN_REL / "compile_session_schedule.py",
    CAMPAIGN_REL / "campaign_orders.py",
    CAMPAIGN_REL / "campaign_plan.py",
    CAMPAIGN_REL / "generate_campaign_plan.py",
    CAMPAIGN_REL / "verify_run_manifests.py",
    CAMPAIGN_REL / "run_combined_campaign.py",
    CAMPAIGN_REL / "catcas_preflight.py",
    CAMPAIGN_REL / "collect_target_engineering_evidence.py",
    CAMPAIGN_REL / "make_executor_source_bundle.py",
    CAMPAIGN_REL / "test_executor_source_bundle.py",
    CAMPAIGN_REL / "test_campaign_plan.py",
    CAMPAIGN_REL / "test_catcas_preflight.py",
    CAMPAIGN_REL / "test_orchestrator.py",
    CAMPAIGN_REL / "test_run_manifests.py",
    CAMPAIGN_REL / "test_session_determinism.py",
    CAMPAIGN_REL / "test_session_schedule.py",
    CAMPAIGN_REL / "test_target_engineering_evidence.py",
    GATE_REL / "verify_combined_plan_binding.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_sidecar(path: Path, sidecar: Path) -> None:
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")


def verify_sha256_sidecar(sidecar: Path, expected_path: Path) -> None:
    tokens = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if len(tokens) != 2:
        raise RuntimeError(f"invalid checksum sidecar: {sidecar.name}")
    digest, recorded_name = tokens
    recorded_name = recorded_name.lstrip("*")
    if recorded_name != expected_path.name:
        raise RuntimeError(
            f"checksum sidecar filename mismatch: {sidecar.name} names {recorded_name}, "
            f"expected {expected_path.name}"
        )
    if digest != sha256_file(expected_path):
        raise RuntimeError(f"checksum sidecar digest mismatch: {sidecar.name}")


def rename_bundle_with_sidecar(
    root: Path, old_name: str, new_name: str
) -> tuple[Path, Path]:
    old_path = root / old_name
    old_sidecar = root / f"{Path(old_name).stem}.sha256"
    new_path = root / new_name
    new_sidecar = root / f"{Path(new_name).stem}.sha256"
    old_path.rename(new_path)
    old_sidecar.rename(new_sidecar)
    write_sha256_sidecar(new_path, new_sidecar)
    verify_sha256_sidecar(new_sidecar, new_path)
    return new_path, new_sidecar


def load_object(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object")
    return value


def valid_commit(commit: str) -> bool:
    return bool(COMMIT_RE.fullmatch(commit)) and set(commit) != {"0"}


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(args)}\n{completed.stdout}"
        )
    return completed


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git command failed ({completed.returncode}): git {' '.join(args)}\n"
            f"{completed.stderr}"
        )
    return completed.stdout.strip()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_without_bytecode(path: Path, name: str):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return import_module(path, name)
    finally:
        sys.dont_write_bytecode = previous


def copy_committed_source(repo: Path, commit: str, destination: Path) -> None:
    for relative in SOURCE_PATHS:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative.as_posix()}"],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cannot read committed source {relative}: "
                f"{result.stderr.decode(errors='replace')}"
            )
        target = destination / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout)
        working = repo / relative
        committed_hash = git(repo, "rev-parse", f"{commit}:{relative.as_posix()}")
        working_hash = git(repo, "hash-object", relative.as_posix()) if working.is_file() else ""
        if working_hash != committed_hash:
            raise RuntimeError(f"working tree differs from committed source: {relative}")


def validate_session_bundle(directory: Path) -> None:
    manifest = load_object(directory / "session_manifest.json", "session manifest")
    if manifest.get("schema_id") != "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1":
        raise RuntimeError(f"unexpected session manifest schema: {directory.name}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"invalid session files table: {directory.name}")
    for name, binding in files.items():
        if not isinstance(name, str) or not isinstance(binding, dict):
            raise RuntimeError(f"invalid session binding: {directory.name}/{name!r}")
        path = (directory / name).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError as exc:
            raise RuntimeError(f"session path escapes bundle: {name}") from exc
        if not path.is_file():
            raise RuntimeError(f"missing session file: {path}")
        if path.stat().st_size != binding.get("size"):
            raise RuntimeError(f"session size mismatch: {path}")
        if sha256_file(path) != binding.get("sha256"):
            raise RuntimeError(f"session hash mismatch: {path}")


def add_file_bindings(root: Path) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"source_bundle.json", "source_bundle.sha256"}:
            continue
        bindings[relative] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return bindings


def verify_file_bindings(root: Path, files: Any) -> None:
    if not isinstance(files, dict) or not files:
        raise RuntimeError("bundle has no file bindings")
    expected: set[str] = set()
    for relative, binding in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(binding, dict):
            raise RuntimeError(f"invalid file binding: {relative!r}")
        expected.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"bundle path escapes root: {relative}") from exc
        if not path.is_file():
            raise RuntimeError(f"missing bound file: {relative}")
        if path.stat().st_size != binding.get("size"):
            raise RuntimeError(f"size mismatch: {relative}")
        if sha256_file(path) != binding.get("sha256"):
            raise RuntimeError(f"sha256 mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } - {"source_bundle.json", "source_bundle.sha256"}
    if "target_evidence_manifest.json" not in expected:
        actual.discard("target_evidence_manifest.json")
    if actual != expected:
        raise RuntimeError(
            f"bound file set mismatch: missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )


def verify_bundle(root: Path, expected_commit: str | None = None) -> dict[str, Any]:
    bundle_path = root / "source_bundle.json"
    checksum_path = root / "source_bundle.sha256"
    bundle = load_object(bundle_path, "source bundle")
    verify_sha256_sidecar(checksum_path, bundle_path)
    if expected_commit is not None and bundle.get("executor_commit") != expected_commit:
        raise RuntimeError("executor commit mismatch")
    if bundle.get("acquisition_authorized") is not False:
        raise RuntimeError("bundle must not authorize acquisition")
    if bundle.get("restoration_authorized") is not False:
        raise RuntimeError("bundle must not authorize restoration")
    verify_file_bindings(root, bundle.get("files"))
    return bundle


def verify_clean_repo(repo: Path, commit: str) -> None:
    if not valid_commit(commit):
        raise ValueError("commit must be nonzero lowercase 40-character hex")
    if git(repo, "rev-parse", "HEAD") != commit:
        raise RuntimeError("HEAD does not equal requested commit")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("working tree must be clean")
    git(repo, "cat-file", "-e", f"{commit}^{{commit}}")


def build_source_transfer(
    repo: Path, commit: str, plan_dir: Path, binding_path: Path, output: Path
) -> dict[str, Any]:
    verify_clean_repo(repo, commit)
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    copy_committed_source(repo, commit, output)
    plan_target = output / "plan"
    plan_target.mkdir()
    for name in ("campaign_plan.json", "campaign_manifest.json"):
        shutil.copy2(plan_dir / name, plan_target / name)
    shutil.copy2(binding_path, output / "COMBINED_CAMPAIGN_BINDING.json")

    campaign = output / "sources" / CAMPAIGN_REL
    plan_module = import_without_bytecode(campaign / "generate_campaign_plan.py", "transfer_plan")
    compiler_module = import_without_bytecode(
        campaign / "compile_session_schedule.py", "transfer_compile"
    )
    plan_errors = plan_module.verify(plan_target)
    if plan_errors:
        raise RuntimeError(f"plan validation failed: {plan_errors}")
    plan = load_object(plan_target / "campaign_plan.json", "campaign plan")
    sessions = plan.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 12 or not all(
        isinstance(session, dict) for session in sessions
    ):
        raise RuntimeError("expected exactly 12 plan sessions")
    compiled = output / "compiled_sessions"
    compiled.mkdir()
    for session in sessions:
        session_id = session.get("session_id")
        if not isinstance(session_id, str):
            raise RuntimeError("session missing session_id")
        directory = compiled / session_id
        compiler_module.write_session(plan_target / "campaign_plan.json", session_id, directory)
        validate_session_bundle(directory)
    if list(output.rglob("__pycache__")) or list(output.rglob("*.pyc")):
        raise RuntimeError("generated Python bytecode is forbidden in source transfer")

    bundle = {
        "schema_id": "CAT_CAS_PHASE6_EXECUTOR_SOURCE_TRANSFER_V1",
        "artifact_class": "ENGINEERING_BUNDLE_NOT_ACQUISITION_AUTHORIZATION",
        "executor_commit": commit,
        "campaign_plan_sha256": sha256_file(plan_target / "campaign_plan.json"),
        "campaign_manifest_sha256": sha256_file(plan_target / "campaign_manifest.json"),
        "validation_sessions_expected": 12,
        "files": add_file_bindings(output),
        "acquisition_authorized": False,
        "restoration_authorized": False,
    }
    bundle_path = output / "source_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_sha256_sidecar(bundle_path, output / "source_bundle.sha256")
    verify_bundle(output, commit)
    return bundle


def write_target_manifest(root: Path, commit: str) -> dict[str, Any]:
    if not valid_commit(commit):
        raise ValueError("commit must be nonzero lowercase 40-character hex")
    path = root / "target_evidence_manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing existing output: {path}")
    manifest = {
        "schema_id": "CAT_CAS_PHASE6_TARGET_EVIDENCE_V1",
        "executor_commit": commit,
        "files": add_file_bindings(root),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest



def validate_target_validation_evidence(target_root: Path) -> dict[str, Any]:
    report_path = target_root / "evidence" / "validation_report.json"
    report = load_object(report_path, "validation report")
    if report.get("schema_id") != "CAT_CAS_PHASE6_VALIDATION_EVIDENCE_V1":
        raise RuntimeError("unexpected validation evidence schema")
    runs_relative = report.get("runs_root")
    if not isinstance(runs_relative, str) or not runs_relative:
        raise RuntimeError("validation report missing runs_root")
    runs_root = (target_root / runs_relative).resolve()
    try:
        runs_root.relative_to(target_root.resolve())
    except ValueError as exc:
        raise RuntimeError("validation runs_root escapes target evidence") from exc
    if not runs_root.is_dir():
        raise RuntimeError("validation runs_root is missing")

    directories = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if len(directories) != 12:
        raise RuntimeError(f"expected 12 raw validation runs, found {len(directories)}")
    actual_records: list[dict[str, Any]] = []
    for directory in directories:
        manifest = load_object(directory / "run_manifest.json", f"{directory.name} run manifest")
        files = manifest.get("files")
        if not isinstance(files, dict) or "run_manifest.json" in files:
            raise RuntimeError(f"{directory.name}: invalid run manifest files table")
        for name, binding in files.items():
            if not isinstance(name, str) or not isinstance(binding, dict):
                raise RuntimeError(f"{directory.name}: invalid run manifest binding")
            path = (directory / name).resolve()
            try:
                path.relative_to(directory.resolve())
            except ValueError as exc:
                raise RuntimeError(f"{directory.name}: run manifest path escape") from exc
            if not path.is_file():
                raise RuntimeError(f"{directory.name}: missing {name}")
            if path.stat().st_size != binding.get("size"):
                raise RuntimeError(f"{directory.name}: size mismatch {name}")
            if sha256_file(path) != binding.get("sha256"):
                raise RuntimeError(f"{directory.name}: sha256 mismatch {name}")
        run = load_object(directory / "run.json", f"{directory.name} validation run")
        if not (
            run.get("status") == "VALIDATION_ONLY_HARDWARE_NOT_EXECUTED"
            and run.get("hardware_executed") is False
            and run.get("scientific_acquisition_authorized") is False
            and run.get("restoration_authorized") is False
            and run.get("physical_carrier_restoration_claimed") is False
        ):
            raise RuntimeError(f"{directory.name}: validation-only contract failed")
        actual_records.append({
            "session_id": directory.name,
            "runner_exit_code": 0,
            "hardware_executed": False,
            "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
        })

    records = report.get("records")
    if not (
        report.get("sessions_expected") == 12
        and report.get("sessions_passed") == 12
        and report.get("all_pass") is True
        and report.get("hardware_touched") is False
        and report.get("errors") == []
        and isinstance(records, list)
        and records == actual_records
    ):
        raise RuntimeError("validation report does not match raw 12-session evidence")
    return report

def validate_target_report(
    target_root: Path, source_bundle: dict[str, Any], source_bundle_sha256: str, commit: str
) -> dict[str, Any]:
    report_path = target_root / "evidence" / "target_engineering_report.json"
    report = load_object(report_path, "target engineering report")
    if report.get("schema_id") != "CAT_CAS_PHASE6_TARGET_ENGINEERING_EVIDENCE_V1":
        raise RuntimeError("unexpected target engineering evidence schema")
    if report.get("executor_commit") != commit:
        raise RuntimeError("target engineering report commit mismatch")
    if report.get("source_transfer_bundle_sha256") != source_bundle_sha256:
        raise RuntimeError("target engineering report source-transfer mismatch")
    if report.get("all_pass") is not True:
        raise RuntimeError("target engineering report is not all-pass")
    if report.get("scientific_acquisition_started") is not False:
        raise RuntimeError("target report indicates scientific acquisition")
    if report.get("physical_carrier_restoration_claimed") is not False:
        raise RuntimeError("target report promotes physical carrier restoration")
    runner = target_root / "combined_pdn_runner"
    if not runner.is_file() or report.get("executor_sha256") != sha256_file(runner):
        raise RuntimeError("target report executor hash mismatch")
    source_commit = source_bundle.get("executor_commit")
    if source_commit != commit:
        raise RuntimeError("source transfer commit mismatch")
    return report


def seal_target_evidence(
    source_root: Path, target_root: Path, output: Path, commit: str
) -> dict[str, Any]:
    source = verify_bundle(source_root, commit)
    if source.get("schema_id") != "CAT_CAS_PHASE6_EXECUTOR_SOURCE_TRANSFER_V1":
        raise RuntimeError("expected a source-transfer bundle")
    target_manifest = load_object(
        target_root / "target_evidence_manifest.json", "target evidence manifest"
    )
    if target_manifest.get("schema_id") != "CAT_CAS_PHASE6_TARGET_EVIDENCE_V1":
        raise RuntimeError("unexpected target evidence schema")
    if target_manifest.get("executor_commit") != commit:
        raise RuntimeError("target evidence commit mismatch")
    verify_file_bindings(target_root, target_manifest.get("files"))
    source_bundle_sha256 = sha256_file(source_root / "source_bundle.json")
    validate_target_report(target_root, source, source_bundle_sha256, commit)
    validate_target_validation_evidence(target_root)
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    shutil.copytree(source_root, output)
    rename_bundle_with_sidecar(
        output, "source_bundle.json", "source_transfer_bundle.json"
    )
    for path in sorted(target_root.rglob("*")):
        if not path.is_file() or path.name == "target_evidence_manifest.json":
            continue
        relative = path.relative_to(target_root)
        destination = output / relative
        if destination.exists():
            raise RuntimeError(f"target evidence collides with source bundle: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(
        target_root / "target_evidence_manifest.json",
        output / "target_evidence_manifest.json",
    )
    runner = output / "combined_pdn_runner"
    bundle = {
        "schema_id": "CAT_CAS_PHASE6_EXECUTOR_SOURCE_BUNDLE_V1",
        "artifact_class": "ENGINEERING_BUNDLE_NOT_ACQUISITION_AUTHORIZATION",
        "executor_commit": commit,
        "executor_sha256": sha256_file(runner),
        "campaign_plan_sha256": source["campaign_plan_sha256"],
        "campaign_manifest_sha256": source["campaign_manifest_sha256"],
        "source_transfer_bundle_sha256": source_bundle_sha256,
        "validation_sessions_expected": 12,
        "evidence": {
            "strict_test_log": "evidence/strict_tests.log",
            "sanitizer_test_log": "evidence/sanitizer_tests.log",
            "python_test_log": "evidence/python_tests.log",
            "validation_report": "evidence/validation_report.json",
            "target_engineering_report": "evidence/target_engineering_report.json",
        },
        "files": add_file_bindings(output),
        "acquisition_authorized": False,
        "restoration_authorized": False,
    }
    bundle_path = output / "source_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_sha256_sidecar(bundle_path, output / "source_bundle.sha256")
    verify_bundle(output, commit)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="deprecated compatibility flag; source transfer is now the only build mode",
    )
    parser.add_argument("--verify-bundle", type=Path)
    parser.add_argument("--write-target-manifest", type=Path)
    parser.add_argument("--seal-target-evidence", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    args = parser.parse_args()
    try:
        if args.verify_bundle:
            bundle = verify_bundle(args.verify_bundle.resolve(), args.commit)
        elif args.write_target_manifest:
            bundle = write_target_manifest(args.write_target_manifest.resolve(), args.commit)
        elif args.seal_target_evidence:
            if not args.source_bundle or not args.output:
                raise ValueError("sealing requires --source-bundle and --output")
            bundle = seal_target_evidence(
                args.source_bundle.resolve(),
                args.seal_target_evidence.resolve(),
                args.output.resolve(),
                args.commit,
            )
        else:
            if not all((args.repo_root, args.plan_dir, args.binding, args.output)):
                raise ValueError("bundle generation requires repo, plan, binding, and output")
            bundle = build_source_transfer(
                args.repo_root.resolve(),
                args.commit,
                args.plan_dir.resolve(),
                args.binding.resolve(),
                args.output.resolve(),
            )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
