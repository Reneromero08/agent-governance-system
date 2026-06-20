#!/usr/bin/env python3
"""Build a deterministic, commit-bound executor bundle and engineering evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
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
    CAMPAIGN_REL / "make_executor_source_bundle.py",
    CAMPAIGN_REL / "test_executor_source_bundle.py",
    CAMPAIGN_REL / "test_campaign_plan.py",
    CAMPAIGN_REL / "test_orchestrator.py",
    CAMPAIGN_REL / "test_run_manifests.py",
    CAMPAIGN_REL / "test_session_determinism.py",
    CAMPAIGN_REL / "test_session_schedule.py",
    GATE_REL / "verify_combined_plan_binding.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: list[str], *, cwd: Path | None = None,
        env: dict[str, str] | None = None,
        log: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"{completed.stdout}")
    return completed


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"git command failed ({completed.returncode}): git {' '.join(args)}\n"
            f"{completed.stderr}")
    return completed.stdout.strip()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_committed_source(repo: Path, commit: str, destination: Path) -> None:
    for relative in SOURCE_PATHS:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=repo,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"cannot read committed source {relative}: "
                f"{result.stderr.decode(errors='replace')}")
        target = destination / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout)
        working = repo / relative
        committed_hash = git(repo, "rev-parse", f"{commit}:{relative.as_posix()}")
        working_hash = git(repo, "hash-object", relative.as_posix()) if working.is_file() else ""
        if working_hash != committed_hash:
            raise RuntimeError(f"working tree differs from committed source: {relative}")


def validate_session_bundle(directory: Path) -> None:
    manifest = json.loads(
        (directory / "session_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_id") != "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1":
        raise RuntimeError(f"unexpected session manifest schema: {directory.name}")
    for name, binding in manifest.get("files", {}).items():
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
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"source_bundle.json", "source_bundle.sha256"}:
            continue
        bindings[relative] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return bindings


def verify_file_bindings(root: Path, files: dict[str, Any]) -> None:
    if not isinstance(files, dict) or not files:
        raise RuntimeError("bundle has no file bindings")
    expected = set(files)
    for relative, binding in sorted(files.items()):
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
        for path in root.rglob("*") if path.is_file()
    } - {"source_bundle.json", "source_bundle.sha256"}
    if "target_evidence_manifest.json" not in expected:
        actual.discard("target_evidence_manifest.json")
    if actual != expected:
        raise RuntimeError(
            f"bound file set mismatch: missing={sorted(expected-actual)} "
            f"unexpected={sorted(actual-expected)}")


def verify_bundle(root: Path, expected_commit: str | None = None) -> dict[str, Any]:
    bundle_path = root / "source_bundle.json"
    checksum_path = root / "source_bundle.sha256"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    expected_checksum = checksum_path.read_text(encoding="utf-8").split()[0]
    if sha256_file(bundle_path) != expected_checksum:
        raise RuntimeError("source_bundle.sha256 mismatch")
    if expected_commit is not None and bundle.get("executor_commit") != expected_commit:
        raise RuntimeError("executor commit mismatch")
    if bundle.get("acquisition_authorized") is not False:
        raise RuntimeError("bundle must not authorize acquisition")
    if bundle.get("restoration_authorized") is not False:
        raise RuntimeError("bundle must not authorize restoration")
    verify_file_bindings(root, bundle.get("files"))
    return bundle


def build_source_transfer(repo: Path, commit: str, plan_dir: Path,
                          binding_path: Path, output: Path) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(commit) or set(commit) == {"0"}:
        raise ValueError("commit must be nonzero lowercase 40-character hex")
    if git(repo, "rev-parse", "HEAD") != commit:
        raise RuntimeError("HEAD does not equal requested commit")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("working tree must be clean")
    git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
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
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        plan_module = import_module(campaign / "generate_campaign_plan.py", "transfer_plan")
        compiler_module = import_module(campaign / "compile_session_schedule.py", "transfer_compile")
    finally:
        sys.dont_write_bytecode = previous_bytecode
    plan_errors = plan_module.verify(plan_target)
    if plan_errors:
        raise RuntimeError(f"plan validation failed: {plan_errors}")
    plan = json.loads((plan_target / "campaign_plan.json").read_text(encoding="utf-8"))
    compiled = output / "compiled_sessions"
    compiled.mkdir()
    for session in plan.get("sessions", []):
        directory = compiled / session["session_id"]
        compiler_module.write_session(plan_target / "campaign_plan.json", session["session_id"], directory)
        validate_session_bundle(directory)
    if len(list(compiled.iterdir())) != 12:
        raise RuntimeError("expected exactly 12 compiled sessions")
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
    (output / "source_bundle.sha256").write_text(
        sha256_file(bundle_path) + "  source_bundle.json\n", encoding="utf-8")
    verify_bundle(output, commit)
    return bundle


def write_target_manifest(root: Path, commit: str) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(commit) or set(commit) == {"0"}:
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


def seal_target_evidence(source_root: Path, target_root: Path,
                         output: Path, commit: str) -> dict[str, Any]:
    source = verify_bundle(source_root, commit)
    manifest = json.loads((target_root / "target_evidence_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_id") != "CAT_CAS_PHASE6_TARGET_EVIDENCE_V1":
        raise RuntimeError("unexpected target evidence schema")
    if manifest.get("executor_commit") != commit:
        raise RuntimeError("target evidence commit mismatch")
    verify_file_bindings(target_root, manifest.get("files"))
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    shutil.copytree(source_root, output)
    (output / "source_bundle.json").rename(output / "source_transfer_bundle.json")
    (output / "source_bundle.sha256").rename(output / "source_transfer_bundle.sha256")
    for path in sorted(target_root.rglob("*")):
        if not path.is_file() or path.name == "target_evidence_manifest.json":
            continue
        relative = path.relative_to(target_root)
        destination = output / relative
        if destination.exists():
            raise RuntimeError(f"target evidence collides with source bundle: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(target_root / "target_evidence_manifest.json", output / "target_evidence_manifest.json")
    validation = json.loads((output / "evidence" / "validation_report.json").read_text(encoding="utf-8"))
    if validation.get("sessions_expected") != 12 or validation.get("sessions_passed") != 12 or validation.get("all_pass") is not True or len(validation.get("records", [])) != 12:
        raise RuntimeError("target evidence does not prove 12/12 validation")
    runner = output / "combined_pdn_runner"
    bundle = {
        "schema_id": "CAT_CAS_PHASE6_EXECUTOR_SOURCE_BUNDLE_V1",
        "artifact_class": "ENGINEERING_BUNDLE_NOT_ACQUISITION_AUTHORIZATION",
        "executor_commit": commit,
        "executor_sha256": sha256_file(runner),
        "campaign_plan_sha256": source["campaign_plan_sha256"],
        "campaign_manifest_sha256": source["campaign_manifest_sha256"],
        "source_transfer_bundle_sha256": sha256_file(source_root / "source_bundle.json"),
        "validation_sessions_expected": 12,
        "evidence": {
            "strict_test_log": "evidence/strict_tests.log",
            "sanitizer_test_log": "evidence/sanitizer_tests.log",
            "python_test_log": "evidence/python_tests.log",
            "validation_report": "evidence/validation_report.json",
        },
        "files": add_file_bindings(output),
        "acquisition_authorized": False,
        "restoration_authorized": False,
    }
    bundle_path = output / "source_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "source_bundle.sha256").write_text(
        sha256_file(bundle_path) + "  source_bundle.json\n", encoding="utf-8")
    verify_bundle(output, commit)
    return bundle


def build_bundle(repo: Path, commit: str, plan_dir: Path,
                 binding_path: Path, output: Path) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(commit) or set(commit) == {"0"}:
        raise ValueError("commit must be nonzero lowercase 40-character hex")
    head = git(repo, "rev-parse", "HEAD")
    if head != commit:
        raise RuntimeError(f"HEAD {head} does not equal requested commit {commit}")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("working tree must be clean")
    git(repo, "cat-file", "-e", f"{commit}^{{commit}}")

    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    evidence = output / "evidence"
    evidence.mkdir()
    copy_committed_source(repo, commit, output)

    plan_target = output / "plan"
    plan_target.mkdir()
    for name in ("campaign_plan.json", "campaign_manifest.json"):
        shutil.copy2(plan_dir / name, plan_target / name)
    shutil.copy2(binding_path, output / "COMBINED_CAMPAIGN_BINDING.json")

    generator_path = output / "sources" / CAMPAIGN_REL / "generate_campaign_plan.py"
    compiler_path = output / "sources" / CAMPAIGN_REL / "compile_session_schedule.py"
    verifier_path = output / "sources" / CAMPAIGN_REL / "verify_run_manifests.py"
    plan_module = import_module(generator_path, "bundle_generate_campaign_plan")
    compiler_module = import_module(compiler_path, "bundle_compile_session_schedule")
    verifier_module = import_module(verifier_path, "bundle_verify_run_manifests")
    plan_errors = plan_module.verify(plan_target)
    if plan_errors:
        raise RuntimeError(f"plan validation failed: {plan_errors}")

    runtime = output / "sources" / RUNTIME_REL
    strict_log = evidence / "strict_tests.log"
    sanitizer_log = evidence / "sanitizer_tests.log"
    python_log = evidence / "python_tests.log"
    strict_compile = [
        "gcc", "-std=c11", "-O2", "-pthread", "-Wall", "-Wextra", "-Werror",
        "combined_pdn_runner.c", "combined_pdn_hardware.c",
        "-o", "combined_pdn_runner", "-lm",
    ]
    run(strict_compile, cwd=runtime)
    strict_result = run(
        [sys.executable, "test_combined_pdn_runner.py"],
        cwd=runtime, log=strict_log)
    if "OK" not in strict_result.stdout:
        raise RuntimeError("strict test log lacks unittest OK marker")
    shutil.copy2(runtime / "combined_pdn_runner", output / "combined_pdn_runner")
    os.chmod(output / "combined_pdn_runner", 0o755)

    sanitizer_compile = [
        "gcc", "-std=c11", "-O1", "-g", "-pthread", "-Wall", "-Wextra",
        "-Werror", "-fsanitize=address,undefined",
        "combined_pdn_runner.c", "combined_pdn_hardware.c",
        "-o", "combined_pdn_runner", "-lm",
    ]
    run(sanitizer_compile, cwd=runtime)
    sanitizer_env = os.environ.copy()
    sanitizer_env["ASAN_OPTIONS"] = "detect_leaks=1"
    sanitizer_env["UBSAN_OPTIONS"] = "halt_on_error=1"
    sanitizer_result = run(
        [sys.executable, "test_combined_pdn_runner.py"],
        cwd=runtime, env=sanitizer_env, log=sanitizer_log)
    if "OK" not in sanitizer_result.stdout:
        raise RuntimeError("sanitizer test log lacks unittest OK marker")

    run(strict_compile, cwd=runtime)
    shutil.copy2(runtime / "combined_pdn_runner", output / "combined_pdn_runner")
    os.chmod(output / "combined_pdn_runner", 0o755)

    campaign_test_dir = repo / CAMPAIGN_REL
    tests = [
        campaign_test_dir / "test_orchestrator.py",
        campaign_test_dir / "test_run_manifests.py",
    ]
    existing = [str(path) for path in tests if path.is_file()]
    if existing:
        run([sys.executable, "-m", "unittest", *existing],
            cwd=campaign_test_dir, log=python_log)
    else:
        python_log.write_text("NO_CAMPAIGN_PYTHON_TESTS_DISCOVERED\n", encoding="utf-8")

    plan = json.loads((plan_target / "campaign_plan.json").read_text(encoding="utf-8"))
    compiled = output / "compiled_sessions"
    compiled.mkdir()
    validation_runs = output / "validation_runs"
    validation_runs.mkdir()
    passed = 0
    records: list[dict[str, Any]] = []
    for session in plan.get("sessions", []):
        session_id = session["session_id"]
        session_dir = compiled / session_id
        compiler_module.write_session(
            plan_target / "campaign_plan.json", session_id, session_dir)
        validate_session_bundle(session_dir)
        route = session["route"]
        if route == "v4s5":
            victim, sender = 4, 5
        elif route == "v2s3":
            victim, sender = 2, 3
        else:
            raise RuntimeError(f"unsupported route: {route}")
        run_dir = validation_runs / session_id
        result = run([
            str(output / "combined_pdn_runner"),
            "--session-dir", str(session_dir),
            "--output-dir", str(run_dir),
            "--victim", str(victim),
            "--sender", str(sender),
            "--validate-only",
        ])
        manifest_errors = verifier_module.verify(validation_runs)
        if manifest_errors:
            raise RuntimeError(f"validation manifest errors: {manifest_errors}")
        passed += 1
        records.append({
            "session_id": session_id,
            "route": route,
            "runner_exit_code": result.returncode,
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        })

    validation_report = {
        "schema_id": "CAT_CAS_PHASE6_VALIDATION_EVIDENCE_V1",
        "sessions_expected": len(plan.get("sessions", [])),
        "sessions_passed": passed,
        "all_pass": passed == 12 and passed == len(plan.get("sessions", [])),
        "records": records,
    }
    (evidence / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if not validation_report["all_pass"]:
        raise RuntimeError(f"expected exactly 12 passing validation sessions: {passed}")

    bundle = {
        "schema_id": "CAT_CAS_PHASE6_EXECUTOR_SOURCE_BUNDLE_V1",
        "artifact_class": "ENGINEERING_BUNDLE_NOT_ACQUISITION_AUTHORIZATION",
        "executor_commit": commit,
        "executor_sha256": sha256_file(output / "combined_pdn_runner"),
        "campaign_plan_sha256": sha256_file(plan_target / "campaign_plan.json"),
        "campaign_manifest_sha256": sha256_file(plan_target / "campaign_manifest.json"),
        "validation_sessions_expected": 12,
        "evidence": {
            "strict_test_log": "evidence/strict_tests.log",
            "sanitizer_test_log": "evidence/sanitizer_tests.log",
            "python_test_log": "evidence/python_tests.log",
            "validation_report": "evidence/validation_report.json",
        },
        "files": add_file_bindings(output),
        "acquisition_authorized": False,
        "restoration_authorized": False,
    }
    bundle_path = output / "source_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "source_bundle.sha256").write_text(
        sha256_file(bundle_path) + "  source_bundle.json\n", encoding="utf-8")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-only", action="store_true")
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
                args.source_bundle.resolve(), args.seal_target_evidence.resolve(),
                args.output.resolve(), args.commit)
        else:
            if not all((args.repo_root, args.plan_dir, args.binding, args.output)):
                raise ValueError("bundle generation requires repo, plan, binding, and output")
            builder = build_source_transfer if args.source_only else build_bundle
            bundle = builder(
                args.repo_root.resolve(), args.commit, args.plan_dir.resolve(),
                args.binding.resolve(), args.output.resolve())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
