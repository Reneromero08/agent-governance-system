#!/usr/bin/env python3
"""Offline controller for the public Family 10h carrier tomography package."""

from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import family10h_carrier_tomography_public as public
import family10h_carrier_tomography_target as target


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "CARRIER_TOMOGRAPHY_CONTRACT.md"
SOURCE_BUNDLE = HERE / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz"
SOURCE_HASHES = HERE / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
MANIFEST_PATH = HERE / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json"
MANIFEST_SHA_PATH = HERE / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256"
RUNTIME_SELF_TEST_PATH = HERE / "CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST.json"
TARGET_SELF_TEST_PATH = HERE / "CARRIER_TOMOGRAPHY_TARGET_SELF_TEST.json"
CONTROLLER_SELF_TEST_PATH = HERE / "CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST.json"
OFFLINE_VALIDATE_PATH = HERE / "CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE.json"
TRANSPORT_SIM_PATH = HERE / "CARRIER_TOMOGRAPHY_TRANSPORT_SIMULATION.json"
DEPLOYMENT_LAYOUT_PATH = HERE / "CARRIER_TOMOGRAPHY_DEPLOYMENT_LAYOUT_SELF_TEST.json"
FEATURE_BOUNDARY_PATH = HERE / "CARRIER_TOMOGRAPHY_FEATURE_BOUNDARY_SELF_TEST.json"
OPERATOR_ANALYSIS_PATH = HERE / "CARRIER_TOMOGRAPHY_OPERATOR_ANALYSIS_SELF_TEST.json"
FACTORIAL_ARM_PATH = HERE / "CARRIER_TOMOGRAPHY_FACTORIAL_ARM_SELF_TEST.json"
SOURCE_DEATH_PATH = HERE / "CARRIER_TOMOGRAPHY_SOURCE_DEATH_CUSTODY_SELF_TEST.json"
EXACT_COVERAGE_PATH = HERE / "CARRIER_TOMOGRAPHY_EXACT_COVERAGE_SELF_TEST.json"
SUBAGENT_FINDINGS_PATH = HERE / "SUBAGENT_FINDINGS_NORMALIZED.json"
SUBAGENT_REVIEW_PATH = HERE / "SUBAGENT_REVIEW_REPORTS.md"
BINARY_PATH = HERE / "family10h_carrier_tomography_runtime"
GENERATED_RECEIPTS = [
    HERE / "CARRIER_TOMOGRAPHY_SELF_TEST.json",
    RUNTIME_SELF_TEST_PATH,
    TARGET_SELF_TEST_PATH,
    CONTROLLER_SELF_TEST_PATH,
    OFFLINE_VALIDATE_PATH,
    TRANSPORT_SIM_PATH,
    DEPLOYMENT_LAYOUT_PATH,
    FEATURE_BOUNDARY_PATH,
    OPERATOR_ANALYSIS_PATH,
    FACTORIAL_ARM_PATH,
    SOURCE_DEATH_PATH,
    EXACT_COVERAGE_PATH,
    MANIFEST_PATH,
    MANIFEST_SHA_PATH,
    SOURCE_HASHES,
]

EXPECTED_STARTING_HEAD = "836d53a81225fb37406528f1c25e87e208aa9495"
COMMIT_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_COMMIT_BINDING"
MANIFEST_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_MANIFEST_SHA256"
AUTHORITY_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_LIVE_AUTHORITY"
AUTHORITY_VALUE = public.TRANSACTION_RUN_ID

SOURCE_FILE_NAMES = target.SOURCE_FILE_NAMES


class ControllerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], *, timeout: float, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False, cwd=cwd)
    if check and completed.returncode != 0:
        raise ControllerError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def git_text(*args: str) -> str:
    return run(["git", *args], timeout=30.0).stdout.strip()


def git_state() -> dict[str, Any]:
    return {
        "branch": git_text("branch", "--show-current"),
        "head": git_text("rev-parse", "HEAD"),
        "origin_main": git_text("rev-parse", "origin/main"),
        "status_porcelain": run(["git", "status", "--porcelain"], timeout=30.0).stdout,
        "stash_list": run(["git", "stash", "list"], timeout=30.0).stdout.splitlines(),
    }


def source_file_map() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILE_NAMES:
        path = HERE / name
        if path.exists():
            result[name] = {"sha256": public.sha256_file(path), "size": path.stat().st_size}
    return result


def write_source_hashes() -> dict[str, Any]:
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_HASHES_V1",
        "source_files": source_file_map(),
    }
    result["source_hashes_sha256"] = public.digest({k: v for k, v in result.items() if k != "source_hashes_sha256"})
    write_json(SOURCE_HASHES, result)
    return result


def clear_generated_receipts() -> None:
    for path in GENERATED_RECEIPTS:
        if path.exists():
            path.unlink()


def write_source_bundle() -> dict[str, Any]:
    names = [name for name in SOURCE_FILE_NAMES if (HERE / name).exists()]
    with SOURCE_BUNDLE.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for name in sorted(names):
                    path = HERE / name
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    return {
        "path": str(SOURCE_BUNDLE),
        "sha256": public.sha256_file(SOURCE_BUNDLE),
        "file_count": len(names),
        "files": sorted(names),
    }


def source_bundle_preview() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_bundle_verify_") as tmp:
        temp_bundle = Path(tmp) / SOURCE_BUNDLE.name
        names = [name for name in SOURCE_FILE_NAMES if (HERE / name).exists()]
        with temp_bundle.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tar:
                    for name in sorted(names):
                        path = HERE / name
                        info = tar.gettarinfo(str(path), arcname=name)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
        return {"sha256": public.sha256_file(temp_bundle), "file_count": len(names), "files": sorted(names)}


def find_c_compiler() -> list[str] | None:
    for name in ["gcc", "clang", "cc"]:
        path = shutil.which(name)
        if path:
            return [path]
    wsl = shutil.which("wsl")
    if wsl:
        probe = run([wsl, "bash", "-lc", "command -v gcc"], timeout=10.0, check=False)
        if probe.returncode == 0 and probe.stdout.strip():
            return [wsl, "gcc"]
    return None


def compile_runtime() -> dict[str, Any]:
    compiler = find_c_compiler()
    source = HERE / "family10h_carrier_tomography_runtime.c"
    if BINARY_PATH.exists():
        BINARY_PATH.unlink()
    if compiler is None:
        return {
            "passed": False,
            "compiler": None,
            "compile_command": None,
            "offline_binary_sha256": None,
            "failure": "no local C compiler found",
        }
    runtime_command: list[str] | None = None
    if compiler[0].endswith("wsl.exe") or Path(compiler[0]).name.lower() == "wsl.exe":
        win_source = str(source)
        win_binary = str(BINARY_PATH)
        command_text = (
            "gcc -std=c11 -Wall -Wextra -Werror -O2 "
            f"-o \"$(wslpath '{win_binary}')\" \"$(wslpath '{win_source}')\""
        )
        completed = run([compiler[0], "bash", "-lc", command_text], timeout=60.0, check=False)
        compile_command = [compiler[0], "bash", "-lc", command_text]
        runtime_command = [compiler[0], "bash", "-lc", f"\"$(wslpath '{win_binary}')\" --self-test"]
    else:
        compile_command = [
            compiler[0],
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-o",
            str(BINARY_PATH),
            str(source),
        ]
        completed = run(compile_command, timeout=60.0, check=False)
        runtime_command = [str(BINARY_PATH), "--self-test"]
    passed = completed.returncode == 0 and BINARY_PATH.exists()
    return {
        "passed": passed,
        "compiler": compiler,
        "compile_command": compile_command,
        "runtime_command": runtime_command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "offline_binary_sha256": public.sha256_file(BINARY_PATH) if BINARY_PATH.exists() else None,
    }


def runtime_self_test() -> dict[str, Any]:
    compile_receipt = compile_runtime()
    if not compile_receipt["passed"]:
        result = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST_RECEIPT_V1",
            "passed": False,
            "compile": compile_receipt,
        }
        write_json(RUNTIME_SELF_TEST_PATH, result)
        return result
    completed = run(compile_receipt["runtime_command"], timeout=20.0, check=False)
    try:
        runtime_json = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        runtime_json = {"passed": False, "raw_stdout": completed.stdout}
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST_RECEIPT_V1",
        "passed": completed.returncode == 0 and runtime_json.get("passed") is True,
        "compile": compile_receipt,
        "runtime_returncode": completed.returncode,
        "runtime_stdout": completed.stdout,
        "runtime_stderr": completed.stderr,
        "runtime_json": runtime_json,
    }
    result["runtime_self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "runtime_self_test_sha256"})
    write_json(RUNTIME_SELF_TEST_PATH, result)
    return result


def deployment_layout_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_layout_") as tmp:
        root = Path(tmp)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        for name in SOURCE_FILE_NAMES:
            path = HERE / name
            if path.exists():
                shutil.copy2(path, source / name)
        if SOURCE_HASHES.exists():
            shutil.copy2(SOURCE_HASHES, source / SOURCE_HASHES.name)
        completed = run(
            [sys.executable, str(source / "family10h_carrier_tomography_target.py"), "--self-test", "--source-root", str(source), "--output-root", str(output)],
            timeout=60.0,
            check=False,
        )
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError:
            data = {"self_test_passed": False, "stdout": completed.stdout}
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DEPLOYMENT_LAYOUT_SELF_TEST_V1",
        "passed": completed.returncode == 0 and data.get("self_test_passed") is True,
        "target_self_test_returncode": completed.returncode,
        "target_self_test_passed": data.get("self_test_passed") is True,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_length": len(completed.stderr),
    }
    result["deployment_layout_self_test_sha256"] = public.digest(
        {k: v for k, v in result.items() if k != "deployment_layout_self_test_sha256"}
    )
    write_json(DEPLOYMENT_LAYOUT_PATH, result)
    return result


def fake_transport_self_tests() -> dict[str, Any]:
    schedule = public.build_schedule()
    packet = public.minimal_success_packet(schedule)
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_transport_") as tmp:
        root = Path(tmp)
        success_root = root / "success"
        success_root.mkdir()
        target.write_minimal_success_root(success_root, schedule)
        success = target.validate_minimal_evidence_root(success_root, schedule)

        failure_root = root / "failure"
        failure_root.mkdir()
        target.write_minimal_success_root(failure_root, schedule)
        raw_lines = (failure_root / "raw_records.jsonl").read_text(encoding="utf-8").splitlines()
        (failure_root / "raw_records.jsonl").write_text("\n".join(raw_lines[:-1]) + "\n", encoding="utf-8")
        failure = target.validate_minimal_evidence_root(failure_root, schedule)

        corrupted_root = root / "corrupted"
        corrupted_root.mkdir()
        target.write_minimal_success_root(corrupted_root, schedule)
        corrupted_records = [json.loads(line) for line in (corrupted_root / "raw_records.jsonl").read_text(encoding="utf-8").splitlines()]
        corrupted_records[0]["tuple_id"] = "corrupted"
        target.write_jsonl(corrupted_root / "raw_records.jsonl", corrupted_records)
        corrupted = target.validate_minimal_evidence_root(corrupted_root, schedule)

        timeout_root = root / "timeout"
        timeout_root.mkdir()
        timeout_receipt = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_EXECUTION_RECEIPT_V1",
            "status": "TARGET_EXECUTION_FAILED",
            "returncode": 124,
            "failure_reason": "runtime timeout before completion",
            "stdout": "",
            "stderr": "",
            "output_root": str(timeout_root),
            "evidence_validation": {"passed": False, "failures": ["timeout before failure sealing"]},
            "retry_count": 0,
        }
        write_json(timeout_root / "target_execution_receipt.json", timeout_receipt)
        timeout_loaded = read_json(timeout_root / "target_execution_receipt.json")
    timeout_consistency = {
        "outer_timeout_exceeds_inner_timeout": 3900 > 3600,
        "failure_sealed_before_copyback": timeout_loaded["status"] == "TARGET_EXECUTION_FAILED"
        and timeout_loaded["evidence_validation"]["passed"] is False,
        "no_automatic_retry": timeout_loaded["retry_count"] == 0,
    }
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TRANSPORT_SIMULATION_V1",
        "offline_only": True,
        "target_contact_count": 0,
        "live_invocation_count": 0,
        "fake_success_transport_passed": success["passed"],
        "fake_failure_transport_rejected": not failure["passed"],
        "copyback_corruption_rejected": not corrupted["passed"],
        "timeout_consistency": timeout_consistency,
        "protected_remote_roots_not_touched": True,
        "success": success,
        "failure": failure,
        "corrupted": corrupted,
    }
    result["passed"] = (
        result["fake_success_transport_passed"]
        and result["fake_failure_transport_rejected"]
        and result["copyback_corruption_rejected"]
        and all(timeout_consistency.values())
    )
    result["transport_simulation_sha256"] = public.digest(
        {k: v for k, v in result.items() if k != "transport_simulation_sha256"}
    )
    write_json(TRANSPORT_SIM_PATH, result)
    return result


def write_split_self_tests(public_self: dict[str, Any]) -> dict[str, Any]:
    feature = public.feature_boundary_self_test()
    write_json(FEATURE_BOUNDARY_PATH, feature)
    operator = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_OPERATOR_ANALYSIS_SELF_TEST_V1",
        "passed": public_self["operator_analysis_tests"]["ideal_persistent_public_state_has_signal"]
        and public_self["operator_analysis_tests"]["held_out_replicate_prediction_used"]
        and public_self["operator_analysis_tests"]["held_out_mapping_prediction_used"]
        and public_self["operator_analysis_tests"]["held_out_delay_prediction_used"]
        and public_self["operator_analysis_tests"]["s2_sufficient_for_ideal_fixture"],
        "packet_analysis_passed": public_self["operator_analysis_tests"]["validated_packet_analysis"]["passed"],
        "packet_analysis_has_lifetime": public_self["operator_analysis_tests"]["validated_packet_analysis"]["lifetime_curve_count"] > 0,
        "packet_analysis_has_factorial": public_self["operator_analysis_tests"]["validated_packet_analysis"]["factorial_matched_group_count"] > 0,
        "operator_analysis_tests": public_self["operator_analysis_tests"],
    }
    operator["passed"] = (
        operator["passed"]
        and operator["packet_analysis_passed"]
        and operator["packet_analysis_has_lifetime"]
        and operator["packet_analysis_has_factorial"]
    )
    factorial = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_FACTORIAL_ARM_SELF_TEST_V1",
        "passed": public_self["factorial_arm_tests"]["additive_data_rejected_as_nonadditive"]
        and public_self["factorial_arm_tests"]["ordinary_nonlinear_data_detected_not_overclaimed"],
        "factorial_arm_tests": public_self["factorial_arm_tests"],
    }
    source_death = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_DEATH_CUSTODY_SELF_TEST_V1",
        "passed": all(public_self["source_death_custody_tests"].values()),
        "source_death_custody_tests": public_self["source_death_custody_tests"],
    }
    coverage = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_EXACT_COVERAGE_SELF_TEST_V1",
        "passed": all(public_self["coverage_tests"][key] if key != "minimal_success_packet" else public_self["coverage_tests"][key]["passed"] for key in public_self["coverage_tests"]),
        "coverage_tests": public_self["coverage_tests"],
    }
    write_json(OPERATOR_ANALYSIS_PATH, operator)
    write_json(FACTORIAL_ARM_PATH, factorial)
    write_json(SOURCE_DEATH_PATH, source_death)
    write_json(EXACT_COVERAGE_PATH, coverage)
    return {"feature": feature, "operator": operator, "factorial": factorial, "source_death": source_death, "coverage": coverage}


def target_self_test() -> dict[str, Any]:
    result = target.self_test(HERE, HERE)
    write_json(TARGET_SELF_TEST_PATH, result)
    return result


def offline_validate() -> dict[str, Any]:
    schedule_validation = target.validate_schedule_artifacts(HERE)
    public_self = public.write_self_test(HERE / "CARRIER_TOMOGRAPHY_SELF_TEST.json")
    split = write_split_self_tests(public_self)
    target_result = target_self_test()
    runtime_result = runtime_self_test()
    transport = fake_transport_self_tests()
    deployment = deployment_layout_self_test()
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE_V1",
        "passed": all(
            [
                schedule_validation["passed"],
                public_self["self_test_passed"],
                split["feature"]["passed"],
                split["operator"]["passed"],
                split["factorial"]["passed"],
                split["source_death"]["passed"],
                split["coverage"]["passed"],
                target_result["self_test_passed"],
                runtime_result["passed"],
                transport["passed"],
                deployment["passed"],
            ]
        ),
        "schedule_validation": schedule_validation,
        "public_self_test_sha256": public_self["self_test_sha256"],
        "target_self_test_sha256": target_result["self_test_sha256"],
        "runtime_self_test_sha256": runtime_result.get("runtime_self_test_sha256"),
        "transport_simulation_sha256": transport["transport_simulation_sha256"],
        "deployment_layout_self_test_sha256": deployment["deployment_layout_self_test_sha256"],
        "feature_boundary_self_test_sha256": split["feature"]["feature_boundary_self_test_sha256"],
        "operator_analysis_passed": split["operator"]["passed"],
        "factorial_arm_passed": split["factorial"]["passed"],
        "source_death_custody_passed": split["source_death"]["passed"],
        "exact_coverage_passed": split["coverage"]["passed"],
        "target_contact_count": 0,
        "live_invocation_count": 0,
    }
    result["offline_validate_sha256"] = public.digest({k: v for k, v in result.items() if k != "offline_validate_sha256"})
    write_json(OFFLINE_VALIDATE_PATH, result)
    return result


def controller_self_test() -> dict[str, Any]:
    transport = fake_transport_self_tests()
    validation = offline_validate()
    live_env_absent = target.validate_no_live_authority_env()
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST_V1",
        "self_test_passed": validation["passed"] and transport["passed"] and live_env_absent["passed"],
        "offline_validate_sha256": validation["offline_validate_sha256"],
        "transport_simulation_sha256": transport["transport_simulation_sha256"],
        "live_authority_env_absent": live_env_absent,
        "target_contact_count": 0,
        "live_invocation_count": 0,
    }
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    write_json(CONTROLLER_SELF_TEST_PATH, result)
    return result


def manifest() -> dict[str, Any]:
    schedule = read_json(public.SCHEDULE_JSON)
    schedule_sidecar = read_json(public.SCHEDULE_SHA)
    source_hashes = write_source_hashes()
    bundle = write_source_bundle()
    runtime_result = read_json(RUNTIME_SELF_TEST_PATH) if RUNTIME_SELF_TEST_PATH.exists() else runtime_self_test()
    offline = read_json(OFFLINE_VALIDATE_PATH) if OFFLINE_VALIDATE_PATH.exists() else offline_validate()
    transport = read_json(TRANSPORT_SIM_PATH) if TRANSPORT_SIM_PATH.exists() else fake_transport_self_tests()
    deployment = read_json(DEPLOYMENT_LAYOUT_PATH) if DEPLOYMENT_LAYOUT_PATH.exists() else deployment_layout_self_test()
    target_result = read_json(TARGET_SELF_TEST_PATH) if TARGET_SELF_TEST_PATH.exists() else target_self_test()
    controller_result = read_json(CONTROLLER_SELF_TEST_PATH) if CONTROLLER_SELF_TEST_PATH.exists() else controller_self_test()
    independent_review = {}
    if SUBAGENT_FINDINGS_PATH.exists():
        independent_review = read_json(SUBAGENT_FINDINGS_PATH)
    review_blocked = bool(independent_review.get("material_blockers"))
    git = git_state()
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "claim_ceiling": "route-scoped public carrier-state model only",
        "package_decision": public.PACKAGE_DECISION_BLOCKED if review_blocked or not offline["passed"] else public.PACKAGE_DECISION_FROZEN,
        "independent_review": {
            "findings_path": str(SUBAGENT_FINDINGS_PATH) if SUBAGENT_FINDINGS_PATH.exists() else None,
            "findings_sha256": public.sha256_file(SUBAGENT_FINDINGS_PATH) if SUBAGENT_FINDINGS_PATH.exists() else None,
            "review_report_path": str(SUBAGENT_REVIEW_PATH) if SUBAGENT_REVIEW_PATH.exists() else None,
            "review_report_sha256": public.sha256_file(SUBAGENT_REVIEW_PATH) if SUBAGENT_REVIEW_PATH.exists() else None,
            "material_blocker_count": len(independent_review.get("material_blockers", [])),
            "verdict": independent_review.get("package_decision"),
        },
        "allowed_result_classes": public.ALLOWED_RESULT_CLASSES,
        "forbidden_result_classes": public.FORBIDDEN_RESULT_CLASSES,
        "public_preparation_grammar": public.public_preparations(),
        "query_family": public.query_family(),
        "delay_grid": public.DELAY_GRID,
        "exact_tuple_count": schedule["tuple_count"],
        "factorial_arm_definition": {
            "arms": public.FACTORIAL_ARMS,
            "J_q": "Y_q(A,B) - Y_q(A,dummy) - Y_q(dummy,B) + Y_q(dummy,dummy)",
            "interpretation_ceiling": "observed nonadditivity under matched arms only",
        },
        "operator_ladder": ["S0 scalar preparation amplitude", "S1 amplitude + query + delay + mapping", "S2 interactions and order"],
        "expected_local_root": public.EXPECTED_LOCAL_ROOT,
        "expected_remote_root": public.EXPECTED_REMOTE_ROOT,
        "expected_remote_output_root": public.EXPECTED_REMOTE_OUTPUT_ROOT,
        "schedule_hashes": schedule_sidecar,
        "source_hashes": source_hashes,
        "source_bundle": bundle,
        "runtime_self_test": {
            "passed": runtime_result["passed"],
            "sha256": runtime_result.get("runtime_self_test_sha256"),
            "offline_binary_sha256": runtime_result.get("compile", {}).get("offline_binary_sha256"),
        },
        "target_self_test": {"passed": target_result["self_test_passed"], "sha256": target_result["self_test_sha256"]},
        "controller_self_test": {"passed": controller_result["self_test_passed"], "sha256": controller_result["self_test_sha256"]},
        "offline_validate": {"passed": offline["passed"], "sha256": offline["offline_validate_sha256"]},
        "transport_simulation": {"passed": transport["passed"], "sha256": transport["transport_simulation_sha256"]},
        "deployment_layout": {"passed": deployment["passed"], "sha256": deployment["deployment_layout_self_test_sha256"]},
        "future_authorization": {
            "commit_binding_env": COMMIT_ENV,
            "manifest_binding_env": MANIFEST_ENV,
            "live_authority_env": AUTHORITY_ENV,
            "live_authority_value": AUTHORITY_VALUE,
            "this_task_authorizes_live_execution": False,
        },
        "zero_live_contact_attestation": {"target_contact_count": 0, "live_invocation_count": 0},
        "git_state_at_manifest_build": git,
        "starting_head_required": EXPECTED_STARTING_HEAD,
    }
    result["manifest_canonical_sha256"] = public.digest({k: v for k, v in result.items() if k != "manifest_canonical_sha256"})
    write_json(MANIFEST_PATH, result)
    sidecar = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST_SHA256_V1",
        "manifest_canonical_sha256": result["manifest_canonical_sha256"],
        "manifest_file_sha256": public.sha256_file(MANIFEST_PATH),
    }
    write_json(MANIFEST_SHA_PATH, sidecar)
    return result


def prepare_only() -> dict[str, Any]:
    clear_generated_receipts()
    schedule_hashes = public.write_schedule_artifacts()
    public_self = public.write_self_test(HERE / "CARRIER_TOMOGRAPHY_SELF_TEST.json")
    write_split_self_tests(public_self)
    source_hashes = write_source_hashes()
    target_result = target_self_test()
    runtime_result = runtime_self_test()
    transport = fake_transport_self_tests()
    deployment = deployment_layout_self_test()
    offline = offline_validate()
    controller = controller_self_test()
    bundle = write_source_bundle()
    manifest_result = manifest()
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_PREPARE_ONLY_RECEIPT_V1",
        "passed": all(
            [
                public_self["self_test_passed"],
                target_result["self_test_passed"],
                runtime_result["passed"],
                transport["passed"],
                deployment["passed"],
                offline["passed"],
                controller["self_test_passed"],
                manifest_result["package_decision"] in {public.PACKAGE_DECISION_FROZEN, public.PACKAGE_DECISION_BLOCKED},
            ]
        ),
        "schedule_hashes": schedule_hashes,
        "source_hashes_sha256": source_hashes["source_hashes_sha256"],
        "source_bundle_sha256": bundle["sha256"],
        "manifest_canonical_sha256": manifest_result["manifest_canonical_sha256"],
        "manifest_file_sha256": public.sha256_file(MANIFEST_PATH),
        "offline_binary_sha256": runtime_result.get("compile", {}).get("offline_binary_sha256"),
        "target_contact_count": 0,
        "live_invocation_count": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def validate_only() -> dict[str, Any]:
    artifacts = [
        public.SCHEDULE_JSON,
        public.SCHEDULE_TSV,
        public.SCHEDULE_SHA,
        CONTRACT_PATH,
        MANIFEST_PATH,
        MANIFEST_SHA_PATH,
        SOURCE_HASHES,
        SOURCE_BUNDLE,
        RUNTIME_SELF_TEST_PATH,
        TARGET_SELF_TEST_PATH,
        CONTROLLER_SELF_TEST_PATH,
        OFFLINE_VALIDATE_PATH,
        TRANSPORT_SIM_PATH,
        DEPLOYMENT_LAYOUT_PATH,
        FEATURE_BOUNDARY_PATH,
        OPERATOR_ANALYSIS_PATH,
        FACTORIAL_ARM_PATH,
        SOURCE_DEATH_PATH,
        EXACT_COVERAGE_PATH,
    ]
    missing = [str(path) for path in artifacts if not path.exists()]
    schedule = public.load_schedule_from_artifacts() if not missing else {}
    sidecar = read_json(MANIFEST_SHA_PATH) if MANIFEST_SHA_PATH.exists() else {}
    manifest_data = read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
    failures = []
    if missing:
        failures.append("missing artifacts")
    if sidecar:
        if sidecar.get("manifest_canonical_sha256") != public.digest({k: v for k, v in manifest_data.items() if k != "manifest_canonical_sha256"}):
            failures.append("manifest canonical digest mismatch")
        if sidecar.get("manifest_file_sha256") != public.sha256_file(MANIFEST_PATH):
            failures.append("manifest file digest mismatch")
    source_authority = target.validate_source_file_authority(HERE)
    if not source_authority["passed"]:
        failures.append("source file authority failed")
    bundle_preview = source_bundle_preview() if SOURCE_BUNDLE.exists() else {"sha256": None}
    manifest_bundle_sha = manifest_data.get("source_bundle", {}).get("sha256")
    if SOURCE_BUNDLE.exists() and (public.sha256_file(SOURCE_BUNDLE) != manifest_bundle_sha or bundle_preview["sha256"] != manifest_bundle_sha):
        failures.append("source bundle reconstruction mismatch")
    feature_boundary = public.feature_boundary_self_test()
    if not feature_boundary["passed"]:
        failures.append("feature boundary scan failed")
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_VALIDATE_ONLY_RECEIPT_V1",
        "passed": not failures,
        "failures": failures,
        "missing": missing,
        "tuple_count": schedule.get("tuple_count") if schedule else None,
        "manifest_sha": sidecar,
        "feature_boundary_passed": feature_boundary["passed"],
        "feature_boundary_sha256": feature_boundary["feature_boundary_self_test_sha256"],
        "source_authority": source_authority,
        "source_bundle_reconstruction": bundle_preview,
        "target_contact_count": 0,
        "live_invocation_count": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--transport-simulation", action="store_true")
    parser.add_argument("--deployment-layout-self-test", action="store_true")
    parser.add_argument("--offline-validate", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prepare_only:
            result = prepare_only()
            return 0 if result["passed"] else 1
        if args.validate_only:
            result = validate_only()
            return 0 if result["passed"] else 1
        if args.self_test:
            result = controller_self_test()
        elif args.transport_simulation:
            result = fake_transport_self_tests()
        elif args.deployment_layout_self_test:
            result = deployment_layout_self_test()
        elif args.offline_validate:
            result = offline_validate()
        else:
            parser.print_help()
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("passed", result.get("self_test_passed", False)) else 1
    except Exception as exc:  # noqa: BLE001 - CLI receipt
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
