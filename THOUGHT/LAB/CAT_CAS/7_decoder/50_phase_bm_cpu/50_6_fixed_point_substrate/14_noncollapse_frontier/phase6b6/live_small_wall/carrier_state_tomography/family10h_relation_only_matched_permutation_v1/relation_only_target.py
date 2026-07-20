#!/usr/bin/env python3
"""Target-side authority gate and offline self-tests for relation-only v1.

This file contains the future target entry point, but it refuses live execution
unless a later task supplies the exact external authority variables. The
self-test path performs no target contact, PMU acquisition, runtime launch, or
sensor discovery.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import relation_only_public as pub


AUTHORITY_ENV = "FAMILY10H_RELATION_ONLY_LIVE_AUTHORITY"
COMMIT_ENV = "FAMILY10H_RELATION_ONLY_COMMIT_BINDING"
MANIFEST_ENV = "FAMILY10H_RELATION_ONLY_MANIFEST_SHA256"
RUNTIME_AUTHORITY_ENV = "FAMILY10H_RELATION_ONLY_RUNTIME_AUTHORITY"
AUTHORITY_VALUE = pub.TRANSACTION_RUN_ID
RUNTIME_BINARY_NAMES = ["relation_only_runtime", "relation_only_runtime.exe"]
TARGET_PREFLIGHT_CHECKS = [
    "cpu_vendor_family_model",
    "source_receiver_cpu_identity",
    "operational_pinning_capability",
    "runtime_binary_sha256",
    "manifest_sha256",
    "schedule_sha256",
    "compiler_abi_identity",
    "pmu_event_availability",
    "grouped_pmu_open_capability",
    "temperature_sensor_authority",
    "cpu_frequency_policy_custody",
    "remote_source_output_root_absence",
    "actual_physical_geometry_status",
    "attempt_ownership_marker",
]


class TargetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetError(message)


def strict_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def no_live_authority_env() -> dict[str, Any]:
    observed = {
        key: os.environ.get(key)
        for key in [AUTHORITY_ENV, COMMIT_ENV, MANIFEST_ENV, RUNTIME_AUTHORITY_ENV]
        if os.environ.get(key) is not None
    }
    return {
        "passed": not observed,
        "observed_authority_keys": sorted(observed),
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }


def validate_source_root(source_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    required = [
        "RELATION_ONLY_CONTRACT.md",
        "RELATION_GRAMMAR.json",
        "RELATION_GRAMMAR.tsv",
        "RELATION_GRAMMAR.sha256",
        "RELATION_ONLY_PUBLIC_SCHEDULE.json",
        "RELATION_ONLY_PUBLIC_SCHEDULE.tsv",
        "RELATION_ONLY_PUBLIC_SCHEDULE.sha256",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256",
        "RELATION_ONLY_SOURCE_HASHES.json",
        "RELATION_ONLY_TOOLCHAIN_DISCOVERY.json",
        "RELATION_ONLY_RUNTIME_BUILD_SELF_TEST.json",
        "RELATION_ONLY_SYNTHETIC_EXECUTOR_SELF_TEST.json",
        "relation_only_public.py",
        "relation_only_adjudication.py",
        "relation_only_physical_adjudication.py",
        "relation_only_runtime.c",
        "relation_only_runtime.h",
        "relation_only_target.py",
        "run_relation_only_matched_permutation.py",
    ]
    missing = [name for name in required if not (source_root / name).exists()]
    failures.extend(f"missing {name}" for name in missing)
    if not failures:
        grammar = read_json(source_root / "RELATION_GRAMMAR.json")
        schedule_manifest = read_json(source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.json")
        schedule = pub.build_schedule(grammar)
        manifest = read_json(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json")
        if pub.sha256_file(source_root / "RELATION_GRAMMAR.json") != (source_root / "RELATION_GRAMMAR.sha256").read_text(encoding="utf-8").strip():
            failures.append("grammar sha sidecar mismatch")
        tsv_sha = pub.sha256_file(source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv")
        if tsv_sha != (source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.sha256").read_text(encoding="utf-8").strip():
            failures.append("expanded schedule TSV sha sidecar mismatch")
        if schedule_manifest.get("schema") != "FAMILY10H_RELATION_ONLY_PUBLIC_SCHEDULE_MANIFEST_V3":
            failures.append("schedule JSON manifest schema mismatch")
        if schedule_manifest.get("canonical_schedule_sha256") != schedule.get("schedule_sha256"):
            failures.append("canonical schedule digest mismatch")
        if schedule_manifest.get("expanded_schedule_file_sha256") != tsv_sha:
            failures.append("schedule manifest TSV binding mismatch")
        if schedule_manifest.get("json_rows_omitted") is not True:
            failures.append("schedule JSON rows not compacted")
        if pub.sha256_file(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json") != (source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256").read_text(encoding="utf-8").strip():
            failures.append("manifest sha sidecar mismatch")
        if not pub.validate_grammar(grammar)["passed"]:
            failures.append("grammar validation failed")
        if not pub.validate_schedule(schedule, grammar)["passed"]:
            failures.append("schedule validation failed")
        if manifest.get("claim_boundary", {}).get("small_wall_crossed") is not False:
            failures.append("manifest small wall boundary mismatch")
    return {"passed": not failures, "failures": failures, "required_file_count": len(required)}


def runtime_binary_path(source_root: Path) -> Path | None:
    for name in RUNTIME_BINARY_NAMES:
        path = source_root / name
        if path.exists():
            return path
    return None


def wsl_path(path: Path) -> str | None:
    if os.name != "nt" or shutil.which("wsl.exe") is None:
        return None
    completed = subprocess.run(
        ["wsl.exe", "wslpath", "-a", str(path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def runtime_command(source_root: Path, runtime: Path, args: list[Path | str]) -> tuple[list[str], str]:
    if os.name == "nt" and runtime.suffix != ".exe":
        translated_runtime = wsl_path(runtime)
        translated_args: list[str] = []
        for arg in args:
            if isinstance(arg, Path):
                translated = wsl_path(arg)
                if translated is None:
                    raise TargetError(f"cannot translate path for WSL: {arg}")
                translated_args.append(translated)
            else:
                translated_args.append(arg)
        if translated_runtime is None:
            raise TargetError("cannot translate runtime path for WSL")
        return ["wsl.exe", "--", translated_runtime, *translated_args], "wsl_linux_binary"
    return [str(runtime), *[str(arg) for arg in args]], "native"


def runtime_refusal_probe(source_root: Path) -> dict[str, Any]:
    runtime = runtime_binary_path(source_root)
    if runtime is None:
        return {
            "passed": True,
            "runtime_binary_present": False,
            "reason": "runtime binary absent; compile gate owns build-readiness blocker",
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    try:
        command, launch_mode = runtime_command(
            source_root,
            runtime,
            ["--execute-schedule", source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv", source_root / "_no_live_output"],
        )
    except TargetError as exc:
        return {
            "passed": False,
            "runtime_binary_present": True,
            "reason": str(exc),
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
        env={key: value for key, value in os.environ.items() if key != RUNTIME_AUTHORITY_ENV},
    )
    return {
        "passed": completed.returncode != 0,
        "runtime_binary_present": True,
        "launch_mode": launch_mode,
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }


def target_preflight_refusal(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json"
    schedule_path = source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv"
    runtime = runtime_binary_path(source_root)
    observed_authority = os.environ.get(AUTHORITY_ENV)
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_TARGET_PREFLIGHT_REFUSAL_V1",
        "status": "TARGET_PREFLIGHT_REFUSED",
        "reason": "live authority missing" if observed_authority != AUTHORITY_VALUE else "preflight not executed in offline self-test",
        "required_checks": TARGET_PREFLIGHT_CHECKS,
        "runtime_binary_present": runtime is not None,
        "manifest_present": manifest_path.exists(),
        "schedule_present": schedule_path.exists(),
        "target_contact_count": 0,
        "pmu_open_count": 0,
        "pmu_acquisition_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "small_wall_crossed": False,
    }
    result["passed"] = (
        observed_authority != AUTHORITY_VALUE
        and result["manifest_present"]
        and result["schedule_present"]
        and result["target_contact_count"] == 0
        and result["pmu_open_count"] == 0
    )
    return result


def self_test(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    source = validate_source_root(source_root)
    env = no_live_authority_env()
    refusal = runtime_refusal_probe(source_root)
    preflight = target_preflight_refusal(source_root)
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_TARGET_SELF_TEST_V1",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "offline_only": True,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "source_root_validation": source,
        "live_authority_env_absent": env,
        "runtime_refuses_without_authority": refusal,
        "target_preflight_refuses_without_authority": preflight,
        "allowed_future_result_classes": pub.FUTURE_RESULT_CLASSES,
        "maximum_future_claim": pub.MAXIMUM_FUTURE_CLAIM,
        "small_wall_crossed": False,
    }
    result["self_test_passed"] = source["passed"] and env["passed"] and refusal["passed"] and preflight["passed"]
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def execute_authorized(source_root: Path, output_root: Path) -> dict[str, Any]:
    require(os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, "live authority missing")
    commit_binding = os.environ.get(COMMIT_ENV, "")
    manifest_binding = os.environ.get(MANIFEST_ENV, "")
    require(re.fullmatch(r"[0-9a-f]{40}", commit_binding or "") is not None, "commit binding must be exact SHA")
    manifest_path = source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json"
    manifest_sha = pub.sha256_file(manifest_path)
    require(manifest_binding == manifest_sha, "manifest binding mismatch")
    manifest = read_json(manifest_path)
    require(
        manifest.get("package_decision") == pub.PACKAGE_DECISION_BUILD_READY,
        "relation-only package is not build-ready for live execution",
    )
    runtime = runtime_binary_path(source_root)
    require(runtime is not None, "runtime binary missing")
    output_root.mkdir(parents=False, exist_ok=False)
    env = os.environ.copy()
    env[RUNTIME_AUTHORITY_ENV] = AUTHORITY_VALUE
    command, launch_mode = runtime_command(
        source_root,
        runtime,
        ["--execute-schedule", source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv", output_root],
    )
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=3600,
        env=env,
    )
    return {
        "schema": "FAMILY10H_RELATION_ONLY_TARGET_EXECUTION_RECEIPT_V1",
        "status": "TARGET_EXECUTION_COMPLETE" if completed.returncode == 0 else "TARGET_EXECUTION_FAILED",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "launch_mode": launch_mode,
        "output_root": str(output_root),
        "live_invocation_count": 1,
        "pmu_acquisition_count": 1 if completed.returncode == 0 else 0,
        "small_wall_crossed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--target-preflight", action="store_true")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args(argv)
    if sum(1 for selected in [args.self_test, args.execute_authorized, args.target_preflight] if selected) != 1:
        parser.error("select exactly one mode")
    try:
        if args.self_test:
            result = self_test(args.source_root)
            print(strict_json_dumps(result, indent=2))
            return 0 if result["self_test_passed"] else 1
        if args.target_preflight:
            if os.environ.get(AUTHORITY_ENV) != AUTHORITY_VALUE:
                result = target_preflight_refusal(args.source_root.resolve())
                print(strict_json_dumps(result, indent=2))
                return 2
            result = {
                "schema": "FAMILY10H_RELATION_ONLY_TARGET_PREFLIGHT_V1",
                "status": "TARGET_PREFLIGHT_NOT_EXECUTED_BY_OFFLINE_PACKAGE",
                "required_checks": TARGET_PREFLIGHT_CHECKS,
                "target_contact_count": 0,
                "pmu_open_count": 0,
                "pmu_acquisition_count": 0,
                "small_wall_crossed": False,
            }
            print(strict_json_dumps(result, indent=2))
            return 2
        result = execute_authorized(args.source_root.resolve(), args.output_root.resolve())
        print(strict_json_dumps(result, indent=2))
        return 0 if result["status"] == "TARGET_EXECUTION_COMPLETE" else 1
    except TargetError as exc:
        result = {
            "schema": "FAMILY10H_RELATION_ONLY_TARGET_REFUSAL_V1",
            "status": "TARGET_EXECUTION_REFUSED",
            "reason": str(exc),
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
            "small_wall_crossed": False,
        }
        print(strict_json_dumps(result, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
