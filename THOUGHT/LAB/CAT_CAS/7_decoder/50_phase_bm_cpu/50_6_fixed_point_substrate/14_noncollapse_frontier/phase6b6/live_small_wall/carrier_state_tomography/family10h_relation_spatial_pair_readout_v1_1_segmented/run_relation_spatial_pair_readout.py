#!/usr/bin/env python3
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

import relation_spatial_adjudication as synthetic_adjudication
import relation_spatial_physical_adjudication as physical_adjudication
import relation_spatial_public as pub


HERE = Path(__file__).resolve().parent
GRAMMAR_JSON = HERE / "RELATION_GRAMMAR.json"
GRAMMAR_TSV = HERE / "RELATION_GRAMMAR.tsv"
GRAMMAR_SHA = HERE / "RELATION_GRAMMAR.sha256"
SCHEDULE_JSON = HERE / "RELATION_SPATIAL_PUBLIC_SCHEDULE.json"
SCHEDULE_TSV = HERE / "RELATION_SPATIAL_PUBLIC_SCHEDULE.tsv"
SCHEDULE_SHA = HERE / "RELATION_SPATIAL_PUBLIC_SCHEDULE.sha256"
MANIFEST_JSON = HERE / "RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.json"
MANIFEST_SHA = HERE / "RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.sha256"
SOURCE_HASHES_JSON = HERE / "RELATION_SPATIAL_SOURCE_HASHES.json"
SOURCE_BUNDLE = HERE / "RELATION_SPATIAL_SOURCE_BUNDLE.tar.gz"
SENSOR_BINDING_JSON = HERE / "RELATION_SPATIAL_SENSOR_AUTHORITY_BINDING.json"
RUNTIME_BUILD_JSON = HERE / "RELATION_SPATIAL_RUNTIME_BUILD_SELF_TEST.json"
SYNTHETIC_EXECUTOR_JSON = HERE / "RELATION_SPATIAL_SYNTHETIC_EXECUTOR_SELF_TEST.json"
PHYSICAL_SELF_TEST_JSON = HERE / "RELATION_SPATIAL_PHYSICAL_ADJUDICATOR_SELF_TEST.json"
THRESHOLD_JSON = HERE / "RELATION_SPATIAL_PHYSICAL_THRESHOLD_CONTRACT.json"
SELF_TEST_JSON = HERE / "RELATION_SPATIAL_SELF_TEST.json"
READINESS_JSON = HERE / "RELATION_SPATIAL_BUILD_READINESS.json"
VALIDATE_JSON = HERE / "RELATION_SPATIAL_OFFLINE_VALIDATE.json"
TOOLCHAIN_JSON = HERE / "RELATION_SPATIAL_TOOLCHAIN_DISCOVERY.json"
PROOF_JSON = HERE / "RELATION_MARGINAL_EQUALITY_PROOF.json"
ADVERSARY_JSON = HERE / "RELATION_SPATIAL_ADVERSARY_TEST.json"
CONTRACT_MD = HERE / "RELATION_SPATIAL_CONTRACT.md"
RUNTIME_C = HERE / "relation_spatial_runtime.c"
RUNTIME_BIN = HERE / "relation_spatial_runtime"
PMU_C = HERE / "relation_spatial_pmu_preflight.c"
PMU_BIN = HERE / "relation_spatial_pmu_preflight"
SYNTHETIC_OUTPUT = HERE / "_relation_spatial_synthetic_executor_self_test_output"

SOURCE_FILES = [
    "relation_spatial_public.py",
    "relation_spatial_adjudication.py",
    "relation_spatial_physical_adjudication.py",
    "relation_spatial_runtime.c",
    "relation_spatial_runtime.h",
    "relation_spatial_pmu_preflight.c",
    "relation_spatial_target.py",
    "relation_spatial_live_controller.py",
    "run_relation_spatial_pair_readout.py",
]


def run(command: list[str], *, timeout: int = 120, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, cwd=cwd or HERE, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def git_text(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], text=True, capture_output=True, cwd=HERE, check=False)


def wsl_path(path: Path) -> str:
    completed = run(["wsl.exe", "wslpath", "-a", str(path.resolve())], timeout=20)
    if completed["returncode"] == 0 and completed["stdout"].strip():
        return completed["stdout"].strip()
    drive = path.resolve().drive.rstrip(":").lower()
    suffix = "/".join(path.resolve().parts[1:])
    return f"/mnt/{drive}/{suffix}"


def discover_toolchain() -> dict[str, Any]:
    candidates = []
    wsl = shutil.which("wsl.exe")
    if not wsl:
        return {"schema": "FAMILY10H_RELATION_SPATIAL_TOOLCHAIN_DISCOVERY_V1", "passed": False, "blockers": ["wsl.exe missing"], "candidates": []}
    gcc = run(["wsl.exe", "--", "bash", "-lc", "command -v gcc || command -v cc"], timeout=30)
    path = gcc["stdout"].strip().splitlines()[0] if gcc["returncode"] == 0 and gcc["stdout"].strip() else ""
    candidates.append({"launcher": "wsl.exe --", "compiler": path, "probe": gcc})
    return {
        "schema": "FAMILY10H_RELATION_SPATIAL_TOOLCHAIN_DISCOVERY_V1",
        "passed": bool(path),
        "target_compatible_compiler": {"available": bool(path), "launcher": ["wsl.exe", "--"], "path": path, "abi": "linux_x86_64"},
        "candidates": candidates,
    }


def ensure_runtime_schedule_sha(schedule_sha: str) -> None:
    text = RUNTIME_C.read_text(encoding="utf-8")
    text = re.sub(
        r'#define RELATION_SPATIAL_CANONICAL_SCHEDULE_SHA256 ".*"',
        f'#define RELATION_SPATIAL_CANONICAL_SCHEDULE_SHA256 "{schedule_sha}"',
        text,
    )
    if f'RELATION_SPATIAL_CANONICAL_SCHEDULE_SHA256 "{schedule_sha}"' not in text:
        text = text.replace('__SCHEDULE_SHA256__', schedule_sha)
    RUNTIME_C.write_text(text, encoding="utf-8", newline="\n")


def compile_c(toolchain: dict[str, Any], source: Path, output: Path, extra: list[str] | None = None) -> dict[str, Any]:
    compiler = toolchain.get("target_compatible_compiler", {})
    if not compiler.get("available"):
        return {"passed": False, "compile_attempted": False, "blockers": ["no target-compatible compiler"]}
    command = [
        "wsl.exe",
        "--",
        compiler["path"],
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-g",
        "-o",
        wsl_path(output),
        wsl_path(source),
        *(extra or []),
    ]
    result = run(command, timeout=120)
    file_probe = run(["wsl.exe", "--", "file", wsl_path(output)], timeout=30) if output.exists() else {"returncode": 1, "stdout": "", "stderr": "binary missing"}
    return {
        "passed": result["returncode"] == 0 and output.exists() and "ELF 64-bit" in file_probe.get("stdout", ""),
        "compile_attempted": True,
        "compile_command": command,
        "compile": result,
        "file_probe": file_probe,
        "compiled_binary_sha256": pub.sha256_file(output) if output.exists() else None,
    }


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.rstrip(b"\r\n"))


def run_runtime_build(schedule: dict[str, Any]) -> dict[str, Any]:
    toolchain = discover_toolchain()
    pub.write_json(TOOLCHAIN_JSON, toolchain)
    ensure_runtime_schedule_sha(schedule["schedule_sha256"])
    runtime_compile = compile_c(toolchain, RUNTIME_C, RUNTIME_BIN, ["-lm"])
    pmu_compile = compile_c(toolchain, PMU_C, PMU_BIN)
    runtime_self = run(["wsl.exe", "--", wsl_path(RUNTIME_BIN), "--self-test"], timeout=60) if runtime_compile["passed"] else {"returncode": 1, "stdout": "", "stderr": "runtime compile failed"}
    if SYNTHETIC_OUTPUT.exists():
        shutil.rmtree(SYNTHETIC_OUTPUT)
    synthetic = (
        run(["wsl.exe", "--", wsl_path(RUNTIME_BIN), "--synthetic-execute-schedule", wsl_path(SCHEDULE_TSV), wsl_path(SYNTHETIC_OUTPUT)], timeout=1200)
        if runtime_compile["passed"]
        else {"returncode": 1, "stdout": "", "stderr": "runtime compile failed"}
    )
    synthetic_counts = {
        "raw_record_count": count_jsonl(SYNTHETIC_OUTPUT / "raw_records.jsonl") if (SYNTHETIC_OUTPUT / "raw_records.jsonl").exists() else 0,
        "pair_observation_count": count_jsonl(SYNTHETIC_OUTPUT / "pair_observations.jsonl") if (SYNTHETIC_OUTPUT / "pair_observations.jsonl").exists() else 0,
        "source_death_receipt_count": count_jsonl(SYNTHETIC_OUTPUT / "source_death_receipts.jsonl") if (SYNTHETIC_OUTPUT / "source_death_receipts.jsonl").exists() else 0,
    }
    receipt = {
        "schema": "FAMILY10H_RELATION_SPATIAL_RUNTIME_BUILD_SELF_TEST_V1",
        "toolchain": toolchain,
        "runtime_compile": runtime_compile,
        "pmu_preflight_helper_compile": pmu_compile,
        "runtime_self_test": runtime_self,
        "synthetic_execution": synthetic,
        "synthetic_counts": synthetic_counts,
        "runtime_binary_authority": {
            "present": RUNTIME_BIN.exists(),
            "compiled_binary_sha256": pub.sha256_file(RUNTIME_BIN) if RUNTIME_BIN.exists() else None,
            "compiler_identity": toolchain.get("target_compatible_compiler", {}),
        },
        "pmu_preflight_helper_authority": {
            "present": PMU_BIN.exists(),
            "compiled_binary_sha256": pub.sha256_file(PMU_BIN) if PMU_BIN.exists() else None,
            "helper_c_sha256": pub.sha256_file(PMU_C) if PMU_C.exists() else None,
        },
    }
    receipt["passed"] = (
        runtime_compile["passed"]
        and pmu_compile["passed"]
        and runtime_self["returncode"] == 0
        and synthetic["returncode"] == 0
        and synthetic_counts["raw_record_count"] == schedule["tuple_count"]
        and synthetic_counts["pair_observation_count"] == schedule["expected_pair_observation_count"]
        and synthetic_counts["source_death_receipt_count"] == schedule["tuple_count"]
    )
    pub.write_json(RUNTIME_BUILD_JSON, receipt)
    pub.write_json(SYNTHETIC_EXECUTOR_JSON, {"schema": "FAMILY10H_RELATION_SPATIAL_SYNTHETIC_EXECUTOR_SELF_TEST_V1", **synthetic_counts, "passed": receipt["passed"]})
    return receipt


def schedule_json_manifest(schedule: dict[str, Any], tsv_sha256: str) -> dict[str, Any]:
    manifest = {
        "schema": "FAMILY10H_RELATION_SPATIAL_PUBLIC_SCHEDULE_MANIFEST_V3",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "canonical_expanded_schedule_artifact": SCHEDULE_TSV.name,
        "canonical_schedule_sha256": schedule["schedule_sha256"],
        "expanded_schedule_file_sha256": tsv_sha256,
        "tuple_count": schedule["tuple_count"],
        "pair_sample_count_per_row": pub.PAIR_SAMPLE_COUNT,
        "expected_pair_observation_count": schedule["expected_pair_observation_count"],
        "schedule_columns": pub.SCHEDULE_COLUMNS,
        "json_rows_omitted": True,
        "deterministic_generator": "relation_spatial_public.build_schedule",
        "claim_boundary": schedule["claim_boundary"],
    }
    manifest["schedule_manifest_sha256"] = pub.digest({k: v for k, v in manifest.items() if k != "schedule_manifest_sha256"})
    return manifest


def write_grammar_tsv(grammar: dict[str, Any]) -> None:
    lines = ["relation_id\tshift\tpermutation_sha256\tpair_distance_histogram_sha256"]
    for relation_id, item in sorted(grammar["relation_definitions"].items()):
        lines.append(f"{relation_id}\t{item['shift']}\t{item['permutation_sha256']}\t{item['pair_distance_histogram_sha256']}")
    GRAMMAR_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contract() -> None:
    CONTRACT_MD.write_text(
        "\n".join(
            [
                "# Family 10h Spatial Pair Readout V1",
                "",
                "This frozen package tests whether relation geometry is readable through spatially resolved first-touch latency pairs.",
                "",
                "- q is fixed at 0.",
                "- source_lifetime is fixed to alive_during_query.",
                "- each row measures 256 deterministic A/B line pairs exactly once.",
                "- primary coordinate: C_pair Spearman rank correlation.",
                "- relation coordinate: R_spatial = 0.5 * (r0/r0 + r1/r1 - r0/r1 - r1/r0).",
                "- calibrated result requires exceeding the frozen matched-permutation q99 null plus all stability and custody gates.",
                "- no result in this package may claim full tomography, relation memory, R2 restoration, catalytic borrowing, or SMALL_WALL_CROSSED.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def source_authority_validation(relation_source_authority: str | None) -> dict[str, Any]:
    value = relation_source_authority or pub.RELATION_SOURCE_AUTHORITY_UNSET
    failures = []
    if not re.fullmatch(r"[0-9a-f]{40}", value or ""):
        failures.append("relation source authority is not a 40-char git SHA")
    if value in pub.SCALAR_EVIDENCE_COMMITS:
        failures.append("relation source authority points at scalar evidence")
    if not failures:
        exists = git_text(["cat-file", "-e", f"{value}^{{commit}}"])
        if exists.returncode != 0:
            failures.append("relation source authority commit does not resolve")
    return {"passed": not failures, "failures": failures, "relation_source_authority_commit": value}


def source_authority_regressions(relation_source_authority: str | None) -> dict[str, Any]:
    value = relation_source_authority or pub.RELATION_SOURCE_AUTHORITY_UNSET
    return {
        "schema": "FAMILY10H_RELATION_SPATIAL_SOURCE_AUTHORITY_REGRESSION_V1",
        "passed": re.fullmatch(r"[0-9a-f]{40}", value or "") is not None and value not in pub.SCALAR_EVIDENCE_COMMITS,
        "rejects_scalar_evidence_commits": True,
        "rejects_unset_source_authority": value != pub.RELATION_SOURCE_AUTHORITY_UNSET,
    }


def build_source_hashes(relation_source_authority: str | None, runtime_build: dict[str, Any]) -> dict[str, Any]:
    source_paths = [HERE / name for name in SOURCE_FILES]
    bundle = pub.write_source_bundle(SOURCE_BUNDLE, source_paths)
    return {
        "schema": "FAMILY10H_RELATION_SPATIAL_SOURCE_HASHES_V1",
        "files": {path.name: {"sha256": pub.sha256_file(path), "size_bytes": path.stat().st_size} for path in source_paths},
        "source_bundle": bundle,
        "runtime_binary_authority": runtime_build["runtime_binary_authority"],
        "pmu_preflight_helper_authority": runtime_build["pmu_preflight_helper_authority"],
        "relation_source_authority_validation": source_authority_validation(relation_source_authority),
        "source_authority_regression_tests": source_authority_regressions(relation_source_authority),
    }


def sensor_binding() -> dict[str, Any]:
    return {
        "schema": "FAMILY10H_RELATION_SPATIAL_SENSOR_AUTHORITY_BINDING_V1",
        "approved_sensor_identity_sha256": pub.APPROVED_SENSOR_IDENTITY_SHA256,
        "approved_sensor_authority_sha256": pub.APPROVED_SENSOR_AUTHORITY_SHA256,
        "approved_sensor_identity": pub.APPROVED_SENSOR_IDENTITY,
        "approved_target_identity_sha256": pub.APPROVED_TARGET_IDENTITY_SHA256,
        "approved_target_identity": pub.APPROVED_TARGET_IDENTITY,
        "unlabeled_legacy_temp1_input_approved": True,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
    }


def implementation_manifest(
    grammar: dict[str, Any],
    schedule: dict[str, Any],
    runtime_build: dict[str, Any],
    source_hashes: dict[str, Any],
    self_test: dict[str, Any],
    validate: dict[str, Any],
    package_decision: str,
    blockers: list[str],
) -> dict[str, Any]:
    manifest = {
        "schema": "FAMILY10H_RELATION_SPATIAL_IMPLEMENTATION_MANIFEST_V1",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "package_decision": package_decision,
        "blockers": blockers,
        "grammar_sha256": grammar["grammar_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_tuple_count": schedule["tuple_count"],
        "pair_measurements_per_row": pub.PAIR_SAMPLE_COUNT,
        "expected_pair_observation_count": schedule["expected_pair_observation_count"],
        "runtime_binary_sha256": runtime_build["runtime_binary_authority"]["compiled_binary_sha256"],
        "pmu_preflight_helper_sha256": runtime_build["pmu_preflight_helper_authority"]["compiled_binary_sha256"],
        "source_bundle_sha256": source_hashes["source_bundle"]["sha256"],
        "self_test_sha256": self_test.get("self_test_sha256"),
        "offline_validate_sha256": validate.get("offline_validate_sha256"),
        "authority_binding": {
            "scalar_evidence_provenance": dict(pub.SCALAR_EVIDENCE_PROVENANCE),
            "relation_source_authority_commit": source_hashes["relation_source_authority_validation"]["relation_source_authority_commit"],
            "relation_manifest_freeze_commit_policy": pub.RELATION_FREEZE_AUTHORITY_POLICY,
            "relation_manifest_freeze_commit_not_embedded": True,
        },
        "claim_boundary": {
            "maximum_future_claim": pub.MAXIMUM_FUTURE_CLAIM,
            "forbidden_promotions": pub.FORBIDDEN_PROMOTIONS,
            "small_wall_crossed": False,
        },
    }
    manifest["manifest_sha256"] = pub.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return manifest


def self_test_result(grammar: dict[str, Any], schedule: dict[str, Any], runtime_build: dict[str, Any], physical_self: dict[str, Any]) -> dict[str, Any]:
    adversary = synthetic_adjudication.run_adversary_tests(schedule)
    pub.write_json(ADVERSARY_JSON, adversary)
    checks = {
        "grammar_validation_passed": pub.validate_grammar(grammar)["passed"],
        "schedule_validation_passed": pub.validate_schedule(schedule, grammar)["passed"],
        "runtime_compile_and_self_test_passed": runtime_build["passed"],
        "exact_pair_coverage_validation_passed": pub.relation_marginal_equality_proof(grammar, schedule)["checks"]["exact_pair_coverage"],
        "physical_adjudicator_self_test_passed": physical_self["passed"],
        "synthetic_adversary_cases_recorded": adversary["passed"],
    }
    result = {
        "schema": "FAMILY10H_RELATION_SPATIAL_SELF_TEST_V1",
        "tests": checks,
        "passed": all(checks.values()),
        "claim_boundary": {"small_wall_crossed": False},
    }
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def validate_artifacts() -> dict[str, Any]:
    failures = []
    grammar = json.loads(GRAMMAR_JSON.read_text(encoding="utf-8"))
    schedule = pub.build_schedule(grammar)
    schedule_manifest = json.loads(SCHEDULE_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    self_test = json.loads(SELF_TEST_JSON.read_text(encoding="utf-8"))
    source_hashes = json.loads(SOURCE_HASHES_JSON.read_text(encoding="utf-8"))
    if GRAMMAR_SHA.read_text(encoding="utf-8").strip() != pub.sha256_file(GRAMMAR_JSON):
        failures.append("grammar sha sidecar mismatch")
    if SCHEDULE_SHA.read_text(encoding="utf-8").strip() != pub.sha256_file(SCHEDULE_TSV):
        failures.append("schedule TSV sha sidecar mismatch")
    if MANIFEST_SHA.read_text(encoding="utf-8").strip() != pub.sha256_file(MANIFEST_JSON):
        failures.append("manifest sha sidecar mismatch")
    if not pub.validate_grammar(grammar)["passed"]:
        failures.append("grammar validation failed")
    if not pub.validate_schedule(schedule, grammar)["passed"]:
        failures.append("schedule validation failed")
    if not pub.validate_tsv(SCHEDULE_TSV, schedule)["passed"]:
        failures.append("schedule TSV validation failed")
    if schedule_manifest.get("schedule_manifest_sha256") != pub.digest({k: v for k, v in schedule_manifest.items() if k != "schedule_manifest_sha256"}):
        failures.append("schedule manifest digest mismatch")
    if schedule_manifest.get("canonical_schedule_sha256") != schedule["schedule_sha256"]:
        failures.append("schedule manifest canonical binding mismatch")
    if manifest.get("manifest_sha256") != pub.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        failures.append("manifest canonical digest mismatch")
    if manifest.get("authority_binding", {}).get("scalar_evidence_provenance") != dict(pub.SCALAR_EVIDENCE_PROVENANCE):
        failures.append("scalar provenance binding mismatch")
    for name, receipt in source_hashes.get("files", {}).items():
        if pub.sha256_file(HERE / name) != receipt.get("sha256"):
            failures.append(f"source hash mismatch: {name}")
    if SOURCE_BUNDLE.exists() and pub.sha256_file(SOURCE_BUNDLE) != source_hashes.get("source_bundle", {}).get("sha256"):
        failures.append("source bundle hash mismatch")
    package_decision = manifest.get("package_decision", pub.PACKAGE_DECISION_BLOCKED)
    return {
        "schema": "FAMILY10H_RELATION_SPATIAL_OFFLINE_VALIDATE_V1",
        "passed": not failures,
        "failures": failures,
        "package_decision": package_decision,
        "build_ready": package_decision == pub.PACKAGE_DECISION_BUILD_READY,
        "self_test_passed": self_test.get("passed") is True,
    }


def prepare(relation_source_authority: str | None = None) -> dict[str, Any]:
    write_contract()
    grammar = pub.relation_grammar(pub.PACKAGE_DECISION_BLOCKED)
    pub.write_json(GRAMMAR_JSON, grammar)
    write_grammar_tsv(grammar)
    GRAMMAR_SHA.write_text(pub.sha256_file(GRAMMAR_JSON) + "\n", encoding="utf-8")
    schedule = pub.build_schedule(grammar)
    pub.write_schedule_tsv(schedule, SCHEDULE_TSV)
    schedule_tsv_sha = pub.sha256_file(SCHEDULE_TSV)
    pub.write_compact_json(SCHEDULE_JSON, schedule_json_manifest(schedule, schedule_tsv_sha))
    SCHEDULE_SHA.write_text(schedule_tsv_sha + "\n", encoding="utf-8")
    runtime_build = run_runtime_build(schedule)
    proof = pub.relation_marginal_equality_proof(grammar, schedule, runtime_receipt=runtime_build)
    pub.write_json(PROOF_JSON, proof)
    threshold = physical_adjudication.physical_threshold_contract()
    pub.write_json(THRESHOLD_JSON, threshold)
    physical_self = physical_adjudication.run_self_test(schedule)
    pub.write_json(PHYSICAL_SELF_TEST_JSON, physical_self)
    self_test = self_test_result(grammar, schedule, runtime_build, physical_self)
    pub.write_json(SELF_TEST_JSON, self_test)
    pub.write_json(SENSOR_BINDING_JSON, sensor_binding())
    source_hashes = build_source_hashes(relation_source_authority, runtime_build)
    pub.write_json(SOURCE_HASHES_JSON, source_hashes)
    source_ready = source_hashes["relation_source_authority_validation"]["passed"] and source_hashes["source_authority_regression_tests"]["passed"]
    blockers = []
    if not runtime_build["passed"]:
        blockers.append("runtime or PMU helper build/self-test failed")
    if not physical_self["passed"]:
        blockers.append("physical adjudicator self-test failed")
    if not self_test["passed"]:
        blockers.append("package self-test failed")
    if relation_source_authority is None:
        blockers.append("relation source authority not bound until source commit")
    elif not source_ready:
        blockers.append("relation source authority validation failed")
    package_decision = pub.PACKAGE_DECISION_BUILD_READY if not blockers else pub.PACKAGE_DECISION_BLOCKED
    validate_stub = {"offline_validate_sha256": None}
    manifest = implementation_manifest(grammar, schedule, runtime_build, source_hashes, self_test, validate_stub, package_decision, blockers)
    pub.write_json(MANIFEST_JSON, manifest)
    MANIFEST_SHA.write_text(pub.sha256_file(MANIFEST_JSON) + "\n", encoding="utf-8")
    validate = validate_artifacts()
    pub.write_json(VALIDATE_JSON, validate)
    manifest = implementation_manifest(grammar, schedule, runtime_build, source_hashes, self_test, validate, package_decision, blockers)
    pub.write_json(MANIFEST_JSON, manifest)
    MANIFEST_SHA.write_text(pub.sha256_file(MANIFEST_JSON) + "\n", encoding="utf-8")
    validate = validate_artifacts()
    pub.write_json(VALIDATE_JSON, validate)
    readiness = {
        "schema": "FAMILY10H_RELATION_SPATIAL_BUILD_READINESS_V1",
        "package_decision": package_decision,
        "blockers": blockers,
        "schedule_row_count": schedule["tuple_count"],
        "pair_measurements_per_row": pub.PAIR_SAMPLE_COUNT,
        "expected_pair_observation_count": schedule["expected_pair_observation_count"],
        "line_coverage_proof": proof,
        "runtime_build_passed": runtime_build["passed"],
        "self_test_passed": self_test["passed"],
        "offline_validate_passed": validate["passed"],
        "target_contact_count": 0,
        "pmu_acquisition_count": 0,
        "small_wall_crossed": False,
    }
    readiness["readiness_sha256"] = pub.digest({k: v for k, v in readiness.items() if k != "readiness_sha256"})
    pub.write_json(READINESS_JSON, readiness)
    return {"passed": validate["passed"] and self_test["passed"], "package_decision": package_decision, "blockers": blockers, "manifest_sha256": manifest["manifest_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--relation-source-authority")
    args = parser.parse_args(argv)
    if args.prepare_only:
        result = prepare(args.relation_source_authority)
    elif args.validate_only:
        result = validate_artifacts()
        pub.write_json(VALIDATE_JSON, result)
    else:
        parser.error("select --prepare-only or --validate-only")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
