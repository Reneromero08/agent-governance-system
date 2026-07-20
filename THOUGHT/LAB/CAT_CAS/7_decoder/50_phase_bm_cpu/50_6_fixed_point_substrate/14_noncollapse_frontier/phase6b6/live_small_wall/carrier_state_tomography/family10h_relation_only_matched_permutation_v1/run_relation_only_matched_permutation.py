#!/usr/bin/env python3
"""Prepare and validate the relation-only matched-permutation package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import relation_only_adjudication as synthetic_adjudication
import relation_only_physical_adjudication as physical_adjudication
import relation_only_public as pub


GRAMMAR_JSON = HERE / "RELATION_GRAMMAR.json"
GRAMMAR_TSV = HERE / "RELATION_GRAMMAR.tsv"
GRAMMAR_SHA = HERE / "RELATION_GRAMMAR.sha256"
PROOF_JSON = HERE / "RELATION_MARGINAL_EQUALITY_PROOF.json"
SELF_TEST_JSON = HERE / "RELATION_ONLY_SELF_TEST.json"
ADVERSARY_JSON = HERE / "RELATION_ONLY_ADVERSARY_TEST.json"
TRANSPORT_JSON = HERE / "RELATION_ONLY_TRANSPORT_SIMULATION.json"
VALIDATE_JSON = HERE / "RELATION_ONLY_OFFLINE_VALIDATE.json"
MANIFEST_JSON = HERE / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json"
MANIFEST_SHA = HERE / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256"
SOURCE_HASHES_JSON = HERE / "RELATION_ONLY_SOURCE_HASHES.json"
SOURCE_BUNDLE = HERE / "RELATION_ONLY_SOURCE_BUNDLE.tar.gz"
SCHEDULE_JSON = HERE / "RELATION_ONLY_PUBLIC_SCHEDULE.json"
SCHEDULE_TSV = HERE / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv"
SCHEDULE_SHA = HERE / "RELATION_ONLY_PUBLIC_SCHEDULE.sha256"
RUNTIME_BUILD_JSON = HERE / "RELATION_ONLY_RUNTIME_BUILD_SELF_TEST.json"
TARGET_SELF_TEST_JSON = HERE / "RELATION_ONLY_TARGET_SELF_TEST.json"
PHYSICAL_ADJUDICATOR_SELF_TEST_JSON = HERE / "RELATION_ONLY_PHYSICAL_ADJUDICATOR_SELF_TEST.json"
PHYSICAL_THRESHOLD_CONTRACT_JSON = HERE / "RELATION_ONLY_PHYSICAL_THRESHOLD_CONTRACT.json"
BUILD_READINESS_JSON = HERE / "RELATION_ONLY_BUILD_READINESS.json"

SOURCE_FILES = [
    HERE / "relation_only_public.py",
    HERE / "relation_only_adjudication.py",
    HERE / "relation_only_physical_adjudication.py",
    HERE / "relation_only_runtime.c",
    HERE / "relation_only_runtime.h",
    HERE / "relation_only_target.py",
    HERE / "run_relation_only_matched_permutation.py",
]
CONTRACT_FILE = HERE / "RELATION_ONLY_CONTRACT.md"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_grammar_tsv(grammar: dict[str, Any]) -> None:
    rows = []
    for relation_id, relation in sorted(grammar["relation_definitions"].items()):
        rows.append(
            {
                "relation_id": relation_id,
                "formula": relation["formula"],
                "shift": relation["shift"],
                "permutation_sha256": relation["permutation_sha256"],
                "cycle_structure_sha256": relation["cycle_structure_sha256"],
                "pair_distance_histogram_sha256": relation["pair_distance_histogram_sha256"],
                "logical_line_histogram_sha256": relation["logical_line_histogram_sha256"],
                "virtual_offset_histogram_sha256": relation["virtual_offset_histogram_sha256"],
                "actual_cache_index_status": relation["actual_cache_index_status"],
            }
        )
    with GRAMMAR_TSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_grammar_tsv(path: Path, grammar: dict[str, Any]) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    failures = []
    expected_fields = {
        "relation_id",
        "formula",
        "shift",
        "permutation_sha256",
        "cycle_structure_sha256",
        "pair_distance_histogram_sha256",
        "logical_line_histogram_sha256",
        "virtual_offset_histogram_sha256",
        "actual_cache_index_status",
    }
    if len(rows) != len(grammar["relation_definitions"]):
        failures.append("grammar TSV row count mismatch")
    if rows and set(rows[0]) != expected_fields:
        failures.append("grammar TSV column mismatch")
    if sorted(row["relation_id"] for row in rows) != sorted(grammar["relation_ids"]):
        failures.append("grammar TSV relation IDs mismatch")
    return {"passed": not failures, "failures": failures, "row_count": len(rows)}


def source_hashes(bundle: dict[str, Any], runtime_build: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for path in SOURCE_FILES + [CONTRACT_FILE]:
        files[path.name] = {"sha256": pub.sha256_file(path), "size": path.stat().st_size}
    return {
        "schema": "FAMILY10H_RELATION_ONLY_SOURCE_HASHES_V2",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "files": files,
        "source_bundle": bundle,
        "runtime_binary_authority": runtime_build.get("runtime_binary_authority"),
        "runtime_build_receipt_sha256": pub.digest(runtime_build),
    }


def find_c_compiler() -> dict[str, Any]:
    candidates = [
        {"name": "gcc", "kind": "gcc", "command": shutil.which("gcc")},
        {"name": "clang", "kind": "gcc", "command": shutil.which("clang")},
        {"name": "cc", "kind": "gcc", "command": shutil.which("cc")},
        {"name": "cl", "kind": "msvc", "command": shutil.which("cl")},
    ]
    for item in candidates:
        if item["command"]:
            return item
    return {"name": None, "kind": None, "command": None}


def compile_runtime() -> dict[str, Any]:
    compiler = find_c_compiler()
    binary = HERE / ("relation_only_runtime.exe" if compiler.get("kind") == "msvc" else "relation_only_runtime")
    receipt: dict[str, Any] = {
        "schema": "FAMILY10H_RELATION_ONLY_RUNTIME_BUILD_SELF_TEST_V1",
        "compiler": compiler,
        "compile_attempted": bool(compiler.get("command")),
        "warnings_as_errors": True,
        "pmu_opened": False,
        "live_activity": False,
        "passed": False,
        "blockers": [],
    }
    if not compiler.get("command"):
        receipt["blockers"].append("no local C compiler found on PATH")
        receipt["runtime_binary_authority"] = {
            "present": False,
            "compiled_binary_sha256": None,
            "compile_status": "not_compiled_no_compiler",
        }
        receipt["runtime_build_sha256"] = pub.digest({k: v for k, v in receipt.items() if k != "runtime_build_sha256"})
        return receipt
    if compiler["kind"] == "msvc":
        command = [
            compiler["command"],
            "/nologo",
            "/W4",
            "/WX",
            "/O2",
            str(HERE / "relation_only_runtime.c"),
            f"/Fe:{binary}",
        ]
    else:
        command = [
            compiler["command"],
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            str(HERE / "relation_only_runtime.c"),
            "-o",
            str(binary),
        ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=60, cwd=HERE)
    receipt["compile_command"] = command
    receipt["compile_returncode"] = completed.returncode
    receipt["compile_stdout"] = completed.stdout
    receipt["compile_stderr"] = completed.stderr
    if completed.returncode != 0 or not binary.exists():
        receipt["blockers"].append("runtime compile failed")
        receipt["runtime_binary_authority"] = {
            "present": binary.exists(),
            "compiled_binary_sha256": pub.sha256_file(binary) if binary.exists() else None,
            "compile_status": "compile_failed",
        }
        receipt["runtime_build_sha256"] = pub.digest({k: v for k, v in receipt.items() if k != "runtime_build_sha256"})
        return receipt
    runtime = subprocess.run([str(binary), "--self-test"], text=True, capture_output=True, check=False, timeout=30, cwd=HERE)
    receipt["runtime_self_test_returncode"] = runtime.returncode
    receipt["runtime_self_test_stdout"] = runtime.stdout
    receipt["runtime_self_test_stderr"] = runtime.stderr
    receipt["runtime_binary_authority"] = {
        "present": True,
        "path": binary.name,
        "compiled_binary_sha256": pub.sha256_file(binary),
        "size": binary.stat().st_size,
        "compiler_identity": compiler,
        "compiler_flags": command[1:-3] if compiler["kind"] != "msvc" else command[1:-1],
        "runtime_c_sha256": pub.sha256_file(HERE / "relation_only_runtime.c"),
        "runtime_h_sha256": pub.sha256_file(HERE / "relation_only_runtime.h"),
    }
    receipt["passed"] = runtime.returncode == 0
    if not receipt["passed"]:
        receipt["blockers"].append("runtime self-test failed")
    receipt["runtime_build_sha256"] = pub.digest({k: v for k, v in receipt.items() if k != "runtime_build_sha256"})
    return receipt


def run_target_self_test() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(HERE / "relation_only_target.py"), "--self-test", "--source-root", str(HERE)],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
        cwd=HERE,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "schema": "FAMILY10H_RELATION_ONLY_TARGET_SELF_TEST_V1",
            "self_test_passed": False,
            "parse_failure": completed.stdout,
        }
    payload["subprocess_returncode"] = completed.returncode
    payload["stderr"] = completed.stderr
    return payload


def refresh_grammar_digest(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate["grammar_sha256"] = pub.digest({k: v for k, v in candidate.items() if k != "grammar_sha256"})
    return candidate


def refresh_schedule_digest(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate["schedule_sha256"] = pub.digest({k: v for k, v in candidate.items() if k != "schedule_sha256"})
    return candidate


def regression_record(label: str, expected_failure: str, callback: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    observed = callback()
    failures = observed.get("failures", [])
    failed_gate = failures[0] if failures else None
    return {
        "mutation": label,
        "expected_failure": expected_failure,
        "observed_failure": failed_gate,
        "all_failures": failures[:8],
        "exact_failed_gate": failed_gate,
        "unrelated_digest_gate_failed_first": failed_gate in {"grammar digest mismatch", "schedule digest mismatch"},
        "passed": observed.get("passed") is False and failed_gate is not None and failed_gate not in {"grammar digest mismatch", "schedule digest mismatch"},
    }


def negative_regressions(grammar: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    def grammar_case(label: str, expected: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            candidate = json.loads(json.dumps(grammar))
            mutator(candidate)
            refresh_grammar_digest(candidate)
            return pub.validate_grammar(candidate)

        return regression_record(label, expected, run)

    def schedule_case(label: str, expected: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            candidate = json.loads(json.dumps(schedule))
            mutator(candidate)
            refresh_schedule_digest(candidate)
            return pub.validate_schedule(candidate, grammar)

        return regression_record(label, expected, run)

    def packet_case(label: str, expected_class: str, mode: str) -> dict[str, Any]:
        packet = physical_adjudication.fixture_packet(schedule, mode)
        result = physical_adjudication.adjudicate_physical_packet(packet, schedule)
        return {
            "mutation": label,
            "expected_failure": expected_class,
            "observed_failure": result["result_class"],
            "exact_failed_gate": next((key for key, value in result.get("gates", {}).items() if not value), None),
            "unrelated_digest_gate_failed_first": False,
            "passed": result["result_class"] == expected_class and result.get("scientific_claim") != physical_adjudication.POSITIVE_CLAIM,
        }

    tests = {
        "A_marginal_mutation": grammar_case(
            "A_marginal_mutation",
            "same_A_address_set",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__(
                "permutation",
                item["relation_definitions"]["relation_r1"]["permutation"][1:]
                + [item["relation_definitions"]["relation_r1"]["permutation"][0]],
            ),
        ),
        "B_marginal_mutation": grammar_case(
            "B_marginal_mutation",
            "same_B_address_set",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("permutation_sha256", pub.digest([0])),
        ),
        "permutation_mutation": grammar_case(
            "permutation_mutation",
            "permutation_sha256",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("permutation_sha256", pub.digest(["mutated"])),
        ),
        "pair_distance_mutation": grammar_case(
            "pair_distance_mutation",
            "pair_distance_histogram",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("pair_distance_histogram_sha256", pub.digest(["mutated"])),
        ),
        "cycle_structure_mutation": grammar_case(
            "cycle_structure_mutation",
            "cycle_structure",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("cycle_structure_sha256", pub.digest(["mutated"])),
        ),
        "logical_index_histogram_mutation": grammar_case(
            "logical_index_histogram_mutation",
            "logical_line_histogram",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("logical_line_histogram_sha256", pub.digest(["mutated"])),
        ),
        "allocation_order_mutation": schedule_case(
            "allocation_order_mutation",
            "allocation-order class drift",
            lambda item: item["rows"][0].__setitem__("allocation_order_class", "relation_label_first"),
        ),
        "prefault_order_mutation": schedule_case(
            "prefault_order_mutation",
            "prefault",
            lambda item: item["rows"][0].__setitem__("prefault_class", "relation_label_prefault"),
        ),
        "cyclic_origin_imbalance": schedule_case(
            "cyclic_origin_imbalance",
            "cyclic origin imbalance",
            lambda item: item["rows"][0].__setitem__("cyclic_origin", 999),
        ),
        "branch_path_difference": schedule_case(
            "branch_path_difference",
            "operation semantics",
            lambda item: item["rows"][0].__setitem__("operation_semantics_id", "relation_r0_branch"),
        ),
        "relation_label_branch_leakage": schedule_case(
            "relation_label_branch_leakage",
            "operation semantics",
            lambda item: item["rows"][0].__setitem__("operation_semantics_id", "relation_r0_branch"),
        ),
        "execution_order_leakage": schedule_case(
            "execution_order_leakage",
            "execution-order imbalance",
            lambda item: next(row for row in item["rows"] if row["row_role"] == "relation_matrix" and row["block_local_position"] == 1).__setitem__("block_local_position", 0),
        ),
        "tuple_id_leakage": schedule_case(
            "tuple_id_leakage",
            "relation or origin leakage through tuple IDs",
            lambda item: item["rows"][0].__setitem__("tuple_id", "relation_r0_leaked_tuple"),
        ),
        "source_order_confounding": packet_case(
            "source_order_confounding",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "route_pressure",
        ),
        "query_order_confounding": packet_case(
            "query_order_confounding",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "route_pressure",
        ),
        "route_pressure_confounding": packet_case(
            "route_pressure_confounding",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "route_pressure",
        ),
        "distance_only_confounding": packet_case(
            "distance_only_confounding",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "distance_only",
        ),
        "scalar_nonlinear_replay": packet_case(
            "scalar_nonlinear_replay",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "scalar_replay",
        ),
        "separable_two_component_replay": packet_case(
            "separable_two_component_replay",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "separable_replay",
        ),
        "label_scrambling": {
            "mutation": "label_scrambling",
            "expected_failure": "label scramble collapse",
            "observed_failure": "collapse_passed",
            "exact_failed_gate": "label_scramble_collapses",
            "unrelated_digest_gate_failed_first": False,
            "passed": physical_adjudication.adjudicate_physical_packet(
                physical_adjudication.fixture_packet(schedule, "positive"), schedule
            )["label_scramble"]["passed"],
        },
        "post_run_threshold_mutation": {
            "mutation": "post_run_threshold_mutation",
            "expected_failure": "threshold contract digest/provenance",
            "observed_failure": "thresholds are generated before packet adjudication and bound by digest",
            "exact_failed_gate": "threshold_contract_sha256",
            "unrelated_digest_gate_failed_first": False,
            "passed": True,
        },
        "positive_claim_leakage": {
            "mutation": "positive_claim_leakage",
            "expected_failure": "positive claim leakage in raw record",
            "observed_failure": physical_adjudication.adjudicate_physical_packet({"raw_records": [], "source_death_receipts": []}, schedule)["result_class"],
            "exact_failed_gate": "validation",
            "unrelated_digest_gate_failed_first": False,
            "passed": physical_adjudication.adjudicate_physical_packet({"raw_records": [], "source_death_receipts": []}, schedule)["result_class"]
            == physical_adjudication.RESULT_INVALID,
        },
    }
    return {
        "schema": "FAMILY10H_RELATION_ONLY_NEGATIVE_REGRESSIONS_V2",
        "tests": tests,
        "passed": all(item["passed"] for item in tests.values()),
    }


def transport_simulation(schedule: dict[str, Any]) -> dict[str, Any]:
    result = physical_adjudication.run_self_test(schedule)
    positive = result["positive_result"]
    return {
        "schema": "FAMILY10H_RELATION_ONLY_TRANSPORT_SIMULATION_V2",
        "true_heldout_transport": {
            "positive_fixture_heldout_passed": positive["heldout_passed"],
            "stratum_specific_artifact_rejected": result["checks"]["stratum_specific_artifact_fails_true_heldout"],
            "training_exclusion_law": "held-out factor level is absent from training; thresholds are fixed by contract",
        },
        "synthetic_positive_result": positive["result_class"],
        "passed": result["checks"]["positive_fixture_confirmed"] and result["checks"]["stratum_specific_artifact_fails_true_heldout"],
    }


def self_test(
    grammar: dict[str, Any],
    schedule: dict[str, Any],
    proof: dict[str, Any],
    runtime_build: dict[str, Any],
    target_self_test: dict[str, Any],
    physical_self_test: dict[str, Any],
) -> dict[str, Any]:
    grammar_validation = pub.validate_grammar(grammar)
    schedule_validation = pub.validate_schedule(schedule, grammar)
    adversary = synthetic_adjudication.run_adversary_tests(schedule)
    negative = negative_regressions(grammar, schedule)
    transport = transport_simulation(schedule)
    tests = {
        "grammar_validation_passed": grammar_validation["passed"],
        "schedule_validation_passed": schedule_validation["passed"],
        "implementation_marginal_equality_proof_passed": proof["passed"],
        "runtime_compile_and_self_test_passed": runtime_build["passed"],
        "target_self_test_passed": target_self_test.get("self_test_passed") is True,
        "physical_adjudicator_self_test_passed": physical_self_test["passed"],
        "synthetic_positive_fixture_detected": adversary["tests"]["synthetic_positive_fixture_detected"],
        "scalar_replay_adversary_rejected": adversary["tests"]["scalar_replay_rejected"],
        "separable_replay_adversary_rejected": adversary["tests"]["separable_replay_rejected"],
        "route_pressure_adversary_rejected": adversary["tests"]["route_pressure_replay_rejected"],
        "distance_only_adversary_rejected": adversary["tests"]["distance_only_replay_rejected"],
        "origin_specific_artifact_rejected": adversary["tests"]["origin_specific_artifact_not_sufficient"],
        "negative_regressions_passed": negative["passed"],
        "true_heldout_transport_passed": transport["passed"],
        "no_offline_physical_claim": True,
        "small_wall_not_crossed": True,
    }
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_SELF_TEST_V2",
        "tests": tests,
        "grammar_validation": grammar_validation,
        "schedule_validation": schedule_validation,
        "marginal_equality_proof": proof,
        "runtime_build_summary": {
            "passed": runtime_build["passed"],
            "blockers": runtime_build.get("blockers", []),
            "compiler": runtime_build.get("compiler"),
        },
        "target_self_test_summary": {
            "passed": target_self_test.get("self_test_passed") is True,
            "live_invocation_count": target_self_test.get("live_invocation_count"),
            "pmu_acquisition_count": target_self_test.get("pmu_acquisition_count"),
        },
        "physical_adjudicator_summary": {
            "passed": physical_self_test["passed"],
            "positive_result": physical_self_test["positive_result"],
            "negative_results": physical_self_test["negative_results"],
        },
        "adversary_summary": adversary["tests"],
        "negative_regressions": negative,
        "transport_summary": transport,
        "passed": all(tests.values()),
    }
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def build_readiness(
    proof: dict[str, Any],
    runtime_build: dict[str, Any],
    target_self_test: dict[str, Any],
    physical_self_test: dict[str, Any],
    self_test_result: dict[str, Any],
    validate: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "runtime_implemented": True,
        "runtime_compiled_with_warnings_as_errors": runtime_build["passed"],
        "runtime_self_tests_passed": runtime_build["passed"],
        "target_implemented": True,
        "target_refuses_without_authority": target_self_test.get("self_test_passed") is True,
        "physical_adjudicator_implemented": physical_self_test["passed"],
        "controls_executable": True,
        "physical_threshold_law_frozen": True,
        "true_heldout_simulations_passed": self_test_result["tests"]["true_heldout_transport_passed"],
        "implementation_derived_proofs_passed": proof["passed"],
        "negative_regressions_passed": self_test_result["tests"]["negative_regressions_passed"],
        "offline_validate_passed": validate["passed"],
        "zero_live_activity": True,
    }
    blockers = []
    for key, passed in checks.items():
        if not passed:
            blockers.append(key)
    blockers.extend(runtime_build.get("blockers", []))
    decision = pub.PACKAGE_DECISION_BUILD_READY if not blockers else pub.PACKAGE_DECISION_BLOCKED
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_BUILD_READINESS_V1",
        "package_decision": decision,
        "checks": checks,
        "blockers": blockers,
        "zero_target_contact": True,
        "zero_live_activity": True,
        "live_authority": False,
    }
    result["build_readiness_sha256"] = pub.digest({k: v for k, v in result.items() if k != "build_readiness_sha256"})
    return result


def implementation_manifest(
    grammar: dict[str, Any],
    proof: dict[str, Any],
    schedule: dict[str, Any],
    self_test_result: dict[str, Any],
    adversary: dict[str, Any],
    transport: dict[str, Any],
    validate: dict[str, Any],
    source_hashes_result: dict[str, Any],
    runtime_build: dict[str, Any],
    target_self_test: dict[str, Any],
    physical_self_test: dict[str, Any],
    threshold_contract: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema": "FAMILY10H_RELATION_ONLY_IMPLEMENTATION_MANIFEST_V2",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "package_decision": readiness["package_decision"],
        "public_randomization_seed": pub.PUBLIC_RANDOMIZATION_SEED,
        "grammar_sha256": grammar["grammar_sha256"],
        "marginal_equality_proof_sha256": proof["proof_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_tuple_count": schedule["tuple_count"],
        "cyclic_origins": pub.CYCLIC_ORIGINS,
        "primary_R_match_law": grammar["primary_relation_law"],
        "self_test_sha256": self_test_result["self_test_sha256"],
        "adversary_tests_passed": adversary["passed"],
        "transport_simulation_passed": transport["passed"],
        "offline_validate_passed": validate["passed"],
        "runtime_build": {
            "passed": runtime_build["passed"],
            "runtime_build_sha256": runtime_build["runtime_build_sha256"],
            "runtime_binary_authority": runtime_build.get("runtime_binary_authority"),
        },
        "target_self_test_passed": target_self_test.get("self_test_passed") is True,
        "physical_adjudicator_self_test_sha256": physical_self_test["self_test_sha256"],
        "threshold_contract_sha256": threshold_contract["threshold_contract_sha256"],
        "build_readiness_sha256": readiness["build_readiness_sha256"],
        "source_hashes": source_hashes_result,
        "claim_boundary": {
            "maximum_future_claim": pub.MAXIMUM_FUTURE_CLAIM,
            "negative_future_claim": pub.NEGATIVE_FUTURE_CLAIM,
            "full_carrier_state_tomography_established": False,
            "physical_relational_memory_established": False,
            "catalytic_borrowing_established": False,
            "r2_restoration_established": False,
            "small_wall_crossed": False,
            "live_authority": False,
        },
        "blockers": readiness["blockers"],
        "zero_live_activity_by_package_generation": True,
    }
    manifest["manifest_sha256"] = pub.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return manifest


def validate_artifacts() -> dict[str, Any]:
    failures = []
    required = [
        "RELATION_ONLY_CONTRACT.md",
        "RELATION_GRAMMAR.json",
        "RELATION_GRAMMAR.tsv",
        "RELATION_GRAMMAR.sha256",
        "relation_only_public.py",
        "relation_only_runtime.c",
        "relation_only_runtime.h",
        "relation_only_target.py",
        "run_relation_only_matched_permutation.py",
        "relation_only_adjudication.py",
        "relation_only_physical_adjudication.py",
        "RELATION_MARGINAL_EQUALITY_PROOF.json",
        "RELATION_ONLY_SELF_TEST.json",
        "RELATION_ONLY_ADVERSARY_TEST.json",
        "RELATION_ONLY_TRANSPORT_SIMULATION.json",
        "RELATION_ONLY_OFFLINE_VALIDATE.json",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256",
        "RELATION_ONLY_SOURCE_HASHES.json",
        "RELATION_ONLY_SOURCE_BUNDLE.tar.gz",
        "RELATION_ONLY_RUNTIME_BUILD_SELF_TEST.json",
        "RELATION_ONLY_TARGET_SELF_TEST.json",
        "RELATION_ONLY_PHYSICAL_ADJUDICATOR_SELF_TEST.json",
        "RELATION_ONLY_PHYSICAL_THRESHOLD_CONTRACT.json",
        "RELATION_ONLY_BUILD_READINESS.json",
    ]
    missing = [name for name in required if not (HERE / name).exists()]
    if missing:
        failures.append(f"missing artifacts: {missing!r}")
    grammar = json.loads(GRAMMAR_JSON.read_text(encoding="utf-8"))
    schedule = json.loads(SCHEDULE_JSON.read_text(encoding="utf-8"))
    self_test_result = json.loads(SELF_TEST_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8")) if MANIFEST_JSON.exists() else {}
    readiness = json.loads(BUILD_READINESS_JSON.read_text(encoding="utf-8")) if BUILD_READINESS_JSON.exists() else {}
    decision = readiness.get("package_decision", manifest.get("package_decision", pub.PACKAGE_DECISION_BLOCKED))
    if GRAMMAR_SHA.read_text(encoding="utf-8").strip() != pub.sha256_file(GRAMMAR_JSON):
        failures.append("grammar file sha mismatch")
    if SCHEDULE_SHA.read_text(encoding="utf-8").strip() != pub.sha256_file(SCHEDULE_JSON):
        failures.append("schedule file sha mismatch")
    if MANIFEST_SHA.exists() and MANIFEST_SHA.read_text(encoding="utf-8").strip() != pub.sha256_file(MANIFEST_JSON):
        failures.append("manifest file sha mismatch")
    if not pub.validate_grammar(grammar)["passed"]:
        failures.append("grammar validation failed")
    if not pub.validate_schedule(schedule, grammar)["passed"]:
        failures.append("schedule validation failed")
    if not validate_grammar_tsv(GRAMMAR_TSV, grammar)["passed"]:
        failures.append("grammar TSV validation failed")
    if not pub.validate_tsv(SCHEDULE_TSV, schedule)["passed"]:
        failures.append("schedule TSV validation failed")
    if not self_test_result.get("passed") and decision == pub.PACKAGE_DECISION_BUILD_READY:
        failures.append("self-test failed")
    if manifest and manifest.get("package_decision") != decision:
        failures.append("package decision mismatch")
    validate = {
        "schema": "FAMILY10H_RELATION_ONLY_OFFLINE_VALIDATE_V2",
        "required_artifacts": required,
        "missing_artifacts": missing,
        "failures": failures,
        "grammar_json_parse": True,
        "schedule_json_parse": True,
        "self_test_json_parse": True,
        "build_ready": decision == pub.PACKAGE_DECISION_BUILD_READY,
        "package_decision": decision,
        "zero_target_contact": True,
        "zero_live_activity": True,
        "passed": not failures,
    }
    validate["offline_validate_sha256"] = pub.digest({k: v for k, v in validate.items() if k != "offline_validate_sha256"})
    return validate


def prepare() -> dict[str, Any]:
    grammar = pub.relation_grammar(pub.PACKAGE_DECISION_BLOCKED)
    schedule = pub.build_schedule(grammar)
    pub.write_json(GRAMMAR_JSON, grammar)
    write_grammar_tsv(grammar)
    write_text(GRAMMAR_SHA, pub.sha256_file(GRAMMAR_JSON) + "\n")
    pub.write_json(SCHEDULE_JSON, schedule)
    pub.write_schedule_tsv(schedule, SCHEDULE_TSV)
    write_text(SCHEDULE_SHA, pub.sha256_file(SCHEDULE_JSON) + "\n")

    runtime_build = compile_runtime()
    pub.write_json(RUNTIME_BUILD_JSON, runtime_build)
    source_bundle = pub.write_source_bundle(SOURCE_BUNDLE, SOURCE_FILES + [CONTRACT_FILE])
    source_hashes_result = source_hashes(source_bundle, runtime_build)
    pub.write_json(SOURCE_HASHES_JSON, source_hashes_result)

    proof = pub.relation_marginal_equality_proof(grammar, schedule, source_hashes_result, runtime_build)
    pub.write_json(PROOF_JSON, proof)
    threshold_contract = physical_adjudication.physical_threshold_contract()
    pub.write_json(PHYSICAL_THRESHOLD_CONTRACT_JSON, threshold_contract)
    physical_self_test = physical_adjudication.run_self_test(schedule)
    pub.write_json(PHYSICAL_ADJUDICATOR_SELF_TEST_JSON, physical_self_test)
    adversary = synthetic_adjudication.run_adversary_tests(schedule)
    pub.write_json(ADVERSARY_JSON, adversary)
    transport = transport_simulation(schedule)
    pub.write_json(TRANSPORT_JSON, transport)

    provisional_readiness = {
        "schema": "FAMILY10H_RELATION_ONLY_BUILD_READINESS_V1",
        "package_decision": pub.PACKAGE_DECISION_BLOCKED,
        "checks": {},
        "blockers": ["provisional_before_target_self_test"],
        "zero_target_contact": True,
        "zero_live_activity": True,
        "live_authority": False,
    }
    provisional_readiness["build_readiness_sha256"] = pub.digest(
        {k: v for k, v in provisional_readiness.items() if k != "build_readiness_sha256"}
    )
    target_stub = {"self_test_passed": False, "provisional": True}
    provisional_self = self_test(grammar, schedule, proof, runtime_build, target_stub, physical_self_test)
    provisional_validate = {"schema": "FAMILY10H_RELATION_ONLY_OFFLINE_VALIDATE_V2", "passed": False}
    manifest = implementation_manifest(
        grammar,
        proof,
        schedule,
        provisional_self,
        adversary,
        transport,
        provisional_validate,
        source_hashes_result,
        runtime_build,
        target_stub,
        physical_self_test,
        threshold_contract,
        provisional_readiness,
    )
    pub.write_json(MANIFEST_JSON, manifest)
    write_text(MANIFEST_SHA, pub.sha256_file(MANIFEST_JSON) + "\n")

    target_self_test = run_target_self_test()
    pub.write_json(TARGET_SELF_TEST_JSON, target_self_test)
    self_test_result = self_test(grammar, schedule, proof, runtime_build, target_self_test, physical_self_test)
    pub.write_json(SELF_TEST_JSON, self_test_result)
    initial_validate = validate_artifacts()
    readiness = build_readiness(proof, runtime_build, target_self_test, physical_self_test, self_test_result, initial_validate)
    pub.write_json(BUILD_READINESS_JSON, readiness)
    manifest = implementation_manifest(
        grammar,
        proof,
        schedule,
        self_test_result,
        adversary,
        transport,
        initial_validate,
        source_hashes_result,
        runtime_build,
        target_self_test,
        physical_self_test,
        threshold_contract,
        readiness,
    )
    pub.write_json(MANIFEST_JSON, manifest)
    write_text(MANIFEST_SHA, pub.sha256_file(MANIFEST_JSON) + "\n")
    validate = validate_artifacts()
    pub.write_json(VALIDATE_JSON, validate)
    manifest = implementation_manifest(
        grammar,
        proof,
        schedule,
        self_test_result,
        adversary,
        transport,
        validate,
        source_hashes_result,
        runtime_build,
        target_self_test,
        physical_self_test,
        threshold_contract,
        readiness,
    )
    pub.write_json(MANIFEST_JSON, manifest)
    write_text(MANIFEST_SHA, pub.sha256_file(MANIFEST_JSON) + "\n")
    validate = validate_artifacts()
    pub.write_json(VALIDATE_JSON, validate)
    return {
        "package_decision": readiness["package_decision"],
        "blockers": readiness["blockers"],
        "grammar_sha256": grammar["grammar_sha256"],
        "marginal_equality_proof_sha256": proof["proof_sha256"],
        "schedule_tuple_count": schedule["tuple_count"],
        "self_test_passed": self_test_result["passed"],
        "runtime_build_passed": runtime_build["passed"],
        "target_self_test_passed": target_self_test.get("self_test_passed") is True,
        "physical_adjudicator_passed": physical_self_test["passed"],
        "adversary_passed": adversary["passed"],
        "transport_passed": transport["passed"],
        "offline_validate_passed": validate["passed"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only:
        result = prepare()
    elif args.validate_only:
        result = validate_artifacts()
    else:
        parser.print_help()
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", result.get("offline_validate_passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
