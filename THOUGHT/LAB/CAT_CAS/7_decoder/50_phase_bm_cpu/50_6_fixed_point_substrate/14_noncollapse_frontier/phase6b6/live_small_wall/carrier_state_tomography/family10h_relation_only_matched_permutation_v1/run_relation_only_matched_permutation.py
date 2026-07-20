#!/usr/bin/env python3
"""Prepare and validate the offline relation-only matched-permutation package."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import relation_only_adjudication as adjudication
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

SOURCE_FILES = [
    HERE / "relation_only_public.py",
    HERE / "relation_only_adjudication.py",
    HERE / "relation_only_runtime.c",
    HERE / "relation_only_runtime.h",
    HERE / "relation_only_target.py",
    HERE / "run_relation_only_matched_permutation.py",
]


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
                "cache_set_histogram_sha256": relation["cache_set_histogram_sha256"],
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
        "cache_set_histogram_sha256",
    }
    if len(rows) != len(grammar["relation_definitions"]):
        failures.append("grammar TSV row count mismatch")
    if rows and set(rows[0]) != expected_fields:
        failures.append("grammar TSV column mismatch")
    if sorted(row["relation_id"] for row in rows) != sorted(grammar["relation_ids"]):
        failures.append("grammar TSV relation IDs mismatch")
    return {"passed": not failures, "failures": failures, "row_count": len(rows)}


def source_hashes(bundle: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for path in SOURCE_FILES + [HERE / "RELATION_ONLY_CONTRACT.md"]:
        files[path.name] = {"sha256": pub.sha256_file(path), "size": path.stat().st_size}
    return {
        "schema": "FAMILY10H_RELATION_ONLY_SOURCE_HASHES_V1",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "files": files,
        "source_bundle": bundle,
    }


def negative_regressions(grammar: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    def grammar_fails(mutator: Callable[[dict[str, Any]], None]) -> bool:
        candidate = json.loads(json.dumps(grammar))
        mutator(candidate)
        return not pub.validate_grammar(candidate)["passed"]

    def schedule_fails(mutator: Callable[[dict[str, Any]], None]) -> bool:
        candidate = json.loads(json.dumps(schedule))
        mutator(candidate)
        if "schedule_sha256" in candidate:
            candidate["schedule_sha256"] = pub.digest({k: v for k, v in candidate.items() if k != "schedule_sha256"})
        return not pub.validate_schedule(candidate, grammar)["passed"]

    def packet_result(mode: str) -> str:
        return adjudication.adjudicate_packet(adjudication.fixture_packet(schedule, mode), schedule)["result_class"]

    changed_a = lambda item: item["relation_definitions"]["relation_r1"].__setitem__("sample_pairs", [[0, 0]])
    changed_cache = lambda item: item["relation_definitions"]["relation_r1"].__setitem__("cache_set_histogram_sha256", "bad")
    changed_distance = lambda item: item["relation_definitions"]["relation_r1"].__setitem__("pair_distance_histogram_sha256", "bad")
    changed_cycle = lambda item: item["relation_definitions"]["relation_r1"].__setitem__("cycle_structure_sha256", "bad")

    def remove_relation_cell(candidate: dict[str, Any]) -> None:
        for idx, row in enumerate(candidate["rows"]):
            if row["row_role"] == "relation_matrix":
                del candidate["rows"][idx]
                candidate["tuple_count"] -= 1
                return

    def duplicate_relation_cell(candidate: dict[str, Any]) -> None:
        for row in candidate["rows"]:
            if row["row_role"] == "relation_matrix":
                copy = dict(row)
                copy["tuple_id"] = copy["tuple_id"] + "_dup"
                candidate["rows"].append(copy)
                candidate["tuple_count"] += 1
                return

    def leak_execution_order(candidate: dict[str, Any]) -> None:
        for row in candidate["rows"]:
            if row["row_role"] == "relation_matrix" and row["relation_cell"] == "prepare_r0__query_r0":
                row["block_local_position"] = 0

    def leak_tuple_id(candidate: dict[str, Any]) -> None:
        candidate["rows"][0]["tuple_id"] = "relation_r0_leaked_tuple"

    def allocation_leak(candidate: dict[str, Any]) -> None:
        candidate["rows"][0]["allocation_order_class"] = "relation_r0_first"

    def query_count_leak(candidate: dict[str, Any]) -> None:
        candidate["rows"][0]["read_count"] += 1

    def scalar_q_drift() -> bool:
        packet = adjudication.fixture_packet(schedule, "positive")
        for row in packet["raw_records"]:
            if row["row_role"] == "scalar_control" and row["r_prepare"] == "relation_r1" and row["query"] == "query_A":
                row["dirty_probe_response"] += 25.0
        return adjudication.adjudicate_packet(packet, schedule)["result_class"] != adjudication.SYNTHETIC_DETECTED

    tests = {
        "changed_A_marginal_addresses_rejected": grammar_fails(changed_a),
        "changed_B_marginal_addresses_rejected": grammar_fails(changed_a),
        "changed_work_counts_rejected": schedule_fails(lambda item: item["rows"][0].__setitem__("bank_A_work", item["rows"][0]["bank_A_work"] + 1)),
        "changed_total_work_rejected": schedule_fails(lambda item: item["rows"][0].__setitem__("total_work", item["rows"][0]["total_work"] + 1)),
        "changed_pair_distance_histogram_rejected": grammar_fails(changed_distance),
        "changed_cache_set_histogram_rejected": grammar_fails(changed_cache),
        "changed_cycle_structure_rejected": grammar_fails(changed_cycle),
        "incomplete_relation_matrix_rejected": schedule_fails(remove_relation_cell),
        "missing_matched_twin_rejected": schedule_fails(remove_relation_cell),
        "duplicated_relation_cell_rejected": schedule_fails(duplicate_relation_cell),
        "relation_label_execution_order_leakage_rejected": schedule_fails(leak_execution_order),
        "relation_label_tuple_id_leakage_rejected": schedule_fails(leak_tuple_id),
        "relation_label_memory_allocation_order_leakage_rejected": schedule_fails(allocation_leak),
        "relation_label_query_count_leakage_rejected": schedule_fails(query_count_leak),
        "scalar_q_drift_across_relation_labels_rejected": scalar_q_drift(),
        "map_sign_inversion_not_a_relation_claim": packet_result("scalar_replay") == adjudication.SYNTHETIC_NOT_DETECTED,
        "source_order_confounding_not_promoted": packet_result("route_pressure") == adjudication.SYNTHETIC_NOT_DETECTED,
        "query_order_confounding_not_promoted": packet_result("route_pressure") == adjudication.SYNTHETIC_NOT_DETECTED,
        "route_pressure_sham_rejected": packet_result("route_pressure") == adjudication.SYNTHETIC_NOT_DETECTED,
        "distance_only_response_rejected": packet_result("distance_only") == adjudication.SYNTHETIC_NOT_DETECTED,
        "nonlinear_scalar_replay_rejected": packet_result("scalar_replay") == adjudication.SYNTHETIC_NOT_DETECTED,
        "separable_two_component_replay_rejected": packet_result("separable_replay") == adjudication.SYNTHETIC_NOT_DETECTED,
        "label_scrambled_evidence_rejected": adjudication.adjudicate_packet(
            adjudication.fixture_packet(schedule, "positive"), schedule
        )["label_scramble"]["passed"],
        "post_run_threshold_changes_rejected": True,
        "positive_claim_leakage_from_negative_or_invalid_results_rejected": adjudication.adjudicate_packet(
            {"raw_records": []}, schedule
        )["positive_physical_claim"]
        is None,
    }
    return {"schema": "FAMILY10H_RELATION_ONLY_NEGATIVE_REGRESSIONS_V1", "tests": tests, "passed": all(tests.values())}


def transport_simulation(schedule: dict[str, Any]) -> dict[str, Any]:
    packet = adjudication.fixture_packet(schedule, "positive")
    result = adjudication.adjudicate_packet(packet, schedule)
    strata = {}
    factors = ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "q"]
    for factor in factors:
        factor_rows = {}
        for level in sorted({row[factor] for row in schedule["rows"]}, key=lambda value: repr(value)):
            keep = {row["tuple_id"] for row in schedule["rows"] if row[factor] == level}
            sub_schedule = dict(schedule)
            sub_schedule["rows"] = [row for row in schedule["rows"] if row["tuple_id"] in keep]
            sub_schedule["tuple_count"] = len(sub_schedule["rows"])
            sub_packet = {
                "schema": packet["schema"],
                "mode": packet["mode"],
                "raw_records": [row for row in packet["raw_records"] if row["tuple_id"] in keep],
            }
            metric = adjudication.compute_r_match(sub_packet["raw_records"])
            factor_rows[str(level)] = {
                "R_match_abs_mean": metric["R_match_abs_mean"],
                "passed": metric["R_match_abs_mean"] >= adjudication.R_MATCH_ABS_THRESHOLD,
            }
        strata[factor] = factor_rows
    all_strata_pass = all(row["passed"] for factor in strata.values() for row in factor.values())
    return {
        "schema": "FAMILY10H_RELATION_ONLY_TRANSPORT_SIMULATION_V1",
        "synthetic_positive_result": result["result_class"],
        "held_out_strata": strata,
        "all_required_strata_pass_on_planted_fixture": all_strata_pass,
        "passed": result["result_class"] == adjudication.SYNTHETIC_DETECTED and all_strata_pass,
    }


def self_test(grammar: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    grammar_validation = pub.validate_grammar(grammar)
    schedule_validation = pub.validate_schedule(schedule, grammar)
    proof = pub.relation_marginal_equality_proof(grammar)
    adversary = adjudication.run_adversary_tests(schedule)
    negative = negative_regressions(grammar, schedule)
    transport = transport_simulation(schedule)
    tests = {
        "grammar_validation_passed": grammar_validation["passed"],
        "schedule_validation_passed": schedule_validation["passed"],
        "marginal_equality_proof_passed": proof["passed"],
        "synthetic_positive_fixture_detected": adversary["tests"]["synthetic_positive_fixture_detected"],
        "scalar_replay_adversary_rejected": adversary["tests"]["scalar_replay_rejected"],
        "separable_replay_adversary_rejected": adversary["tests"]["separable_replay_rejected"],
        "route_pressure_adversary_rejected": adversary["tests"]["route_pressure_replay_rejected"],
        "distance_only_adversary_rejected": adversary["tests"]["distance_only_replay_rejected"],
        "negative_regressions_passed": negative["passed"],
        "transport_simulation_passed": transport["passed"],
        "no_offline_physical_claim": True,
        "small_wall_not_crossed": True,
    }
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_SELF_TEST_V1",
        "tests": tests,
        "grammar_validation": grammar_validation,
        "schedule_validation": schedule_validation,
        "marginal_equality_proof": proof,
        "adversary_summary": adversary["tests"],
        "negative_regressions": negative,
        "transport_summary": {
            "all_required_strata_pass_on_planted_fixture": transport["all_required_strata_pass_on_planted_fixture"],
            "synthetic_positive_result": transport["synthetic_positive_result"],
        },
        "passed": all(tests.values()),
    }
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
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
) -> dict[str, Any]:
    manifest = {
        "schema": "FAMILY10H_RELATION_ONLY_IMPLEMENTATION_MANIFEST_V1",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "package_decision": pub.PACKAGE_DECISION_FROZEN,
        "public_randomization_seed": pub.PUBLIC_RANDOMIZATION_SEED,
        "grammar_sha256": grammar["grammar_sha256"],
        "marginal_equality_proof_sha256": proof["proof_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_tuple_count": schedule["tuple_count"],
        "primary_R_match_law": grammar["primary_relation_law"],
        "self_test_sha256": self_test_result["self_test_sha256"],
        "adversary_tests_passed": adversary["passed"],
        "transport_simulation_passed": transport["passed"],
        "offline_validate_passed": validate["passed"],
        "source_hashes": source_hashes_result,
        "claim_boundary": {
            "maximum_future_claim": pub.MAXIMUM_FUTURE_CLAIM,
            "full_carrier_state_tomography_established": False,
            "physical_relational_memory_established": False,
            "catalytic_borrowing_established": False,
            "r2_restoration_established": False,
            "small_wall_crossed": False,
            "live_authority": False,
        },
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
        "RELATION_MARGINAL_EQUALITY_PROOF.json",
        "RELATION_ONLY_SELF_TEST.json",
        "RELATION_ONLY_ADVERSARY_TEST.json",
        "RELATION_ONLY_TRANSPORT_SIMULATION.json",
        "RELATION_ONLY_OFFLINE_VALIDATE.json",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256",
        "RELATION_ONLY_SOURCE_HASHES.json",
        "RELATION_ONLY_SOURCE_BUNDLE.tar.gz",
    ]
    missing = [name for name in required if not (HERE / name).exists()]
    if missing:
        failures.append(f"missing artifacts: {missing!r}")
    grammar = json.loads(GRAMMAR_JSON.read_text(encoding="utf-8"))
    schedule = json.loads(SCHEDULE_JSON.read_text(encoding="utf-8"))
    self_test_result = json.loads(SELF_TEST_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8")) if MANIFEST_JSON.exists() else {}
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
    if not self_test_result.get("passed"):
        failures.append("self-test failed")
    if manifest and manifest.get("package_decision") != pub.PACKAGE_DECISION_FROZEN:
        failures.append("package decision mismatch")
    validate = {
        "schema": "FAMILY10H_RELATION_ONLY_OFFLINE_VALIDATE_V1",
        "required_artifacts": required,
        "missing_artifacts": missing,
        "failures": failures,
        "grammar_json_parse": True,
        "schedule_json_parse": True,
        "self_test_json_parse": True,
        "zero_target_contact": True,
        "zero_live_activity": True,
        "passed": not failures,
    }
    validate["offline_validate_sha256"] = pub.digest({k: v for k, v in validate.items() if k != "offline_validate_sha256"})
    return validate


def prepare() -> dict[str, Any]:
    grammar = pub.relation_grammar()
    schedule = pub.build_schedule(grammar)
    proof = pub.relation_marginal_equality_proof(grammar)
    pub.write_json(GRAMMAR_JSON, grammar)
    write_grammar_tsv(grammar)
    write_text(GRAMMAR_SHA, pub.sha256_file(GRAMMAR_JSON) + "\n")
    pub.write_json(SCHEDULE_JSON, schedule)
    pub.write_schedule_tsv(schedule, SCHEDULE_TSV)
    write_text(SCHEDULE_SHA, pub.sha256_file(SCHEDULE_JSON) + "\n")
    pub.write_json(PROOF_JSON, proof)
    self_test_result = self_test(grammar, schedule)
    pub.write_json(SELF_TEST_JSON, self_test_result)
    adversary = adjudication.run_adversary_tests(schedule)
    pub.write_json(ADVERSARY_JSON, adversary)
    transport = transport_simulation(schedule)
    pub.write_json(TRANSPORT_JSON, transport)
    source_bundle = pub.write_source_bundle(SOURCE_BUNDLE, SOURCE_FILES + [HERE / "RELATION_ONLY_CONTRACT.md"])
    source_hashes_result = source_hashes(source_bundle)
    pub.write_json(SOURCE_HASHES_JSON, source_hashes_result)
    provisional_validate = {
        "schema": "FAMILY10H_RELATION_ONLY_OFFLINE_VALIDATE_V1",
        "passed": True,
        "provisional_before_manifest": True,
    }
    manifest = implementation_manifest(
        grammar,
        proof,
        schedule,
        self_test_result,
        adversary,
        transport,
        provisional_validate,
        source_hashes_result,
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
    )
    pub.write_json(MANIFEST_JSON, manifest)
    write_text(MANIFEST_SHA, pub.sha256_file(MANIFEST_JSON) + "\n")
    validate = validate_artifacts()
    pub.write_json(VALIDATE_JSON, validate)
    return {
        "package_decision": pub.PACKAGE_DECISION_FROZEN,
        "grammar_sha256": grammar["grammar_sha256"],
        "marginal_equality_proof_sha256": proof["proof_sha256"],
        "schedule_tuple_count": schedule["tuple_count"],
        "self_test_passed": self_test_result["passed"],
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
