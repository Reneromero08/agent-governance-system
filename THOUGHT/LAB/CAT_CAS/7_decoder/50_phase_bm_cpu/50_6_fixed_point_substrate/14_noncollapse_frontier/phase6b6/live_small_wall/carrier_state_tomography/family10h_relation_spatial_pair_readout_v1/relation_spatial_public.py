#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent

SCIENCE_PACKAGE_ID = "family10h_relation_spatial_pair_readout_v1"
TRANSACTION_RUN_ID = "family10h_relation_spatial_pair_readout_v1_0"
PUBLIC_RANDOMIZATION_SEED = "family10h-relation-spatial-pair-readout-v1-public-seed-2a91db36"
SCALAR_EVIDENCE_SOURCE_AUTHORITY_COMMIT = "cac5d33536768e00aa0de5f515e626fecccdeeda"
SCALAR_EVIDENCE_MANIFEST_FREEZE_COMMIT = "354ca8ab2d62458fca41481d74ff98c1b39ab6ed"
SCALAR_EVIDENCE_POSTRUN_SEAL_COMMIT = "c126700b8d46e6501ff39cfa360bf32a9fbdb2ac"
RELATION_ONLY_EVIDENCE_COMMIT = "aaec66edfe536995a2d56498d72b6219a6084466"
RELATION_LIFETIME_EVIDENCE_COMMIT = "e3b77a56a919fd1afb090a7c9907daa1772d56dc"
SCALAR_EVIDENCE_PROVENANCE = {
    "source_authority_commit": SCALAR_EVIDENCE_SOURCE_AUTHORITY_COMMIT,
    "manifest_freeze_commit": SCALAR_EVIDENCE_MANIFEST_FREEZE_COMMIT,
    "postrun_seal_commit": SCALAR_EVIDENCE_POSTRUN_SEAL_COMMIT,
    "role": "sealed scalar q-readout baseline only; not used as relation-spatial target data",
}
SCALAR_EVIDENCE_COMMITS = {
    SCALAR_EVIDENCE_SOURCE_AUTHORITY_COMMIT,
    SCALAR_EVIDENCE_MANIFEST_FREEZE_COMMIT,
    SCALAR_EVIDENCE_POSTRUN_SEAL_COMMIT,
}
RELATION_SOURCE_AUTHORITY_UNSET = "RELATION_SOURCE_AUTHORITY_UNSET_UNTIL_SOURCE_COMMIT"
RELATION_FREEZE_AUTHORITY_POLICY = "controller_verifies_local_head_and_origin_equal_freeze_commit_at_authorization_time"
SYNTHETIC_RELATION_FREEZE_COMMIT = "f" * 40
DEPLOYMENT_CUSTODY_FILENAME = "RELATION_SPATIAL_DEPLOYMENT_CUSTODY.json"
SENSOR_AUTHORITY_BINDING_FILENAME = "RELATION_SPATIAL_SENSOR_AUTHORITY_BINDING.json"
OWNED_OUTPUT_PARENT_NAME = "_relation_spatial_owned_outputs"
ATTEMPT_CEILING = 1

PACKAGE_DECISION_BLOCKED = "FAMILY10H_SPATIAL_RELATION_READOUT_BUILD_READINESS_BLOCKED"
PACKAGE_DECISION_BUILD_READY = "FAMILY10H_SPATIAL_RELATION_READOUT_BUILD_READY_AWAITING_AUTHORIZATION"
FUTURE_RESULT_CLASSES = [
    "FAMILY10H_SPATIAL_RELATION_READOUT_CALIBRATED_PROSPECTIVE",
    "FAMILY10H_SPATIAL_RELATION_READOUT_NOT_OBSERVED_PROSPECTIVE",
    "FAMILY10H_SPATIAL_RELATION_READOUT_CANDIDATE_PROSPECTIVE",
    "FAMILY10H_SPATIAL_RELATION_READOUT_CUSTODY_INVALID",
]
MAXIMUM_FUTURE_CLAIM = "PUBLIC_POST_SOURCE_SPATIALLY_RESOLVED_FAMILY10H_RELATION_PAIR_READOUT_CALIBRATED"
NEGATIVE_FUTURE_CLAIM = "PUBLIC_POST_SOURCE_SPATIALLY_RESOLVED_FAMILY10H_RELATION_PAIR_READOUT_NOT_ESTABLISHED"
FORBIDDEN_PROMOTIONS = [
    "FULL_CARRIER_STATE_TOMOGRAPHY_ESTABLISHED",
    "PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED",
    "CATALYTIC_BORROWING_ESTABLISHED",
    "R2_RESTORATION_ESTABLISHED",
    "SMALL_WALL_CROSSED",
]

APPROVED_SENSOR_IDENTITY_SHA256 = "a993bb09ee6c38819f75f3be133cee859acc18ec8f62aa4ed8a57ee484fe1137"
APPROVED_SENSOR_AUTHORITY_SHA256 = "72ab7571611259b1844aacb808ab39b8549369b1e3e1baa23b16b33d2e5a0a0f"
APPROVED_SENSOR_IDENTITY = {
    "class_path": "/sys/class/hwmon/hwmon0/temp1_input",
    "device_driver": "k10temp",
    "device_modalias": "pci:v00001022d00001203sv00000000sd00000000bc06sc00i00",
    "device_subsystem": "pci",
    "hwmon_name": "k10temp",
    "identity_sha256": APPROVED_SENSOR_IDENTITY_SHA256,
    "input_st_dev": 22,
    "input_st_ino": 30803,
    "input_st_mode": 33060,
    "resolved_device_path": "/sys/devices/pci0000:00/0000:00:18.3",
    "resolved_driver_path": "/sys/bus/pci/drivers/k10temp",
    "resolved_hwmon_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon0",
    "resolved_input_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon0/temp1_input",
    "resolved_subsystem_path": "/sys/bus/pci",
    "sensor_input": "temp1_input",
    "sensor_label_present": False,
    "sensor_label_value": None,
    "sensor_semantic_profile": "LEGACY_FAMILY10H_K10TEMP_TEMP1_V1",
    "sensor_semantic_role": "Tctl",
}
APPROVED_TARGET_IDENTITY = {
    "vendor": "AuthenticAMD",
    "family": 16,
    "model": 10,
    "processor_count": 6,
    "source_cpu": 4,
    "receiver_cpu": 5,
    "processors": [
        {"processor": 0, "vendor_id": "AuthenticAMD", "cpu_family": 16, "model": 10},
        {"processor": 1, "vendor_id": "AuthenticAMD", "cpu_family": 16, "model": 10},
        {"processor": 2, "vendor_id": "AuthenticAMD", "cpu_family": 16, "model": 10},
        {"processor": 3, "vendor_id": "AuthenticAMD", "cpu_family": 16, "model": 10},
        {"processor": 4, "vendor_id": "AuthenticAMD", "cpu_family": 16, "model": 10},
        {"processor": 5, "vendor_id": "AuthenticAMD", "cpu_family": 16, "model": 10},
    ],
    "runtime_abi": "linux_x86_64_perf_event_open",
    "source": "sealed_family10h_temperature_sensor_authority_target_platform",
}

LINE_COUNT = 4096
LINE_BYTES = 64
LANE_BYTES = LINE_COUNT * LINE_BYTES
PAGE_COUNT_PER_LANE = 64
TOTAL_WORK = 4096
M = 2048
SESSIONS = ["session_0", "session_1"]
REPLICATES = [0, 1]
MAPPINGS = ["map0", "map1"]
SOURCE_ORDERS = ["A_then_B", "B_then_A"]
QUERY_ORDERS = ["AB", "BA"]
CYCLIC_ORIGINS = [0, 1024, 2048, 3072]
BLOCK_REPEATS = [0, 1, 2, 3]
DELAY_GRID = [{"delay_label": "same_window", "delay_ns": 0}]
SOURCE_LIFETIMES = ["alive_during_query"]
LIFETIME_HOLD_NS = 25_000_000
RELATIONS = ["relation_r0", "relation_r1"]
RELATION_CELLS = [
    ("relation_r0", "relation_r0"),
    ("relation_r0", "relation_r1"),
    ("relation_r1", "relation_r0"),
    ("relation_r1", "relation_r1"),
]
CONTROL_ROWS = [
    "relation_sham",
    "scrambled_pair_control",
    "route_pressure_control",
    "distance_matched_control",
]
PAIR_SAMPLE_COUNT = 256
PAIR_SAMPLE_STRIDE = 16
MATCHED_PERMUTATION_SEED = "family10h-spatial-pair-matched-null-seed-b6d4c002"
MATCHED_PERMUTATION_COUNT = 127

PMU_GROUP = {
    "name": "family10h_public_relation_match_group",
    "events": {
        "cpu_cycles_not_halted": {"event": "0x76", "umask": "0x00"},
        "cache_block_commands_change_to_dirty": {"event": "0xea", "umask": "0x20"},
        "probe_responses_dirty": {"event": "0xec", "umask": "0x0c"},
    },
}
PHYSICAL_GEOMETRY_STATUS = {
    "logical_line_index_histogram": "proven_offline_from_generated_tables",
    "virtual_offset_histogram": "proven_offline_from_fixed_64_byte_line_layout",
    "actual_physical_cache_index": "not_claimed_by_spatial_calibration",
}
RUNTIME_OPERATION_SEMANTICS = {
    "relation_matrix": "fresh carrier; alive source prepares public relation; receiver records 256 one-touch A/B latency pairs and preserves pair correspondence",
    "relation_sham": "fresh carrier; relation label is present but receiver uses a block-local sham pairing with matched A/B marginals",
    "scrambled_pair_control": "fresh carrier; receiver uses deterministic scrambled B partners while preserving selected A and B line sets",
    "route_pressure_control": "fresh carrier; receiver preserves two-load route pressure but destroys relation-pair correspondence",
    "distance_matched_control": "fresh carrier; receiver preserves circular pair-distance class but not the preparation/query relation law",
}
SCHEDULE_COLUMNS = [
    "tuple_id",
    "execution_ordinal",
    "block_id",
    "block_local_position",
    "row_role",
    "q",
    "bank_A_work",
    "bank_B_work",
    "total_work",
    "r_prepare",
    "r_query",
    "relation_match",
    "query",
    "relation_cell",
    "session",
    "replicate",
    "mapping",
    "delay_label",
    "delay_ns",
    "source_lifetime",
    "lifetime_pair_id",
    "lifetime_execution_order",
    "lifetime_hold_ns",
    "source_order",
    "query_order",
    "cyclic_origin",
    "route_pressure_class",
    "distance_control_class",
    "allocation_order_class",
    "prefault_class",
    "operation_semantics_id",
    "control_semantics_id",
    "source_cpu_expected",
    "receiver_cpu_expected",
    "source_loop_count",
    "receiver_loop_count",
    "read_count",
    "write_count",
    "page_count_A",
    "page_count_B",
    "line_count_A",
    "line_count_B",
    "logical_line_histogram_sha256",
    "virtual_offset_histogram_sha256",
    "actual_cache_index_status",
    "pair_distance_histogram_sha256",
    "permutation_cycle_structure_sha256",
    "matched_twin_group",
    "matched_twin_pair",
    "expected_pmu_group",
    "requires_pmu",
    "post_observation_scheduling",
]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


APPROVED_TARGET_IDENTITY_SHA256 = digest(APPROVED_TARGET_IDENTITY)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def write_compact_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def stable_token(value: Any, n: int = 20) -> str:
    return hashlib.sha256((PUBLIC_RANDOMIZATION_SEED + ":" + digest(value)).encode("utf-8")).hexdigest()[:n]


def relation_permutation(shift: int) -> list[int]:
    return [(idx + shift) % LINE_COUNT for idx in range(LINE_COUNT)]


def selected_pair_indices(execution_ordinal: int) -> list[int]:
    return [((execution_ordinal % LINE_COUNT) + idx * PAIR_SAMPLE_STRIDE) % LINE_COUNT for idx in range(PAIR_SAMPLE_COUNT)]


def circular_distance(a: int, b: int) -> int:
    delta = abs(a - b) % LINE_COUNT
    return min(delta, LINE_COUNT - delta)


def histogram(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def relation_definition(relation_id: str, shift: int) -> dict[str, Any]:
    perm = relation_permutation(shift)
    line_hist = histogram([x for idx, b in enumerate(perm) for x in (idx, b)])
    offset_hist = histogram([x for idx, b in enumerate(perm) for x in ((idx * LINE_BYTES) % 4096, (b * LINE_BYTES) % 4096)])
    dist_hist = histogram([circular_distance(idx, b) for idx, b in enumerate(perm)])
    cycle = [LINE_COUNT]
    return {
        "relation_id": relation_id,
        "formula": f"B_index = (A_index {'+' if shift > 0 else '-'} 1) mod 4096",
        "shift": shift,
        "permutation": perm,
        "permutation_sha256": digest(perm),
        "cycle_structure": cycle,
        "cycle_structure_sha256": digest(cycle),
        "pair_distance_histogram": dist_hist,
        "pair_distance_histogram_sha256": digest(dist_hist),
        "logical_line_histogram": line_hist,
        "logical_line_histogram_sha256": digest(line_hist),
        "virtual_offset_histogram": offset_hist,
        "virtual_offset_histogram_sha256": digest(offset_hist),
        "sample_pairs": [[idx, perm[idx]] for idx in range(16)],
    }


def relation_grammar(package_decision: str = PACKAGE_DECISION_BLOCKED) -> dict[str, Any]:
    relations = {"relation_r0": relation_definition("relation_r0", 1), "relation_r1": relation_definition("relation_r1", -1)}
    grammar = {
        "schema": "FAMILY10H_RELATION_SPATIAL_GRAMMAR_V1",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "public_randomization_seed": PUBLIC_RANDOMIZATION_SEED,
        "relation_definitions": relations,
        "relation_ids": RELATIONS,
        "cyclic_origins": CYCLIC_ORIGINS,
        "pair_sampling": {
            "pair_sample_count_per_row": PAIR_SAMPLE_COUNT,
            "pair_sample_stride": PAIR_SAMPLE_STRIDE,
            "pair_index_formula": "pair_index = (execution_ordinal + sample_index * 16) mod 4096",
            "coverage_law": "across 4096 rows each A line and each relation-permuted B line appears exactly 256 times",
            "post_observation_selection": False,
        },
        "physical_marginals": {
            "q": 0,
            "source_lifetime": "alive_during_query",
            "source_cpu_expected": 4,
            "receiver_cpu_expected": 5,
            "pmu_group": PMU_GROUP,
            "line_count_per_lane": LINE_COUNT,
            "line_bytes": LINE_BYTES,
            "pair_measurements_per_row": PAIR_SAMPLE_COUNT,
        },
        "operation_semantics": RUNTIME_OPERATION_SEMANTICS,
        "primary_relation_law": {
            "name": "R_spatial",
            "row_coordinate": "C_pair = Spearman(centered A first-touch latencies, paired B first-touch latencies)",
            "formula": "0.5 * (C_pair(r0,r0) + C_pair(r1,r1) - C_pair(r0,r1) - C_pair(r1,r0))",
            "threshold": "observed abs mean must exceed frozen matched-permutation q99",
            "matched_permutation_seed": MATCHED_PERMUTATION_SEED,
            "matched_permutation_count": MATCHED_PERMUTATION_COUNT,
        },
        "claim_boundary": {
            "offline_package_decision": package_decision,
            "maximum_future_claim": MAXIMUM_FUTURE_CLAIM,
            "negative_future_claim": NEGATIVE_FUTURE_CLAIM,
            "forbidden_promotions": FORBIDDEN_PROMOTIONS,
            "live_authority": False,
            "small_wall_crossed": False,
        },
    }
    grammar["grammar_sha256"] = digest({k: v for k, v in grammar.items() if k != "grammar_sha256"})
    return grammar


def relation_marginal_equality_proof(
    grammar: dict[str, Any],
    schedule: dict[str, Any] | None = None,
    source_hashes: dict[str, Any] | None = None,
    runtime_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rels = grammar["relation_definitions"]
    rows = schedule.get("rows", []) if schedule else []
    selected = Counter()
    for row in rows:
        for idx in selected_pair_indices(int(row["execution_ordinal"])):
            selected[idx] += 1
    coverage_ok = not rows or (len(selected) == LINE_COUNT and min(selected.values()) == max(selected.values()) == PAIR_SAMPLE_COUNT)
    checks = {
        "relations_distinct": rels["relation_r0"]["permutation_sha256"] != rels["relation_r1"]["permutation_sha256"],
        "same_A_address_set": True,
        "same_B_address_set": sorted(rels["relation_r0"]["permutation"]) == sorted(rels["relation_r1"]["permutation"]) == list(range(LINE_COUNT)),
        "same_logical_line_histogram": rels["relation_r0"]["logical_line_histogram_sha256"] == rels["relation_r1"]["logical_line_histogram_sha256"],
        "same_virtual_offset_histogram": rels["relation_r0"]["virtual_offset_histogram_sha256"] == rels["relation_r1"]["virtual_offset_histogram_sha256"],
        "same_pair_distance_histogram": rels["relation_r0"]["pair_distance_histogram_sha256"] == rels["relation_r1"]["pair_distance_histogram_sha256"],
        "exact_pair_coverage": coverage_ok,
        "q_zero_only": not rows or {row["q"] for row in rows} == {0},
        "alive_only": not rows or {row["source_lifetime"] for row in rows} == {"alive_during_query"},
        "small_wall_not_claimed": True,
    }
    return {
        "schema": "FAMILY10H_RELATION_SPATIAL_MARGINAL_EQUALITY_PROOF_V1",
        "passed": all(checks.values()),
        "checks": checks,
        "relation_r0": rels["relation_r0"],
        "relation_r1": rels["relation_r1"],
        "pair_sample_count_per_row": PAIR_SAMPLE_COUNT,
        "line_coverage_count": min(selected.values()) if selected else None,
        "line_coverage_sha256": digest(dict(sorted((str(k), v) for k, v in selected.items()))) if selected else None,
    }


def rotated(items: list[Any], offset: int) -> list[Any]:
    if not items:
        return []
    offset %= len(items)
    return items[offset:] + items[:offset]


def base_conditions() -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for session in SESSIONS:
        for replicate in REPLICATES:
            for mapping in MAPPINGS:
                for source_order in SOURCE_ORDERS:
                    for query_order in QUERY_ORDERS:
                        for cyclic_origin in CYCLIC_ORIGINS:
                            for block_repeat in BLOCK_REPEATS:
                                for delay in DELAY_GRID:
                                    conditions.append(
                                        {
                                            "session": session,
                                            "replicate": replicate,
                                            "mapping": mapping,
                                            "source_order": source_order,
                                            "query_order": query_order,
                                            "cyclic_origin": cyclic_origin,
                                            "block_repeat": block_repeat,
                                            "delay_label": delay["delay_label"],
                                            "delay_ns": delay["delay_ns"],
                                        }
                                    )
    return sorted(conditions, key=lambda item: stable_token(item))


def relation_cell_name(r_prepare: str, r_query: str) -> str:
    return f"prepare_{r_prepare[-2:]}__query_{r_query[-2:]}"


def schedule_row_base(condition: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "q": 0,
        "bank_A_work": M,
        "bank_B_work": M,
        "total_work": TOTAL_WORK,
        "session": condition["session"],
        "replicate": condition["replicate"],
        "mapping": condition["mapping"],
        "delay_label": condition["delay_label"],
        "delay_ns": condition["delay_ns"],
        "source_lifetime": "alive_during_query",
        "lifetime_pair_id": f"spatial_block_repeat_{condition['block_repeat']}",
        "lifetime_execution_order": "alive_only",
        "lifetime_hold_ns": LIFETIME_HOLD_NS,
        "source_order": condition["source_order"],
        "query_order": condition["query_order"],
        "cyclic_origin": condition["cyclic_origin"],
        "route_pressure_class": "spatial_two_load_pair_probe",
        "distance_control_class": "one_step_distance_matched_for_r0_r1",
        "allocation_order_class": "relation_label_independent_prefault_spatial_v1",
        "prefault_class": "all_lanes_prefaulted_before_source_fork",
        "source_cpu_expected": 4,
        "receiver_cpu_expected": 5,
        "source_loop_count": TOTAL_WORK,
        "receiver_loop_count": PAIR_SAMPLE_COUNT * 2,
        "read_count": PAIR_SAMPLE_COUNT * 2,
        "write_count": TOTAL_WORK,
        "page_count_A": PAGE_COUNT_PER_LANE,
        "page_count_B": PAGE_COUNT_PER_LANE,
        "line_count_A": LINE_COUNT,
        "line_count_B": LINE_COUNT,
        "actual_cache_index_status": PHYSICAL_GEOMETRY_STATUS["actual_physical_cache_index"],
        "expected_pmu_group": PMU_GROUP["name"],
        "requires_pmu": True,
        "post_observation_scheduling": False,
        **hashes,
    }


def build_schedule(grammar: dict[str, Any]) -> dict[str, Any]:
    proof = relation_marginal_equality_proof(grammar)
    hashes = {
        "logical_line_histogram_sha256": proof["relation_r0"]["logical_line_histogram_sha256"],
        "virtual_offset_histogram_sha256": proof["relation_r0"]["virtual_offset_histogram_sha256"],
        "pair_distance_histogram_sha256": proof["relation_r0"]["pair_distance_histogram_sha256"],
        "permutation_cycle_structure_sha256": proof["relation_r0"]["cycle_structure_sha256"],
    }
    rows: list[dict[str, Any]] = []
    ordinal = 0
    conditions = base_conditions()
    for block_index, condition in enumerate(conditions):
        block_id = f"spatial_block_{block_index:05d}_{stable_token({'condition': condition}, 10)}"
        local_rows: list[dict[str, Any]] = []
        for local_position, (r_prepare, r_query) in enumerate(rotated(RELATION_CELLS, block_index)):
            match = r_prepare == r_query
            local_rows.append(
                {
                    **schedule_row_base(condition, hashes),
                    "block_id": block_id,
                    "block_local_position": local_position,
                    "row_role": "relation_matrix",
                    "r_prepare": r_prepare,
                    "r_query": r_query,
                    "relation_match": match,
                    "query": "query_relation_pair",
                    "relation_cell": relation_cell_name(r_prepare, r_query),
                    "operation_semantics_id": "relation_matrix",
                    "control_semantics_id": "none",
                    "matched_twin_group": f"{block_id}:relation_matrix",
                    "matched_twin_pair": f"{block_id}:relation_pair_{local_position}",
                }
            )
        for control_position, control in enumerate(rotated(CONTROL_ROWS, block_index), start=4):
            local_rows.append(
                {
                    **schedule_row_base(condition, hashes),
                    "block_id": block_id,
                    "block_local_position": control_position,
                    "row_role": "relation_control",
                    "r_prepare": RELATIONS[(block_index + control_position) % 2],
                    "r_query": RELATIONS[(block_index + control_position + 1) % 2],
                    "relation_match": False,
                    "query": control,
                    "relation_cell": control,
                    "operation_semantics_id": control,
                    "control_semantics_id": control,
                    "matched_twin_group": f"{block_id}:control:{control}",
                    "matched_twin_pair": f"{block_id}:control:{control}",
                }
            )
        for row in sorted(local_rows, key=lambda item: item["block_local_position"]):
            row["execution_ordinal"] = ordinal
            row["tuple_id"] = f"{TRANSACTION_RUN_ID}:{ordinal:06d}:{stable_token({'ordinal': ordinal, 'block': block_id})}"
            rows.append(row)
            ordinal += 1
    schedule = {
        "schema": "FAMILY10H_RELATION_SPATIAL_PUBLIC_SCHEDULE_V1",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "public_randomization_seed": PUBLIC_RANDOMIZATION_SEED,
        "schedule_columns": SCHEDULE_COLUMNS,
        "tuple_count": len(rows),
        "base_condition_count": len(conditions),
        "rows_per_base_condition": 8,
        "pair_sample_count_per_row": PAIR_SAMPLE_COUNT,
        "pair_sample_stride": PAIR_SAMPLE_STRIDE,
        "expected_pair_observation_count": len(rows) * PAIR_SAMPLE_COUNT,
        "expected_tuple_count_derivation": {
            "sessions": len(SESSIONS),
            "replicates": len(REPLICATES),
            "mappings": len(MAPPINGS),
            "source_orders": len(SOURCE_ORDERS),
            "query_orders": len(QUERY_ORDERS),
            "cyclic_origins": len(CYCLIC_ORIGINS),
            "block_repeats": len(BLOCK_REPEATS),
            "rows_per_block": 8,
            "expected_tuple_count": 4096,
        },
        "physical_label_scramble_rows": 0,
        "rows": rows,
        "claim_boundary": {"post_observation_scheduling_allowed": False, "live_authority": False, "small_wall_crossed": False},
    }
    schedule["schedule_sha256"] = digest({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    return schedule


def validate_grammar(grammar: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if grammar.get("schema") != "FAMILY10H_RELATION_SPATIAL_GRAMMAR_V1":
        failures.append("schema mismatch")
    expected = digest({k: v for k, v in grammar.items() if k != "grammar_sha256"})
    if grammar.get("grammar_sha256") != expected:
        failures.append("grammar digest mismatch")
    rels = grammar.get("relation_definitions", {})
    if set(rels) != set(RELATIONS):
        failures.append("relation keyset mismatch")
    for relation_id, shift in [("relation_r0", 1), ("relation_r1", -1)]:
        expected_def = relation_definition(relation_id, shift)
        observed = rels.get(relation_id, {})
        if observed.get("permutation") != expected_def["permutation"]:
            failures.append(f"{relation_id} permutation table mismatch")
    return {"passed": not failures, "failures": failures, "proof": relation_marginal_equality_proof(grammar)}


def validate_schedule(schedule: dict[str, Any], grammar: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if schedule.get("schema") != "FAMILY10H_RELATION_SPATIAL_PUBLIC_SCHEDULE_V1":
        failures.append("schedule schema mismatch")
    expected = digest({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    if schedule.get("schedule_sha256") != expected:
        failures.append("schedule digest mismatch")
    rows = schedule.get("rows", [])
    if len(rows) != 4096 or schedule.get("tuple_count") != 4096:
        failures.append("tuple count is not the frozen 4096-row spatial schedule")
    if [row.get("execution_ordinal") for row in rows] != list(range(len(rows))):
        failures.append("execution ordinal sequence mismatch")
    proof = relation_marginal_equality_proof(grammar, schedule)
    if not proof["checks"]["exact_pair_coverage"]:
        failures.append("selected pair coverage imbalance")
    block_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        block_rows[row["block_id"]].append(row)
        if row.get("q") != 0:
            failures.append("nonzero q row present")
            break
        if row.get("source_lifetime") != "alive_during_query":
            failures.append("non-alive source_lifetime row present")
            break
        if row.get("post_observation_scheduling") is not False:
            failures.append("post-observation scheduling enabled")
            break
        if row.get("source_cpu_expected") != 4 or row.get("receiver_cpu_expected") != 5:
            failures.append("CPU custody drift")
            break
    if len(block_rows) != 512:
        failures.append("block count is not 512")
    for block_id, block in block_rows.items():
        if len(block) != 8:
            failures.append(f"{block_id} row count drift")
            continue
        relation_rows = [row for row in block if row["row_role"] == "relation_matrix"]
        control_rows = [row for row in block if row["row_role"] == "relation_control"]
        if sorted((row["r_prepare"], row["r_query"]) for row in relation_rows) != sorted(RELATION_CELLS):
            failures.append(f"{block_id} relation cell set mismatch")
        if sorted(row["query"] for row in control_rows) != sorted(CONTROL_ROWS):
            failures.append(f"{block_id} control set mismatch")
    factor_counts = {
        "session": Counter(row["session"] for row in rows if row["block_local_position"] == 0),
        "mapping": Counter(row["mapping"] for row in rows if row["block_local_position"] == 0),
        "source_order": Counter(row["source_order"] for row in rows if row["block_local_position"] == 0),
        "query_order": Counter(row["query_order"] for row in rows if row["block_local_position"] == 0),
        "cyclic_origin": Counter(row["cyclic_origin"] for row in rows if row["block_local_position"] == 0),
    }
    for name, counts in factor_counts.items():
        if counts and len(set(counts.values())) != 1:
            failures.append(f"{name} imbalance")
    return {
        "passed": not failures,
        "failures": failures,
        "tuple_count": len(rows),
        "block_count": len(block_rows),
        "factor_counts": {k: dict(v) for k, v in factor_counts.items()},
        "proof": proof,
    }


def write_schedule_tsv(schedule: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULE_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in schedule["rows"]:
            writer.writerow({key: row[key] for key in SCHEDULE_COLUMNS})


def validate_tsv(path: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    failures = []
    if len(rows) != schedule["tuple_count"]:
        failures.append("TSV row count mismatch")
    if rows and list(rows[0].keys()) != SCHEDULE_COLUMNS:
        failures.append("TSV column mismatch")
    return {"passed": not failures, "failures": failures, "row_count": len(rows)}


def write_source_bundle(path: Path, source_paths: list[Path]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tar_payload = BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w") as tf:
        for source in sorted(source_paths, key=lambda item: item.name):
            data = source.read_bytes()
            info = tarfile.TarInfo(name=source.name)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tf.addfile(info, BytesIO(data))
    with path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, compresslevel=9, mtime=0) as gz:
            gz.write(tar_payload.getvalue())
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "members": [source.name for source in sorted(source_paths, key=lambda item: item.name)],
    }
