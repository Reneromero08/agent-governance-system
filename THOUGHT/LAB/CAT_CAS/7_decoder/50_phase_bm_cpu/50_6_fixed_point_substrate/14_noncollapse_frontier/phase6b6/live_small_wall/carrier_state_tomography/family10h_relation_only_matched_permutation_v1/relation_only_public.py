#!/usr/bin/env python3
"""Public relation-only matched-permutation build-readiness helpers.

The package freezes a future Family 10h relation-only experiment, but this
module performs only offline construction and validation. It does not authorize
target contact, PMU acquisition, runtime execution, cleanup, or a Small Wall
promotion.
"""

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

SCIENCE_PACKAGE_ID = "family10h_relation_only_matched_permutation_v1"
TRANSACTION_RUN_ID = "family10h_relation_only_matched_permutation_v1_0"
PUBLIC_RANDOMIZATION_SEED = "family10h-relation-only-matched-permutation-v1-public-seed-1bf2c63e"
SOURCE_AUTHORITY_COMMIT = "cac5d33536768e00aa0de5f515e626fecccdeeda"
MANIFEST_FREEZE_COMMIT = "354ca8ab2d62458fca41481d74ff98c1b39ab6ed"
POSTRUN_SEAL_COMMIT = "c126700b8d46e6501ff39cfa360bf32a9fbdb2ac"
APPROVED_SENSOR_IDENTITY_SHA256 = "a993bb09ee6c38819f75f3be133cee859acc18ec8f62aa4ed8a57ee484fe1137"
OWNED_OUTPUT_PARENT_NAME = "_relation_only_owned_outputs"
ATTEMPT_CEILING = 1

PACKAGE_DECISION_BLOCKED = "FAMILY10H_RELATION_ONLY_BUILD_READINESS_BLOCKED"
PACKAGE_DECISION_BUILD_READY = "FAMILY10H_RELATION_ONLY_BUILD_READY_AWAITING_AUTHORIZATION"

FUTURE_RESULT_CLASSES = [
    "FAMILY10H_RELATION_MATCH_COORDINATE_CONFIRMED_PROSPECTIVE",
    "FAMILY10H_RELATION_MATCH_COORDINATE_NOT_CONFIRMED_PROSPECTIVE",
    "FAMILY10H_RELATION_MATCH_COORDINATE_CANDIDATE_PROSPECTIVE",
    "FAMILY10H_RELATION_MATCH_COORDINATE_CUSTODY_INVALID",
]
MAXIMUM_FUTURE_CLAIM = "PUBLIC_POST_SOURCE_RELATION_MATCH_COORDINATE_CONFIRMED"
NEGATIVE_FUTURE_CLAIM = "PUBLIC_POST_SOURCE_RELATION_MATCH_COORDINATE_NOT_ESTABLISHED"
FORBIDDEN_PROMOTIONS = [
    "FULL_CARRIER_STATE_TOMOGRAPHY_ESTABLISHED",
    "PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED",
    "CATALYTIC_BORROWING_ESTABLISHED",
    "R2_RESTORATION_ESTABLISHED",
    "SMALL_WALL_CROSSED",
]

LINE_COUNT = 4096
LINE_BYTES = 64
LANE_BYTES = LINE_COUNT * LINE_BYTES
PAGE_COUNT_PER_LANE = 64
LOGICAL_CACHE_SET_COUNT = 64
TOTAL_WORK = 4096
M = 2048
Q_VALUES = [-1536, -1024, -512, 0, 512, 1024, 1536]
SESSIONS = ["session_0", "session_1"]
REPLICATES = [0, 1]
MAPPINGS = ["map0", "map1"]
SOURCE_ORDERS = ["A_then_B", "B_then_A"]
QUERY_ORDERS = ["AB", "BA"]
CYCLIC_ORIGINS = [0, 1024, 2048, 3072]
DELAY_GRID = [
    {"delay_label": "0ns", "delay_ns": 0},
    {"delay_label": "1ms", "delay_ns": 1_000_000},
    {"delay_label": "10ms", "delay_ns": 10_000_000},
]
RELATIONS = ["relation_r0", "relation_r1"]
RELATION_CELLS = [
    ("relation_r0", "relation_r0"),
    ("relation_r0", "relation_r1"),
    ("relation_r1", "relation_r0"),
    ("relation_r1", "relation_r1"),
]
SCALAR_QUERIES = ["query_A", "query_B"]
CONTROL_ROWS = [
    "relation_sham",
    "route_pressure_sham",
    "independent_marginal_replay",
    "distance_control",
]

PMU_GROUP = {
    "name": "family10h_public_relation_match_group",
    "events": {
        "cpu_cycles_not_halted": {"event": "0x76", "umask": "0x00"},
        "cache_block_commands_change_to_dirty": {"event": "0xea", "umask": "0x20"},
        "probe_responses_dirty": {"event": "0xec", "umask": "0x0c"},
    },
}

RUNTIME_OPERATION_SEMANTICS = {
    "relation_matrix": (
        "fresh carrier; source walks A from cyclic_origin for 4096 steps, maps B through "
        "the selected public permutation table, then receiver walks the selected query "
        "permutation with the same origin"
    ),
    "scalar_control": (
        "fresh carrier; source uses the selected public relation table; receiver queries "
        "one scalar lane to preserve the confirmed D_single baseline"
    ),
    "relation_sham": (
        "fresh carrier; paired traversal uses a public block-local sham relation that "
        "matches scalar marginals and route pressure but is not stable across prepare/query"
    ),
    "route_pressure_sham": (
        "fresh carrier; interleaved A/B route pressure and instruction count are matched "
        "while pair identity is relation neutral"
    ),
    "independent_marginal_replay": (
        "fresh carrier; A and B marginal work is replayed independently, destroying the "
        "pair correspondence"
    ),
    "distance_control": (
        "fresh carrier; pair-distance histogram is preserved while the preparation/query "
        "correspondence law is intentionally mismatched"
    ),
}

PHYSICAL_GEOMETRY_STATUS = {
    "logical_line_index_histogram": "proven_offline_from_generated_tables",
    "virtual_offset_histogram": "proven_offline_from_fixed_64_byte_line_layout",
    "actual_physical_cache_index": "unresolved_until_authorized_target_preflight",
    "future_fail_closed_rule": (
        "if target-side owned-page physical geometry cannot be resolved when required, "
        "lower or invalidate the future relation claim"
    ),
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_compact_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def stable_token(value: Any, n: int = 20) -> str:
    return hashlib.sha256((PUBLIC_RANDOMIZATION_SEED + ":" + digest(value)).encode("utf-8")).hexdigest()[:n]


def relation_permutation(shift: int) -> list[int]:
    return [(idx + shift) % LINE_COUNT for idx in range(LINE_COUNT)]


def a_sequence(origin: int) -> list[int]:
    return [(origin + step) % LINE_COUNT for step in range(LINE_COUNT)]


def b_sequence_for_relation(shift: int, origin: int) -> list[int]:
    perm = relation_permutation(shift)
    return [perm[index] for index in a_sequence(origin)]


def cycle_structure(perm: list[int]) -> list[int]:
    seen = [False] * len(perm)
    cycles: list[int] = []
    for start in range(len(perm)):
        if seen[start]:
            continue
        current = start
        length = 0
        while not seen[current]:
            seen[current] = True
            length += 1
            current = perm[current]
        cycles.append(length)
    return sorted(cycles)


def histogram(values: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def circular_distance(a: int, b: int) -> int:
    delta = abs(a - b) % LINE_COUNT
    return min(delta, LINE_COUNT - delta)


def logical_line_hist_for_perm(perm: list[int]) -> dict[str, int]:
    values = []
    for idx, mapped in enumerate(perm):
        values.append(idx)
        values.append(mapped)
    return histogram(values)


def virtual_offset_hist_for_perm(perm: list[int]) -> dict[str, int]:
    values = []
    for idx, mapped in enumerate(perm):
        values.append((idx * LINE_BYTES) % 4096)
        values.append((mapped * LINE_BYTES) % 4096)
    return histogram(values)


def logical_cache_set_hist_for_perm(perm: list[int]) -> dict[str, int]:
    values = []
    for idx, mapped in enumerate(perm):
        values.append(idx % LOGICAL_CACHE_SET_COUNT)
        values.append(mapped % LOGICAL_CACHE_SET_COUNT)
    return histogram(values)


def pair_distance_hist_for_perm(perm: list[int]) -> dict[str, int]:
    return histogram([circular_distance(idx, mapped) for idx, mapped in enumerate(perm)])


def relation_definition(relation_id: str, shift: int) -> dict[str, Any]:
    perm = relation_permutation(shift)
    line_hist = logical_line_hist_for_perm(perm)
    offset_hist = virtual_offset_hist_for_perm(perm)
    distance_hist = pair_distance_hist_for_perm(perm)
    cycle = cycle_structure(perm)
    return {
        "relation_id": relation_id,
        "formula": f"B_index = (A_index {'+' if shift > 0 else '-'} 1) mod 4096",
        "shift": shift,
        "permutation": perm,
        "permutation_sha256": digest(perm),
        "cycle_structure": cycle,
        "cycle_structure_sha256": digest(cycle),
        "pair_distance_histogram": distance_hist,
        "pair_distance_histogram_sha256": digest(distance_hist),
        "logical_line_histogram": line_hist,
        "logical_line_histogram_sha256": digest(line_hist),
        "virtual_offset_histogram": offset_hist,
        "virtual_offset_histogram_sha256": digest(offset_hist),
        "logical_cache_set_histogram": logical_cache_set_hist_for_perm(perm),
        "logical_cache_set_histogram_sha256": digest(logical_cache_set_hist_for_perm(perm)),
        "actual_cache_index_status": PHYSICAL_GEOMETRY_STATUS["actual_physical_cache_index"],
        "sample_pairs": [[idx, perm[idx]] for idx in range(16)],
    }


def relation_grammar(package_decision: str = PACKAGE_DECISION_BLOCKED) -> dict[str, Any]:
    relations = {
        "relation_r0": relation_definition("relation_r0", 1),
        "relation_r1": relation_definition("relation_r1", -1),
    }
    grammar = {
        "schema": "FAMILY10H_RELATION_ONLY_GRAMMAR_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "public_randomization_seed": PUBLIC_RANDOMIZATION_SEED,
        "relation_definitions": relations,
        "relation_ids": RELATIONS,
        "cyclic_origins": CYCLIC_ORIGINS,
        "physical_marginals": {
            "A_address_set": "logical A line indices 0..4095",
            "B_address_set": "logical B line indices 0..4095",
            "page_count_per_lane": PAGE_COUNT_PER_LANE,
            "line_count_per_lane": LINE_COUNT,
            "line_bytes": LINE_BYTES,
            "logical_cache_set_count": LOGICAL_CACHE_SET_COUNT,
            "source_cpu_expected": 4,
            "receiver_cpu_expected": 5,
            "pmu_group": PMU_GROUP,
            "source_loop_count": TOTAL_WORK,
            "receiver_loop_count": TOTAL_WORK,
            "branch_structure": "same compiled hot loops; relation is selected by public table pointer before the loop",
            "allocation_and_prefault": "allocate all owned lanes before schedule, prefault every lane in identical order",
        },
        "operation_semantics": RUNTIME_OPERATION_SEMANTICS,
        "physical_geometry_status": PHYSICAL_GEOMETRY_STATUS,
        "primary_relation_law": {
            "name": "R_match",
            "formula": "0.5 * ((Y_r0_r0 + Y_r1_r1) - (Y_r0_r1 + Y_r1_r0))",
            "positive_or_negative_allowed": True,
            "primary_endpoint": "dirty_probe_response",
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
    r0 = rels["relation_r0"]
    r1 = rels["relation_r1"]
    schedule_rows = schedule.get("rows", []) if schedule else []
    origin_counts = Counter(row["cyclic_origin"] for row in schedule_rows) if schedule_rows else Counter()
    relation_origin_counts = Counter(
        (row["r_prepare"], row["r_query"], row["cyclic_origin"])
        for row in schedule_rows
        if row.get("row_role") == "relation_matrix"
    )
    expected_relation_origin_count = None
    if relation_origin_counts:
        expected_relation_origin_count = min(relation_origin_counts.values()) == max(relation_origin_counts.values())
    runtime_gates = (runtime_receipt or {}).get("implementation_gates", {})
    binary_authority = (runtime_receipt or {}).get("runtime_binary_authority", {})
    full_sequences = {
        relation_id: {
            str(origin): {
                "A_sequence_sha256": digest(a_sequence(origin)),
                "B_sequence_sha256": digest(b_sequence_for_relation(int(defn["shift"]), origin)),
                "read_sequence_length": LINE_COUNT,
                "write_sequence_length": LINE_COUNT,
            }
            for origin in CYCLIC_ORIGINS
        }
        for relation_id, defn in rels.items()
    }
    checks = {
        "relations_distinct": r0["permutation_sha256"] != r1["permutation_sha256"],
        "same_A_address_set": sorted(a_sequence(0)) == list(range(LINE_COUNT)),
        "same_B_address_set": sorted(r0["permutation"]) == sorted(r1["permutation"]) == list(range(LINE_COUNT)),
        "same_A_work_count": TOTAL_WORK == TOTAL_WORK,
        "same_B_work_count": TOTAL_WORK == TOTAL_WORK,
        "same_total_work": (M + max(Q_VALUES)) + (M - max(Q_VALUES)) == TOTAL_WORK,
        "same_read_count": LINE_COUNT == LINE_COUNT,
        "same_write_count": LINE_COUNT == LINE_COUNT,
        "same_page_count": PAGE_COUNT_PER_LANE == PAGE_COUNT_PER_LANE,
        "same_line_count": LINE_COUNT == LINE_COUNT,
        "same_logical_line_histogram": r0["logical_line_histogram_sha256"] == r1["logical_line_histogram_sha256"],
        "same_virtual_offset_histogram": r0["virtual_offset_histogram_sha256"] == r1["virtual_offset_histogram_sha256"],
        "actual_cache_index_unresolved_not_claimed": r0["actual_cache_index_status"].startswith("unresolved")
        and r1["actual_cache_index_status"].startswith("unresolved"),
        "same_pair_distance_histogram": r0["pair_distance_histogram_sha256"] == r1["pair_distance_histogram_sha256"],
        "same_permutation_cycle_structure_class": r0["cycle_structure_sha256"] == r1["cycle_structure_sha256"],
        "same_source_loop_length": TOTAL_WORK == LINE_COUNT,
        "same_receiver_loop_length": TOTAL_WORK == LINE_COUNT,
        "same_branch_structure": runtime_gates.get("hot_loop_relation_branch_free", False),
        "same_relation_table_lookup_count": LINE_COUNT == LINE_COUNT,
        "same_allocation_order": runtime_gates.get("fresh_carrier_per_tuple_implemented", False),
        "same_prefault_order": runtime_gates.get("prefault_implemented", False),
        "same_source_and_receiver_cpu": runtime_gates.get("source_cpu_pinning_implemented", False)
        and runtime_gates.get("receiver_cpu_pinning_implemented", False),
        "same_pmu_event_group": runtime_gates.get("pmu_group_open_implemented", False)
        and runtime_gates.get("pmu_group_read_implemented", False),
        "same_delay_distribution": runtime_gates.get("delay_enforcement_implemented", False),
        "same_source_order_and_query_order_counts": runtime_gates.get("source_order_control_implemented", False)
        and runtime_gates.get("query_order_control_implemented", False),
        "cyclic_origin_balance": bool(origin_counts) and sorted(origin_counts) == CYCLIC_ORIGINS and len(set(origin_counts.values())) == 1
        if schedule_rows
        else True,
        "relation_cells_balanced_across_origins": expected_relation_origin_count if expected_relation_origin_count is not None else True,
        "runtime_self_test_receipt_bound": runtime_receipt is not None,
        "runtime_schedule_executor_bound": runtime_gates.get("runtime_schedule_executor_implemented", False),
        "synthetic_executor_bound": runtime_gates.get("synthetic_executor_complete_schedule_passed", False),
        "compiled_binary_hash_bound": binary_authority.get("compiled_binary_sha256") is not None,
        "source_hashes_bound": source_hashes is not None,
    }
    proof = {
        "schema": "FAMILY10H_RELATION_MARGINAL_EQUALITY_PROOF_V2",
        "grammar_sha256": grammar["grammar_sha256"],
        "proof_basis": "generated full permutation tables, generated origin-rotated A/B sequences, schedule rows, source hashes, and runtime build receipt",
        "relation_r0": {
            "formula": r0["formula"],
            "permutation_sha256": r0["permutation_sha256"],
            "cycle_structure_sha256": r0["cycle_structure_sha256"],
            "pair_distance_histogram_sha256": r0["pair_distance_histogram_sha256"],
            "logical_line_histogram_sha256": r0["logical_line_histogram_sha256"],
            "virtual_offset_histogram_sha256": r0["virtual_offset_histogram_sha256"],
            "actual_cache_index_status": r0["actual_cache_index_status"],
        },
        "relation_r1": {
            "formula": r1["formula"],
            "permutation_sha256": r1["permutation_sha256"],
            "cycle_structure_sha256": r1["cycle_structure_sha256"],
            "pair_distance_histogram_sha256": r1["pair_distance_histogram_sha256"],
            "logical_line_histogram_sha256": r1["logical_line_histogram_sha256"],
            "virtual_offset_histogram_sha256": r1["virtual_offset_histogram_sha256"],
            "actual_cache_index_status": r1["actual_cache_index_status"],
        },
        "full_sequence_digests": full_sequences,
        "origin_counts": dict(sorted((str(k), v) for k, v in origin_counts.items())),
        "relation_origin_counts_sha256": digest(dict(sorted((str(k), v) for k, v in relation_origin_counts.items()))),
        "source_hashes_sha256": digest(source_hashes) if source_hashes is not None else None,
        "runtime_build_receipt_sha256": digest(runtime_receipt) if runtime_receipt is not None else None,
        "runtime_build_receipt_passed": runtime_receipt.get("passed") if runtime_receipt else False,
        "checks": checks,
        "passed": all(checks.values()),
    }
    proof["proof_sha256"] = digest({k: v for k, v in proof.items() if k != "proof_sha256"})
    return proof


def rotated(items: list[Any], amount: int) -> list[Any]:
    amount = amount % len(items)
    return items[amount:] + items[:amount]


def base_conditions() -> list[dict[str, Any]]:
    conditions = []
    for session in SESSIONS:
        for replicate in REPLICATES:
            for mapping in MAPPINGS:
                for delay in DELAY_GRID:
                    for q in Q_VALUES:
                        for source_order in SOURCE_ORDERS:
                            for query_order in QUERY_ORDERS:
                                for origin in CYCLIC_ORIGINS:
                                    conditions.append(
                                        {
                                            "session": session,
                                            "replicate": replicate,
                                            "mapping": mapping,
                                            "delay_label": delay["delay_label"],
                                            "delay_ns": delay["delay_ns"],
                                            "q": q,
                                            "source_order": source_order,
                                            "query_order": query_order,
                                            "cyclic_origin": origin,
                                        }
                                    )
    return sorted(conditions, key=lambda item: stable_token(item))


def relation_cell_name(r_prepare: str, r_query: str) -> str:
    return f"prepare_{r_prepare[-2:]}__query_{r_query[-2:]}"


def schedule_row_base(condition: dict[str, Any], common_hashes: dict[str, str]) -> dict[str, Any]:
    q = int(condition["q"])
    return {
        "q": q,
        "bank_A_work": M + q,
        "bank_B_work": M - q,
        "total_work": TOTAL_WORK,
        "session": condition["session"],
        "replicate": condition["replicate"],
        "mapping": condition["mapping"],
        "delay_label": condition["delay_label"],
        "delay_ns": condition["delay_ns"],
        "source_order": condition["source_order"],
        "query_order": condition["query_order"],
        "cyclic_origin": condition["cyclic_origin"],
        "route_pressure_class": "matched_relation_route_pressure_v2",
        "distance_control_class": "distance_histogram_preserved_one_step_circular",
        "allocation_order_class": "relation_label_independent_prefault_v2",
        "prefault_class": "all_lanes_prefaulted_before_source_fork",
        "source_cpu_expected": 4,
        "receiver_cpu_expected": 5,
        "source_loop_count": TOTAL_WORK,
        "receiver_loop_count": TOTAL_WORK,
        "read_count": TOTAL_WORK,
        "write_count": TOTAL_WORK,
        "page_count_A": PAGE_COUNT_PER_LANE,
        "page_count_B": PAGE_COUNT_PER_LANE,
        "line_count_A": LINE_COUNT,
        "line_count_B": LINE_COUNT,
        "actual_cache_index_status": PHYSICAL_GEOMETRY_STATUS["actual_physical_cache_index"],
        "expected_pmu_group": PMU_GROUP["name"],
        "requires_pmu": True,
        "post_observation_scheduling": False,
        **common_hashes,
    }


def build_schedule(grammar: dict[str, Any]) -> dict[str, Any]:
    proof_stub = relation_marginal_equality_proof(grammar)
    common_relation_hashes = {
        "logical_line_histogram_sha256": proof_stub["relation_r0"]["logical_line_histogram_sha256"],
        "virtual_offset_histogram_sha256": proof_stub["relation_r0"]["virtual_offset_histogram_sha256"],
        "pair_distance_histogram_sha256": proof_stub["relation_r0"]["pair_distance_histogram_sha256"],
        "permutation_cycle_structure_sha256": proof_stub["relation_r0"]["cycle_structure_sha256"],
    }
    rows: list[dict[str, Any]] = []
    ordinal = 0
    conditions = base_conditions()
    for block_index, condition in enumerate(conditions):
        block_id = f"relation_block_{block_index:05d}_{stable_token({'condition': condition}, 10)}"
        relation_order = rotated(RELATION_CELLS, block_index)
        scalar_order = rotated(SCALAR_QUERIES, block_index)
        control_order = rotated(CONTROL_ROWS, block_index)
        local_rows: list[dict[str, Any]] = []
        for local_position, (r_prepare, r_query) in enumerate(relation_order):
            relation_match = r_prepare == r_query
            local_rows.append(
                {
                    **schedule_row_base(condition, common_relation_hashes),
                    "block_id": block_id,
                    "block_local_position": local_position,
                    "row_role": "relation_matrix",
                    "r_prepare": r_prepare,
                    "r_query": r_query,
                    "relation_match": relation_match,
                    "query": "query_relation_pair",
                    "relation_cell": relation_cell_name(r_prepare, r_query),
                    "operation_semantics_id": "relation_matrix",
                    "control_semantics_id": "none",
                    "matched_twin_group": f"{block_id}:relation_matrix",
                    "matched_twin_pair": f"{block_id}:pair_{local_position // 2}",
                }
            )
        for scalar_index, query in enumerate(scalar_order):
            for relation_index, relation in enumerate(RELATIONS):
                local_rows.append(
                    {
                        **schedule_row_base(condition, common_relation_hashes),
                        "block_id": block_id,
                        "block_local_position": 4 + scalar_index * 2 + relation_index,
                        "row_role": "scalar_control",
                        "r_prepare": relation,
                        "r_query": relation,
                        "relation_match": True,
                        "query": query,
                        "relation_cell": f"{query}_{relation}",
                        "operation_semantics_id": "scalar_control",
                        "control_semantics_id": "scalar_q_baseline",
                        "matched_twin_group": f"{block_id}:scalar:{query}",
                        "matched_twin_pair": f"{block_id}:scalar:{query}",
                    }
                )
        for control_index, control in enumerate(control_order):
            local_rows.append(
                {
                    **schedule_row_base(condition, common_relation_hashes),
                    "block_id": block_id,
                    "block_local_position": 8 + control_index,
                    "row_role": "relation_control",
                    "r_prepare": "control",
                    "r_query": "control",
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
        "schema": "FAMILY10H_RELATION_ONLY_PUBLIC_SCHEDULE_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "public_randomization_seed": PUBLIC_RANDOMIZATION_SEED,
        "schedule_columns": SCHEDULE_COLUMNS,
        "tuple_count": len(rows),
        "base_condition_count": len(conditions),
        "rows_per_base_condition": len(rows) // len(conditions),
        "cyclic_origins": CYCLIC_ORIGINS,
        "physical_label_scramble_rows": 0,
        "label_scramble_policy": "offline_adjudication_regression_only",
        "rows": rows,
        "claim_boundary": {
            "post_observation_scheduling_allowed": False,
            "live_authority": False,
            "small_wall_crossed": False,
        },
    }
    schedule["schedule_sha256"] = digest({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    return schedule


def validate_grammar(grammar: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if grammar.get("schema") != "FAMILY10H_RELATION_ONLY_GRAMMAR_V2":
        failures.append("schema mismatch")
    expected = digest({k: v for k, v in grammar.items() if k != "grammar_sha256"})
    if grammar.get("grammar_sha256") != expected:
        failures.append("grammar digest mismatch")
    rels = grammar.get("relation_definitions", {})
    if set(rels) != set(RELATIONS):
        failures.append("relation keyset mismatch")
    for relation_id, shift in [("relation_r0", 1), ("relation_r1", -1)]:
        observed = rels.get(relation_id, {})
        expected_def = relation_definition(relation_id, shift)
        if observed.get("permutation") != expected_def["permutation"]:
            failures.append(f"{relation_id} permutation table mismatch")
        for key in [
            "permutation_sha256",
            "cycle_structure_sha256",
            "pair_distance_histogram_sha256",
            "logical_line_histogram_sha256",
            "virtual_offset_histogram_sha256",
        ]:
            if observed.get(key) != expected_def[key]:
                failures.append(f"{relation_id} {key} mismatch")
                break
    proof = relation_marginal_equality_proof(grammar)
    if not proof["checks"]["actual_cache_index_unresolved_not_claimed"]:
        failures.append("actual physical cache-index equality is overclaimed")
    return {"passed": not failures, "failures": failures, "proof": proof}


def validate_schedule(schedule: dict[str, Any], grammar: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if schedule.get("schema") != "FAMILY10H_RELATION_ONLY_PUBLIC_SCHEDULE_V2":
        failures.append("schedule schema mismatch")
    expected = digest({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    if schedule.get("schedule_sha256") != expected:
        failures.append("schedule digest mismatch")
    rows = schedule.get("rows", [])
    if schedule.get("tuple_count") != len(rows):
        failures.append("tuple count mismatch")
    if [row.get("execution_ordinal") for row in rows] != list(range(len(rows))):
        failures.append("execution ordinal sequence mismatch")
    forbidden_tokens = ["relation_r0", "relation_r1", "prepare_r0", "prepare_r1", "cyclic_origin", "origin_"]
    if any(token in row["tuple_id"] for row in rows for token in forbidden_tokens):
        failures.append("relation or origin leakage through tuple IDs")
    proof = relation_marginal_equality_proof(grammar, schedule)
    for row in rows:
        if row.get("query") == "label_scramble_control":
            failures.append("physical label-scramble row present")
            break
        if row.get("post_observation_scheduling") is not False:
            failures.append("post-observation scheduling enabled")
            break
        if row.get("total_work") != TOTAL_WORK:
            failures.append("total work drift")
            break
        if row.get("bank_A_work") + row.get("bank_B_work") != TOTAL_WORK:
            failures.append("A/B work drift")
            break
        if row.get("read_count") != TOTAL_WORK or row.get("write_count") != TOTAL_WORK:
            failures.append("read/write count drift")
            break
        if row.get("source_loop_count") != TOTAL_WORK or row.get("receiver_loop_count") != TOTAL_WORK:
            failures.append("loop count drift")
            break
        if row.get("allocation_order_class") != "relation_label_independent_prefault_v2":
            failures.append("allocation-order class drift")
            break
        if row.get("prefault_class") != "all_lanes_prefaulted_before_source_fork":
            failures.append("prefault class drift")
            break
        allowed_semantics = set(RUNTIME_OPERATION_SEMANTICS) | {"scalar_control"}
        if row.get("operation_semantics_id") not in allowed_semantics:
            failures.append("operation semantics drift")
            break
        if row.get("cyclic_origin") not in CYCLIC_ORIGINS:
            failures.append("invalid cyclic origin")
            break
        if row.get("page_count_A") != PAGE_COUNT_PER_LANE or row.get("page_count_B") != PAGE_COUNT_PER_LANE:
            failures.append("page count drift")
            break
        if row.get("line_count_A") != LINE_COUNT or row.get("line_count_B") != LINE_COUNT:
            failures.append("line count drift")
            break
        if row.get("actual_cache_index_status") != PHYSICAL_GEOMETRY_STATUS["actual_physical_cache_index"]:
            failures.append("actual cache-index status drift")
            break
        if row.get("logical_line_histogram_sha256") != proof["relation_r0"]["logical_line_histogram_sha256"]:
            failures.append("logical-line histogram drift")
            break
        if row.get("virtual_offset_histogram_sha256") != proof["relation_r0"]["virtual_offset_histogram_sha256"]:
            failures.append("virtual-offset histogram drift")
            break
        if row.get("pair_distance_histogram_sha256") != proof["relation_r0"]["pair_distance_histogram_sha256"]:
            failures.append("pair-distance histogram drift")
            break
    block_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        block_rows[row["block_id"]].append(row)
    position_counts: dict[str, dict[int, int]] = {relation_cell_name(*cell): {} for cell in RELATION_CELLS}
    origin_counts: Counter[int] = Counter()
    relation_origin_counts: Counter[tuple[str, str, int]] = Counter()
    for block_id, block in block_rows.items():
        relation_rows = [row for row in block if row["row_role"] == "relation_matrix"]
        if len(relation_rows) != 4:
            failures.append(f"{block_id} incomplete relation matrix")
            continue
        cells = [(row["r_prepare"], row["r_query"]) for row in relation_rows]
        if sorted(cells) != sorted(RELATION_CELLS):
            failures.append(f"{block_id} relation cell set mismatch")
        if len(set(cells)) != 4:
            failures.append(f"{block_id} duplicated relation cell")
        ordered_positions = sorted(row["block_local_position"] for row in relation_rows)
        if ordered_positions != [0, 1, 2, 3]:
            failures.append(f"{block_id} relation rows not adjacent")
        for row in relation_rows:
            cell_name = relation_cell_name(row["r_prepare"], row["r_query"])
            position_counts[cell_name][row["block_local_position"]] = position_counts[cell_name].get(row["block_local_position"], 0) + 1
            relation_origin_counts[(row["r_prepare"], row["r_query"], row["cyclic_origin"])] += 1
        control_rows = [row for row in block if row["row_role"] == "relation_control"]
        if sorted(row["query"] for row in control_rows) != sorted(CONTROL_ROWS):
            failures.append(f"{block_id} relation controls incomplete")
        scalar_rows = [row for row in block if row["row_role"] == "scalar_control"]
        if len(scalar_rows) != 4:
            failures.append(f"{block_id} scalar controls incomplete")
        origin_counts[block[0]["cyclic_origin"]] += 1
    for cell_name, counts in position_counts.items():
        if set(counts) != {0, 1, 2, 3}:
            failures.append(f"{cell_name} not counterbalanced over relation positions")
        elif max(counts.values()) - min(counts.values()) > 1:
            failures.append(f"{cell_name} execution-order imbalance")
    if sorted(origin_counts) != CYCLIC_ORIGINS or len(set(origin_counts.values())) != 1:
        failures.append("cyclic origin imbalance")
    if relation_origin_counts and len(set(relation_origin_counts.values())) != 1:
        failures.append("relation cell origin imbalance")
    return {
        "passed": not failures,
        "failures": failures,
        "tuple_count": len(rows),
        "block_count": len(block_rows),
        "origin_counts": dict(sorted((str(k), v) for k, v in origin_counts.items())),
        "relation_position_counts": position_counts,
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
    if rows and set(rows[0].keys()) != set(SCHEDULE_COLUMNS):
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
