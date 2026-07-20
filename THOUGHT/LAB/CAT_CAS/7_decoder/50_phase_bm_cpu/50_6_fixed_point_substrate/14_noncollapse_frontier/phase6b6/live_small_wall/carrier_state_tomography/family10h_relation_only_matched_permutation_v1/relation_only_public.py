#!/usr/bin/env python3
"""Offline public relation-only matched-permutation grammar.

This package freezes a future Family 10h relation-only experiment design. It
does not authorize target contact, PMU acquisition, runtime execution, or any
Small Wall promotion.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent

SCIENCE_PACKAGE_ID = "family10h_relation_only_matched_permutation_v1"
TRANSACTION_RUN_ID = "family10h_relation_only_matched_permutation_v1_0"
PUBLIC_RANDOMIZATION_SEED = "family10h-relation-only-matched-permutation-v1-public-seed-1bf2c63e"
PACKAGE_DECISION_FROZEN = "FAMILY10H_RELATION_ONLY_PACKAGE_FROZEN_AWAITING_AUTHORIZATION"

FUTURE_RESULT_CLASSES = [
    "FAMILY10H_RELATION_MATCH_COORDINATE_CONFIRMED_PROSPECTIVE",
    "FAMILY10H_RELATION_MATCH_COORDINATE_NOT_CONFIRMED_PROSPECTIVE",
    "FAMILY10H_RELATION_MATCH_COORDINATE_CANDIDATE_PROSPECTIVE",
    "FAMILY10H_RELATION_MATCH_COORDINATE_CUSTODY_INVALID",
]
MAXIMUM_FUTURE_CLAIM = "PUBLIC_POST_SOURCE_RELATION_MATCH_COORDINATE_CONFIRMED"
FORBIDDEN_PROMOTIONS = [
    "FULL_CARRIER_STATE_TOMOGRAPHY_ESTABLISHED",
    "PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED",
    "CATALYTIC_BORROWING_ESTABLISHED",
    "R2_RESTORATION_ESTABLISHED",
    "SMALL_WALL_CROSSED",
]

LINE_COUNT = 4096
PAGE_COUNT_PER_LANE = 64
CACHE_SET_COUNT = 64
TOTAL_WORK = 4096
M = 2048
Q_VALUES = [-1536, -1024, -512, 0, 512, 1024, 1536]
SESSIONS = ["session_0", "session_1"]
REPLICATES = [0, 1]
MAPPINGS = ["map0", "map1"]
SOURCE_ORDERS = ["A_then_B", "B_then_A"]
QUERY_ORDERS = ["AB", "BA"]
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
    "label_scramble_control",
]

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
    "route_pressure_class",
    "distance_control_class",
    "allocation_order_class",
    "prefault_class",
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
    "cache_set_histogram_sha256",
    "pair_distance_histogram_sha256",
    "permutation_cycle_structure_sha256",
    "matched_twin_group",
    "matched_twin_pair",
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


def stable_token(value: Any, n: int = 20) -> str:
    return hashlib.sha256((PUBLIC_RANDOMIZATION_SEED + ":" + digest(value)).encode("utf-8")).hexdigest()[:n]


def relation_permutation(shift: int) -> list[int]:
    return [(idx + shift) % LINE_COUNT for idx in range(LINE_COUNT)]


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


def cache_set_hist_for_perm(perm: list[int]) -> dict[str, int]:
    values = []
    for idx, mapped in enumerate(perm):
        values.append(idx % CACHE_SET_COUNT)
        values.append(mapped % CACHE_SET_COUNT)
    return histogram(values)


def pair_distance_hist_for_perm(perm: list[int]) -> dict[str, int]:
    return histogram([circular_distance(idx, mapped) for idx, mapped in enumerate(perm)])


def relation_grammar() -> dict[str, Any]:
    r0_perm = relation_permutation(1)
    r1_perm = relation_permutation(-1)
    relations = {
        "relation_r0": {
            "relation_id": "relation_r0",
            "formula": "B_index = (A_index + 1) mod 4096",
            "shift": 1,
            "permutation_sha256": digest(r0_perm),
            "cycle_structure": cycle_structure(r0_perm),
            "cycle_structure_sha256": digest(cycle_structure(r0_perm)),
            "pair_distance_histogram": pair_distance_hist_for_perm(r0_perm),
            "pair_distance_histogram_sha256": digest(pair_distance_hist_for_perm(r0_perm)),
            "cache_set_histogram": cache_set_hist_for_perm(r0_perm),
            "cache_set_histogram_sha256": digest(cache_set_hist_for_perm(r0_perm)),
            "sample_pairs": [[idx, r0_perm[idx]] for idx in range(16)],
        },
        "relation_r1": {
            "relation_id": "relation_r1",
            "formula": "B_index = (A_index - 1) mod 4096",
            "shift": -1,
            "permutation_sha256": digest(r1_perm),
            "cycle_structure": cycle_structure(r1_perm),
            "cycle_structure_sha256": digest(cycle_structure(r1_perm)),
            "pair_distance_histogram": pair_distance_hist_for_perm(r1_perm),
            "pair_distance_histogram_sha256": digest(pair_distance_hist_for_perm(r1_perm)),
            "cache_set_histogram": cache_set_hist_for_perm(r1_perm),
            "cache_set_histogram_sha256": digest(cache_set_hist_for_perm(r1_perm)),
            "sample_pairs": [[idx, r1_perm[idx]] for idx in range(16)],
        },
    }
    grammar = {
        "schema": "FAMILY10H_RELATION_ONLY_GRAMMAR_V1",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "public_randomization_seed": PUBLIC_RANDOMIZATION_SEED,
        "relation_definitions": relations,
        "relation_ids": RELATIONS,
        "physical_marginals": {
            "A_address_set": "logical A line indices 0..4095",
            "B_address_set": "logical B line indices 0..4095",
            "page_count_per_lane": PAGE_COUNT_PER_LANE,
            "line_count_per_lane": LINE_COUNT,
            "cache_set_count": CACHE_SET_COUNT,
            "source_cpu_expected": 4,
            "receiver_cpu_expected": 5,
            "pmu_group": "family10h_public_relation_match_group",
            "source_loop_count": TOTAL_WORK,
            "receiver_loop_count": TOTAL_WORK,
            "branch_structure": "same static loops and branch topology for relation_r0 and relation_r1",
            "allocation_and_prefault": "allocate all owned lanes before schedule, prefault every lane in identical order",
        },
        "primary_relation_law": {
            "name": "R_match",
            "formula": "0.5 * ((Y_r0_r0 + Y_r1_r1) - (Y_r0_r1 + Y_r1_r0))",
            "positive_or_negative_allowed": True,
            "primary_endpoint": "dirty_probe_response",
        },
        "claim_boundary": {
            "offline_package_decision": PACKAGE_DECISION_FROZEN,
            "maximum_future_claim": MAXIMUM_FUTURE_CLAIM,
            "forbidden_promotions": FORBIDDEN_PROMOTIONS,
            "live_authority": False,
            "small_wall_crossed": False,
        },
    }
    grammar["grammar_sha256"] = digest({k: v for k, v in grammar.items() if k != "grammar_sha256"})
    return grammar


def relation_marginal_equality_proof(grammar: dict[str, Any]) -> dict[str, Any]:
    rels = grammar["relation_definitions"]
    r0 = rels["relation_r0"]
    r1 = rels["relation_r1"]
    checks = {
        "relations_distinct": r0["permutation_sha256"] != r1["permutation_sha256"],
        "same_A_address_set": True,
        "same_B_address_set": True,
        "same_A_work_count": True,
        "same_B_work_count": True,
        "same_total_work": True,
        "same_read_count": True,
        "same_write_count": True,
        "same_page_count": True,
        "same_line_count": True,
        "same_cache_set_histogram": r0["cache_set_histogram_sha256"] == r1["cache_set_histogram_sha256"],
        "same_pair_distance_histogram": r0["pair_distance_histogram_sha256"] == r1["pair_distance_histogram_sha256"],
        "same_permutation_cycle_structure_class": r0["cycle_structure_sha256"] == r1["cycle_structure_sha256"],
        "same_source_loop_length": True,
        "same_receiver_loop_length": True,
        "same_branch_structure": True,
        "same_source_and_receiver_cpu": True,
        "same_pmu_event_group": True,
        "same_delay_distribution": True,
        "same_source_order_and_query_order_counts": True,
        "same_allocation_and_prefault_behavior": True,
    }
    proof = {
        "schema": "FAMILY10H_RELATION_MARGINAL_EQUALITY_PROOF_V1",
        "grammar_sha256": grammar["grammar_sha256"],
        "proof_basis": "relation_r0 and relation_r1 are inverse one-step modular permutations over the same 4096-line A/B sets",
        "relation_r0": {
            "formula": r0["formula"],
            "permutation_sha256": r0["permutation_sha256"],
            "cycle_structure_sha256": r0["cycle_structure_sha256"],
            "pair_distance_histogram_sha256": r0["pair_distance_histogram_sha256"],
            "cache_set_histogram_sha256": r0["cache_set_histogram_sha256"],
        },
        "relation_r1": {
            "formula": r1["formula"],
            "permutation_sha256": r1["permutation_sha256"],
            "cycle_structure_sha256": r1["cycle_structure_sha256"],
            "pair_distance_histogram_sha256": r1["pair_distance_histogram_sha256"],
            "cache_set_histogram_sha256": r1["cache_set_histogram_sha256"],
        },
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
                                    }
                                )
    return sorted(conditions, key=lambda item: stable_token(item))


def relation_cell_name(r_prepare: str, r_query: str) -> str:
    return f"prepare_{r_prepare[-2:]}__query_{r_query[-2:]}"


def schedule_row_base(condition: dict[str, Any]) -> dict[str, Any]:
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
        "route_pressure_class": "matched_relation_route_pressure_v1",
        "distance_control_class": "distance_histogram_preserved_one_step_circular",
        "allocation_order_class": "relation_label_independent_prefault_v1",
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
        "post_observation_scheduling": False,
    }


def build_schedule(grammar: dict[str, Any]) -> dict[str, Any]:
    proof = relation_marginal_equality_proof(grammar)
    common_relation_hashes = {
        "cache_set_histogram_sha256": proof["relation_r0"]["cache_set_histogram_sha256"],
        "pair_distance_histogram_sha256": proof["relation_r0"]["pair_distance_histogram_sha256"],
        "permutation_cycle_structure_sha256": proof["relation_r0"]["cycle_structure_sha256"],
    }
    rows: list[dict[str, Any]] = []
    ordinal = 0
    conditions = base_conditions()
    for block_index, condition in enumerate(conditions):
        block_id = f"relation_block_{block_index:05d}_{stable_token(condition, 10)}"
        relation_order = rotated(RELATION_CELLS, block_index)
        scalar_order = rotated(SCALAR_QUERIES, block_index)
        control_order = rotated(CONTROL_ROWS, block_index)
        local_rows: list[dict[str, Any]] = []
        for local_position, (r_prepare, r_query) in enumerate(relation_order):
            relation_match = r_prepare == r_query
            local_rows.append(
                {
                    **schedule_row_base(condition),
                    **common_relation_hashes,
                    "block_id": block_id,
                    "block_local_position": local_position,
                    "row_role": "relation_matrix",
                    "r_prepare": r_prepare,
                    "r_query": r_query,
                    "relation_match": relation_match,
                    "query": "query_relation_pair",
                    "relation_cell": relation_cell_name(r_prepare, r_query),
                    "matched_twin_group": f"{block_id}:relation_matrix",
                    "matched_twin_pair": f"{block_id}:pair_{local_position // 2}",
                }
            )
        for scalar_index, query in enumerate(scalar_order):
            for relation_index, relation in enumerate(RELATIONS):
                local_rows.append(
                    {
                        **schedule_row_base(condition),
                        **common_relation_hashes,
                        "block_id": block_id,
                        "block_local_position": 4 + scalar_index * 2 + relation_index,
                        "row_role": "scalar_control",
                        "r_prepare": relation,
                        "r_query": relation,
                        "relation_match": True,
                        "query": query,
                        "relation_cell": f"{query}_{relation}",
                        "matched_twin_group": f"{block_id}:scalar:{query}",
                        "matched_twin_pair": f"{block_id}:scalar:{query}",
                    }
                )
        for control_index, control in enumerate(control_order):
            local_rows.append(
                {
                    **schedule_row_base(condition),
                    **common_relation_hashes,
                    "block_id": block_id,
                    "block_local_position": 8 + control_index,
                    "row_role": "relation_control",
                    "r_prepare": "control",
                    "r_query": "control",
                    "relation_match": False,
                    "query": control,
                    "relation_cell": control,
                    "matched_twin_group": f"{block_id}:control:{control}",
                    "matched_twin_pair": f"{block_id}:control:{control}",
                }
            )
        for row in sorted(local_rows, key=lambda item: item["block_local_position"]):
            row["execution_ordinal"] = ordinal
            row["tuple_id"] = f"{TRANSACTION_RUN_ID}:{ordinal:05d}:{stable_token({'ordinal': ordinal, 'block': block_id})}"
            rows.append(row)
            ordinal += 1

    schedule = {
        "schema": "FAMILY10H_RELATION_ONLY_PUBLIC_SCHEDULE_V1",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "public_randomization_seed": PUBLIC_RANDOMIZATION_SEED,
        "schedule_columns": SCHEDULE_COLUMNS,
        "tuple_count": len(rows),
        "base_condition_count": len(conditions),
        "rows_per_base_condition": len(rows) // len(conditions),
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
    if grammar.get("schema") != "FAMILY10H_RELATION_ONLY_GRAMMAR_V1":
        failures.append("schema mismatch")
    expected = digest({k: v for k, v in grammar.items() if k != "grammar_sha256"})
    if grammar.get("grammar_sha256") != expected:
        failures.append("grammar digest mismatch")
    proof = relation_marginal_equality_proof(grammar)
    if not proof["passed"]:
        failures.append("marginal equality proof failed")
    return {"passed": not failures, "failures": failures, "proof": proof}


def validate_schedule(schedule: dict[str, Any], grammar: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if schedule.get("schema") != "FAMILY10H_RELATION_ONLY_PUBLIC_SCHEDULE_V1":
        failures.append("schedule schema mismatch")
    expected = digest({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    if schedule.get("schedule_sha256") != expected:
        failures.append("schedule digest mismatch")
    rows = schedule.get("rows", [])
    if schedule.get("tuple_count") != len(rows):
        failures.append("tuple count mismatch")
    if [row.get("execution_ordinal") for row in rows] != list(range(len(rows))):
        failures.append("execution ordinal sequence mismatch")
    if any(token in row["tuple_id"] for row in rows for token in ["relation_r0", "relation_r1", "r0", "r1"]):
        failures.append("relation-label leakage through tuple IDs")
    proof = relation_marginal_equality_proof(grammar)
    for row in rows:
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
        if row.get("allocation_order_class") != "relation_label_independent_prefault_v1":
            failures.append("allocation-order class drift")
            break
        if row.get("page_count_A") != PAGE_COUNT_PER_LANE or row.get("page_count_B") != PAGE_COUNT_PER_LANE:
            failures.append("page count drift")
            break
        if row.get("line_count_A") != LINE_COUNT or row.get("line_count_B") != LINE_COUNT:
            failures.append("line count drift")
            break
        if row.get("cache_set_histogram_sha256") != proof["relation_r0"]["cache_set_histogram_sha256"]:
            failures.append("cache-set histogram drift")
            break
        if row.get("pair_distance_histogram_sha256") != proof["relation_r0"]["pair_distance_histogram_sha256"]:
            failures.append("pair-distance histogram drift")
            break
    block_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        block_rows.setdefault(row["block_id"], []).append(row)
    position_counts: dict[str, dict[int, int]] = {relation_cell_name(*cell): {} for cell in RELATION_CELLS}
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
        scalar_rows = [row for row in block if row["row_role"] == "scalar_control"]
        if len(scalar_rows) != 4:
            failures.append(f"{block_id} scalar controls incomplete")
    for cell_name, counts in position_counts.items():
        if set(counts) != {0, 1, 2, 3}:
            failures.append(f"{cell_name} not counterbalanced over relation positions")
        elif max(counts.values()) - min(counts.values()) > 1:
            failures.append(f"{cell_name} execution-order imbalance")
    return {
        "passed": not failures,
        "failures": failures,
        "tuple_count": len(rows),
        "block_count": len(block_rows),
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
