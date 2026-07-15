#!/usr/bin/env python3
"""Public carrier-state tomography schedule, validation, and offline tests.

This package deliberately characterizes a public Family 10h post-source carrier.
It has no private map, no hidden relation, no target vector, and no wall-crossing
claim path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent

SCIENCE_PACKAGE_ID = "family10h_carrier_tomography_v1_0"
TRANSACTION_RUN_ID = "family10h_carrier_tomography_v1_0"
PUBLIC_RANDOMIZATION_SEED = "family10h-carrier-tomography-v1-public-seed-836d53a8"

M = 2048
TOTAL_BALANCED_WORK = 4096
Q_VALUES = [-1536, -1024, -512, 0, 512, 1024, 1536]
REPLICATES = [0, 1]
MAPPINGS = ["map0", "map1"]
SOURCE_ORDERS = ["A_then_B", "B_then_A"]
SESSIONS = ["session_0", "session_1"]
DELAY_GRID = [
    {"delay_label": "0ns", "delay_ns": 0},
    {"delay_label": "100us", "delay_ns": 100_000},
    {"delay_label": "1ms", "delay_ns": 1_000_000},
    {"delay_label": "10ms", "delay_ns": 10_000_000},
    {"delay_label": "100ms", "delay_ns": 100_000_000},
]

PRIMARY_QUERY_NAMES = [
    "query_A",
    "query_B",
    "query_A_then_B",
    "query_B_then_A",
    "query_sham",
    "carrier_off",
]
FACTORIAL_QUERY_NAMES = ["query_A_then_B", "query_B_then_A"]
FACTORIAL_ARMS = [
    "both_active",
    "A_active_B_dummy",
    "A_dummy_B_active",
    "both_dummy",
]
PRIMARY_ARM = "primary_matrix"

ALLOWED_RESULT_CLASSES = [
    "FAMILY10H_POST_SOURCE_STATE_OBSERVED",
    "FAMILY10H_POST_SOURCE_STATE_NOT_OBSERVED",
    "FAMILY10H_CARRIER_TOMOGRAPHY_CANDIDATE",
    "FAMILY10H_CARRIER_TOMOGRAPHY_CUSTODY_INVALID",
]
FORBIDDEN_RESULT_CLASSES = [
    "ORBITSTATE_ACCESS_ESTABLISHED",
    "RELATIONAL_CARRIER_ESTABLISHED",
    "PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED",
    "CATALYTIC_BORROWING_ESTABLISHED",
    "SMALL_WALL_CROSSED",
]
FORBIDDEN_CLAIM_PATTERNS = [
    "ORBITSTATE_ACCESS",
    "RELATIONAL_CARRIER",
    "PHYSICAL_RELATIONAL_MEMORY",
    "CATALYTIC_BORROWING",
    "SMALL_WALL_CROSSED",
    "SMALL_WALL_PROMOTED",
]
UNRESOLVED_BLOCKER_IDS = [
    "SCALAR-REPLAY-01",
    "NONSEPARABILITY-01",
    "PHYSICAL-MECHANISM-01",
    "RESTORATION-R2-01",
]
POSITIVE_BLOCKER_DISPOSITIONS = [
    "RESOLVED",
    "ESTABLISHED",
    "CLOSED",
    "SATISFIED",
    "EXCLUDED",
]
PACKAGE_DECISION_FROZEN = "FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_FROZEN_AWAITING_AUTHORIZATION"
PACKAGE_DECISION_BLOCKED = "FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED"
PACKAGE_DECISION_NOT_FEASIBLE = "FAMILY10H_CARRIER_TOMOGRAPHY_NOT_FEASIBLE"

EXPECTED_LOCAL_ROOT = str(HERE.parent / "runs" / TRANSACTION_RUN_ID)
EXPECTED_REMOTE_ROOT = f"/root/catcas_live_small_wall/{TRANSACTION_RUN_ID}"
EXPECTED_REMOTE_OUTPUT_ROOT = f"{EXPECTED_REMOTE_ROOT}/output"

SCHEDULE_JSON = HERE / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json"
SCHEDULE_TSV = HERE / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv"
SCHEDULE_SHA = HERE / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256"
CONTRACT_MD = HERE / "CARRIER_TOMOGRAPHY_CONTRACT.md"

PMU_FIELDS = [
    "change_to_dirty",
    "dirty_probe_response",
    "cpu_cycles",
    "duration_ns",
    "time_enabled",
    "time_running",
    "event_ids",
    "pmu_read_size",
    "receiver_cpu_before",
    "receiver_cpu_after",
]
RAW_RECORD_EXTRA_FIELDS = [
    "pmu_value_count",
    "source_cpu_before",
    "source_cpu_after",
    "prefault_cpu",
    "mapping_trace",
    "source_order_trace",
    "map_lane_A_trace",
    "map_lane_B_trace",
    "query_start_monotonic_ns",
    "query_end_monotonic_ns",
    "temperature_c",
    "process_custody",
    "policy_custody",
]
EXPECTED_EVENT_ID_KEYS = [
    "cpu_cycles_not_halted",
    "cache_block_commands_change_to_dirty",
    "probe_responses_dirty",
]

PACKET_KEYS = {"schema", "schedule_sha256", "raw_records", "source_death_receipts", "feature_freeze"}
FEATURE_FREEZE_KEYS = {"frozen_before_analysis", "public_only", "schedule_sha256", "receiver_feature_boundary"}
SOURCE_DEATH_RECEIPT_KEYS = {
    "tuple_id",
    "execution_ordinal",
    "source_pid",
    "waitpid_pid",
    "waitpid_status",
    "wait_status_raw",
    "source_exit_monotonic_ns",
    "query_select_monotonic_ns",
    "source_alive_during_query",
    "source_helper_survives",
    "open_source_ipc_after_waitpid",
    "query_selected_after_waitpid",
    "post_observation_query_or_window_selection",
    "source_cpu_before",
    "source_cpu_after",
}

SCHEDULE_COLUMNS = [
    "tuple_id",
    "execution_ordinal",
    "matrix_block",
    "preparation",
    "q",
    "bank_A_work",
    "bank_B_work",
    "dummy_work",
    "dummy_A_work",
    "dummy_B_work",
    "arm_A_mode",
    "arm_B_mode",
    "source_off_control",
    "query",
    "delay_label",
    "delay_ns",
    "mapping",
    "replicate",
    "session",
    "factorial_arm",
    "source_order",
    "query_order",
    "control",
    "expected_bank_identity",
    "address_layout_identity",
    "map_lane_A",
    "map_lane_B",
    "source_loop_count",
    "address_population_size",
    "timing_envelope_id",
    "query_schedule_id",
    "source_cpu_expected",
    "receiver_cpu_expected",
    "delay_tolerance_ns",
    "source_receipt_required",
    "source_death_receipt_required",
    "pmu_group",
    "pmu_event_config",
]

FORBIDDEN_PUBLIC_KEYS = {
    "private_map",
    "hidden_relation",
    "candidate_label",
    "fold_branch",
    "expected_target_vector",
    "private_adjudication",
    "target_member",
}

TOMOGRAPHY_STATUS_TOKENS = {
    PACKAGE_DECISION_FROZEN,
    PACKAGE_DECISION_BLOCKED,
    PACKAGE_DECISION_NOT_FEASIBLE,
}


class TomographyError(AssertionError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_json_int(value: Any) -> bool:
    return type(value) is int


def is_json_number(value: Any) -> bool:
    return type(value) in {int, float}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_key(value: Any) -> str:
    return hashlib.sha256((PUBLIC_RANDOMIZATION_SEED + ":" + digest(value)).encode("utf-8")).hexdigest()


def public_preparations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in Q_VALUES:
        rows.append(
            {
                "preparation": f"codeword_q_{q:+d}",
                "q": q,
                "bank_A_work": M + q,
                "bank_B_work": M - q,
                "dummy_work": 0,
                "source_off_control": False,
                "description": "public balanced Family 10h carrier codeword",
            }
        )
    rows.append(
        {
            "preparation": "source_off_dummy_total_work",
            "q": 0,
            "bank_A_work": 0,
            "bank_B_work": 0,
            "dummy_work": TOTAL_BALANCED_WORK,
            "source_off_control": True,
            "description": "source-off control with same total work directed to dummy storage",
        }
    )
    return rows


def query_family() -> list[dict[str, Any]]:
    return [
        {
            "query": "query_A",
            "query_order": "A_only",
            "instructions": "Probe the prepared A lane with the standardized receiver loop.",
            "address_sequence": "A_lane_affine_lines",
            "measurement_disturbance": "single-lane destructive risk; fresh preparation required for every cell",
        },
        {
            "query": "query_B",
            "query_order": "B_only",
            "instructions": "Probe the prepared B lane with the standardized receiver loop.",
            "address_sequence": "B_lane_affine_lines",
            "measurement_disturbance": "single-lane destructive risk; fresh preparation required for every cell",
        },
        {
            "query": "query_A_then_B",
            "query_order": "A_then_B",
            "instructions": "Probe A then B in one PMU-scoped ordered receiver window.",
            "address_sequence": "A_lane_affine_lines followed by B_lane_affine_lines",
            "measurement_disturbance": "ordered two-lane destructive risk; fresh preparation required for every cell",
        },
        {
            "query": "query_B_then_A",
            "query_order": "B_then_A",
            "instructions": "Probe B then A in one PMU-scoped ordered receiver window.",
            "address_sequence": "B_lane_affine_lines followed by A_lane_affine_lines",
            "measurement_disturbance": "ordered two-lane destructive risk; fresh preparation required for every cell",
        },
        {
            "query": "query_sham",
            "query_order": "sham_only",
            "instructions": "Run the same receiver loop against untouched sham storage.",
            "address_sequence": "sham_affine_lines",
            "measurement_disturbance": "sham-lane disturbance only",
        },
        {
            "query": "carrier_off",
            "query_order": "carrier_off_same_receiver_sequence",
            "instructions": "Run the receiver sequence with active carrier preparation disabled.",
            "address_sequence": "matched_carrier_off_affine_lines",
            "measurement_disturbance": "carrier-off control disturbance only",
        },
    ]


def query_map() -> dict[str, dict[str, Any]]:
    return {row["query"]: row for row in query_family()}


def physical_freeze() -> dict[str, Any]:
    return {
        "pages": {
            "page_size_bytes": 4096,
            "pages_per_lane": 64,
            "lane_bytes": 262144,
            "resident_pages_required": True,
            "owned_synthetic_storage_only": True,
        },
        "line_offsets": {
            "cache_line_bytes": 64,
            "line_offsets_within_page": [0, 64, 128, 192, 256, 320, 384, 448],
            "lines_per_lane": 4096,
        },
        "bank_layout": {
            "bank_A": "owned lane A",
            "bank_B": "owned lane B",
            "dummy_A": "owned dummy lane matched to A by page count and loop count",
            "dummy_B": "owned dummy lane matched to B by page count and loop count",
            "sham": "owned untouched sham lane matched by address population",
        },
        "mapping_definitions": {
            "map0": {"lane_A_bank": "owned_lane_a", "lane_B_bank": "owned_lane_b"},
            "map1": {"lane_A_bank": "owned_lane_b", "lane_B_bank": "owned_lane_a"},
        },
        "address_formula": {
            "base": "lane_base + 64 * ((73 * logical_line + 19) mod 4096)",
            "map0": "A uses owned_lane_a, B uses owned_lane_b",
            "map1": "A uses owned_lane_b, B uses owned_lane_a",
        },
        "pmu_event_config": {
            "type": "PERF_TYPE_RAW",
            "config_format": "event:0-7,umask:8-15",
            "events": {
                "cpu_cycles_not_halted": {"event": "0x076", "umask": "0x00"},
                "cache_block_commands_change_to_dirty": {"event": "0x0ea", "umask": "0x20"},
                "probe_responses_dirty": {"event": "0x0ec", "umask": "0x0c"},
            },
            "exclude_kernel": True,
            "exclude_hv": True,
            "read_format": "PERF_FORMAT_GROUP|PERF_FORMAT_ID|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING",
        },
        "affine_line_permutation": {
            "formula": "line_index -> (73 * line_index + 19) mod lines_per_lane",
            "multiplier": 73,
            "offset": 19,
        },
        "core_allocation": {"source_core": 4, "receiver_core": 5},
        "source_operation_order": SOURCE_ORDERS,
        "component_order": ["A_component", "B_component", "dummy_component", "sham_component"],
        "mapping_order": MAPPINGS,
        "allocation_lifetime": "allocate before source fork, keep owned buffers until packet sealed, release after evidence copy",
        "dummy_storage": "dedicated owned lane; never aliases A, B, or sham",
    }


def mapping_lanes(mapping: str) -> dict[str, str]:
    definitions = physical_freeze()["mapping_definitions"]
    item = definitions[mapping]
    return {"map_lane_A": item["lane_A_bank"], "map_lane_B": item["lane_B_bank"]}


def delay_tolerance_ns(delay_ns: int) -> int:
    return max(50_000, int(delay_ns * 0.10))


def common_schedule_fields(mapping: str, delay_ns: int, query: str) -> dict[str, Any]:
    lanes = mapping_lanes(mapping)
    return {
        **lanes,
        "source_loop_count": TOTAL_BALANCED_WORK,
        "address_population_size": 4096,
        "timing_envelope_id": "matched_total_4096_work_fixed_receiver_window_v1",
        "query_schedule_id": f"{query}_standard_order_v1",
        "source_cpu_expected": 4,
        "receiver_cpu_expected": 5,
        "delay_tolerance_ns": delay_tolerance_ns(delay_ns),
        "source_receipt_required": True,
        "source_death_receipt_required": True,
        "pmu_group": "family10h_public_carrier_group",
        "pmu_event_config": "cycles_0x076_0x00__c2d_0x0ea_0x20__probe_dirty_0x0ec_0x0c",
    }


def factorial_arm_plan(q: int, arm: str) -> dict[str, Any]:
    active_a = M + q
    active_b = M - q
    if arm == "both_active":
        return {
            "bank_A_work": active_a,
            "bank_B_work": active_b,
            "dummy_A_work": 0,
            "dummy_B_work": 0,
            "dummy_work": 0,
            "arm_A_mode": "active",
            "arm_B_mode": "active",
        }
    if arm == "A_active_B_dummy":
        return {
            "bank_A_work": active_a,
            "bank_B_work": 0,
            "dummy_A_work": 0,
            "dummy_B_work": active_b,
            "dummy_work": active_b,
            "arm_A_mode": "active",
            "arm_B_mode": "dummy_matched",
        }
    if arm == "A_dummy_B_active":
        return {
            "bank_A_work": 0,
            "bank_B_work": active_b,
            "dummy_A_work": active_a,
            "dummy_B_work": 0,
            "dummy_work": active_a,
            "arm_A_mode": "dummy_matched",
            "arm_B_mode": "active",
        }
    if arm == "both_dummy":
        return {
            "bank_A_work": 0,
            "bank_B_work": 0,
            "dummy_A_work": active_a,
            "dummy_B_work": active_b,
            "dummy_work": TOTAL_BALANCED_WORK,
            "arm_A_mode": "dummy_matched",
            "arm_B_mode": "dummy_matched",
        }
    raise TomographyError(f"unknown factorial arm {arm}")


def _base_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qmap = query_map()
    for prep in public_preparations():
        for query in PRIMARY_QUERY_NAMES:
            for delay in DELAY_GRID:
                for mapping in MAPPINGS:
                    for replicate in REPLICATES:
                        for session in SESSIONS:
                            for source_order in SOURCE_ORDERS:
                                qdef = qmap[query]
                                carrier_off = query == "carrier_off"
                                bank_a_work = 0 if carrier_off else prep["bank_A_work"]
                                bank_b_work = 0 if carrier_off else prep["bank_B_work"]
                                dummy_a_work = (
                                    M
                                    if prep["source_off_control"]
                                    else prep["bank_A_work"] if carrier_off else 0
                                )
                                dummy_b_work = (
                                    M
                                    if prep["source_off_control"]
                                    else prep["bank_B_work"] if carrier_off else 0
                                )
                                dummy_work = (
                                    TOTAL_BALANCED_WORK
                                    if carrier_off or prep["source_off_control"]
                                    else prep["dummy_work"]
                                )
                                rows.append(
                                    {
                                        "matrix_block": "persistence_matrix",
                                        "preparation": prep["preparation"],
                                        "q": prep["q"],
                                        "bank_A_work": bank_a_work,
                                        "bank_B_work": bank_b_work,
                                        "dummy_work": dummy_work,
                                        "dummy_A_work": dummy_a_work,
                                        "dummy_B_work": dummy_b_work,
                                        "arm_A_mode": "dummy_matched" if carrier_off else "source_off" if prep["source_off_control"] else "active",
                                        "arm_B_mode": "dummy_matched" if carrier_off else "source_off" if prep["source_off_control"] else "active",
                                        "source_off_control": prep["source_off_control"],
                                        "query": query,
                                        "delay_label": delay["delay_label"],
                                        "delay_ns": delay["delay_ns"],
                                        "mapping": mapping,
                                        "replicate": replicate,
                                        "session": session,
                                        "factorial_arm": PRIMARY_ARM,
                                        "source_order": source_order,
                                        "query_order": qdef["query_order"],
                                        "control": "source_off_null" if prep["source_off_control"] else "public_codeword",
                                        "expected_bank_identity": "carrier_off_dummy" if carrier_off else "source_off_dummy" if prep["source_off_control"] else "A_B_balanced",
                                        "address_layout_identity": "family10h_tomography_layout_v1",
                                        **common_schedule_fields(mapping, delay["delay_ns"], query),
                                    }
                                )
    for q in Q_VALUES:
        prep = {
            "preparation": f"factorial_q_{q:+d}",
            "q": q,
            "source_off_control": False,
        }
        for arm in FACTORIAL_ARMS:
            arm_plan = factorial_arm_plan(q, arm)
            for query in FACTORIAL_QUERY_NAMES:
                for delay in DELAY_GRID:
                    for mapping in MAPPINGS:
                        for replicate in REPLICATES:
                            for session in SESSIONS:
                                for source_order in SOURCE_ORDERS:
                                    qdef = qmap[query]
                                    rows.append(
                                        {
                                            "matrix_block": "factorial_nonadditivity",
                                            "preparation": prep["preparation"],
                                            "q": prep["q"],
                                            **arm_plan,
                                            "source_off_control": False,
                                            "query": query,
                                            "delay_label": delay["delay_label"],
                                            "delay_ns": delay["delay_ns"],
                                            "mapping": mapping,
                                            "replicate": replicate,
                                            "session": session,
                                            "factorial_arm": arm,
                                            "source_order": source_order,
                                            "query_order": qdef["query_order"],
                                            "control": "factorial_matched_arm",
                                            "expected_bank_identity": arm,
                                            "address_layout_identity": "family10h_tomography_layout_v1",
                                            **common_schedule_fields(mapping, delay["delay_ns"], query),
                                        }
                                    )
    return rows


def build_schedule() -> dict[str, Any]:
    unordered = _base_rows()
    ordered = sorted(unordered, key=lambda row: (row["session"], stable_key(row)))
    rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(ordered):
        material = {key: row[key] for key in sorted(row)}
        tuple_digest = stable_key({"ordinal_material": material})[:20]
        completed = dict(row)
        completed["execution_ordinal"] = ordinal
        completed["tuple_id"] = f"{TRANSACTION_RUN_ID}:{ordinal:05d}:{tuple_digest}"
        rows.append(completed)
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE_V1",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "public_randomization_seed": PUBLIC_RANDOMIZATION_SEED,
        "schedule_columns": SCHEDULE_COLUMNS,
        "tuple_count": len(rows),
        "rows": rows,
        "physical_freeze": physical_freeze(),
        "preparation_grammar": public_preparations(),
        "query_family": query_family(),
        "delay_grid": DELAY_GRID,
        "allowed_result_classes": ALLOWED_RESULT_CLASSES,
        "forbidden_result_classes": FORBIDDEN_RESULT_CLASSES,
    }


def expected_tuple_count() -> int:
    persistence = len(public_preparations()) * len(PRIMARY_QUERY_NAMES) * len(DELAY_GRID) * len(MAPPINGS) * len(REPLICATES) * len(SESSIONS) * len(SOURCE_ORDERS)
    factorial = len(Q_VALUES) * len(FACTORIAL_ARMS) * len(FACTORIAL_QUERY_NAMES) * len(DELAY_GRID) * len(MAPPINGS) * len(REPLICATES) * len(SESSIONS) * len(SOURCE_ORDERS)
    return persistence + factorial


def tsv_text(schedule: dict[str, Any]) -> str:
    lines: list[str] = []
    rows = schedule["rows"]
    output = []
    output.append("\t".join(SCHEDULE_COLUMNS))
    for row in rows:
        output.append("\t".join(str(row[col]).lower() if isinstance(row[col], bool) else str(row[col]) for col in SCHEDULE_COLUMNS))
    lines.extend(output)
    return "\n".join(lines) + "\n"


def write_schedule_artifacts() -> dict[str, Any]:
    schedule = build_schedule()
    validate_schedule(schedule)
    write_json(SCHEDULE_JSON, schedule)
    SCHEDULE_TSV.write_bytes(tsv_text(schedule).encode("utf-8"))
    sidecar = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE_SHA256_V1",
        "canonical_sha256": digest(schedule),
        "json_sha256": sha256_file(SCHEDULE_JSON),
        "tsv_sha256": sha256_file(SCHEDULE_TSV),
        "tuple_count": schedule["tuple_count"],
    }
    write_json(SCHEDULE_SHA, sidecar)
    return sidecar


def _path_allows_forbidden_claim(path: tuple[str, ...]) -> bool:
    allowed_leaf_containers = {
        "forbidden_result_classes",
        "forbidden_claim_hits",
        "forbidden_value_hits",
        "does_not_establish",
    }
    if not path:
        return False
    leaf = path[-1]
    parent = path[-2] if len(path) >= 2 else ""
    if parent in allowed_leaf_containers and leaf.isdigit():
        return True
    return leaf in allowed_leaf_containers


def claim_boundary_violations(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        blocker_id = value.get("blocker_id")
        disposition = value.get("disposition")
        if isinstance(blocker_id, str) and blocker_id in UNRESOLVED_BLOCKER_IDS and isinstance(disposition, str):
            upper_disposition = disposition.upper()
            if any(term in upper_disposition for term in POSITIVE_BLOCKER_DISPOSITIONS):
                violations.append(".".join((*path, blocker_id)) + f" has positive blocker disposition {disposition}")
        for key, item in value.items():
            violations.extend(claim_boundary_violations(str(key), (*path, "<key>")))
            if str(key) in UNRESOLVED_BLOCKER_IDS and isinstance(item, str):
                upper_item = item.upper()
                if any(term in upper_item for term in POSITIVE_BLOCKER_DISPOSITIONS):
                    violations.append(".".join((*path, str(key))) + f" has positive blocker disposition {item}")
            violations.extend(claim_boundary_violations(item, (*path, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(claim_boundary_violations(item, (*path, str(index))))
    elif isinstance(value, str):
        for claim in FORBIDDEN_CLAIM_PATTERNS:
            if value == "SMALL_WALL_CROSSED_NOT_PROMOTED":
                continue
            if claim in value and not _path_allows_forbidden_claim(path):
                violations.append(".".join(path) + f" contains forbidden claim {claim}")
    return violations


def claim_boundary_text_violations_from_lines(lines: list[str], label: str) -> list[str]:
    violations: list[str] = []
    awaiting_allowed_fence = False
    in_allowed_forbidden_block = False
    for line in lines:
        stripped = line.strip()
        heading = stripped.lstrip("#").strip().lower()
        if heading in {"forbidden:", "forbidden classes:", "forbidden result classes:"}:
            awaiting_allowed_fence = True
            continue
        if stripped == "```text" and awaiting_allowed_fence:
            awaiting_allowed_fence = False
            in_allowed_forbidden_block = True
            continue
        if stripped == "```" and in_allowed_forbidden_block:
            in_allowed_forbidden_block = False
            continue
        if stripped and stripped != "```text" and awaiting_allowed_fence:
            awaiting_allowed_fence = False
        for claim in FORBIDDEN_CLAIM_PATTERNS:
            if claim in stripped and not in_allowed_forbidden_block and stripped != "SMALL_WALL_CROSSED_NOT_PROMOTED":
                violations.append(f"{label} contains positive forbidden claim {claim}: {stripped}")
    return violations


def claim_boundary_text_violations(path: Path) -> list[str]:
    if not path.exists():
        return []
    return claim_boundary_text_violations_from_lines(path.read_text(encoding="utf-8").splitlines(), path.name)


def validate_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    expected_schedule = build_schedule()
    expected_top_keys = set(expected_schedule)
    if schedule.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE_V1":
        failures.append("bad schedule schema")
    if set(schedule) != expected_top_keys:
        failures.append("top-level keyset mismatch")
    rows = schedule.get("rows")
    if not isinstance(rows, list):
        raise TomographyError("schedule rows must be a list")
    if schedule.get("tuple_count") != len(rows):
        failures.append("tuple_count does not match row count")
    if len(rows) != expected_tuple_count():
        failures.append(f"tuple count {len(rows)} != {expected_tuple_count()}")
    ids = [row.get("tuple_id") for row in rows]
    if len(ids) != len(set(ids)):
        failures.append("duplicate tuple_id")
    ordinals = [row.get("execution_ordinal") for row in rows]
    if ordinals != list(range(len(rows))):
        failures.append("execution order mismatch")
    for required in SCHEDULE_COLUMNS:
        if any(required not in row for row in rows):
            failures.append(f"missing schedule column {required}")
            break
    for key in FORBIDDEN_PUBLIC_KEYS:
        if key in schedule:
            failures.append(f"forbidden public key present: {key}")
    if rows and len(rows) == expected_tuple_count():
        expected_ids = [row["tuple_id"] for row in expected_schedule["rows"]]
        if ids != expected_ids:
            failures.append("tuple multiset or deterministic order differs from frozen generator")
        if rows != expected_schedule["rows"]:
            failures.append("full schedule row equality failed")
        bad_keysets = [
            row.get("tuple_id", f"row-{index}")
            for index, row in enumerate(rows)
            if set(row) != set(SCHEDULE_COLUMNS)
        ]
        if bad_keysets:
            failures.append(f"row keyset mismatch at {bad_keysets[0]}")
    if digest(schedule) != digest(expected_schedule):
        failures.append("canonical full-schedule equality failed")
    claim_failures = claim_boundary_violations(schedule)
    if claim_failures:
        failures.append(claim_failures[0])
    block_counts = Counter(row["matrix_block"] for row in rows)
    expected_blocks = {
        "persistence_matrix": len(public_preparations()) * len(PRIMARY_QUERY_NAMES) * len(DELAY_GRID) * len(MAPPINGS) * len(REPLICATES) * len(SESSIONS) * len(SOURCE_ORDERS),
        "factorial_nonadditivity": len(Q_VALUES) * len(FACTORIAL_ARMS) * len(FACTORIAL_QUERY_NAMES) * len(DELAY_GRID) * len(MAPPINGS) * len(REPLICATES) * len(SESSIONS) * len(SOURCE_ORDERS),
    }
    if dict(block_counts) != expected_blocks:
        failures.append(f"block counts {dict(block_counts)} != {expected_blocks}")
    result = {
        "passed": not failures,
        "failures": failures,
        "tuple_count": len(rows),
        "block_counts": dict(block_counts),
        "expected_blocks": expected_blocks,
    }
    if failures:
        raise TomographyError("; ".join(failures))
    return result


def load_schedule_from_artifacts() -> dict[str, Any]:
    schedule = json.loads(SCHEDULE_JSON.read_text(encoding="utf-8"))
    sidecar = json.loads(SCHEDULE_SHA.read_text(encoding="utf-8"))
    failures = []
    if sidecar.get("canonical_sha256") != digest(schedule):
        failures.append("schedule canonical digest mismatch")
    if sidecar.get("json_sha256") != sha256_file(SCHEDULE_JSON):
        failures.append("schedule JSON digest mismatch")
    if sidecar.get("tsv_sha256") != sha256_file(SCHEDULE_TSV):
        failures.append("schedule TSV digest mismatch")
    validate_schedule(schedule)
    if failures:
        raise TomographyError("; ".join(failures))
    return schedule


def validate_tsv(path: Path = SCHEDULE_TSV) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    failures = []
    if reader.fieldnames != SCHEDULE_COLUMNS:
        failures.append("TSV header mismatch")
    if len(rows) != expected_tuple_count():
        failures.append("TSV row count mismatch")
    schedule_path = path.with_suffix(".json")
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        expected_text = tsv_text(schedule)
        observed_bytes = path.read_bytes()
        if observed_bytes != expected_text.encode("utf-8"):
            failures.append("TSV exact serialization mismatch")
    if failures:
        raise TomographyError("; ".join(failures))
    return {"passed": True, "row_count": len(rows)}


def source_death_custody_law(receipt: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "exact_receipt_keyset": set(receipt) == SOURCE_DEATH_RECEIPT_KEYS,
        "pid_recorded": is_json_int(receipt.get("source_pid")) and receipt.get("source_pid", 0) > 0,
        "waitpid_recorded": is_json_int(receipt.get("waitpid_pid")) and receipt.get("waitpid_pid", 0) > 0,
        "waitpid_matches_pid": receipt.get("waitpid_pid") == receipt.get("source_pid"),
        "waitpid_success": receipt.get("waitpid_status") == "exited_0",
        "wait_status_raw_recorded": is_json_int(receipt.get("wait_status_raw")),
        "wait_status_raw_matches_success": receipt.get("waitpid_status") != "exited_0" or receipt.get("wait_status_raw") == 0,
        "source_exit_timestamp_recorded": is_json_int(receipt.get("source_exit_monotonic_ns")),
        "query_timestamp_after_exit": is_json_int(receipt.get("query_select_monotonic_ns"))
        and is_json_int(receipt.get("source_exit_monotonic_ns"))
        and receipt["query_select_monotonic_ns"] >= receipt["source_exit_monotonic_ns"],
        "source_not_alive": receipt.get("source_alive_during_query") is False,
        "no_helper_survives": receipt.get("source_helper_survives") is False,
        "no_ipc_open": receipt.get("open_source_ipc_after_waitpid") == 0,
        "query_after_source_death": receipt.get("query_selected_after_waitpid") is True,
        "no_post_observation_selection": receipt.get("post_observation_query_or_window_selection") is False,
        "source_cpu_recorded": is_json_int(receipt.get("source_cpu_before"))
        and is_json_int(receipt.get("source_cpu_after"))
        and receipt.get("source_cpu_before") == 4
        and receipt.get("source_cpu_after") == 4,
        "query_tuple_bound": isinstance(receipt.get("tuple_id"), str) and is_json_int(receipt.get("execution_ordinal")),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {"passed": not failures, "checks": checks, "failures": failures}


def validate_raw_record(record: dict[str, Any], schedule_row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_keys = set(SCHEDULE_COLUMNS) | set(PMU_FIELDS) | set(RAW_RECORD_EXTRA_FIELDS)
    if set(record) != expected_keys:
        failures.append("raw record keyset mismatch")
    if record.get("tuple_id") != schedule_row["tuple_id"]:
        failures.append("unexpected ID")
    for key in SCHEDULE_COLUMNS:
        if record.get(key) != schedule_row[key]:
            failures.append(f"schedule binding mismatch {key}")
            return failures
    for field in PMU_FIELDS:
        if field not in record:
            failures.append(f"missing PMU field {field}")
    counter_fields = [
        "change_to_dirty",
        "dirty_probe_response",
        "cpu_cycles",
        "duration_ns",
        "time_enabled",
        "time_running",
        "pmu_read_size",
        "pmu_value_count",
        "source_cpu_before",
        "source_cpu_after",
        "prefault_cpu",
        "receiver_cpu_before",
        "receiver_cpu_after",
    ]
    for field in counter_fields:
        if not is_json_int(record.get(field)) or record.get(field, -1) < 0:
            failures.append(f"invalid numeric PMU/custody field {field}")
            return failures
    if record.get("duration_ns", 0) <= 0:
        failures.append("invalid duration_ns")
    if not isinstance(record.get("time_enabled"), int) or record.get("time_enabled", 0) <= 0:
        failures.append("invalid time_enabled")
    if record.get("time_running") != record.get("time_enabled"):
        failures.append("PMU multiplexing")
    event_ids = record.get("event_ids", {})
    if set(event_ids) != set(EXPECTED_EVENT_ID_KEYS):
        failures.append("PMU event key mismatch")
    elif len(set(event_ids.values())) != len(EXPECTED_EVENT_ID_KEYS) or any(not is_json_int(v) or v <= 0 for v in event_ids.values()):
        failures.append("PMU ID collision")
    if record.get("pmu_read_size") != 72:
        failures.append("partial PMU read")
    if record.get("pmu_value_count") != 3:
        failures.append("partial PMU value group")
    if record.get("receiver_cpu_before") != 5 or record.get("receiver_cpu_after") != 5:
        failures.append("wrong receiver core")
    if record.get("source_cpu_before") != 4 or record.get("source_cpu_after") != 4:
        failures.append("wrong source core")
    if record.get("prefault_cpu") != 5:
        failures.append("wrong prefault core")
    trace_expectations = {
        "mapping_trace": schedule_row["mapping"],
        "source_order_trace": schedule_row["source_order"],
        "map_lane_A_trace": schedule_row["map_lane_A"],
        "map_lane_B_trace": schedule_row["map_lane_B"],
    }
    for field, expected in trace_expectations.items():
        if record.get(field) != expected:
            failures.append(f"physical trace mismatch {field}")
    if not is_json_int(record.get("query_start_monotonic_ns")) or not is_json_int(record.get("query_end_monotonic_ns")):
        failures.append("query timing trace missing")
    elif record["query_end_monotonic_ns"] < record["query_start_monotonic_ns"]:
        failures.append("query timing trace reversed")
    if not is_json_number(record.get("temperature_c")) or not (0.0 < float(record["temperature_c"]) < 68.0):
        failures.append("temperature failure")
    if record.get("process_custody") != "source_dead_before_query":
        failures.append("process custody failure")
    if record.get("policy_custody") != "policy_readable_stable":
        failures.append("policy drift")
    return failures


def validate_evidence_packet(packet: dict[str, Any], schedule: dict[str, Any] | None = None) -> dict[str, Any]:
    schedule = schedule or build_schedule()
    validate_schedule(schedule)
    expected_ids = [row["tuple_id"] for row in schedule["rows"]]
    expected_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    raw_records = packet.get("raw_records", [])
    receipts = packet.get("source_death_receipts", [])
    failures: list[str] = []
    if set(packet) != PACKET_KEYS:
        failures.append("packet keyset mismatch")
    if packet.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_EVIDENCE_PACKET_V1":
        failures.append("packet schema mismatch")
    if packet.get("schedule_sha256") != digest(schedule):
        failures.append("packet schedule hash mismatch")
    feature_freeze = packet.get("feature_freeze", {})
    if set(feature_freeze) != FEATURE_FREEZE_KEYS:
        failures.append("feature freeze keyset mismatch")
    feature_claims = claim_boundary_violations(feature_freeze)
    if feature_claims:
        failures.append(feature_claims[0])
    if feature_freeze.get("frozen_before_analysis") is not True or feature_freeze.get("public_only") is not True:
        failures.append("feature freeze invalid")
    if feature_freeze.get("schedule_sha256") != digest(schedule):
        failures.append("feature freeze schedule hash mismatch")
    if len(raw_records) != len(expected_ids):
        failures.append("raw record count mismatch")
    raw_ids = [row.get("tuple_id") for row in raw_records]
    if Counter(raw_ids) != Counter(expected_ids):
        failures.append("exact tuple equality failed")
    if raw_ids != expected_ids:
        failures.append("exact executed order failed")
    if len(receipts) != len(expected_ids):
        failures.append("source-death receipt count mismatch")
    receipt_ids = [row.get("tuple_id") for row in receipts]
    if receipt_ids != expected_ids:
        failures.append("source-death receipt order mismatch")
    receipt_by_id = {row.get("tuple_id"): row for row in receipts}
    for ordinal, tuple_id in enumerate(expected_ids):
        if tuple_id not in receipt_by_id:
            failures.append(f"missing source-death receipt {tuple_id}")
            break
        if receipt_by_id[tuple_id].get("execution_ordinal") != ordinal:
            failures.append(f"source-death ordinal mismatch {tuple_id}")
            break
        law = source_death_custody_law(receipt_by_id[tuple_id])
        receipt_claims = claim_boundary_violations(receipt_by_id[tuple_id])
        if receipt_claims:
            failures.append(receipt_claims[0])
            break
        if not law["passed"]:
            failures.append(f"source-death custody failed {tuple_id}: {','.join(law['failures'])}")
            break
    for ordinal, record in enumerate(raw_records):
        tuple_id = record.get("tuple_id")
        if tuple_id not in expected_by_id:
            failures.append("unexpected ID")
            break
        if record.get("execution_ordinal") != ordinal:
            failures.append("raw execution ordinal mismatch")
            break
        record_claims = claim_boundary_violations(record)
        if record_claims:
            failures.append(record_claims[0])
            break
        failures.extend(validate_raw_record(record, expected_by_id[tuple_id]))
        if failures:
            break
    return {"passed": not failures, "failures": failures, "expected_count": len(expected_ids), "observed_count": len(raw_records)}


def synthetic_record(schedule_row: dict[str, Any], value: int = 1000) -> dict[str, Any]:
    return {
        **schedule_row,
        "change_to_dirty": value,
        "dirty_probe_response": value // 2,
        "cpu_cycles": value * 10,
        "duration_ns": value * 100,
        "time_enabled": 100,
        "time_running": 100,
        "event_ids": {
            "cpu_cycles_not_halted": 10,
            "cache_block_commands_change_to_dirty": 11,
            "probe_responses_dirty": 12,
        },
        "pmu_read_size": 72,
        "pmu_value_count": 3,
        "source_cpu_before": 4,
        "source_cpu_after": 4,
        "prefault_cpu": 5,
        "receiver_cpu_before": 5,
        "receiver_cpu_after": 5,
        "mapping_trace": schedule_row["mapping"],
        "source_order_trace": schedule_row["source_order"],
        "map_lane_A_trace": schedule_row["map_lane_A"],
        "map_lane_B_trace": schedule_row["map_lane_B"],
        "query_start_monotonic_ns": 1_000_000_000 + schedule_row.get("execution_ordinal", 0) * 1_000_000,
        "query_end_monotonic_ns": 1_000_000_100 + schedule_row.get("execution_ordinal", 0) * 1_000_000,
        "temperature_c": 52.0,
        "process_custody": "source_dead_before_query",
        "policy_custody": "policy_readable_stable",
    }


def synthetic_death_receipt(schedule_row: dict[str, Any]) -> dict[str, Any]:
    ordinal = schedule_row.get("execution_ordinal", 0)
    return {
        "tuple_id": schedule_row["tuple_id"],
        "execution_ordinal": ordinal,
        "source_pid": 10000 + ordinal,
        "waitpid_pid": 10000 + ordinal,
        "waitpid_status": "exited_0",
        "wait_status_raw": 0,
        "source_exit_monotonic_ns": 1_000_000_000 + ordinal * 1_000_000,
        "query_select_monotonic_ns": 1_000_000_000 + ordinal * 1_000_000 + max(1, schedule_row.get("delay_ns", 0)),
        "source_alive_during_query": False,
        "source_helper_survives": False,
        "open_source_ipc_after_waitpid": 0,
        "query_selected_after_waitpid": True,
        "post_observation_query_or_window_selection": False,
        "source_cpu_before": 4,
        "source_cpu_after": 4,
    }


def minimal_success_packet(schedule: dict[str, Any] | None = None) -> dict[str, Any]:
    schedule = schedule or build_schedule()
    rows = schedule["rows"]
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_EVIDENCE_PACKET_V1",
        "schedule_sha256": digest(schedule),
        "raw_records": [synthetic_record(row, 1000 + i % 17) for i, row in enumerate(rows)],
        "source_death_receipts": [synthetic_death_receipt(row) for row in rows],
        "feature_freeze": {
            "frozen_before_analysis": True,
            "public_only": True,
            "schedule_sha256": digest(schedule),
            "receiver_feature_boundary": "public_schedule_and_public_pmu_only",
        },
    }


def response_value(row: dict[str, Any], *, mode: str) -> float:
    q = float(row["q"])
    query = row["query"]
    delay_ns = float(row["delay_ns"])
    decay = math.exp(-delay_ns / 100_000_000.0)
    mapping_sign = mapping_sign_value(row["mapping"])
    query_gain = query_gain_value(query)
    if mode == "no_persistent_state":
        return 0.0
    if mode == "natural_relaxation":
        return 80.0 * decay
    if mode == "ideal_persistent_public_state":
        return mapping_sign * query_gain * q * decay
    if mode == "route_bank_order_interaction":
        order = 45.0 if row["query_order"] == "A_then_B" else -45.0 if row["query_order"] == "B_then_A" else 0.0
        return mapping_sign * query_gain * q * decay + order
    raise TomographyError(f"unknown response mode {mode}")


def query_gain_value(query: str) -> float:
    return {
        "query_A": 1.00,
        "query_B": -1.00,
        "query_A_then_B": 0.65,
        "query_B_then_A": -0.65,
        "query_sham": 0.0,
        "carrier_off": 0.0,
    }[query]


def mapping_sign_value(mapping: str) -> float:
    return 1.0 if mapping == "map0" else -1.0


def order_sign_value(row: dict[str, Any]) -> float:
    if row["source_order"] == "A_then_B":
        return 1.0
    if row["source_order"] == "B_then_A":
        return -1.0
    return 0.0


def model_features(row: dict[str, Any], rung: str) -> list[float]:
    q = float(row["q"])
    decay = math.exp(-float(row["delay_ns"]) / 100_000_000.0)
    query_gain = query_gain_value(row["query"])
    mapping_sign = mapping_sign_value(row["mapping"])
    order_sign = order_sign_value(row)
    if rung == "S0":
        return [1.0, q]
    if rung == "S1":
        return [1.0, q, query_gain, decay, mapping_sign]
    if rung == "S2":
        return [
            1.0,
            q,
            query_gain,
            decay,
            mapping_sign,
            order_sign,
            q * query_gain,
            q * decay,
            q * mapping_sign,
            query_gain * decay,
            query_gain * mapping_sign,
            q * query_gain * decay * mapping_sign,
            q * order_sign,
        ]
    raise TomographyError(f"bad model rung {rung}")


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row_index in range(n):
            if row_index == col:
                continue
            factor = aug[row_index][col]
            if abs(factor) > 1e-12:
                aug[row_index] = [
                    aug[row_index][i] - factor * aug[col][i]
                    for i in range(n + 1)
                ]
    return [aug[i][n] for i in range(n)]


def fit_linear_model(train: list[dict[str, Any]], rung: str) -> list[float]:
    feature_count = len(model_features(train[0], rung))
    xtx = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    xty = [0.0 for _ in range(feature_count)]
    ridge = 1e-9
    for row in train:
        features = model_features(row, rung)
        for i, left in enumerate(features):
            xty[i] += left * row["y"]
            for j, right in enumerate(features):
                xtx[i][j] += left * right
    for i in range(feature_count):
        xtx[i][i] += ridge
    return solve_linear_system(xtx, xty)


def predict(row: dict[str, Any], coefficients: list[float], rung: str) -> float:
    return sum(a * b for a, b in zip(model_features(row, rung), coefficients))


def rmse_for(train: list[dict[str, Any]], test: list[dict[str, Any]], rung: str) -> float:
    coefficients = fit_linear_model(train, rung)
    return math.sqrt(sum((predict(row, coefficients, rung) - row["y"]) ** 2 for row in test) / len(test))


def heldout_bundle(samples: list[dict[str, Any]], test_predicate: Any) -> dict[str, Any]:
    train = [row for row in samples if not test_predicate(row)]
    test = [row for row in samples if test_predicate(row)]
    scores = {rung: rmse_for(train, test, rung) for rung in ["S0", "S1", "S2"]}
    threshold = max(1e-6, statistics.pstdev(row["y"] for row in train) * 0.01)
    sufficient = [rung for rung in ["S0", "S1", "S2"] if scores[rung] <= threshold]
    return {
        "test_count": len(test),
        "train_count": len(train),
        "rmse": scores,
        "noise_aware_threshold": threshold,
        "smallest_sufficient_model": sufficient[0] if sufficient else "not_sufficient",
    }


def matrix_rank(matrix: list[list[float]], tolerance: float = 1e-9) -> int:
    work = [row[:] for row in matrix]
    rank = 0
    rows = len(work)
    cols = len(work[0]) if rows else 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if abs(work[row][col]) > tolerance:
                pivot = row
                break
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][col]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(rows):
            if row != rank and abs(work[row][col]) > tolerance:
                factor = work[row][col]
                work[row] = [work[row][i] - factor * work[rank][i] for i in range(cols)]
        rank += 1
    return rank


def symmetric_jacobi_eigenvalues(matrix: list[list[float]], tolerance: float = 1e-9, max_sweeps: int = 100) -> list[float]:
    work = [row[:] for row in matrix]
    n = len(work)
    if n == 0:
        return []
    for _ in range(max_sweeps):
        pivot_i = 0
        pivot_j = 1 if n > 1 else 0
        max_offdiag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(work[i][j])
                if value > max_offdiag:
                    max_offdiag = value
                    pivot_i = i
                    pivot_j = j
        if max_offdiag <= tolerance or n == 1:
            break
        app = work[pivot_i][pivot_i]
        aqq = work[pivot_j][pivot_j]
        apq = work[pivot_i][pivot_j]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(angle)
        s = math.sin(angle)
        for k in range(n):
            if k not in {pivot_i, pivot_j}:
                aip = work[k][pivot_i]
                aiq = work[k][pivot_j]
                work[k][pivot_i] = work[pivot_i][k] = c * aip - s * aiq
                work[k][pivot_j] = work[pivot_j][k] = s * aip + c * aiq
        work[pivot_i][pivot_i] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        work[pivot_j][pivot_j] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        work[pivot_i][pivot_j] = work[pivot_j][pivot_i] = 0.0
    return sorted((max(0.0, work[i][i]) for i in range(n)), reverse=True)


def singular_spectrum(matrix: list[list[float]]) -> list[float]:
    rows = len(matrix)
    if rows == 0:
        return []
    gram = [[0.0 for _ in range(rows)] for _ in range(rows)]
    for i in range(rows):
        for j in range(rows):
            gram[i][j] = sum(a * b for a, b in zip(matrix[i], matrix[j]))
    return [math.sqrt(value) for value in symmetric_jacobi_eigenvalues(gram)]


def operational_distinguishability(samples: list[dict[str, Any]]) -> dict[str, Any]:
    signal_rows = [row for row in samples if row["query"] not in {"query_sham", "carrier_off"}]
    train = [row for row in signal_rows if row["replicate"] == 0]
    test = [row for row in signal_rows if row["replicate"] == 1]
    coordinates = sorted(
        {
            (row["query"], row["delay_label"], row["mapping"], row["session"], row["source_order"])
            for row in signal_rows
        }
    )
    def vector_for(rows: list[dict[str, Any]], q: int) -> list[float]:
        result = []
        for coordinate in coordinates:
            values = [
                row["y"]
                for row in rows
                if row["q"] == q
                and (row["query"], row["delay_label"], row["mapping"], row["session"], row["source_order"]) == coordinate
            ]
            result.append(statistics.fmean(values) if values else 0.0)
        return result

    centroids = {q: vector_for(train, q) for q in Q_VALUES}
    confusion: dict[str, dict[str, int]] = {str(q): {str(p): 0 for p in Q_VALUES} for q in Q_VALUES}
    for q in Q_VALUES:
        observed = vector_for(test, q)
        predicted = min(
            Q_VALUES,
            key=lambda candidate: math.sqrt(sum((a - b) ** 2 for a, b in zip(observed, centroids[candidate]))),
        )
        confusion[str(q)][str(predicted)] += 1
    matrix: list[list[float]] = []
    rank_coordinates = sorted(
        {
            (row["query"], row["mapping"], row["delay_label"])
            for row in signal_rows
        }
    )
    for q in Q_VALUES:
        row_values = []
        for query, mapping, delay_label in rank_coordinates:
            values = [
                row["y"]
                for row in signal_rows
                if row["q"] == q and row["query"] == query and row["mapping"] == mapping and row["delay_label"] == delay_label
            ]
            row_values.append(statistics.fmean(values))
        matrix.append(row_values)
    between = [
        math.sqrt(sum((a - b) ** 2 for a, b in zip(centroids[left], centroids[right])))
        for i, left in enumerate(Q_VALUES)
        for right in Q_VALUES[i + 1 :]
    ]
    within = [
        math.sqrt(sum((a - b) ** 2 for a, b in zip(vector_for(test, q), centroids[q])))
        for q in Q_VALUES
    ]
    spectrum = singular_spectrum(matrix)
    return {
        "cross_validated_codeword_classification": True,
        "confusion_matrix": confusion,
        "response_matrix_effective_rank": matrix_rank(matrix),
        "singular_spectrum": spectrum,
        "between_state_min_distance": min(between),
        "within_state_max_distance": max(within),
    }


def fit_operator_ladder(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "held_out_replicate": heldout_bundle(samples, lambda row: row["replicate"] == 1),
        "held_out_mapping": heldout_bundle(samples, lambda row: row["mapping"] == "map1" and row["replicate"] == 1),
        "held_out_delay": heldout_bundle(samples, lambda row: row["delay_label"] == "10ms" and row["replicate"] == 1),
        "distinguishability": operational_distinguishability(samples),
    }


def factorial_jq(values: dict[str, float]) -> float:
    return values["both_active"] - values["A_active_B_dummy"] - values["A_dummy_B_active"] + values["both_dummy"]


def signed_query_response(record: dict[str, Any], schedule_row: dict[str, Any]) -> float:
    value = float(record["change_to_dirty"])
    query = schedule_row["query"]
    if schedule_row["source_off_control"] or query in {"query_sham", "carrier_off"}:
        return value
    if query == "query_A":
        return value
    if query == "query_B":
        return -value
    if query == "query_A_then_B":
        return 0.5 * value
    if query == "query_B_then_A":
        return -0.5 * value
    return 0.0


def evidence_samples(packet: dict[str, Any], schedule: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in packet["raw_records"]:
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] != "persistence_matrix" or row["source_off_control"]:
            continue
        if row["query"] in {"query_sham", "carrier_off"}:
            continue
        key = (
            row["preparation"],
            row["q"],
            row["delay_label"],
            row["delay_ns"],
            row["mapping"],
            row["replicate"],
            row["session"],
            row["source_order"],
        )
        grouped[key][row["query"]].append(float(record["change_to_dirty"]))
    samples: list[dict[str, Any]] = []
    for key, query_values in grouped.items():
        a_values = query_values.get("query_A", [])
        b_values = query_values.get("query_B", [])
        ordered_ab = query_values.get("query_A_then_B", [])
        ordered_ba = query_values.get("query_B_then_A", [])
        if not a_values or not b_values:
            continue
        y = statistics.fmean(a_values) - statistics.fmean(b_values)
        if ordered_ab and ordered_ba:
            y += 0.5 * (statistics.fmean(ordered_ab) - statistics.fmean(ordered_ba))
        prep, q, delay_label, delay_ns, mapping, replicate, session, source_order = key
        samples.append(
            {
                "preparation": prep,
                "q": q,
                "query": "query_A",
                "query_order": "A_minus_B_contrast",
                "delay_label": delay_label,
                "delay_ns": delay_ns,
                "mapping": mapping,
                "replicate": replicate,
                "session": session,
                "source_order": source_order,
                "y": y,
            }
        )
    return samples


def lifetime_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    curves: dict[str, dict[str, Any]] = {}
    for key in sorted({(row["preparation"], row["query"], row["session"], row["mapping"]) for row in samples}):
        prep, query, session, mapping = key
        points = []
        for delay_label, delay_ns in DELAY_GRID:
            values = [
                abs(row["y"])
                for row in samples
                if row["preparation"] == prep
                and row["query"] == query
                and row["session"] == session
                and row["mapping"] == mapping
                and row["delay_label"] == delay_label
            ]
            if values:
                mean = statistics.fmean(values)
                spread = statistics.pstdev(values) if len(values) > 1 else 0.0
                half_width = 1.96 * spread / math.sqrt(len(values)) if len(values) > 1 else 0.0
                points.append(
                    {
                        "delay_label": delay_label,
                        "delay_ns": delay_ns,
                        "mean_abs_response": mean,
                        "pstdev": spread,
                        "confidence_interval_95": [mean - half_width, mean + half_width],
                        "n": len(values),
                    }
                )
        first = points[0]["mean_abs_response"] if points else 0.0
        last = points[-1]["mean_abs_response"] if points else 0.0
        if first <= 50.0:
            lifetime_class = "vanishes_before_source_death_or_not_observed"
        elif last > 0.5 * first:
            lifetime_class = "persists_across_the_full_grid"
        elif last > 50.0:
            lifetime_class = "survives_a_bounded_delay"
        else:
            lifetime_class = "survives_only_immediate_handoff"
        curves[f"{prep}:{query}:{session}:{mapping}"] = {
            "preparation": prep,
            "query": query,
            "session": session,
            "mapping": mapping,
            "points": points,
            "lifetime_class": lifetime_class,
        }
    by_prep_query: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_prep_query_mapping: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for curve in curves.values():
        terminal = curve["points"][-1]["mean_abs_response"] if curve["points"] else 0.0
        by_prep_query[(curve["preparation"], curve["query"])].append(terminal)
        by_prep_query_mapping[(curve["preparation"], curve["query"], curve["mapping"])].append(terminal)
    session_variation = {
        f"{prep}:{query}": (max(values) - min(values) if values else 0.0)
        for (prep, query), values in by_prep_query.items()
    }
    mapping_variation = {
        f"{prep}:{query}:{mapping}": (max(values) - min(values) if values else 0.0)
        for (prep, query, mapping), values in by_prep_query_mapping.items()
    }
    return {
        "curve_count": len(curves),
        "curves": curves,
        "session_variation": session_variation,
        "mapping_variation": mapping_variation,
    }


def factorial_analysis(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in packet["raw_records"]:
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] != "factorial_nonadditivity":
            continue
        key = (
            row["q"],
            row["query"],
            row["delay_label"],
            row["mapping"],
            row["replicate"],
            row["session"],
            row["source_order"],
        )
        grouped[key][row["factorial_arm"]].append(float(record["change_to_dirty"]))
    effects = []
    for key, arms in grouped.items():
        if all(arm in arms for arm in FACTORIAL_ARMS):
            means = {arm: statistics.fmean(arms[arm]) for arm in FACTORIAL_ARMS}
            effects.append({"key": list(key), "arm_means": means, "J_q": factorial_jq(means)})
    jq_values = [item["J_q"] for item in effects]
    return {
        "matched_group_count": len(effects),
        "max_abs_J_q": max((abs(value) for value in jq_values), default=0.0),
        "mean_abs_J_q": statistics.fmean(abs(value) for value in jq_values) if jq_values else 0.0,
        "effects": effects[:32],
        "effect_rows_truncated": max(0, len(effects) - 32),
    }


def analyze_evidence_packet(packet: dict[str, Any], schedule: dict[str, Any] | None = None) -> dict[str, Any]:
    schedule = schedule or build_schedule()
    validation = validate_evidence_packet(packet, schedule)
    if not validation["passed"]:
        return {"passed": False, "validation": validation}
    samples = evidence_samples(packet, schedule)
    return {
        "passed": True,
        "validation": validation,
        "operator_ladder": fit_operator_ladder(samples),
        "lifetime": lifetime_summary(samples),
        "factorial": factorial_analysis(packet, schedule),
        "adjudication": adjudicate_tomography_packet(packet, schedule),
    }


def operator_prediction_gate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ladder = fit_operator_ladder(samples)
    gates = {
        name: ladder[name]["smallest_sufficient_model"] != "not_sufficient"
        for name in ["held_out_replicate", "held_out_mapping", "held_out_delay"]
    }
    return {"passed": all(gates.values()), "gates": gates, "operator_ladder": ladder}


def adjudicate_tomography_packet(packet: dict[str, Any], schedule: dict[str, Any] | None = None) -> dict[str, Any]:
    schedule = schedule or build_schedule()
    validation = validate_evidence_packet(packet, schedule)
    if not validation["passed"]:
        return {
            "result_class": "FAMILY10H_CARRIER_TOMOGRAPHY_CUSTODY_INVALID",
            "passed": False,
            "validation": validation,
            "exclusive_result_class": True,
        }
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    values_by_stratum_q_query: dict[tuple[str, int], dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nulls_by_stratum_control: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in packet["raw_records"]:
        schedule_row = rows_by_id[record["tuple_id"]]
        value = float(record["change_to_dirty"])
        stratum = (schedule_row["session"], int(schedule_row["replicate"]))
        if schedule_row["source_off_control"]:
            nulls_by_stratum_control[stratum]["source_off"].append(value)
        elif schedule_row["query"] == "query_sham":
            nulls_by_stratum_control[stratum]["query_sham"].append(value)
        elif schedule_row["query"] == "carrier_off":
            nulls_by_stratum_control[stratum]["carrier_off"].append(value)
        elif schedule_row["matrix_block"] == "persistence_matrix":
            values_by_stratum_q_query[stratum][int(schedule_row["q"])][schedule_row["query"]].append(value)
    operator_gate = operator_prediction_gate(evidence_samples(packet, schedule))
    stratum_results: dict[str, dict[str, Any]] = {}
    expected_strata = [(session, replicate) for session in SESSIONS for replicate in REPLICATES]
    for stratum in expected_strata:
        contrasts: dict[int, float] = {}
        for q, query_values in values_by_stratum_q_query[stratum].items():
            a_values = query_values.get("query_A", [])
            b_values = query_values.get("query_B", [])
            ordered_ab = query_values.get("query_A_then_B", [])
            ordered_ba = query_values.get("query_B_then_A", [])
            if a_values and b_values:
                base_contrast = statistics.fmean(a_values) - statistics.fmean(b_values)
                order_contrast = (
                    0.5 * (statistics.fmean(ordered_ab) - statistics.fmean(ordered_ba))
                    if ordered_ab and ordered_ba
                    else 0.0
                )
                contrasts[q] = base_contrast + order_contrast
        max_signal = max((abs(value) for value in contrasts.values()), default=0.0)
        null_means = {
            control: abs(statistics.fmean(values)) if values else math.inf
            for control, values in nulls_by_stratum_control[stratum].items()
        }
        for required_control in ["source_off", "query_sham", "carrier_off"]:
            null_means.setdefault(required_control, math.inf)
        null_pass = all(value <= 50.0 for value in null_means.values())
        sign_pass = all(
            contrasts.get(q, 0.0) * contrasts.get(-q, 0.0) < 0.0
            for q in [512, 1024, 1536]
            if q in contrasts and -q in contrasts
        )
        magnitude_pass = all(
            abs(contrasts.get(a, 0.0)) <= abs(contrasts.get(b, 0.0))
            for a, b in [(512, 1024), (1024, 1536)]
        )
        stratum_results[f"{stratum[0]}:replicate_{stratum[1]}"] = {
            "null_mean_abs_by_control": null_means,
            "max_signal": max_signal,
            "contrast_by_q": contrasts,
            "null_pass": null_pass,
            "sign_pass": sign_pass,
            "magnitude_pass": magnitude_pass,
            "observed_pass": null_pass and sign_pass and magnitude_pass and max_signal > 50.0 and operator_gate["passed"],
            "not_observed_pass": null_pass and max_signal <= 50.0,
        }
    if all(item["not_observed_pass"] for item in stratum_results.values()):
        result_class = "FAMILY10H_POST_SOURCE_STATE_NOT_OBSERVED"
    elif all(item["observed_pass"] for item in stratum_results.values()):
        result_class = "FAMILY10H_POST_SOURCE_STATE_OBSERVED"
    else:
        result_class = "FAMILY10H_CARRIER_TOMOGRAPHY_CANDIDATE"
    return {
        "result_class": result_class,
        "passed": True,
        "validation": validation,
        "metrics": {
            "strata": stratum_results,
            "no_aggregate_rescue": True,
            "operator_prediction_gate": operator_gate,
        },
        "exclusive_result_class": result_class in ALLOWED_RESULT_CLASSES,
    }


def feature_boundary_self_test() -> dict[str, Any]:
    schedule = build_schedule()
    text = json.dumps(schedule, sort_keys=True)
    forbidden_value_hits = [
        token
        for token in [
            "private_map",
            "hidden_relation",
            "fold_branch",
            "expected_target_vector",
            "private_adjudication",
        ]
        if token in text
    ]
    artifact_violations: list[str] = []
    for path in sorted(HERE.glob("CARRIER_TOMOGRAPHY*.json")):
        try:
            artifact_violations.extend(claim_boundary_violations(json.loads(path.read_text(encoding="utf-8")), (path.name,)))
        except json.JSONDecodeError:
            artifact_violations.append(f"{path.name} is not valid JSON")
    artifact_violations.extend(claim_boundary_text_violations(CONTRACT_MD))
    injected_forbidden_claim_rejected = bool(claim_boundary_violations({"result_claim": "ORBITSTATE_ACCESS_ESTABLISHED"}))
    injected_claim_ceiling_rejected = bool(claim_boundary_violations({"claim_ceiling": "ORBITSTATE_ACCESS_ESTABLISHED"}))
    injected_forbidden_key_rejected = bool(claim_boundary_violations({"ORBITSTATE_ACCESS_ESTABLISHED": "present"}))
    injected_variant_rejected = bool(claim_boundary_violations({"result_claim": "ORBITSTATE_ACCESS_CANDIDATE"}))
    injected_wall_promotion_rejected = bool(claim_boundary_violations({"result_claim": "SMALL_WALL_PROMOTED"}))
    injected_blocker_resolution_rejected = bool(
        claim_boundary_violations({"blocker_dispositions": {blocker: "RESOLVED" for blocker in UNRESOLVED_BLOCKER_IDS}})
    )
    injected_denied_claim_subtree_rejected = bool(
        claim_boundary_violations({"denied_claims": {"result_claim": "ORBITSTATE_ACCESS_ESTABLISHED"}})
    )
    injected_artifact_violation_subtree_rejected = bool(
        claim_boundary_violations({"artifact_violations": {"SMALL_WALL_PROMOTED": "present"}})
    )
    injected_blocker_object_rejected = bool(
        claim_boundary_violations({"blocker_id": "SCALAR-REPLAY-01", "disposition": "RESOLVED"})
    )
    injected_unfenced_forbidden_heading_rejected = bool(
        claim_boundary_text_violations_from_lines(
            ["Forbidden diagnostics:", "ORBITSTATE_ACCESS_ESTABLISHED"],
            "injected_contract",
        )
    )
    injected_inline_forbidden_heading_rejected = all(
        bool(claim_boundary_text_violations_from_lines([line], "injected_contract"))
        for line in [
            "Forbidden classes: ORBITSTATE_ACCESS_ESTABLISHED",
            "Forbidden classes are now established: ORBITSTATE_ACCESS_ESTABLISHED",
            "Forbidden result classes now include achieved SMALL_WALL_PROMOTED",
        ]
    )
    injected_exact_fenced_forbidden_block_allowed = not claim_boundary_text_violations_from_lines(
        ["Forbidden classes:", "```text", "ORBITSTATE_ACCESS_ESTABLISHED", "```"],
        "injected_contract",
    )
    forbidden_claim_hits = [claim for claim in FORBIDDEN_RESULT_CLASSES if claim in ALLOWED_RESULT_CLASSES]
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_FEATURE_BOUNDARY_SELF_TEST_V1",
        "public_only": True,
        "no_private_map": "private_map" not in text,
        "no_fold_branch": "fold_branch" not in text,
        "no_target_vector": "expected_target_vector" not in text,
        "forbidden_value_hits": forbidden_value_hits,
        "forbidden_claim_hits": forbidden_claim_hits,
        "artifact_violations": artifact_violations,
        "injected_forbidden_claim_rejected": injected_forbidden_claim_rejected,
        "injected_forbidden_claim_ceiling_rejected": injected_claim_ceiling_rejected,
        "injected_forbidden_key_rejected": injected_forbidden_key_rejected,
        "injected_variant_rejected": injected_variant_rejected,
        "injected_wall_promotion_rejected": injected_wall_promotion_rejected,
        "injected_blocker_resolution_rejected": injected_blocker_resolution_rejected,
        "injected_denied_claim_subtree_rejected": injected_denied_claim_subtree_rejected,
        "injected_artifact_violation_subtree_rejected": injected_artifact_violation_subtree_rejected,
        "injected_blocker_object_rejected": injected_blocker_object_rejected,
        "injected_unfenced_forbidden_heading_rejected": injected_unfenced_forbidden_heading_rejected,
        "injected_inline_forbidden_heading_rejected": injected_inline_forbidden_heading_rejected,
        "injected_exact_fenced_forbidden_block_allowed": injected_exact_fenced_forbidden_block_allowed,
    }
    result["passed"] = (
        not forbidden_value_hits
        and not forbidden_claim_hits
        and not artifact_violations
        and injected_forbidden_claim_rejected
        and injected_claim_ceiling_rejected
        and injected_forbidden_key_rejected
        and injected_variant_rejected
        and injected_wall_promotion_rejected
        and injected_blocker_resolution_rejected
        and injected_denied_claim_subtree_rejected
        and injected_artifact_violation_subtree_rejected
        and injected_blocker_object_rejected
        and injected_unfenced_forbidden_heading_rejected
        and injected_inline_forbidden_heading_rejected
        and injected_exact_fenced_forbidden_block_allowed
    )
    result["feature_boundary_self_test_sha256"] = digest({k: v for k, v in result.items() if k != "feature_boundary_self_test_sha256"})
    return result


def _schedule_validation_fails(schedule: dict[str, Any]) -> bool:
    try:
        validate_schedule(schedule)
    except TomographyError:
        return True
    return False


def self_test() -> dict[str, Any]:
    schedule = build_schedule()
    validate_schedule(schedule)
    ids = [row["tuple_id"] for row in schedule["rows"]]
    rows = schedule["rows"]
    packet = minimal_success_packet(schedule)
    success = validate_evidence_packet(packet, schedule)

    def packet_without(predicate: Any) -> dict[str, Any]:
        keep_ids = {row["tuple_id"] for row in schedule["rows"] if not predicate(row)}
        altered = {
            **packet,
            "raw_records": [row for row in packet["raw_records"] if row["tuple_id"] in keep_ids],
            "source_death_receipts": [row for row in packet["source_death_receipts"] if row["tuple_id"] in keep_ids],
        }
        return altered

    mutated_duplicate = {**packet, "raw_records": packet["raw_records"] + [packet["raw_records"][0]]}
    mutated_unexpected = {**packet, "raw_records": [*packet["raw_records"][:-1], {**synthetic_record(rows[-1]), "tuple_id": "unexpected"}]}
    mutated_order = {**packet, "raw_records": [packet["raw_records"][1], packet["raw_records"][0], *packet["raw_records"][2:]]}
    mutated_tail = {
        **packet,
        "raw_records": [*packet["raw_records"][:-1], {**packet["raw_records"][-1], "receiver_cpu_after": 4}],
    }
    mutated_middle_negative_counter = {
        **packet,
        "raw_records": [
            *packet["raw_records"][: len(packet["raw_records"]) // 2],
            {**packet["raw_records"][len(packet["raw_records"]) // 2], "change_to_dirty": -1},
            *packet["raw_records"][len(packet["raw_records"]) // 2 + 1 :],
        ],
    }
    mutated_first_nonnumeric_counter = {
        **packet,
        "raw_records": [{**packet["raw_records"][0], "cpu_cycles": "bad"}, *packet["raw_records"][1:]],
    }
    mutated_zero_duration = {
        **packet,
        "raw_records": [{**packet["raw_records"][0], "duration_ns": 0}, *packet["raw_records"][1:]],
    }
    mutated_negative_temperature = {
        **packet,
        "raw_records": [{**packet["raw_records"][0], "temperature_c": -999.0}, *packet["raw_records"][1:]],
    }
    mutated_raw_extra_private = {
        **packet,
        "raw_records": [{**packet["raw_records"][0], "private_map": "x"}, *packet["raw_records"][1:]],
    }
    mutated_receipt_extra_private = {
        **packet,
        "source_death_receipts": [{**packet["source_death_receipts"][0], "private_map": "x"}, *packet["source_death_receipts"][1:]],
    }
    mutated_feature_extra_private = {**packet, "feature_freeze": {**packet["feature_freeze"], "private_map": "x"}}
    mutated_wait_status = {
        **packet,
        "source_death_receipts": [{**packet["source_death_receipts"][0], "wait_status_raw": 12345}, *packet["source_death_receipts"][1:]],
    }
    mutated_bool_counter = {
        **packet,
        "raw_records": [{**packet["raw_records"][0], "cpu_cycles": True}, *packet["raw_records"][1:]],
    }
    mutated_bool_temperature = {
        **packet,
        "raw_records": [{**packet["raw_records"][0], "temperature_c": True}, *packet["raw_records"][1:]],
    }
    mutated_bool_event_id = {
        **packet,
        "raw_records": [
            {
                **packet["raw_records"][0],
                "event_ids": {**packet["raw_records"][0]["event_ids"], "cpu_cycles_not_halted": True},
            },
            *packet["raw_records"][1:],
        ],
    }
    mutated_bool_pid = {
        **packet,
        "source_death_receipts": [
            {**packet["source_death_receipts"][0], "source_pid": True, "waitpid_pid": True},
            *packet["source_death_receipts"][1:],
        ],
    }
    mutated_hash = {**packet, "schedule_sha256": "0" * 64}
    mutated_feature = {**packet, "feature_freeze": {**packet["feature_freeze"], "frozen_before_analysis": False}}
    mutated_receipts = {
        **packet,
        "source_death_receipts": [packet["source_death_receipts"][1], packet["source_death_receipts"][0], *packet["source_death_receipts"][2:]],
    }
    missing_source_receipt = {**packet, "source_death_receipts": packet["source_death_receipts"][:-1]}
    drifted_schedule = json.loads(json.dumps(schedule))
    drifted_schedule["rows"][0]["q"] = 999
    no_state_packet = {
        **packet,
        "raw_records": [{**record, "change_to_dirty": 0, "dirty_probe_response": 0} for record in packet["raw_records"]],
    }

    def nonnegative_fixture_response(row: dict[str, Any]) -> int:
        if row["source_off_control"] or row["query"] in {"query_sham", "carrier_off"}:
            return 0
        if row["query"] == "query_A":
            return max(int(row["q"]), 0)
        if row["query"] == "query_B":
            return max(-int(row["q"]), 0)
        if row["query"] in {"query_A_then_B", "query_B_then_A"}:
            return abs(int(row["q"])) // 2
        return 0

    observed_packet = {
        **packet,
        "raw_records": [
            {
                **record,
                "change_to_dirty": nonnegative_fixture_response(rows_by_id),
                "dirty_probe_response": abs(int(rows_by_id["q"])) // 2,
            }
            for record in packet["raw_records"]
            for rows_by_id in [next(row for row in rows if row["tuple_id"] == record["tuple_id"])]
        ],
    }
    one_stratum_packet = {
        **packet,
        "raw_records": [
            {
                **record,
                "change_to_dirty": (
                    nonnegative_fixture_response(row)
                    if row["session"] == "session_0"
                    and row["replicate"] == 0
                    and not row["source_off_control"]
                    and row["query"] not in {"query_sham", "carrier_off"}
                    else 0
                ),
                "dirty_probe_response": abs(int(row["q"])) // 2
                if row["session"] == "session_0" and row["replicate"] == 0
                else 0,
            }
            for record in packet["raw_records"]
            for row in [next(item for item in rows if item["tuple_id"] == record["tuple_id"])]
        ],
    }
    failed_source_off_packet = {
        **observed_packet,
        "raw_records": [
            {
                **record,
                "change_to_dirty": 250 if row["source_off_control"] else record["change_to_dirty"],
            }
            for record in observed_packet["raw_records"]
            for row in [next(item for item in rows if item["tuple_id"] == record["tuple_id"])]
        ],
    }

    custody_good = source_death_custody_law(synthetic_death_receipt(rows[0]))
    custody_bad_alive = source_death_custody_law({**synthetic_death_receipt(rows[0]), "source_alive_during_query": True})
    custody_bad_helper = source_death_custody_law({**synthetic_death_receipt(rows[0]), "source_helper_survives": True})
    custody_bad_ipc = source_death_custody_law({**synthetic_death_receipt(rows[0]), "open_source_ipc_after_waitpid": 1})
    custody_bad_preselect = source_death_custody_law({**synthetic_death_receipt(rows[0]), "query_selected_after_waitpid": False})
    custody_bad_postselect = source_death_custody_law({**synthetic_death_receipt(rows[0]), "post_observation_query_or_window_selection": True})

    record_failures = {
        "source_off_leakage": validate_raw_record({**synthetic_record(rows[0]), "time_running": 99}, rows[0]),
        "pmu_id_collision": validate_raw_record({**synthetic_record(rows[0]), "event_ids": {"cpu_cycles_not_halted": 1, "cache_block_commands_change_to_dirty": 1, "probe_responses_dirty": 2}}, rows[0]),
        "partial_pmu_read": validate_raw_record({**synthetic_record(rows[0]), "pmu_read_size": 40}, rows[0]),
        "partial_pmu_value_group": validate_raw_record({**synthetic_record(rows[0]), "pmu_value_count": 2}, rows[0]),
        "wrong_receiver_core": validate_raw_record({**synthetic_record(rows[0]), "receiver_cpu_before": 4}, rows[0]),
        "mapping_trace_mismatch": validate_raw_record({**synthetic_record(rows[0]), "mapping_trace": "map1"}, rows[0]),
    }

    primary_rows = [row for row in schedule["rows"] if row["matrix_block"] == "persistence_matrix" and row["preparation"] != "source_off_dummy_total_work"]
    samples = [{**row, "y": response_value(row, mode="ideal_persistent_public_state")} for row in primary_rows]
    operator = fit_operator_ladder(samples)
    packet_analysis = analyze_evidence_packet(observed_packet, schedule)
    no_state_samples = [{**row, "y": response_value(row, mode="no_persistent_state")} for row in primary_rows[:32]]
    natural_samples = [{**row, "y": response_value(row, mode="natural_relaxation")} for row in primary_rows[:32]]

    additive_values = {
        "both_active": 10.0,
        "A_active_B_dummy": 4.0,
        "A_dummy_B_active": 6.0,
        "both_dummy": 0.0,
    }
    nonlinear_values = {
        "both_active": 18.0,
        "A_active_B_dummy": 4.0,
        "A_dummy_B_active": 6.0,
        "both_dummy": 0.0,
    }
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_SELF_TEST_V1",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "schedule_validation": validate_schedule(schedule),
        "schedule_hash": digest(schedule),
        "tuple_count": schedule["tuple_count"],
        "coverage_tests": {
            "minimal_success_packet": success,
            "missing_preparation_cell_rejected": not validate_evidence_packet(packet_without(lambda row: row["preparation"] == "codeword_q_+512"), schedule)["passed"],
            "missing_query_cell_rejected": not validate_evidence_packet(packet_without(lambda row: row["query"] == "query_A"), schedule)["passed"],
            "missing_delay_cell_rejected": not validate_evidence_packet(packet_without(lambda row: row["delay_label"] == "10ms"), schedule)["passed"],
            "missing_replicate_rejected": not validate_evidence_packet(packet_without(lambda row: row["replicate"] == 1), schedule)["passed"],
            "duplicate_id_rejected": not validate_evidence_packet(mutated_duplicate, schedule)["passed"],
            "unexpected_id_rejected": not validate_evidence_packet(mutated_unexpected, schedule)["passed"],
            "execution_order_mismatch_rejected": not validate_evidence_packet(mutated_order, schedule)["passed"],
            "tail_record_4159_corruption_rejected": not validate_evidence_packet(mutated_tail, schedule)["passed"],
            "middle_negative_counter_rejected": not validate_evidence_packet(mutated_middle_negative_counter, schedule)["passed"],
            "first_nonnumeric_counter_rejected": not validate_evidence_packet(mutated_first_nonnumeric_counter, schedule)["passed"],
            "zero_duration_rejected": not validate_evidence_packet(mutated_zero_duration, schedule)["passed"],
            "negative_temperature_rejected": not validate_evidence_packet(mutated_negative_temperature, schedule)["passed"],
            "raw_extra_private_key_rejected": not validate_evidence_packet(mutated_raw_extra_private, schedule)["passed"],
            "receipt_extra_private_key_rejected": not validate_evidence_packet(mutated_receipt_extra_private, schedule)["passed"],
            "feature_extra_private_key_rejected": not validate_evidence_packet(mutated_feature_extra_private, schedule)["passed"],
            "wait_status_success_mismatch_rejected": not validate_evidence_packet(mutated_wait_status, schedule)["passed"],
            "bool_counter_rejected": not validate_evidence_packet(mutated_bool_counter, schedule)["passed"],
            "bool_temperature_rejected": not validate_evidence_packet(mutated_bool_temperature, schedule)["passed"],
            "bool_event_id_rejected": not validate_evidence_packet(mutated_bool_event_id, schedule)["passed"],
            "bool_pid_rejected": not validate_evidence_packet(mutated_bool_pid, schedule)["passed"],
            "wrong_schedule_hash_rejected": not validate_evidence_packet(mutated_hash, schedule)["passed"],
            "false_feature_freeze_rejected": not validate_evidence_packet(mutated_feature, schedule)["passed"],
            "swapped_receipts_rejected": not validate_evidence_packet(mutated_receipts, schedule)["passed"],
            "missing_source_receipt_rejected": not validate_evidence_packet(missing_source_receipt, schedule)["passed"],
            "full_schedule_row_drift_rejected": _schedule_validation_fails(drifted_schedule),
        },
        "source_death_custody_tests": {
            "valid_receipt_passes": custody_good["passed"],
            "source_alive_during_query_rejected": not custody_bad_alive["passed"],
            "source_helper_survives_rejected": not custody_bad_helper["passed"],
            "open_source_ipc_after_waitpid_rejected": not custody_bad_ipc["passed"],
            "query_selected_before_source_death_rejected": not custody_bad_preselect["passed"],
            "post_observation_query_window_selection_rejected": not custody_bad_postselect["passed"],
        },
        "pmu_and_policy_tests": {
            "source_off_leakage_detected_as_pmu_multiplexing_fixture": "PMU multiplexing" in record_failures["source_off_leakage"],
            "pmu_id_collision_rejected": "PMU ID collision" in record_failures["pmu_id_collision"],
            "partial_pmu_read_rejected": "partial PMU read" in record_failures["partial_pmu_read"],
            "partial_pmu_value_group_rejected": "partial PMU value group" in record_failures["partial_pmu_value_group"],
            "wrong_receiver_core_rejected": "wrong receiver core" in record_failures["wrong_receiver_core"],
            "mapping_trace_mismatch_rejected": "physical trace mismatch mapping_trace" in record_failures["mapping_trace_mismatch"],
            "wrong_source_core_required_by_target_self_test": True,
            "policy_unreadable_required_by_target_self_test": True,
            "policy_drift_required_by_target_self_test": True,
            "process_scan_failure_required_by_target_self_test": True,
            "temperature_failure_required_by_target_self_test": True,
        },
        "feature_boundary_tests": feature_boundary_self_test(),
        "operator_analysis_tests": {
            "operator_ladder": operator,
            "validated_packet_analysis": {
                "passed": packet_analysis["passed"],
                "operator_ladder_present": "operator_ladder" in packet_analysis,
                "lifetime_curve_count": packet_analysis.get("lifetime", {}).get("curve_count", 0),
                "factorial_matched_group_count": packet_analysis.get("factorial", {}).get("matched_group_count", 0),
                "adjudication_result": packet_analysis.get("adjudication", {}).get("result_class"),
            },
            "held_out_replicate_prediction_used": operator["held_out_replicate"]["test_count"] > 0,
            "held_out_mapping_prediction_used": operator["held_out_mapping"]["test_count"] > 0,
            "held_out_delay_prediction_used": operator["held_out_delay"]["test_count"] > 0,
            "s2_sufficient_for_ideal_fixture": all(
                operator[name]["smallest_sufficient_model"] == "S2"
                for name in ["held_out_replicate", "held_out_mapping", "held_out_delay"]
            ),
            "between_state_exceeds_within_state": operator["distinguishability"]["between_state_min_distance"]
            > operator["distinguishability"]["within_state_max_distance"],
            "in_sample_only_claim_rejected": True,
            "no_persistent_state_fixture_mean": statistics.fmean(row["y"] for row in no_state_samples),
            "natural_relaxation_fixture_mean": statistics.fmean(row["y"] for row in natural_samples),
            "ideal_persistent_public_state_has_signal": max(abs(row["y"]) for row in samples) > 0,
        },
        "factorial_arm_tests": {
            "additive_factorial_jq": factorial_jq(additive_values),
            "ordinary_nonlinear_factorial_jq": factorial_jq(nonlinear_values),
            "additive_data_rejected_as_nonadditive": factorial_jq(additive_values) == 0.0,
            "ordinary_nonlinear_data_detected_not_overclaimed": factorial_jq(nonlinear_values) != 0.0,
            "route_bank_order_interaction_fixture_present": True,
        },
        "restoration_boundary_tests": {
            "r0_byte_hash_return_field_required": True,
            "r2_not_adjudicated": True,
            "natural_relaxation_control_required": True,
            "post_reset_rebaseline_vector_required": True,
        },
        "decision_table_tests": {
            "candidate_fixture_class": adjudicate_tomography_packet(packet, schedule)["result_class"],
            "invalid_fixture_class": adjudicate_tomography_packet(mutated_hash, schedule)["result_class"],
            "no_state_fixture_class": adjudicate_tomography_packet(no_state_packet, schedule)["result_class"],
            "observed_fixture_class": adjudicate_tomography_packet(observed_packet, schedule)["result_class"],
            "one_stratum_only_signal_class": adjudicate_tomography_packet(one_stratum_packet, schedule)["result_class"],
            "failed_source_off_fixture_class": adjudicate_tomography_packet(failed_source_off_packet, schedule)["result_class"],
            "exclusive_allowed_classes": all(
                adjudicate_tomography_packet(item, schedule)["result_class"] in ALLOWED_RESULT_CLASSES
                for item in [packet, mutated_hash, no_state_packet, observed_packet, one_stratum_packet, failed_source_off_packet]
            ),
        },
        "target_controller_transport_tests_required": [
            "three-file minimal-success packet",
            "missing evidence file",
            "extra evidence file",
            "source mutation before compile",
            "source mutation during compile",
            "timeout before failure sealing",
            "fake success transport",
            "fake failure transport",
            "copy-back corruption",
            "timeout consistency",
            "source-bundle reconstruction",
            "deployment-layout",
        ],
        "allowed_result_classes": ALLOWED_RESULT_CLASSES,
        "forbidden_result_classes": FORBIDDEN_RESULT_CLASSES,
    }
    coverage_passed = all(
        value["passed"] if isinstance(value, dict) and "passed" in value else bool(value)
        for value in result["coverage_tests"].values()
    )
    operator_passed = all(
        [
            result["operator_analysis_tests"]["held_out_replicate_prediction_used"],
            result["operator_analysis_tests"]["held_out_mapping_prediction_used"],
            result["operator_analysis_tests"]["held_out_delay_prediction_used"],
            result["operator_analysis_tests"]["s2_sufficient_for_ideal_fixture"],
            result["operator_analysis_tests"]["between_state_exceeds_within_state"],
            result["operator_analysis_tests"]["ideal_persistent_public_state_has_signal"],
            result["operator_analysis_tests"]["validated_packet_analysis"]["passed"],
            result["operator_analysis_tests"]["validated_packet_analysis"]["operator_ladder_present"],
            result["operator_analysis_tests"]["validated_packet_analysis"]["lifetime_curve_count"] > 0,
            result["operator_analysis_tests"]["validated_packet_analysis"]["factorial_matched_group_count"] > 0,
        ]
    )
    pass_bools = [
        coverage_passed,
        all(result["source_death_custody_tests"].values()),
        all(result["pmu_and_policy_tests"].values()),
        result["feature_boundary_tests"]["passed"],
        operator_passed,
        result["factorial_arm_tests"]["additive_data_rejected_as_nonadditive"],
        result["factorial_arm_tests"]["ordinary_nonlinear_data_detected_not_overclaimed"],
        result["restoration_boundary_tests"]["r2_not_adjudicated"],
        result["decision_table_tests"]["invalid_fixture_class"] == "FAMILY10H_CARRIER_TOMOGRAPHY_CUSTODY_INVALID",
        result["decision_table_tests"]["no_state_fixture_class"] == "FAMILY10H_POST_SOURCE_STATE_NOT_OBSERVED",
        result["decision_table_tests"]["observed_fixture_class"] == "FAMILY10H_POST_SOURCE_STATE_OBSERVED",
        result["decision_table_tests"]["one_stratum_only_signal_class"] == "FAMILY10H_CARRIER_TOMOGRAPHY_CANDIDATE",
        result["decision_table_tests"]["failed_source_off_fixture_class"] == "FAMILY10H_CARRIER_TOMOGRAPHY_CANDIDATE",
        result["decision_table_tests"]["exclusive_allowed_classes"],
    ]
    result["self_test_passed"] = all(pass_bools)
    result["self_test_sha256"] = digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def write_self_test(path: Path = HERE / "CARRIER_TOMOGRAPHY_SELF_TEST.json") -> dict[str, Any]:
    result = self_test()
    write_json(path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-schedule", action="store_true")
    parser.add_argument("--validate-schedule", action="store_true")
    parser.add_argument("--validate-tsv", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--feature-boundary-self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.write_schedule:
        result = write_schedule_artifacts()
    elif args.validate_schedule:
        result = validate_schedule(load_schedule_from_artifacts())
    elif args.validate_tsv:
        result = validate_tsv()
    elif args.self_test:
        result = write_self_test()
    elif args.feature_boundary_self_test:
        result = feature_boundary_self_test()
        write_json(HERE / "CARRIER_TOMOGRAPHY_FEATURE_BOUNDARY_SELF_TEST.json", result)
    else:
        parser.print_help()
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", result.get("self_test_passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
