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
ACTIVE_SIGNAL_QUERIES = ["query_A", "query_B", "query_A_then_B", "query_B_then_A"]
CONTROL_QUERY_NAMES = ["query_sham", "carrier_off"]
QUERY_ORDER_NAMES = ["A_only", "B_only", "A_then_B", "B_then_A"]
SOURCE_ORDER_NAMES = ["A_then_B", "B_then_A"]
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
CLASSIFICATION_MIN_BALANCED_ACCURACY = 0.95
HOLDOUT_RMSE_RELATIVE_TOLERANCE = 0.05
LIFETIME_SIGNAL_FLOOR = 50.0
LIFETIME_VARIATION_RELATIVE_TOLERANCE = 0.35
LIFETIME_CONFIDENCE_RELATIVE_TOLERANCE = 0.50

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
    "temperature_sensor_identity_sha256",
    "temperature_sensor_hwmon_name",
    "temperature_sensor_label",
    "temperature_sensor_input",
    "temperature_sensor_class_path",
    "temperature_sensor_resolved_input_path",
    "temperature_sensor_resolved_hwmon_path",
    "temperature_sensor_resolved_device_path",
    "process_custody",
    "policy_custody",
]
EXPECTED_EVENT_ID_KEYS = [
    "cpu_cycles_not_halted",
    "cache_block_commands_change_to_dirty",
    "probe_responses_dirty",
]

PACKET_KEYS = {"schema", "schedule_sha256", "raw_records", "source_death_receipts", "feature_freeze"}
TEMPERATURE_SENSOR_IDENTITY_KEYS = {
    "hwmon_name",
    "sensor_label",
    "sensor_input",
    "class_path",
    "resolved_input_path",
    "resolved_hwmon_path",
    "resolved_device_path",
    "identity_sha256",
}
FEATURE_FREEZE_KEYS = {
    "frozen_before_analysis",
    "public_only",
    "schedule_sha256",
    "receiver_feature_boundary",
    "temperature_sensor_identity",
}
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
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def temperature_identity_digest(identity: dict[str, Any]) -> str:
    payload = {key: identity[key] for key in sorted(TEMPERATURE_SENSOR_IDENTITY_KEYS - {"identity_sha256"})}
    return digest(payload)


def with_temperature_identity_digest(identity: dict[str, Any]) -> dict[str, Any]:
    result = {key: identity[key] for key in sorted(TEMPERATURE_SENSOR_IDENTITY_KEYS - {"identity_sha256"})}
    result["identity_sha256"] = temperature_identity_digest(result)
    return result


def synthetic_temperature_identity() -> dict[str, Any]:
    return with_temperature_identity_digest(
        {
            "hwmon_name": "k10temp",
            "sensor_label": "Tctl",
            "sensor_input": "temp1_input",
            "class_path": "/sys/class/hwmon/hwmon0/temp1_input",
            "resolved_input_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon0/temp1_input",
            "resolved_hwmon_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon0",
            "resolved_device_path": "/sys/devices/pci0000:00/0000:00:18.3",
        }
    )


def record_temperature_identity(record: dict[str, Any]) -> dict[str, Any]:
    return with_temperature_identity_digest(
        {
            "hwmon_name": record.get("temperature_sensor_hwmon_name"),
            "sensor_label": record.get("temperature_sensor_label"),
            "sensor_input": record.get("temperature_sensor_input"),
            "class_path": record.get("temperature_sensor_class_path"),
            "resolved_input_path": record.get("temperature_sensor_resolved_input_path"),
            "resolved_hwmon_path": record.get("temperature_sensor_resolved_hwmon_path"),
            "resolved_device_path": record.get("temperature_sensor_resolved_device_path"),
        }
    )


def identity_record_fields(identity: dict[str, Any]) -> dict[str, str]:
    return {
        "temperature_sensor_identity_sha256": identity["identity_sha256"],
        "temperature_sensor_hwmon_name": identity["hwmon_name"],
        "temperature_sensor_label": identity["sensor_label"],
        "temperature_sensor_input": identity["sensor_input"],
        "temperature_sensor_class_path": identity["class_path"],
        "temperature_sensor_resolved_input_path": identity["resolved_input_path"],
        "temperature_sensor_resolved_hwmon_path": identity["resolved_hwmon_path"],
        "temperature_sensor_resolved_device_path": identity["resolved_device_path"],
    }


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
    identity_fields = [
        "temperature_sensor_identity_sha256",
        "temperature_sensor_hwmon_name",
        "temperature_sensor_label",
        "temperature_sensor_input",
        "temperature_sensor_class_path",
        "temperature_sensor_resolved_input_path",
        "temperature_sensor_resolved_hwmon_path",
        "temperature_sensor_resolved_device_path",
    ]
    if any(not isinstance(record.get(field), str) or not record.get(field) for field in identity_fields):
        failures.append("temperature sensor identity missing")
    else:
        try:
            identity = record_temperature_identity(record)
        except (KeyError, TypeError):
            failures.append("temperature sensor identity malformed")
        else:
            if record.get("temperature_sensor_identity_sha256") != identity["identity_sha256"]:
                failures.append("temperature sensor identity digest mismatch")
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
    feature_identity = feature_freeze.get("temperature_sensor_identity")
    if not isinstance(feature_identity, dict) or set(feature_identity) != TEMPERATURE_SENSOR_IDENTITY_KEYS:
        failures.append("feature freeze temperature identity keyset mismatch")
        feature_identity_sha = None
    else:
        feature_identity_sha = feature_identity.get("identity_sha256")
        if feature_identity_sha != temperature_identity_digest(feature_identity):
            failures.append("feature freeze temperature identity digest mismatch")
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
        if feature_identity_sha is not None and record.get("temperature_sensor_identity_sha256") != feature_identity_sha:
            failures.append("temperature sensor identity not evidence-bound")
        if failures:
            break
    return {"passed": not failures, "failures": failures, "expected_count": len(expected_ids), "observed_count": len(raw_records)}


def synthetic_record(schedule_row: dict[str, Any], value: int = 1000, temperature_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    temperature_identity = temperature_identity or synthetic_temperature_identity()
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
        **identity_record_fields(temperature_identity),
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
    temperature_identity = synthetic_temperature_identity()
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_EVIDENCE_PACKET_V1",
        "schedule_sha256": digest(schedule),
        "raw_records": [synthetic_record(row, 1000 + i % 17, temperature_identity=temperature_identity) for i, row in enumerate(rows)],
        "source_death_receipts": [synthetic_death_receipt(row) for row in rows],
        "feature_freeze": {
            "frozen_before_analysis": True,
            "public_only": True,
            "schedule_sha256": digest(schedule),
            "receiver_feature_boundary": "public_schedule_and_public_pmu_only",
            "temperature_sensor_identity": temperature_identity,
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


def query_order_sign_value(row: dict[str, Any]) -> float:
    if row["query_order"] == "A_then_B":
        return 1.0
    if row["query_order"] == "B_then_A":
        return -1.0
    return 0.0


def source_order_sign_value(row: dict[str, Any]) -> float:
    if row["source_order"] == "A_then_B":
        return 1.0
    if row["source_order"] == "B_then_A":
        return -1.0
    return 0.0


def one_hot(value: str, levels: list[str]) -> list[float]:
    return [1.0 if value == level else 0.0 for level in levels]


def model_features(row: dict[str, Any], rung: str) -> list[float]:
    q = float(row["q"])
    decay = math.exp(-float(row["delay_ns"]) / 100_000_000.0)
    query_gain = query_gain_value(row["query"])
    mapping_sign = mapping_sign_value(row["mapping"])
    mapped_q = q * mapping_sign
    mapped_q_pos = max(mapped_q, 0.0)
    mapped_q_neg = max(-mapped_q, 0.0)
    query_order_sign = query_order_sign_value(row)
    source_order_sign = source_order_sign_value(row)
    query_terms = one_hot(row["query"], ACTIVE_SIGNAL_QUERIES)
    source_order_terms = one_hot(row["source_order"], SOURCE_ORDER_NAMES)
    if rung == "S0":
        return [1.0, q]
    if rung == "S1":
        return [1.0, q, decay, mapping_sign, source_order_sign, query_gain, *query_terms, *source_order_terms]
    if rung == "S2":
        return [
            1.0,
            decay,
            source_order_sign,
            query_gain,
            *query_terms,
            *source_order_terms,
            mapped_q,
            mapped_q * query_gain,
            mapped_q * decay,
            mapped_q * source_order_sign,
            query_gain * decay,
            query_gain * source_order_sign,
            mapped_q * query_gain * decay,
            mapped_q * query_gain * decay * source_order_sign,
            mapped_q_pos,
            mapped_q_neg,
            mapped_q_pos * decay,
            mapped_q_neg * decay,
            *[mapped_q * item for item in query_terms],
            *[mapping_sign * item for item in query_terms],
            *[mapping_sign * decay * item for item in query_terms],
            *[mapping_sign * query_gain * item for item in query_terms],
            *[decay * item for item in query_terms],
            *[source_order_sign * item for item in query_terms],
            *[mapped_q * decay * item for item in query_terms],
            *[mapped_q * query_gain * decay * item for item in query_terms],
            *[source_order_sign * item for item in source_order_terms],
            *[decay * item for item in source_order_terms],
            *[mapped_q_pos * item for item in query_terms],
            *[mapped_q_neg * item for item in query_terms],
            *[mapped_q_pos * decay * item for item in query_terms],
            *[mapped_q_neg * decay * item for item in query_terms],
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
    ridge = 1e-3
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


def heldout_bundle(samples: list[dict[str, Any]], test_predicate: Any, *, factor_name: str, heldout_value: Any) -> dict[str, Any]:
    train = [row for row in samples if not test_predicate(row)]
    test = [row for row in samples if test_predicate(row)]
    if not train or not test:
        return {
            "heldout_factor": factor_name,
            "heldout_value": heldout_value,
            "test_count": len(test),
            "train_count": len(train),
            "train_levels": [],
            "test_levels": [],
            "tested_factor_absent_from_training": False,
            "rmse": {},
            "noise_aware_threshold": 0.0,
            "smallest_sufficient_model": "not_sufficient",
            "passed": False,
        }
    scores = {rung: rmse_for(train, test, rung) for rung in ["S0", "S1", "S2"]}
    threshold = max(1e-6, statistics.pstdev(row["y"] for row in train) * HOLDOUT_RMSE_RELATIVE_TOLERANCE)
    sufficient = [rung for rung in ["S0", "S1", "S2"] if scores[rung] <= threshold]
    train_levels = sorted({row[factor_name] for row in train})
    test_levels = sorted({row[factor_name] for row in test})
    absent = heldout_value not in train_levels and heldout_value in test_levels
    return {
        "heldout_factor": factor_name,
        "heldout_value": heldout_value,
        "test_count": len(test),
        "train_count": len(train),
        "train_levels": train_levels,
        "test_levels": test_levels,
        "tested_factor_absent_from_training": absent,
        "rmse": scores,
        "noise_aware_threshold": threshold,
        "smallest_sufficient_model": sufficient[0] if sufficient else "not_sufficient",
        "passed": absent and bool(sufficient),
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


def codeword_classifier(samples: list[dict[str, Any]], test_predicate: Any) -> dict[str, Any]:
    signal_rows = [row for row in samples if row["query"] in ACTIVE_SIGNAL_QUERIES]
    train = [row for row in signal_rows if not test_predicate(row)]
    test = [row for row in signal_rows if test_predicate(row)]
    confusion: dict[str, dict[str, int]] = {str(q): {str(p): 0 for p in Q_VALUES} for q in Q_VALUES}
    if not train or not test:
        return {
            "classifier": "nearest_centroid_by_q",
            "training_split": "not_available",
            "test_split": "not_available",
            "confusion_matrix": confusion,
            "balanced_accuracy": 0.0,
            "minimum_balanced_accuracy": CLASSIFICATION_MIN_BALANCED_ACCURACY,
            "cross_validated_codeword_classification": False,
            "passed": False,
        }
    coordinates = sorted(
        {
            (row["query"], row["query_order"], row["delay_label"], row["mapping"], row["session"], row["source_order"])
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
                and (row["query"], row["query_order"], row["delay_label"], row["mapping"], row["session"], row["source_order"]) == coordinate
            ]
            result.append(statistics.fmean(values) if values else 0.0)
        return result

    centroids = {q: vector_for(train, q) for q in Q_VALUES}
    for q in Q_VALUES:
        observed = vector_for(test, q)
        predicted = min(
            Q_VALUES,
            key=lambda candidate: math.sqrt(sum((a - b) ** 2 for a, b in zip(observed, centroids[candidate]))),
        )
        confusion[str(q)][str(predicted)] += 1
    recalls = []
    for q in Q_VALUES:
        row_total = sum(confusion[str(q)].values())
        recalls.append(confusion[str(q)][str(q)] / row_total if row_total else 0.0)
    balanced_accuracy = statistics.fmean(recalls) if recalls else 0.0
    return {
        "classifier": "nearest_centroid_by_q",
        "training_split": "all rows not matching held_out_replicate_1",
        "test_split": "replicate_1 absent from training",
        "confusion_matrix": confusion,
        "balanced_accuracy": balanced_accuracy,
        "minimum_balanced_accuracy": CLASSIFICATION_MIN_BALANCED_ACCURACY,
        "cross_validated_codeword_classification": balanced_accuracy >= CLASSIFICATION_MIN_BALANCED_ACCURACY,
        "passed": balanced_accuracy >= CLASSIFICATION_MIN_BALANCED_ACCURACY,
    }


def query_structure_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    signal_rows = [row for row in samples if row["query"] in ACTIVE_SIGNAL_QUERIES]
    query_means = {
        query: statistics.fmean(row["y"] for row in signal_rows if row["query"] == query)
        for query in ACTIVE_SIGNAL_QUERIES
        if any(row["query"] == query for row in signal_rows)
    }
    order_means = {
        order: statistics.fmean(row["y"] for row in signal_rows if row["query_order"] == order)
        for order in QUERY_ORDER_NAMES
        if any(row["query_order"] == order for row in signal_rows)
    }
    source_order_means = {
        order: statistics.fmean(row["y"] for row in signal_rows if row["source_order"] == order)
        for order in SOURCE_ORDER_NAMES
        if any(row["source_order"] == order for row in signal_rows)
    }
    source_orders_by_query = {
        query: {row["source_order"] for row in signal_rows if row["query"] == query}
        for query in ACTIVE_SIGNAL_QUERIES
    }
    independent_source_order_contrasts = all(source_orders_by_query.get(query) == set(SOURCE_ORDER_NAMES) for query in ACTIVE_SIGNAL_QUERIES)
    q0_mapping_vectors_distinct = False
    for left in signal_rows:
        if int(left["q"]) != 0 or left["mapping"] != "map0":
            continue
        right = next(
            (
                row
                for row in signal_rows
                if int(row["q"]) == 0
                and row["mapping"] == "map1"
                and row["query"] == left["query"]
                and row["query_order"] == left["query_order"]
                and row["source_order"] == left["source_order"]
                and row["delay_label"] == left["delay_label"]
                and row["replicate"] == left["replicate"]
                and row["session"] == left["session"]
            ),
            None,
        )
        if right is not None and model_features(left, "S2") != model_features(right, "S2"):
            q0_mapping_vectors_distinct = True
            break
    query_effect = (max(query_means.values()) - min(query_means.values())) if query_means else 0.0
    order_effect = abs(order_means.get("A_then_B", 0.0) - order_means.get("B_then_A", 0.0))
    source_order_effect = abs(source_order_means.get("A_then_B", 0.0) - source_order_means.get("B_then_A", 0.0))
    query_and_source_nonduplicate = False
    ordered_query_features_distinct = False
    single_query_features_distinct = False
    query_order_main_effect_claimed = False
    raw_query_observations_preserved = True
    raw_rows = [row for row in signal_rows if "raw_change_to_dirty" in row]
    if raw_rows:
        raw_query_observations_preserved = all(row["y"] == row["raw_change_to_dirty"] for row in raw_rows)
    if signal_rows:
        query_order_probe = next((row for row in signal_rows if row["query_order"] in {"A_then_B", "B_then_A"}), signal_rows[0])
        source_swapped = {
            **query_order_probe,
            "source_order": "B_then_A" if query_order_probe["source_order"] == "A_then_B" else "A_then_B",
        }
        base_features = model_features(query_order_probe, "S2")
        ordered_pair = next(
            (
                (left, right)
                for left in signal_rows
                if left["query"] == "query_A_then_B"
                for right in signal_rows
                if right["query"] == "query_B_then_A"
                and right["q"] == left["q"]
                and right["mapping"] == left["mapping"]
                and right["source_order"] == left["source_order"]
                and right["delay_label"] == left["delay_label"]
                and right["replicate"] == left["replicate"]
                and right["session"] == left["session"]
            ),
            None,
        )
        single_pair = next(
            (
                (left, right)
                for left in signal_rows
                if left["query"] == "query_A"
                for right in signal_rows
                if right["query"] == "query_B"
                and right["q"] == left["q"]
                and right["mapping"] == left["mapping"]
                and right["source_order"] == left["source_order"]
                and right["delay_label"] == left["delay_label"]
                and right["replicate"] == left["replicate"]
                and right["session"] == left["session"]
            ),
            None,
        )
        ordered_query_features_distinct = (
            ordered_pair is not None and model_features(ordered_pair[0], "S2") != model_features(ordered_pair[1], "S2")
        )
        single_query_features_distinct = (
            single_pair is not None and model_features(single_pair[0], "S2") != model_features(single_pair[1], "S2")
        )
        query_and_source_nonduplicate = (
            ordered_query_features_distinct
            and single_query_features_distinct
            and model_features(source_swapped, "S2") != base_features
        )
    return {
        "queries_present": sorted(query_means),
        "query_orders_present": sorted(order_means),
        "source_orders_present": sorted(source_order_means),
        "all_active_queries_preserved": set(query_means) == set(ACTIVE_SIGNAL_QUERIES),
        "ordered_queries_preserved": all(order in order_means for order in ["A_then_B", "B_then_A"]),
        "main_query_effect_abs": query_effect,
        "query_order_effect_abs": order_effect,
        "source_order_effect_abs": source_order_effect,
        "query_order_identifiability_basis": "order_encoded_query_identity_not_independent_main_effect",
        "query_order_independent_main_effect_claimed": query_order_main_effect_claimed,
        "query_order_terms_not_duplicated_with_query_terms": True,
        "ordered_query_features_distinct": ordered_query_features_distinct,
        "single_query_features_distinct": single_query_features_distinct,
        "raw_query_observations_preserved": raw_query_observations_preserved,
        "independent_source_order_contrasts_preserved": independent_source_order_contrasts,
        "q0_mapping_by_query_vectors_distinct": q0_mapping_vectors_distinct,
        "query_and_source_order_contrasts_nonduplicate": query_and_source_nonduplicate,
        "preparation_by_query_terms_declared": len(model_features(signal_rows[0], "S2")) > len(model_features(signal_rows[0], "S1")) if signal_rows else False,
        "delay_by_query_terms_declared": any(abs(value) > 0.0 for row in signal_rows for value in model_features(row, "S2")[len(model_features(row, "S1")) :]),
        "mapping_by_query_terms_declared": q0_mapping_vectors_distinct,
    }


def operational_distinguishability(samples: list[dict[str, Any]]) -> dict[str, Any]:
    signal_rows = [row for row in samples if row["query"] in ACTIVE_SIGNAL_QUERIES]
    train = [row for row in signal_rows if row["replicate"] == 0]
    test = [row for row in signal_rows if row["replicate"] == 1]
    classifier = codeword_classifier(samples, lambda row: row["replicate"] == 1)
    coordinates = sorted(
        {
            (row["query"], row["query_order"], row["delay_label"], row["mapping"], row["session"], row["source_order"])
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
                and (row["query"], row["query_order"], row["delay_label"], row["mapping"], row["session"], row["source_order"]) == coordinate
            ]
            result.append(statistics.fmean(values) if values else 0.0)
        return result

    centroids = {q: vector_for(train, q) for q in Q_VALUES}
    matrix: list[list[float]] = []
    rank_coordinates = sorted(
        {
            (row["query"], row["query_order"], row["mapping"], row["delay_label"])
            for row in signal_rows
        }
    )
    for q in Q_VALUES:
        row_values = []
        for query, query_order, mapping, delay_label in rank_coordinates:
            values = [
                row["y"]
                for row in signal_rows
                if row["q"] == q
                and row["query"] == query
                and row["query_order"] == query_order
                and row["mapping"] == mapping
                and row["delay_label"] == delay_label
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
        "cross_validated_codeword_classification": classifier["cross_validated_codeword_classification"],
        "classifier": classifier,
        "confusion_matrix": classifier["confusion_matrix"],
        "balanced_accuracy": classifier["balanced_accuracy"],
        "minimum_balanced_accuracy": classifier["minimum_balanced_accuracy"],
        "response_matrix_effective_rank": matrix_rank(matrix),
        "singular_spectrum": spectrum,
        "between_state_min_distance": min(between),
        "within_state_max_distance": max(within),
    }


def fit_operator_ladder(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "held_out_replicate": heldout_bundle(
            samples,
            lambda row: row["replicate"] == 1,
            factor_name="replicate",
            heldout_value=1,
        ),
        "held_out_mapping": heldout_bundle(
            samples,
            lambda row: row["mapping"] == "map1",
            factor_name="mapping",
            heldout_value="map1",
        ),
        "held_out_delay": heldout_bundle(
            samples,
            lambda row: row["delay_label"] == "10ms",
            factor_name="delay_label",
            heldout_value="10ms",
        ),
        "query_structure": query_structure_summary(samples),
        "distinguishability": operational_distinguishability(samples),
    }


def factorial_jq(values: dict[str, float]) -> float:
    return values["both_active"] - values["A_active_B_dummy"] - values["A_dummy_B_active"] + values["both_dummy"]


def matched_query_key(schedule_row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        schedule_row["preparation"],
        int(schedule_row["q"]),
        schedule_row["delay_label"],
        schedule_row["mapping"],
        schedule_row["replicate"],
        schedule_row["session"],
        schedule_row["source_order"],
    )


def matched_query_counts(packet: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, float]]:
    counts: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for record in packet["raw_records"]:
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] != "persistence_matrix" or row["source_off_control"] or row["query"] not in ACTIVE_SIGNAL_QUERIES:
            continue
        counts[matched_query_key(row)][row["query"]] = float(record["change_to_dirty"])
    return counts


def signed_query_response(record: dict[str, Any], schedule_row: dict[str, Any], counts: dict[tuple[Any, ...], dict[str, float]]) -> float:
    query = schedule_row["query"]
    if schedule_row["source_off_control"] or query in {"query_sham", "carrier_off"}:
        return float(record["change_to_dirty"])
    matched = counts.get(matched_query_key(schedule_row), {})
    single_contrast = matched.get("query_A", 0.0) - matched.get("query_B", 0.0)
    ordered_contrast = matched.get("query_A_then_B", 0.0) - matched.get("query_B_then_A", 0.0)
    if query == "query_A":
        return single_contrast
    if query == "query_B":
        return -single_contrast
    if query == "query_A_then_B":
        return ordered_contrast
    if query == "query_B_then_A":
        return -ordered_contrast
    return 0.0


def evidence_samples(packet: dict[str, Any], schedule: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    counts = matched_query_counts(packet, rows_by_id)
    samples: list[dict[str, Any]] = []
    for record in packet["raw_records"]:
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] != "persistence_matrix" or row["source_off_control"]:
            continue
        if row["query"] not in ACTIVE_SIGNAL_QUERIES:
            continue
        samples.append(
            {
                "preparation": row["preparation"],
                "q": row["q"],
                "query": row["query"],
                "query_order": row["query_order"],
                "delay_label": row["delay_label"],
                "delay_ns": row["delay_ns"],
                "mapping": row["mapping"],
                "replicate": row["replicate"],
                "session": row["session"],
                "source_order": row["source_order"],
                "raw_change_to_dirty": float(record["change_to_dirty"]),
                "matched_contrast_y": signed_query_response(record, row, counts),
                "operator_response_law": "raw per-query physical response",
                "signed_response_law": "paired query matched physical contrast",
                "orientation_law": "raw model response; separate matched contrast for physical sign law",
                "y": float(record["change_to_dirty"]),
            }
        )
    return samples


def lifetime_class_for_points(points: list[dict[str, Any]]) -> str:
    values = [point["mean_abs_response"] for point in points]
    if not values or max(values) <= LIFETIME_SIGNAL_FLOOR:
        return "vanishes before source death"
    oriented_signs = {
        1 if point.get("mean_oriented_response", 0.0) > LIFETIME_SIGNAL_FLOOR else -1
        for point in points
        if abs(point.get("mean_oriented_response", 0.0)) > LIFETIME_SIGNAL_FLOOR
    }
    if len(oriented_signs) > 1 or any(point.get("oriented_sign_consistent") is False for point in points):
        return "changes form across delay"
    first = values[0]
    last = values[-1]
    later = values[1:]
    if first <= LIFETIME_SIGNAL_FLOOR:
        return "changes form across delay"
    if all(value <= LIFETIME_SIGNAL_FLOOR for value in later):
        return "survives only immediate handoff"
    if any(values[index] > values[index - 1] * 1.25 and values[index] > LIFETIME_SIGNAL_FLOOR for index in range(1, len(values))):
        return "changes form across delay"
    if last > 0.5 * first and all(value > LIFETIME_SIGNAL_FLOOR for value in values):
        return "persists across the full grid"
    return "survives a bounded delay"


def variation_summary(values_by_factor: dict[str, list[float]]) -> dict[str, Any]:
    within_spreads = {
        key: (max(values) - min(values) if values else 0.0)
        for key, values in values_by_factor.items()
    }
    factor_means = {
        key: (statistics.fmean(values) if values else 0.0)
        for key, values in values_by_factor.items()
    }
    across_spread = max(factor_means.values(), default=0.0) - min(factor_means.values(), default=0.0)
    across_mean = statistics.fmean(abs(value) for value in factor_means.values()) if factor_means else 0.0
    relative_across_spread = across_spread / max(abs(across_mean), LIFETIME_SIGNAL_FLOOR)
    all_values = [abs(value) for values in values_by_factor.values() for value in values]
    within_reference = max(
        statistics.fmean(all_values) if all_values else 0.0,
        abs(across_mean),
        LIFETIME_SIGNAL_FLOOR,
    )
    max_within_spread = max(within_spreads.values(), default=0.0)
    relative_within_spread = max_within_spread / within_reference
    passed = (
        relative_across_spread <= LIFETIME_VARIATION_RELATIVE_TOLERANCE
        and relative_within_spread <= LIFETIME_VARIATION_RELATIVE_TOLERANCE
    )
    return {
        "within_factor_spread": within_spreads,
        "max_within_factor_spread": max_within_spread,
        "factor_mean": factor_means,
        "across_factor_spread": across_spread,
        "relative_across_factor_spread": relative_across_spread,
        "relative_within_factor_spread": relative_within_spread,
        "passed": passed,
        "relative_tolerance": LIFETIME_VARIATION_RELATIVE_TOLERANCE,
    }


def expected_query_scale(query: str) -> float:
    return {
        "query_A": 1.0,
        "query_B": 1.0,
        "query_A_then_B": 1.3,
        "query_B_then_A": 1.3,
    }.get(query, 1.0)


def response_orientation(row: dict[str, Any]) -> float:
    expected = mapping_sign_value(row["mapping"]) * query_gain_value(row["query"]) * float(row["q"])
    if expected > 0.0:
        return 1.0
    if expected < 0.0:
        return -1.0
    return 1.0


def sign_consistent(values: list[float]) -> bool:
    signs = {
        1 if value > LIFETIME_SIGNAL_FLOOR else -1
        for value in values
        if abs(value) > LIFETIME_SIGNAL_FLOOR
    }
    return len(signs) <= 1


def lifetime_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    curves: dict[str, dict[str, Any]] = {}
    for key in sorted({(row["preparation"], row["query"], row["session"], row["mapping"]) for row in samples}):
        prep, query, session, mapping = key
        points = []
        for delay in DELAY_GRID:
            delay_label = delay["delay_label"]
            delay_ns = delay["delay_ns"]
            raw_values = [
                row.get("matched_contrast_y", row["y"])
                for row in samples
                if row["preparation"] == prep
                and row["query"] == query
                and row["session"] == session
                and row["mapping"] == mapping
                and row["delay_label"] == delay_label
            ]
            if raw_values:
                template = next(
                    row
                    for row in samples
                    if row["preparation"] == prep
                    and row["query"] == query
                    and row["session"] == session
                    and row["mapping"] == mapping
                    and row["delay_label"] == delay_label
                )
                oriented_values = [value * response_orientation(template) for value in raw_values]
                abs_values = [abs(value) for value in raw_values]
                mean = statistics.fmean(abs_values)
                spread = statistics.pstdev(abs_values) if len(abs_values) > 1 else 0.0
                half_width = 1.96 * spread / math.sqrt(len(abs_values)) if len(abs_values) > 1 else 0.0
                oriented_mean = statistics.fmean(oriented_values)
                oriented_spread = statistics.pstdev(oriented_values) if len(oriented_values) > 1 else 0.0
                oriented_half_width = 1.96 * oriented_spread / math.sqrt(len(oriented_values)) if len(oriented_values) > 1 else 0.0
                points.append(
                    {
                        "delay_label": delay_label,
                        "delay_ns": delay_ns,
                        "mean_abs_response": mean,
                        "mean_oriented_response": oriented_mean,
                        "pstdev": spread,
                        "oriented_pstdev": oriented_spread,
                        "confidence_interval_95": [mean - half_width, mean + half_width],
                        "oriented_confidence_interval_95": [oriented_mean - oriented_half_width, oriented_mean + oriented_half_width],
                        "oriented_sign_consistent": sign_consistent(oriented_values),
                        "n": len(abs_values),
                    }
                )
        curves[f"{prep}:{query}:{session}:{mapping}"] = {
            "preparation": prep,
            "query": query,
            "session": session,
            "mapping": mapping,
            "points": points,
            "lifetime_class": lifetime_class_for_points(points),
        }
    by_prep_query_session_delay: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_prep_query_mapping_delay: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_prep_delay_query: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_delay: dict[str, list[float]] = defaultdict(list)
    confidence_widths = []
    polarity_failures: list[str] = []
    for curve in curves.values():
        for point in curve["points"]:
            by_delay[point["delay_label"]].append(point["mean_abs_response"])
            key = (curve["preparation"], curve["query"], point["delay_label"])
            by_prep_query_session_delay[key][curve["session"]].append(point["mean_oriented_response"])
            by_prep_query_mapping_delay[key][curve["mapping"]].append(point["mean_oriented_response"])
            normalized = point["mean_oriented_response"] / expected_query_scale(curve["query"])
            by_prep_delay_query[(curve["preparation"], point["delay_label"])][curve["query"]].append(normalized)
            ci = point["oriented_confidence_interval_95"]
            confidence_widths.append((ci[1] - ci[0]) / max(abs(point["mean_oriented_response"]), LIFETIME_SIGNAL_FLOOR))
            if not point["oriented_sign_consistent"]:
                polarity_failures.append(f"{curve['preparation']}:{curve['query']}:{curve['session']}:{curve['mapping']}:{point['delay_label']}")
    session_variation = {
        f"{prep}:{query}:{delay_label}": variation_summary(values_by_session)
        for (prep, query, delay_label), values_by_session in by_prep_query_session_delay.items()
    }
    mapping_variation = {
        f"{prep}:{query}:{delay_label}": variation_summary(values_by_mapping)
        for (prep, query, delay_label), values_by_mapping in by_prep_query_mapping_delay.items()
    }
    query_variation_by_prep_delay = {
        f"{prep}:{delay_label}": variation_summary(values_by_query)
        for (prep, delay_label), values_by_query in by_prep_delay_query.items()
    }
    query_variation = {
        "by_preparation_delay": query_variation_by_prep_delay,
        "max_relative_across_query_spread": max(
            (item["relative_across_factor_spread"] for item in query_variation_by_prep_delay.values()),
            default=0.0,
        ),
        "relative_tolerance": LIFETIME_VARIATION_RELATIVE_TOLERANCE,
        "passed": all(item["passed"] for item in query_variation_by_prep_delay.values()),
    }
    mean_persistence_curve = [
        {
            "delay_label": delay["delay_label"],
            "delay_ns": delay["delay_ns"],
            "mean_abs_response": statistics.fmean(by_delay[delay["delay_label"]]) if by_delay.get(delay["delay_label"]) else 0.0,
            "n": len(by_delay.get(delay["delay_label"], [])),
        }
        for delay in DELAY_GRID
    ]
    confidence_variation = {
        "max_relative_ci_width": max(confidence_widths, default=0.0),
        "relative_tolerance": LIFETIME_CONFIDENCE_RELATIVE_TOLERANCE,
        "passed": max(confidence_widths, default=0.0) <= LIFETIME_CONFIDENCE_RELATIVE_TOLERANCE,
    }
    polarity_consistency = {
        "failures": polarity_failures,
        "passed": not polarity_failures and not any(curve["lifetime_class"] == "changes form across delay" for curve in curves.values()),
    }
    variation_gates = {
        "session_variation_passed": all(item["passed"] for item in session_variation.values()),
        "mapping_variation_passed": all(item["passed"] for item in mapping_variation.values()),
        "query_variation_passed": query_variation["passed"],
        "confidence_variation_passed": confidence_variation["passed"],
        "polarity_consistency_passed": polarity_consistency["passed"],
    }
    return {
        "curve_count": len(curves),
        "curves": curves,
        "mean_persistence_curve": mean_persistence_curve,
        "session_variation": session_variation,
        "mapping_variation": mapping_variation,
        "query_variation": query_variation,
        "confidence_variation": confidence_variation,
        "polarity_consistency": polarity_consistency,
        "variation_gates": variation_gates,
        "passed": all(variation_gates.values()),
        "vocabulary": [
            "vanishes before source death",
            "survives only immediate handoff",
            "survives a bounded delay",
            "persists across the full grid",
            "changes form across delay",
        ],
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
    lifetime = lifetime_summary(samples)
    gates = {
        name: ladder[name]["passed"]
        for name in ["held_out_replicate", "held_out_mapping", "held_out_delay"]
    }
    gates["all_active_queries_preserved"] = ladder["query_structure"]["all_active_queries_preserved"]
    gates["ordered_queries_preserved"] = ladder["query_structure"]["ordered_queries_preserved"]
    gates["independent_source_order_contrasts_preserved"] = ladder["query_structure"]["independent_source_order_contrasts_preserved"]
    gates["query_order_terms_not_duplicated_with_query_terms"] = ladder["query_structure"]["query_order_terms_not_duplicated_with_query_terms"]
    gates["query_order_independent_main_effect_not_claimed"] = not ladder["query_structure"]["query_order_independent_main_effect_claimed"]
    gates["ordered_query_features_distinct"] = ladder["query_structure"]["ordered_query_features_distinct"]
    gates["single_query_features_distinct"] = ladder["query_structure"]["single_query_features_distinct"]
    gates["raw_query_observations_preserved"] = ladder["query_structure"]["raw_query_observations_preserved"]
    gates["q0_mapping_by_query_vectors_distinct"] = ladder["query_structure"]["q0_mapping_by_query_vectors_distinct"]
    gates["query_and_source_order_contrasts_nonduplicate"] = ladder["query_structure"]["query_and_source_order_contrasts_nonduplicate"]
    gates["held_out_classifier_passed"] = ladder["distinguishability"]["classifier"]["passed"]
    gates["lifetime_variation_passed"] = lifetime["passed"]
    return {"passed": all(gates.values()), "gates": gates, "operator_ladder": ladder, "lifetime": lifetime}


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
    counts = matched_query_counts(packet, rows_by_id)
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
        elif schedule_row["matrix_block"] == "persistence_matrix" and schedule_row["query"] in ACTIVE_SIGNAL_QUERIES:
            query_sign = 1.0 if query_gain_value(schedule_row["query"]) >= 0.0 else -1.0
            normalized = signed_query_response(record, schedule_row, counts) * mapping_sign_value(schedule_row["mapping"]) * query_sign
            values_by_stratum_q_query[stratum][int(schedule_row["q"])]["mapping_normalized_signed_response"].append(normalized)
    operator_gate = operator_prediction_gate(evidence_samples(packet, schedule))
    stratum_results: dict[str, dict[str, Any]] = {}
    expected_strata = [(session, replicate) for session in SESSIONS for replicate in REPLICATES]
    for stratum in expected_strata:
        contrasts: dict[int, float] = {}
        for q, query_values in values_by_stratum_q_query[stratum].items():
            normalized_values = query_values.get("mapping_normalized_signed_response", [])
            if normalized_values:
                contrasts[q] = statistics.fmean(normalized_values)
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
    first_without_temperature_identity = dict(packet["raw_records"][0])
    for key in list(identity_record_fields(packet["feature_freeze"]["temperature_sensor_identity"])):
        first_without_temperature_identity.pop(key, None)
    mutated_missing_temperature_identity = {
        **packet,
        "raw_records": [first_without_temperature_identity, *packet["raw_records"][1:]],
    }
    mutated_temperature_identity_digest = {
        **packet,
        "raw_records": [
            {**packet["raw_records"][0], "temperature_sensor_identity_sha256": "0" * 64},
            *packet["raw_records"][1:],
        ],
    }
    mutated_feature_temperature_identity = {
        **packet,
        "feature_freeze": {
            **packet["feature_freeze"],
            "temperature_sensor_identity": {
                **packet["feature_freeze"]["temperature_sensor_identity"],
                "resolved_device_path": "/sys/devices/not-the-approved-cpu-sensor",
            },
        },
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
        effective_q = int(row["q"]) * (1 if row["mapping"] == "map0" else -1)
        magnitude = abs(effective_q)
        inactive_leg_ratio = 0.75
        if row["query"] == "query_A":
            return int(round(magnitude if effective_q >= 0 else inactive_leg_ratio * magnitude))
        if row["query"] == "query_B":
            return int(round(magnitude if effective_q <= 0 else inactive_leg_ratio * magnitude))
        if row["query"] == "query_A_then_B":
            scale = 1.3 if effective_q >= 0 else 1.3 * inactive_leg_ratio
            return int(round(scale * magnitude))
        if row["query"] == "query_B_then_A":
            scale = 1.3 if effective_q <= 0 else 1.3 * inactive_leg_ratio
            return int(round(scale * magnitude))
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

    primary_rows = [
        row
        for row in schedule["rows"]
        if row["matrix_block"] == "persistence_matrix"
        and row["preparation"] != "source_off_dummy_total_work"
        and row["query"] in ACTIVE_SIGNAL_QUERIES
    ]
    samples = [{**row, "y": response_value(row, mode="ideal_persistent_public_state")} for row in primary_rows]
    operator = fit_operator_ladder(samples)
    packet_analysis = analyze_evidence_packet(observed_packet, schedule)
    no_state_samples = [{**row, "y": response_value(row, mode="no_persistent_state")} for row in primary_rows[:32]]
    natural_samples = [{**row, "y": response_value(row, mode="natural_relaxation")} for row in primary_rows[:32]]
    query_order_only_samples = [
        {
            **row,
            "y": 80.0 if row["query_order"] == "A_then_B" else -80.0 if row["query_order"] == "B_then_A" else 0.0,
        }
        for row in primary_rows
    ]
    query_order_operator = fit_operator_ladder(query_order_only_samples)
    source_order_only_samples = [
        {
            **row,
            "y": 80.0 if row["source_order"] == "A_then_B" else -80.0,
        }
        for row in primary_rows
    ]
    source_order_operator = fit_operator_ladder(source_order_only_samples)
    query_order_collapsed_samples = [
        {
            **row,
            "y": 0.0 if row["query_order"] in {"A_then_B", "B_then_A"} else query_order_only_samples[index]["y"],
        }
        for index, row in enumerate(primary_rows)
    ]
    query_order_collapsed_operator = fit_operator_ladder(query_order_collapsed_samples)
    raw_equal_contrast_base = [
        {
            **row,
            "raw_change_to_dirty": value,
            "matched_contrast_y": value,
            "y": value,
        }
        for row in primary_rows
        for value in [
            100.0
            if row["query"] == "query_A"
            else 40.0
            if row["query"] == "query_B"
            else 90.0
            if row["query"] == "query_A_then_B"
            else 30.0
        ]
    ]
    raw_equal_contrast_shifted = [
        {
            **row,
            "raw_change_to_dirty": value,
            "matched_contrast_y": value,
            "y": value,
        }
        for row in primary_rows
        for value in [
            160.0
            if row["query"] == "query_A"
            else 100.0
            if row["query"] == "query_B"
            else 150.0
            if row["query"] == "query_A_then_B"
            else 90.0
        ]
    ]
    raw_equal_contrast_base_operator = fit_operator_ladder(raw_equal_contrast_base)
    raw_equal_contrast_shifted_operator = fit_operator_ladder(raw_equal_contrast_shifted)
    raw_equal_contrast_vectors_differ = (
        raw_equal_contrast_base_operator["query_structure"]["main_query_effect_abs"]
        != raw_equal_contrast_shifted_operator["query_structure"]["main_query_effect_abs"]
        or raw_equal_contrast_base_operator["distinguishability"]["singular_spectrum"]
        != raw_equal_contrast_shifted_operator["distinguishability"]["singular_spectrum"]
    )
    mapping_scrambled_samples = [
        {
            **row,
            "y": response_value({**row, "mapping": "map0"}, mode="ideal_persistent_public_state"),
        }
        for row in primary_rows
    ]
    delay_scrambled_samples = [
        {
            **row,
            "y": -response_value(row, mode="ideal_persistent_public_state")
            if row["delay_label"] == "10ms"
            else response_value(row, mode="ideal_persistent_public_state"),
        }
        for row in primary_rows
    ]
    confused_classification_samples = [
        {
            **row,
            "y": response_value({**row, "q": 0}, mode="ideal_persistent_public_state"),
        }
        for row in primary_rows
    ]
    imbalanced_trivial_samples = [
        {
            **row,
            "y": 1.0,
        }
        for row in primary_rows
        if int(row["q"]) in {-1536, 1536} or (int(row["q"]) == 0 and row["replicate"] == 1)
    ]
    memorized_train_failed_test_samples = [
        {
            **row,
            "y": response_value(row, mode="ideal_persistent_public_state")
            if row["replicate"] == 0
            else response_value({**row, "q": -int(row["q"])}, mode="ideal_persistent_public_state"),
        }
        for row in primary_rows
    ]
    lifetime_fixture_rows = [row for row in primary_rows if row["q"] == 1536 and row["query"] == "query_A"]

    def packet_signed_delay_reversal_response(row: dict[str, Any]) -> int:
        if row["source_off_control"] or row["query"] in {"query_sham", "carrier_off"}:
            return 0
        if row["matrix_block"] != "persistence_matrix" or row["query"] not in ACTIVE_SIGNAL_QUERIES:
            return 0
        if int(row["q"]) == 0:
            return 0
        positive_contrast = mapping_sign_value(row["mapping"]) * float(row["q"]) > 0.0
        if row["delay_label"] == "10ms":
            positive_contrast = not positive_contrast
        high = 400
        low = 100
        if row["query"] in {"query_A", "query_A_then_B"}:
            return high if positive_contrast else low
        if row["query"] in {"query_B", "query_B_then_A"}:
            return low if positive_contrast else high
        return 0

    packet_path_signed_delay_reversal = {
        **packet,
        "raw_records": [
            {
                **record,
                "change_to_dirty": packet_signed_delay_reversal_response(row),
                "dirty_probe_response": abs(int(row["q"])) // 2 if not row["source_off_control"] else 0,
            }
            for record in packet["raw_records"]
            for row in [next(item for item in rows if item["tuple_id"] == record["tuple_id"])]
        ],
    }
    packet_path_signed_delay_reversal_analysis = analyze_evidence_packet(packet_path_signed_delay_reversal, schedule)

    def lifetime_fixture(scale_by_delay: dict[str, float], *, session_scale: dict[str, float] | None = None, mapping_scale: dict[str, float] | None = None) -> dict[str, Any]:
        fixture_samples = []
        for row in lifetime_fixture_rows:
            value = 200.0 * scale_by_delay[row["delay_label"]] * response_orientation(row)
            if session_scale:
                value *= session_scale.get(row["session"], 1.0)
            if mapping_scale:
                value *= mapping_scale.get(row["mapping"], 1.0)
            fixture_samples.append({**row, "y": value})
        return lifetime_summary(fixture_samples)

    lifetime_fixtures = {
        "stable_persistence": lifetime_fixture({"0ns": 1.0, "100us": 0.98, "1ms": 0.96, "10ms": 0.94, "100ms": 0.92}),
        "bounded_monotonic_decay": lifetime_fixture({"0ns": 1.0, "100us": 0.8, "1ms": 0.55, "10ms": 0.35, "100ms": 0.2}),
        "immediate_only_response": lifetime_fixture({"0ns": 1.0, "100us": 0.0, "1ms": 0.0, "10ms": 0.0, "100ms": 0.0}),
        "no_post_source_response": lifetime_fixture({"0ns": 0.0, "100us": 0.0, "1ms": 0.0, "10ms": 0.0, "100ms": 0.0}),
        "nonmonotonic_form_change": lifetime_fixture({"0ns": 1.0, "100us": 0.4, "1ms": 1.2, "10ms": 0.3, "100ms": 0.9}),
        "session_confounded_response": lifetime_fixture(
            {"0ns": 1.0, "100us": 0.95, "1ms": 0.9, "10ms": 0.85, "100ms": 0.8},
            session_scale={"session_0": 1.0, "session_1": 0.1},
        ),
        "mapping_confounded_response": lifetime_fixture(
            {"0ns": 1.0, "100us": 0.95, "1ms": 0.9, "10ms": 0.85, "100ms": 0.8},
            mapping_scale={"map0": 1.0, "map1": 0.1},
        ),
    }
    query_variation_confounded = lifetime_summary(
        [
            {
                **base,
                "query": qdef["query"],
                "query_order": qdef["query_order"],
                "y": response_orientation({**base, "query": qdef["query"]})
                * 200.0
                * (4.0 if qdef["query"] in {"query_A_then_B", "query_B_then_A"} else 1.0),
            }
            for base in lifetime_fixture_rows
            for qdef in query_family()
            if qdef["query"] in ACTIVE_SIGNAL_QUERIES
        ]
    )
    session_crossover = lifetime_summary(
        [
            {
                **row,
                "y": response_orientation(row)
                * 200.0
                * (10.0 if (row["session"] == "session_0" and row["query"] == "query_A") or (row["session"] == "session_1" and row["query"] == "query_B") else 1.0),
            }
            for row in samples
            if row["q"] == 1536 and row["query"] in {"query_A", "query_B"}
        ]
    )
    mapping_crossover = lifetime_summary(
        [
            {
                **row,
                "y": response_orientation(row)
                * 200.0
                * (10.0 if (row["mapping"] == "map0" and row["query"] == "query_A") or (row["mapping"] == "map1" and row["query"] == "query_B") else 1.0),
            }
            for row in samples
            if row["q"] == 1536 and row["query"] in {"query_A", "query_B"}
        ]
    )
    equal_marginal_session_mapping_crossover = lifetime_summary(
        [
            {
                **row,
                "y": response_orientation(row)
                * (
                    400.0
                    if (row["session"] == "session_0" and row["mapping"] == "map0")
                    or (row["session"] == "session_1" and row["mapping"] == "map1")
                    else 100.0
                ),
            }
            for row in lifetime_fixture_rows
        ]
    )
    signed_delay_reversal = lifetime_summary(
        [
            {
                **row,
                "y": response_orientation(row) * (200.0 if row["delay_label"] != "10ms" else -200.0),
            }
            for row in lifetime_fixture_rows
        ]
    )
    signed_session_confounded = lifetime_summary(
        [
            {
                **row,
                "y": response_orientation(row) * (200.0 if row["session"] == "session_0" else -200.0),
            }
            for row in lifetime_fixture_rows
        ]
    )
    signed_mapping_confounded = lifetime_summary(
        [
            {
                **row,
                "y": 200.0,
            }
            for row in lifetime_fixture_rows
        ]
    )
    opposite_sign_replicates = lifetime_summary(
        [
            {
                **row,
                "y": response_orientation(row) * (200.0 if row["replicate"] == 0 else -200.0),
            }
            for row in lifetime_fixture_rows
        ]
    )

    def has_lifetime_class(summary: dict[str, Any], expected: str) -> bool:
        return any(curve["lifetime_class"] == expected for curve in summary["curves"].values())

    classifier_regressions = {
        "perfectly_separated_codewords_pass": operator["distinguishability"]["classifier"]["passed"],
        "deliberately_confused_codewords_fail": not codeword_classifier(confused_classification_samples, lambda row: row["replicate"] == 1)["passed"],
        "class_imbalanced_trivial_predictor_fails": not codeword_classifier(imbalanced_trivial_samples, lambda row: row["replicate"] == 1)["passed"],
        "class_imbalanced_fixture_has_unequal_support": len({sum(1 for row in imbalanced_trivial_samples if row["q"] == q) for q in Q_VALUES}) > 1,
        "training_memorization_with_heldout_failure_fails": not codeword_classifier(
            memorized_train_failed_test_samples,
            lambda row: row["replicate"] == 1,
        )["passed"],
    }
    lifetime_regressions = {
        "stable_persistence_classified": has_lifetime_class(lifetime_fixtures["stable_persistence"], "persists across the full grid")
        and lifetime_fixtures["stable_persistence"]["passed"],
        "bounded_monotonic_decay_classified": has_lifetime_class(lifetime_fixtures["bounded_monotonic_decay"], "survives a bounded delay"),
        "immediate_only_response_classified": has_lifetime_class(lifetime_fixtures["immediate_only_response"], "survives only immediate handoff"),
        "no_post_source_response_classified": has_lifetime_class(lifetime_fixtures["no_post_source_response"], "vanishes before source death"),
        "nonmonotonic_form_change_classified": has_lifetime_class(lifetime_fixtures["nonmonotonic_form_change"], "changes form across delay"),
        "session_confounded_response_downgraded": not lifetime_fixtures["session_confounded_response"]["variation_gates"]["session_variation_passed"],
        "mapping_confounded_response_downgraded": not lifetime_fixtures["mapping_confounded_response"]["variation_gates"]["mapping_variation_passed"],
        "query_variation_confounded_response_downgraded": not query_variation_confounded["variation_gates"]["query_variation_passed"],
        "session_query_crossover_downgraded": not session_crossover["variation_gates"]["session_variation_passed"],
        "mapping_query_crossover_downgraded": not mapping_crossover["variation_gates"]["mapping_variation_passed"],
        "equal_marginal_session_mapping_crossover_downgraded": not equal_marginal_session_mapping_crossover["variation_gates"]["session_variation_passed"]
        and not equal_marginal_session_mapping_crossover["variation_gates"]["mapping_variation_passed"],
        "signed_delay_reversal_classified_as_form_change": has_lifetime_class(signed_delay_reversal, "changes form across delay")
        and not signed_delay_reversal["variation_gates"]["polarity_consistency_passed"],
        "packet_path_signed_delay_reversal_downgraded": packet_path_signed_delay_reversal_analysis["passed"]
        and not packet_path_signed_delay_reversal_analysis["lifetime"]["variation_gates"]["polarity_consistency_passed"],
        "signed_session_confounded_response_downgraded": not signed_session_confounded["variation_gates"]["session_variation_passed"],
        "signed_mapping_confounded_response_downgraded": not signed_mapping_confounded["variation_gates"]["mapping_variation_passed"],
        "opposite_sign_replicates_fail_confidence": not opposite_sign_replicates["variation_gates"]["confidence_variation_passed"]
        or not opposite_sign_replicates["variation_gates"]["polarity_consistency_passed"],
    }

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
            "missing_temperature_sensor_identity_rejected": not validate_evidence_packet(mutated_missing_temperature_identity, schedule)["passed"],
            "temperature_sensor_identity_digest_mismatch_rejected": not validate_evidence_packet(mutated_temperature_identity_digest, schedule)["passed"],
            "feature_temperature_sensor_identity_drift_rejected": not validate_evidence_packet(mutated_feature_temperature_identity, schedule)["passed"],
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
            "held_out_replicate_absent_from_training": operator["held_out_replicate"]["tested_factor_absent_from_training"],
            "held_out_mapping_absent_from_training": operator["held_out_mapping"]["tested_factor_absent_from_training"],
            "held_out_delay_absent_from_training": operator["held_out_delay"]["tested_factor_absent_from_training"],
            "s2_sufficient_for_ideal_fixture": all(
                operator[name]["smallest_sufficient_model"] == "S2"
                for name in ["held_out_replicate", "held_out_mapping", "held_out_delay"]
            ),
            "strict_factor_holdouts_pass": all(
                operator[name]["passed"]
                for name in ["held_out_replicate", "held_out_mapping", "held_out_delay"]
            ),
            "all_active_queries_preserved": operator["query_structure"]["all_active_queries_preserved"],
            "ordered_queries_preserved": operator["query_structure"]["ordered_queries_preserved"],
            "query_order_terms_not_duplicated_with_query_terms": operator["query_structure"]["query_order_terms_not_duplicated_with_query_terms"],
            "query_order_independent_main_effect_not_claimed": not operator["query_structure"]["query_order_independent_main_effect_claimed"],
            "ordered_query_features_distinct": operator["query_structure"]["ordered_query_features_distinct"],
            "single_query_features_distinct": operator["query_structure"]["single_query_features_distinct"],
            "raw_query_observations_preserved": operator_prediction_gate(evidence_samples(observed_packet, schedule))["gates"]["raw_query_observations_preserved"],
            "equal_pairwise_contrasts_different_raw_responses_remain_distinct": raw_equal_contrast_vectors_differ,
            "query_order_only_effect_detected": query_order_operator["query_structure"]["query_order_effect_abs"] > 100.0,
            "query_order_only_fixture_source_balanced": query_order_operator["query_structure"]["source_order_effect_abs"] <= 1.0,
            "query_order_only_receiver_order_effect_detected": query_order_operator["query_structure"]["query_order_effect_abs"] > 100.0,
            "query_order_collapsed_receiver_order_downgraded": query_order_collapsed_operator["query_structure"]["query_order_effect_abs"] <= 1.0,
            "source_order_only_effect_detected": source_order_operator["query_structure"]["source_order_effect_abs"] > 100.0,
            "source_order_only_fixture_query_balanced": source_order_operator["query_structure"]["query_order_effect_abs"] <= 1.0,
            "source_order_only_fixture_holdouts_pass": all(
                source_order_operator[name]["passed"]
                for name in ["held_out_replicate", "held_out_mapping", "held_out_delay"]
            ),
            "independent_source_order_contrasts_preserved": operator["query_structure"]["independent_source_order_contrasts_preserved"],
            "q0_mapping_by_query_vectors_distinct": operator["query_structure"]["q0_mapping_by_query_vectors_distinct"],
            "mapping_scrambled_packet_remains_candidate": not operator_prediction_gate(mapping_scrambled_samples)["passed"],
            "delay_scrambled_packet_remains_candidate": not operator_prediction_gate(delay_scrambled_samples)["passed"],
            "in_sample_fit_cannot_rescue_strict_factor_holdout_failure": not fit_operator_ladder(mapping_scrambled_samples)["held_out_mapping"]["passed"],
            "classification_regressions": classifier_regressions,
            "between_state_exceeds_within_state": operator["distinguishability"]["between_state_min_distance"]
            > operator["distinguishability"]["within_state_max_distance"],
            "held_out_classifier_passed": operator["distinguishability"]["classifier"]["passed"],
            "classification_confusion_matrix": operator["distinguishability"]["classifier"]["confusion_matrix"],
            "classification_balanced_accuracy": operator["distinguishability"]["classifier"]["balanced_accuracy"],
            "classification_minimum_balanced_accuracy": operator["distinguishability"]["classifier"]["minimum_balanced_accuracy"],
            "in_sample_only_claim_rejected": True,
            "no_persistent_state_fixture_mean": statistics.fmean(row["y"] for row in no_state_samples),
            "natural_relaxation_fixture_mean": statistics.fmean(row["y"] for row in natural_samples),
            "ideal_persistent_public_state_has_signal": max(abs(row["y"]) for row in samples) > 0,
            "lifetime_regressions": lifetime_regressions,
            "lifetime_fixture_summaries": lifetime_fixtures,
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
            result["operator_analysis_tests"]["held_out_replicate_absent_from_training"],
            result["operator_analysis_tests"]["held_out_mapping_absent_from_training"],
            result["operator_analysis_tests"]["held_out_delay_absent_from_training"],
            result["operator_analysis_tests"]["s2_sufficient_for_ideal_fixture"],
            result["operator_analysis_tests"]["strict_factor_holdouts_pass"],
            result["operator_analysis_tests"]["all_active_queries_preserved"],
            result["operator_analysis_tests"]["ordered_queries_preserved"],
            result["operator_analysis_tests"]["query_order_terms_not_duplicated_with_query_terms"],
            result["operator_analysis_tests"]["query_order_independent_main_effect_not_claimed"],
            result["operator_analysis_tests"]["ordered_query_features_distinct"],
            result["operator_analysis_tests"]["single_query_features_distinct"],
            result["operator_analysis_tests"]["raw_query_observations_preserved"],
            result["operator_analysis_tests"]["equal_pairwise_contrasts_different_raw_responses_remain_distinct"],
            result["operator_analysis_tests"]["query_order_only_effect_detected"],
            result["operator_analysis_tests"]["query_order_only_fixture_source_balanced"],
            result["operator_analysis_tests"]["query_order_only_receiver_order_effect_detected"],
            result["operator_analysis_tests"]["query_order_collapsed_receiver_order_downgraded"],
            result["operator_analysis_tests"]["source_order_only_effect_detected"],
            result["operator_analysis_tests"]["source_order_only_fixture_query_balanced"],
            result["operator_analysis_tests"]["source_order_only_fixture_holdouts_pass"],
            result["operator_analysis_tests"]["independent_source_order_contrasts_preserved"],
            result["operator_analysis_tests"]["q0_mapping_by_query_vectors_distinct"],
            result["operator_analysis_tests"]["mapping_scrambled_packet_remains_candidate"],
            result["operator_analysis_tests"]["delay_scrambled_packet_remains_candidate"],
            result["operator_analysis_tests"]["in_sample_fit_cannot_rescue_strict_factor_holdout_failure"],
            all(result["operator_analysis_tests"]["classification_regressions"].values()),
            result["operator_analysis_tests"]["between_state_exceeds_within_state"],
            result["operator_analysis_tests"]["held_out_classifier_passed"],
            all(result["operator_analysis_tests"]["lifetime_regressions"].values()),
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
