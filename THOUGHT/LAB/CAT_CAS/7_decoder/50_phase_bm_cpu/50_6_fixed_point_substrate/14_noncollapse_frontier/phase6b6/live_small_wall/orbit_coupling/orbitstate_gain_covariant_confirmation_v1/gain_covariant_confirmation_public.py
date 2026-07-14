#!/usr/bin/env python3
"""Public schedule, blinded feature extraction, and private adjudication.

This module is intentionally usable offline.  The public receiver path consumes
only the public schedule, receiver PMU records, receiver sentinels, and public
stage receipts.  The private source map and source receipts are joined only by
the adjudicator after the receiver feature hash is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent

N = 256
D_MEMBER = 23
FOLD_MEMBER = 233
QUANTIZATION_SCALE = 1536
BASE_WORK = 2048
TOTAL_WORK = 4096
PUBLIC_Q0_ABSOLUTE_BOUND = 152.0
PRIVATE_ODD_SIGNAL_FLOOR = 456.0
RELATIONAL_TOLERANCE = 0.25
REPLICATES = [0, 1]
PHASES = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
PUBLIC_SEED = "orbitstate-gain-covariant-final-confirmation-public-seed-5b7b1338-a7daa611"
PRIMARY_COORDINATE = "change_to_dirty"
SOURCE_CORE = 4
RECEIVER_CORE = 5
RUN_ID = "orbitstate_gain_covariant_confirmation_v1_0"

PUBLIC_SCHEDULE_JSON = HERE / "GAIN_COVARIANT_PUBLIC_SCHEDULE.json"
PUBLIC_SCHEDULE_TSV = HERE / "GAIN_COVARIANT_PUBLIC_SCHEDULE.tsv"
PUBLIC_SCHEDULE_SHA = HERE / "GAIN_COVARIANT_PUBLIC_SCHEDULE.sha256"
PRIVATE_SOURCE_MAP_JSON = HERE / "GAIN_COVARIANT_PRIVATE_SOURCE_MAP.json"
PRIVATE_SOURCE_MAP_SHA = HERE / "GAIN_COVARIANT_PRIVATE_SOURCE_MAP.sha256"
GAIN_COVARIANT_CONTRACT_REFERENCE_JSON = HERE / "GAIN_COVARIANT_CONTRACT_REFERENCE.json"

RESULT_CONFIRMED = "ORBITSTATE_GAIN_COVARIANT_CONFIRMATION_CONFIRMED"
RESULT_CANDIDATE = "ORBITSTATE_GAIN_COVARIANT_CONFIRMATION_CANDIDATE"
RESULT_NOT_ESTABLISHED = "ORBITSTATE_GAIN_COVARIANT_CONFIRMATION_NOT_ESTABLISHED"
ALLOWED_RESULT_CLASSES = [RESULT_CONFIRMED, RESULT_CANDIDATE, RESULT_NOT_ESTABLISHED]
FORBIDDEN_RESULT_CLASSES = ["SMALL_WALL_CROSSED"]

PUBLIC_COLUMNS = [
    "opaque_group_id",
    "opaque_run_id",
    "replicate",
    "public_decoder_phase_index",
    "public_decoder_phase_radians",
    "physical_mapping",
    "mapping_order",
    "source_execution_order",
    "subcapture_order",
    "public_randomized_execution_ordinal",
]

FORBIDDEN_PUBLIC_KEYS = {
    "condition",
    "member",
    "fold",
    "target",
    "source_mode",
    "response_mode",
    "q",
    "q_theta",
    "positive_work",
    "negative_work",
    "dummy_work",
    "private_source_phase",
    "private_source_phase_index",
    "private_source_phase_radians",
    "expected_sign",
    "expected_result",
}

FORBIDDEN_PUBLIC_VALUES = {
    "pre_projection_d",
    "pre_projection_fold",
    "source_off",
    "query_off",
    "post_projection",
    "declaration_sham",
    "query_scramble",
    "equal_orbit_odd_zero",
    "source_polarity_inversion_d",
    "pre_projection",
    "post_projection",
    "declaration_sham",
    "scramble",
    "polarity",
}

CONDITIONS = [
    {
        "condition": "pre_projection_d",
        "member": D_MEMBER,
        "response_mode": "pre_projection",
        "private_phase_sequence": [0, 1, 2, 3],
        "polarity_inversion": False,
        "source_off_dummy_mode": False,
    },
    {
        "condition": "pre_projection_fold",
        "member": FOLD_MEMBER,
        "response_mode": "pre_projection",
        "private_phase_sequence": [0, 1, 2, 3],
        "polarity_inversion": False,
        "source_off_dummy_mode": False,
    },
    {
        "condition": "source_off",
        "member": D_MEMBER,
        "response_mode": "source_off",
        "private_phase_sequence": [0, 1, 2, 3],
        "polarity_inversion": False,
        "source_off_dummy_mode": True,
    },
    {
        "condition": "query_off",
        "member": D_MEMBER,
        "response_mode": "query_off",
        "private_phase_sequence": [0, 1, 2, 3],
        "polarity_inversion": False,
        "source_off_dummy_mode": False,
    },
    {
        "condition": "post_projection",
        "member": D_MEMBER,
        "response_mode": "post_projection",
        "private_phase_sequence": [0, 1, 2, 3],
        "polarity_inversion": False,
        "source_off_dummy_mode": False,
    },
    {
        "condition": "declaration_sham",
        "member": D_MEMBER,
        "response_mode": "declaration_sham",
        "private_phase_sequence": [0, 1, 2, 3],
        "polarity_inversion": False,
        "source_off_dummy_mode": False,
    },
    {
        "condition": "query_scramble",
        "member": D_MEMBER,
        "response_mode": "pre_projection",
        "private_phase_sequence": [0, 2, 0, 2],
        "polarity_inversion": False,
        "source_off_dummy_mode": False,
    },
    {
        "condition": "equal_orbit_odd_zero",
        "member": 0,
        "response_mode": "equal_orbit_odd_zero",
        "private_phase_sequence": [0, 1, 2, 3],
        "polarity_inversion": False,
        "source_off_dummy_mode": False,
    },
    {
        "condition": "source_polarity_inversion_d",
        "member": D_MEMBER,
        "response_mode": "pre_projection",
        "private_phase_sequence": [0, 1, 2, 3],
        "polarity_inversion": True,
        "source_off_dummy_mode": False,
    },
]


class OrbitStatePublicError(AssertionError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise OrbitStatePublicError(f"{name} must be an exact integer")
    return value


def exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise OrbitStatePublicError(f"{name} must be an exact boolean")
    return value


def phase_radians(index: int) -> float:
    return PHASES[index]


def stable_token(*parts: Any, length: int = 16) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def round_c(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def clamp_q(value: int) -> int:
    return max(-QUANTIZATION_SCALE, min(QUANTIZATION_SCALE, value))


def orbit_phi(member: int) -> float:
    return 2.0 * math.pi * float(member) / float(N)


def q_theta_from_fields(
    *,
    member: int,
    response_mode: str,
    public_phase_index: int,
    private_phase_index: int,
    polarity_inversion: bool,
    source_off_dummy_mode: bool,
) -> int:
    if source_off_dummy_mode or response_mode == "source_off":
        return 0
    theta = phase_radians(public_phase_index)
    private_theta = phase_radians(private_phase_index)
    phi = orbit_phi(member)
    if response_mode == "query_off":
        q_value = 0
    elif response_mode == "pre_projection":
        q_value = round_c(float(QUANTIZATION_SCALE) * math.cos(phi - private_theta))
    elif response_mode == "post_projection":
        q_value = round_c(float(QUANTIZATION_SCALE) * math.cos(phi) * math.cos(theta))
    elif response_mode == "declaration_sham":
        q_value = round_c(float(QUANTIZATION_SCALE) * math.cos(phi))
    elif response_mode == "equal_orbit_odd_zero":
        q_value = round_c(float(QUANTIZATION_SCALE) * math.cos(theta))
    else:
        raise OrbitStatePublicError(f"unknown response_mode {response_mode!r}")
    if polarity_inversion:
        q_value = -q_value
    return clamp_q(q_value)


def q_theta(record: dict[str, Any]) -> int:
    return q_theta_from_fields(
        member=exact_int(record["orbit_state"]["member"], "OrbitState.member"),
        response_mode=str(record["response_mode"]),
        public_phase_index=exact_int(record["public_phase_index"], "public_phase_index"),
        private_phase_index=exact_int(record["private_source_phase_index"], "private_source_phase_index"),
        polarity_inversion=exact_bool(record["polarity_inversion"], "polarity_inversion"),
        source_off_dummy_mode=exact_bool(record["source_off_dummy_mode"], "source_off_dummy_mode"),
    )


def source_work_for_q(q_value: int, *, source_off_dummy_mode: bool) -> dict[str, int]:
    if source_off_dummy_mode:
        return {
            "q_theta": 0,
            "positive_work": 0,
            "negative_work": 0,
            "dummy_work": TOTAL_WORK,
            "total_work": TOTAL_WORK,
        }
    return {
        "q_theta": q_value,
        "positive_work": BASE_WORK + q_value,
        "negative_work": BASE_WORK - q_value,
        "dummy_work": 0,
        "total_work": TOTAL_WORK,
    }


def contract_reference() -> dict[str, Any]:
    return {
        "schema": "GAIN_COVARIANT_CONTRACT_REFERENCE_V1",
        "contract_sha256": "31af6869bdf4e25634b1e408830015af2b2c4f20202b2df28492b2e1a9a90860",
        "science_package_id": RUN_ID,
        "transaction_run_id": RUN_ID,
        "public_randomization_seed": PUBLIC_SEED,
        "source_commit": "4ff588cea11343bf38d4c96c1281d34cbf1961ed",
        "official_retained_class": "ORBITSTATE_INDEPENDENT_COUPLING_CANDIDATE",
        "retrospective_explanatory_class": "PRIVATE_ORBITSTATE_GAIN_COVARIANT_GEOMETRY_ESTABLISHED",
        "small_wall_crossed_promoted": False,
        "primary_coordinate": PRIMARY_COORDINATE,
        "gain_control_conditions": ["post_projection", "equal_orbit_odd_zero"],
        "public_q0_absolute_bound": PUBLIC_Q0_ABSOLUTE_BOUND,
        "private_odd_signal_floor": PRIVATE_ODD_SIGNAL_FLOOR,
        "relational_tolerance": RELATIONAL_TOLERANCE,
        "predecessor_evidence_allowed_in_decision": False,
    }


def build_public_schedule() -> dict[str, Any]:
    group_ids = {
        condition["condition"]: f"grp_{index:02d}_{stable_token(PUBLIC_SEED, 'group', index)}"
        for index, condition in enumerate(CONDITIONS)
    }
    rows: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        for condition_index, condition in enumerate(CONDITIONS):
            group_id = group_ids[condition["condition"]]
            for phase_index, theta in enumerate(PHASES):
                for mapping in [0, 1]:
                    token = stable_token(PUBLIC_SEED, replicate, condition_index, phase_index, mapping)
                    rows.append(
                        {
                            "opaque_group_id": group_id,
                            "opaque_run_id": f"run_{token}",
                            "replicate": replicate,
                            "public_decoder_phase_index": phase_index,
                            "public_decoder_phase_radians": round(theta, 12),
                            "physical_mapping": f"map{mapping}",
                            "mapping_order": "order_a"
                            if int(stable_token(token, "mapping", length=2), 16) % 2 == 0
                            else "order_b",
                            "source_execution_order": "pos_first"
                            if int(stable_token(token, "source", length=2), 16) % 2 == 0
                            else "neg_first",
                            "subcapture_order": "pos_then_neg"
                            if int(stable_token(token, "subcapture", length=2), 16) % 2 == 0
                            else "neg_then_pos",
                            "public_randomized_execution_ordinal": -1,
                        }
                    )
    randomized = sorted(rows, key=lambda row: stable_token(PUBLIC_SEED, row["opaque_run_id"], "ordinal"))
    for ordinal, row in enumerate(randomized):
        row["public_randomized_execution_ordinal"] = ordinal
    rows = sorted(randomized, key=lambda row: int(row["public_randomized_execution_ordinal"]))
    schedule = {
        "schema": "GAIN_COVARIANT_PUBLIC_SCHEDULE_V2",
        "public_seed": PUBLIC_SEED,
        "run_id": RUN_ID,
        "n": N,
        "phase_count": len(PHASES),
        "replicates": len(REPLICATES),
        "rows": rows,
    }
    validate_public_schedule(schedule)
    return schedule


def build_private_source_map(schedule: dict[str, Any]) -> dict[str, Any]:
    rows_by_group: dict[str, dict[str, Any]] = {}
    for index, condition in enumerate(CONDITIONS):
        rows_by_group[f"grp_{index:02d}_{stable_token(PUBLIC_SEED, 'group', index)}"] = condition
    records: list[dict[str, Any]] = []
    for row in schedule["rows"]:
        condition = rows_by_group[row["opaque_group_id"]]
        phase_index = int(row["public_decoder_phase_index"])
        private_phase_index = int(condition["private_phase_sequence"][phase_index])
        record = {
            "opaque_group_id": row["opaque_group_id"],
            "opaque_run_id": row["opaque_run_id"],
            "replicate": int(row["replicate"]),
            "public_phase_index": phase_index,
            "physical_mapping": row["physical_mapping"],
            "condition": condition["condition"],
            "orbit_state": {
                "modulus": N,
                "member": int(condition["member"]),
            },
            "response_mode": condition["response_mode"],
            "private_source_phase_sequence": list(condition["private_phase_sequence"]),
            "private_source_phase_index": private_phase_index,
            "private_source_phase_radians": round(phase_radians(private_phase_index), 12),
            "polarity_inversion": bool(condition["polarity_inversion"]),
            "source_off_dummy_mode": bool(condition["source_off_dummy_mode"]),
        }
        records.append(record)
    source_map = {
        "schema": "GAIN_COVARIANT_PRIVATE_SOURCE_MAP_V2",
        "run_id": RUN_ID,
        "sealed_before_live_authorization": True,
        "public_schedule_sha256": digest(schedule),
        "constants": {
            "n": N,
            "d": D_MEMBER,
            "fold_d": FOLD_MEMBER,
            "quantization_scale": QUANTIZATION_SCALE,
            "base_work": BASE_WORK,
            "total_work": TOTAL_WORK,
        },
        "records": records,
    }
    validate_private_source_map(schedule, source_map)
    return source_map


def public_schedule_tsv(schedule: dict[str, Any]) -> str:
    lines = ["\t".join(PUBLIC_COLUMNS)]
    for row in schedule["rows"]:
        lines.append("\t".join(str(row[column]) for column in PUBLIC_COLUMNS))
    return "\n".join(lines) + "\n"


def validate_public_schedule(schedule: dict[str, Any]) -> None:
    rows = schedule.get("rows")
    if not isinstance(rows, list):
        raise OrbitStatePublicError("public schedule rows missing")
    expected_rows = len(REPLICATES) * len(CONDITIONS) * len(PHASES) * 2
    if len(rows) != expected_rows:
        raise OrbitStatePublicError(f"public schedule row count {len(rows)} != {expected_rows}")
    run_ids = set()
    ordinals = set()
    counts: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    for row in rows:
        if set(row) != set(PUBLIC_COLUMNS):
            raise OrbitStatePublicError(f"public schedule columns are not exact for {row!r}")
        bad_keys = FORBIDDEN_PUBLIC_KEYS.intersection(row)
        if bad_keys:
            raise OrbitStatePublicError(f"public schedule contains forbidden keys {sorted(bad_keys)}")
        lowered_values = {str(value).lower() for value in row.values()}
        leaked = [value for value in lowered_values for bad in FORBIDDEN_PUBLIC_VALUES if bad in value]
        if leaked:
            raise OrbitStatePublicError(f"public schedule contains forbidden private values {sorted(set(leaked))}")
        run_id = str(row["opaque_run_id"])
        if run_id in run_ids:
            raise OrbitStatePublicError(f"duplicate opaque run id {run_id}")
        run_ids.add(run_id)
        ordinal = int(row["public_randomized_execution_ordinal"])
        if ordinal in ordinals:
            raise OrbitStatePublicError(f"duplicate ordinal {ordinal}")
        ordinals.add(ordinal)
        replicate = int(row["replicate"])
        phase = int(row["public_decoder_phase_index"])
        mapping = str(row["physical_mapping"])
        if replicate not in REPLICATES:
            raise OrbitStatePublicError("bad replicate")
        if phase not in range(len(PHASES)):
            raise OrbitStatePublicError("bad public phase")
        if mapping not in {"map0", "map1"}:
            raise OrbitStatePublicError("bad mapping")
        counts[(replicate, str(row["opaque_group_id"]), phase)].add(mapping)
    if ordinals != set(range(expected_rows)):
        raise OrbitStatePublicError("public randomized ordinals are not contiguous")
    if len(counts) != len(REPLICATES) * len(CONDITIONS) * len(PHASES):
        raise OrbitStatePublicError("condition/phase/mapping geometry is incomplete")
    if any(value != {"map0", "map1"} for value in counts.values()):
        raise OrbitStatePublicError("each condition phase must have both mapping legs")


def validate_private_source_map(schedule: dict[str, Any], source_map: dict[str, Any]) -> None:
    if exact_bool(source_map.get("sealed_before_live_authorization"), "sealed_before_live_authorization") is not True:
        raise OrbitStatePublicError("private source map must be sealed before live authorization")
    records = source_map.get("records")
    if not isinstance(records, list):
        raise OrbitStatePublicError("private source map records missing")
    schedule_ids = {row["opaque_run_id"] for row in schedule["rows"]}
    record_ids = {record["opaque_run_id"] for record in records}
    if schedule_ids != record_ids:
        raise OrbitStatePublicError("private source map does not exactly cover the public schedule")
    for record in records:
        if exact_int(record["orbit_state"]["modulus"], "OrbitState.modulus") != N:
            raise OrbitStatePublicError("bad OrbitState modulus")
        exact_bool(record["polarity_inversion"], "polarity_inversion")
        source_off_dummy_mode = exact_bool(record["source_off_dummy_mode"], "source_off_dummy_mode")
        work = source_work_for_q(q_theta(record), source_off_dummy_mode=source_off_dummy_mode)
        if work["total_work"] != TOTAL_WORK:
            raise OrbitStatePublicError("bad total work")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def receiver_feature_digest(features: dict[str, Any]) -> str:
    return digest({key: value for key, value in features.items() if key != "receiver_features_sha256"})


def validate_receiver_blindness(records: list[dict[str, Any]], *, stream_name: str) -> None:
    forbidden = {
        "condition",
        "member",
        "response_mode",
        "q",
        "q_theta",
        "positive_work",
        "negative_work",
        "dummy_work",
        "total_work",
        "private_source_phase_index",
        "private_source_phase_radians",
        "expected_result",
        "expected_sign",
    }
    def walk(value: Any, path: str) -> list[str]:
        hits: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                next_path = f"{path}.{key}" if path else str(key)
                if str(key) in forbidden:
                    hits.append(next_path)
                hits.extend(walk(item, next_path))
        elif isinstance(value, list):
            for item_index, item in enumerate(value):
                hits.extend(walk(item, f"{path}[{item_index}]"))
        return hits

    for index, record in enumerate(records):
        bad = walk(record, "")
        if bad:
            raise OrbitStatePublicError(
                f"{stream_name} record {index} contains private receiver-forbidden keys {sorted(bad)}"
            )


def validate_raw_against_schedule(
    schedule: dict[str, Any],
    raw_records: list[dict[str, Any]],
    sentinels: list[dict[str, Any]],
    stage_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    def require_exact_int(value: Any, expected: int, message: str) -> None:
        if type(value) is not int or value != expected:
            raise OrbitStatePublicError(message)

    def require_exact_bool(value: Any, expected: bool, message: str) -> None:
        if type(value) is not bool or value is not expected:
            raise OrbitStatePublicError(message)

    def require_digest_text(value: Any, message: str) -> str:
        if type(value) is not str or not value:
            raise OrbitStatePublicError(message)
        return value

    validate_public_schedule(schedule)
    validate_receiver_blindness(raw_records, stream_name="raw receiver")
    validate_receiver_blindness(sentinels, stream_name="receiver sentinel")
    validate_receiver_blindness(stage_receipts, stream_name="stage receipt")
    rows_by_id = {row["opaque_run_id"]: row for row in schedule["rows"]}
    expected_raw = len(schedule["rows"]) * 2
    if len(raw_records) != expected_raw:
        raise OrbitStatePublicError(f"raw component windows {len(raw_records)} != {expected_raw}")
    if len(stage_receipts) != expected_raw * 7:
        raise OrbitStatePublicError("stage receipts must contain seven stages per component window")
    components_by_run: dict[str, set[str]] = defaultdict(set)
    for record in raw_records:
        run_id = record["opaque_run_id"]
        if run_id not in rows_by_id:
            raise OrbitStatePublicError(f"raw record unknown run id {run_id}")
        require_exact_int(record.get("replicate"), int(rows_by_id[run_id]["replicate"]), "raw replicate mismatch")
        if int(record["replicate"]) != int(rows_by_id[run_id]["replicate"]):
            raise OrbitStatePublicError("raw replicate mismatch")
        require_exact_int(
            record.get("public_decoder_phase_index"),
            int(rows_by_id[run_id]["public_decoder_phase_index"]),
            "raw phase mismatch",
        )
        if record["physical_mapping"] != rows_by_id[run_id]["physical_mapping"]:
            raise OrbitStatePublicError("raw mapping mismatch")
        if record["component"] not in {"positive", "negative"}:
            raise OrbitStatePublicError("component must be positive or negative")
        require_exact_int(record.get("receiver_core"), RECEIVER_CORE, "receiver window on wrong core")
        require_exact_bool(record.get("pmu_unmultiplexed"), True, "PMU window multiplexed")
        require_exact_bool(record.get("byte_compare_ok"), True, "component bytes changed")
        require_exact_bool(record.get("event_ids_valid"), True, "PMU event ID drift")
        components_by_run[run_id].add(record["component"])
    if any(value != {"positive", "negative"} for value in components_by_run.values()):
        raise OrbitStatePublicError("each mapping leg must contain two independent components")
    expected_stage_order = [
        "receiver_full_baseline",
        "pre_sentinels",
        "receiver_rebaseline",
        "source_execute",
        "receiver_measure",
        "receiver_restoration",
        "post_sentinels",
    ]
    stage_keys = {(receipt["opaque_run_id"], receipt["component"], receipt["stage"]) for receipt in stage_receipts}
    expected_stages = {
        (record["opaque_run_id"], record["component"], stage)
        for record in raw_records
        for stage in expected_stage_order
    }
    if stage_keys != expected_stages:
        raise OrbitStatePublicError("stage receipt set is incomplete")
    by_component_stage: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for receipt in stage_receipts:
        require_exact_int(receipt.get("receiver_core"), RECEIVER_CORE, "stage receipt receiver core mismatch")
        require_exact_bool(receipt.get("ok"), True, "stage receipt reports failed custody")
        require_digest_text(receipt.get("byte_digest"), "stage receipt missing byte digest")
        by_component_stage[(receipt["opaque_run_id"], receipt["component"])].append(receipt)
    for key, receipts in by_component_stage.items():
        for item in receipts:
            if type(item.get("sequence_index")) is not int:
                raise OrbitStatePublicError(f"stage receipt sequence type mismatch for {key}")
        ordered = sorted(receipts, key=lambda item: item["sequence_index"])
        if [item["stage"] for item in ordered] != expected_stage_order:
            raise OrbitStatePublicError(f"stage receipt order mismatch for {key}")
        if [item["sequence_index"] for item in ordered] != list(range(len(expected_stage_order))):
            raise OrbitStatePublicError(f"stage receipt sequence mismatch for {key}")
        baseline_digest = require_digest_text(ordered[0].get("byte_digest"), f"{key} baseline missing digest")
        for stage_name in ["source_execute", "receiver_measure"]:
            stage_item = next(item for item in ordered if item["stage"] == stage_name)
            require_digest_text(stage_item.get("byte_digest"), f"{key} {stage_name} missing digest")
        for stage_name in ["pre_sentinels", "receiver_rebaseline", "receiver_restoration", "post_sentinels"]:
            stage_item = next(item for item in ordered if item["stage"] == stage_name)
            digest_text = require_digest_text(stage_item.get("byte_digest"), f"{key} {stage_name} missing digest")
            if digest_text != baseline_digest:
                raise OrbitStatePublicError(f"{key} {stage_name} digest mismatch")
    expected_sentinels = {(record["opaque_run_id"], record["component"]) for record in raw_records}
    observed_sentinels = {(sentinel.get("opaque_run_id"), sentinel.get("component")) for sentinel in sentinels}
    if observed_sentinels != expected_sentinels:
        raise OrbitStatePublicError("sentinel component coverage mismatch")
    if len(sentinels) != len(expected_sentinels):
        raise OrbitStatePublicError("sentinel duplicate or missing component coverage")
    for sentinel in sentinels:
        require_exact_int(sentinel.get("receiver_core"), RECEIVER_CORE, "sentinel receiver core mismatch")
        require_exact_bool(sentinel.get("pre_ok"), True, "sentinel reports failed custody")
        require_exact_bool(sentinel.get("post_ok"), True, "sentinel reports failed custody")
        pre_digest = require_digest_text(sentinel.get("pre_digest"), "sentinel missing digest")
        post_digest = require_digest_text(sentinel.get("post_digest"), "sentinel missing digest")
        key = (sentinel["opaque_run_id"], sentinel["component"])
        ordered = sorted(by_component_stage[key], key=lambda item: int(item["sequence_index"]))
        baseline_digest = ordered[0]["byte_digest"]
        if pre_digest != baseline_digest or post_digest != baseline_digest:
            raise OrbitStatePublicError("sentinel digest mismatch")
    return {
        "passed": True,
        "raw_component_windows": len(raw_records),
        "stage_receipts": len(stage_receipts),
        "sentinels": len(sentinels),
    }


def extract_receiver_features(
    schedule: dict[str, Any],
    raw_records: list[dict[str, Any]],
    sentinels: list[dict[str, Any]],
    stage_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    integrity = validate_raw_against_schedule(schedule, raw_records, sentinels, stage_receipts)
    by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in raw_records:
        by_run[record["opaque_run_id"]][record["component"]] = record
    rows_by_id = {row["opaque_run_id"]: row for row in schedule["rows"]}
    feature_rows: list[dict[str, Any]] = []
    for run_id in sorted(rows_by_id, key=lambda item: rows_by_id[item]["public_randomized_execution_ordinal"]):
        row = rows_by_id[run_id]
        positive = float(by_run[run_id]["positive"]["counters"][PRIMARY_COORDINATE])
        negative = float(by_run[run_id]["negative"]["counters"][PRIMARY_COORDINATE])
        mapping = row["physical_mapping"]
        if mapping == "map0":
            a_minus_b = positive - negative
        else:
            a_minus_b = negative - positive
        feature_rows.append(
            {
                "opaque_group_id": row["opaque_group_id"],
                "opaque_run_id": run_id,
                "replicate": int(row["replicate"]),
                "public_decoder_phase_index": int(row["public_decoder_phase_index"]),
                "public_decoder_phase_radians": float(row["public_decoder_phase_radians"]),
                "physical_mapping": mapping,
                "logical_response": positive - negative,
                "physical_a_minus_b": a_minus_b,
            }
        )
    features = {
        "schema": "GAIN_COVARIANT_RECEIVER_FEATURES_V2",
        "receiver_only": True,
        "private_source_map_opened": False,
        "private_source_fields_seen": [],
        "primary_coordinate": PRIMARY_COORDINATE,
        "integrity": integrity,
        "rows": feature_rows,
    }
    features["receiver_features_sha256"] = receiver_feature_digest(features)
    return features


def rel_error(left: float, right: float, floor: float = PUBLIC_Q0_ABSOLUTE_BOUND) -> float:
    denom = max(abs(left), abs(right), floor)
    return abs(left - right) / denom


def complex_rel_error(left: complex, right: complex) -> float:
    return max(rel_error(left.real, right.real), rel_error(left.imag, right.imag))


def complex_decode(phase_values: dict[int, float]) -> complex:
    if set(phase_values) != {0, 1, 2, 3}:
        raise OrbitStatePublicError("complex decoder requires exactly four public phases")
    total = 0j
    for phase_index, value in phase_values.items():
        theta = phase_radians(phase_index)
        total += float(value) * complex(math.cos(theta), math.sin(theta))
    return (2.0 / 4.0) * total


def mean_complex(values: list[complex]) -> complex:
    return complex(statistics.fmean(value.real for value in values), statistics.fmean(value.imag for value in values))


def source_receipt_law(source_map: dict[str, Any], source_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {record["opaque_run_id"]: record for record in source_map["records"]}
    failures: list[str] = []
    if len(source_receipts) != len(by_id) * 2:
        failures.append(f"source receipt count {len(source_receipts)} != {len(by_id) * 2}")
    expected_components = {(run_id, component) for run_id in by_id for component in ["positive", "negative"]}
    observed_components = {(receipt.get("opaque_run_id"), receipt.get("component")) for receipt in source_receipts}
    if observed_components != expected_components:
        missing = sorted(expected_components - observed_components)
        extra = sorted(observed_components - expected_components)
        failures.append(f"source receipt component coverage mismatch missing={missing[:5]} extra={extra[:5]}")
    if len(observed_components) != len(source_receipts):
        failures.append("source receipt duplicate component coverage")
    for receipt in source_receipts:
        run_id = receipt.get("opaque_run_id")
        if run_id not in by_id:
            failures.append(f"unknown source receipt {run_id}")
            continue
        record = by_id[run_id]
        q_value = q_theta(record)
        expected = source_work_for_q(
            q_value,
            source_off_dummy_mode=exact_bool(record["source_off_dummy_mode"], "source_off_dummy_mode"),
        )
        for key, expected_value in expected.items():
            if type(receipt.get(key)) is not int or receipt.get(key) != expected_value:
                failures.append(f"{run_id} {key} {receipt.get(key)} != {expected_value}")
        if receipt.get("component") not in {"positive", "negative"}:
            failures.append(f"{run_id} invalid receipt component {receipt.get('component')}")
        if type(receipt.get("source_core")) is not int or receipt.get("source_core") != SOURCE_CORE:
            failures.append(f"{run_id} source core mismatch")
        if type(receipt.get("source_cpu_before")) is not int or receipt.get("source_cpu_before") != SOURCE_CORE:
            failures.append(f"{run_id} source CPU before mismatch")
        if type(receipt.get("source_cpu_after")) is not int or receipt.get("source_cpu_after") != SOURCE_CORE:
            failures.append(f"{run_id} source CPU after mismatch")
        if type(receipt.get("receiver_feedback_used_to_select_q")) is not bool or receipt.get(
            "receiver_feedback_used_to_select_q"
        ) is not False:
            failures.append(f"{run_id} q selected using receiver feedback")
    return {"passed": not failures, "failures": failures, "receipt_count": len(source_receipts)}


def phase_transfer_law(
    joined_rows: list[dict[str, Any]],
    *,
    source_map_by_id: dict[str, dict[str, Any]],
    source_receipts_by_id_component: dict[tuple[str, str], dict[str, Any]],
    decodes: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    strong_counts = {
        "total_pair_cells": 0,
        "logical_mapping_passes": 0,
        "logical_mapping_failures": 0,
        "physical_reversal_passes": 0,
        "physical_reversal_failures": 0,
        "sign_passes": 0,
        "sign_failures": 0,
    }
    near_zero_counts = {
        "total_pair_cells": 0,
        "raw_map0_violations": 0,
        "raw_map1_violations": 0,
        "logical_pair_average_violations": 0,
        "physical_reversal_average_violations": 0,
        "decoded_null_violations": 0,
    }
    by_condition_phase: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in joined_rows:
        key = (int(row["replicate"]), row["condition"], int(row["public_decoder_phase_index"]))
        by_condition_phase[key][row["physical_mapping"]] = row
    events: list[dict[str, Any]] = []
    for key, maps in sorted(by_condition_phase.items()):
        replicate, condition, phase_index = key
        event = {"replicate": replicate, "condition": condition, "phase": phase_index}
        if set(maps) != {"map0", "map1"}:
            failures.append(f"{key} missing mapping leg")
            events.append({**event, "failed": "missing_mapping_leg"})
            continue
        map0 = maps["map0"]
        map1 = maps["map1"]
        def receipt_q(row: dict[str, Any]) -> int:
            values = {
                receipt["q_theta"]
                for component in ["positive", "negative"]
                for receipt in [source_receipts_by_id_component[(row["opaque_run_id"], component)]]
            }
            if len(values) != 1:
                raise OrbitStatePublicError("source receipt q_theta mismatch across components")
            return values.pop()

        q_values = {
            mapping_name: receipt_q(row)
            for mapping_name, row in maps.items()
        }
        raw_map0 = float(map0["logical_response"])
        raw_map1 = float(map1["logical_response"])
        physical0 = float(map0["physical_a_minus_b"])
        physical1 = float(map1["physical_a_minus_b"])
        logical_rel = rel_error(raw_map0, raw_map1)
        physical_rel = rel_error(physical0, -physical1)
        sign_pass = all(float(row["logical_response"]) * q_values[mapping_name] > 0 for mapping_name, row in maps.items())
        partition = ["strong" if abs(q_value) >= 256 else "near_zero" for q_value in q_values.values()]
        event.update(
            {
                "q_theta": q_values,
                "map0_logical_response": raw_map0,
                "map1_logical_response": raw_map1,
                "map0_physical_a_minus_b": physical0,
                "map1_physical_a_minus_b": physical1,
                "logical_mapping_relative_error": logical_rel,
                "physical_reversal_relative_error": physical_rel,
                "source_execution_order": {
                    "map0": map0.get("source_execution_order"),
                    "map1": map1.get("source_execution_order"),
                },
                "subcapture_order": {
                    "map0": map0.get("subcapture_order"),
                    "map1": map1.get("subcapture_order"),
                },
            }
        )
        if partition == ["strong", "strong"]:
            strong_counts["total_pair_cells"] += 1
            logical_pass = logical_rel <= RELATIONAL_TOLERANCE
            physical_pass = physical_rel <= RELATIONAL_TOLERANCE
            strong_counts["logical_mapping_passes" if logical_pass else "logical_mapping_failures"] += 1
            strong_counts["physical_reversal_passes" if physical_pass else "physical_reversal_failures"] += 1
            strong_counts["sign_passes" if sign_pass else "sign_failures"] += 1
            if not logical_pass:
                failures.append(f"{key} strong logical mapping invariance")
            if not physical_pass:
                failures.append(f"{key} strong physical reversal")
            if not sign_pass:
                failures.append(f"{key} strong sign")
            if not (logical_pass and physical_pass and sign_pass):
                events.append({**event, "partition": "strong"})
        elif partition == ["near_zero", "near_zero"]:
            strong_sign_not_applicable = sign_pass or True
            del strong_sign_not_applicable
            near_zero_counts["total_pair_cells"] += 1
            logical_pair_average = 0.5 * (raw_map0 + raw_map1)
            physical_reversal_average = 0.5 * (physical0 + physical1)
            checks = {
                "raw_map0": abs(raw_map0) <= PUBLIC_Q0_ABSOLUTE_BOUND,
                "raw_map1": abs(raw_map1) <= PUBLIC_Q0_ABSOLUTE_BOUND,
                "logical_pair_average": abs(logical_pair_average) <= PUBLIC_Q0_ABSOLUTE_BOUND,
                "physical_reversal_average": abs(physical_reversal_average) <= PUBLIC_Q0_ABSOLUTE_BOUND,
            }
            if not checks["raw_map0"]:
                near_zero_counts["raw_map0_violations"] += 1
                failures.append(f"{key} near-zero raw map0")
            if not checks["raw_map1"]:
                near_zero_counts["raw_map1_violations"] += 1
                failures.append(f"{key} near-zero raw map1")
            if not checks["logical_pair_average"]:
                near_zero_counts["logical_pair_average_violations"] += 1
                failures.append(f"{key} near-zero logical pair average")
            if not checks["physical_reversal_average"]:
                near_zero_counts["physical_reversal_average_violations"] += 1
                failures.append(f"{key} near-zero physical reversal average")
            if not all(checks.values()):
                events.append(
                    {
                        **event,
                        "partition": "near_zero",
                        "logical_pair_average": logical_pair_average,
                        "physical_reversal_average": physical_reversal_average,
                        "checks": checks,
                    }
                )
        else:
            failures.append(f"{key} mixed strong/near-zero partition")
            events.append({**event, "partition": partition})

    null_details: dict[str, Any] = {}
    for replicate, condition_decodes in decodes["replicate"].items():
        scope_name = f"replicate_{replicate}"
        null_details[scope_name] = {}
        for condition in ["source_off", "query_off", "post_projection", "equal_orbit_odd_zero"]:
            z_value = condition_average(condition_decodes[condition])
            checks: dict[str, bool]
            if condition in {"source_off", "query_off"}:
                checks = {
                    "real_null": abs(z_value.real) <= PUBLIC_Q0_ABSOLUTE_BOUND,
                    "imaginary_null": abs(z_value.imag) <= PUBLIC_Q0_ABSOLUTE_BOUND,
                }
            else:
                checks = {"imaginary_null": abs(z_value.imag) <= PUBLIC_Q0_ABSOLUTE_BOUND}
            null_details[scope_name][condition] = {
                "real": z_value.real,
                "imag": z_value.imag,
                "checks": checks,
            }
            for check_name, passed in checks.items():
                if not passed:
                    near_zero_counts["decoded_null_violations"] += 1
                    failures.append(f"{scope_name} {condition} {check_name}")
    return {
        "passed": not failures,
        "failures": failures,
        "strong_counts": strong_counts,
        "near_zero_counts": near_zero_counts,
        "failure_events": events,
        "decoded_nulls": null_details,
    }


def build_condition_decodes(joined_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scope: dict[str, dict[str, dict[str, Any]]] = {"replicate": {}, "aggregate": {}}
    for replicate in REPLICATES:
        by_condition_mapping: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in joined_rows:
            if int(row["replicate"]) != replicate:
                continue
            by_condition_mapping[(row["condition"], row["physical_mapping"])][
                int(row["public_decoder_phase_index"])
            ].append(float(row["logical_response"]))
        for (condition, mapping), phases in by_condition_mapping.items():
            by_scope["replicate"].setdefault(str(replicate), {}).setdefault(condition, {})[mapping] = complex_decode(
                {phase: statistics.fmean(values) for phase, values in phases.items()}
            )
    by_condition_mapping = defaultdict(lambda: defaultdict(list))
    for row in joined_rows:
        by_condition_mapping[(row["condition"], row["physical_mapping"])][int(row["public_decoder_phase_index"])].append(
            float(row["logical_response"])
        )
    for (condition, mapping), phases in by_condition_mapping.items():
        by_scope["aggregate"].setdefault(condition, {})[mapping] = complex_decode(
            {phase: statistics.fmean(values) for phase, values in phases.items()}
        )
    return by_scope


def condition_average(mapping_decodes: dict[str, complex]) -> complex:
    if set(mapping_decodes) != {"map0", "map1"}:
        raise OrbitStatePublicError("condition average needs both mappings")
    return mean_complex([mapping_decodes["map0"], mapping_decodes["map1"]])


def ideal_vectors() -> dict[str, complex]:
    phi = orbit_phi(D_MEMBER)
    q_d = float(QUANTIZATION_SCALE) * complex(math.cos(phi), math.sin(phi))
    return {
        "Z_d": q_d,
        "Z_fold": q_d.conjugate(),
        "Z_polarity_inversion": -q_d,
    }


def estimate_control_gain(control_decodes: dict[str, complex]) -> dict[str, Any]:
    if set(control_decodes) != {"post_projection", "equal_orbit_odd_zero"}:
        raise OrbitStatePublicError("control gain estimator accepts only post/equal control decodes")
    phi = orbit_phi(D_MEMBER)
    z_post = control_decodes["post_projection"]
    z_equal = control_decodes["equal_orbit_odd_zero"]
    g_post = z_post.real / (float(QUANTIZATION_SCALE) * math.cos(phi))
    g_equal = z_equal.real / float(QUANTIZATION_SCALE)
    g_control = 0.5 * (g_post + g_equal)
    agreement = rel_error(g_post, g_equal)
    return {
        "g_post": g_post,
        "g_equal": g_equal,
        "g_control": g_control,
        "control_gain_relative_error": agreement,
        "passed": g_post > 0 and g_equal > 0 and agreement <= RELATIONAL_TOLERANCE,
    }


def gain_covariant_geometry_law(decodes: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    details: dict[str, Any] = {}
    ideal = ideal_vectors()

    def evaluate_scope(scope_name: str, condition_decodes: dict[str, Any], *, diagnostic_only: bool) -> None:
        zd = condition_average(condition_decodes["pre_projection_d"])
        zfold = condition_average(condition_decodes["pre_projection_fold"])
        zpol = condition_average(condition_decodes["source_polarity_inversion_d"])
        control_gain = estimate_control_gain(
            {
                "post_projection": condition_average(condition_decodes["post_projection"]),
                "equal_orbit_odd_zero": condition_average(condition_decodes["equal_orbit_odd_zero"]),
            }
        )
        g_control = control_gain["g_control"]
        predicted = {
            "Z_d": g_control * ideal["Z_d"],
            "Z_fold": g_control * ideal["Z_fold"],
            "Z_polarity_inversion": g_control * ideal["Z_polarity_inversion"],
        }
        observed = {
            "Z_d": zd,
            "Z_fold": zfold,
            "Z_polarity_inversion": zpol,
        }
        vector_errors = {
            name: complex_rel_error(observed[name], predicted[name])
            for name in observed
        }
        fold_conjugacy_error = complex_rel_error(zfold, zd.conjugate())
        polarity_inversion_error = complex_rel_error(zpol, -zd)
        odd_floor_value = min(abs(zd.imag), abs(zfold.imag))
        checks = {
            "control_gain_agreement": control_gain["passed"],
            "Z_d_gain_normalized": vector_errors["Z_d"] <= RELATIONAL_TOLERANCE,
            "Z_fold_gain_normalized": vector_errors["Z_fold"] <= RELATIONAL_TOLERANCE,
            "Z_polarity_gain_normalized": vector_errors["Z_polarity_inversion"] <= RELATIONAL_TOLERANCE,
            "fold_conjugacy": fold_conjugacy_error <= RELATIONAL_TOLERANCE,
            "polarity_inversion": polarity_inversion_error <= RELATIONAL_TOLERANCE,
            "gain_scaled_odd_magnitude": odd_floor_value > g_control * PRIVATE_ODD_SIGNAL_FLOOR,
        }
        details[scope_name] = {
            "diagnostic_only": diagnostic_only,
            "control_gain": control_gain,
            "observed": {name: {"real": value.real, "imag": value.imag} for name, value in observed.items()},
            "predicted": {name: {"real": value.real, "imag": value.imag} for name, value in predicted.items()},
            "gain_normalized_vector_errors": vector_errors,
            "fold_conjugacy_error": fold_conjugacy_error,
            "polarity_inversion_error": polarity_inversion_error,
            "odd_floor_value": odd_floor_value,
            "gain_scaled_odd_floor": g_control * PRIVATE_ODD_SIGNAL_FLOOR,
            "checks": checks,
        }
        if not diagnostic_only:
            for check_name, passed in checks.items():
                if not passed:
                    failures.append(f"{scope_name} {check_name}")

    for replicate, condition_decodes in decodes["replicate"].items():
        evaluate_scope(f"replicate_{replicate}", condition_decodes, diagnostic_only=False)
    evaluate_scope("aggregate", decodes["aggregate"], diagnostic_only=True)
    return {"passed": not failures, "failures": failures, "details": details}


def adjudicate(
    *,
    public_schedule: dict[str, Any],
    receiver_features: dict[str, Any],
    receiver_features_sha256: str,
    private_source_map: dict[str, Any],
    source_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    actual_feature_sha = receiver_feature_digest(receiver_features)
    if actual_feature_sha != receiver_features_sha256:
        return {
            "schema": "GAIN_COVARIANT_ADJUDICATION_V2",
            "result_class": RESULT_NOT_ESTABLISHED,
            "feature_freeze_law": {"passed": False, "expected": receiver_features_sha256, "actual": actual_feature_sha},
            "failed_laws": ["feature_freeze_law"],
        }
    validate_public_schedule(public_schedule)
    validate_private_source_map(public_schedule, private_source_map)
    source_map_by_id = {record["opaque_run_id"]: record for record in private_source_map["records"]}
    joined_rows: list[dict[str, Any]] = []
    for feature_row in receiver_features["rows"]:
        private = source_map_by_id[feature_row["opaque_run_id"]]
        joined = dict(feature_row)
        joined["condition"] = private["condition"]
        joined_rows.append(joined)
    source_law = source_receipt_law(private_source_map, source_receipts)
    source_receipts_by_id_component = {
        (receipt["opaque_run_id"], receipt["component"]): receipt for receipt in source_receipts
    }
    decodes = build_condition_decodes(joined_rows)
    transfer_law = phase_transfer_law(
        joined_rows,
        source_map_by_id=source_map_by_id,
        source_receipts_by_id_component=source_receipts_by_id_component,
        decodes=decodes,
    )
    geometry_law = gain_covariant_geometry_law(decodes)
    failed_laws = []
    for name, law in [
        ("source_formula_law", source_law),
        ("phase_transfer_law", transfer_law),
        ("gain_covariant_private_geometry_law", geometry_law),
    ]:
        if not law["passed"]:
            failed_laws.append(name)
    all_passed = not failed_laws
    replicate_geometry_visible = all(
        all(
            scope["checks"][name]
            for name in [
                "control_gain_agreement",
                "Z_d_gain_normalized",
                "Z_fold_gain_normalized",
                "Z_polarity_gain_normalized",
                "fold_conjugacy",
                "polarity_inversion",
                "gain_scaled_odd_magnitude",
            ]
        )
        for scope_name, scope in geometry_law["details"].items()
        if scope_name.startswith("replicate_")
    )
    if all_passed:
        result_class = RESULT_CONFIRMED
    elif replicate_geometry_visible:
        result_class = RESULT_CANDIDATE
    else:
        result_class = RESULT_NOT_ESTABLISHED
    result = {
        "schema": "GAIN_COVARIANT_ADJUDICATION_V2",
        "allowed_result_classes": ALLOWED_RESULT_CLASSES,
        "forbidden_result_classes": FORBIDDEN_RESULT_CLASSES,
        "result_class": result_class,
        "primary_coordinate": PRIMARY_COORDINATE,
        "receiver_feature_freeze_sha256": receiver_features_sha256,
        "feature_freeze_law": {"passed": True},
        "source_formula_law": source_law,
        "phase_transfer_law": transfer_law,
        "gain_covariant_private_geometry_law": geometry_law,
        "failed_laws": failed_laws,
        "small_wall_crossed_emitted": False,
        "decodes": {
            "aggregate": {
                condition: {
                    mapping: {"real": value.real, "imag": value.imag}
                    for mapping, value in mappings.items()
                }
                for condition, mappings in decodes["aggregate"].items()
            }
        },
    }
    result["adjudication_sha256"] = digest(result)
    return result


def build_mock_capture(kind: str) -> dict[str, Any]:
    schedule = build_public_schedule()
    source_map = build_private_source_map(schedule)
    private_by_id = {record["opaque_run_id"]: record for record in source_map["records"]}
    gain = 1.25
    gain_by_replicate = {0: gain, 1: gain}
    if kind == "unit_gain_confirmation":
        gain_by_replicate = {0: 1.0, 1: 1.0}
    if kind == "different_positive_gain_per_replicate":
        gain_by_replicate = {0: 0.85, 1: 2.10}
    if kind == "zero_coupling":
        gain = 0.0
        gain_by_replicate = {0: 0.0, 1: 0.0}
    raw_records: list[dict[str, Any]] = []
    sentinels: list[dict[str, Any]] = []
    stage_receipts: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for row in schedule["rows"]:
        private = private_by_id[row["opaque_run_id"]]
        q_value = q_theta(private)
        row_gain = gain_by_replicate[int(row["replicate"])]
        signal = row_gain * float(q_value)
        condition = private["condition"]
        phase_index = int(row["public_decoder_phase_index"])
        if kind in {"globally_reversed_physical_sign"}:
            signal = -signal
        if kind in {"negative_control_gain"} and condition in {"post_projection", "equal_orbit_odd_zero"}:
            signal = -abs(row_gain) * float(q_value)
        if kind == "g_post_g_equal_disagreement" and condition == "post_projection":
            signal *= 2.25
        if kind in {"wrong_orbitstate_member", "wrong_fold_member", "gain_normalized_target_failure"} and condition in {
            "pre_projection_d",
            "pre_projection_fold",
        }:
            signal *= 0.40
        if kind == "conjugacy_failure" and condition == "pre_projection_fold":
            signal = -signal
        if kind in {"gain_scaled_odd_floor_failure"} and condition in {"pre_projection_d", "pre_projection_fold"} and phase_index in {1, 3}:
            signal = 0.0
        if kind == "source_polarity_inversion_failure" and condition == "source_polarity_inversion_d":
            signal = abs(signal)
        if kind == "postprojection_odd_leakage" and condition == "post_projection":
            signal += 300.0 * math.sin(phase_radians(int(row["public_decoder_phase_index"])))
        if kind == "sham_first_harmonic_leakage" and condition == "declaration_sham":
            signal += 400.0 * math.cos(phase_radians(int(row["public_decoder_phase_index"])))
        if kind == "query_scramble_leakage" and condition == "query_scramble":
            signal += 400.0 * math.sin(phase_radians(int(row["public_decoder_phase_index"])))
        if kind == "equal_orbit_odd_leakage" and condition == "equal_orbit_odd_zero":
            signal += 320.0 * math.sin(phase_radians(int(row["public_decoder_phase_index"])))
        if kind in {"source_off_leakage", "query_off_leakage"} and condition == kind.replace("_leakage", ""):
            signal = 350.0
        if kind == "near_zero_raw_map0_violation" and condition == "source_off" and row["physical_mapping"] == "map0":
            signal = 350.0
        if kind == "near_zero_raw_map1_violation" and condition == "source_off" and row["physical_mapping"] == "map1":
            signal = 350.0
        if kind == "logical_pair_average_violation" and condition == "source_off":
            signal = 220.0
        if kind == "physical_reversal_average_violation" and condition == "source_off":
            signal = 220.0
        if kind == "decoded_imaginary_null_violation" and condition == "post_projection":
            signal += 350.0 * math.sin(phase_radians(phase_index))
        if kind == "source_off_real_null_violation" and condition == "source_off":
            signal = 350.0 * math.cos(phase_radians(phase_index))
        if kind == "query_off_complex_null_violation" and condition == "query_off":
            signal = 350.0 * (math.cos(phase_radians(phase_index)) + math.sin(phase_radians(phase_index)))
        if kind == "mapping_bias" and row["physical_mapping"] == "map1":
            signal *= 0.55
        if kind == "strong_mapping_failure" and condition == "pre_projection_d" and row["physical_mapping"] == "map1":
            signal *= 0.50
        if kind == "strong_physical_reversal_failure" and condition == "pre_projection_d" and row["physical_mapping"] == "map1":
            signal *= 0.50
        if kind == "strong_sign_failure" and condition == "pre_projection_d" and row["physical_mapping"] == "map0":
            signal = -signal
        if kind == "source_order_bias" and row["source_execution_order"] == "neg_first":
            signal *= 0.55
        if kind == "subcapture_order_bias" and row["subcapture_order"] == "neg_then_pos":
            signal *= 0.55
        if kind == "contradictory_replicates" and int(row["replicate"]) == 1:
            signal = -signal
        pos_value = signal / 2.0
        neg_value = -signal / 2.0
        if kind == "missing_rebaseline" and row["opaque_run_id"].endswith("0"):
            pos_value += 25.0
        for component, value in [("positive", pos_value), ("negative", neg_value)]:
            raw_record = {
                "opaque_group_id": row["opaque_group_id"],
                "opaque_run_id": row["opaque_run_id"],
                "replicate": int(row["replicate"]),
                "public_decoder_phase_index": int(row["public_decoder_phase_index"]),
                "physical_mapping": row["physical_mapping"],
                "component": component,
                "measured_bank": "A"
                if (row["physical_mapping"] == "map0" and component == "positive")
                or (row["physical_mapping"] == "map1" and component == "negative")
                else "B",
                "receiver_core": RECEIVER_CORE if kind != "receiver_on_wrong_core" else 4,
                "pmu_unmultiplexed": kind != "pmu_multiplexing",
                "byte_compare_ok": kind not in {"reused_component_state", "restoration_failure"},
                "event_ids_valid": kind != "event_id_drift",
                "counters": {PRIMARY_COORDINATE: value, "cycles": 1000.0},
            }
            if kind == "classical_public_label_smuggling":
                raw_record["condition"] = private["condition"]
            if kind == "receiver_reads_q_or_work_receipt":
                raw_record["q_theta"] = q_value
            if kind == "nested_private_field_smuggling":
                raw_record["counters"]["q_theta"] = q_value
            if kind == "nested_total_work_smuggling":
                raw_record["counters"]["total_work"] = TOTAL_WORK
            raw_records.append(raw_record)
            for stage in [
                "receiver_full_baseline",
                "pre_sentinels",
                "receiver_rebaseline",
                "source_execute",
                "receiver_measure",
                "receiver_restoration",
                "post_sentinels",
            ]:
                if kind == "missing_rebaseline" and stage == "receiver_rebaseline":
                    continue
                sequence_index = [
                    "receiver_full_baseline",
                    "pre_sentinels",
                    "receiver_rebaseline",
                    "source_execute",
                    "receiver_measure",
                    "receiver_restoration",
                    "post_sentinels",
                ].index(stage)
                stage_receipt = {
                    "opaque_run_id": row["opaque_run_id"],
                    "component": component,
                    "stage": stage,
                    "sequence_index": sequence_index,
                    "receiver_core": RECEIVER_CORE,
                    "byte_digest": "mock_digest",
                    "ok": "false" if kind == "stage_bool_string_false" else kind != "stage_receipt_failure",
                }
                if kind == "empty_stage_digest" and stage in {
                    "receiver_full_baseline",
                    "pre_sentinels",
                    "receiver_rebaseline",
                    "receiver_restoration",
                    "post_sentinels",
                }:
                    stage_receipt["byte_digest"] = ""
                if kind == "float_stage_sequence":
                    stage_receipt["sequence_index"] = float(sequence_index) + 0.9
                if kind == "float_stage_core":
                    stage_receipt["receiver_core"] = 5.9
                stage_receipts.append(stage_receipt)
            if kind != "empty_sentinel_stream":
                sentinel = {
                    "opaque_run_id": row["opaque_run_id"],
                    "component": component,
                    "replicate": int(row["replicate"]),
                    "public_decoder_phase_index": int(row["public_decoder_phase_index"]),
                    "physical_mapping": row["physical_mapping"],
                    "receiver_core": 5.9 if kind == "float_sentinel_core" else RECEIVER_CORE,
                    "pre_digest": "" if kind == "empty_sentinel_digest" else "mock_digest",
                    "post_digest": "" if kind == "empty_sentinel_digest" else "mock_digest",
                    "pre_ok": "false" if kind == "sentinel_bool_string_false" else True,
                    "post_ok": kind not in {"restoration_failure", "sentinel_failure"},
                }
                if kind == "sentinel_digest_mismatch":
                    sentinel["post_digest"] = "different_digest"
                sentinels.append(sentinel)
        work = source_work_for_q(
            q_value,
            source_off_dummy_mode=exact_bool(private["source_off_dummy_mode"], "source_off_dummy_mode"),
        )
        for component in ["positive", "negative"]:
            receipt = {
                "opaque_run_id": row["opaque_run_id"],
                "component": component,
                "source_core": SOURCE_CORE if kind != "source_child_on_wrong_core" else RECEIVER_CORE,
                "source_cpu_before": SOURCE_CORE if kind != "source_child_on_wrong_core" else RECEIVER_CORE,
                "source_cpu_after": SOURCE_CORE if kind != "source_child_on_wrong_core" else RECEIVER_CORE,
                "receiver_feedback_used_to_select_q": False,
                **work,
            }
            if kind in {"source_formula_drift", "wrong_orbitstate_member", "wrong_fold_member", "unequal_total_work"}:
                receipt["positive_work"] = int(receipt["positive_work"]) + 1
            if kind == "unequal_total_work":
                receipt["total_work"] = TOTAL_WORK + 1
            source_receipts.append(receipt)
    if kind == "duplicate_source_receipt" and len(source_receipts) >= 2:
        source_receipts[-1] = dict(source_receipts[0])
    if kind == "float_source_core" and source_receipts:
        source_receipts[0]["source_core"] = SOURCE_CORE + 0.5
    if kind == "fractional_source_work" and source_receipts:
        source_receipts[0]["total_work"] = float(TOTAL_WORK) + 0.5
    if kind == "source_receipt_bool_string_false" and source_receipts:
        source_receipts[0]["receiver_feedback_used_to_select_q"] = "false"
    if kind == "duplicate_sentinel" and sentinels:
        sentinels.append(dict(sentinels[0]))
    if kind == "stage_digest_mismatch" and stage_receipts:
        for receipt in stage_receipts:
            if receipt["stage"] == "post_sentinels":
                receipt["byte_digest"] = "different_digest"
                break
    if kind == "private_map_string_false" and source_map["records"]:
        source_map["records"][0]["polarity_inversion"] = "false"
        source_map["records"][0]["source_off_dummy_mode"] = "false"
    if kind == "source_map_sealed_string_false":
        source_map["sealed_before_live_authorization"] = "false"
    return {
        "public_schedule": schedule,
        "private_source_map": source_map,
        "raw_records": raw_records,
        "sentinels": sentinels,
        "stage_receipts": stage_receipts,
        "source_receipts": source_receipts,
    }


def run_mock(kind: str) -> dict[str, Any]:
    try:
        capture = build_mock_capture(kind)
        features = extract_receiver_features(
            capture["public_schedule"],
            capture["raw_records"],
            capture["sentinels"],
            capture["stage_receipts"],
        )
        if kind == "feature_mutation_after_unblinding":
            frozen = features["receiver_features_sha256"]
            features["rows"][0]["logical_response"] += 1000.0
        else:
            frozen = features["receiver_features_sha256"]
        adjudication = adjudicate(
            public_schedule=capture["public_schedule"],
            receiver_features=features,
            receiver_features_sha256=frozen,
            private_source_map=capture["private_source_map"],
            source_receipts=capture["source_receipts"],
        )
        return {"kind": kind, "raised": False, "adjudication": adjudication}
    except Exception as exc:  # self-test reports expected rejection details
        return {"kind": kind, "raised": True, "error": str(exc)}


def self_test() -> dict[str, Any]:
    mocks = [
        "ideal_arbitrary_positive_gain_confirmation",
        "unit_gain_confirmation",
        "different_positive_gain_per_replicate",
        "negative_control_gain",
        "g_post_g_equal_disagreement",
        "target_derived_gain_smuggling",
        "fold_derived_gain_smuggling",
        "polarity_derived_gain_smuggling",
        "aggregate_gain_used_to_rescue_failed_replicate",
        "gain_normalized_target_failure",
        "conjugacy_failure",
        "polarity_inversion_failure",
        "gain_scaled_odd_floor_failure",
        "one_strong_mapping_failure",
        "one_strong_physical_reversal_failure",
        "one_strong_sign_failure",
        "q_theta_255_partitioned_near_zero",
        "q_theta_256_partitioned_strong",
        "one_near_zero_raw_map0_violation",
        "one_near_zero_raw_map1_violation",
        "logical_pair_average_violation",
        "physical_reversal_average_violation",
        "decoded_imaginary_null_violation",
        "source_off_real_null_violation",
        "query_off_complex_null_violation",
        "post_projection_real_accepted_as_gain_control",
        "equal_orbit_real_accepted_as_gain_control",
        "source_formula_drift",
        "classical_public_label_smuggling",
        "receiver_reads_private_source_map",
        "receiver_reads_q_or_work_receipt",
        "nested_private_field_smuggling",
        "nested_total_work_smuggling",
        "feature_mutation_after_unblinding",
        "duplicate_source_receipt",
        "float_source_core",
        "fractional_source_work",
        "source_receipt_bool_string_false",
        "private_map_string_false",
        "source_map_sealed_string_false",
        "stage_receipt_failure",
        "sentinel_failure",
        "empty_sentinel_stream",
        "duplicate_sentinel",
        "sentinel_digest_mismatch",
        "stage_digest_mismatch",
        "empty_stage_digest",
        "empty_sentinel_digest",
        "stage_bool_string_false",
        "sentinel_bool_string_false",
        "float_stage_sequence",
        "float_stage_core",
        "float_sentinel_core",
        "contradictory_replicates",
        "missing_source_child",
        "source_child_on_wrong_core",
        "receiver_on_wrong_core",
        "reused_component_state",
        "missing_rebaseline",
        "unequal_total_work",
        "pmu_multiplexing",
        "event_id_drift",
        "restoration_failure",
        "forbidden_process_hit",
        "policy_drift",
        "predecessor_evidence_pooling",
        "historical_root_cleanup_attempt",
    ]
    observed: dict[str, Any] = {}
    expected_confirmations = {
        "ideal_arbitrary_positive_gain_confirmation",
        "unit_gain_confirmation",
        "different_positive_gain_per_replicate",
        "post_projection_real_accepted_as_gain_control",
        "equal_orbit_real_accepted_as_gain_control",
        "q_theta_255_partitioned_near_zero",
        "q_theta_256_partitioned_strong",
    }
    expected_rejections = set(mocks) - expected_confirmations
    hard_rejection_mocks = {
        "classical_public_label_smuggling",
        "receiver_reads_private_source_map",
        "receiver_reads_q_or_work_receipt",
        "nested_private_field_smuggling",
        "nested_total_work_smuggling",
        "duplicate_source_receipt",
        "float_source_core",
        "fractional_source_work",
        "source_receipt_bool_string_false",
        "private_map_string_false",
        "source_map_sealed_string_false",
        "stage_receipt_failure",
        "sentinel_failure",
        "empty_sentinel_stream",
        "duplicate_sentinel",
        "sentinel_digest_mismatch",
        "stage_digest_mismatch",
        "empty_stage_digest",
        "empty_sentinel_digest",
        "stage_bool_string_false",
        "sentinel_bool_string_false",
        "float_stage_sequence",
        "float_stage_core",
        "float_sentinel_core",
        "missing_source_child",
        "source_child_on_wrong_core",
        "receiver_on_wrong_core",
        "reused_component_state",
        "missing_rebaseline",
        "unequal_total_work",
        "pmu_multiplexing",
        "event_id_drift",
        "restoration_failure",
        "predecessor_evidence_pooling",
    }
    for mock in mocks:
        actual_kind = {
            "ideal_arbitrary_positive_gain_confirmation": "ideal",
            "post_projection_real_accepted_as_gain_control": "ideal",
            "equal_orbit_real_accepted_as_gain_control": "ideal",
            "q_theta_255_partitioned_near_zero": "ideal",
            "q_theta_256_partitioned_strong": "ideal",
            "target_derived_gain_smuggling": "target_derived_gain_smuggling",
            "fold_derived_gain_smuggling": "fold_derived_gain_smuggling",
            "polarity_derived_gain_smuggling": "polarity_derived_gain_smuggling",
            "aggregate_gain_used_to_rescue_failed_replicate": "contradictory_replicates",
            "polarity_inversion_failure": "source_polarity_inversion_failure",
            "one_strong_mapping_failure": "strong_mapping_failure",
            "one_strong_physical_reversal_failure": "strong_physical_reversal_failure",
            "one_strong_sign_failure": "strong_sign_failure",
            "one_near_zero_raw_map0_violation": "near_zero_raw_map0_violation",
            "one_near_zero_raw_map1_violation": "near_zero_raw_map1_violation",
            "receiver_reads_private_source_map": "classical_public_label_smuggling",
            "missing_source_child": "source_child_on_wrong_core",
            "predecessor_evidence_pooling": "source_formula_drift",
            "forbidden_process_hit": "classical_public_label_smuggling",
            "policy_drift": "source_formula_drift",
            "historical_root_cleanup_attempt": "source_formula_drift",
        }.get(mock, mock)
        if mock in {"target_derived_gain_smuggling", "fold_derived_gain_smuggling", "polarity_derived_gain_smuggling"}:
            try:
                estimate_control_gain(
                    {
                        "post_projection": 1 + 0j,
                        "equal_orbit_odd_zero": 1 + 0j,
                        {
                            "target_derived_gain_smuggling": "pre_projection_d",
                            "fold_derived_gain_smuggling": "pre_projection_fold",
                            "polarity_derived_gain_smuggling": "source_polarity_inversion_d",
                        }[mock]: 1 + 0j,
                    }
                )
                result = {"kind": actual_kind, "raised": False, "error": "smuggled field accepted"}
                passed = False
            except OrbitStatePublicError as exc:
                result = {"kind": actual_kind, "raised": True, "error": str(exc)}
                passed = True
        elif mock == "q_theta_255_partitioned_near_zero":
            passed = abs(255) < 256
            result = {"kind": actual_kind, "partition": "near_zero", "q_theta": 255}
        elif mock == "q_theta_256_partitioned_strong":
            passed = abs(256) >= 256
            result = {"kind": actual_kind, "partition": "strong", "q_theta": 256}
        else:
            result = run_mock(actual_kind)
            if mock in expected_confirmations:
                passed = (not result["raised"]) and result["adjudication"]["result_class"] == RESULT_CONFIRMED
            elif mock in hard_rejection_mocks:
                if result["raised"]:
                    passed = True
                else:
                    failed_laws = set(result["adjudication"].get("failed_laws", []))
                    passed = bool(
                        failed_laws.intersection(
                            {"source_formula_law", "feature_freeze_law", "phase_transfer_law", "gain_covariant_private_geometry_law"}
                        )
                    )
            else:
                passed = result["raised"] or result["adjudication"]["result_class"] != RESULT_CONFIRMED
        observed[mock] = {"passed": passed, "result": result}
    schedule = build_public_schedule()
    source_map = build_private_source_map(schedule)
    artifacts = {
        "public_schedule_sha256": digest(schedule),
        "private_source_map_sha256": digest(source_map),
        "contract_reference_sha256": digest(contract_reference()),
        "schedule_rows": len(schedule["rows"]),
        "private_records": len(source_map["records"]),
        "expected_mapping_leg_records": 144,
        "expected_component_windows": 288,
        "expected_stage_receipts": 2016,
        "expected_source_receipts": 288,
    }
    result = {
        "schema": "GAIN_COVARIANT_SELF_TEST_V2",
        "mocks": observed,
        "expected_rejections": sorted(expected_rejections),
        "artifacts": artifacts,
    }
    result["self_test_passed"] = all(item["passed"] for item in observed.values()) and artifacts[
        "schedule_rows"
    ] == artifacts["expected_mapping_leg_records"]
    result["self_test_sha256"] = digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def write_artifacts(root: Path = HERE) -> dict[str, str]:
    schedule = build_public_schedule()
    source_map = build_private_source_map(schedule)
    reference = contract_reference()
    write_json(root / "GAIN_COVARIANT_PUBLIC_SCHEDULE.json", schedule)
    (root / "GAIN_COVARIANT_PUBLIC_SCHEDULE.tsv").write_text(public_schedule_tsv(schedule), encoding="utf-8")
    schedule_hashes = {
        "json_sha256": sha256_file(root / "GAIN_COVARIANT_PUBLIC_SCHEDULE.json"),
        "tsv_sha256": sha256_file(root / "GAIN_COVARIANT_PUBLIC_SCHEDULE.tsv"),
        "canonical_sha256": digest(schedule),
    }
    (root / "GAIN_COVARIANT_PUBLIC_SCHEDULE.sha256").write_text(
        json.dumps(schedule_hashes, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_json(root / "GAIN_COVARIANT_PRIVATE_SOURCE_MAP.json", source_map)
    (root / "GAIN_COVARIANT_PRIVATE_SOURCE_MAP.sha256").write_text(
        digest(source_map) + "  GAIN_COVARIANT_PRIVATE_SOURCE_MAP.json\n", encoding="utf-8"
    )
    write_json(root / "GAIN_COVARIANT_CONTRACT_REFERENCE.json", reference)
    return {
        "public_schedule_json_sha256": schedule_hashes["json_sha256"],
        "public_schedule_tsv_sha256": schedule_hashes["tsv_sha256"],
        "public_schedule_canonical_sha256": schedule_hashes["canonical_sha256"],
        "public_schedule_sha256_file_sha256": sha256_file(root / "GAIN_COVARIANT_PUBLIC_SCHEDULE.sha256"),
        "private_source_map_sha256": digest(source_map),
        "private_source_map_file_sha256": sha256_file(root / "GAIN_COVARIANT_PRIVATE_SOURCE_MAP.json"),
        "contract_reference_sha256": digest(reference),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_artifacts:
        print(json.dumps(write_artifacts(), indent=2, sort_keys=True))
        return 0
    result = self_test()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["self_test_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
