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
PUBLIC_SEED = "orbitstate-independent-v2-public-seed-4f2e8c41"
PRIMARY_COORDINATE = "change_to_dirty"
SOURCE_CORE = 4
RECEIVER_CORE = 5
RUN_ID = "orbitstate_independent_v2_0"

PUBLIC_SCHEDULE_JSON = HERE / "ORBITSTATE_PUBLIC_SCHEDULE.json"
PUBLIC_SCHEDULE_TSV = HERE / "ORBITSTATE_PUBLIC_SCHEDULE.tsv"
PUBLIC_SCHEDULE_SHA = HERE / "ORBITSTATE_PUBLIC_SCHEDULE.sha256"
PRIVATE_SOURCE_MAP_JSON = HERE / "ORBITSTATE_PRIVATE_SOURCE_MAP.json"
PRIVATE_SOURCE_MAP_SHA = HERE / "ORBITSTATE_PRIVATE_SOURCE_MAP.sha256"
PUBLIC_TRANSDUCER_REFERENCE_JSON = HERE / "PUBLIC_TRANSDUCER_REFERENCE.json"

RESULT_CONFIRMED = "ORBITSTATE_INDEPENDENT_COUPLING_CONFIRMED"
RESULT_CANDIDATE = "ORBITSTATE_INDEPENDENT_COUPLING_CANDIDATE"
RESULT_NOT_ESTABLISHED = "ORBITSTATE_INDEPENDENT_COUPLING_NOT_ESTABLISHED"
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


def public_transducer_reference() -> dict[str, Any]:
    return {
        "source_commit": "4762b5b49b308ae4aca8e141113e4fafe4b0f81e",
        "retained_result_class": "PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_CANDIDATE",
        "engineering_state": [
            "PUBLIC_CHANGE_TO_DIRTY_ODD_TRANSDUCER_REPRODUCED",
            "PUBLIC_MAX_NULL_GATE_NOT_A_VALID_REASON_TO_DELAY_PRIVATE_COUPLING",
        ],
        "primary_coordinate": PRIMARY_COORDINATE,
        "aggregate_change_to_dirty_means": {
            "-1536": -1901.3125,
            "-1024": -1274.625,
            "-512": -633.9375,
            "0": 4.4375,
            "512": 633.8125,
            "1024": 1276.5,
            "1536": 1913.6875,
        },
        "retained_hashes": {
            "raw": "0c3ea99f1cd2a771980012a1f53e22677e48e7cd3778054744af41fa251aac2d",
            "sentinels": "6c8a48219874fff2126d29b2056383ec4caeeb2a519ee83eb418c2679a218fb8",
            "stage_receipts": "cf4586c09a94cdb4de5ef438151e5f8c7bfec76271e13056cca786a98de40992",
            "source_receipts": "402d1e436dafdca7d7beb1172c48b2c3ab57ac6787720174edb4786c23d70bf5",
            "features": "3601718af96514755fff8e548fbf61df667f98587d350de600b6171b813298ae",
            "adjudication": "2b0445c5110aca5b1630064c071ea18f0755aa89829b88618697cb3345cd7f43",
        },
        "public_q0_absolute_bound": PUBLIC_Q0_ABSOLUTE_BOUND,
        "private_odd_signal_floor": PRIVATE_ODD_SIGNAL_FLOOR,
        "bound_freeze_rationale": {
            "aggregate_build_ceiling": 76.0,
            "predeclared_physical_pair_bound": 152.0,
            "largest_retained_heldout_mapping": 100.0,
            "largest_retained_heldout_pair_residual": 127.0,
            "refit_after_private_evidence": False,
        },
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
        "schema": "ORBITSTATE_PUBLIC_SCHEDULE_V2",
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
        "schema": "ORBITSTATE_PRIVATE_SOURCE_MAP_V2",
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
        "schema": "ORBITSTATE_RECEIVER_FEATURES_V2",
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
        if type(receipt.get("receiver_feedback_used_to_select_q")) is not bool or receipt.get(
            "receiver_feedback_used_to_select_q"
        ) is not False:
            failures.append(f"{run_id} q selected using receiver feedback")
    return {"passed": not failures, "failures": failures, "receipt_count": len(source_receipts)}


def phase_transfer_law(
    joined_rows: list[dict[str, Any]],
    *,
    source_map_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    by_condition_phase: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in joined_rows:
        key = (int(row["replicate"]), row["condition"], int(row["public_decoder_phase_index"]))
        by_condition_phase[key][row["physical_mapping"]] = row
    for key, maps in by_condition_phase.items():
        if set(maps) != {"map0", "map1"}:
            failures.append(f"{key} missing mapping leg")
            continue
        left = float(maps["map0"]["logical_response"])
        right = float(maps["map1"]["logical_response"])
        if rel_error(left, right) > RELATIONAL_TOLERANCE:
            failures.append(f"{key} logical mapping invariance")
        if rel_error(float(maps["map0"]["physical_a_minus_b"]), -float(maps["map1"]["physical_a_minus_b"])) > RELATIONAL_TOLERANCE:
            failures.append(f"{key} physical reversal")
        for mapping_name, row in maps.items():
            private = source_map_by_id[row["opaque_run_id"]]
            q_value = q_theta(private)
            response = float(row["logical_response"])
            if abs(q_value) >= 256 and response * q_value <= 0:
                failures.append(f"{key} {mapping_name} sign response does not follow q")
            if abs(q_value) < 256 and abs(response) > PUBLIC_Q0_ABSOLUTE_BOUND:
                failures.append(f"{key} {mapping_name} near-zero response outside fixed bound")
    return {"passed": not failures, "failures": failures}


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


def target_fold_laws(decodes: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    details: dict[str, Any] = {}

    def evaluate_scope(scope_name: str, condition_decodes: dict[str, Any]) -> None:
        zd = condition_average(condition_decodes["pre_projection_d"])
        zfold = condition_average(condition_decodes["pre_projection_fold"])
        zpol = condition_average(condition_decodes["source_polarity_inversion_d"])
        nulls = {
            name: condition_average(condition_decodes[name])
            for name in [
                "source_off",
                "query_off",
                "post_projection",
                "declaration_sham",
                "query_scramble",
                "equal_orbit_odd_zero",
            ]
        }
        expected_phi = orbit_phi(D_MEMBER)
        expected_re = QUANTIZATION_SCALE * math.cos(expected_phi)
        if rel_error(zd.real, expected_re) > RELATIONAL_TOLERANCE:
            failures.append(f"{scope_name} Re(Z_d) geometry")
        if rel_error(zfold.real, expected_re) > RELATIONAL_TOLERANCE:
            failures.append(f"{scope_name} Re(Z_fold) geometry")
        if zd.imag * zfold.imag >= 0:
            failures.append(f"{scope_name} fold imaginary sign")
        if rel_error(abs(zd.imag), abs(zfold.imag)) > RELATIONAL_TOLERANCE:
            failures.append(f"{scope_name} fold imaginary magnitude")
        if min(abs(zd.imag), abs(zfold.imag)) <= PRIVATE_ODD_SIGNAL_FLOOR:
            failures.append(f"{scope_name} private odd signal floor")
        if complex_rel_error(zfold, zd.conjugate()) > RELATIONAL_TOLERANCE:
            failures.append(f"{scope_name} complex conjugate")
        if complex_rel_error(zpol, -zd) > RELATIONAL_TOLERANCE:
            failures.append(f"{scope_name} polarity inversion")
        for name, z_value in nulls.items():
            if abs(z_value.imag) > PUBLIC_Q0_ABSOLUTE_BOUND:
                failures.append(f"{scope_name} {name} imaginary null")
        for name in ["source_off", "query_off", "declaration_sham", "query_scramble"]:
            if abs(nulls[name].real) > PUBLIC_Q0_ABSOLUTE_BOUND or abs(nulls[name].imag) > PUBLIC_Q0_ABSOLUTE_BOUND:
                failures.append(f"{scope_name} {name} complex null")
        details[scope_name] = {
            "Z_d": {"real": zd.real, "imag": zd.imag},
            "Z_fold": {"real": zfold.real, "imag": zfold.imag},
            "Z_polarity": {"real": zpol.real, "imag": zpol.imag},
            "nulls": {name: {"real": z.real, "imag": z.imag} for name, z in nulls.items()},
        }

    for replicate, condition_decodes in decodes["replicate"].items():
        evaluate_scope(f"replicate_{replicate}", condition_decodes)
    evaluate_scope("aggregate", decodes["aggregate"])
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
            "schema": "ORBITSTATE_ADJUDICATION_V2",
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
    transfer_law = phase_transfer_law(joined_rows, source_map_by_id=source_map_by_id)
    decodes = build_condition_decodes(joined_rows)
    geometry_law = target_fold_laws(decodes)
    failed_laws = []
    for name, law in [
        ("source_formula_law", source_law),
        ("phase_transfer_law", transfer_law),
        ("target_fold_geometry_law", geometry_law),
    ]:
        if not law["passed"]:
            failed_laws.append(name)
    all_passed = not failed_laws
    d_aggregate = condition_average(decodes["aggregate"]["pre_projection_d"])
    fold_aggregate = condition_average(decodes["aggregate"]["pre_projection_fold"])
    geometry_visible = (
        min(abs(d_aggregate.imag), abs(fold_aggregate.imag)) > PRIVATE_ODD_SIGNAL_FLOOR
        and d_aggregate.imag * fold_aggregate.imag < 0
    )
    if all_passed:
        result_class = RESULT_CONFIRMED
    elif geometry_visible:
        result_class = RESULT_CANDIDATE
    else:
        result_class = RESULT_NOT_ESTABLISHED
    result = {
        "schema": "ORBITSTATE_ADJUDICATION_V2",
        "allowed_result_classes": ALLOWED_RESULT_CLASSES,
        "forbidden_result_classes": FORBIDDEN_RESULT_CLASSES,
        "result_class": result_class,
        "primary_coordinate": PRIMARY_COORDINATE,
        "receiver_feature_freeze_sha256": receiver_features_sha256,
        "feature_freeze_law": {"passed": True},
        "source_formula_law": source_law,
        "phase_transfer_law": transfer_law,
        "target_fold_geometry_law": geometry_law,
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
    if kind == "zero_coupling":
        gain = 0.0
    raw_records: list[dict[str, Any]] = []
    sentinels: list[dict[str, Any]] = []
    stage_receipts: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for row in schedule["rows"]:
        private = private_by_id[row["opaque_run_id"]]
        q_value = q_theta(private)
        signal = gain * float(q_value)
        if kind in {"globally_reversed_physical_sign"}:
            signal = -signal
        if kind in {"wrong_orbitstate_member", "wrong_fold_member"} and private["condition"] in {
            "pre_projection_d",
            "pre_projection_fold",
        }:
            signal *= 0.40
        if kind == "source_polarity_inversion_failure" and private["condition"] == "source_polarity_inversion_d":
            signal = abs(signal)
        if kind == "postprojection_odd_leakage" and private["condition"] == "post_projection":
            signal += 300.0 * math.sin(phase_radians(int(row["public_decoder_phase_index"])))
        if kind == "sham_first_harmonic_leakage" and private["condition"] == "declaration_sham":
            signal += 400.0 * math.cos(phase_radians(int(row["public_decoder_phase_index"])))
        if kind == "query_scramble_leakage" and private["condition"] == "query_scramble":
            signal += 400.0 * math.sin(phase_radians(int(row["public_decoder_phase_index"])))
        if kind == "equal_orbit_odd_leakage" and private["condition"] == "equal_orbit_odd_zero":
            signal += 320.0 * math.sin(phase_radians(int(row["public_decoder_phase_index"])))
        if kind in {"source_off_leakage", "query_off_leakage"} and private["condition"] == kind.replace("_leakage", ""):
            signal = 350.0
        if kind == "mapping_bias" and row["physical_mapping"] == "map1":
            signal *= 0.55
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
        "ideal_d_fold_conjugate_coupling",
        "globally_reversed_physical_sign",
        "zero_coupling",
        "classical_public_label_smuggling",
        "receiver_reads_private_source_map",
        "receiver_reads_q_or_work_receipt",
        "nested_private_field_smuggling",
        "nested_total_work_smuggling",
        "feature_mutation_after_unblinding",
        "wrong_orbitstate_member",
        "wrong_fold_member",
        "source_formula_drift",
        "duplicate_source_receipt",
        "float_source_core",
        "fractional_source_work",
        "source_receipt_bool_string_false",
        "private_map_string_false",
        "source_map_sealed_string_false",
        "source_polarity_inversion_failure",
        "postprojection_odd_leakage",
        "sham_first_harmonic_leakage",
        "query_scramble_leakage",
        "equal_orbit_odd_leakage",
        "source_off_leakage",
        "query_off_leakage",
        "mapping_bias",
        "source_order_bias",
        "subcapture_order_bias",
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
        "missing_re_encoding",
        "unequal_total_work",
        "pmu_multiplexing",
        "event_id_drift",
        "restoration_failure",
        "historical_evidence_pooling",
        "post_hoc_decoder_fitting",
    ]
    observed: dict[str, Any] = {}
    expected_rejections = set(mocks) - {"ideal_d_fold_conjugate_coupling"}
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
        "missing_re_encoding",
        "unequal_total_work",
        "pmu_multiplexing",
        "event_id_drift",
        "restoration_failure",
        "historical_evidence_pooling",
    }
    for mock in mocks:
        actual_kind = {
            "ideal_d_fold_conjugate_coupling": "ideal",
            "receiver_reads_private_source_map": "classical_public_label_smuggling",
            "missing_source_child": "source_child_on_wrong_core",
            "missing_re_encoding": "source_formula_drift",
            "historical_evidence_pooling": "source_formula_drift",
            "post_hoc_decoder_fitting": "feature_mutation_after_unblinding",
        }.get(mock, mock)
        result = run_mock(actual_kind)
        if mock == "ideal_d_fold_conjugate_coupling":
            passed = (not result["raised"]) and result["adjudication"]["result_class"] == RESULT_CONFIRMED
        elif mock in hard_rejection_mocks:
            if result["raised"]:
                passed = True
            else:
                failed_laws = set(result["adjudication"].get("failed_laws", []))
                passed = bool(
                    failed_laws.intersection(
                        {"source_formula_law", "feature_freeze_law", "phase_transfer_law", "target_fold_geometry_law"}
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
        "public_transducer_reference_sha256": digest(public_transducer_reference()),
        "schedule_rows": len(schedule["rows"]),
        "private_records": len(source_map["records"]),
        "expected_mapping_leg_records": 144,
        "expected_component_windows": 288,
        "expected_stage_receipts": 2016,
        "expected_source_receipts": 288,
    }
    result = {
        "schema": "ORBITSTATE_SELF_TEST_V2",
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
    reference = public_transducer_reference()
    write_json(root / "ORBITSTATE_PUBLIC_SCHEDULE.json", schedule)
    (root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv").write_text(public_schedule_tsv(schedule), encoding="utf-8")
    schedule_hashes = {
        "json_sha256": sha256_file(root / "ORBITSTATE_PUBLIC_SCHEDULE.json"),
        "tsv_sha256": sha256_file(root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv"),
        "canonical_sha256": digest(schedule),
    }
    (root / "ORBITSTATE_PUBLIC_SCHEDULE.sha256").write_text(
        json.dumps(schedule_hashes, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_json(root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json", source_map)
    (root / "ORBITSTATE_PRIVATE_SOURCE_MAP.sha256").write_text(
        digest(source_map) + "  ORBITSTATE_PRIVATE_SOURCE_MAP.json\n", encoding="utf-8"
    )
    write_json(root / "PUBLIC_TRANSDUCER_REFERENCE.json", reference)
    return {
        "public_schedule_json_sha256": schedule_hashes["json_sha256"],
        "public_schedule_tsv_sha256": schedule_hashes["tsv_sha256"],
        "public_schedule_canonical_sha256": schedule_hashes["canonical_sha256"],
        "public_schedule_sha256_file_sha256": sha256_file(root / "ORBITSTATE_PUBLIC_SCHEDULE.sha256"),
        "private_source_map_sha256": digest(source_map),
        "private_source_map_file_sha256": sha256_file(root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json"),
        "public_transducer_reference_sha256": digest(reference),
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
