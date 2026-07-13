#!/usr/bin/env python3
"""Independent-window public transducer V3 schedule and adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent

RUN_ID = "independent_window_transducer_v3_0"
PRIMARY_COORDINATE = "change_to_dirty"
PUBLIC_SEED = "CAT_CAS_INDEPENDENT_WINDOW_TRANSDUCER_V3_PUBLIC_SEED"

SCHEDULE_SCHEMA = "CAT_CAS_INDEPENDENT_WINDOW_PUBLIC_SCHEDULE_V3"
RAW_SCHEMA = "CAT_CAS_INDEPENDENT_WINDOW_RAW_COMPONENT_RECORD_V3"
SENTINEL_SCHEMA = "CAT_CAS_INDEPENDENT_WINDOW_RESTORATION_SENTINEL_V3"
FEATURE_SCHEMA = "CAT_CAS_INDEPENDENT_WINDOW_FEATURES_V3"
ADJUDICATION_SCHEMA = "CAT_CAS_INDEPENDENT_WINDOW_ADJUDICATION_V3"
SELF_TEST_SCHEMA = "CAT_CAS_INDEPENDENT_WINDOW_SELF_TEST_V3"

CLASS_CONFIRMED = "PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_CONFIRMED"
CLASS_CANDIDATE = "PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_CANDIDATE"
CLASS_NOT_ESTABLISHED = "PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_NOT_ESTABLISHED"
ALLOWED_CLASSES = (CLASS_CONFIRMED, CLASS_CANDIDATE, CLASS_NOT_ESTABLISHED)
FORBIDDEN_CLASSES = (
    "BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED",
    "V1_PARTIAL_CONFIRMED",
    "ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE",
    "SMALL_WALL_CROSSED",
)

Q_LADDER = (-1536, -1024, -512, 0, 512, 1024, 1536)
NONZERO_Q = tuple(q for q in Q_LADDER if q != 0)
REPLICATES = (0, 1)
MAPPINGS = (0, 1)
SOURCE_ORDERS = ("positive_first", "negative_first")
SUBCAPTURE_ORDERS = ("positive_subcapture_first", "negative_subcapture_first")
SUBCAPTURE_STAGE_SEQUENCE = (
    "receiver_baseline",
    "pre_sentinel",
    "rebaseline",
    "source_encoding",
    "measure_logical_bank",
    "restore_both_banks",
    "post_sentinel",
)
BASE_WORK = 2048
TOTAL_WORK = 4096
SOURCE_WORK_PER_SUBCAPTURE = 4096
SOURCE_WORK_PER_MAPPING_LEG = 8192
BANK_LINES = 4096
LINE_BYTES = 64
SOURCE_CORE = 4
RECEIVER_CORE = 5
PERM_A = 257
PERM_B = 43

Q0_REPEATS = (0, 1)
NONZERO_PAIRS_PER_REPLICATE = len(NONZERO_Q) * len(SOURCE_ORDERS) * len(SUBCAPTURE_ORDERS)
Q0_PAIRS_PER_REPLICATE = len(SOURCE_ORDERS) * len(SUBCAPTURE_ORDERS) * len(Q0_REPEATS)
PAIRS_PER_REPLICATE = NONZERO_PAIRS_PER_REPLICATE + Q0_PAIRS_PER_REPLICATE
TRIALS_PER_REPLICATE = PAIRS_PER_REPLICATE * len(MAPPINGS)
TOTAL_TRIALS = len(REPLICATES) * TRIALS_PER_REPLICATE
TOTAL_COMPONENT_WINDOWS = TOTAL_TRIALS * 2

COORDINATES = (
    "change_to_dirty",
    "probe_dirty",
    "cycles",
    "duration_ns",
    "change_to_dirty_per_cycle",
    "probe_dirty_per_cycle",
)
ABS_FLOOR_BY_COORDINATE = {
    "change_to_dirty": 1.0,
    "probe_dirty": 1.0,
    "cycles": 1.0,
    "duration_ns": 1.0,
    "change_to_dirty_per_cycle": 1.0e-6,
    "probe_dirty_per_cycle": 1.0e-6,
}
RESTORATION_FLOOR_BY_COORDINATE = {
    "change_to_dirty": 1024.0,
    "probe_dirty": 1024.0,
    "cycles": 10000.0,
    "duration_ns": 10000.0,
    "change_to_dirty_per_cycle": 1.0e-4,
    "probe_dirty_per_cycle": 1.0e-4,
}
GAIN_MULTIPLIER = 3.0
ODDNESS_TOL = 0.25
POINTER_SWAP_TOL = 0.25
ORDER_TOL = 0.25
REPLICATE_TOL = 0.35
SENTINEL_TOL = 0.25

FORBIDDEN_DERIVED_FIELDS = {
    "logical_cycles_delta",
    "logical_change_to_dirty_delta",
    "logical_probe_dirty_delta",
    "logical_duration_ns_delta",
    "physical_a_minus_b_cycles_delta",
    "physical_a_minus_b_change_to_dirty_delta",
    "physical_a_minus_b_probe_dirty_delta",
    "physical_a_minus_b_duration_ns_delta",
}


class IndependentWindowError(AssertionError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def q0_role(q: int, repeat_index: int) -> str:
    if q != 0:
        return "signal"
    if repeat_index == 0:
        return "null_build"
    if repeat_index == 1:
        return "null_test"
    raise IndependentWindowError("q0 repeat out of range")


def source_bool(order: str) -> bool:
    if order not in SOURCE_ORDERS:
        raise IndependentWindowError(f"bad source order: {order}")
    return order == "positive_first"


def subcapture_bool(order: str) -> bool:
    if order not in SUBCAPTURE_ORDERS:
        raise IndependentWindowError(f"bad subcapture order: {order}")
    return order == "positive_subcapture_first"


def build_schedule() -> dict[str, Any]:
    rng = random.Random(PUBLIC_SEED)
    trials: list[dict[str, Any]] = []
    for rep in REPLICATES:
        pairs: list[dict[str, Any]] = []
        cell = 0
        for q in NONZERO_Q:
            for source_order in SOURCE_ORDERS:
                for subcapture_order in SUBCAPTURE_ORDERS:
                    pairs.append(_pair_template(rep, q, 0, "signal", source_order, subcapture_order, cell))
                    cell += 1
        for source_order in SOURCE_ORDERS:
            for subcapture_order in SUBCAPTURE_ORDERS:
                for repeat in Q0_REPEATS:
                    pairs.append(_pair_template(rep, 0, repeat, q0_role(0, repeat), source_order, subcapture_order, cell))
                    cell += 1
        if len(pairs) != PAIRS_PER_REPLICATE:
            raise IndependentWindowError("internal schedule pair count drift")
        rng.shuffle(pairs)
        for pair_index, pair in enumerate(pairs):
            mapping_first = (rep + pair_index + int(pair["q"] != 0) + int(pair["repeat_index"])) & 1
            for leg_index, mapping in enumerate((mapping_first, 1 - mapping_first)):
                trial_index = pair_index * 2 + leg_index
                trials.append(
                    {
                        **pair,
                        "pair_index": pair_index,
                        "leg_index": leg_index,
                        "trial_index": trial_index,
                        "global_trial_index": rep * TRIALS_PER_REPLICATE + trial_index,
                        "mapping": mapping,
                        "mapping_order_first": mapping_first,
                        "logical_positive_physical": "A" if mapping == 0 else "B",
                        "logical_negative_physical": "B" if mapping == 0 else "A",
                        "positive_work": BASE_WORK + int(pair["q"]),
                        "negative_work": BASE_WORK - int(pair["q"]),
                        "source_work_per_subcapture": SOURCE_WORK_PER_SUBCAPTURE,
                        "source_work_per_mapping_leg": SOURCE_WORK_PER_MAPPING_LEG,
                    }
                )
    schedule = {
        "schema_id": SCHEDULE_SCHEMA,
        "run_id": RUN_ID,
        "public_seed": PUBLIC_SEED,
        "primary_coordinate": PRIMARY_COORDINATE,
        "q_ladder": list(Q_LADDER),
        "base_work": BASE_WORK,
        "total_work": TOTAL_WORK,
        "bank_lines": BANK_LINES,
        "line_bytes": LINE_BYTES,
        "line_permutation": {"a": PERM_A, "b": PERM_B, "modulus": BANK_LINES},
        "source_core": SOURCE_CORE,
        "receiver_core": RECEIVER_CORE,
        "replicates": len(REPLICATES),
        "pairs_per_replicate": PAIRS_PER_REPLICATE,
        "trials_per_replicate": TRIALS_PER_REPLICATE,
        "total_mapping_leg_records": TOTAL_TRIALS,
        "total_component_measurement_windows": TOTAL_COMPONENT_WINDOWS,
        "q0_split": "repeat 0 is null_build; repeat 1 is held-out null_test",
        "factors": {
            "source_order": list(SOURCE_ORDERS),
            "subcapture_order": list(SUBCAPTURE_ORDERS),
            "mapping": list(MAPPINGS),
            "fresh_process_replicate": list(REPLICATES),
        },
        "trials": trials,
    }
    schedule["schedule_semantic_sha256"] = schedule_semantic_hash(schedule)
    validate_schedule(schedule)
    schedule["schedule_sha256"] = digest({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    return schedule


def _pair_template(
    rep: int,
    q: int,
    repeat: int,
    role: str,
    source_order: str,
    subcapture_order: str,
    cell_index: int,
) -> dict[str, Any]:
    return {
        "replicate_index": rep,
        "q": q,
        "repeat_index": repeat,
        "q0_role": role,
        "source_order": source_order,
        "source_positive_first": source_bool(source_order),
        "subcapture_order": subcapture_order,
        "positive_subcapture_first": subcapture_bool(subcapture_order),
        "cell_index": cell_index,
        "bank_allocation_id": f"rep{rep}_q{q}_src{SOURCE_ORDERS.index(source_order)}_sub{SUBCAPTURE_ORDERS.index(subcapture_order)}_r{repeat}",
    }


def schedule_semantic_hash(schedule: dict[str, Any]) -> str:
    rows = [
        {
            "replicate_index": int(t["replicate_index"]),
            "pair_index": int(t["pair_index"]),
            "leg_index": int(t["leg_index"]),
            "trial_index": int(t["trial_index"]),
            "q": int(t["q"]),
            "q0_role": str(t["q0_role"]),
            "repeat_index": int(t["repeat_index"]),
            "mapping": int(t["mapping"]),
            "mapping_order_first": int(t["mapping_order_first"]),
            "source_order": str(t["source_order"]),
            "subcapture_order": str(t["subcapture_order"]),
            "bank_allocation_id": str(t["bank_allocation_id"]),
        }
        for t in sorted(schedule["trials"], key=lambda item: int(item["global_trial_index"]))
    ]
    return digest(
        {
            "schema_id": schedule["schema_id"],
            "run_id": schedule["run_id"],
            "q_ladder": schedule["q_ladder"],
            "line_permutation": schedule["line_permutation"],
            "rows": rows,
        }
    )


def schedule_tsv(schedule: dict[str, Any]) -> str:
    rows = []
    for trial in sorted(schedule["trials"], key=lambda item: (int(item["replicate_index"]), int(item["pair_index"]), int(item["leg_index"]))):
        role_code = {"signal": 0, "null_build": 1, "null_test": 2}[str(trial["q0_role"])]
        rows.append(
            "\t".join(
                str(int(value))
                for value in (
                    trial["replicate_index"],
                    trial["pair_index"],
                    trial["leg_index"],
                    trial["trial_index"],
                    trial["repeat_index"],
                    trial["q"],
                    trial["mapping"],
                    trial["mapping_order_first"],
                    1 if trial["source_positive_first"] else 0,
                    1 if trial["positive_subcapture_first"] else 0,
                    role_code,
                )
            )
        )
    return "\n".join(rows) + "\n"


def validate_schedule(schedule: dict[str, Any]) -> None:
    if schedule.get("schema_id") != SCHEDULE_SCHEMA:
        raise IndependentWindowError("schedule schema mismatch")
    if schedule.get("run_id") != RUN_ID:
        raise IndependentWindowError("schedule run ID mismatch")
    trials = schedule.get("trials")
    if not isinstance(trials, list) or len(trials) != TOTAL_TRIALS:
        raise IndependentWindowError("schedule trial count mismatch")
    pair_seen: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    trial_seen: set[tuple[int, int]] = set()
    cell_counts: dict[tuple[int, int, int, str, str], int] = defaultdict(int)
    q0_counts: dict[tuple[int, str], int] = defaultdict(int)
    for trial in trials:
        rep = int(trial["replicate_index"])
        pair = int(trial["pair_index"])
        leg = int(trial["leg_index"])
        trial_index = int(trial["trial_index"])
        q = int(trial["q"])
        mapping = int(trial["mapping"])
        repeat = int(trial["repeat_index"])
        source_order = str(trial["source_order"])
        subcapture_order = str(trial["subcapture_order"])
        role = str(trial["q0_role"])
        if rep not in REPLICATES or pair < 0 or pair >= PAIRS_PER_REPLICATE:
            raise IndependentWindowError("replicate or pair index out of range")
        if leg not in MAPPINGS or mapping not in MAPPINGS:
            raise IndependentWindowError("leg or mapping out of range")
        if (rep, trial_index) in trial_seen or trial_index < 0 or trial_index >= TRIALS_PER_REPLICATE:
            raise IndependentWindowError("trial index drift")
        trial_seen.add((rep, trial_index))
        if q not in Q_LADDER:
            raise IndependentWindowError("q outside frozen ladder")
        if source_order not in SOURCE_ORDERS or subcapture_order not in SUBCAPTURE_ORDERS:
            raise IndependentWindowError("factor order outside frozen levels")
        if role != q0_role(q, repeat):
            raise IndependentWindowError("q0 role drift")
        if int(trial["positive_work"]) != BASE_WORK + q or int(trial["negative_work"]) != BASE_WORK - q:
            raise IndependentWindowError("source work drift")
        if int(trial["source_work_per_subcapture"]) != SOURCE_WORK_PER_SUBCAPTURE:
            raise IndependentWindowError("subcapture source work drift")
        if int(trial["source_work_per_mapping_leg"]) != SOURCE_WORK_PER_MAPPING_LEG:
            raise IndependentWindowError("mapping-leg source work drift")
        pair_seen[(rep, pair)].append(trial)
        cell_counts[(rep, q, repeat, source_order, subcapture_order)] += 1
        if q == 0 and leg == 0:
            q0_counts[(rep, role)] += 1
    for rep in REPLICATES:
        for pair in range(PAIRS_PER_REPLICATE):
            legs = sorted(pair_seen[(rep, pair)], key=lambda item: int(item["leg_index"]))
            if len(legs) != 2 or {int(item["mapping"]) for item in legs} != {0, 1}:
                raise IndependentWindowError("mapping pair is not a two-leg crossover")
            shared = ("q", "q0_role", "repeat_index", "source_order", "subcapture_order", "bank_allocation_id", "mapping_order_first")
            for key in shared:
                if legs[0][key] != legs[1][key]:
                    raise IndependentWindowError(f"mapping pair does not share {key}")
            if int(legs[0]["mapping_order_first"]) != int(legs[0]["mapping"]):
                raise IndependentWindowError("mapping order does not match first leg")
        for q in NONZERO_Q:
            for source_order in SOURCE_ORDERS:
                for subcapture_order in SUBCAPTURE_ORDERS:
                    if cell_counts[(rep, q, 0, source_order, subcapture_order)] != 2:
                        raise IndependentWindowError("nonzero cell count mismatch")
        for source_order in SOURCE_ORDERS:
            for subcapture_order in SUBCAPTURE_ORDERS:
                for repeat in Q0_REPEATS:
                    if cell_counts[(rep, 0, repeat, source_order, subcapture_order)] != 2:
                        raise IndependentWindowError("q0 cell count mismatch")
        if q0_counts[(rep, "null_build")] != 4 or q0_counts[(rep, "null_test")] != 4:
            raise IndependentWindowError("q0 build/test split mismatch")


def write_schedule_artifacts(root: Path = HERE) -> dict[str, str]:
    schedule = build_schedule()
    json_path = root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json"
    tsv_path = root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv"
    sha_path = root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256"
    write_json(json_path, schedule)
    tsv_path.write_text(schedule_tsv(schedule), encoding="utf-8", newline="\n")
    schedule_hash = sha256_file(json_path)
    sha_path.write_text(schedule_hash + "\n", encoding="utf-8", newline="\n")
    return {
        "schedule_json_sha256": schedule_hash,
        "schedule_tsv_sha256": sha256_file(tsv_path),
        "schedule_semantic_sha256": schedule["schedule_semantic_sha256"],
    }


def rel(left: float, right: float, floor: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


def oddness(left: float, right: float, floor: float) -> float:
    return abs(left + right) / max(abs(left), abs(right), floor)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def coord_floor(coord: str) -> float:
    return ABS_FLOOR_BY_COORDINATE[coord]


def gain_floor(coord: str, null_ceiling: float) -> float:
    return max(GAIN_MULTIPLIER * null_ceiling, coord_floor(coord))


def _window_ok(record: dict[str, Any], prefix: str) -> bool:
    try:
        ids = record[f"{prefix}_event_ids"]
        return (
            record[f"{prefix}_opened"] is True
            and record[f"{prefix}_read_ok"] is True
            and record[f"{prefix}_event_order_ok"] is True
            and record[f"{prefix}_unmultiplexed"] is True
            and int(record[f"{prefix}_open_errno"]) == 0
            and int(record[f"{prefix}_read_errno"]) == 0
            and int(record[f"{prefix}_time_enabled"]) > 0
            and int(record[f"{prefix}_time_enabled"]) == int(record[f"{prefix}_time_running"])
            and int(record[f"{prefix}_cpu_before"]) == RECEIVER_CORE
            and int(record[f"{prefix}_cpu_after"]) == RECEIVER_CORE
            and isinstance(ids, list)
            and len(ids) == 3
        )
    except (KeyError, TypeError, ValueError):
        return False


def _component_value(record: dict[str, Any], component: str, coord: str) -> float:
    if coord == "change_to_dirty":
        return float(record[f"{component}_measure_change_to_dirty"])
    if coord == "probe_dirty":
        return float(record[f"{component}_measure_probe_dirty"])
    if coord == "cycles":
        return float(record[f"{component}_measure_cycles"])
    if coord == "duration_ns":
        return float(record[f"{component}_measure_duration_ns"])
    if coord == "change_to_dirty_per_cycle":
        cycles = float(record[f"{component}_measure_cycles"])
        return float(record[f"{component}_measure_change_to_dirty"]) / cycles if cycles else 0.0
    if coord == "probe_dirty_per_cycle":
        cycles = float(record[f"{component}_measure_cycles"])
        return float(record[f"{component}_measure_probe_dirty"]) / cycles if cycles else 0.0
    raise KeyError(coord)


def _sentinel_value(sentinel: dict[str, Any], component: str, phase: str, bank: str, coord: str) -> float:
    prefix = f"{component}_{phase}_{bank}"
    if coord == "change_to_dirty":
        return float(sentinel[f"{prefix}_change_to_dirty"])
    if coord == "probe_dirty":
        return float(sentinel[f"{prefix}_probe_dirty"])
    if coord == "cycles":
        return float(sentinel[f"{prefix}_cycles"])
    if coord == "duration_ns":
        return float(sentinel[f"{prefix}_duration_ns"])
    if coord == "change_to_dirty_per_cycle":
        cycles = float(sentinel[f"{prefix}_cycles"])
        return float(sentinel[f"{prefix}_change_to_dirty"]) / cycles if cycles else 0.0
    if coord == "probe_dirty_per_cycle":
        cycles = float(sentinel[f"{prefix}_cycles"])
        return float(sentinel[f"{prefix}_probe_dirty"]) / cycles if cycles else 0.0
    raise KeyError(coord)


def _window_signature(record: dict[str, Any], prefix: str) -> tuple[Any, ...]:
    return (
        tuple(record.get(f"{prefix}_event_ids", [])),
        record.get(f"{prefix}_cycles"),
        record.get(f"{prefix}_change_to_dirty"),
        record.get(f"{prefix}_probe_dirty"),
        record.get(f"{prefix}_duration_ns"),
        record.get(f"{prefix}_time_enabled"),
    )


def validate_raw_against_schedule(
    schedule: dict[str, Any],
    raw_records: list[dict[str, Any]],
    sentinels: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_schedule(schedule)
    trial_map = {
        (int(trial["replicate_index"]), int(trial["trial_index"])): trial
        for trial in schedule["trials"]
    }
    sentinel_map = {
        (int(s["replicate_index"]), int(s["trial_index"])): s
        for s in sentinels
        if s.get("schema_id") == SENTINEL_SCHEMA
    }
    failures: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    receipt_seen: dict[str, tuple[int, int]] = {}
    if len(raw_records) != TOTAL_TRIALS:
        failures.append({"law": "raw_record_count", "actual": len(raw_records), "expected": TOTAL_TRIALS})
    if len(sentinels) != TOTAL_TRIALS:
        failures.append({"law": "sentinel_record_count", "actual": len(sentinels), "expected": TOTAL_TRIALS})
    for record in raw_records:
        if record.get("schema_id") != RAW_SCHEMA:
            failures.append({"law": "raw_schema", "schema": record.get("schema_id")})
            continue
        key = (int(record["replicate_index"]), int(record["trial_index"]))
        seen.add(key)
        trial = trial_map.get(key)
        sentinel = sentinel_map.get(key)
        if trial is None:
            failures.append({"law": "raw_schedule_membership", "key": key})
            continue
        if sentinel is None:
            failures.append({"law": "sentinel_pairing", "key": key})
            continue
        if any(field in record for field in FORBIDDEN_DERIVED_FIELDS):
            failures.append({"law": "forged_or_precomputed_derived_field", "key": key})
        for field in (
            "pair_index",
            "leg_index",
            "repeat_index",
            "q",
            "mapping",
            "mapping_order_first",
            "source_order",
            "subcapture_order",
            "q0_role",
            "bank_allocation_id",
        ):
            if record.get(field) != trial.get(field):
                failures.append({"law": "schedule_field_drift", "field": field, "key": key, "actual": record.get(field), "expected": trial.get(field)})
        if record.get("logical_positive_physical") != trial.get("logical_positive_physical"):
            failures.append({"law": "mapping_role_drift", "field": "logical_positive_physical", "key": key})
        if record.get("logical_negative_physical") != trial.get("logical_negative_physical"):
            failures.append({"law": "mapping_role_drift", "field": "logical_negative_physical", "key": key})
        if int(record.get("positive_source_total_work", -1)) != SOURCE_WORK_PER_SUBCAPTURE:
            failures.append({"law": "positive_subcapture_source_work", "key": key})
        if int(record.get("negative_source_total_work", -1)) != SOURCE_WORK_PER_SUBCAPTURE:
            failures.append({"law": "negative_subcapture_source_work", "key": key})
        if int(record.get("mapping_leg_source_work", -1)) != SOURCE_WORK_PER_MAPPING_LEG:
            failures.append({"law": "mapping_leg_source_work", "key": key})
        if int(record.get("positive_source_positive_work", -1)) != BASE_WORK + int(record["q"]):
            failures.append({"law": "positive_subcapture_positive_work", "key": key})
        if int(record.get("negative_source_positive_work", -1)) != BASE_WORK + int(record["q"]):
            failures.append({"law": "negative_subcapture_positive_work", "key": key})
        if int(record.get("positive_source_negative_work", -1)) != BASE_WORK - int(record["q"]):
            failures.append({"law": "positive_subcapture_negative_work", "key": key})
        if int(record.get("negative_source_negative_work", -1)) != BASE_WORK - int(record["q"]):
            failures.append({"law": "negative_subcapture_negative_work", "key": key})
        if record.get("line_permutation_a") != PERM_A or record.get("line_permutation_b") != PERM_B:
            failures.append({"law": "different_line_permutation", "key": key})
        if not record.get("positive_prefix_unique") or not record.get("negative_prefix_unique"):
            failures.append({"law": "duplicate_prefix_lines", "key": key})
        expected_order = ["positive", "negative"] if subcapture_bool(str(record["subcapture_order"])) else ["negative", "positive"]
        if record.get("subcapture_execution_order") != expected_order:
            failures.append({"law": "component_order_drift", "key": key, "actual": record.get("subcapture_execution_order"), "expected": expected_order})
        if int(record.get("positive_sequence_index", -1)) != expected_order.index("positive"):
            failures.append({"law": "positive_sequence_index_drift", "key": key})
        if int(record.get("negative_sequence_index", -1)) != expected_order.index("negative"):
            failures.append({"law": "negative_sequence_index_drift", "key": key})
        all_receipt_ids = []
        for component in ("positive", "negative"):
            if record.get(f"{component}_stage_sequence") != list(SUBCAPTURE_STAGE_SEQUENCE):
                failures.append(
                    {
                        "law": f"{component}_stage_sequence_drift",
                        "key": key,
                        "actual": record.get(f"{component}_stage_sequence"),
                        "expected": list(SUBCAPTURE_STAGE_SEQUENCE),
                    }
                )
            receipt_fields = (
                "baseline_receipt_id",
                "pre_sentinel_receipt_id",
                "rebaseline_receipt_id",
                "source_receipt_id",
                "measure_receipt_id",
                "restore_receipt_id",
                "post_sentinel_receipt_id",
            )
            component_receipts = []
            for receipt_field in receipt_fields:
                receipt = record.get(f"{component}_{receipt_field}")
                if not isinstance(receipt, str) or not receipt:
                    failures.append({"law": f"{component}_missing_{receipt_field}", "key": key})
                else:
                    component_receipts.append(receipt)
                    all_receipt_ids.append(receipt)
                    previous = receipt_seen.get(receipt)
                    if previous is not None and previous != key:
                        failures.append({"law": "receipt_reused_across_trials", "key": key, "previous_key": previous})
                    elif previous is None:
                        receipt_seen[receipt] = key
            if len(set(component_receipts)) != len(component_receipts):
                failures.append({"law": f"{component}_receipt_reuse", "key": key})
            if int(record.get(f"{component}_baseline_rc", -1)) != 0:
                failures.append({"law": f"{component}_missing_initial_baseline", "key": key})
            if int(record.get(f"{component}_rebaseline_rc", -1)) != 0:
                failures.append({"law": f"{component}_missing_rebaseline", "key": key})
            if int(record.get(f"{component}_source_rc", -1)) != 0:
                failures.append({"law": f"{component}_missing_source_encoding", "key": key})
            if int(record.get(f"{component}_restore_rc", -1)) != 0:
                failures.append({"law": f"{component}_restore_rc", "key": key})
            if int(record.get(f"{component}_measure_rc", -1)) != 0:
                failures.append({"law": f"{component}_measure_rc", "key": key})
            if not _window_ok(record, f"{component}_measure"):
                failures.append({"law": f"{component}_measure_pmu_window", "key": key})
            for phase in ("pre_a", "pre_b", "post_a", "post_b"):
                if not _window_ok(sentinel, f"{component}_{phase}"):
                    failures.append({"law": f"{component}_{phase}_sentinel_window", "key": key})
        if _window_signature(record, "positive_measure") == _window_signature(record, "negative_measure"):
            failures.append({"law": "same_raw_window_reused_twice", "key": key})
        if len(set(all_receipt_ids)) != len(all_receipt_ids):
            failures.append({"law": "cross_component_receipt_reuse", "key": key})
        if not bool(record.get("byte_compare_passed")) or not bool(sentinel.get("byte_compare_passed")):
            failures.append({"law": "byte_restoration_failure", "key": key})
        if not bool(record.get("restoration_passed")) or not bool(sentinel.get("bytes_unchanged")):
            failures.append({"law": "ownership_sentinel_restoration_failure", "key": key})
        if not bool(record.get("trial_ok")):
            failures.append({"law": "trial_ok_false", "key": key})
    missing = sorted(set(trial_map) - seen)
    if missing:
        failures.append({"law": "missing_raw_trials", "keys": missing[:8], "missing_count": len(missing)})
    return {
        "schedule_matched": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "raw_record_count": len(raw_records),
        "sentinel_record_count": len(sentinels),
    }


def extract_features(
    schedule: dict[str, Any],
    raw_records: list[dict[str, Any]],
    sentinels: list[dict[str, Any]],
) -> dict[str, Any]:
    integrity = validate_raw_against_schedule(schedule, raw_records, sentinels)
    sentinel_map = {
        (int(s["replicate_index"]), int(s["trial_index"])): s
        for s in sentinels
        if s.get("schema_id") == SENTINEL_SCHEMA
    }
    features = []
    for record in raw_records:
        if record.get("schema_id") != RAW_SCHEMA:
            continue
        key = (int(record["replicate_index"]), int(record["trial_index"]))
        sentinel = sentinel_map.get(key, {})
        coordinates = {}
        for coord in COORDINATES:
            positive = _component_value(record, "positive", coord)
            negative = _component_value(record, "negative", coord)
            logical = positive - negative
            if int(record["mapping"]) == 0:
                physical = positive - negative
            else:
                physical = negative - positive
            sentinel_values = []
            try:
                for component in ("positive", "negative"):
                    for phase in ("pre", "post"):
                        a_val = _sentinel_value(sentinel, component, phase, "a", coord)
                        b_val = _sentinel_value(sentinel, component, phase, "b", coord)
                        sentinel_values.append(a_val - b_val)
            except (KeyError, TypeError, ValueError):
                sentinel_values.append(float("inf"))
            coordinates[coord] = {
                "logical": logical,
                "physical_a_minus_b": physical,
                "positive_component": positive,
                "negative_component": negative,
                "sentinel_variation": max(abs(value) for value in sentinel_values) if sentinel_values else 0.0,
                "sentinel_values": sentinel_values,
            }
        features.append(
            {
                "replicate_index": int(record["replicate_index"]),
                "pair_index": int(record["pair_index"]),
                "leg_index": int(record["leg_index"]),
                "trial_index": int(record["trial_index"]),
                "repeat_index": int(record["repeat_index"]),
                "q": int(record["q"]),
                "q0_role": str(record["q0_role"]),
                "mapping": int(record["mapping"]),
                "mapping_order_first": int(record["mapping_order_first"]),
                "source_order": str(record["source_order"]),
                "subcapture_order": str(record["subcapture_order"]),
                "bank_allocation_id": str(record["bank_allocation_id"]),
                "logical_positive_physical": str(record["logical_positive_physical"]),
                "logical_negative_physical": str(record["logical_negative_physical"]),
                "source_receipts": {
                    "positive_subcapture": {
                        "positive_work": int(record.get("positive_source_positive_work", -1)),
                        "negative_work": int(record.get("positive_source_negative_work", -1)),
                        "total_work": int(record.get("positive_source_total_work", -1)),
                    },
                    "negative_subcapture": {
                        "positive_work": int(record.get("negative_source_positive_work", -1)),
                        "negative_work": int(record.get("negative_source_negative_work", -1)),
                        "total_work": int(record.get("negative_source_total_work", -1)),
                    },
                    "mapping_leg_source_work": int(record.get("mapping_leg_source_work", -1)),
                },
                "trial_ok": bool(record.get("trial_ok")),
                "restoration_passed": bool(record.get("restoration_passed")),
                "byte_compare_passed": bool(record.get("byte_compare_passed")),
                "coordinates": coordinates,
            }
        )
    payload = {
        "schema_id": FEATURE_SCHEMA,
        "run_id": RUN_ID,
        "source_kind": "fresh_independent_window_v3",
        "fresh_only": True,
        "schedule_sha256": schedule.get("schedule_sha256"),
        "schedule_semantic_sha256": schedule.get("schedule_semantic_sha256"),
        "raw_record_count": len(raw_records),
        "sentinel_record_count": len(sentinels),
        "unique_trial_key_count": len({(row["replicate_index"], row["trial_index"]) for row in features}),
        "component_measurement_window_count": len(features) * 2,
        "integrity": integrity,
        "trial_features": features,
        "raw_records_sha256": digest(raw_records),
        "restoration_sentinels_sha256": digest(sentinels),
    }
    payload["fresh_evidence_seal_sha256"] = digest(
        {
            "run_id": payload["run_id"],
            "schedule": payload["schedule_sha256"],
            "raw": payload["raw_records_sha256"],
            "sentinels": payload["restoration_sentinels_sha256"],
            "kind": payload["source_kind"],
        }
    )
    payload["features_sha256"] = digest({k: v for k, v in payload.items() if k != "features_sha256"})
    return payload


def select_rows(
    rows: list[dict[str, Any]],
    *,
    q: int | None = None,
    mapping: int | None = None,
    source_order: str | None = None,
    subcapture_order: str | None = None,
    mapping_order_first: int | None = None,
    q0_role: str | None = None,
    replicate_index: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if q is not None and int(row["q"]) != q:
            continue
        if mapping is not None and int(row["mapping"]) != mapping:
            continue
        if source_order is not None and str(row["source_order"]) != source_order:
            continue
        if subcapture_order is not None and str(row["subcapture_order"]) != subcapture_order:
            continue
        if mapping_order_first is not None and int(row["mapping_order_first"]) != mapping_order_first:
            continue
        if q0_role is not None and str(row["q0_role"]) != q0_role:
            continue
        if replicate_index is not None and int(row["replicate_index"]) != replicate_index:
            continue
        selected.append(row)
    return selected


def coord_value(row: dict[str, Any], coord: str, *, physical: bool = False) -> float:
    key = "physical_a_minus_b" if physical else "logical"
    return float(row["coordinates"][coord][key])


def mean_coord(rows: list[dict[str, Any]], coord: str, *, physical: bool = False, **kwargs: Any) -> float:
    return mean([coord_value(row, coord, physical=physical) for row in select_rows(rows, **kwargs)])


def null_ceiling(rows: list[dict[str, Any]], coord: str, *, include_fresh: bool) -> dict[str, float]:
    floor = coord_floor(coord)
    build_rows = select_rows(rows, q=0, q0_role="null_build")
    reps = sorted({int(row["replicate_index"]) for row in rows})
    q0_values = [abs(coord_value(row, coord)) for row in build_rows]
    mapping_residuals = []
    source_residuals = []
    subcapture_residuals = []
    mapping_order_residuals = []
    for rep in reps:
        rep_rows = [row for row in build_rows if int(row["replicate_index"]) == rep]
        mapping_residuals.append(abs(mean_coord(rep_rows, coord, mapping=0) - mean_coord(rep_rows, coord, mapping=1)))
        source_residuals.append(abs(mean_coord(rep_rows, coord, source_order="positive_first") - mean_coord(rep_rows, coord, source_order="negative_first")))
        subcapture_residuals.append(abs(mean_coord(rep_rows, coord, subcapture_order="positive_subcapture_first") - mean_coord(rep_rows, coord, subcapture_order="negative_subcapture_first")))
        mapping_order_residuals.append(abs(mean_coord(rep_rows, coord, mapping_order_first=0) - mean_coord(rep_rows, coord, mapping_order_first=1)))
    fresh_variation = 0.0
    if include_fresh and len(reps) >= 2:
        per_rep = [mean([coord_value(row, coord) for row in build_rows if int(row["replicate_index"]) == rep]) for rep in reps]
        fresh_variation = max(per_rep) - min(per_rep) if per_rep else 0.0
    sentinel_variation = max([abs(float(row["coordinates"][coord]["sentinel_variation"])) for row in build_rows], default=0.0)
    parts = {
        "q0_build_abs": max(q0_values, default=0.0),
        "q0_build_mapping_residual": max(mapping_residuals, default=0.0),
        "q0_build_source_order_residual": max(source_residuals, default=0.0),
        "q0_build_subcapture_order_residual": max(subcapture_residuals, default=0.0),
        "q0_build_mapping_order_residual": max(mapping_order_residuals, default=0.0),
        "restoration_sentinel_variation": sentinel_variation,
        "fresh_process_variation": fresh_variation,
        "absolute_floor": floor,
    }
    parts["complete_null_ceiling"] = max(parts.values())
    return parts


def pair_rows(rows: list[dict[str, Any]], *, include_zero: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not include_zero and int(row["q"]) == 0:
            continue
        groups[(int(row["replicate_index"]), int(row["pair_index"]))].append(row)
    pairs = []
    for (_, _), items in groups.items():
        if len(items) != 2:
            continue
        by_mapping = {int(row["mapping"]): row for row in items}
        if set(by_mapping) != {0, 1}:
            continue
        first = items[0]
        pairs.append(
            {
                "replicate_index": int(first["replicate_index"]),
                "pair_index": int(first["pair_index"]),
                "q": int(first["q"]),
                "q0_role": str(first["q0_role"]),
                "source_order": str(first["source_order"]),
                "subcapture_order": str(first["subcapture_order"]),
                "mapping_order_first": int(first["mapping_order_first"]),
                "mapping0": by_mapping[0],
                "mapping1": by_mapping[1],
            }
        )
    return pairs


def q0_null_semantics(rows: list[dict[str, Any]], coord: str, null: dict[str, float]) -> dict[str, Any]:
    ceiling = null["complete_null_ceiling"]
    pair_bound = ceiling
    physical_bound = 2.0 * ceiling
    heldout_rows = select_rows(rows, q=0, q0_role="null_test")
    max_mapping_abs = max([abs(coord_value(row, coord)) for row in heldout_rows], default=0.0)
    max_f0_abs = max_mapping_abs
    max_pair_residual = 0.0
    max_physical_pair_sum = 0.0
    pair_entries = []
    for pair in pair_rows(rows, include_zero=True):
        if int(pair["q"]) != 0 or pair["q0_role"] != "null_test":
            continue
        m0 = pair["mapping0"]
        m1 = pair["mapping1"]
        residual = coord_value(m0, coord) - coord_value(m1, coord)
        physical_sum = coord_value(m0, coord, physical=True) + coord_value(m1, coord, physical=True)
        pair_entry = {
            "replicate_index": pair["replicate_index"],
            "pair_index": pair["pair_index"],
            "source_order": pair["source_order"],
            "subcapture_order": pair["subcapture_order"],
            "mapping0": coord_value(m0, coord),
            "mapping1": coord_value(m1, coord),
            "logical_pair_residual": residual,
            "physical_pair_sum": physical_sum,
        }
        pair_entries.append(pair_entry)
        max_pair_residual = max(max_pair_residual, abs(residual))
        max_physical_pair_sum = max(max_physical_pair_sum, abs(physical_sum))
    return {
        "ceiling": ceiling,
        "pair_null_bound": pair_bound,
        "physical_pair_sum_bound": physical_bound,
        "max_heldout_mapping_abs": max_mapping_abs,
        "max_heldout_f0_abs": max_f0_abs,
        "max_heldout_logical_pair_residual": max_pair_residual,
        "max_heldout_physical_pair_sum": max_physical_pair_sum,
        "heldout_mapping_inside_null": max_mapping_abs <= ceiling,
        "heldout_f0_inside_null": max_f0_abs <= ceiling,
        "heldout_logical_pair_inside_bound": max_pair_residual <= pair_bound,
        "heldout_physical_pair_inside_bound": max_physical_pair_sum <= physical_bound,
        "heldout_pairs": pair_entries,
        "null_test_used_to_construct_ceiling": False,
    }


def paired_crossover_law(rows: list[dict[str, Any]], coord: str, null: dict[str, float]) -> dict[str, Any]:
    floor = coord_floor(coord)
    entries = []
    max_logical: dict[str, Any] | None = None
    max_physical: dict[str, Any] | None = None
    for pair in pair_rows(rows, include_zero=False):
        m0 = pair["mapping0"]
        m1 = pair["mapping1"]
        logical0 = coord_value(m0, coord)
        logical1 = coord_value(m1, coord)
        physical0 = coord_value(m0, coord, physical=True)
        physical1 = coord_value(m1, coord, physical=True)
        logical_den = max(abs(logical0), abs(logical1), floor)
        physical_den = max(abs(physical0), abs(physical1), floor)
        entry = {
            "q": pair["q"],
            "replicate_index": pair["replicate_index"],
            "pair_index": pair["pair_index"],
            "source_order": pair["source_order"],
            "subcapture_order": pair["subcapture_order"],
            "mapping_order_first": pair["mapping_order_first"],
            "logical_mapping0": logical0,
            "logical_mapping1": logical1,
            "logical_pair_residual": logical0 - logical1,
            "logical_denominator": logical_den,
            "logical_relative_error": abs(logical0 - logical1) / logical_den,
            "physical_mapping0": physical0,
            "physical_mapping1": physical1,
            "physical_pair_sum_residual": physical0 + physical1,
            "physical_denominator": physical_den,
            "physical_relative_error": abs(physical0 + physical1) / physical_den,
        }
        entries.append(entry)
        if max_logical is None or entry["logical_relative_error"] > max_logical["logical_relative_error"]:
            max_logical = entry
        if max_physical is None or entry["physical_relative_error"] > max_physical["physical_relative_error"]:
            max_physical = entry
    q0 = q0_null_semantics(rows, coord, null)
    logical_max = max([entry["logical_relative_error"] for entry in entries], default=0.0)
    physical_max = max([entry["physical_relative_error"] for entry in entries], default=0.0)
    return {
        "logical_mapping_invariance_law": (
            logical_max <= POINTER_SWAP_TOL
            and q0["heldout_mapping_inside_null"]
            and q0["heldout_logical_pair_inside_bound"]
            and q0["heldout_f0_inside_null"]
        ),
        "physical_mapping_reversal_law": physical_max <= POINTER_SWAP_TOL and q0["heldout_physical_pair_inside_bound"],
        "max_nonzero_logical_relative_error": logical_max,
        "max_nonzero_physical_relative_error": physical_max,
        "max_logical_entry": max_logical,
        "max_physical_entry": max_physical,
        "q0_null_semantics": q0,
    }


def factor_invariance_law(rows: list[dict[str, Any]], coord: str, factor: str) -> dict[str, Any]:
    levels = SOURCE_ORDERS if factor == "source_order" else SUBCAPTURE_ORDERS
    errors = []
    floor = coord_floor(coord)
    for q in NONZERO_Q:
        left = mean([coord_value(row, coord) for row in rows if int(row["q"]) == q and row[factor] == levels[0]])
        right = mean([coord_value(row, coord) for row in rows if int(row["q"]) == q and row[factor] == levels[1]])
        denominator = max(abs(left), abs(right), floor)
        errors.append(
            {
                "q": q,
                "left_level": levels[0],
                "right_level": levels[1],
                "left_mean": left,
                "right_mean": right,
                "absolute_residual": abs(left - right),
                "denominator": denominator,
                "relative_error": abs(left - right) / denominator,
            }
        )
    max_entry = max(errors, key=lambda item: item["relative_error"]) if errors else None
    return {
        "passed": max([entry["relative_error"] for entry in errors], default=0.0) <= ORDER_TOL,
        "max_relative_error": max([entry["relative_error"] for entry in errors], default=0.0),
        "max_entry": max_entry,
        "per_q": errors,
    }


def level_transfer(rows: list[dict[str, Any]], coord: str, factor: str, level: Any, null_value: float) -> dict[str, Any]:
    subset = [row for row in rows if row[factor] == level]
    return basic_transfer_laws(subset, coord, null_value)


def basic_transfer_laws(rows: list[dict[str, Any]], coord: str, null_value: float) -> dict[str, Any]:
    means = {q: mean([coord_value(row, coord) for row in rows if int(row["q"]) == q]) for q in Q_LADDER}
    nonzero = [q for q in NONZERO_Q if means[q] != 0.0]
    positives = [means[q] for q in (512, 1024, 1536)]
    negatives = [means[q] for q in (-512, -1024, -1536)]
    direct = all(value > 0.0 for value in positives) and all(value < 0.0 for value in negatives)
    reversed_sign = all(value < 0.0 for value in positives) and all(value > 0.0 for value in negatives)
    convention = "direct" if direct else "reversed" if reversed_sign else "none"
    odd_errors = {
        str(q): oddness(means[q], means[-q], coord_floor(coord))
        for q in (512, 1024, 1536)
    }
    abs_by_q = [abs(means[q]) for q in (512, 1024, 1536)]
    monotonic = abs_by_q == sorted(abs_by_q) and len(set(abs_by_q)) == len(abs_by_q)
    gain = min(abs(means[q]) for q in (-1536, -1024, 1024, 1536)) > gain_floor(coord, null_value)
    return {
        "passed": convention != "none" and all(value <= ODDNESS_TOL for value in odd_errors.values()) and gain and monotonic,
        "means_by_q": {str(k): v for k, v in means.items()},
        "zero_test_mean": means[0],
        "sign_convention": convention,
        "oddness_law": all(value <= ODDNESS_TOL for value in odd_errors.values()),
        "oddness_errors": odd_errors,
        "gain_law": gain,
        "gain_floor": gain_floor(coord, null_value),
        "monotonicity_law": monotonic,
    }


def gain_normalized_factor_law(rows: list[dict[str, Any]], coord: str, factor: str, null_value: float) -> dict[str, Any]:
    levels = SOURCE_ORDERS if factor == "source_order" else SUBCAPTURE_ORDERS
    items = [level_transfer(rows, coord, factor, level, null_value) for level in levels]
    mean_gains = []
    for item in items:
        gains = [abs(float(item["means_by_q"][str(q)])) / abs(q) for q in NONZERO_Q]
        mean_gains.append(mean(gains))
    gain_error = rel(mean_gains[0], mean_gains[1], coord_floor(coord) / min(abs(q) for q in NONZERO_Q))
    return {
        "factor": factor,
        "levels": [{"level": level, **item} for level, item in zip(levels, items)],
        "gain_agreement_error": gain_error,
        "gain_agreement_law": gain_error <= ORDER_TOL,
        "passed": all(item["passed"] for item in items) and gain_error <= ORDER_TOL,
    }


def stratum_law(rows: list[dict[str, Any]], coord: str, null_value: float) -> dict[str, Any]:
    strata = []
    for mapping in MAPPINGS:
        for source_order in SOURCE_ORDERS:
            for subcapture_order in SUBCAPTURE_ORDERS:
                subset = [
                    row for row in rows
                    if int(row["mapping"]) == mapping
                    and row["source_order"] == source_order
                    and row["subcapture_order"] == subcapture_order
                ]
                laws = basic_transfer_laws(subset, coord, null_value)
                strata.append(
                    {
                        "mapping": mapping,
                        "source_order": source_order,
                        "subcapture_order": subcapture_order,
                        "passed": laws["passed"],
                        "sign_convention": laws["sign_convention"],
                        "oddness_law": laws["oddness_law"],
                        "gain_law": laws["gain_law"],
                        "monotonicity_law": laws["monotonicity_law"],
                    }
                )
    return {"passed": all(item["passed"] for item in strata), "strata": strata}


def restoration_law(rows: list[dict[str, Any]], coord: str) -> dict[str, Any]:
    floor = RESTORATION_FLOOR_BY_COORDINATE[coord]
    max_error = 0.0
    max_entry = None
    for row in rows:
        for value in row["coordinates"][coord]["sentinel_values"]:
            error = abs(value) / max(abs(value), floor)
            if error > max_error:
                max_error = error
                max_entry = {
                    "replicate_index": row["replicate_index"],
                    "pair_index": row["pair_index"],
                    "trial_index": row["trial_index"],
                    "q": row["q"],
                    "mapping": row["mapping"],
                    "sentinel_value": value,
                    "relative_error": error,
                }
    integrity = all(row["trial_ok"] and row["restoration_passed"] and row["byte_compare_passed"] for row in rows)
    return {"passed": integrity and max_error <= SENTINEL_TOL, "max_error": max_error, "floor": floor, "max_entry": max_entry}


def evaluate_scope(rows: list[dict[str, Any]], coord: str, *, include_fresh: bool) -> dict[str, Any]:
    null = null_ceiling(rows, coord, include_fresh=include_fresh)
    null_value = null["complete_null_ceiling"]
    base = basic_transfer_laws(rows, coord, null_value)
    paired = paired_crossover_law(rows, coord, null)
    source_order = factor_invariance_law(rows, coord, "source_order")
    subcapture_order = factor_invariance_law(rows, coord, "subcapture_order")
    source_gain = gain_normalized_factor_law(rows, coord, "source_order", null_value)
    subcapture_gain = gain_normalized_factor_law(rows, coord, "subcapture_order", null_value)
    strata = stratum_law(rows, coord, null_value)
    restoration = restoration_law(rows, coord)
    laws = {
        "heldout_q0_null_law": (
            paired["q0_null_semantics"]["heldout_mapping_inside_null"]
            and paired["q0_null_semantics"]["heldout_logical_pair_inside_bound"]
            and paired["q0_null_semantics"]["heldout_f0_inside_null"]
            and paired["q0_null_semantics"]["heldout_physical_pair_inside_bound"]
        ),
        "sign_law": base["sign_convention"] != "none",
        "oddness_law": base["oddness_law"],
        "gain_law": base["gain_law"],
        "monotonicity_law": base["monotonicity_law"],
        "paired_logical_mapping_invariance_law": paired["logical_mapping_invariance_law"],
        "paired_physical_reversal_law": paired["physical_mapping_reversal_law"],
        "source_order_invariance_law": source_order["passed"],
        "independent_subcapture_order_invariance_law": subcapture_order["passed"],
        "gain_normalized_source_order_law": source_gain["passed"],
        "gain_normalized_subcapture_order_law": subcapture_gain["passed"],
        "stratum_transfer_law": strata["passed"],
        "restoration_law": restoration["passed"],
    }
    return {
        "passed": all(laws.values()),
        "laws": laws,
        "means_by_q": base["means_by_q"],
        "zero_test_mean": base["zero_test_mean"],
        "null_components": null,
        "sign_convention": base["sign_convention"],
        "oddness_errors": base["oddness_errors"],
        "gain_floor": base["gain_floor"],
        "paired_crossover": paired,
        "source_order_direct_nonzero": source_order,
        "subcapture_order_direct_nonzero": subcapture_order,
        "source_order_gain_normalized": source_gain,
        "subcapture_order_gain_normalized": subcapture_gain,
        "stratum_transfer": strata,
        "restoration": restoration,
    }


def adjudicate(features: dict[str, Any]) -> dict[str, Any]:
    if features.get("schema_id") != FEATURE_SCHEMA or features.get("source_kind") != "fresh_independent_window_v3":
        raise IndependentWindowError("V3 adjudication requires fresh independent-window features")
    expected_features_sha = digest({k: v for k, v in features.items() if k != "features_sha256"})
    if features.get("features_sha256") != expected_features_sha:
        return {
            "schema_id": ADJUDICATION_SCHEMA,
            "status": CLASS_NOT_ESTABLISHED,
            "allowed_classes": list(ALLOWED_CLASSES),
            "forbidden_claims_not_emitted": list(FORBIDDEN_CLASSES),
            "primary_coordinate": PRIMARY_COORDINATE,
            "fresh_only": True,
            "features_sha256": features.get("features_sha256"),
            "expected_features_sha256": expected_features_sha,
            "primary_coordinate_result": {
                "coordinate": PRIMARY_COORDINATE,
                "passed": False,
                "integrity_gate": False,
                "meaningful_transfer_shape": False,
                "failed_laws": [{"law": "features_sha256_mismatch"}],
            },
            "integrity": features.get("integrity", {}),
            "adjudication_sha256": digest(
                {
                    "schema_id": ADJUDICATION_SCHEMA,
                    "status": CLASS_NOT_ESTABLISHED,
                    "features_sha256": features.get("features_sha256"),
                    "expected_features_sha256": expected_features_sha,
                    "failed_law": "features_sha256_mismatch",
                }
            ),
        }
    rows = features["trial_features"]
    aggregate = evaluate_scope(rows, PRIMARY_COORDINATE, include_fresh=True)
    per_replicate = {
        str(rep): evaluate_scope([row for row in rows if int(row["replicate_index"]) == rep], PRIMARY_COORDINATE, include_fresh=False)
        for rep in REPLICATES
    }
    conventions = [per_replicate[str(rep)]["sign_convention"] for rep in REPLICATES]
    same_convention = aggregate["sign_convention"] != "none" and all(item == aggregate["sign_convention"] for item in conventions)
    rep_errors = []
    for q in NONZERO_Q:
        left = float(per_replicate["0"]["means_by_q"][str(q)])
        right = float(per_replicate["1"]["means_by_q"][str(q)])
        rep_errors.append(rel(left, right, coord_floor(PRIMARY_COORDINATE)))
    replicate_consistency = max(rep_errors, default=0.0) <= REPLICATE_TOL
    failed_laws = []
    if not features["integrity"]["schedule_matched"]:
        failed_laws.extend(features["integrity"]["failures"])
    for law, ok in aggregate["laws"].items():
        if not ok:
            failed_laws.append({"scope": "aggregate", "law": law})
    for rep in REPLICATES:
        for law, ok in per_replicate[str(rep)]["laws"].items():
            if not ok:
                failed_laws.append({"scope": f"replicate_{rep}", "law": law})
    if not same_convention:
        failed_laws.append({"scope": "replicate", "law": "same_sign_convention", "conventions": conventions})
    if not replicate_consistency:
        failed_laws.append({"scope": "replicate", "law": "fresh_process_consistency", "max_error": max(rep_errors, default=0.0)})
    confirmed = (
        features["integrity"]["schedule_matched"]
        and aggregate["passed"]
        and all(per_replicate[str(rep)]["passed"] for rep in REPLICATES)
        and same_convention
        and replicate_consistency
    )
    shape = (
        aggregate["laws"]["sign_law"]
        and (aggregate["laws"]["oddness_law"] or aggregate["laws"]["monotonicity_law"])
        and same_convention
    )
    integrity_gate = features["integrity"]["schedule_matched"]
    status = CLASS_CONFIRMED if confirmed else CLASS_CANDIDATE if integrity_gate and shape else CLASS_NOT_ESTABLISHED
    result = {
        "schema_id": ADJUDICATION_SCHEMA,
        "status": status,
        "allowed_classes": list(ALLOWED_CLASSES),
        "forbidden_claims_not_emitted": list(FORBIDDEN_CLASSES),
        "primary_coordinate": PRIMARY_COORDINATE,
        "fresh_only": True,
        "features_sha256": features["features_sha256"],
        "primary_coordinate_result": {
            "coordinate": PRIMARY_COORDINATE,
            "passed": confirmed,
            "integrity_gate": integrity_gate,
            "meaningful_transfer_shape": shape,
            "aggregate": aggregate,
            "per_replicate": per_replicate,
            "same_replicate_sign_convention": same_convention,
            "replicate_sign_conventions": conventions,
            "fresh_process_consistency_law": replicate_consistency,
            "max_replicate_error": max(rep_errors, default=0.0),
            "failed_laws": failed_laws,
        },
        "integrity": features["integrity"],
    }
    result["adjudication_sha256"] = digest({k: v for k, v in result.items() if k != "adjudication_sha256"})
    return result


def _mock_window(prefix: str, value: float, *, ids: tuple[int, int, int], cpu: int = RECEIVER_CORE, ok: bool = True) -> dict[str, Any]:
    cycles = max(1.0, 100000.0 + abs(value))
    return {
        f"{prefix}_opened": ok,
        f"{prefix}_read_ok": ok,
        f"{prefix}_event_order_ok": ok,
        f"{prefix}_unmultiplexed": ok,
        f"{prefix}_open_errno": 0 if ok else 1,
        f"{prefix}_read_errno": 0 if ok else 1,
        f"{prefix}_cpu_before": cpu,
        f"{prefix}_cpu_after": cpu,
        f"{prefix}_cycles": cycles,
        f"{prefix}_change_to_dirty": value,
        f"{prefix}_probe_dirty": 2.0 * value,
        f"{prefix}_duration_ns": 50000.0 + abs(value),
        f"{prefix}_time_enabled": 1000,
        f"{prefix}_time_running": 1000 if ok else 900,
        f"{prefix}_event_ids": list(ids),
    }


def _mock_sentinel_fields(component: str, event_seed: int, *, broken: bool = False, wrong_core: bool = False, multiplexed: bool = False) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    prefixes = (f"{component}_pre_a", f"{component}_pre_b", f"{component}_post_a", f"{component}_post_b")
    for offset, prefix in enumerate(prefixes):
        value = 5000.0 if broken and prefix.endswith("post_a") else 0.0
        ids = (event_seed + offset * 3, event_seed + offset * 3 + 1, event_seed + offset * 3 + 2)
        fields.update(_mock_window(prefix, value, ids=ids, cpu=4 if wrong_core else RECEIVER_CORE, ok=not multiplexed))
    return fields


def _stage_receipts(rep: int, trial_index: int, component: str, *, kind: str) -> dict[str, Any]:
    receipts = {
        "baseline_receipt_id": f"r{rep}_{trial_index}_{component}_baseline",
        "pre_sentinel_receipt_id": f"r{rep}_{trial_index}_{component}_pre_sentinel",
        "rebaseline_receipt_id": f"r{rep}_{trial_index}_{component}_rebaseline",
        "source_receipt_id": f"r{rep}_{trial_index}_{component}_source",
        "measure_receipt_id": f"r{rep}_{trial_index}_{component}_measure",
        "restore_receipt_id": f"r{rep}_{trial_index}_{component}_restore",
        "post_sentinel_receipt_id": f"r{rep}_{trial_index}_{component}_post_sentinel",
    }
    if kind == "reused_baseline_receipt" and component == "negative":
        receipts["baseline_receipt_id"] = f"r{rep}_{trial_index}_positive_baseline"
    return {f"{component}_{key}": value for key, value in receipts.items()}


def _f_value(kind: str, q: int, rep: int) -> float:
    if kind in {"ideal", "legacy_carryover", "single_q0_outlier", "missing_reencoding", "missing_rebaseline", "unequal_source_work",
                "different_permutation", "same_raw_window_reused", "component_order_drift", "mapping_role_drift",
                "q0_null_test_leakage", "fixed_physical_bank_bias", "subcapture_order_bias", "multiplexed_pmu",
                "event_id_drift", "wrong_core", "byte_restoration_failure", "ownership_sentinel_restoration_failure",
                "forged_derived_delta", "out_of_order_zero_rc", "reused_baseline_receipt", "cross_trial_receipt_reuse"}:
        return 1.25 * q
    if kind == "contradictory_replicates":
        return 1.25 * q if rep == 0 else -1.25 * q
    if kind == "non_odd_response":
        return 1.25 * abs(q)
    if kind == "non_monotonic_response":
        return {-1536: -1500.0, -1024: -1800.0, -512: -700.0, 0: 0.0, 512: 700.0, 1024: 1800.0, 1536: 1500.0}[q]
    if kind == "insufficient_gain":
        return 0.5 * q / 512.0
    if kind == "independent_no_carryover":
        return 1.25 * q
    return 1.25 * q


def build_mock_capture(kind: str) -> dict[str, Any]:
    schedule = build_schedule()
    raw_records: list[dict[str, Any]] = []
    sentinels: list[dict[str, Any]] = []
    event_id = 1000
    for trial in schedule["trials"]:
        rep = int(trial["replicate_index"])
        q = int(trial["q"])
        mapping = int(trial["mapping"])
        f_value = _f_value(kind, q, rep)
        if q == 0 and kind in {"legacy_carryover", "single_q0_outlier", "missing_reencoding", "missing_rebaseline"}:
            if rep == 0 and trial["q0_role"] == "null_test" and int(trial["pair_index"]) % 4 == 1 and mapping == 1:
                f_value = -101.0
            elif rep == 0 and trial["q0_role"] == "null_test" and int(trial["pair_index"]) % 4 == 1 and mapping == 0:
                f_value = 1.0
        if q == 0 and kind == "q0_null_test_leakage" and trial["q0_role"] == "null_test":
            f_value = 1000.0
        if kind == "subcapture_order_bias" and trial["subcapture_order"] == "negative_subcapture_first":
            f_value += 500.0 if q != 0 else 10.0
        positive_value = 200000.0 + f_value / 2.0
        negative_value = 200000.0 - f_value / 2.0
        if kind == "fixed_physical_bank_bias":
            if mapping == 0:
                positive_value += 1000.0
            else:
                negative_value += 1000.0
        if kind == "mapping_role_drift" and rep == 0 and int(trial["trial_index"]) == 0:
            logical_positive_physical = "B" if mapping == 0 else "A"
        else:
            logical_positive_physical = trial["logical_positive_physical"]
        logical_negative_physical = "B" if logical_positive_physical == "A" else "A"
        pos_ids = (event_id, event_id + 1, event_id + 2)
        neg_ids = (event_id + 3, event_id + 4, event_id + 5)
        if kind == "same_raw_window_reused":
            neg_ids = pos_ids
            negative_value = positive_value
        if kind == "event_id_drift":
            neg_ids = (pos_ids[0], pos_ids[1], pos_ids[2])
        pos_ok = kind not in {"multiplexed_pmu"}
        neg_ok = kind not in {"multiplexed_pmu"}
        cpu = 4 if kind == "wrong_core" else RECEIVER_CORE
        expected_order = ["positive", "negative"] if subcapture_bool(trial["subcapture_order"]) else ["negative", "positive"]
        if kind == "component_order_drift":
            observed_order = list(reversed(expected_order))
        else:
            observed_order = expected_order
        positive_stage_sequence = list(SUBCAPTURE_STAGE_SEQUENCE)
        negative_stage_sequence = list(SUBCAPTURE_STAGE_SEQUENCE)
        if kind == "out_of_order_zero_rc":
            positive_stage_sequence = [
                "receiver_baseline",
                "pre_sentinel",
                "source_encoding",
                "rebaseline",
                "measure_logical_bank",
                "restore_both_banks",
                "post_sentinel",
            ]
        record = {
            "schema_id": RAW_SCHEMA if kind != "translated_v1_v2_smuggling" else "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_RAW_RECORD_V2",
            "replicate_index": rep,
            "pair_index": int(trial["pair_index"]),
            "leg_index": int(trial["leg_index"]),
            "trial_index": int(trial["trial_index"]),
            "repeat_index": int(trial["repeat_index"]),
            "q": q,
            "q0_role": trial["q0_role"],
            "mapping": mapping,
            "mapping_order_first": int(trial["mapping_order_first"]),
            "bank_allocation_id": trial["bank_allocation_id"],
            "source_order": trial["source_order"],
            "subcapture_order": trial["subcapture_order"],
            "subcapture_execution_order": observed_order,
            "positive_sequence_index": observed_order.index("positive"),
            "negative_sequence_index": observed_order.index("negative"),
            "positive_stage_sequence": positive_stage_sequence,
            "negative_stage_sequence": negative_stage_sequence,
            "logical_positive_physical": logical_positive_physical,
            "logical_negative_physical": logical_negative_physical,
            "line_permutation_a": PERM_A if kind != "different_permutation" else PERM_A + 2,
            "line_permutation_b": PERM_B,
            "positive_prefix_unique": kind != "duplicate_prefix_lines",
            "negative_prefix_unique": True,
            "positive_baseline_rc": 0,
            "positive_rebaseline_rc": 1 if kind == "missing_rebaseline" else 0,
            "positive_source_rc": 1 if kind == "missing_reencoding" else 0,
            "positive_restore_rc": 0,
            "positive_measure_rc": 0,
            "negative_baseline_rc": 0,
            "negative_rebaseline_rc": 1 if kind == "missing_rebaseline" else 0,
            "negative_source_rc": 1 if kind == "missing_reencoding" else 0,
            "negative_restore_rc": 0,
            "negative_measure_rc": 0,
            "positive_source_positive_work": BASE_WORK + q,
            "positive_source_negative_work": BASE_WORK - q,
            "positive_source_total_work": SOURCE_WORK_PER_SUBCAPTURE,
            "negative_source_positive_work": BASE_WORK + q,
            "negative_source_negative_work": BASE_WORK - q,
            "negative_source_total_work": SOURCE_WORK_PER_SUBCAPTURE - (1 if kind == "unequal_source_work" else 0),
            "mapping_leg_source_work": SOURCE_WORK_PER_MAPPING_LEG - (1 if kind == "unequal_source_work" else 0),
            "byte_compare_passed": kind != "byte_restoration_failure",
            "restoration_passed": kind not in {"ownership_sentinel_restoration_failure", "byte_restoration_failure"},
            "trial_ok": kind not in {
                "missing_reencoding",
                "missing_rebaseline",
                "unequal_source_work",
                "different_permutation",
                "same_raw_window_reused",
                "component_order_drift",
                "mapping_role_drift",
                "multiplexed_pmu",
                "event_id_drift",
                "wrong_core",
                "byte_restoration_failure",
                "ownership_sentinel_restoration_failure",
                "duplicate_prefix_lines",
                "forged_derived_delta",
                "out_of_order_zero_rc",
                "reused_baseline_receipt",
            },
        }
        record.update(_stage_receipts(rep, int(trial["trial_index"]), "positive", kind=kind))
        record.update(_stage_receipts(rep, int(trial["trial_index"]), "negative", kind=kind))
        if kind == "forged_derived_delta":
            record["logical_change_to_dirty_delta"] = positive_value - negative_value
        record.update(_mock_window("positive_measure", positive_value, ids=pos_ids, cpu=cpu, ok=pos_ok))
        record.update(_mock_window("negative_measure", negative_value, ids=neg_ids, cpu=cpu, ok=neg_ok))
        if kind == "event_id_drift":
            record["negative_measure_event_order_ok"] = False
        sentinel = {
            "schema_id": SENTINEL_SCHEMA if kind != "translated_v1_v2_smuggling" else "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_RESTORATION_SENTINEL_V2",
            "replicate_index": rep,
            "pair_index": int(trial["pair_index"]),
            "leg_index": int(trial["leg_index"]),
            "trial_index": int(trial["trial_index"]),
            "repeat_index": int(trial["repeat_index"]),
            "q": q,
            "q0_role": trial["q0_role"],
            "mapping": mapping,
            "mapping_order_first": int(trial["mapping_order_first"]),
            "bank_allocation_id": trial["bank_allocation_id"],
            "source_order": trial["source_order"],
            "subcapture_order": trial["subcapture_order"],
            "bytes_unchanged": kind != "ownership_sentinel_restoration_failure",
            "byte_compare_passed": kind != "byte_restoration_failure",
        }
        sentinel.update(_mock_sentinel_fields("positive", event_id + 6, broken=kind == "ownership_sentinel_restoration_failure", wrong_core=kind == "wrong_core", multiplexed=kind == "multiplexed_pmu"))
        sentinel.update(_mock_sentinel_fields("negative", event_id + 18, broken=False, wrong_core=kind == "wrong_core", multiplexed=kind == "multiplexed_pmu"))
        raw_records.append(record)
        sentinels.append(sentinel)
        event_id += 64
    if kind == "retained_evidence_pooling":
        raw_records.extend(raw_records[:2])
    if kind == "cross_trial_receipt_reuse" and len(raw_records) >= 2:
        for component in ("positive", "negative"):
            for suffix in (
                "baseline_receipt_id",
                "pre_sentinel_receipt_id",
                "rebaseline_receipt_id",
                "source_receipt_id",
                "measure_receipt_id",
                "restore_receipt_id",
                "post_sentinel_receipt_id",
            ):
                raw_records[1][f"{component}_{suffix}"] = raw_records[0][f"{component}_{suffix}"]
    return {"schedule": schedule, "raw_records": raw_records, "sentinels": sentinels}


def run_mock(kind: str) -> dict[str, Any]:
    capture = build_mock_capture(kind)
    try:
        features = extract_features(capture["schedule"], capture["raw_records"], capture["sentinels"])
        return adjudicate(features)
    except Exception as exc:
        return {
            "schema_id": ADJUDICATION_SCHEMA,
            "status": CLASS_NOT_ESTABLISHED,
            "exception": type(exc).__name__,
            "message": str(exc),
            "forbidden_claims_not_emitted": list(FORBIDDEN_CLASSES),
        }


def self_test() -> dict[str, Any]:
    schedule = build_schedule()
    cases = {
        "independent_no_carryover": (CLASS_CONFIRMED, "equals"),
        "legacy_carryover": (CLASS_CANDIDATE, "equals"),
        "single_q0_outlier": (CLASS_CANDIDATE, "equals"),
        "missing_reencoding": (CLASS_NOT_ESTABLISHED, "equals"),
        "missing_rebaseline": (CLASS_NOT_ESTABLISHED, "equals"),
        "unequal_source_work": (CLASS_NOT_ESTABLISHED, "equals"),
        "different_permutation": (CLASS_NOT_ESTABLISHED, "equals"),
        "same_raw_window_reused": (CLASS_NOT_ESTABLISHED, "equals"),
        "component_order_drift": (CLASS_NOT_ESTABLISHED, "equals"),
        "mapping_role_drift": (CLASS_NOT_ESTABLISHED, "equals"),
        "q0_null_test_leakage": (CLASS_CANDIDATE, "equals"),
        "fixed_physical_bank_bias": (CLASS_CANDIDATE, "equals"),
        "subcapture_order_bias": (CLASS_CANDIDATE, "equals"),
        "contradictory_replicates": (CLASS_NOT_ESTABLISHED, "equals"),
        "non_odd_response": (CLASS_NOT_ESTABLISHED, "equals"),
        "non_monotonic_response": (CLASS_CANDIDATE, "equals"),
        "insufficient_gain": (CLASS_CANDIDATE, "equals"),
        "multiplexed_pmu": (CLASS_NOT_ESTABLISHED, "equals"),
        "event_id_drift": (CLASS_NOT_ESTABLISHED, "equals"),
        "wrong_core": (CLASS_NOT_ESTABLISHED, "equals"),
        "byte_restoration_failure": (CLASS_NOT_ESTABLISHED, "equals"),
        "ownership_sentinel_restoration_failure": (CLASS_NOT_ESTABLISHED, "equals"),
        "translated_v1_v2_smuggling": (CLASS_NOT_ESTABLISHED, "equals"),
        "retained_evidence_pooling": (CLASS_NOT_ESTABLISHED, "equals"),
        "forged_derived_delta": (CLASS_NOT_ESTABLISHED, "equals"),
        "out_of_order_zero_rc": (CLASS_NOT_ESTABLISHED, "equals"),
        "reused_baseline_receipt": (CLASS_NOT_ESTABLISHED, "equals"),
        "cross_trial_receipt_reuse": (CLASS_NOT_ESTABLISHED, "equals"),
    }
    observed = {}
    for kind, (expected, mode) in cases.items():
        adjudication = run_mock(kind)
        status = adjudication["status"]
        passed = status == expected if mode == "equals" else status != expected
        failed_laws = adjudication.get("primary_coordinate_result", {}).get("failed_laws", [])
        observed[kind] = {
            "expected": expected,
            "mode": mode,
            "actual": status,
            "passed": passed,
            "failed_law_count": len(failed_laws),
            "failed_law_sample": failed_laws[:8],
        }
    legacy = run_mock("legacy_carryover")
    independent = run_mock("independent_no_carryover")
    regressions = {
        "schedule_total_records_128": len(schedule["trials"]) == 128,
        "schedule_per_replicate_64": all(len([t for t in schedule["trials"] if int(t["replicate_index"]) == rep]) == 64 for rep in REPLICATES),
        "q0_build_pairs_per_replicate_4": all(len([t for t in schedule["trials"] if int(t["replicate_index"]) == rep and int(t["q"]) == 0 and t["q0_role"] == "null_build" and int(t["leg_index"]) == 0]) == 4 for rep in REPLICATES),
        "q0_test_pairs_per_replicate_4": all(len([t for t in schedule["trials"] if int(t["replicate_index"]) == rep and int(t["q"]) == 0 and t["q0_role"] == "null_test" and int(t["leg_index"]) == 0]) == 4 for rep in REPLICATES),
        "component_windows_256": schedule["total_component_measurement_windows"] == 256,
        "legacy_carryover_distinguished": legacy["status"] != CLASS_CONFIRMED and independent["status"] == CLASS_CONFIRMED,
        "null_test_not_used_to_build_ceiling": independent["primary_coordinate_result"]["aggregate"]["paired_crossover"]["q0_null_semantics"]["null_test_used_to_construct_ceiling"] is False,
    }
    tamper_capture = build_mock_capture("independent_no_carryover")
    tamper_features = extract_features(tamper_capture["schedule"], tamper_capture["raw_records"], tamper_capture["sentinels"])
    tamper_features["trial_features"][0]["coordinates"][PRIMARY_COORDINATE]["logical"] += 1.0
    tamper_adjudication = adjudicate(tamper_features)
    regressions["tampered_features_hash_rejected"] = tamper_adjudication["status"] == CLASS_NOT_ESTABLISHED
    result = {"schema_id": SELF_TEST_SCHEMA, "cases": observed, "regressions": regressions}
    result["self_test_passed"] = all(item["passed"] for item in observed.values()) and all(regressions.values())
    result["self_test_sha256"] = digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-schedule", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mock", choices=(
        "independent_no_carryover",
        "legacy_carryover",
        "single_q0_outlier",
        "missing_reencoding",
        "missing_rebaseline",
        "unequal_source_work",
        "different_permutation",
        "same_raw_window_reused",
        "component_order_drift",
        "mapping_role_drift",
        "q0_null_test_leakage",
        "fixed_physical_bank_bias",
        "subcapture_order_bias",
        "contradictory_replicates",
        "non_odd_response",
        "non_monotonic_response",
        "insufficient_gain",
        "multiplexed_pmu",
        "event_id_drift",
        "wrong_core",
        "byte_restoration_failure",
        "ownership_sentinel_restoration_failure",
        "translated_v1_v2_smuggling",
        "retained_evidence_pooling",
        "forged_derived_delta",
        "out_of_order_zero_rc",
        "reused_baseline_receipt",
        "cross_trial_receipt_reuse",
    ))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_schedule:
        result = write_schedule_artifacts()
    elif args.schedule:
        result = build_schedule()
    elif args.self_test:
        result = self_test()
    elif args.mock:
        result = run_mock(args.mock)
    else:
        result = {"schema_id": "CAT_CAS_INDEPENDENT_WINDOW_PUBLIC_INFO_V3", "schedule": build_schedule()}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not args.self_test or result["self_test_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
