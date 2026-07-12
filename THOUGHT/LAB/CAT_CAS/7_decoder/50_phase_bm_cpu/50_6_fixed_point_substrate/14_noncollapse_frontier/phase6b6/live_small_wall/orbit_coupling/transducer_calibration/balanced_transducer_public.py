#!/usr/bin/env python3
"""Public schedule, feature extraction, and adjudication for balanced calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


SCHEDULE_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_PUBLIC_SCHEDULE_V1"
FEATURE_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_FEATURES_V1"
ADJUDICATION_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_ADJUDICATION_V1"
RAW_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_RAW_RECORD_V1"
SENTINEL_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_RESTORATION_SENTINEL_V1"

PUBLIC_SEED = "CAT_CAS_BALANCED_TRANSDUCER_PUBLIC_SEED_V1"
Q_LADDER = (-1536, -1024, -512, 0, 512, 1024, 1536)
REPLICATES = (0, 1)
MAPPINGS = (0, 1)
TRIALS_PER_Q_MAPPING = 8
PAIR_COUNT_PER_REPLICATE = len(Q_LADDER) * TRIALS_PER_Q_MAPPING
TRIALS_PER_REPLICATE = len(Q_LADDER) * len(MAPPINGS) * TRIALS_PER_Q_MAPPING
BASE_WORK = 2048
TOTAL_WORK = 4096
BANK_LINES = 4096
PERM_A = 257
PERM_B = 43
ABS_FLOOR = 1.0
GAIN_MULTIPLIER = 3.0
ODDNESS_TOL = 0.25
POINTER_SWAP_TOL = 0.25
ORDER_TOL = 0.25
REPLICATE_TOL = 0.35
SENTINEL_TOL = 0.25
SENTINEL_ABS_FLOOR = 1024.0
COORDINATE_PRIORITY = (
    "change_to_dirty",
    "probe_dirty",
    "change_to_dirty_per_cycle",
    "probe_dirty_per_cycle",
    "cycles",
)
COORDINATES = (
    "change_to_dirty",
    "probe_dirty",
    "cycles",
    "duration_ns",
    "change_to_dirty_per_cycle",
    "probe_dirty_per_cycle",
)
DIAGNOSTIC_COORDINATES = ("duration_ns",)
RESTORATION_FLOOR_BY_COORDINATE = {
    "change_to_dirty": 1024.0,
    "probe_dirty": 1024.0,
    "cycles": 1024.0,
    "duration_ns": 1024.0,
    "change_to_dirty_per_cycle": 1.0e-6,
    "probe_dirty_per_cycle": 1.0e-6,
}


class TransducerError(AssertionError):
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_schedule() -> dict[str, Any]:
    rng = random.Random(PUBLIC_SEED)
    all_trials: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        pair_rows: list[dict[str, Any]] = []
        for repeat_index in range(TRIALS_PER_Q_MAPPING):
            q_order = list(Q_LADDER)
            rng.shuffle(q_order)
            source_positive_first = (repeat_index & 1) == 0
            receiver_positive_first = ((repeat_index >> 1) & 1) == 0
            for q in q_order:
                mapping_order_first = (repeat_index + (q // 512) + replicate) & 1
                pair_rows.append(
                    {
                        "replicate_index": replicate,
                        "q": q,
                        "repeat_index": repeat_index,
                        "q0_role": "null_build" if q == 0 and repeat_index < 4 else (
                            "null_test" if q == 0 else "signal"
                        ),
                        "source_order": "positive_first" if source_positive_first else "negative_first",
                        "source_positive_first": source_positive_first,
                        "measurement_order": "positive_first" if receiver_positive_first else "negative_first",
                        "receiver_positive_first": receiver_positive_first,
                        "mapping_order_first": mapping_order_first,
                    }
                )
        rng.shuffle(pair_rows)
        pair_index = 0
        trial_index = 0
        replicate_trials: list[dict[str, Any]] = []
        for pair in pair_rows:
            mappings = [pair["mapping_order_first"], 1 - pair["mapping_order_first"]]
            for leg_index, mapping in enumerate(mappings):
                replicate_trials.append(
                    {
                        **pair,
                        "pair_index": pair_index,
                        "leg_index": leg_index,
                        "mapping": mapping,
                        "logical_positive_physical": "A" if mapping == 0 else "B",
                        "logical_negative_physical": "B" if mapping == 0 else "A",
                        "trial_index": trial_index,
                        "global_trial_index": replicate * TRIALS_PER_REPLICATE + trial_index,
                    }
                )
                trial_index += 1
            pair_index += 1
        all_trials.extend(replicate_trials)
    schedule = {
        "schema_id": SCHEDULE_SCHEMA,
        "public_seed": PUBLIC_SEED,
        "q_ladder": list(Q_LADDER),
        "base_work": BASE_WORK,
        "total_work": TOTAL_WORK,
        "bank_lines": BANK_LINES,
        "line_permutation": {"a": PERM_A, "b": PERM_B, "modulus": BANK_LINES},
        "fresh_process_replicates": len(REPLICATES),
        "trials_per_q_mapping": TRIALS_PER_Q_MAPPING,
        "pair_count_per_replicate": PAIR_COUNT_PER_REPLICATE,
        "trials_per_replicate": TRIALS_PER_REPLICATE,
        "trials": all_trials,
    }
    validate_schedule(schedule)
    schedule["schedule_sha256"] = digest({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    return schedule


def schedule_tsv(schedule: dict[str, Any]) -> str:
    rows = []
    for trial in sorted(schedule["trials"], key=lambda item: (item["replicate_index"], item["pair_index"], item["leg_index"])):
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
                    1 if trial["receiver_positive_first"] else 0,
                )
            )
        )
    return "\n".join(rows) + "\n"


def validate_schedule(schedule: dict[str, Any]) -> None:
    if schedule.get("schema_id") != SCHEDULE_SCHEMA:
        raise TransducerError("schedule schema mismatch")
    trials = schedule.get("trials")
    if not isinstance(trials, list) or len(trials) != len(REPLICATES) * TRIALS_PER_REPLICATE:
        raise TransducerError("schedule trial count mismatch")
    seen: set[tuple[int, int]] = set()
    pair_seen: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    counts: dict[tuple[int, int, int, str, str], int] = defaultdict(int)
    for trial in trials:
        rep = int(trial["replicate_index"])
        idx = int(trial["trial_index"])
        pair = int(trial["pair_index"])
        leg = int(trial["leg_index"])
        repeat = int(trial["repeat_index"])
        q = int(trial["q"])
        mapping = int(trial["mapping"])
        order = str(trial["measurement_order"])
        source_order = str(trial["source_order"])
        if rep not in REPLICATES or idx < 0 or idx >= TRIALS_PER_REPLICATE:
            raise TransducerError("schedule replicate or trial index out of range")
        if pair < 0 or pair >= PAIR_COUNT_PER_REPLICATE or leg not in (0, 1):
            raise TransducerError("schedule pair or leg index out of range")
        if repeat < 0 or repeat >= TRIALS_PER_Q_MAPPING:
            raise TransducerError("schedule repeat index out of range")
        if (rep, idx) in seen:
            raise TransducerError("duplicate schedule trial index")
        seen.add((rep, idx))
        pair_seen[(rep, pair)].append(trial)
        if q not in Q_LADDER or mapping not in MAPPINGS:
            raise TransducerError("schedule q or mapping out of range")
        if order not in ("positive_first", "negative_first"):
            raise TransducerError("schedule measurement order out of range")
        if source_order not in ("positive_first", "negative_first"):
            raise TransducerError("schedule source order out of range")
        counts[(rep, q, mapping, source_order, order)] += 1
    for rep in REPLICATES:
        for pair in range(PAIR_COUNT_PER_REPLICATE):
            legs = sorted(pair_seen[(rep, pair)], key=lambda item: int(item["leg_index"]))
            if len(legs) != 2:
                raise TransducerError("schedule pair does not have two legs")
            if {int(item["mapping"]) for item in legs} != {0, 1}:
                raise TransducerError("schedule pair does not cross both mappings")
            for key in ("q", "repeat_index", "source_order", "measurement_order", "mapping_order_first"):
                if legs[0][key] != legs[1][key]:
                    raise TransducerError(f"schedule pair has mismatched {key}")
            if int(legs[0]["mapping"]) != int(legs[0]["mapping_order_first"]):
                raise TransducerError("schedule pair leg order does not match mapping_order_first")
        for q in Q_LADDER:
            for mapping in MAPPINGS:
                for source_order in ("positive_first", "negative_first"):
                    for receiver_order in ("positive_first", "negative_first"):
                        if counts[(rep, q, mapping, source_order, receiver_order)] != 2:
                            raise TransducerError("full-factor schedule balance failed")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _logical_value(record: dict[str, Any], coord: str) -> float:
    if coord == "change_to_dirty":
        return float(record["logical_change_to_dirty_delta"])
    if coord == "probe_dirty":
        return float(record["logical_probe_dirty_delta"])
    if coord == "cycles":
        return float(record["logical_cycles_delta"])
    if coord == "duration_ns":
        return float(record["logical_duration_ns_delta"])
    if coord == "change_to_dirty_per_cycle":
        return _ratio(float(record["positive_change_to_dirty"]), float(record["positive_cycles"])) - _ratio(
            float(record["negative_change_to_dirty"]), float(record["negative_cycles"])
        )
    if coord == "probe_dirty_per_cycle":
        return _ratio(float(record["positive_probe_dirty"]), float(record["positive_cycles"])) - _ratio(
            float(record["negative_probe_dirty"]), float(record["negative_cycles"])
        )
    raise KeyError(coord)


def _physical_value(record: dict[str, Any], coord: str) -> float:
    if coord == "change_to_dirty":
        return float(record["physical_a_minus_b_change_to_dirty_delta"])
    if coord == "probe_dirty":
        return float(record["physical_a_minus_b_probe_dirty_delta"])
    if coord == "cycles":
        return float(record["physical_a_minus_b_cycles_delta"])
    if coord == "duration_ns":
        return float(record["physical_a_minus_b_duration_ns_delta"])
    if coord == "change_to_dirty_per_cycle":
        return _ratio(float(record["physical_a_change_to_dirty"]), float(record["physical_a_cycles"])) - _ratio(
            float(record["physical_b_change_to_dirty"]), float(record["physical_b_cycles"])
        )
    if coord == "probe_dirty_per_cycle":
        return _ratio(float(record["physical_a_probe_dirty"]), float(record["physical_a_cycles"])) - _ratio(
            float(record["physical_b_probe_dirty"]), float(record["physical_b_cycles"])
        )
    raise KeyError(coord)


def _sentinel_pre(sentinel: dict[str, Any], coord: str) -> float:
    if coord == "change_to_dirty":
        return float(sentinel["pre_a_minus_b_change_to_dirty_delta"])
    if coord == "probe_dirty":
        return float(sentinel["pre_a_minus_b_probe_dirty_delta"])
    if coord == "cycles":
        return float(sentinel["pre_a_minus_b_cycles_delta"])
    if coord == "duration_ns":
        return float(sentinel["pre_a_minus_b_duration_ns_delta"])
    if coord == "change_to_dirty_per_cycle":
        return _ratio(float(sentinel["pre_a_change_to_dirty"]), float(sentinel["pre_a_cycles"])) - _ratio(
            float(sentinel["pre_b_change_to_dirty"]), float(sentinel["pre_b_cycles"])
        )
    if coord == "probe_dirty_per_cycle":
        return _ratio(float(sentinel["pre_a_probe_dirty"]), float(sentinel["pre_a_cycles"])) - _ratio(
            float(sentinel["pre_b_probe_dirty"]), float(sentinel["pre_b_cycles"])
        )
    raise KeyError(coord)


def _sentinel_post(sentinel: dict[str, Any], coord: str) -> float:
    if coord == "change_to_dirty":
        return float(sentinel["post_a_minus_b_change_to_dirty_delta"])
    if coord == "probe_dirty":
        return float(sentinel["post_a_minus_b_probe_dirty_delta"])
    if coord == "cycles":
        return float(sentinel["post_a_minus_b_cycles_delta"])
    if coord == "duration_ns":
        return float(sentinel["post_a_minus_b_duration_ns_delta"])
    if coord == "change_to_dirty_per_cycle":
        return _ratio(float(sentinel["post_a_change_to_dirty"]), float(sentinel["post_a_cycles"])) - _ratio(
            float(sentinel["post_b_change_to_dirty"]), float(sentinel["post_b_cycles"])
        )
    if coord == "probe_dirty_per_cycle":
        return _ratio(float(sentinel["post_a_probe_dirty"]), float(sentinel["post_a_cycles"])) - _ratio(
            float(sentinel["post_b_probe_dirty"]), float(sentinel["post_b_cycles"])
        )
    raise KeyError(coord)


def _sentinel_bank(sentinel: dict[str, Any], coord: str, bank: str, phase: str) -> float:
    prefix = f"{phase}_{bank}"
    if coord == "change_to_dirty":
        return float(sentinel[f"{prefix}_change_to_dirty"])
    if coord == "probe_dirty":
        return float(sentinel[f"{prefix}_probe_dirty"])
    if coord == "cycles":
        return float(sentinel[f"{prefix}_cycles"])
    if coord == "duration_ns":
        return float(sentinel[f"{prefix}_duration_ns"])
    if coord == "change_to_dirty_per_cycle":
        return _ratio(float(sentinel[f"{prefix}_change_to_dirty"]), float(sentinel[f"{prefix}_cycles"]))
    if coord == "probe_dirty_per_cycle":
        return _ratio(float(sentinel[f"{prefix}_probe_dirty"]), float(sentinel[f"{prefix}_cycles"]))
    raise KeyError(coord)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _relative_error(left: float, right: float, floor: float = ABS_FLOOR) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


def _oddness_error(left: float, right: float) -> float:
    return abs(left + right) / max(abs(left), abs(right), ABS_FLOOR)


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _record_pmu_ok(record: dict[str, Any], prefix: str) -> bool:
    return (
        bool(record[f"{prefix}_opened"])
        and bool(record[f"{prefix}_read_ok"])
        and bool(record[f"{prefix}_event_order_ok"])
        and bool(record[f"{prefix}_unmultiplexed"])
        and int(record[f"{prefix}_time_enabled"]) == int(record[f"{prefix}_time_running"])
        and int(record[f"{prefix}_time_enabled"]) > 0
        and int(record[f"{prefix}_cpu_before"]) == 5
        and int(record[f"{prefix}_cpu_after"]) == 5
    )


def _sentinel_pmu_ok(sentinel: dict[str, Any], prefix: str) -> bool:
    return (
        bool(sentinel[f"{prefix}_opened"])
        and bool(sentinel[f"{prefix}_read_ok"])
        and bool(sentinel[f"{prefix}_event_order_ok"])
        and bool(sentinel[f"{prefix}_unmultiplexed"])
        and int(sentinel[f"{prefix}_time_enabled"]) == int(sentinel[f"{prefix}_time_running"])
        and int(sentinel[f"{prefix}_time_enabled"]) > 0
        and int(sentinel[f"{prefix}_cpu_before"]) == 5
        and int(sentinel[f"{prefix}_cpu_after"]) == 5
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
    if len(raw_records) != len(trial_map) or len(sentinels) != len(trial_map):
        raise TransducerError("raw, sentinel, and schedule counts differ")
    seen_raw: set[tuple[int, int]] = set()
    seen_sentinel: set[tuple[int, int]] = set()
    failures: list[str] = []
    for record in raw_records:
        if record.get("schema_id") != RAW_SCHEMA:
            failures.append("raw schema mismatch")
            continue
        key = (int(record["replicate_index"]), int(record["trial_index"]))
        seen_raw.add(key)
        trial = trial_map.get(key)
        if trial is None:
            failures.append(f"raw trial not in schedule: {key}")
            continue
        for schedule_key in ("pair_index", "leg_index", "repeat_index"):
            if int(record[schedule_key]) != int(trial[schedule_key]):
                failures.append(f"raw {schedule_key} drift: {key}")
        if int(record["q"]) != int(trial["q"]) or int(record["mapping"]) != int(trial["mapping"]):
            failures.append(f"raw schedule drift: {key}")
        if int(record["mapping_order_first"]) != int(trial["mapping_order_first"]):
            failures.append(f"raw mapping-order drift: {key}")
        if str(record["measurement_order"]) != str(trial["measurement_order"]):
            failures.append(f"raw order drift: {key}")
        if str(record["source_order"]) != str(trial["source_order"]):
            failures.append(f"raw source-order drift: {key}")
        expected_positive = BASE_WORK + int(record["q"])
        expected_negative = BASE_WORK - int(record["q"])
        if int(record["positive_work"]) != expected_positive or int(record["negative_work"]) != expected_negative:
            failures.append(f"work law drift: {key}")
        if int(record["source_total_work"]) != TOTAL_WORK:
            failures.append(f"unequal source work: {key}")
        if int(record["positive_work"]) >= BANK_LINES or int(record["negative_work"]) >= BANK_LINES:
            failures.append(f"work prefix reaches full bank: {key}")
        if int(record["line_permutation_a"]) != PERM_A or int(record["line_permutation_b"]) != PERM_B:
            failures.append(f"line permutation drift: {key}")
        if not bool(record["positive_prefix_unique"]) or not bool(record["negative_prefix_unique"]):
            failures.append(f"duplicate source prefix: {key}")
        if not _record_pmu_ok(record, "positive") or not _record_pmu_ok(record, "negative"):
            failures.append(f"main PMU custody failed: {key}")
        if not bool(record["byte_compare_passed"]):
            failures.append(f"byte compare failed: {key}")
        if not bool(record["restoration_passed"]) or not bool(record["trial_ok"]):
            failures.append(f"trial restoration/integrity failed: {key}")
    for sentinel in sentinels:
        if sentinel.get("schema_id") != SENTINEL_SCHEMA:
            failures.append("sentinel schema mismatch")
            continue
        key = (int(sentinel["replicate_index"]), int(sentinel["trial_index"]))
        seen_sentinel.add(key)
        trial = trial_map.get(key)
        if trial is None:
            failures.append(f"sentinel trial not in schedule: {key}")
            continue
        for schedule_key in ("pair_index", "leg_index", "repeat_index"):
            if int(sentinel[schedule_key]) != int(trial[schedule_key]):
                failures.append(f"sentinel {schedule_key} drift: {key}")
        if int(sentinel["q"]) != int(trial["q"]) or int(sentinel["mapping"]) != int(trial["mapping"]):
            failures.append(f"sentinel schedule drift: {key}")
        if int(sentinel["mapping_order_first"]) != int(trial["mapping_order_first"]):
            failures.append(f"sentinel mapping-order drift: {key}")
        if str(sentinel["source_order"]) != str(trial["source_order"]):
            failures.append(f"sentinel source-order drift: {key}")
        if str(sentinel["measurement_order"]) != str(trial["measurement_order"]):
            failures.append(f"sentinel receiver-order drift: {key}")
        if not bool(sentinel["bytes_unchanged"]):
            failures.append(f"sentinel byte restoration failed: {key}")
        if not bool(sentinel["byte_compare_passed"]):
            failures.append(f"sentinel byte compare failed: {key}")
        if not all(_sentinel_pmu_ok(sentinel, name) for name in ("pre_a", "pre_b", "post_a", "post_b")):
            failures.append(f"sentinel PMU custody failed: {key}")
    if seen_raw != set(trial_map):
        failures.append("raw trial set differs from schedule")
    if seen_sentinel != set(trial_map):
        failures.append("sentinel trial set differs from schedule")
    return {
        "schedule_matched": not failures,
        "failure_count": len(failures),
        "failures": failures[:32],
    }


def extract_features(
    schedule: dict[str, Any],
    raw_records: list[dict[str, Any]],
    sentinels: list[dict[str, Any]],
) -> dict[str, Any]:
    integrity = validate_raw_against_schedule(schedule, raw_records, sentinels)
    sentinel_by_key = {
        (int(item["replicate_index"]), int(item["trial_index"])): item for item in sentinels
    }
    trial_features: list[dict[str, Any]] = []
    for record in sorted(raw_records, key=lambda item: (int(item["replicate_index"]), int(item["trial_index"]))):
        key = (int(record["replicate_index"]), int(record["trial_index"]))
        sentinel = sentinel_by_key[key]
        coords = {
            coord: {
                "logical": _logical_value(record, coord),
                "physical_a_minus_b": _physical_value(record, coord),
                "sentinel_pre_a_minus_b": _sentinel_pre(sentinel, coord),
                "sentinel_post_a_minus_b": _sentinel_post(sentinel, coord),
                "sentinel_variation": abs(_sentinel_post(sentinel, coord) - _sentinel_pre(sentinel, coord)),
            }
            for coord in COORDINATES
        }
        trial_features.append(
            {
                "replicate_index": int(record["replicate_index"]),
                "pair_index": int(record["pair_index"]),
                "leg_index": int(record["leg_index"]),
                "trial_index": int(record["trial_index"]),
                "repeat_index": int(record["repeat_index"]),
                "q": int(record["q"]),
                "q0_role": str(next(
                    trial["q0_role"]
                    for trial in schedule["trials"]
                    if int(trial["replicate_index"]) == int(record["replicate_index"])
                    and int(trial["trial_index"]) == int(record["trial_index"])
                )),
                "mapping": int(record["mapping"]),
                "mapping_order_first": int(record["mapping_order_first"]),
                "source_order": str(record["source_order"]),
                "measurement_order": str(record["measurement_order"]),
                "coordinates": coords,
                "sentinel": sentinel,
                "trial_ok": bool(record["trial_ok"]),
                "restoration_passed": bool(record["restoration_passed"]),
                "byte_compare_passed": bool(record["byte_compare_passed"]),
            }
        )
    features = {
        "schema_id": FEATURE_SCHEMA,
        "schedule_sha256": schedule["schedule_sha256"],
        "integrity": integrity,
        "trial_features": trial_features,
    }
    features["features_sha256"] = digest({k: v for k, v in features.items() if k != "features_sha256"})
    return features


def _select_trials(features: dict[str, Any], replicate: int | None = None) -> list[dict[str, Any]]:
    rows = features["trial_features"]
    if replicate is None:
        return list(rows)
    return [row for row in rows if int(row["replicate_index"]) == replicate]


def _coord_values(
    rows: list[dict[str, Any]],
    coord: str,
    *,
    q: int | None = None,
    mapping: int | None = None,
    order: str | None = None,
    source_order: str | None = None,
    mapping_order_first: int | None = None,
    q0_role: str | None = None,
    physical: bool = False,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if q is not None and int(row["q"]) != q:
            continue
        if mapping is not None and int(row["mapping"]) != mapping:
            continue
        if order is not None and str(row["measurement_order"]) != order:
            continue
        if source_order is not None and str(row["source_order"]) != source_order:
            continue
        if mapping_order_first is not None and int(row["mapping_order_first"]) != mapping_order_first:
            continue
        if q0_role is not None and str(row["q0_role"]) != q0_role:
            continue
        key = "physical_a_minus_b" if physical else "logical"
        values.append(float(row["coordinates"][coord][key]))
    return values


def _mean_coord(
    rows: list[dict[str, Any]],
    coord: str,
    *,
    q: int,
    mapping: int | None = None,
    order: str | None = None,
    source_order: str | None = None,
    physical: bool = False,
) -> float:
    return _mean(
        _coord_values(
            rows,
            coord,
            q=q,
            mapping=mapping,
            order=order,
            source_order=source_order,
            physical=physical,
        )
    )


def _null_ceiling(rows: list[dict[str, Any]], coord: str, include_fresh: bool) -> dict[str, float]:
    q0_values = [abs(value) for value in _coord_values(rows, coord, q=0, q0_role="null_build")]
    mapping_residuals = []
    order_residuals = []
    source_order_residuals = []
    mapping_order_residuals = []
    for rep in sorted({int(row["replicate_index"]) for row in rows}):
        rep_rows = [row for row in rows if int(row["replicate_index"]) == rep]
        mapping_residuals.append(
            abs(_mean_coord(rep_rows, coord, q=0, mapping=0) - _mean_coord(rep_rows, coord, q=0, mapping=1))
        )
        order_residuals.append(
            abs(
                _mean_coord(rep_rows, coord, q=0, order="positive_first")
                - _mean_coord(rep_rows, coord, q=0, order="negative_first")
            )
        )
        source_order_residuals.append(
            abs(
                _mean_coord(rep_rows, coord, q=0, source_order="positive_first")
                - _mean_coord(rep_rows, coord, q=0, source_order="negative_first")
            )
        )
        mapping_order_residuals.append(
            abs(
                _mean(_coord_values(rep_rows, coord, q=0, mapping_order_first=0))
                - _mean(_coord_values(rep_rows, coord, q=0, mapping_order_first=1))
            )
        )
    sentinel_variations = [abs(float(row["coordinates"][coord]["sentinel_variation"])) for row in rows]
    fresh_variation = 0.0
    if include_fresh:
        per_rep_zero = [
            _mean(_coord_values([row for row in rows if int(row["replicate_index"]) == rep], coord, q=0, q0_role="null_build"))
            for rep in REPLICATES
        ]
        fresh_variation = abs(per_rep_zero[0] - per_rep_zero[1])
    parts = {
        "q0_abs": max(q0_values, default=0.0),
        "q0_mapping_residual": max(mapping_residuals, default=0.0),
        "q0_measurement_order_residual": max(order_residuals, default=0.0),
        "q0_source_order_residual": max(source_order_residuals, default=0.0),
        "q0_mapping_order_residual": max(mapping_order_residuals, default=0.0),
        "restoration_sentinel_variation": max(sentinel_variations, default=0.0),
        "fresh_process_variation": fresh_variation,
        "absolute_floor": ABS_FLOOR,
    }
    parts["complete_null_ceiling"] = max(parts.values())
    return parts


def _sentinel_law(rows: list[dict[str, Any]], coord: str) -> dict[str, Any]:
    errors = []
    floor = RESTORATION_FLOOR_BY_COORDINATE[coord]
    for row in rows:
        sentinel = row["sentinel"]
        pre_a = _sentinel_bank(sentinel, coord, "a", "pre")
        pre_b = _sentinel_bank(sentinel, coord, "b", "pre")
        post_a = _sentinel_bank(sentinel, coord, "a", "post")
        post_b = _sentinel_bank(sentinel, coord, "b", "post")
        pairs = (
            (pre_a, post_a),
            (pre_b, post_b),
            (pre_a - pre_b, post_a - post_b),
            (pre_a + pre_b, post_a + post_b),
        )
        for pre, post in pairs:
            errors.append(abs(post - pre) / max(abs(pre), abs(post), floor))
    max_error = max(errors, default=0.0)
    return {"passed": max_error <= SENTINEL_TOL, "max_error": max_error, "floor": floor}


def _evaluate_scope(rows: list[dict[str, Any]], coord: str, *, include_fresh: bool) -> dict[str, Any]:
    null_parts = _null_ceiling(rows, coord, include_fresh)
    null_ceiling = null_parts["complete_null_ceiling"]
    means = {q: _mean_coord(rows, coord, q=q) for q in Q_LADDER}
    zero_test_values = _coord_values(rows, coord, q=0, q0_role="null_test")
    zero_test_mean = _mean(zero_test_values)
    q0_law = abs(zero_test_mean) <= null_ceiling
    oddness_errors = {
        q: _oddness_error(means[q], means[-q])
        for q in (512, 1024, 1536)
    }
    oddness_law = all(value <= ODDNESS_TOL for value in oddness_errors.values())
    resolved = {q: means[q] for q in Q_LADDER if q != 0 and abs(means[q]) > null_ceiling}
    direct_sign = bool(resolved) and all(_sign(value) == _sign(q) for q, value in resolved.items())
    reversed_sign = bool(resolved) and all(_sign(value) == -_sign(q) for q, value in resolved.items())
    sign_law = direct_sign or reversed_sign
    convention = "direct" if direct_sign else ("reversed" if reversed_sign else "none")
    gain_floor = GAIN_MULTIPLIER * null_ceiling
    high_values = [abs(means[q]) for q in (1024, -1024, 1536, -1536)]
    gain_law = min(high_values) > gain_floor
    positive_monotonic = abs(means[512]) < abs(means[1024]) < abs(means[1536])
    negative_monotonic = abs(means[-512]) < abs(means[-1024]) < abs(means[-1536])
    monotonicity_law = positive_monotonic and negative_monotonic
    pointer_errors = []
    physical_errors = []
    for q in Q_LADDER:
        m0 = _mean_coord(rows, coord, q=q, mapping=0)
        m1 = _mean_coord(rows, coord, q=q, mapping=1)
        pointer_errors.append(_relative_error(m0, m1))
        p0 = _mean_coord(rows, coord, q=q, mapping=0, physical=True)
        p1 = _mean_coord(rows, coord, q=q, mapping=1, physical=True)
        physical_errors.append(abs(p0 + p1) / max(abs(p0), abs(p1), ABS_FLOOR))
    pointer_swap_law = max(pointer_errors, default=0.0) <= POINTER_SWAP_TOL
    physical_reversal_law = max(physical_errors, default=0.0) <= POINTER_SWAP_TOL
    order_errors = []
    source_order_errors = []
    for q in Q_LADDER:
        positive_first = _mean_coord(rows, coord, q=q, order="positive_first")
        negative_first = _mean_coord(rows, coord, q=q, order="negative_first")
        order_errors.append(_relative_error(positive_first, negative_first))
        source_positive_first = _mean_coord(rows, coord, q=q, source_order="positive_first")
        source_negative_first = _mean_coord(rows, coord, q=q, source_order="negative_first")
        source_order_errors.append(_relative_error(source_positive_first, source_negative_first))
    measurement_order_law = max(order_errors, default=0.0) <= ORDER_TOL
    source_order_law = max(source_order_errors, default=0.0) <= ORDER_TOL
    stratum_results: list[dict[str, Any]] = []
    for mapping in MAPPINGS:
        for source_order in ("positive_first", "negative_first"):
            for receiver_order in ("positive_first", "negative_first"):
                stratum_rows = [
                    row for row in rows
                    if int(row["mapping"]) == mapping
                    and str(row["source_order"]) == source_order
                    and str(row["measurement_order"]) == receiver_order
                ]
                stratum_means = {q: _mean_coord(stratum_rows, coord, q=q) for q in Q_LADDER}
                stratum_resolved = {
                    q: stratum_means[q]
                    for q in Q_LADDER
                    if q != 0 and abs(stratum_means[q]) > null_ceiling
                }
                stratum_direct = bool(stratum_resolved) and all(
                    _sign(value) == _sign(q) for q, value in stratum_resolved.items()
                )
                stratum_reversed = bool(stratum_resolved) and all(
                    _sign(value) == -_sign(q) for q, value in stratum_resolved.items()
                )
                stratum_oddness = all(
                    _oddness_error(stratum_means[q], stratum_means[-q]) <= ODDNESS_TOL
                    for q in (512, 1024, 1536)
                )
                stratum_gain = min(abs(stratum_means[q]) for q in (1024, -1024, 1536, -1536)) > gain_floor
                stratum_results.append(
                    {
                        "mapping": mapping,
                        "source_order": source_order,
                        "measurement_order": receiver_order,
                        "sign_convention": "direct" if stratum_direct else ("reversed" if stratum_reversed else "none"),
                        "sign_law": stratum_direct or stratum_reversed,
                        "oddness_law": stratum_oddness,
                        "gain_law": stratum_gain,
                        "passed": (stratum_direct or stratum_reversed) and stratum_oddness and stratum_gain,
                    }
                )
    stratum_law = all(item["passed"] and item["sign_convention"] == convention for item in stratum_results)
    sentinel = _sentinel_law(rows, coord)
    integrity_law = all(
        bool(row["trial_ok"]) and bool(row["restoration_passed"]) and bool(row["byte_compare_passed"])
        for row in rows
    )
    restoration_law = integrity_law and sentinel["passed"]
    laws = {
        "zero_law": q0_law,
        "oddness_law": oddness_law,
        "sign_law": sign_law,
        "gain_law": gain_law,
        "monotonicity_law": monotonicity_law,
        "logical_pointer_swap_invariance_law": pointer_swap_law,
        "physical_pointer_swap_reversal_law": physical_reversal_law,
        "source_order_law": source_order_law,
        "measurement_order_law": measurement_order_law,
        "stratum_transfer_law": stratum_law,
        "restoration_law": restoration_law,
    }
    return {
        "passed": all(laws.values()),
        "laws": laws,
        "means_by_q": {str(k): v for k, v in means.items()},
        "zero_test_mean": zero_test_mean,
        "null_components": null_parts,
        "gain_floor": gain_floor,
        "oddness_errors": {str(k): v for k, v in oddness_errors.items()},
        "sign_convention": convention,
        "max_pointer_swap_error": max(pointer_errors, default=0.0),
        "max_physical_reversal_error": max(physical_errors, default=0.0),
        "max_measurement_order_error": max(order_errors, default=0.0),
        "max_source_order_error": max(source_order_errors, default=0.0),
        "stratum_results": stratum_results,
        "restoration_sentinel": sentinel,
    }


def adjudicate(features: dict[str, Any]) -> dict[str, Any]:
    if features.get("schema_id") != FEATURE_SCHEMA:
        raise TransducerError("feature schema mismatch")
    coordinate_results: dict[str, Any] = {}
    eligible: list[str] = []
    partial_candidates: list[str] = []
    hard_integrity = bool(features["integrity"]["schedule_matched"])
    for coord in COORDINATES:
        per_replicate = {
            str(rep): _evaluate_scope(_select_trials(features, rep), coord, include_fresh=False)
            for rep in REPLICATES
        }
        aggregate = _evaluate_scope(_select_trials(features), coord, include_fresh=True)
        replicate_conventions = [per_replicate[str(rep)]["sign_convention"] for rep in REPLICATES]
        same_replicate_convention = (
            replicate_conventions[0] == replicate_conventions[1] == aggregate["sign_convention"]
            and aggregate["sign_convention"] != "none"
        )
        replicate_errors = []
        for q in Q_LADDER:
            if q == 0:
                continue
            left = float(per_replicate["0"]["means_by_q"][str(q)])
            right = float(per_replicate["1"]["means_by_q"][str(q)])
            replicate_errors.append(_relative_error(left, right))
        replicate_consistency = max(replicate_errors, default=0.0) <= REPLICATE_TOL
        diagnostic_only = coord in DIAGNOSTIC_COORDINATES
        coord_passed = (
            hard_integrity
            and not diagnostic_only
            and aggregate["passed"]
            and all(per_replicate[str(rep)]["passed"] for rep in REPLICATES)
            and same_replicate_convention
            and replicate_consistency
        )
        if coord_passed:
            eligible.append(coord)
        shape_present = same_replicate_convention and aggregate["laws"]["sign_law"] and (
            aggregate["laws"]["oddness_law"] or aggregate["laws"]["monotonicity_law"]
        ) and all(
            per_replicate[str(rep)]["laws"]["sign_law"]
            and (
                per_replicate[str(rep)]["laws"]["oddness_law"]
                or per_replicate[str(rep)]["laws"]["monotonicity_law"]
            )
            for rep in REPLICATES
        )
        if shape_present and not coord_passed:
            partial_candidates.append(coord)
        coordinate_results[coord] = {
            "passed": coord_passed,
            "diagnostic_only": diagnostic_only,
            "hard_integrity": hard_integrity,
            "same_replicate_sign_convention": same_replicate_convention,
            "replicate_sign_conventions": replicate_conventions,
            "replicate_consistency_law": replicate_consistency,
            "max_replicate_error": max(replicate_errors, default=0.0),
            "aggregate": aggregate,
            "per_replicate": per_replicate,
        }
    primary = next((coord for coord in COORDINATE_PRIORITY if coord in eligible), None)
    if primary is not None:
        status = "BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED"
    elif partial_candidates:
        status = "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL"
    else:
        status = "BALANCED_PHYSICAL_TRANSDUCER_NOT_ESTABLISHED"
    result = {
        "schema_id": ADJUDICATION_SCHEMA,
        "status": status,
        "claim_ceiling": "public balanced transducer only; no OrbitState coupling candidate and no Small Wall crossing claim",
        "eligible_coordinates": eligible,
        "primary_coordinate": primary,
        "partial_candidate_coordinates": partial_candidates,
        "coordinate_priority": list(COORDINATE_PRIORITY),
        "coordinate_results": coordinate_results,
        "integrity": features["integrity"],
    }
    result["adjudication_sha256"] = digest({k: v for k, v in result.items() if k != "adjudication_sha256"})
    return result


def _mock_window(prefix: str, value: float, cycles_base: float = 100000.0) -> dict[str, Any]:
    return {
        f"{prefix}_opened": True,
        f"{prefix}_read_ok": True,
        f"{prefix}_event_order_ok": True,
        f"{prefix}_unmultiplexed": True,
        f"{prefix}_open_errno": 0,
        f"{prefix}_read_errno": 0,
        f"{prefix}_cpu_before": 5,
        f"{prefix}_cpu_after": 5,
        f"{prefix}_cycles": int(cycles_base + value),
        f"{prefix}_change_to_dirty": int(200000 + value),
        f"{prefix}_probe_dirty": int(400000 + 2.0 * value),
        f"{prefix}_duration_ns": int(80000 + value),
        f"{prefix}_time_enabled": 1000,
        f"{prefix}_time_running": 1000,
        f"{prefix}_event_ids": [101, 102, 103],
    }


def build_mock_capture(kind: str) -> dict[str, Any]:
    schedule = build_schedule()

    def transfer(q: int, rep: int) -> float:
        if kind == "ideal_odd":
            return 4.0 * q
        if kind == "global_sign_reversed":
            return -4.0 * q
        if kind == "zero":
            return 0.0
        if kind == "nonlinear_monotonic":
            return math.copysign((abs(q) / 512.0) ** 2 * 2000.0, q) if q else 0.0
        if kind == "non_odd":
            return abs(q) * 4.0
        if kind == "contradictory_fresh_process":
            return (4.0 * q) if rep == 0 else (-4.0 * q)
        if kind == "unequal_contradictory_fresh_process":
            return (4.0 * q) if rep == 0 else (-2.0 * q)
        return 4.0 * q

    raw_records: list[dict[str, Any]] = []
    sentinels: list[dict[str, Any]] = []
    for trial in schedule["trials"]:
        rep = int(trial["replicate_index"])
        q = int(trial["q"])
        mapping = int(trial["mapping"])
        f_value = transfer(q, rep)
        bias = 2000.0 if kind == "fixed_physical_bank_bias" else 0.0
        if mapping == 0:
            physical_a = 200000.0 + f_value / 2.0 + bias
            physical_b = 200000.0 - f_value / 2.0 - bias
            positive_value = physical_a
            negative_value = physical_b
        else:
            physical_a = 200000.0 - f_value / 2.0 + bias
            physical_b = 200000.0 + f_value / 2.0 - bias
            positive_value = physical_b
            negative_value = physical_a
        if kind == "measurement_order_bias":
            if trial["receiver_positive_first"]:
                positive_value += 2500.0
            else:
                negative_value += 2500.0
        positive_delta = positive_value - 200000.0
        negative_delta = negative_value - 200000.0
        if kind == "source_order_bias":
            if trial["source_order"] == "positive_first":
                positive_delta += 2500.0
            else:
                negative_delta += 2500.0
        pos_window = _mock_window("positive", positive_delta)
        neg_window = _mock_window("negative", negative_delta)
        if kind == "multiplexed_pmu":
            pos_window["positive_time_running"] = pos_window["positive_time_enabled"] - 1
        positive_work = BASE_WORK + q
        negative_work = BASE_WORK - q
        record = {
            "schema_id": RAW_SCHEMA,
            "replicate_index": rep,
            "pair_index": int(trial["pair_index"]),
            "leg_index": int(trial["leg_index"]),
            "trial_index": int(trial["trial_index"]),
            "repeat_index": int(trial["repeat_index"]),
            "q": q,
            "mapping": mapping,
            "mapping_order_first": int(trial["mapping_order_first"]),
            "source_order": trial["source_order"],
            "measurement_order": trial["measurement_order"],
            "logical_positive_physical": "A" if mapping == 0 else "B",
            "logical_negative_physical": "B" if mapping == 0 else "A",
            "positive_work": positive_work,
            "negative_work": negative_work,
            "source_total_work": TOTAL_WORK,
            "line_permutation_a": PERM_A,
            "line_permutation_b": PERM_B,
            "positive_prefix_unique": kind != "duplicate_line_prefix",
            "negative_prefix_unique": True,
            "baseline_rc": 0,
            "source_rc": 0,
            "positive_rc": 0,
            "negative_rc": 0,
            "restoration_passed": kind != "restoration_failure",
            "byte_compare_passed": kind != "restoration_failure",
            "pmu_windows_ok": kind != "multiplexed_pmu",
            "trial_ok": kind not in ("multiplexed_pmu", "duplicate_line_prefix", "unequal_work", "restoration_failure"),
            "initial_a_digest": "abc",
            "initial_b_digest": "abc",
            "baseline_a_digest": "abc",
            "baseline_b_digest": "abc",
            "final_a_digest": "abc",
            "final_b_digest": "abc",
            **pos_window,
            **neg_window,
            "logical_cycles_delta": positive_delta - negative_delta,
            "logical_change_to_dirty_delta": positive_delta - negative_delta,
            "logical_probe_dirty_delta": 2.0 * (positive_delta - negative_delta),
            "logical_duration_ns_delta": positive_delta - negative_delta,
            "physical_a_cycles": int(100000 + physical_a - 200000.0),
            "physical_b_cycles": int(100000 + physical_b - 200000.0),
            "physical_a_change_to_dirty": int(physical_a),
            "physical_b_change_to_dirty": int(physical_b),
            "physical_a_probe_dirty": int(2.0 * physical_a),
            "physical_b_probe_dirty": int(2.0 * physical_b),
            "physical_a_duration_ns": int(80000 + physical_a - 200000.0),
            "physical_b_duration_ns": int(80000 + physical_b - 200000.0),
            "physical_a_minus_b_cycles_delta": physical_a - physical_b,
            "physical_a_minus_b_change_to_dirty_delta": physical_a - physical_b,
            "physical_a_minus_b_probe_dirty_delta": 2.0 * (physical_a - physical_b),
            "physical_a_minus_b_duration_ns_delta": physical_a - physical_b,
        }
        if kind == "unequal_work":
            record["source_total_work"] = TOTAL_WORK - 1
        raw_records.append(record)
        sentinel = {
            "schema_id": SENTINEL_SCHEMA,
            "replicate_index": rep,
            "pair_index": int(trial["pair_index"]),
            "leg_index": int(trial["leg_index"]),
            "trial_index": int(trial["trial_index"]),
            "repeat_index": int(trial["repeat_index"]),
            "q": q,
            "mapping": mapping,
            "mapping_order_first": int(trial["mapping_order_first"]),
            "source_order": trial["source_order"],
            "measurement_order": trial["measurement_order"],
            "pre_a_rc": 0,
            "pre_b_rc": 0,
            "post_a_rc": 0,
            "post_b_rc": 0,
            "bytes_unchanged": kind != "restoration_failure",
            "byte_compare_passed": kind != "restoration_failure",
            **_mock_window("pre_a", 0.0),
            **_mock_window("pre_b", 0.0),
            **_mock_window("post_a", 0.0 if kind != "restoration_failure" else 5000.0),
            **_mock_window("post_b", 0.0),
            "pre_a_minus_b_cycles_delta": 0.0,
            "post_a_minus_b_cycles_delta": 0.0 if kind != "restoration_failure" else 5000.0,
            "pre_a_minus_b_change_to_dirty_delta": 0.0,
            "post_a_minus_b_change_to_dirty_delta": 0.0 if kind != "restoration_failure" else 5000.0,
            "pre_a_minus_b_probe_dirty_delta": 0.0,
            "post_a_minus_b_probe_dirty_delta": 0.0 if kind != "restoration_failure" else 10000.0,
            "pre_a_minus_b_duration_ns_delta": 0.0,
            "post_a_minus_b_duration_ns_delta": 0.0 if kind != "restoration_failure" else 5000.0,
        }
        sentinels.append(sentinel)
    return {"schedule": schedule, "raw_records": raw_records, "sentinels": sentinels}


def self_test() -> dict[str, Any]:
    expected = {
        "ideal_odd": "BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED",
        "global_sign_reversed": "BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED",
        "zero": "BALANCED_PHYSICAL_TRANSDUCER_NOT_ESTABLISHED",
        "fixed_physical_bank_bias": "BALANCED_PHYSICAL_TRANSDUCER_NOT_ESTABLISHED",
        "source_order_bias": "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "measurement_order_bias": "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "nonlinear_monotonic": "BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED",
        "non_odd": "BALANCED_PHYSICAL_TRANSDUCER_NOT_ESTABLISHED",
        "multiplexed_pmu": "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "duplicate_line_prefix": "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "unequal_work": "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "restoration_failure": "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "contradictory_fresh_process": "BALANCED_PHYSICAL_TRANSDUCER_NOT_ESTABLISHED",
        "unequal_contradictory_fresh_process": "BALANCED_PHYSICAL_TRANSDUCER_NOT_ESTABLISHED",
    }
    cases: dict[str, Any] = {}
    for kind, status in expected.items():
        capture = build_mock_capture(kind)
        features = extract_features(capture["schedule"], capture["raw_records"], capture["sentinels"])
        adjudication = adjudicate(features)
        cases[kind] = {
            "expected": status,
            "actual": adjudication["status"],
            "passed": adjudication["status"] == status,
            "primary_coordinate": adjudication["primary_coordinate"],
            "eligible_coordinates": adjudication["eligible_coordinates"],
        }
    result = {
        "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_PUBLIC_SELF_TEST_V1",
        "cases": cases,
    }
    result["self_test_passed"] = all(item["passed"] for item in cases.values())
    result["self_test_sha256"] = digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--mock-runtime", choices=(
        "ideal_odd",
        "global_sign_reversed",
        "zero",
        "fixed_physical_bank_bias",
        "source_order_bias",
        "measurement_order_bias",
        "nonlinear_monotonic",
        "non_odd",
        "multiplexed_pmu",
        "duplicate_line_prefix",
        "unequal_work",
        "restoration_failure",
        "contradictory_fresh_process",
        "unequal_contradictory_fresh_process",
    ))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        result = self_test()
        exit_ok = result["self_test_passed"]
    elif args.schedule:
        result = build_schedule()
        exit_ok = True
    elif args.mock_runtime:
        capture = build_mock_capture(args.mock_runtime)
        features = extract_features(capture["schedule"], capture["raw_records"], capture["sentinels"])
        result = {
            "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_MOCK_RUNTIME_V1",
            "mode": args.mock_runtime,
            "features": features,
            "adjudication": adjudicate(features),
        }
        exit_ok = True
    else:
        result = {"schema_id": "CAT_CAS_BALANCED_TRANSDUCER_PUBLIC_INFO_V1", "schedule": build_schedule()}
        exit_ok = True
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if exit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
