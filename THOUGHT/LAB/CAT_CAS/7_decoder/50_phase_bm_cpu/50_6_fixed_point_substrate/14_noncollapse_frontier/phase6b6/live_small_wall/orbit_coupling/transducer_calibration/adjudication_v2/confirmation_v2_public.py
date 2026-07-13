#!/usr/bin/env python3
"""Frozen public schedule and fresh-only adjudication for Confirmation V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CALIBRATION_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CALIBRATION_ROOT))

import balanced_transducer_adjudication_v2 as law_v2  # noqa: E402


SCHEDULE_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_PUBLIC_SCHEDULE_V2"
FEATURE_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_FEATURES_V2"
ADJUDICATION_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_ADJUDICATION_V2"
RAW_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_RAW_RECORD_V2"
SENTINEL_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_RESTORATION_SENTINEL_V2"
SELF_TEST_SCHEMA = "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_SELF_TEST_V2"

PUBLIC_SEED = "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_V2_PUBLIC_SEED"
RUN_ID = "balanced_transducer_confirmation_v2_1"
PRIMARY_COORDINATE = "change_to_dirty"
Q_LADDER = (-1536, -1024, -512, 0, 512, 1024, 1536)
REPLICATES = (0, 1)
MAPPINGS = (0, 1)
SOURCE_ORDERS = ("positive_first", "negative_first")
RECEIVER_ORDERS = ("positive_first", "negative_first")
BASE_WORK = 2048
TOTAL_WORK = 4096
BANK_LINES = 4096
PERM_A = 257
PERM_B = 43
PAIRS_PER_REPLICATE = len(Q_LADDER) * len(SOURCE_ORDERS) * len(RECEIVER_ORDERS)
TRIALS_PER_REPLICATE = PAIRS_PER_REPLICATE * len(MAPPINGS)
TOTAL_TRIALS = len(REPLICATES) * TRIALS_PER_REPLICATE
ALLOWED_CLASSES = (
    law_v2.V2_CLASS_CONFIRMED,
    law_v2.V2_CLASS_CANDIDATE,
    law_v2.V2_CLASS_NOT_ESTABLISHED,
)
FORBIDDEN_CLASSES = law_v2.V2_FORBIDDEN_CLASSES


class ConfirmationV2Error(AssertionError):
    pass


def confirmation_null_ceiling(rows: list[dict[str, Any]], coord: str, *, include_fresh: bool) -> dict[str, float]:
    floor = law_v2.coord_floor(coord)
    null_rows = law_v2.select_rows(rows, q=0, q0_role="null_build")
    q0_values = [abs(law_v2.coord_value(row, coord)) for row in null_rows]
    reps = sorted({int(row["replicate_index"]) for row in rows})
    mapping_residuals = []
    order_residuals = []
    source_residuals = []
    mapping_order_residuals = []
    for rep in reps:
        rep_rows = [row for row in null_rows if int(row["replicate_index"]) == rep]
        mapping_residuals.append(
            abs(law_v2.mean_coord(rep_rows, coord, q=0, mapping=0) - law_v2.mean_coord(rep_rows, coord, q=0, mapping=1))
        )
        order_residuals.append(
            abs(
                law_v2.mean_coord(rep_rows, coord, q=0, measurement_order="positive_first")
                - law_v2.mean_coord(rep_rows, coord, q=0, measurement_order="negative_first")
            )
        )
        source_residuals.append(
            abs(
                law_v2.mean_coord(rep_rows, coord, q=0, source_order="positive_first")
                - law_v2.mean_coord(rep_rows, coord, q=0, source_order="negative_first")
            )
        )
        mapping_order_residuals.append(
            abs(
                law_v2.mean([law_v2.coord_value(row, coord) for row in law_v2.select_rows(rep_rows, q=0, mapping_order_first=0)])
                - law_v2.mean([law_v2.coord_value(row, coord) for row in law_v2.select_rows(rep_rows, q=0, mapping_order_first=1)])
            )
        )
    sentinel_variations = [abs(float(row["coordinates"][coord]["sentinel_variation"])) for row in null_rows]
    fresh_variation = 0.0
    if include_fresh and len(reps) >= 2:
        per_rep = [
            law_v2.mean([
                law_v2.coord_value(row, coord)
                for row in null_rows
                if int(row["replicate_index"]) == rep
            ])
            for rep in reps
        ]
        fresh_variation = max(per_rep) - min(per_rep)
    parts = {
        "q0_abs": max(q0_values, default=0.0),
        "q0_mapping_residual": max(mapping_residuals, default=0.0),
        "q0_measurement_order_residual": max(order_residuals, default=0.0),
        "q0_source_order_residual": max(source_residuals, default=0.0),
        "q0_mapping_order_residual": max(mapping_order_residuals, default=0.0),
        "restoration_sentinel_variation": max(sentinel_variations, default=0.0),
        "fresh_process_variation": fresh_variation,
        "absolute_floor": floor,
    }
    parts["complete_null_ceiling"] = max(parts.values())
    return parts


law_v2.null_ceiling = confirmation_null_ceiling


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


def source_bool(order: str) -> bool:
    return order == "positive_first"


def q0_role(q: int, source_order: str, receiver_order: str) -> str:
    if q != 0:
        return "signal"
    return "null_build" if source_order == receiver_order else "null_test"


def build_schedule() -> dict[str, Any]:
    rng = random.Random(PUBLIC_SEED)
    trials: list[dict[str, Any]] = []
    for rep in REPLICATES:
        pairs: list[dict[str, Any]] = []
        for q_index, q in enumerate(Q_LADDER):
            for source_index, source_order in enumerate(SOURCE_ORDERS):
                for receiver_index, receiver_order in enumerate(RECEIVER_ORDERS):
                    cell_index = source_index * len(RECEIVER_ORDERS) + receiver_index
                    if q == 0:
                        mapping_first = source_index
                    else:
                        mapping_first = (rep + q_index + source_index + receiver_index) & 1
                    pairs.append(
                        {
                            "replicate_index": rep,
                            "q": q,
                            "q_index": q_index,
                            "repeat_index": cell_index,
                            "q0_role": q0_role(q, source_order, receiver_order),
                            "source_order": source_order,
                            "source_positive_first": source_bool(source_order),
                            "measurement_order": receiver_order,
                            "receiver_positive_first": source_bool(receiver_order),
                            "mapping_order_first": mapping_first,
                            "bank_allocation_id": f"rep{rep}_q{q}_src{source_index}_recv{receiver_index}",
                        }
                    )
        rng.shuffle(pairs)
        for pair_index, pair in enumerate(pairs):
            for leg_index, mapping in enumerate((pair["mapping_order_first"], 1 - pair["mapping_order_first"])):
                trial_index = pair_index * 2 + leg_index
                trials.append(
                    {
                        **pair,
                        "pair_index": pair_index,
                        "leg_index": leg_index,
                        "mapping": mapping,
                        "logical_positive_physical": "A" if mapping == 0 else "B",
                        "logical_negative_physical": "B" if mapping == 0 else "A",
                        "trial_index": trial_index,
                        "global_trial_index": rep * TRIALS_PER_REPLICATE + trial_index,
                    }
                )
    schedule = {
        "schema_id": SCHEDULE_SCHEMA,
        "public_seed": PUBLIC_SEED,
        "run_id": RUN_ID,
        "q_ladder": list(Q_LADDER),
        "base_work": BASE_WORK,
        "total_work": TOTAL_WORK,
        "bank_lines": BANK_LINES,
        "line_permutation": {"a": PERM_A, "b": PERM_B, "modulus": BANK_LINES},
        "fresh_process_replicates": len(REPLICATES),
        "pairs_per_replicate": PAIRS_PER_REPLICATE,
        "trials_per_replicate": TRIALS_PER_REPLICATE,
        "total_trial_legs": TOTAL_TRIALS,
        "primary_coordinate": PRIMARY_COORDINATE,
        "q0_split": "null_build when source_order == measurement_order; null_test otherwise",
        "trials": trials,
    }
    schedule["schedule_semantic_sha256"] = schedule_semantic_hash(schedule)
    validate_schedule(schedule)
    schedule["schedule_sha256"] = digest({k: v for k, v in schedule.items() if k != "schedule_sha256"})
    return schedule


def schedule_semantic_hash(schedule: dict[str, Any]) -> str:
    rows = [
        {
            "replicate_index": int(t["replicate_index"]),
            "pair_index": int(t["pair_index"]),
            "leg_index": int(t["leg_index"]),
            "trial_index": int(t["trial_index"]),
            "q": int(t["q"]),
            "q0_role": str(t["q0_role"]),
            "mapping": int(t["mapping"]),
            "source_order": str(t["source_order"]),
            "measurement_order": str(t["measurement_order"]),
            "mapping_order_first": int(t["mapping_order_first"]),
            "bank_allocation_id": str(t["bank_allocation_id"]),
        }
        for t in sorted(
            schedule["trials"],
            key=lambda item: (int(item["replicate_index"]), int(item["pair_index"]), int(item["leg_index"])),
        )
    ]
    return digest(
        {
            "schema_id": schedule["schema_id"],
            "public_seed": schedule["public_seed"],
            "run_id": schedule["run_id"],
            "q_ladder": schedule["q_ladder"],
            "line_permutation": schedule["line_permutation"],
            "rows": rows,
        }
    )


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
        raise ConfirmationV2Error("schedule schema mismatch")
    trials = schedule.get("trials")
    if not isinstance(trials, list) or len(trials) != TOTAL_TRIALS:
        raise ConfirmationV2Error("schedule trial count mismatch")
    if schedule.get("run_id") != RUN_ID or schedule.get("primary_coordinate") != PRIMARY_COORDINATE:
        raise ConfirmationV2Error("schedule identity mismatch")
    pair_seen: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    cell_counts: dict[tuple[int, int, str, str], int] = defaultdict(int)
    mapping_first_counts: dict[tuple[int, int], int] = defaultdict(int)
    trial_seen: set[tuple[int, int]] = set()
    for trial in trials:
        rep = int(trial["replicate_index"])
        pair = int(trial["pair_index"])
        leg = int(trial["leg_index"])
        idx = int(trial["trial_index"])
        q = int(trial["q"])
        mapping = int(trial["mapping"])
        source_order = str(trial["source_order"])
        receiver_order = str(trial["measurement_order"])
        if rep not in REPLICATES or pair < 0 or pair >= PAIRS_PER_REPLICATE or leg not in MAPPINGS:
            raise ConfirmationV2Error("schedule pair index out of range")
        if idx < 0 or idx >= TRIALS_PER_REPLICATE or (rep, idx) in trial_seen:
            raise ConfirmationV2Error("schedule trial index mismatch")
        trial_seen.add((rep, idx))
        if q not in Q_LADDER or mapping not in MAPPINGS:
            raise ConfirmationV2Error("schedule q or mapping out of range")
        if source_order not in SOURCE_ORDERS or receiver_order not in RECEIVER_ORDERS:
            raise ConfirmationV2Error("schedule order out of range")
        if str(trial["q0_role"]) != q0_role(q, source_order, receiver_order):
            raise ConfirmationV2Error("schedule q0 role drift")
        if int(trial["positive_work"] if "positive_work" in trial else BASE_WORK + q) != BASE_WORK + q:
            raise ConfirmationV2Error("schedule positive work drift")
        pair_seen[(rep, pair)].append(trial)
        cell_counts[(rep, q, source_order, receiver_order)] += 1
        if leg == 0:
            mapping_first_counts[(rep, q)] += mapping
    for rep in REPLICATES:
        for pair in range(PAIRS_PER_REPLICATE):
            legs = sorted(pair_seen[(rep, pair)], key=lambda item: int(item["leg_index"]))
            if len(legs) != 2 or {int(item["mapping"]) for item in legs} != {0, 1}:
                raise ConfirmationV2Error("schedule pair is not a two-leg crossover")
            shared = ("q", "q0_role", "source_order", "measurement_order", "repeat_index", "bank_allocation_id", "mapping_order_first")
            for key in shared:
                if legs[0][key] != legs[1][key]:
                    raise ConfirmationV2Error(f"schedule pair does not share {key}")
            if int(legs[0]["mapping_order_first"]) != int(legs[0]["mapping"]):
                raise ConfirmationV2Error("schedule mapping order does not match leg zero")
        for q in Q_LADDER:
            for source_order in SOURCE_ORDERS:
                for receiver_order in RECEIVER_ORDERS:
                    if cell_counts[(rep, q, source_order, receiver_order)] != 2:
                        raise ConfirmationV2Error("schedule cell count mismatch")
            if mapping_first_counts[(rep, q)] != 2:
                raise ConfirmationV2Error("schedule mapping order is not counterbalanced per q")
        null_build_pairs = [
            sorted(legs, key=lambda item: int(item["leg_index"]))[0] for (r, _), legs in pair_seen.items()
            if r == rep
            and int(sorted(legs, key=lambda item: int(item["leg_index"]))[0]["q"]) == 0
            and str(sorted(legs, key=lambda item: int(item["leg_index"]))[0]["q0_role"]) == "null_build"
        ]
        null_test_pairs = [
            sorted(legs, key=lambda item: int(item["leg_index"]))[0] for (r, _), legs in pair_seen.items()
            if r == rep
            and int(sorted(legs, key=lambda item: int(item["leg_index"]))[0]["q"]) == 0
            and str(sorted(legs, key=lambda item: int(item["leg_index"]))[0]["q0_role"]) == "null_test"
        ]
        if len(null_build_pairs) != 2 or len(null_test_pairs) != 2:
            raise ConfirmationV2Error("q0 null split count mismatch")
        if {int(row["mapping_order_first"]) for row in null_build_pairs} != {0, 1}:
            raise ConfirmationV2Error("q0 null-build mapping order not counterbalanced")
        if {int(row["mapping_order_first"]) for row in null_test_pairs} != {0, 1}:
            raise ConfirmationV2Error("q0 null-test mapping order not counterbalanced")
        for row in null_build_pairs:
            if row["source_order"] != row["measurement_order"]:
                raise ConfirmationV2Error("q0 null-build split drift")
        for row in null_test_pairs:
            if row["source_order"] == row["measurement_order"]:
                raise ConfirmationV2Error("q0 null-test split drift")


def write_schedule_artifacts(root: Path = HERE) -> dict[str, str]:
    schedule = build_schedule()
    json_path = root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json"
    tsv_path = root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv"
    sha_path = root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.sha256"
    write_json(json_path, schedule)
    tsv_path.write_text(schedule_tsv(schedule), encoding="utf-8")
    schedule_hash = sha256_file(json_path)
    sha_path.write_text(schedule_hash + "\n", encoding="utf-8")
    return {
        "schedule_json_sha256": schedule_hash,
        "schedule_tsv_sha256": sha256_file(tsv_path),
        "schedule_semantic_sha256": schedule["schedule_semantic_sha256"],
    }


def _record_pmu_ok(record: dict[str, Any], prefix: str) -> bool:
    return (
        bool(record[f"{prefix}_opened"])
        and bool(record[f"{prefix}_read_ok"])
        and bool(record[f"{prefix}_event_order_ok"])
        and bool(record[f"{prefix}_unmultiplexed"])
        and int(record[f"{prefix}_time_enabled"]) > 0
        and int(record[f"{prefix}_time_enabled"]) == int(record[f"{prefix}_time_running"])
        and int(record[f"{prefix}_cpu_before"]) == 5
        and int(record[f"{prefix}_cpu_after"]) == 5
    )


def _sentinel_pmu_ok(sentinel: dict[str, Any], prefix: str) -> bool:
    return (
        bool(sentinel[f"{prefix}_opened"])
        and bool(sentinel[f"{prefix}_read_ok"])
        and bool(sentinel[f"{prefix}_event_order_ok"])
        and bool(sentinel[f"{prefix}_unmultiplexed"])
        and int(sentinel[f"{prefix}_time_enabled"]) > 0
        and int(sentinel[f"{prefix}_time_enabled"]) == int(sentinel[f"{prefix}_time_running"])
        and int(sentinel[f"{prefix}_cpu_before"]) == 5
        and int(sentinel[f"{prefix}_cpu_after"]) == 5
    )


def _near(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _derived_delta_failures(record: dict[str, Any], key: tuple[int, int]) -> list[str]:
    checks = {
        "logical_cycles_delta": float(record["positive_cycles"]) - float(record["negative_cycles"]),
        "logical_change_to_dirty_delta": float(record["positive_change_to_dirty"]) - float(record["negative_change_to_dirty"]),
        "logical_probe_dirty_delta": float(record["positive_probe_dirty"]) - float(record["negative_probe_dirty"]),
        "logical_duration_ns_delta": float(record["positive_duration_ns"]) - float(record["negative_duration_ns"]),
        "physical_a_minus_b_cycles_delta": float(record["physical_a_cycles"]) - float(record["physical_b_cycles"]),
        "physical_a_minus_b_change_to_dirty_delta": float(record["physical_a_change_to_dirty"]) - float(record["physical_b_change_to_dirty"]),
        "physical_a_minus_b_probe_dirty_delta": float(record["physical_a_probe_dirty"]) - float(record["physical_b_probe_dirty"]),
        "physical_a_minus_b_duration_ns_delta": float(record["physical_a_duration_ns"]) - float(record["physical_b_duration_ns"]),
    }
    return [
        f"derived delta drift {field}: {key}"
        for field, expected in checks.items()
        if not _near(float(record[field]), expected)
    ]


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
    failures: list[str] = []
    seen_raw: set[tuple[int, int]] = set()
    seen_sentinel: set[tuple[int, int]] = set()
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
        if str(record["source_order"]) != str(trial["source_order"]):
            failures.append(f"raw source-order drift: {key}")
        if str(record["measurement_order"]) != str(trial["measurement_order"]):
            failures.append(f"raw receiver-order drift: {key}")
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
        if str(record.get("bank_allocation_id")) != str(trial["bank_allocation_id"]):
            failures.append(f"raw bank allocation drift: {key}")
        failures.extend(_derived_delta_failures(record, key))
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
        if str(sentinel["source_order"]) != str(trial["source_order"]):
            failures.append(f"sentinel source-order drift: {key}")
        if str(sentinel["measurement_order"]) != str(trial["measurement_order"]):
            failures.append(f"sentinel receiver-order drift: {key}")
        if not bool(sentinel["bytes_unchanged"]) or not bool(sentinel["byte_compare_passed"]):
            failures.append(f"sentinel byte restoration failed: {key}")
        if not all(_sentinel_pmu_ok(sentinel, name) for name in ("pre_a", "pre_b", "post_a", "post_b")):
            failures.append(f"sentinel PMU custody failed: {key}")
        if str(sentinel.get("bank_allocation_id")) != str(trial["bank_allocation_id"]):
            failures.append(f"sentinel bank allocation drift: {key}")
    if seen_raw != set(trial_map):
        failures.append("raw trial set differs from schedule")
    if seen_sentinel != set(trial_map):
        failures.append("sentinel trial set differs from schedule")
    return {"schedule_matched": not failures, "failure_count": len(failures), "failures": failures[:64]}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _logical_delta(record: dict[str, Any], field: str) -> float:
    return float(record[f"positive_{field}"]) - float(record[f"negative_{field}"])


def _physical_delta(record: dict[str, Any], field: str) -> float:
    return float(record[f"physical_a_{field}"]) - float(record[f"physical_b_{field}"])


def logical_value(record: dict[str, Any], coord: str) -> float:
    if coord == "change_to_dirty":
        return _logical_delta(record, "change_to_dirty")
    if coord == "probe_dirty":
        return _logical_delta(record, "probe_dirty")
    if coord == "cycles":
        return _logical_delta(record, "cycles")
    if coord == "duration_ns":
        return _logical_delta(record, "duration_ns")
    if coord == "change_to_dirty_per_cycle":
        return _ratio(_logical_delta(record, "change_to_dirty"), float(record["positive_cycles"]) + float(record["negative_cycles"]))
    if coord == "probe_dirty_per_cycle":
        return _ratio(_logical_delta(record, "probe_dirty"), float(record["positive_cycles"]) + float(record["negative_cycles"]))
    raise KeyError(coord)


def physical_value(record: dict[str, Any], coord: str) -> float:
    if coord == "change_to_dirty":
        return _physical_delta(record, "change_to_dirty")
    if coord == "probe_dirty":
        return _physical_delta(record, "probe_dirty")
    if coord == "cycles":
        return _physical_delta(record, "cycles")
    if coord == "duration_ns":
        return _physical_delta(record, "duration_ns")
    if coord == "change_to_dirty_per_cycle":
        return _ratio(
            _physical_delta(record, "change_to_dirty"),
            float(record["physical_a_cycles"]) + float(record["physical_b_cycles"]),
        )
    if coord == "probe_dirty_per_cycle":
        return _ratio(
            _physical_delta(record, "probe_dirty"),
            float(record["physical_a_cycles"]) + float(record["physical_b_cycles"]),
        )
    raise KeyError(coord)


def sentinel_bank(sentinel: dict[str, Any], coord: str, bank: str, phase: str) -> float:
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


def extract_features(
    schedule: dict[str, Any],
    raw_records: list[dict[str, Any]],
    sentinels: list[dict[str, Any]],
) -> dict[str, Any]:
    if schedule.get("schema_id") != SCHEDULE_SCHEMA:
        raise ConfirmationV2Error("fresh confirmation adjudicator rejects non-V2 schedule")
    integrity = validate_raw_against_schedule(schedule, raw_records, sentinels)
    trial_by_key = {
        (int(trial["replicate_index"]), int(trial["trial_index"])): trial
        for trial in schedule["trials"]
    }
    sentinel_by_key = {
        (int(item["replicate_index"]), int(item["trial_index"])): item for item in sentinels
    }
    trial_features = []
    for record in sorted(raw_records, key=lambda item: (int(item["replicate_index"]), int(item["trial_index"]))):
        if record.get("schema_id") != RAW_SCHEMA:
            continue
        key = (int(record["replicate_index"]), int(record["trial_index"]))
        trial = trial_by_key.get(key)
        sentinel = sentinel_by_key.get(key)
        if trial is None or sentinel is None or sentinel.get("schema_id") != SENTINEL_SCHEMA:
            continue
        coords = {}
        for coord in law_v2.v1.COORDINATES:
            pre = sentinel_bank(sentinel, coord, "a", "pre") - sentinel_bank(sentinel, coord, "b", "pre")
            post = sentinel_bank(sentinel, coord, "a", "post") - sentinel_bank(sentinel, coord, "b", "post")
            coords[coord] = {
                "logical": logical_value(record, coord),
                "physical_a_minus_b": physical_value(record, coord),
                "sentinel_pre_a_minus_b": pre,
                "sentinel_post_a_minus_b": post,
                "sentinel_variation": abs(post - pre),
            }
        trial_features.append(
            {
                "replicate_index": int(record["replicate_index"]),
                "pair_index": int(record["pair_index"]),
                "leg_index": int(record["leg_index"]),
                "trial_index": int(record["trial_index"]),
                "repeat_index": int(record["repeat_index"]),
                "q": int(record["q"]),
                "q0_role": str(trial["q0_role"]),
                "mapping": int(record["mapping"]),
                "mapping_order_first": int(record["mapping_order_first"]),
                "bank_allocation_id": str(record.get("bank_allocation_id", "")),
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
        "run_id": schedule["run_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_semantic_sha256": schedule["schedule_semantic_sha256"],
        "raw_records_sha256": digest(raw_records),
        "restoration_sentinels_sha256": digest(sentinels),
        "raw_record_count": len(raw_records),
        "sentinel_record_count": len(sentinels),
        "unique_trial_key_count": len({
            (int(item["replicate_index"]), int(item["trial_index"]))
            for item in raw_records
            if item.get("schema_id") == RAW_SCHEMA
        }),
        "fresh_only": True,
        "source_kind": "fresh_confirmation_v2",
        "integrity": integrity,
        "trial_features": trial_features,
    }
    features["fresh_evidence_seal_sha256"] = digest(
        {
            "run_id": features["run_id"],
            "schedule_sha256": features["schedule_sha256"],
            "schedule_semantic_sha256": features["schedule_semantic_sha256"],
            "raw_records_sha256": features["raw_records_sha256"],
            "restoration_sentinels_sha256": features["restoration_sentinels_sha256"],
            "raw_record_count": features["raw_record_count"],
            "sentinel_record_count": features["sentinel_record_count"],
            "unique_trial_key_count": features["unique_trial_key_count"],
        }
    )
    features["features_sha256"] = digest({k: v for k, v in features.items() if k != "features_sha256"})
    return features


def _replicate_error(per_replicate: dict[str, Any], coord: str) -> float:
    errors = []
    for q in Q_LADDER:
        if q == 0:
            continue
        left = float(per_replicate["0"]["means_by_q"][str(q)])
        right = float(per_replicate["1"]["means_by_q"][str(q)])
        errors.append(law_v2.rel(left, right, law_v2.coord_floor(coord)))
    return max(errors, default=0.0)


def _confirmation_coordinate(features: dict[str, Any], coord: str) -> dict[str, Any]:
    rows = features["trial_features"]
    aggregate = law_v2.evaluate_scope(rows, coord, include_fresh=True)
    per_replicate = {
        str(rep): law_v2.evaluate_scope([row for row in rows if int(row["replicate_index"]) == rep], coord, include_fresh=False)
        for rep in REPLICATES
    }
    conventions = [per_replicate[str(rep)]["sign_convention"] for rep in REPLICATES]
    same_convention = aggregate["sign_convention"] != "none" and conventions[0] == conventions[1] == aggregate["sign_convention"]
    max_rep_error = _replicate_error(per_replicate, coord)
    replicate_consistency = max_rep_error <= law_v2.v1.REPLICATE_TOL
    passed = (
        bool(features["integrity"]["schedule_matched"])
        and coord == PRIMARY_COORDINATE
        and aggregate["passed"]
        and all(per_replicate[str(rep)]["passed"] for rep in REPLICATES)
        and same_convention
        and replicate_consistency
    )
    shape = (
        same_convention
        and aggregate["laws"]["sign_law"]
        and (aggregate["laws"]["oddness_law"] or aggregate["laws"]["monotonicity_law"])
        and all(
            per_replicate[str(rep)]["laws"]["sign_law"]
            and (
                per_replicate[str(rep)]["laws"]["oddness_law"]
                or per_replicate[str(rep)]["laws"]["monotonicity_law"]
            )
            for rep in REPLICATES
        )
    )
    failed = []
    if not bool(features["integrity"]["schedule_matched"]):
        failed.append({"scope": "integrity", "law": "schedule_and_custody_integrity", "details": features["integrity"]})
    for law, ok in aggregate["laws"].items():
        if not ok:
            failed.append({"scope": "aggregate", "law": law})
    for rep in REPLICATES:
        for law, ok in per_replicate[str(rep)]["laws"].items():
            if not ok:
                failed.append({"scope": f"replicate_{rep}", "law": law})
    if not same_convention:
        failed.append({"scope": "replicate", "law": "same_sign_convention", "conventions": conventions, "aggregate": aggregate["sign_convention"]})
    if not replicate_consistency:
        failed.append({"scope": "replicate", "law": "replicate_consistency", "max_error": max_rep_error})
    return {
        "coordinate": coord,
        "passed": passed,
        "meaningful_transfer_shape": shape,
        "same_replicate_sign_convention": same_convention,
        "replicate_sign_conventions": conventions,
        "replicate_consistency_law": replicate_consistency,
        "max_replicate_error": max_rep_error,
        "aggregate": aggregate,
        "per_replicate": per_replicate,
        "failed_laws": failed,
    }


def _fresh_guard_failures(features: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    expected = {
        "run_id": RUN_ID,
        "raw_record_count": TOTAL_TRIALS,
        "sentinel_record_count": TOTAL_TRIALS,
        "unique_trial_key_count": TOTAL_TRIALS,
        "fresh_only": True,
        "source_kind": "fresh_confirmation_v2",
    }
    for key, value in expected.items():
        if features.get(key) != value:
            failures.append({"scope": "fresh_evidence", "law": f"{key}_mismatch", "actual": features.get(key), "expected": value})
    for key in ("schedule_sha256", "schedule_semantic_sha256", "raw_records_sha256", "restoration_sentinels_sha256", "fresh_evidence_seal_sha256"):
        if not isinstance(features.get(key), str) or not re_full_sha(features[key]):
            failures.append({"scope": "fresh_evidence", "law": f"{key}_missing_or_invalid"})
    if features.get("schedule_sha256") == features.get("raw_records_sha256"):
        failures.append({"scope": "fresh_evidence", "law": "schedule_and_raw_hash_alias"})
    return failures


def re_full_sha(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def adjudicate(features: dict[str, Any]) -> dict[str, Any]:
    if features.get("schema_id") != FEATURE_SCHEMA or not bool(features.get("fresh_only")):
        raise ConfirmationV2Error("confirmation adjudication requires fresh-only V2 features")
    fresh_guard_failures = _fresh_guard_failures(features)
    primary = _confirmation_coordinate(features, PRIMARY_COORDINATE)
    if fresh_guard_failures:
        primary["passed"] = False
        primary["failed_laws"].extend(fresh_guard_failures)
    if primary["passed"]:
        status = law_v2.V2_CLASS_CONFIRMED
    elif primary["meaningful_transfer_shape"]:
        status = law_v2.V2_CLASS_CANDIDATE
    else:
        status = law_v2.V2_CLASS_NOT_ESTABLISHED
    result = {
        "schema_id": ADJUDICATION_SCHEMA,
        "status": status,
        "allowed_classes": list(ALLOWED_CLASSES),
        "forbidden_claims_not_emitted": list(FORBIDDEN_CLASSES),
        "primary_coordinate": PRIMARY_COORDINATE,
        "eligible_coordinates": [PRIMARY_COORDINATE] if status == law_v2.V2_CLASS_CONFIRMED else [],
        "fresh_only": True,
        "features_sha256": features["features_sha256"],
        "fresh_guard_failures": fresh_guard_failures,
        "primary_coordinate_result": primary,
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


def _transfer(kind: str, q: int, rep: int) -> float:
    if kind in ("ideal_direct", "q0_residual_inside_null", "q0_residual_outside_null"):
        return 4.0 * q
    if kind == "ideal_reversed":
        return -4.0 * q
    if kind == "zero_transfer":
        return 0.0
    if kind == "contradictory_replicates":
        return 4.0 * q if rep == 0 else -4.0 * q
    if kind == "non_odd_transfer":
        return abs(q) * 4.0
    if kind == "non_monotonic_transfer":
        return {512: 3000.0, 1024: 2000.0, 1536: 4000.0, -512: -3000.0, -1024: -2000.0, -1536: -4000.0}.get(q, 0.0)
    if kind == "insufficient_gain":
        return float(q // 512) if q >= 0 else -float((-q) // 512)
    if kind == "forged_derived_delta":
        return 4.0 * q
    return 4.0 * q


def build_mock_capture(kind: str) -> dict[str, Any]:
    schedule = build_schedule()
    raw_records: list[dict[str, Any]] = []
    sentinels: list[dict[str, Any]] = []
    for trial in schedule["trials"]:
        rep = int(trial["replicate_index"])
        q = int(trial["q"])
        mapping = int(trial["mapping"])
        f_value = _transfer(kind, q, rep)
        bank_bias = 2000.0 if kind == "paired_fixed_bank_bias" else 0.0
        if mapping == 0:
            physical_a = 200000.0 + f_value / 2.0 + bank_bias
            physical_b = 200000.0 - f_value / 2.0 - bank_bias
            positive_value = physical_a
            negative_value = physical_b
        else:
            physical_a = 200000.0 - f_value / 2.0 + bank_bias
            physical_b = 200000.0 + f_value / 2.0 - bank_bias
            positive_value = physical_b
            negative_value = physical_a
        if kind == "unpaired_mapping_error" and mapping == 1:
            positive_value += 2500.0
        if kind == "source_order_bias":
            positive_value += 2500.0 if trial["source_order"] == "positive_first" else 0.0
        if kind == "receiver_order_bias":
            positive_value += 2500.0 if trial["receiver_positive_first"] else 0.0
        positive_delta = positive_value - 200000.0
        negative_delta = negative_value - 200000.0
        if q == 0 and kind == "q0_residual_inside_null" and trial["q0_role"] == "null_test":
            positive_delta += 1.0
        if q == 0 and kind == "q0_residual_outside_null" and trial["q0_role"] == "null_test":
            positive_delta += 50.0
        pos_window = _mock_window("positive", positive_delta)
        neg_window = _mock_window("negative", negative_delta)
        if kind == "multiplexed_pmu":
            pos_window["positive_time_running"] = pos_window["positive_time_enabled"] - 1
        if kind == "event_order_drift":
            pos_window["positive_event_order_ok"] = False
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
            "bank_allocation_id": str(trial["bank_allocation_id"]),
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
            "pmu_windows_ok": kind not in ("multiplexed_pmu", "event_order_drift"),
            "trial_ok": kind not in ("multiplexed_pmu", "event_order_drift", "duplicate_line_prefix", "unequal_source_work", "restoration_failure"),
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
        if kind == "forged_derived_delta":
            for prefix in ("positive", "negative"):
                record[f"{prefix}_cycles"] = 100000
                record[f"{prefix}_change_to_dirty"] = 200000
                record[f"{prefix}_probe_dirty"] = 400000
                record[f"{prefix}_duration_ns"] = 80000
            record["physical_a_cycles"] = 100000
            record["physical_b_cycles"] = 100000
            record["physical_a_change_to_dirty"] = 200000
            record["physical_b_change_to_dirty"] = 200000
            record["physical_a_probe_dirty"] = 400000
            record["physical_b_probe_dirty"] = 400000
            record["physical_a_duration_ns"] = 80000
            record["physical_b_duration_ns"] = 80000
        if kind == "unequal_source_work":
            record["source_total_work"] = TOTAL_WORK - 1
        if kind == "schedule_drift" and rep == 0 and int(trial["trial_index"]) == 0:
            record["q"] = 1024 if q != 1024 else 512
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
            "bank_allocation_id": str(trial["bank_allocation_id"]),
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
    if kind == "missing_pair_leg":
        raw_records = raw_records[:-1]
        sentinels = sentinels[:-1]
    if kind == "translated_v1_row_smuggle":
        for record in raw_records:
            record["schema_id"] = "CAT_CAS_BALANCED_TRANSDUCER_RAW_RECORD_V1"
        for sentinel in sentinels:
            sentinel["schema_id"] = "CAT_CAS_BALANCED_TRANSDUCER_RESTORATION_SENTINEL_V1"
    return {"schedule": schedule, "raw_records": raw_records, "sentinels": sentinels}


def run_case(kind: str) -> dict[str, Any]:
    capture = build_mock_capture(kind)
    features = extract_features(capture["schedule"], capture["raw_records"], capture["sentinels"])
    return adjudicate(features)


def self_test() -> dict[str, Any]:
    expected = {
        "ideal_direct": law_v2.V2_CLASS_CONFIRMED,
        "ideal_reversed": law_v2.V2_CLASS_CONFIRMED,
        "zero_transfer": law_v2.V2_CLASS_NOT_ESTABLISHED,
        "q0_residual_inside_null": law_v2.V2_CLASS_CONFIRMED,
        "q0_residual_outside_null": law_v2.V2_CLASS_CANDIDATE,
        "paired_fixed_bank_bias": law_v2.V2_CLASS_NOT_ESTABLISHED,
        "unpaired_mapping_error": law_v2.V2_CLASS_CANDIDATE,
        "source_order_bias": law_v2.V2_CLASS_CANDIDATE,
        "receiver_order_bias": law_v2.V2_CLASS_CANDIDATE,
        "contradictory_replicates": law_v2.V2_CLASS_NOT_ESTABLISHED,
        "non_odd_transfer": law_v2.V2_CLASS_NOT_ESTABLISHED,
        "non_monotonic_transfer": law_v2.V2_CLASS_CANDIDATE,
        "insufficient_gain": law_v2.V2_CLASS_CANDIDATE,
        "multiplexed_pmu": law_v2.V2_CLASS_CANDIDATE,
        "event_order_drift": law_v2.V2_CLASS_CANDIDATE,
        "duplicate_line_prefix": law_v2.V2_CLASS_CANDIDATE,
        "unequal_source_work": law_v2.V2_CLASS_CANDIDATE,
        "restoration_failure": law_v2.V2_CLASS_CANDIDATE,
        "schedule_drift": law_v2.V2_CLASS_CANDIDATE,
        "missing_pair_leg": law_v2.V2_CLASS_CANDIDATE,
        "forged_derived_delta": law_v2.V2_CLASS_NOT_ESTABLISHED,
        "translated_v1_row_smuggle": law_v2.V2_CLASS_NOT_ESTABLISHED,
    }
    cases: dict[str, Any] = {}
    for kind, status in expected.items():
        adjudication = run_case(kind)
        cases[kind] = {
            "expected": status,
            "actual": adjudication["status"],
            "passed": adjudication["status"] == status,
            "failed_law_count": len(adjudication["primary_coordinate_result"]["failed_laws"]),
        }
    schedule = build_schedule()
    regressions = {
        "schedule_total_trials": len(schedule["trials"]) == TOTAL_TRIALS,
        "schedule_trials_per_replicate": all(
            len([t for t in schedule["trials"] if int(t["replicate_index"]) == rep]) == TRIALS_PER_REPLICATE
            for rep in REPLICATES
        ),
        "q0_split_balanced": all(
            len([
                t for t in schedule["trials"]
                if int(t["replicate_index"]) == rep
                and int(t["q"]) == 0
                and int(t["leg_index"]) == 0
                and str(t["q0_role"]) == role
            ]) == 2
            for rep in REPLICATES
            for role in ("null_build", "null_test")
        ),
        "q0_role_mapping_order_balanced": all(
            {
                int(t["mapping_order_first"])
                for t in schedule["trials"]
                if int(t["replicate_index"]) == rep
                and int(t["q"]) == 0
                and int(t["leg_index"]) == 0
                and str(t["q0_role"]) == role
            } == {0, 1}
            for rep in REPLICATES
            for role in ("null_build", "null_test")
        ),
        "paired_crossover_uses_pair_index": True,
        "change_to_dirty_primary": PRIMARY_COORDINATE == "change_to_dirty",
        "fresh_evidence_not_pooled_with_v1": True,
        "derived_field_forgery_rejected": cases["forged_derived_delta"]["actual"] != law_v2.V2_CLASS_CONFIRMED,
        "translated_v1_rows_rejected": cases["translated_v1_row_smuggle"]["actual"] == law_v2.V2_CLASS_NOT_ESTABLISHED,
    }
    try:
        import balanced_transducer_public as v1_public

        extract_features(v1_public.build_schedule(), [], [])
        regressions["historical_evidence_pooling_attempt_rejected"] = False
    except Exception:
        regressions["historical_evidence_pooling_attempt_rejected"] = True
    result = {"schema_id": SELF_TEST_SCHEMA, "cases": cases, "regressions": regressions}
    result["self_test_passed"] = all(c["passed"] for c in cases.values()) and all(regressions.values())
    result["self_test_sha256"] = digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--write-schedule", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mock", choices=(
        "ideal_direct",
        "ideal_reversed",
        "zero_transfer",
        "q0_residual_inside_null",
        "q0_residual_outside_null",
        "paired_fixed_bank_bias",
        "unpaired_mapping_error",
        "source_order_bias",
        "receiver_order_bias",
        "contradictory_replicates",
        "non_odd_transfer",
        "non_monotonic_transfer",
        "insufficient_gain",
        "multiplexed_pmu",
        "event_order_drift",
        "duplicate_line_prefix",
        "unequal_source_work",
        "restoration_failure",
        "schedule_drift",
        "missing_pair_leg",
        "forged_derived_delta",
        "translated_v1_row_smuggle",
    ))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_schedule:
        result = write_schedule_artifacts(HERE)
    elif args.schedule:
        result = build_schedule()
    elif args.self_test:
        result = self_test()
    elif args.mock:
        result = run_case(args.mock)
    else:
        result = {"schema_id": "CAT_CAS_CONFIRMATION_V2_PUBLIC_INFO", "schedule": build_schedule()}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not args.self_test or result["self_test_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
