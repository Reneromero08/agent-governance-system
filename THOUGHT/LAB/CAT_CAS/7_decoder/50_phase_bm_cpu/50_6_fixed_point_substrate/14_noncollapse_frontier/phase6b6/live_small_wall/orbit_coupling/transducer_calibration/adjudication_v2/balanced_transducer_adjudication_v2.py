#!/usr/bin/env python3
"""Offline V2 adjudication audit for retained balanced-transducer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CALIBRATION_ROOT = HERE.parent
RUN_ROOT = CALIBRATION_ROOT / "runs" / "balanced_transducer_calibration_1"
sys.path.insert(0, str(CALIBRATION_ROOT))

import balanced_transducer_public as v1  # noqa: E402


SCHEMA_ID = "CAT_CAS_BALANCED_TRANSDUCER_ADJUDICATION_AUDIT_V2"
SELF_TEST_SCHEMA_ID = "CAT_CAS_BALANCED_TRANSDUCER_ADJUDICATION_V2_SELF_TEST"
EXPECTED_HASHES = {
    "PUBLIC_TRIAL_SCHEDULE.json": "3c6d499c6085ad9e9168a238ca30c63d0f642f8d26a3af13d25fd2b8f12adff1",
    "RAW_TRANSDUCER_CAPTURE.jsonl": "942328fde50ed2fca6d0fb620e97e3ebe15c1bd2b7de887e6b04e73d7ba96dab",
    "RESTORATION_SENTINELS.jsonl": "b516983c5440ab58c7ea73e043d91234bff40f4a1c7e2e2eea4f5e42dcb14299",
}
ABS_FLOOR_BY_COORDINATE = {
    "change_to_dirty": 1.0,
    "probe_dirty": 1.0,
    "cycles": 1.0,
    "duration_ns": 1.0,
    "change_to_dirty_per_cycle": 1.0e-6,
    "probe_dirty_per_cycle": 1.0e-6,
}
V2_CLASS_CANDIDATE = "V1_PARTIAL_V2_TRANSFER_CANDIDATE"
V2_CLASS_CONFIRMED = "V1_PARTIAL_CONFIRMED"
V2_CLASS_NOT_ESTABLISHED = "V1_PARTIAL_V2_NOT_ESTABLISHED"
V2_FORBIDDEN_CLASSES = (
    "BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED",
    "ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE",
    "SMALL_WALL_CROSSED",
)


class V2Error(AssertionError):
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def rel(left: float, right: float, floor: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


def oddness(left: float, right: float, floor: float) -> float:
    return abs(left + right) / max(abs(left), abs(right), floor)


def coord_floor(coord: str) -> float:
    return ABS_FLOOR_BY_COORDINATE[coord]


def gain_floor(coord: str) -> float:
    return coord_floor(coord) / min(abs(q) for q in v1.Q_LADDER if q != 0)


def coord_value(row: dict[str, Any], coord: str, *, physical: bool = False) -> float:
    key = "physical_a_minus_b" if physical else "logical"
    return float(row["coordinates"][coord][key])


def select_rows(
    rows: list[dict[str, Any]],
    *,
    q: int | None = None,
    mapping: int | None = None,
    source_order: str | None = None,
    measurement_order: str | None = None,
    mapping_order_first: int | None = None,
    q0_role: str | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if q is not None and int(row["q"]) != q:
            continue
        if mapping is not None and int(row["mapping"]) != mapping:
            continue
        if source_order is not None and str(row["source_order"]) != source_order:
            continue
        if measurement_order is not None and str(row["measurement_order"]) != measurement_order:
            continue
        if mapping_order_first is not None and int(row["mapping_order_first"]) != mapping_order_first:
            continue
        if q0_role is not None and str(row["q0_role"]) != q0_role:
            continue
        selected.append(row)
    return selected


def mean_coord(rows: list[dict[str, Any]], coord: str, *, physical: bool = False, **kwargs: Any) -> float:
    return mean([coord_value(row, coord, physical=physical) for row in select_rows(rows, **kwargs)])


def load_retained_evidence(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    schedule = json.loads((run_root / "PUBLIC_TRIAL_SCHEDULE.json").read_text(encoding="utf-8"))
    raw_records = v1.load_jsonl(run_root / "RAW_TRANSDUCER_CAPTURE.jsonl")
    sentinels = v1.load_jsonl(run_root / "RESTORATION_SENTINELS.jsonl")
    return schedule, raw_records, sentinels


def verify_hashes(run_root: Path) -> dict[str, Any]:
    observed = {
        name: sha256_file(run_root / name)
        for name in EXPECTED_HASHES
    }
    checks = {
        name: {
            "expected": expected,
            "observed": observed[name],
            "passed": observed[name] == expected,
        }
        for name, expected in EXPECTED_HASHES.items()
    }
    return {
        "checks": checks,
        "all_passed": all(item["passed"] for item in checks.values()),
    }


def build_features_and_reproduce_v1(
    schedule: dict[str, Any],
    raw_records: list[dict[str, Any]],
    sentinels: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    features = v1.extract_features(schedule, raw_records, sentinels)
    reproduced = v1.adjudicate(features)
    report = {
        "status": reproduced["status"],
        "expected_status": "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "status_matches_expected": reproduced["status"] == "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "eligible_coordinates": reproduced["eligible_coordinates"],
        "primary_coordinate": reproduced["primary_coordinate"],
        "reproduced_internal_sha256": reproduced["adjudication_sha256"],
        "note": (
            "V1 classification is reproduced from schedule/raw/sentinels only. "
            "The retained V1 adjudication file is not read by the V2 tool."
        ),
    }
    return features, reproduced, report


def null_ceiling(rows: list[dict[str, Any]], coord: str, *, include_fresh: bool) -> dict[str, float]:
    floor = coord_floor(coord)
    q0_values = [abs(coord_value(row, coord)) for row in select_rows(rows, q=0, q0_role="null_build")]
    reps = sorted({int(row["replicate_index"]) for row in rows})
    mapping_residuals = []
    order_residuals = []
    source_residuals = []
    mapping_order_residuals = []
    for rep in reps:
        rep_rows = [row for row in rows if int(row["replicate_index"]) == rep]
        mapping_residuals.append(
            abs(mean_coord(rep_rows, coord, q=0, mapping=0) - mean_coord(rep_rows, coord, q=0, mapping=1))
        )
        order_residuals.append(
            abs(
                mean_coord(rep_rows, coord, q=0, measurement_order="positive_first")
                - mean_coord(rep_rows, coord, q=0, measurement_order="negative_first")
            )
        )
        source_residuals.append(
            abs(
                mean_coord(rep_rows, coord, q=0, source_order="positive_first")
                - mean_coord(rep_rows, coord, q=0, source_order="negative_first")
            )
        )
        mapping_order_residuals.append(
            abs(
                mean([coord_value(row, coord) for row in select_rows(rep_rows, q=0, mapping_order_first=0)])
                - mean([coord_value(row, coord) for row in select_rows(rep_rows, q=0, mapping_order_first=1)])
            )
        )
    sentinel_variations = [abs(float(row["coordinates"][coord]["sentinel_variation"])) for row in rows]
    fresh_variation = 0.0
    if include_fresh and len(reps) >= 2:
        per_rep = [
            mean([coord_value(row, coord) for row in select_rows(rows, q=0, q0_role="null_build") if int(row["replicate_index"]) == rep])
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
        return float(sentinel[f"{prefix}_change_to_dirty"]) / float(sentinel[f"{prefix}_cycles"])
    if coord == "probe_dirty_per_cycle":
        return float(sentinel[f"{prefix}_probe_dirty"]) / float(sentinel[f"{prefix}_cycles"])
    raise KeyError(coord)


def restoration_law(rows: list[dict[str, Any]], coord: str) -> dict[str, Any]:
    floor = v1.RESTORATION_FLOOR_BY_COORDINATE[coord]
    mode_maxima: dict[str, dict[str, Any]] = {}
    all_errors = []
    for row in rows:
        sentinel = row["sentinel"]
        values = {
            "bank_a": (sentinel_bank(sentinel, coord, "a", "pre"), sentinel_bank(sentinel, coord, "a", "post")),
            "bank_b": (sentinel_bank(sentinel, coord, "b", "pre"), sentinel_bank(sentinel, coord, "b", "post")),
        }
        values["differential"] = (values["bank_a"][0] - values["bank_b"][0], values["bank_a"][1] - values["bank_b"][1])
        values["common_mode"] = (values["bank_a"][0] + values["bank_b"][0], values["bank_a"][1] + values["bank_b"][1])
        for mode, (pre, post) in values.items():
            error = abs(post - pre) / max(abs(pre), abs(post), floor)
            all_errors.append(error)
            current = mode_maxima.get(mode)
            if current is None or error > current["relative_error"]:
                mode_maxima[mode] = {
                    "relative_error": error,
                    "trial_index": int(row["trial_index"]),
                    "replicate_index": int(row["replicate_index"]),
                    "pair_index": int(row["pair_index"]),
                    "q": int(row["q"]),
                    "mapping": int(row["mapping"]),
                    "source_order": str(row["source_order"]),
                    "measurement_order": str(row["measurement_order"]),
                    "pre_value": pre,
                    "post_value": post,
                    "absolute_residual": abs(post - pre),
                    "denominator": max(abs(pre), abs(post), floor),
                }
    max_error = max(all_errors, default=0.0)
    return {
        "passed": max_error <= v1.SENTINEL_TOL,
        "max_error": max_error,
        "floor": floor,
        "mode_maxima": mode_maxima,
    }


def q0_null_semantics(rows: list[dict[str, Any]], coord: str, null: dict[str, float]) -> dict[str, Any]:
    ceiling = null["complete_null_ceiling"]
    q0_pairs = pair_rows(rows, include_zero=True)
    max_logical_mapping_abs = 0.0
    max_logical_pair_residual = 0.0
    max_physical_pair_sum = 0.0
    max_null_test_abs = 0.0
    for row in select_rows(rows, q=0, q0_role="null_test"):
        max_null_test_abs = max(max_null_test_abs, abs(coord_value(row, coord)))
    for pair in q0_pairs:
        if int(pair["q"]) != 0:
            continue
        m0 = pair["mapping0"]
        m1 = pair["mapping1"]
        logical0 = coord_value(m0, coord)
        logical1 = coord_value(m1, coord)
        physical_sum = abs(coord_value(m0, coord, physical=True) + coord_value(m1, coord, physical=True))
        max_logical_mapping_abs = max(max_logical_mapping_abs, abs(logical0), abs(logical1))
        max_logical_pair_residual = max(max_logical_pair_residual, abs(logical0 - logical1))
        max_physical_pair_sum = max(max_physical_pair_sum, physical_sum)
    return {
        "ceiling": ceiling,
        "physical_pair_sum_ceiling": 2.0 * ceiling,
        "max_q0_null_test_abs": max_null_test_abs,
        "max_q0_logical_mapping_abs": max_logical_mapping_abs,
        "max_q0_logical_pair_residual": max_logical_pair_residual,
        "max_q0_physical_pair_sum_residual": max_physical_pair_sum,
        "null_test_inside_null": max_null_test_abs <= ceiling,
        "logical_mappings_inside_null": max_logical_mapping_abs <= ceiling,
        "physical_pair_sum_inside_pair_null_bound": max_physical_pair_sum <= 2.0 * ceiling,
        "physical_reversal_not_applicable": True,
        "note": (
            "q=0 logical mappings are tested as null-region membership. "
            "q=0 physical A-minus-B reversal is not required because equal work was applied; "
            "its pair sum is bounded against 2C rather than C."
        ),
    }


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
                "source_order": str(first["source_order"]),
                "measurement_order": str(first["measurement_order"]),
                "mapping_order_first": int(first["mapping_order_first"]),
                "mapping0": by_mapping[0],
                "mapping1": by_mapping[1],
            }
        )
    return pairs


def paired_crossover_law(rows: list[dict[str, Any]], coord: str, null: dict[str, float]) -> dict[str, Any]:
    floor = coord_floor(coord)
    entries = []
    grouped: dict[str, dict[str, Any]] = {}
    max_logical: dict[str, Any] | None = None
    max_physical: dict[str, Any] | None = None
    for pair in pair_rows(rows, include_zero=False):
        q = int(pair["q"])
        m0 = pair["mapping0"]
        m1 = pair["mapping1"]
        logical0 = coord_value(m0, coord)
        logical1 = coord_value(m1, coord)
        physical0 = coord_value(m0, coord, physical=True)
        physical1 = coord_value(m1, coord, physical=True)
        logical_residual = logical0 - logical1
        physical_residual = physical0 + physical1
        logical_den = max(abs(logical0), abs(logical1), floor)
        physical_den = max(abs(physical0), abs(physical1), floor)
        entry = {
            "q": q,
            "replicate_index": pair["replicate_index"],
            "pair_index": pair["pair_index"],
            "source_order": pair["source_order"],
            "receiver_order": pair["measurement_order"],
            "mapping_order_first": pair["mapping_order_first"],
            "logical_mapping0": logical0,
            "logical_mapping1": logical1,
            "logical_pair_residual": logical_residual,
            "logical_denominator": logical_den,
            "logical_relative_error": abs(logical_residual) / logical_den,
            "physical_mapping0": physical0,
            "physical_mapping1": physical1,
            "physical_pair_sum_residual": physical_residual,
            "physical_denominator": physical_den,
            "physical_relative_error": abs(physical_residual) / physical_den,
        }
        entries.append(entry)
        if max_logical is None or entry["logical_relative_error"] > max_logical["logical_relative_error"]:
            max_logical = entry
        if max_physical is None or entry["physical_relative_error"] > max_physical["physical_relative_error"]:
            max_physical = entry
        group_key = json.dumps(
            {
                "q": q,
                "replicate_index": pair["replicate_index"],
                "source_order": pair["source_order"],
                "receiver_order": pair["measurement_order"],
                "mapping_order_first": pair["mapping_order_first"],
            },
            sort_keys=True,
        )
        grouped.setdefault(group_key, {"count": 0, "max_logical_error": 0.0, "max_physical_error": 0.0})
        grouped[group_key]["count"] += 1
        grouped[group_key]["max_logical_error"] = max(grouped[group_key]["max_logical_error"], entry["logical_relative_error"])
        grouped[group_key]["max_physical_error"] = max(grouped[group_key]["max_physical_error"], entry["physical_relative_error"])
    q0 = q0_null_semantics(rows, coord, null)
    logical_max = max([entry["logical_relative_error"] for entry in entries], default=0.0)
    physical_max = max([entry["physical_relative_error"] for entry in entries], default=0.0)
    return {
        "logical_pointer_swap_invariance_law": (
            logical_max <= v1.POINTER_SWAP_TOL
            and q0["null_test_inside_null"]
            and q0["logical_mappings_inside_null"]
        ),
        "physical_pointer_swap_reversal_law": physical_max <= v1.POINTER_SWAP_TOL,
        "max_logical_relative_error": logical_max,
        "max_physical_relative_error": physical_max,
        "max_logical_entry": max_logical,
        "max_physical_entry": max_physical,
        "q0_null_semantics": q0,
        "grouped_distribution": [
            {"group": json.loads(key), **value}
            for key, value in sorted(grouped.items())
        ],
    }


def nonzero_order_law(rows: list[dict[str, Any]], coord: str, factor: str) -> dict[str, Any]:
    floor = coord_floor(coord)
    if factor == "source_order":
        levels = ("positive_first", "negative_first")
        key = "source_order"
    elif factor == "measurement_order":
        levels = ("positive_first", "negative_first")
        key = "measurement_order"
    else:
        raise KeyError(factor)
    errors = []
    for q in v1.Q_LADDER:
        if q == 0:
            continue
        left = mean([coord_value(row, coord) for row in rows if int(row["q"]) == q and str(row[key]) == levels[0]])
        right = mean([coord_value(row, coord) for row in rows if int(row["q"]) == q and str(row[key]) == levels[1]])
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
        "passed": max((item["relative_error"] for item in errors), default=0.0) <= v1.ORDER_TOL,
        "max_relative_error": max((item["relative_error"] for item in errors), default=0.0),
        "max_entry": max_entry,
        "per_q": errors,
    }


def level_transfer(rows: list[dict[str, Any]], coord: str, factor: str, level: Any, null_ceiling: float) -> dict[str, Any]:
    if factor == "source_order":
        subset = [row for row in rows if str(row["source_order"]) == level]
    elif factor == "measurement_order":
        subset = [row for row in rows if str(row["measurement_order"]) == level]
    elif factor == "mapping":
        subset = [row for row in rows if int(row["mapping"]) == int(level)]
    elif factor == "mapping_order_first":
        subset = [row for row in rows if int(row["mapping_order_first"]) == int(level)]
    else:
        raise KeyError(factor)
    means = {q: mean([coord_value(row, coord) for row in subset if int(row["q"]) == q]) for q in v1.Q_LADDER}
    gains = {q: means[q] / q for q in v1.Q_LADDER if q != 0}
    resolved = {q: means[q] for q in v1.Q_LADDER if q != 0 and abs(means[q]) > null_ceiling}
    direct = bool(resolved) and all(sign(value) == sign(q) for q, value in resolved.items())
    reversed_sign = bool(resolved) and all(sign(value) == -sign(q) for q, value in resolved.items())
    convention = "direct" if direct else ("reversed" if reversed_sign else "none")
    odd = all(oddness(means[q], means[-q], coord_floor(coord)) <= v1.ODDNESS_TOL for q in (512, 1024, 1536))
    gain = min(abs(means[q]) for q in (1024, -1024, 1536, -1536)) > v1.GAIN_MULTIPLIER * null_ceiling
    return {
        "level": level,
        "means_by_q": {str(k): v for k, v in means.items()},
        "gain_by_q": {str(k): v for k, v in gains.items()},
        "mean_gain": mean(list(gains.values())),
        "sign_convention": convention,
        "oddness_law": odd,
        "gain_law": gain,
        "intercept_q0": means[0],
        "intercept_inside_null": abs(means[0]) <= null_ceiling,
        "passed": convention != "none" and odd and gain and abs(means[0]) <= null_ceiling,
    }


def gain_normalized_law(rows: list[dict[str, Any]], coord: str, factor: str, null_ceiling: float) -> dict[str, Any]:
    levels: tuple[Any, ...]
    if factor in ("source_order", "measurement_order"):
        levels = ("positive_first", "negative_first")
    elif factor in ("mapping", "mapping_order_first"):
        levels = (0, 1)
    else:
        raise KeyError(factor)
    items = [level_transfer(rows, coord, factor, level, null_ceiling) for level in levels]
    conventions = [item["sign_convention"] for item in items]
    same_convention = len(set(conventions)) == 1 and conventions[0] != "none"
    gain_values = [item["mean_gain"] for item in items]
    floor = gain_floor(coord)
    gain_agreement_error = rel(gain_values[0], gain_values[1], floor) if len(gain_values) == 2 else 0.0
    return {
        "factor": factor,
        "levels": items,
        "gain_space_floor": floor,
        "same_sign_convention": same_convention,
        "gain_agreement_error": gain_agreement_error,
        "gain_agreement_law": gain_agreement_error <= v1.ORDER_TOL,
        "passed": same_convention and all(item["passed"] for item in items) and gain_agreement_error <= v1.ORDER_TOL,
    }


def stratum_law(rows: list[dict[str, Any]], coord: str, convention: str, null_ceiling: float) -> dict[str, Any]:
    results = []
    for mapping in v1.MAPPINGS:
        for source_order in ("positive_first", "negative_first"):
            for measurement_order in ("positive_first", "negative_first"):
                subset = [
                    row for row in rows
                    if int(row["mapping"]) == mapping
                    and str(row["source_order"]) == source_order
                    and str(row["measurement_order"]) == measurement_order
                ]
                means = {q: mean([coord_value(row, coord) for row in subset if int(row["q"]) == q]) for q in v1.Q_LADDER}
                resolved = {q: means[q] for q in v1.Q_LADDER if q != 0 and abs(means[q]) > null_ceiling}
                direct = bool(resolved) and all(sign(value) == sign(q) for q, value in resolved.items())
                reversed_sign = bool(resolved) and all(sign(value) == -sign(q) for q, value in resolved.items())
                stratum_convention = "direct" if direct else ("reversed" if reversed_sign else "none")
                odd = all(oddness(means[q], means[-q], coord_floor(coord)) <= v1.ODDNESS_TOL for q in (512, 1024, 1536))
                gain = min(abs(means[q]) for q in (1024, -1024, 1536, -1536)) > v1.GAIN_MULTIPLIER * null_ceiling
                results.append(
                    {
                        "mapping": mapping,
                        "source_order": source_order,
                        "measurement_order": measurement_order,
                        "sign_convention": stratum_convention,
                        "sign_law": stratum_convention != "none",
                        "oddness_law": odd,
                        "gain_law": gain,
                        "passed": stratum_convention != "none" and odd and gain and stratum_convention == convention,
                    }
                )
    return {"passed": all(item["passed"] for item in results), "strata": results}


def evaluate_scope(rows: list[dict[str, Any]], coord: str, *, include_fresh: bool) -> dict[str, Any]:
    floor = coord_floor(coord)
    null = null_ceiling(rows, coord, include_fresh=include_fresh)
    null_value = null["complete_null_ceiling"]
    means = {q: mean_coord(rows, coord, q=q) for q in v1.Q_LADDER}
    zero_test_mean = mean([coord_value(row, coord) for row in select_rows(rows, q=0, q0_role="null_test")])
    zero_law = abs(zero_test_mean) <= null_value
    odd_errors = {q: oddness(means[q], means[-q], floor) for q in (512, 1024, 1536)}
    odd_law = all(value <= v1.ODDNESS_TOL for value in odd_errors.values())
    resolved = {q: means[q] for q in v1.Q_LADDER if q != 0 and abs(means[q]) > null_value}
    direct = bool(resolved) and all(sign(value) == sign(q) for q, value in resolved.items())
    reversed_sign = bool(resolved) and all(sign(value) == -sign(q) for q, value in resolved.items())
    convention = "direct" if direct else ("reversed" if reversed_sign else "none")
    gain_floor = v1.GAIN_MULTIPLIER * null_value
    gain_law = min(abs(means[q]) for q in (1024, -1024, 1536, -1536)) > gain_floor
    monotonicity = abs(means[512]) < abs(means[1024]) < abs(means[1536]) and abs(means[-512]) < abs(means[-1024]) < abs(means[-1536])
    paired = paired_crossover_law(rows, coord, null)
    source_direct = nonzero_order_law(rows, coord, "source_order")
    receiver_direct = nonzero_order_law(rows, coord, "measurement_order")
    source_gain = gain_normalized_law(rows, coord, "source_order", null_value)
    receiver_gain = gain_normalized_law(rows, coord, "measurement_order", null_value)
    mapping_order_gain = gain_normalized_law(rows, coord, "mapping_order_first", null_value)
    strata = stratum_law(rows, coord, convention, null_value)
    restoration = restoration_law(rows, coord)
    integrity = all(row["trial_ok"] and row["restoration_passed"] and row["byte_compare_passed"] for row in rows)
    restoration_passed = integrity and restoration["passed"]
    laws = {
        "zero_law": zero_law,
        "oddness_law": odd_law,
        "sign_law": convention != "none",
        "gain_law": gain_law,
        "monotonicity_law": monotonicity,
        "paired_logical_pointer_invariance_law": paired["logical_pointer_swap_invariance_law"],
        "paired_physical_pointer_reversal_law": paired["physical_pointer_swap_reversal_law"],
        "nonzero_q_source_order_invariance_law": source_direct["passed"],
        "nonzero_q_receiver_order_invariance_law": receiver_direct["passed"],
        "gain_normalized_source_order_law": source_gain["passed"],
        "gain_normalized_receiver_order_law": receiver_gain["passed"],
        "stratum_transfer_law": strata["passed"],
        "restoration_law": restoration_passed,
    }
    return {
        "passed": all(laws.values()),
        "laws": laws,
        "means_by_q": {str(k): v for k, v in means.items()},
        "zero_test_mean": zero_test_mean,
        "null_components": null,
        "gain_floor": gain_floor,
        "resolved_nonzero_q": sorted(resolved),
        "oddness_errors": {str(k): v for k, v in odd_errors.items()},
        "sign_convention": convention,
        "paired_crossover": paired,
        "source_order_direct_nonzero": source_direct,
        "receiver_order_direct_nonzero": receiver_direct,
        "source_order_gain_normalized": source_gain,
        "receiver_order_gain_normalized": receiver_gain,
        "mapping_order_gain_normalized": mapping_order_gain,
        "stratum_transfer": strata,
        "restoration": restoration,
    }


def v1_failed_maxima_table(rows: list[dict[str, Any]], coord: str) -> list[dict[str, Any]]:
    floor = 1.0
    table = []
    definitions = (
        ("logical_pointer_swap_invariance", "mapping=0 vs mapping=1 logical response"),
        ("physical_pointer_swap_reversal", "mapping=0 plus mapping=1 physical A-minus-B response"),
        ("source_order_equality", "source positive-first vs source negative-first logical response"),
        ("measurement_order_equality", "receiver positive-first vs receiver negative-first logical response"),
    )
    for law, description in definitions:
        entries = []
        for q in v1.Q_LADDER:
            if law == "logical_pointer_swap_invariance":
                left = mean_coord(rows, coord, q=q, mapping=0)
                right = mean_coord(rows, coord, q=q, mapping=1)
                residual = abs(left - right)
                denominator = max(abs(left), abs(right), floor)
                strata = "mapping 0 logical mean vs mapping 1 logical mean"
            elif law == "physical_pointer_swap_reversal":
                left = mean_coord(rows, coord, q=q, mapping=0, physical=True)
                right = mean_coord(rows, coord, q=q, mapping=1, physical=True)
                residual = abs(left + right)
                denominator = max(abs(left), abs(right), floor)
                strata = "mapping 0 physical A-B plus mapping 1 physical A-B"
            elif law == "source_order_equality":
                left = mean_coord(rows, coord, q=q, source_order="positive_first")
                right = mean_coord(rows, coord, q=q, source_order="negative_first")
                residual = abs(left - right)
                denominator = max(abs(left), abs(right), floor)
                strata = "source positive-first logical mean vs source negative-first logical mean"
            else:
                left = mean_coord(rows, coord, q=q, measurement_order="positive_first")
                right = mean_coord(rows, coord, q=q, measurement_order="negative_first")
                residual = abs(left - right)
                denominator = max(abs(left), abs(right), floor)
                strata = "receiver positive-first logical mean vs receiver negative-first logical mean"
            entries.append(
                {
                    "q": q,
                    "left_value": left,
                    "right_value": right,
                    "absolute_residual": residual,
                    "denominator": denominator,
                    "relative_error": residual / denominator,
                    "strata_involved": strata,
                }
            )
        max_entry = max(entries, key=lambda item: item["relative_error"])
        nonzero_max = max([entry for entry in entries if entry["q"] != 0], key=lambda item: item["relative_error"])
        table.append(
            {
                "coordinate": coord,
                "law": law,
                "description": description,
                "maximum_error": max_entry["relative_error"],
                "q_producing_maximum": max_entry["q"],
                "absolute_residual": max_entry["absolute_residual"],
                "denominator": max_entry["denominator"],
                "strata_involved": max_entry["strata_involved"],
                "max_entry": max_entry,
                "max_nonzero_q_entry": nonzero_max,
                "v1_law_failed": max_entry["relative_error"] > v1.POINTER_SWAP_TOL,
                "q0_singularity_dominated": max_entry["q"] == 0 and nonzero_max["relative_error"] <= v1.POINTER_SWAP_TOL,
            }
        )
    return table


def evaluate_v2(features: dict[str, Any], v1_reproduction: dict[str, Any]) -> dict[str, Any]:
    rows = features["trial_features"]
    hard_integrity = bool(features["integrity"]["schedule_matched"])
    coordinate_results = {}
    eligible = []
    partial = []
    failed_maxima = {}
    for coord in v1.COORDINATES:
        aggregate = evaluate_scope(rows, coord, include_fresh=True)
        per_replicate = {
            str(rep): evaluate_scope([row for row in rows if int(row["replicate_index"]) == rep], coord, include_fresh=False)
            for rep in v1.REPLICATES
        }
        replicate_conventions = [per_replicate[str(rep)]["sign_convention"] for rep in v1.REPLICATES]
        same_replicate_convention = (
            aggregate["sign_convention"] != "none"
            and replicate_conventions[0] == replicate_conventions[1] == aggregate["sign_convention"]
        )
        replicate_errors = []
        for q in v1.Q_LADDER:
            if q == 0:
                continue
            left = float(per_replicate["0"]["means_by_q"][str(q)])
            right = float(per_replicate["1"]["means_by_q"][str(q)])
            replicate_errors.append(rel(left, right, coord_floor(coord)))
        replicate_consistency = max(replicate_errors, default=0.0) <= v1.REPLICATE_TOL
        diagnostic_only = coord in v1.DIAGNOSTIC_COORDINATES
        passed = (
            hard_integrity
            and not diagnostic_only
            and aggregate["passed"]
            and all(per_replicate[str(rep)]["passed"] for rep in v1.REPLICATES)
            and same_replicate_convention
            and replicate_consistency
        )
        if passed:
            eligible.append(coord)
        shape_present = (
            same_replicate_convention
            and aggregate["laws"]["sign_law"]
            and (aggregate["laws"]["oddness_law"] or aggregate["laws"]["monotonicity_law"])
            and all(
                per_replicate[str(rep)]["laws"]["sign_law"]
                and (
                    per_replicate[str(rep)]["laws"]["oddness_law"]
                    or per_replicate[str(rep)]["laws"]["monotonicity_law"]
                )
                for rep in v1.REPLICATES
            )
        )
        if shape_present and not passed:
            partial.append(coord)
        coordinate_results[coord] = {
            "passed": passed,
            "diagnostic_only": diagnostic_only,
            "hard_integrity": hard_integrity,
            "same_replicate_sign_convention": same_replicate_convention,
            "replicate_sign_conventions": replicate_conventions,
            "replicate_consistency_law": replicate_consistency,
            "max_replicate_error": max(replicate_errors, default=0.0),
            "aggregate": aggregate,
            "per_replicate": per_replicate,
        }
        failed_maxima[coord] = v1_failed_maxima_table(rows, coord)
    if eligible:
        status = V2_CLASS_CANDIDATE
    elif partial:
        status = V2_CLASS_CONFIRMED
    else:
        status = V2_CLASS_NOT_ESTABLISHED
    result = {
        "schema_id": SCHEMA_ID,
        "status": status,
        "v1_preserved_status": v1_reproduction["status"],
        "forbidden_claims_not_emitted": list(V2_FORBIDDEN_CLASSES),
        "eligible_coordinates": eligible,
        "primary_v2_coordinate": next((coord for coord in v1.COORDINATE_PRIORITY if coord in eligible), None),
        "partial_candidate_coordinates": partial,
        "coordinate_priority": list(v1.COORDINATE_PRIORITY),
        "coordinate_absolute_floors": ABS_FLOOR_BY_COORDINATE,
        "coordinate_results": coordinate_results,
        "v1_failed_maxima_table": failed_maxima,
        "mechanism_diagnosis": diagnose_mechanisms(coordinate_results, failed_maxima),
        "integrity": features["integrity"],
    }
    result["audit_sha256"] = digest({k: val for k, val in result.items() if k != "audit_sha256"})
    return result


def diagnose_mechanisms(coordinate_results: dict[str, Any], failed_maxima: dict[str, Any]) -> dict[str, Any]:
    diagnosis = {}
    for coord, result in coordinate_results.items():
        agg = result["aggregate"]
        q0_singular = [
            item["law"] for item in failed_maxima[coord]
            if item["q0_singularity_dominated"]
        ]
        additive_order_bias = []
        multiplicative_gain_difference = []
        for factor, direct_key, gain_key in (
            ("source_order", "source_order_direct_nonzero", "source_order_gain_normalized"),
            ("receiver_order", "receiver_order_direct_nonzero", "receiver_order_gain_normalized"),
        ):
            direct = agg[direct_key]
            gain = agg[gain_key]
            if not direct["passed"] and gain["passed"]:
                additive_order_bias.append(factor)
            if not gain["passed"]:
                multiplicative_gain_difference.append(factor)
        mapping_order = agg["mapping_order_gain_normalized"]
        genuine_nonzero = [
            law for law in (
                "paired_logical_pointer_invariance_law",
                "paired_physical_pointer_reversal_law",
                "nonzero_q_source_order_invariance_law",
                "nonzero_q_receiver_order_invariance_law",
            )
            if not agg["laws"][law]
        ]
        diagnosis[coord] = {
            "q0_singularity": q0_singular,
            "additive_order_bias": additive_order_bias,
            "multiplicative_gain_difference": multiplicative_gain_difference,
            "mapping_order_carryover": not mapping_order["passed"],
            "genuine_nonzero_q_instability": genuine_nonzero,
            "summary": (
                "candidate" if result["passed"] else (
                    "visible transfer but corrected controls still fail"
                    if result["same_replicate_sign_convention"] and agg["laws"]["sign_law"] and agg["laws"]["oddness_law"]
                    else "not established under V2"
                )
            ),
        }
    return diagnosis


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(CALIBRATION_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def source_hashes() -> dict[str, str]:
    paths = [
        CALIBRATION_ROOT / "BALANCED_TRANSDUCER_CONTRACT.md",
        CALIBRATION_ROOT / "balanced_transducer_public.py",
        CALIBRATION_ROOT / "balanced_transducer_runtime.c",
        HERE / "balanced_transducer_adjudication_v2.py",
        HERE / "CONFIRMATION_CONTRACT_V2.md",
    ]
    return {
        relative_path(path): sha256_file(path)
        for path in paths
        if path.exists()
    }


def threshold_manifest() -> dict[str, Any]:
    return {
        "q_ladder": list(v1.Q_LADDER),
        "replicates": list(v1.REPLICATES),
        "coordinate_priority": list(v1.COORDINATE_PRIORITY),
        "coordinates": list(v1.COORDINATES),
        "diagnostic_coordinates": list(v1.DIAGNOSTIC_COORDINATES),
        "coordinate_absolute_floors": ABS_FLOOR_BY_COORDINATE,
        "gain_space_floors": {coord: gain_floor(coord) for coord in v1.COORDINATES},
        "gain_multiplier": v1.GAIN_MULTIPLIER,
        "oddness_tolerance": v1.ODDNESS_TOL,
        "pointer_swap_tolerance": v1.POINTER_SWAP_TOL,
        "order_tolerance": v1.ORDER_TOL,
        "replicate_tolerance": v1.REPLICATE_TOL,
        "sentinel_tolerance": v1.SENTINEL_TOL,
        "allowed_result_classes": [
            V2_CLASS_CONFIRMED,
            V2_CLASS_CANDIDATE,
            V2_CLASS_NOT_ESTABLISHED,
        ],
        "forbidden_claim_classes": list(V2_FORBIDDEN_CLASSES),
    }


def build_manifest(hash_report: dict[str, Any]) -> dict[str, Any]:
    tests = self_test()
    manifest = {
        "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_ADJUDICATION_V2_MANIFEST",
        "retained_evidence_hashes": hash_report["checks"],
        "source_hashes": source_hashes(),
        "thresholds": threshold_manifest(),
        "self_test": {
            "passed": tests["self_test_passed"],
            "self_test_sha256": tests["self_test_sha256"],
            "case_statuses": {
                name: {
                    "expected": item["expected"],
                    "actual": item["actual"],
                    "passed": item["passed"],
                }
                for name, item in tests["cases"].items()
            },
        },
    }
    manifest["manifest_sha256"] = digest({k: val for k, val in manifest.items() if k != "manifest_sha256"})
    return manifest


def run_audit(run_root: Path) -> dict[str, Any]:
    hash_report = verify_hashes(run_root)
    if not hash_report["all_passed"]:
        raise V2Error("retained evidence hash mismatch")
    schedule, raw_records, sentinels = load_retained_evidence(run_root)
    features, reproduced_v1, reproduction = build_features_and_reproduce_v1(schedule, raw_records, sentinels)
    if reproduction["status"] != "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL":
        raise V2Error("V1 reproduction failed")
    v2_result = evaluate_v2(features, reproduction)
    result = {
        "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_ADJUDICATION_AUDIT_PACKET_V2",
        "run_root": str(run_root),
        "retained_hashes": hash_report,
        "manifest": build_manifest(hash_report),
        "v1_reproduction": reproduction,
        "v2_result": v2_result,
    }
    result["packet_sha256"] = digest({k: val for k, val in result.items() if k != "packet_sha256"})
    return result


def mock_audit(kind: str) -> dict[str, Any]:
    capture = v1.build_mock_capture(kind)
    features = v1.extract_features(capture["schedule"], capture["raw_records"], capture["sentinels"])
    reproduced_v1 = v1.adjudicate(features)
    reproduction = {
        "status": reproduced_v1["status"],
        "expected_status": "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "status_matches_expected": reproduced_v1["status"] == "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
        "eligible_coordinates": reproduced_v1["eligible_coordinates"],
        "primary_coordinate": reproduced_v1["primary_coordinate"],
        "reproduced_internal_sha256": reproduced_v1["adjudication_sha256"],
        "note": "mock V1 classification reproduced from synthetic schedule/raw/sentinels",
    }
    return evaluate_v2(features, reproduction)


def q0_singularity_mock() -> dict[str, Any]:
    capture = v1.build_mock_capture("ideal_odd")
    for record in capture["raw_records"]:
        if int(record["q"]) != 0:
            continue
        offset = 0.5 if int(record["mapping"]) == 0 else 0.0
        record["logical_change_to_dirty_delta"] = offset
        record["logical_probe_dirty_delta"] = 2.0 * offset
        record["logical_cycles_delta"] = offset
        record["logical_duration_ns_delta"] = offset
        if int(record["mapping"]) == 0:
            record["positive_change_to_dirty"] += int(offset)
    features = v1.extract_features(capture["schedule"], capture["raw_records"], capture["sentinels"])
    reproduced_v1 = v1.adjudicate(features)
    reproduction = {
        "status": reproduced_v1["status"],
        "expected_status": "synthetic q=0 singularity regression",
        "status_matches_expected": True,
        "eligible_coordinates": reproduced_v1["eligible_coordinates"],
        "primary_coordinate": reproduced_v1["primary_coordinate"],
        "reproduced_internal_sha256": reproduced_v1["adjudication_sha256"],
        "note": "synthetic q=0 singularity regression",
    }
    return evaluate_v2(features, reproduction)


def self_test() -> dict[str, Any]:
    cases = {}
    expected = {
        "ideal_odd": V2_CLASS_CANDIDATE,
        "zero": V2_CLASS_NOT_ESTABLISHED,
        "fixed_physical_bank_bias": V2_CLASS_NOT_ESTABLISHED,
        "source_order_bias": V2_CLASS_CONFIRMED,
        "measurement_order_bias": V2_CLASS_CONFIRMED,
        "contradictory_fresh_process": V2_CLASS_NOT_ESTABLISHED,
        "unequal_contradictory_fresh_process": V2_CLASS_NOT_ESTABLISHED,
        "restoration_failure": V2_CLASS_CONFIRMED,
        "q0_singularity_regression": V2_CLASS_CANDIDATE,
    }
    for kind, expected_status in expected.items():
        audit = q0_singularity_mock() if kind == "q0_singularity_regression" else mock_audit(kind)
        actual = audit["status"]
        cases[kind] = {
            "expected": expected_status,
            "actual": actual,
            "passed": actual == expected_status,
            "eligible_coordinates": audit["eligible_coordinates"],
            "primary_v2_coordinate": audit["primary_v2_coordinate"],
        }
    result = {"schema_id": SELF_TEST_SCHEMA_ID, "cases": cases}
    result["self_test_passed"] = all(item["passed"] for item in cases.values())
    result["self_test_sha256"] = digest({k: val for k, val in result.items() if k != "self_test_sha256"})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mock", choices=(
        "ideal_odd",
        "zero",
        "fixed_physical_bank_bias",
        "source_order_bias",
        "measurement_order_bias",
        "contradictory_fresh_process",
        "unequal_contradictory_fresh_process",
        "restoration_failure",
        "q0_singularity_regression",
    ))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        result = self_test()
        ok = result["self_test_passed"]
    elif args.mock:
        result = q0_singularity_mock() if args.mock == "q0_singularity_regression" else mock_audit(args.mock)
        ok = True
    else:
        result = run_audit(args.run_root.resolve())
        ok = True
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        write_json(args.output, result)
    else:
        print(payload, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
