#!/usr/bin/env python3
"""Fail-closed relation-only adjudication helpers for synthetic/offline tests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import relation_only_public as pub


SYNTHETIC_DETECTED = "SYNTHETIC_RELATION_MATCH_SIGNAL_DETECTED"
SYNTHETIC_NOT_DETECTED = "SYNTHETIC_RELATION_MATCH_SIGNAL_NOT_DETECTED"
SYNTHETIC_CUSTODY_INVALID = "SYNTHETIC_RELATION_MATCH_CUSTODY_INVALID"

R_MATCH_ABS_THRESHOLD = 50.0
SCALAR_EQUIVALENCE_TOLERANCE = 5.0


def validate_packet(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    rows = packet.get("raw_records")
    if not isinstance(rows, list):
        return {"passed": False, "failures": ["raw_records missing"]}
    schedule_rows = schedule["rows"]
    if len(rows) != len(schedule_rows):
        failures.append("raw record count mismatch")
    tuple_ids = [row.get("tuple_id") for row in rows]
    if len(tuple_ids) != len(set(tuple_ids)):
        failures.append("duplicate tuple_id")
    expected_by_id = {row["tuple_id"]: row for row in schedule_rows}
    for row in rows:
        tuple_id = row.get("tuple_id")
        expected = expected_by_id.get(tuple_id)
        if expected is None:
            failures.append("unexpected tuple_id")
            continue
        if row.get("execution_ordinal") != expected["execution_ordinal"]:
            failures.append("execution ordinal mismatch")
        if row.get("row_role") != expected["row_role"]:
            failures.append("row_role mismatch")
        if row.get("r_prepare") != expected["r_prepare"] or row.get("r_query") != expected["r_query"]:
            failures.append("relation label mismatch")
        value = row.get("dirty_probe_response")
        if type(value) not in {int, float}:
            failures.append("dirty_probe_response not numeric")
        if row.get("positive_physical_claim") is True:
            failures.append("positive physical claim leakage")
        if len(failures) > 20:
            break
    return {"passed": not failures, "failures": failures}


def compute_r_match(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_block: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        if row.get("row_role") != "relation_matrix":
            continue
        by_block[row["block_id"]][(row["r_prepare"], row["r_query"])] = float(row["dirty_probe_response"])
    values = []
    failures = []
    for block_id, cells in by_block.items():
        missing = [cell for cell in pub.RELATION_CELLS if cell not in cells]
        if missing:
            failures.append(f"{block_id} missing relation cells {missing!r}")
            continue
        r_match = 0.5 * (
            (cells[("relation_r0", "relation_r0")] + cells[("relation_r1", "relation_r1")])
            - (cells[("relation_r0", "relation_r1")] + cells[("relation_r1", "relation_r0")])
        )
        values.append(r_match)
    mean = sum(values) / len(values) if values else 0.0
    return {
        "passed": not failures,
        "failures": failures,
        "block_count": len(by_block),
        "R_match_values": values,
        "R_match_mean": mean,
        "R_match_abs_mean": abs(mean),
    }


def scalar_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("row_role") != "scalar_control":
            continue
        by_relation[row["r_prepare"]][row["query"]].append(float(row["dirty_probe_response"]))
    relation_d = {}
    for relation, by_query in by_relation.items():
        a = by_query.get("query_A", [])
        b = by_query.get("query_B", [])
        if not a or not b:
            continue
        relation_d[relation] = (sum(a) / len(a)) - (sum(b) / len(b))
    drift = abs(relation_d.get("relation_r0", 0.0) - relation_d.get("relation_r1", 0.0))
    return {
        "relation_D_single": relation_d,
        "relation_D_single_drift": drift,
        "tolerance": SCALAR_EQUIVALENCE_TOLERANCE,
        "passed": drift <= SCALAR_EQUIVALENCE_TOLERANCE,
    }


def label_scramble_collapse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scrambled = []
    for row in rows:
        mutated = dict(row)
        if mutated.get("row_role") == "relation_matrix" and mutated.get("execution_ordinal", 0) // 12 % 2 == 0:
            mutated["r_query"] = "relation_r1" if mutated["r_query"] == "relation_r0" else "relation_r0"
        scrambled.append(mutated)
    metric = compute_r_match(scrambled)
    return {
        "scrambled_R_match_abs_mean": metric["R_match_abs_mean"],
        "passed": metric["R_match_abs_mean"] < R_MATCH_ABS_THRESHOLD,
    }


def adjudicate_packet(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    validation = validate_packet(packet, schedule)
    if not validation["passed"]:
        return {
            "result_class": SYNTHETIC_CUSTODY_INVALID,
            "positive_physical_claim": None,
            "reproducible_relation_match_coordinate_observed": False,
            "validation": validation,
        }
    rows = packet["raw_records"]
    relation = compute_r_match(rows)
    scalar = scalar_equivalence(rows)
    scramble = label_scramble_collapse(rows)
    detected = (
        relation["passed"]
        and relation["R_match_abs_mean"] >= R_MATCH_ABS_THRESHOLD
        and scalar["passed"]
        and scramble["passed"]
    )
    return {
        "result_class": SYNTHETIC_DETECTED if detected else SYNTHETIC_NOT_DETECTED,
        "positive_physical_claim": None,
        "reproducible_relation_match_coordinate_observed": detected,
        "R_match": relation,
        "scalar_equivalence": scalar,
        "label_scramble": scramble,
        "future_physical_result_classes_predeclared": pub.FUTURE_RESULT_CLASSES,
        "maximum_future_claim": pub.MAXIMUM_FUTURE_CLAIM,
        "small_wall_crossed": False,
    }


def fixture_packet(schedule: dict[str, Any], mode: str) -> dict[str, Any]:
    rows = []
    for expected in schedule["rows"]:
        q = float(expected["q"])
        scalar = 1000.0 + 0.75 * q
        value = scalar
        if expected["row_role"] == "relation_matrix":
            if mode == "positive":
                value += 80.0 if expected["relation_match"] else -80.0
            elif mode == "scalar_replay":
                value += 0.25 * q
            elif mode == "separable_replay":
                value += (15.0 if expected["r_prepare"] == "relation_r0" else -15.0)
                value += (20.0 if expected["r_query"] == "relation_r0" else -20.0)
            elif mode == "route_pressure":
                value += 30.0
            elif mode == "distance_only":
                value += 25.0
        elif expected["row_role"] == "scalar_control":
            if expected["query"] == "query_A":
                value += 40.0
            elif expected["query"] == "query_B":
                value -= 40.0
        rows.append(
            {
                "tuple_id": expected["tuple_id"],
                "execution_ordinal": expected["execution_ordinal"],
                "block_id": expected["block_id"],
                "row_role": expected["row_role"],
                "r_prepare": expected["r_prepare"],
                "r_query": expected["r_query"],
                "relation_match": expected["relation_match"],
                "query": expected["query"],
                "dirty_probe_response": value,
                "change_to_dirty": 1 if value > scalar else 0,
                "cpu_cycles": 1000000,
                "duration_ns": 100000,
                "positive_physical_claim": False,
            }
        )
    return {"schema": "SYNTHETIC_RELATION_ONLY_FIXTURE_PACKET_V1", "mode": mode, "raw_records": rows}


def run_adversary_tests(schedule: dict[str, Any]) -> dict[str, Any]:
    positive = adjudicate_packet(fixture_packet(schedule, "positive"), schedule)
    scalar = adjudicate_packet(fixture_packet(schedule, "scalar_replay"), schedule)
    separable = adjudicate_packet(fixture_packet(schedule, "separable_replay"), schedule)
    route = adjudicate_packet(fixture_packet(schedule, "route_pressure"), schedule)
    distance = adjudicate_packet(fixture_packet(schedule, "distance_only"), schedule)
    invalid_packet = fixture_packet(schedule, "positive")
    invalid_packet["raw_records"] = invalid_packet["raw_records"][:-1]
    invalid = adjudicate_packet(invalid_packet, schedule)
    tests = {
        "synthetic_positive_fixture_detected": positive["result_class"] == SYNTHETIC_DETECTED,
        "scalar_replay_rejected": scalar["result_class"] == SYNTHETIC_NOT_DETECTED,
        "separable_replay_rejected": separable["result_class"] == SYNTHETIC_NOT_DETECTED,
        "route_pressure_replay_rejected": route["result_class"] == SYNTHETIC_NOT_DETECTED,
        "distance_only_replay_rejected": distance["result_class"] == SYNTHETIC_NOT_DETECTED,
        "invalid_packet_no_positive_claim": invalid["result_class"] == SYNTHETIC_CUSTODY_INVALID
        and not invalid["reproducible_relation_match_coordinate_observed"],
    }
    return {
        "schema": "FAMILY10H_RELATION_ONLY_ADVERSARY_TEST_V1",
        "threshold": R_MATCH_ABS_THRESHOLD,
        "positive": positive,
        "scalar_replay": scalar,
        "separable_replay": separable,
        "route_pressure": route,
        "distance_only": distance,
        "invalid_packet": invalid,
        "tests": tests,
        "passed": all(tests.values()),
    }
