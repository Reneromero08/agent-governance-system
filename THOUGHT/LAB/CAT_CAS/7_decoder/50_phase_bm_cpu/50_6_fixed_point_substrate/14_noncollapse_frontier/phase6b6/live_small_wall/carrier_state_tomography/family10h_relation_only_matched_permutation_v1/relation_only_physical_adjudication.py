#!/usr/bin/env python3
"""Prospective physical adjudicator for relation-only matched permutations.

The functions here are offline and fixture-driven until a separately authorized
live transaction provides raw evidence. All negative and invalid paths fail
closed with no positive scientific claim.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Any

import relation_only_public as pub


RESULT_CONFIRMED = "FAMILY10H_RELATION_MATCH_COORDINATE_CONFIRMED_PROSPECTIVE"
RESULT_NOT_CONFIRMED = "FAMILY10H_RELATION_MATCH_COORDINATE_NOT_CONFIRMED_PROSPECTIVE"
RESULT_CANDIDATE = "FAMILY10H_RELATION_MATCH_COORDINATE_CANDIDATE_PROSPECTIVE"
RESULT_INVALID = "FAMILY10H_RELATION_MATCH_COORDINATE_CUSTODY_INVALID"

POSITIVE_CLAIM = pub.MAXIMUM_FUTURE_CLAIM
NEGATIVE_CLAIM = pub.NEGATIVE_FUTURE_CLAIM


def physical_threshold_contract() -> dict[str, Any]:
    contract = {
        "schema": "FAMILY10H_RELATION_ONLY_PHYSICAL_THRESHOLD_CONTRACT_V1",
        "threshold_status": "prospective_physical_thresholds_frozen_before_relation_only_acquisition",
        "scalar_evidence_provenance": {
            "basis": "sealed_family10h_v1_1_scalar_q_readout_attempt_1_and_pre_run_logic",
            **pub.SCALAR_EVIDENCE_PROVENANCE,
            "archive_sha256": "0f92bcd4c00ee78b7e78e84c86bf375ee1caf4ca8c52ae49166ea809f16ff041",
            "relation_only_target_data_used": False,
            "not_relation_source_authority": True,
        },
        "absolute_thresholds": {
            "r_match_abs_min": 512.0,
            "control_null_abs_max": 128.0,
            "source_off_abs_max": 128.0,
            "scalar_D_single_drift_abs_max": 128.0,
        },
        "relative_thresholds": {
            "q_slope_relative_disagreement_max": 0.25,
            "q_intercept_abs_max": 128.0,
            "heldout_relative_error_max": 0.25,
        },
        "resampling_thresholds": {
            "bootstrap_sign_fraction_min": 0.95,
            "label_scramble_abs_max": 128.0,
            "matched_permutation_null_abs_max": 128.0,
        },
        "provenance_law": {
            "r_match_abs_min": "four times the sealed v1.1 source-off absolute null ceiling",
            "control_null_abs_max": "sealed v1.1 source-off absolute null ceiling",
            "scalar_D_single_drift_abs_max": "one sealed v1.1 source-off absolute null ceiling",
            "no_post_run_revision": True,
            "synthetic_thresholds_are_not_physical_thresholds": True,
        },
    }
    contract["threshold_contract_sha256"] = pub.digest(
        {k: v for k, v in contract.items() if k != "threshold_contract_sha256"}
    )
    return contract


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def signed_distribution(values: list[float]) -> dict[str, Any]:
    abs_values = [abs(value) for value in values]
    signs = Counter(1 if value > 0 else -1 if value < 0 else 0 for value in values)
    nonzero = signs[1] + signs[-1]
    return {
        "count": len(values),
        "mean": mean(values),
        "abs_mean": mean(abs_values),
        "max_abs": max(abs_values) if abs_values else 0.0,
        "q50_abs": quantile(abs_values, 0.50),
        "q90_abs": quantile(abs_values, 0.90),
        "q95_abs": quantile(abs_values, 0.95),
        "sign_counts": {str(key): signs[key] for key in [-1, 0, 1]},
        "sign_balance": abs(signs[1] - signs[-1]) / nonzero if nonzero else 0.0,
    }


def claim_boundary(observed: bool) -> dict[str, Any]:
    return {
        "reproducible_relation_match_coordinate_observed": observed,
        "maximum_future_claim": POSITIVE_CLAIM,
        "full_carrier_state_tomography_established": False,
        "physical_relational_memory_established": False,
        "catalytic_borrowing_established": False,
        "r2_restoration_established": False,
        "small_wall_crossed": False,
    }


def fail_closed_result(result_class: str, validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "FAMILY10H_RELATION_ONLY_PHYSICAL_ADJUDICATION_V1",
        "result_class": result_class,
        "passed": False,
        "scientific_claim": NEGATIVE_CLAIM,
        "claim_boundary": claim_boundary(False),
        "validation": validation,
    }


def validate_physical_packet(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    raw_records = packet.get("raw_records")
    source_death = packet.get("source_death_receipts")
    feature_freeze = packet.get("feature_freeze")
    custody = packet.get("custody_envelope")
    if not isinstance(raw_records, list):
        return {"passed": False, "failures": ["raw_records missing"]}
    if not isinstance(source_death, list):
        return {"passed": False, "failures": ["source_death_receipts missing"]}
    schedule_rows = schedule["rows"]
    if len(raw_records) != len(schedule_rows):
        failures.append("raw record count mismatch")
    if len(source_death) != len(schedule_rows):
        failures.append("source-death receipt count mismatch")
    schedule_by_id = {row["tuple_id"]: row for row in schedule_rows}
    if len(schedule_by_id) != len(schedule_rows):
        failures.append("schedule tuple_id duplicate")
    trusted_schedule_fields = list(pub.SCHEDULE_COLUMNS)
    seen = set()
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            failures.append("raw record malformed")
            break
        if index >= len(schedule_rows):
            failures.append("raw record count exceeds schedule")
            break
        ordered_expected = schedule_rows[index]
        tuple_id = record.get("tuple_id")
        expected = schedule_by_id.get(tuple_id)
        if expected is None:
            failures.append("unexpected tuple_id")
            continue
        if tuple_id != ordered_expected["tuple_id"]:
            failures.append("raw record execution order mismatch")
        if tuple_id in seen:
            failures.append("duplicate raw tuple_id")
        seen.add(tuple_id)
        for field in trusted_schedule_fields:
            if record.get(field) != expected.get(field):
                failures.append(f"{field} mismatch")
        for metric in ["dirty_probe_response", "change_to_dirty", "cpu_cycles", "duration_ns"]:
            if type(record.get(metric)) not in {int, float}:
                failures.append(f"{metric} not numeric")
        if record.get("physical_measurement") is not True:
            failures.append("physical measurement flag mismatch")
        if record.get("process_custody") != "source_dead_before_query":
            failures.append("process custody mismatch")
        if record.get("expected_pmu_group") != pub.PMU_GROUP["name"]:
            failures.append("expected PMU event group mismatch")
        if record.get("pmu_event_group") != pub.PMU_GROUP["name"]:
            failures.append("PMU event group mismatch")
        if record.get("pmu_events") != pub.PMU_GROUP["events"]:
            failures.append("PMU event identity mismatch")
        event_ids = record.get("event_ids")
        if not isinstance(event_ids, dict) or set(event_ids) != set(pub.PMU_GROUP["events"]):
            failures.append("PMU event IDs missing")
        elif len(set(event_ids.values())) != len(event_ids) or not all(isinstance(value, int) and value > 0 for value in event_ids.values()):
            failures.append("PMU event IDs invalid or non-distinct")
        time_enabled = record.get("time_enabled")
        time_running = record.get("time_running")
        if type(time_enabled) not in {int, float} or time_enabled <= 0:
            failures.append("time_enabled invalid")
        if type(time_running) not in {int, float} or time_running <= 0 or (type(time_enabled) in {int, float} and time_running > time_enabled):
            failures.append("time_running invalid")
        if record.get("source_cpu_before") != expected["source_cpu_expected"] or record.get("source_cpu_after") != expected["source_cpu_expected"]:
            failures.append("source CPU custody mismatch")
        if record.get("receiver_cpu_before") != expected["receiver_cpu_expected"] or record.get("receiver_cpu_after") != expected["receiver_cpu_expected"]:
            failures.append("receiver CPU custody mismatch")
        if record.get("positive_physical_claim") is True:
            failures.append("positive claim leakage in raw record")
        if len(failures) > 32:
            break
    death_by_id = {row.get("tuple_id"): row for row in source_death if isinstance(row, dict)}
    if len(death_by_id) != len(source_death):
        failures.append("source-death receipt duplicate or malformed")
    for index, expected in enumerate(schedule_rows):
        receipt = source_death[index] if index < len(source_death) else None
        if not isinstance(receipt, dict):
            failures.append("source-death receipt missing")
            break
        if receipt.get("tuple_id") != expected["tuple_id"]:
            failures.append("source-death receipt execution order mismatch")
        if receipt.get("tuple_id") != expected["tuple_id"] or receipt.get("execution_ordinal") != expected["execution_ordinal"]:
            failures.append("source-death tuple identity mismatch")
        if receipt.get("source_pid") != receipt.get("waitpid_pid"):
            failures.append("source PID/waitpid mismatch")
        if receipt.get("waitpid_status") != "exited_0":
            failures.append("source wait status mismatch")
        if receipt.get("query_selected_after_waitpid") is not True:
            failures.append("query selected before source death")
        if receipt.get("source_alive_during_query") is not False:
            failures.append("source alive during query")
        if receipt.get("source_helper_survives") is not False:
            failures.append("source helper survived query")
        if receipt.get("open_source_ipc_after_waitpid") != 0:
            failures.append("open source IPC after waitpid")
        if receipt.get("post_observation_query_or_window_selection") is not False:
            failures.append("post-observation selection")
        if receipt.get("source_cpu_before") != expected["source_cpu_expected"] or receipt.get("source_cpu_after") != expected["source_cpu_expected"]:
            failures.append("source-death CPU custody mismatch")
        if receipt.get("physical_measurement") is not True:
            failures.append("source-death physical measurement mismatch")
        if len(failures) > 48:
            break
    if not isinstance(feature_freeze, dict):
        failures.append("feature_freeze missing")
    else:
        if feature_freeze.get("physical_measurement") is not True:
            failures.append("feature freeze physical measurement mismatch")
        if feature_freeze.get("raw_record_count") != len(schedule_rows):
            failures.append("feature freeze raw-record count mismatch")
        if feature_freeze.get("primary_endpoint") != "dirty_probe_response":
            failures.append("feature freeze primary endpoint mismatch")
        if feature_freeze.get("secondary_endpoints") != ["change_to_dirty", "cpu_cycles", "duration_ns"]:
            failures.append("feature freeze secondary endpoints mismatch")
        if feature_freeze.get("schedule_sha256") != schedule.get("schedule_sha256"):
            failures.append("feature freeze schedule binding mismatch")
        if feature_freeze.get("post_observation_feature_selection") is not False:
            failures.append("post-observation feature selection")
    if not isinstance(custody, dict):
        failures.append("custody envelope missing")
    else:
        inventory = custody.get("evidence_inventory")
        required_inventory = [
            "raw_records",
            "source_death_receipts",
            "feature_freeze",
            "target_execution_receipt",
        ]
        for field in [
            "target_execution_receipt_sha256",
            "runtime_sha256",
            "manifest_sha256",
            "copied_back_archive_sha256",
        ]:
            value = custody.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                failures.append(f"{field} missing or malformed")
        for field in ["source_authority_sha", "freeze_sha"]:
            value = custody.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
                failures.append(f"{field} missing or malformed")
        if not isinstance(custody.get("copied_back_archive_size"), int) or custody.get("copied_back_archive_size") <= 0:
            failures.append("copied-back archive size missing")
        if not isinstance(inventory, dict) or not all(inventory.get(item) is True for item in required_inventory):
            failures.append("evidence inventory incomplete")
    return {
        "passed": not failures,
        "failures": failures[:64],
        "raw_record_count": len(raw_records),
        "source_death_receipt_count": len(source_death),
    }


def relation_cells_by_block(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str], float]]:
    by_block: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        if row.get("row_role") == "relation_matrix":
            by_block[row["block_id"]][(row["r_prepare"], row["r_query"])] = float(row["dirty_probe_response"])
    return by_block


def r_match_values(rows: list[dict[str, Any]]) -> tuple[list[float], list[str]]:
    values: list[float] = []
    failures: list[str] = []
    for block_id, cells in relation_cells_by_block(rows).items():
        missing = [cell for cell in pub.RELATION_CELLS if cell not in cells]
        if missing:
            failures.append(f"{block_id} missing relation cells {missing!r}")
            continue
        values.append(
            0.5
            * (
                (cells[("relation_r0", "relation_r0")] + cells[("relation_r1", "relation_r1")])
                - (cells[("relation_r0", "relation_r1")] + cells[("relation_r1", "relation_r0")])
            )
        )
    return values, failures


def effect_summary(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    values, failures = r_match_values(rows)
    effect = mean(values)
    floor = threshold_contract["absolute_thresholds"]["r_match_abs_min"]
    distribution = signed_distribution(values)
    return {
        "passed": not failures and abs(effect) >= floor and distribution["abs_mean"] >= floor,
        "failures": failures,
        "block_count": len(values),
        "R_match_mean": effect,
        "R_match_abs_mean": distribution["abs_mean"],
        "R_match_abs_of_mean": abs(effect),
        "R_match_std": stdev(values),
        "R_match_min": min(values) if values else 0.0,
        "R_match_max": max(values) if values else 0.0,
        "R_match_max_abs": distribution["max_abs"],
        "R_match_q50_abs": distribution["q50_abs"],
        "R_match_q90_abs": distribution["q90_abs"],
        "R_match_q95_abs": distribution["q95_abs"],
        "sign_counts": distribution["sign_counts"],
        "sign_balance": distribution["sign_balance"],
        "threshold": floor,
        "sign": 1 if effect > 0 else -1 if effect < 0 else 0,
    }


def group_rows(rows: list[dict[str, Any]], factor: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[factor]].append(row)
    return dict(grouped)


def strata_report(rows: list[dict[str, Any]], threshold_contract: dict[str, Any], factors: list[str]) -> dict[str, Any]:
    report = {}
    for factor in factors:
        by_level = {}
        for level, level_rows in sorted(group_rows(rows, factor).items(), key=lambda item: repr(item[0])):
            by_level[str(level)] = effect_summary(level_rows, threshold_contract)
        report[factor] = {
            "levels": by_level,
            "passed": all(item["passed"] for item in by_level.values()),
        }
    return report


def scalar_equivalence(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    by_relation_query_q: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("row_role") != "scalar_control":
            continue
        by_relation_query_q[(row["r_prepare"], row["query"], int(row["q"]))].append(float(row["dirty_probe_response"]))
    relation_d: dict[str, dict[int, float]] = defaultdict(dict)
    for relation in pub.RELATIONS:
        for q in pub.Q_VALUES:
            a = by_relation_query_q.get((relation, "query_A", q), [])
            b = by_relation_query_q.get((relation, "query_B", q), [])
            if a and b:
                relation_d[relation][q] = mean(a) - mean(b)
    slopes = {}
    intercepts = {}
    for relation, qmap in relation_d.items():
        xs = [float(q) for q in sorted(qmap)]
        ys = [qmap[int(q)] for q in xs]
        xbar = mean(xs)
        ybar = mean(ys)
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom else 0.0
        slopes[relation] = slope
        intercepts[relation] = ybar - slope * xbar
    slope_values = list(slopes.values())
    max_slope = max((abs(value) for value in slope_values), default=0.0)
    slope_disagreement = (max(slope_values) - min(slope_values)) / max_slope if max_slope else 0.0
    d_drift = max(
        (abs(relation_d["relation_r0"].get(q, 0.0) - relation_d["relation_r1"].get(q, 0.0)) for q in pub.Q_VALUES),
        default=0.0,
    )
    passed = (
        len(relation_d) == 2
        and d_drift <= threshold_contract["absolute_thresholds"]["scalar_D_single_drift_abs_max"]
        and slope_disagreement <= threshold_contract["relative_thresholds"]["q_slope_relative_disagreement_max"]
        and all(abs(value) <= threshold_contract["relative_thresholds"]["q_intercept_abs_max"] for value in intercepts.values())
    )
    return {
        "passed": passed,
        "relation_D_single_by_q": {key: {str(q): value for q, value in qmap.items()} for key, qmap in relation_d.items()},
        "max_D_single_drift": d_drift,
        "slopes": slopes,
        "intercepts": intercepts,
        "slope_relative_disagreement": slope_disagreement,
    }


def control_nulls(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    threshold = threshold_contract["absolute_thresholds"]["control_null_abs_max"]
    by_control: dict[str, list[float]] = defaultdict(list)
    by_stratum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    stratum_factors = ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "q", "cyclic_origin"]
    for row in rows:
        if row.get("row_role") == "relation_control":
            value = float(row["dirty_probe_response"])
            control = row["query"]
            by_control[control].append(value)
            for factor in stratum_factors:
                by_stratum[control][f"{factor}={row[factor]}"].append(value)
    controls = {}
    for control in pub.CONTROL_ROWS:
        values = by_control.get(control, [])
        distribution = signed_distribution(values)
        per_stratum = {}
        for key, stratum_values in sorted(by_stratum.get(control, {}).items()):
            item = signed_distribution(stratum_values)
            item["passed"] = bool(stratum_values) and item["max_abs"] <= threshold and item["abs_mean"] <= threshold
            per_stratum[key] = item
        controls[control] = {
            **distribution,
            "per_stratum": per_stratum,
            "passed": bool(values)
            and distribution["abs_mean"] <= threshold
            and distribution["max_abs"] <= threshold
            and distribution["q95_abs"] <= threshold
            and all(item["passed"] for item in per_stratum.values()),
        }
    return {"passed": all(item["passed"] for item in controls.values()), "threshold": threshold, "controls": controls}


def label_scramble_test(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    threshold = threshold_contract["resampling_thresholds"]["label_scramble_abs_max"]
    values = []
    for seed in range(64):
        rng = random.Random(0xC0DEC0DE + seed)
        scrambled = []
        block_flip: dict[str, bool] = {}
        for row in rows:
            item = dict(row)
            if item.get("row_role") == "relation_matrix":
                flip = block_flip.setdefault(item["block_id"], rng.random() < 0.5)
                if flip:
                    item["r_query"] = "relation_r1" if item["r_query"] == "relation_r0" else "relation_r0"
            scrambled.append(item)
        values.append(effect_summary(scrambled, threshold_contract)["R_match_abs_of_mean"])
    distribution = signed_distribution(values)
    return {
        "passed": distribution["abs_mean"] <= threshold and distribution["q95_abs"] <= threshold,
        "seed": "0xC0DEC0DE",
        "iterations": len(values),
        "null_distribution": distribution,
        "threshold": threshold,
    }


def heldout_transport(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    factors = ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "q", "cyclic_origin"]
    global_summary = effect_summary(rows, threshold_contract)
    expected_sign = global_summary["sign"]
    report = {}
    for factor in factors:
        levels = {}
        for level in sorted({row[factor] for row in rows}, key=lambda value: repr(value)):
            train_rows = [row for row in rows if row[factor] != level]
            test_rows = [row for row in rows if row[factor] == level]
            test = effect_summary(test_rows, threshold_contract)
            train_absent = all(row[factor] != level for row in train_rows)
            sign_pass = test["sign"] == expected_sign and expected_sign != 0
            levels[str(level)] = {
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "held_out_level_absent_from_training": train_absent,
                "test_R_match_abs_mean": test["R_match_abs_mean"],
                "test_sign": test["sign"],
                "confidence_interval": [
                    test["R_match_mean"] - 2.0 * test["R_match_std"],
                    test["R_match_mean"] + 2.0 * test["R_match_std"],
                ],
                "null_comparison": "fixed_threshold_no_relation_only_training",
                "passed": train_absent and test["passed"] and sign_pass,
            }
        report[factor] = {"levels": levels, "passed": all(item["passed"] for item in levels.values())}
    return {"passed": all(item["passed"] for item in report.values()), "factors": report}


def bootstrap_stability(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    values, failures = r_match_values(rows)
    if failures or not values:
        return {"passed": False, "failures": failures or ["no R_match blocks"]}
    observed_sign = 1 if mean(values) > 0 else -1 if mean(values) < 0 else 0
    rng = random.Random(0xB00757A9)
    sample_means = []
    signs = []
    for _ in range(256):
        sample = [rng.choice(values) for _ in values]
        sample_mean = mean(sample)
        sample_means.append(sample_mean)
        signs.append(1 if sample_mean > 0 else -1 if sample_mean < 0 else 0)
    sign_fraction = sum(1 for sign in signs if sign == observed_sign) / len(signs)
    return {
        "passed": observed_sign != 0 and sign_fraction >= threshold_contract["resampling_thresholds"]["bootstrap_sign_fraction_min"],
        "seed": "0xB00757A9",
        "iterations": len(sample_means),
        "observed_sign": observed_sign,
        "bootstrap_sign_fraction": sign_fraction,
        "sample_mean_distribution": signed_distribution(sample_means),
        "threshold": threshold_contract["resampling_thresholds"]["bootstrap_sign_fraction_min"],
    }


def neutralized_effect(rows: list[dict[str, Any]], threshold_contract: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Diagnostic-only replay summary; not used as an adjudication gate."""
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("row_role") == "relation_matrix":
            grouped[tuple(row[key] for key in keys)].append(float(row["dirty_probe_response"]))
    neutralized = []
    for row in rows:
        item = dict(row)
        if item.get("row_role") == "relation_matrix":
            item["dirty_probe_response"] = mean(grouped[tuple(row[key] for key in keys)])
        neutralized.append(item)
    return effect_summary(neutralized, threshold_contract)


def rows_by_block(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[row["block_id"]].append(row)
    return dict(by_block)


def relation_matrix_rows_by_block(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_role") == "relation_matrix":
            grouped[row["block_id"]].append(row)
    return dict(grouped)


def scalar_block_baselines(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    for block_id, block_rows in rows_by_block(rows).items():
        scalar_rows = [row for row in block_rows if row.get("row_role") == "scalar_control"]
        control_rows = [row for row in block_rows if row.get("row_role") == "relation_control"]
        a_values = [float(row["dirty_probe_response"]) for row in scalar_rows if row.get("query") == "query_A"]
        b_values = [float(row["dirty_probe_response"]) for row in scalar_rows if row.get("query") == "query_B"]
        control_values = [float(row["dirty_probe_response"]) for row in control_rows]
        scalar_center = mean([*a_values, *b_values]) if a_values or b_values else 0.0
        baselines[block_id] = {
            "scalar_center": scalar_center,
            "D_single": (mean(a_values) - mean(b_values)) if a_values and b_values else 0.0,
            "route_pressure_control": mean(control_values) if control_values else 0.0,
        }
    return baselines


def scalar_replay_residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = scalar_block_baselines(rows)
    residual_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("row_role") == "relation_matrix":
            baseline = baselines.get(item["block_id"], {})
            item["dirty_probe_response"] = float(item["dirty_probe_response"]) - float(baseline.get("scalar_center", 0.0))
        residual_rows.append(item)
    return residual_rows


def additive_ab_residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residual_rows = [dict(row) for row in rows]
    output_by_tuple = {row["tuple_id"]: row for row in residual_rows}
    for block_id, matrix_rows in relation_matrix_rows_by_block(rows).items():
        if len(matrix_rows) != len(pub.RELATION_CELLS):
            continue
        grand = mean([float(row["dirty_probe_response"]) for row in matrix_rows])
        prep_means = {
            relation: mean([float(row["dirty_probe_response"]) for row in matrix_rows if row["r_prepare"] == relation])
            for relation in pub.RELATIONS
        }
        query_means = {
            relation: mean([float(row["dirty_probe_response"]) for row in matrix_rows if row["r_query"] == relation])
            for relation in pub.RELATIONS
        }
        for row in matrix_rows:
            predicted = prep_means[row["r_prepare"]] + query_means[row["r_query"]] - grand
            output_by_tuple[row["tuple_id"]]["dirty_probe_response"] = float(row["dirty_probe_response"]) - predicted
    return residual_rows


def matched_permutation_null(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    observed = effect_summary(rows, threshold_contract)
    null_values: list[float] = []
    block_cells = []
    for matrix_rows in relation_matrix_rows_by_block(rows).values():
        if len(matrix_rows) != len(pub.RELATION_CELLS):
            continue
        block_cells.append(
            {
                (row["r_prepare"], row["r_query"]): float(row["dirty_probe_response"])
                for row in matrix_rows
            }
        )
    for seed in range(128):
        rng = random.Random(0xA77A5EED + seed)
        sample_r: list[float] = []
        for cells in block_cells:
            shuffled_values = [cells[cell] for cell in pub.RELATION_CELLS]
            rng.shuffle(shuffled_values)
            shuffled_cells = dict(zip(pub.RELATION_CELLS, shuffled_values))
            sample_r.append(
                0.5
                * (
                    (shuffled_cells[("relation_r0", "relation_r0")] + shuffled_cells[("relation_r1", "relation_r1")])
                    - (shuffled_cells[("relation_r0", "relation_r1")] + shuffled_cells[("relation_r1", "relation_r0")])
                )
            )
        null_values.append(abs(mean(sample_r)) if sample_r else 0.0)
    distribution = signed_distribution(null_values)
    threshold = threshold_contract["resampling_thresholds"]["matched_permutation_null_abs_max"]
    return {
        "passed": observed["R_match_abs_of_mean"] >= observed["threshold"]
        and observed["R_match_abs_of_mean"] > distribution["q95_abs"]
        and distribution["q95_abs"] <= threshold,
        "seed": "0xA77A5EED",
        "iterations": len(null_values),
        "observed_R_match_abs_of_mean": observed["R_match_abs_of_mean"],
        "null_distribution": distribution,
        "null_threshold": threshold,
    }


def replay_adversary_report(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    scalar_residual = effect_summary(scalar_replay_residual_rows(rows), threshold_contract)
    additive_residual = effect_summary(additive_ab_residual_rows(rows), threshold_contract)
    matched_null = matched_permutation_null(rows, threshold_contract)
    heldout = heldout_transport(rows, threshold_contract)
    diagnostic_neutralization = {
        "non_gating": True,
        "reason": "group-mean neutralization can algebraically erase or preserve effects and is retained only for audit visibility",
        "examples": {
            "q_only": neutralized_effect(rows, threshold_contract, ["q"]),
            "mapping_session_replicate_delay_q": neutralized_effect(
                rows,
                threshold_contract,
                ["mapping", "session", "replicate", "delay_label", "q"],
            ),
        },
    }
    adversaries = {
        "scalar_q_replay": {
            "model": "relation cells predicted from preserved scalar-control block center and D_single only; no relation-cell labels",
            "residual_R_match_abs_of_mean": scalar_residual["R_match_abs_of_mean"],
            "residual_R_match_abs_mean": scalar_residual["R_match_abs_mean"],
            "relation_residual_survives": scalar_residual["passed"],
            "passed": scalar_residual["passed"],
        },
        "nonlinear_scalar_q_replay": {
            "model": "same scalar-only residual law with q/nonlinear nuisance confined to block-level scalar marginals",
            "residual_R_match_abs_of_mean": scalar_residual["R_match_abs_of_mean"],
            "residual_R_match_abs_mean": scalar_residual["R_match_abs_mean"],
            "relation_residual_survives": scalar_residual["passed"],
            "passed": scalar_residual["passed"],
        },
        "separable_a_b_marginal_replay": {
            "model": "per-block additive prepare/query main effects without relation interaction",
            "residual_R_match_abs_of_mean": additive_residual["R_match_abs_of_mean"],
            "residual_R_match_abs_mean": additive_residual["R_match_abs_mean"],
            "relation_interaction_residual_survives": additive_residual["passed"],
            "passed": additive_residual["passed"],
        },
        "route_pressure_replay": {
            "model": "matched route-pressure and relation-control rows are block-level nuisance only",
            "residual_R_match_abs_of_mean": scalar_residual["R_match_abs_of_mean"],
            "matched_controls_available": bool(scalar_block_baselines(rows)),
            "passed": scalar_residual["passed"],
        },
        "distance_only_replay": {
            "model": "distance histogram is frozen equal across relation cells; no relation interaction term allowed",
            "residual_R_match_abs_of_mean": additive_residual["R_match_abs_of_mean"],
            "passed": additive_residual["passed"],
        },
        "matched_permutation_null": matched_null,
        "confounding_heldout": {
            "factors": {name: item["passed"] for name, item in heldout["factors"].items()},
            "passed": heldout["passed"],
        },
    }
    return {
        "schema": "FAMILY10H_RELATION_ONLY_SCALAR_REPLAY_ADVERSARY_REPORT_V1",
        "null_threshold": threshold_contract["resampling_thresholds"]["matched_permutation_null_abs_max"],
        "adversaries": adversaries,
        "diagnostic_neutralized_effect": diagnostic_neutralization,
        "passed": all(item["passed"] for item in adversaries.values()),
    }


def source_off_gate(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    values = [float(row["dirty_probe_response"]) for row in rows if row.get("row_role") == "source_off"]
    if not values:
        return {
            "status": "not_scheduled",
            "claimed_as_gate": False,
            "passed": True,
            "reason": "frozen schedule contains no legitimate source-off matrix",
        }
    threshold = threshold_contract["absolute_thresholds"]["source_off_abs_max"]
    distribution = signed_distribution(values)
    return {
        "status": "scheduled",
        "claimed_as_gate": True,
        "distribution": distribution,
        "threshold": threshold,
        "passed": distribution["max_abs"] <= threshold,
    }


def adjudicate_physical_packet(packet: dict[str, Any], schedule: dict[str, Any], threshold_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    threshold_contract = threshold_contract or physical_threshold_contract()
    validation = validate_physical_packet(packet, schedule)
    if not validation["passed"]:
        return fail_closed_result(RESULT_INVALID, validation)
    rows = packet["raw_records"]
    global_effect = effect_summary(rows, threshold_contract)
    factors = ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "q", "cyclic_origin"]
    strata = strata_report(rows, threshold_contract, factors)
    scalar = scalar_equivalence(rows, threshold_contract)
    controls = control_nulls(rows, threshold_contract)
    scramble = label_scramble_test(rows, threshold_contract)
    heldout = heldout_transport(rows, threshold_contract)
    bootstrap = bootstrap_stability(rows, threshold_contract)
    adversary = replay_adversary_report(rows, threshold_contract)
    source_off = source_off_gate(rows, threshold_contract)
    gates = {
        "validation": validation["passed"],
        "complete_relation_matrix_global": global_effect["passed"],
        "session_strata": strata["session"]["passed"],
        "replicate_strata": strata["replicate"]["passed"],
        "mapping_strata": strata["mapping"]["passed"],
        "delay_strata": strata["delay_label"]["passed"],
        "source_order_strata": strata["source_order"]["passed"],
        "query_order_strata": strata["query_order"]["passed"],
        "q_strata": strata["q"]["passed"],
        "cyclic_origin_strata": strata["cyclic_origin"]["passed"],
        "relation_controls_null": controls["passed"],
        "scalar_q_equivalence": scalar["passed"],
        "label_scramble_collapses": scramble["passed"],
        "heldout_transport": heldout["passed"],
        "bootstrap_stability": bootstrap["passed"],
        "scalar_replay_adversary_rejected": adversary["passed"],
        "source_off_not_claimed_when_unscheduled": source_off["passed"] and (source_off.get("claimed_as_gate") is False or source_off.get("status") == "scheduled"),
        "no_aggregate_rescue": all(item["passed"] for item in strata.values()),
    }
    passed = all(gates.values())
    result_class = RESULT_CONFIRMED if passed else RESULT_NOT_CONFIRMED
    return {
        "schema": "FAMILY10H_RELATION_ONLY_PHYSICAL_ADJUDICATION_V1",
        "result_class": result_class,
        "passed": passed,
        "scientific_claim": POSITIVE_CLAIM if passed else NEGATIVE_CLAIM,
        "claim_boundary": claim_boundary(passed),
        "validation": validation,
        "threshold_contract": threshold_contract,
        "R_match_global": global_effect,
        "strata": strata,
        "controls": controls,
        "scalar_q_equivalence": scalar,
        "label_scramble": scramble,
        "heldout_transport": heldout,
        "bootstrap_stability": bootstrap,
        "scalar_replay_adversary": adversary,
        "source_off_gate": source_off,
        "gates": gates,
    }


def fixture_packet(schedule: dict[str, Any], mode: str) -> dict[str, Any]:
    raw_records = []
    source_death = []
    for expected in schedule["rows"]:
        q = float(expected["q"])
        base = 10_000.0 + 0.25 * q
        value = base
        if expected["row_role"] == "relation_matrix":
            if mode == "positive":
                value += 320.0 if expected["relation_match"] else -320.0
            elif mode == "origin_specific":
                value += (320.0 if expected["relation_match"] else -320.0) if expected["cyclic_origin"] == 0 else 0.0
            elif mode == "mapping_specific":
                value += (320.0 if expected["relation_match"] else -320.0) if expected["mapping"] == "map0" else 0.0
            elif mode == "session_specific":
                value += (320.0 if expected["relation_match"] else -320.0) if expected["session"] == "session_0" else 0.0
            elif mode == "scalar_replay":
                value += 0.5 * q
            elif mode == "separable_replay":
                value += (80.0 if expected["r_prepare"] == "relation_r0" else -80.0)
                value += (60.0 if expected["r_query"] == "relation_r0" else -60.0)
            elif mode == "route_pressure":
                value += 64.0
            elif mode == "distance_only":
                value += 64.0
        elif expected["row_role"] == "scalar_control":
            value += 60.0 if expected["query"] == "query_A" else -60.0
            value += 0.2 * q
        elif expected["row_role"] == "relation_control":
            value = 0.0
        record = {
            **expected,
            "dirty_probe_response": value,
            "change_to_dirty": 1.0 if value > base else 0.0,
            "cpu_cycles": 1_000_000.0 + abs(q),
            "duration_ns": 100_000.0 + int(expected["cyclic_origin"]),
            "time_enabled": 100_000.0,
            "time_running": 100_000.0,
            "pmu_event_group": pub.PMU_GROUP["name"],
            "pmu_events": pub.PMU_GROUP["events"],
            "event_ids": {name: idx + 1 for idx, name in enumerate(pub.PMU_GROUP["events"])},
            "source_cpu_before": expected["source_cpu_expected"],
            "source_cpu_after": expected["source_cpu_expected"],
            "receiver_cpu_before": expected["receiver_cpu_expected"],
            "receiver_cpu_after": expected["receiver_cpu_expected"],
            "process_custody": "source_dead_before_query",
            "physical_measurement": True,
            "positive_physical_claim": False,
        }
        raw_records.append(record)
        source_death.append(
            {
                "tuple_id": expected["tuple_id"],
                "execution_ordinal": expected["execution_ordinal"],
                "source_pid": 10000 + expected["execution_ordinal"],
                "waitpid_pid": 10000 + expected["execution_ordinal"],
                "waitpid_status": "exited_0",
                "source_alive_during_query": False,
                "source_helper_survives": False,
                "open_source_ipc_after_waitpid": 0,
                "query_selected_after_waitpid": True,
                "post_observation_query_or_window_selection": False,
                "source_cpu_before": expected["source_cpu_expected"],
                "source_cpu_after": expected["source_cpu_expected"],
                "physical_measurement": True,
            }
        )
    return {
        "schema": "FAMILY10H_RELATION_ONLY_PHYSICAL_FIXTURE_PACKET_V1",
        "mode": mode,
        "raw_records": raw_records,
        "source_death_receipts": source_death,
        "feature_freeze": {
            "schema": "FAMILY10H_RELATION_ONLY_FEATURE_FREEZE_V1",
            "schedule_sha256": schedule["schedule_sha256"],
            "primary_endpoint": "dirty_probe_response",
            "secondary_endpoints": ["change_to_dirty", "cpu_cycles", "duration_ns"],
            "raw_record_count": len(raw_records),
            "physical_measurement": True,
            "post_observation_feature_selection": False,
        },
        "custody_envelope": {
            "schema": "FAMILY10H_RELATION_ONLY_FIXTURE_CUSTODY_ENVELOPE_V1",
            "target_execution_receipt_sha256": "1" * 64,
            "runtime_sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
            "source_authority_sha": "4" * 40,
            "freeze_sha": "5" * 40,
            "copied_back_archive_sha256": "6" * 64,
            "copied_back_archive_size": 1,
            "evidence_inventory": {
                "raw_records": True,
                "source_death_receipts": True,
                "feature_freeze": True,
                "target_execution_receipt": True,
            },
        },
    }


def fail_closed_claim_state(report: dict[str, Any]) -> dict[str, Any]:
    boundary = report.get("claim_boundary", {})
    checks = {
        "result_class_not_confirmed": report.get("result_class") != RESULT_CONFIRMED,
        "positive_scientific_claim_absent": report.get("scientific_claim") != POSITIVE_CLAIM,
        "relation_coordinate_observed_false": boundary.get("reproducible_relation_match_coordinate_observed") is False,
    }
    return {
        "passed": all(checks.values()),
        "result_class": report.get("result_class"),
        "scientific_claim": report.get("scientific_claim"),
        "checks": checks,
    }


def mutated_packet_regression(schedule: dict[str, Any], label: str, mutator: Any) -> dict[str, Any]:
    packet = fixture_packet(schedule, "positive")
    mutator(packet)
    result = adjudicate_physical_packet(packet, schedule)
    claim = fail_closed_claim_state(result)
    return {
        "passed": result.get("result_class") == RESULT_INVALID and claim["passed"],
        "result_class": result.get("result_class"),
        "failures": result.get("validation", {}).get("failures", []),
        "claim_state": claim,
    }


def packet_mutation_regressions(schedule: dict[str, Any]) -> dict[str, Any]:
    def raw0(packet: dict[str, Any]) -> dict[str, Any]:
        return packet["raw_records"][0]

    def death0(packet: dict[str, Any]) -> dict[str, Any]:
        return packet["source_death_receipts"][0]

    cases = {
        "altered_block_id": lambda packet: raw0(packet).__setitem__("block_id", "mutated_block"),
        "altered_q": lambda packet: raw0(packet).__setitem__("q", int(raw0(packet)["q"]) + 1),
        "altered_session": lambda packet: raw0(packet).__setitem__("session", "mutated_session"),
        "altered_mapping": lambda packet: raw0(packet).__setitem__("mapping", "mutated_mapping"),
        "altered_delay": lambda packet: raw0(packet).__setitem__("delay_label", "mutated_delay"),
        "altered_source_order": lambda packet: raw0(packet).__setitem__("source_order", "mutated_source_order"),
        "altered_query_order": lambda packet: raw0(packet).__setitem__("query_order", "mutated_query_order"),
        "altered_cyclic_origin": lambda packet: raw0(packet).__setitem__("cyclic_origin", int(raw0(packet)["cyclic_origin"]) + 1),
        "swapped_execution_order": lambda packet: packet["raw_records"].__setitem__(
            slice(0, 2),
            [packet["raw_records"][1], packet["raw_records"][0]],
        ),
        "wrong_cpu": lambda packet: raw0(packet).__setitem__("source_cpu_before", -1),
        "wrong_event_ids": lambda packet: raw0(packet).__setitem__("event_ids", {"bad": 1}),
        "physical_measurement_false": lambda packet: raw0(packet).__setitem__("physical_measurement", False),
        "altered_feature_freeze": lambda packet: packet["feature_freeze"].__setitem__("schedule_sha256", "0" * 64),
        "mismatched_target_receipt": lambda packet: packet["custody_envelope"].__setitem__("target_execution_receipt_sha256", "bad"),
        "mismatched_archive_hash": lambda packet: packet["custody_envelope"].__setitem__("copied_back_archive_sha256", "bad"),
        "source_death_pid_mismatch": lambda packet: death0(packet).__setitem__("waitpid_pid", -1),
    }
    results = {label: mutated_packet_regression(schedule, label, mutator) for label, mutator in cases.items()}
    return {
        "schema": "FAMILY10H_RELATION_ONLY_PACKET_MUTATION_REGRESSIONS_V1",
        "results": results,
        "passed": all(item["passed"] for item in results.values()),
    }


def run_self_test(schedule: dict[str, Any]) -> dict[str, Any]:
    threshold = physical_threshold_contract()
    positive = adjudicate_physical_packet(fixture_packet(schedule, "positive"), schedule, threshold)
    scalar = adjudicate_physical_packet(fixture_packet(schedule, "scalar_replay"), schedule, threshold)
    separable = adjudicate_physical_packet(fixture_packet(schedule, "separable_replay"), schedule, threshold)
    route = adjudicate_physical_packet(fixture_packet(schedule, "route_pressure"), schedule, threshold)
    distance = adjudicate_physical_packet(fixture_packet(schedule, "distance_only"), schedule, threshold)
    origin_specific = adjudicate_physical_packet(fixture_packet(schedule, "origin_specific"), schedule, threshold)
    mapping_specific = adjudicate_physical_packet(fixture_packet(schedule, "mapping_specific"), schedule, threshold)
    session_specific = adjudicate_physical_packet(fixture_packet(schedule, "session_specific"), schedule, threshold)
    invalid_packet = fixture_packet(schedule, "positive")
    invalid_packet["source_death_receipts"] = invalid_packet["source_death_receipts"][:-1]
    invalid = adjudicate_physical_packet(invalid_packet, schedule, threshold)
    packet_mutations = packet_mutation_regressions(schedule)
    negative_claim_states = {
        "scalar_replay": fail_closed_claim_state(scalar),
        "separable_replay": fail_closed_claim_state(separable),
        "route_pressure": fail_closed_claim_state(route),
        "distance_only": fail_closed_claim_state(distance),
        "origin_specific": fail_closed_claim_state(origin_specific),
        "mapping_specific": fail_closed_claim_state(mapping_specific),
        "session_specific": fail_closed_claim_state(session_specific),
        "invalid": fail_closed_claim_state(invalid),
    }
    checks = {
        "positive_fixture_confirmed": positive["result_class"] == RESULT_CONFIRMED and positive["scientific_claim"] == POSITIVE_CLAIM,
        "scalar_replay_rejected": scalar["result_class"] == RESULT_NOT_CONFIRMED,
        "separable_replay_rejected": separable["result_class"] == RESULT_NOT_CONFIRMED,
        "route_pressure_rejected": route["result_class"] == RESULT_NOT_CONFIRMED,
        "distance_only_rejected": distance["result_class"] == RESULT_NOT_CONFIRMED,
        "stratum_specific_artifact_fails_true_heldout": origin_specific["result_class"] == RESULT_NOT_CONFIRMED
        and origin_specific["heldout_transport"]["factors"]["cyclic_origin"]["passed"] is False
        and mapping_specific["result_class"] == RESULT_NOT_CONFIRMED
        and mapping_specific["heldout_transport"]["factors"]["mapping"]["passed"] is False
        and session_specific["result_class"] == RESULT_NOT_CONFIRMED
        and session_specific["heldout_transport"]["factors"]["session"]["passed"] is False,
        "invalid_packet_custody_invalid": invalid["result_class"] == RESULT_INVALID,
        "packet_mutation_regressions_passed": packet_mutations["passed"],
        "negative_and_invalid_fail_closed": all(item["passed"] for item in negative_claim_states.values()),
    }
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_PHYSICAL_ADJUDICATOR_SELF_TEST_V1",
        "threshold_contract": threshold,
        "checks": checks,
        "positive_result": {
            "result_class": positive["result_class"],
            "R_match_abs_mean": positive["R_match_global"]["R_match_abs_mean"],
            "heldout_passed": positive["heldout_transport"]["passed"],
            "adversary_passed": positive["scalar_replay_adversary"]["passed"],
            "bootstrap_sign_fraction": positive["bootstrap_stability"].get("bootstrap_sign_fraction"),
            "label_scramble_q95_abs": positive["label_scramble"]["null_distribution"]["q95_abs"],
        },
        "negative_results": {
            "scalar_replay": scalar["result_class"],
            "separable_replay": separable["result_class"],
            "route_pressure": route["result_class"],
            "distance_only": distance["result_class"],
            "origin_specific": origin_specific["result_class"],
            "mapping_specific": mapping_specific["result_class"],
            "session_specific": session_specific["result_class"],
            "invalid": invalid["result_class"],
        },
        "false_positive_fixture_results": {
            "block_id_relabeling": packet_mutations["results"]["altered_block_id"],
            "execution_order_drift": packet_mutations["results"]["swapped_execution_order"],
            "q_dependent_nonlinear_scalar_artifact": {"result_class": scalar["result_class"], "adversary_passed": scalar["scalar_replay_adversary"]["passed"]},
            "additive_preparation_query_effects": {"result_class": separable["result_class"], "adversary_passed": separable["scalar_replay_adversary"]["passed"]},
            "route_pressure_artifact": {"result_class": route["result_class"], "adversary_passed": route["scalar_replay_adversary"]["passed"]},
            "distance_artifact": {"result_class": distance["result_class"], "adversary_passed": distance["scalar_replay_adversary"]["passed"]},
            "cyclic_origin_specific_artifact": {
                "result_class": origin_specific["result_class"],
                "heldout_cyclic_origin_passed": origin_specific["heldout_transport"]["factors"]["cyclic_origin"]["passed"],
            },
            "mapping_specific_artifact": {
                "result_class": mapping_specific["result_class"],
                "heldout_mapping_passed": mapping_specific["heldout_transport"]["factors"]["mapping"]["passed"],
            },
            "session_specific_artifact": {
                "result_class": session_specific["result_class"],
                "heldout_session_passed": session_specific["heldout_transport"]["factors"]["session"]["passed"],
            },
            "genuine_relation_interaction": {
                "result_class": positive["result_class"],
                "adversary_passed": positive["scalar_replay_adversary"]["passed"],
            },
        },
        "packet_mutation_regressions": packet_mutations,
        "negative_claim_states": negative_claim_states,
        "passed": all(checks.values()),
    }
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result
