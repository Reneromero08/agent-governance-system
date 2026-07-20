#!/usr/bin/env python3
"""Prospective physical adjudicator for relation-only matched permutations.

The functions here are offline and fixture-driven until a separately authorized
live transaction provides raw evidence. All negative and invalid paths fail
closed with no positive scientific claim.
"""

from __future__ import annotations

import math
import random
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
    seen = set()
    for record in raw_records:
        tuple_id = record.get("tuple_id")
        expected = schedule_by_id.get(tuple_id)
        if expected is None:
            failures.append("unexpected tuple_id")
            continue
        if tuple_id in seen:
            failures.append("duplicate raw tuple_id")
        seen.add(tuple_id)
        for field in [
            "execution_ordinal",
            "row_role",
            "r_prepare",
            "r_query",
            "query",
            "cyclic_origin",
            "expected_pmu_group",
        ]:
            if record.get(field) != expected[field]:
                failures.append(f"{field} mismatch")
        for metric in ["dirty_probe_response", "change_to_dirty", "cpu_cycles", "duration_ns"]:
            if type(record.get(metric)) not in {int, float}:
                failures.append(f"{metric} not numeric")
        if record.get("process_custody") != "source_dead_before_query":
            failures.append("process custody mismatch")
        if record.get("pmu_event_group") != pub.PMU_GROUP["name"]:
            failures.append("PMU event group mismatch")
        if record.get("positive_physical_claim") is True:
            failures.append("positive claim leakage in raw record")
        if len(failures) > 32:
            break
    death_by_id = {row.get("tuple_id"): row for row in source_death if isinstance(row, dict)}
    if len(death_by_id) != len(source_death):
        failures.append("source-death receipt duplicate or malformed")
    for tuple_id, expected in schedule_by_id.items():
        receipt = death_by_id.get(tuple_id)
        if not isinstance(receipt, dict):
            failures.append("source-death receipt missing")
            break
        if receipt.get("query_selected_after_waitpid") is not True:
            failures.append("query selected before source death")
        if receipt.get("source_alive_during_query") is not False:
            failures.append("source alive during query")
        if receipt.get("post_observation_query_or_window_selection") is not False:
            failures.append("post-observation selection")
    if not isinstance(feature_freeze, dict):
        failures.append("feature_freeze missing")
    else:
        if feature_freeze.get("primary_endpoint") != "dirty_probe_response":
            failures.append("feature freeze primary endpoint mismatch")
        if feature_freeze.get("schedule_sha256") != schedule.get("schedule_sha256"):
            failures.append("feature freeze schedule binding mismatch")
        if feature_freeze.get("post_observation_feature_selection") is not False:
            failures.append("post-observation feature selection")
    return {"passed": not failures, "failures": failures[:64], "raw_record_count": len(raw_records)}


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


def replay_adversary_report(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    null_threshold = threshold_contract["resampling_thresholds"]["matched_permutation_null_abs_max"]
    specs = {
        "scalar_q_replay": ["q"],
        "nonlinear_scalar_q_replay": ["q", "mapping", "delay_label", "source_order", "query_order", "cyclic_origin"],
        "separable_a_b_marginal_replay": ["r_prepare", "q", "mapping", "delay_label", "source_order", "query_order", "cyclic_origin"],
        "route_pressure_replay": ["query_order", "source_order", "delay_label", "cyclic_origin"],
        "distance_only_replay": ["q", "cyclic_origin"],
        "source_order_confounding_replay": ["source_order", "q", "mapping", "delay_label"],
        "query_order_confounding_replay": ["query_order", "q", "mapping", "delay_label"],
        "cyclic_origin_confounding_replay": ["cyclic_origin", "q", "mapping", "delay_label"],
        "mapping_session_replicate_delay_transport_replay": ["mapping", "session", "replicate", "delay_label", "q"],
    }
    adversaries = {}
    for name, keys in specs.items():
        summary = neutralized_effect(rows, threshold_contract, keys)
        adversaries[name] = {
            "conditioning_keys": keys,
            "R_match_abs_of_mean_after_replay": summary["R_match_abs_of_mean"],
            "R_match_abs_mean_after_replay": summary["R_match_abs_mean"],
            "passed": summary["R_match_abs_of_mean"] <= null_threshold,
        }
    return {
        "schema": "FAMILY10H_RELATION_ONLY_SCALAR_REPLAY_ADVERSARY_REPORT_V1",
        "null_threshold": null_threshold,
        "adversaries": adversaries,
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
            "event_ids": {name: idx + 1 for idx, name in enumerate(pub.PMU_GROUP["events"])},
            "source_cpu_before": expected["source_cpu_expected"],
            "source_cpu_after": expected["source_cpu_expected"],
            "receiver_cpu_before": expected["receiver_cpu_expected"],
            "receiver_cpu_after": expected["receiver_cpu_expected"],
            "process_custody": "source_dead_before_query",
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
            "post_observation_feature_selection": False,
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


def run_self_test(schedule: dict[str, Any]) -> dict[str, Any]:
    threshold = physical_threshold_contract()
    positive = adjudicate_physical_packet(fixture_packet(schedule, "positive"), schedule, threshold)
    scalar = adjudicate_physical_packet(fixture_packet(schedule, "scalar_replay"), schedule, threshold)
    separable = adjudicate_physical_packet(fixture_packet(schedule, "separable_replay"), schedule, threshold)
    route = adjudicate_physical_packet(fixture_packet(schedule, "route_pressure"), schedule, threshold)
    distance = adjudicate_physical_packet(fixture_packet(schedule, "distance_only"), schedule, threshold)
    origin_specific = adjudicate_physical_packet(fixture_packet(schedule, "origin_specific"), schedule, threshold)
    invalid_packet = fixture_packet(schedule, "positive")
    invalid_packet["source_death_receipts"] = invalid_packet["source_death_receipts"][:-1]
    invalid = adjudicate_physical_packet(invalid_packet, schedule, threshold)
    negative_claim_states = {
        "scalar_replay": fail_closed_claim_state(scalar),
        "separable_replay": fail_closed_claim_state(separable),
        "route_pressure": fail_closed_claim_state(route),
        "distance_only": fail_closed_claim_state(distance),
        "origin_specific": fail_closed_claim_state(origin_specific),
        "invalid": fail_closed_claim_state(invalid),
    }
    checks = {
        "positive_fixture_confirmed": positive["result_class"] == RESULT_CONFIRMED and positive["scientific_claim"] == POSITIVE_CLAIM,
        "scalar_replay_rejected": scalar["result_class"] == RESULT_NOT_CONFIRMED,
        "separable_replay_rejected": separable["result_class"] == RESULT_NOT_CONFIRMED,
        "route_pressure_rejected": route["result_class"] == RESULT_NOT_CONFIRMED,
        "distance_only_rejected": distance["result_class"] == RESULT_NOT_CONFIRMED,
        "stratum_specific_artifact_fails_true_heldout": origin_specific["result_class"] == RESULT_NOT_CONFIRMED
        and origin_specific["heldout_transport"]["factors"]["cyclic_origin"]["passed"] is False,
        "invalid_packet_custody_invalid": invalid["result_class"] == RESULT_INVALID,
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
            "invalid": invalid["result_class"],
        },
        "negative_claim_states": negative_claim_states,
        "passed": all(checks.values()),
    }
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result
