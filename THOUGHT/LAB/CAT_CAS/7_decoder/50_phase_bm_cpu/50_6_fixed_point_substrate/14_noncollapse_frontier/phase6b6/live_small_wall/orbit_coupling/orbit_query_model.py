#!/usr/bin/env python3
"""Offline OrbitState public-query model and mock runtime.

This file is non-driving. It never contacts the lab device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orbit_query_public as public_model


SCHEMA_ID = "CAT_CAS_ORBIT_QUERY_MODEL_V1"
N = 256
D = 23
FOLD_D = N - D
EQUAL_ORBIT_MEMBER = 0
PHASES = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
PHASE_NAMES = ("0", "pi/2", "pi", "3pi/2")
SCRAMBLED_SOURCE_PHASES = (0.0, math.pi, 0.0, math.pi)
BASE_WORK = 2048
QUANT_SCALE = 1024
BANK_LINES = 4096
REL_BALANCE_TOL = 0.35
REL_REAL_TOL = 0.35
ABS_FLOOR = 1.0
NULL_MULTIPLIER = 3.0
EXPECTED_REPLICATES = (0, 1)
FORBIDDEN_RECEIVER_TERMS = (
    "condition",
    "member",
    "target",
    "plus",
    "minus",
    "lower",
    "orientation",
    "branch",
    "quantized",
    "positive_work",
    "negative_work",
    "dummy_work",
)


@dataclass(frozen=True)
class Condition:
    name: str
    member: int | None
    response_mode: str
    source_phases: tuple[float, ...] = PHASES
    bank_swap: bool = False
    public_label_swap: bool = False


CONDITIONS: tuple[Condition, ...] = (
    Condition("pre_projection_d", D, "pre"),
    Condition("pre_projection_fold", FOLD_D, "pre"),
    Condition("source_off", None, "source_off"),
    Condition("query_off", D, "query_off"),
    Condition("post_projection", D, "post"),
    Condition("declaration_sham", D, "declaration_sham"),
    Condition("query_scramble", D, "pre", SCRAMBLED_SOURCE_PHASES),
    Condition("equal_orbit_odd_zero", EQUAL_ORBIT_MEMBER, "pre"),
    Condition("physical_bank_swap", D, "pre", PHASES, True),
    Condition("public_label_swap", D, "pre", PHASES, False, True),
)
RESPONSE_MODE_CODES = {
    "pre": 0,
    "source_off": 1,
    "query_off": 2,
    "post": 3,
    "declaration_sham": 4,
}
CONDITION_ORDER = (4, 0, 6, 2, 8, 1, 5, 9, 3, 7)
OPAQUE_GROUP_IDS = (
    "g7c19",
    "g2f80",
    "g9a04",
    "g41de",
    "gc633",
    "g0b72",
    "gde58",
    "g6f31",
    "g83aa",
    "g14e5",
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def quantize(value: float) -> int:
    q = int(round(value * QUANT_SCALE))
    return max(-BASE_WORK, min(BASE_WORK, q))


def response_for(condition: Condition, phase_index: int) -> float:
    if condition.response_mode == "source_off":
        return 0.0
    if condition.response_mode == "query_off":
        return 0.0
    if condition.member is None:
        return 0.0
    phi = 2.0 * math.pi * condition.member / N
    theta = condition.source_phases[phase_index]
    if condition.response_mode == "post":
        return math.cos(phi) * math.cos(PHASES[phase_index])
    if condition.response_mode == "declaration_sham":
        return math.cos(phi)
    if condition.response_mode == "pre":
        value = math.cos(phi - theta)
        return -value if condition.bank_swap else value
    raise ValueError(f"unknown mode: {condition.response_mode}")


def source_response_for_work(condition: Condition, phase_index: int) -> float:
    if condition.response_mode == "source_off":
        return 0.0
    if condition.response_mode == "query_off":
        return 0.0
    if condition.member is None:
        return 0.0
    phi = 2.0 * math.pi * condition.member / N
    theta = condition.source_phases[phase_index]
    if condition.response_mode == "post":
        return math.cos(phi) * math.cos(PHASES[phase_index])
    if condition.response_mode == "declaration_sham":
        return math.cos(phi)
    if condition.response_mode == "pre":
        return math.cos(phi - theta)
    raise ValueError(f"unknown mode: {condition.response_mode}")


def work_allocation(condition: Condition, phase_index: int) -> dict[str, int]:
    if condition.response_mode == "source_off":
        return {
            "positive_work": 0,
            "negative_work": 0,
            "dummy_work": 2 * BASE_WORK,
            "total_work": 2 * BASE_WORK,
            "q_theta": 0,
        }
    q = quantize(source_response_for_work(condition, phase_index))
    if condition.bank_swap:
        q = -q
    return {
        "positive_work": BASE_WORK + q,
        "negative_work": BASE_WORK - q,
        "dummy_work": 0,
        "total_work": 2 * BASE_WORK,
        "q_theta": q,
    }


def public_phase_ordinal(public_index: int, replicate: int, phase_index: int, label_swap: bool = False) -> int:
    offset = (public_index * 3 + replicate + (1 if label_swap else 0)) % 4
    return (phase_index + offset) % 4


def reconstruct_complex(responses: list[float], phases: tuple[float, ...] = PHASES) -> complex:
    if len(responses) != len(phases):
        raise ValueError("response and phase lengths differ")
    total = 0.0 + 0.0j
    for response, phase in zip(responses, phases):
        total += response * complex(math.cos(phase), math.sin(phase))
    return (2.0 / float(len(phases))) * total


def build_manifest_and_unblind() -> tuple[dict[str, Any], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    unblind: list[dict[str, Any]] = []
    for replicate in range(2):
        for public_index, condition_index in enumerate(CONDITION_ORDER):
            condition = CONDITIONS[condition_index]
            group_id = f"{OPAQUE_GROUP_IDS[public_index]}r{replicate}"
            records = []
            for phase_index, phase in enumerate(PHASES):
                records.append(
                    {
                        "opaque_run_id": f"{OPAQUE_GROUP_IDS[public_index]}p{phase_index}r{replicate}",
                        "phase_ordinal": public_phase_ordinal(
                            public_index, replicate, phase_index, condition.public_label_swap
                        ),
                        "decoder_phase_index": phase_index,
                        "decoder_phase_radians": phase,
                    }
                )
            groups.append(
                {
                    "group_id": group_id,
                    "replicate_index": replicate,
                    "records": records,
                    "operator": "byte_preserving_same_value_store",
                    "base_work_units": BASE_WORK,
                    "total_source_work_units": 2 * BASE_WORK,
                }
            )
            unblind.append(
                {
                    "group_id": group_id,
                    "replicate_index": replicate,
                    "condition": condition.name,
                    "member": condition.member,
                    "response_mode": condition.response_mode,
                    "bank_swap": condition.bank_swap,
                    "public_label_swap": condition.public_label_swap,
                    "quantized_work": [
                        {"phase_ordinal": phase_index, **work_allocation(condition, phase_index)}
                        for phase_index in range(4)
                    ],
                    "source_work_receipt": [
                        {"phase_ordinal": phase_index, "source_rc": 0, **work_allocation(condition, phase_index)}
                        for phase_index in range(4)
                    ],
                }
            )
    manifest = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_CAPTURE_MANIFEST_V1",
        "modulus": N,
        "phase_names": list(PHASE_NAMES),
        "decoder": "Z=(2/K)*sum(response_k*exp(i*theta_k))",
        "groups": groups,
    }
    return manifest, {"schema_id": "CAT_CAS_ORBIT_QUERY_UNBLINDING_MAP_V1", "groups": unblind}


def assert_receiver_manifest_blind(manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest, sort_keys=True).lower()
    leaked = [term for term in FORBIDDEN_RECEIVER_TERMS if term in text]
    if leaked:
        raise AssertionError(f"receiver manifest leaks private labels: {leaked}")
    for group in manifest["groups"]:
        text = json.dumps(group, sort_keys=True).lower()
        leaked = [term for term in FORBIDDEN_RECEIVER_TERMS if term in text]
        if leaked:
            raise AssertionError(f"group {group['group_id']} leaks {leaked}")


def model_summary() -> dict[str, Any]:
    condition_results: dict[str, Any] = {}
    for condition in CONDITIONS:
        responses = [response_for(condition, phase) for phase in range(4)]
        z = reconstruct_complex(responses)
        allocations = [work_allocation(condition, phase) for phase in range(4)]
        condition_results[condition.name] = {
            "responses": responses,
            "z": {"real": z.real, "imag": z.imag, "abs": abs(z)},
            "allocations": allocations,
            "equal_total_work": all(item["total_work"] == 2 * BASE_WORK for item in allocations),
        }

    null_names = (
        "source_off",
        "query_off",
        "post_projection",
        "declaration_sham",
        "query_scramble",
        "equal_orbit_odd_zero",
    )
    null_ceiling = max(abs(condition_results[name]["z"]["imag"]) for name in null_names)
    pre = condition_results["pre_projection_d"]["z"]
    fold = condition_results["pre_projection_fold"]["z"]
    bank_swap = condition_results["physical_bank_swap"]["z"]
    label_swap = condition_results["public_label_swap"]["z"]
    signal_floor = NULL_MULTIPLIER * max(null_ceiling, 1.0e-12)
    acceptance = {
        "passive_fold_even_equality": abs(math.cos(2 * math.pi * D / N) - math.cos(2 * math.pi * FOLD_D / N)) < 1e-12,
        "pre_projection_odd_sign_reversal": pre["imag"] * fold["imag"] < 0.0,
        "pre_projection_real_match": abs(pre["real"] - fold["real"]) <= 1e-12,
        "post_projection_odd_null": abs(condition_results["post_projection"]["z"]["imag"]) <= 1e-12,
        "source_off_odd_null": abs(condition_results["source_off"]["z"]["imag"]) <= 1e-12,
        "query_off_odd_null": abs(condition_results["query_off"]["z"]["imag"]) <= 1e-12,
        "declaration_sham_odd_null": abs(condition_results["declaration_sham"]["z"]["imag"]) <= 1e-12,
        "query_scramble_odd_null": abs(condition_results["query_scramble"]["z"]["imag"]) <= 1e-12,
        "equal_orbit_odd_null": abs(condition_results["equal_orbit_odd_zero"]["z"]["imag"]) <= 1e-12,
        "bank_swap_transformation": abs(bank_swap["imag"] + pre["imag"]) <= 1e-12 and abs(bank_swap["real"] + pre["real"]) <= 1e-12,
        "public_label_swap_invariant": abs(label_swap["imag"] - pre["imag"]) <= 1e-12 and abs(label_swap["real"] - pre["real"]) <= 1e-12,
        "equal_total_physical_work": all(result["equal_total_work"] for result in condition_results.values()),
        "pre_projection_exceeds_null_floor": min(abs(pre["imag"]), abs(fold["imag"])) > signal_floor,
    }
    manifest, unblind = build_manifest_and_unblind()
    assert_receiver_manifest_blind(manifest)
    acceptance["no_private_label_in_receiver_payload"] = True
    result = {
        "schema_id": SCHEMA_ID,
        "constants": {
            "N": N,
            "d": D,
            "fold_d": FOLD_D,
            "base_work": BASE_WORK,
            "quant_scale": QUANT_SCALE,
            "phases": list(PHASES),
        },
        "condition_results": condition_results,
        "null_ceiling": null_ceiling,
        "acceptance": acceptance,
        "model_passed": all(acceptance.values()),
        "capture_manifest_sha256": digest(manifest),
        "unblinding_map_sha256": digest(unblind),
    }
    result["result_sha256"] = digest({k: v for k, v in result.items() if k != "result_sha256"})
    return result


def build_mock_capture(signal_present: bool = True) -> dict[str, Any]:
    manifest, unblind = build_manifest_and_unblind()
    raw_records: list[dict[str, Any]] = []
    gain = 100.0 if signal_present else 0.0
    for group in manifest["groups"]:
        hidden = next(item for item in unblind["groups"] if item["group_id"] == group["group_id"])
        condition = next(item for item in CONDITIONS if item.name == hidden["condition"])
        for record in group["records"]:
            phase_index = int(record["decoder_phase_index"])
            response = gain * response_for(condition, phase_index)
            raw_records.append(
                {
                    "schema_id": "CAT_CAS_ORBIT_QUERY_RAW_RECORD_V1",
                    "opaque_run_id": record["opaque_run_id"],
                    "group_id": group["group_id"],
                    "replicate_index": group["replicate_index"],
                    "phase_ordinal": int(record["phase_ordinal"]),
                    "decoder_phase_index": record["decoder_phase_index"],
                    "change_to_dirty_delta": response,
                    "probe_dirty_delta": 2.0 * response,
                    "duration_delta_ns": 0.5 * response,
                    "restoration_passed": True,
                }
            )
    return {"manifest": manifest, "unblind": unblind, "raw_records": raw_records}


def extract_features(manifest: dict[str, Any], raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_raw_keys = {
        "schema_id",
        "opaque_run_id",
        "group_id",
        "replicate_index",
        "phase_ordinal",
        "decoder_phase_index",
        "measurement_order",
        "positive_cycles",
        "negative_cycles",
        "positive_change_to_dirty",
        "negative_change_to_dirty",
        "positive_probe_dirty",
        "negative_probe_dirty",
        "positive_duration_ns",
        "negative_duration_ns",
        "change_to_dirty_delta",
        "probe_dirty_delta",
        "duration_delta_ns",
        "restoration_passed",
        "perf_available",
        "initial_positive_digest",
        "initial_negative_digest",
        "final_positive_digest",
        "final_negative_digest",
    }
    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in raw_records:
        extra = set(record) - allowed_raw_keys
        if extra:
            raise AssertionError(f"raw record contains unexpected fields: {sorted(extra)}")
        by_group.setdefault(str(record["group_id"]), []).append(record)
    features: list[dict[str, Any]] = []
    for group in manifest["groups"]:
        manifest_ids = {record["opaque_run_id"] for record in group["records"]}
        records = sorted(by_group[group["group_id"]], key=lambda item: int(item["phase_ordinal"]))
        if len(records) != 4:
            raise AssertionError(f"group {group['group_id']} does not have 4 records")
        if {int(record["phase_ordinal"]) for record in records} != {0, 1, 2, 3}:
            raise AssertionError(f"group {group['group_id']} has invalid phase ordinals")
        if {record["opaque_run_id"] for record in records} != manifest_ids:
            raise AssertionError(f"group {group['group_id']} raw IDs do not match manifest")
        for record in records:
            if int(record["decoder_phase_index"]) != int(record["phase_ordinal"]):
                raise AssertionError(f"group {group['group_id']} decoder phase drift")
        decoded: dict[str, Any] = {}
        for key in ("change_to_dirty_delta", "probe_dirty_delta", "duration_delta_ns"):
            values = [float(record[key]) for record in records]
            z = reconstruct_complex(values)
            decoded[key] = {"real": z.real, "imag": z.imag, "abs": abs(z), "responses": values}
        features.append(
            {
                "group_id": group["group_id"],
                "replicate_index": group["replicate_index"],
                "decoded": decoded,
                "restoration_passed": all(bool(record["restoration_passed"]) for record in records),
            }
        )
    result = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_FEATURES_FROZEN_V1",
        "primary_response": "change_to_dirty_delta",
        "features": features,
    }
    result["features_sha256"] = digest({k: v for k, v in result.items() if k != "features_sha256"})
    return result


assert_receiver_manifest_blind = public_model.assert_receiver_manifest_blind
extract_features = public_model.extract_features


def adjudicate(features: dict[str, Any], unblind: dict[str, Any]) -> dict[str, Any]:
    hidden = {item["group_id"]: item for item in unblind["groups"]}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for feature in features["features"]:
        condition = hidden[feature["group_id"]]["condition"]
        grouped.setdefault(condition, []).append(feature)
    feature_replicates = sorted({int(item["replicate_index"]) for item in features["features"]})
    condition_names = {condition.name for condition in CONDITIONS}
    expected_feature_count = len(EXPECTED_REPLICATES) * len(CONDITIONS)

    def z_for(condition: str, replicate: int, key: str = "change_to_dirty_delta") -> complex:
        matches = [item for item in grouped[condition] if int(item["replicate_index"]) == replicate]
        if len(matches) != 1:
            raise AssertionError(f"expected one feature for {condition} replicate {replicate}")
        item = matches[0]
        return complex(item["decoded"][key]["real"], item["decoded"][key]["imag"])

    def mean_z(condition: str, key: str = "change_to_dirty_delta") -> complex:
        zs = [
            complex(item["decoded"][key]["real"], item["decoded"][key]["imag"])
            for item in grouped[condition]
        ]
        return sum(zs, 0j) / len(zs)

    def hidden_for(condition: str, replicate: int) -> dict[str, Any]:
        matches = [
            item for item in hidden.values()
            if item["condition"] == condition and int(item["replicate_index"]) == replicate
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one unblind group for {condition} replicate {replicate}")
        return matches[0]

    def condition_from_unblind(group: dict[str, Any]) -> Condition:
        name = str(group["condition"])
        base = next(item for item in CONDITIONS if item.name == name)
        member = group["member"]
        if member != base.member:
            raise AssertionError(f"unblind member drift for {name}")
        response_mode = group["response_mode"]
        if isinstance(response_mode, str):
            mode_matches = response_mode == base.response_mode
        else:
            mode_matches = int(response_mode) == RESPONSE_MODE_CODES[base.response_mode]
        if not mode_matches:
            raise AssertionError(f"unblind response mode drift for {name}")
        if not isinstance(group["bank_swap"], bool) or not isinstance(group["public_label_swap"], bool):
            raise AssertionError(f"unblind boolean type drift for {name}")
        if bool(group["bank_swap"]) != base.bank_swap:
            raise AssertionError(f"unblind bank swap drift for {name}")
        if bool(group["public_label_swap"]) != base.public_label_swap:
            raise AssertionError(f"unblind public label swap drift for {name}")
        return Condition(
            name=name,
            member=None if member is None else int(member),
            response_mode=base.response_mode,
            source_phases=base.source_phases,
            bank_swap=bool(group["bank_swap"]),
            public_label_swap=bool(group["public_label_swap"]),
        )

    def work_row_ok(recomputed: dict[str, Any], expected: dict[str, Any], receipt: dict[str, Any], condition: str) -> bool:
        positive = int(receipt["positive_work"])
        negative = int(receipt["negative_work"])
        dummy = int(receipt["dummy_work"])
        total = int(receipt["total_work"])
        expected_matches_recomputed = (
            int(expected["q_theta"]) == int(recomputed["q_theta"])
            and int(expected["positive_work"]) == int(recomputed["positive_work"])
            and int(expected["negative_work"]) == int(recomputed["negative_work"])
            and int(expected["dummy_work"]) == int(recomputed["dummy_work"])
            and int(expected["total_work"]) == int(recomputed["total_work"])
        )
        receipt_matches_recomputed = (
            positive == int(recomputed["positive_work"])
            and negative == int(recomputed["negative_work"])
            and dummy == int(recomputed["dummy_work"])
            and total == int(recomputed["total_work"])
        )
        source_ok = "source_rc" in receipt and int(receipt["source_rc"]) == 0
        balanced_total = positive + negative + dummy == 2 * BASE_WORK == total
        measured_bounded = positive < BANK_LINES and negative < BANK_LINES
        dummy_bounded = dummy <= BANK_LINES
        if condition == "source_off":
            control_shape = positive == 0 and negative == 0 and dummy == 2 * BASE_WORK
        elif condition == "query_off":
            control_shape = positive == BASE_WORK and negative == BASE_WORK and dummy == 0
        else:
            control_shape = dummy == 0
        return (
            source_ok
            and expected_matches_recomputed
            and receipt_matches_recomputed
            and balanced_total
            and measured_bounded
            and dummy_bounded
            and control_shape
        )

    def work_group_result(condition: str, replicate: int) -> dict[str, Any]:
        group = hidden_for(condition, replicate)
        expected_rows = sorted(group["quantized_work"], key=lambda item: int(item["phase_ordinal"]))
        receipt_rows = sorted(group["source_work_receipt"], key=lambda item: int(item["phase_ordinal"]))
        condition_spec = condition_from_unblind(group)
        recomputed_rows = [
            {"phase_ordinal": phase, **work_allocation(condition_spec, phase)}
            for phase in range(4)
        ]
        phases_expected = [int(item["phase_ordinal"]) for item in expected_rows]
        phases_receipt = [int(item["phase_ordinal"]) for item in receipt_rows]
        rows_ok = (
            phases_expected == [0, 1, 2, 3]
            and phases_receipt == [0, 1, 2, 3]
            and all(
                work_row_ok(recomputed, expected, receipt, condition)
                for recomputed, expected, receipt in zip(recomputed_rows, expected_rows, receipt_rows)
            )
        )
        return {
            "condition": condition,
            "replicate_index": replicate,
            "phase_ordinals_expected": phases_expected,
            "phase_ordinals_receipt": phases_receipt,
            "rows_passed": rows_ok,
        }

    null_conditions = [
        "source_off",
        "query_off",
        "post_projection",
        "declaration_sham",
        "query_scramble",
        "equal_orbit_odd_zero",
    ]
    pre = mean_z("pre_projection_d")
    fold = mean_z("pre_projection_fold")
    bank_swap = mean_z("physical_bank_swap")
    label_swap = mean_z("public_label_swap")
    null_ceiling = max(abs(mean_z(name).imag) for name in null_conditions)
    required_floor = NULL_MULTIPLIER * max(null_ceiling, ABS_FLOOR)
    balance_denominator = max(abs(pre.imag), abs(fold.imag), ABS_FLOOR)
    real_denominator = max(abs(pre.real), abs(fold.real), ABS_FLOOR)
    work_results = [
        work_group_result(condition.name, replicate)
        for replicate in EXPECTED_REPLICATES
        for condition in CONDITIONS
    ]
    replicate_results: list[dict[str, Any]] = []
    for replicate in EXPECTED_REPLICATES:
        r_pre = z_for("pre_projection_d", replicate)
        r_fold = z_for("pre_projection_fold", replicate)
        r_bank = z_for("physical_bank_swap", replicate)
        r_label = z_for("public_label_swap", replicate)
        r_null_ceiling = max(abs(z_for(name, replicate).imag) for name in null_conditions)
        r_required_floor = NULL_MULTIPLIER * max(r_null_ceiling, ABS_FLOOR)
        r_balance_denominator = max(abs(r_pre.imag), abs(r_fold.imag), ABS_FLOOR)
        r_real_denominator = max(abs(r_pre.real), abs(r_fold.real), ABS_FLOOR)
        complex_tolerance = max(ABS_FLOOR, REL_BALANCE_TOL * max(abs(r_pre), ABS_FLOOR))
        checks = {
            "opposed_fold_odd_signs": r_pre.imag * r_fold.imag < 0.0,
            "fold_odd_exceeds_null_ceiling": min(abs(r_pre.imag), abs(r_fold.imag)) > r_required_floor,
            "fold_odd_magnitudes_balanced": abs(abs(r_pre.imag) - abs(r_fold.imag)) / r_balance_denominator <= REL_BALANCE_TOL,
            "fold_even_reals_match": abs(r_pre.real - r_fold.real) / r_real_denominator <= REL_REAL_TOL,
            "null_controls_below_ceiling": all(abs(z_for(name, replicate).imag) <= max(r_null_ceiling, ABS_FLOOR) for name in null_conditions),
            "bank_swap_complex_transform": abs(r_bank + r_pre) <= complex_tolerance,
            "label_swap_complex_invariant": abs(r_label - r_pre) <= complex_tolerance,
            "restoration_passed": all(
                item["restoration_passed"] and int(item["replicate_index"]) == replicate
                for items in grouped.values()
                for item in items
                if int(item["replicate_index"]) == replicate
            ),
            "source_work_receipt_matches_frozen_law": all(
                item["rows_passed"] for item in work_results if int(item["replicate_index"]) == replicate
            ),
        }
        replicate_results.append(
            {
                "replicate_index": replicate,
                "z": {
                    "pre_projection_d": {"real": r_pre.real, "imag": r_pre.imag, "abs": abs(r_pre)},
                    "pre_projection_fold": {"real": r_fold.real, "imag": r_fold.imag, "abs": abs(r_fold)},
                    "physical_bank_swap": {"real": r_bank.real, "imag": r_bank.imag, "abs": abs(r_bank)},
                    "public_label_swap": {"real": r_label.real, "imag": r_label.imag, "abs": abs(r_label)},
                    "null_ceiling": r_null_ceiling,
                    "required_floor": r_required_floor,
                },
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    acceptance = {
        "expected_replicates_present": feature_replicates == list(EXPECTED_REPLICATES),
        "expected_condition_set_present": set(grouped) == condition_names,
        "expected_feature_count_present": len(features["features"]) == expected_feature_count,
        "all_replicates_pass_frozen_law": all(item["passed"] for item in replicate_results),
        "aggregate_opposed_fold_odd_signs": pre.imag * fold.imag < 0.0,
        "aggregate_fold_odd_exceeds_null_ceiling": min(abs(pre.imag), abs(fold.imag)) > required_floor,
        "aggregate_fold_odd_magnitudes_balanced": abs(abs(pre.imag) - abs(fold.imag)) / balance_denominator <= REL_BALANCE_TOL,
        "aggregate_fold_even_reals_match": abs(pre.real - fold.real) / real_denominator <= REL_REAL_TOL,
        "aggregate_bank_swap_complex_transform": abs(bank_swap + pre) <= max(ABS_FLOOR, REL_BALANCE_TOL * max(abs(pre), ABS_FLOOR)),
        "aggregate_label_swap_complex_invariant": abs(label_swap - pre) <= max(ABS_FLOOR, REL_BALANCE_TOL * max(abs(pre), ABS_FLOOR)),
        "restoration_passed": all(item["restoration_passed"] for item in features["features"]),
        "source_work_receipt_matches_frozen_law": all(item["rows_passed"] for item in work_results),
    }
    status = (
        "ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE"
        if all(acceptance.values())
        else "ORBITSTATE_PHYSICAL_QUERY_COUPLING_NOT_ESTABLISHED"
    )
    result = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_ADJUDICATION_V1",
        "status": status,
        "claim_ceiling": "ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE; SMALL_WALL_CROSSED forbidden",
        "primary_response": "change_to_dirty_delta",
        "z_summary": {
            "pre_projection_d": {"real": pre.real, "imag": pre.imag, "abs": abs(pre)},
            "pre_projection_fold": {"real": fold.real, "imag": fold.imag, "abs": abs(fold)},
            "physical_bank_swap": {"real": bank_swap.real, "imag": bank_swap.imag, "abs": abs(bank_swap)},
            "public_label_swap": {"real": label_swap.real, "imag": label_swap.imag, "abs": abs(label_swap)},
            "null_ceiling": null_ceiling,
            "required_floor": required_floor,
        },
        "acceptance": acceptance,
        "replicate_results": replicate_results,
        "source_work_results": work_results,
    }
    result["adjudication_sha256"] = digest({k: v for k, v in result.items() if k != "adjudication_sha256"})
    return result


def self_test() -> dict[str, Any]:
    model = model_summary()
    if not model["model_passed"]:
        raise AssertionError("model acceptance failed")
    present = build_mock_capture(signal_present=True)
    features = extract_features(present["manifest"], present["raw_records"])
    frozen_hash = features["features_sha256"]
    adjudication = adjudicate(features, present["unblind"])
    absent = build_mock_capture(signal_present=False)
    absent_features = extract_features(absent["manifest"], absent["raw_records"])
    absent_adjudication = adjudicate(absent_features, absent["unblind"])
    smuggled = json.loads(json.dumps(present["manifest"]))
    smuggled["groups"][0]["condition"] = "pre_projection_d"
    smuggled_rejected = False
    try:
        assert_receiver_manifest_blind(smuggled)
    except AssertionError:
        smuggled_rejected = True
    if not smuggled_rejected:
        raise AssertionError("smuggled condition label was not rejected")
    leaked_raw = json.loads(json.dumps(present["raw_records"]))
    leaked_raw[0]["positive_work_units"] = 2048
    leaked_raw_rejected = False
    try:
        extract_features(present["manifest"], leaked_raw)
    except AssertionError:
        leaked_raw_rejected = True
    if not leaked_raw_rejected:
        raise AssertionError("pre-freeze work-unit leak was not rejected")
    drifted_raw = json.loads(json.dumps(present["raw_records"]))
    drifted_raw[0]["decoder_phase_index"] = (int(drifted_raw[0]["decoder_phase_index"]) + 1) % 4
    raw_manifest_drift_rejected = False
    try:
        extract_features(present["manifest"], drifted_raw)
    except AssertionError:
        raw_manifest_drift_rejected = True
    if not raw_manifest_drift_rejected:
        raise AssertionError("raw-to-manifest decoder drift was not rejected")
    unequal_unblind = json.loads(json.dumps(present["unblind"]))
    unequal_unblind["groups"][0]["source_work_receipt"][0]["total_work"] = 1
    unequal_adjudication = adjudicate(features, unequal_unblind)
    missing_source_rc_unblind = json.loads(json.dumps(present["unblind"]))
    del missing_source_rc_unblind["groups"][0]["source_work_receipt"][0]["source_rc"]
    missing_source_rc_adjudication = adjudicate(features, missing_source_rc_unblind)
    q_drift_unblind = json.loads(json.dumps(present["unblind"]))
    q_drift_unblind["groups"][0]["quantized_work"][0]["q_theta"] += 1
    q_drift_adjudication = adjudicate(features, q_drift_unblind)
    result = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_MODEL_SELF_TEST_V1",
        "model_passed": model["model_passed"],
        "signal_present_status": adjudication["status"],
        "signal_absent_status": absent_adjudication["status"],
        "smuggled_label_rejected": True,
        "pre_freeze_work_leak_rejected": leaked_raw_rejected,
        "raw_manifest_drift_rejected": raw_manifest_drift_rejected,
        "unequal_work_rejected": not unequal_adjudication["acceptance"]["source_work_receipt_matches_frozen_law"],
        "missing_source_rc_rejected": not missing_source_rc_adjudication["acceptance"]["source_work_receipt_matches_frozen_law"],
        "q_theta_drift_rejected": not q_drift_adjudication["acceptance"]["source_work_receipt_matches_frozen_law"],
        "features_hash_before_unblind": frozen_hash,
    }
    result["self_test_passed"] = (
        result["signal_present_status"] == "ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE"
        and result["signal_absent_status"] == "ORBITSTATE_PHYSICAL_QUERY_COUPLING_NOT_ESTABLISHED"
        and result["smuggled_label_rejected"]
        and result["pre_freeze_work_leak_rejected"]
        and result["raw_manifest_drift_rejected"]
        and result["unequal_work_rejected"]
        and result["missing_source_rc_rejected"]
        and result["q_theta_drift_rejected"]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mock-runtime", choices=("signal-present", "signal-absent"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        result = self_test()
    elif args.mock_runtime:
        capture = build_mock_capture(signal_present=args.mock_runtime == "signal-present")
        features = extract_features(capture["manifest"], capture["raw_records"])
        result = {
            "schema_id": "CAT_CAS_ORBIT_QUERY_MOCK_RUNTIME_V1",
            "mode": args.mock_runtime,
            "features": features,
            "adjudication": adjudicate(features, capture["unblind"]),
        }
    else:
        result = model_summary()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result.get("self_test_passed", result.get("model_passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
