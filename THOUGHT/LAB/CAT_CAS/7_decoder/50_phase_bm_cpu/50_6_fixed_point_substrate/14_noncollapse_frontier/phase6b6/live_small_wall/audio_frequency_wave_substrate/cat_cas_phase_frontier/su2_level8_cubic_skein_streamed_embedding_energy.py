#!/usr/bin/env python3
"""M222 cell-streamed exact embedding energy for the M221 carrier.

M221 removed raw and candidate vector materialization, but it still formed an
aggregate residual norm and a candidate cyclotomic norm at every line-search
point.  M222 uses exact trace linearity to accumulate the same all-embedding
energy one residual cell at a time.  No aggregate residual, actual, or
candidate norm field is materialized.

The accepted recurrence remains equally available to compact classical exact
software.  This module tests a measured material-state obstruction; it does
not assert a distinct phase resource.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import su2_level8_cubic_skein_ledger_native_gauge as m221


sys.set_int_max_str_digits(0)
braid = m221.braid
UNITS = m221.UNITS
UNIT_RANK = m221.UNIT_RANK
CASES = m221.CASES
PRIMARY = m221.PRIMARY
REUSE = m221.REUSE
MAX_BRACKET_DOUBLINGS = m221.MAX_BRACKET_DOUBLINGS


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Work(m221.Work):
    streamed_embedding_energy_evaluations: int = 0
    streamed_embedding_cells_scanned: int = 0
    streamed_embedding_norm_field_multiplications: int = 0
    streamed_embedding_weight_field_multiplications: int = 0
    streamed_embedding_trace_terms: int = 0
    aggregate_residual_norm_fields_materialized: int = 0
    aggregate_actual_norm_fields_materialized: int = 0
    aggregate_candidate_norm_fields_materialized: int = 0


def streamed_embedding_energy(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    weight: braid.K,
    work: Work,
    *,
    live_integers: tuple[int, ...] = (),
    context_prefix: str,
) -> int:
    """Return Trace(weight * sum(v*conj(v))) without forming the sum."""

    work.exact_trace_energy_evaluations += 1
    work.streamed_embedding_energy_evaluations += 1
    accumulator = 0
    for index, value in enumerate(residual):
        work.streamed_embedding_cells_scanned += 1
        conjugated = m221.m220.conjugate(value)
        norm_cell = value * conjugated
        work.norm_field_multiplications += 1
        work.streamed_embedding_norm_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(weight, conjugated, norm_cell),
            integers=live_integers + (index, accumulator),
            context=f"{context_prefix}_CELL_NORM",
        )
        weighted = norm_cell if weight == braid.ONE else weight * norm_cell
        if weight != braid.ONE:
            work.unit_norm_field_multiplications += 1
            work.streamed_embedding_weight_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(weight, norm_cell, weighted),
            integers=live_integers + (index, accumulator),
            context=f"{context_prefix}_WEIGHTED_CELL",
        )
        term = m221.m220.field_trace(weighted)
        updated = accumulator + term
        work.streamed_embedding_trace_terms += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(weight, weighted),
            integers=live_integers + (index, accumulator, term, updated),
            context=f"{context_prefix}_INTEGER_ACCUMULATE",
        )
        accumulator = updated
    if accumulator < 0:
        raise RuntimeError("streamed exact embedding energy became negative")
    return accumulator


def line_minimum(
    scale_norm: braid.K,
    direction: m221.m220.UnitDirection,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
) -> tuple[int, int]:
    """Find the exact line minimum with no energy cache or aggregate norm."""

    work.line_searches += 1

    def energy(exponent: int, integers: tuple[int, ...]) -> int:
        factor = m221.norm_factor(
            direction, exponent, residual, ledger, scratch, work
        )
        combined_weight = factor * scale_norm
        work.unit_norm_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(scale_norm, factor, combined_weight),
            integers=integers + (exponent,),
            context="STREAMED_TRACE_LINE_COMBINED_WEIGHT",
        )
        return streamed_embedding_energy(
            residual,
            ledger,
            scratch,
            combined_weight,
            work,
            live_integers=integers + (exponent,),
            context_prefix="STREAMED_TRACE_LINE",
        )

    zero = energy(0, (0,))
    positive = energy(1, (0, 1, -1, zero))
    negative = energy(-1, (0, 1, -1, zero, positive))
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale_norm,),
        integers=(zero, positive, negative),
        context="STREAMED_TRACE_INITIAL_DIRECTION_ENERGIES",
    )
    if min(positive, negative) >= zero:
        return 0, zero
    direction_sign = 1 if positive < negative else -1
    previous = 0
    current = direction_sign
    current_energy = positive if direction_sign > 0 else negative
    del zero, positive, negative
    for _ in range(MAX_BRACKET_DOUBLINGS):
        following = 2 * current
        work.line_bracket_doublings += 1
        following_energy = energy(
            following, (previous, current, following, current_energy)
        )
        if following_energy >= current_energy:
            low, high = sorted((previous, following))
            break
        previous, current, current_energy = current, following, following_energy
    else:
        raise RuntimeError("unit line minimum was not bracketed")
    while high - low > 8:
        work.line_ternary_steps += 1
        first = low + (high - low) // 3
        second = high - (high - low) // 3
        first_energy = energy(first, (low, high, first, second))
        second_energy = energy(second, (low, high, first, second, first_energy))
        if first_energy <= second_energy:
            high = second - 1
        else:
            low = first + 1
    selected = low
    selected_energy = energy(low, (low, high))
    for exponent in range(low + 1, high + 1):
        candidate_energy = energy(
            exponent, (low, high, selected, selected_energy)
        )
        if (candidate_energy, exponent) < (selected_energy, selected):
            selected, selected_energy = exponent, candidate_energy
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale_norm,),
        integers=(low, high, selected, selected_energy),
        context="STREAMED_TRACE_LINE_SELECTED_ENERGY",
    )
    return selected, selected_energy


def balance_resident(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    scale: braid.K,
) -> dict[str, Any]:
    work.balance_calls += 1
    scale_conjugate = m221.m220.conjugate(scale)
    scale_norm = scale * scale_conjugate
    work.norm_field_multiplications += 1
    work.observe(
        residual,
        ledger,
        scratch,
        scalars=(scale, scale_conjugate, scale_norm),
        context="STREAMED_ENERGY_LEDGER_SCALE_NORM",
    )
    zero_ledger = [0] * UNIT_RANK
    raw_payload = m221.stream_scaled_payload(
        residual,
        scale,
        residual,
        ledger,
        scratch,
        work,
        candidate=False,
        live_scalars=(scale_norm,),
    )
    identity_total = raw_payload + m221.m220.ledger_payload_bits(zero_ledger)
    identity_energy = streamed_embedding_energy(
        residual,
        ledger,
        scratch,
        scale_norm,
        work,
        context_prefix="STREAMED_IDENTITY_ENERGY",
    )
    best_key = (
        identity_total,
        raw_payload,
        identity_energy,
        tuple(zero_ledger),
    )
    best_multiplier = scale
    best_ledger = zero_ledger
    candidate_exponents: list[int] = []
    for index, direction in enumerate(UNITS):
        exponent, energy = line_minimum(
            scale_norm, direction, residual, ledger, scratch, work
        )
        candidate_exponents.append(exponent)
        if exponent == 0:
            continue
        factor = m221.residual_factor(
            direction, exponent, residual, ledger, scratch, work
        )
        multiplier = factor * scale
        work.ledger_scale_field_multiplications += 1
        work.observe(
            residual,
            ledger,
            scratch,
            scalars=(scale_norm, best_multiplier, scale, factor, multiplier),
            integers=(index, exponent, energy),
            context="STREAMED_ENERGY_CANDIDATE_NET_UNIT_MULTIPLIER",
        )
        candidate_ledger = [0] * UNIT_RANK
        candidate_ledger[index] = exponent
        candidate_payload = m221.stream_scaled_payload(
            residual,
            multiplier,
            residual,
            ledger,
            scratch,
            work,
            candidate=True,
            live_scalars=(scale_norm, best_multiplier),
            live_integers=(index, exponent, energy),
        )
        key = (
            candidate_payload + m221.m220.ledger_payload_bits(candidate_ledger),
            candidate_payload,
            energy,
            tuple(candidate_ledger),
        )
        if key < best_key:
            best_key = key
            best_multiplier = multiplier
            best_ledger = candidate_ledger
    m221.apply_selected_net(
        residual,
        ledger,
        scratch,
        best_multiplier,
        work,
        live_scalars=(),
    )
    ledger[:] = best_ledger
    return {
        "raw_payload_bits": raw_payload,
        "balanced_residual_payload_bits": braid.field_payload_bits(residual),
        "unit_ledger_payload_bits": m221.m220.ledger_payload_bits(ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            identity_total - best_key[0]
        ),
        "selected_unit_ledger": list(ledger),
        "per_direction_trace_energy_minimizing_exponents": candidate_exponents,
        "selected_exact_embedding_energy_bits": m221.signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(ledger),
    }


def install_streamed_energy_path() -> None:
    m221.Work = Work
    m221.balance_resident = balance_resident


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_cubic_skein_streamed_embedding_energy.py "
            "SEPARATE_REFERENCE_JSON"
        )
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M222 reference is forbidden")
    reference = json.loads(reference_path.read_text())
    expected_schema = (
        "cat_cas.su2_level8_cubic_skein_streamed_embedding_energy_reference.v1"
    )
    if reference.get("schema") != expected_schema:
        raise RuntimeError("M222 separate-reference schema changed")
    install_streamed_energy_path()
    cases = [m221.execute_case(*case) for case in CASES]
    if [m221.reference_case_view(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M222 independent case and resource parity failed")
    reuse = m221.reuse_result()
    for section in ("primary", "reuse", "fresh_reuse"):
        for key in (
            "boundary_commitment",
            "forward_state_commitment",
            "forward_raw_payload_bits",
            "final_balance",
            "restoration_error_field_cells",
            "canonical_post_restoration_state_exact",
            "declared_live_payload_reduction_vs_raw_bits",
        ):
            if reuse[section][key] != reference["reuse"][section][key]:
                raise RuntimeError(
                    f"M222 independent reuse parity failed: {section}.{key}"
                )
    all_controls = m221.controls()
    positive_controls = {
        key: value
        for key, value in all_controls.items()
        if key
        not in {
            "raw_actual_vector_materialized",
            "candidate_residual_vector_materialized",
            "intermediate_actual_vector_projected",
            "snapshot_command_available",
        }
    }
    if (
        not all(positive_controls.values())
        or all_controls["raw_actual_vector_materialized"]
        or all_controls["candidate_residual_vector_materialized"]
        or all_controls["intermediate_actual_vector_projected"]
        or all_controls["snapshot_command_available"]
    ):
        raise RuntimeError("M222 control failed")
    selected = [
        {
            "strands": case["strands"],
            "rounds": case["rounds"],
            "forward_raw_payload_bits": case["forward_raw_payload_bits"],
            "balanced_residual_plus_ledger_payload_bits": case["final_balance"][
                "balanced_residual_plus_ledger_payload_bits"
            ],
            "streamed_energy_maximum_declared_live_payload_bits": case["work"][
                "maximum_declared_live_payload_bits"
            ],
            "raw_maximum_declared_live_payload_bits": case[
                "matched_raw_recurrence"
            ]["maximum_declared_live_payload_bits"],
            "declared_live_payload_reduction_vs_raw_bits": case[
                "declared_live_payload_reduction_vs_raw_bits"
            ],
            "streamed_energy_maximum_context": case["work"][
                "maximum_declared_live_context"
            ],
            "raw_maximum_context": case["matched_raw_recurrence"][
                "maximum_declared_live_context"
            ],
        }
        for case in cases
        if (case["strands"], case["rounds"]) in ((4, 4), (6, 2), (8, 1))
    ]
    all_nontrivial_live_smaller = all(
        case["declared_live_payload_reduction_vs_raw_bits"] > 0
        for case in cases
        if case["rounds"] > 1
    )
    every_declared_case_above_raw = all(
        case["declared_live_payload_reduction_vs_raw_bits"] < 0 for case in cases
    )
    dominant_contexts = sorted(
        {case["work"]["maximum_declared_live_context"] for case in cases}
    )
    if all_nontrivial_live_smaller:
        result_name = "PASS_BOUNDED_EXACT_STREAMED_EMBEDDING_ENERGY_LIFECYCLE_REDUCTION"
        claim = (
            "BOUNDED_EXACT_STREAMED_ALL_EMBEDDING_UNIT_ENERGY_FROM_LEDGER_NATIVE_"
            "TOPOLOGY_LOCAL_CUBIC_SKEIN_ELIMINATES_AGGREGATE_CYCLOTOMIC_NORM_"
            "FIELDS_AND_REDUCES_DECLARED_EXACT_LIVE_PAYLOAD_BELOW_THE_MATCHED_RAW_"
            "RECURRENCE_ON_EVERY_DECLARED_DEPTH_ABOVE_ONE_WITH_FINAL_ONLY_BOUNDARY_"
            "EXACT_RESTORATION_REUSE_BUT_THE_IDENTICAL_CLASSICAL_RECURRENCE_REMAINS"
        )
    elif every_declared_case_above_raw:
        result_name = "PASS_BOUNDED_EXACT_STREAMED_EMBEDDING_ENERGY_PERSISTING_HEIGHT_NO_GO"
        claim = (
            "BOUNDED_EXACT_STREAMED_ALL_EMBEDDING_UNIT_ENERGY_FROM_LEDGER_NATIVE_"
            "TOPOLOGY_LOCAL_CUBIC_SKEIN_ELIMINATES_AGGREGATE_CYCLOTOMIC_NORM_"
            "FIELD_MATERIALIZATION_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_"
            "REUSE_BUT_DECLARED_EXACT_LIVE_PAYLOAD_REMAINS_ABOVE_THE_MATCHED_"
            "RAW_RECURRENCE_ON_EVERY_DECLARED_CASE_BECAUSE_ONE_STREAMED_WEIGHTED_"
            "CELL_FIELD_PRODUCT_DOMINATES_AND_THE_IDENTICAL_CLASSICAL_STREAMED_"
            "ENERGY_RECURRENCE_REMAINS"
        )
    else:
        result_name = "PASS_BOUNDED_EXACT_STREAMED_EMBEDDING_ENERGY_MIXED_LIFECYCLE_RESULT"
        claim = (
            "BOUNDED_EXACT_STREAMED_ALL_EMBEDDING_UNIT_ENERGY_FROM_LEDGER_NATIVE_"
            "TOPOLOGY_LOCAL_CUBIC_SKEIN_ELIMINATES_AGGREGATE_CYCLOTOMIC_NORM_"
            "FIELD_MATERIALIZATION_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_"
            "REUSE_AND_A_MIXED_DECLARED_LIVE_PAYLOAD_RESULT_AGAINST_THE_MATCHED_"
            "RAW_RECURRENCE_WHILE_THE_IDENTICAL_CLASSICAL_STREAMED_ENERGY_"
            "RECURRENCE_REMAINS"
        )
    here = Path(__file__).resolve().parent
    result = {
        "schema": "cat_cas.su2_level8_cubic_skein_streamed_embedding_energy.v1",
        "result": result_name,
        "claim": claim,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PRETRUNCATION_QZETA40_M219_CUBIC_SKEIN_M220_UNIT_M221_LEDGER_NATIVE_PARAMETERS3_7_9_11_13_17_19_CELL_STREAMED_EXACT_TRACE_LINE_SEARCH_FAMILY0_STRANDS4_DEPTH1TO4_STRANDS6_DEPTH1TO2_STRANDS8_DEPTH1_PRIMARY4_DEPTH4_REUSE4_DEPTH2_FAMILY1_DIRECT_PROCESS_ONLY",
        "mechanism": {
            "represented_state": "ACTUAL_EQUALS_GLOBAL_CYCLOTOMIC_UNIT_LEDGER_SCALE_TIMES_RESIDUAL",
            "energy_identity": "TRACE_LINEARITY_OVER_SUM_OF_WEIGHTED_RESIDUAL_CELL_HERMITIAN_PRODUCTS",
            "aggregate_residual_norm_field_materialized": False,
            "aggregate_actual_norm_field_materialized": False,
            "aggregate_candidate_norm_field_materialized": False,
            "raw_actual_vectors_materialized": False,
            "candidate_residual_vectors_materialized": False,
            "retained_inverse_value_history": 0,
        },
        "cases": cases,
        "selected_cases": selected,
        "lifecycle_law": {
            "all_declared_depth_above_one_smaller_than_matched_raw": all_nontrivial_live_smaller,
            "every_declared_case_above_matched_raw": every_declared_case_above_raw,
            "dominant_streamed_energy_contexts": dominant_contexts,
            "aggregate_norm_field_materialization_eliminated": all(
                case["work"]["aggregate_residual_norm_fields_materialized"] == 0
                and case["work"]["aggregate_actual_norm_fields_materialized"] == 0
                and case["work"]["aggregate_candidate_norm_fields_materialized"] == 0
                for case in cases
            ),
            "raw_and_candidate_vector_materialization_eliminated": all(
                case["work"]["raw_actual_vectors_materialized"] == 0
                and case["work"]["candidate_residual_vectors_materialized"] == 0
                for case in cases
            ),
            "raw_baseline_uses_same_exact_field_and_cubic_skein_recurrence": True,
            "logical_exact_live_intervals_not_process_rss": True,
        },
        "separate_reference": {
            "imports_m222_production": reference.get("imports_m222_production"),
            "imports_m221_production": reference.get("imports_m221_production"),
            "uses_prior_standalone_m221_reference_substrate": reference.get(
                "uses_prior_standalone_m221_reference_substrate"
            ),
            "case_state_boundary_balance_resource_restoration_parity": True,
            "reuse_parity": True,
        },
        "reuse": reuse,
        "controls": all_controls,
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_LEDGER_NATIVE_CELL_STREAMED_EXACT_EMBEDDING_ENERGY_RECURRENCE",
            "matched_raw": "IDENTICAL_RAW_LINK_PATTERN_CUBIC_SKEIN_RECURRENCE_WITH_THE_SAME_DECLARED_SCALAR_TEMPORARY_LAW",
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "resource_law": {
            "carrier": "LINK_PATTERN_RESIDUAL_PLUS_SEVEN_SIGNED_UNIT_EXPONENTS_PLUS_EQUAL_SKEIN_SCRATCH",
            "streamed_energy_cell_products_and_integer_accumulators_counted": True,
            "unit_power_combined_weight_mutation_inverse_and_projection_work_counted": True,
            "line_search_retains_no_energy_cache": True,
            "whole_process_and_python_object_overhead_bounded": False,
            "excluded_not_zero": "PYTHON_CONTAINER_CAPACITY_ALLOCATOR_INTERPRETER_PROCESS_IMAGE_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_RSS",
        },
        "claim_limits": {
            "global_unit_lattice_optimum": False,
            "asymptotic_height_bound": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m221_production_sha256": sha256_file(
                here / "su2_level8_cubic_skein_ledger_native_gauge.py"
            ),
            "m222_production_sha256": sha256_file(Path(__file__).resolve()),
            "m222_separate_reference_code_sha256": sha256_file(
                here / "su2_level8_cubic_skein_streamed_embedding_energy_separate_reference.py"
            ),
            "m222_separate_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
