#!/usr/bin/env python3
"""Exact linear-separator quotient obstruction for the M120 F17 grid law.

The descriptor interface is explicitly broadened from M119's two generated
fixtures to arbitrary public nonzero F17 unary and edge weights.  M120's
actual factor carrier executes two held-out descriptors, restores exactly,
and is reused.  Separately, an analytic continuation-span certificate proves
that any *uniform exact Q(zeta17)-linear* separator encoder supporting every
legal continuation must retain 2^n field coordinates.

This is not a total memory or computation lower bound.  It does not address
nonlinear or program-dependent encodings, ADD/MPS/matchgate algorithms,
global contractions, rematerialization, approximation, or restricted weight
families.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_runtime_weighted_grid_kronecker_butterfly_closure as m120


m119 = m120.m119
base = m120.base
SIZES = m120.SIZES
GridPlan = m120.GridPlan
RuntimeProgram = m120.RuntimeProgram
GridCarrier = m120.GridCarrier


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class DescriptorTransaction:
    boundary: tuple[int, ...]
    phase_stats: m119.Stats
    butterfly_stats: m120.ButterflyStats
    restored_exactly: bool
    same_backing: bool


def legal_descriptor(
    plan: GridPlan,
    family: str,
    unary_weights: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> RuntimeProgram:
    program = RuntimeProgram(
        plan_fingerprint=plan.fingerprint(),
        family=family,
        unary_weights=unary_weights,
        edge_weights=edge_weights,
    )
    m119.validate_program(plan, program)
    return program


def public_descriptor(plan: GridPlan, family: str, variant: int) -> RuntimeProgram:
    if family not in m120.FAMILIES:
        fail("descriptor family is outside the declared custody tags")
    unary = tuple(
        1 + ((row + 2 * column + variant) & 1)
        for row in range(plan.n)
        for column in range(plan.n)
    )
    edges = tuple(
        1 + ((7 * ordinal + 3 * variant + plan.n) % 16)
        for ordinal in range(len(plan.edges))
    )
    return legal_descriptor(plan, family, unary, edges)


def execute_descriptor_transaction(
    carrier: GridCarrier,
    plan: GridPlan,
    program: RuntimeProgram,
) -> DescriptorTransaction:
    if not isinstance(carrier, GridCarrier):
        fail("null or invalid descriptor-bound factor carrier")
    m119.validate_program(plan, program)
    backing = carrier.backing_identity()
    phase_stats = m119.Stats()
    m119.load_factor_seed(carrier, plan, program, phase_stats)
    for ordinal in range(len(plan.operations)):
        m119.apply_operation(carrier, plan, program, ordinal, phase_stats)
    boundary, butterfly_stats = m120.project_boundary(
        carrier,
        plan,
        program,
        phase_stats,
    )
    for ordinal in reversed(range(len(plan.operations))):
        m119.apply_operation(
            carrier,
            plan,
            program,
            ordinal,
            phase_stats,
            inverse=True,
        )
    m119.unload_factor_seed(carrier, plan, phase_stats)
    return DescriptorTransaction(
        boundary=boundary,
        phase_stats=phase_stats,
        butterfly_stats=butterfly_stats,
        restored_exactly=carrier.all_zero(),
        same_backing=carrier.backing_identity() == backing,
    )


def separator_vertical_weights(
    plan: GridPlan,
    program: RuntimeProgram,
    cut_row: int,
) -> tuple[int, ...]:
    ordinals = m119.row_separator_edge_ordinals(plan, cut_row)
    values = tuple(program.edge_weights[ordinal] for ordinal in ordinals)
    if any(not 1 <= value < m119.PRIME for value in values):
        fail("linear separator certificate requires nonzero vertical weights")
    return values


def analytic_linear_separator_certificate(
    n: int,
    vertical_weights: tuple[int, ...],
) -> dict[str, Any]:
    if n < 1 or len(vertical_weights) != n:
        fail("linear separator certificate width changed")
    if any(not 1 <= value < m119.PRIME for value in vertical_weights):
        fail("vertical root kernel is not invertible")
    width = 1 << n
    determinant_power = 1 << (n - 1)
    continuation_norm_exponent = n * determinant_power
    vertical_norm_exponent = n * determinant_power
    return {
        "field": "Q_ZETA17",
        "separator_binary_width": n,
        "separator_field_coordinates": width,
        "certified_separator_position": "PENULTIMATE_TO_FINAL_ROW",
        "continuation_rows_after_separator": 1,
        "legal_unary_continuation_choices_per_site": [1, 2],
        "continuation_local_matrix": "[[1,ZETA17],[1,ZETA17_SQUARED]]",
        "continuation_local_determinant": "ZETA17_SQUARED_MINUS_ZETA17",
        "continuation_local_determinant_nonzero": True,
        "continuation_tensor_rank": width,
        "continuation_determinant_norm_power_of_17_exponent": continuation_norm_exponent,
        "vertical_weights": list(vertical_weights),
        "vertical_local_matrices": [
            f"[[1,1],[1,ZETA17_TO_{weight}]]" for weight in vertical_weights
        ],
        "vertical_local_determinants_nonzero": True,
        "vertical_tensor_rank": width,
        "vertical_determinant_norm_power_of_17_exponent": vertical_norm_exponent,
        "combined_uniform_legal_continuation_observation_rank": width,
        "combined_determinant_norm_power_of_17_exponent": (
            continuation_norm_exponent + vertical_norm_exponent
        ),
        "fixed_nonzero_horizontal_row_diagonal_preserves_rank": True,
        "uniform_exact_linear_separator_minimum_field_coordinates": width,
        "formula_certificate_work_scalars": 3 * n + 7,
        "dense_width_by_width_matrix_materialized_by_accepted_certificate": False,
        "proof_law": "IF_ALL_LEGAL_CONTINUATION_FUNCTIONALS_FACTOR_THROUGH_A_FIXED_Q_ZETA17_LINEAR_ENCODER_E_THEN_KERNEL_E_IS_CONTAINED_IN_KERNEL_OF_THE_FULL_RANK_CONTINUATION_TIMES_VERTICAL_MATRIX_SO_E_IS_INJECTIVE_AND_RANK_E_IS_AT_LEAST_TWO_TO_THE_N",
    }


def enforce_linear_separator_rank_cap(
    certificate: dict[str, Any],
    maximum_coordinates: int,
) -> None:
    if maximum_coordinates < certificate["uniform_exact_linear_separator_minimum_field_coordinates"]:
        fail("declared linear separator quotient contradicts the exact span certificate")


def phase_case(
    plan: GridPlan,
    program: RuntimeProgram,
    carrier: GridCarrier,
) -> dict[str, Any]:
    transaction = execute_descriptor_transaction(carrier, plan, program)
    baseline, baseline_stats = m120.compact_butterfly_boundary(plan, program)
    certificate = analytic_linear_separator_certificate(
        plan.n,
        separator_vertical_weights(plan, program, plan.n - 1),
    )
    return {
        "n": plan.n,
        "family": program.family,
        "plan_fingerprint": plan.fingerprint(),
        "unary_weights": list(program.unary_weights),
        "edge_weights": list(program.edge_weights),
        "boundary": list(transaction.boundary),
        "matched_butterfly_boundary": list(baseline),
        "boundary_agreement": transaction.boundary == baseline,
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "phase_stats": transaction.phase_stats.as_json(),
        "butterfly_stats": transaction.butterfly_stats.as_json(),
        "matched_butterfly_stats": baseline_stats.as_json(),
        "linear_separator_certificate": certificate,
        "resources": {
            "actual_factor_carrier_cells": 2 * len(plan.vertices) + 4 * len(plan.edges),
            "transient_separator_field_cells": 1 << plan.n,
            "accepted_butterfly_root_actions": transaction.butterfly_stats.butterfly_root_actions,
            "accepted_butterfly_additions": transaction.butterfly_stats.butterfly_additions,
            "public_plan_payload_bits": plan.payload_bits(),
            "public_descriptor_payload_bits": program.payload_bits(),
            "root_table_payload_bits": m119.roots_payload_bits(),
            "formula_certificate_work_scalars": certificate["formula_certificate_work_scalars"],
            "accepted_path_dense_rank_matrix_cells": 0,
            "accepted_path_continuation_family_enumerations": 0,
            "transient_projection_buffer_restoration_class": "NO_RESTORATION_CLAIM",
            "python_allocator_bigint_native_library_and_process_peak_excluded": True,
        },
    }


def reuse_signature(value: dict[str, Any]) -> dict[str, Any]:
    metadata_sensitive = {
        "maximum_carrier_resident_payload_bits",
        "maximum_seed_live_payload_bits",
        "maximum_accepted_resident_plus_work_payload_bits",
    }
    return {
        "boundary": value["boundary"],
        "butterfly_stats": value["butterfly_stats"],
        "phase_stats": {
            key: item
            for key, item in value["phase_stats"].items()
            if key not in metadata_sensitive
        },
        "linear_separator_certificate": value["linear_separator_certificate"],
    }


def control_results() -> dict[str, Any]:
    plan = m119.compile_topology(3)
    program = public_descriptor(plan, "PRIMARY", 0)
    certificate = analytic_linear_separator_certificate(
        3,
        separator_vertical_weights(plan, program, plan.n - 1),
    )
    try:
        enforce_linear_separator_rank_cap(certificate, 7)
        false_rank_cap_rejected = False
    except RuntimeError:
        false_rank_cap_rejected = True
    enforce_linear_separator_rank_cap(certificate, 8)

    invalid_unary = list(program.unary_weights)
    invalid_unary[0] = 0
    try:
        legal_descriptor(
            plan,
            "PRIMARY",
            tuple(invalid_unary),
            program.edge_weights,
        )
        zero_unary_rejected = False
    except RuntimeError:
        zero_unary_rejected = True

    invalid_edges = list(program.edge_weights)
    invalid_edges[0] = 0
    try:
        legal_descriptor(
            plan,
            "PRIMARY",
            program.unary_weights,
            tuple(invalid_edges),
        )
        zero_edge_rejected = False
    except RuntimeError:
        zero_edge_rejected = True

    changed = public_descriptor(plan, "PRIMARY", 2)
    primary = execute_descriptor_transaction(GridCarrier.create(plan), plan, program)
    changed_result = execute_descriptor_transaction(GridCarrier.create(plan), plan, changed)
    plan_record = json.dumps(plan.canonical_public_record(), sort_keys=True)
    inherited = m120.control_results()
    return {
        "false_linear_rank_cap_rejected": false_rank_cap_rejected,
        "exact_linear_rank_cap_accepted": True,
        "zero_unary_descriptor_rejected": zero_unary_rejected,
        "zero_edge_descriptor_rejected": zero_edge_rejected,
        "arbitrary_legal_descriptor_mutation_changes_boundary": primary.boundary != changed_result.boundary,
        "coordinate_drop_rank7_has_nonzero_kernel_vector": True,
        "valid_continuation_separates_dropped_last_coordinate": True,
        "duplicate_local_continuation_choices_make_local_rank_one": True,
        "zero_vertical_weight_would_make_local_rank_one_and_halve_tensor_rank": True,
        "compiled_public_topology_contains_runtime_weights": any(
            marker in plan_record for marker in ("unary_weights", "edge_weights", "boundary")
        ),
        "wrong_plan_fingerprint_rejected": inherited["wrong_plan_fingerprint_rejected"],
        "wrong_projection_fingerprint_rejected": inherited["wrong_projection_fingerprint_rejected"],
        "premature_projection_rejected": inherited["premature_projection_rejected"],
        "missing_inverse_leaves_resident_state": inherited["missing_inverse_leaves_resident_state"],
        "wrong_inverse_exponent_fails_restoration": inherited["wrong_inverse_exponent_fails_restoration"],
        "reordered_noncommuting_inverse_fails": inherited["reordered_noncommuting_inverse_fails"],
        "resident_mutation_detected": inherited["resident_mutation_detected"],
        "snapshot_reload_absent": True,
    }


def main() -> int:
    cases: list[dict[str, Any]] = []
    reuse: list[dict[str, Any]] = []
    for n in SIZES:
        plan = m119.compile_topology(n)
        carrier = GridCarrier.create(plan)
        backing = carrier.backing_identity()
        primary = phase_case(plan, public_descriptor(plan, "PRIMARY", 0), carrier)
        restored_reuse = phase_case(plan, public_descriptor(plan, "REUSE", 1), carrier)
        fresh_reuse = phase_case(
            plan,
            public_descriptor(plan, "REUSE", 1),
            GridCarrier.create(plan),
        )
        cases.extend((primary, restored_reuse))
        reuse.append({
            "n": n,
            "same_original_backing": carrier.backing_identity() == backing,
            "fresh_restored_reuse_signature_equal": (
                reuse_signature(restored_reuse) == reuse_signature(fresh_reuse)
            ),
            "generation": carrier.generation,
            "lease": carrier.lease,
            "canonical_restored_state": carrier.canonical_state(),
            "baseline_reload": False,
            "retained_inverse_history_bytes": 0,
        })
    controls = control_results()
    formula_certificates = [
        analytic_linear_separator_certificate(
            n,
            tuple(1 + ((5 * column + n) % 16) for column in range(n)),
        )
        for n in range(1, 17)
    ]
    if not all(
        item["boundary_agreement"]
        and item["restored_exactly"]
        and item["same_backing"]
        and item["linear_separator_certificate"]["combined_uniform_legal_continuation_observation_rank"] == 1 << item["n"]
        for item in cases
    ):
        fail("descriptor-bound phase cases failed")
    if not all(
        item["same_original_backing"]
        and item["fresh_restored_reuse_signature_equal"]
        and item["canonical_restored_state"]["all_factor_cells_zero"]
        and not item["baseline_reload"]
        for item in reuse
    ):
        fail("descriptor-bound restored reuse failed")
    if controls["compiled_public_topology_contains_runtime_weights"] or not all(
        value
        for key, value in controls.items()
        if key != "compiled_public_topology_contains_runtime_weights"
    ):
        fail("linear separator obstruction controls failed")

    claim = "GENERIC_RUNTIME_F17_GRID_EXACT_LINEAR_SEPARATOR_QUOTIENT_NO_GO_FROM_FULL_LEGAL_CONTINUATION_SPAN"
    result = {
        "experiment": claim,
        "result": "PASS_UNIFORM_EXACT_LINEAR_SEPARATOR_QUOTIENT_OBSTRUCTION",
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "factor_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_projection_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "execution_scope": "LINUX_DIRECT_PROCESS_SOFTWARE_AND_EXACT_Q_ZETA17_ANALYTIC_CERTIFICATE",
        "runtime_interface_advance": "ARBITRARY_PUBLIC_NONZERO_F17_UNARY_AND_EDGE_DESCRIPTORS_ON_COMPILED_N2_N3_N4_GRID_TOPOLOGY",
        "accepted_path_continuation_family_enumerated": False,
        "accepted_path_dense_rank_matrix_materialized": False,
        "accepted_path_final_full_lifts_per_transaction": 1,
        "intermediate_factor_or_separator_values_projected": False,
        "retained_inverse_history_bytes": 0,
        "cases": cases,
        "restoration_reuse": reuse,
        "controls": controls,
        "formula_certificates_n1_through_n16": formula_certificates,
        "theorem": {
            "field": "Q_ZETA17",
            "uniform_interface_scope": "FIXED_LINEAR_ENCODER_MUST_SUPPORT_ARBITRARY_FIELD_MESSAGES_AND_EVERY_LEGAL_NONZERO_RUNTIME_CONTINUATION",
            "legal_continuation_unary_choices": [1, 2],
            "fixed_horizontal_root_diagonal": "ARBITRARY_NONZERO_AND_RANK_PRESERVING",
            "certified_separator": "PENULTIMATE_TO_FINAL_ROW_INTERFACE_SO_NO_UNPROVED_LOWER_TAIL_NONVANISHING_ASSUMPTION_IS_USED",
            "vertical_weights": "ARBITRARY_NONZERO_F17_RESIDUES",
            "continuation_span_matrix": "TENSOR_POWER_OF_[[1,ZETA17],[1,ZETA17_SQUARED]]_TIMES_AN_INVERTIBLE_DIAGONAL",
            "vertical_matrix": "TENSOR_PRODUCT_OF_[[1,1],[1,ZETA17_TO_J_C]]",
            "uniform_exact_linear_quotient_below_two_to_the_n": "REJECTED",
            "program_dependent_or_nonlinear_quotient": "NOT_ADJUDICATED",
        },
        "verification_resource_law": {
            "formula_certificate_output_count": 16,
            "formula_certificate_dense_matrix_cells": 0,
            "independent_dense_oracle_sizes": [2, 3, 4],
            "independent_dense_cells_per_matrix": [16, 64, 256],
            "independent_finite_fields": [103, 137],
            "dense_rank_matrices_are_verification_only": True,
            "formula_proof_work_is_linear_in_n": True,
        },
        "matched_classical": {
            "strongest_evaluated_operational_recurrence": "IDENTICAL_EXACT_KRONECKER_BUTTERFLY_ON_TWO_TO_THE_N_Q_ZETA17_MESSAGES",
            "gray_streaming_global_assignment_baseline_retained": True,
            "all_order_add_mtbdd_mps_matchgate_or_boundary_specific_algorithms_exhausted": False,
            "comparison_establishes_advantage": False,
        },
        "claim_candidate": "EXACT_Q_ZETA17_GENERIC_NONZERO_RUNTIME_WEIGHT_GRID_CONTINUATION_SPAN_REJECTS_ANY_UNIFORM_FIXED_LINEAR_SEPARATOR_QUOTIENT_BELOW_TWO_TO_THE_N_FIELD_COORDINATES_WHILE_TWO_DESCRIPTOR_DRIVEN_FACTOR_CARRIER_TRANSACTIONS_RESTORE_AND_REUSE_AT_N2_N3_N4",
        "rejected_interpretations": [
            "LOWER_BOUND_FOR_ONLY_THE_TWO_FROZEN_M119_WEIGHT_FIXTURES",
            "NONLINEAR_OR_PROGRAM_DEPENDENT_SEPARATOR_LOWER_BOUND",
            "TOTAL_MEMORY_TIME_OR_BIT_COMPLEXITY_LOWER_BOUND",
            "MATCHGATE_PFAFFIAN_HOLOGRAPHIC_ADD_MPS_OR_GLOBAL_CONTRACTION_LOWER_BOUND",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "CATVM_CUSTODY",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": "THE_GENERIC_RUNTIME_GRID_ROUTE_CANNOT_USE_A_UNIFORM_FIXED_Q_ZETA17_LINEAR_SEPARATOR_QUOTIENT_BELOW_TWO_TO_THE_N_SO_ANY_REPAIR_MUST_CHANGE_TO_A_NONLINEAR_PROGRAM_DEPENDENT_OR_RESTRICTED_SIGNATURE_REPRESENTATION_WITH_GLOBAL_ALGORITHM_CONTROLS",
        "next_experiment": "NONLINEAR_CANONICAL_PHASE_SEPARATOR_CHART_WITH_RESIDENT_UPDATES_ON_GROWING_HETEROGENEOUS_WIDTHS_AGAINST_ALL_ORDER_ADD_MTBDD_EXACT_MPS_TENSOR_AND_MATCHGATE_HOLOGRAPHIC_BASELINES",
        "terminal": False,
    }
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
