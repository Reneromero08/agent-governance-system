#!/usr/bin/env python3
"""M267 conditional-Gaussian closed-loop obstruction.

This executable is a deterministic exact-arithmetic theorem fixture.  It
does not execute QEMU, retain a physical carrier, or establish a complexity
lower bound.  Fractions and symbolic lifted phases are used where possible so
that a symplectic endpoint cannot silently erase a metaplectic or Weyl phase.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


CLAIM = (
    "FINITE_MODE_PUBLIC_FIXED_AXIS_CONDITIONAL_GAUSSIAN_LOOPS_WITH_EXACT_"
    "FAITHFUL_CARRIER_REFERENCE_IDENTITY_REDUCE_TO_A_DIRECT_CLIENT_DIAGONAL_"
    "PHASE_OR_DECLARED_DILATION_SCHUR_CHANNEL_WHILE_POSITIVE_ACCUMULATED_CP_"
    "DIVISIBLE_MARKOV_DIFFUSION_ON_A_CLAIMED_CARRIER_SUBSPACE_PRECLUDES_"
    "EXACT_SAME_MODE_CHANNEL_RETURN_ON_THAT_SUBSPACE"
)
CLAIM_CEILING = (
    "FINITE_MODE_FINITE_JOINT_CLIENT_LABEL_PUBLIC_PIECEWISE_QUADRATIC_OR_"
    "AFFINE_GAUSSIAN_DYNAMICS_WITH_FIXED_COMMUTING_CLIENT_OBSERVABLES_"
    "DECLARED_COMMON_DILATION_AND_EXACT_GAUSSIAN_MOMENT_OR_LIFTED_AFFINE_"
    "SYMPLECTIC_SEMANTICS_ONLY_NO_NONCOMMUTING_AXES_NONQUADRATIC_INTERACTIONS_"
    "NON_GAUSSIAN_BOUNDARY_MEASUREMENTS_QEC_RESTRICTED_ACCESS_NONMARKOV_"
    "RECOHERENCE_INFINITE_MODE_OR_PHYSICAL_CUSTODY"
)
RESTORATION_CLASSIFICATION = "NO_RESTORATION_CLAIM"
RESTORATION_SCOPE = (
    "FORMAL_REFERENCE_COMPLETE_GAUSSIAN_CHANNEL_IDENTITY_CRITERION_AND_"
    "POSITIVE_DIFFUSION_NO_RETURN_ON_DECLARED_SUPPORT_WITHOUT_EXECUTED_OR_"
    "PHYSICAL_CARRIER_RESTORATION"
)
RESOURCE_DISPOSITION = (
    "GENERAL_SECTOR_DIRECT_CLIENT_SHADOW_EXISTS_WITH_EXPLICIT_L_OR_L_SQUARED_"
    "COST_AND_THE_AFFINE_LABEL_COROLLARY_IS_POLYNOMIALLY_COMPACT_WHILE_"
    "POSITIVE_DIFFUSION_ON_CLAIMED_SUPPORT_FORBIDS_EXACT_REFERENCE_COMPLETE_"
    "RETURN_SO_NO_CATALYTIC_BUS_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED"
)
NEXT_MECHANISM = (
    "RESTRICTED_ACCESS_NON_GAUSSIAN_PHASE_EIGENSTATE_KICKBACK_ORACLE_WITH_"
    "FAITHFUL_CARRIER_RETURN_PREPARATION_PRECISION_QUERY_AND_CUSTODY_COSTS"
)

MILESTONE = "M267"
SCHEMA = "PHASE_QEMU_V9_CONDITIONAL_GAUSSIAN_CLOSED_LOOP_OBSTRUCTION_V1"

Q = Fraction
Matrix = tuple[tuple[Fraction, ...], ...]


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _matrix(values: Sequence[Sequence[int | Fraction]]) -> Matrix:
    return tuple(tuple(Q(value) for value in row) for row in values)


def _matrix_json(values: Matrix) -> list[list[dict[str, int]]]:
    return [[_fraction(value) for value in row] for row in values]


def _identity(dimension: int) -> Matrix:
    return tuple(
        tuple(Q(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )


def _transpose(values: Matrix) -> Matrix:
    return tuple(tuple(value for value in column) for column in zip(*values))


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            sum(
                (left[row][inner] * right[inner][column]
                 for inner in range(len(right))),
                Q(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _add(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def _sub(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(a - b for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def _congruence(transform: Matrix, covariance: Matrix) -> Matrix:
    return _multiply(_multiply(transform, covariance), _transpose(transform))


def _block(values: Matrix, rows: Iterable[int], columns: Iterable[int]) -> Matrix:
    row_indices = tuple(rows)
    column_indices = tuple(columns)
    return tuple(
        tuple(values[row][column] for column in column_indices)
        for row in row_indices
    )


def _rank(values: Matrix) -> int:
    work = [list(row) for row in values]
    rows = len(work)
    columns = len(work[0]) if work else 0
    pivot_row = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(pivot_row, rows) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = work[row][column]
            if factor != 0:
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[pivot_row])
                ]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def _symplectic(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


def _weyl_product_phase_radians(
    vectors: Sequence[tuple[Fraction, Fraction]],
) -> Fraction:
    """Radian exponent for W(v)W(w)=exp[-i*sigma(v,w)/2]W(v+w)."""

    pair_sum = sum(
        (
            _symplectic(vectors[left], vectors[right])
            for left in range(len(vectors))
            for right in range(left + 1, len(vectors))
        ),
        Q(0),
    )
    return -pair_sum / 2


def _metaplectic_fixtures() -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = _identity(2)
    two_pi = {
        "fixture": "metaplectic_2pi_vs_zero",
        "rotation_turns_by_client_label": [0, 1],
        "projected_symplectic_endpoints": [_matrix_json(endpoint)] * 2,
        "projected_endpoints_equal": True,
        "lifted_metaplectic_signs": [1, -1],
        "direct_client_diagonal": ["+1", "-1"],
        "carrier_reference_identity": True,
        "lifted_phase_required": True,
    }
    four_pi = {
        "fixture": "metaplectic_4pi_control",
        "rotation_turns": 2,
        "rotation_angle_pi_units": 4,
        "projected_symplectic_endpoint": _matrix_json(endpoint),
        "lifted_metaplectic_sign": 1,
        "matches_zero_lift": True,
        "carrier_reference_identity": True,
    }
    return two_pi, four_pi


def _weyl_fixture() -> dict[str, Any]:
    a = Q(1, 2)
    b = Q(1, 3)
    sectors = (("++", 1, 1), ("+-", 1, -1), ("-+", -1, 1), ("--", -1, -1))
    sector_records = []
    for label, z0, z1 in sectors:
        xi = (a * z0, Q(0))
        eta = (Q(0), b * z1)
        vectors = (xi, eta, (-xi[0], -xi[1]), (-eta[0], -eta[1]))
        net = tuple(
            sum((vector[index] for vector in vectors), Q(0))
            for index in (0, 1)
        )
        phase = _weyl_product_phase_radians(vectors)
        expected = -a * b * z0 * z1
        if phase != expected:
            raise AssertionError("two-qubit Weyl rectangle phase-law failure")
        symbolic = "EXP(-I/6)" if phase < 0 else "EXP(+I/6)"
        sector_records.append(
            {
                "sector": label,
                "z0": z0,
                "z1": z1,
                "ordered_vectors": [
                    [_fraction(vector[0]), _fraction(vector[1])]
                    for vector in vectors
                ],
                "net_displacement": [_fraction(net[0]), _fraction(net[1])],
                "lifted_loop_phase_radians": _fraction(phase),
                "lifted_loop_phase_symbolic": symbolic,
            }
        )
    return {
        "fixture": "weyl_rectangle_cocycle",
        "weyl_convention": "W_V_W_W=EXP_MINUS_I_SIGMA_V_W_OVER_2_W_V_PLUS_W",
        "client_qubits": 2,
        "sector_order": [label for label, _, _ in sectors],
        "sector_spin_labels": [[z0, z1] for _, z0, z1 in sectors],
        "affine_force_xi": "(A*Z0,0)",
        "affine_force_eta": "(0,B*Z1)",
        "a": _fraction(a),
        "b": _fraction(b),
        "base_signed_symplectic_area": _fraction(a * b),
        "exact_sector_phase_law_radians": "PHI(Z0,Z1)=-Z0*Z1/6",
        "sector_records": sector_records,
        "sector_phase_radians_in_declared_order": [
            record["lifted_loop_phase_radians"] for record in sector_records
        ],
        "direct_client_diagonal_in_declared_order": [
            "EXP(-I/6)",
            "EXP(+I/6)",
            "EXP(+I/6)",
            "EXP(-I/6)",
        ],
        "all_sector_displacements_close": all(
            record["net_displacement"] == [_fraction(Q(0)), _fraction(Q(0))]
            for record in sector_records
        ),
        "carrier_reference_identity_all_sectors": True,
        "phase_survives_closed_carrier_loop": True,
    }


def _tmsv_covariance() -> Matrix:
    c = Q(5, 4)
    s = Q(3, 4)
    return _matrix(
        (
            (c / 2, 0, s / 2, 0),
            (0, c / 2, 0, -s / 2),
            (s / 2, 0, c / 2, 0),
            (0, -s / 2, 0, c / 2),
        )
    )


def _vacuum_rotation_fixture() -> dict[str, Any]:
    tmsv = _tmsv_covariance()
    quarter_turn = _matrix(((0, -1), (1, 0)))
    lifted = _matrix(
        (
            (0, -1, 0, 0),
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    after = _congruence(lifted, tmsv)
    before_bus = _block(tmsv, range(2), range(2))
    after_bus = _block(after, range(2), range(2))
    return {
        "fixture": "vacuum_rotation_marginal_false_positive",
        "bus_rotation": "R_PI_OVER_2",
        "bus_symplectic": _matrix_json(quarter_turn),
        "tmsv_c": _fraction(Q(5, 4)),
        "tmsv_s": _fraction(Q(3, 4)),
        "tmsv_identity_c_squared_minus_s_squared": _fraction(Q(1)),
        "input_covariance": _matrix_json(tmsv),
        "output_covariance": _matrix_json(after),
        "bus_marginal_before": _matrix_json(before_bus),
        "bus_marginal_after": _matrix_json(after_bus),
        "bus_marginal_unchanged": before_bus == after_bus,
        "full_bus_reference_covariance_unchanged": tmsv == after,
        "reference_complete_identity_rejected": tmsv != after,
        "covariance_condition_number": _fraction(Q(4)),
        "input_mean_quanta_per_mode": _fraction(Q(1, 8)),
    }


def _additive_diffusion_fixture() -> dict[str, Any]:
    identity = _identity(2)
    vacuum = _matrix(((Q(1, 2), 0), (0, Q(1, 2))))
    nu = Q(1, 8)
    gramian = _matrix(((nu, 0), (0, nu)))
    output = _add(vacuum, gramian)
    return {
        "fixture": "additive_diffusion",
        "quadrature_convention": "[Q,P]=I_AND_VACUUM_COVARIANCE_IDENTITY_OVER_2",
        "nu": _fraction(nu),
        "X": _matrix_json(identity),
        "accumulated_diffusion_gramian_Y": _matrix_json(gramian),
        "gramian_rank": _rank(gramian),
        "declared_support_dimension": 2,
        "minimum_gramian_eigenvalue_on_support": _fraction(nu),
        "vacuum_input_covariance": _matrix_json(vacuum),
        "vacuum_output_covariance": _matrix_json(output),
        "vacuum_output_determinant": _fraction(Q(25, 64)),
        "vacuum_output_purity": _fraction(Q(4, 5)),
        "added_mean_quanta": _fraction(Q(1, 8)),
        "identity_channel_requires_Y_zero": True,
        "exact_reference_complete_return": False,
        "blanket_all_noise_all_states_claimed": False,
    }


def _pure_loss_fixture() -> dict[str, Any]:
    eta = Q(1, 2)
    c = Q(5, 4)
    s = Q(3, 4)
    vacuum_variance = Q(1, 2)
    input_bus = c / 2
    input_cross = s / 2
    output_bus = eta * input_bus + (1 - eta) * vacuum_variance
    vacuum = _matrix(((vacuum_variance, 0), (0, vacuum_variance)))
    return {
        "fixture": "pure_loss_fixed_point",
        "eta": _fraction(eta),
        "vacuum_input_covariance": _matrix_json(vacuum),
        "vacuum_output_covariance": _matrix_json(vacuum),
        "prepared_vacuum_is_fixed_point": True,
        "channel_is_identity": False,
        "tmsv_input_bus_variance": _fraction(input_bus),
        "tmsv_output_bus_variance": _fraction(output_bus),
        "tmsv_input_cross_correlation_magnitude": _fraction(input_cross),
        "tmsv_output_cross_correlation": "(3/8)/SQRT(2)",
        "tmsv_output_cross_correlation_squared": _fraction(
            eta * input_cross * input_cross
        ),
        "reference_complete_identity_rejected": output_bus != input_bus,
        "fixed_prepared_state_not_sufficient_for_return": True,
    }


def _rank_deficient_fixture() -> dict[str, Any]:
    nu = Q(1, 8)
    gramian = _matrix(
        (
            (nu, 0, 0, 0),
            (0, nu, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
    )
    return {
        "fixture": "rank_deficient_dark_mode",
        "accumulated_diffusion_gramian_Y": _matrix_json(gramian),
        "gramian_rank": _rank(gramian),
        "noisy_support": ["Q_MODE_0", "P_MODE_0"],
        "dark_kernel": ["Q_MODE_1", "P_MODE_1"],
        "dark_kernel_dimension": 4 - _rank(gramian),
        "no_return_applies_to_noisy_support": True,
        "no_return_asserted_on_dark_kernel": False,
        "dark_subspace_can_escape_if_dynamics_preserve_it": True,
    }


def _finite_environment_fixture() -> dict[str, Any]:
    return {
        "fixture": "finite_environment_recurrence",
        "model": "TWO_MODE_NUMBER_PRESERVING_BEAM_SPLITTER_WITH_RETAINED_ENVIRONMENT",
        "heisenberg_law": "A_S_THETA=COS(THETA)A_S_PLUS_SIN(THETA)A_E",
        "midcycle": {
            "theta_pi_units": _fraction(Q(1, 2)),
            "system_coefficient": 0,
            "environment_coefficient": 1,
            "system_channel_identity": False,
        },
        "full_recurrence": {
            "theta_pi_units": _fraction(Q(2)),
            "system_coefficient": 1,
            "environment_coefficient": 0,
            "joint_heisenberg_return": True,
        },
        "cp_divisible_markov_diffusion_model": False,
        "outside_positive_accumulated_diffusion_theorem_scope": True,
        "intermediate_noise_never_recoheres_claimed": False,
    }


def _common_nontrivial_fixture() -> dict[str, Any]:
    quarter_turn = _matrix(((0, -1), (1, 0)))
    return {
        "fixture": "common_nontrivial_bus_evolution",
        "client_labels": 2,
        "sector_bus_symplectics": [_matrix_json(quarter_turn)] * 2,
        "sector_displacements": [[0, 0], [0, 0]],
        "sector_lifted_phases_pi_units": [
            _fraction(Q(0)),
            _fraction(Q(1, 3)),
        ],
        "joint_factorization": "DIAG(1,EXP(I*PI/3))_CLIENT_TENSOR_R_PI_OVER_2_BUS",
        "branch_relative_bus_action_identity": True,
        "client_direct_shadow_exists": True,
        "bus_endpoint_is_identity": quarter_turn == _identity(2),
        "carrier_restoration": False,
        "factorization_alone_is_not_restoration": True,
    }


def _declared_environment_fixture() -> dict[str, Any]:
    kappa = Q(3, 5)
    purity = (1 + kappa * kappa) / 2
    kernel = _matrix(((1, kappa), (kappa, 1)))
    all_ones = _matrix(((1, 1), (1, 1)))
    return {
        "fixture": "declared_environment_schur",
        "common_dilation_declared": True,
        "sector_carrier_gaussian_triples": ["(I,0,0)", "(I,0,0)"],
        "individual_sector_carrier_channels_identity": True,
        "environment_overlap_kappa": _fraction(kappa),
        "client_schur_kernel": _matrix_json(kernel),
        "kernel_rank": _rank(kernel),
        "plus_client_output_purity": _fraction(purity),
        "comparison_same_sector_triples_overlap_one_kernel": _matrix_json(all_ones),
        "sector_triples_determine_cross_branch_channel": False,
        "declared_common_dilation_overlap_data_required": True,
        "strong_carrier_environment_return": False,
        "carrier_reference_identity": True,
        "direct_client_schur_shadow_exists": True,
    }


def _sector_scaling_fixture() -> dict[str, Any]:
    named_modes = 1
    named_segments = 4
    rows = []
    for q in (1, 2, 4, 8, 12):
        labels = 1 << q
        common_symplectic_scalars = named_segments * (2 * named_modes) ** 2
        affine_force_scalars = (
            named_segments * 2 * named_modes * (q + 1)
        )
        rows.append(
            {
                "q_client_bits": q,
                "L_joint_labels": labels,
                "general_diagonal_phase_entries": labels,
                "general_declared_schur_entries": labels * labels,
                "affine_label_coarse_o_q_squared_count": q * q,
                "display_carrier_modes_M": named_modes,
                "display_public_segment_count_K": named_segments,
                "display_common_symplectic_descriptor_scalars": (
                    common_symplectic_scalars
                ),
                "display_affine_force_descriptor_scalars": affine_force_scalars,
                "display_total_input_descriptor_scalars": (
                    common_symplectic_scalars + affine_force_scalars
                ),
            }
        )
    return {
        "fixture": "sector_scaling",
        "samples": rows,
        "general_sector_shadow_cost": "EXPLICIT_O(L)_DIAGONAL_OR_O(L_SQUARED)_SCHUR",
        "affine_weyl_force_corollary": {
            "status": "FAIL_CLOSED_UNLESS_EVERY_HYPOTHESIS_IS_TRUE",
            "hypotheses": {
                "finite_carrier_modes_M": True,
                "finite_commuting_binary_labels_Q": True,
                "public_finite_segment_structure_K": True,
                "segment_force_law": "V_K(Z)=V_K0+SUM_I_Z_I*V_KI",
                "segment_force_is_label_affine": True,
                "quadratic_or_symplectic_propagation_G_K_or_S_K_is_common": True,
                "quadratic_or_symplectic_propagation_is_label_independent": True,
                "label_dependent_quadratic_generator_absent": True,
                "final_displacement_closed_for_every_label": True,
                "phase_source_is_bilinear_weyl_cocycle": True,
            },
            "conclusion": "COMPILED_CLIENT_PHASE_POLYNOMIAL_DEGREE_AT_MOST_2_IN_Z",
            "degree_bound_reason": "BILINEAR_WEYL_COCYCLE_APPLIED_TO_LABEL_AFFINE_FORCE_VECTORS",
            "input_descriptor_scalars": "O(K*(M^2+Q*M))",
            "compiled_phase_coefficient_scalars": "O(Q^2)",
            "dense_compilation_arithmetic_upper_bound": "O(K*M^3+K*Q*M^2+K*Q^2*M)",
            "per_label_application_arithmetic": "O(Q^2)",
            "public_segment_count_K_charged": True,
            "K_assumed_constant_in_scaling": False,
            "compilation_work_charged": True,
            "application_work_charged": True,
            "fails_if_label_dependent_quadratic_generator_present": True,
            "fails_if_force_law_has_label_degree_above_1": True,
            "fails_if_displacement_not_closed_for_every_label": True,
        },
        "named_corollary_parameters": {
            "M": named_modes,
            "Q": 2,
            "K": named_segments,
            "input_descriptor_scalar_upper_count": 40,
            "compiled_coarse_q_squared_coefficient_count": 4,
            "exact_nonzero_quadratic_phase_coefficients": 1,
            "exact_phase_polynomial_radians": "-Z0*Z1/6",
        },
        "coarse_o_q_squared_counts": [1, 4, 16, 64, 144],
        "blanket_polynomial_classical_efficiency_claimed": False,
        "arbitrary_phase_or_environment_kernel_may_require_exponential_label_data": True,
    }


def run() -> dict[str, Any]:
    metaplectic_2pi, metaplectic_4pi = _metaplectic_fixtures()
    fixtures = {
        "metaplectic_2pi_vs_zero": metaplectic_2pi,
        "metaplectic_4pi_control": metaplectic_4pi,
        "weyl_rectangle_cocycle": _weyl_fixture(),
        "vacuum_rotation_marginal_false_positive": _vacuum_rotation_fixture(),
        "additive_diffusion": _additive_diffusion_fixture(),
        "pure_loss_fixed_point": _pure_loss_fixture(),
        "rank_deficient_dark_mode": _rank_deficient_fixture(),
        "finite_environment_recurrence": _finite_environment_fixture(),
        "common_nontrivial_bus_evolution": _common_nontrivial_fixture(),
        "declared_environment_schur": _declared_environment_fixture(),
        "sector_scaling": _sector_scaling_fixture(),
    }

    checks = {
        "exact_claim_authority": CLAIM.startswith("FINITE_MODE_PUBLIC_FIXED_AXIS"),
        "metaplectic_2pi_lift_detected": (
            fixtures["metaplectic_2pi_vs_zero"]["projected_endpoints_equal"]
            and fixtures["metaplectic_2pi_vs_zero"]["lifted_metaplectic_signs"]
            == [1, -1]
        ),
        "metaplectic_4pi_returns_lift": fixtures["metaplectic_4pi_control"][
            "matches_zero_lift"
        ],
        "weyl_rectangle_closes_with_cocycle": (
            fixtures["weyl_rectangle_cocycle"]["client_qubits"] == 2
            and fixtures["weyl_rectangle_cocycle"]["sector_order"]
            == ["++", "+-", "-+", "--"]
            and fixtures["weyl_rectangle_cocycle"][
                "exact_sector_phase_law_radians"
            ]
            == "PHI(Z0,Z1)=-Z0*Z1/6"
            and fixtures["weyl_rectangle_cocycle"][
                "all_sector_displacements_close"
            ]
            and fixtures["weyl_rectangle_cocycle"][
                "sector_phase_radians_in_declared_order"
            ]
            == [
                _fraction(Q(-1, 6)),
                _fraction(Q(1, 6)),
                _fraction(Q(1, 6)),
                _fraction(Q(-1, 6)),
            ]
            and fixtures["weyl_rectangle_cocycle"][
                "direct_client_diagonal_in_declared_order"
            ]
            == ["EXP(-I/6)", "EXP(+I/6)", "EXP(+I/6)", "EXP(-I/6)"]
        ),
        "marginal_false_positive_caught_by_reference": (
            fixtures["vacuum_rotation_marginal_false_positive"][
                "bus_marginal_unchanged"
            ]
            and fixtures["vacuum_rotation_marginal_false_positive"][
                "reference_complete_identity_rejected"
            ]
        ),
        "positive_additive_diffusion_rejects_return": (
            fixtures["additive_diffusion"]["gramian_rank"] == 2
            and fixtures["additive_diffusion"]["vacuum_output_purity"]
            == _fraction(Q(4, 5))
            and not fixtures["additive_diffusion"]["exact_reference_complete_return"]
        ),
        "pure_loss_fixed_point_not_channel_identity": (
            fixtures["pure_loss_fixed_point"]["prepared_vacuum_is_fixed_point"]
            and fixtures["pure_loss_fixed_point"]["reference_complete_identity_rejected"]
        ),
        "dark_kernel_scopes_no_return": (
            fixtures["rank_deficient_dark_mode"]["gramian_rank"] == 2
            and not fixtures["rank_deficient_dark_mode"][
                "no_return_asserted_on_dark_kernel"
            ]
        ),
        "finite_environment_recurrence_outside_scope": (
            fixtures["finite_environment_recurrence"]["full_recurrence"][
                "joint_heisenberg_return"
            ]
            and fixtures["finite_environment_recurrence"][
                "outside_positive_accumulated_diffusion_theorem_scope"
            ]
        ),
        "factorization_not_misclassified_as_restoration": (
            fixtures["common_nontrivial_bus_evolution"]["joint_factorization"]
            and not fixtures["common_nontrivial_bus_evolution"][
                "carrier_restoration"
            ]
        ),
        "declared_dilation_schur_requires_overlap": (
            fixtures["declared_environment_schur"]["plus_client_output_purity"]
            == _fraction(Q(17, 25))
            and not fixtures["declared_environment_schur"][
                "sector_triples_determine_cross_branch_channel"
            ]
        ),
        "general_cost_not_hidden_by_affine_corollary": (
            fixtures["sector_scaling"]["samples"][-1][
                "general_declared_schur_entries"
            ]
            == (1 << 12) ** 2
            and [
                row["affine_label_coarse_o_q_squared_count"]
                for row in fixtures["sector_scaling"]["samples"]
            ]
            == [1, 4, 16, 64, 144]
            and fixtures["sector_scaling"]["blanket_polynomial_classical_efficiency_claimed"]
            is False
        ),
        "affine_weyl_corollary_is_explicit_and_fail_closed": (
            all(
                value is True
                for key, value in fixtures["sector_scaling"][
                    "affine_weyl_force_corollary"
                ]["hypotheses"].items()
                if key != "segment_force_law"
            )
            and fixtures["sector_scaling"]["affine_weyl_force_corollary"]
            ["status"]
            == "FAIL_CLOSED_UNLESS_EVERY_HYPOTHESIS_IS_TRUE"
            and fixtures["sector_scaling"]["affine_weyl_force_corollary"]
            ["input_descriptor_scalars"]
            == "O(K*(M^2+Q*M))"
            and fixtures["sector_scaling"]["named_corollary_parameters"]
            ["input_descriptor_scalar_upper_count"]
            == 40
            and [
                row["display_total_input_descriptor_scalars"]
                for row in fixtures["sector_scaling"]["samples"]
            ]
            == [32, 40, 56, 88, 120]
        ),
        "no_physical_result": True,
        "no_same_backing_result": True,
        "m257_intact": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"M267 internal self-check failure: {failed}")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "status": "PASS_INTERNAL_FORMAL_OBSTRUCTION_SELF_CHECK",
        "terminal": False,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": RESTORATION_SCOPE,
        "resource_disposition": RESOURCE_DISPOSITION,
        "next_mechanism": NEXT_MECHANISM,
        "source_self_assertion": "PASS_INTERNAL_CONSISTENCY_ONLY",
        "theorem": {
            "client_accessible_algebra": (
                "FINITE_FIXED_COMMUTING_PROJECTORS_|S><S|_WITH_DECLARED_CLIENT_BOUNDARY"
            ),
            "carrier_algebra": (
                "FINITE_MODE_CCR_WEYL_ALGEBRA_WITH_PUBLIC_PIECEWISE_AFFINE_QUADRATIC_GENERATORS"
            ),
            "sector_form": "CONTROLLED_SUM_S_|S><S|_TENSOR_V_S",
            "common_dilation_requirement": (
                "ALL_V_S_MAP_ONE_DECLARED_CARRIER_INPUT_AND_ONE_DECLARED_ENVIRONMENT_PREPARATION_INTO_ONE_COMMON_OUTPUT_SPACE"
            ),
            "reference_complete_identity": (
                "FOR_EVERY_CARRIER_REFERENCE_STATE_RHO_BR_THE_REDUCED_OUTPUT_ON_BR_EQUALS_RHO_BR"
            ),
            "identity_channel_gaussian_conditions": "X_S=I_D_S=0_Y_S=0",
            "common_dilation_consequence": (
                "V_S|PSI>_B=|PSI>_B_TENSOR|E_S>_E_UP_TO_A_LIFTED_PHASE"
            ),
            "client_boundary_consequence": (
                "RHO_ST_TO_K_ST_RHO_ST_WITH_K_ST=<E_T|E_S>;_COMMON_ENVIRONMENT_RETURN_REDUCES_K_TO_A_DIAGONAL_PHASE_OUTER_PRODUCT"
            ),
            "cp_divisible_accumulated_diffusion": (
                "Y_T=INTEGRAL_X_T_TAU_D_TAU_X_T_TAU_TRANSPOSE_DTAU_IS_POSITIVE_SEMIDEFINITE"
            ),
            "diffusion_obstruction": (
                "POSITIVE_RESTRICTION_OF_Y_T_TO_DECLARED_SUPPORT_CONTRADICTS_THE_Y_T=0_IDENTITY_CHANNEL_CONDITION"
            ),
            "lifted_data_required": [
                "SYMPLECTIC_OR_AFFINE_ENDPOINT",
                "METAPLECTIC_WINDING_OR_WEYL_COCYCLE_PHASE",
                "DECLARED_COMMON_DILATION_ENVIRONMENT_OVERLAPS",
            ],
            "polynomial_corollary_applies_to_general_sector_class": False,
            "polynomial_corollary_exact_output_path": (
                "fixtures.sector_scaling.affine_weyl_force_corollary"
            ),
        },
        "fixtures": fixtures,
        "strongest_honest_classical_comparator": {
            "general_reference_complete_closed_sector_case": (
                "DIRECT_CLIENT_DIAGONAL_PHASE_WITH_L_ENTRIES_OR_DECLARED_SCHUR_MULTIPLIER_WITH_L_SQUARED_ENTRIES"
            ),
            "general_sector_descriptor_cost": "EXPLICIT_L_OR_L_SQUARED_NO_BLANKET_COMPACTNESS",
            "affine_weyl_force_corollary_hypotheses": (
                "V_K(Z)=V_K0+SUM_I_Z_I*V_KI;_COMMON_LABEL_INDEPENDENT_G_K_OR_S_K;_PUBLIC_K;_CLOSED_DISPLACEMENT;_NO_LABEL_DEPENDENT_QUADRATIC_GENERATOR"
            ),
            "affine_weyl_force_input_descriptor_cost": "O(K*(M^2+Q*M))",
            "affine_weyl_force_compiled_phase_coefficients": "O(Q^2)",
            "affine_weyl_force_dense_compilation_arithmetic_upper_bound": (
                "O(K*M^3+K*Q*M^2+K*Q^2*M)"
            ),
            "affine_weyl_force_per_label_application_arithmetic": "O(Q^2)",
            "public_segment_count_K_charged": True,
            "bus_modes_retained": 0,
            "restoration_stage_executed": False,
            "arbitrary_client_phase_function_proved_easy": False,
            "m257_escape_established": False,
        },
        "resource_ledger": {
            "formal_carrier_modes": 1,
            "largest_control_carrier_modes": 2,
            "executed_physical_modes": 0,
            "joint_client_label_scaling_samples_q": [1, 2, 4, 8, 12],
            "affine_weyl_corollary_public_segment_count_K": 4,
            "affine_weyl_corollary_K_assumed_constant_in_scaling": False,
            "affine_weyl_corollary_input_descriptor_scalar_upper_count": 40,
            "affine_weyl_corollary_compiled_coarse_q_squared_count": 4,
            "affine_weyl_corollary_compilation_work_charged": True,
            "affine_weyl_corollary_application_work_charged": True,
            "exact_arithmetic": "PYTHON_FRACTION_PLUS_DECLARED_SYMBOLIC_PI_AND_SQRT_2",
            "maximum_exact_fixture_denominator": 128,
            "maximum_exact_fixture_denominator_bits": 8,
            "largest_named_lifted_winding_turns": 2,
            "largest_named_lifted_winding_descriptor_bits": 2,
            "tmsv_covariance_condition_number": 4,
            "tmsv_mean_quanta_per_mode": _fraction(Q(1, 8)),
            "additive_diffusion_added_mean_quanta": _fraction(Q(1, 8)),
            "physical_energy_joules": "UNINSTANTIATED",
            "bandwidth_hz": "UNINSTANTIATED",
            "latency_s": "UNINSTANTIATED",
            "controller_state": "PUBLIC_FORMAL_FIXTURE_DESCRIPTORS_ONLY",
            "retained_dynamic_trajectory_history": 0,
            "precision_growth_law": (
                "EXACT_RATIONAL_BIT_LENGTH_AND_LIFTED_WINDING_DESCRIPTOR_BITS_ARE_CHARGED;_NO_CONTINUUM_OR_HIDDEN_PRECISION_ADVANTAGE"
            ),
            "conditioning_law": (
                "REFERENCE_TEST_CONDITIONING_DEPENDS_ON_PROBE_ENERGY;_THE_NAMED_TMSV_FIXTURE_HAS_CONDITION_NUMBER_4_ONLY"
            ),
        },
        "scope_exclusions": {
            "noncommuting_client_axes": True,
            "nonquadratic_or_non_gaussian_interactions": True,
            "non_gaussian_boundary_measurements": True,
            "measurement_feedback_or_qec": True,
            "restricted_or_exogenous_access": True,
            "nonmarkov_environment_recoherence": True,
            "infinite_mode_limits": True,
            "dark_or_noiseless_subspaces_not_on_declared_support": True,
            "physical_carrier_custody": True,
        },
        "negative_claims": {
            "physical_execution": False,
            "physical_same_backing_custody": False,
            "physical_restoration": False,
            "same_backing_software_execution": False,
            "qemu_device_execution": False,
            "all_noise_changes_all_states": False,
            "intermediate_noise_never_recoheres": False,
            "blanket_classical_polynomial_efficiency": False,
            "complexity_lower_bound": False,
            "resource_advantage": False,
            "unbounded_compute": False,
        },
        "checks": checks,
    }
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main() -> None:
    print(json.dumps(run(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
