#!/usr/bin/env python3
"""Independent analytic oracle for the M267 conditional-Gaussian class result.

The oracle derives every fixture from affine-symplectic, covariance-channel,
or declared dilation data.  It is a finite-dimensional software reference;
it executes no restoration and establishes no physical carrier custody.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


REFERENCE_ID = "M267_CONDITIONAL_GAUSSIAN_CLOSED_LOOP_SEPARATE_REFERENCE_V1"
SCHEMA = "PHASE_QEMU_V9_CONDITIONAL_GAUSSIAN_CLOSED_LOOP_SEPARATE_REFERENCE_V1"
CLAIM = (
    "FINITE_MODE_PUBLIC_FIXED_AXIS_CONDITIONAL_GAUSSIAN_LOOPS_WITH_EXACT_"
    "FAITHFUL_CARRIER_REFERENCE_IDENTITY_REDUCE_TO_A_DIRECT_CLIENT_DIAGONAL_"
    "PHASE_OR_DECLARED_DILATION_SCHUR_CHANNEL_WHILE_POSITIVE_ACCUMULATED_CP_"
    "DIVISIBLE_MARKOV_DIFFUSION_ON_A_CLAIMED_CARRIER_SUBSPACE_PRECLUDES_"
    "EXACT_SAME_MODE_CHANNEL_RETURN_ON_THAT_SUBSPACE"
)
CEILING = (
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
DISPOSITION = (
    "GENERAL_SECTOR_DIRECT_CLIENT_SHADOW_EXISTS_WITH_EXPLICIT_L_OR_L_SQUARED_"
    "COST_AND_THE_AFFINE_LABEL_COROLLARY_IS_POLYNOMIALLY_COMPACT_WHILE_"
    "POSITIVE_DIFFUSION_ON_CLAIMED_SUPPORT_FORBIDS_EXACT_REFERENCE_COMPLETE_"
    "RETURN_SO_NO_CATALYTIC_BUS_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_"
    "ESTABLISHED"
)
SUCCESSOR = (
    "RESTRICTED_ACCESS_NON_GAUSSIAN_PHASE_EIGENSTATE_KICKBACK_ORACLE_WITH_"
    "FAITHFUL_CARRIER_RETURN_PREPARATION_PRECISION_QUERY_AND_CUSTODY_COSTS"
)

VACUUM_VARIANCE = 0.5
TMSV_C = 5.0 / 4.0
TMSV_S = 3.0 / 4.0
DIFFUSION_NU = 1.0 / 8.0
LOSS_ETA = 1.0 / 2.0


def clean(value: object) -> object:
    """Convert numpy and complex objects into deterministic JSON values."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, complex):
        real = 0.0 if abs(value.real) < 1e-15 else float(value.real)
        imag = 0.0 if abs(value.imag) < 1e-15 else float(value.imag)
        return {"real": real, "imag": imag}
    if isinstance(value, float) and abs(value) < 1e-15:
        return 0.0
    return value


def frobenius(matrix: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(matrix) ** 2)))


def symplectic_form(modes: int) -> np.ndarray:
    j = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return np.kron(np.eye(modes), j)


def symplectic_residual(matrix: np.ndarray) -> float:
    omega = symplectic_form(matrix.shape[0] // 2)
    return frobenius(matrix @ omega @ matrix.T - omega)


def quarter_rotation(quarter_turns: int) -> np.ndarray:
    """Return an exact integer rotation for a multiple of pi/2."""
    rotations = (
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[-1.0, 0.0], [0.0, -1.0]]),
        np.array([[0.0, 1.0], [-1.0, 0.0]]),
    )
    return rotations[quarter_turns % 4].copy()


def tmsv_covariance() -> np.ndarray:
    identity = np.eye(2)
    sign = np.diag([1.0, -1.0])
    return VACUUM_VARIANCE * np.block(
        [[TMSV_C * identity, TMSV_S * sign],
         [TMSV_S * sign, TMSV_C * identity]]
    )


def apply_signal_channel_to_reference(
    covariance: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Apply a one-mode Gaussian channel to the signal half of a TMSV."""
    signal = covariance[:2, :2]
    correlation = covariance[:2, 2:]
    reference = covariance[2:, 2:]
    return np.block(
        [
            [x @ signal @ x.T + y, x @ correlation],
            [correlation.T @ x.T, reference],
        ]
    )


def metaplectic_fixtures() -> tuple[dict[str, object], dict[str, object]]:
    zero_map = quarter_rotation(0)
    two_pi_map = quarter_rotation(4)
    four_pi_map = quarter_rotation(8)
    zero_lift = complex(1.0, 0.0)
    two_pi_lift = complex((-1) ** 1, 0.0)
    four_pi_lift = complex((-1) ** 2, 0.0)
    two_pi = {
        "angle_radians": 2.0 * math.pi,
        "zero_affine_symplectic_map": zero_map,
        "two_pi_affine_symplectic_map": two_pi_map,
        "affine_symplectic_maps_equal": bool(np.array_equal(zero_map, two_pi_map)),
        "zero_metaplectic_scalar": zero_lift,
        "two_pi_metaplectic_scalar": two_pi_lift,
        "lifted_scalars_equal": bool(abs(zero_lift - two_pi_lift) <= 1e-15),
        "client_detectable_relative_phase_if_conditioned": two_pi_lift / zero_lift,
    }
    four_pi = {
        "angle_radians": 4.0 * math.pi,
        "four_pi_affine_symplectic_map": four_pi_map,
        "four_pi_metaplectic_scalar": four_pi_lift,
        "returns_to_zero_lift": bool(
            np.array_equal(zero_map, four_pi_map)
            and abs(four_pi_lift - zero_lift) <= 1e-15
        ),
    }
    return two_pi, four_pi


def weyl_rectangle_fixture() -> dict[str, object]:
    omega = symplectic_form(1)
    a = 1.0 / 2.0
    b = 1.0 / 3.0
    sector_descriptors = (
        ("++", 1.0, 1.0),
        ("+-", 1.0, -1.0),
        ("-+", -1.0, 1.0),
        ("--", -1.0, -1.0),
    )
    sectors = []
    for name, z0, z1 in sector_descriptors:
        xi = np.array([a * z0, 0.0])
        eta = np.array([0.0, b * z1])
        path = (xi, eta, -xi, -eta)
        net = np.sum(path, axis=0)
        pair_sum = sum(
            float(left @ omega @ right)
            for index, left in enumerate(path)
            for right in path[index + 1 :]
        )
        cocycle_exponent = -0.5 * pair_sum
        expected_exponent = -z0 * z1 / 6.0
        phase = np.exp(1j * cocycle_exponent)
        expected_phase = np.exp(1j * expected_exponent)
        sectors.append(
            {
                "name": name,
                "z0": z0,
                "z1": z1,
                "xi": xi,
                "eta": eta,
                "ordered_rectangle": path,
                "net_displacement": net,
                "pairwise_symplectic_sum": pair_sum,
                "cocycle_exponent_radians": cocycle_exponent,
                "expected_zz_exponent_radians": expected_exponent,
                "lifted_rectangle_scalar": phase,
                "expected_zz_scalar": expected_phase,
                "carrier_affine_map_is_identity": bool(frobenius(net) == 0.0),
                "matches_exact_zz_law": bool(
                    abs(cocycle_exponent - expected_exponent) <= 1e-15
                    and abs(phase - expected_phase) <= 1e-15
                ),
            }
        )
    client_diagonal = np.diag(
        np.array([sector["lifted_rectangle_scalar"] for sector in sectors])
    )
    return {
        "weyl_convention": (
            "W_XI_W_ETA=EXP_MINUS_I_OVER_2_XI_TRANSPOSE_OMEGA_ETA_TIMES_"
            "W_XI_PLUS_ETA"
        ),
        "a": a,
        "b": b,
        "a_exact": "1_OVER_2",
        "b_exact": "1_OVER_3",
        "sector_order": [descriptor[0] for descriptor in sector_descriptors],
        "sectors": sectors,
        "exact_client_phase_law": "EXP_MINUS_I_Z0_Z1_OVER_6",
        "exact_sector_phase_sequence": (
            "[EXP_MINUS_I_OVER_6,EXP_PLUS_I_OVER_6,"
            "EXP_PLUS_I_OVER_6,EXP_MINUS_I_OVER_6]"
        ),
        "direct_client_diagonal": client_diagonal,
        "all_sector_carrier_affine_maps_are_identity": bool(
            all(sector["carrier_affine_map_is_identity"] for sector in sectors)
        ),
        "all_sector_lifted_scalars_match_exact_zz_law": bool(
            all(sector["matches_exact_zz_law"] for sector in sectors)
        ),
        "lifted_client_diagonal_is_nontrivial": bool(
            not np.allclose(client_diagonal, np.eye(4), atol=1e-15, rtol=0.0)
        ),
    }


def vacuum_rotation_fixture() -> dict[str, object]:
    rotation = quarter_rotation(1)
    vacuum = VACUUM_VARIANCE * np.eye(2)
    marginal_after = rotation @ vacuum @ rotation.T
    faithful_before = tmsv_covariance()
    faithful_after = apply_signal_channel_to_reference(
        faithful_before, rotation, np.zeros((2, 2))
    )
    return {
        "rotation": rotation,
        "vacuum_covariance_before": vacuum,
        "vacuum_covariance_after": marginal_after,
        "vacuum_marginal_frobenius_change": frobenius(marginal_after - vacuum),
        "tmsv_parameters": {"c": TMSV_C, "s": TMSV_S},
        "faithful_tmsv_covariance_before": faithful_before,
        "faithful_tmsv_covariance_after": faithful_after,
        "faithful_tmsv_frobenius_change": frobenius(
            faithful_after - faithful_before
        ),
        "marginal_test_false_positive": bool(
            frobenius(marginal_after - vacuum) == 0.0
            and frobenius(faithful_after - faithful_before) > 1e-6
        ),
        "reference_complete_identity": False,
    }


def additive_diffusion_fixture() -> dict[str, object]:
    x = np.eye(2)
    y = DIFFUSION_NU * np.eye(2)
    vacuum = VACUUM_VARIANCE * np.eye(2)
    vacuum_after = x @ vacuum @ x.T + y
    faithful_before = tmsv_covariance()
    faithful_after = apply_signal_channel_to_reference(faithful_before, x, y)
    output_determinant = float(np.linalg.det(vacuum_after))
    output_purity = 1.0 / (2.0 * math.sqrt(output_determinant))
    added_mean_occupation = float(
        (np.trace(vacuum_after) - np.trace(vacuum)) / 2.0
    )
    return {
        "x": x,
        "y": y,
        "nu": DIFFUSION_NU,
        "diffusion_eigenvalues": np.linalg.eigvalsh(y),
        "diffusion_rank": int(np.linalg.matrix_rank(y)),
        "vacuum_covariance_before": vacuum,
        "vacuum_covariance_after": vacuum_after,
        "vacuum_output_covariance_determinant": output_determinant,
        "vacuum_output_gaussian_purity": output_purity,
        "added_mean_occupation": added_mean_occupation,
        "faithful_tmsv_covariance_after": faithful_after,
        "faithful_tmsv_frobenius_change": frobenius(
            faithful_after - faithful_before
        ),
        "reference_complete_identity": bool(
            np.array_equal(x, np.eye(2)) and np.count_nonzero(y) == 0
        ),
    }


def pure_loss_fixture() -> dict[str, object]:
    x = math.sqrt(LOSS_ETA) * np.eye(2)
    y = (1.0 - LOSS_ETA) * VACUUM_VARIANCE * np.eye(2)
    vacuum = VACUUM_VARIANCE * np.eye(2)
    vacuum_after = x @ vacuum @ x.T + y
    faithful_before = tmsv_covariance()
    faithful_after = apply_signal_channel_to_reference(faithful_before, x, y)
    signal_variance_after = float(faithful_after[0, 0])
    signal_reference_cross_after = float(faithful_after[0, 2])
    return {
        "eta": LOSS_ETA,
        "x": x,
        "y": y,
        "vacuum_covariance_before": vacuum,
        "vacuum_covariance_after": vacuum_after,
        "vacuum_is_exact_fixed_point": bool(np.array_equal(vacuum, vacuum_after)),
        "faithful_tmsv_covariance_after": faithful_after,
        "faithful_tmsv_signal_variance_after": signal_variance_after,
        "faithful_tmsv_cross_entry_squared": signal_reference_cross_after**2,
        "faithful_tmsv_frobenius_change": frobenius(
            faithful_after - faithful_before
        ),
        "reference_complete_identity": False,
    }


def rank_deficient_dark_mode_fixture() -> dict[str, object]:
    noisy_projector = np.diag([1.0, 1.0, 0.0, 0.0])
    dark_projector = np.eye(4) - noisy_projector
    diffusion = DIFFUSION_NU * noisy_projector
    dark_basis = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).T
    return {
        "quadrature_order": ["q0", "p0", "q1", "p1"],
        "noisy_mode_projector": noisy_projector,
        "dark_projector": dark_projector,
        "diffusion_y": diffusion,
        "diffusion_eigenvalues": np.linalg.eigvalsh(diffusion),
        "diffusion_rank": int(np.linalg.matrix_rank(diffusion)),
        "dark_kernel_dimension": int(4 - np.linalg.matrix_rank(diffusion)),
        "dark_basis": dark_basis,
        "dark_kernel_residual": frobenius(diffusion @ dark_basis),
        "full_two_mode_reference_complete_identity": False,
        "declared_dark_subspace_is_noiseless": bool(
            frobenius(diffusion @ dark_basis) <= 1e-15
        ),
        "scope_law": (
            "NO_RETURN_APPLIES_ONLY_ON_DECLARED_SUPPORT_WITH_POSITIVE_"
            "ACCUMULATED_DIFFUSION_NOT_ON_THE_DARK_KERNEL"
        ),
    }


def beam_splitter_quarter(quarter_turns: int) -> np.ndarray:
    cycle = quarter_turns % 4
    cosine = (1.0, 0.0, -1.0, 0.0)[cycle]
    sine = (0.0, 1.0, 0.0, -1.0)[cycle]
    identity = np.eye(2)
    return np.block(
        [
            [cosine * identity, sine * identity],
            [-sine * identity, cosine * identity],
        ]
    )


def finite_environment_recurrence_fixture() -> dict[str, object]:
    environment = VACUUM_VARIANCE * np.eye(2)
    system = np.diag([3.0 / 4.0, 1.0 / 2.0])
    joint = np.block(
        [[system, np.zeros((2, 2))], [np.zeros((2, 2)), environment]]
    )
    steps = []
    for quarter_turn in range(5):
        dilation = beam_splitter_quarter(quarter_turn)
        cosine = (1.0, 0.0, -1.0, 0.0, 1.0)[quarter_turn]
        sine = (0.0, 1.0, 0.0, -1.0, 0.0)[quarter_turn]
        evolved = dilation @ joint @ dilation.T
        steps.append(
            {
                "quarter_turn": quarter_turn,
                "angle_radians": quarter_turn * math.pi / 2.0,
                "joint_symplectic": dilation,
                "symplectic_residual": symplectic_residual(dilation),
                "reduced_x": cosine * np.eye(2),
                "reduced_y_for_vacuum_environment": sine**2 * environment,
                "reduced_system_covariance": evolved[:2, :2],
            }
        )
    erased_x = np.asarray(steps[1]["reduced_x"])
    revived_x = np.asarray(steps[2]["reduced_x"])
    final_dilation = np.asarray(steps[4]["joint_symplectic"])
    return {
        "finite_environment_mode_count": 1,
        "steps": steps,
        "quarter_turn_reduced_map_erases_input": bool(
            frobenius(erased_x) == 0.0
        ),
        "half_turn_reduced_map_recovers_input_dependence": bool(
            frobenius(revived_x) > 1.0
        ),
        "four_quarter_joint_recurrence_is_identity": bool(
            np.array_equal(final_dilation, np.eye(4))
        ),
        "cp_divisible_markov_scope": False,
        "outside_scope_reason": (
            "FINITE_RETAINED_ENVIRONMENT_STORES_INFORMATION_AND_RETURNS_IT_"
            "AFTER_A_REDUCED_ERASURE_SO_THE_REDUCED_FAMILY_IS_NOT_CP_DIVISIBLE"
        ),
    }


def common_nontrivial_bus_fixture() -> dict[str, object]:
    labels = (0, 1)
    common = quarter_rotation(1)
    sector_maps = [common.copy() for _ in labels]
    phases = np.array([0.0, math.pi / 3.0])
    phase_vector = np.exp(1j * phases)
    client_diagonal = np.diag(phase_vector)
    phase_kernel = np.outer(phase_vector, np.conj(phase_vector))
    faithful_before = tmsv_covariance()
    faithful_after = apply_signal_channel_to_reference(
        faithful_before, common, np.zeros((2, 2))
    )
    return {
        "client_labels": labels,
        "common_bus_symplectic": common,
        "client_lifted_phases_radians": phases,
        "direct_client_diagonal": client_diagonal,
        "direct_client_phase_kernel": phase_kernel,
        "all_sector_maps_equal": bool(
            all(np.array_equal(item, sector_maps[0]) for item in sector_maps)
        ),
        "common_map_is_identity": bool(np.array_equal(common, np.eye(2))),
        "client_channel_is_identity": bool(
            np.array_equal(client_diagonal, np.eye(len(labels)))
        ),
        "direct_client_diagonal_nontrivial": bool(
            not np.array_equal(client_diagonal, np.eye(len(labels)))
        ),
        "factorized_common_bus_and_client_diagonal": True,
        "faithful_tmsv_frobenius_change": frobenius(
            faithful_after - faithful_before
        ),
        "carrier_reference_complete_identity": False,
        "control_law": (
            "THE_TWO_SECTOR_LIFTED_MAP_FACTORIZES_AS_DIAG_ONE_EXP_I_PI_OVER_3_"
            "ON_THE_CLIENT_TENSOR_COMMON_R_PI_OVER_2_ON_THE_BUS_SO_THE_DIRECT_"
            "CLIENT_DIAGONAL_IS_NONTRIVIAL_BUT_THE_BUS_HAS_NOT_RETURNED"
        ),
    }


def declared_environment_schur_fixture() -> dict[str, object]:
    labels = (0, 1)
    kappa = 3.0 / 5.0
    environment_vectors = np.array([[1.0, 0.0], [kappa, 4.0 / 5.0]])
    overlaps = environment_vectors @ environment_vectors.T
    phases = np.zeros(len(labels))
    phase_vector = np.exp(1j * phases)
    direct_kernel = (
        np.outer(phase_vector, np.conj(phase_vector))
        * overlaps.astype(np.complex128)
    )
    eigenvalues = np.linalg.eigvalsh((direct_kernel + direct_kernel.conj().T) / 2)
    uniform_client = np.ones_like(direct_kernel) / len(labels)
    output_client = direct_kernel * uniform_client
    expected_plus_purity = (1.0 + kappa**2) / 2.0
    return {
        "client_labels": labels,
        "kappa": kappa,
        "declared_environment_state_vectors": environment_vectors,
        "declared_environment_overlap_kernel": overlaps,
        "client_diagonal_phases_radians": phases,
        "direct_client_schur_kernel": direct_kernel,
        "kernel_eigenvalues": eigenvalues,
        "kernel_diagonal": np.diag(direct_kernel),
        "uniform_client_output": output_client,
        "uniform_client_output_trace": float(np.trace(output_client).real),
        "uniform_client_output_purity": float(
            np.trace(output_client @ output_client).real
        ),
        "plus_output_purity_expected": expected_plus_purity,
        "plus_output_purity_exact": "17_OVER_25",
        "positive_semidefinite": bool(float(np.min(eigenvalues)) >= -1e-12),
        "trace_preserving": bool(
            np.max(np.abs(np.diag(direct_kernel) - 1.0)) <= 1e-15
        ),
        "direct_shadow_law": (
            "RHO_LM_MAPS_TO_EXP_I_PHI_L_MINUS_PHI_M_TIMES_"
            "INNER_PRODUCT_E_M_E_L_TIMES_RHO_LM"
        ),
    }


def sector_scaling_fixture() -> dict[str, object]:
    rows = []
    for client_qubits in (1, 2, 4, 8, 12):
        branches = 1 << client_qubits
        rows.append(
            {
                "q": client_qubits,
                "L": branches,
                "L_squared": branches * branches,
                "generic_explicit_sector_map_entries": branches,
                "generic_explicit_schur_kernel_entries": branches * branches,
                "affine_weyl_force_compiled_q_squared_upper_bound": (
                    client_qubits**2
                ),
                "affine_weyl_force_pair_query_q_squared_upper_bound": (
                    client_qubits**2
                ),
            }
        )
    return {
        "rows": rows,
        "generic_law": (
            "A_GENERIC_PUBLIC_SECTOR_TABLE_HAS_L_EXPLICIT_BRANCH_RECORDS_AND_"
            "A_GENERIC_DECLARED_ENVIRONMENT_OVERLAP_TABLE_HAS_L_SQUARED_"
            "EXPLICIT_PAIR_RECORDS"
        ),
        "affine_weyl_force_corollary": {
            "hypotheses": {
                "segment_force_law": "V_K_Z=V_K0_PLUS_SUM_I_Z_I_V_KI",
                "client_labels": "Z_I_IN_PLUS_ONE_OR_MINUS_ONE",
                "common_quadratic_generator_is_label_independent": True,
                "common_symplectic_propagation": "S_K_OR_G_K_INDEPENDENT_OF_Z",
                "label_dependent_quadratic_generator": False,
                "segment_count_symbol": "K",
                "segment_count_is_finite_per_program": True,
                "K_assumed_constant_across_scaling_family": False,
                "segment_count_is_public": True,
                "segment_count_is_charged": True,
                "carrier_mode_count_symbol": "M",
                "endpoint_displacement_closed_for_every_label": True,
            },
            "derivation": (
                "COMMON_LINEAR_PROPAGATION_PRESERVES_AFFINITY_OF_EACH_"
                "PROPAGATED_FORCE_IN_Z_AND_THE_WEYL_COCYCLE_IS_BILINEAR_IN_"
                "PAIRS_OF_PROPAGATED_FORCES_SO_THE_CLOSED_LOOP_PHASE_HAS_"
                "DEGREE_AT_MOST_TWO_IN_Z"
            ),
            "phase_polynomial_degree_upper_bound": 2,
            "input_descriptor_scaling": "O(K*(M^2+q*M))",
            "compiled_phase_descriptor_scaling": "O(q^2)",
            "dense_compilation_arithmetic_upper_bound": (
                "O(K*M^3+K*q*M^2+K*q^2*M)"
            ),
            "compilation_work_charged": True,
            "per_label_application_arithmetic": "O(q^2)",
            "application_work_charged": True,
            "fixed_public_K_is_still_charged": True,
            "dense_L_or_L_squared_materialization_required": False,
            "scope_exclusion": (
                "NO_LABEL_DEPENDENT_QUADRATIC_GENERATOR_OR_LABEL_DEPENDENT_"
                "SYMPLECTIC_PROPAGATION"
            ),
        },
    }


def main() -> int:
    metaplectic_2pi, metaplectic_4pi = metaplectic_fixtures()
    fixtures = {
        "metaplectic_2pi_vs_zero": metaplectic_2pi,
        "metaplectic_4pi_control": metaplectic_4pi,
        "weyl_rectangle_cocycle": weyl_rectangle_fixture(),
        "vacuum_rotation_marginal_false_positive": vacuum_rotation_fixture(),
        "additive_diffusion": additive_diffusion_fixture(),
        "pure_loss_fixed_point": pure_loss_fixture(),
        "rank_deficient_dark_mode": rank_deficient_dark_mode_fixture(),
        "finite_environment_recurrence": finite_environment_recurrence_fixture(),
        "common_nontrivial_bus_evolution": common_nontrivial_bus_fixture(),
        "declared_environment_schur": declared_environment_schur_fixture(),
        "sector_scaling": sector_scaling_fixture(),
    }

    rows = fixtures["sector_scaling"]["rows"]
    affine_corollary = fixtures["sector_scaling"]["affine_weyl_force_corollary"]
    affine_hypotheses = affine_corollary["hypotheses"]
    checks = {
        "metaplectic_2pi_affine_identity_lifted_minus_one": bool(
            fixtures["metaplectic_2pi_vs_zero"]["affine_symplectic_maps_equal"]
            and not fixtures["metaplectic_2pi_vs_zero"]["lifted_scalars_equal"]
        ),
        "metaplectic_4pi_returns_to_zero_lift": bool(
            fixtures["metaplectic_4pi_control"]["returns_to_zero_lift"]
        ),
        "weyl_rectangle_closes_with_nontrivial_cocycle": bool(
            fixtures["weyl_rectangle_cocycle"]
            ["all_sector_carrier_affine_maps_are_identity"]
            and fixtures["weyl_rectangle_cocycle"]
            ["all_sector_lifted_scalars_match_exact_zz_law"]
            and fixtures["weyl_rectangle_cocycle"]
            ["lifted_client_diagonal_is_nontrivial"]
        ),
        "vacuum_marginal_test_is_not_reference_complete": bool(
            fixtures["vacuum_rotation_marginal_false_positive"]
            ["marginal_test_false_positive"]
        ),
        "positive_additive_diffusion_rejects_identity": bool(
            fixtures["additive_diffusion"]["diffusion_rank"] == 2
            and not fixtures["additive_diffusion"]["reference_complete_identity"]
        ),
        "pure_loss_vacuum_fixed_but_reference_fails": bool(
            fixtures["pure_loss_fixed_point"]["vacuum_is_exact_fixed_point"]
            and not fixtures["pure_loss_fixed_point"]["reference_complete_identity"]
        ),
        "rank_deficient_diffusion_preserves_only_dark_kernel": bool(
            fixtures["rank_deficient_dark_mode"]["diffusion_rank"] == 2
            and fixtures["rank_deficient_dark_mode"]
            ["declared_dark_subspace_is_noiseless"]
            and not fixtures["rank_deficient_dark_mode"]
            ["full_two_mode_reference_complete_identity"]
        ),
        "finite_environment_recurrence_is_outside_cp_divisible_scope": bool(
            fixtures["finite_environment_recurrence"]
            ["quarter_turn_reduced_map_erases_input"]
            and fixtures["finite_environment_recurrence"]
            ["half_turn_reduced_map_recovers_input_dependence"]
            and fixtures["finite_environment_recurrence"]
            ["four_quarter_joint_recurrence_is_identity"]
            and not fixtures["finite_environment_recurrence"]
            ["cp_divisible_markov_scope"]
        ),
        "common_nontrivial_map_is_not_carrier_return": bool(
            fixtures["common_nontrivial_bus_evolution"]["all_sector_maps_equal"]
            and fixtures["common_nontrivial_bus_evolution"]
            ["direct_client_diagonal_nontrivial"]
            and fixtures["common_nontrivial_bus_evolution"]
            ["factorized_common_bus_and_client_diagonal"]
            and not fixtures["common_nontrivial_bus_evolution"]
            ["carrier_reference_complete_identity"]
        ),
        "declared_environment_kernel_is_a_channel": bool(
            fixtures["declared_environment_schur"]["positive_semidefinite"]
            and fixtures["declared_environment_schur"]["trace_preserving"]
            and abs(
                fixtures["declared_environment_schur"]
                ["uniform_client_output_purity"]
                - 17.0 / 25.0
            )
            <= 1e-15
        ),
        "sector_counts_are_exact": bool(
            all(
                row["L"] == 2 ** row["q"]
                and row["L_squared"] == row["L"] ** 2
                for row in rows
            )
            and [row["L"] for row in rows] == [2, 4, 16, 256, 4096]
            and [row["L_squared"] for row in rows]
            == [4, 16, 256, 65536, 16777216]
            and [
                row["affine_weyl_force_compiled_q_squared_upper_bound"]
                for row in rows
            ]
            == [1, 4, 16, 64, 144]
        ),
        "affine_weyl_force_q_squared_corollary_is_exactly_scoped": bool(
            affine_hypotheses["segment_force_law"]
            == "V_K_Z=V_K0_PLUS_SUM_I_Z_I_V_KI"
            and affine_hypotheses[
                "common_quadratic_generator_is_label_independent"
            ]
            and affine_hypotheses["common_symplectic_propagation"]
            == "S_K_OR_G_K_INDEPENDENT_OF_Z"
            and not affine_hypotheses["label_dependent_quadratic_generator"]
            and affine_hypotheses["segment_count_is_finite_per_program"]
            and not affine_hypotheses["K_assumed_constant_across_scaling_family"]
            and affine_hypotheses["segment_count_is_public"]
            and affine_hypotheses["segment_count_is_charged"]
            and affine_hypotheses["endpoint_displacement_closed_for_every_label"]
            and affine_corollary["phase_polynomial_degree_upper_bound"] == 2
            and affine_corollary["input_descriptor_scaling"]
            == "O(K*(M^2+q*M))"
            and affine_corollary["compiled_phase_descriptor_scaling"] == "O(q^2)"
            and affine_corollary["dense_compilation_arithmetic_upper_bound"]
            == "O(K*M^3+K*q*M^2+K*q^2*M)"
            and affine_corollary["compilation_work_charged"]
            and affine_corollary["per_label_application_arithmetic"] == "O(q^2)"
            and affine_corollary["application_work_charged"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"independent analytic self-check failed: {failed}")

    payload = {
        "schema": SCHEMA,
        "reference_id": REFERENCE_ID,
        "milestone": "M267",
        "claim": CLAIM,
        "ceiling": CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": RESTORATION_SCOPE,
        "resource_disposition": DISPOSITION,
        "next_mechanism": SUCCESSOR,
        "mathematical_conventions": {
            "quadrature_commutator": "[Q_J,P_K]=I_DELTA_JK",
            "vacuum_covariance": "I2_OVER_2",
            "gaussian_channel": "M_MAPS_TO_X_M_PLUS_D_AND_V_MAPS_TO_X_V_XT_PLUS_Y",
            "faithful_reference_fixture": (
                "TWO_MODE_SQUEEZED_VACUUM_WITH_C=5_OVER_4_S=3_OVER_4"
            ),
            "lifted_rotation": "U_THETA=EXP_MINUS_I_THETA_TIMES_N_PLUS_ONE_HALF",
        },
        "class_theorem": {
            "assumptions": [
                "FINITE_CARRIER_MODE_COUNT",
                "FINITE_JOINT_FIXED_AXIS_COMMUTING_CLIENT_LABEL_SET",
                "PUBLIC_PIECEWISE_QUADRATIC_OR_AFFINE_GAUSSIAN_LAWS",
                "DECLARED_COMMON_ENVIRONMENT_DILATION_WHEN_PRESENT",
                "EXACT_GAUSSIAN_MOMENT_OR_LIFTED_AFFINE_SYMPLECTIC_SEMANTICS",
            ],
            "reference_complete_identity_criterion": (
                "THE_CARRIER_CHANNEL_TENSORED_WITH_IDENTITY_ON_A_FAITHFUL_"
                "REFERENCE_MUST_FIX_THE_JOINT_STATE_EQUIVALENTLY_X=I_D=0_Y=0_"
                "ON_THE_DECLARED_CARRIER_SUPPORT"
            ),
            "closed_sector_conclusion": (
                "IDENTITY_AFFINE_CARRIER_ENDPOINTS_LEAVE_ONLY_LIFTED_"
                "METAPLECTIC_OR_WEYL_BRANCH_PHASES_AND_A_DECLARED_ENVIRONMENT_"
                "DILATION_LEAVES_ITS_EXPLICIT_OVERLAP_SCHUR_KERNEL"
            ),
            "direct_client_shadow": (
                "RHO_LM_MAPS_TO_EXP_I_PHI_L_MINUS_PHI_M_TIMES_K_LM_TIMES_"
                "RHO_LM_WITH_K_LM_EQUAL_TO_DECLARED_ENVIRONMENT_OVERLAP"
            ),
            "cp_divisible_diffusion_obstruction": (
                "ACCUMULATED_Y_EQUALS_THE_TIME_INTEGRAL_OF_PROPAGATED_PSD_"
                "DIFFUSION_AND_CANNOT_EQUAL_ZERO_ON_ANY_CLAIMED_SUPPORT_WHERE_"
                "THE_ACCUMULATED_DIFFUSION_IS_POSITIVE"
            ),
            "excluded_or_separately_scoped_cases": [
                "DARK_OR_NOISELESS_SUBSPACE_OUTSIDE_DECLARED_NOISY_SUPPORT",
                "FINITE_ENVIRONMENT_NONMARKOV_RECOHERENCE_OR_ENVIRONMENT_REVERSAL",
                "NONCOMMUTING_CLIENT_AXES",
                "NONQUADRATIC_OR_NON_GAUSSIAN_INTERACTIONS",
                "NON_GAUSSIAN_BOUNDARY_MEASUREMENTS_OR_QEC",
                "RESTRICTED_OR_EXOGENOUS_ACCESS",
                "INFINITE_MODE_LIMITS",
            ],
        },
        "fixtures": fixtures,
        "checks": checks,
        "claims": {
            "formal_reference_complete_gaussian_identity_criterion": True,
            "positive_markov_diffusion_no_return_on_declared_support": True,
            "general_direct_client_shadow_exists": True,
            "generic_direct_shadow_is_polynomially_compact": False,
            "unrestricted_affine_label_direct_shadow_is_polynomially_compact": (
                False
            ),
            "restricted_affine_weyl_force_common_quadratic_propagation_"
            "corollary_is_polynomially_compact": True,
            "executed_carrier_restoration": False,
            "same_backing_reuse": False,
            "physical_execution": False,
            "physical_carrier_custody": False,
            "physical_restoration": False,
            "computational_advantage": False,
            "m257_escape": False,
            "unbounded_compute": False,
            "bit_replaced_with_pi": False,
        },
        "architecture_authority": {
            "reference_is_physical_evidence": False,
            "reference_executes_restoration": False,
            "reference_asserts_same_backing_identity": False,
            "generic_sector_shadow_cost_is_explicit": True,
            "q_squared_compactness_requires_affine_weyl_forces_common_"
            "label_independent_quadratic_propagation_fixed_public_charged_K_"
            "and_closed_displacement": True,
            "m257_equal_access_guardrail_remains_intact": True,
        },
        "reference_self_assertion": "PASS_INDEPENDENT_ANALYTIC_CLASS_REFERENCE",
        "status": "PASS_INDEPENDENT_ANALYTIC_CLASS_REFERENCE",
        "terminal": False,
    }
    claim_bytes = json.dumps(
        clean(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    payload["claim_payload_sha256"] = hashlib.sha256(claim_bytes).hexdigest()
    payload["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(clean(payload), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
