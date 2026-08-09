#!/usr/bin/env python3
"""Independent finite-Fock reference for the M265 Weyl-loop bus.

This file deliberately contains its own client algebra, oscillator
construction, spectral pulse exponentiation, density reductions, fixtures,
and ideal-law oracle.  It is a numerical reference, not physical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


REFERENCE_ID = "M265_QUANTUM_WEYL_LOOP_BUS_SEPARATE_REFERENCE_V1"
CUTOFFS = (16, 32, 64, 128)
CLIENT_DIMENSION = 8

LAMBDA_A = 1.0 / 2.0
MU_A = math.pi / 4.0
THETA_A = LAMBDA_A * MU_A
LAMBDA_B = 2.0 / 3.0
MU_B = math.pi / 4.0
THETA_B = LAMBDA_B * MU_B

COHERENT_ALPHA = 0.65 + 0.20j
THERMAL_MEAN_OCCUPATION = 0.4
SQUEEZING_R = 0.45
SQUEEZING_PHASE = 0.30
PULSE_RELATIVE_MISMATCH = 1.0e-3
FREE_ROTATION_PER_GAP = 2.0e-3
KERR_PHASE_PER_GAP = 5.0e-4
DECLARED_LOSS_RATE_PER_GAP = 1.0e-3

I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
PLUS3 = np.ones(CLIENT_DIMENSION, dtype=np.complex128) / math.sqrt(CLIENT_DIMENSION)


def tensor3(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(first, second), third)


CLIENT_OPERATORS = {
    "X0": tensor3(X2, I2, I2),
    "X1": tensor3(I2, X2, I2),
    "X2": tensor3(I2, I2, X2),
    "Y0Z1": tensor3(Y2, Z2, I2),
    "Z0Y1": tensor3(Z2, Y2, I2),
    "Z1Y2": tensor3(I2, Z2, Y2),
    "X0X1": tensor3(X2, X2, I2),
    "X1X2": tensor3(I2, X2, X2),
}


def z_eigenvalue(basis_index: int, qubit: int) -> int:
    bit = (basis_index >> (2 - qubit)) & 1
    return 1 if bit == 0 else -1


def fock_quadratures(cutoff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    annihilation = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for number in range(1, cutoff):
        annihilation[number - 1, number] = math.sqrt(number)
    creation = annihilation.conj().T
    position = (annihilation + creation) / math.sqrt(2.0)
    momentum = (annihilation - creation) / (1.0j * math.sqrt(2.0))
    number_operator = np.diag(np.arange(cutoff, dtype=np.float64)).astype(np.complex128)
    return position, momentum, number_operator


def spectral_exponential(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    phases = np.exp(1.0j * coefficient * eigenvalues)
    return (eigenvectors * phases[np.newaxis, :]) @ eigenvectors.conj().T


def make_loop_family(
    position: np.ndarray,
    momentum: np.ndarray,
    lam: float,
    mu: float,
    *,
    closing_x_scale: float = 1.0,
    omit_closing_x: bool = False,
    gap_unitary: np.ndarray | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    x_values, x_vectors = np.linalg.eigh(position)
    p_values, p_vectors = np.linalg.eigh(momentum)
    identity = np.eye(position.shape[0], dtype=np.complex128)
    gap = identity if gap_unitary is None else gap_unitary
    loops: dict[tuple[int, int], np.ndarray] = {}
    for a_value in (-1, 1):
        for b_value in (-1, 1):
            # Chronological state action is U1, U2, U3, U4.  Matrix products
            # consequently place U4 on the left.
            pulse_1 = spectral_exponential(p_values, p_vectors, +mu * b_value)
            pulse_2 = spectral_exponential(x_values, x_vectors, +lam * a_value)
            pulse_3 = spectral_exponential(p_values, p_vectors, -mu * b_value)
            if omit_closing_x:
                pulse_4 = identity
            else:
                pulse_4 = spectral_exponential(
                    x_values,
                    x_vectors,
                    -lam * closing_x_scale * a_value,
                )
            loops[(a_value, b_value)] = pulse_4 @ gap @ pulse_3 @ gap @ pulse_2 @ gap @ pulse_1
    return loops


def apply_loop(
    state: np.ndarray,
    loops: dict[tuple[int, int], np.ndarray],
    first_qubit: int,
    second_qubit: int,
) -> np.ndarray:
    result = np.empty_like(state)
    for client_index in range(CLIENT_DIMENSION):
        a_value = z_eigenvalue(client_index, first_qubit)
        b_value = z_eigenvalue(client_index, second_qubit)
        result[client_index] = loops[(a_value, b_value)] @ state[client_index]
    return result


def ideal_phase_vector(theta_a: float = THETA_A, theta_b: float = THETA_B) -> np.ndarray:
    phases = np.empty(CLIENT_DIMENSION, dtype=np.complex128)
    for client_index in range(CLIENT_DIMENSION):
        za = z_eigenvalue(client_index, 0) * z_eigenvalue(client_index, 1)
        zb = z_eigenvalue(client_index, 1) * z_eigenvalue(client_index, 2)
        phases[client_index] = np.exp(-1.0j * (theta_a * za + theta_b * zb))
    return phases


def client_density(state: np.ndarray) -> np.ndarray:
    flattened = state.reshape(CLIENT_DIMENSION, -1)
    density = flattened @ flattened.conj().T
    return (density + density.conj().T) / 2.0


def bus_density(state: np.ndarray) -> np.ndarray:
    if state.ndim == 2:
        density = np.einsum("cn,cm->nm", state, state.conj(), optimize=True)
    elif state.ndim == 3:
        density = np.einsum("cnr,cmr->nm", state, state.conj(), optimize=True)
    else:
        raise ValueError(f"unsupported state rank {state.ndim}")
    return (density + density.conj().T) / 2.0


def trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    delta = (first - second + (first - second).conj().T) / 2.0
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))


def pure_trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    first_unit = first_flat / np.linalg.norm(first_flat)
    second_unit = second_flat / np.linalg.norm(second_flat)
    overlap = np.vdot(first_unit, second_unit)
    if abs(overlap) > 0.0:
        second_unit = second_unit * overlap.conjugate() / abs(overlap)
    # For normalized pure states, after phase alignment, d=||u-v|| and
    # D=sqrt(1-|<u|v>|^2)=d*sqrt(1-d^2/4).  This form remains accurate when
    # D is near machine epsilon, unlike subtracting |<u|v>|^2 from one.
    aligned_l2 = float(np.linalg.norm(first_unit - second_unit))
    return min(1.0, aligned_l2 * math.sqrt(max(0.0, 1.0 - aligned_l2**2 / 4.0)))


def vector_l2(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm((first - second).reshape(-1)))


def expectation(density: np.ndarray, operator: np.ndarray) -> float:
    value = np.trace(density @ operator)
    if abs(value.imag) > 2.0e-11:
        raise AssertionError(f"non-real Hermitian expectation {value}")
    return float(value.real)


def coherent_state(cutoff: int) -> np.ndarray:
    coefficients = np.empty(cutoff, dtype=np.complex128)
    coefficients[0] = np.exp(-0.5 * abs(COHERENT_ALPHA) ** 2)
    for number in range(1, cutoff):
        coefficients[number] = coefficients[number - 1] * COHERENT_ALPHA / math.sqrt(number)
    return coefficients / np.linalg.norm(coefficients)


def squeezed_vacuum(cutoff: int) -> np.ndarray:
    coefficients = np.zeros(cutoff, dtype=np.complex128)
    for pair_number in range((cutoff + 1) // 2):
        number = 2 * pair_number
        if number >= cutoff:
            break
        log_magnitude = (
            0.5 * math.lgamma(2 * pair_number + 1)
            - pair_number * math.log(2.0)
            - math.lgamma(pair_number + 1)
            - 0.5 * math.log(math.cosh(SQUEEZING_R))
            + pair_number * math.log(math.tanh(SQUEEZING_R))
        )
        coefficients[number] = math.exp(log_magnitude) * np.exp(
            1.0j * pair_number * (SQUEEZING_PHASE + math.pi)
        )
    return coefficients / np.linalg.norm(coefficients)


def pure_bus_fixtures(cutoff: int) -> dict[str, np.ndarray]:
    vacuum = np.zeros(cutoff, dtype=np.complex128)
    vacuum[0] = 1.0
    fock_one = np.zeros(cutoff, dtype=np.complex128)
    fock_one[1] = 1.0
    zero_plus_i_three = np.zeros(cutoff, dtype=np.complex128)
    zero_plus_i_three[0] = 1.0 / math.sqrt(2.0)
    zero_plus_i_three[3] = 1.0j / math.sqrt(2.0)
    return {
        "vacuum": vacuum,
        "coherent_0p65_plus_0p20i": coherent_state(cutoff),
        "squeezed_r0p45_phi0p30": squeezed_vacuum(cutoff),
        "fock_1": fock_one,
        "zero_plus_i_three": zero_plus_i_three,
    }


def thermal_purification(cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    ratio = THERMAL_MEAN_OCCUPATION / (THERMAL_MEAN_OCCUPATION + 1.0)
    probabilities = (1.0 - ratio) * ratio ** np.arange(cutoff, dtype=np.float64)
    probabilities /= float(np.sum(probabilities))
    purification = np.diag(np.sqrt(probabilities)).astype(np.complex128)
    return purification, probabilities


def phi4_bus_reference(cutoff: int) -> np.ndarray:
    purification = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for number in range(4):
        purification[number, number] = 0.5
    return purification


def initial_joint(bus_state: np.ndarray) -> np.ndarray:
    if bus_state.ndim == 1:
        return PLUS3[:, np.newaxis] * bus_state[np.newaxis, :]
    if bus_state.ndim == 2:
        return PLUS3[:, np.newaxis, np.newaxis] * bus_state[np.newaxis, :, :]
    raise ValueError("bus state must be a vector or bus-reference matrix")


def ideal_joint(initial: np.ndarray, include_query_b: bool) -> np.ndarray:
    if include_query_b:
        phases = ideal_phase_vector()
    else:
        phases = ideal_phase_vector(theta_b=0.0)
    reshape = (CLIENT_DIMENSION,) + (1,) * (initial.ndim - 1)
    return initial * phases.reshape(reshape)


def run_fixture(
    bus_state: np.ndarray,
    loops_a: dict[tuple[int, int], np.ndarray],
    loops_b: dict[tuple[int, int], np.ndarray],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    initial = initial_joint(bus_state)
    after_a = apply_loop(initial, loops_a, 0, 1)
    final = apply_loop(after_a, loops_b, 1, 2)
    ideal_a = ideal_joint(initial, include_query_b=False)
    ideal_final = ideal_joint(initial, include_query_b=True)
    initial_bus_density = bus_density(initial)
    record = {
        "query_a_joint_trace_distance_to_ideal": pure_trace_distance(after_a, ideal_a),
        "query_a_joint_vector_l2_to_ideal": vector_l2(after_a, ideal_a),
        "combined_joint_trace_distance_to_ideal": pure_trace_distance(final, ideal_final),
        "combined_joint_vector_l2_to_ideal": vector_l2(final, ideal_final),
        "combined_client_trace_distance_to_direct": trace_distance(
            client_density(final), client_density(ideal_final)
        ),
        "combined_bus_trace_distance_return": trace_distance(
            bus_density(final), initial_bus_density
        ),
    }
    return record, after_a, final


def client_direct_density(theta_a: float, theta_b: float) -> np.ndarray:
    state = PLUS3.copy()
    for client_index in range(CLIENT_DIMENSION):
        za = z_eigenvalue(client_index, 0) * z_eigenvalue(client_index, 1)
        zb = z_eigenvalue(client_index, 1) * z_eigenvalue(client_index, 2)
        state[client_index] *= np.exp(-1.0j * (theta_a * za + theta_b * zb))
    return np.outer(state, state.conj())


def top_fock_control(
    cutoff: int,
    position: np.ndarray,
    momentum: np.ndarray,
    loops_a: dict[tuple[int, int], np.ndarray],
) -> dict[str, float]:
    identity = np.eye(cutoff, dtype=np.complex128)
    commutator_defect = position @ momentum - momentum @ position - 1.0j * identity
    defect_norm = float(np.linalg.norm(commutator_defect, ord=2))
    top = np.zeros(cutoff, dtype=np.complex128)
    top[-1] = 1.0
    actual = loops_a[(1, 1)] @ top
    ideal = np.exp(-1.0j * THETA_A) * top
    return {
        "ccr_defect_operator_norm": defect_norm,
        "ccr_defect_expected_exact": float(cutoff),
        "top_fock_return_trace_distance": pure_trace_distance(actual, ideal),
        "top_fock_vector_l2_to_ideal": vector_l2(actual, ideal),
    }


def sham_dephasing_control(probabilities: np.ndarray) -> dict[str, float]:
    amplitudes = np.sqrt(probabilities)
    ideal_subspace_density = np.outer(amplitudes, amplitudes)
    dephased_subspace_density = np.diag(probabilities)
    return {
        "bus_marginal_trace_distance": 0.0,
        "bus_reference_trace_distance": trace_distance(
            ideal_subspace_density, dephased_subspace_density
        ),
        "purity_before": 1.0,
        "purity_after": float(np.sum(probabilities**2)),
        "fixed_marginal_does_not_imply_identity_channel": True,
    }


def number_expectation(state: np.ndarray, number_operator: np.ndarray) -> float:
    return float(np.vdot(state, number_operator @ state).real)


def excursion_account(
    position: np.ndarray,
    momentum: np.ndarray,
    number_operator: np.ndarray,
    lam: float,
    mu: float,
) -> dict[str, float]:
    x_values, x_vectors = np.linalg.eigh(position)
    p_values, p_vectors = np.linalg.eigh(momentum)
    vacuum = np.zeros(position.shape[0], dtype=np.complex128)
    vacuum[0] = 1.0
    maxima: list[float] = []
    exposures: list[float] = []
    for a_value in (-1, 1):
        for b_value in (-1, 1):
            pulses = (
                spectral_exponential(p_values, p_vectors, +mu * b_value),
                spectral_exponential(x_values, x_vectors, +lam * a_value),
                spectral_exponential(p_values, p_vectors, -mu * b_value),
                spectral_exponential(x_values, x_vectors, -lam * a_value),
            )
            state = vacuum.copy()
            path_occupations = []
            for pulse in pulses:
                state = pulse @ state
                path_occupations.append(number_expectation(state, number_operator))
            maxima.append(max(path_occupations))
            exposures.append(sum(path_occupations[:-1]))
    maximum = max(maxima)
    exposure = max(exposures)
    return {
        "maximum_vacuum_path_mean_occupation": maximum,
        "three_gap_occupation_exposure_proxy": exposure,
        "declared_loss_rate_per_gap": DECLARED_LOSS_RATE_PER_GAP,
        "loss_which_path_proxy_not_a_channel_simulation": DECLARED_LOSS_RATE_PER_GAP
        * exposure,
    }


def perturbation_controls(
    cutoff: int,
    position: np.ndarray,
    momentum: np.ndarray,
    number_operator: np.ndarray,
    ideal_loops_a: dict[tuple[int, int], np.ndarray],
) -> dict[str, object]:
    vacuum = pure_bus_fixtures(cutoff)["vacuum"]
    initial = initial_joint(vacuum)
    ideal = ideal_joint(initial, include_query_b=False)
    initial_bus_density = bus_density(initial)

    mismatch_loops = make_loop_family(
        position,
        momentum,
        LAMBDA_A,
        MU_A,
        closing_x_scale=1.0 + PULSE_RELATIVE_MISMATCH,
    )
    mismatch_state = apply_loop(initial, mismatch_loops, 0, 1)

    omitted_loops = make_loop_family(
        position,
        momentum,
        LAMBDA_A,
        MU_A,
        omit_closing_x=True,
    )
    omitted_state = apply_loop(initial, omitted_loops, 0, 1)

    numbers = np.arange(cutoff, dtype=np.float64)
    gap_phases = np.exp(
        -1.0j
        * (
            FREE_ROTATION_PER_GAP * numbers
            + KERR_PHASE_PER_GAP * numbers * (numbers - 1.0)
        )
    )
    gap = np.diag(gap_phases).astype(np.complex128)
    gap_loops = make_loop_family(
        position,
        momentum,
        LAMBDA_A,
        MU_A,
        gap_unitary=gap,
    )
    gap_state = apply_loop(initial, gap_loops, 0, 1)

    ideal_state = apply_loop(initial, ideal_loops_a, 0, 1)
    return {
        "pulse_relative_mismatch": PULSE_RELATIVE_MISMATCH,
        "mismatch_joint_trace_distance_to_ideal": pure_trace_distance(mismatch_state, ideal),
        "mismatch_bus_trace_distance_return": trace_distance(
            bus_density(mismatch_state), initial_bus_density
        ),
        "omitted_closing_pulse_joint_trace_distance_to_ideal": pure_trace_distance(
            omitted_state, ideal
        ),
        "omitted_closing_pulse_bus_trace_distance_return": trace_distance(
            bus_density(omitted_state), initial_bus_density
        ),
        "free_rotation_per_gap": FREE_ROTATION_PER_GAP,
        "kerr_phase_per_gap": KERR_PHASE_PER_GAP,
        "free_plus_kerr_joint_trace_distance_to_ideal": pure_trace_distance(gap_state, ideal),
        "free_plus_kerr_bus_trace_distance_return": trace_distance(
            bus_density(gap_state), initial_bus_density
        ),
        "unperturbed_joint_trace_distance_to_ideal": pure_trace_distance(ideal_state, ideal),
        "loss_channel_simulated": False,
        "loss_proxy_is_physical_evidence": False,
        "number_operator_trace": float(np.trace(number_operator).real),
    }


def main() -> int:
    start = time.perf_counter()
    cutoff_records: dict[str, object] = {}
    largest_after_a: np.ndarray | None = None
    largest_final: np.ndarray | None = None
    largest_initial: np.ndarray | None = None
    largest_position: np.ndarray | None = None
    largest_momentum: np.ndarray | None = None
    largest_number: np.ndarray | None = None
    largest_loops_a: dict[tuple[int, int], np.ndarray] | None = None

    for cutoff in CUTOFFS:
        position, momentum, number_operator = fock_quadratures(cutoff)
        loops_a = make_loop_family(position, momentum, LAMBDA_A, MU_A)
        loops_b = make_loop_family(position, momentum, LAMBDA_B, MU_B)

        fixtures: dict[str, object] = {}
        for name, bus_state in pure_bus_fixtures(cutoff).items():
            fixture_record, after_a, final = run_fixture(bus_state, loops_a, loops_b)
            fixtures[name] = fixture_record
            if cutoff == CUTOFFS[-1] and name == "vacuum":
                largest_after_a = after_a
                largest_final = final
                largest_initial = initial_joint(bus_state)

        thermal, _thermal_probabilities = thermal_purification(cutoff)
        fixtures["thermal_nbar_0p4"] = run_fixture(thermal, loops_a, loops_b)[0] | {
            "mixed_state_implemented_by_bus_reference_purification": True
        }
        phi4 = phi4_bus_reference(cutoff)
        fixtures["phi4_bus_reference"] = run_fixture(phi4, loops_a, loops_b)[0] | {
            "bus_reference_dimension": cutoff,
            "occupied_schmidt_rank": 4,
        }

        cutoff_records[str(cutoff)] = {
            "fixtures": fixtures,
            "top_fock_nonuniformity_control": top_fock_control(
                cutoff, position, momentum, loops_a
            ),
            "joint_client_bus_dimension": CLIENT_DIMENSION * cutoff,
            "purified_joint_dimension": CLIENT_DIMENSION * cutoff * cutoff,
        }
        if cutoff == CUTOFFS[-1]:
            largest_position = position
            largest_momentum = momentum
            largest_number = number_operator
            largest_loops_a = loops_a

    if any(
        item is None
        for item in (
            largest_after_a,
            largest_final,
            largest_initial,
            largest_position,
            largest_momentum,
            largest_number,
            largest_loops_a,
        )
    ):
        raise AssertionError("largest-cutoff witnesses were not retained")

    assert largest_after_a is not None
    assert largest_final is not None
    assert largest_initial is not None
    assert largest_position is not None
    assert largest_momentum is not None
    assert largest_number is not None
    assert largest_loops_a is not None

    rho_after_a = client_density(largest_after_a)
    rho_final = client_density(largest_final)
    rho_direct_a = client_direct_density(THETA_A, 0.0)
    rho_direct_final = client_direct_density(THETA_A, THETA_B)
    rho_wrong_sign = client_direct_density(-THETA_A, -THETA_B)
    boundary_moments = {
        name: expectation(rho_final, operator) for name, operator in CLIENT_OPERATORS.items()
    }
    direct_boundary_moments = {
        name: expectation(rho_direct_final, operator)
        for name, operator in CLIENT_OPERATORS.items()
    }
    phase_sign = {
        "query_a_Y0Z1": expectation(rho_after_a, CLIENT_OPERATORS["Y0Z1"]),
        "query_a_direct_Y0Z1": expectation(rho_direct_a, CLIENT_OPERATORS["Y0Z1"]),
        "combined_client_trace_distance_to_correct_sign": trace_distance(
            rho_final, rho_direct_final
        ),
        "combined_client_trace_distance_to_opposite_sign": trace_distance(
            rho_final, rho_wrong_sign
        ),
        "combined_boundary_moments": boundary_moments,
        "direct_compiled_boundary_moments": direct_boundary_moments,
    }

    _, thermal_probabilities = thermal_purification(CUTOFFS[-1])
    phi4_probabilities = np.zeros(CUTOFFS[-1], dtype=np.float64)
    phi4_probabilities[:4] = 0.25
    shams = {
        "thermal_dephasing": sham_dephasing_control(thermal_probabilities),
        "phi4_dephasing": sham_dephasing_control(phi4_probabilities),
    }

    perturbations = perturbation_controls(
        CUTOFFS[-1],
        largest_position,
        largest_momentum,
        largest_number,
        largest_loops_a,
    )
    energy_account = {
        "query_a": excursion_account(
            largest_position,
            largest_momentum,
            largest_number,
            LAMBDA_A,
            MU_A,
        ),
        "query_b": excursion_account(
            largest_position,
            largest_momentum,
            largest_number,
            LAMBDA_B,
            MU_B,
        ),
    }

    last = cutoff_records[str(CUTOFFS[-1])]
    assert isinstance(last, dict)
    last_fixtures = last["fixtures"]
    assert isinstance(last_fixtures, dict)
    low_energy_final_errors = [
        float(record["combined_joint_trace_distance_to_ideal"])
        for record in last_fixtures.values()
        if isinstance(record, dict)
    ]
    convergence_fixture_names = (
        "coherent_0p65_plus_0p20i",
        "squeezed_r0p45_phi0p30",
        "thermal_nbar_0p4",
        "phi4_bus_reference",
    )
    cutoff_16_fixtures = cutoff_records["16"]["fixtures"]
    cutoff_128_fixtures = cutoff_records["128"]["fixtures"]
    converged_l2_by_fixture = {
        name: float(cutoff_128_fixtures[name]["combined_joint_vector_l2_to_ideal"])
        < float(cutoff_16_fixtures[name]["combined_joint_vector_l2_to_ideal"])
        for name in convergence_fixture_names
    }
    top_errors = [
        float(
            cutoff_records[str(cutoff)]["top_fock_nonuniformity_control"][
                "top_fock_return_trace_distance"
            ]
        )
        for cutoff in CUTOFFS
    ]
    ccr_defect_errors = [
        abs(
            float(
                cutoff_records[str(cutoff)]["top_fock_nonuniformity_control"][
                    "ccr_defect_operator_norm"
                ]
            )
            - cutoff
        )
        for cutoff in CUTOFFS
    ]

    checks = {
        "theta_a_exact_pi_over_8": abs(THETA_A - math.pi / 8.0) <= 2.0e-16,
        "theta_b_exact_pi_over_6": abs(THETA_B - math.pi / 6.0) <= 2.0e-16,
        "largest_cutoff_low_energy_joint_error_le_2e_minus_7": max(
            low_energy_final_errors
        )
        <= 2.0e-7,
        "declared_low_energy_fixtures_improve_from_n16_to_n128": all(
            converged_l2_by_fixture.values()
        ),
        "largest_cutoff_correct_sign_client_error_le_2e_minus_10": phase_sign[
            "combined_client_trace_distance_to_correct_sign"
        ]
        <= 2.0e-10,
        "opposite_sign_is_rejected": phase_sign[
            "combined_client_trace_distance_to_opposite_sign"
        ]
        >= 0.5,
        "ccr_defect_norm_equals_cutoff": max(ccr_defect_errors) <= 5.0e-12,
        "top_fock_witness_rejects_uniform_return": min(top_errors) >= 0.10,
        "phi4_sham_has_identical_marginal": shams["phi4_dephasing"][
            "bus_marginal_trace_distance"
        ]
        == 0.0,
        "phi4_reference_detects_sham": shams["phi4_dephasing"][
            "bus_reference_trace_distance"
        ]
        >= 0.70,
        "omitted_pulse_is_detected": perturbations[
            "omitted_closing_pulse_joint_trace_distance_to_ideal"
        ]
        >= 0.1,
        "mismatched_pulse_is_detected": perturbations[
            "mismatch_joint_trace_distance_to_ideal"
        ]
        >= 1.0e-5,
        "free_rotation_and_kerr_are_detected": perturbations[
            "free_plus_kerr_joint_trace_distance_to_ideal"
        ]
        >= 1.0e-4,
    }

    record: dict[str, object] = {
        "reference_id": REFERENCE_ID,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "PASS_SEPARATE_REFERENCE" if all(checks.values()) else "FAIL_SEPARATE_REFERENCE",
        "model": {
            "client_qubits": 3,
            "query_a": {
                "first_generator": "Z0",
                "second_generator": "Z1",
                "lambda": LAMBDA_A,
                "mu": MU_A,
                "theta": THETA_A,
            },
            "query_b": {
                "first_generator": "Z1",
                "second_generator": "Z2",
                "lambda": LAMBDA_B,
                "mu": MU_B,
                "theta": THETA_B,
            },
            "chronological_pulses": [
                "exp(+i mu B P)",
                "exp(+i lambda A X)",
                "exp(-i mu B P)",
                "exp(-i lambda A X)",
            ],
            "ideal_compiled_law": "exp(-i*pi/6*Z1Z2) exp(-i*pi/8*Z0Z1) |+++>",
            "cutoffs": list(CUTOFFS),
            "precision": "complex128",
        },
        "execution_custody_semantics": {
            "same_three_client_state_used_for_query_a_then_query_b": True,
            "query_b_consumes_query_a_joint_client_bus_result": True,
            "fresh_client_detach_between_queries": False,
            "query_a_metrics_are_verifier_only": True,
            "query_a_boundary_released": False,
            "combined_final_client_boundary_released": True,
            "software_same_backing_claim": False,
        },
        "cutoff_sweep": cutoff_records,
        "declared_cutoff_convergence": converged_l2_by_fixture,
        "phase_sign_and_boundaries": phase_sign,
        "bus_reference_shams": shams,
        "perturbation_controls": perturbations,
        "energy_and_loss_account": energy_account,
        "checks": checks,
        "claims": {
            "ideal_regular_ccr_factorization_is_analytic_law": True,
            "finite_cutoff_exact_arbitrary_state_return": False,
            "uniform_arbitrary_state_return": False,
            "same_backing_restoration": False,
            "physical_restoration": False,
            "physical_execution": False,
            "computational_advantage": False,
            "m257_escape": False,
            "unbounded_compute": False,
            "bit_replaced_with_pi": False,
            "restoration_classification": "NO_RESTORATION_CLAIM",
            "finite_model_classification": "ENERGY_CONSTRAINED_CUTOFF_CONVERGENCE_ONLY",
        },
        "resource_accounting": {
            "reference_algorithm": "dense_complex128_fock_quadratures_plus_independent_hermitian_eigendecomposition_and_spectral_pulses",
            "maximum_joint_client_bus_dimension": CLIENT_DIMENSION * CUTOFFS[-1],
            "maximum_purified_client_bus_reference_dimension": CLIENT_DIMENSION
            * CUTOFFS[-1]
            * CUTOFFS[-1],
            "minimum_named_dense_complex_cells_at_maximum": 11 * CUTOFFS[-1] ** 2,
            "exact_peak_temporary_storage_claimed": False,
            "spectral_dense_work_scaling": "O(sum_N N^3) with explicit client blocks",
            "retained_initial_fixture_for_verification": True,
            "verification_fixture_used_for_state_transition": False,
            "direct_compiled_shadow": {
                "client_dimension": CLIENT_DIMENSION,
                "stored_complex_phases": CLIENT_DIMENSION,
                "bus_dimension": 0,
                "inverse_or_restoration_stage": False,
                "exact_final_client_law": True,
            },
            "oracle_is_charged": True,
            "resource_advantage_claim": False,
        },
        "ceiling": "FINITE_COMPLEX128_DETERMINISTIC_SOFTWARE_REFERENCE_WITH_TRUNCATED_FOCK_CCR_AND_DIRECT_CLIENT_ORACLE_NO_PHYSICAL_OR_SAME_BACKING_RESTORATION",
        "disposition": "IDEAL_WEYL_LOOP_IS_A_CATALYTIC_INTERACTION_LAW_BUT_FINITE_CUTOFF_RETURN_IS_ONLY_ENERGY_CONSTRAINED_AND_THE_EXACT_DIRECT_COMPILED_CLIENT_SHADOW_RETAINS_M257",
    }
    claim_payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    record["claim_payload_sha256"] = hashlib.sha256(claim_payload.encode("utf-8")).hexdigest()
    record["wall_seconds_not_used_for_claim"] = time.perf_counter() - start
    print(json.dumps(record, sort_keys=True, indent=2, allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
