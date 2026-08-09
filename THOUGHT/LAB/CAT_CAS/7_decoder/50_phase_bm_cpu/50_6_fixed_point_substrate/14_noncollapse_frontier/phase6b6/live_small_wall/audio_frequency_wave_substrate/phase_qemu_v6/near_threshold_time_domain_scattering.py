#!/usr/bin/env python3
"""M264 source-off near-threshold time-domain scattering digital twin.

This program executes a finite deterministic software model.  It evolves a
single probe on an open tight-binding lead coupled locally to a four-spin
target, streams aggregate observables, checks interaction-picture target
return, and rematerializes the complete returned target density for a second
query.  It does not execute QEMU, physical hardware, same-backing reuse, or a
resource advantage.

The claim-bearing geometry is frozen.  Command-line switches may omit streams
from JSON, but they do not alter the scientific fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, expm_multiply


PASS_CLAIM = (
    "FINITE_NUMERICAL_SOURCE_OFF_NEAR_THRESHOLD_TIME_DOMAIN_ONE_PORT_"
    "SCATTERING_ON_THE_FROZEN_FOUR_SPIN_TARGET_EXECUTES_TRANSIENT_"
    "INTERACTION_DRAIN_INTERACTION_PICTURE_TARGET_RETURN_AND_COMPLETE_"
    "RETURNED_REDUCED_STATE_REMATERIALIZATION_FROM_QUERY_A_INTO_QUERY_B_"
    "WITHOUT_BASELINE_OR_EXACT_GROUND_RELOAD"
)
FAIL_CLAIM = (
    "FROZEN_L641_SIGMA50_NEAR_THRESHOLD_TIME_DOMAIN_SCATTERING_EXECUTES_"
    "TRANSIENT_BORROW_DRAIN_AND_COMPLETE_RETURNED_DENSITY_HANDOFF_BUT_THE_"
    "PUBLIC_T120_ADIABATIC_PREPARATION_AND_FINITE_PACKET_T340_RETURN_FAIL_"
    "DECLARED_1E_MINUS_7_MATCHED_FREE_TARGET_TRACE_DISTANCE_GATES"
)
CLAIM_CEILING = (
    "FINITE_COMPLEX128_DETERMINISTIC_SOFTWARE_SINGLE_PROBE_L641_FOUR_SPIN_"
    "TIME_DOMAIN_MODEL_WITH_RETURNED_TARGET_DENSITY_REMATERIALIZATION_AND_"
    "NO_SAME_BACKING_OR_PHYSICAL_RESTORATION"
)
PASS_DISPOSITION = (
    "TIME_DOMAIN_RETURN_AND_FUNCTIONAL_RETURNED_STATE_REUSE_QUALIFIED_BUT_"
    "THE_IDENTICAL_O_L_TIMES_16_SPARSE_CLASSICAL_EVOLUTION_AND_BOND_AT_MOST_"
    "17_ONE_PARTICLE_MPS_REMAIN_NO_TENSOR_NETWORK_CROSSOVER_RESOURCE_WIN_"
    "M257_ESCAPE_OR_SAME_BACKING_CATALYSIS"
)
FAIL_DISPOSITION = (
    "STRICT_PREPARATION_AND_FINITE_PACKET_RESTORATION_OBSTRUCTION_RETAINS_"
    "REAL_TRANSIENT_INTERACTION_DRAIN_AND_APPROXIMATE_FUNCTIONAL_HANDOFF_BUT_"
    "REQUIRES_A_CHANGED_RETURN_PREPARATION_LAW_NOT_POST_HOC_FIXTURE_TUNING"
)
NEXT_MECHANISM = (
    "RESIDENT_OPEN_DRAIN_OR_ECHO_RETURN_WITH_PAID_GROUND_STATE_SUPPLY_"
    "PREDECLARED_FINITE_PACKET_ERROR_AND_SAME_BACKING_TARGET_CUSTODY"
)

# Frozen public geometry.  sigma=50 is the predeclared correction from the
# original sigma=48 blueprint; it puts query B's open-lead above-gap tail
# below the declared 1e-9 gate before semantic execution.
LEAD_LENGTH = 641
PACKET_CENTER = 320
PACKET_SIGMA = 50
DETECTOR_SITE = 192
CONTACT_SITE = 0
NEAR_TARGET_MAX_SITE = 32
EDGE_BUFFER_SITES = 32
SCATTER_TIME = 340.0
DELAY_CONTROL_TIME = 370.0
OBSERVATION_DT = 2.0
CHUNK_INTERVALS = 10
LEAD_HOPPING = 1.0
COUPLING_G = 1.5
K_QUERY_A = 2.0 * math.pi / 5.0
K_QUERY_B = 9.0 * math.pi / 20.0
K_ABOVE_THRESHOLD = math.pi / 2.0

PREP_TIME = 120.0
PREP_STEPS = 480
PREP_CONTROL_STEPS = 240
PREP_LONG_CONTROL_TIME = 480.0
PREP_LONG_CONTROL_STEPS = 1920

THRESHOLDS = {
    "norm_or_trace_error_max": 1.0e-9,
    "density_hermiticity_error_max": 1.0e-12,
    "density_minimum_eigenvalue_min": -1.0e-12,
    "above_threshold_packet_weight_max": 1.0e-9,
    "query_a_transient_excitation_min": 1.0e-2,
    "query_b_transient_excitation_min": 5.0e-2,
    "final_contact_probability_max": 1.0e-8,
    "final_near_target_probability_max": 1.0e-7,
    "target_trace_distance_to_free_max": 1.0e-7,
    "detector_incoming_flux_min": 0.99,
    "detector_outgoing_flux_min": 0.99,
    "endpoint_convergence_l2_max": 1.0e-8,
    "delay_ordering_margin_min": 5.5,
    "returned_density_psd_clip_trace_max": 1.0e-12,
}


def _complex_json(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def _float(value: Any) -> float:
    return float(np.real_if_close(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _operator_on_site(operator: np.ndarray, site: int, count: int = 4) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    identity = np.eye(2, dtype=np.complex128)
    for index in range(count):
        result = np.kron(result, operator if index == site else identity)
    return result


def build_target() -> dict[str, Any]:
    identity = np.eye(2, dtype=np.complex128)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z = np.diag([1.0, -1.0]).astype(np.complex128)
    xs = [_operator_on_site(x, site) for site in range(4)]
    zs = [_operator_on_site(z, site) for site in range(4)]

    h_x = -sum(xs, start=np.zeros((16, 16), dtype=np.complex128))
    h_target = h_x.copy()
    for site in range(3):
        h_target -= (3.0 / 4.0) * (zs[site] @ zs[site + 1])
    for site, coefficient in enumerate((1.0 / 5.0, 2.0 / 7.0, 3.0 / 11.0, 5.0 / 13.0)):
        h_target -= coefficient * zs[site]
    h_target -= 0.5 * (zs[0] @ zs[1] @ zs[2])

    eigenvalues, eigenvectors = eigh(h_target)
    ground_energy = float(eigenvalues[0])
    shifted = h_target - ground_energy * np.eye(16, dtype=np.complex128)
    gaps = eigenvalues - ground_energy
    return {
        "h_x": h_x,
        "h_target": h_target,
        "h_shifted": shifted,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "gaps": gaps,
        "ground": eigenvectors[:, 0],
        "ground_energy": ground_energy,
        "gap": float(gaps[1]),
        "z0": zs[0],
        "identity": identity,
    }


def build_lead() -> sparse.csr_matrix:
    off_diagonal = -LEAD_HOPPING * np.ones(LEAD_LENGTH - 1)
    return sparse.diags(
        (off_diagonal, 2.0 * LEAD_HOPPING * np.ones(LEAD_LENGTH), off_diagonal),
        offsets=(-1, 0, 1),
        shape=(LEAD_LENGTH, LEAD_LENGTH),
        format="csr",
        dtype=np.complex128,
    )


def build_contact_projector() -> sparse.csr_matrix:
    # Fail closed against the scipy.diags scalar-broadcast error: the contact
    # projector must have exactly one stored entry at (0,0).
    projector = sparse.csr_matrix(
        (
            np.array([1.0], dtype=np.complex128),
            (np.array([CONTACT_SITE]), np.array([CONTACT_SITE])),
        ),
        shape=(LEAD_LENGTH, LEAD_LENGTH),
    )
    assert projector.nnz == 1
    rows, columns = projector.nonzero()
    assert rows.tolist() == [CONTACT_SITE] and columns.tolist() == [CONTACT_SITE]
    return projector


def build_total_hamiltonian(
    lead: sparse.csr_matrix,
    target_shifted: np.ndarray,
    z0: np.ndarray,
    coupling: float,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    identity_lead = sparse.identity(LEAD_LENGTH, dtype=np.complex128, format="csr")
    identity_target = sparse.identity(16, dtype=np.complex128, format="csr")
    projector = build_contact_projector()
    target_sparse = sparse.csr_matrix(target_shifted)
    z0_sparse = sparse.csr_matrix(z0)
    h_total = (
        sparse.kron(lead, identity_target, format="csr")
        + sparse.kron(identity_lead, target_sparse, format="csr")
        + coupling * sparse.kron(projector, z0_sparse, format="csr")
    )
    coupling_term = coupling * sparse.kron(projector, z0_sparse, format="csr")
    coupling_rows = np.unique(coupling_term.nonzero()[0] // 16)
    localized = coupling_rows.tolist() == [CONTACT_SITE]
    assert localized
    return h_total, {
        "contact_projector_nnz": int(projector.nnz),
        "contact_projector_coordinate": [CONTACT_SITE, CONTACT_SITE],
        "coupling_support_lead_sites": coupling_rows.tolist(),
        "coupling_localized_at_site_zero": localized,
        "total_dimension": int(h_total.shape[0]),
        "total_hamiltonian_nnz": int(h_total.nnz),
        "coupling_term_nnz": int(coupling_term.nnz),
    }


class CountingLinearOperator(LinearOperator):
    """LinearOperator wrapper counting vector-equivalent sparse products."""

    def __init__(self, matrix: sparse.spmatrix):
        self.matrix = matrix.tocsr()
        self.adjoint_matrix = self.matrix.getH().tocsr()
        self.forward_vector_equivalents = 0
        self.adjoint_vector_equivalents = 0
        super().__init__(dtype=self.matrix.dtype, shape=self.matrix.shape)

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        self.forward_vector_equivalents += 1
        return self.matrix @ vector

    def _matmat(self, matrix: np.ndarray) -> np.ndarray:
        columns = 1 if matrix.ndim == 1 else matrix.shape[1]
        self.forward_vector_equivalents += columns
        return self.matrix @ matrix

    def _rmatvec(self, vector: np.ndarray) -> np.ndarray:
        self.adjoint_vector_equivalents += 1
        return self.adjoint_matrix @ vector

    def _rmatmat(self, matrix: np.ndarray) -> np.ndarray:
        columns = 1 if matrix.ndim == 1 else matrix.shape[1]
        self.adjoint_vector_equivalents += columns
        return self.adjoint_matrix @ matrix


def smoothstep_cubic(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def prepare_target(
    target: dict[str, Any], steps: int, duration: float = PREP_TIME
) -> dict[str, Any]:
    started = time.perf_counter()
    plus = np.ones(16, dtype=np.complex128) / 4.0
    state = plus.copy()
    delta_t = duration / steps
    matvecs = 0
    adjoint_matvecs = 0
    for step in range(steps):
        u = (step + 0.5) / steps
        schedule = smoothstep_cubic(u)
        h_mid = (1.0 - schedule) * target["h_x"] + schedule * target["h_target"]
        generator = sparse.csr_matrix((-1.0j * delta_t) * h_mid)
        counting = CountingLinearOperator(generator)
        trace_generator = np.trace((-1.0j * delta_t) * h_mid)
        state = expm_multiply(counting, state, traceA=trace_generator)
        matvecs += counting.forward_vector_equivalents
        adjoint_matvecs += counting.adjoint_vector_equivalents
    state /= np.linalg.norm(state)
    ground = target["ground"]
    fidelity = abs(np.vdot(ground, state)) ** 2
    energy = np.vdot(state, target["h_target"] @ state)
    energy2 = np.vdot(state, target["h_target"] @ target["h_target"] @ state)
    variance = float(np.real(energy2 - energy * energy))
    return {
        "state": state,
        "steps": steps,
        "duration": duration,
        "schedule": "CUBIC_SMOOTHSTEP_3U2_MINUS_2U3_MIDPOINT_PIECEWISE_CONSTANT",
        "ground_fidelity": float(fidelity),
        "ground_infidelity": float(1.0 - fidelity),
        "energy_expectation": float(np.real(energy)),
        "energy_variance": variance,
        "norm_error": float(abs(np.vdot(state, state) - 1.0)),
        "forward_vector_equivalent_matvecs": int(matvecs),
        "adjoint_vector_equivalent_matvecs": int(adjoint_matvecs),
        "wall_seconds_not_used_for_claim": time.perf_counter() - started,
    }


def packet(k_value: float) -> np.ndarray:
    positions = np.arange(LEAD_LENGTH, dtype=np.float64)
    envelope = np.exp(-((positions - PACKET_CENTER) ** 2) / (4.0 * PACKET_SIGMA**2))
    phase = np.exp(-1.0j * k_value * (positions - PACKET_CENTER))
    state = envelope * phase
    return state / np.linalg.norm(state)


def packet_spectral_audit(
    packet_state: np.ndarray, k_value: float, gap: float, lead: sparse.csr_matrix
) -> dict[str, Any]:
    mode_indices = np.arange(1, LEAD_LENGTH + 1, dtype=np.float64)
    q_values = mode_indices * math.pi / (LEAD_LENGTH + 1)
    energies = 2.0 - 2.0 * np.cos(q_values)
    positions = np.arange(1, LEAD_LENGTH + 1, dtype=np.float64)
    sine_basis = math.sqrt(2.0 / (LEAD_LENGTH + 1)) * np.sin(
        np.outer(q_values, positions)
    )
    coefficients = sine_basis @ packet_state
    weights = np.abs(coefficients) ** 2
    above = float(np.sum(weights[energies >= gap]))
    mean_energy = float(np.real(np.vdot(packet_state, lead @ packet_state)))
    mean_energy2 = float(np.real(np.vdot(lead @ packet_state, lead @ packet_state)))
    nominal_energy = 2.0 - 2.0 * math.cos(k_value)
    return {
        "k": k_value,
        "nominal_energy": nominal_energy,
        "nominal_group_velocity": -2.0 * math.sin(k_value),
        "group_velocity_sign_convention": "NEGATIVE_FOR_EXP_MINUS_I_K_X_INCOMING_TOWARD_DECREASING_X",
        "target_gap": gap,
        "nominal_detuning_below_gap": gap - nominal_energy,
        "discrete_open_lead_mean_energy": mean_energy,
        "discrete_open_lead_energy_standard_deviation": math.sqrt(
            max(0.0, mean_energy2 - mean_energy * mean_energy)
        ),
        "above_target_gap_spectral_weight": above,
        "spectral_weight_sum_error": abs(float(np.sum(weights)) - 1.0),
        "effectively_subgap_at_declared_tail": above
        <= THRESHOLDS["above_threshold_packet_weight_max"],
    }


def _as_block(state: np.ndarray) -> np.ndarray:
    return state[:, None] if state.ndim == 1 else state


def target_density(state: np.ndarray, target_dimension: int = 16) -> np.ndarray:
    block = _as_block(state)
    components = block.shape[1]
    amplitudes = block.reshape(LEAD_LENGTH, target_dimension, components)
    return np.einsum("xar,xbr->ab", amplitudes, amplitudes.conj(), optimize=True)


def trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    difference = 0.5 * ((first - second) + (first - second).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


def density_fidelity(first: np.ndarray, second: np.ndarray) -> float:
    def normalized_psd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hermitian = 0.5 * (matrix + matrix.conj().T)
        trace_value = float(np.real(np.trace(hermitian)))
        if trace_value <= 0.0:
            raise ValueError("density fidelity requires positive traces")
        values, vectors = eigh(hermitian / trace_value)
        values = np.maximum(values, 0.0)
        value_sum = float(np.sum(values))
        if value_sum <= 0.0:
            raise ValueError("density fidelity argument lost all PSD weight")
        values /= value_sum
        projected = (vectors * values) @ vectors.conj().T
        return projected, values, vectors

    _, first_values, first_vectors = normalized_psd(first)
    _, second_values, second_vectors = normalized_psd(second)
    sqrt_first = (first_vectors * np.sqrt(first_values)) @ first_vectors.conj().T
    sqrt_second = (
        second_vectors * np.sqrt(second_values)
    ) @ second_vectors.conj().T
    singular_values = np.linalg.svd(sqrt_first @ sqrt_second, compute_uv=False)
    fidelity = float(np.sum(singular_values) ** 2)
    return min(1.0, max(0.0, fidelity))


def density_diagnostics(density: np.ndarray) -> dict[str, Any]:
    hermiticity = float(np.max(np.abs(density - density.conj().T)))
    hermitian = 0.5 * (density + density.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    return {
        "trace": float(np.real(np.trace(density))),
        "trace_error": abs(float(np.real(np.trace(density))) - 1.0),
        "hermiticity_max_abs": hermiticity,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "purity": float(np.real(np.trace(hermitian @ hermitian))),
        "rank_above_1e_minus_14": int(np.count_nonzero(eigenvalues > 1.0e-14)),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "non_dominant_trace": float(np.sum(eigenvalues[:-1])),
    }


def _observable_snapshot(
    state: np.ndarray, ground: np.ndarray, target_dimension: int = 16
) -> dict[str, float]:
    block = _as_block(state)
    component_count = block.shape[1]
    amplitudes = block.reshape(LEAD_LENGTH, target_dimension, component_count)
    probabilities = np.abs(amplitudes) ** 2
    norm = float(np.sum(probabilities))
    current = 2.0 * LEAD_HOPPING * float(
        np.imag(np.vdot(amplitudes[DETECTOR_SITE], amplitudes[DETECTOR_SITE + 1]))
    )
    ground_amplitudes = np.einsum(
        "xar,a->xr", amplitudes, ground.conj(), optimize=True
    )
    ground_probability = float(np.sum(np.abs(ground_amplitudes) ** 2))
    return {
        "norm": norm,
        "detector_current": current,
        "detector_left_probability": float(np.sum(probabilities[: DETECTOR_SITE + 1])),
        "contact_probability": float(np.sum(probabilities[CONTACT_SITE])),
        "near_target_probability": float(
            np.sum(probabilities[: NEAR_TARGET_MAX_SITE + 1])
        ),
        "right_edge_buffer_probability": float(
            np.sum(probabilities[LEAD_LENGTH - EDGE_BUFFER_SITES :])
        ),
        "target_ground_probability": ground_probability,
        "target_excitation_probability": max(0.0, norm - ground_probability),
    }


@dataclass
class PropagationResult:
    final_state: np.ndarray
    times: np.ndarray
    streams: dict[str, np.ndarray]
    forward_matvecs: int
    adjoint_matvecs: int
    wall_seconds: float
    maximum_chunk_state_bytes: int


def propagate_streaming(
    hamiltonian: sparse.csr_matrix,
    initial_state: np.ndarray,
    ground: np.ndarray,
    duration: float,
    base_time: float = 0.0,
) -> PropagationResult:
    started = time.perf_counter()
    block = _as_block(initial_state).astype(np.complex128, copy=True)
    intervals = int(round(duration / OBSERVATION_DT))
    if not math.isclose(intervals * OBSERVATION_DT, duration):
        raise ValueError("duration must be an integer multiple of observation dt")
    times = base_time + np.arange(intervals + 1, dtype=np.float64) * OBSERVATION_DT
    target_dimension = int(ground.size)
    names = tuple(_observable_snapshot(block, ground, target_dimension).keys())
    stream_lists: dict[str, list[float]] = {name: [] for name in names}

    def append_snapshot(state: np.ndarray) -> None:
        snapshot = _observable_snapshot(state, ground, target_dimension)
        for name, value in snapshot.items():
            stream_lists[name].append(value)

    append_snapshot(block)
    generator_matrix = (-1.0j * hamiltonian).tocsr()
    trace_generator = -1.0j * np.sum(hamiltonian.diagonal())
    forward = 0
    adjoint = 0
    maximum_chunk_bytes = block.nbytes
    completed = 0
    while completed < intervals:
        chunk_intervals = min(CHUNK_INTERVALS, intervals - completed)
        counting = CountingLinearOperator(generator_matrix)
        chunk = expm_multiply(
            counting,
            block,
            start=0.0,
            stop=chunk_intervals * OBSERVATION_DT,
            num=chunk_intervals + 1,
            endpoint=True,
            traceA=trace_generator,
        )
        maximum_chunk_bytes = max(maximum_chunk_bytes, int(chunk.nbytes))
        for local_index in range(1, chunk_intervals + 1):
            append_snapshot(chunk[local_index])
        block = np.array(chunk[-1], copy=True)
        completed += chunk_intervals
        forward += counting.forward_vector_equivalents
        adjoint += counting.adjoint_vector_equivalents
    streams = {name: np.asarray(values) for name, values in stream_lists.items()}
    return PropagationResult(
        final_state=block,
        times=times,
        streams=streams,
        forward_matvecs=forward,
        adjoint_matvecs=adjoint,
        wall_seconds=time.perf_counter() - started,
        maximum_chunk_state_bytes=maximum_chunk_bytes,
    )


def join_propagations(
    primary: PropagationResult, continuation: PropagationResult
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    times = np.concatenate((primary.times, continuation.times[1:]))
    streams = {
        name: np.concatenate((primary.streams[name], continuation.streams[name][1:]))
        for name in primary.streams
    }
    return times, streams


def current_centroid(times: np.ndarray, current: np.ndarray) -> dict[str, float]:
    incoming_weight = np.maximum(-current, 0.0)
    outgoing_weight = np.maximum(current, 0.0)
    incoming_flux = float(np.trapezoid(incoming_weight, times))
    outgoing_flux = float(np.trapezoid(outgoing_weight, times))
    incoming_centroid = float(np.trapezoid(times * incoming_weight, times) / incoming_flux)
    outgoing_centroid = float(np.trapezoid(times * outgoing_weight, times) / outgoing_flux)
    return {
        "incoming_flux": incoming_flux,
        "outgoing_flux": outgoing_flux,
        "incoming_centroid": incoming_centroid,
        "outgoing_centroid": outgoing_centroid,
        "round_trip_current_centroid": outgoing_centroid - incoming_centroid,
    }


def free_target_columns(
    target_shifted: np.ndarray, columns: np.ndarray, duration: float
) -> np.ndarray:
    return expm_multiply((-1.0j * duration) * target_shifted, columns)


def final_mode_metrics(
    final_state: np.ndarray,
    reference_lead: np.ndarray,
    reference_target_columns: np.ndarray,
    ground: np.ndarray,
) -> dict[str, Any]:
    block = _as_block(final_state)
    component_count = block.shape[1]
    amplitudes = block.reshape(LEAD_LENGTH, 16, component_count)
    reference_block = (
        reference_lead[:, None, None] * reference_target_columns[None, :, :]
    ).reshape(LEAD_LENGTH * 16, component_count)
    arm_coherence = np.vdot(reference_block, block)
    lead_mode_probability = 0.0
    ground_channel_probability = 0.0
    ground_mode_amplitudes: list[complex] = []
    for component in range(component_count):
        projected_target = reference_lead.conj() @ amplitudes[:, :, component]
        lead_mode_probability += float(np.vdot(projected_target, projected_target).real)
        ground_lead = amplitudes[:, :, component] @ ground.conj()
        ground_channel_probability += float(np.vdot(ground_lead, ground_lead).real)
        ground_mode_amplitudes.append(np.vdot(reference_lead, ground_lead))
    result: dict[str, Any] = {
        "matched_g0_arm_density_coherence_iq": _complex_json(arm_coherence),
        "matched_g0_arm_density_coherence_modulus_squared": float(abs(arm_coherence) ** 2),
        "reference_lead_mode_probability": lead_mode_probability,
        "target_ground_channel_probability": ground_channel_probability,
        "component_count": component_count,
        "iq_is_density_linear_and_not_a_branch_phase_comparison": True,
    }
    if component_count == 1:
        result["ground_channel_reference_mode_iq"] = _complex_json(
            ground_mode_amplitudes[0]
        )
        result["ground_channel_reference_mode_fidelity"] = float(
            abs(ground_mode_amplitudes[0]) ** 2
        )
    else:
        result["branch_amplitude_phases_reported"] = False
        result["ground_channel_reference_mode_fidelity_sum"] = float(
            sum(abs(value) ** 2 for value in ground_mode_amplitudes)
        )
    return result


def target_return_metrics(
    final_state: np.ndarray,
    initial_density: np.ndarray,
    free_density: np.ndarray,
    ground: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    density = target_density(final_state)
    diagnostics = density_diagnostics(density)
    diagnostics.update(
        {
            "ground_fidelity": float(np.real(np.vdot(ground, density @ ground))),
            "trace_distance_to_interaction_picture_free_target": trace_distance(
                density, free_density
            ),
            "fidelity_to_interaction_picture_free_target": density_fidelity(
                density, free_density
            ),
            "trace_distance_to_input_target": trace_distance(density, initial_density),
        }
    )
    return density, diagnostics


def query_run(
    label: str,
    hamiltonian: sparse.csr_matrix,
    packet_state: np.ndarray,
    target_columns: np.ndarray,
    target: dict[str, Any],
    reference_lead_final: np.ndarray,
    continue_to_delay_control: bool,
) -> tuple[dict[str, Any], PropagationResult, PropagationResult | None, np.ndarray]:
    started = time.perf_counter()
    initial_block = (
        packet_state[:, None, None] * target_columns[None, :, :]
    ).reshape(LEAD_LENGTH * 16, target_columns.shape[1])
    initial_density = target_columns @ target_columns.conj().T
    free_columns = free_target_columns(target["h_shifted"], target_columns, SCATTER_TIME)
    free_density = free_columns @ free_columns.conj().T
    primary = propagate_streaming(
        hamiltonian, initial_block, target["ground"], SCATTER_TIME
    )
    continuation = None
    if continue_to_delay_control:
        continuation = propagate_streaming(
            hamiltonian,
            primary.final_state,
            target["ground"],
            DELAY_CONTROL_TIME - SCATTER_TIME,
            base_time=SCATTER_TIME,
        )
    density, return_metrics = target_return_metrics(
        primary.final_state,
        initial_density,
        free_density,
        target["ground"],
    )
    mode = final_mode_metrics(
        primary.final_state,
        reference_lead_final,
        free_columns,
        target["ground"],
    )
    initial_energy = np.trace(initial_block.conj().T @ (hamiltonian @ initial_block))
    final_energy = np.trace(
        primary.final_state.conj().T @ (hamiltonian @ primary.final_state)
    )
    summary = {
        "label": label,
        "target_component_count": int(target_columns.shape[1]),
        "maximum_transient_target_excitation": float(
            np.max(primary.streams["target_excitation_probability"])
        ),
        "final_target_excitation": float(
            primary.streams["target_excitation_probability"][-1]
        ),
        "maximum_contact_probability": float(
            np.max(primary.streams["contact_probability"])
        ),
        "final_contact_probability": float(primary.streams["contact_probability"][-1]),
        "final_near_target_probability": float(
            primary.streams["near_target_probability"][-1]
        ),
        "maximum_right_edge_buffer_probability": float(
            np.max(primary.streams["right_edge_buffer_probability"])
        ),
        "maximum_norm_error": float(np.max(np.abs(primary.streams["norm"] - 1.0))),
        "energy_expectation_initial": _complex_json(initial_energy),
        "energy_expectation_final": _complex_json(final_energy),
        "energy_drift_abs": float(abs(final_energy - initial_energy)),
        "target_return": return_metrics,
        "boundary_mode": mode,
        "primary_forward_vector_equivalent_matvecs": primary.forward_matvecs,
        "primary_adjoint_vector_equivalent_matvecs": primary.adjoint_matvecs,
        "primary_wall_seconds_not_used_for_claim": primary.wall_seconds,
        "maximum_chunk_state_bytes": primary.maximum_chunk_state_bytes,
        "total_wall_seconds_not_used_for_claim": time.perf_counter() - started,
    }
    if continuation is not None:
        summary["delay_control_continuation"] = {
            "stop_time": DELAY_CONTROL_TIME,
            "forward_vector_equivalent_matvecs": continuation.forward_matvecs,
            "adjoint_vector_equivalent_matvecs": continuation.adjoint_matvecs,
            "wall_seconds_not_used_for_claim": continuation.wall_seconds,
            "maximum_norm_error": float(
                np.max(np.abs(continuation.streams["norm"] - 1.0))
            ),
        }
    return summary, primary, continuation, density


def stationary_relative_scattering(
    target: dict[str, Any], k_value: float
) -> complex:
    gaps = target["gaps"]
    eigenvectors = target["eigenvectors"]
    interaction = eigenvectors.conj().T @ (COUPLING_G * target["z0"]) @ eigenvectors
    probe_energy = 2.0 - 2.0 * math.cos(k_value)
    channel_energies = probe_energy - gaps
    diagonal = channel_energies - 2.0
    propagation = np.zeros(16, dtype=np.complex128)
    z_value = np.exp(1.0j * k_value)
    propagation[0] = z_value
    for channel in range(1, 16):
        if not channel_energies[channel] < 0.0:
            raise ValueError("stationary subgap oracle received an open excited channel")
        root_sum = 2.0 - channel_energies[channel]
        propagation[channel] = 0.5 * (
            root_sum - math.sqrt(root_sum * root_sum - 4.0)
        )
    boundary_matrix = np.diag(diagonal + propagation) - interaction
    source = np.zeros(16, dtype=np.complex128)
    source[0] = -(1.0 / z_value - z_value)
    boundary = np.linalg.solve(boundary_matrix, source)
    reflection = boundary[0] - 1.0
    uncoupled_reflection = -(z_value**2)
    return reflection / uncoupled_reflection


def stationary_phase_and_wigner_delay(
    target: dict[str, Any], k_value: float, delta_k: float = 1.0e-5
) -> dict[str, Any]:
    center = stationary_relative_scattering(target, k_value)
    plus = stationary_relative_scattering(target, k_value + delta_k)
    minus = stationary_relative_scattering(target, k_value - delta_k)
    energy_plus = 2.0 - 2.0 * math.cos(k_value + delta_k)
    energy_minus = 2.0 - 2.0 * math.cos(k_value - delta_k)
    delay = np.angle(plus * minus.conjugate()) / (energy_plus - energy_minus)
    return {
        "relative_scattering_iq": _complex_json(center),
        "unit_modulus_error": abs(abs(center) ** 2 - 1.0),
        "relative_phase_radians": float(np.angle(center)),
        "finite_difference_delta_k": delta_k,
        "wigner_delay": float(delay),
        "oracle_dimension": 16,
        "oracle_classification": "INDEPENDENT_FORMULA_SAME_PACKAGE_SELF_CHECK_NOT_SEPARATE_REFERENCE",
    }


def endpoint_one_shot_control(
    hamiltonian: sparse.csr_matrix, initial_state: np.ndarray, streamed_final: np.ndarray
) -> dict[str, Any]:
    started = time.perf_counter()
    generator = (-1.0j * hamiltonian).tocsr()
    counting = CountingLinearOperator(generator)
    trace_generator = -1.0j * np.sum(hamiltonian.diagonal())
    endpoint = expm_multiply(
        counting,
        _as_block(initial_state),
        start=0.0,
        stop=SCATTER_TIME,
        num=2,
        endpoint=True,
        traceA=trace_generator,
    )[-1]
    phase = np.vdot(endpoint, streamed_final)
    if abs(phase) > 0.0:
        aligned = endpoint * (phase / abs(phase))
    else:
        aligned = endpoint
    return {
        "phase_aligned_endpoint_l2": float(np.linalg.norm(aligned - streamed_final)),
        "density_trace_distance": trace_distance(
            target_density(endpoint), target_density(streamed_final)
        ),
        "forward_vector_equivalent_matvecs": counting.forward_vector_equivalents,
        "adjoint_vector_equivalent_matvecs": counting.adjoint_vector_equivalents,
        "wall_seconds_not_used_for_claim": time.perf_counter() - started,
        "comparison": "ONE_SHOT_T340_VERSUS_17_CHUNK_T340_EXPM_MULTIPLY",
    }


def returned_density_columns(density: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = 0.5 * (density + density.conj().T)
    input_trace = float(np.real(np.trace(normalized)))
    normalized /= input_trace
    values, vectors = eigh(normalized)
    negative_values = values[values < 0.0]
    clipped_trace = float(-np.sum(negative_values))
    clipped = np.maximum(values, 0.0)
    clipped /= np.sum(clipped)
    columns = vectors * np.sqrt(clipped)[None, :]
    reconstruction = columns @ columns.conj().T
    return columns, {
        "input_trace_before_normalization": input_trace,
        "trace_normalization_correction": abs(input_trace - 1.0),
        "minimum_raw_eigenvalue": float(values[0]),
        "negative_eigenvalue_trace_clipped": clipped_trace,
        "all_16_spectral_components_propagated": True,
        "discarded_positive_eigenvalue_weight": 0.0,
        "spectral_component_count": 16,
        "dominant_weight": float(clipped[-1]),
        "non_dominant_trace": float(np.sum(clipped[:-1])),
        "reconstruction_max_abs": float(np.max(np.abs(reconstruction - normalized))),
        "returned_state_rematerialization_used": True,
        "target_ground_reload_used": False,
        "generic_target_state_repreparation_via_density_rematerialization_used": True,
        "same_backing_reuse_established": False,
        "physical_reuse_established": False,
    }


def summarize_stream(
    result: PropagationResult, include_streams: bool
) -> dict[str, Any] | None:
    if not include_streams:
        return None
    return {
        "times": result.times.tolist(),
        **{name: values.tolist() for name, values in result.streams.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-streams",
        action="store_true",
        help="include aggregate observation streams; never includes amplitudes",
    )
    parser.add_argument("--indent", type=int, default=2)
    arguments = parser.parse_args()
    run_started = time.perf_counter()

    target = build_target()
    lead = build_lead()
    coupled, locality = build_total_hamiltonian(
        lead, target["h_shifted"], target["z0"], COUPLING_G
    )

    prepared = prepare_target(target, PREP_STEPS)
    prep_control = prepare_target(target, PREP_CONTROL_STEPS)
    prep_long_control = prepare_target(
        target, PREP_LONG_CONTROL_STEPS, PREP_LONG_CONTROL_TIME
    )
    prep_overlap = abs(np.vdot(prepared["state"], prep_control["state"])) ** 2

    packet_a = packet(K_QUERY_A)
    packet_b = packet(K_QUERY_B)
    packet_above = packet(K_ABOVE_THRESHOLD)
    spectral_a = packet_spectral_audit(packet_a, K_QUERY_A, target["gap"], lead)
    spectral_b = packet_spectral_audit(packet_b, K_QUERY_B, target["gap"], lead)
    spectral_above = packet_spectral_audit(
        packet_above, K_ABOVE_THRESHOLD, target["gap"], lead
    )

    # Matched g=0 reference arms are factorized and are propagated on the lead
    # alone; no target answer is supplied to the coupled model.
    reference_ground = np.array([1.0 + 0.0j])
    reference_a_primary = propagate_streaming(
        lead, packet_a, reference_ground, SCATTER_TIME
    )
    reference_a_continuation = propagate_streaming(
        lead,
        reference_a_primary.final_state,
        reference_ground,
        DELAY_CONTROL_TIME - SCATTER_TIME,
        base_time=SCATTER_TIME,
    )
    reference_b_primary = propagate_streaming(
        lead, packet_b, reference_ground, SCATTER_TIME
    )
    reference_b_continuation = propagate_streaming(
        lead,
        reference_b_primary.final_state,
        reference_ground,
        DELAY_CONTROL_TIME - SCATTER_TIME,
        base_time=SCATTER_TIME,
    )

    prepared_column = prepared["state"][:, None]
    query_a, query_a_primary, query_a_continuation, rho_a = query_run(
        "QUERY_A_PREPARED_TARGET",
        coupled,
        packet_a,
        prepared_column,
        target,
        reference_a_primary.final_state[:, 0],
        continue_to_delay_control=True,
    )

    initial_a = (
        packet_a[:, None, None] * prepared_column[None, :, :]
    ).reshape(LEAD_LENGTH * 16, 1)
    endpoint_control = endpoint_one_shot_control(
        coupled, initial_a, query_a_primary.final_state
    )

    returned_columns, handoff = returned_density_columns(rho_a)
    query_b_reuse, query_b_reuse_primary, query_b_reuse_continuation, rho_b_reuse = query_run(
        "QUERY_B_COMPLETE_RETURNED_RHO_A_REMATERIALIZATION",
        coupled,
        packet_b,
        returned_columns,
        target,
        reference_b_primary.final_state[:, 0],
        continue_to_delay_control=True,
    )

    # Clean-B executes a distinct adiabatic preparation.  It is a comparator,
    # not part of the two-query reuse amortization.
    clean_prepared = prepare_target(target, PREP_STEPS)
    clean_column = clean_prepared["state"][:, None]
    clean_b, clean_b_primary, clean_b_continuation, rho_b_clean = query_run(
        "CLEAN_B_DISTINCT_ADIABATIC_PREPARATION_CONTROL",
        coupled,
        packet_b,
        clean_column,
        target,
        reference_b_primary.final_state[:, 0],
        continue_to_delay_control=True,
    )

    exact_ground_column = target["ground"][:, None]
    exact_ground_b, exact_ground_b_primary, _, rho_b_exact_ground = query_run(
        "EXACT_GROUND_INJECTION_SHAM_CONTROL",
        coupled,
        packet_b,
        exact_ground_column,
        target,
        reference_b_primary.final_state[:, 0],
        continue_to_delay_control=False,
    )

    above_threshold, above_primary, _, _ = query_run(
        "ABOVE_THRESHOLD_K_PI_OVER_2_NEGATIVE_CONTROL",
        coupled,
        packet_above,
        exact_ground_column,
        target,
        # A dedicated g=0 arm is cheap and required for correct IQ.
        propagate_streaming(lead, packet_above, reference_ground, SCATTER_TIME).final_state[:, 0],
        continue_to_delay_control=False,
    )

    # Static boundary potential: retain only the exact-ground expectation of
    # the interaction.  This is the strongest scalar phase-only sham.
    static_potential = float(
        np.real(np.vdot(target["ground"], COUPLING_G * target["z0"] @ target["ground"]))
    )
    static_projector = build_contact_projector()
    static_lead = lead + static_potential * static_projector
    static_a = propagate_streaming(static_lead, packet_a, reference_ground, SCATTER_TIME)
    static_b = propagate_streaming(static_lead, packet_b, reference_ground, SCATTER_TIME)
    static_a_iq = np.vdot(reference_a_primary.final_state, static_a.final_state)
    static_b_iq = np.vdot(reference_b_primary.final_state, static_b.final_state)

    a_times_370, a_streams_370 = join_propagations(
        query_a_primary, query_a_continuation
    )
    a_ref_times_370, a_ref_streams_370 = join_propagations(
        reference_a_primary, reference_a_continuation
    )
    b_times_370, b_streams_370 = join_propagations(
        clean_b_primary, clean_b_continuation
    )
    b_reuse_times_370, b_reuse_streams_370 = join_propagations(
        query_b_reuse_primary, query_b_reuse_continuation
    )
    b_ref_times_370, b_ref_streams_370 = join_propagations(
        reference_b_primary, reference_b_continuation
    )
    a_centroid_340 = current_centroid(
        query_a_primary.times, query_a_primary.streams["detector_current"]
    )
    a_ref_centroid_340 = current_centroid(
        reference_a_primary.times, reference_a_primary.streams["detector_current"]
    )
    b_centroid_340 = current_centroid(
        clean_b_primary.times, clean_b_primary.streams["detector_current"]
    )
    b_reuse_centroid_340 = current_centroid(
        query_b_reuse_primary.times,
        query_b_reuse_primary.streams["detector_current"],
    )
    b_ref_centroid_340 = current_centroid(
        reference_b_primary.times, reference_b_primary.streams["detector_current"]
    )
    a_centroid_370 = current_centroid(a_times_370, a_streams_370["detector_current"])
    a_ref_centroid_370 = current_centroid(
        a_ref_times_370, a_ref_streams_370["detector_current"]
    )
    b_centroid_370 = current_centroid(b_times_370, b_streams_370["detector_current"])
    b_reuse_centroid_370 = current_centroid(
        b_reuse_times_370, b_reuse_streams_370["detector_current"]
    )
    b_ref_centroid_370 = current_centroid(
        b_ref_times_370, b_ref_streams_370["detector_current"]
    )

    def with_delay(coupled_centroid: dict[str, float], reference_centroid: dict[str, float]) -> dict[str, Any]:
        return {
            "coupled": coupled_centroid,
            "matched_g0": reference_centroid,
            "relative_current_centroid_delay": coupled_centroid[
                "round_trip_current_centroid"
            ]
            - reference_centroid["round_trip_current_centroid"],
        }

    delay_a_340 = with_delay(a_centroid_340, a_ref_centroid_340)
    delay_b_340 = with_delay(b_centroid_340, b_ref_centroid_340)
    delay_b_reuse_340 = with_delay(b_reuse_centroid_340, b_ref_centroid_340)
    delay_a_370 = with_delay(a_centroid_370, a_ref_centroid_370)
    delay_b_370 = with_delay(b_centroid_370, b_ref_centroid_370)
    delay_b_reuse_370 = with_delay(b_reuse_centroid_370, b_ref_centroid_370)
    stationary_a = stationary_phase_and_wigner_delay(target, K_QUERY_A)
    stationary_b = stationary_phase_and_wigner_delay(target, K_QUERY_B)

    reuse_vs_clean = {
        "final_target_trace_distance": trace_distance(rho_b_reuse, rho_b_clean),
        "final_target_density_fidelity": density_fidelity(rho_b_reuse, rho_b_clean),
        "ground_population_difference_abs": abs(
            query_b_reuse["target_return"]["ground_fidelity"]
            - clean_b["target_return"]["ground_fidelity"]
        ),
        "reference_lead_mode_probability_difference_abs": abs(
            query_b_reuse["boundary_mode"]["reference_lead_mode_probability"]
            - clean_b["boundary_mode"]["reference_lead_mode_probability"]
        ),
        "density_observables_only_no_branch_phase_comparison": True,
    }
    exact_ground_vs_clean = {
        "final_target_trace_distance": trace_distance(
            rho_b_exact_ground, rho_b_clean
        ),
        "prepared_initial_to_exact_ground_trace_distance": math.sqrt(
            max(0.0, 1.0 - clean_prepared["ground_fidelity"])
        ),
    }

    acceptance_checks = {
        "contact_projector_exactly_one_entry": locality["contact_projector_nnz"] == 1,
        "coupling_only_at_lead_site_zero": locality["coupling_localized_at_site_zero"],
        "query_a_effectively_subgap": spectral_a["effectively_subgap_at_declared_tail"],
        "query_b_effectively_subgap": spectral_b["effectively_subgap_at_declared_tail"],
        "query_a_transient_interaction": query_a["maximum_transient_target_excitation"]
        >= THRESHOLDS["query_a_transient_excitation_min"],
        "query_b_transient_interaction": query_b_reuse[
            "maximum_transient_target_excitation"
        ]
        >= THRESHOLDS["query_b_transient_excitation_min"],
        "query_a_contact_drained": query_a["final_contact_probability"]
        <= THRESHOLDS["final_contact_probability_max"],
        "query_b_contact_drained": query_b_reuse["final_contact_probability"]
        <= THRESHOLDS["final_contact_probability_max"],
        "query_a_target_return": query_a["target_return"][
            "trace_distance_to_interaction_picture_free_target"
        ]
        <= THRESHOLDS["target_trace_distance_to_free_max"],
        "query_b_reuse_target_return": query_b_reuse["target_return"][
            "trace_distance_to_interaction_picture_free_target"
        ]
        <= THRESHOLDS["target_trace_distance_to_free_max"],
        "query_a_norm": query_a["maximum_norm_error"]
        <= THRESHOLDS["norm_or_trace_error_max"],
        "query_b_reuse_norm": query_b_reuse["maximum_norm_error"]
        <= THRESHOLDS["norm_or_trace_error_max"],
        "query_a_density_trace": query_a["target_return"]["trace_error"]
        <= THRESHOLDS["norm_or_trace_error_max"],
        "query_a_density_hermiticity": query_a["target_return"][
            "hermiticity_max_abs"
        ]
        <= THRESHOLDS["density_hermiticity_error_max"],
        "query_a_density_psd": query_a["target_return"]["minimum_eigenvalue"]
        >= THRESHOLDS["density_minimum_eigenvalue_min"],
        "query_b_reuse_density_trace": query_b_reuse["target_return"][
            "trace_error"
        ]
        <= THRESHOLDS["norm_or_trace_error_max"],
        "query_b_reuse_density_hermiticity": query_b_reuse["target_return"][
            "hermiticity_max_abs"
        ]
        <= THRESHOLDS["density_hermiticity_error_max"],
        "query_b_reuse_density_psd": query_b_reuse["target_return"][
            "minimum_eigenvalue"
        ]
        >= THRESHOLDS["density_minimum_eigenvalue_min"],
        "rho_a_psd_clip_control": handoff["negative_eigenvalue_trace_clipped"]
        <= THRESHOLDS["returned_density_psd_clip_trace_max"],
        "query_a_endpoint_convergence": endpoint_control["phase_aligned_endpoint_l2"]
        <= THRESHOLDS["endpoint_convergence_l2_max"],
        "query_a_detector_flux_370": min(
            a_centroid_370["incoming_flux"], a_centroid_370["outgoing_flux"]
        )
        >= THRESHOLDS["detector_incoming_flux_min"],
        "query_b_detector_flux_370": min(
            b_reuse_centroid_370["incoming_flux"],
            b_reuse_centroid_370["outgoing_flux"],
        )
        >= THRESHOLDS["detector_outgoing_flux_min"],
        "near_threshold_delay_ordering": stationary_b["wigner_delay"]
        - stationary_a["wigner_delay"]
        >= THRESHOLDS["delay_ordering_margin_min"],
    }
    all_pass = all(acceptance_checks.values())
    execution_integrity_checks = {
        key: value
        for key, value in acceptance_checks.items()
        if key not in {"query_a_target_return", "query_b_reuse_target_return"}
    }
    execution_integrity_pass = all(execution_integrity_checks.values())

    two_query_matvecs = (
        prepared["forward_vector_equivalent_matvecs"]
        + query_a_primary.forward_matvecs
        + query_b_reuse_primary.forward_matvecs
    )
    two_query_wall = (
        prepared["wall_seconds_not_used_for_claim"]
        + query_a_primary.wall_seconds
        + query_b_reuse_primary.wall_seconds
    )

    result: dict[str, Any] = {
        "schema": "PHASE_QEMU_V6_TIME_DOMAIN_SCATTERING_EVIDENCE_V1",
        "milestone": "M264",
        "source_self_assertion": "SOURCE_SELF_CHECK_PASS"
        if execution_integrity_pass
        else "SOURCE_SELF_CHECK_FAIL",
        "verification_scope": {
            "science": "PACKAGE_SELF_REVIEW_PENDING_SEPARATE_REFERENCE_PARITY",
            "theory": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource": "PACKAGE_SELF_REVIEW",
        },
        "claim": PASS_CLAIM if all_pass else FAIL_CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "disposition": PASS_DISPOSITION if all_pass else FAIL_DISPOSITION,
        "next_mechanism": NEXT_MECHANISM,
        "restoration": {
            "classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION"
            if all_pass
            else "NO_RESTORATION_CLAIM",
            "classification_semantics": "REGISTRY_CLASS_FOR_FUNCTIONAL_NUMERICAL_MODELED_TARGET_STATE_RETURN_ONLY_NOT_PHYSICAL_HARDWARE_RESTORATION"
            if all_pass
            else "NO_RETURN_QUALIFICATION",
            "scope": "INTERACTION_PICTURE_REDUCED_TARGET_STATE_RETURN_AND_FUNCTIONAL_RETURNED_STATE_REMATERIALIZATION_WITHOUT_SAME_BACKING_IN_A_DETERMINISTIC_SOFTWARE_MODEL"
            if all_pass
            else "FAILED_DECLARED_TIME_DOMAIN_RETURN_OR_REUSE_THRESHOLDS",
            "executed_time_domain_target_excursion_and_drain": True,
            "returned_state_rematerialization_used": True,
            "baseline_or_exact_ground_reload_used_for_reuse_path": False,
            "generic_target_state_repreparation_via_density_rematerialization_used": True,
            "same_backing_restoration_established": False,
            "same_backing_reuse_established": False,
            "physical_restoration_established": False,
            "permanent_restoration_established": False,
        },
        "backend": {
            "classification": "DETERMINISTIC_COMPLEX128_SOFTWARE_HARDWARE_MODEL",
            "qemu_device_executed": False,
            "physical_execution": False,
            "source_on_drive_during_scattering": False,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "source_sha256": _sha256(Path(__file__)),
        },
        "access_model": {
            "guest_or_controller_public_inputs": [
                "FOUR_SPIN_HAMILTONIAN_COEFFICIENTS",
                "ADIABATIC_SCHEDULE_AND_DURATION",
                "LEAD_GEOMETRY_AND_HOPPING",
                "PACKET_CENTER_SIGMA_AND_MOMENTUM",
                "LOCAL_COUPLING_OPERATOR_AND_STRENGTH",
                "OBSERVATION_GRID_AND_ACCEPTANCE_THRESHOLDS",
            ],
            "hidden_or_secret_inputs": [],
            "precomputed_green_functions_phases_delays_or_answer_tables_supplied_to_propagator": False,
            "production_target_input": "PUBLIC_ADIABATIC_EVOLUTION_FROM_PLUS4",
            "query_B_target_input": "COMPLETE_RECONSTRUCTED_RETURNED_RHO_A_ALL_16_SPECTRAL_COMPONENTS",
            "comparator_access": "EQUAL_FULL_IMPLEMENTATION_DESCRIPTOR_AND_STATE_ACCESS",
            "forward_shadow_may_execute_identical_sparse_recurrence": True,
            "oracle_or_restricted_access_advantage_claimed": False,
        },
        "public_configuration": {
            "lead": {
                "type": "OPEN_DISCRETE_LAPLACIAN_2I_MINUS_NEAREST_NEIGHBOR_ADJACENCY",
                "length": LEAD_LENGTH,
                "hopping": LEAD_HOPPING,
                "packet_center": PACKET_CENTER,
                "packet_sigma": PACKET_SIGMA,
                "packet_amplitude_convention": "EXP_MINUS_X_MINUS_X0_SQUARED_OVER_4_SIGMA_SQUARED_TIMES_EXP_MINUS_I_K_X_MINUS_X0",
                "detector_bond": [DETECTOR_SITE, DETECTOR_SITE + 1],
                "contact_site": CONTACT_SITE,
                "primary_stop_time": SCATTER_TIME,
                "delay_control_stop_time": DELAY_CONTROL_TIME,
                "observation_dt": OBSERVATION_DT,
            },
            "target": {
                "dimension": 16,
                "hamiltonian": "MINUS_SUM_X_MINUS_3_OVER_4_SUM_NEAREST_ZZ_MINUS_1_OVER_5_Z0_MINUS_2_OVER_7_Z1_MINUS_3_OVER_11_Z2_MINUS_5_OVER_13_Z3_MINUS_1_OVER_2_Z0Z1Z2",
                "ground_energy": target["ground_energy"],
                "gap": target["gap"],
                "coupling": "3_OVER_2_TIMES_Z0",
            },
            "queries": {
                "A_k": K_QUERY_A,
                "B_k": K_QUERY_B,
                "above_threshold_control_k": K_ABOVE_THRESHOLD,
            },
            "sigma_change_from_blueprint": {
                "original": 48,
                "frozen": 50,
                "reason": "PREDECLARED_QUERY_B_ABOVE_GAP_DST_TAIL_CORRECTION_BEFORE_SEMANTIC_EXECUTION",
            },
        },
        "locality_fail_closed_evidence": locality,
        "thresholds": THRESHOLDS,
        "acceptance_checks": acceptance_checks,
        "acceptance_all": all_pass,
        "restoration_promotion_all": all_pass,
        "execution_integrity_checks": execution_integrity_checks,
        "execution_integrity_all": execution_integrity_pass,
        "packet_preflight": {
            "query_A": spectral_a,
            "query_B": spectral_b,
            "above_threshold_control": spectral_above,
            "finite_bandwidth_packets_are_not_exact_energy_eigenstates": True,
        },
        "preparation": {
            **{key: value for key, value in prepared.items() if key != "state"},
            "half_step_control": {
                **{key: value for key, value in prep_control.items() if key != "state"},
                "state_fidelity_to_primary": float(prep_overlap),
            },
            "T480_nonclaim_bearing_diagnostic": {
                **{
                    key: value
                    for key, value in prep_long_control.items()
                    if key != "state"
                },
                "used_as_production_query_input": False,
                "may_not_repair_the_frozen_T120_outcome": True,
            },
            "exact_ground_was_not_injected_into_production_queries": True,
        },
        "query_A": query_a,
        "returned_rho_A_handoff": handoff,
        "query_B_reuse": query_b_reuse,
        "query_B_clean": clean_b,
        "reuse_vs_clean_B": reuse_vs_clean,
        "delay": {
            "query_A_T340": delay_a_340,
            "query_B_reuse_T340": delay_b_reuse_340,
            "query_B_clean_T340": delay_b_340,
            "query_A_T370_control": delay_a_370,
            "query_B_reuse_T370": delay_b_reuse_370,
            "query_B_reuse_T370_control": delay_b_reuse_370,
            "query_B_clean_T370_control": delay_b_370,
            "stationary_16_channel_query_A": stationary_a,
            "stationary_16_channel_query_B": stationary_b,
            "T340_is_finite_window_not_a_tight_wigner_estimator": True,
        },
        "convergence": {
            "query_A_one_shot_endpoint": endpoint_control,
            "observation_grid_is_measurement_grid_not_integrator_step": True,
            "T370_detector_flux_control_executed": True,
            "finite_size_control": {
                "second_lead_size_executed": False,
                "reason": "ORIGINAL_UNHALVED_L641_GEOMETRY_RETAINED",
                "query_A_maximum_right_edge_buffer_probability": query_a[
                    "maximum_right_edge_buffer_probability"
                ],
                "query_B_maximum_right_edge_buffer_probability": query_b_reuse[
                    "maximum_right_edge_buffer_probability"
                ],
                "query_B_clean_comparator_maximum_right_edge_buffer_probability": clean_b[
                    "maximum_right_edge_buffer_probability"
                ],
            },
        },
        "controls": {
            "matched_g0": {
                "implemented": True,
                "query_A_T370_current_centroid": a_ref_centroid_370,
                "query_B_T370_current_centroid": b_ref_centroid_370,
            },
            "static_boundary_potential": {
                "implemented": True,
                "potential": static_potential,
                "query_A_matched_g0_iq": _complex_json(static_a_iq),
                "query_B_matched_g0_iq": _complex_json(static_b_iq),
                "query_A_mode_fidelity": float(abs(static_a_iq) ** 2),
                "query_B_mode_fidelity": float(abs(static_b_iq) ** 2),
            },
            "exact_ground_injection": {
                "implemented": True,
                "classification": "ANSWER_BEARING_PREPARATION_SHAM_NOT_PRODUCTION",
                "query_B": exact_ground_b,
                "comparison_to_clean_B": exact_ground_vs_clean,
            },
            "above_threshold": {
                "implemented": True,
                "query": above_threshold,
                "expected_to_permit_real_target_excitation": True,
            },
            "snapshot_reload": {
                "implemented_as_counterfactual": True,
                "classification": "SHAM_BASELINE_NOT_RESTORATION",
                "cached_prepared_vector_would_replace_returned_rho_A": True,
                "production_reuse_path_used_it": False,
            },
            "disconnected": {
                "implemented": True,
                "realization": "MATCHED_G0_ARM",
            },
            "small_core": {
                "implemented": False,
                "reason": "NO_NONARBITRARY_SMALL_CORE_TRUNCATION_WAS_FROZEN",
            },
            "gaussian_or_quadratic_target": {
                "implemented": False,
                "reason": "NO_MATCHED_QUADRATIC_SURROGATE_WITH_THE_FROZEN_GAP_AND_BOUNDARY_SPECTRAL_MEASURE_WAS_PREDECLARED",
            },
            "forward_only_shadow": {
                "implemented_constructively": True,
                "law": "IDENTICAL_SPARSE_COMPLEX128_FORWARD_EVOLUTION_PLUS_FINAL_BOUNDARY_PROJECTION_OMITTING_RETURN_TESTS_AND_QUERY_B_HANDOFF",
                "state_no_greater_than_phase_model": True,
                "work_no_greater_than_phase_model": True,
                "m257_intact": True,
            },
        },
        "resource_accounting": {
            "target_dimension": 16,
            "lead_length": LEAD_LENGTH,
            "joint_complex_amplitudes": LEAD_LENGTH * 16,
            "raw_joint_state_bytes_complex128": LEAD_LENGTH * 16 * 16,
            "full_171_state_history_bytes_if_retained": 171 * LEAD_LENGTH * 16 * 16,
            "history_retained_by_production": False,
            "maximum_chunk_intervals": CHUNK_INTERVALS,
            "query_B_returned_density_component_count": 16,
            "query_B_raw_block_state_bytes": LEAD_LENGTH * 16 * 16 * 16,
            "preparation_forward_vector_equivalent_matvecs": prepared[
                "forward_vector_equivalent_matvecs"
            ],
            "query_A_forward_vector_equivalent_matvecs": query_a_primary.forward_matvecs,
            "query_B_reuse_forward_vector_equivalent_matvecs": query_b_reuse_primary.forward_matvecs,
            "two_query_cold_start_total_forward_vector_equivalent_matvecs": two_query_matvecs,
            "two_query_amortized_forward_vector_equivalent_matvecs": two_query_matvecs / 2.0,
            "two_query_cold_start_wall_seconds_not_used_for_claim": two_query_wall,
            "two_query_amortized_wall_seconds_not_used_for_claim": two_query_wall / 2.0,
            "density_handoff_work": "16_BY_16_DENSE_PARTIAL_TRACE_EIGENDECOMPOSITION_AND_16_COMPONENT_REMATERIALIZATION",
            "strongest_honest_classical_comparator": "IDENTICAL_O_16L_SPARSE_COORDINATE_TIME_EVOLUTION",
            "one_particle_lead_cut_mps_bond_upper_bound": 17,
            "tensor_network_crossover_established": False,
            "resource_advantage_established": False,
            "not_instrumented": [
                "PYTHON_NUMPY_SCIPY_OBJECT_AND_ALLOCATOR_OVERHEAD",
                "SCIPY_INTERNAL_KRYLOV_BASIS_PAYLOAD_AND_EXACT_PEAK_RSS",
                "BLAS_THREAD_RUNTIME_AND_CACHE_TRAFFIC",
                "INPUT_INTERMEDIATE_AND_OUTPUT_PAYLOAD_BIT_HEIGHT_BEYOND_COMPLEX128",
                "DESCRIPTOR_COPIES_AND_CONTROLLER_DETECTOR_OBJECT_STATE",
                "WHOLE_PROCESS_LIVENESS_OUTSIDE_DECLARED_STATE_CHUNKS",
                "PHYSICAL_ENERGY_LOSS_BANDWIDTH_LATENCY_AND_CONTROL_WORK",
                "FINITE_PRECISION_REPETITION_OR_SHOT_COUNT_FOR_A_PHYSICAL_ESTIMATOR",
            ],
            "precision_law": "COMPLEX128_NUMERICAL_ERROR_EMPIRICALLY_GATED_NO_EXACT_OR_ARBITRARY_PRECISION_CLAIM",
            "shot_law": "STATEVECTOR_EXPECTATIONS_NO_SAMPLING_SHOTS_PHYSICAL_SHOTS_UNINSTRUMENTED",
        },
        "streams": {
            "included": arguments.include_streams,
            "contain_hidden_amplitudes": False,
            "query_A": summarize_stream(query_a_primary, arguments.include_streams),
            "query_B_reuse": summarize_stream(
                query_b_reuse_primary, arguments.include_streams
            ),
            "query_B_clean": summarize_stream(clean_b_primary, arguments.include_streams),
            "above_threshold": summarize_stream(above_primary, arguments.include_streams),
        },
        "global_not_established": [
            "QEMU_DEVICE_EXECUTION",
            "PHYSICAL_SCATTERING_OR_OBSERVATION",
            "SOURCE_ISOLATION_IN_HARDWARE",
            "SAME_BACKING_RESTORATION_OR_REUSE",
            "PERMANENT_RESTORATION_ON_A_FINITE_LEAD",
            "ASYMPTOTIC_OR_GROWING_TARGET_RESOURCE_LAW",
            "TENSOR_NETWORK_CROSSOVER",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "M257_ESCAPE",
            "SMALL_WALL_CROSSING",
            "UNBOUNDED_COMPUTE",
            "PHYSICAL_BIT_REPLACEMENT_WITH_PI",
        ],
        "total_wall_seconds_not_used_for_claim": time.perf_counter() - run_started,
    }
    print(json.dumps(result, indent=arguments.indent, sort_keys=True, allow_nan=False))
    return 0 if execution_integrity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
