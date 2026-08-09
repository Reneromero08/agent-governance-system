#!/usr/bin/env python3
"""M265 energy-constrained truncated-Fock Weyl-loop calibration.

This is deterministic complex128 software emulation.  It executes four actual
conditional quadrature pulses per program on one in-place client+bus ensemble
backing.  It is not physical execution, a physical same-mode custody proof, a
resource advantage, or an escape from M257.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import expm
from scipy.special import gammaln


HERE = Path(__file__).resolve().parent
CUTOFFS = (16, 32, 64, 128)
FINAL_CUTOFF = 128
TOP_EDGE_WIDTH = 4
CLIENT_DIMENSION = 8
COMPLEX_COMPONENT_BITS = 128

PROGRAM_A = {
    "name": "A_Z0_Z1_PI_OVER_8",
    "a_client": 0,
    "b_client": 1,
    "lambda": 0.5,
    "mu": math.pi / 4.0,
    "theta": math.pi / 8.0,
}
PROGRAM_B = {
    "name": "B_Z1_Z2_PI_OVER_6",
    "a_client": 1,
    "b_client": 2,
    "lambda": 2.0 / 3.0,
    "mu": math.pi / 4.0,
    "theta": math.pi / 6.0,
}

THRESHOLDS = {
    "final_client_trace_distance_max": 1e-10,
    "final_bus_trace_distance_after_a_max": 1e-9,
    "final_bus_trace_distance_after_b_max": 2e-9,
    "boundary_max_abs_error": 1e-10,
    "density_trace_error_max": 1e-12,
    "density_hermiticity_max": 1e-12,
    "density_min_eigenvalue_min": -1e-12,
    "reference_br_trace_distance_max": 1e-9,
    "reference_entanglement_infidelity_max": 1e-9,
    "complete_joint_factorization_frobenius_max": 2e-9,
    "vacuum_midloop_mutual_information_min_nats": 1.0,
    "top_fock_client_trace_distance_min": 0.5,
    "top_fock_bus_trace_distance_min": 0.5,
    "ideal_control_return_max": 1e-9,
    "sign_sensitive_separation_min": 1.0,
    "omission_bus_trace_distance_min": 0.05,
    "noise_area_client_and_bus_min": 1e-4,
    "noise_rotation_client_and_bus_min": 1e-3,
    "noise_kerr_client_min": 1e-4,
    "noise_kerr_bus_min": 1e-3,
}

PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.diag([1.0, -1.0]).astype(np.complex128)
PLUS3 = np.ones(CLIENT_DIMENSION, dtype=np.complex128) / math.sqrt(CLIENT_DIMENSION)


def metric(value: float) -> float:
    return float(f"{float(value):.13g}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hermitize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0


def trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(hermitize(first - second))
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def density_integrity(matrix: np.ndarray) -> dict[str, float]:
    hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
    eigenvalues = np.linalg.eigvalsh(hermitize(matrix))
    return {
        "trace_error": metric(abs(float(np.trace(matrix).real) - 1.0)),
        "hermiticity_max_abs": metric(hermiticity),
        "minimum_eigenvalue": metric(float(np.min(eigenvalues))),
    }


def entropy_nats(matrix: np.ndarray) -> float:
    eigenvalues = np.clip(np.linalg.eigvalsh(hermitize(matrix)).real, 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues / total
    nonzero = probabilities[probabilities > 0.0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def client_z(client_index: int, basis_index: int) -> int:
    bit = (basis_index >> (2 - client_index)) & 1
    return 1 if bit == 0 else -1


def kron3(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(first, second), third)


OBSERVABLES = {
    "X0": kron3(PAULI_X, PAULI_I, PAULI_I),
    "X1": kron3(PAULI_I, PAULI_X, PAULI_I),
    "X2": kron3(PAULI_I, PAULI_I, PAULI_X),
    "Y0Z1": kron3(PAULI_Y, PAULI_Z, PAULI_I),
    "Z0Y1": kron3(PAULI_Z, PAULI_Y, PAULI_I),
    "Z1Y2": kron3(PAULI_I, PAULI_Z, PAULI_Y),
    "X0X1": kron3(PAULI_X, PAULI_X, PAULI_I),
    "X1X2": kron3(PAULI_I, PAULI_X, PAULI_X),
}

BOUNDARY_A_NAMES = ("X0", "X1", "X2", "Y0Z1", "Z0Y1", "X0X1")
BOUNDARY_COMBINED_NAMES = (
    "X0",
    "X1",
    "X2",
    "Y0Z1",
    "Z0Y1",
    "Z1Y2",
    "X0X1",
    "X1X2",
)


def observe(client_density: np.ndarray, names: Iterable[str]) -> dict[str, float]:
    return {
        name: metric(float(np.trace(client_density @ OBSERVABLES[name]).real))
        for name in names
    }


def boundary_error(actual: dict[str, float], expected: dict[str, float]) -> float:
    return max(abs(actual[name] - expected[name]) for name in expected)


def compiled_client_vector(programs: Iterable[dict[str, Any]]) -> np.ndarray:
    state = PLUS3.copy()
    for program in programs:
        for basis_index in range(CLIENT_DIMENSION):
            product = client_z(program["a_client"], basis_index) * client_z(
                program["b_client"], basis_index
            )
            state[basis_index] *= np.exp(-1j * program["theta"] * product)
    return state


COMPILED_A_VECTOR = compiled_client_vector((PROGRAM_A,))
COMPILED_COMBINED_VECTOR = compiled_client_vector((PROGRAM_A, PROGRAM_B))
COMPILED_A_DENSITY = np.outer(COMPILED_A_VECTOR, COMPILED_A_VECTOR.conj())
COMPILED_COMBINED_DENSITY = np.outer(
    COMPILED_COMBINED_VECTOR, COMPILED_COMBINED_VECTOR.conj()
)
COMPILED_A_BOUNDARY = observe(COMPILED_A_DENSITY, BOUNDARY_A_NAMES)
COMPILED_COMBINED_BOUNDARY = observe(
    COMPILED_COMBINED_DENSITY, BOUNDARY_COMBINED_NAMES
)

ANALYTIC_A_BOUNDARY = {
    "X0": 1.0 / math.sqrt(2.0),
    "X1": 1.0 / math.sqrt(2.0),
    "X2": 1.0,
    "Y0Z1": 1.0 / math.sqrt(2.0),
    "Z0Y1": 1.0 / math.sqrt(2.0),
    "X0X1": 1.0,
}
ANALYTIC_COMBINED_BOUNDARY = {
    "X0": 1.0 / math.sqrt(2.0),
    "X1": 1.0 / (2.0 * math.sqrt(2.0)),
    "X2": 0.5,
    "Y0Z1": 1.0 / math.sqrt(2.0),
    "Z0Y1": 1.0 / (2.0 * math.sqrt(2.0)),
    "Z1Y2": math.sqrt(3.0) / 2.0,
    "X0X1": 0.5,
    "X1X2": 1.0 / math.sqrt(2.0),
}


@dataclass
class Fixture:
    name: str
    ensemble: np.ndarray
    reference_dimension: int = 1


class PulseCache:
    def __init__(self, cutoff: int):
        self.cutoff = cutoff
        annihilation = np.diag(np.sqrt(np.arange(1, cutoff)), 1).astype(np.complex128)
        creation = annihilation.conj().T
        self.x = (annihilation + creation) / math.sqrt(2.0)
        self.p = (annihilation - creation) / (1j * math.sqrt(2.0))
        self.number = creation @ annihilation
        self._cache: dict[tuple[str, float], np.ndarray] = {}
        self.expm_calls = 0

    def pulse(self, quadrature: str, amount: float) -> np.ndarray:
        key = (quadrature, float(amount))
        if key not in self._cache:
            generator = self.x if quadrature == "X" else self.p
            self._cache[key] = expm(-1j * amount * generator)
            self.expm_calls += 1
        return self._cache[key]

    @property
    def cached_complex_cells(self) -> int:
        return len(self._cache) * self.cutoff * self.cutoff


def coherent_vector(cutoff: int, alpha: complex) -> np.ndarray:
    values = np.zeros(cutoff, dtype=np.complex128)
    values[0] = np.exp(-abs(alpha) ** 2 / 2.0)
    for index in range(1, cutoff):
        values[index] = values[index - 1] * alpha / math.sqrt(index)
    values /= np.linalg.norm(values)
    return values


def squeezed_vacuum_vector(cutoff: int, squeeze: float, angle: float) -> np.ndarray:
    values = np.zeros(cutoff, dtype=np.complex128)
    for pair_index in range(cutoff // 2):
        index = 2 * pair_index
        log_magnitude = (
            -0.5 * math.log(math.cosh(squeeze))
            + pair_index * math.log(math.tanh(squeeze))
            + 0.5 * gammaln(2 * pair_index + 1)
            - pair_index * math.log(2.0)
            - gammaln(pair_index + 1)
        )
        phase = pair_index * (angle + math.pi)
        values[index] = np.exp(log_magnitude + 1j * phase)
    values /= np.linalg.norm(values)
    return values


def pure_fixture(name: str, vector: np.ndarray) -> Fixture:
    normalized = vector.astype(np.complex128, copy=True)
    normalized /= np.linalg.norm(normalized)
    return Fixture(name=name, ensemble=normalized[None, :, None])


def fixtures(cutoff: int) -> list[Fixture]:
    vacuum = np.zeros(cutoff, dtype=np.complex128)
    vacuum[0] = 1.0
    fock_one = np.zeros(cutoff, dtype=np.complex128)
    fock_one[1] = 1.0
    non_gaussian = np.zeros(cutoff, dtype=np.complex128)
    non_gaussian[0] = 1.0
    non_gaussian[3] = 1j

    mean_thermal = 0.4
    ratio = mean_thermal / (1.0 + mean_thermal)
    probabilities = (1.0 - ratio) * ratio ** np.arange(cutoff)
    probabilities /= np.sum(probabilities)
    thermal_ensemble = np.diag(np.sqrt(probabilities)).astype(np.complex128)[:, :, None]

    phi4 = np.zeros((1, cutoff, 4), dtype=np.complex128)
    for index in range(4):
        phi4[0, index, index] = 0.5

    return [
        pure_fixture("vacuum", vacuum),
        pure_fixture("coherent_alpha_0p65_plus_0p20i", coherent_vector(cutoff, 0.65 + 0.20j)),
        Fixture("thermal_nbar_0p4", thermal_ensemble),
        pure_fixture("squeezed_r_0p45_phi_0p30", squeezed_vacuum_vector(cutoff, 0.45, 0.30)),
        pure_fixture("fock_1_non_gaussian", fock_one),
        pure_fixture("fock_superposition_0_plus_i3", non_gaussian),
        Fixture("phi4_bus_reference", phi4, reference_dimension=4),
    ]


def supplied_state(fixture: Fixture) -> np.ndarray:
    # Canonical exact ensemble/purification representation:
    # (ensemble component, client basis, bus level, inert reference).
    return np.einsum("j,rnq->rjnq", PLUS3, fixture.ensemble).astype(np.complex128)


def client_density(state: np.ndarray) -> np.ndarray:
    return np.einsum("rjnq,rknq->jk", state, state.conj(), optimize=True)


def bus_density(state: np.ndarray) -> np.ndarray:
    return np.einsum("rjnq,rjmq->nm", state, state.conj(), optimize=True)


def bus_reference_density(state: np.ndarray) -> np.ndarray:
    rank, _, cutoff, reference_dimension = state.shape
    dimension = cutoff * reference_dimension
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for component in range(rank):
        for client_index in range(CLIENT_DIMENSION):
            vector = state[component, client_index].reshape(dimension)
            result += np.outer(vector, vector.conj())
    return result


def complete_joint_factorization_frobenius(
    state: np.ndarray,
    expected_client: np.ndarray,
    expected_bus_reference: np.ndarray,
) -> float:
    """Stable exact Frobenius distance for the complete mixed joint density.

    One bus-reference block is materialized at a time.  This avoids both a
    potentially 4096-by-4096 joint matrix and the square-root cancellation of
    a purity-identity implementation near exact factorization.
    """

    rank, client_dimension, cutoff, reference_dimension = state.shape
    br_dimension = cutoff * reference_dimension
    vectors = state.reshape(rank, client_dimension, br_dimension)
    squared = 0.0
    for row in range(client_dimension):
        for column in range(client_dimension):
            actual_block = vectors[:, row, :].T @ vectors[:, column, :].conj()
            difference = actual_block - expected_client[row, column] * expected_bus_reference
            squared += float(np.sum(np.abs(difference) ** 2).real)
    return float(math.sqrt(squared))


def bus_diagnostics(state: np.ndarray, cache: PulseCache) -> dict[str, float]:
    density = bus_density(state)
    populations = np.real(np.diag(density))
    return {
        "mean_excitation": metric(float(np.dot(np.arange(cache.cutoff), populations))),
        "top_edge_probability": metric(float(np.sum(populations[-TOP_EDGE_WIDTH:]))),
    }


def apply_conditional_pulse_in_place(
    state: np.ndarray,
    cache: PulseCache,
    quadrature: str,
    amount: float,
    client_index: int,
) -> None:
    rank, _, cutoff, reference_dimension = state.shape
    for basis_index in range(CLIENT_DIMENSION):
        signed_amount = amount * client_z(client_index, basis_index)
        unitary = cache.pulse(quadrature, signed_amount)
        rows = state[:, basis_index, :, :].transpose(0, 2, 1).reshape(
            rank * reference_dimension, cutoff
        )
        updated = rows @ unitary.T
        state[:, basis_index, :, :] = updated.reshape(
            rank, reference_dimension, cutoff
        ).transpose(0, 2, 1)


def accepted_schedule(program: dict[str, Any]) -> list[tuple[str, float, int]]:
    # Product Q(+lambda) R(+mu) Q(-lambda) R(-mu) is applied right-to-left.
    return [
        ("P", -program["mu"], program["b_client"]),
        ("X", -program["lambda"], program["a_client"]),
        ("P", +program["mu"], program["b_client"]),
        ("X", +program["lambda"], program["a_client"]),
    ]


def run_schedule_in_place(
    state: np.ndarray,
    cache: PulseCache,
    schedule: list[tuple[str, float, int]],
    inter_pulse: np.ndarray | None = None,
) -> list[dict[str, float]]:
    diagnostics: list[dict[str, float]] = []
    for pulse_index, (quadrature, amount, client_index) in enumerate(schedule):
        apply_conditional_pulse_in_place(state, cache, quadrature, amount, client_index)
        if inter_pulse is not None and pulse_index < len(schedule) - 1:
            rank, _, cutoff, reference_dimension = state.shape
            for basis_index in range(CLIENT_DIMENSION):
                rows = state[:, basis_index, :, :].transpose(0, 2, 1).reshape(
                    rank * reference_dimension, cutoff
                )
                updated = rows @ inter_pulse.T
                state[:, basis_index, :, :] = updated.reshape(
                    rank, reference_dimension, cutoff
                ).transpose(0, 2, 1)
        diagnostics.append(bus_diagnostics(state, cache))
    return diagnostics


def fixture_run(fixture: Fixture, cache: PulseCache) -> dict[str, Any]:
    state = supplied_state(fixture)
    backing_object_id = id(state)
    backing_pointer = int(state.__array_interface__["data"][0])
    initial_bus = bus_density(state).copy()  # verifier-only; never read by pulse execution
    initial_process_br = bus_reference_density(state).copy()
    initial_br = initial_process_br if fixture.reference_dimension > 1 else None

    diagnostics_a = run_schedule_in_place(state, cache, accepted_schedule(PROGRAM_A))
    client_after_a = client_density(state)
    bus_after_a = bus_density(state)
    br_after_a = bus_reference_density(state) if initial_br is not None else None
    boundary_a = observe(client_after_a, BOUNDARY_A_NAMES)
    factorization_after_a = complete_joint_factorization_frobenius(
        state, COMPILED_A_DENSITY, initial_process_br
    )

    # Program B consumes the same coherent client+bus state in the same ndarray.
    # There is no client detach, CPTP replacement, carrier reload, or re-factorization.
    diagnostics_b = run_schedule_in_place(state, cache, accepted_schedule(PROGRAM_B))
    client_after_b = client_density(state)
    bus_after_b = bus_density(state)
    br_after_b = bus_reference_density(state) if initial_br is not None else None
    boundary_combined = observe(client_after_b, BOUNDARY_COMBINED_NAMES)
    factorization_after_b = complete_joint_factorization_frobenius(
        state, COMPILED_COMBINED_DENSITY, initial_process_br
    )

    reference_metrics: dict[str, Any] | None = None
    if initial_br is not None and br_after_a is not None and br_after_b is not None:
        fidelity_a = float(np.trace(initial_br @ br_after_a).real)
        fidelity_b = float(np.trace(initial_br @ br_after_b).real)
        reference_metrics = {
            "dimension": initial_br.shape[0],
            "after_a_trace_distance": metric(trace_distance(br_after_a, initial_br)),
            "after_b_trace_distance": metric(trace_distance(br_after_b, initial_br)),
            "after_a_entanglement_infidelity": metric(max(0.0, 1.0 - fidelity_a)),
            "after_b_entanglement_infidelity": metric(max(0.0, 1.0 - fidelity_b)),
            "after_a_density_integrity": density_integrity(br_after_a),
            "after_b_density_integrity": density_integrity(br_after_b),
        }

    all_diagnostics = diagnostics_a + diagnostics_b
    return {
        "fixture": fixture.name,
        "reference_dimension": fixture.reference_dimension,
        "ensemble_rank": fixture.ensemble.shape[0],
        "logical_resident_custody": {
            "allocation_object_unchanged": id(state) == backing_object_id,
            "allocation_base_pointer_unchanged": int(state.__array_interface__["data"][0])
            == backing_pointer,
            "carrier_supply_count": 1,
            "program_count": 2,
            "client_detach_count": 0,
            "client_replacement_count": 0,
            "post_supply_carrier_state_set_count": 0,
            "snapshot_count": 0,
            "reload_count": 0,
            "reinitialize_count": 0,
            "privileged_midpoint_boundary_reads": 1,
            "guest_boundary_release_count": 1,
            "generation_sequence": [0, 1, 2],
            "physical_same_mode_custody_established": False,
        },
        "program_a_privileged_nondestructive_boundary": {
            "client_trace_distance_to_direct_compiler": metric(
                trace_distance(client_after_a, COMPILED_A_DENSITY)
            ),
            "bus_trace_distance_to_supplied": metric(trace_distance(bus_after_a, initial_bus)),
            "boundary": boundary_a,
            "boundary_max_abs_error_to_compiler": metric(
                boundary_error(boundary_a, COMPILED_A_BOUNDARY)
            ),
            "complete_joint_to_compiled_client_tensor_supplied_br_frobenius": metric(
                factorization_after_a
            ),
            "client_density_integrity": density_integrity(client_after_a),
            "bus_density_integrity": density_integrity(bus_after_a),
        },
        "combined_a_then_b_released_boundary": {
            "client_trace_distance_to_direct_compiler": metric(
                trace_distance(client_after_b, COMPILED_COMBINED_DENSITY)
            ),
            "bus_trace_distance_to_supplied": metric(trace_distance(bus_after_b, initial_bus)),
            "boundary": boundary_combined,
            "boundary_max_abs_error_to_compiler": metric(
                boundary_error(boundary_combined, COMPILED_COMBINED_BOUNDARY)
            ),
            "complete_joint_to_compiled_client_tensor_supplied_br_frobenius": metric(
                factorization_after_b
            ),
            "client_density_integrity": density_integrity(client_after_b),
            "bus_density_integrity": density_integrity(bus_after_b),
        },
        "bus_reference_coherence": reference_metrics,
        "process_diagnostics": {
            "maximum_mean_excitation": metric(
                max(record["mean_excitation"] for record in all_diagnostics)
            ),
            "maximum_top_edge_probability": metric(
                max(record["top_edge_probability"] for record in all_diagnostics)
            ),
            "per_pulse": all_diagnostics,
        },
        "resource_accounting": {
            "canonical_joint_backing_complex_cells": int(state.size),
            "canonical_joint_backing_bytes": int(state.nbytes),
            "verifier_bus_baseline_complex_cells": int(initial_bus.size),
            "verifier_br_baseline_complex_cells": int(initial_process_br.size),
            "largest_explicit_row_update_complex_cells": int(
                state.shape[0] * state.shape[3] * state.shape[2]
            ),
            "largest_verifier_factorization_block_complex_cells": int(
                (state.shape[2] * state.shape[3]) ** 2
            ),
            "ensemble_factorization_is_exact_but_resource_counted": True,
            "unreported_linear_algebra_scratch_exists": True,
        },
    }


def pure_vacuum_state(cutoff: int) -> np.ndarray:
    vector = np.zeros(cutoff, dtype=np.complex128)
    vector[0] = 1.0
    return supplied_state(pure_fixture("vacuum_control", vector))


def control_result(
    cache: PulseCache,
    schedule: list[tuple[str, float, int]],
    inter_pulse: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = pure_vacuum_state(cache.cutoff)
    initial_bus = bus_density(state).copy()
    run_schedule_in_place(state, cache, schedule, inter_pulse=inter_pulse)
    return client_density(state), bus_density(state), initial_bus


def run_controls(cutoff: int = 64) -> dict[str, Any]:
    cache = PulseCache(cutoff)
    accepted = accepted_schedule(PROGRAM_A)

    reverse_schedule = [
        ("X", -PROGRAM_A["lambda"], PROGRAM_A["a_client"]),
        ("P", -PROGRAM_A["mu"], PROGRAM_A["b_client"]),
        ("X", +PROGRAM_A["lambda"], PROGRAM_A["a_client"]),
        ("P", +PROGRAM_A["mu"], PROGRAM_A["b_client"]),
    ]
    reverse_client, reverse_bus, reverse_initial = control_result(cache, reverse_schedule)
    reverse_boundary = observe(reverse_client, BOUNDARY_A_NAMES)

    commuting_schedule = [
        ("X" if quadrature == "P" else quadrature, amount, client_index)
        for quadrature, amount, client_index in accepted
    ]
    commuting_client, commuting_bus, commuting_initial = control_result(cache, commuting_schedule)
    commuting_boundary = observe(commuting_client, BOUNDARY_A_NAMES)

    omitted_client, omitted_bus, omitted_initial = control_result(cache, accepted[:-1])

    sham_state = pure_vacuum_state(cutoff)
    sham_snapshot = sham_state.copy()
    sham_initial_bus = bus_density(sham_state).copy()
    run_schedule_in_place(sham_state, cache, accepted[:-1])
    sham_pre_reload_bus_distance = trace_distance(bus_density(sham_state), sham_initial_bus)
    sham_state[:] = sham_snapshot
    sham_post_reload_bus_distance = trace_distance(bus_density(sham_state), sham_initial_bus)

    area_schedule = accepted.copy()
    final_q, final_amount, final_client = area_schedule[-1]
    area_schedule[-1] = (final_q, final_amount * 1.05, final_client)
    area_client, area_bus, area_initial = control_result(cache, area_schedule)

    free_rotation = expm(-1j * 0.05 * cache.number)
    rotation_client, rotation_bus, rotation_initial = control_result(
        cache, accepted, inter_pulse=free_rotation
    )

    kerr = expm(-1j * 0.02 * (cache.number @ (cache.number - np.eye(cutoff))))
    kerr_client, kerr_bus, kerr_initial = control_result(cache, accepted, inter_pulse=kerr)

    # Exact coherence-sensitive sham on |Phi_4>_BR.  Complete Fock dephasing
    # keeps the bus marginal I_4/4 but destroys bus-reference coherence.
    phi = np.zeros(16, dtype=np.complex128)
    for index in range(4):
        phi[4 * index + index] = 0.5
    phi_density = np.outer(phi, phi.conj())
    dephased = np.zeros_like(phi_density)
    for index in range(4):
        location = 4 * index + index
        dephased[location, location] = 0.25
    phi_tensor = phi_density.reshape(4, 4, 4, 4)
    dephased_tensor = dephased.reshape(4, 4, 4, 4)
    phi_bus = np.einsum("nqmq->nm", phi_tensor)
    dephased_bus = np.einsum("nqmq->nm", dephased_tensor)

    return {
        "cutoff": cutoff,
        "reverse_rectangle": {
            "bus_trace_distance": metric(trace_distance(reverse_bus, reverse_initial)),
            "Y0Z1": reverse_boundary["Y0Z1"],
            "sign_sensitive_separation_from_accepted": metric(
                abs(reverse_boundary["Y0Z1"] - COMPILED_A_BOUNDARY["Y0Z1"])
            ),
        },
        "commuting_quadrature_sham": {
            "bus_trace_distance": metric(trace_distance(commuting_bus, commuting_initial)),
            "client_trace_distance_to_initial_plus": metric(
                trace_distance(commuting_client, np.outer(PLUS3, PLUS3.conj()))
            ),
            "X0": commuting_boundary["X0"],
            "Y0Z1": commuting_boundary["Y0Z1"],
        },
        "omitted_final_pulse": {
            "bus_trace_distance": metric(trace_distance(omitted_bus, omitted_initial)),
            "client_trace_distance_to_compiled": metric(
                trace_distance(omitted_client, COMPILED_A_DENSITY)
            ),
        },
        "snapshot_reload_sham": {
            "classification": "SNAPSHOT_RELOAD",
            "pre_reload_bus_trace_distance": metric(sham_pre_reload_bus_distance),
            "post_reload_bus_trace_distance": metric(sham_post_reload_bus_distance),
            "snapshot_count": 1,
            "reload_count": 1,
            "native_restoration_count": 0,
        },
        "final_q_area_error_0p05": {
            "client_trace_distance_to_compiled": metric(
                trace_distance(area_client, COMPILED_A_DENSITY)
            ),
            "bus_trace_distance": metric(trace_distance(area_bus, area_initial)),
        },
        "free_rotation_0p05_between_pulses": {
            "client_trace_distance_to_compiled": metric(
                trace_distance(rotation_client, COMPILED_A_DENSITY)
            ),
            "bus_trace_distance": metric(trace_distance(rotation_bus, rotation_initial)),
        },
        "kerr_n_n_minus_1_0p02_between_pulses": {
            "client_trace_distance_to_compiled": metric(
                trace_distance(kerr_client, COMPILED_A_DENSITY)
            ),
            "bus_trace_distance": metric(trace_distance(kerr_bus, kerr_initial)),
        },
        "marginal_only_dephasing_sham": {
            "bus_marginal_trace_distance": metric(trace_distance(phi_bus, dephased_bus)),
            "bus_reference_trace_distance": metric(trace_distance(phi_density, dephased)),
            "entanglement_fidelity": metric(float(np.trace(phi_density @ dephased).real)),
        },
        "pulse_cache_complex_cells": cache.cached_complex_cells,
        "pulse_expm_calls": cache.expm_calls,
    }


def top_fock_counterexample(cache: PulseCache) -> dict[str, Any]:
    vector = np.zeros(cache.cutoff, dtype=np.complex128)
    vector[-1] = 1.0
    state = supplied_state(pure_fixture("top_fock", vector))
    initial_bus = bus_density(state).copy()
    run_schedule_in_place(state, cache, accepted_schedule(PROGRAM_A))
    return {
        "client_trace_distance_to_ideal": metric(
            trace_distance(client_density(state), COMPILED_A_DENSITY)
        ),
        "bus_trace_distance_to_supplied": metric(
            trace_distance(bus_density(state), initial_bus)
        ),
    }


def vacuum_midloop_causality(cutoff: int = FINAL_CUTOFF) -> dict[str, float]:
    cache = PulseCache(cutoff)
    state = pure_vacuum_state(cutoff)
    first_pulse = accepted_schedule(PROGRAM_A)[0]
    apply_conditional_pulse_in_place(state, cache, *first_pulse)
    client_entropy = entropy_nats(client_density(state))
    overlap = math.exp(-(PROGRAM_A["mu"] ** 2))
    eigenvalue = (1.0 + overlap) / 2.0
    exact_entropy = -eigenvalue * math.log(eigenvalue) - (1.0 - eigenvalue) * math.log(
        1.0 - eigenvalue
    )
    return {
        "coherent_branch_overlap_exact": metric(overlap),
        "client_entropy_numeric_nats": metric(client_entropy),
        "client_entropy_exact_nats": metric(exact_entropy),
        "pure_joint_mutual_information_numeric_nats": metric(2.0 * client_entropy),
        "pure_joint_mutual_information_exact_nats": metric(2.0 * exact_entropy),
    }


def finite_commutator_record(cache: PulseCache) -> dict[str, float]:
    cutoff = cache.cutoff
    top = np.zeros((cutoff, cutoff), dtype=np.complex128)
    top[-1, -1] = 1.0
    commutator = cache.x @ cache.p - cache.p @ cache.x
    exact_finite_law = 1j * (np.eye(cutoff) - cutoff * top)
    residual = float(np.max(np.abs(commutator - exact_finite_law)))
    defect_norm = float(np.linalg.norm(commutator - 1j * np.eye(cutoff), ord=2))
    return {
        "finite_law_max_abs_residual": metric(residual),
        "operator_norm_distance_from_infinite_ccr": metric(defect_norm),
        "expected_operator_norm_distance": float(cutoff),
    }


def integrity_passes(record: dict[str, float]) -> bool:
    return (
        record["trace_error"] <= THRESHOLDS["density_trace_error_max"]
        and record["hermiticity_max_abs"] <= THRESHOLDS["density_hermiticity_max"]
        and record["minimum_eigenvalue"] >= THRESHOLDS["density_min_eigenvalue_min"]
    )


def build_result() -> dict[str, Any]:
    start = time.perf_counter()
    cutoff_records: dict[str, Any] = {}
    all_fixture_records: list[dict[str, Any]] = []
    top_fock_records: dict[str, Any] = {}
    commutator_records: dict[str, Any] = {}
    cutoff_resources: dict[str, Any] = {}

    for cutoff in CUTOFFS:
        cache = PulseCache(cutoff)
        records = [fixture_run(fixture, cache) for fixture in fixtures(cutoff)]
        top_record = top_fock_counterexample(cache)
        commutator_record = finite_commutator_record(cache)
        cutoff_records[str(cutoff)] = records
        all_fixture_records.extend(records)
        top_fock_records[str(cutoff)] = top_record
        commutator_records[str(cutoff)] = commutator_record
        cutoff_resources[str(cutoff)] = {
            "pulse_cache_complex_cells": cache.cached_complex_cells,
            "pulse_cache_bytes": cache.cached_complex_cells * 16,
            "unique_truncated_pulse_exponentials": cache.expm_calls,
            "maximum_canonical_joint_backing_complex_cells": max(
                record["resource_accounting"]["canonical_joint_backing_complex_cells"]
                for record in records
            ),
            "maximum_canonical_joint_backing_bytes": max(
                record["resource_accounting"]["canonical_joint_backing_bytes"]
                for record in records
            ),
            "dense_exponential_and_matrix_multiply_work_scales_as": "THETA_N_CUBED",
            "canonical_mixed_ensemble_state_scales_as": "8_N_SQUARED_COMPLEX_CELLS",
        }

    controls = run_controls()
    midloop = vacuum_midloop_causality()

    analytic_a_error = boundary_error(COMPILED_A_BOUNDARY, ANALYTIC_A_BOUNDARY)
    analytic_combined_error = boundary_error(
        COMPILED_COMBINED_BOUNDARY, ANALYTIC_COMBINED_BOUNDARY
    )
    final_records = cutoff_records[str(FINAL_CUTOFF)]
    initial_records_by_name = {
        record["fixture"]: record for record in cutoff_records[str(CUTOFFS[0])]
    }
    final_records_by_name = {record["fixture"]: record for record in final_records}
    convergence_fixture_names = (
        "coherent_alpha_0p65_plus_0p20i",
        "squeezed_r_0p45_phi_0p30",
        "thermal_nbar_0p4",
        "phi4_bus_reference",
    )
    declared_named_fixture_convergence = {
        name: final_records_by_name[name]["combined_a_then_b_released_boundary"]
        ["complete_joint_to_compiled_client_tensor_supplied_br_frobenius"]
        < initial_records_by_name[name]["combined_a_then_b_released_boundary"]
        ["complete_joint_to_compiled_client_tensor_supplied_br_frobenius"]
        for name in convergence_fixture_names
    }
    checks: dict[str, bool] = {
        "compiled_a_matches_independent_analytic_moments": analytic_a_error <= 1e-12,
        "compiled_combined_matches_independent_analytic_moments": analytic_combined_error
        <= 1e-12,
        "all_cutoffs_executed": tuple(int(value) for value in cutoff_records) == CUTOFFS,
        "declared_named_fixture_sequences_improve_from_n16_to_n128": all(
            declared_named_fixture_convergence.values()
        ),
        "all_final_client_a_gates_match_direct_compiler": all(
            record["program_a_privileged_nondestructive_boundary"][
                "client_trace_distance_to_direct_compiler"
            ]
            <= THRESHOLDS["final_client_trace_distance_max"]
            for record in final_records
        ),
        "all_final_combined_client_gates_match_direct_compiler": all(
            record["combined_a_then_b_released_boundary"][
                "client_trace_distance_to_direct_compiler"
            ]
            <= THRESHOLDS["final_client_trace_distance_max"]
            for record in final_records
        ),
        "all_final_bus_returns_after_a": all(
            record["program_a_privileged_nondestructive_boundary"][
                "bus_trace_distance_to_supplied"
            ]
            <= THRESHOLDS["final_bus_trace_distance_after_a_max"]
            for record in final_records
        ),
        "all_final_bus_returns_after_combined_reuse": all(
            record["combined_a_then_b_released_boundary"][
                "bus_trace_distance_to_supplied"
            ]
            <= THRESHOLDS["final_bus_trace_distance_after_b_max"]
            for record in final_records
        ),
        "all_final_boundaries_match": all(
            record["program_a_privileged_nondestructive_boundary"][
                "boundary_max_abs_error_to_compiler"
            ]
            <= THRESHOLDS["boundary_max_abs_error"]
            and record["combined_a_then_b_released_boundary"][
                "boundary_max_abs_error_to_compiler"
            ]
            <= THRESHOLDS["boundary_max_abs_error"]
            for record in final_records
        ),
        "all_final_complete_joint_states_factorize": all(
            record["program_a_privileged_nondestructive_boundary"][
                "complete_joint_to_compiled_client_tensor_supplied_br_frobenius"
            ]
            <= THRESHOLDS["complete_joint_factorization_frobenius_max"]
            and record["combined_a_then_b_released_boundary"][
                "complete_joint_to_compiled_client_tensor_supplied_br_frobenius"
            ]
            <= THRESHOLDS["complete_joint_factorization_frobenius_max"]
            for record in final_records
        ),
        "all_final_density_integrity_gates": all(
            integrity_passes(
                record["program_a_privileged_nondestructive_boundary"][
                    "client_density_integrity"
                ]
            )
            and integrity_passes(
                record["program_a_privileged_nondestructive_boundary"][
                    "bus_density_integrity"
                ]
            )
            and integrity_passes(
                record["combined_a_then_b_released_boundary"]["client_density_integrity"]
            )
            and integrity_passes(
                record["combined_a_then_b_released_boundary"]["bus_density_integrity"]
            )
            for record in final_records
        ),
        "all_trials_keep_one_logical_backing_across_a_and_b": all(
            record["logical_resident_custody"]["allocation_object_unchanged"]
            and record["logical_resident_custody"]["allocation_base_pointer_unchanged"]
            and record["logical_resident_custody"]["carrier_supply_count"] == 1
            and record["logical_resident_custody"]["client_detach_count"] == 0
            and record["logical_resident_custody"]["client_replacement_count"] == 0
            and record["logical_resident_custody"][
                "post_supply_carrier_state_set_count"
            ]
            == 0
            and record["logical_resident_custody"]["snapshot_count"] == 0
            and record["logical_resident_custody"]["reload_count"] == 0
            and record["logical_resident_custody"]["guest_boundary_release_count"] == 1
            for record in all_fixture_records
        ),
        "phi4_reference_coherence_returns": sum(
            record["bus_reference_coherence"] is not None for record in final_records
        )
        == 1
        and final_records_by_name["phi4_bus_reference"]["reference_dimension"] == 4
        and final_records_by_name["phi4_bus_reference"]["bus_reference_coherence"]
        is not None
        and final_records_by_name["phi4_bus_reference"]["bus_reference_coherence"][
            "after_a_trace_distance"
        ]
        <= THRESHOLDS["reference_br_trace_distance_max"]
        and final_records_by_name["phi4_bus_reference"]["bus_reference_coherence"][
            "after_b_trace_distance"
        ]
        <= THRESHOLDS["reference_br_trace_distance_max"]
        and final_records_by_name["phi4_bus_reference"]["bus_reference_coherence"][
            "after_a_entanglement_infidelity"
        ]
        <= THRESHOLDS["reference_entanglement_infidelity_max"]
        and final_records_by_name["phi4_bus_reference"]["bus_reference_coherence"][
            "after_b_entanglement_infidelity"
        ]
        <= THRESHOLDS["reference_entanglement_infidelity_max"]
        and integrity_passes(
            final_records_by_name["phi4_bus_reference"]["bus_reference_coherence"][
                "after_a_density_integrity"
            ]
        )
        and integrity_passes(
            final_records_by_name["phi4_bus_reference"]["bus_reference_coherence"][
                "after_b_density_integrity"
            ]
        ),
        "vacuum_midloop_is_causally_entangled": midloop[
            "pure_joint_mutual_information_numeric_nats"
        ]
        >= THRESHOLDS["vacuum_midloop_mutual_information_min_nats"],
        "finite_ccr_law_reconstructed": all(
            record["finite_law_max_abs_residual"] <= 1e-12
            and abs(
                record["operator_norm_distance_from_infinite_ccr"]
                - record["expected_operator_norm_distance"]
            )
            <= 1e-10
            for record in commutator_records.values()
        ),
        "top_fock_rejects_uniform_finite_state_return": all(
            record["client_trace_distance_to_ideal"]
            >= THRESHOLDS["top_fock_client_trace_distance_min"]
            and record["bus_trace_distance_to_supplied"]
            >= THRESHOLDS["top_fock_bus_trace_distance_min"]
            for record in top_fock_records.values()
        ),
        "reverse_rectangle_returns_bus_and_flips_phase": controls["reverse_rectangle"][
            "bus_trace_distance"
        ]
        <= THRESHOLDS["ideal_control_return_max"]
        and controls["reverse_rectangle"]["sign_sensitive_separation_from_accepted"]
        >= THRESHOLDS["sign_sensitive_separation_min"],
        "commuting_quadrature_sham_has_no_geometric_phase": controls[
            "commuting_quadrature_sham"
        ]["bus_trace_distance"]
        <= THRESHOLDS["ideal_control_return_max"]
        and controls["commuting_quadrature_sham"]["client_trace_distance_to_initial_plus"]
        <= THRESHOLDS["ideal_control_return_max"]
        and abs(controls["commuting_quadrature_sham"]["Y0Z1"]) <= 1e-10,
        "omitted_final_pulse_fails_return": controls["omitted_final_pulse"][
            "bus_trace_distance"
        ]
        >= THRESHOLDS["omission_bus_trace_distance_min"],
        "snapshot_control_is_reload_not_restoration": controls["snapshot_reload_sham"][
            "pre_reload_bus_trace_distance"
        ]
        >= THRESHOLDS["omission_bus_trace_distance_min"]
        and controls["snapshot_reload_sham"]["post_reload_bus_trace_distance"] <= 1e-12
        and controls["snapshot_reload_sham"]["reload_count"] == 1,
        "reference_dephasing_detects_coherence_loss": controls[
            "marginal_only_dephasing_sham"
        ]["bus_marginal_trace_distance"]
        <= 1e-12
        and abs(
            controls["marginal_only_dephasing_sham"]["bus_reference_trace_distance"]
            - 0.75
        )
        <= 1e-12
        and abs(
            controls["marginal_only_dephasing_sham"]["entanglement_fidelity"] - 0.25
        )
        <= 1e-12,
        "area_error_is_detected": controls["final_q_area_error_0p05"][
            "client_trace_distance_to_compiled"
        ]
        >= THRESHOLDS["noise_area_client_and_bus_min"]
        and controls["final_q_area_error_0p05"]["bus_trace_distance"]
        >= THRESHOLDS["noise_area_client_and_bus_min"],
        "free_rotation_is_detected": controls["free_rotation_0p05_between_pulses"][
            "client_trace_distance_to_compiled"
        ]
        >= THRESHOLDS["noise_rotation_client_and_bus_min"]
        and controls["free_rotation_0p05_between_pulses"]["bus_trace_distance"]
        >= THRESHOLDS["noise_rotation_client_and_bus_min"],
        "kerr_is_detected": controls["kerr_n_n_minus_1_0p02_between_pulses"][
            "client_trace_distance_to_compiled"
        ]
        >= THRESHOLDS["noise_kerr_client_min"]
        and controls["kerr_n_n_minus_1_0p02_between_pulses"]["bus_trace_distance"]
        >= THRESHOLDS["noise_kerr_bus_min"],
    }

    wall_seconds = time.perf_counter() - start
    result = {
        "schema": "PHASE_QEMU_V7_WEYL_LOOP_BUS_RESULT_V1",
        "milestone": "M265",
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "contract_sha256": sha256_file(HERE / "PHASE_QEMU_V7_WEYL_LOOP_CONTRACT.md"),
        "ideal_infinite_ccr_law": {
            "quadratures": "X=(a+a_dagger)/sqrt(2), P=(a-a_dagger)/(i*sqrt(2)), [X,P]=i",
            "operator_product": "Q_A(+lambda) R_B(+mu) Q_A(-lambda) R_B(-mu)",
            "chronological_pulses": [
                "R_B(-mu)",
                "Q_A(-lambda)",
                "R_B(+mu)",
                "Q_A(+lambda)",
            ],
            "central_commutator": "[-i lambda A X,-i mu B P]=-i lambda mu A B",
            "exact_factorization": "exp(-i lambda mu A B) tensor I_bus",
            "scope": "COMMUTING_BOUNDED_CLIENT_OPERATORS_AND_ARBITRARY_NORMAL_JOINT_INPUT_ON_THE_IDEAL_INFINITE_CCR_REPRESENTATION",
            "finite_cutoff_uniform_arbitrary_state_claim": False,
        },
        "programs": [PROGRAM_A, PROGRAM_B],
        "direct_compiled_oracle": {
            "program_a_boundary": COMPILED_A_BOUNDARY,
            "combined_boundary": COMPILED_COMBINED_BOUNDARY,
            "analytic_program_a_boundary": {
                key: metric(value) for key, value in ANALYTIC_A_BOUNDARY.items()
            },
            "analytic_combined_boundary": {
                key: metric(value) for key, value in ANALYTIC_COMBINED_BOUNDARY.items()
            },
            "analytic_program_a_max_abs_error": metric(analytic_a_error),
            "analytic_combined_max_abs_error": metric(analytic_combined_error),
            "direct_zz_phase_gates": 2,
            "weyl_conditional_pulses": 8,
            "selected_boundary_compact_formula_work": "O_SUM_OF_SELECTED_VERTEX_DEGREES",
            "software_forward_shadow_omits_bus_and_return": True,
        },
        "thresholds": THRESHOLDS,
        "cutoff_sweep": cutoff_records,
        "declared_named_fixture_convergence": declared_named_fixture_convergence,
        "finite_cutoff_commutator": commutator_records,
        "top_fock_nonuniform_counterexample": top_fock_records,
        "midloop_causal_witness": midloop,
        "controls": controls,
        "acceptance_checks": checks,
        "accepted_energy_constrained_numerical_return": all(checks.values()),
        "resource_ledger": {
            "physical_modes_in_model": 1,
            "client_qubits": 3,
            "client_role": "BOUNDED_NON_GAUSSIAN_CALIBRATION_LOAD_ONLY",
            "cutoffs": list(CUTOFFS),
            "complex_component_bits": COMPLEX_COMPONENT_BITS,
            "cutoff_resources": cutoff_resources,
            "conditional_pulses_per_program": 4,
            "conditional_pulses_total": 8,
            "program_descriptor_size_is_constant_for_this_fixture": True,
            "program_a_max_rectangle_excursion": metric(
                math.sqrt(PROGRAM_A["lambda"] ** 2 + PROGRAM_A["mu"] ** 2)
                / math.sqrt(2.0)
            ),
            "program_b_max_rectangle_excursion": metric(
                math.sqrt(PROGRAM_B["lambda"] ** 2 + PROGRAM_B["mu"] ** 2)
                / math.sqrt(2.0)
            ),
            "joules_uninstantiated_without_hardware_mapping": True,
            "bandwidth_and_pulse_duration_uninstantiated_without_hardware_mapping": True,
            "retained_public_program_descriptors": 2,
            "public_schedule_entries": 8,
            "retained_dynamic_trajectory_history_complex_cells": 0,
            "retained_inverse_history_entries": 0,
            "privileged_full_initial_density_baselines_retained_for_validation": True,
            "privileged_validation_baselines_readable_by_dynamics": False,
            "history_free_complete_experiment_claim": False,
            "linear_algebra_library_internal_scratch_not_instrumented": True,
            "process_max_rss_kib_not_used_for_claim": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "wall_seconds_not_used_for_claim": metric(wall_seconds),
        },
        "classical_comparison": {
            "best_fixture_comparator": "DIRECT_COMPILED_COMMUTING_ZZ_PHASE_AND_SELECTED_MOMENT_PRODUCT_FORMULAS",
            "phase_qemu_emulator_has_resource_advantage": False,
            "m257_escape_established": False,
            "growing_edge_family_bus_pulses": "4E",
            "growing_edge_family_direct_compiled_gates": "E",
            "generic_client_state_and_boundary_costs_not_removed": True,
        },
        "claim": "IDEAL_INFINITE_CCR_WEYL_COMMUTATOR_FACTORIZATION_WITH_ARBITRARY_NORMAL_STATE_BUS_IDENTITY_AND_FINITE_ENERGY_CONSTRAINED_TRUNCATED_FOCK_NUMERICAL_CONVERGENCE_ON_A_BOUNDED_THREE_QUBIT_CALIBRATION_LOAD",
        "claim_ceiling": "DETERMINISTIC_COMPLEX128_SOFTWARE_EMULATION_WITH_LOGICAL_RESIDENT_ARRAY_CUSTODY_DIRECT_COMPILED_FORWARD_SHADOW_AND_NO_PHYSICAL_SAME_MODE_CUSTODY",
        "restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "restoration_scope": "ENERGY_CONSTRAINED_COMPLEX128_LOGICAL_RESIDENT_BACKING_BUS_AND_REFERENCE_RETURN_WITH_CLIENT_TRANSFORMATION_AND_COMPLETE_FACTORIZATION_AT_CUTOFF128_WITHOUT_PHYSICAL_SAME_MODE_CUSTODY",
        "registry_taxonomy_label_does_not_establish_physical_hardware": True,
        "resource_disposition": "DIRECT_COMPILED_ZZ_FORWARD_SHADOW_STRICTLY_OMITS_THE_BUS_LOOP_AND_NO_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED",
        "next_mechanism": "MULTIMODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_WEYL_LOOP_DIGITAL_TWIN_WITH_HEATING_SPECTATOR_MODE_CLOSURE_CONTROLLER_COST_AND_ENERGY_CONSTRAINED_SAME_MODE_REUSE",
        "physical_execution": False,
        "physical_same_mode_restoration": False,
        "phase_native_client_architecture": False,
        "replace_the_bit_with_pi_established": False,
        "computational_advantage": False,
        "unbounded_compute": False,
        "m257_intact": True,
        "source_self_assertion": "PASS_ENERGY_CONSTRAINED_LOGICAL_RETURN",
        "status": "PASS_ENERGY_CONSTRAINED_LOGICAL_RETURN",
        "terminal": False,
    }
    return result


def main() -> int:
    result = build_result()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted_energy_constrained_numerical_return"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
