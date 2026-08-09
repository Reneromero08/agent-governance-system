#!/usr/bin/env python3
"""Independent M264 near-threshold time-domain scattering oracle.

This source intentionally does not import or execute the production model.  It
reconstructs the public four-spin/one-port fixture and uses an independent
fourth-order split-operator route: the open lead is exponentiated by an
orthonormal DST-I, the target by dense diagonalization, and the site-zero
coupling by a separate dense exponential.  The target preparation is checked
both with the public midpoint schedule and an independent continuous DOP853
integration.

Only aggregate diagnostics are emitted.  No wavefunctions, amplitudes, or time
histories are serialized.  Passing the source self-check is not package,
physical, restoration, same-backing, resource-advantage, or M257 qualification.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
from scipy.fft import dst
from scipy.integrate import solve_ivp


STATUS = "SOURCE_SELF_CHECK_PASS"
CLASSIFICATION = "NO_PHYSICAL_RESTORATION_SAME_BACKING_RESOURCE_ADVANTAGE_OR_M257_CLAIM"
SCOPE = "EXECUTED_FINITE_NUMERICAL_MODEL_AND_SEPARATE_REFERENCE_PARITY_ONLY"
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
FAIL_DISPOSITION = (
    "STRICT_PREPARATION_AND_FINITE_PACKET_RESTORATION_OBSTRUCTION_RETAINS_"
    "REAL_TRANSIENT_INTERACTION_DRAIN_AND_APPROXIMATE_FUNCTIONAL_HANDOFF_BUT_"
    "REQUIRES_A_CHANGED_RETURN_PREPARATION_LAW_NOT_POST_HOC_FIXTURE_TUNING"
)
NEXT_MECHANISM = (
    "RESIDENT_OPEN_DRAIN_OR_ECHO_RETURN_WITH_PAID_GROUND_STATE_SUPPLY_"
    "PREDECLARED_FINITE_PACKET_ERROR_AND_SAME_BACKING_TARGET_CUSTODY"
)

N_SPINS = 4
TARGET_DIM = 1 << N_SPINS
LEAD_LENGTH = 641
X0 = 320
SIGMA_X = 50
DETECTOR_LEFT = 192
DETECTOR_RIGHT = 193
FINAL_TIME = 340.0
DELAY_CONTROL_TIME = 370.0
OBSERVATION_DT = 2.0
INTERNAL_DT = 0.125
COARSE_DT = 0.25
HOPPING = 1.0
COUPLING = 1.5
K_A = 2.0 * math.pi / 5.0
K_B = 9.0 * math.pi / 20.0
K_ABOVE = math.pi / 2.0
PREP_TIME = 120.0
PREP_STEPS = 480
TAIL_LIMIT = 1.0e-9
MATCHED_FREE_RETURN_LIMIT = 1.0e-7
MIXED_DENSITY_DISCARDED_TRACE_LIMIT = 1.0e-12


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def sf(value: float) -> float:
    """Stable aggregate serialization without using rounded values in tests."""
    return float(f"{float(value):.13g}")


def complex_record(value: complex) -> dict[str, float]:
    return {"real": sf(value.real), "imag": sf(value.imag)}


def wrap_phase(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def tensor_word(operators: dict[int, np.ndarray]) -> np.ndarray:
    answer = np.array([[1.0 + 0.0j]])
    for site in range(N_SPINS):
        answer = np.kron(answer, operators.get(site, I2))
    return answer


XS = [tensor_word({site: X}) for site in range(N_SPINS)]
ZS = [tensor_word({site: Z}) for site in range(N_SPINS)]


def make_target(disconnect_boundary: bool = False, remove_three_body: bool = False) -> np.ndarray:
    h = -sum(XS)
    for site in range(N_SPINS - 1):
        if disconnect_boundary and site == 0:
            continue
        h = h - (3.0 / 4.0) * (ZS[site] @ ZS[site + 1])
    fields = (1.0 / 5.0, 2.0 / 7.0, 3.0 / 11.0, 5.0 / 13.0)
    for field, operator in zip(fields, ZS):
        h = h - field * operator
    if not remove_three_body:
        h = h - 0.5 * (ZS[0] @ ZS[1] @ ZS[2])
    return np.asarray(h, dtype=np.complex128)


H_INITIAL = -sum(XS)
H_TARGET = make_target()
TARGET_ENERGIES, TARGET_VECTORS = np.linalg.eigh(H_TARGET)
GROUND_ENERGY = float(TARGET_ENERGIES[0])
GROUND = TARGET_VECTORS[:, 0]
EXCITATION_ENERGIES = TARGET_ENERGIES - GROUND_ENERGY
H_SHIFTED = H_TARGET - GROUND_ENERGY * np.eye(TARGET_DIM)
TARGET_GAP = float(EXCITATION_ENERGIES[1])
BOUNDARY_OPERATOR = ZS[0]


def smoothstep(u: float) -> float:
    return 3.0 * u * u - 2.0 * u * u * u


def preparation_hamiltonian(t: float) -> np.ndarray:
    s = smoothstep(min(1.0, max(0.0, t / PREP_TIME)))
    return (1.0 - s) * H_INITIAL + s * H_TARGET


def prepare_midpoint_law(duration: float, steps: int) -> np.ndarray:
    state = np.full(TARGET_DIM, 1.0 / math.sqrt(TARGET_DIM), dtype=np.complex128)
    dt = duration / steps
    for step in range(steps):
        u = (step + 0.5) / steps
        s = smoothstep(u)
        h_mid = (1.0 - s) * H_INITIAL + s * H_TARGET
        state = scipy.linalg.expm(-1j * dt * h_mid) @ state
    return state / np.linalg.norm(state)


def prepare_midpoint() -> np.ndarray:
    return prepare_midpoint_law(PREP_TIME, PREP_STEPS)


def prepare_continuous_reference() -> tuple[np.ndarray, int]:
    initial = np.full(TARGET_DIM, 1.0 / math.sqrt(TARGET_DIM), dtype=np.complex128)

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        return -1j * (preparation_hamiltonian(t) @ state)

    result = solve_ivp(
        rhs,
        (0.0, PREP_TIME),
        initial,
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
        t_eval=[PREP_TIME],
    )
    assert result.success
    state = result.y[:, -1]
    return state / np.linalg.norm(state), int(result.nfev)


PREPARED = prepare_midpoint()
PREPARED_T480_CONTROL = prepare_midpoint_law(480.0, 1920)
PREPARED_CONTINUOUS, PREP_CONTINUOUS_NFEV = prepare_continuous_reference()


def expectation(state: np.ndarray, operator: np.ndarray) -> complex:
    return complex(np.vdot(state, operator @ state))


def non_gaussian_witness() -> tuple[float, tuple[int, int, int, int]]:
    majoranas: list[np.ndarray] = []
    for site in range(N_SPINS):
        x_string: dict[int, np.ndarray] = {earlier: X for earlier in range(site)}
        majoranas.append(tensor_word({**x_string, site: Z}))
        majoranas.append(tensor_word({**x_string, site: Y}))
    best = (-1.0, (0, 1, 2, 3))
    for a, b, c, d in itertools.combinations(range(2 * N_SPINS), 4):
        ab = expectation(GROUND, majoranas[a] @ majoranas[b])
        ac = expectation(GROUND, majoranas[a] @ majoranas[c])
        ad = expectation(GROUND, majoranas[a] @ majoranas[d])
        bc = expectation(GROUND, majoranas[b] @ majoranas[c])
        bd = expectation(GROUND, majoranas[b] @ majoranas[d])
        cd = expectation(GROUND, majoranas[c] @ majoranas[d])
        four = expectation(GROUND, majoranas[a] @ majoranas[b] @ majoranas[c] @ majoranas[d])
        residual = abs(four - (ab * cd - ac * bd + ad * bc))
        if residual > best[0]:
            best = (float(residual), (a, b, c, d))
    return best


NON_GAUSSIAN_RESIDUAL, NON_GAUSSIAN_TUPLE = non_gaussian_witness()


def target_mps_diagnostics() -> list[dict[str, float | int]]:
    records = []
    for cut in range(1, N_SPINS):
        singular = np.linalg.svd(GROUND.reshape(1 << cut, -1), compute_uv=False)
        probabilities = singular * singular
        entropy = -sum(float(p) * math.log(float(p)) for p in probabilities if p > 1.0e-16)
        records.append(
            {
                "cut": cut,
                "exact_schmidt_rank_at_1e-12": int(np.count_nonzero(singular > 1.0e-12)),
                "entropy_nats": sf(entropy),
            }
        )
    return records


def lead_eigenvalues(length: int) -> np.ndarray:
    modes = np.arange(1, length + 1, dtype=np.float64)
    return 2.0 - 2.0 * np.cos(math.pi * modes / (length + 1.0))


def packet(length: int, momentum: float, x0: int = X0, sigma: float = SIGMA_X) -> np.ndarray:
    sites = np.arange(length, dtype=np.float64)
    state = np.exp(-((sites - x0) ** 2) / (4.0 * sigma * sigma)) * np.exp(-1j * momentum * sites)
    return state / np.linalg.norm(state)


@dataclass(frozen=True)
class PacketSpectrum:
    momentum: float
    central_energy: float
    mean_energy: float
    std_energy: float
    above_gap_weight: float


def packet_spectrum(momentum: float, length: int = LEAD_LENGTH) -> PacketSpectrum:
    state = packet(length, momentum)
    coefficients = dst(state, type=1, axis=0, norm="ortho")
    weights = np.abs(coefficients) ** 2
    energies = lead_eigenvalues(length)
    mean = float(np.dot(weights, energies))
    variance = float(np.dot(weights, (energies - mean) ** 2))
    return PacketSpectrum(
        momentum=momentum,
        central_energy=2.0 - 2.0 * math.cos(momentum),
        mean_energy=mean,
        std_energy=math.sqrt(max(0.0, variance)),
        above_gap_weight=float(np.sum(weights[energies >= TARGET_GAP])),
    )


SPECTRUM_A = packet_spectrum(K_A)
SPECTRUM_B = packet_spectrum(K_B)
SPECTRUM_ABOVE = packet_spectrum(K_ABOVE)


class SplitPropagator:
    """Fourth-order Yoshida composition of exact A and B exponential actions."""

    def __init__(
        self,
        length: int,
        target_hamiltonian: np.ndarray,
        boundary_operator: np.ndarray,
        coupling: float,
        dt: float,
    ) -> None:
        self.length = length
        self.dt = dt
        self.lead_spectrum = lead_eigenvalues(length)
        self.target_values, self.target_vectors = np.linalg.eigh(target_hamiltonian)
        self.boundary_values, self.boundary_vectors = np.linalg.eigh(boundary_operator)
        cube_root_two = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cube_root_two)
        w0 = -cube_root_two / (2.0 - cube_root_two)
        self.a_times = (0.5 * w1 * dt, 0.5 * (w1 + w0) * dt, 0.5 * (w0 + w1) * dt, 0.5 * w1 * dt)
        self.b_times = (w1 * dt, w0 * dt, w1 * dt)
        self.coupling = coupling
        self.a_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self.b_cache: dict[float, np.ndarray] = {}

    def apply_a(self, state: np.ndarray, duration: float) -> np.ndarray:
        cached = self.a_cache.get(duration)
        if cached is None:
            lead_phase = np.exp(-1j * duration * self.lead_spectrum)
            target_unitary = (
                self.target_vectors
                * np.exp(-1j * duration * self.target_values)[None, :]
            ) @ self.target_vectors.conj().T
            cached = (lead_phase, target_unitary)
            self.a_cache[duration] = cached
        lead_phase, target_unitary = cached
        spectral = dst(state, type=1, axis=0, norm="ortho")
        spectral *= lead_phase.reshape((self.length,) + (1,) * (state.ndim - 1))
        state = dst(spectral, type=1, axis=0, norm="ortho")
        if state.ndim == 2:
            return state @ target_unitary.T
        assert state.ndim == 3
        return np.einsum("ab,xbr->xar", target_unitary, state, optimize=True)

    def apply_b(self, state: np.ndarray, duration: float) -> np.ndarray:
        unitary = self.b_cache.get(duration)
        if unitary is None:
            unitary = (
                self.boundary_vectors
                * np.exp(-1j * duration * self.coupling * self.boundary_values)[None, :]
            ) @ self.boundary_vectors.conj().T
            self.b_cache[duration] = unitary
        if state.ndim == 2:
            state[0, :] = state[0, :] @ unitary.T
        else:
            assert state.ndim == 3
            state[0, :, :] = np.einsum("ab,br->ar", unitary, state[0, :, :], optimize=True)
        return state

    def step(self, state: np.ndarray) -> np.ndarray:
        state = self.apply_a(state, self.a_times[0])
        state = self.apply_b(state, self.b_times[0])
        state = self.apply_a(state, self.a_times[1])
        state = self.apply_b(state, self.b_times[1])
        state = self.apply_a(state, self.a_times[2])
        state = self.apply_b(state, self.b_times[2])
        return self.apply_a(state, self.a_times[3])


def target_density(state: np.ndarray) -> np.ndarray:
    if state.ndim == 2:
        return np.einsum("xa,xb->ab", state, state.conj(), optimize=True)
    assert state.ndim == 3
    return np.einsum("xar,xbr->ab", state, state.conj(), optimize=True)


def trace_distance_to_pure(rho: np.ndarray, reference: np.ndarray) -> float:
    difference = rho - np.outer(reference, reference.conj())
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(rho - sigma))))


def entropy_from_density(rho: np.ndarray) -> float:
    values = np.maximum(np.linalg.eigvalsh(rho), 0.0)
    return -sum(float(v) * math.log(float(v)) for v in values if v > 1.0e-16)


def apply_total_hamiltonian(
    state: np.ndarray,
    target_hamiltonian: np.ndarray,
    boundary_operator: np.ndarray,
    coupling: float,
) -> np.ndarray:
    answer = 2.0 * state.copy()
    answer[:-1, :] -= state[1:, :]
    answer[1:, :] -= state[:-1, :]
    if state.ndim == 2:
        answer += state @ target_hamiltonian.T
        answer[0, :] += coupling * (state[0, :] @ boundary_operator.T)
    else:
        assert state.ndim == 3
        answer += np.einsum("ab,xbr->xar", target_hamiltonian, state, optimize=True)
        answer[0, :, :] += coupling * np.einsum("ab,br->ar", boundary_operator, state[0, :, :], optimize=True)
    return answer


def detector_current(state: np.ndarray) -> float:
    return float(2.0 * np.imag(np.vdot(state[DETECTOR_LEFT, ...], state[DETECTOR_RIGHT, ...])))


@dataclass
class RunResult:
    momentum: float
    dt: float
    length: int
    final_state: np.ndarray
    times: np.ndarray
    currents: np.ndarray
    max_excitation: float
    max_excitation_time: float
    final_excitation: float
    peak_site_zero_probability: float
    final_site_zero_probability: float
    final_trace_distance: float
    final_matched_free_trace_distance: float
    final_purity_deficit: float
    max_target_lead_entropy: float
    energy_drift: float
    norm_error: float
    input_ground_phase: complex


@dataclass
class MixedRunResult:
    momentum: float
    dt: float
    length: int
    final_density: np.ndarray
    times: np.ndarray
    currents: np.ndarray
    max_excitation: float
    max_excitation_time: float
    final_excitation: float
    peak_site_zero_probability: float
    final_site_zero_probability: float
    final_ground_trace_distance: float
    final_matched_free_trace_distance: float
    final_purity_deficit: float
    max_target_lead_entropy: float
    energy_drift: float
    norm_error: float
    propagated_rank: int
    discarded_density_trace: float


def run_wavepacket(
    momentum: float,
    target_state: np.ndarray,
    *,
    length: int = LEAD_LENGTH,
    dt: float = INTERNAL_DT,
    target_hamiltonian: np.ndarray = H_SHIFTED,
    boundary_operator: np.ndarray = BOUNDARY_OPERATOR,
    coupling: float = COUPLING,
    reference_ground: np.ndarray | None = None,
) -> RunResult:
    assert abs(FINAL_TIME / dt - round(FINAL_TIME / dt)) < 1.0e-12
    assert abs(OBSERVATION_DT / dt - round(OBSERVATION_DT / dt)) < 1.0e-12
    if reference_ground is None:
        reference_ground = GROUND if target_state.size == TARGET_DIM else np.eye(target_state.size, dtype=np.complex128)[:, 0]
    assert reference_ground.shape == target_state.shape
    input_ground_overlap = complex(np.vdot(reference_ground, target_state))
    assert abs(input_ground_overlap) > 1.0e-8
    input_ground_phase = input_ground_overlap / abs(input_ground_overlap)
    state = packet(length, momentum)[:, None] * target_state[None, :]
    propagator = SplitPropagator(length, target_hamiltonian, boundary_operator, coupling, dt)
    observation_stride = int(round(OBSERVATION_DT / dt))
    total_steps = int(round(FINAL_TIME / dt))
    times: list[float] = []
    currents: list[float] = []
    excitations: list[float] = []
    site_zero: list[float] = []
    entropies: list[float] = []
    initial_norm = float(np.vdot(state, state).real)
    initial_energy = float(np.vdot(state, apply_total_hamiltonian(state, target_hamiltonian, boundary_operator, coupling)).real)
    for step in range(total_steps + 1):
        if step % observation_stride == 0:
            rho = target_density(state)
            ground_probability = float(np.vdot(state @ reference_ground.conj(), state @ reference_ground.conj()).real)
            times.append(step * dt)
            currents.append(detector_current(state))
            excitations.append(max(0.0, 1.0 - ground_probability))
            site_zero.append(float(np.vdot(state[0, :], state[0, :]).real))
            entropies.append(entropy_from_density(rho))
        if step != total_steps:
            state = propagator.step(state)
    final_norm = float(np.vdot(state, state).real)
    final_energy = float(np.vdot(state, apply_total_hamiltonian(state, target_hamiltonian, boundary_operator, coupling)).real)
    final_rho = target_density(state)
    target_free_unitary = scipy.linalg.expm(-1j * FINAL_TIME * target_hamiltonian)
    matched_free_target = target_free_unitary @ target_state
    matched_free_target /= np.linalg.norm(matched_free_target)
    max_index = int(np.argmax(excitations))
    return RunResult(
        momentum=momentum,
        dt=dt,
        length=length,
        final_state=state,
        times=np.asarray(times),
        currents=np.asarray(currents),
        max_excitation=float(excitations[max_index]),
        max_excitation_time=float(times[max_index]),
        final_excitation=float(excitations[-1]),
        peak_site_zero_probability=float(max(site_zero)),
        final_site_zero_probability=float(site_zero[-1]),
        final_trace_distance=trace_distance_to_pure(final_rho, reference_ground),
        final_matched_free_trace_distance=trace_distance_to_pure(final_rho, matched_free_target),
        final_purity_deficit=max(0.0, 1.0 - float(np.trace(final_rho @ final_rho).real)),
        max_target_lead_entropy=float(max(entropies)),
        energy_drift=abs(final_energy - initial_energy),
        norm_error=abs(final_norm - initial_norm),
        input_ground_phase=input_ground_phase,
    )


def run_mixed_wavepacket(momentum: float, target_rho: np.ndarray, *, dt: float = INTERNAL_DT) -> MixedRunResult:
    """Propagate every nonnegative spectral component of a target density.

    The rank axis is a classical ensemble/purification index and is never
    coherently summed.  This is a numerical rematerialization of the returned
    density, not resident same-backing reuse.
    """
    target_rho = 0.5 * (target_rho + target_rho.conj().T)
    target_rho = target_rho / float(np.trace(target_rho).real)
    values, vectors = np.linalg.eigh(target_rho)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    values /= float(np.sum(values))
    cumulative = np.cumsum(values)
    rank = int(np.searchsorted(cumulative, 1.0 - MIXED_DENSITY_DISCARDED_TRACE_LIMIT, side="left") + 1)
    rank = min(values.size, max(2, rank))
    discarded_density_trace = max(0.0, 1.0 - float(np.sum(values[:rank])))
    values = values[:rank]
    vectors = vectors[:, :rank]
    values /= float(np.sum(values))
    branches = vectors * np.sqrt(values)[None, :]
    state = packet(LEAD_LENGTH, momentum)[:, None, None] * branches[None, :, :]
    propagator = SplitPropagator(LEAD_LENGTH, H_SHIFTED, BOUNDARY_OPERATOR, COUPLING, dt)
    observation_stride = int(round(OBSERVATION_DT / dt))
    total_steps = int(round(FINAL_TIME / dt))
    times: list[float] = []
    currents: list[float] = []
    excitations: list[float] = []
    site_zero: list[float] = []
    entropies: list[float] = []
    initial_norm = float(np.vdot(state, state).real)
    initial_energy = float(np.vdot(state, apply_total_hamiltonian(state, H_SHIFTED, BOUNDARY_OPERATOR, COUPLING)).real)
    for step in range(total_steps + 1):
        if step % observation_stride == 0:
            rho = target_density(state)
            ground_amplitudes = np.einsum("a,xar->xr", GROUND.conj(), state, optimize=True)
            ground_probability = float(np.vdot(ground_amplitudes, ground_amplitudes).real)
            times.append(step * dt)
            currents.append(detector_current(state))
            excitations.append(max(0.0, 1.0 - ground_probability))
            site_zero.append(float(np.vdot(state[0, ...], state[0, ...]).real))
            entropies.append(entropy_from_density(rho))
        if step != total_steps:
            state = propagator.step(state)
    final_norm = float(np.vdot(state, state).real)
    final_energy = float(np.vdot(state, apply_total_hamiltonian(state, H_SHIFTED, BOUNDARY_OPERATOR, COUPLING)).real)
    final_rho = target_density(state)
    target_free = scipy.linalg.expm(-1j * FINAL_TIME * H_SHIFTED)
    matched_free_rho = target_free @ target_rho @ target_free.conj().T
    max_index = int(np.argmax(excitations))
    return MixedRunResult(
        momentum=momentum,
        dt=dt,
        length=LEAD_LENGTH,
        final_density=final_rho,
        times=np.asarray(times),
        currents=np.asarray(currents),
        max_excitation=float(excitations[max_index]),
        max_excitation_time=float(times[max_index]),
        final_excitation=float(excitations[-1]),
        peak_site_zero_probability=float(max(site_zero)),
        final_site_zero_probability=float(site_zero[-1]),
        final_ground_trace_distance=trace_distance_to_pure(final_rho, GROUND),
        final_matched_free_trace_distance=trace_distance(final_rho, matched_free_rho),
        final_purity_deficit=max(0.0, 1.0 - float(np.trace(final_rho @ final_rho).real)),
        max_target_lead_entropy=float(max(entropies)),
        energy_drift=abs(final_energy - initial_energy),
        norm_error=abs(final_norm - initial_norm),
        propagated_rank=rank,
        discarded_density_trace=discarded_density_trace,
    )


def exact_free_lead_state(momentum: float, time: float, length: int = LEAD_LENGTH) -> np.ndarray:
    initial = packet(length, momentum)
    coefficients = dst(initial, type=1, axis=0, norm="ortho")
    coefficients *= np.exp(-1j * lead_eigenvalues(length) * time)
    return dst(coefficients, type=1, axis=0, norm="ortho")


def free_current_series(momentum: float, times: np.ndarray, length: int = LEAD_LENGTH) -> np.ndarray:
    currents = []
    for time in times:
        state = exact_free_lead_state(momentum, float(time), length)
        currents.append(float(2.0 * np.imag(np.conj(state[DETECTOR_LEFT]) * state[DETECTOR_RIGHT])))
    return np.asarray(currents)


def current_centroid(
    times: np.ndarray,
    currents: np.ndarray,
    *,
    outgoing: bool,
) -> tuple[float, float]:
    mask = times >= 180.0 if outgoing else times <= 140.0
    selected_times = times[mask]
    signed = np.maximum(currents[mask], 0.0) if outgoing else np.maximum(-currents[mask], 0.0)
    flux = float(np.trapezoid(signed, selected_times))
    assert flux > 1.0e-8
    centroid = float(np.trapezoid(selected_times * signed, selected_times) / flux)
    return centroid, flux


def iq_metrics(run: RunResult) -> dict[str, float | dict[str, float]]:
    # Quotient the arbitrary global phase of the input target vector.  This is
    # essential for a principal eigenvector reconstructed from a density
    # matrix; its eigensolver gauge has no physical IQ meaning.
    outgoing_ground = (run.final_state @ GROUND.conj()) / run.input_ground_phase
    free_final = exact_free_lead_state(run.momentum, FINAL_TIME, run.length)
    norm_product = math.sqrt(float(np.vdot(outgoing_ground, outgoing_ground).real) * float(np.vdot(free_final, free_final).real))
    same = complex(np.vdot(free_final, outgoing_ground) / norm_product)

    def negative_overlap(delay: float) -> float:
        shifted = exact_free_lead_state(run.momentum, FINAL_TIME - delay, run.length)
        overlap = np.vdot(shifted, outgoing_ground)
        denominator = math.sqrt(float(np.vdot(shifted, shifted).real) * float(np.vdot(outgoing_ground, outgoing_ground).real))
        return -float(abs(overlap) ** 2 / (denominator * denominator))

    optimum = scipy.optimize.minimize_scalar(
        negative_overlap,
        bounds=(-10.0, 25.0),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    delay = float(optimum.x)
    shifted = exact_free_lead_state(run.momentum, FINAL_TIME - delay, run.length)
    denominator = math.sqrt(float(np.vdot(shifted, shifted).real) * float(np.vdot(outgoing_ground, outgoing_ground).real))
    aligned = complex(np.vdot(shifted, outgoing_ground) / denominator)
    return {
        "same_time_iq": complex_record(same),
        "same_time_phase": sf(np.angle(same)),
        "same_time_visibility": sf(abs(same)),
        "best_mode_delay": sf(delay),
        "aligned_iq": complex_record(aligned),
        "aligned_phase": sf(np.angle(aligned)),
        "aligned_visibility": sf(abs(aligned)),
        "aligned_mode_fidelity": sf(abs(aligned) ** 2),
    }


def time_domain_metrics(run: RunResult) -> dict[str, object]:
    free_currents = free_current_series(run.momentum, run.times, run.length)
    candidate_out, candidate_out_flux = current_centroid(run.times, run.currents, outgoing=True)
    free_out, free_out_flux = current_centroid(run.times, free_currents, outgoing=True)
    candidate_in, candidate_in_flux = current_centroid(run.times, run.currents, outgoing=False)
    free_in, free_in_flux = current_centroid(run.times, free_currents, outgoing=False)
    traversal_delay = (candidate_out - candidate_in) - (free_out - free_in)
    return {
        "max_transient_target_excitation": sf(run.max_excitation),
        "max_excitation_time": sf(run.max_excitation_time),
        "final_target_excitation": sf(run.final_excitation),
        "peak_site_zero_probability": sf(run.peak_site_zero_probability),
        "final_site_zero_probability": sf(run.final_site_zero_probability),
        "final_target_trace_distance_from_ground": sf(run.final_trace_distance),
        "final_target_trace_distance_from_matched_free_target": sf(run.final_matched_free_trace_distance),
        "final_target_purity_deficit": sf(run.final_purity_deficit),
        "max_target_lead_entropy_nats": sf(run.max_target_lead_entropy),
        "candidate_incoming_current_centroid": sf(candidate_in),
        "free_incoming_current_centroid": sf(free_in),
        "candidate_outgoing_current_centroid": sf(candidate_out),
        "free_outgoing_current_centroid": sf(free_out),
        "current_centroid_delay": sf(traversal_delay),
        "candidate_integrated_incoming_current": sf(candidate_in_flux),
        "free_integrated_incoming_current": sf(free_in_flux),
        "candidate_integrated_outgoing_current": sf(candidate_out_flux),
        "free_integrated_outgoing_current": sf(free_out_flux),
        "energy_drift": sf(run.energy_drift),
        "norm_error": sf(run.norm_error),
        "iq": iq_metrics(run),
    }


def mixed_time_domain_metrics(run: MixedRunResult) -> dict[str, object]:
    free_currents = free_current_series(run.momentum, run.times, run.length)
    candidate_out, candidate_out_flux = current_centroid(run.times, run.currents, outgoing=True)
    free_out, free_out_flux = current_centroid(run.times, free_currents, outgoing=True)
    candidate_in, candidate_in_flux = current_centroid(run.times, run.currents, outgoing=False)
    free_in, free_in_flux = current_centroid(run.times, free_currents, outgoing=False)
    return {
        "propagated_density_rank": run.propagated_rank,
        "discarded_input_density_trace_bound": sf(run.discarded_density_trace),
        "max_transient_target_excitation": sf(run.max_excitation),
        "max_excitation_time": sf(run.max_excitation_time),
        "final_target_excitation": sf(run.final_excitation),
        "peak_site_zero_probability": sf(run.peak_site_zero_probability),
        "final_site_zero_probability": sf(run.final_site_zero_probability),
        "final_target_trace_distance_from_ground": sf(run.final_ground_trace_distance),
        "final_target_trace_distance_from_matched_free_returned_density": sf(run.final_matched_free_trace_distance),
        "final_target_purity_deficit": sf(run.final_purity_deficit),
        "max_target_lead_entropy_nats": sf(run.max_target_lead_entropy),
        "candidate_incoming_current_centroid": sf(candidate_in),
        "free_incoming_current_centroid": sf(free_in),
        "candidate_outgoing_current_centroid": sf(candidate_out),
        "free_outgoing_current_centroid": sf(free_out),
        "current_centroid_delay": sf((candidate_out - candidate_in) - (free_out - free_in)),
        "candidate_integrated_incoming_current": sf(candidate_in_flux),
        "free_integrated_incoming_current": sf(free_in_flux),
        "candidate_integrated_outgoing_current": sf(candidate_out_flux),
        "free_integrated_outgoing_current": sf(free_out_flux),
        "energy_drift": sf(run.energy_drift),
        "norm_error": sf(run.norm_error),
        "iq": "NOT_DEFINED_BY_GAUGE_ARBITRARY_DENSITY_EIGENVECTORS_WITHOUT_AN_EXPLICIT_REFERENCE_ARM_CHANNEL",
    }


BOUNDARY_IN_EIGENBASIS = TARGET_VECTORS.conj().T @ BOUNDARY_OPERATOR @ TARGET_VECTORS


def stationary_reflection(energy: float, coupling: float = COUPLING) -> complex:
    assert 0.0 < energy < TARGET_GAP
    k = math.acos(1.0 - 0.5 * energy)
    diagonal = np.empty(TARGET_DIM, dtype=np.complex128)
    diagonal[0] = np.exp(-1j * k)
    for channel in range(1, TARGET_DIM):
        z = 2.0 + float(EXCITATION_ENERGIES[channel]) - energy
        decay = 0.5 * (z - math.sqrt(z * z - 4.0))
        diagonal[channel] = 1.0 / decay
    matrix = np.diag(diagonal) + coupling * BOUNDARY_IN_EIGENBASIS
    source = np.zeros(TARGET_DIM, dtype=np.complex128)
    source[0] = -2j * math.sin(k)
    endpoint = np.linalg.solve(matrix, source)
    return complex(endpoint[0] - 1.0)


def free_reflection(energy: float) -> complex:
    k = math.acos(1.0 - 0.5 * energy)
    return complex(-np.exp(2j * k))


def stationary_record(momentum: float) -> dict[str, object]:
    energy = 2.0 - 2.0 * math.cos(momentum)
    h = min(1.0e-5, energy / 10.0, (TARGET_GAP - energy) / 10.0)

    def derivative(function) -> complex:
        return (
            -function(energy + 2.0 * h)
            + 8.0 * function(energy + h)
            - 8.0 * function(energy - h)
            + function(energy - 2.0 * h)
        ) / (12.0 * h)

    reflection = stationary_reflection(energy)
    reference = free_reflection(energy)
    delay = float(np.imag(derivative(stationary_reflection) / reflection - derivative(free_reflection) / reference))
    h2 = h / 2.0

    def derivative_h2(function) -> complex:
        return (
            -function(energy + 2.0 * h2)
            + 8.0 * function(energy + h2)
            - 8.0 * function(energy - h2)
            + function(energy - 2.0 * h2)
        ) / (12.0 * h2)

    delay_h2 = float(np.imag(derivative_h2(stationary_reflection) / reflection - derivative_h2(free_reflection) / reference))
    relative = reflection / reference
    return {
        "energy": sf(energy),
        "reflection": complex_record(reflection),
        "unit_modulus_residual": sf(abs(abs(reflection) - 1.0)),
        "relative_phase": sf(np.angle(relative)),
        "relative_wigner_delay": sf(delay_h2),
        "derivative_h_vs_h_over_2_difference": sf(abs(delay - delay_h2)),
    }


def packet_weighted_stationary_delay(momentum: float) -> tuple[float, float]:
    initial = packet(LEAD_LENGTH, momentum)
    weights = np.abs(dst(initial, type=1, axis=0, norm="ortho")) ** 2
    energies = lead_eigenvalues(LEAD_LENGTH)
    weighted_delay = 0.0
    retained = 0.0
    for weight, energy in zip(weights, energies):
        if weight < 1.0e-17 or not (1.0e-6 < energy < TARGET_GAP - 3.0e-5):
            continue
        h = min(1.0e-5, energy / 10.0, (TARGET_GAP - energy) / 10.0)
        rp = stationary_reflection(float(energy + h))
        rm = stationary_reflection(float(energy - h))
        r = stationary_reflection(float(energy))
        fp = free_reflection(float(energy + h))
        fm = free_reflection(float(energy - h))
        fr = free_reflection(float(energy))
        derivative_r = (rp - rm) / (2.0 * h)
        derivative_f = (fp - fm) / (2.0 * h)
        tau = float(np.imag(derivative_r / r - derivative_f / fr))
        weighted_delay += float(weight) * tau
        retained += float(weight)
    return weighted_delay / retained, retained


def run_scalar_static(momentum: float, potential: float) -> RunResult:
    return run_wavepacket(
        momentum,
        np.ones(1, dtype=np.complex128),
        target_hamiltonian=np.zeros((1, 1), dtype=np.complex128),
        boundary_operator=np.ones((1, 1), dtype=np.complex128),
        coupling=potential,
    )


def scalar_metrics(run: RunResult) -> dict[str, object]:
    free_currents = free_current_series(run.momentum, run.times, run.length)
    candidate_out, _ = current_centroid(run.times, run.currents, outgoing=True)
    free_out, _ = current_centroid(run.times, free_currents, outgoing=True)
    candidate_in, candidate_in_flux = current_centroid(run.times, run.currents, outgoing=False)
    free_in, free_in_flux = current_centroid(run.times, free_currents, outgoing=False)
    outgoing = run.final_state[:, 0]
    free_final = exact_free_lead_state(run.momentum, FINAL_TIME, run.length)
    same = np.vdot(free_final, outgoing) / math.sqrt(float(np.vdot(free_final, free_final).real * np.vdot(outgoing, outgoing).real))
    return {
        "current_centroid_delay": sf((candidate_out - candidate_in) - (free_out - free_in)),
        "candidate_integrated_incoming_current": sf(candidate_in_flux),
        "free_integrated_incoming_current": sf(free_in_flux),
        "same_time_phase": sf(np.angle(same)),
        "same_time_visibility": sf(abs(same)),
        "peak_site_zero_probability": sf(run.peak_site_zero_probability),
        "final_site_zero_probability": sf(run.final_site_zero_probability),
        "norm_error": sf(run.norm_error),
    }


def run_record(run: RunResult) -> dict[str, object]:
    return time_domain_metrics(run)


def continue_delay_control(run: RunResult) -> dict[str, float]:
    """Continue only the pure-query T340 state to the public T370 delay seal."""
    state = run.final_state.copy()
    propagator = SplitPropagator(run.length, H_SHIFTED, BOUNDARY_OPERATOR, COUPLING, run.dt)
    extra_steps = int(round((DELAY_CONTROL_TIME - FINAL_TIME) / run.dt))
    observation_stride = int(round(OBSERVATION_DT / run.dt))
    extra_times: list[float] = []
    extra_currents: list[float] = []
    for step in range(1, extra_steps + 1):
        state = propagator.step(state)
        if step % observation_stride == 0:
            extra_times.append(FINAL_TIME + step * run.dt)
            extra_currents.append(detector_current(state))
    times = np.concatenate((run.times, np.asarray(extra_times)))
    currents = np.concatenate((run.currents, np.asarray(extra_currents)))
    free_currents = free_current_series(run.momentum, times, run.length)
    candidate_out, candidate_out_flux = current_centroid(times, currents, outgoing=True)
    free_out, free_out_flux = current_centroid(times, free_currents, outgoing=True)
    candidate_in, candidate_in_flux = current_centroid(times, currents, outgoing=False)
    free_in, free_in_flux = current_centroid(times, free_currents, outgoing=False)
    return {
        "stop_time": sf(DELAY_CONTROL_TIME),
        "current_centroid_delay": sf((candidate_out - candidate_in) - (free_out - free_in)),
        "candidate_integrated_incoming_current": sf(candidate_in_flux),
        "free_integrated_incoming_current": sf(free_in_flux),
        "candidate_integrated_outgoing_current": sf(candidate_out_flux),
        "free_integrated_outgoing_current": sf(free_out_flux),
        "final_site_zero_probability": sf(float(np.vdot(state[0, :], state[0, :]).real)),
    }


# A one-entry sparse projector is constructed explicitly.  A single-valued
# scipy.sparse.diags main diagonal would broadcast across the entire lead.
P0 = scipy.sparse.csr_matrix(([1.0], ([0], [0])), shape=(LEAD_LENGTH, LEAD_LENGTH))
assert P0.nnz == 1
assert P0.indices.tolist() == [0]
assert P0.indptr[1] == 1 and P0.indptr[-1] == 1


RUN_A = run_wavepacket(K_A, PREPARED)
RUN_B_CLEAN = run_wavepacket(K_B, PREPARED)
RUN_A_COARSE = run_wavepacket(K_A, PREPARED, dt=COARSE_DT)
RUN_A_LONG_LEAD = run_wavepacket(K_A, PREPARED, length=769)
RUN_B_EXACT_GROUND = run_wavepacket(K_B, GROUND)
RUN_ABOVE = run_wavepacket(K_ABOVE, GROUND)

RHO_AFTER_A = target_density(RUN_A.final_state)
RHO_AFTER_A = 0.5 * (RHO_AFTER_A + RHO_AFTER_A.conj().T)
RHO_AFTER_A /= float(np.trace(RHO_AFTER_A).real)
RETURN_VALUES, RETURN_VECTORS = np.linalg.eigh(RHO_AFTER_A)
RETURN_ORDER = np.argsort(RETURN_VALUES)[::-1]
RETURN_VALUES = RETURN_VALUES[RETURN_ORDER]
RETURN_VECTORS = RETURN_VECTORS[:, RETURN_ORDER]
RETURN_PRINCIPAL_WEIGHT = float(max(0.0, RETURN_VALUES[0]))
RETURN_NONPRINCIPAL_TRACE = max(0.0, 1.0 - RETURN_PRINCIPAL_WEIGHT)
RETURN_NEGATIVE_NUMERICAL_MASS = float(np.sum(np.maximum(-RETURN_VALUES, 0.0)))
RETURN_PRINCIPAL = RETURN_VECTORS[:, 0]
RUN_B_RETURNED_MIXED = run_mixed_wavepacket(K_B, RHO_AFTER_A)

GROUND_BOUNDARY_MEAN = float(expectation(GROUND, BOUNDARY_OPERATOR).real)
STATIC_POTENTIAL = COUPLING * GROUND_BOUNDARY_MEAN
RUN_STATIC_A = run_scalar_static(K_A, STATIC_POTENTIAL)
RUN_STATIC_B = run_scalar_static(K_B, STATIC_POTENTIAL)

METRICS_A = run_record(RUN_A)
METRICS_B = run_record(RUN_B_CLEAN)
DELAY_CONTROL_A = continue_delay_control(RUN_A)
DELAY_CONTROL_B = continue_delay_control(RUN_B_CLEAN)
METRICS_RETURNED_B = mixed_time_domain_metrics(RUN_B_RETURNED_MIXED)
METRICS_EXACT_B = run_record(RUN_B_EXACT_GROUND)
METRICS_ABOVE = run_record(RUN_ABOVE)

STATIONARY_A = stationary_record(K_A)
STATIONARY_B = stationary_record(K_B)
WEIGHTED_DELAY_A, WEIGHT_RETAINED_A = packet_weighted_stationary_delay(K_A)
WEIGHTED_DELAY_B, WEIGHT_RETAINED_B = packet_weighted_stationary_delay(K_B)


def metric_phase(record: dict[str, object]) -> float:
    return float(record["iq"]["same_time_phase"])  # type: ignore[index]


def metric_delay(record: dict[str, object]) -> float:
    return float(record["current_centroid_delay"])


REUSE_DELAY_DIFFERENCE = abs(metric_delay(METRICS_RETURNED_B) - metric_delay(METRICS_B))
REUSE_FINAL_DENSITY_TRACE_DISTANCE = trace_distance(
    RUN_B_RETURNED_MIXED.final_density,
    target_density(RUN_B_CLEAN.final_state),
)
PREP_PHASE_DIFFERENCE = abs(wrap_phase(metric_phase(METRICS_EXACT_B) - metric_phase(METRICS_B)))
PREP_DELAY_DIFFERENCE = abs(metric_delay(METRICS_EXACT_B) - metric_delay(METRICS_B))

COARSE_A_METRICS = run_record(RUN_A_COARSE)
LONG_A_METRICS = run_record(RUN_A_LONG_LEAD)
STEP_DELAY_DIFFERENCE = abs(metric_delay(METRICS_A) - metric_delay(COARSE_A_METRICS))
STEP_PHASE_DIFFERENCE = abs(wrap_phase(metric_phase(METRICS_A) - metric_phase(COARSE_A_METRICS)))
# Yoshida is fourth order.  The h/2 result's leading Richardson estimate is
# |fine-coarse|/(2^4-1); this is an estimate, not a rigorous truncation bound.
STEP_FINE_DELAY_ERROR_ESTIMATE = STEP_DELAY_DIFFERENCE / 15.0
STEP_FINE_PHASE_ERROR_ESTIMATE = STEP_PHASE_DIFFERENCE / 15.0


def shots(visibility: float, phase_error: float, alpha: float = 0.01) -> int:
    return int(math.ceil(4.0 * math.log(4.0 / alpha) / (visibility * visibility * phase_error * phase_error)))


VISIBILITY_A = float(METRICS_A["iq"]["aligned_visibility"])  # type: ignore[index]
VISIBILITY_B = float(METRICS_B["iq"]["aligned_visibility"])  # type: ignore[index]


PREP_GROUND_INFIDELITY = max(0.0, 1.0 - abs(np.vdot(GROUND, PREPARED)) ** 2)
PREP_T480_CONTROL_GROUND_INFIDELITY = max(0.0, 1.0 - abs(np.vdot(GROUND, PREPARED_T480_CONTROL)) ** 2)
PREP_CONTINUOUS_INFIDELITY = max(0.0, 1.0 - abs(np.vdot(GROUND, PREPARED_CONTINUOUS)) ** 2)
PREP_DISCRETIZATION_INFIDELITY = max(0.0, 1.0 - abs(np.vdot(PREPARED_CONTINUOUS, PREPARED)) ** 2)
CONNECTED_ZZ01 = float(
    (expectation(GROUND, ZS[0] @ ZS[1]) - expectation(GROUND, ZS[0]) * expectation(GROUND, ZS[1])).real
)


assert SPECTRUM_A.above_gap_weight < TAIL_LIMIT
assert SPECTRUM_B.above_gap_weight < TAIL_LIMIT
assert SPECTRUM_ABOVE.above_gap_weight > 0.9
assert NON_GAUSSIAN_RESIDUAL > 1.0e-2
assert abs(CONNECTED_ZZ01) > 1.0e-2
assert PREP_GROUND_INFIDELITY < 1.0e-7
assert RUN_A.max_excitation > 1.0e-2
assert RUN_A.final_excitation < 1.0e-6
assert RUN_B_CLEAN.max_excitation > 1.0e-2
assert RUN_B_CLEAN.final_excitation < 1.0e-5
assert RUN_ABOVE.final_excitation > 1.0e-4
assert RUN_A.norm_error < 1.0e-9 and RUN_B_CLEAN.norm_error < 1.0e-9
assert RUN_A.final_matched_free_trace_distance > MATCHED_FREE_RETURN_LIMIT
assert RUN_B_CLEAN.final_matched_free_trace_distance > MATCHED_FREE_RETURN_LIMIT
assert float(STATIONARY_A["unit_modulus_residual"]) < 1.0e-10
assert float(STATIONARY_B["unit_modulus_residual"]) < 1.0e-10
assert REUSE_DELAY_DIFFERENCE < 1.0e-4
assert REUSE_FINAL_DENSITY_TRACE_DISTANCE < 1.0e-4
assert RUN_B_RETURNED_MIXED.discarded_density_trace <= MIXED_DENSITY_DISCARDED_TRACE_LIMIT
assert STEP_FINE_DELAY_ERROR_ESTIMATE < 0.03
assert STEP_FINE_PHASE_ERROR_ESTIMATE < 2.0e-3
assert abs(metric_delay(METRICS_A) - metric_delay(LONG_A_METRICS)) < 1.0e-4


def spectrum_record(value: PacketSpectrum) -> dict[str, float]:
    return {
        "momentum": sf(value.momentum),
        "central_energy": sf(value.central_energy),
        "dst_mean_energy": sf(value.mean_energy),
        "dst_std_energy": sf(value.std_energy),
        "dst_weight_at_or_above_target_gap": sf(value.above_gap_weight),
    }


OUTPUT = {
    "schema": "PHASE_QEMU_V6_TIME_DOMAIN_SCATTERING_SEPARATE_REFERENCE_EVIDENCE_V1",
    "milestone": "M264",
    "source_self_assertion": STATUS,
    "verification_scope": {
        "science": "SEPARATE_REFERENCE_PARITY",
        "theory": "FORMAL_DERIVATION_SOURCE_AUDITED",
        "resource": "PACKAGE_SELF_REVIEW",
    },
    "claim": FAIL_CLAIM,
    "claim_ceiling": CLAIM_CEILING,
    "disposition": FAIL_DISPOSITION,
    "next_mechanism": NEXT_MECHANISM,
    "restoration": {
        "classification": "NO_RESTORATION_CLAIM",
        "scope": "FAILED_DECLARED_TIME_DOMAIN_RETURN_OR_REUSE_THRESHOLDS",
        "executed_time_domain_target_excursion_and_drain": True,
        "returned_state_rematerialization_used": True,
        "generic_target_state_repreparation_via_density_rematerialization_used": True,
        "same_backing_restoration_established": False,
        "same_backing_reuse_established": False,
        "physical_restoration_established": False,
        "permanent_restoration_established": False,
    },
    "metadata": {
        "experiment": "NEAR_THRESHOLD_NONINTEGRABLE_BOUNDARY_RESOLVENT_WITH_EXPLICIT_WIGNER_DELAY_FINITE_BANDWIDTH_PRECISION_PREPARATION_AMORTIZATION_AND_TENSOR_NETWORK_RESOURCE_CROSSOVER",
        "role": "SEPARATE_REFERENCE_ORACLE",
        "derivation_scope": "FORMULAS_RECONSTRUCTED_WITHOUT_PRODUCTION_IMPORT_OR_CALL",
        "status": STATUS,
        "classification": CLASSIFICATION,
        "scope": SCOPE,
        "claims": {
            "executed_numerical_model": True,
            "physical_observation": False,
            "physical_restoration": False,
            "same_backing_reuse": False,
            "resource_advantage": False,
            "m257_escape": False,
            "package_qualification": False,
        },
    },
    "package_qualification": {
        "result": "FAIL_MATCHED_FREE_TARGET_RETURN_GATE",
        "promotion": "DENIED",
        "predeclared_matched_free_target_trace_distance_limit": sf(MATCHED_FREE_RETURN_LIMIT),
        "query_A_observed": sf(RUN_A.final_matched_free_trace_distance),
        "query_B_observed": sf(RUN_B_CLEAN.final_matched_free_trace_distance),
        "obstruction": "TINY_FINAL_GROUND_EXCITATION_DOES_NOT_IMPLY_RETURN_TO_THE_MATCHED_FREELY_EVOLVED_PREPARED_TARGET_STATE",
        "exact_ground_and_T480_results": "DECOMPOSITION_CONTROLS_ONLY_NOT_A_GATE_REPAIR",
        "source_self_check_pass_does_not_imply_package_pass": True,
    },
    "public_fixture": {
        "target_spins": N_SPINS,
        "target_dimension": TARGET_DIM,
        "lead_length": LEAD_LENGTH,
        "joint_state_dimension": LEAD_LENGTH * TARGET_DIM,
        "packet_center_site": X0,
        "packet_sigma_x": SIGMA_X,
        "detector_bond": [DETECTOR_LEFT, DETECTOR_RIGHT],
        "final_time": sf(FINAL_TIME),
        "observation_grid_dt": sf(OBSERVATION_DT),
        "independent_internal_dt": sf(INTERNAL_DT),
        "coupling": sf(COUPLING),
        "coupling_operator": "(3/2)|0><0|_lead tensor Z0_target",
        "source_off_after_packet_initialization": True,
        "site_zero_projector": {
            "construction": "EXPLICIT_SINGLE_ENTRY_CSR_NOT_SPARSE_DIAGS_BROADCAST",
            "nnz": int(P0.nnz),
            "only_coordinate": [0, 0],
            "support_assertion_pass": True,
        },
    },
    "target": {
        "hamiltonian": "-sum_j X_j -(3/4)sum_j Z_jZ_{j+1} -(1/5 Z0+2/7 Z1+3/11 Z2+5/13 Z3) -(1/2)Z0Z1Z2",
        "ground_energy": sf(GROUND_ENERGY),
        "gap": sf(TARGET_GAP),
        "connected_Z0_Z1_correlation": sf(CONNECTED_ZZ01),
        "maximum_four_majorana_wick_residual": sf(NON_GAUSSIAN_RESIDUAL),
        "witness_majorana_indices": list(NON_GAUSSIAN_TUPLE),
        "witness_interpretation": "NON_GAUSSIAN_DIAGNOSTIC_NOT_A_HARDNESS_PROOF",
        "ground_mps_cuts": target_mps_diagnostics(),
    },
    "preparation": {
        "law": "H(s)=(1-s)(-sum X)+s H_target; s(u)=3u^2-2u^3",
        "duration": sf(PREP_TIME),
        "public_midpoint_steps": PREP_STEPS,
        "midpoint_ground_infidelity": sf(PREP_GROUND_INFIDELITY),
        "T480_duration_control": {
            "duration": 480,
            "midpoint_steps": 1920,
            "ground_infidelity": sf(PREP_T480_CONTROL_GROUND_INFIDELITY),
            "role": "DECOMPOSITION_CONTROL_ONLY_NOT_A_REPAIR_OR_PROMOTION",
        },
        "continuous_DOP853_ground_infidelity": sf(PREP_CONTINUOUS_INFIDELITY),
        "midpoint_vs_continuous_infidelity": sf(PREP_DISCRETIZATION_INFIDELITY),
        "continuous_reference_rhs_evaluations": PREP_CONTINUOUS_NFEV,
        "exact_ground_injection_is_control_only": True,
        "amortization_law": "C_total_per_query(Q)=C_prep/Q+C_query",
        "amortization_query_counts_reported": [1, 2, 4, 8, 16],
        "preparation_cost_not_hidden": True,
    },
    "finite_bandwidth": {
        "query_A": spectrum_record(SPECTRUM_A),
        "query_B": spectrum_record(SPECTRUM_B),
        "above_threshold_control": spectrum_record(SPECTRUM_ABOVE),
        "target_gap": sf(TARGET_GAP),
        "tail_limit": sf(TAIL_LIMIT),
        "sigma_correction": "SIGMA_50_REPLACES_PRE_SEMANTIC_SIGMA_48_SO_QUERY_B_DST_ABOVE_GAP_WEIGHT_IS_BELOW_1E-9",
    },
    "stationary_dense_channel_reference": {
        "query_A": STATIONARY_A,
        "query_B": STATIONARY_B,
        "query_A_packet_weighted_relative_delay": sf(WEIGHTED_DELAY_A),
        "query_A_packet_weight_retained": sf(WEIGHT_RETAINED_A),
        "query_B_packet_weighted_relative_delay": sf(WEIGHTED_DELAY_B),
        "query_B_packet_weight_retained": sf(WEIGHT_RETAINED_B),
        "role": "INDEPENDENT_SMALL_TARGET_CHANNEL_RECONSTRUCTION_NOT_TIME_DOMAIN_PROPAGATOR",
    },
    "time_domain": {
        "solver": "FOURTH_ORDER_YOSHIDA_SPLIT_OPERATOR_WITH_ORTHONORMAL_DST_I_LEAD_AND_DENSE_TARGET_AND_ENDPOINT_EXPONENTIALS",
        "query_A": METRICS_A,
        "query_B_clean": METRICS_B,
        "T370_delay_control": {
            "query_A": DELAY_CONTROL_A,
            "query_B": DELAY_CONTROL_B,
            "role": "FINITE_WINDOW_CONVERGENCE_CONTROL_NOT_THE_T340_RETURN_GATE",
        },
        "formal_stationary_vs_time_domain_delay_difference_A": sf(abs(WEIGHTED_DELAY_A - metric_delay(METRICS_A))),
        "formal_stationary_vs_time_domain_delay_difference_B": sf(abs(WEIGHTED_DELAY_B - metric_delay(METRICS_B))),
        "step_halving_control_A": {
            "coarse_dt": sf(COARSE_DT),
            "fine_dt": sf(INTERNAL_DT),
            "current_delay_difference": sf(STEP_DELAY_DIFFERENCE),
            "same_time_phase_difference": sf(STEP_PHASE_DIFFERENCE),
            "fine_delay_Richardson_error_estimate": sf(STEP_FINE_DELAY_ERROR_ESTIMATE),
            "fine_phase_Richardson_error_estimate": sf(STEP_FINE_PHASE_ERROR_ESTIMATE),
            "maximum_excitation_difference": sf(abs(RUN_A.max_excitation - RUN_A_COARSE.max_excitation)),
        },
        "finite_lead_control_A": {
            "lengths": [LEAD_LENGTH, 769],
            "current_delay_difference": sf(abs(metric_delay(METRICS_A) - metric_delay(LONG_A_METRICS))),
            "same_time_phase_difference": sf(abs(wrap_phase(metric_phase(METRICS_A) - metric_phase(LONG_A_METRICS)))),
            "final_excitation_difference": sf(abs(RUN_A.final_excitation - RUN_A_LONG_LEAD.final_excitation)),
        },
    },
    "executed_distinct_energy_reuse": {
        "procedure": "TRACE_QUERY_A_PROBE_COMPUTE_TARGET_RHO_DIAGONALIZE_AND_CONSTRUCT_NEW_PACKET_TENSOR_MULTIPLE_RETURNED_DENSITY_EIGENCOMPONENTS_FOR_QUERY_B_THEN_INCOHERENTLY_SUM_WITH_DECLARED_DISCARDED_TRACE_BOUND",
        "query_A_final_target_principal_weight": sf(RETURN_PRINCIPAL_WEIGHT),
        "query_A_nonprincipal_trace": sf(RETURN_NONPRINCIPAL_TRACE),
        "negative_density_eigenvalue_mass_clipped_as_numerical_roundoff": sf(RETURN_NEGATIVE_NUMERICAL_MASS),
        "returned_principal_ground_infidelity": sf(1.0 - abs(np.vdot(GROUND, RETURN_PRINCIPAL)) ** 2),
        "query_B_returned_controlled_mixed_density": METRICS_RETURNED_B,
        "clean_vs_returned_current_delay_difference": sf(REUSE_DELAY_DIFFERENCE),
        "clean_vs_returned_final_excitation_difference": sf(abs(RUN_B_CLEAN.final_excitation - RUN_B_RETURNED_MIXED.final_excitation)),
        "clean_vs_returned_final_target_density_trace_distance": sf(REUSE_FINAL_DENSITY_TRACE_DISTANCE),
        "iq_parity": "NOT_CLAIMED_WITHOUT_EXPLICIT_MIXED_TARGET_REFERENCE_ARM_CHANNEL",
        "classification": "FUNCTIONAL_NUMERICAL_CARRYOVER_VIA_COMPUTED_STATE_REMATERIALIZATION_NOT_RESIDENT_PHYSICAL_TARGET_REUSE_OR_RESTORATION",
        "returned_state_rematerialization_used": True,
        "full_returned_density_matrix_propagated": False,
        "controlled_spectral_truncation_used": True,
        "discarded_density_trace_limit": sf(MIXED_DENSITY_DISCARDED_TRACE_LIMIT),
        "baseline_or_exact_ground_reload_used": False,
        "same_backing": False,
    },
    "controls": {
        "free": {
            "implementation": "EXACT_DST_I_OPEN_LEAD_REFERENCE",
            "target_interaction": False,
            "source_replay_after_initialization": False,
        },
        "exact_ground_query_B": METRICS_EXACT_B,
        "prepared_vs_exact_query_B": {
            "same_time_phase_difference": sf(PREP_PHASE_DIFFERENCE),
            "current_delay_difference": sf(PREP_DELAY_DIFFERENCE),
            "maximum_excitation_difference": sf(abs(RUN_B_CLEAN.max_excitation - RUN_B_EXACT_GROUND.max_excitation)),
        },
        "static_boundary_mean": {
            "ground_Z0_expectation": sf(GROUND_BOUNDARY_MEAN),
            "endpoint_potential": sf(STATIC_POTENTIAL),
            "query_A": scalar_metrics(RUN_STATIC_A),
            "query_B": scalar_metrics(RUN_STATIC_B),
            "interpretation": "EXACT_ONE_CHANNEL_STATIC_DESCRIPTOR_CONTROL_WITH_NO_TARGET_EXCITATION",
        },
        "above_threshold": {
            "initial_target": "EXACT_GROUND",
            "query": METRICS_ABOVE,
            "interpretation": "OPEN_INELASTIC_CHANNEL_CONTROL_EXPECTED_NOT_TO_RESTORE_TARGET",
        },
        "not_materialized_here_but_required_for_package_promotion": [
            "DISCONNECTED_BOUNDARY_CORE",
            "GAUSSIAN_OR_HARMONIC_MATCHED_CONTROL",
            "BETHE_OR_FACTORIZED_CONTROL",
            "MPS_CORRECTION_VECTOR_AND_REAL_TIME_BOND_DIMENSION_SWEEPS",
            "SPARSE_KRYLOV_CHEBYSHEV_LANCZOS_EQUAL_ACCESS_FORWARD_ONLY_SHADOWS",
            "LOSS_DEPHASING_AND_STATIC_COEFFICIENT_ERROR_SWEEPS",
        ],
    },
    "precision_and_shots": {
        "formula": "N_per_quadrature=ceil(4 ln(4/alpha)/(visibility^2 delta_phase^2))",
        "alpha": 0.01,
        "detector_efficiency_assumed_for_counts": 1.0,
        "query_A": {
            "visibility": sf(VISIBILITY_A),
            "shots_per_quadrature_delta_phase_1e-2": shots(VISIBILITY_A, 1.0e-2),
            "shots_per_quadrature_delta_phase_1e-3": shots(VISIBILITY_A, 1.0e-3),
        },
        "query_B": {
            "visibility": sf(VISIBILITY_B),
            "shots_per_quadrature_delta_phase_1e-2": shots(VISIBILITY_B, 1.0e-2),
            "shots_per_quadrature_delta_phase_1e-3": shots(VISIBILITY_B, 1.0e-3),
        },
        "efficiency_scaling": "DIVIDE_COUNTS_BY_DETECTOR_EFFICIENCY",
        "coherent_phase_estimation_credited": False,
        "noise_or_loss_robustness_claim": False,
    },
    "resource_accounting": {
        "complex_state_amplitudes": LEAD_LENGTH * TARGET_DIM,
        "complex128_state_bytes": LEAD_LENGTH * TARGET_DIM * 16,
        "returned_density_retained_rank": RUN_B_RETURNED_MIXED.propagated_rank,
        "returned_density_retained_complex128_bytes": LEAD_LENGTH * TARGET_DIM * RUN_B_RETURNED_MIXED.propagated_rank * 16,
        "full_rank_16_mixed_complex128_bytes": LEAD_LENGTH * TARGET_DIM * TARGET_DIM * 16,
        "separate_reference_algorithmic_work": "O((T/dt)*L*2^n*log L) PLUS DENSE_2^n_TARGET_ACTIONS",
        "target_boundary_krylov_rank_ceiling": TARGET_DIM,
        "fixed_fixture_tensor_network_crossover_claim": "NONE",
        "asymptotic_resource_advantage_claim": "NONE",
        "strongest_caveat": "ONE_DIMENSIONAL_GAPPED_FINITE_BANDWIDTH_LOCAL_SCATTERING_CAN_REMAIN_COMPACT_FOR_MPS_CORRECTION_VECTOR_AND_SPARSE_FORWARD_SHADOWS",
        "near_threshold_cost_caveat": "DELAY_ENHANCEMENT_ALSO_CHARGES_PACKET_DURATION_DWELL_TIME_LOSS_SENSITIVITY_AND_RESOLVENT_CONDITIONING",
        "growing_exact_rank_is_approximation_lower_bound": False,
        "forward_only_shadow_may_omit_restoration_tail": True,
        "promotion_requires_scaling_family": True,
        "qualifier_runtime_caveat": "MULTICOMPONENT_DENSITY_HANDOFF_COST_IS_MATERIAL_AND_MUST_BE_COUNTED_SEPARATELY_FROM_THE_PURE_QUERY",
    },
    "assertions": {
        "site_zero_projector_support_exact": True,
        "subgap_packet_tails_below_limit": True,
        "target_connected_and_non_gaussian": True,
        "preparation_ground_infidelity_below_1e-7": True,
        "subgap_transient_excitation_observed": True,
        "subgap_final_target_excitation_below_declared_limits": True,
        "matched_free_target_return_gate_pass": False,
        "matched_free_target_return_obstruction_detected": True,
        "above_threshold_nonreturn_control_observed": True,
        "stationary_subgap_reflections_unit_modulus": True,
        "split_step_norm_error_below_1e-9": True,
        "step_halving_and_finite_lead_controls_pass": True,
        "clean_vs_returned_query_B_parity_below_1e-4_with_bound": True,
        "package_qualification": "FAIL_MATCHED_FREE_TARGET_RETURN_GATE",
        "status": STATUS,
    },
}


print(json.dumps(OUTPUT, sort_keys=True, separators=(",", ":")))
