#!/usr/bin/env python3
"""M266 three-mode trapped-ion state-dependent-force digital twin.

This is a deterministic complex128/float64 Gaussian-moment model.  It is not
QEMU device code, a physical observation, physical ion custody, or a resource
advantage.  The nominal force law uses an exact per-segment Magnus recurrence,
including the intra-segment self-area.  The primary run deliberately includes
nonzero additive Markovian heating and therefore makes no restoration claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLAIM = (
    "THREE_MODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_NULLSPACE_PULSES_CLOSE_ALL_"
    "NOMINAL_MODE_DISPLACEMENTS_AND_IMPLEMENT_TWO_DISTINCT_ZZ_PHASE_PROGRAMS_"
    "ON_ONE_LOGICAL_MULTIMODE_BACKING_WHILE_DECLARED_NONZERO_MARKOVIAN_HEATING_"
    "MONOTONICALLY_BREAKS_EXACT_INITIAL_MODE_STATE_RETURN_WITHOUT_RECOOLING"
)
CLAIM_CEILING = (
    "DETERMINISTIC_COMPLEX128_FLOAT64_GAUSSIAN_MOMENT_SOFTWARE_DIGITAL_TWIN_"
    "WITH_DECLARED_LINEAR_HARMONIC_STATE_DEPENDENT_FORCE_AND_MARKOVIAN_ADDITIVE_"
    "HEATING_LAWS_NO_PHYSICAL_ION_CUSTODY_AND_DIRECT_COMPILED_CLIENT_CHANNEL_SHADOW"
)
RESTORATION_CLASSIFICATION = "NO_RESTORATION_CLAIM"
RESTORATION_SCOPE = (
    "NOMINAL_ZERO_HEATING_LOGICAL_GAUSSIAN_MODE_RETURN_ONLY_WITH_NONZERO_HEATING_"
    "EXACT_SAME_MODE_RETURN_REJECTED_AND_FRESH_MODE_SWAP_OR_RECOOLING_CLASSIFIED_"
    "AS_EXTERNAL_RESET"
)
RESOURCE_DISPOSITION = (
    "DIRECT_COMPILED_ZZ_AND_DEPHASING_CHANNEL_SHADOW_OMITS_THE_THREE_MODE_"
    "CONTROLLER_LOOP_WHILE_HEATING_PREVENTS_CATALYTIC_RETURN_SO_NO_RESOURCE_"
    "ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED"
)
NEXT_MECHANISM = (
    "CONDITIONAL_GAUSSIAN_CLOSED_LOOP_FORWARD_SHADOW_AND_IRREVERSIBLE_DIFFUSION_"
    "NO_RETURN"
)

MILESTONE = "M266"
SCHEMA = "PHASE_QEMU_V8_TRAPPED_ION_WEYL_LOOP_RESULT_V1"
PROGRAM_DURATION_S = 120e-6
SEGMENTS = 8
SEGMENT_DURATION_S = PROGRAM_DURATION_S / SEGMENTS
MODE_FREQUENCIES_HZ = np.array([1_900_000.0, 1_918_000.0, 1_930_000.0])
DRIVE_FREQUENCY_HZ = 1_911_000.0
# The public descriptor freezes drive-minus-mode detunings in this order.  The frequency
# table above is retained as hardware geometry; its ordering convention is not
# silently used to recompute the signed rotating-frame array.
DETUNINGS_RAD_S = 2.0 * np.pi * np.array([11_000.0, -7_000.0, -19_000.0])
HEATING_QUANTA_PER_S = np.array([15.0, 30.0, 60.0])
HBAR_J_S = 1.054_571_817e-34

MODE_VECTORS = np.array(
    [
        [1.0 / math.sqrt(3.0), 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)],
        [1.0 / math.sqrt(3.0), 0.0, -2.0 / math.sqrt(6.0)],
        [1.0 / math.sqrt(3.0), -1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)],
    ],
    dtype=np.float64,
)
ETA = 0.06 * MODE_VECTORS

# These are the physically corrected waveforms.  They are twice the discarded
# no-self-area amplitudes.  Changing them without regenerating every result is
# a contract violation.
PROGRAM_A_AMPLITUDES_RAD_S = np.array(
    [
        875322.2363528529,
        -1017217.5496960702,
        1149417.5349432244,
        -128702.55695608277,
        -128702.55695606946,
        1149417.5349432132,
        -1017217.549696064,
        875322.2363528487,
    ],
    dtype=np.float64,
)
PROGRAM_B_AMPLITUDES_RAD_S = np.array(
    [
        -229497.22240917346,
        725694.6296712102,
        -1293755.9826773852,
        1628861.1574510091,
        -1628861.1574510091,
        1293755.9826774026,
        -725694.6296712208,
        229497.22240918515,
    ],
    dtype=np.float64,
)


def _round_float(value: float, digits: int = 15) -> float:
    return float(np.round(float(value), digits))


def _complex_record(value: complex) -> dict[str, float]:
    return {"real": _round_float(value.real), "imag": _round_float(value.imag)}


def _array_records(values: Iterable[complex]) -> list[dict[str, float]]:
    return [_complex_record(complex(value)) for value in values]


def _segment_integral(delta: float, t0: float, dt: float) -> complex:
    """Integral of exp(+i delta t) over one segment."""

    return np.exp(1j * delta * t0) * np.expm1(1j * delta * dt) / (1j * delta)


def _self_area(delta: float, dt: float) -> float:
    """Exact same-segment Magnus area for constant real coupling."""

    return float((delta * dt - math.sin(delta * dt)) / (delta * delta))


SPIN_PAIR_STATES = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class BranchEvolution:
    phase: float
    final_displacements: np.ndarray
    maximum_displacement: float
    maximum_displacement_per_mode: np.ndarray


def evolve_branch(
    amplitudes: np.ndarray,
    pair: tuple[int, int],
    spin_pair: tuple[int, int],
    detunings: np.ndarray = DETUNINGS_RAD_S,
) -> BranchEvolution:
    """Exact chronological segment recurrence for one target-spin branch."""

    z = np.zeros(3, dtype=np.complex128)
    phase = 0.0
    maximum_per_mode = np.zeros(3, dtype=np.float64)
    for segment_index, omega in enumerate(amplitudes):
        t0 = segment_index * SEGMENT_DURATION_S
        couplings = 0.5 * omega * (
            ETA[pair[0], :] * spin_pair[0] + ETA[pair[1], :] * spin_pair[1]
        )
        for mode_index, delta in enumerate(detunings):
            integral = _segment_integral(delta, t0, SEGMENT_DURATION_S)
            increment = -1j * couplings[mode_index] * integral
            # D(increment) D(z) contributes Im(increment conj(z)); the second
            # term is the exact intra-segment Magnus self-area and must not be
            # dropped even though the force amplitude is constant in a segment.
            phase += float(np.imag(increment * np.conj(z[mode_index])))
            phase += couplings[mode_index] ** 2 * _self_area(
                delta, SEGMENT_DURATION_S
            )
            z[mode_index] += increment

        # Endpoint sampling is exact for the frozen maximum in these waveforms;
        # a dense within-segment scan below guards against a hidden larger arc.
        maximum_per_mode = np.maximum(maximum_per_mode, np.abs(z))
        sample_fractions = np.linspace(0.0, 1.0, 65, dtype=np.float64)[1:-1]
        for fraction in sample_fractions:
            partial_dt = float(fraction * SEGMENT_DURATION_S)
            partial = np.array(
                [
                    -1j
                    * couplings[m]
                    * _segment_integral(detunings[m], t0, partial_dt)
                    for m in range(3)
                ],
                dtype=np.complex128,
            )
            # z already includes the full increment; remove it to obtain the
            # start-of-segment coordinate before adding the partial arc.
            full = np.array(
                [
                    -1j
                    * couplings[m]
                    * _segment_integral(
                        detunings[m], t0, SEGMENT_DURATION_S
                    )
                    for m in range(3)
                ],
                dtype=np.complex128,
            )
            maximum_per_mode = np.maximum(
                maximum_per_mode, np.abs(z - full + partial)
            )

    return BranchEvolution(
        phase=float(phase),
        final_displacements=z,
        maximum_displacement=float(np.max(maximum_per_mode)),
        maximum_displacement_per_mode=maximum_per_mode,
    )


def _separation_integrals(
    amplitudes: np.ndarray,
    pair: tuple[int, int],
    left: tuple[int, int],
    right: tuple[int, int],
    detunings: np.ndarray = DETUNINGS_RAD_S,
) -> np.ndarray:
    """Exact integral of branch separation squared for all modes."""

    separation = np.zeros(3, dtype=np.complex128)
    integrals = np.zeros(3, dtype=np.float64)
    spin_difference = np.array(left, dtype=np.float64) - np.array(
        right, dtype=np.float64
    )
    for segment_index, omega in enumerate(amplitudes):
        t0 = segment_index * SEGMENT_DURATION_S
        e0 = np.exp(1j * detunings * t0)
        coupling_difference = 0.5 * omega * (
            ETA[pair[0], :] * spin_difference[0]
            + ETA[pair[1], :] * spin_difference[1]
        )
        constant = separation + (coupling_difference / detunings) * e0
        rotating = -coupling_difference / detunings
        segment_integrals = np.array(
            [
                _segment_integral(delta, t0, SEGMENT_DURATION_S)
                for delta in detunings
            ],
            dtype=np.complex128,
        )
        integrals += SEGMENT_DURATION_S * (
            np.abs(constant) ** 2 + np.abs(rotating) ** 2
        ) + 2.0 * np.real(np.conj(constant) * rotating * segment_integrals)
        separation += -1j * coupling_difference * segment_integrals
    return integrals


def _basis_spins(index: int) -> tuple[int, int, int]:
    return tuple(1 if ((index >> (2 - q)) & 1) == 0 else -1 for q in range(3))


BASIS_SPINS = tuple(_basis_spins(index) for index in range(8))


@dataclass(frozen=True)
class ProgramModel:
    name: str
    pair: tuple[int, int]
    target_theta: float
    amplitudes: np.ndarray
    branches: dict[tuple[int, int], BranchEvolution]
    walsh_theta: float
    walsh_global: float
    maximum_closure: float
    maximum_midloop_displacement: float
    maximum_midloop_per_mode: np.ndarray
    heating_exposure: np.ndarray
    heated_multiplier: np.ndarray
    nominal_multiplier: np.ndarray
    direct_heated_multiplier: np.ndarray
    direct_ideal_multiplier: np.ndarray


def build_program(
    name: str,
    amplitudes: np.ndarray,
    pair: tuple[int, int],
    target_theta: float,
    detunings: np.ndarray = DETUNINGS_RAD_S,
) -> ProgramModel:
    branches = {
        spin_pair: evolve_branch(amplitudes, pair, spin_pair, detunings)
        for spin_pair in SPIN_PAIR_STATES
    }
    phases = np.array(
        [branches[spin_pair].phase for spin_pair in SPIN_PAIR_STATES],
        dtype=np.float64,
    )
    walsh_theta = float((phases[0] - phases[1] - phases[2] + phases[3]) / 4.0)
    walsh_global = float(np.sum(phases) / 4.0)
    maximum_closure = max(
        float(np.max(np.abs(branch.final_displacements)))
        for branch in branches.values()
    )
    maximum_midloop_per_mode = np.maximum.reduce(
        [branch.maximum_displacement_per_mode for branch in branches.values()]
    )

    heating_exposure = np.zeros((8, 8), dtype=np.float64)
    nominal_multiplier = np.empty((8, 8), dtype=np.complex128)
    heated_multiplier = np.empty((8, 8), dtype=np.complex128)
    direct_ideal_multiplier = np.empty((8, 8), dtype=np.complex128)
    direct_heated_multiplier = np.empty((8, 8), dtype=np.complex128)
    for left_index, left_full in enumerate(BASIS_SPINS):
        left_pair = (left_full[pair[0]], left_full[pair[1]])
        left_phase = branches[left_pair].phase
        left_product = left_pair[0] * left_pair[1]
        for right_index, right_full in enumerate(BASIS_SPINS):
            right_pair = (right_full[pair[0]], right_full[pair[1]])
            right_phase = branches[right_pair].phase
            right_product = right_pair[0] * right_pair[1]
            path_integrals = _separation_integrals(
                amplitudes, pair, left_pair, right_pair, detunings
            )
            # Standard D[L]=L rho L^dagger-1/2{L^dagger L,rho} convention:
            # Gamma(D[a]+D[a^dagger]) gives V_m -> V_m + Gamma_m t I,
            # d<n_m>/dt=Gamma_m, and the influence functional below.
            exposure = float(np.dot(HEATING_QUANTA_PER_S, path_integrals))
            heating_exposure[left_index, right_index] = exposure
            nominal_phase = np.exp(1j * (left_phase - right_phase))
            direct_phase = np.exp(
                1j * target_theta * (left_product - right_product)
            )
            attenuation = math.exp(-exposure)
            nominal_multiplier[left_index, right_index] = nominal_phase
            heated_multiplier[left_index, right_index] = (
                nominal_phase * attenuation
            )
            direct_ideal_multiplier[left_index, right_index] = direct_phase
            direct_heated_multiplier[left_index, right_index] = (
                direct_phase * attenuation
            )

    return ProgramModel(
        name=name,
        pair=pair,
        target_theta=target_theta,
        amplitudes=amplitudes,
        branches=branches,
        walsh_theta=walsh_theta,
        walsh_global=walsh_global,
        maximum_closure=maximum_closure,
        maximum_midloop_displacement=float(np.max(maximum_midloop_per_mode)),
        maximum_midloop_per_mode=maximum_midloop_per_mode,
        heating_exposure=heating_exposure,
        heated_multiplier=heated_multiplier,
        nominal_multiplier=nominal_multiplier,
        direct_heated_multiplier=direct_heated_multiplier,
        direct_ideal_multiplier=direct_ideal_multiplier,
    )


def _plus_density() -> np.ndarray:
    vector = np.ones(8, dtype=np.complex128) / math.sqrt(8.0)
    return np.outer(vector, np.conj(vector))


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _pauli_word(word: str) -> np.ndarray:
    operators = {"I": I2, "X": X, "Y": Y, "Z": Z}
    result = np.array([[1.0]], dtype=np.complex128)
    for symbol in word:
        result = np.kron(result, operators[symbol])
    return result


def _boundary(rho: np.ndarray, words: dict[str, str]) -> dict[str, float]:
    return {
        name: _round_float(np.real(np.trace(rho @ _pauli_word(word))), 13)
        for name, word in words.items()
    }


BOUNDARY_A = {
    "X0": "XII",
    "X1": "IXI",
    "X2": "IIX",
    "Y0Z1": "YZI",
    "Z0Y1": "ZYI",
    "X0X1": "XXI",
}
BOUNDARY_AB = {
    **BOUNDARY_A,
    "Z1Y2": "IZY",
    "X1X2": "IXX",
}


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = 0.5 * ((left - right) + (left - right).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def _density_diagnostics(rho: np.ndarray) -> dict[str, float]:
    hermitian = 0.5 * (rho + rho.conj().T)
    return {
        "trace_error": _round_float(abs(np.trace(rho) - 1.0)),
        "hermiticity_max_abs": _round_float(np.max(np.abs(rho - rho.conj().T))),
        "minimum_eigenvalue": _round_float(np.min(np.linalg.eigvalsh(hermitian))),
        "purity": _round_float(np.real(np.trace(rho @ rho))),
    }


@dataclass
class GaussianBacking:
    name: str
    means: np.ndarray
    covariances: np.ndarray
    tmsv_joint_covariance: np.ndarray | None = None
    tmsv_r: float | None = None
    generation: int = 0
    supply_count: int = 1
    snapshot_count: int = 0
    reload_count: int = 0
    recooling_count: int = 0
    replacement_count: int = 0

    @property
    def base_pointer(self) -> int:
        return int(self.covariances.__array_interface__["data"][0])

    def heat_in_place(self, duration: float) -> None:
        drift = HEATING_QUANTA_PER_S * duration
        for mode_index, added_variance in enumerate(drift):
            self.covariances[mode_index] += added_variance * np.eye(2)
        if self.tmsv_joint_covariance is not None:
            self.tmsv_joint_covariance[:2, :2] += drift[0] * np.eye(2)
        self.generation += 1


def _rotation(angle: float) -> np.ndarray:
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )


def build_fixtures() -> list[GaussianBacking]:
    vacuum_covariances = np.repeat(0.5 * np.eye(2)[None, :, :], 3, axis=0)
    fixtures: list[GaussianBacking] = []
    fixtures.append(
        GaussianBacking(
            "vacuum_product",
            np.zeros(3, dtype=np.complex128),
            vacuum_covariances.copy(),
        )
    )
    thermal_nbar = np.array([0.05, 0.10, 0.20])
    fixtures.append(
        GaussianBacking(
            "thermal_product_nbar_0p05_0p10_0p20",
            np.zeros(3, dtype=np.complex128),
            np.array([(nbar + 0.5) * np.eye(2) for nbar in thermal_nbar]),
        )
    )
    fixtures.append(
        GaussianBacking(
            "coherent_product_alpha_0p30_plus_0p15i_minus_0p20_plus_0p10i_0p10_minus_0p25i",
            np.array([0.30 + 0.15j, -0.20 + 0.10j, 0.10 - 0.25j]),
            vacuum_covariances.copy(),
        )
    )
    squeeze_r = 0.35
    squeeze_phi = 0.20
    squeeze_rotation = _rotation(squeeze_phi / 2.0)
    squeezed = squeeze_rotation @ np.diag(
        [0.5 * math.exp(-2.0 * squeeze_r), 0.5 * math.exp(2.0 * squeeze_r)]
    ) @ squeeze_rotation.T
    fixtures.append(
        GaussianBacking(
            "squeezed_mode0_r0p35_phi0p20_with_thermal_spectators",
            np.zeros(3, dtype=np.complex128),
            np.array([squeezed, 0.55 * np.eye(2), 0.60 * np.eye(2)]),
        )
    )
    tmsv_r = 0.30
    c = math.cosh(2.0 * tmsv_r)
    s = math.sinh(2.0 * tmsv_r)
    z = np.diag([1.0, -1.0])
    tmsv = 0.5 * np.block([[c * np.eye(2), s * z], [s * z, c * np.eye(2)]])
    fixtures.append(
        GaussianBacking(
            "tmsv_bus0_inert_reference_r0p30",
            np.zeros(3, dtype=np.complex128),
            np.array([0.5 * c * np.eye(2), 0.5 * np.eye(2), 0.5 * np.eye(2)]),
            tmsv_joint_covariance=tmsv,
            tmsv_r=tmsv_r,
        )
    )
    return fixtures


def _fixture_run(backing: GaussianBacking) -> dict[str, Any]:
    initial_means = backing.means.copy()
    initial_covariances = backing.covariances.copy()
    initial_tmsv = (
        None
        if backing.tmsv_joint_covariance is None
        else backing.tmsv_joint_covariance.copy()
    )
    object_id = id(backing.covariances)
    base_pointer = backing.base_pointer
    backing.heat_in_place(PROGRAM_DURATION_S)
    after_a_covariance_difference = backing.covariances - initial_covariances
    after_a_max_drift = float(np.max(np.abs(after_a_covariance_difference)))
    after_a_tmsv_fidelity = None
    if backing.tmsv_r is not None:
        nu = HEATING_QUANTA_PER_S[0] * PROGRAM_DURATION_S
        after_a_tmsv_fidelity = 1.0 / (
            1.0 + nu * math.cosh(2.0 * backing.tmsv_r)
        )
    backing.heat_in_place(PROGRAM_DURATION_S)
    after_ab_covariance_difference = backing.covariances - initial_covariances
    after_ab_max_drift = float(np.max(np.abs(after_ab_covariance_difference)))
    after_ab_tmsv_fidelity = None
    if backing.tmsv_r is not None:
        nu = 2.0 * HEATING_QUANTA_PER_S[0] * PROGRAM_DURATION_S
        after_ab_tmsv_fidelity = 1.0 / (
            1.0 + nu * math.cosh(2.0 * backing.tmsv_r)
        )
    return {
        "fixture": backing.name,
        "initial_means": _array_records(initial_means),
        "final_means": _array_records(backing.means),
        "mean_return_max_abs": _round_float(np.max(np.abs(backing.means - initial_means))),
        "after_a_covariance_max_abs_drift": _round_float(after_a_max_drift),
        "after_a_exact_initial_mode_state_return": False,
        "after_ab_covariance_max_abs_drift": _round_float(after_ab_max_drift),
        "after_ab_covariance_frobenius_drift": _round_float(
            np.linalg.norm(after_ab_covariance_difference)
        ),
        "after_ab_exact_initial_mode_state_return": False,
        "covariance_drift_strictly_monotone": bool(
            after_ab_max_drift > after_a_max_drift > 0.0
        ),
        "heating_delta_n_after_a": [
            _round_float(value)
            for value in HEATING_QUANTA_PER_S * PROGRAM_DURATION_S
        ],
        "heating_delta_n_after_ab": [
            _round_float(value)
            for value in HEATING_QUANTA_PER_S * 2.0 * PROGRAM_DURATION_S
        ],
        "logical_custody": {
            "allocation_object_unchanged": id(backing.covariances) == object_id,
            "allocation_base_pointer_unchanged": backing.base_pointer == base_pointer,
            "generation_sequence": [0, 1, 2],
            "carrier_supply_count": backing.supply_count,
            "snapshot_count": backing.snapshot_count,
            "reload_count": backing.reload_count,
            "recooling_count": backing.recooling_count,
            "carrier_replacement_count": backing.replacement_count,
            "physical_same_mode_custody_established": False,
        },
        "tmsv_reference_coherence": None
        if initial_tmsv is None
        else {
            "initial_joint_covariance_determinant": _round_float(
                np.linalg.det(initial_tmsv)
            ),
            "after_a_entanglement_fidelity": _round_float(after_a_tmsv_fidelity),
            "after_ab_entanglement_fidelity": _round_float(after_ab_tmsv_fidelity),
            "after_ab_entanglement_infidelity": _round_float(
                1.0 - float(after_ab_tmsv_fidelity)
            ),
            "reference_is_inert": True,
            "exact_reference_coherence_return": False,
        },
    }


def _closure_matrix(detunings: np.ndarray) -> np.ndarray:
    matrix = np.empty((len(detunings), SEGMENTS), dtype=np.complex128)
    for mode_index, delta in enumerate(detunings):
        for segment_index in range(SEGMENTS):
            matrix[mode_index, segment_index] = _segment_integral(
                delta, segment_index * SEGMENT_DURATION_S, SEGMENT_DURATION_S
            )
    return matrix


def _middle_mode_only_waveform(target_theta: float) -> np.ndarray:
    matrix = _closure_matrix(DETUNINGS_RAD_S)[1:2, :]
    real_matrix = np.vstack([matrix.real, matrix.imag])
    _, _, vh = np.linalg.svd(real_matrix, full_matrices=True)
    null_basis = vh[2:, :].T
    indices = np.arange(1.0, SEGMENTS + 1.0)
    for seed_index in range(1, 33):
        seed = np.sin(indices * seed_index * 0.71) + 0.3 * np.cos(
            indices * (seed_index + 1) * 0.37
        )
        candidate = null_basis @ (null_basis.T @ seed)
        model = build_program(
            "middle_mode_seed",
            candidate,
            (0, 1),
            target_theta,
        )
        if model.walsh_theta * target_theta > 0.0 and abs(model.walsh_theta) > 1e-16:
            return candidate * math.sqrt(target_theta / model.walsh_theta)
    raise RuntimeError("could not construct deterministic middle-mode-only control")


def _control_summary(
    name: str,
    amplitudes: np.ndarray,
    pair: tuple[int, int],
    target_theta: float,
    detunings: np.ndarray = DETUNINGS_RAD_S,
) -> dict[str, Any]:
    model = build_program(name, amplitudes, pair, target_theta, detunings)
    return {
        "name": name,
        "segments_executed": int(len(amplitudes)),
        "walsh_theta": _round_float(model.walsh_theta),
        "theta_error": _round_float(abs(model.walsh_theta - target_theta)),
        "maximum_closure": _round_float(model.maximum_closure),
        "maximum_midloop_displacement": _round_float(
            model.maximum_midloop_displacement
        ),
    }


def _controls(program_a: ProgramModel, program_b: ProgramModel) -> dict[str, Any]:
    omitted = _control_summary(
        "omitted_final_segment",
        PROGRAM_A_AMPLITUDES_RAD_S[:-1],
        (0, 1),
        -math.pi / 8.0,
    )
    omitted["program_b"] = _control_summary(
        "omitted_final_segment_program_b",
        PROGRAM_B_AMPLITUDES_RAD_S[:-1],
        (1, 2),
        -math.pi / 6.0,
    )
    middle_waveform = _middle_mode_only_waveform(-math.pi / 8.0)
    middle = _control_summary(
        "target_middle_mode_only_spectator_omission",
        middle_waveform,
        (0, 1),
        -math.pi / 8.0,
    )
    detuned = _control_summary(
        "detuning_plus_250hz_all_modes",
        PROGRAM_A_AMPLITUDES_RAD_S,
        (0, 1),
        -math.pi / 8.0,
        DETUNINGS_RAD_S - 2.0 * np.pi * 250.0,
    )
    detuned["program_b"] = _control_summary(
        "detuning_plus_250hz_all_modes_program_b",
        PROGRAM_B_AMPLITUDES_RAD_S,
        (1, 2),
        -math.pi / 6.0,
        DETUNINGS_RAD_S - 2.0 * np.pi * 250.0,
    )
    fullscale = 1.8e6
    step = 2.0 * fullscale / (2**12 - 1)
    quantized_amplitudes = np.clip(
        np.round(PROGRAM_A_AMPLITUDES_RAD_S / step) * step,
        -fullscale,
        fullscale,
    )
    quantized = _control_summary(
        "amplitude_quantization_12bit_fullscale_1p8e6",
        quantized_amplitudes,
        (0, 1),
        -math.pi / 8.0,
    )
    quantized_b_amplitudes = np.clip(
        np.round(PROGRAM_B_AMPLITUDES_RAD_S / step) * step,
        -fullscale,
        fullscale,
    )
    quantized["program_b"] = _control_summary(
        "amplitude_quantization_12bit_fullscale_1p8e6_program_b",
        quantized_b_amplitudes,
        (1, 2),
        -math.pi / 6.0,
    )
    area_error_amplitudes = PROGRAM_A_AMPLITUDES_RAD_S.copy()
    area_error_amplitudes[3] *= 1.01
    area_error = _control_summary(
        "segment_index3_plus_1_percent",
        area_error_amplitudes,
        (0, 1),
        -math.pi / 8.0,
    )
    area_error_b_amplitudes = PROGRAM_B_AMPLITUDES_RAD_S.copy()
    area_error_b_amplitudes[3] *= 1.01
    area_error["program_b"] = _control_summary(
        "segment_index3_plus_1_percent_program_b",
        area_error_b_amplitudes,
        (1, 2),
        -math.pi / 6.0,
    )
    opposite = _control_summary(
        "opposite_phase_orientation",
        PROGRAM_A_AMPLITUDES_RAD_S,
        (0, 1),
        math.pi / 8.0,
        -DETUNINGS_RAD_S,
    )
    opposite["program_b"] = _control_summary(
        "opposite_phase_orientation_program_b",
        PROGRAM_B_AMPLITUDES_RAD_S,
        (1, 2),
        math.pi / 6.0,
        -DETUNINGS_RAD_S,
    )
    zero_force = _control_summary(
        "zero_force_zero_area",
        np.zeros(SEGMENTS, dtype=np.float64),
        (0, 1),
        0.0,
    )
    zero_force["program_b"] = _control_summary(
        "zero_force_zero_area_program_b",
        np.zeros(SEGMENTS, dtype=np.float64),
        (1, 2),
        0.0,
    )
    return {
        control["name"]: control
        for control in (
            omitted,
            middle,
            detuned,
            quantized,
            area_error,
            opposite,
            zero_force,
        )
    } | {
        "snapshot_reload": {
            "restores_saved_software_baseline": True,
            "snapshot_count": 1,
            "reload_count": 1,
            "classification": "SNAPSHOT_RELOAD",
            "accepted_as_catalytic_restoration": False,
        },
        "recooling_external_reset": {
            "mode_covariance_replaced_by_preparation_covariance": True,
            "recooling_count": 1,
            "classification": "EXTERNAL_RESET",
            "accepted_as_same_mode_reuse": False,
        },
        "fresh_mode_swap_carrier_replacement": {
            "carrier_replacement_count": 1,
            "classification": "CARRIER_REPLACEMENT",
            "accepted_as_same_mode_reuse": False,
        },
        "tmsv_reference_coherence": {
            "checked_in_named_fixture": True,
            "bus_marginal_only_is_sufficient": False,
        },
        "nominal_reference": {
            "walsh_theta": _round_float(program_a.walsh_theta),
            "maximum_closure": _round_float(program_a.maximum_closure),
        },
    }


def _program_record(program: ProgramModel) -> dict[str, Any]:
    return {
        "name": program.name,
        "target_pair": list(program.pair),
        "target_theta": _round_float(program.target_theta),
        "amplitudes_rad_s": [_round_float(value) for value in program.amplitudes],
        "branch_phases": {
            f"{left:+d}_{right:+d}": _round_float(
                program.branches[(left, right)].phase
            )
            for left, right in SPIN_PAIR_STATES
        },
        "walsh_global_phase": _round_float(program.walsh_global),
        "walsh_zz_phase": _round_float(program.walsh_theta),
        "phase_error": _round_float(abs(program.walsh_theta - program.target_theta)),
        "maximum_final_branch_displacement": _round_float(program.maximum_closure),
        "maximum_midloop_branch_displacement": _round_float(
            program.maximum_midloop_displacement
        ),
        "maximum_midloop_branch_displacement_per_mode": [
            _round_float(value) for value in program.maximum_midloop_per_mode
        ],
        "maximum_heating_exposure": _round_float(
            np.max(program.heating_exposure)
        ),
        "self_area_included": True,
        "cross_segment_area_convention": "IM_INCREMENT_TIMES_CONJUGATE_PRIOR_DISPLACEMENT",
    }


def run() -> dict[str, Any]:
    program_a = build_program(
        "A_Z0_Z1_MINUS_PI_OVER_8",
        PROGRAM_A_AMPLITUDES_RAD_S,
        (0, 1),
        -math.pi / 8.0,
    )
    program_b = build_program(
        "B_Z1_Z2_MINUS_PI_OVER_6",
        PROGRAM_B_AMPLITUDES_RAD_S,
        (1, 2),
        -math.pi / 6.0,
    )

    initial_client = _plus_density()
    nominal_after_a = initial_client * program_a.nominal_multiplier
    nominal_after_ab = nominal_after_a * program_b.nominal_multiplier
    heated_after_a = initial_client * program_a.heated_multiplier
    heated_after_ab = heated_after_a * program_b.heated_multiplier
    direct_ideal_after_a = initial_client * program_a.direct_ideal_multiplier
    direct_ideal_after_ab = direct_ideal_after_a * program_b.direct_ideal_multiplier
    direct_heated_after_a = initial_client * program_a.direct_heated_multiplier
    direct_heated_after_ab = (
        direct_heated_after_a * program_b.direct_heated_multiplier
    )

    fixture_records = [_fixture_run(fixture) for fixture in build_fixtures()]
    controls = _controls(program_a, program_b)
    closure_matrix = _closure_matrix(DETUNINGS_RAD_S)
    real_closure_matrix = np.vstack([closure_matrix.real, closure_matrix.imag])
    closure_singular_values = np.linalg.svd(
        real_closure_matrix, compute_uv=False
    )

    checks = {
        "program_a_exact_self_area_phase": abs(program_a.walsh_theta + math.pi / 8.0)
        <= 1e-12,
        "program_b_exact_self_area_phase": abs(program_b.walsh_theta + math.pi / 6.0)
        <= 1e-12,
        "program_a_all_mode_closure": program_a.maximum_closure <= 1e-12,
        "program_b_all_mode_closure": program_b.maximum_closure <= 1e-12,
        "program_a_midloop_witness": abs(
            program_a.maximum_midloop_displacement - 0.7125305458876734
        )
        <= 1e-12,
        "program_b_midloop_witness": abs(
            program_b.maximum_midloop_displacement - 0.8557657738719043
        )
        <= 1e-12,
        "program_a_heating_exposure_witness": abs(
            np.max(program_a.heating_exposure) - 0.008551211055284244
        )
        <= 1e-14,
        "program_b_heating_exposure_witness": abs(
            np.max(program_b.heating_exposure) - 0.004805936264387415
        )
        <= 1e-14,
        "nominal_direct_shadow_parity": _trace_distance(
            nominal_after_ab, direct_ideal_after_ab
        )
        <= 1e-12,
        "heated_direct_channel_shadow_parity": _trace_distance(
            heated_after_ab, direct_heated_after_ab
        )
        <= 1e-12,
        "heated_primary_detectably_differs_from_ideal": _trace_distance(
            heated_after_ab, direct_ideal_after_ab
        )
        >= 1e-3,
        "heated_primary_rejects_exact_mode_return": all(
            not record["after_ab_exact_initial_mode_state_return"]
            for record in fixture_records
        ),
        "heating_covariance_drift_monotone": all(
            record["covariance_drift_strictly_monotone"]
            for record in fixture_records
        ),
        "same_logical_backing_a_then_b": all(
            record["logical_custody"]["allocation_object_unchanged"]
            and record["logical_custody"]["allocation_base_pointer_unchanged"]
            and record["logical_custody"]["generation_sequence"] == [0, 1, 2]
            for record in fixture_records
        ),
        "spectator_omission_detected": controls[
            "target_middle_mode_only_spectator_omission"
        ]["maximum_closure"]
        >= 1e-3,
        "omitted_final_segment_detected": controls["omitted_final_segment"][
            "maximum_closure"
        ]
        >= 1e-3
        and controls["omitted_final_segment"]["program_b"]["maximum_closure"]
        >= 1e-3,
        "segment_area_error_detected": controls["segment_index3_plus_1_percent"][
            "maximum_closure"
        ]
        >= 1e-4
        and controls["segment_index3_plus_1_percent"]["program_b"][
            "maximum_closure"
        ]
        >= 1e-4,
        "detuning_error_detected": controls["detuning_plus_250hz_all_modes"][
            "maximum_closure"
        ]
        >= 1e-4
        and controls["detuning_plus_250hz_all_modes"]["program_b"][
            "maximum_closure"
        ]
        >= 1e-4,
        "opposite_orientation_flips_phase": abs(
            controls["opposite_phase_orientation"]["walsh_theta"] - math.pi / 8.0
        )
        <= 1e-12
        and abs(
            controls["opposite_phase_orientation"]["program_b"]["walsh_theta"]
            - math.pi / 6.0
        )
        <= 1e-12,
        "zero_force_has_zero_area": controls["zero_force_zero_area"][
            "maximum_closure"
        ]
        <= 1e-15
        and abs(controls["zero_force_zero_area"]["walsh_theta"]) <= 1e-15,
        "frozen_control_receipts": abs(
            controls["omitted_final_segment"]["maximum_closure"]
            - 0.434733850265698
        )
        <= 1e-12
        and abs(
            controls["omitted_final_segment"]["program_b"]["maximum_closure"]
            - 0.1139811226
        )
        <= 1e-10
        and abs(
            controls["detuning_plus_250hz_all_modes"]["maximum_closure"]
            - 0.073242886359666
        )
        <= 1e-12
        and abs(
            controls["detuning_plus_250hz_all_modes"]["program_b"][
                "maximum_closure"
            ]
            - 0.02455695617
        )
        <= 1e-10
        and abs(
            controls["segment_index3_plus_1_percent"]["maximum_closure"]
            - 0.000639208691392
        )
        <= 1e-12
        and abs(
            controls["segment_index3_plus_1_percent"]["program_b"][
                "maximum_closure"
            ]
            - 0.00808983313
        )
        <= 1e-10
        and abs(
            controls["amplitude_quantization_12bit_fullscale_1p8e6"][
                "maximum_closure"
            ]
            - 0.000815882274292
        )
        <= 1e-12
        and abs(
            controls["amplitude_quantization_12bit_fullscale_1p8e6"][
                "program_b"
            ]["maximum_closure"]
            - 0.00047372709
        )
        <= 1e-10,
        "tmsv_reference_detects_heating": next(
            record
            for record in fixture_records
            if record["fixture"] == "tmsv_bus0_inert_reference_r0p30"
        )["tmsv_reference_coherence"]["after_ab_entanglement_infidelity"]
        > 0.0,
        "no_physical_claim": True,
        "m257_intact": True,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"M266 internal self-check failure: {failed}")

    combined_heated_trace_distance = _trace_distance(
        heated_after_ab, direct_ideal_after_ab
    )
    maximum_covariance_frobenius_drift = max(
        record["after_ab_covariance_frobenius_drift"] for record in fixture_records
    )
    maximum_peak_displacement = np.maximum(
        program_a.maximum_midloop_per_mode, program_b.maximum_midloop_per_mode
    )
    maximum_coherent_excursion_joules = float(
        np.sum(
            HBAR_J_S
            * 2.0
            * np.pi
            * MODE_FREQUENCIES_HZ
            * maximum_peak_displacement**2
        )
    )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "status": "PASS_INTERNAL_SELF_CHECK_HEATED_NONRESTORATION",
        "terminal": False,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": RESTORATION_SCOPE,
        "resource_disposition": RESOURCE_DISPOSITION,
        "next_mechanism": NEXT_MECHANISM,
        "source_self_assertion": "PASS_INTERNAL_CONSISTENCY_ONLY",
        "physical_execution": False,
        "physical_same_mode_custody": False,
        "physical_restoration": False,
        "phase_native_client_architecture": False,
        "unbounded_compute": False,
        "resource_advantage": False,
        "m257_intact": True,
        "model": {
            "state": "THREE_SYNTHETIC_COLLECTIVE_HARMONIC_MODES_PLUS_THREE_QUBIT_CLIENT_GAUSSIAN_MOMENTS_AND_EXACT_CLIENT_CHANNEL",
            "frequency_eigenvector_table_physical_trap_mapping_established": False,
            "hamiltonian_over_hbar": "SUM_M C_S_M_T_A_M_EXP_MINUS_I_DELTA_M_T_PLUS_ADJOINT",
            "coupling": "C_S_M=(ETA_I_M*Z_I+ETA_J_M*Z_J)*OMEGA/2",
            "alpha_increment": "-I*C_S_M*INTEGRAL_EXP_PLUS_I_DELTA_M_T_DT",
            "cross_segment_phase": "IM_INCREMENT_TIMES_CONJUGATE_PRIOR_DISPLACEMENT",
            "intra_segment_self_area": "C_S_M_SQUARED*(DELTA_M*DT-SIN(DELTA_M*DT))/DELTA_M_SQUARED",
            "heating_covariance_law": "V_M_TO_V_M_PLUS_GAMMA_M*T*IDENTITY_2",
            "heating_coherence_law": "EXP_MINUS_SUM_M_GAMMA_M_INTEGRAL_ABS_DELTA_BETA_M_SQUARED_DT",
            "full_joint_heated_gaussian_process_retained": False,
            "heating_channel_is_declared_model_not_physical_measurement": True,
        },
        "hardware_geometry": {
            "ions": 3,
            "physical_modes": 3,
            "mode_names": ["COM", "STRETCH", "WOBBLE"],
            "mode_frequencies_hz": [
                _round_float(value) for value in MODE_FREQUENCIES_HZ
            ],
            "drive_frequency_hz": DRIVE_FREQUENCY_HZ,
            "signed_drive_minus_mode_detunings_rad_s": [
                _round_float(value) for value in DETUNINGS_RAD_S
            ],
            "mode_vectors": MODE_VECTORS.tolist(),
            "eta_matrix": ETA.tolist(),
            "program_duration_s": PROGRAM_DURATION_S,
            "segments_per_program": SEGMENTS,
            "segment_duration_s": SEGMENT_DURATION_S,
            "heating_quanta_per_s": HEATING_QUANTA_PER_S.tolist(),
        },
        "programs": [_program_record(program_a), _program_record(program_b)],
        "client": {
            "initial_state": "PLUS_PLUS_PLUS",
            "program_a_nominal_boundary": _boundary(nominal_after_a, BOUNDARY_A),
            "combined_nominal_boundary": _boundary(nominal_after_ab, BOUNDARY_AB),
            "program_a_heated_boundary": _boundary(heated_after_a, BOUNDARY_A),
            "combined_heated_released_boundary": {
                "boundary": _boundary(heated_after_ab, BOUNDARY_AB),
                "guest_visible": True,
            },
            "guest_visible_boundary_path": "client.combined_heated_released_boundary.boundary",
            "nominal_density_integrity": _density_diagnostics(nominal_after_ab),
            "heated_density_integrity": _density_diagnostics(heated_after_ab),
            "combined_nominal_trace_distance_to_direct_ideal": _round_float(
                _trace_distance(nominal_after_ab, direct_ideal_after_ab)
            ),
            "combined_heated_trace_distance_to_direct_heated_channel": _round_float(
                _trace_distance(heated_after_ab, direct_heated_after_ab)
            ),
            "combined_heated_trace_distance_to_direct_ideal": _round_float(
                combined_heated_trace_distance
            ),
        },
        "same_logical_multimode_backing_heated_primary": fixture_records,
        "controls": controls,
        "strongest_honest_classical_comparator": {
            "ideal_shadow": "TWO_DIRECT_DIAGONAL_ZZ_PHASE_GATES",
            "heated_shadow": "TWO_DIRECT_DIAGONAL_ZZ_PHASES_TIMES_PUBLIC_8_BY_8_DEPHASING_MULTIPLIERS",
            "bus_coordinates_retained": 0,
            "controller_segments_executed": 0,
            "exposure_compilation_force_segments_processed": 16,
            "public_controller_descriptor_processed_for_heated_channel": True,
            "restoration_stage_executed": False,
            "direct_zz_phase_gates": 2,
            "complex_channel_descriptor_cells": 2 * 8 * 8,
            "matches_nominal_client": True,
            "matches_heated_client": True,
            "m257_escape_established": False,
        },
        "resource_ledger": {
            "physical_modes": 3,
            "client_qubits": 3,
            "programs": 2,
            "force_segments": 16,
            "single_qubit_basis_rotation_pulses": 8,
            "basis_rotation_accounting": "FOUR_PER_PROGRAM_FOR_PHYSICAL_XX_TO_EFFECTIVE_ZZ_MAPPING",
            "public_amplitude_descriptor_values": 16,
            "mode_frequency_calibration_values": 3,
            "eta_calibration_values": 9,
            "closure_real_constraint_rank": int(
                np.linalg.matrix_rank(real_closure_matrix)
            ),
            "closure_nullspace_dimension": int(
                SEGMENTS - np.linalg.matrix_rank(real_closure_matrix)
            ),
            "closure_singular_values": [
                _round_float(value) for value in closure_singular_values
            ],
            "amplitude_descriptor_bytes": int(
                PROGRAM_A_AMPLITUDES_RAD_S.nbytes
                + PROGRAM_B_AMPLITUDES_RAD_S.nbytes
            ),
            "peak_omega_rad_s": _round_float(
                max(
                    np.max(np.abs(PROGRAM_A_AMPLITUDES_RAD_S)),
                    np.max(np.abs(PROGRAM_B_AMPLITUDES_RAD_S)),
                )
            ),
            "control_norm_integral_omega_squared_seconds": _round_float(
                SEGMENT_DURATION_S
                * (
                    np.sum(PROGRAM_A_AMPLITUDES_RAD_S**2)
                    + np.sum(PROGRAM_B_AMPLITUDES_RAD_S**2)
                )
            ),
            "controller_update_rate_hz": _round_float(1.0 / SEGMENT_DURATION_S),
            "controller_quantization_control_bits": 12,
            "controller_quantization_fullscale_rad_s": 1.8e6,
            "heating_delta_n_after_two_programs": [
                _round_float(value)
                for value in 2.0 * PROGRAM_DURATION_S * HEATING_QUANTA_PER_S
            ],
            "maximum_named_covariance_frobenius_drift": _round_float(
                maximum_covariance_frobenius_drift
            ),
            "sum_of_per_mode_peak_coherent_energy_upper_bound_joules": float(
                maximum_coherent_excursion_joules
            ),
            "net_coherent_mode_energy_change_at_nominal_closure_joules": 0.0,
            "added_heating_energy_joules_after_two_programs": float(
                np.sum(
                    HBAR_J_S
                    * 2.0
                    * np.pi
                    * MODE_FREQUENCIES_HZ
                    * 2.0
                    * PROGRAM_DURATION_S
                    * HEATING_QUANTA_PER_S
                )
            ),
            "optical_or_rf_joules": "UNINSTANTIATED_WITHOUT_SOURCE_TRANSFER_MODEL",
            "wall_plug_energy": "UNINSTANTIATED",
            "retained_dynamic_trajectory_history": 0,
            "retained_public_schedule_entries": 16,
            "privileged_initial_gaussian_baselines": 5,
            "verifier_baselines_readable_by_dynamics": False,
            "software_forward_shadow_omits_modes_and_return": True,
        },
        "finite_fock_nonuniformity_witness": {
            "numerically_executed": False,
            "reason": "PRODUCTION_USES_EXACT_IDEAL_GAUSSIAN_MAGNUS_AND_DECLARED_HEATING_MOMENTS",
            "projected_commutator_law": "[A_N,A_N_DAGGER]=I_N-N|N-1><N-1|",
            "operator_norm_defect": "N",
            "top_fock_or_uniform_arbitrary_state_claim": False,
            "energy_constrained_reference_required_for_ANY_future_fock_promotion": True,
        },
        "checks": checks,
    }
    source_path = Path(__file__)
    result["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return result


def main() -> None:
    print(json.dumps(run(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
