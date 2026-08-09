#!/usr/bin/env python3
"""Independent continuum reference for the M266 three-mode SDF loop.

The public descriptor is repeated literally.  An exact constant-envelope
segment recurrence is compared with a separately written RK4 continuum
integrator.  This is software evidence and makes no restoration claim.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


REFERENCE_ID = "M266_MULTIMODE_TRAPPED_ION_WEYL_LOOP_SEPARATE_REFERENCE_V1"
SCHEMA = "PHASE_QEMU_V8_TRAPPED_ION_WEYL_LOOP_SEPARATE_REFERENCE_V1"
CLAIM = (
    "THREE_MODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_NULLSPACE_PULSES_CLOSE_ALL_"
    "NOMINAL_MODE_DISPLACEMENTS_AND_IMPLEMENT_TWO_DISTINCT_ZZ_PHASE_PROGRAMS_"
    "ON_ONE_LOGICAL_MULTIMODE_BACKING_WHILE_DECLARED_NONZERO_MARKOVIAN_HEATING_"
    "MONOTONICALLY_BREAKS_EXACT_INITIAL_MODE_STATE_RETURN_WITHOUT_RECOOLING"
)
CEILING = (
    "DETERMINISTIC_COMPLEX128_FLOAT64_GAUSSIAN_MOMENT_SOFTWARE_DIGITAL_TWIN_"
    "WITH_DECLARED_LINEAR_HARMONIC_STATE_DEPENDENT_FORCE_AND_MARKOVIAN_"
    "ADDITIVE_HEATING_LAWS_NO_PHYSICAL_ION_CUSTODY_AND_DIRECT_COMPILED_CLIENT_"
    "CHANNEL_SHADOW"
)
RESTORATION_CLASSIFICATION = "NO_RESTORATION_CLAIM"
RESTORATION_SCOPE = (
    "NOMINAL_ZERO_HEATING_LOGICAL_GAUSSIAN_MODE_RETURN_ONLY_WITH_NONZERO_"
    "HEATING_EXACT_SAME_MODE_RETURN_REJECTED_AND_FRESH_MODE_SWAP_OR_RECOOLING_"
    "CLASSIFIED_AS_EXTERNAL_RESET"
)
DISPOSITION = (
    "DIRECT_COMPILED_ZZ_AND_DEPHASING_CHANNEL_SHADOW_OMITS_THE_THREE_MODE_"
    "CONTROLLER_LOOP_WHILE_HEATING_PREVENTS_CATALYTIC_RETURN_SO_NO_RESOURCE_"
    "ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED"
)
SUCCESSOR = (
    "CONDITIONAL_GAUSSIAN_CLOSED_LOOP_FORWARD_SHADOW_AND_IRREVERSIBLE_"
    "DIFFUSION_NO_RETURN"
)

T = 120.0e-6
SEGMENTS = 8
H = T / SEGMENTS
DELTA_HZ = np.array([11_000.0, -7_000.0, -19_000.0])
DELTA = 2.0 * math.pi * DELTA_HZ
GAMMA = np.array([15.0, 30.0, 60.0])
ETA = 0.06 * np.array(
    [
        [1 / math.sqrt(3), 1 / math.sqrt(2), 1 / math.sqrt(6)],
        [1 / math.sqrt(3), 0.0, -2 / math.sqrt(6)],
        [1 / math.sqrt(3), -1 / math.sqrt(2), 1 / math.sqrt(6)],
    ]
)
PULSE_A = np.array(
    [
        875322.2363528529,
        -1017217.5496960702,
        1149417.5349432244,
        -128702.55695608277,
        -128702.55695606946,
        1149417.5349432132,
        -1017217.549696064,
        875322.2363528487,
    ]
)
PULSE_B = np.array(
    [
        -229497.22240917346,
        725694.6296712102,
        -1293755.9826773852,
        1628861.1574510091,
        -1628861.1574510091,
        1293755.9826774026,
        -725694.6296712208,
        229497.22240918515,
    ]
)
PROGRAMS = (
    ("A_Z0_Z1_MINUS_PI_OVER_8", (0, 1), PULSE_A, -math.pi / 8),
    ("B_Z1_Z2_MINUS_PI_OVER_6", (1, 2), PULSE_B, -math.pi / 6),
)
GRIDS = (1024, 2048, 4096)

I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def clean(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def exact_recurrence(
    pulses: np.ndarray, detuning: np.ndarray = DELTA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact segment recurrence including continuous intra-segment area."""
    alpha = np.zeros(3, dtype=np.complex128)
    area = np.zeros(3)
    kick_area = np.zeros(3)
    for segment, omega in enumerate(pulses):
        q = omega / 2.0
        angle = detuning * H
        increment = (
            -(q / detuning)
            * np.exp(1j * detuning * segment * H)
            * (np.exp(1j * angle) - 1)
        )
        between = np.imag(np.conj(alpha) * increment)
        self_area = q**2 / detuning**2 * (angle - np.sin(angle))
        kick_area += between
        area += between + self_area
        alpha += increment
    return alpha, area, kick_area


def rk4(pulses: np.ndarray, ions: tuple[int, int], steps: int) -> dict[str, object]:
    if steps % SEGMENTS:
        raise ValueError("grid does not align with segment boundaries")
    dt = T / steps
    alpha = np.zeros(3, dtype=np.complex128)
    area = np.zeros(3)
    exposure = np.zeros(3)
    maximum_conditional = 0.0

    def derivative(t: float, state: np.ndarray, omega: float):
        velocity = -0.5j * omega * np.exp(1j * DELTA * t)
        return velocity, np.imag(np.conj(state) * velocity), np.abs(state) ** 2

    for step in range(steps):
        omega = float(pulses[step // (steps // SEGMENTS)])
        instant = step * dt
        k1 = derivative(instant, alpha, omega)
        k2 = derivative(instant + dt / 2, alpha + dt * k1[0] / 2, omega)
        k3 = derivative(instant + dt / 2, alpha + dt * k2[0] / 2, omega)
        k4 = derivative(instant + dt, alpha + dt * k3[0], omega)
        stage_states = (
            alpha,
            alpha + dt * k1[0] / 2,
            alpha + dt * k2[0] / 2,
            alpha + dt * k3[0],
        )
        for state in stage_states:
            for first_spin, second_spin in itertools.product((1.0, -1.0), repeat=2):
                scale = ETA[ions[0]] * first_spin + ETA[ions[1]] * second_spin
                maximum_conditional = max(
                    maximum_conditional, float(np.max(np.abs(state * scale)))
                )
        alpha += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        area += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        exposure += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
    zz = 2 * float(np.sum(area * ETA[ions[0]] * ETA[ions[1]]))
    return {
        "steps": steps,
        "base_displacement": alpha,
        "base_magnus_area_by_mode": area,
        "base_displacement_exposure_seconds_by_mode": exposure,
        "zz_exponent_coefficient_rad": zz,
        "maximum_single_mode_conditional_displacement": maximum_conditional,
    }


def make_record(
    name: str, ions: tuple[int, int], pulses: np.ndarray, target: float
) -> dict[str, object]:
    alpha, area, kick_area = exact_recurrence(pulses)
    exact_zz = 2 * float(np.sum(area * ETA[ions[0]] * ETA[ions[1]]))
    kick_zz = 2 * float(np.sum(kick_area * ETA[ions[0]] * ETA[ions[1]]))
    grids = {str(size): rk4(pulses, ions, size) for size in GRIDS}
    exposure = np.asarray(
        grids[str(GRIDS[-1])]["base_displacement_exposure_seconds_by_mode"]
    )
    maximum_heating_exponent = 0.0
    for left in itertools.product((1.0, -1.0), repeat=2):
        for right in itertools.product((1.0, -1.0), repeat=2):
            difference = (
                ETA[ions[0]] * (left[0] - right[0])
                + ETA[ions[1]] * (left[1] - right[1])
            )
            maximum_heating_exponent = max(
                maximum_heating_exponent,
                float(np.sum(GAMMA * exposure * difference**2)),
            )
    return {
        "name": name,
        "ions": ions,
        "target_zz_exponent_coefficient_rad": target,
        "exact_segment_recurrence": {
            "base_displacement": alpha,
            "base_magnus_area_by_mode": area,
            "maximum_base_displacement_abs": float(np.max(np.abs(alpha))),
            "zz_exponent_coefficient_rad": exact_zz,
        },
        "rk4_continuum_grids": grids,
        "intra_segment_self_area_control": {
            "kick_only_zz_exponent_coefficient_rad": kick_zz,
            "abs_phase_error_when_self_area_is_omitted": abs(kick_zz - exact_zz),
        },
        "maximum_heating_dephasing_exponent": maximum_heating_exponent,
    }


def tensor3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


OBSERVABLES = {
    "X0": tensor3(X2, I2, I2),
    "X1": tensor3(I2, X2, I2),
    "X2": tensor3(I2, I2, X2),
    "Y0Z1": tensor3(Y2, Z2, I2),
    "Z0Y1": tensor3(Z2, Y2, I2),
    "Z1Y2": tensor3(I2, Z2, Y2),
    "X0X1": tensor3(X2, X2, I2),
    "X1X2": tensor3(I2, X2, X2),
}


def spins(index: int) -> np.ndarray:
    return np.array(
        [1.0 if ((index >> (2 - qubit)) & 1) == 0 else -1.0 for qubit in range(3)]
    )


def expectation(rho: np.ndarray, operator: np.ndarray) -> float:
    value = np.trace(rho @ operator)
    if abs(value.imag) > 1e-11:
        raise AssertionError(f"non-real expectation {value}")
    return float(value.real)


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = (a - b + (a - b).conj().T) / 2
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))


def client_channel(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    ideal = np.ones((8, 8), dtype=np.complex128) / 8
    heated = ideal.copy()
    basis = tuple(spins(index) for index in range(8))
    for descriptor, record in zip(PROGRAMS, records, strict=True):
        _, ions, _, _ = descriptor
        coefficient = float(
            record["exact_segment_recurrence"]["zz_exponent_coefficient_rad"]
        )
        exposure = np.asarray(
            record["rk4_continuum_grids"][str(GRIDS[-1])][
                "base_displacement_exposure_seconds_by_mode"
            ]
        )
        for row, sr in enumerate(basis):
            for column, sc in enumerate(basis):
                phase = coefficient * (
                    sr[ions[0]] * sr[ions[1]] - sc[ions[0]] * sc[ions[1]]
                )
                difference = (
                    ETA[ions[0]] * (sr[ions[0]] - sc[ions[0]])
                    + ETA[ions[1]] * (sr[ions[1]] - sc[ions[1]])
                )
                exponent = float(np.sum(GAMMA * exposure * difference**2))
                ideal[row, column] *= np.exp(1j * phase)
                heated[row, column] *= np.exp(1j * phase - exponent)
    return {
        "ideal_boundary": {
            name: expectation(ideal, operator) for name, operator in OBSERVABLES.items()
        },
        "heated_boundary": {
            name: expectation(heated, operator)
            for name, operator in OBSERVABLES.items()
        },
        "heated_client_trace_distance_to_ideal": trace_distance(heated, ideal),
        "heated_client_purity": float(np.trace(heated @ heated).real),
        "heated_client_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(heated))),
        "direct_shadow_law": (
            "TWO_DIAGONAL_ZZ_PHASES_PLUS_ANALYTIC_CORRELATED_DEPHASING_WITHOUT_"
            "MOTIONAL_MODE_OR_CONTROLLER_LOOP_EMULATION"
        ),
    }


def fixture_ledger() -> dict[str, object]:
    combined_delta_n = 2 * T * GAMMA
    covariance_drift = math.sqrt(2 * float(np.sum(combined_delta_n**2)))
    vacuum_survival = float(np.prod(1 / (1 + combined_delta_n)))
    r, phi = 0.35, 0.20
    rotation = np.array(
        [
            [math.cos(phi / 2), -math.sin(phi / 2)],
            [math.sin(phi / 2), math.cos(phi / 2)],
        ]
    )
    squeezed_covariance = rotation @ np.diag(
        [math.exp(-2 * r) / 2, math.exp(2 * r) / 2]
    ) @ rotation.T
    return {
        "named_initial_fixtures": {
            "vacuum": {"nbar": [0.0, 0.0, 0.0]},
            "thermal": {"nbar": [0.05, 0.10, 0.20]},
            "coherent": {
                "alpha": [
                    {"real": 0.30, "imag": 0.15},
                    {"real": -0.20, "imag": 0.10},
                    {"real": 0.10, "imag": -0.25},
                ]
            },
            "squeezed_mode0_r0p35_phi0p20_with_thermal_spectators": {
                "mode0_covariance": squeezed_covariance,
                "spectator_nbar": [0.10, 0.20],
            },
            "tmsv_bus0_inert_reference_r0p30": {
                "squeezing_r": 0.30,
                "other_modes_nbar": [0.0, 0.0],
                "reference_is_inert": True,
            },
        },
        "nominal_zero_heating": {
            "means_return": True,
            "covariances_return": True,
            "bus_reference_cross_covariance_returns": True,
            "logical_gaussian_return_only": True,
        },
        "declared_heating_after_a_then_b": {
            "delta_n_by_mode": combined_delta_n,
            "covariance_drift_frobenius": covariance_drift,
            "vacuum_fixture_trace_distance_from_initial": 1 - vacuum_survival,
            "means_unchanged_does_not_establish_state_return": True,
            "exact_initial_mode_state_return": False,
        },
    }


def controls(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    result: dict[str, object] = {}
    drift = DELTA + 2 * math.pi * np.array([50.0, -75.0, 100.0])
    for descriptor, record in zip(PROGRAMS, records, strict=True):
        name, ions, pulses, _ = descriptor
        exact_zz = float(
            record["exact_segment_recurrence"]["zz_exponent_coefficient_rad"]
        )
        omitted = pulses.copy()
        omitted[-1] = 0.0
        omitted_alpha, omitted_area, _ = exact_recurrence(omitted)
        omitted_zz = 2 * float(np.sum(omitted_area * ETA[ions[0]] * ETA[ions[1]]))
        drift_alpha, drift_area, _ = exact_recurrence(pulses, drift)
        drift_zz = 2 * float(np.sum(drift_area * ETA[ions[0]] * ETA[ions[1]]))
        phase = np.asarray(
            record["exact_segment_recurrence"]["base_magnus_area_by_mode"]
        )
        no_spectator_zz = 2 * float(
            np.sum(phase[:2] * ETA[ions[0], :2] * ETA[ions[1], :2])
        )
        result[name] = {
            "last_segment_omission": {
                "maximum_base_displacement_abs": float(np.max(np.abs(omitted_alpha))),
                "zz_exponent_coefficient_rad": omitted_zz,
            },
            "mode2_spectator_omission": {
                "zz_exponent_coefficient_rad": no_spectator_zz,
                "abs_phase_error": abs(no_spectator_zz - exact_zz),
            },
            "detuning_drift_hz_50_minus75_plus100": {
                "maximum_base_displacement_abs": float(np.max(np.abs(drift_alpha))),
                "zz_exponent_coefficient_rad": drift_zz,
                "abs_phase_error": abs(drift_zz - exact_zz),
            },
        }
    result["architecture_shams"] = {
        "direct_client_gate": "MATCHES_BOUNDARY_BUT_OMITS_CAUSAL_MODE_LOOP",
        "recooling": "EXTERNAL_RESET_NOT_CATALYTIC_RESTORATION",
        "snapshot_reload": "FORBIDDEN_HISTORY_BASED_RESET",
        "fresh_mode_swap": (
            "BLACK_BOX_EQUIVALENT_REQUIRES_OUT_OF_BAND_CUSTODY_AND_IS_NOT_"
            "ESTABLISHED_HERE"
        ),
        "forced_zero_heating": "SHAM_FOR_THE_DECLARED_NONZERO_HEATING_MODEL",
    }
    return result


def main() -> int:
    records = tuple(make_record(*descriptor) for descriptor in PROGRAMS)
    channel = client_channel(records)
    fixtures = fixture_ledger()
    named_controls = controls(records)

    exact_phase_errors = []
    convergences = []
    finest_errors = []
    for descriptor, record in zip(PROGRAMS, records, strict=True):
        exact_zz = float(
            record["exact_segment_recurrence"]["zz_exponent_coefficient_rad"]
        )
        exact_phase_errors.append(abs(exact_zz - descriptor[3]))
        errors = [
            abs(
                float(
                    record["rk4_continuum_grids"][str(grid)][
                        "zz_exponent_coefficient_rad"
                    ]
                )
                - exact_zz
            )
            for grid in GRIDS
        ]
        convergences.append(errors[2] < errors[1] < errors[0])
        finest_errors.append(errors[-1])

    checks = {
        "exact_closure_max_le_5e_minus_12": max(
            float(record["exact_segment_recurrence"]["maximum_base_displacement_abs"])
            for record in records
        )
        <= 5e-12,
        "target_phase_abs_error_le_2e_minus_12": max(exact_phase_errors) <= 2e-12,
        "rk4_strictly_converges_1024_2048_4096": all(convergences),
        "finest_rk4_phase_abs_error_le_2e_minus_10": max(finest_errors) <= 2e-10,
        "intra_segment_self_area_omission_detected": min(
            float(
                record["intra_segment_self_area_control"][
                    "abs_phase_error_when_self_area_is_omitted"
                ]
            )
            for record in records
        )
        >= 1e-3,
        "heated_client_channel_differs": float(
            channel["heated_client_trace_distance_to_ideal"]
        )
        >= 2e-3,
        "heated_client_purity_decreases": float(channel["heated_client_purity"])
        < 0.995,
        "heated_client_channel_is_positive": float(
            channel["heated_client_minimum_eigenvalue"]
        )
        >= -1e-12,
        "heating_rejects_exact_mode_return": not bool(
            fixtures["declared_heating_after_a_then_b"][
                "exact_initial_mode_state_return"
            ]
        ),
        "pulse_omission_detected": min(
            float(
                named_controls[descriptor[0]]["last_segment_omission"][
                    "maximum_base_displacement_abs"
                ]
            )
            for descriptor in PROGRAMS
        )
        >= 1,
        "spectator_omission_detected": min(
            float(
                named_controls[descriptor[0]]["mode2_spectator_omission"][
                    "abs_phase_error"
                ]
            )
            for descriptor in PROGRAMS
        )
        >= 1e-2,
        "detuning_drift_detected": min(
            float(
                named_controls[descriptor[0]][
                    "detuning_drift_hz_50_minus75_plus100"
                ]["maximum_base_displacement_abs"]
            )
            for descriptor in PROGRAMS
        )
        >= 1e-2,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"independent reference self-check failed: {failed}")

    descriptor = {
        "duration_seconds": T,
        "segment_count": SEGMENTS,
        "detuning_hz": DELTA_HZ,
        "eta": ETA,
        "heating_quanta_per_second": GAMMA,
        "lindblad_law": (
            "SUM_M_GAMMA_M_TIMES_D_A_M_PLUS_D_A_M_DAGGER_WITH_"
            "D_L_RHO=L_RHO_L_DAGGER-MINUS_ONE_HALF_ANTICOMMUTATOR_L_DAGGER_L_RHO"
        ),
        "covariance_and_occupation_law": "V_DOT_M=GAMMA_M*I2_NBAR_DOT_M=GAMMA_M",
        "coherence_law": (
            "EXP_MINUS_SUM_M_GAMMA_M_INTEGRAL_ABS_DELTA_BETA_M_T_SQUARED_DT"
        ),
        "programs": [
            {
                "name": name,
                "ions": ions,
                "pulses_rad_per_second": pulses,
                "target_zz_exponent_coefficient_rad": target,
            }
            for name, ions, pulses, target in PROGRAMS
        ],
    }
    payload = {
        "schema": SCHEMA,
        "reference_id": REFERENCE_ID,
        "milestone": "M266",
        "claim": CLAIM,
        "ceiling": CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": RESTORATION_SCOPE,
        "resource_disposition": DISPOSITION,
        "next_mechanism": SUCCESSOR,
        "public_descriptor": descriptor,
        "program_records": records,
        "gaussian_fixture_ledger": fixtures,
        "direct_compiled_client_channel": channel,
        "named_controls": named_controls,
        "checks": checks,
        "claims": {
            "nominal_zero_heating_logical_gaussian_mode_return": True,
            "nonzero_heating_exact_initial_mode_state_return": False,
            "same_backing_restoration": False,
            "physical_execution": False,
            "physical_same_mode_custody": False,
            "physical_restoration": False,
            "fresh_mode_swap_is_restoration": False,
            "recooling_is_catalytic_restoration": False,
            "computational_advantage": False,
            "m257_escape": False,
            "unbounded_compute": False,
            "bit_replaced_with_pi": False,
        },
        "architecture_authority": {
            "reference_is_physical_evidence": False,
            "equal_access_direct_compiled_shadow_is_allowed": True,
            "carrier_identity_requires_out_of_band_custody": True,
            "digital_twin_does_not_authenticate_physical_custody": True,
            "retained_dynamic_trajectory_history": False,
            "retained_inverse_history": False,
        },
        "reference_self_assertion": "PASS_INDEPENDENT_CONTINUUM_REFERENCE_SELF_CHECK",
        "status": "PASS_INDEPENDENT_CONTINUUM_REFERENCE_SELF_CHECK",
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
