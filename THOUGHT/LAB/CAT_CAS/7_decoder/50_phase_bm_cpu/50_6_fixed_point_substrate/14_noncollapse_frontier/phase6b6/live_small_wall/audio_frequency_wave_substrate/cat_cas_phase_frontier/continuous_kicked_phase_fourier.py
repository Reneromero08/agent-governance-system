#!/usr/bin/env python3
"""Continuous kicked-phase/Fourier carrier and compactness kill test."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass

import numpy as np
from scipy import fft as scipy_fft
from scipy.special import jv


K_PRIMARY = math.sqrt(2.0)
TAU_PRIMARY = math.sqrt(3.0)
EPSILON_ENERGY = 1.0e-12
EPSILON_SWEEP = (1.0e-10, 1.0e-12, 1.0e-14)
RESTORATION_TOLERANCE = 5.0e-11
NORM_TOLERANCE = 5.0e-11
CROSS_GRID_TOLERANCE = 2.0e-10
BESSEL_L2_TOLERANCE = 2.0e-12
PERIODIC_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
SCRAMBLED_DEPTHS = PERIODIC_DEPTHS
GRID_SIZES = (256, 512, 1024, 2048)
SCRAMBLED_GRID_SIZES = (4096, 8192)


def fail(message: str) -> None:
    raise RuntimeError(message)


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def phase_offset(step: int, schedule: str) -> float:
    if schedule == "PERIODIC":
        return 0.0
    if schedule == "SCRAMBLED_17":
        return float((step * step) % 17) * 2.0 * math.pi / 17.0
    if schedule == "REUSE":
        return float((5 * step + 3) % 19) * 2.0 * math.pi / 19.0
    fail("continuous kicked-phase schedule invalid")


@dataclass
class Carrier:
    samples: np.ndarray
    generation: int = 0


def zero_momentum_carrier(grid_size: int) -> Carrier:
    return Carrier(
        np.ones(grid_size, dtype=np.complex128) / math.sqrt(grid_size)
    )


def topology(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(grid_size, dtype=np.float64)
    theta *= 2.0 * math.pi / grid_size
    modes = np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    return theta, modes


def forward_step(
    samples: np.ndarray,
    theta: np.ndarray,
    modes: np.ndarray,
    kick_strength: float,
    free_time: float,
    offset: float,
) -> None:
    samples *= np.exp(
        -1j * kick_strength * np.cos(theta + offset)
    )
    momentum = np.fft.fft(samples, norm="ortho")
    momentum *= np.exp(-0.5j * free_time * modes * modes)
    samples[:] = np.fft.ifft(momentum, norm="ortho")


def inverse_step(
    samples: np.ndarray,
    theta: np.ndarray,
    modes: np.ndarray,
    kick_strength: float,
    free_time: float,
    offset: float,
) -> None:
    momentum = np.fft.fft(samples, norm="ortho")
    momentum *= np.exp(0.5j * free_time * modes * modes)
    samples[:] = np.fft.ifft(momentum, norm="ortho")
    samples *= np.exp(
        1j * kick_strength * np.cos(theta + offset)
    )


def boundary(
    initial: np.ndarray, samples: np.ndarray, theta: np.ndarray
) -> dict[str, object]:
    probability = np.abs(samples) ** 2
    return {
        "loschmidt_amplitude": complex_pair(
            complex(np.vdot(initial, samples))
        ),
        "cosine_moment": float(
            np.sum(probability * np.cos(theta))
        ),
        "sine_moment": float(
            np.sum(probability * np.sin(theta))
        ),
        "norm": float(np.linalg.norm(samples)),
    }


def boundary_distance(
    left: dict[str, object], right: dict[str, object]
) -> float:
    left_amp = complex(*left["loschmidt_amplitude"])
    right_amp = complex(*right["loschmidt_amplitude"])
    return max(
        abs(left_amp - right_amp),
        abs(float(left["cosine_moment"]) - float(right["cosine_moment"])),
        abs(float(left["sine_moment"]) - float(right["sine_moment"])),
        abs(float(left["norm"]) - float(right["norm"])),
    )


def fourier_radius(
    samples: np.ndarray, epsilon_energy: float = EPSILON_ENERGY
) -> tuple[int, float]:
    momentum = np.fft.fft(samples, norm="ortho")
    modes = np.fft.fftfreq(len(samples), d=1.0 / len(samples))
    energy = np.abs(momentum) ** 2
    for radius in range(len(samples) // 2):
        tail = float(np.sum(energy[np.abs(modes) > radius]))
        if tail <= epsilon_energy:
            return radius, tail
    fail("continuous kicked-phase grid aliases declared bandwidth")


def evolve_checkpoints(
    grid_size: int,
    depths: tuple[int, ...],
    schedule: str,
) -> dict[str, object]:
    carrier = zero_momentum_carrier(grid_size)
    initial = carrier.samples.copy()
    theta, modes = topology(grid_size)
    checkpoints: list[dict[str, object]] = []
    selected = set(depths)
    for step in range(1, max(depths) + 1):
        forward_step(
            carrier.samples,
            theta,
            modes,
            K_PRIMARY,
            TAU_PRIMARY,
            phase_offset(step, schedule),
        )
        if step in selected:
            radius, tail = fourier_radius(carrier.samples)
            checkpoints.append(
                {
                    "depth": step,
                    "epsilon_fourier_radius": radius,
                    "retained_modes": 2 * radius + 1,
                    "tail_energy": tail,
                    "boundary": boundary(
                        initial, carrier.samples, theta
                    ),
                }
            )
    return {
        "grid_size": grid_size,
        "schedule": schedule,
        "epsilon_energy": EPSILON_ENERGY,
        "checkpoints": checkpoints,
        "final_epsilon_sweep_radii": {
            f"{epsilon:.0e}": fourier_radius(
                carrier.samples, epsilon
            )[0]
            for epsilon in EPSILON_SWEEP
        },
    }


def high_precision_replay(
    grid_size: int, depth: int
) -> dict[str, object]:
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        fail("continuous kicked-phase extended precision unavailable")
    real_type = np.longdouble
    complex_type = np.clongdouble
    theta = np.arange(grid_size, dtype=real_type)
    theta *= real_type(2.0) * real_type(np.pi) / real_type(grid_size)
    modes = scipy_fft.fftfreq(
        grid_size, d=1.0 / grid_size
    ).astype(real_type)
    samples = np.ones(grid_size, dtype=complex_type)
    samples /= np.sqrt(real_type(grid_size))
    initial = samples.copy()
    kick_strength = np.sqrt(real_type(2.0))
    free_time = np.sqrt(real_type(3.0))
    kick = np.exp(-1j * kick_strength * np.cos(theta))
    free = np.exp(-0.5j * free_time * modes * modes)
    for _ in range(depth):
        samples *= kick
        momentum = scipy_fft.fft(samples, norm="ortho")
        momentum *= free
        samples[:] = scipy_fft.ifft(momentum, norm="ortho")
    momentum = scipy_fft.fft(samples, norm="ortho")
    energy = np.abs(momentum) ** 2
    radii: dict[str, int] = {}
    for epsilon in EPSILON_SWEEP:
        for radius in range(grid_size // 2):
            if np.sum(energy[np.abs(modes) > radius]) <= real_type(
                epsilon
            ):
                radii[f"{epsilon:.0e}"] = radius
                break
    loschmidt = np.vdot(initial, samples)
    probability = np.abs(samples) ** 2
    return {
        "grid_size": grid_size,
        "depth": depth,
        "real_mantissa_bits": int(np.finfo(real_type).nmant),
        "epsilon_sweep_radii": radii,
        "loschmidt_amplitude": [
            float(loschmidt.real),
            float(loschmidt.imag),
        ],
        "cosine_moment": float(
            np.sum(probability * np.cos(theta))
        ),
        "sine_moment": float(
            np.sum(probability * np.sin(theta))
        ),
        "norm": float(np.linalg.norm(samples)),
    }


def transaction(
    carrier: Carrier,
    depth: int,
    schedule: str,
    kick_strength: float,
    free_time: float,
) -> dict[str, object]:
    initial = carrier.samples.copy()
    theta, modes = topology(len(carrier.samples))
    for step in range(1, depth + 1):
        forward_step(
            carrier.samples,
            theta,
            modes,
            kick_strength,
            free_time,
            phase_offset(step, schedule),
        )
    latched_boundary = boundary(initial, carrier.samples, theta)
    latched_copy = json.loads(json.dumps(latched_boundary))
    for step in range(depth, 0, -1):
        inverse_step(
            carrier.samples,
            theta,
            modes,
            kick_strength,
            free_time,
            phase_offset(step, schedule),
        )
    restoration_error = float(
        np.max(np.abs(carrier.samples - initial))
    )
    if restoration_error > RESTORATION_TOLERANCE:
        fail("continuous kicked-phase actual restoration failed")
    if boundary_distance(latched_boundary, latched_copy) != 0.0:
        fail("continuous kicked-phase boundary latch changed")
    carrier.generation += 1
    return {
        "boundary": latched_boundary,
        "restoration_error": restoration_error,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "snapshot_loaded": False,
        "retained_inverse_history_bytes": 0,
        "inverse_topology_rematerialized": True,
    }


def inverse_controls(grid_size: int, depth: int) -> dict[str, float]:
    initial = zero_momentum_carrier(grid_size).samples
    theta, modes = topology(grid_size)

    def forward_copy() -> np.ndarray:
        samples = initial.copy()
        for step in range(1, depth + 1):
            forward_step(
                samples,
                theta,
                modes,
                K_PRIMARY,
                TAU_PRIMARY,
                phase_offset(step, "PERIODIC"),
            )
        return samples

    missing = float(np.max(np.abs(forward_copy() - initial)))
    wrong = forward_copy()
    for step in range(depth, 0, -1):
        inverse_step(
            wrong,
            theta,
            modes,
            K_PRIMARY * 1.01,
            TAU_PRIMARY,
            phase_offset(step, "PERIODIC"),
        )
    wrong_error = float(np.max(np.abs(wrong - initial)))

    reordered = forward_copy()
    kick_inverse = np.exp(1j * K_PRIMARY * np.cos(theta))
    free_inverse = np.exp(0.5j * TAU_PRIMARY * modes * modes)
    for _ in range(depth, 0, -1):
        reordered *= kick_inverse
        momentum = np.fft.fft(reordered, norm="ortho")
        momentum *= free_inverse
        reordered[:] = np.fft.ifft(momentum, norm="ortho")
    reordered_error = float(
        np.max(np.abs(reordered - initial))
    )

    disabled = initial.copy()
    for _ in range(depth):
        disabled *= np.exp(-1j * K_PRIMARY * np.cos(theta))
    fourier_disabled_difference = float(
        np.max(np.abs(disabled - forward_copy()))
    )
    return {
        "missing_inverse_error": missing,
        "wrong_inverse_error": wrong_error,
        "reordered_inverse_error": reordered_error,
        "fourier_disabled_difference": fourier_disabled_difference,
    }


def bessel_baseline(
    fft_samples: np.ndarray, depth: int
) -> dict[str, object]:
    kernel_radius = 16
    kernel_modes = np.arange(-kernel_radius, kernel_radius + 1)
    kernel = (
        np.power(-1j, kernel_modes)
        * jv(kernel_modes, K_PRIMARY)
    )
    fft_momentum = np.fft.fft(fft_samples, norm="ortho")
    fft_modes = np.fft.fftfreq(
        len(fft_samples), d=1.0 / len(fft_samples)
    ).astype(np.int64)
    candidate_errors: list[dict[str, object]] = []
    selected: tuple[
        int, np.ndarray, np.ndarray, float
    ] | None = None
    for maximum_mode in range(24, 81, 4):
        modes = np.arange(-maximum_mode, maximum_mode + 1)
        state = np.zeros(
            2 * maximum_mode + 1, dtype=np.complex128
        )
        state[maximum_mode] = 1.0
        free = np.exp(-0.5j * TAU_PRIMARY * modes * modes)
        for _ in range(depth):
            state = np.convolve(state, kernel, mode="same")
            state *= free
        reference = np.array(
            [
                fft_momentum[
                    int(np.flatnonzero(fft_modes == mode)[0])
                ]
                for mode in modes
            ]
        )
        inside_error = float(np.linalg.norm(state - reference))
        outside_error = float(
            np.linalg.norm(
                fft_momentum[np.abs(fft_modes) > maximum_mode]
            )
        )
        total_l2_error = math.hypot(
            inside_error, outside_error
        )
        candidate_errors.append(
            {
                "maximum_mode": maximum_mode,
                "total_l2_error": total_l2_error,
            }
        )
        if (
            selected is None
            and total_l2_error <= BESSEL_L2_TOLERANCE
        ):
            selected = (maximum_mode, state, free, total_l2_error)
    if selected is None:
        fail("continuous kicked-phase Bessel guard search failed")
    maximum_mode, state, free, total_l2_error = selected
    tail_modes = np.arange(kernel_radius + 1, 128)
    kernel_tail_energy = float(
        2.0 * np.sum(np.square(jv(tail_modes, K_PRIMARY)))
    )
    cumulative_kernel_tail_l2_bound = (
        depth * math.sqrt(kernel_tail_energy)
    )
    resident_complex_cells = len(state)
    compiled_complex_cells = len(kernel) + len(free)
    scratch_complex_cells = len(state)
    complex_multiply_add_pairs = (
        depth * len(state) * len(kernel)
    )
    return {
        "maximum_mode": maximum_mode,
        "retained_modes": len(state),
        "kernel_radius": kernel_radius,
        "kernel_coefficients": len(kernel),
        "kernel_tail_energy": kernel_tail_energy,
        "cumulative_kernel_tail_l2_bound": (
            cumulative_kernel_tail_l2_bound
        ),
        "total_l2_error_against_fft": total_l2_error,
        "public_guard_search": candidate_errors,
        "resident_state_bytes": resident_complex_cells * 16,
        "compiled_program_bytes": compiled_complex_cells * 16,
        "scratch_bytes": scratch_complex_cells * 16,
        "complex_multiply_add_pairs": (
            complex_multiply_add_pairs
        ),
        "projection_complex_operations": len(state),
        "verification_reference_state_bytes": fft_samples.nbytes,
        "verification_fft_conceptual_complex_operations": int(
            len(fft_samples) * math.log2(len(fft_samples))
        ),
        "classical_equivalent_of_phase_harmonic_recurrence": True,
    }


def warm_runtime(
    grid_size: int, depth: int, repetitions: int
) -> dict[str, int]:
    theta, modes = topology(grid_size)

    def run_fft() -> None:
        samples = zero_momentum_carrier(grid_size).samples
        for step in range(1, depth + 1):
            forward_step(
                samples,
                theta,
                modes,
                K_PRIMARY,
                TAU_PRIMARY,
                phase_offset(step, "PERIODIC"),
            )

    maximum_mode = 48
    kernel_radius = 16
    harmonic_modes = np.arange(-maximum_mode, maximum_mode + 1)
    kernel_modes = np.arange(-kernel_radius, kernel_radius + 1)
    kernel = (
        np.power(-1j, kernel_modes)
        * jv(kernel_modes, K_PRIMARY)
    )
    free = np.exp(
        -0.5j * TAU_PRIMARY * harmonic_modes * harmonic_modes
    )

    def run_sparse() -> None:
        state = np.zeros(
            2 * maximum_mode + 1, dtype=np.complex128
        )
        state[maximum_mode] = 1.0
        for _ in range(depth):
            state = np.convolve(state, kernel, mode="same")
            state *= free

    run_fft()
    run_sparse()
    fft_times: list[int] = []
    sparse_times: list[int] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        run_fft()
        fft_times.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        run_sparse()
        sparse_times.append(time.perf_counter_ns() - start)
    return {
        "grid_size": grid_size,
        "depth": depth,
        "repetitions": repetitions,
        "warm_fft_median_ns": int(statistics.median(fft_times)),
        "warm_sparse_bessel_median_ns": int(
            statistics.median(sparse_times)
        ),
    }


def main() -> None:
    periodic_runs = [
        evolve_checkpoints(size, PERIODIC_DEPTHS, "PERIODIC")
        for size in GRID_SIZES
    ]
    reference_periodic = periodic_runs[-1]
    reference_boundaries = {
        item["depth"]: item["boundary"]
        for item in reference_periodic["checkpoints"]
    }
    cross_grid_max = 0.0
    for run in periodic_runs[:-1]:
        for item in run["checkpoints"]:
            cross_grid_max = max(
                cross_grid_max,
                boundary_distance(
                    item["boundary"],
                    reference_boundaries[item["depth"]],
                ),
            )
    if cross_grid_max > CROSS_GRID_TOLERANCE:
        fail("continuous kicked-phase grid convergence failed")

    scrambled_runs = [
        evolve_checkpoints(size, SCRAMBLED_DEPTHS, "SCRAMBLED_17")
        for size in SCRAMBLED_GRID_SIZES
    ]
    scrambled = scrambled_runs[-1]
    scrambled_reference_boundaries = {
        item["depth"]: item["boundary"]
        for item in scrambled["checkpoints"]
    }
    scrambled_cross_grid_max = 0.0
    for item in scrambled_runs[0]["checkpoints"]:
        scrambled_cross_grid_max = max(
            scrambled_cross_grid_max,
            boundary_distance(
                item["boundary"],
                scrambled_reference_boundaries[item["depth"]],
            ),
        )
    if scrambled_cross_grid_max > CROSS_GRID_TOLERANCE:
        fail("continuous kicked-phase scrambled grid convergence failed")
    periodic_radii = [
        item["epsilon_fourier_radius"]
        for item in reference_periodic["checkpoints"]
    ]
    scrambled_radii = [
        item["epsilon_fourier_radius"]
        for item in scrambled["checkpoints"]
    ]
    localized_window = periodic_radii[6:]
    if max(localized_window) - min(localized_window) > 2:
        fail("continuous kicked-phase localization window failed")
    if scrambled_radii[-1] <= 20 * max(localized_window):
        fail("continuous kicked-phase scrambled growth control failed")
    for run in periodic_runs + scrambled_runs:
        for item in run["checkpoints"]:
            if abs(float(item["boundary"]["norm"]) - 1.0) > NORM_TOLERANCE:
                fail("continuous kicked-phase norm drift failed")

    extended = high_precision_replay(512, 2048)
    double_extended_reference = periodic_runs[1]["checkpoints"][-1]
    extended_boundary = {
        "loschmidt_amplitude": extended["loschmidt_amplitude"],
        "cosine_moment": extended["cosine_moment"],
        "sine_moment": extended["sine_moment"],
        "norm": extended["norm"],
    }
    extended_boundary_error = boundary_distance(
        extended_boundary, double_extended_reference["boundary"]
    )
    if extended_boundary_error > CROSS_GRID_TOLERANCE:
        fail("continuous kicked-phase extended precision mismatch")
    if (
        extended["epsilon_sweep_radii"]
        != periodic_runs[1]["final_epsilon_sweep_radii"]
    ):
        fail("continuous kicked-phase precision radius mismatch")

    accepted_carrier = zero_momentum_carrier(2048)
    primary = transaction(
        accepted_carrier,
        2048,
        "PERIODIC",
        K_PRIMARY,
        TAU_PRIMARY,
    )
    reuse = transaction(
        accepted_carrier,
        31,
        "REUSE",
        math.sqrt(3.0),
        math.sqrt(5.0),
    )
    repeated_errors: list[float] = []
    for _ in range(8):
        repeated = transaction(
            accepted_carrier,
            64,
            "PERIODIC",
            K_PRIMARY,
            TAU_PRIMARY,
        )
        repeated_errors.append(float(repeated["restoration_error"]))

    final_samples = zero_momentum_carrier(2048).samples
    theta, modes = topology(2048)
    for step in range(1, 2049):
        forward_step(
            final_samples,
            theta,
            modes,
            K_PRIMARY,
            TAU_PRIMARY,
            phase_offset(step, "PERIODIC"),
        )
    bessel = bessel_baseline(final_samples, 2048)
    if (
        float(bessel["total_l2_error_against_fft"])
        > BESSEL_L2_TOLERANCE
    ):
        fail("continuous kicked-phase Bessel baseline mismatch")

    snapshot_source = zero_momentum_carrier(2048)
    snapshot_image = snapshot_source.samples.copy()
    snapshot_working = snapshot_image.copy()
    for step in range(1, 2049):
        forward_step(
            snapshot_working,
            theta,
            modes,
            K_PRIMARY,
            TAU_PRIMARY,
            phase_offset(step, "PERIODIC"),
        )
    snapshot_boundary = boundary(snapshot_image, snapshot_working, theta)
    snapshot_working[:] = snapshot_image
    snapshot_primary_reload_error = float(
        np.max(np.abs(snapshot_working - snapshot_image))
    )
    for step in range(1, 32):
        forward_step(
            snapshot_working,
            theta,
            modes,
            math.sqrt(3.0),
            math.sqrt(5.0),
            phase_offset(step, "REUSE"),
        )
    snapshot_reuse_boundary = boundary(
        snapshot_image, snapshot_working, theta
    )
    snapshot_working[:] = snapshot_image
    snapshot_reuse_reload_error = float(
        np.max(np.abs(snapshot_working - snapshot_image))
    )
    snapshot_match = boundary_distance(
        snapshot_boundary, primary["boundary"]
    )
    if snapshot_match > CROSS_GRID_TOLERANCE:
        fail("continuous kicked-phase snapshot boundary mismatch")

    controls = inverse_controls(2048, 32)
    if not all(value > 1.0e-5 for value in controls.values()):
        fail("continuous kicked-phase negative control failed")

    fft_grid_size = 2048
    fft_state_bytes = fft_grid_size * 16
    fft_topology_bytes = 2 * fft_grid_size * 8
    fft_peak_step_temporary_bytes = 2 * fft_grid_size * 16
    fft_restoration_verification_copy_bytes = fft_state_bytes
    fft_per_step_temporary_allocation_payload_bytes = (
        4 * fft_grid_size * 16
    )
    fft_forward_conceptual_complex_operations = int(
        2048
        * (
            2 * fft_grid_size * math.log2(fft_grid_size)
            + 2 * fft_grid_size
        )
    )
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_NUMERICAL_CONTINUOUS_IRRATIONAL_KICKED_PHASE_"
            "COHERENT_FOURIER_LOCALIZATION_CONTRAST_WITH_EFFECTIVE_"
            "BANDWIDTH_PLATEAU_ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "PERIODIC_K_SQRT2_TAU_SQRT3_GRIDS256_TO2048_"
            "DEPTH2048_EPSILON1E_MINUS12_SOFTWARE_FLOAT64"
        ),
        "predeclared_tolerances": {
            "epsilon_tail_energy": EPSILON_ENERGY,
            "restoration_linf": RESTORATION_TOLERANCE,
            "norm_l2": NORM_TOLERANCE,
            "cross_grid_boundary": CROSS_GRID_TOLERANCE,
            "sparse_bessel_l2": BESSEL_L2_TOLERANCE,
        },
        "periodic_grid_runs": periodic_runs,
        "periodic_cross_grid_boundary_max_error": cross_grid_max,
        "extended_precision_replay": extended,
        "extended_precision_boundary_error": extended_boundary_error,
        "periodic_localization_window_depths": list(
            PERIODIC_DEPTHS[6:]
        ),
        "periodic_localization_window_radii": localized_window,
        "periodic_effective_fixed_bandwidth_observed": True,
        "scrambled_control_grid_runs": scrambled_runs,
        "scrambled_control_radii": scrambled_radii,
        "scrambled_cross_grid_boundary_max_error": (
            scrambled_cross_grid_max
        ),
        "scrambled_control_asymptotic_delocalization_established": False,
        "primary": primary,
        "reuse": reuse,
        "repeated_reuse_restoration_errors": repeated_errors,
        "maximum_repeated_reuse_restoration_error": max(
            repeated_errors
        ),
        "controls": controls,
        "snapshot_sham": {
            "boundary": snapshot_boundary,
            "boundary_match_error": snapshot_match,
            "snapshot_loaded": True,
            "actual_inverse_restoration": False,
            "restoration_generation": 0,
            "creation_payload_copy_bytes": fft_state_bytes,
            "execution_load_payload_copy_bytes": fft_state_bytes,
            "restoration_reload_payload_copy_bytes": fft_state_bytes,
            "reuse_restoration_reload_payload_copy_bytes": (
                fft_state_bytes
            ),
            "total_payload_copy_bytes": 4 * fft_state_bytes,
            "primary_reload_error": snapshot_primary_reload_error,
            "reuse_boundary": snapshot_reuse_boundary,
            "reuse_consumed_reloaded_execution_carrier": True,
            "reuse_reload_error": snapshot_reuse_reload_error,
        },
        "fft_phase_path_resources": {
            "resident_state_bytes": fft_state_bytes,
            "public_topology_bytes": fft_topology_bytes,
            "compiled_scalar_program_bytes": 32,
            "peak_step_temporary_payload_bytes": (
                fft_peak_step_temporary_bytes
            ),
            "restoration_verification_copy_bytes": (
                fft_restoration_verification_copy_bytes
            ),
            "forward_temporary_allocation_payload_bytes": (
                2048
                * fft_per_step_temporary_allocation_payload_bytes
            ),
            "inverse_temporary_allocation_payload_bytes": (
                2048
                * fft_per_step_temporary_allocation_payload_bytes
            ),
            "resource_accounting_kind": (
                "NUMPY_ARRAY_PAYLOAD_NOT_PYTHON_ALLOCATOR_TRAFFIC"
            ),
            "retained_inverse_history_bytes": 0,
            "forward_conceptual_complex_operations": (
                fft_forward_conceptual_complex_operations
            ),
            "inverse_conceptual_complex_operations": (
                fft_forward_conceptual_complex_operations
            ),
            "projection_complex_operations": fft_grid_size,
            "restoration_verification_complex_operations": (
                fft_grid_size
            ),
            "total_lifecycle_conceptual_complex_operations": (
                2 * fft_forward_conceptual_complex_operations
                + 2 * fft_grid_size
            ),
            "controller_backend_traffic_bytes": 0,
        },
        "matched_sparse_bessel_baseline": bessel,
        "warm_runtime": warm_runtime(2048, 256, 5),
        "exact_finite_fourier_support": False,
        "infinite_bessel_support_of_generic_kick": True,
        "phase_is_primitive_wave_amplitude": True,
        "unresolved_phase_preserved_across_composition": True,
        "machine_enforced_hidden_intermediate": False,
        "distinct_phase_resource_established": False,
        "asymptotic_dynamical_localization_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
        "physical_waveform_execution": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(2) from error
