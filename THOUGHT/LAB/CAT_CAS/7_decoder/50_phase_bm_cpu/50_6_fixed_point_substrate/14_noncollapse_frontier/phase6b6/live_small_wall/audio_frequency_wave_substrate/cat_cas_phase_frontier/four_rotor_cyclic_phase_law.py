#!/usr/bin/env python3
"""Exact finite-torus four-rotor phase law with history-free inversion."""

from __future__ import annotations

import json
import math
import resource
import time
from dataclasses import dataclass

import numpy as np
from scipy import fft as scipy_fft


ROTORS = 4
GRID_SIZE = 17
PRIMARY_DEPTH = 32
PRIMARY_KICK = math.sqrt(2.0)
PRIMARY_COUPLING = math.sqrt(5.0) / 7.0
PRIMARY_FREE_TIME = math.sqrt(3.0)
REUSE_DEPTH = 11
REUSE_KICK = math.sqrt(7.0) / 3.0
REUSE_COUPLING = math.sqrt(11.0) / 13.0
REUSE_FREE_TIME = math.sqrt(13.0)
RESTORATION_L2_TOLERANCE = 2.0e-11
BOUNDARY_TOLERANCE = 2.0e-11
CONTROL_FLOOR = 1.0e-4
DEPTH_SWEEP = (1, 2, 4, 8, 16, 32, 64)


def fail(message: str) -> None:
    raise RuntimeError(message)


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


@dataclass
class Carrier:
    samples: np.ndarray
    generation: int = 0


@dataclass(frozen=True)
class Plan:
    theta: np.ndarray
    modes: np.ndarray
    free_factor: np.ndarray
    retained_payload_bytes: int
    compilation_peak_payload_bytes: int


@dataclass
class Stats:
    forward_steps: int = 0
    inverse_steps: int = 0
    onsite_phase_updates: int = 0
    pair_phase_updates: int = 0
    free_phase_updates: int = 0
    fft_transforms: int = 0
    phase_factor_rematerializations: int = 0
    rematerialized_phase_factor_cells: int = 0
    maximum_phase_factor_cells: int = 0
    maximum_accounted_engine_array_bytes: int = 0
    maximum_accounted_wrapper_array_bytes: int = 0
    fft_backing_buffer_preserved: bool = True
    verification_baseline_bytes: int = 0


def zero_momentum_carrier(grid_size: int) -> Carrier:
    cell_count = grid_size**ROTORS
    samples = np.full(
        (grid_size,) * ROTORS,
        1.0 / math.sqrt(cell_count),
        dtype=np.complex128,
    )
    return Carrier(samples)


def compile_plan(grid_size: int, free_time: float) -> Plan:
    theta = (
        np.arange(grid_size, dtype=np.float64)
        * (2.0 * math.pi / grid_size)
    )
    modes = np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    free_argument = np.empty(grid_size, dtype=np.float64)
    np.square(modes, out=free_argument)
    free_argument *= -0.5 * free_time
    free_factor = np.empty(grid_size, dtype=np.complex128)
    free_factor.real = 0.0
    free_factor.imag = free_argument
    np.exp(free_factor, out=free_factor)
    retained = theta.nbytes + modes.nbytes + free_factor.nbytes
    compilation_peak = retained + free_argument.nbytes
    return Plan(
        theta,
        modes,
        free_factor,
        retained,
        compilation_peak,
    )


def account(
    carrier: Carrier,
    plan: Plan,
    stats: Stats,
    *arrays: np.ndarray,
) -> None:
    allocations: dict[int, np.ndarray] = {}
    for array in arrays:
        allocation = array
        while isinstance(allocation.base, np.ndarray):
            allocation = allocation.base
        allocations[id(allocation)] = allocation
    transient_bytes = sum(
        allocation.nbytes for allocation in allocations.values()
    )
    engine_bytes = (
        carrier.samples.nbytes
        + max(
            plan.retained_payload_bytes + transient_bytes,
            plan.compilation_peak_payload_bytes,
        )
    )
    stats.maximum_accounted_engine_array_bytes = max(
        stats.maximum_accounted_engine_array_bytes,
        engine_bytes,
    )
    stats.maximum_accounted_wrapper_array_bytes = max(
        stats.maximum_accounted_wrapper_array_bytes,
        engine_bytes + stats.verification_baseline_bytes,
    )


def public_offset(step: int, program_tag: int, channel: int) -> float:
    residue = (
        step * step
        + (2 * channel + 3) * step
        + 7 * program_tag
        + channel * channel
    ) % 31
    return 2.0 * math.pi * residue / 31.0


def apply_potential(
    carrier: Carrier,
    plan: Plan,
    step: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    conjugate: bool,
    stats: Stats,
) -> None:
    sign = 1.0 if conjugate else -1.0
    grid_size = len(plan.theta)
    for rotor in range(ROTORS):
        offset = public_offset(step, program_tag, rotor)
        phase_argument = np.empty(grid_size, dtype=np.float64)
        np.add(plan.theta, offset, out=phase_argument)
        np.cos(phase_argument, out=phase_argument)
        phase_argument *= sign * kick_strength
        factor = np.empty(grid_size, dtype=np.complex128)
        factor.real = 0.0
        factor.imag = phase_argument
        np.exp(factor, out=factor)
        shape = [1] * ROTORS
        shape[rotor] = grid_size
        carrier.samples *= factor.reshape(shape)
        stats.onsite_phase_updates += 1
        stats.phase_factor_rematerializations += 1
        stats.rematerialized_phase_factor_cells += int(factor.size)
        stats.maximum_phase_factor_cells = max(
            stats.maximum_phase_factor_cells, int(factor.size)
        )
        account(carrier, plan, stats, phase_argument, factor)
    for edge in range(ROTORS - 1):
        offset = public_offset(step, program_tag, ROTORS + edge)
        phase_argument = np.empty(
            (grid_size, grid_size), dtype=np.float64
        )
        np.subtract(
            plan.theta[:, np.newaxis],
            plan.theta[np.newaxis, :],
            out=phase_argument,
        )
        phase_argument += offset
        np.cos(phase_argument, out=phase_argument)
        phase_argument *= sign * coupling_strength
        factor = np.empty(
            (grid_size, grid_size), dtype=np.complex128
        )
        factor.real = 0.0
        factor.imag = phase_argument
        np.exp(factor, out=factor)
        shape = [1] * ROTORS
        shape[edge] = grid_size
        shape[edge + 1] = grid_size
        carrier.samples *= factor.reshape(shape)
        stats.pair_phase_updates += 1
        stats.phase_factor_rematerializations += 1
        stats.rematerialized_phase_factor_cells += int(factor.size)
        stats.maximum_phase_factor_cells = max(
            stats.maximum_phase_factor_cells, int(factor.size)
        )
        account(carrier, plan, stats, phase_argument, factor)


def apply_free(
    carrier: Carrier,
    plan: Plan,
    conjugate: bool,
    stats: Stats,
) -> None:
    address_before = carrier.samples.__array_interface__["data"][0]
    transformed = scipy_fft.fftn(
        carrier.samples,
        norm="ortho",
        overwrite_x=True,
        workers=1,
    )
    if not np.shares_memory(carrier.samples, transformed):
        stats.fft_backing_buffer_preserved = False
    carrier.samples = transformed
    if conjugate:
        factor = np.empty_like(plan.free_factor)
        np.conjugate(plan.free_factor, out=factor)
    else:
        factor = plan.free_factor
    for rotor in range(ROTORS):
        shape = [1] * ROTORS
        shape[rotor] = len(factor)
        carrier.samples *= factor.reshape(shape)
        stats.free_phase_updates += 1
    transformed = scipy_fft.ifftn(
        carrier.samples,
        norm="ortho",
        overwrite_x=True,
        workers=1,
    )
    if not np.shares_memory(carrier.samples, transformed):
        stats.fft_backing_buffer_preserved = False
    carrier.samples = transformed
    address_after = carrier.samples.__array_interface__["data"][0]
    if address_before != address_after:
        stats.fft_backing_buffer_preserved = False
    stats.fft_transforms += 2
    account(carrier, plan, stats, factor)


def forward_step(
    carrier: Carrier,
    plan: Plan,
    step: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    stats: Stats,
) -> None:
    apply_potential(
        carrier,
        plan,
        step,
        program_tag,
        kick_strength,
        coupling_strength,
        False,
        stats,
    )
    apply_free(carrier, plan, False, stats)
    stats.forward_steps += 1


def inverse_step(
    carrier: Carrier,
    plan: Plan,
    step: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    stats: Stats,
) -> None:
    apply_free(carrier, plan, True, stats)
    apply_potential(
        carrier,
        plan,
        step,
        program_tag,
        kick_strength,
        coupling_strength,
        True,
        stats,
    )
    stats.inverse_steps += 1


def boundary(
    carrier: Carrier, plan: Plan, stats: Stats
) -> dict[str, object]:
    probability = np.empty(carrier.samples.shape, dtype=np.float64)
    np.absolute(carrier.samples, out=probability)
    np.square(probability, out=probability)
    uniform_overlap = complex(
        np.sum(carrier.samples) / math.sqrt(carrier.samples.size)
    )
    local_marginal = np.sum(probability, axis=(1, 2, 3))
    local_kernel = np.empty(len(plan.theta), dtype=np.float64)
    np.cos(plan.theta, out=local_kernel)
    account(
        carrier,
        plan,
        stats,
        probability,
        local_marginal,
        local_kernel,
    )
    local_cosine = float(np.dot(local_marginal, local_kernel))
    del local_marginal, local_kernel
    pair_marginal = np.sum(probability, axis=(2, 3))
    pair_kernel = np.empty(pair_marginal.shape, dtype=np.float64)
    np.subtract(
        plan.theta[:, np.newaxis],
        plan.theta[np.newaxis, :],
        out=pair_kernel,
    )
    np.cos(pair_kernel, out=pair_kernel)
    account(
        carrier,
        plan,
        stats,
        probability,
        pair_marginal,
        pair_kernel,
    )
    np.multiply(pair_marginal, pair_kernel, out=pair_marginal)
    pair_cosine = float(np.sum(pair_marginal))
    return {
        "zero_momentum_amplitude": complex_pair(uniform_overlap),
        "local_cosine": local_cosine,
        "neighbor_phase_correlation": pair_cosine,
        "norm": float(np.linalg.norm(carrier.samples)),
    }


def boundary_distance(
    left: dict[str, object], right: dict[str, object]
) -> float:
    left_amplitude = complex(*left["zero_momentum_amplitude"])
    right_amplitude = complex(*right["zero_momentum_amplitude"])
    return max(
        abs(left_amplitude - right_amplitude),
        abs(float(left["local_cosine"]) - float(right["local_cosine"])),
        abs(
            float(left["neighbor_phase_correlation"])
            - float(right["neighbor_phase_correlation"])
        ),
        abs(float(left["norm"]) - float(right["norm"])),
    )


def transaction(
    carrier: Carrier,
    depth: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    baseline = carrier.samples.copy()
    baseline_address = baseline.__array_interface__["data"][0]
    carrier_address = carrier.samples.__array_interface__["data"][0]
    if baseline_address == carrier_address:
        fail("verification baseline aliases the carrier")
    plan = compile_plan(len(carrier.samples), free_time)
    stats = Stats(verification_baseline_bytes=baseline.nbytes)
    account(carrier, plan, stats)
    start = time.perf_counter_ns()
    for step in range(1, depth + 1):
        forward_step(
            carrier,
            plan,
            step,
            program_tag,
            kick_strength,
            coupling_strength,
            stats,
        )
    latched = boundary(carrier, plan, stats)
    for step in range(depth, 0, -1):
        inverse_step(
            carrier,
            plan,
            step,
            program_tag,
            kick_strength,
            coupling_strength,
            stats,
        )
    restoration_error = float(np.linalg.norm(carrier.samples - baseline))
    if restoration_error > RESTORATION_L2_TOLERANCE:
        fail("cyclic phase-law restoration failed")
    if carrier.samples.__array_interface__["data"][0] != carrier_address:
        fail("cyclic phase-law carrier backing changed")
    carrier.generation += 1
    elapsed = time.perf_counter_ns() - start
    wrapper_peak = stats.maximum_accounted_wrapper_array_bytes
    return {
        "boundary": latched,
        "restoration_error": restoration_error,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse_ready": True,
        "verification_baseline_bytes": baseline.nbytes,
        "verification_baseline_reload_count": 0,
        "verification_baseline_used_for_restoration": False,
        "resources": {
            "carrier_complex_cells": int(carrier.samples.size),
            "carrier_payload_bytes": carrier.samples.nbytes,
            "retained_public_plan_bytes": plan.retained_payload_bytes,
            "plan_compilation_peak_payload_bytes": (
                plan.compilation_peak_payload_bytes
            ),
            "maximum_phase_factor_cells": stats.maximum_phase_factor_cells,
            "maximum_accounted_engine_array_bytes": (
                stats.maximum_accounted_engine_array_bytes
            ),
            "wrapper_accounted_peak_array_bytes": wrapper_peak,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "forward_steps": stats.forward_steps,
            "inverse_steps": stats.inverse_steps,
            "onsite_phase_updates": stats.onsite_phase_updates,
            "pair_phase_updates": stats.pair_phase_updates,
            "free_phase_updates": stats.free_phase_updates,
            "fft_transforms": stats.fft_transforms,
            "phase_factor_rematerializations": (
                stats.phase_factor_rematerializations
            ),
            "rematerialized_phase_factor_cells": (
                stats.rematerialized_phase_factor_cells
            ),
            "retained_inverse_history_bytes": 0,
            "dense_operator_materialization_bytes": 0,
            "decoded_intermediate_bytes": 0,
            "fft_backing_buffer_preserved": (
                stats.fft_backing_buffer_preserved
            ),
            "elapsed_ns": elapsed,
        },
    }


def forward_only(
    depth: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    carrier = zero_momentum_carrier(GRID_SIZE)
    plan = compile_plan(GRID_SIZE, free_time)
    stats = Stats()
    start = time.perf_counter_ns()
    for step in range(1, depth + 1):
        forward_step(
            carrier,
            plan,
            step,
            program_tag,
            kick_strength,
            coupling_strength,
            stats,
        )
    elapsed = time.perf_counter_ns() - start
    return {
        "boundary": boundary(carrier, plan, stats),
        "elapsed_ns": elapsed,
        "carrier_payload_bytes": carrier.samples.nbytes,
        "maximum_accounted_engine_array_bytes": (
            stats.maximum_accounted_engine_array_bytes
        ),
        "matched_algorithm": "DIRECT_CYCLIC_FFT_PHASE_ARRAY",
    }


def controls() -> dict[str, float]:
    initial = zero_momentum_carrier(GRID_SIZE)
    plan = compile_plan(GRID_SIZE, PRIMARY_FREE_TIME)
    forward = Carrier(initial.samples.copy())
    forward_step(
        forward,
        plan,
        1,
        1,
        PRIMARY_KICK,
        PRIMARY_COUPLING,
        Stats(),
    )
    missing = float(np.linalg.norm(forward.samples - initial.samples))

    wrong = Carrier(forward.samples.copy())
    inverse_step(
        wrong,
        plan,
        1,
        1,
        PRIMARY_KICK,
        PRIMARY_COUPLING * 1.1,
        Stats(),
    )

    reordered = Carrier(forward.samples.copy())
    reordered_stats = Stats()
    apply_potential(
        reordered,
        plan,
        1,
        1,
        PRIMARY_KICK,
        PRIMARY_COUPLING,
        True,
        reordered_stats,
    )
    apply_free(reordered, plan, True, reordered_stats)
    return {
        "missing_inverse_error": missing,
        "wrong_inverse_error": float(
            np.linalg.norm(wrong.samples - initial.samples)
        ),
        "reordered_inverse_error": float(
            np.linalg.norm(reordered.samples - initial.samples)
        ),
    }


def depth_growth() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for depth in DEPTH_SWEEP:
        result = transaction(
            zero_momentum_carrier(GRID_SIZE),
            depth,
            1,
            PRIMARY_KICK,
            PRIMARY_COUPLING,
            PRIMARY_FREE_TIME,
        )
        resources = result["resources"]
        assert isinstance(resources, dict)
        rows.append(
            {
                "depth": depth,
                "restoration_error": result["restoration_error"],
                "carrier_payload_bytes": resources[
                    "carrier_payload_bytes"
                ],
                "maximum_accounted_engine_array_bytes": resources[
                    "maximum_accounted_engine_array_bytes"
                ],
                "wrapper_accounted_peak_array_bytes": resources[
                    "wrapper_accounted_peak_array_bytes"
                ],
                "retained_inverse_history_bytes": resources[
                    "retained_inverse_history_bytes"
                ],
                "fft_backing_buffer_preserved": resources[
                    "fft_backing_buffer_preserved"
                ],
            }
        )
    return rows


def main() -> None:
    carrier = zero_momentum_carrier(GRID_SIZE)
    primary = transaction(
        carrier,
        PRIMARY_DEPTH,
        1,
        PRIMARY_KICK,
        PRIMARY_COUPLING,
        PRIMARY_FREE_TIME,
    )
    direct = forward_only(
        PRIMARY_DEPTH,
        1,
        PRIMARY_KICK,
        PRIMARY_COUPLING,
        PRIMARY_FREE_TIME,
    )
    direct_boundary_error = boundary_distance(
        primary["boundary"], direct["boundary"]
    )
    reuse = transaction(
        carrier,
        REUSE_DEPTH,
        2,
        REUSE_KICK,
        REUSE_COUPLING,
        REUSE_FREE_TIME,
    )
    fresh_carrier = zero_momentum_carrier(GRID_SIZE)
    fresh = transaction(
        fresh_carrier,
        REUSE_DEPTH,
        2,
        REUSE_KICK,
        REUSE_COUPLING,
        REUSE_FREE_TIME,
    )
    fresh_restored_boundary_error = boundary_distance(
        reuse["boundary"], fresh["boundary"]
    )
    growth = depth_growth()
    memory_signature = {
        (
            row["carrier_payload_bytes"],
            row["maximum_accounted_engine_array_bytes"],
            row["wrapper_accounted_peak_array_bytes"],
            row["retained_inverse_history_bytes"],
        )
        for row in growth
    }
    control = controls()
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_DENSE_FINITE_TORUS_CYCLIC_PHASE_UPDATE_LAW_"
            "WITH_DEPTH_INDEPENDENT_EXPLICIT_NUMPY_ARRAY_PAYLOAD_"
            "ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTORS_GRID17_DEPTHS_1_TO_64_"
            "SCIPY_POCKETFFT_SOFTWARE_COMPLEX128_ARRAY_PAYLOAD_"
            "ACCOUNTING_NO_FFT_INTERNAL_WORKSPACE_BOUND"
        ),
        "phase_primitive": (
            "U1_ANGLE_PHASE_MULTIPLICATION_AND_ORTHONORMAL_FOURIER_"
            "MOMENTUM_PHASE_ROTATION"
        ),
        "primary": primary,
        "matched_direct_baseline": direct,
        "matched_direct_boundary_error": direct_boundary_error,
        "reuse": reuse,
        "fresh_reuse": fresh,
        "fresh_restored_boundary_error": (
            fresh_restored_boundary_error
        ),
        "depth_growth": growth,
        "depth_independent_memory_signature": len(memory_signature) == 1,
        "controls": control,
        "dense_cyclic_path_avoids_sector_inverse_work": {
            "sector_rhs_rematerializations": 0,
            "gram_rematerializations": 0,
            "inverse_sector_closures": 0,
            "retained_inverse_history_bytes": 0,
        },
        "matched_classical_cyclic_fft_identical": True,
        "compact_tt_advantage_established": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
        "physical_waveform_execution": False,
        "terminal": False,
    }
    if (
        primary["restoration_error"] > RESTORATION_L2_TOLERANCE
        or reuse["restoration_generation"] != 2
        or fresh_restored_boundary_error > BOUNDARY_TOLERANCE
        or direct_boundary_error > BOUNDARY_TOLERANCE
        or len(memory_signature) != 1
        or not all(
            row["fft_backing_buffer_preserved"] for row in growth
        )
        or min(control.values()) <= CONTROL_FLOOR
    ):
        fail("cyclic phase-law qualification failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(2) from error
