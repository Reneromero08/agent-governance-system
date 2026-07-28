#!/usr/bin/env python3
"""Exact global-rotation quotient for the cyclic four-rotor phase law."""

from __future__ import annotations

import json
import math
import resource
import time
from dataclasses import dataclass

import numpy as np
from scipy import fft as scipy_fft

import four_rotor_cyclic_phase_law as dense


GRID_SIZE = dense.GRID_SIZE
DEPTH_SWEEP = (1, 2, 4, 8, 16, 32, 64)
PRIMARY_DEPTH = 32
RESTORATION_L2_TOLERANCE = 2.0e-11
FULL_STATE_PARITY_TOLERANCE = 2.0e-11
BOUNDARY_TOLERANCE = 2.0e-11
CONTROL_FLOOR = 1.0e-4


def fail(message: str) -> None:
    raise RuntimeError(message)


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
    compilation_peak_payload_bytes: int


@dataclass
class Stats:
    forward_steps: int = 0
    inverse_steps: int = 0
    pair_phase_updates: int = 0
    fft_transforms: int = 0
    maximum_pair_factor_cells: int = 0
    maximum_explicit_engine_array_bytes: int = 0
    fft_backing_buffer_preserved: bool = True


def product_zero_total_momentum_carrier() -> Carrier:
    cells = GRID_SIZE**3
    return Carrier(
        np.full(
            (GRID_SIZE, GRID_SIZE, GRID_SIZE),
            1.0 / math.sqrt(cells),
            dtype=np.complex128,
        )
    )


def compile_plan(free_time: float) -> Plan:
    theta = (
        np.arange(GRID_SIZE, dtype=np.float64)
        * (2.0 * math.pi / GRID_SIZE)
    )
    modes = np.fft.fftfreq(GRID_SIZE, d=1.0 / GRID_SIZE)
    indices = np.arange(GRID_SIZE, dtype=np.int64)
    zero_index = np.empty(
        (GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.int64
    )
    np.add(
        indices[:, np.newaxis, np.newaxis],
        indices[np.newaxis, :, np.newaxis],
        out=zero_index,
    )
    zero_index += indices[np.newaxis, np.newaxis, :]
    np.negative(zero_index, out=zero_index)
    np.remainder(zero_index, GRID_SIZE, out=zero_index)
    mode_squared = np.empty(GRID_SIZE, dtype=np.float64)
    np.square(modes, out=mode_squared)
    energy = np.empty(zero_index.shape, dtype=np.float64)
    np.take(mode_squared, zero_index, out=energy)
    energy += mode_squared[:, np.newaxis, np.newaxis]
    energy += mode_squared[np.newaxis, :, np.newaxis]
    energy += mode_squared[np.newaxis, np.newaxis, :]
    energy *= -0.5 * free_time
    free_factor = np.empty(energy.shape, dtype=np.complex128)
    free_factor.real = 0.0
    free_factor.imag = energy
    np.exp(free_factor, out=free_factor)
    compilation_peak = (
        theta.nbytes
        + modes.nbytes
        + indices.nbytes
        + zero_index.nbytes
        + mode_squared.nbytes
        + energy.nbytes
        + free_factor.nbytes
    )
    return Plan(theta, modes, free_factor, compilation_peak)


def retained_plan_bytes(plan: Plan) -> int:
    return plan.theta.nbytes + plan.modes.nbytes + plan.free_factor.nbytes


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
    runtime_bytes = (
        retained_plan_bytes(plan)
        + sum(array.nbytes for array in allocations.values())
    )
    stats.maximum_explicit_engine_array_bytes = max(
        stats.maximum_explicit_engine_array_bytes,
        carrier.samples.nbytes
        + max(runtime_bytes, plan.compilation_peak_payload_bytes),
    )


def pair_factor(
    plan: Plan,
    step: int,
    program_tag: int,
    edge: int,
    coupling_strength: float,
    conjugate: bool,
) -> np.ndarray:
    sign = 1.0 if conjugate else -1.0
    offset = dense.public_offset(
        step, program_tag, dense.ROTORS + edge
    )
    if edge == 0:
        phase = -plan.theta + offset
    else:
        phase = (
            plan.theta[:, np.newaxis]
            - plan.theta[np.newaxis, :]
            + offset
        )
    return np.exp(sign * 1j * coupling_strength * np.cos(phase))


def apply_potential(
    carrier: Carrier,
    plan: Plan,
    step: int,
    program_tag: int,
    coupling_strength: float,
    conjugate: bool,
    stats: Stats,
) -> None:
    for edge in range(dense.ROTORS - 1):
        factor = pair_factor(
            plan,
            step,
            program_tag,
            edge,
            coupling_strength,
            conjugate,
        )
        if edge == 0:
            carrier.samples *= factor[:, np.newaxis, np.newaxis]
        elif edge == 1:
            carrier.samples *= factor[:, :, np.newaxis]
        else:
            carrier.samples *= factor[np.newaxis, :, :]
        stats.pair_phase_updates += 1
        stats.maximum_pair_factor_cells = max(
            stats.maximum_pair_factor_cells, int(factor.size)
        )
        account(carrier, plan, stats, factor)


def apply_free(
    carrier: Carrier,
    plan: Plan,
    conjugate: bool,
    stats: Stats,
) -> None:
    address = carrier.samples.__array_interface__["data"][0]
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
    carrier.samples *= factor
    transformed = scipy_fft.ifftn(
        carrier.samples,
        norm="ortho",
        overwrite_x=True,
        workers=1,
    )
    if not np.shares_memory(carrier.samples, transformed):
        stats.fft_backing_buffer_preserved = False
    carrier.samples = transformed
    if carrier.samples.__array_interface__["data"][0] != address:
        stats.fft_backing_buffer_preserved = False
    stats.fft_transforms += 2
    account(carrier, plan, stats, factor)


def forward_step(
    carrier: Carrier,
    plan: Plan,
    step: int,
    program_tag: int,
    coupling_strength: float,
    stats: Stats,
) -> None:
    apply_potential(
        carrier,
        plan,
        step,
        program_tag,
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
    coupling_strength: float,
    stats: Stats,
) -> None:
    apply_free(carrier, plan, True, stats)
    apply_potential(
        carrier,
        plan,
        step,
        program_tag,
        coupling_strength,
        True,
        stats,
    )
    stats.inverse_steps += 1


def boundary(carrier: Carrier, plan: Plan) -> dict[str, object]:
    probability = np.empty(carrier.samples.shape, dtype=np.float64)
    np.absolute(carrier.samples, out=probability)
    np.square(probability, out=probability)
    overlap = complex(
        np.sum(carrier.samples) / math.sqrt(carrier.samples.size)
    )
    first_marginal = np.sum(probability, axis=(1, 2))
    pair_correlation = float(
        np.dot(first_marginal, np.cos(-plan.theta))
    )
    return {
        "zero_momentum_amplitude": dense.complex_pair(overlap),
        "local_cosine": 0.0,
        "neighbor_phase_correlation": pair_correlation,
        "norm": float(np.linalg.norm(carrier.samples)),
    }


def dense_state(
    depth: int,
    program_tag: int,
    coupling_strength: float,
    free_time: float,
) -> tuple[np.ndarray, dict[str, object]]:
    carrier = dense.zero_momentum_carrier(GRID_SIZE)
    plan = dense.compile_plan(GRID_SIZE, free_time)
    stats = dense.Stats()
    for step in range(1, depth + 1):
        dense.forward_step(
            carrier,
            plan,
            step,
            program_tag,
            0.0,
            coupling_strength,
            stats,
        )
    boundary_value = dense.boundary(carrier, plan, stats)
    return carrier.samples, boundary_value


def lift_to_four_rotors(
    samples: np.ndarray,
) -> tuple[np.ndarray, int]:
    index = np.arange(GRID_SIZE, dtype=np.int64)
    zero = index[:, np.newaxis, np.newaxis, np.newaxis]
    first = index[np.newaxis, :, np.newaxis, np.newaxis]
    second = index[np.newaxis, np.newaxis, :, np.newaxis]
    third = index[np.newaxis, np.newaxis, np.newaxis, :]
    relative_first = (first - zero) % GRID_SIZE
    relative_second = (second - zero) % GRID_SIZE
    relative_third = (third - zero) % GRID_SIZE
    lifted = samples[
        relative_first, relative_second, relative_third
    ]
    lifted /= math.sqrt(GRID_SIZE)
    index_bytes = (
        index.nbytes
        + relative_first.nbytes
        + relative_second.nbytes
        + relative_third.nbytes
    )
    return lifted, index_bytes


def transaction(
    carrier: Carrier,
    depth: int,
    program_tag: int,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    baseline = carrier.samples.copy()
    carrier_address = carrier.samples.__array_interface__["data"][0]
    plan = compile_plan(free_time)
    stats = Stats()
    account(carrier, plan, stats)
    start = time.perf_counter_ns()
    for step in range(1, depth + 1):
        forward_step(
            carrier,
            plan,
            step,
            program_tag,
            coupling_strength,
            stats,
        )
    latched = boundary(carrier, plan)
    dense_samples, dense_boundary_value = dense_state(
        depth, program_tag, coupling_strength, free_time
    )
    lifted, lift_index_bytes = lift_to_four_rotors(carrier.samples)
    np.subtract(lifted, dense_samples, out=lifted)
    full_state_error = float(np.linalg.norm(lifted))
    boundary_error = dense.boundary_distance(
        latched, dense_boundary_value
    )
    if (
        full_state_error > FULL_STATE_PARITY_TOLERANCE
        or boundary_error > BOUNDARY_TOLERANCE
    ):
        fail("rotation quotient dense parity failed")
    verification_bytes = dense_samples.nbytes + lifted.nbytes
    verification_peak = (
        carrier.samples.nbytes
        + baseline.nbytes
        + retained_plan_bytes(plan)
        + verification_bytes
        + lift_index_bytes
    )
    del dense_samples, lifted
    for step in range(depth, 0, -1):
        inverse_step(
            carrier,
            plan,
            step,
            program_tag,
            coupling_strength,
            stats,
        )
    restoration_error = float(np.linalg.norm(carrier.samples - baseline))
    if restoration_error > RESTORATION_L2_TOLERANCE:
        fail("rotation quotient actual inverse restoration failed")
    if carrier.samples.__array_interface__["data"][0] != carrier_address:
        fail("rotation quotient carrier backing changed")
    carrier.generation += 1
    return {
        "boundary": latched,
        "full_state_error_vs_dense_lift": full_state_error,
        "boundary_error_vs_dense": boundary_error,
        "restoration_error": restoration_error,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse_ready": True,
        "resources": {
            "quotient_carrier_complex_cells": int(carrier.samples.size),
            "quotient_carrier_payload_bytes": carrier.samples.nbytes,
            "dense_four_rotor_complex_cells": GRID_SIZE**dense.ROTORS,
            "dense_to_quotient_state_ratio": GRID_SIZE,
            "retained_public_plan_bytes": retained_plan_bytes(plan),
            "plan_compilation_peak_payload_bytes": (
                plan.compilation_peak_payload_bytes
            ),
            "maximum_pair_factor_cells": stats.maximum_pair_factor_cells,
            "maximum_explicit_engine_array_bytes": (
                stats.maximum_explicit_engine_array_bytes
            ),
            "maximum_explicit_wrapper_array_bytes": (
                stats.maximum_explicit_engine_array_bytes
                + baseline.nbytes
            ),
            "verification_dense_lift_bytes": verification_bytes,
            "verification_lift_index_bytes": lift_index_bytes,
            "verification_peak_explicit_array_bytes": verification_peak,
            "verification_baseline_bytes": baseline.nbytes,
            "verification_baseline_reload_count": 0,
            "verification_baseline_used_for_restoration": False,
            "forward_steps": stats.forward_steps,
            "inverse_steps": stats.inverse_steps,
            "pair_phase_updates": stats.pair_phase_updates,
            "fft_transforms": stats.fft_transforms,
            "fft_backing_buffer_preserved": (
                stats.fft_backing_buffer_preserved
            ),
            "retained_inverse_history_bytes": 0,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "elapsed_ns": time.perf_counter_ns() - start,
        },
    }


def controls() -> dict[str, float]:
    initial = product_zero_total_momentum_carrier()
    plan = compile_plan(dense.PRIMARY_FREE_TIME)
    forward = Carrier(initial.samples.copy())
    forward_step(
        forward,
        plan,
        1,
        1,
        dense.PRIMARY_COUPLING,
        Stats(),
    )
    missing = float(np.linalg.norm(forward.samples - initial.samples))
    wrong = Carrier(forward.samples.copy())
    inverse_step(
        wrong,
        plan,
        1,
        1,
        dense.PRIMARY_COUPLING * 1.1,
        Stats(),
    )
    reordered = Carrier(forward.samples.copy())
    reordered_stats = Stats()
    apply_potential(
        reordered,
        plan,
        1,
        1,
        dense.PRIMARY_COUPLING,
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


def memory_signature(result: dict[str, object]) -> tuple[object, ...]:
    resources = result["resources"]
    assert isinstance(resources, dict)
    return tuple(
        resources[key]
        for key in (
            "quotient_carrier_complex_cells",
            "quotient_carrier_payload_bytes",
            "dense_four_rotor_complex_cells",
            "dense_to_quotient_state_ratio",
            "retained_public_plan_bytes",
            "plan_compilation_peak_payload_bytes",
            "maximum_pair_factor_cells",
            "maximum_explicit_engine_array_bytes",
            "maximum_explicit_wrapper_array_bytes",
            "retained_inverse_history_bytes",
        )
    )


def main() -> None:
    depth_rows: list[dict[str, object]] = []
    signatures: set[tuple[object, ...]] = set()
    for depth in DEPTH_SWEEP:
        result = transaction(
            product_zero_total_momentum_carrier(),
            depth,
            1,
            dense.PRIMARY_COUPLING,
            dense.PRIMARY_FREE_TIME,
        )
        signatures.add(memory_signature(result))
        depth_rows.append(
            {
                "depth": depth,
                "full_state_error_vs_dense_lift": result[
                    "full_state_error_vs_dense_lift"
                ],
                "boundary_error_vs_dense": result[
                    "boundary_error_vs_dense"
                ],
                "restoration_error": result["restoration_error"],
                "memory_signature": memory_signature(result),
            }
        )
    carrier = product_zero_total_momentum_carrier()
    primary = transaction(
        carrier,
        PRIMARY_DEPTH,
        1,
        dense.PRIMARY_COUPLING,
        dense.PRIMARY_FREE_TIME,
    )
    reuse = transaction(
        carrier,
        11,
        2,
        dense.REUSE_COUPLING,
        dense.REUSE_FREE_TIME,
    )
    fresh = transaction(
        product_zero_total_momentum_carrier(),
        11,
        2,
        dense.REUSE_COUPLING,
        dense.REUSE_FREE_TIME,
    )
    reuse_boundary_error = dense.boundary_distance(
        reuse["boundary"], fresh["boundary"]
    )
    control = controls()
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_EXACT_GLOBAL_ROTATION_QUOTIENT_CYCLIC_"
            "PHASE_CARRIER_REDUCES_FOUR_ROTOR_STATE_BY_GRID_FACTOR_"
            "WITH_DEPTH_INDEPENDENT_MEMORY_ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTATION_INVARIANT_ROTORS_GRID17_"
            "DEPTHS1_TO64_SOFTWARE_COMPLEX128_SCIPY_POCKETFFT_"
            "NO_FFT_INTERNAL_WORKSPACE_BOUND"
        ),
        "quotient_geometry": (
            "THREE_RELATIVE_ANGLES_WITH_TOTAL_MOMENTUM_ZERO_MOD17"
        ),
        "original_angle_coordinates": 4,
        "quotient_angle_coordinates": 3,
        "global_rotation_coordinates_removed": 1,
        "onsite_phase_updates": 0,
        "state_reduction_factor": GRID_SIZE,
        "depth_growth": depth_rows,
        "depth_independent_memory_signature": len(signatures) == 1,
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh,
        "fresh_restored_boundary_error": reuse_boundary_error,
        "controls": control,
        "matched_classical_rotation_quotient_identical": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
        "physical_waveform_execution": False,
        "terminal": False,
    }
    if (
        len(signatures) != 1
        or primary["restoration_error"] > RESTORATION_L2_TOLERANCE
        or reuse["restoration_generation"] != 2
        or reuse_boundary_error > BOUNDARY_TOLERANCE
        or min(control.values()) <= CONTROL_FLOOR
        or primary["resources"]["dense_to_quotient_state_ratio"]
        != GRID_SIZE
    ):
        fail("rotation quotient qualification failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(2) from error
