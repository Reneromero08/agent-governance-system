#!/usr/bin/env python3
"""Stream an unprojected topology-derived total-momentum coordinate."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import fft as scipy_fft

import four_rotor_cyclic_phase_law as dense
import four_rotor_rotation_quotient_phase as quotient


GRID_SIZE = quotient.GRID_SIZE
DEPTH_SWEEP = quotient.DEPTH_SWEEP
PRIMARY_DEPTH = quotient.PRIMARY_DEPTH
RESTORATION_L2_TOLERANCE = quotient.RESTORATION_L2_TOLERANCE
FULL_STATE_PARITY_TOLERANCE = quotient.FULL_STATE_PARITY_TOLERANCE
BOUNDARY_TOLERANCE = quotient.BOUNDARY_TOLERANCE
CONTROL_FLOOR = quotient.CONTROL_FLOOR


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class Plan:
    theta: np.ndarray
    modes: np.ndarray
    mode_squared: np.ndarray
    retained_payload_bytes: int
    compilation_peak_payload_bytes: int


@dataclass
class Stats:
    forward_steps: int = 0
    inverse_steps: int = 0
    pair_phase_updates: int = 0
    fft_transforms: int = 0
    total_momentum_coordinate_closures: int = 0
    maximum_pair_factor_cells: int = 0
    maximum_free_slice_cells: int = 0
    maximum_explicit_engine_array_bytes: int = 0
    fft_backing_buffer_preserved: bool = True


def compile_plan() -> Plan:
    theta = (
        np.arange(GRID_SIZE, dtype=np.float64)
        * (2.0 * math.pi / GRID_SIZE)
    )
    modes = np.fft.fftfreq(GRID_SIZE, d=1.0 / GRID_SIZE)
    mode_squared = np.empty(GRID_SIZE, dtype=np.float64)
    np.square(modes, out=mode_squared)
    retained = theta.nbytes + modes.nbytes + mode_squared.nbytes
    return Plan(theta, modes, mode_squared, retained, retained)


def account(
    carrier: quotient.Carrier,
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
    runtime = plan.retained_payload_bytes + sum(
        array.nbytes for array in allocations.values()
    )
    stats.maximum_explicit_engine_array_bytes = max(
        stats.maximum_explicit_engine_array_bytes,
        carrier.samples.nbytes
        + max(runtime, plan.compilation_peak_payload_bytes),
    )


def pair_factor(
    plan: Plan,
    step: int,
    program_tag: int,
    edge: int,
    coupling_strength: float,
    conjugate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sign = 1.0 if conjugate else -1.0
    offset = dense.public_offset(
        step, program_tag, dense.ROTORS + edge
    )
    if edge == 0:
        phase = np.empty(GRID_SIZE, dtype=np.float64)
        np.negative(plan.theta, out=phase)
        phase += offset
    else:
        phase = np.empty(
            (GRID_SIZE, GRID_SIZE), dtype=np.float64
        )
        np.subtract(
            plan.theta[:, np.newaxis],
            plan.theta[np.newaxis, :],
            out=phase,
        )
        phase += offset
    np.cos(phase, out=phase)
    phase *= sign * coupling_strength
    factor = np.empty(phase.shape, dtype=np.complex128)
    factor.real = 0.0
    factor.imag = phase
    np.exp(factor, out=factor)
    return phase, factor


def apply_potential(
    carrier: quotient.Carrier,
    plan: Plan,
    step: int,
    program_tag: int,
    coupling_strength: float,
    conjugate: bool,
    stats: Stats,
) -> None:
    for edge in range(dense.ROTORS - 1):
        phase, factor = pair_factor(
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
        account(carrier, plan, stats, phase, factor)


def apply_free(
    carrier: quotient.Carrier,
    plan: Plan,
    free_time: float,
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
    third_indices = np.arange(GRID_SIZE, dtype=np.int64)
    zero_indices = np.empty(GRID_SIZE, dtype=np.int64)
    energy = np.empty(GRID_SIZE, dtype=np.float64)
    factor = np.empty(GRID_SIZE, dtype=np.complex128)
    sign = 1.0 if conjugate else -1.0
    for first in range(GRID_SIZE):
        for second in range(GRID_SIZE):
            np.add(
                third_indices,
                first + second,
                out=zero_indices,
            )
            np.negative(zero_indices, out=zero_indices)
            np.remainder(
                zero_indices, GRID_SIZE, out=zero_indices
            )
            np.take(plan.mode_squared, zero_indices, out=energy)
            energy += (
                plan.mode_squared[first]
                + plan.mode_squared[second]
            )
            energy += plan.mode_squared
            energy *= sign * 0.5 * free_time
            factor.real = 0.0
            factor.imag = energy
            np.exp(factor, out=factor)
            carrier.samples[first, second, :] *= factor
            stats.total_momentum_coordinate_closures += 1
    stats.maximum_free_slice_cells = GRID_SIZE
    account(
        carrier,
        plan,
        stats,
        third_indices,
        zero_indices,
        energy,
        factor,
    )
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


def forward_step(
    carrier: quotient.Carrier,
    plan: Plan,
    step: int,
    program_tag: int,
    coupling_strength: float,
    free_time: float,
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
    apply_free(carrier, plan, free_time, False, stats)
    stats.forward_steps += 1


def inverse_step(
    carrier: quotient.Carrier,
    plan: Plan,
    step: int,
    program_tag: int,
    coupling_strength: float,
    free_time: float,
    stats: Stats,
) -> None:
    apply_free(carrier, plan, free_time, True, stats)
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


def boundary(
    carrier: quotient.Carrier, plan: Plan, stats: Stats
) -> dict[str, object]:
    probability = np.empty(carrier.samples.shape, dtype=np.float64)
    np.absolute(carrier.samples, out=probability)
    np.square(probability, out=probability)
    overlap = complex(
        np.sum(carrier.samples) / math.sqrt(carrier.samples.size)
    )
    first_marginal = np.sum(probability, axis=(1, 2))
    pair_kernel = np.empty(GRID_SIZE, dtype=np.float64)
    np.negative(plan.theta, out=pair_kernel)
    np.cos(pair_kernel, out=pair_kernel)
    account(
        carrier,
        plan,
        stats,
        probability,
        first_marginal,
        pair_kernel,
    )
    return {
        "zero_momentum_amplitude": dense.complex_pair(overlap),
        "local_cosine": 0.0,
        "neighbor_phase_correlation": float(
            np.dot(first_marginal, pair_kernel)
        ),
        "norm": float(np.linalg.norm(carrier.samples)),
    }


def transaction(
    carrier: quotient.Carrier,
    depth: int,
    program_tag: int,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    baseline = carrier.samples.copy()
    carrier_address = carrier.samples.__array_interface__["data"][0]
    plan = compile_plan()
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
            free_time,
            stats,
        )
    latched = boundary(carrier, plan, stats)
    dense_samples, dense_boundary_value = quotient.dense_state(
        depth, program_tag, coupling_strength, free_time
    )
    lifted, lift_index_bytes = quotient.lift_to_four_rotors(
        carrier.samples
    )
    np.subtract(lifted, dense_samples, out=lifted)
    full_state_error = float(np.linalg.norm(lifted))
    boundary_error = dense.boundary_distance(
        latched, dense_boundary_value
    )
    if (
        full_state_error > FULL_STATE_PARITY_TOLERANCE
        or boundary_error > BOUNDARY_TOLERANCE
    ):
        fail("streamed momentum-coordinate dense parity failed")
    verification_dense_bytes = dense_samples.nbytes + lifted.nbytes
    verification_peak = (
        carrier.samples.nbytes
        + baseline.nbytes
        + plan.retained_payload_bytes
        + verification_dense_bytes
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
            free_time,
            stats,
        )
    restoration_error = float(np.linalg.norm(carrier.samples - baseline))
    if restoration_error > RESTORATION_L2_TOLERANCE:
        fail("streamed momentum-coordinate restoration failed")
    if carrier.samples.__array_interface__["data"][0] != carrier_address:
        fail("streamed momentum-coordinate carrier backing changed")
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
            "retained_public_plan_bytes": plan.retained_payload_bytes,
            "plan_compilation_peak_payload_bytes": (
                plan.compilation_peak_payload_bytes
            ),
            "maximum_explicit_engine_array_bytes": (
                stats.maximum_explicit_engine_array_bytes
            ),
            "maximum_explicit_wrapper_array_bytes": (
                stats.maximum_explicit_engine_array_bytes
                + baseline.nbytes
            ),
            "maximum_pair_factor_cells": (
                stats.maximum_pair_factor_cells
            ),
            "maximum_free_slice_cells": stats.maximum_free_slice_cells,
            "total_momentum_coordinate_closures": (
                stats.total_momentum_coordinate_closures
            ),
            "forward_steps": stats.forward_steps,
            "inverse_steps": stats.inverse_steps,
            "pair_phase_updates": stats.pair_phase_updates,
            "fft_transforms": stats.fft_transforms,
            "fft_backing_buffer_preserved": (
                stats.fft_backing_buffer_preserved
            ),
            "dense_free_phase_plan_materialized": False,
            "retained_inverse_history_bytes": 0,
            "verification_baseline_bytes": baseline.nbytes,
            "verification_baseline_reload_count": 0,
            "verification_baseline_used_for_restoration": False,
            "verification_dense_lift_bytes": verification_dense_bytes,
            "verification_lift_index_bytes": lift_index_bytes,
            "verification_peak_explicit_array_bytes": verification_peak,
            "elapsed_ns": time.perf_counter_ns() - start,
        },
    }


def controls() -> dict[str, float]:
    initial = quotient.product_zero_total_momentum_carrier()
    plan = compile_plan()
    forward = quotient.Carrier(initial.samples.copy())
    forward_step(
        forward,
        plan,
        1,
        1,
        dense.PRIMARY_COUPLING,
        dense.PRIMARY_FREE_TIME,
        Stats(),
    )
    missing = float(np.linalg.norm(forward.samples - initial.samples))
    wrong = quotient.Carrier(forward.samples.copy())
    inverse_step(
        wrong,
        plan,
        1,
        1,
        dense.PRIMARY_COUPLING * 1.1,
        dense.PRIMARY_FREE_TIME,
        Stats(),
    )
    reordered = quotient.Carrier(forward.samples.copy())
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
    apply_free(
        reordered,
        plan,
        dense.PRIMARY_FREE_TIME,
        True,
        reordered_stats,
    )
    return {
        "missing_inverse_error": missing,
        "wrong_inverse_error": float(
            np.linalg.norm(wrong.samples - initial.samples)
        ),
        "reordered_inverse_error": float(
            np.linalg.norm(reordered.samples - initial.samples)
        ),
    }


def signature(result: dict[str, object]) -> tuple[object, ...]:
    resources = result["resources"]
    assert isinstance(resources, dict)
    return tuple(
        resources[key]
        for key in (
            "quotient_carrier_complex_cells",
            "quotient_carrier_payload_bytes",
            "retained_public_plan_bytes",
            "plan_compilation_peak_payload_bytes",
            "maximum_explicit_engine_array_bytes",
            "maximum_explicit_wrapper_array_bytes",
            "maximum_pair_factor_cells",
            "maximum_free_slice_cells",
            "retained_inverse_history_bytes",
        )
    )


def main() -> None:
    depth_rows: list[dict[str, object]] = []
    signatures: set[tuple[object, ...]] = set()
    for depth in DEPTH_SWEEP:
        result = transaction(
            quotient.product_zero_total_momentum_carrier(),
            depth,
            1,
            dense.PRIMARY_COUPLING,
            dense.PRIMARY_FREE_TIME,
        )
        signatures.add(signature(result))
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
                "resource_signature": signature(result),
            }
        )
    carrier = quotient.product_zero_total_momentum_carrier()
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
        quotient.product_zero_total_momentum_carrier(),
        11,
        2,
        dense.REUSE_COUPLING,
        dense.REUSE_FREE_TIME,
    )
    reuse_boundary_error = dense.boundary_distance(
        reuse["boundary"], fresh["boundary"]
    )
    control = controls()
    dense_plan_bytes = GRID_SIZE**3 * 16 + GRID_SIZE * 16
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_TOPOLOGY_STREAMED_TOTAL_MOMENTUM_COORDINATE_"
            "PHASE_CLOSURE_ELIMINATES_DENSE_QUOTIENT_FREE_PLAN_"
            "WITH_ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTATION_INVARIANT_ROTORS_GRID17_"
            "DEPTHS1_TO64_SOFTWARE_COMPLEX128_SCIPY_POCKETFFT_"
            "NO_FFT_INTERNAL_WORKSPACE_BOUND"
        ),
        "depth_growth": depth_rows,
        "depth_independent_resource_signature": len(signatures) == 1,
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh,
        "fresh_restored_boundary_error": reuse_boundary_error,
        "controls": control,
        "previous_dense_quotient_retained_plan_bytes": dense_plan_bytes,
        "streamed_retained_plan_bytes": primary["resources"][
            "retained_public_plan_bytes"
        ],
        "retained_plan_reduction_factor": (
            dense_plan_bytes
            / primary["resources"]["retained_public_plan_bytes"]
        ),
        "unprojected_topology_derived_internal_coordinate": (
            "TOTAL_MOMENTUM_N0_MOD17"
        ),
        "internal_coordinate_projected": False,
        "phase_resident_unresolved_internal_port_established": False,
        "warm_time_reduction_established": False,
        "matched_classical_streamed_quotient_identical": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    if (
        len(signatures) != 1
        or primary["restoration_error"] > RESTORATION_L2_TOLERANCE
        or reuse["restoration_generation"] != 2
        or reuse_boundary_error > BOUNDARY_TOLERANCE
        or min(control.values()) <= CONTROL_FLOOR
    ):
        fail("streamed momentum-coordinate qualification failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(2) from error
