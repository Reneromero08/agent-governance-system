#!/usr/bin/env python3
"""Compact TT test for the exact finite-torus cyclic four-rotor law."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import fft as scipy_fft

import four_rotor_cyclic_phase_law as dense
import four_rotor_kicked_phase_tt as tt
import four_rotor_post_inverse_canonical_closure as canonical


GRID_SIZE = dense.GRID_SIZE
PRIMARY_DEPTH = 5
DEPTH_SWEEP = (1, 2, 3, 4, 5)
PAIR_GATE_L2_TOLERANCE = 2.0e-12
RESTORATION_L2_TOLERANCE = 2.0e-9
BOUNDARY_TOLERANCE = 2.0e-9
CONTROL_FLOOR = 1.0e-4


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class Stats:
    pair_phase_closures: int = 0
    local_phase_updates: int = 0
    local_free_updates: int = 0
    svd_factorizations: int = 0
    maximum_bond_rank: int = 1
    maximum_carrier_cells: int = 0
    maximum_two_site_core_cells: int = 0
    discarded_l2_sum: float = 0.0


def product_uniform_carrier() -> tt.Carrier:
    vector = np.full(
        GRID_SIZE, 1.0 / math.sqrt(GRID_SIZE), dtype=np.complex128
    )
    tensors = [
        vector.reshape(1, GRID_SIZE, 1).copy()
        for _ in range(dense.ROTORS)
    ]
    return tt.Carrier(tensors, (GRID_SIZE - 1) // 2)


def update_stats(carrier: tt.Carrier, stats: Stats) -> None:
    cells = sum(int(tensor.size) for tensor in carrier.tensors)
    stats.maximum_carrier_cells = max(stats.maximum_carrier_cells, cells)
    stats.maximum_bond_rank = max(
        stats.maximum_bond_rank,
        *(int(tensor.shape[2]) for tensor in carrier.tensors[:-1]),
    )


def free_matrix(free_time: float, conjugate: bool) -> np.ndarray:
    plan = dense.compile_plan(GRID_SIZE, free_time)
    factor = plan.free_factor.conj() if conjugate else plan.free_factor
    identity = np.eye(GRID_SIZE, dtype=np.complex128)
    momentum = scipy_fft.fft(
        identity, axis=0, norm="ortho", workers=1
    )
    momentum *= factor[:, np.newaxis]
    return scipy_fft.ifft(
        momentum, axis=0, norm="ortho", workers=1
    )


def apply_local_matrix(
    carrier: tt.Carrier,
    matrix: np.ndarray,
    stats: Stats,
    free: bool,
) -> None:
    for index, tensor in enumerate(carrier.tensors):
        carrier.tensors[index] = np.einsum(
            "nm,amb->anb", matrix, tensor, optimize=True
        )
        if free:
            stats.local_free_updates += 1
        else:
            stats.local_phase_updates += 1
    update_stats(carrier, stats)


def onsite_factor(
    step: int,
    program_tag: int,
    rotor: int,
    strength: float,
    conjugate: bool,
) -> np.ndarray:
    plan = dense.compile_plan(GRID_SIZE, 0.0)
    sign = 1.0 if conjugate else -1.0
    offset = dense.public_offset(step, program_tag, rotor)
    return np.exp(
        sign * 1j * strength * np.cos(plan.theta + offset)
    )


def pair_factor(
    step: int,
    program_tag: int,
    edge: int,
    strength: float,
    conjugate: bool,
) -> np.ndarray:
    plan = dense.compile_plan(GRID_SIZE, 0.0)
    sign = 1.0 if conjugate else -1.0
    offset = dense.public_offset(
        step, program_tag, dense.ROTORS + edge
    )
    return np.exp(
        sign
        * 1j
        * strength
        * np.cos(
            plan.theta[:, np.newaxis]
            - plan.theta[np.newaxis, :]
            + offset
        )
    )


def apply_onsite(
    carrier: tt.Carrier,
    step: int,
    program_tag: int,
    strength: float,
    conjugate: bool,
    stats: Stats,
) -> None:
    for rotor, tensor in enumerate(carrier.tensors):
        factor = onsite_factor(
            step, program_tag, rotor, strength, conjugate
        )
        tensor *= factor[np.newaxis, :, np.newaxis]
        stats.local_phase_updates += 1
    update_stats(carrier, stats)


def retained_rank(singular: np.ndarray) -> tuple[int, float]:
    squared = singular * singular
    for keep in range(1, len(singular) + 1):
        discarded_squared = float(np.sum(squared[keep:]))
        if discarded_squared <= PAIR_GATE_L2_TOLERANCE**2:
            return keep, math.sqrt(discarded_squared)
    return len(singular), 0.0


def apply_pair(
    carrier: tt.Carrier,
    edge: int,
    factor: np.ndarray,
    stats: Stats,
) -> list[float]:
    gauge = tt.Stats()
    tt.left_canonicalize(carrier.tensors, edge, gauge)
    tt.right_canonicalize(carrier.tensors, edge + 1, gauge)
    left_tensor = carrier.tensors[edge]
    right_tensor = carrier.tensors[edge + 1]
    left_rank, physical, bond = left_tensor.shape
    _, _, right_rank = right_tensor.shape
    joined = np.einsum(
        "aib,bjc->aijc",
        left_tensor,
        right_tensor,
        optimize=True,
    )
    joined *= factor[np.newaxis, :, :, np.newaxis]
    matrix = joined.reshape(
        left_rank * physical, physical * right_rank
    )
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    keep, discarded = retained_rank(singular)
    carrier.tensors[edge] = np.ascontiguousarray(
        u[:, :keep].reshape(left_rank, physical, keep)
    )
    carrier.tensors[edge + 1] = (
        singular[:keep, np.newaxis] * vh[:keep]
    ).reshape(keep, physical, right_rank)
    stats.pair_phase_closures += 1
    stats.svd_factorizations += 1
    stats.maximum_two_site_core_cells = max(
        stats.maximum_two_site_core_cells, int(joined.size)
    )
    stats.discarded_l2_sum += discarded
    update_stats(carrier, stats)
    normalized = singular[:keep] / np.linalg.norm(singular)
    return [float(value) for value in normalized]


def forward_step(
    carrier: tt.Carrier,
    step: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
    stats: Stats,
) -> None:
    apply_onsite(
        carrier,
        step,
        program_tag,
        kick_strength,
        False,
        stats,
    )
    for edge in range(dense.ROTORS - 1):
        apply_pair(
            carrier,
            edge,
            pair_factor(
                step,
                program_tag,
                edge,
                coupling_strength,
                False,
            ),
            stats,
        )
    apply_local_matrix(
        carrier, free_matrix(free_time, False), stats, True
    )


def inverse_step(
    carrier: tt.Carrier,
    step: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
    stats: Stats,
) -> None:
    apply_local_matrix(
        carrier, free_matrix(free_time, True), stats, True
    )
    for edge in range(dense.ROTORS - 2, -1, -1):
        apply_pair(
            carrier,
            edge,
            pair_factor(
                step,
                program_tag,
                edge,
                coupling_strength,
                True,
            ),
            stats,
        )
    apply_onsite(
        carrier,
        step,
        program_tag,
        kick_strength,
        True,
        stats,
    )


def product_overlap(carrier: tt.Carrier) -> complex:
    vector = np.full(
        GRID_SIZE, 1.0 / math.sqrt(GRID_SIZE), dtype=np.complex128
    )
    environment = np.ones(1, dtype=np.complex128)
    for tensor in carrier.tensors:
        environment = np.einsum(
            "a,aib,i->b",
            environment,
            tensor,
            vector.conj(),
            optimize=True,
        )
    return complex(environment[0])


def central_spectrum(carrier: tt.Carrier) -> np.ndarray:
    copy = tt.copy_carrier(carrier)
    gauge = tt.Stats()
    tt.left_canonicalize(copy.tensors, 1, gauge)
    tt.right_canonicalize(copy.tensors, 2, gauge)
    left = copy.tensors[1]
    right = copy.tensors[2]
    matrix = np.einsum(
        "aib,bjc->aijc", left, right, optimize=True
    ).reshape(left.shape[0] * GRID_SIZE, GRID_SIZE * right.shape[2])
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = PAIR_GATE_L2_TOLERANCE
    return singular[singular > threshold]


def boundary(carrier: tt.Carrier) -> dict[str, object]:
    theta = dense.compile_plan(GRID_SIZE, 0.0).theta
    cosine = np.cos(theta)
    sine = np.sin(theta)
    pair = (
        tt.diagonal_expectation(carrier, {0: cosine, 1: cosine})
        + tt.diagonal_expectation(carrier, {0: sine, 1: sine})
    )
    singular = central_spectrum(carrier)
    normalized = singular / np.linalg.norm(singular)
    entropy = -float(
        np.sum(
            np.square(normalized)
            * np.log(np.square(normalized))
        )
    )
    return {
        "zero_momentum_amplitude": dense.complex_pair(
            product_overlap(carrier)
        ),
        "local_cosine": tt.diagonal_expectation(
            carrier, {0: cosine}
        ),
        "neighbor_phase_correlation": pair,
        "norm": float(tt.inner(carrier, carrier).real),
        "central_schmidt_rank": int(len(singular)),
        "central_schmidt_entropy": entropy,
    }


def shared_boundary_distance(
    compact: dict[str, object], reference: dict[str, object]
) -> float:
    return dense.boundary_distance(compact, reference)


def transaction(
    carrier: tt.Carrier,
    depth: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    initial = tt.copy_carrier(carrier)
    stats = Stats()
    update_stats(carrier, stats)
    rank_history: list[list[int]] = []
    start = time.perf_counter_ns()
    for step in range(1, depth + 1):
        forward_step(
            carrier,
            step,
            program_tag,
            kick_strength,
            coupling_strength,
            free_time,
            stats,
        )
        rank_history.append(
            [tensor.shape[2] for tensor in carrier.tensors[:-1]]
        )
    latched = boundary(carrier)
    for step in range(depth, 0, -1):
        inverse_step(
            carrier,
            step,
            program_tag,
            kick_strength,
            coupling_strength,
            free_time,
            stats,
        )
    inverse_error = tt.physical_distance(initial, carrier)
    if inverse_error > RESTORATION_L2_TOLERANCE:
        fail("compact cyclic TT actual inverse restoration failed")
    closure = canonical.canonical_round_actual(
        carrier, RESTORATION_L2_TOLERANCE / 4.0
    )
    restoration_error = tt.physical_distance(initial, carrier)
    if restoration_error > RESTORATION_L2_TOLERANCE:
        fail("compact cyclic TT restoration failed")
    carrier.generation += 1
    return {
        "boundary": latched,
        "rank_history": rank_history,
        "inverse_restoration_error": inverse_error,
        "postclosure_restoration_error": restoration_error,
        "closure": closure,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse_ready": True,
        "resources": {
            "maximum_bond_rank": stats.maximum_bond_rank,
            "maximum_logical_tt_cells": stats.maximum_carrier_cells,
            "maximum_logical_tt_payload_bytes": (
                stats.maximum_carrier_cells * 16
            ),
            "maximum_two_site_core_cells": (
                stats.maximum_two_site_core_cells
            ),
            "pair_phase_closures": stats.pair_phase_closures,
            "svd_factorizations": stats.svd_factorizations,
            "discarded_l2_sum": stats.discarded_l2_sum,
            "retained_inverse_history_bytes": 0,
            "physical_coordinate_dense_wave_materialized": False,
            "simultaneous_peak_payload_established": False,
            "elapsed_ns": time.perf_counter_ns() - start,
        },
    }


def dense_boundary(
    depth: int,
    program_tag: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    return dense.forward_only(
        depth,
        program_tag,
        kick_strength,
        coupling_strength,
        free_time,
    )["boundary"]


def controls() -> dict[str, float]:
    initial = product_uniform_carrier()
    forward = tt.copy_carrier(initial)
    forward_step(
        forward,
        1,
        1,
        dense.PRIMARY_KICK,
        dense.PRIMARY_COUPLING,
        dense.PRIMARY_FREE_TIME,
        Stats(),
    )
    missing = tt.physical_distance(initial, forward)
    wrong = tt.copy_carrier(forward)
    inverse_step(
        wrong,
        1,
        1,
        dense.PRIMARY_KICK,
        dense.PRIMARY_COUPLING * 1.1,
        dense.PRIMARY_FREE_TIME,
        Stats(),
    )
    reordered = tt.copy_carrier(forward)
    reordered_stats = Stats()
    apply_onsite(
        reordered,
        1,
        1,
        dense.PRIMARY_KICK,
        True,
        reordered_stats,
    )
    for edge in range(dense.ROTORS - 2, -1, -1):
        apply_pair(
            reordered,
            edge,
            pair_factor(
                1,
                1,
                edge,
                dense.PRIMARY_COUPLING,
                True,
            ),
            reordered_stats,
        )
    apply_local_matrix(
        reordered,
        free_matrix(dense.PRIMARY_FREE_TIME, True),
        reordered_stats,
        True,
    )
    return {
        "missing_inverse_error": missing,
        "wrong_inverse_error": tt.physical_distance(initial, wrong),
        "reordered_inverse_error": tt.physical_distance(
            initial, reordered
        ),
    }


def main() -> None:
    depth_rows: list[dict[str, object]] = []
    for depth in DEPTH_SWEEP:
        result = transaction(
            product_uniform_carrier(),
            depth,
            1,
            dense.PRIMARY_KICK,
            dense.PRIMARY_COUPLING,
            dense.PRIMARY_FREE_TIME,
        )
        reference_boundary = dense_boundary(
            depth,
            1,
            dense.PRIMARY_KICK,
            dense.PRIMARY_COUPLING,
            dense.PRIMARY_FREE_TIME,
        )
        depth_rows.append(
            {
                "depth": depth,
                "rank_history": result["rank_history"],
                "central_boundary_rank": result["boundary"][
                    "central_schmidt_rank"
                ],
                "boundary_error_vs_dense": shared_boundary_distance(
                    result["boundary"], reference_boundary
                ),
                "postclosure_restoration_error": result[
                    "postclosure_restoration_error"
                ],
                "maximum_logical_tt_cells": result["resources"][
                    "maximum_logical_tt_cells"
                ],
                "maximum_two_site_core_cells": result["resources"][
                    "maximum_two_site_core_cells"
                ],
            }
        )
    carrier = product_uniform_carrier()
    primary = transaction(
        carrier,
        PRIMARY_DEPTH,
        1,
        dense.PRIMARY_KICK,
        dense.PRIMARY_COUPLING,
        dense.PRIMARY_FREE_TIME,
    )
    reuse = transaction(
        carrier,
        2,
        2,
        dense.REUSE_KICK,
        dense.REUSE_COUPLING,
        dense.REUSE_FREE_TIME,
    )
    fresh = transaction(
        product_uniform_carrier(),
        2,
        2,
        dense.REUSE_KICK,
        dense.REUSE_COUPLING,
        dense.REUSE_FREE_TIME,
    )
    reuse_boundary_error = shared_boundary_distance(
        reuse["boundary"], fresh["boundary"]
    )
    control = controls()
    final_rank = depth_rows[-1]["central_boundary_rank"]
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_CYCLIC_PHASE_TT_NATIVE_PAIR_CLOSURE_"
            "SATURATES_GRID17_CENTRAL_INTERFACE_AT_DEPTH5_"
            "WITH_ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTORS_GRID17_DEPTHS1_TO5_PAIR_"
            "SVD_L2_2E_MINUS12_SOFTWARE_COMPLEX128"
        ),
        "depth_growth": depth_rows,
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh,
        "fresh_restored_boundary_error": reuse_boundary_error,
        "controls": control,
        "exact_central_rank_ceiling": GRID_SIZE * GRID_SIZE,
        "central_rank_fraction_of_exact_width": (
            final_rank / (GRID_SIZE * GRID_SIZE)
        ),
        "central_rank_near_saturated_exact_width": (
            final_rank >= math.floor(0.95 * GRID_SIZE * GRID_SIZE)
        ),
        "dense_cyclic_carrier_cells": GRID_SIZE**dense.ROTORS,
        "logical_tt_over_dense_cell_ratio": (
            primary["resources"]["maximum_logical_tt_cells"]
            / (GRID_SIZE**dense.ROTORS)
        ),
        "dense_equivalent_interface_core_materialized": (
            primary["resources"]["maximum_two_site_core_cells"]
            == GRID_SIZE**dense.ROTORS
        ),
        "identified_obstruction": (
            "COMPACT_CYCLIC_PHASE_TT_CENTRAL_RANK_SATURATES_"
            "DENSE_INTERFACE"
        ),
        "matched_classical_cyclic_tt_identical": True,
        "compact_fixed_rank_across_depth_established": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    if (
        max(row["boundary_error_vs_dense"] for row in depth_rows)
        > BOUNDARY_TOLERANCE
        or primary["postclosure_restoration_error"]
        > RESTORATION_L2_TOLERANCE
        or reuse["restoration_generation"] != 2
        or reuse_boundary_error > BOUNDARY_TOLERANCE
        or min(control.values()) <= CONTROL_FLOOR
        or final_rank != GRID_SIZE * GRID_SIZE
        or primary["resources"]["maximum_logical_tt_cells"]
        <= GRID_SIZE**dense.ROTORS
    ):
        fail("compact cyclic TT qualification failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(2) from error
