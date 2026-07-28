#!/usr/bin/env python3
"""Private backend law for the incremental four-rotor CATVM triad."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass

import numpy as np

import four_rotor_incremental_schmidt_closure as phase
import four_rotor_kicked_phase_tt as reference
import four_rotor_kicked_phase_tt_matrix_free as matrix_free


@dataclass(frozen=True)
class Program:
    depth: int
    kick_strength: float
    coupling_strength: float
    free_time: float


PROGRAMS = {
    "PRIMARY": Program(
        reference.PRIMARY_DEPTH,
        reference.PRIMARY_K,
        reference.PRIMARY_G,
        reference.PRIMARY_TAU,
    ),
    "REUSE": Program(2, 0.9, 0.22, math.sqrt(5.0)),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def warm_runtime() -> None:
    matrix = np.asfortranarray(
        np.array(
            [[1.0 + 0.0j, 0.25j], [0.5 + 0.0j, 1.0 - 0.25j]],
            dtype=np.complex128,
        )
    )
    np.linalg.svd(matrix, full_matrices=False)


def process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def carrier_payload_bytes(carrier: reference.Carrier) -> int:
    return matrix_free.carrier_backing_cells(carrier) * 16


def public_boundary(
    boundary: dict[str, object],
) -> dict[str, object]:
    amplitude = boundary["zero_product_amplitude"]
    if not isinstance(amplitude, list) or len(amplitude) != 2:
        fail("four-rotor public amplitude malformed")
    return {
        "zero_product_amplitude": [
            float(amplitude[0]),
            float(amplitude[1]),
        ],
        "norm": float(boundary["norm"]),
        "neighbor_momentum_correlation": float(
            boundary["neighbor_momentum_correlation"]
        ),
        "central_schmidt_rank": int(
            boundary["central_schmidt_rank"]
        ),
        "central_schmidt_entropy": float(
            boundary["central_schmidt_entropy"]
        ),
        "maximum_local_epsilon_radius": int(
            boundary["maximum_local_epsilon_radius"]
        ),
    }


def custody_receipt(
    arm: str,
    program: str,
    transaction_id: int,
    generation: int,
    boundary: dict[str, object],
) -> str:
    # This receipt identifies custody through public transaction identity and
    # the declared final boundary. It never serializes the resident TT state.
    payload = json.dumps(
        {
            "law": "CATVM_FOUR_ROTOR_INCREMENTAL_V1",
            "arm": arm,
            "program": program,
            "transaction_id": transaction_id,
            "generation": generation,
            "boundary": boundary,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def public_operator_payload_bytes(
    program: Program, inverse: bool
) -> int:
    dimension = 2 * reference.MODE_RADIUS + 1
    terms = 2 * reference.COUPLING_KERNEL_RADIUS + 1
    per_round_cells = (
        dimension * dimension
        + dimension
        + 3 * terms * 2 * dimension * dimension
    )
    passes = 2 if inverse else 1
    return passes * program.depth * per_round_cells * 16


def common_metrics(
    program: Program,
    stats: phase.Stats,
    elapsed_ns: int,
    verification_baseline_bytes: int,
    snapshot_resident_bytes: int,
) -> dict[str, object]:
    engine_peak = stats.maximum_total_live_cells * 16
    wrapper_peak = (
        engine_peak
        + verification_baseline_bytes
        + snapshot_resident_bytes
    )
    return {
        "engine_execution_ns": elapsed_ns,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "engine_accounted_peak_array_bytes": engine_peak,
        "wrapper_accounted_peak_array_bytes": wrapper_peak,
        "carrier_creation_payload_bytes": (
            (2 * reference.MODE_RADIUS + 1) * reference.ROTORS * 16
        ),
        "verification_baseline_bytes": verification_baseline_bytes,
        "verification_baseline_reload_count": 0,
        "verification_baseline_used_for_restoration": False,
        "snapshot_resident_bytes": snapshot_resident_bytes,
        "public_operator_materialization_bytes_total": (
            public_operator_payload_bytes(
                program, verification_baseline_bytes > 0
            )
        ),
        "native_coupling_applications": stats.coupling_applications,
        "native_incremental_updates": stats.incremental_updates,
        "projection_calls": 1,
        "projection_output_scalar_count": 7,
        "projection_output_payload_bytes": 56,
        "retained_inverse_history_bytes": 0,
        "rematerialized_public_bessel_terms": (
            stats.incremental_updates
        ),
        "combined_declared_l2_bound": (
            stats.discarded_l2_bound
            + stats.bessel_kernel_tail_l2_bound
        ),
    }


def forward_only(
    arm: str,
    carrier: reference.Carrier,
    program_name: str,
    transaction_id: int,
    snapshot_resident_bytes: int = 0,
) -> dict[str, object]:
    program = PROGRAMS.get(program_name)
    if program is None:
        fail("four-rotor public program invalid")
    stats = phase.Stats()
    start = time.perf_counter_ns()
    central: list[float] = [1.0]
    for _ in range(program.depth):
        central = phase.forward_round(
            carrier,
            program.kick_strength,
            program.coupling_strength,
            program.free_time,
            stats,
        )
    boundary = public_boundary(reference.boundary(carrier, central))
    elapsed = time.perf_counter_ns() - start
    return {
        "program": program_name,
        "transaction_id": transaction_id,
        "final_boundary": boundary,
        "actual_inverse_restoration": False,
        "canonical_restoration": False,
        "restoration_error": None,
        "restoration_generation": 0,
        "custody_receipt": custody_receipt(
            arm,
            program_name,
            transaction_id,
            0,
            boundary,
        ),
        "resources": common_metrics(
            program,
            stats,
            elapsed,
            0,
            snapshot_resident_bytes,
        ),
    }


def in_place(
    carrier: reference.Carrier,
    program_name: str,
    transaction_id: int,
) -> dict[str, object]:
    program = PROGRAMS.get(program_name)
    if program is None:
        fail("four-rotor public program invalid")
    verification_baseline_bytes = carrier_payload_bytes(carrier)
    start = time.perf_counter_ns()
    result = phase.transaction(
        carrier,
        program.depth,
        program.kick_strength,
        program.coupling_strength,
        program.free_time,
    )
    elapsed = time.perf_counter_ns() - start
    stats = result["stats"]
    if not isinstance(stats, dict):
        fail("four-rotor transaction statistics malformed")
    if not result["actual_inverse_restoration"]:
        fail("four-rotor actual inverse restoration failed")
    boundary = public_boundary(result["boundary"])
    filtered_stats = phase.Stats(
        coupling_applications=int(stats["coupling_applications"]),
        incremental_updates=int(stats["incremental_updates"]),
        maximum_total_live_cells=int(
            stats["maximum_total_live_cells"]
        ),
        discarded_l2_bound=float(stats["discarded_l2_bound"]),
        bessel_kernel_tail_l2_bound=float(
            stats["bessel_kernel_tail_l2_bound"]
        ),
    )
    generation = int(result["restoration_generation"])
    return {
        "program": program_name,
        "transaction_id": transaction_id,
        "final_boundary": boundary,
        "actual_inverse_restoration": True,
        "canonical_restoration": (
            result["closure"]["bond_ranks_after"] == [1, 1, 1]
        ),
        "restoration_error": float(
            result["postclosure_restoration_error"]
        ),
        "restoration_generation": generation,
        "custody_receipt": custody_receipt(
            "IN_PLACE",
            program_name,
            transaction_id,
            generation,
            boundary,
        ),
        "resources": common_metrics(
            program,
            filtered_stats,
            elapsed,
            verification_baseline_bytes,
            0,
        ),
    }


def boundary_distance(
    left: dict[str, object], right: dict[str, object]
) -> float:
    left_amplitude = complex(*left["zero_product_amplitude"])
    right_amplitude = complex(*right["zero_product_amplitude"])
    return max(
        abs(left_amplitude - right_amplitude),
        abs(float(left["norm"]) - float(right["norm"])),
        abs(
            float(left["neighbor_momentum_correlation"])
            - float(right["neighbor_momentum_correlation"])
        ),
    )
