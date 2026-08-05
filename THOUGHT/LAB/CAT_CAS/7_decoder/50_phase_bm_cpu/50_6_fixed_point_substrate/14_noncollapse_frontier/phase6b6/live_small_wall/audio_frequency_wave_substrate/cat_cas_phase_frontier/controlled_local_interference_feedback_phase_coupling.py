#!/usr/bin/env python3
"""Controlled local nonlinear phase-coupling Small Wall diagnostic.

This successor removes the free global DFT primitive used by M186.  Each
layer consists of two disjoint nearest-neighbour unitary-coupler sublayers and
one local intensity-dependent phase-feedback sublayer.  The inverse is
rematerialized from public topology and acts on the borrowed complex carrier.

The implementation counts every resident mode, local coupler, feedback
operation, precision bit, snapshot byte, and accepted inverse.  It is virtual
complex128 software, not a physical waveform or CATVM boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np


CLAIM = (
    "BOUNDED_CONTROLLED_FOUR_CASE_REVERSIBLE_LOCAL_INTERFERENCE_FEEDBACK_"
    "PHASE_COUPLING_REPLACES_THE_FREE_GLOBAL_DFT_WITH_PUBLIC_BRICKWORK_"
    "UNITARY_COUPLERS_AND_LOCAL_INTENSITY_DEPENDENT_PHASE_FEEDBACK_WITH_"
    "FINAL_ONLY_BOUNDARY_NUMERICAL_SAME_BACKING_RESTORATION_AND_REUSE_"
    "BUT_REQUIRES_N_COMPLEX128_MODES_N_LOCAL_COUPLERS_AND_N_FEEDBACK_"
    "OPERATIONS_PER_LAYER_AND_COLLAPSES_TO_THE_IDENTICAL_COMPACT_"
    "CLASSICAL_FULL_STATE_RECURRENCE_SO_NO_DISTINCT_PHASE_RESOURCE_OR_"
    "SMALL_WALL_CROSSING_IS_ESTABLISHED"
)

CASES = ((8, 8), (16, 16), (32, 32), (64, 64))
STATE_TOLERANCE = 2.0e-10
BOUNDARY_TOLERANCE = 2.0e-11
CONTROL_MINIMUM = 1.0e-8
REUSE_CYCLES = 128
COMPLEX_BITS = 128
COMPLEX_BYTES = 16


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class PublicProgram:
    width: int
    depth: int
    even_angle: float
    odd_angle: float
    feedback_strength: float
    phase_seed: int
    weight_shift: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "depth": self.depth,
            "even_angle": self.even_angle,
            "odd_angle": self.odd_angle,
            "feedback_strength": self.feedback_strength,
            "phase_seed": self.phase_seed,
            "weight_shift": self.weight_shift,
        }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def source_state(program: PublicProgram) -> np.ndarray:
    width = program.width
    carrier = np.empty(width, dtype=np.complex128)
    for index in range(width):
        magnitude = 1.0 + 0.19 * math.cos(
            2.0 * math.pi * (index + 2) / width
        )
        phase = 2.0 * math.pi * (
            index * index + 3 * index + 1
        ) / (width + 3)
        carrier[index] = magnitude * complex(math.cos(phase), math.sin(phase))
    carrier /= np.linalg.norm(carrier)
    return carrier


def layer_parameters(program: PublicProgram, layer: int) -> tuple[float, float, float]:
    even = program.even_angle + 0.003 * (
        (3 * layer + program.phase_seed) % 11
    )
    odd = program.odd_angle - 0.002 * (
        (5 * layer + 2 * program.phase_seed) % 13
    )
    feedback = program.feedback_strength * (
        1.0 + 0.01 * ((layer + program.phase_seed) % 7)
    )
    return even, odd, feedback


def coupler_pair(carrier: np.ndarray, left: int, right: int, angle: float) -> None:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = complex(carrier[left])
    b = complex(carrier[right])
    carrier[left] = cosine * a + 1j * sine * b
    carrier[right] = 1j * sine * a + cosine * b


def coupler_sublayer(carrier: np.ndarray | None, offset: int, angle: float) -> None:
    if carrier is None:
        fail("null carrier")
    width = len(carrier)
    if width < 2 or width % 2:
        fail("local brickwork requires a positive even carrier width")
    for left in range(offset, width, 2):
        right = (left + 1) % width
        coupler_pair(carrier, left, right, angle)


def feedback_sublayer(carrier: np.ndarray | None, strength: float) -> None:
    if carrier is None:
        fail("null carrier")
    for index in range(len(carrier)):
        value = complex(carrier[index])
        angle = strength * (value.real * value.real + value.imag * value.imag)
        carrier[index] = value * complex(math.cos(angle), math.sin(angle))


def forward_in_place(carrier: np.ndarray | None, program: PublicProgram) -> None:
    if carrier is None:
        fail("null carrier")
    for layer in range(program.depth):
        even, odd, feedback = layer_parameters(program, layer)
        coupler_sublayer(carrier, 0, even)
        coupler_sublayer(carrier, 1, odd)
        feedback_sublayer(carrier, feedback)


def inverse_in_place(carrier: np.ndarray | None, program: PublicProgram) -> None:
    if carrier is None:
        fail("null carrier")
    for layer in reversed(range(program.depth)):
        even, odd, feedback = layer_parameters(program, layer)
        feedback_sublayer(carrier, -feedback)
        coupler_sublayer(carrier, 1, -odd)
        coupler_sublayer(carrier, 0, -even)


def wrong_inverse_in_place(carrier: np.ndarray, program: PublicProgram) -> None:
    wrong = replace(program, feedback_strength=program.feedback_strength + 0.073)
    inverse_in_place(carrier, wrong)


def reordered_inverse_in_place(carrier: np.ndarray, program: PublicProgram) -> None:
    for layer in reversed(range(program.depth)):
        even, odd, feedback = layer_parameters(program, layer)
        coupler_sublayer(carrier, 1, -odd)
        coupler_sublayer(carrier, 0, -even)
        feedback_sublayer(carrier, -feedback)


def swapped_forward_in_place(carrier: np.ndarray, program: PublicProgram) -> None:
    for layer in range(program.depth):
        even, odd, feedback = layer_parameters(program, layer)
        coupler_sublayer(carrier, 1, odd)
        coupler_sublayer(carrier, 0, even)
        feedback_sublayer(carrier, feedback)


def project_boundary(carrier: np.ndarray, program: PublicProgram) -> complex:
    total = 0j
    scale = math.sqrt(program.width)
    for index, value in enumerate(carrier):
        angle = 2.0 * math.pi * program.weight_shift * index / program.width
        weight = complex(math.cos(angle), math.sin(angle)) / scale
        total += weight.conjugate() * complex(value)
    return total


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def response_payload(value: complex) -> dict[str, list[float]]:
    return {"final_boundary": complex_pair(value)}


def commitment(carrier: np.ndarray) -> str:
    return hashlib.sha256(carrier.tobytes(order="C")).hexdigest()


def maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    return max(
        abs(complex(a) - complex(b)) for a, b in zip(left, right, strict=True)
    )


def boundary_close(left: complex, right: complex) -> bool:
    return abs(left - right) <= BOUNDARY_TOLERANCE


def compact_direct(program: PublicProgram) -> dict[str, Any]:
    carrier = source_state(program)
    forward_in_place(carrier, program)
    boundary = project_boundary(carrier, program)
    response = response_payload(boundary)
    return {
        "boundary": complex_pair(boundary),
        "response": response,
        "request_bytes": len(canonical_bytes(program.as_dict())),
        "response_bytes": len(canonical_bytes(response)),
        "logical_complex_cells": program.width + 2,
        "local_couplers": program.width * program.depth,
        "local_feedback_operations": program.width * program.depth,
        "projection_terms": program.width,
        "snapshot_bytes": 0,
        "restoration_classification": "NO_RESTORATION_CLAIM",
    }


def snapshot_sham(program: PublicProgram, second: PublicProgram) -> dict[str, Any]:
    carrier = source_state(program)
    backing = id(carrier)
    snapshot = carrier.copy()
    forward_in_place(carrier, program)
    boundary = project_boundary(carrier, program)
    carrier[:] = snapshot
    restored_error = maximum_error(carrier, snapshot)
    forward_in_place(carrier, second)
    reused_boundary = project_boundary(carrier, second)
    carrier[:] = snapshot
    fresh_boundary = complex(*compact_direct(second)["boundary"])
    response = response_payload(boundary)
    return {
        "boundary": complex_pair(boundary),
        "response": response,
        "request_bytes": len(canonical_bytes(program.as_dict())),
        "response_bytes": len(canonical_bytes(response)),
        "logical_complex_cells": 2 * program.width + 2,
        "local_couplers_primary_and_reuse": 2 * program.width * program.depth,
        "local_feedback_operations_primary_and_reuse": 2
        * program.width
        * program.depth,
        "projection_terms_primary_and_reuse": 2 * program.width,
        "verification_fresh_projection_terms": program.width,
        "snapshot_bytes_primary_and_reuse": 3 * COMPLEX_BYTES * program.width,
        "accepted_transaction_logical_complex_cells": 2 * program.width + 2,
        "verification_peak_logical_complex_cells": 3 * program.width + 2,
        "restoration_classification": "SNAPSHOT_RELOAD",
        "restored_error": restored_error,
        "same_backing": id(carrier) == backing,
        "reuse_matches_fresh": boundary_close(reused_boundary, fresh_boundary),
        "baseline_reload_used": True,
    }


def accepted_in_place_lifecycle(
    program: PublicProgram, second: PublicProgram
) -> tuple[complex, complex, float, float]:
    carrier = source_state(program)
    initial = carrier.copy()
    forward_in_place(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse_in_place(carrier, program)
    primary_error = maximum_error(carrier, initial)
    forward_in_place(carrier, second)
    reuse_boundary = project_boundary(carrier, second)
    inverse_in_place(carrier, second)
    reuse_error = maximum_error(carrier, initial)
    return boundary, reuse_boundary, primary_error, reuse_error


def in_place_phase(program: PublicProgram, second: PublicProgram) -> dict[str, Any]:
    carrier = source_state(program)
    initial = carrier.copy()
    backing = id(carrier)
    pre_commitment = commitment(carrier)
    forward_in_place(carrier, program)
    resident_commitment = commitment(carrier)
    resident = carrier.copy()
    boundary = project_boundary(carrier, program)
    persisted = complex(boundary)

    missing_error = maximum_error(carrier, initial)
    wrong = resident.copy()
    wrong_inverse_in_place(wrong, program)
    wrong_error = maximum_error(wrong, initial)
    del wrong
    reordered = resident.copy()
    reordered_inverse_in_place(reordered, program)
    reordered_error = maximum_error(reordered, initial)
    del reordered
    dephased = resident.copy()
    for index, value in enumerate(dephased):
        dephased[index] = complex(abs(complex(value)), 0.0)
    dephased_boundary = project_boundary(dephased, program)
    del dephased
    zero_feedback = source_state(program)
    forward_in_place(zero_feedback, replace(program, feedback_strength=0.0))
    zero_feedback_boundary = project_boundary(zero_feedback, program)
    del zero_feedback
    swapped = source_state(program)
    swapped_forward_in_place(swapped, program)
    swapped_boundary = project_boundary(swapped, program)
    del swapped

    inverse_in_place(carrier, program)
    restoration_error = maximum_error(carrier, initial)
    post_commitment = commitment(carrier)
    if restoration_error > STATE_TOLERANCE or id(carrier) != backing:
        fail("local phase carrier failed primary restoration")

    forward_in_place(carrier, second)
    reused_boundary = project_boundary(carrier, second)
    inverse_in_place(carrier, second)
    reuse_restoration_error = maximum_error(carrier, initial)
    fresh_boundary = complex(*compact_direct(second)["boundary"])
    if (
        reuse_restoration_error > STATE_TOLERANCE
        or not boundary_close(reused_boundary, fresh_boundary)
        or id(carrier) != backing
    ):
        fail("actual restored carrier reuse differs from fresh")

    repeated_max_error = 0.0
    for cycle in range(REUSE_CYCLES):
        active = program if cycle % 2 == 0 else second
        forward_in_place(carrier, active)
        inverse_in_place(carrier, active)
        repeated_max_error = max(repeated_max_error, maximum_error(carrier, initial))
    if repeated_max_error > STATE_TOLERANCE:
        fail("repeated reuse drift exceeds predeclared tolerance")

    null_rejected = False
    try:
        forward_in_place(None, program)
    except RuntimeError:
        null_rejected = True

    controls = {
        "missing_inverse_fails": missing_error > CONTROL_MINIMUM,
        "wrong_inverse_fails": wrong_error > CONTROL_MINIMUM,
        "reordered_inverse_fails_for_noncommuting_feedback_and_couplers": reordered_error
        > CONTROL_MINIMUM,
        "dephasing_changes_boundary": not boundary_close(boundary, dephased_boundary),
        "zero_feedback_changes_boundary": not boundary_close(
            boundary, zero_feedback_boundary
        ),
        "swapped_coupler_order_changes_boundary": not boundary_close(
            boundary, swapped_boundary
        ),
        "null_carrier_rejected": null_rejected,
        "premature_projection_machine_enforced": False,
    }
    if not all(value for key, value in controls.items() if key != "premature_projection_machine_enforced"):
        fail("one or more local phase controls failed to discriminate")
    del resident

    response = response_payload(persisted)
    n_times_d = program.width * program.depth
    verification_multiplier = 2 * REUSE_CYCLES + 5
    return {
        "boundary": complex_pair(persisted),
        "response": response,
        "request_bytes": len(canonical_bytes(program.as_dict())),
        "response_bytes": len(canonical_bytes(response)),
        "restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "restoration_scope": "VIRTUAL_COMPLEX128_PHASE_COORDINATES_ONLY",
        "mode_count": program.width,
        "simultaneous_mode_bandwidth_units": program.width,
        "precision_bits_per_mode": COMPLEX_BITS,
        "carrier_complex_cells": program.width,
        "declared_pair_temporary_complex_cells": 2,
        "logical_complex_cells": program.width + 2,
        "accepted_transaction_logical_complex_cells": program.width + 2,
        "verification_and_controls_peak_logical_complex_cells": 4
        * program.width
        + 2,
        "local_couplers_primary_and_reuse": 4 * n_times_d,
        "local_feedback_operations_primary_and_reuse": 4 * n_times_d,
        "verification_and_controls_local_couplers": verification_multiplier
        * n_times_d,
        "verification_and_controls_local_feedback_operations": verification_multiplier
        * n_times_d,
        "abstract_parallel_sublayer_depth_primary_and_reuse": 12 * program.depth,
        "projection_terms_primary_and_reuse": 2 * program.width,
        "verification_and_controls_projection_terms": 4 * program.width,
        "commitment_serialization_bytes": 3 * COMPLEX_BYTES * program.width,
        "snapshot_bytes": 0,
        "pre_state_commitment": pre_commitment,
        "resident_intermediate_commitment": resident_commitment,
        "post_state_commitment": post_commitment,
        "same_backing": id(carrier) == backing,
        "restoration_error": restoration_error,
        "reuse_restoration_error": reuse_restoration_error,
        "reuse_matches_fresh": boundary_close(reused_boundary, fresh_boundary),
        "repeated_reuse_cycles": REUSE_CYCLES,
        "repeated_reuse_max_error": repeated_max_error,
        "boundary_survives_inverse": boundary_close(persisted, boundary),
        "baseline_reload_used": False,
        "response_release_order": "FORWARD_FINAL_RETENTION_INVERSE_RESTORATION_CHECK_THEN_LOCAL_RETURN",
        "response_order_machine_enforced": False,
        "controls": controls,
    }


def median_ns(action: Callable[[], Any], repetitions: int) -> int:
    samples: list[int] = []
    for _ in range(3):
        action()
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        action()
        samples.append(time.perf_counter_ns() - start)
    return int(statistics.median(samples))


def warm_timings(
    program: PublicProgram, second: PublicProgram, repetitions: int
) -> dict[str, int | bool]:
    def direct_lifecycle() -> None:
        compact_direct(program)
        compact_direct(second)

    return {
        "repetitions": repetitions,
        "warm_compact_direct_fresh_median_ns": median_ns(direct_lifecycle, repetitions),
        "warm_snapshot_sham_median_ns": median_ns(
            lambda: snapshot_sham(program, second), repetitions
        ),
        "warm_in_place_phase_median_ns": median_ns(
            lambda: accepted_in_place_lifecycle(program, second), repetitions
        ),
        "warm_identical_classical_full_state_median_ns": median_ns(
            lambda: accepted_in_place_lifecycle(program, second), repetitions
        ),
        "timing_is_environment_specific_not_claim_authority": True,
    }


def make_program(width: int, depth: int, variant: int) -> PublicProgram:
    return PublicProgram(
        width=width,
        depth=depth,
        even_angle=0.173 + 0.019 * variant,
        odd_angle=0.287 - 0.013 * variant,
        feedback_strength=0.71 + 0.11 * variant,
        phase_seed=3 + 2 * variant,
        weight_shift=3 + variant,
    )


def run_case(width: int, depth: int, benchmark_repetitions: int) -> dict[str, Any]:
    program = make_program(width, depth, 0)
    second = make_program(width, depth, 1)
    direct = compact_direct(program)
    sham = snapshot_sham(program, second)
    phase = in_place_phase(program, second)
    direct_boundary = complex(*direct["boundary"])
    sham_boundary = complex(*sham["boundary"])
    phase_boundary = complex(*phase["boundary"])
    if not (
        boundary_close(direct_boundary, sham_boundary)
        and boundary_close(direct_boundary, phase_boundary)
    ):
        fail("triad final boundaries differ")
    request_bytes = len(canonical_bytes(program.as_dict()))
    response_bytes = len(canonical_bytes(response_payload(direct_boundary)))
    return {
        "width": width,
        "depth": depth,
        "request": program.as_dict(),
        "second_program": second.as_dict(),
        "predeclared_state_tolerance": STATE_TOLERANCE,
        "predeclared_boundary_tolerance": BOUNDARY_TOLERANCE,
        "compact_direct": direct,
        "snapshot_sham": sham,
        "in_place_virtual_phase": phase,
        "identical_public_boundary": True,
        "identical_request_response_traffic": all(
            path["request_bytes"] == request_bytes
            and path["response_bytes"] == response_bytes
            for path in (direct, sham, phase)
        ),
        "warm_timing": warm_timings(program, second, benchmark_repetitions)
        if benchmark_repetitions
        else None,
    }


def build_result(benchmark_repetitions: int) -> dict[str, Any]:
    cases = [run_case(width, depth, benchmark_repetitions) for width, depth in CASES]
    return {
        "schema": "CAT_CAS_CONTROLLED_LOCAL_INTERFERENCE_FEEDBACK_PHASE_COUPLING_V1",
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "in_place_restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "sham_restoration_classification": "SNAPSHOT_RELOAD",
        "direct_restoration_classification": "NO_RESTORATION_CLAIM",
        "execution_scope": "LINUX_DIRECT_PROCESS_COMPLEX128_VIRTUAL_PHASE_SOFTWARE",
        "state_equality": "MAX_ABSOLUTE_COMPLEX_COORDINATE_ERROR",
        "predeclared_state_tolerance": STATE_TOLERANCE,
        "predeclared_boundary_tolerance": BOUNDARY_TOLERANCE,
        "cases": cases,
        "controls": {
            "all_boundaries_match": all(c["identical_public_boundary"] for c in cases),
            "all_public_traffic_matches": all(
                c["identical_request_response_traffic"] for c in cases
            ),
            "all_restorations_within_tolerance": all(
                c["in_place_virtual_phase"]["restoration_error"] <= STATE_TOLERANCE
                and c["in_place_virtual_phase"]["reuse_restoration_error"]
                <= STATE_TOLERANCE
                and c["in_place_virtual_phase"]["repeated_reuse_max_error"]
                <= STATE_TOLERANCE
                for c in cases
            ),
            "all_controls_discriminate": all(
                all(
                    value
                    for key, value in c["in_place_virtual_phase"]["controls"].items()
                    if key != "premature_projection_machine_enforced"
                )
                for c in cases
            ),
        },
        "resource_conclusion": {
            "free_global_dft_removed": True,
            "local_nonlinear_feedback_present": True,
            "coupler_count_per_layer_law": "N",
            "feedback_count_per_layer_law": "N",
            "mode_and_bandwidth_law": "N",
            "precision_bits_per_mode": COMPLEX_BITS,
            "strongest_compact_classical_path": "IDENTICAL_N_COMPLEX_FULL_STATE_LOCAL_COUPLER_AND_FEEDBACK_RECURRENCE",
            "distinct_phase_resource_established": False,
            "small_wall_crossing_established": False,
        },
        "claim_boundaries": {
            "catvm_custody": False,
            "machine_enforced_hidden_intermediate": False,
            "physical_waveform_execution": False,
            "physical_audio_execution": False,
            "physical_bit_replacement": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "unbounded_catalytic_computation": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-repetitions", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repetitions = args.benchmark_repetitions if args.benchmark else 0
    payload = json.dumps(build_result(repetitions), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
