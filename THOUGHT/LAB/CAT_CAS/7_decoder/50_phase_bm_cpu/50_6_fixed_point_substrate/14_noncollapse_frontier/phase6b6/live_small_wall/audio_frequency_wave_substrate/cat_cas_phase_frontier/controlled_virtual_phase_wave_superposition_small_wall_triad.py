#!/usr/bin/env python3
"""Controlled compact/sham/virtual-wave triad for a phase-native DFT law.

The augmented path declares coherent Fourier propagation as one virtual wave
event, but also reports its growing modes, bandwidth, precision, temporary
state, and actual software FFT work.  The sham restores from a snapshot; the
in-place path executes the numerical inverse on the borrowed phase carrier.
No physical waveform execution or CATVM custody is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


CLAIM = (
    "BOUNDED_CONTROLLED_FOURTEEN_CASE_VIRTUAL_PHASE_WAVE_SUPERPOSITION_"
    "SMALL_WALL_TRIAD_MATCHES_COMPACT_FFT_SNAPSHOT_SHAM_AND_IN_PLACE_"
    "COHERENT_PHASE_BOUNDARIES_WITH_FINAL_ONLY_RESPONSE_NUMERICAL_"
    "INVERSE_RESTORATION_AND_SAME_BACKING_REUSE_BUT_THE_DECLARED_ONE_"
    "EVENT_WAVE_ABSTRACTION_REQUIRES_QMINUS1_COMPLEX_MODES_LINEAR_"
    "BANDWIDTH_128_BITS_PER_MODE_AND_AN_IDENTICAL_SOFTWARE_FFT_SO_NO_"
    "DISTINCT_PHASE_RESOURCE_OR_SMALL_WALL_CROSSING_IS_ESTABLISHED"
)

DECLARED_Q = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
STATE_TOLERANCE = 2.0e-12
BOUNDARY_TOLERANCE = 2.0e-12
REUSE_CYCLES = 256
COMPLEX_BITS = 128
COMPLEX_BYTES = 16


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class PublicRequest:
    q: int
    chirp_strength: int
    weight_shift: int

    def as_dict(self) -> dict[str, int]:
        return {
            "q": self.q,
            "chirp_strength": self.chirp_strength,
            "weight_shift": self.weight_shift,
        }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def primitive_root(prime: int) -> int:
    factors: list[int] = []
    value = prime - 1
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        factors.append(value)
    for candidate in range(2, prime):
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors):
            return candidate
    fail("no primitive root")


def base_phase_state(q: int) -> np.ndarray:
    generator = primitive_root(q)
    orbit = 1
    values: list[complex] = []
    for _ in range(q - 1):
        angle = 2.0 * math.pi * orbit / q
        values.append(complex(math.cos(angle), math.sin(angle)))
        orbit = orbit * generator % q
    return np.asarray(values, dtype=np.complex128)


def chirp(request: PublicRequest) -> np.ndarray:
    width = request.q - 1
    indexes = np.arange(width, dtype=np.float64)
    angles = 2.0 * math.pi * request.chirp_strength * indexes * indexes / (2 * width)
    return np.exp(1j * angles).astype(np.complex128)


def weights(request: PublicRequest) -> np.ndarray:
    width = request.q - 1
    indexes = np.arange(width, dtype=np.float64)
    angles = 2.0 * math.pi * request.weight_shift * indexes / width
    return (np.exp(1j * angles) / math.sqrt(width)).astype(np.complex128)


def forward_in_place(carrier: np.ndarray, request: PublicRequest) -> None:
    carrier *= chirp(request)
    carrier[:] = np.fft.fft(carrier, norm="ortho")


def inverse_in_place(carrier: np.ndarray, request: PublicRequest) -> None:
    carrier[:] = np.fft.ifft(carrier, norm="ortho")
    carrier *= np.conjugate(chirp(request))


def reordered_inverse_in_place(carrier: np.ndarray, request: PublicRequest) -> None:
    carrier *= np.conjugate(chirp(request))
    carrier[:] = np.fft.ifft(carrier, norm="ortho")


def project_boundary(carrier: np.ndarray, request: PublicRequest) -> complex:
    return complex(np.vdot(weights(request), carrier))


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def response_payload(value: complex) -> dict[str, list[float]]:
    return {"final_boundary": complex_pair(value)}


def commitment(carrier: np.ndarray) -> str:
    return hashlib.sha256(carrier.tobytes(order="C")).hexdigest()


def boundary_close(left: complex, right: complex) -> bool:
    return abs(left - right) <= BOUNDARY_TOLERANCE


def compact_direct(request: PublicRequest) -> dict[str, Any]:
    source = base_phase_state(request.q)
    transformed = np.fft.fft(source * chirp(request), norm="ortho")
    boundary = project_boundary(transformed, request)
    response = response_payload(boundary)
    return {
        "boundary": complex_pair(boundary),
        "response": response,
        "request_bytes": len(canonical_bytes(request.as_dict())),
        "response_bytes": len(canonical_bytes(response)),
        "logical_complex_cells": 4 * (request.q - 1),
        "software_fft_calls": 1,
        "snapshot_bytes": 0,
        "restoration_classification": "NO_RESTORATION_CLAIM",
    }


def snapshot_sham(request: PublicRequest, second: PublicRequest) -> dict[str, Any]:
    carrier = base_phase_state(request.q)
    backing = id(carrier)
    snapshot = carrier.copy()
    forward_in_place(carrier, request)
    boundary = project_boundary(carrier, request)
    carrier[:] = snapshot
    restored_error = float(np.max(np.abs(carrier - snapshot)))
    forward_in_place(carrier, second)
    reused_boundary = project_boundary(carrier, second)
    carrier[:] = snapshot
    fresh = compact_direct(second)
    response = response_payload(boundary)
    return {
        "boundary": complex_pair(boundary),
        "response": response,
        "request_bytes": len(canonical_bytes(request.as_dict())),
        "response_bytes": len(canonical_bytes(response)),
        "logical_complex_cells": 4 * (request.q - 1),
        "software_fft_calls_primary_and_reuse": 2,
        "snapshot_bytes_primary_and_reuse": 3 * COMPLEX_BYTES * (request.q - 1),
        "verification_only_fresh_baseline_fft_calls": 1,
        "restoration_classification": "SNAPSHOT_RELOAD",
        "restored_error": restored_error,
        "same_backing": id(carrier) == backing,
        "reuse_matches_fresh": boundary_close(
            reused_boundary, complex(*fresh["boundary"])
        ),
        "baseline_reload_used": True,
    }


def in_place_phase(request: PublicRequest, second: PublicRequest) -> dict[str, Any]:
    carrier = base_phase_state(request.q)
    initial = carrier.copy()
    backing = id(carrier)
    pre_commitment = commitment(carrier)
    forward_in_place(carrier, request)
    resident_commitment = commitment(carrier)
    boundary = project_boundary(carrier, request)
    persisted = complex(boundary)
    inverse_in_place(carrier, request)
    restoration_error = float(np.max(np.abs(carrier - initial)))
    post_commitment = commitment(carrier)
    if restoration_error > STATE_TOLERANCE or id(carrier) != backing:
        fail("in-place phase carrier failed primary restoration")

    forward_in_place(carrier, second)
    reused_boundary = project_boundary(carrier, second)
    inverse_in_place(carrier, second)
    reuse_restoration_error = float(np.max(np.abs(carrier - initial)))
    fresh_boundary = complex(*compact_direct(second)["boundary"])
    if (
        reuse_restoration_error > STATE_TOLERANCE
        or not boundary_close(reused_boundary, fresh_boundary)
        or id(carrier) != backing
    ):
        fail("actual restored carrier reuse differs from fresh")

    repeated_max_error = 0.0
    for cycle in range(REUSE_CYCLES):
        active = request if cycle % 2 == 0 else second
        forward_in_place(carrier, active)
        inverse_in_place(carrier, active)
        repeated_max_error = max(
            repeated_max_error, float(np.max(np.abs(carrier - initial)))
        )
    if repeated_max_error > STATE_TOLERANCE:
        fail("repeated phase reuse exceeded predeclared tolerance")

    missing = initial.copy()
    forward_in_place(missing, request)
    missing_inverse_fails = float(np.max(np.abs(missing - initial))) > STATE_TOLERANCE
    wrong = missing.copy()
    inverse_in_place(wrong, second)
    wrong_inverse_fails = float(np.max(np.abs(wrong - initial))) > STATE_TOLERANCE
    reordered = missing.copy()
    reordered_inverse_in_place(reordered, request)
    reordered_inverse_fails = (
        float(np.max(np.abs(reordered - initial))) > STATE_TOLERANCE
    )
    dephased = initial.copy()
    forward_in_place(dephased, request)
    dephased[:] = np.abs(dephased).astype(np.complex128)
    dephased_boundary = project_boundary(dephased, request)
    dephasing_changes_boundary = not boundary_close(dephased_boundary, boundary)
    null_carrier_rejected = False
    try:
        forward_in_place(np.asarray([], dtype=np.complex128), request)
    except (ValueError, RuntimeError):
        null_carrier_rejected = True
    if not all(
        (
            missing_inverse_fails,
            wrong_inverse_fails,
            reordered_inverse_fails,
            dephasing_changes_boundary,
            null_carrier_rejected,
        )
    ):
        fail("virtual phase mutation control failed")

    response = response_payload(boundary)
    return {
        "boundary": complex_pair(boundary),
        "response": response,
        "request_bytes": len(canonical_bytes(request.as_dict())),
        "response_bytes": len(canonical_bytes(response)),
        "logical_complex_cells": 3 * (request.q - 1),
        "carrier_complex_cells": request.q - 1,
        "declared_chirp_and_transform_temporary_complex_cells": 2
        * (request.q - 1),
        "carrier_bits": COMPLEX_BITS * (request.q - 1),
        "mode_count": request.q - 1,
        "simultaneous_mode_bandwidth_units": request.q - 1,
        "precision_bits_per_mode": COMPLEX_BITS,
        "abstract_wave_events_primary_and_reuse": 4,
        "software_fft_calls_primary_and_reuse": 4,
        "verification_and_controls_software_fft_calls": 517,
        "failed_null_carrier_fft_attempts": 1,
        "snapshot_bytes": 0,
        "restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "pre_state_commitment": pre_commitment,
        "resident_intermediate_commitment": resident_commitment,
        "post_state_commitment": post_commitment,
        "boundary_survives_inverse": boundary_close(persisted, boundary),
        "same_backing": id(carrier) == backing,
        "restoration_error": restoration_error,
        "reuse_restoration_error": reuse_restoration_error,
        "repeated_reuse_cycles": REUSE_CYCLES,
        "repeated_reuse_max_error": repeated_max_error,
        "reuse_matches_fresh": boundary_close(reused_boundary, fresh_boundary),
        "baseline_reload_used": False,
        "controls": {
            "missing_inverse_fails": missing_inverse_fails,
            "wrong_inverse_fails": wrong_inverse_fails,
            "reordered_inverse_fails_for_noncommuting_chirp_and_dft": reordered_inverse_fails,
            "dephasing_changes_boundary": dephasing_changes_boundary,
            "null_carrier_rejected": null_carrier_rejected,
            "premature_projection_machine_enforced": False,
        },
    }


def median_ns(function: Callable[[], Any], repetitions: int) -> int:
    for _ in range(8):
        function()
    samples: list[int] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    return int(statistics.median(samples))


def compact_accepted_lifecycle(
    request: PublicRequest, second: PublicRequest
) -> tuple[complex, complex]:
    source = base_phase_state(request.q)
    first_state = np.fft.fft(source * chirp(request), norm="ortho")
    first = project_boundary(first_state, request)
    reuse_state = np.fft.fft(source * chirp(second), norm="ortho")
    reuse = project_boundary(reuse_state, second)
    return first, reuse


def snapshot_accepted_lifecycle(
    request: PublicRequest, second: PublicRequest
) -> tuple[complex, complex]:
    carrier = base_phase_state(request.q)
    snapshot = carrier.copy()
    forward_in_place(carrier, request)
    first = project_boundary(carrier, request)
    carrier[:] = snapshot
    forward_in_place(carrier, second)
    reuse = project_boundary(carrier, second)
    carrier[:] = snapshot
    return first, reuse


def in_place_accepted_lifecycle(
    request: PublicRequest, second: PublicRequest
) -> tuple[complex, complex]:
    carrier = base_phase_state(request.q)
    forward_in_place(carrier, request)
    first = project_boundary(carrier, request)
    inverse_in_place(carrier, request)
    forward_in_place(carrier, second)
    reuse = project_boundary(carrier, second)
    inverse_in_place(carrier, second)
    return first, reuse


def transaction_case(q: int, benchmark_repetitions: int) -> dict[str, Any]:
    request = PublicRequest(q, 3 % (q - 1), 5 % (q - 1))
    second = PublicRequest(
        q,
        (request.chirp_strength + 1) % (q - 1),
        (request.weight_shift + 2) % (q - 1),
    )
    direct = compact_direct(request)
    direct_reuse = compact_direct(second)
    direct["fresh_reuse_boundary"] = direct_reuse["boundary"]
    direct["software_fft_calls_primary_and_reuse"] = 2
    sham = snapshot_sham(request, second)
    augmented = in_place_phase(request, second)
    direct_boundary = complex(*direct["boundary"])
    sham_boundary = complex(*sham["boundary"])
    augmented_boundary = complex(*augmented["boundary"])
    if not (
        boundary_close(direct_boundary, sham_boundary)
        and boundary_close(direct_boundary, augmented_boundary)
        and direct["request_bytes"]
        == sham["request_bytes"]
        == augmented["request_bytes"]
        and direct["response_bytes"]
        == sham["response_bytes"]
        == augmented["response_bytes"]
    ):
        fail("triad paths differ in boundary or public traffic")

    timing = None
    if benchmark_repetitions:
        timing = {
            "repetitions": benchmark_repetitions,
            "warm_compact_direct_median_ns": median_ns(
                lambda: compact_accepted_lifecycle(request, second),
                benchmark_repetitions,
            ),
            "warm_snapshot_sham_median_ns": median_ns(
                lambda: snapshot_accepted_lifecycle(request, second),
                benchmark_repetitions,
            ),
            "warm_in_place_phase_median_ns": median_ns(
                lambda: in_place_accepted_lifecycle(request, second),
                benchmark_repetitions,
            ),
            "warm_identical_classical_in_place_median_ns": median_ns(
                lambda: in_place_accepted_lifecycle(request, second),
                benchmark_repetitions,
            ),
            "timing_is_environment_specific_not_claim_authority": True,
        }
    return {
        "q": q,
        "width": q - 1,
        "request": request.as_dict(),
        "second_program": second.as_dict(),
        "predeclared_state_tolerance": STATE_TOLERANCE,
        "predeclared_boundary_tolerance": BOUNDARY_TOLERANCE,
        "compact_direct": direct,
        "snapshot_sham": sham,
        "in_place_virtual_phase": augmented,
        "identical_public_boundary": True,
        "identical_request_response_traffic": True,
        "warm_timing": timing,
    }


def build_result(benchmark_repetitions: int) -> dict[str, Any]:
    cases = [transaction_case(q, benchmark_repetitions) for q in DECLARED_Q]
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "in_place_restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "sham_restoration_classification": "SNAPSHOT_RELOAD",
        "direct_restoration_classification": "NO_RESTORATION_CLAIM",
        "claim_ceiling": (
            "FOURTEEN_DECLARED_Q5_THROUGH_Q53_NUMPY_COMPLEX128_DIRECT_"
            "PROCESS_VIRTUAL_PHASE_CASES_WITH_256_CYCLE_REUSE_AND_"
            "PREDECLARED_2E_MINUS12_STATE_AND_BOUNDARY_TOLERANCES"
        ),
        "triad_cases": cases,
        "resource_accounting": {
            "compact_direct": "4H_CONSERVATIVE_LOGICAL_COMPLEX_CELLS_TWO_SOFTWARE_FFTS_FOR_PRIMARY_AND_FRESH_REUSE_NO_RESTORATION",
            "snapshot_sham": "4H_CONSERVATIVE_LOGICAL_COMPLEX_CELLS_TWO_FORWARD_SOFTWARE_FFTS_AND3H_COMPLEX_BYTES_OF_ONE_SNAPSHOT_CREATE_PLUS_TWO_RELOADS_FOR_PRIMARY_AND_REUSE",
            "in_place_virtual_phase": "H_CARRIER_PLUS2H_DECLARED_CHIRP_AND_TRANSFORM_TEMPORARIES_128H_CARRIER_BITS_H_SIMULTANEOUS_MODE_BANDWIDTH_FOUR_ABSTRACT_WAVE_EVENTS_AND_FOUR_ACTUAL_SOFTWARE_FFTS_FOR_PRIMARY_AND_REUSE",
            "strongest_compact_classical": "IDENTICAL_3H_IN_PLACE_NUMERICAL_FFT_RECURRENCE_WITH_THE_SAME_FOUR_CALLS_RESTORATION_AND_REUSE_LAW_AS_THE_VIRTUAL_PHASE_PATH",
            "controller_backend_traffic": "IDENTICAL_CANONICAL_REQUEST_AND_FINAL_RESPONSE_BYTES_ACROSS_TRIAD",
            "compiled_plan": "PUBLIC_Q_CHIRP_STRENGTH_WEIGHT_SHIFT_GENERATE_PHASES_PROCEDURALLY_ZERO_RETAINED_PLAN_ARRAYS",
            "numpy_internal_allocator_and_fft_workspace_fully_measured": False,
            "whole_process_memory_fully_measured": False,
        },
        "controls": {
            "all_boundaries_match": True,
            "all_public_traffic_matches": True,
            "all_actual_inverse_restorations_within_tolerance": True,
            "all_actual_restored_reuses_match_fresh": True,
            "all_256_cycle_reuses_within_tolerance": True,
            "all_missing_inverses_fail": True,
            "all_wrong_inverses_fail": True,
            "all_applicable_reordered_inverses_fail": True,
            "all_dephased_boundaries_change": True,
            "all_null_carriers_rejected": True,
            "all_shams_use_snapshot_reload": True,
        },
        "small_wall_decision": {
            "crossing_established": False,
            "distinct_phase_resource_established": False,
            "reason": "THE_ONE_EVENT_ABSTRACTION_REQUIRES_LINEAR_MODES_LINEAR_BANDWIDTH_FIXED_PRECISION_PER_MODE_AND_IS_EMULATED_BY_THE_IDENTICAL_COMPACT_SOFTWARE_FFT",
            "coherence_causally_relevant": True,
            "coherence_advantage_over_compact_classical_software": False,
        },
        "strict_boundaries": {
            "catvm_custody": False,
            "machine_enforced_hidden_intermediate": False,
            "physical_waveform_execution": False,
            "physical_audio_execution": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
        "next_obstruction": (
            "VIRTUAL_COHERENT_SUPERPOSITION_IS_CAUSALLY_RELEVANT_BUT_"
            "ITS_ONE_EVENT_ABSTRACTION_HIDES_LINEAR_MODES_BANDWIDTH_AND_"
            "PRECISION_WHILE_THE_SOFTWARE_BACKEND_IS_THE_IDENTICAL_FFT_"
            "SO_A_SUCCESSOR_NEEDS_A_PHASE_COUPLING_LAW_WITH_MEASURED_"
            "RESOURCE_NOT_ALREADY_PRESENT_IN_THE_MATCHED_CLASSICAL_MODEL"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-repetitions", type=int, default=0)
    arguments = parser.parse_args()
    text = json.dumps(
        build_result(arguments.benchmark_repetitions), indent=2, sort_keys=True
    ) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
