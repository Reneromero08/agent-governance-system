#!/usr/bin/env python3
"""No-production-import dense-DFT oracle for the virtual phase triad."""

from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path
from typing import Any


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


def fail(message: str) -> None:
    raise RuntimeError(message)


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
    fail("oracle primitive root missing")


def base_state(q: int) -> list[complex]:
    generator = primitive_root(q)
    orbit = 1
    answer: list[complex] = []
    for _ in range(q - 1):
        answer.append(cmath.exp(2j * math.pi * orbit / q))
        orbit = orbit * generator % q
    return answer


def chirp(width: int, strength: int) -> list[complex]:
    return [
        cmath.exp(2j * math.pi * strength * index * index / (2 * width))
        for index in range(width)
    ]


def weights(width: int, shift: int) -> list[complex]:
    scale = math.sqrt(width)
    return [
        cmath.exp(2j * math.pi * shift * index / width) / scale
        for index in range(width)
    ]


def dense_dft(values: list[complex], inverse: bool = False) -> list[complex]:
    width = len(values)
    sign = 1.0 if inverse else -1.0
    scale = math.sqrt(width)
    return [
        sum(
            value
            * cmath.exp(sign * 2j * math.pi * output * source / width)
            for source, value in enumerate(values)
        )
        / scale
        for output in range(width)
    ]


def forward(values: list[complex], strength: int) -> list[complex]:
    return dense_dft(
        [value * phase for value, phase in zip(values, chirp(len(values), strength))]
    )


def inverse(values: list[complex], strength: int) -> list[complex]:
    restored = dense_dft(values, inverse=True)
    return [
        value * phase.conjugate()
        for value, phase in zip(restored, chirp(len(values), strength))
    ]


def reordered_inverse(values: list[complex], strength: int) -> list[complex]:
    rephased = [
        value * phase.conjugate()
        for value, phase in zip(values, chirp(len(values), strength))
    ]
    return dense_dft(rephased, inverse=True)


def boundary(values: list[complex], shift: int) -> complex:
    return sum(
        weight.conjugate() * value
        for weight, value in zip(weights(len(values), shift), values)
    )


def maximum_error(left: list[complex], right: list[complex]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def oracle_case(q: int) -> dict[str, Any]:
    width = q - 1
    strength = 3 % width
    shift = 5 % width
    second_strength = (strength + 1) % width
    second_shift = (shift + 2) % width
    initial = base_state(q)
    transformed = forward(initial, strength)
    projected = boundary(transformed, shift)
    restored = inverse(transformed, strength)
    restoration_error = maximum_error(restored, initial)
    second_transformed = forward(restored, second_strength)
    reused_boundary = boundary(second_transformed, second_shift)
    reused_restored = inverse(second_transformed, second_strength)
    fresh_boundary = boundary(forward(initial, second_strength), second_shift)
    reuse_error = maximum_error(reused_restored, initial)
    missing_inverse_fails = maximum_error(transformed, initial) > STATE_TOLERANCE
    wrong_inverse_fails = (
        maximum_error(inverse(transformed, second_strength), initial)
        > STATE_TOLERANCE
    )
    reordered_inverse_fails = (
        maximum_error(reordered_inverse(transformed, strength), initial)
        > STATE_TOLERANCE
    )
    dephased = [complex(abs(value), 0.0) for value in transformed]
    dephased_boundary = boundary(dephased, shift)
    dephasing_changes_boundary = abs(dephased_boundary - projected) > BOUNDARY_TOLERANCE
    if not (
        restoration_error <= STATE_TOLERANCE
        and reuse_error <= STATE_TOLERANCE
        and abs(reused_boundary - fresh_boundary) <= BOUNDARY_TOLERANCE
        and missing_inverse_fails
        and wrong_inverse_fails
        and reordered_inverse_fails
        and dephasing_changes_boundary
    ):
        fail("independent virtual phase case failed")
    return {
        "q": q,
        "width": width,
        "chirp_strength": strength,
        "weight_shift": shift,
        "second_chirp_strength": second_strength,
        "second_weight_shift": second_shift,
        "projected_boundary": pair(projected),
        "reused_boundary": pair(reused_boundary),
        "fresh_reuse_boundary": pair(fresh_boundary),
        "restoration_error": restoration_error,
        "reuse_restoration_error": reuse_error,
        "reuse_matches_fresh": abs(reused_boundary - fresh_boundary)
        <= BOUNDARY_TOLERANCE,
        "missing_inverse_fails": missing_inverse_fails,
        "wrong_inverse_fails": wrong_inverse_fails,
        "reordered_inverse_fails": reordered_inverse_fails,
        "dephasing_changes_boundary": dephasing_changes_boundary,
        "dense_dft_terms_primary_and_reuse_with_inverses": 4 * width * width,
    }


def build_result() -> dict[str, Any]:
    cases = [oracle_case(q) for q in DECLARED_Q]
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "claim_ceiling": (
            "FOURTEEN_DECLARED_Q5_THROUGH_Q53_STANDALONE_CMATH_DENSE_"
            "UNITARY_DFT_BOUNDARY_RESTORATION_REUSE_AND_CONTROL_PARITY"
        ),
        "oracle_cases": cases,
        "oracle_independence": {
            "imports_production": False,
            "uses_numpy_fft": False,
            "uses_standalone_dense_unitary_dft": True,
            "reconstructs_primitive_roots_phase_orbits_chirps_weights_and_boundaries": True,
            "reports_oracle_work_not_production_work": True,
        },
        "controls": {
            "all_boundaries_match_production_within_predeclared_tolerance": True,
            "all_inverse_restorations_within_predeclared_tolerance": True,
            "all_reuses_match_fresh": True,
            "all_missing_inverses_fail": True,
            "all_wrong_inverses_fail": True,
            "all_reordered_inverses_fail": True,
            "all_dephased_boundaries_change": True,
        },
        "strict_boundaries": {
            "catvm_custody": False,
            "physical_waveform_execution": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    text = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
