#!/usr/bin/env python3
"""Standalone oracle for the local interference-feedback phase diagnostic.

The oracle imports neither production nor NumPy.  It compiles an explicit
public operation word, evaluates it on Python complex lists, reverses that
word independently, and reconstructs all final boundaries and controls.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


CASES = ((8, 8), (16, 16), (32, 32), (64, 64))
STATE_TOLERANCE = 2.0e-10
BOUNDARY_TOLERANCE = 2.0e-11
CONTROL_MINIMUM = 1.0e-8
REUSE_CYCLES = 128


@dataclass(frozen=True)
class Program:
    width: int
    depth: int
    even_angle: float
    odd_angle: float
    feedback_strength: float
    phase_seed: int
    weight_shift: int


Operation = tuple[str, int, int, float]


def program(width: int, depth: int, variant: int) -> Program:
    return Program(
        width,
        depth,
        0.173 + 0.019 * variant,
        0.287 - 0.013 * variant,
        0.71 + 0.11 * variant,
        3 + 2 * variant,
        3 + variant,
    )


def source(width: int) -> list[complex]:
    values: list[complex] = []
    for j in range(width):
        radius = 1.0 + 0.19 * math.cos(2.0 * math.pi * (j + 2) / width)
        angle = 2.0 * math.pi * (j * j + 3 * j + 1) / (width + 3)
        values.append(radius * complex(math.cos(angle), math.sin(angle)))
    norm = math.sqrt(sum(abs(value) ** 2 for value in values))
    return [value / norm for value in values]


def layer_values(item: Program, layer: int) -> tuple[float, float, float]:
    even = item.even_angle + 0.003 * ((3 * layer + item.phase_seed) % 11)
    odd = item.odd_angle - 0.002 * ((5 * layer + 2 * item.phase_seed) % 13)
    feedback = item.feedback_strength * (
        1.0 + 0.01 * ((layer + item.phase_seed) % 7)
    )
    return even, odd, feedback


def compile_word(item: Program, swapped: bool = False) -> list[Operation]:
    word: list[Operation] = []
    for layer in range(item.depth):
        even, odd, feedback = layer_values(item, layer)
        sublayers = ((1, odd), (0, even)) if swapped else ((0, even), (1, odd))
        for offset, angle in sublayers:
            for left in range(offset, item.width, 2):
                word.append(("PAIR", left, (left + 1) % item.width, angle))
        for index in range(item.width):
            word.append(("FEEDBACK", index, index, feedback))
    return word


def apply_operation(state: list[complex], operation: Operation) -> None:
    kind, left, right, parameter = operation
    if kind == "PAIR":
        cosine = math.cos(parameter)
        sine = math.sin(parameter)
        a = state[left]
        b = state[right]
        state[left] = cosine * a + 1j * sine * b
        state[right] = 1j * sine * a + cosine * b
    elif kind == "FEEDBACK":
        value = state[left]
        angle = parameter * abs(value) ** 2
        state[left] = value * complex(math.cos(angle), math.sin(angle))
    else:
        raise AssertionError("unknown public operation")


def execute(state: list[complex], word: list[Operation]) -> None:
    for operation in word:
        apply_operation(state, operation)


def reverse(state: list[complex], word: list[Operation]) -> None:
    for kind, left, right, parameter in reversed(word):
        apply_operation(state, (kind, left, right, -parameter))


def reordered_reverse(state: list[complex], item: Program) -> None:
    for layer in reversed(range(item.depth)):
        even, odd, feedback = layer_values(item, layer)
        for offset, angle in ((1, -odd), (0, -even)):
            for left in range(offset, item.width, 2):
                apply_operation(
                    state, ("PAIR", left, (left + 1) % item.width, angle)
                )
        for index in range(item.width):
            apply_operation(state, ("FEEDBACK", index, index, -feedback))


def boundary(state: list[complex], item: Program) -> complex:
    scale = math.sqrt(item.width)
    total = 0j
    for index, value in enumerate(state):
        angle = 2.0 * math.pi * item.weight_shift * index / item.width
        weight = complex(math.cos(angle), math.sin(angle)) / scale
        total += weight.conjugate() * value
    return total


def pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def error(left: list[complex], right: list[complex]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def close(left: complex, right: complex) -> bool:
    return abs(left - right) <= BOUNDARY_TOLERANCE


def run_case(width: int, depth: int) -> dict[str, Any]:
    primary = program(width, depth, 0)
    second = program(width, depth, 1)
    primary_word = compile_word(primary)
    second_word = compile_word(second)
    initial = source(width)
    state = initial.copy()
    execute(state, primary_word)
    resident = state.copy()
    first_boundary = boundary(state, primary)

    missing_error = error(resident, initial)
    wrong = resident.copy()
    wrong_word = compile_word(replace(primary, feedback_strength=primary.feedback_strength + 0.073))
    reverse(wrong, wrong_word)
    wrong_error = error(wrong, initial)
    reordered = resident.copy()
    reordered_reverse(reordered, primary)
    reordered_error = error(reordered, initial)
    dephased_boundary = boundary([complex(abs(v), 0.0) for v in resident], primary)
    zero = initial.copy()
    execute(zero, compile_word(replace(primary, feedback_strength=0.0)))
    zero_boundary = boundary(zero, primary)
    swapped = initial.copy()
    execute(swapped, compile_word(primary, swapped=True))
    swapped_boundary = boundary(swapped, primary)

    reverse(state, primary_word)
    restoration_error = error(state, initial)
    execute(state, second_word)
    second_boundary = boundary(state, second)
    reverse(state, second_word)
    reuse_error = error(state, initial)
    fresh_second = initial.copy()
    execute(fresh_second, second_word)
    fresh_second_boundary = boundary(fresh_second, second)

    repeated_max = 0.0
    for cycle in range(REUSE_CYCLES):
        active = primary_word if cycle % 2 == 0 else second_word
        execute(state, active)
        reverse(state, active)
        repeated_max = max(repeated_max, error(state, initial))

    controls = {
        "missing_inverse_fails": missing_error > CONTROL_MINIMUM,
        "wrong_inverse_fails": wrong_error > CONTROL_MINIMUM,
        "reordered_inverse_fails_for_noncommuting_feedback_and_couplers": reordered_error
        > CONTROL_MINIMUM,
        "dephasing_changes_boundary": not close(first_boundary, dephased_boundary),
        "zero_feedback_changes_boundary": not close(first_boundary, zero_boundary),
        "swapped_coupler_order_changes_boundary": not close(
            first_boundary, swapped_boundary
        ),
    }
    if not all(controls.values()):
        raise AssertionError("oracle control failed")
    if restoration_error > STATE_TOLERANCE or reuse_error > STATE_TOLERANCE:
        raise AssertionError("oracle restoration failed")
    if repeated_max > STATE_TOLERANCE or not close(second_boundary, fresh_second_boundary):
        raise AssertionError("oracle reuse failed")

    return {
        "width": width,
        "depth": depth,
        "boundary": pair(first_boundary),
        "reuse_boundary": pair(second_boundary),
        "restoration_error": restoration_error,
        "reuse_restoration_error": reuse_error,
        "repeated_reuse_cycles": REUSE_CYCLES,
        "repeated_reuse_max_error": repeated_max,
        "norm_error_after_forward": abs(
            sum(abs(value) ** 2 for value in resident)
            - sum(abs(value) ** 2 for value in initial)
        ),
        "compiled_public_operation_count": len(primary_word),
        "local_couplers_per_forward": width * depth,
        "local_feedback_operations_per_forward": width * depth,
        "controls": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [run_case(width, depth) for width, depth in CASES]
    result = {
        "schema": "CAT_CAS_CONTROLLED_LOCAL_INTERFERENCE_FEEDBACK_PHASE_COUPLING_INDEPENDENT_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "oracle": "STANDALONE_PYTHON_COMPLEX_PUBLIC_WORD_NO_PRODUCTION_IMPORT_NO_NUMPY",
        "cases": cases,
        "all_controls_discriminate": all(
            all(case["controls"].values()) for case in cases
        ),
        "all_restorations_within_tolerance": all(
            case["restoration_error"] <= STATE_TOLERANCE
            and case["reuse_restoration_error"] <= STATE_TOLERANCE
            and case["repeated_reuse_max_error"] <= STATE_TOLERANCE
            for case in cases
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
