#!/usr/bin/env python3
"""Independent exact oracle for the M141 shared-scale diagnostic.

The oracle deliberately does not import the M140 matrix-free implementation or
the M141 measurement code.  It executes the older retained 17-by-17 radial
Fourier recurrence, independently compiles the public programs, and implements
division by pi = 1-zeta_17 directly over integer coefficient vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f17_anisotropic_quartic_radial_phase_quotient_closure as retained
import f17_coherent_veronese_phase_chart_closure as exact
import f17_nonlinear_canonical_mps_separator_chart as backend


P = 17
DEPTHS = (1, 2, 4, 8, 16, 32, 64)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
SHELL_COUNTS = (1, *([18] * 16))


def signed_bits(value: int) -> int:
    return 1 if value == 0 else abs(value).bit_length() + 1


def denominator_exponent(value: int) -> int:
    exponent = 0
    while value > 1 and value % P == 0:
        value //= P
        exponent += 1
    if value != 1:
        raise AssertionError("oracle denominator is not a power of 17")
    return exponent


def divide_pi(element: tuple[int, ...]) -> tuple[int, ...]:
    """Return element/(1-zeta_17), failing unless the quotient is integral."""
    if len(element) != 16:
        raise AssertionError("oracle cyclotomic degree changed")
    total = sum(element)
    if total % P:
        raise ValueError("not divisible by pi")
    last = total // P
    prefix = 0
    quotient = []
    for index, coefficient in enumerate(element):
        prefix += coefficient
        quotient.append(prefix - (index + 1) * last)
    if quotient[-1] != last:
        raise AssertionError("oracle pi quotient recurrence changed")
    return tuple(quotient)


def multiply_pi(element: tuple[int, ...]) -> tuple[int, ...]:
    """Multiply by 1-zeta_17 in the power basis modulo Phi_17."""
    last = element[-1]
    return tuple(
        (element[0] if index == 0 else element[index] - element[index - 1])
        + last
        for index in range(16)
    )


def pi_valuation(element: tuple[int, ...]) -> int | None:
    if not any(element):
        return None
    value = element
    valuation = 0
    while sum(value) % P == 0:
        quotient = divide_pi(value)
        if multiply_pi(quotient) != value:
            raise AssertionError("oracle pi division failed reconstruction")
        value = quotient
        valuation += 1
    return valuation


@dataclass(frozen=True)
class Gate:
    quadratic: int
    linear: int
    constant: int

    def exponent(self, shell: int) -> int:
        return (
            self.quadratic * shell * shell
            + self.linear * shell
            + self.constant
        ) % P


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    gates: tuple[Gate, ...]
    observation_quadratic: int
    observation_linear: int


def gate_parameters(index: int, family: str) -> tuple[int, int, int]:
    bit_weight = index.bit_count()
    gray_weight = (index ^ (index >> 1)).bit_count()
    ternary_weight = 0
    remaining = index
    while remaining:
        ternary_weight += remaining % 3
        remaining //= 3
    if family == "PRIMARY":
        values = (
            3 * index + 5 * bit_weight + 1,
            7 * index + 2 * bit_weight + 2,
            11 * index + bit_weight + 4,
        )
    elif family == "REUSE":
        values = (
            5 * index + 2 * ternary_weight + 3,
            4 * index + 3 * ternary_weight + 6,
            9 * index + ternary_weight + 8,
        )
    elif family == "ALTERNATE":
        values = (
            7 * index + 3 * gray_weight + 2,
            8 * index + 2 * gray_weight + 5,
            6 * index + gray_weight + 1,
        )
    else:
        raise ValueError(family)
    return values[0] % P or 1, values[1] % P, values[2] % P


def compile_program(depth: int, family: str) -> Program:
    return Program(
        depth,
        family,
        tuple(Gate(*gate_parameters(index, family)) for index in range(depth)),
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )


def dense_fourier(geometry: retained.CompiledGeometry, state: list[Any]) -> list[Any]:
    alg = geometry.alg
    return [
        sum_field(
            alg,
            [
                alg.mul(geometry.normalized_fourier[target][source], state[source])
                for source in range(P)
            ],
        )
        for target in range(P)
    ]


def sum_field(alg: backend.Algebra, values: list[Any]) -> Any:
    total = alg.zero
    for value in values:
        total = alg.add(total, value)
    return total


def forward(
    geometry: retained.CompiledGeometry, program: Program
) -> list[Any]:
    alg = geometry.alg
    state = [alg.one for _ in range(P)]
    for gate in program.gates:
        state = [
            alg.mul(value, alg.power(gate.exponent(shell)))
            for shell, value in enumerate(state)
        ]
        state = dense_fourier(geometry, state)
    return state


def inverse(
    geometry: retained.CompiledGeometry,
    program: Program,
    state: list[Any],
) -> list[Any]:
    alg = geometry.alg
    for gate in reversed(program.gates):
        state = dense_fourier(geometry, state)
        state = [
            alg.mul(value, alg.power(-gate.exponent(shell)))
            for shell, value in enumerate(state)
        ]
    return state


def boundary(
    geometry: retained.CompiledGeometry,
    program: Program,
    state: list[Any],
) -> Any:
    alg = geometry.alg
    return sum_field(
        alg,
        [
            alg.mul(
                exact.field_integer(alg, SHELL_COUNTS[shell]),
                alg.mul(
                    alg.power(
                        program.observation_quadratic * shell * shell
                        + program.observation_linear * shell
                    ),
                    value,
                ),
            )
            for shell, value in enumerate(state)
        ],
    )


def commitment(alg: backend.Algebra, state: list[Any]) -> str:
    hasher = hashlib.sha256()
    for value in state:
        record = json.dumps(alg.serialize(value), separators=(",", ":")).encode()
        hasher.update(len(record).to_bytes(8, "big"))
        hasher.update(record)
    return hasher.hexdigest()


def metrics(alg: backend.Algebra, state: list[Any]) -> dict[str, int | bool]:
    serialized = [alg.serialize(value) for value in state]
    denominator = 1
    for cell in serialized:
        for _, coefficient_denominator in cell:
            denominator = math.lcm(denominator, coefficient_denominator)
    integral = [
        tuple(
            numerator * (denominator // coefficient_denominator)
            for numerator, coefficient_denominator in cell
        )
        for cell in serialized
    ]
    rational_content = denominator
    for element in integral:
        for coefficient in element:
            rational_content = math.gcd(rational_content, abs(coefficient))
    denominator //= rational_content
    integral = [
        tuple(coefficient // rational_content for coefficient in element)
        for element in integral
    ]
    valuations = [pi_valuation(element) for element in integral if any(element)]
    if not valuations or any(value is None for value in valuations):
        raise AssertionError("oracle state valuation is undefined")
    common_pi = min(int(value) for value in valuations if value is not None)
    residuals = []
    for element in integral:
        residual = element
        for _ in range(common_pi):
            residual = divide_pi(residual)
        restored = residual
        for _ in range(common_pi):
            restored = multiply_pi(restored)
        if restored != element:
            raise AssertionError("oracle common pi extraction changed state")
        residuals.append(residual)
    integer_payload = sum(
        signed_bits(coefficient)
        for element in integral
        for coefficient in element
    )
    return {
        "raw_repeated_denominator_payload_bits": sum(
            alg.payload_bits(value) for value in state
        ),
        "shared_denominator_payload_bits": (
            integer_payload + signed_bits(denominator_exponent(denominator))
        ),
        "common_pi_normalized_payload_bits": (
            sum(
                signed_bits(coefficient)
                for element in residuals
                for coefficient in element
            )
            + signed_bits(denominator_exponent(denominator))
            + signed_bits(common_pi)
        ),
        "denominator_power_of_17": denominator_exponent(denominator),
        "common_pi_exponent": common_pi,
        "minimum_cell_pi_valuation": min(int(value) for value in valuations),
        "maximum_cell_pi_valuation": max(int(value) for value in valuations),
        "maximum_integral_numerator_signed_bits": max(
            signed_bits(coefficient)
            for element in integral
            for coefficient in element
        ),
        "maximum_pi_residual_signed_bits": max(
            signed_bits(coefficient)
            for element in residuals
            for coefficient in element
        ),
        "exact_reconstruction_equal": True,
    }


def execute_case(
    geometry: retained.CompiledGeometry, depth: int, family: str
) -> dict[str, Any]:
    program = compile_program(depth, family)
    state = forward(geometry, program)
    result = {
        "depth": depth,
        "family": family,
        "final_state_commitment": commitment(geometry.alg, state),
        "final_boundary": geometry.alg.serialize(
            boundary(geometry, program, state)
        ),
        **metrics(geometry.alg, state),
    }
    restored = inverse(geometry, program, state)
    result["exact_inverse_seed_equal"] = restored == [
        geometry.alg.one for _ in range(P)
    ]
    return result


def run(package_path: Path) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    geometry = retained.CompiledGeometry.compile(backend.Algebra("Q_ZETA17"))
    oracle_cases = [
        execute_case(geometry, depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    package_cases = {
        (case["family"], case["depth"]): case for case in package["cases"]
    }
    compared_fields = (
        "final_state_commitment",
        "final_boundary",
        "raw_repeated_denominator_payload_bits",
        "shared_denominator_payload_bits",
        "common_pi_normalized_payload_bits",
        "denominator_power_of_17",
        "common_pi_exponent",
        "minimum_cell_pi_valuation",
        "maximum_cell_pi_valuation",
        "maximum_integral_numerator_signed_bits",
        "maximum_pi_residual_signed_bits",
        "exact_reconstruction_equal",
    )
    comparisons = 0
    for oracle in oracle_cases:
        candidate = package_cases[(oracle["family"], oracle["depth"])]
        for field in compared_fields:
            if candidate[field] != oracle[field]:
                raise AssertionError(
                    f"package/oracle mismatch: {oracle['family']} "
                    f"depth {oracle['depth']} field {field}"
                )
            comparisons += 1
        if not oracle["exact_inverse_seed_equal"]:
            raise AssertionError("oracle exact inverse did not restore seed")

    mutated = json.loads(json.dumps(package))
    mutated["cases"][-1]["common_pi_normalized_payload_bits"] += 1
    mutation_detected = any(
        mutated["cases"][-1][field] != oracle_cases[-1][field]
        for field in compared_fields
    )
    if not mutation_detected:
        raise AssertionError("oracle failed payload mutation control")

    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_SHARED_SCALE_HEIGHT_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "oracle_method": (
            "SEPARATE_RETAINED_17_BY_17_EXACT_FOURIER_RECURRENCE_WITH_"
            "INDEPENDENT_PUBLIC_PROGRAM_COMPILER_AND_DIRECT_INTEGER_PI_DIVISION"
        ),
        "case_count": len(oracle_cases),
        "metric_and_boundary_comparisons": comparisons,
        "all_cases_equal": True,
        "all_exact_inverse_seed_checks_pass": True,
        "payload_mutation_detected": mutation_detected,
        "resource_observation": {
            "oracle_retains_exact_fourier_kernel_cells": 289,
            "package_accepted_path_retains_exact_fourier_kernel_cells": 0,
            "oracle_is_verification_only": True,
            "whole_process_and_allocator_memory_excluded": True,
        },
        "claim_ceiling": (
            "INDEPENDENT_EXACT_PARITY_FOR_21_DECLARED_CASES_THROUGH_DEPTH64;"
            "NO_UNIVERSAL_HEIGHT_LOWER_BOUND_OR_OPTIMALITY_OR_ADVANTAGE"
        ),
        "preserved_subclaims": [
            "EXACT_SHARED_SCALE_METRIC_PARITY_FOR_21_DECLARED_CASES",
            "EXACT_FINAL_BOUNDARY_PARITY",
            "EXACT_INVERSE_SEED_RESTORATION_IN_THE_SEPARATE_RECURRENCE",
        ],
        "rejected_interpretations": [
            "ASYMPTOTIC_OR_UNIVERSAL_HEIGHT_LOWER_BOUND",
            "OPTIMAL_EXACT_REPRESENTATION",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.package)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
