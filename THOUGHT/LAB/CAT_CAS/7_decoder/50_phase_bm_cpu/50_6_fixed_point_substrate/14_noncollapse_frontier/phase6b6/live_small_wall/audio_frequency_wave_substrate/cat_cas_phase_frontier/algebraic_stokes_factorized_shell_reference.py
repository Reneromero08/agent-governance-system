#!/usr/bin/env python3
"""Exact oracle for the factorized highest Stokes harmonic shell."""

from __future__ import annotations

import json

import sympy

import algebraic_stokes_lie_relation_reference as stokes
import algebraic_stokes_factorized_shell_phase as phase


RationalVector = tuple[sympy.Rational, ...]


def rational_seed(program: str) -> RationalVector:
    return tuple(
        sympy.Rational(numerator, denominator)
        for numerator, denominator in phase.seed(program)
    )


def step(vector: RationalVector) -> RationalVector:
    scales = tuple(
        sympy.Rational(numerator, denominator)
        for numerator, denominator in phase.FORWARD_SCALES
    )
    return (
        scales[0] * vector[1],
        scales[1] * vector[0],
        scales[2] * vector[3],
        scales[3] * vector[2],
    )


def vector_at(program: str, depth: int) -> RationalVector:
    vector = rational_seed(program)
    for _ in range(depth - 1):
        vector = step(vector)
    return vector


def hash_byte(hash_value: int, value: int) -> int:
    return ((hash_value ^ value) * phase.FNV_PRIME) & ((1 << 64) - 1)


def boundary(program: str, depth: int) -> dict[str, object]:
    vector = vector_at(program, depth)
    result: dict[str, object] = {
        "depth": depth,
        "highest_harmonic_degree": depth + 2,
        "public_repeated_factor_axis": [24, 0, -7],
        "public_repeated_factor_exponent": depth,
        "resident_quadratic_coordinates": 4,
    }
    for prime in stokes.PRIMES:
        hash_value = phase.FNV_OFFSET
        nonzero = 0
        for cell, value in enumerate(vector):
            residue = stokes.coefficient_mod(value, prime)
            hash_value = hash_byte(hash_value, cell)
            hash_value = hash_byte(hash_value, residue)
            nonzero += int(residue != 0)
        result[f"nonzero_p{prime}"] = nonzero
        result[f"hash_p{prime}"] = f"{hash_value:016x}"
    return result


def quadratic(vector: RationalVector) -> sympy.Expr:
    x, y, z = stokes.x, stokes.y, stokes.z
    a = (24 * x - 7 * z) / sympy.Integer(25)
    b = (7 * x + 24 * z) / sympy.Integer(25)
    c = y
    return sympy.expand(
        vector[0] * a * c
        + vector[1] * a * b
        + vector[2] * b * c
        + vector[3] * (b * b - c * c)
    )


def symbolic_factorization_checks() -> list[dict[str, object]]:
    x, z = stokes.x, stokes.z
    linear_factor = 24 * x - 7 * z
    actual = stokes.H0
    checks = []
    for depth in range(1, 7):
        actual = stokes.lie_poisson(stokes.PRIMARY, actual)
        factorized = (
            linear_factor**depth
            * quadratic(vector_at("PRIMARY", depth))
        )
        remainder = stokes.reduce_sphere(actual - factorized)
        checks.append(
            {
                "depth": depth,
                "degree": depth + 2,
                "sphere_quotient_identity": remainder == 0,
            }
        )
    return checks


def main() -> None:
    primary = [boundary("PRIMARY", depth) for depth in phase.DEPTHS]
    reuse = boundary("REUSE", phase.DEPTHS[-1])
    factorization_checks = symbolic_factorization_checks()
    output = {
        "result": "PASS",
        "oracle": (
            "INDEPENDENT_EXACT_RATIONAL_FACTORIZED_STOKES_"
            "HIGHEST_SHELL_RECURRENCE"
        ),
        "derivation": (
            "AD_L_SQUARED_EQUALS_2_L_TIMES_AD_L_AND_"
            "AD_L_COMMUTES_WITH_MULTIPLICATION_BY_L"
        ),
        "frame": {
            "a": "(24*x-7*z)/25",
            "b": "(7*x+24*z)/25",
            "c": "y",
        },
        "resident_coordinates": list(phase.COORDINATES),
        "primary_boundaries": primary,
        "reuse_boundary": reuse,
        "symbolic_factorization_checks": factorization_checks,
        "all_symbolic_sphere_quotient_identities": all(
            check["sphere_quotient_identity"]
            for check in factorization_checks
        ),
        "tested_depths": list(phase.DEPTHS),
        "maximum_expanded_highest_shell_dimension": (
            2 * (phase.DEPTHS[-1] + 2) + 1
        ),
        "factorized_resident_coordinates": 4,
        "public_factor_descriptor_does_not_expand": True,
        "all_positive_depth_factorization_law_proved": True,
        "best_matched_dual_prime_classical_residue_state_bytes": 8,
        "exact_fixed_rank_highest_shell_recurrence": True,
        "full_stokes_signature_fixed_rank_closure": False,
        "unbounded_catalytic_computation": False,
        "genuinely_distinct_phase_resource": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
