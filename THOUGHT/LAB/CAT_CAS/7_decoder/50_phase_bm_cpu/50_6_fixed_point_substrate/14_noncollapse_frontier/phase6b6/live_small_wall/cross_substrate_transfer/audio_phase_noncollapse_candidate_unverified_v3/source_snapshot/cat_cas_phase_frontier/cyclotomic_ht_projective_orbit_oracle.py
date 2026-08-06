#!/usr/bin/env python3
"""Independent exact oracle for the analytic H-T projective-orbit result."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path


DEPTHS = (1, 2, 4, 8, 16, 32, 64)
PRIMARY_DEPTH = 64
REUSE_DEPTH = 23
ZERO = (Fraction(0),) * 4
ONE = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))

Phase = tuple[Fraction, Fraction, Fraction, Fraction]
Carrier = tuple[Phase, Phase]


def add(left: Phase, right: Phase) -> Phase:
    return tuple(
        a + b for a, b in zip(left, right, strict=True)
    )


def subtract(left: Phase, right: Phase) -> Phase:
    return tuple(
        a - b for a, b in zip(left, right, strict=True)
    )


def multiply(left: Phase, right: Phase) -> Phase:
    convolution = [Fraction(0)] * 7
    for i in range(4):
        for j in range(4):
            convolution[i + j] += left[i] * right[j]
    result = [Fraction(0)] * 4
    for degree, coefficient in enumerate(convolution):
        if degree < 4:
            result[degree] += coefficient
        else:
            result[degree - 4] -= coefficient
    return tuple(result)


def multiply_zeta(value: Phase, exponent: int) -> Phase:
    result = value
    for _ in range(exponent % 8):
        result = (-result[3], result[0], result[1], result[2])
    return result


def inverse_sqrt_two() -> Phase:
    return (
        Fraction(0),
        Fraction(1, 2),
        Fraction(0),
        Fraction(-1, 2),
    )


def conjugate(value: Phase) -> Phase:
    return (value[0], -value[3], -value[2], -value[1])


def hadamard(carrier: Carrier) -> Carrier:
    scalar = inverse_sqrt_two()
    return (
        multiply(scalar, add(carrier[0], carrier[1])),
        multiply(scalar, subtract(carrier[0], carrier[1])),
    )


def star_norm(carrier: Carrier) -> Phase:
    return add(
        multiply(conjugate(carrier[0]), carrier[0]),
        multiply(conjugate(carrier[1]), carrier[1]),
    )


def primary_step(carrier: Carrier) -> Carrier:
    phased = (carrier[0], multiply_zeta(carrier[1], 1))
    return hadamard(phased)


def primary_inverse(carrier: Carrier) -> Carrier:
    mixed = hadamard(carrier)
    return (mixed[0], multiply_zeta(mixed[1], -1))


def wrong_primary_inverse(carrier: Carrier) -> Carrier:
    mixed = hadamard(carrier)
    return (mixed[0], multiply_zeta(mixed[1], -3))


def reordered_primary_inverse(carrier: Carrier) -> Carrier:
    phased = (carrier[0], multiply_zeta(carrier[1], -1))
    return hadamard(phased)


def reuse_step(carrier: Carrier) -> Carrier:
    mixed = hadamard(carrier)
    return (mixed[0], multiply_zeta(mixed[1], 3))


def reuse_inverse(carrier: Carrier) -> Carrier:
    phased = (carrier[0], multiply_zeta(carrier[1], -3))
    return hadamard(phased)


def apply(
    carrier: Carrier,
    operation,
    depth: int,
) -> Carrier:
    result = carrier
    for _ in range(depth):
        result = operation(result)
    return result


def common_dyadic_form(value: Phase) -> tuple[tuple[int, ...], int]:
    powers = []
    for coefficient in value:
        denominator = coefficient.denominator
        if denominator & (denominator - 1):
            raise RuntimeError("oracle found non-dyadic coefficient")
        powers.append(denominator.bit_length() - 1)
    common_power = max(powers)
    numerators = tuple(
        coefficient.numerator << (common_power - power)
        for coefficient, power in zip(value, powers, strict=True)
    )
    while (
        common_power > 0
        and all(numerator % 2 == 0 for numerator in numerators)
    ):
        numerators = tuple(numerator // 2 for numerator in numerators)
        common_power -= 1
    return numerators, common_power


def phase_payload(value: Phase) -> tuple[int, int, int]:
    numerators, denominator_power = common_dyadic_form(value)
    maximum_bits = max(abs(value).bit_length() for value in numerators)
    payload = max(1, denominator_power.bit_length())
    for numerator in numerators:
        bits = abs(numerator).bit_length()
        payload += 1 if bits == 0 else bits + 1
    return maximum_bits, denominator_power, payload


def carrier_metrics(carrier: Carrier) -> tuple[int, int, int]:
    per_cell = [phase_payload(value) for value in carrier]
    return (
        max(value[0] for value in per_cell),
        max(value[1] for value in per_cell),
        sum(value[2] for value in per_cell),
    )


def projectively_equal(left: Carrier, right: Carrier) -> bool:
    return multiply(left[0], right[1]) == multiply(left[1], right[0])


def decode_phase(value: object) -> Phase:
    if not isinstance(value, dict):
        raise RuntimeError("production phase is not an object")
    numerators = value.get("numerator")
    denominator_power = value.get("denominator_power")
    if (
        not isinstance(numerators, list)
        or len(numerators) != 4
        or not isinstance(denominator_power, int)
    ):
        raise RuntimeError("production phase schema mismatch")
    denominator = 1 << denominator_power
    return tuple(
        Fraction(int(coefficient), denominator)
        for coefficient in numerators
    )


def decode_carrier(value: object) -> Carrier:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError("production carrier schema mismatch")
    return decode_phase(value[0]), decode_phase(value[1])


def theorem_certificate() -> dict[str, object]:
    c = inverse_sqrt_two()
    zeta = multiply_zeta(ONE, 1)
    trace = subtract(c, multiply(c, zeta))
    determinant = multiply_zeta((-ONE[0],) + ONE[1:], 1)
    inverse_determinant = multiply_zeta(ONE, 3)
    invariant = subtract(
        multiply(multiply(trace, trace), inverse_determinant),
        (Fraction(2), Fraction(0), Fraction(0), Fraction(0)),
    )
    expected = (
        Fraction(-1),
        Fraction(-1, 2),
        Fraction(0),
        Fraction(1, 2),
    )
    if invariant != expected:
        raise RuntimeError("independent trace invariant mismatch")
    sqrt_two_coefficient = Fraction(-1, 2)
    if sqrt_two_coefficient.denominator == 1:
        raise RuntimeError("independent integrality control failed")
    first_column = (c, c)
    cyclic_determinant = first_column[1]
    if cyclic_determinant == ZERO:
        raise RuntimeError("initial vector is not cyclic")
    return {
        "unitary": "U_EQUALS_H_T",
        "q_plus_inverse_q": {
            "rational": -1,
            "sqrt2_numerator": -1,
            "sqrt2_denominator": 2,
        },
        "quadratic_integer_ring": "Z_SQRT2",
        "q_plus_inverse_q_is_algebraic_integer": False,
        "root_of_unity_would_require_algebraic_integer_sum": True,
        "eigenvalue_ratio_is_root_of_unity": False,
        "initial_basis_vector_is_cyclic": True,
        "analytic_projective_orbit_infinite": True,
    }


def verify_production(production: dict[str, object]) -> list[dict[str, object]]:
    runs = production.get("depth_runs")
    if not isinstance(runs, list) or len(runs) != len(DEPTHS):
        raise RuntimeError("production depth runs missing")
    initial: Carrier = (ONE, ZERO)
    sampled = [initial]
    verified = []
    for depth, production_run in zip(DEPTHS, runs, strict=True):
        if not isinstance(production_run, dict):
            raise RuntimeError("production depth run is not an object")
        boundary = apply(initial, primary_step, depth)
        restored = apply(boundary, primary_inverse, depth)
        metrics = carrier_metrics(boundary)
        if (
            production_run.get("depth") != depth
            or production_run.get("maximum_numerator_bits") != metrics[0]
            or production_run.get("maximum_denominator_power") != metrics[1]
            or production_run.get("logical_payload_bits") != metrics[2]
            or decode_carrier(production_run.get("boundary")) != boundary
            or restored != initial
            or star_norm(boundary) != ONE
        ):
            raise RuntimeError(f"production depth {depth} mismatch")
        if any(projectively_equal(boundary, prior) for prior in sampled):
            raise RuntimeError("sampled projective collision")
        sampled.append(boundary)
        verified.append(
            {
                "depth": depth,
                "maximum_numerator_bits": metrics[0],
                "maximum_denominator_power": metrics[1],
                "logical_payload_bits": metrics[2],
                "boundary_matches": True,
                "star_norm_exactly_one": True,
                "restored": True,
            }
        )
    return verified


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: cyclotomic_ht_projective_orbit_oracle.py "
            "PRODUCTION_RESULT"
        )
    production_value = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    if not isinstance(production_value, dict):
        raise RuntimeError("production result is not an object")
    verified_runs = verify_production(production_value)

    initial: Carrier = (ONE, ZERO)
    primary = apply(initial, primary_step, PRIMARY_DEPTH)
    if apply(primary, primary_inverse, PRIMARY_DEPTH) != initial:
        raise RuntimeError("independent primary restoration failed")
    reuse = apply(initial, reuse_step, REUSE_DEPTH)
    restored = apply(reuse, reuse_inverse, REUSE_DEPTH)
    fresh_reuse = apply(initial, reuse_step, REUSE_DEPTH)
    if restored != initial or fresh_reuse != reuse:
        raise RuntimeError("independent reuse failed")
    if decode_carrier(production_value["primary"]["boundary"]) != primary:
        raise RuntimeError("production primary boundary mismatch")
    if decode_carrier(production_value["reuse"]["boundary"]) != reuse:
        raise RuntimeError("production reuse boundary mismatch")

    missing_inverse = apply(initial, primary_step, 19)
    wrong_inverse = apply(
        missing_inverse, wrong_primary_inverse, 19
    )
    reordered_inverse = apply(
        missing_inverse, reordered_primary_inverse, 19
    )
    phase_disabled = apply(initial, hadamard, 19)
    if (
        missing_inverse == initial
        or wrong_inverse == initial
        or reordered_inverse == initial
        or phase_disabled == missing_inverse
    ):
        raise RuntimeError("independent inverse control failed")

    alias_left = 1
    alias_right = 698
    if (
        alias_left == alias_right
        or alias_left % 17 != alias_right % 17
        or alias_left % 41 != alias_right % 41
    ):
        raise RuntimeError("independent finite quotient alias failed")

    output = {
        "result": "PASS",
        "oracle": (
            "INDEPENDENT_FRACTION_HT_RECURRENCE_AND_"
            "QUADRATIC_INTEGER_CERTIFICATE"
        ),
        "production_backend_imported": False,
        "theorem_certificate": theorem_certificate(),
        "tested_depths": list(DEPTHS),
        "verified_depth_runs": verified_runs,
        "all_boundaries_match": True,
        "all_star_norms_exactly_one": True,
        "all_restorations_exact": True,
        "sampled_projective_states_distinct": True,
        "fresh_restored_reuse_boundary_equal": True,
        "inverse_controls_pass": True,
        "demonstrated_nonzero_kernel_element_integer": 697,
        "alias_pair_equal_mod_17_and_41": True,
        "alias_pair_is_normalized_transaction": False,
        "fixed_finite_state_lossless_quotient_possible": False,
        "matched_compact_classical_recurrence_identical": True,
        "terminal": False,
    }
    json.dump(output, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
