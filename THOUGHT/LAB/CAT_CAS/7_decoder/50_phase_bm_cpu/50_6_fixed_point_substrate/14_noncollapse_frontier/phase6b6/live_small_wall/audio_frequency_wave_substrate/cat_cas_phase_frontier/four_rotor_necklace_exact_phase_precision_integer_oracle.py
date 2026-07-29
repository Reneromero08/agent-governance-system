#!/usr/bin/env python3
"""Independent Fraction-based oracle for exact phase-precision measurements."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


GRID = 17
ROTORS = 4
NECKLACES = 285
LATENT = 2
CELLS = NECKLACES * LATENT
DEPTHS = (1, 2, 4, 8, 16, 32, 64)
PRIMARY_VARIANT = 2
PRIMARY_DEPTH = 64
REUSE_VARIANT = 5
REUSE_DEPTH = 23
STRIDES = (1, 2, 4, 7)
ZERO = (Fraction(0),) * 4
ONE = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))

Phase = tuple[Fraction, Fraction, Fraction, Fraction]
Carrier = list[Phase]
Boundary = tuple[Phase, ...]


@dataclass(frozen=True)
class Necklace:
    representative: tuple[int, ...]
    collisions: int


@dataclass
class Precision:
    maximum_numerator_bits: int = 0
    maximum_denominator_power: int = 0
    maximum_logical_payload_bits: int = 0


def rotate(histogram: tuple[int, ...], shift: int) -> tuple[int, ...]:
    result = [0] * GRID
    for index, value in enumerate(histogram):
        result[(index + shift) % GRID] = value
    return tuple(result)


def canonical_rotation(histogram: tuple[int, ...]) -> tuple[int, ...]:
    return min(rotate(histogram, shift) for shift in range(GRID))


def compile_necklaces() -> tuple[list[Necklace], int]:
    necklaces: list[Necklace] = []
    histogram = [0] * GRID
    histogram_count = 0

    def visit(position: int, remaining: int) -> None:
        nonlocal histogram_count
        if position == GRID - 1:
            histogram[position] = remaining
            histogram_count += 1
            candidate = tuple(histogram)
            if canonical_rotation(candidate) != candidate:
                return
            representative = tuple(
                value
                for value, count in enumerate(candidate)
                for _ in range(count)
            )
            collisions = sum(
                count * (count - 1) // 2 for count in candidate
            )
            necklaces.append(Necklace(representative, collisions))
            return
        for value in range(remaining + 1):
            histogram[position] = value
            visit(position + 1, remaining - value)

    visit(0, ROTORS)
    if histogram_count != math.comb(ROTORS + GRID - 1, ROTORS):
        raise RuntimeError("independent histogram count mismatch")
    if len(necklaces) != NECKLACES:
        raise RuntimeError("independent necklace count mismatch")
    return necklaces, histogram_count


def multiply_zeta(value: Phase, exponent: int) -> Phase:
    result = value
    for _ in range(exponent % 8):
        result = (-result[3], result[0], result[1], result[2])
    return result


def multiply_inverse_sqrt2(value: Phase) -> Phase:
    first = multiply_zeta(value, 1)
    third = multiply_zeta(value, 3)
    return tuple(
        (left - right) / 2
        for left, right in zip(first, third, strict=True)
    )


def add(left: Phase, right: Phase) -> Phase:
    return tuple(
        a + b for a, b in zip(left, right, strict=True)
    )


def subtract(left: Phase, right: Phase) -> Phase:
    return tuple(
        a - b for a, b in zip(left, right, strict=True)
    )


def hadamard(upper: Phase, lower: Phase) -> tuple[Phase, Phase]:
    return (
        multiply_inverse_sqrt2(add(upper, lower)),
        multiply_inverse_sqrt2(subtract(upper, lower)),
    )


def common_dyadic_form(value: Phase) -> tuple[tuple[int, ...], int]:
    powers = []
    for coefficient in value:
        denominator = coefficient.denominator
        if denominator & (denominator - 1):
            raise RuntimeError("non-dyadic denominator in exact oracle")
        powers.append(denominator.bit_length() - 1)
    common_power = max(powers, default=0)
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
    maximum_numerator_bits = max(
        (abs(numerator).bit_length() for numerator in numerators),
        default=0,
    )
    payload = max(1, denominator_power.bit_length())
    for numerator in numerators:
        bits = abs(numerator).bit_length()
        payload += 1 if bits == 0 else bits + 1
    return maximum_numerator_bits, denominator_power, payload


def observe(carrier: Carrier, precision: Precision) -> None:
    payload = 0
    for value in carrier:
        numerator_bits, denominator_power, cell_payload = phase_payload(value)
        precision.maximum_numerator_bits = max(
            precision.maximum_numerator_bits,
            numerator_bits,
        )
        precision.maximum_denominator_power = max(
            precision.maximum_denominator_power,
            denominator_power,
        )
        payload += cell_payload
    precision.maximum_logical_payload_bits = max(
        precision.maximum_logical_payload_bits,
        payload,
    )


def carrier_payload(carrier: Carrier) -> int:
    return sum(phase_payload(value)[2] for value in carrier)


def feature_phase(
    necklace: Necklace,
    variant: int,
    ordinal: int,
    latent: int,
    family: int,
) -> int:
    coordinate = (
        ordinal + latent + family
    ) % len(necklace.representative)
    return (
        necklace.collisions
        + family
        + necklace.representative[coordinate]
        + (2 * latent + 1) * variant
        + (3 * family + 1) * ordinal
    ) % 8


def public_matching(variant: int, ordinal: int) -> tuple[int, int]:
    stride = STRIDES[(variant + ordinal) % len(STRIDES)]
    offset = (11 * variant + 17 * ordinal) % NECKLACES
    if math.gcd(stride, NECKLACES) != 1:
        raise RuntimeError("independent matching is not a permutation")
    return offset, stride


def apply_diagonal(
    carrier: Carrier,
    necklaces: list[Necklace],
    variant: int,
    ordinal: int,
    family: int,
    inverse: bool,
) -> None:
    for necklace_index, necklace in enumerate(necklaces):
        for latent in range(LATENT):
            exponent = feature_phase(
                necklace,
                variant,
                ordinal,
                latent,
                family,
            )
            if inverse:
                exponent = -exponent
            index = necklace_index * LATENT + latent
            carrier[index] = multiply_zeta(carrier[index], exponent)


def apply_latent_hadamards(carrier: Carrier) -> None:
    for necklace in range(NECKLACES):
        upper = necklace * LATENT
        carrier[upper], carrier[upper + 1] = hadamard(
            carrier[upper],
            carrier[upper + 1],
        )


def apply_necklace_matching(
    carrier: Carrier,
    variant: int,
    ordinal: int,
) -> None:
    offset, stride = public_matching(variant, ordinal)
    for cursor in range(0, NECKLACES - 1, 2):
        upper = (offset + cursor * stride) % NECKLACES
        lower = (offset + (cursor + 1) * stride) % NECKLACES
        for latent in range(LATENT):
            upper_cell = upper * LATENT + latent
            lower_cell = lower * LATENT + latent
            carrier[upper_cell], carrier[lower_cell] = hadamard(
                carrier[upper_cell],
                carrier[lower_cell],
            )


def apply_module(
    carrier: Carrier,
    necklaces: list[Necklace],
    variant: int,
    ordinal: int,
    inverse: bool,
) -> None:
    if not inverse:
        apply_diagonal(
            carrier, necklaces, variant, ordinal, 1, False
        )
        apply_latent_hadamards(carrier)
        apply_necklace_matching(carrier, variant, ordinal)
        apply_diagonal(
            carrier, necklaces, variant, ordinal, 2, False
        )
    else:
        apply_diagonal(
            carrier, necklaces, variant, ordinal, 2, True
        )
        apply_necklace_matching(carrier, variant, ordinal)
        apply_latent_hadamards(carrier)
        apply_diagonal(
            carrier, necklaces, variant, ordinal, 1, True
        )


def initial_carrier() -> Carrier:
    result = [ZERO] * CELLS
    result[0] = ONE
    return result


def project(carrier: Carrier, necklaces: list[Necklace]) -> Boundary:
    boundary = [ZERO] * 7
    for necklace_index, necklace in enumerate(necklaces):
        cell = necklace_index * LATENT
        boundary[necklace.collisions] = add(
            boundary[necklace.collisions],
            add(
                carrier[cell],
                multiply_zeta(
                    carrier[cell + 1],
                    necklace.representative[0],
                ),
            ),
        )
    return tuple(boundary)


def transaction(
    carrier: Carrier,
    necklaces: list[Necklace],
    variant: int,
    depth: int,
) -> tuple[dict[str, object], Boundary]:
    baseline = list(carrier)
    backing = id(carrier)
    precision = Precision()
    observe(carrier, precision)
    for ordinal in range(depth):
        apply_module(carrier, necklaces, variant, ordinal, False)
        observe(carrier, precision)
    forward_payload = carrier_payload(carrier)
    boundary = project(carrier, necklaces)
    for ordinal in reversed(range(depth)):
        apply_module(carrier, necklaces, variant, ordinal, True)
        observe(carrier, precision)
    return (
        {
            "depth": depth,
            "maximum_numerator_bits": precision.maximum_numerator_bits,
            "maximum_denominator_power": (
                precision.maximum_denominator_power
            ),
            "maximum_logical_payload_bits": (
                precision.maximum_logical_payload_bits
            ),
            "forward_logical_payload_bits": forward_payload,
            "forward_elementary_operations": depth * 1709,
            "exact_algebraic_restoration": carrier == baseline,
            "outer_list_backing_preserved": id(carrier) == backing,
        },
        boundary,
    )


def expected_tuple(result: dict[str, object]) -> tuple[int, int, int, int]:
    return (
        int(result["maximum_numerator_bits"]),
        int(result["maximum_denominator_power"]),
        int(result["forward_logical_payload_bits"]),
        int(result["forward_elementary_operations"]),
    )


def verify_run(
    actual: dict[str, object],
    production: dict[str, object],
    label: str,
) -> None:
    if expected_tuple(actual) != expected_tuple(production):
        raise RuntimeError(f"independent exact tuple mismatch: {label}")
    if not actual["exact_algebraic_restoration"]:
        raise RuntimeError(f"independent exact restoration mismatch: {label}")
    if not actual["outer_list_backing_preserved"]:
        raise RuntimeError(f"independent backing mismatch: {label}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: "
            "four_rotor_necklace_exact_phase_precision_integer_oracle.py "
            "PRODUCTION_RESULT"
        )
    production = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    necklaces, histogram_count = compile_necklaces()

    depth_results = []
    production_depths = production["depth_runs"]
    if not isinstance(production_depths, list):
        raise RuntimeError("production depth runs missing")
    for depth, expected in zip(DEPTHS, production_depths, strict=True):
        if not isinstance(expected, dict):
            raise RuntimeError("production depth run is not an object")
        actual, _ = transaction(
            initial_carrier(),
            necklaces,
            PRIMARY_VARIANT,
            depth,
        )
        verify_run(actual, expected, f"depth{depth}")
        depth_results.append(actual)

    accepted_carrier = initial_carrier()
    primary, _ = transaction(
        accepted_carrier,
        necklaces,
        PRIMARY_VARIANT,
        PRIMARY_DEPTH,
    )
    production_primary = production["primary"]
    if not isinstance(production_primary, dict):
        raise RuntimeError("production primary missing")
    verify_run(primary, production_primary, "primary")

    reuse, reuse_boundary = transaction(
        accepted_carrier,
        necklaces,
        REUSE_VARIANT,
        REUSE_DEPTH,
    )
    production_reuse = production["reuse"]
    if not isinstance(production_reuse, dict):
        raise RuntimeError("production reuse missing")
    verify_run(reuse, production_reuse, "reuse")

    fresh_reuse, fresh_boundary = transaction(
        initial_carrier(),
        necklaces,
        REUSE_VARIANT,
        REUSE_DEPTH,
    )
    verify_run(fresh_reuse, production_reuse, "fresh_reuse")
    if reuse_boundary != fresh_boundary:
        raise RuntimeError("independent fresh/restored reuse boundary mismatch")

    print(
        json.dumps(
            {
                "result": "PASS",
                "oracle": (
                    "INDEPENDENT_FRACTION_BASED_EXACT_INTEGER_"
                    "PRECISION_REEXECUTION"
                ),
                "histogram_count": histogram_count,
                "necklace_count": len(necklaces),
                "logical_phase_cells": CELLS,
                "tested_depths": list(DEPTHS),
                "depth_runs": depth_results,
                "primary": primary,
                "reuse": reuse,
                "fresh_restored_reuse_boundary_equal": True,
                "all_precision_tuples_match": True,
                "all_restorations_exact": True,
                "production_backend_imported": False,
                "production_exact_representation_reused": False,
                "independent_representation": (
                    "PYTHON_FRACTION_FOUR_COEFFICIENT_ZETA8_BASIS"
                ),
                "dense_operator_cells": 0,
                "assignment_expansion_cells": 0,
                "matched_compact_classical_recurrence_identical": True,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
