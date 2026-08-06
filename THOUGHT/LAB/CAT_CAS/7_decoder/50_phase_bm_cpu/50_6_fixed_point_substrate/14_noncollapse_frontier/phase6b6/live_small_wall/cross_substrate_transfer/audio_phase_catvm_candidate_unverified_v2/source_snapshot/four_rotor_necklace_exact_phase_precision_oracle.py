#!/usr/bin/env python3
"""Independent finite-field oracle for the exact necklace precision law."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


GRID = 17
ROTORS = 4
NECKLACES = 285
LATENT = 2
DEPTHS = (1, 2, 4, 8, 16, 32, 64)
PRIMARY_VARIANT = 2
PRIMARY_DEPTH = 64
REUSE_VARIANT = 5
REUSE_DEPTH = 23
FIELDS = ((17, 2), (41, 3))
STRIDES = (1, 2, 4, 7)


@dataclass(frozen=True)
class Necklace:
    histogram: tuple[int, ...]
    representative: tuple[int, ...]
    collisions: int


def rotate(histogram: tuple[int, ...], shift: int) -> tuple[int, ...]:
    result = [0] * GRID
    for index, value in enumerate(histogram):
        result[(index + shift) % GRID] = value
    return tuple(result)


def canonical(histogram: tuple[int, ...]) -> tuple[int, ...]:
    return min(rotate(histogram, shift) for shift in range(GRID))


def compile_necklaces() -> tuple[list[Necklace], int]:
    result: list[Necklace] = []
    working = [0] * GRID
    histogram_count = 0

    def visit(position: int, remaining: int) -> None:
        nonlocal histogram_count
        if position == GRID - 1:
            working[position] = remaining
            histogram_count += 1
            histogram = tuple(working)
            if canonical(histogram) != histogram:
                return
            representative = tuple(
                value
                for value, count in enumerate(histogram)
                for _ in range(count)
            )
            collisions = sum(
                count * (count - 1) // 2 for count in histogram
            )
            result.append(
                Necklace(histogram, representative, collisions)
            )
            return
        for value in range(remaining + 1):
            working[position] = value
            visit(position + 1, remaining - value)

    visit(0, ROTORS)
    if histogram_count != math.comb(ROTORS + GRID - 1, ROTORS):
        raise RuntimeError("independent histogram count mismatch")
    if len(result) != NECKLACES:
        raise RuntimeError("independent necklace count mismatch")
    return result, histogram_count


def feature_phase(
    necklace: Necklace,
    variant: int,
    ordinal: int,
    latent: int,
    family: int,
) -> int:
    coordinate = (ordinal + latent + family) % len(
        necklace.representative
    )
    return (
        necklace.collisions
        + family
        + necklace.representative[coordinate]
        + (2 * latent + 1) * variant
        + (3 * family + 1) * ordinal
    ) % 8


def public_matching(
    variant: int, ordinal: int, perturbation: bool
) -> tuple[int, int]:
    stride = STRIDES[(variant + ordinal) % len(STRIDES)]
    offset = (
        11 * variant + 17 * ordinal + int(perturbation)
    ) % NECKLACES
    if math.gcd(stride, NECKLACES) != 1:
        raise RuntimeError("independent matching is not a permutation")
    return offset, stride


def hadamard(
    upper: int,
    lower: int,
    inverse_sqrt2: int,
    prime: int,
) -> tuple[int, int]:
    return (
        (upper + lower) * inverse_sqrt2 % prime,
        (upper - lower) * inverse_sqrt2 % prime,
    )


def apply_diagonal(
    carrier: list[int],
    necklaces: list[Necklace],
    variant: int,
    ordinal: int,
    family: int,
    inverse: bool,
    prime: int,
    root: int,
) -> None:
    for necklace_index, necklace in enumerate(necklaces):
        for latent in range(LATENT):
            exponent = feature_phase(
                necklace, variant, ordinal, latent, family
            )
            if inverse:
                exponent = -exponent
            index = necklace_index * LATENT + latent
            carrier[index] = (
                carrier[index] * pow(root, exponent % 8, prime)
            ) % prime


def apply_latent_hadamards(
    carrier: list[int], inverse_sqrt2: int, prime: int
) -> None:
    for necklace in range(NECKLACES):
        upper = necklace * LATENT
        carrier[upper], carrier[upper + 1] = hadamard(
            carrier[upper],
            carrier[upper + 1],
            inverse_sqrt2,
            prime,
        )


def apply_matching(
    carrier: list[int],
    variant: int,
    ordinal: int,
    perturbation: bool,
    inverse_sqrt2: int,
    prime: int,
) -> None:
    offset, stride = public_matching(variant, ordinal, perturbation)
    for cursor in range(0, NECKLACES - 1, 2):
        upper = (offset + cursor * stride) % NECKLACES
        lower = (offset + (cursor + 1) * stride) % NECKLACES
        for latent in range(LATENT):
            upper_cell = upper * LATENT + latent
            lower_cell = lower * LATENT + latent
            carrier[upper_cell], carrier[lower_cell] = hadamard(
                carrier[upper_cell],
                carrier[lower_cell],
                inverse_sqrt2,
                prime,
            )


def apply_module(
    carrier: list[int],
    necklaces: list[Necklace],
    variant: int,
    ordinal: int,
    inverse: bool,
    prime: int,
    root: int,
    phase_disabled: bool = False,
    perturbation: bool = False,
) -> None:
    inverse_sqrt2 = (
        (root - pow(root, 3, prime))
        * pow(2, prime - 2, prime)
    ) % prime
    if inverse_sqrt2 * inverse_sqrt2 % prime != pow(
        2, prime - 2, prime
    ):
        raise RuntimeError("independent inverse sqrt two mismatch")
    if not inverse:
        if not phase_disabled:
            apply_diagonal(
                carrier,
                necklaces,
                variant,
                ordinal,
                1,
                False,
                prime,
                root,
            )
        apply_latent_hadamards(carrier, inverse_sqrt2, prime)
        apply_matching(
            carrier,
            variant,
            ordinal,
            perturbation,
            inverse_sqrt2,
            prime,
        )
        if not phase_disabled:
            apply_diagonal(
                carrier,
                necklaces,
                variant,
                ordinal,
                2,
                False,
                prime,
                root,
            )
    else:
        if not phase_disabled:
            apply_diagonal(
                carrier,
                necklaces,
                variant,
                ordinal,
                2,
                True,
                prime,
                root,
            )
        apply_matching(
            carrier,
            variant,
            ordinal,
            perturbation,
            inverse_sqrt2,
            prime,
        )
        apply_latent_hadamards(carrier, inverse_sqrt2, prime)
        if not phase_disabled:
            apply_diagonal(
                carrier,
                necklaces,
                variant,
                ordinal,
                1,
                True,
                prime,
                root,
            )


def project(
    carrier: list[int],
    necklaces: list[Necklace],
    prime: int,
    root: int,
) -> list[int]:
    result = [0] * 7
    for necklace_index, necklace in enumerate(necklaces):
        result[necklace.collisions] = (
            result[necklace.collisions]
            + carrier[necklace_index * LATENT]
            + carrier[necklace_index * LATENT + 1]
            * pow(root, necklace.representative[0] % 8, prime)
        ) % prime
    return result


def execute(
    necklaces: list[Necklace],
    variant: int,
    depth: int,
    prime: int,
    root: int,
) -> tuple[list[int], bool, int]:
    carrier = [0] * (NECKLACES * LATENT)
    carrier[0] = 1
    baseline = list(carrier)
    for ordinal in range(depth):
        apply_module(
            carrier,
            necklaces,
            variant,
            ordinal,
            False,
            prime,
            root,
        )
    boundary = project(carrier, necklaces, prime, root)

    conjugate_carrier = [0] * (NECKLACES * LATENT)
    conjugate_carrier[0] = 1
    inverse_root = pow(root, 7, prime)
    for ordinal in range(depth):
        apply_module(
            conjugate_carrier,
            necklaces,
            variant,
            ordinal,
            False,
            prime,
            inverse_root,
        )
    norm = sum(
        left * right
        for left, right in zip(carrier, conjugate_carrier, strict=True)
    ) % prime

    for ordinal in reversed(range(depth)):
        apply_module(
            carrier,
            necklaces,
            variant,
            ordinal,
            True,
            prime,
            root,
        )
    return boundary, carrier == baseline, norm


def verify_field(
    production: dict[str, object],
    necklaces: list[Necklace],
    prime: int,
    root: int,
) -> dict[str, object]:
    if pow(root, 8, prime) != 1 or pow(root, 4, prime) != prime - 1:
        raise RuntimeError("independent root is not primitive order eight")
    key = f"boundary_residues_p{prime}"
    fixtures = []
    production_depths = production["depth_runs"]
    if not isinstance(production_depths, list):
        raise RuntimeError("production depth list missing")
    for expected, depth in zip(production_depths, DEPTHS, strict=True):
        boundary, restored, norm = execute(
            necklaces, PRIMARY_VARIANT, depth, prime, root
        )
        if not isinstance(expected, dict) or boundary != expected[key]:
            raise RuntimeError(
                f"independent boundary mismatch p={prime} depth={depth}"
            )
        if not restored or norm != 1:
            raise RuntimeError(
                f"independent restoration mismatch p={prime} depth={depth}"
            )
        fixtures.append(
            {
                "depth": depth,
                "boundary": boundary,
                "restored": restored,
                "norm": norm,
            }
        )

    primary, primary_restored, primary_norm = execute(
        necklaces,
        PRIMARY_VARIANT,
        PRIMARY_DEPTH,
        prime,
        root,
    )
    reuse, reuse_restored, reuse_norm = execute(
        necklaces,
        REUSE_VARIANT,
        REUSE_DEPTH,
        prime,
        root,
    )
    production_primary = production["primary"]
    production_reuse = production["reuse"]
    if not isinstance(production_primary, dict) or not isinstance(
        production_reuse, dict
    ):
        raise RuntimeError("production primary or reuse missing")
    if primary != production_primary[key] or reuse != production_reuse[key]:
        raise RuntimeError(f"independent accepted path mismatch p={prime}")
    if not primary_restored or not reuse_restored:
        raise RuntimeError(f"independent accepted restoration mismatch p={prime}")
    if primary_norm != 1 or reuse_norm != 1:
        raise RuntimeError(f"independent accepted norm mismatch p={prime}")
    return {
        "prime": prime,
        "primitive_eighth_root": root,
        "fixtures": fixtures,
        "primary_boundary": primary,
        "reuse_boundary": reuse,
        "all_boundaries_match": True,
        "all_restorations_exact": True,
        "all_forward_norms_one": True,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: four_rotor_necklace_exact_phase_precision_oracle.py "
            "PRODUCTION_RESULT"
        )
    production = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    necklaces, histogram_count = compile_necklaces()
    fields = [
        verify_field(production, necklaces, prime, root)
        for prime, root in FIELDS
    ]
    print(
        json.dumps(
            {
                "result": "PASS",
                "oracle": (
                    "INDEPENDENT_DUAL_FINITE_FIELD_PUBLIC_TOPOLOGY_"
                    "RECURRENCE"
                ),
                "histogram_count": histogram_count,
                "necklace_count": len(necklaces),
                "logical_phase_cells": NECKLACES * LATENT,
                "tested_depths": list(DEPTHS),
                "fields": fields,
                "production_backend_imported": False,
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
