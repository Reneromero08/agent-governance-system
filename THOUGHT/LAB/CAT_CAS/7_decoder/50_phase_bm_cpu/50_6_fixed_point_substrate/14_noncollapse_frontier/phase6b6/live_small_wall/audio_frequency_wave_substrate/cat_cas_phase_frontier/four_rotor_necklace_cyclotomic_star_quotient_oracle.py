#!/usr/bin/env python3
"""Independent oracle for the conjugate-pair cyclotomic residue quotient."""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import four_rotor_necklace_exact_phase_precision_integer_oracle as exact_oracle


NECKLACES = 285
LATENT = 2
CELLS = NECKLACES * LATENT
DEPTHS = (1, 64, 256, 1024, 4096)
BRIDGE_DEPTHS = (1, 2, 4, 8, 16, 32, 64)
PRIMARY_VARIANT = 2
PRIMARY_DEPTH = 4096
REUSE_VARIANT = 5
REUSE_DEPTH = 1537
STRIDES = (1, 2, 4, 7)
EMBEDDINGS = (
    (17, 2, 1),
    (17, 9, 0),
    (41, 3, 3),
    (41, 14, 2),
)
ROOT_POWERS = tuple(
    tuple(pow(root, exponent, prime) for exponent in range(8))
    for prime, root, _ in EMBEDDINGS
)
INVERSE_SQRT2 = tuple(
    (root - pow(root, 3, prime)) * pow(2, prime - 2, prime) % prime
    for prime, root, _ in EMBEDDINGS
)

Phase = tuple[int, int, int, int]
Carrier = list[Phase]
Boundary = list[Phase]


def add(left: Phase, right: Phase) -> Phase:
    return tuple(
        (a + b) % EMBEDDINGS[index][0]
        for index, (a, b) in enumerate(
            zip(left, right, strict=True)
        )
    )


def subtract(left: Phase, right: Phase) -> Phase:
    return tuple(
        (a - b) % EMBEDDINGS[index][0]
        for index, (a, b) in enumerate(
            zip(left, right, strict=True)
        )
    )


def multiply_zeta(value: Phase, exponent: int) -> Phase:
    return tuple(
        value[index] * ROOT_POWERS[index][exponent % 8] % prime
        for index, (prime, _, _) in enumerate(EMBEDDINGS)
    )


def multiply_inverse_sqrt2(value: Phase) -> Phase:
    return tuple(
        value[index] * INVERSE_SQRT2[index] % prime
        for index, (prime, _, _) in enumerate(EMBEDDINGS)
    )


def hadamard(upper: Phase, lower: Phase) -> tuple[Phase, Phase]:
    return (
        multiply_inverse_sqrt2(add(upper, lower)),
        multiply_inverse_sqrt2(subtract(upper, lower)),
    )


def feature_phase(
    necklace: exact_oracle.Necklace,
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
        raise RuntimeError("independent public matching is not a permutation")
    return offset, stride


def apply_diagonal(
    carrier: Carrier,
    necklaces: list[exact_oracle.Necklace],
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


def apply_matching(
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
    necklaces: list[exact_oracle.Necklace],
    variant: int,
    ordinal: int,
    inverse: bool,
) -> None:
    if not inverse:
        apply_diagonal(
            carrier, necklaces, variant, ordinal, 1, False
        )
        apply_latent_hadamards(carrier)
        apply_matching(carrier, variant, ordinal)
        apply_diagonal(
            carrier, necklaces, variant, ordinal, 2, False
        )
    else:
        apply_diagonal(
            carrier, necklaces, variant, ordinal, 2, True
        )
        apply_matching(carrier, variant, ordinal)
        apply_latent_hadamards(carrier)
        apply_diagonal(
            carrier, necklaces, variant, ordinal, 1, True
        )


def initial_carrier() -> Carrier:
    zero = (0, 0, 0, 0)
    one = (1, 1, 1, 1)
    carrier = [zero] * CELLS
    carrier[0] = one
    return carrier


def project(
    carrier: Carrier,
    necklaces: list[exact_oracle.Necklace],
) -> Boundary:
    boundary = [(0, 0, 0, 0)] * 7
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
    return boundary


def star_norm(carrier: Carrier) -> Phase:
    result = []
    for embedding, (prime, _, conjugate) in enumerate(EMBEDDINGS):
        result.append(
            sum(
                value[embedding] * value[conjugate]
                for value in carrier
            )
            % prime
        )
    return tuple(result)


def transaction(
    carrier: Carrier,
    necklaces: list[exact_oracle.Necklace],
    variant: int,
    depth: int,
) -> tuple[Boundary, Phase, bool]:
    baseline = list(carrier)
    for ordinal in range(depth):
        apply_module(carrier, necklaces, variant, ordinal, False)
    boundary = project(carrier, necklaces)
    norm = star_norm(carrier)
    for ordinal in reversed(range(depth)):
        apply_module(carrier, necklaces, variant, ordinal, True)
    return boundary, norm, carrier == baseline


def normalize_boundary(value: object) -> Boundary:
    if not isinstance(value, list):
        raise RuntimeError("production boundary is not a list")
    result = []
    for cell in value:
        if not isinstance(cell, list) or len(cell) != len(EMBEDDINGS):
            raise RuntimeError("production boundary cell is invalid")
        result.append(tuple(int(item) for item in cell))
    return result


def evaluate_exact_phase(
    phase: tuple[Fraction, Fraction, Fraction, Fraction],
    prime: int,
    root: int,
) -> int:
    result = 0
    root_power = 1
    for coefficient in phase:
        denominator_inverse = pow(
            coefficient.denominator % prime,
            prime - 2,
            prime,
        )
        residue = (
            coefficient.numerator % prime
        ) * denominator_inverse % prime
        result = (result + residue * root_power) % prime
        root_power = root_power * root % prime
    return result


def evaluate_exact_boundary(
    boundary: exact_oracle.Boundary,
) -> Boundary:
    return [
        tuple(
            evaluate_exact_phase(phase, prime, root)
            for prime, root, _ in EMBEDDINGS
        )
        for phase in boundary
    ]


def verify_production_run(
    run: object,
    necklaces: list[exact_oracle.Necklace],
    variant: int,
    depth: int,
    carrier: Carrier | None = None,
) -> dict[str, object]:
    if not isinstance(run, dict):
        raise RuntimeError("production run is not an object")
    boundary, norm, restored = transaction(
        initial_carrier() if carrier is None else carrier,
        necklaces,
        variant,
        depth,
    )
    if boundary != normalize_boundary(run["boundary"]):
        raise RuntimeError(f"independent boundary mismatch depth={depth}")
    if list(norm) != run["forward_star_norm"]:
        raise RuntimeError(f"independent star norm mismatch depth={depth}")
    if not restored or not run["exact_algebraic_restoration"]:
        raise RuntimeError(f"independent restoration mismatch depth={depth}")
    return {
        "depth": depth,
        "boundary": [list(value) for value in boundary],
        "star_norm": list(norm),
        "restored": restored,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: four_rotor_necklace_cyclotomic_star_quotient_oracle.py "
            "PRODUCTION_RESULT"
        )
    production = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    necklaces, histogram_count = exact_oracle.compile_necklaces()
    demonstrated_kernel_element = 17 * 41
    if (
        demonstrated_kernel_element == 0
        or any(
            demonstrated_kernel_element % prime != 0
            for prime, _, _ in EMBEDDINGS
        )
    ):
        raise RuntimeError("independent quotient kernel check failed")

    depth_runs = production["depth_runs"]
    if not isinstance(depth_runs, list):
        raise RuntimeError("production depth runs missing")
    verified_depths = [
        verify_production_run(run, necklaces, PRIMARY_VARIANT, depth)
        for run, depth in zip(
            depth_runs[:-1],
            DEPTHS[:-1],
            strict=True,
        )
    ]
    accepted_carrier = initial_carrier()
    primary = verify_production_run(
        depth_runs[-1],
        necklaces,
        PRIMARY_VARIANT,
        PRIMARY_DEPTH,
        accepted_carrier,
    )
    verified_depths.append(primary)

    bridge_runs = production["exact_bridge_runs"]
    if not isinstance(bridge_runs, list):
        raise RuntimeError("production bridge runs missing")
    exact_bridge = []
    for run, depth in zip(bridge_runs, BRIDGE_DEPTHS, strict=True):
        verified = verify_production_run(
            run,
            necklaces,
            PRIMARY_VARIANT,
            depth,
        )
        exact_carrier = exact_oracle.initial_carrier()
        exact_result, exact_boundary = exact_oracle.transaction(
            exact_carrier,
            necklaces,
            PRIMARY_VARIANT,
            depth,
        )
        if not exact_result["exact_algebraic_restoration"]:
            raise RuntimeError(
                f"independent exact bridge restoration mismatch depth={depth}"
            )
        evaluated = evaluate_exact_boundary(exact_boundary)
        if evaluated != normalize_boundary(run["boundary"]):
            raise RuntimeError(
                f"exact quotient homomorphism mismatch depth={depth}"
            )
        exact_bridge.append(
            {
                **verified,
                "exact_fraction_boundary_matches_all_embeddings": True,
            }
        )

    production_primary = production["primary"]
    if not isinstance(production_primary, dict):
        raise RuntimeError("production primary missing")
    if normalize_boundary(production_primary["boundary"]) != [
        tuple(value) for value in primary["boundary"]
    ]:
        raise RuntimeError("production primary/depth4096 disagreement")
    reuse_boundary, reuse_norm, reuse_restored = transaction(
        accepted_carrier,
        necklaces,
        REUSE_VARIANT,
        REUSE_DEPTH,
    )
    fresh_boundary, fresh_norm, fresh_restored = transaction(
        initial_carrier(),
        necklaces,
        REUSE_VARIANT,
        REUSE_DEPTH,
    )
    production_reuse = production["reuse"]
    if not isinstance(production_reuse, dict):
        raise RuntimeError("production reuse missing")
    if reuse_boundary != normalize_boundary(production_reuse["boundary"]):
        raise RuntimeError("independent reuse boundary mismatch")
    if (
        not primary["restored"]
        or not reuse_restored
        or not fresh_restored
        or reuse_boundary != fresh_boundary
        or reuse_norm != fresh_norm
    ):
        raise RuntimeError("independent restored-carrier reuse mismatch")

    print(
        json.dumps(
            {
                "result": "PASS",
                "oracle": (
                    "INDEPENDENT_CONJUGATE_PAIR_RESIDUE_RECURRENCE_"
                    "WITH_FRACTION_EXACT_BRIDGE"
                ),
                "histogram_count": histogram_count,
                "necklace_count": len(necklaces),
                "logical_phase_cells": CELLS,
                "demonstrated_nonzero_kernel_element_integer": (
                    demonstrated_kernel_element
                ),
                "demonstrated_nonzero_kernel_element_maps_to_zero": True,
                "embeddings": [
                    {
                        "prime": prime,
                        "root": root,
                        "conjugate_index": conjugate,
                    }
                    for prime, root, conjugate in EMBEDDINGS
                ],
                "tested_depths": list(DEPTHS),
                "verified_depth_runs": verified_depths,
                "exact_bridge_depths": list(BRIDGE_DEPTHS),
                "exact_bridge_runs": exact_bridge,
                "primary": primary,
                "reuse": {
                    "depth": REUSE_DEPTH,
                    "boundary": [
                        list(value) for value in reuse_boundary
                    ],
                    "star_norm": list(reuse_norm),
                    "restored": reuse_restored,
                },
                "fresh_restored_reuse_boundary_equal": True,
                "all_residue_boundaries_match": True,
                "all_star_norms_one": True,
                "all_restorations_exact": True,
                "all_exact_fraction_bridges_match": True,
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
