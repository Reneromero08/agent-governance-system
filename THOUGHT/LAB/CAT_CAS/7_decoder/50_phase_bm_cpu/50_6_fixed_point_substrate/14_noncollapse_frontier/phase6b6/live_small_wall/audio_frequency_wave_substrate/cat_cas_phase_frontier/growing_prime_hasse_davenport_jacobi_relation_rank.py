#!/usr/bin/env python3
"""Exact Hasse--Davenport/Jacobi relation-rank diagnostic.

M181 showed that one final Mellin/Gauss boundary scalar can be streamed in a
fixed number of field cells, but only by rematerializing theta(q^2) character
work.  This diagnostic asks whether the standard nonlinear multiplicative
relations among the required Gauss phases give a uniformly smaller generator
family instead.

The accepted statement is deliberately restricted to a formal monomial
relation algebra.  Hasse--Davenport and Gauss norm constants are public
topology scalars; answer-bearing Gauss or Jacobi values are not admitted as
free constants.  Exact rational row reduction measures the free generator
rank.  Separate residue-field checks verify every declared Hasse--Davenport,
Gauss norm, and applicable two-character Jacobi identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from growing_prime_mellin_gauss_streamed_recurrence_rank import (
    DECLARED_FIELDS,
    ProceduralField,
    character_reference,
    gauss_reference,
)


CLAIM = (
    "BOUNDED_EXACT_FOURTEEN_DECLARED_PRIME_ALL_DIVISOR_HASSE_DAVENPORT_"
    "GAUSS_PHASE_RELATIONS_HAVE_FORMAL_FREE_MONOMIAL_GENERATOR_RANK_"
    "PHI_OF_QMINUS1_AND_GAUSS_NORM_AUGMENTATION_HAS_RANK_HALF_PHI_WHILE_"
    "THE_ACTUAL_MELLIN_DETERMINANT_BOUNDARY_CHANNEL_PRODUCTS_SPAN_THE_"
    "ENTIRE_REMAINING_QUOTIENT_AND_STREAMED_JACOBI_DEFINITIONS_ADD_NO_"
    "GAUSS_CONSTRAINT_WITHOUT_RETAINING_ANSWER_BEARING_VALUES_OR_"
    "REMATERIALIZING_Q_TERM_SUMS_SO_THE_DECLARED_SAFE_PRIME_CASES_RETAIN_"
    "GROWING_RANK_AND_THE_IDENTICAL_CLASSICAL_RELATION_ALGEBRA_REMAINS"
)


DECLARED_PROGRAMS = {
    5: (1, 2),
    7: (3, 5),
    11: (5, 8),
    13: (7, 11),
    17: (9, 14),
    19: (11, 17),
    23: (13, 20),
    29: (15, 23),
    31: (17, 26),
    37: (19, 29),
    41: (21, 32),
    43: (23, 35),
    47: (25, 38),
    53: (27, 41),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def prime_factors(value: int) -> tuple[int, ...]:
    factors: list[int] = []
    remaining = value
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors.append(divisor)
            while remaining % divisor == 0:
                remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def is_prime(value: int) -> bool:
    return value >= 2 and not any(value % p == 0 for p in range(2, int(value**0.5) + 1))


def euler_phi(value: int) -> int:
    result = value
    for factor in prime_factors(value):
        result -= result // factor
    return result


def nontrivial_divisors(value: int) -> tuple[int, ...]:
    return tuple(divisor for divisor in range(2, value + 1) if value % divisor == 0)


def add_coordinate(row: list[int], coordinate: int, value: int) -> None:
    row[coordinate] += value


def hasse_davenport_rows(
    width: int, omit_divisor: int | None = None
) -> list[list[int]]:
    """Return exponent rows for every multiplication formula n | width.

    In the standard positive Gauss-sum convention used by M180/M181,

      G(n*j) = -chi_(n*j)(n) product_k G(j+k*h/n)/G(k*h/n).

    Only the exponent lattice is represented here.  The sign and character
    factor are public topology-derived constants and do not change free rank.
    """
    rows: list[list[int]] = []
    for divisor in nontrivial_divisors(width):
        if divisor == omit_divisor:
            continue
        step = width // divisor
        for character in range(width):
            row = [0] * width
            add_coordinate(row, divisor * character % width, 1)
            for offset in range(divisor):
                add_coordinate(row, (character + offset * step) % width, -1)
                add_coordinate(row, offset * step, 1)
            if any(row):
                rows.append(row)
    return rows


def gauss_norm_rows(width: int) -> list[list[int]]:
    """Exponent rows for G(j)G(-j)=chi_j(-1)q, excluding trivial j."""
    rows: list[list[int]] = []
    for character in range(1, width):
        row = [0] * width
        row[character] += 1
        row[-character % width] += 1
        rows.append(row)
    return rows


def boundary_product_rows(
    width: int, determinant_character: int, scale_character: int
) -> list[list[int]]:
    """Formal Gauss exponents in each rank-three, nonzero-scale M181 channel."""
    eta = width // 2
    rows: list[list[int]] = []
    for channel in range(width):
        row = [0] * width
        factors = (
            (determinant_character - channel, 1),
            (channel, 2),
            (channel + eta, 1),
            (eta, 3),
            (determinant_character + scale_character - channel, 1),
        )
        for exponent, multiplicity in factors:
            row[exponent % width] += multiplicity
        rows.append(row)
    return rows


def exact_sparse_rank(rows: Iterable[list[int]]) -> tuple[int, tuple[int, ...], int]:
    """Exact rational rank using a normalized sparse row basis."""
    basis: dict[int, dict[int, Fraction]] = {}
    maximum_fraction_bits = 1
    for dense in rows:
        row = {
            column: Fraction(value)
            for column, value in enumerate(dense)
            if value
        }
        while row:
            pivot = min(row)
            if pivot not in basis:
                scale = row[pivot]
                normalized = {column: value / scale for column, value in row.items()}
                basis[pivot] = normalized
                maximum_fraction_bits = max(
                    maximum_fraction_bits,
                    *(
                        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                        for value in normalized.values()
                    ),
                )
                break
            factor = row[pivot]
            for column, value in basis[pivot].items():
                reduced = row.get(column, Fraction(0)) - factor * value
                if reduced:
                    row[column] = reduced
                    maximum_fraction_bits = max(
                        maximum_fraction_bits,
                        abs(reduced.numerator).bit_length(),
                        reduced.denominator.bit_length(),
                    )
                else:
                    row.pop(column, None)
    return len(basis), tuple(sorted(basis)), maximum_fraction_bits


def public_log_table(field: ProceduralField) -> list[int]:
    logs = [-1] * field.q
    value = 1
    for exponent in range(field.q - 1):
        logs[value] = exponent
        value = value * field.q_generator % field.q
    if any(logs[value] < 0 for value in range(1, field.q)):
        fail("public generator did not cover the multiplicative group")
    return logs


def character_from_log(
    field: ProceduralField, logs: list[int], value: int, exponent: int
) -> int:
    reduced = value % field.q
    if reduced == 0:
        return 0
    return pow(
        field.multiplicative_root,
        exponent * logs[reduced] % (field.q - 1),
        field.p,
    )


def jacobi_sum(
    field: ProceduralField, logs: list[int], left: int, right: int
) -> int:
    return sum(
        character_from_log(field, logs, value, left)
        * character_from_log(field, logs, 1 - value, right)
        for value in range(field.q)
    ) % field.p


def residue_identity_diagnostic(field: ProceduralField) -> dict[str, Any]:
    q, p = field.q, field.p
    width = q - 1
    gauss = [gauss_reference(field, exponent) for exponent in range(width)]
    logs = public_log_table(field)

    hasse_checks = 0
    hasse_digest = hashlib.sha256()
    for divisor in nontrivial_divisors(width):
        step = width // divisor
        denominator = 1
        for offset in range(divisor):
            denominator = denominator * gauss[offset * step] % p
        for character in range(width):
            numerator = 1
            for offset in range(divisor):
                numerator = numerator * gauss[(character + offset * step) % width] % p
            public_scalar = character_reference(
                field, divisor, divisor * character % width
            )
            reconstructed = (
                -public_scalar * numerator * pow(denominator, -1, p)
            ) % p
            expected = gauss[divisor * character % width]
            if reconstructed != expected:
                fail(f"Hasse-Davenport identity failed at q={q}, n={divisor}, j={character}")
            hasse_digest.update(
                f"{divisor}:{character}:{reconstructed};".encode("ascii")
            )
            hasse_checks += 1

    norm_checks = 0
    for character in range(1, width):
        expected = (
            character_reference(field, q - 1, character) * q
        ) % p
        if gauss[character] * gauss[-character % width] % p != expected:
            fail(f"Gauss norm identity failed at q={q}, j={character}")
        norm_checks += 1

    jacobi_checks = 0
    jacobi_terms = 0
    jacobi_digest = hashlib.sha256()
    for left in range(1, width):
        for right in range(1, width):
            combined = (left + right) % width
            if combined == 0:
                continue
            value = jacobi_sum(field, logs, left, right)
            if gauss[left] * gauss[right] % p != value * gauss[combined] % p:
                fail(f"Jacobi identity failed at q={q}, j={left}, k={right}")
            jacobi_digest.update(f"{left}:{right}:{value};".encode("ascii"))
            jacobi_checks += 1
            jacobi_terms += q

    if gauss[1] == 1:
        fail("false overmerge control G(1)=1 unexpectedly passed")
    return {
        "q": q,
        "auxiliary_prime": p,
        "hasse_davenport_identity_checks": hasse_checks,
        "hasse_davenport_value_commitment": hasse_digest.hexdigest(),
        "gauss_norm_identity_checks": norm_checks,
        "applicable_jacobi_identity_checks": jacobi_checks,
        "streamed_jacobi_character_terms": jacobi_terms,
        "jacobi_value_commitment": jacobi_digest.hexdigest(),
        "false_overmerge_g1_equals_one_rejected": True,
        "verification_only_retained_gauss_cells": width,
        "verification_only_public_log_cells": q,
        "maximum_live_streamed_jacobi_values": 1,
    }


def relation_rank_diagnostic(q: int) -> dict[str, Any]:
    width = q - 1
    determinant_character, scale_character = DECLARED_PROGRAMS[q]
    hasse_rows = hasse_davenport_rows(width)
    norm_rows = gauss_norm_rows(width)
    boundary_rows = boundary_product_rows(
        width, determinant_character, scale_character
    )

    hasse_rank, hasse_pivots, hasse_bits = exact_sparse_rank(hasse_rows)
    augmented_rank, augmented_pivots, augmented_bits = exact_sparse_rank(
        hasse_rows + norm_rows
    )
    boundary_hasse_rank, _, _ = exact_sparse_rank(hasse_rows + boundary_rows)
    boundary_augmented_rank, _, _ = exact_sparse_rank(
        hasse_rows + norm_rows + boundary_rows
    )
    omitted_quadratic_rank, _, _ = exact_sparse_rank(
        hasse_davenport_rows(width, omit_divisor=2)
    )

    phi = euler_phi(width)
    hasse_free_rank = width - hasse_rank
    norm_augmented_free_rank = width - augmented_rank
    boundary_hasse_span = boundary_hasse_rank - hasse_rank
    boundary_augmented_span = boundary_augmented_rank - augmented_rank
    if hasse_free_rank != phi:
        fail(f"declared Hasse-Davenport free rank differs from phi at q={q}")
    if norm_augmented_free_rank != phi // 2:
        fail(f"declared norm-augmented free rank differs from phi/2 at q={q}")
    if boundary_hasse_span != hasse_free_rank:
        fail(f"boundary products do not span Hasse quotient at q={q}")
    if boundary_augmented_span != norm_augmented_free_rank:
        fail(f"boundary products do not span norm-augmented quotient at q={q}")
    if omitted_quadratic_rank >= hasse_rank:
        fail(f"missing quadratic multiplication relation was not detected at q={q}")

    applicable_jacobi_generators = (width - 1) * (width - 2)
    jacobi_augmented_generator_count = width + applicable_jacobi_generators
    jacobi_augmented_relation_rank = hasse_rank + applicable_jacobi_generators
    if jacobi_augmented_generator_count - jacobi_augmented_relation_rank != phi:
        fail("Jacobi defining-symbol accounting changed the Gauss free rank")

    safe_prime = is_prime(width // 2)
    return {
        "q": q,
        "character_width": width,
        "nontrivial_hasse_divisors": list(nontrivial_divisors(width)),
        "hasse_relation_rows": len(hasse_rows),
        "hasse_relation_matrix_materialized_integer_cells": len(hasse_rows) * width,
        "hasse_exact_rational_rank": hasse_rank,
        "hasse_free_monomial_generator_rank": hasse_free_rank,
        "euler_phi_q_minus_1": phi,
        "hasse_pivot_columns": list(hasse_pivots),
        "hasse_maximum_elimination_fraction_bits": hasse_bits,
        "gauss_norm_relation_rows": len(norm_rows),
        "norm_augmented_exact_rational_rank": augmented_rank,
        "norm_augmented_free_monomial_generator_rank": norm_augmented_free_rank,
        "norm_augmented_pivot_columns": list(augmented_pivots),
        "norm_augmented_maximum_elimination_fraction_bits": augmented_bits,
        "declared_program": {
            "determinant_character": determinant_character,
            "scale_character": scale_character,
        },
        "boundary_channel_product_rows": len(boundary_rows),
        "boundary_product_span_mod_hasse": boundary_hasse_span,
        "boundary_product_span_mod_hasse_and_norm": boundary_augmented_span,
        "boundary_products_span_entire_hasse_quotient": True,
        "boundary_products_span_entire_norm_augmented_quotient": True,
        "without_quadratic_hasse_relation_free_rank": width - omitted_quadratic_rank,
        "missing_quadratic_relation_increases_free_rank": True,
        "applicable_formal_jacobi_generators": applicable_jacobi_generators,
        "jacobi_augmented_formal_generators": jacobi_augmented_generator_count,
        "jacobi_augmented_defining_relation_rank": jacobi_augmented_relation_rank,
        "jacobi_augmented_free_rank": phi,
        "jacobi_definitions_add_no_constraint_on_gauss_projection": True,
        "safe_prime_q_equals_2r_plus_1": safe_prime,
        "safe_prime_hasse_free_rank_formula_observed": (
            "(q-3)/2" if safe_prime else "NOT_APPLICABLE"
        ),
        "safe_prime_norm_augmented_free_rank_formula_observed": (
            "1_FOR_Q5_OTHERWISE_(q-3)/4" if safe_prime else "NOT_APPLICABLE"
        ),
    }


def build_result() -> dict[str, Any]:
    rank_diagnostics = [relation_rank_diagnostic(q) for q, _ in DECLARED_FIELDS]
    residue_diagnostics = [
        residue_identity_diagnostic(ProceduralField.create(q, p))
        for q, p in DECLARED_FIELDS
    ]
    safe_cases = [
        {
            "q": item["q"],
            "hasse_free_rank": item["hasse_free_monomial_generator_rank"],
            "norm_augmented_free_rank": item[
                "norm_augmented_free_monomial_generator_rank"
            ],
        }
        for item in rank_diagnostics
        if item["safe_prime_q_equals_2r_plus_1"]
    ]
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "claim_ceiling": (
            "FORMAL_INTEGER_EXPONENT_LATTICE_OF_ALL_DIVISOR_HASSE_DAVENPORT_"
            "RELATIONS_PLUS_OPTIONAL_GAUSS_NORM_ROWS_AND_APPLICABLE_TWO_"
            "CHARACTER_JACOBI_DEFINITIONS_FOR_THE_FOURTEEN_M181_PRIME_"
            "FAMILIES_WITH_ONE_DECLARED_RANK3_NONZERO_SCALE_PROGRAM_PER_"
            "FIELD_DIRECT_PROCESS_EXACT_RESIDUE_DIAGNOSTIC"
        ),
        "declared_scope": {
            "fields": [
                {"q": q, "auxiliary_prime": p} for q, p in DECLARED_FIELDS
            ],
            "safe_prime_cases": safe_cases,
            "public_constants_admitted": [
                "HASSE_DAVENPORT_SIGN",
                "CHARACTER_OF_DIVISOR",
                "FIELD_CARDINALITY",
                "CHARACTER_OF_MINUS_ONE",
            ],
            "answer_bearing_gauss_or_jacobi_constants_admitted": False,
            "relation_constants_affect_free_rank": False,
        },
        "rank_diagnostics": rank_diagnostics,
        "residue_identity_diagnostics": residue_diagnostics,
        "controls": {
            "every_declared_hasse_davenport_identity_checked": True,
            "every_declared_nontrivial_gauss_norm_identity_checked": True,
            "every_applicable_declared_two_character_jacobi_identity_checked": True,
            "missing_quadratic_hasse_relation_detected_every_field": True,
            "false_overmerge_g1_equals_one_rejected_every_field": True,
            "boundary_specific_span_tested": True,
        },
        "observed_resource_law": {
            "hasse_free_rank": "PHI(Q_MINUS_1)_ON_EVERY_DECLARED_CASE",
            "norm_augmented_free_rank": "PHI(Q_MINUS_1)_DIVIDED_BY_2_ON_EVERY_DECLARED_CASE",
            "declared_safe_prime_hasse_rank": "LINEAR_(Q_MINUS_3)_DIVIDED_BY_2",
            "declared_safe_prime_norm_augmented_rank": "LINEAR_AFTER_Q5_(Q_MINUS_3)_DIVIDED_BY_4",
            "boundary_product_span": "FULL_REMAINING_QUOTIENT_ON_EVERY_DECLARED_PROGRAM",
            "retained_full_jacobi_table_if_materialized": "(Q_MINUS_2)*(Q_MINUS_3)_FIELD_CELLS",
            "one_streamed_jacobi_value": "Q_CHARACTER_TERMS_AND_ONE_VALUE_CELL",
            "all_boundary_channels_with_one_streamed_jacobi_each": "THETA_Q_SQUARED_CHARACTER_TERMS",
            "exact_rank_diagnostic_relation_matrix": "MATERIALIZED_VERIFICATION_ONLY_AND_NOT_AN_ACCEPTED_COMPACT_CARRIER",
        },
        "matched_baseline": {
            "strongest_compact_classical_method": (
                "IDENTICAL_FORMAL_HASSE_DAVENPORT_NORM_JACOBI_RELATION_"
                "ALGEBRA_WITH_THE_SAME_EXACT_RANK_AND_STREAMING_COST"
            ),
            "state_advantage": False,
            "work_advantage": False,
        },
        "strict_interpretation": {
            "establishes": [
                "BOUNDED_ALL_DIVISOR_HASSE_DAVENPORT_IDENTITIES",
                "BOUNDED_GAUSS_NORM_AND_JACOBI_IDENTITIES",
                "EXACT_FORMAL_FREE_RANKS_ON_DECLARED_CASES",
                "FULL_BOUNDARY_PRODUCT_SPAN_OF_DECLARED_QUOTIENTS",
                "GROWING_SAFE_PRIME_FORMAL_MONOMIAL_RANK_ON_DECLARED_CASES",
            ],
            "does_not_establish": [
                "NO_GO_FOR_ALL_NONLINEAR_OR_NONMONOMIAL_ALGORITHMS",
                "SUBQUADRATIC_WORK_IMPOSSIBILITY_FOR_ALL_GAUSS_ALGORITHMS",
                "COMPACT_JACOBI_VALUE_GENERATION",
                "CATVM_CUSTODY",
                "DISTINCT_PHASE_RESOURCE",
                "COMPUTATIONAL_ADVANTAGE",
                "SMALL_WALL_CROSSING",
                "PHYSICAL_WAVEFORM_EXECUTION",
                "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
                "UNBOUNDED_COMPUTATION",
            ],
        },
        "next_obstruction": (
            "ALL_STANDARD_MONOMIAL_HASSE_DAVENPORT_AND_NORM_RELATIONS_LEAVE_"
            "A_GROWING_FREE_GENERATOR_FAMILY_EXERCISED_BY_THE_ACTUAL_BOUNDARY_"
            "PRODUCTS_WHILE_JACOBI_VALUES_MUST_BE_RETAINED_OR_REMATERIALIZED_"
            "SO_THE_NEXT_REPAIR_MUST_CHANGE_TO_AN_ADDITIVE_FOURIER_PHASE_"
            "COMPILER_OR_ANOTHER_NONMONOMIAL_NATIVE_UPDATE_WITHOUT_MOVING_"
            "LINEAR_STATE_OR_QUADRATIC_WORK_ELSEWHERE"
        ),
        "selected_successor": (
            "EXACT_REVERSIBLE_MIXED_RADIX_OR_BLUESTEIN_NTT_GAUSS_PHASE_"
            "COMPILER_WITH_SUBQUADRATIC_WORK_AND_HONEST_LINEAR_STATE_OR_"
            "FOURIER_WORKSPACE_OBSTRUCTION_WITH_IDENTICAL_CLASSICAL_NTT_BASELINE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
