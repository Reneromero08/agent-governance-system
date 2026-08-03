#!/usr/bin/env python3
"""Independent oracle for the M128 linear phase-module no-go.

The oracle never imports the M128 production package.  It reconstructs the
public primitive word on occupation coordinates using the already independent
M127 matching reference, implements its own symmetric shear recurrence, and
rebuilds the power-sum/projector/connectivity certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import f17_exchange_symmetric_latent_geometry_closure_oracle as m127ref
import f17_nonlinear_canonical_mps_separator_chart as backend


PRIME = 17
STRUCTURAL_K = (1, 2, 3, 4)
EXACT_CASES = ((1, "PRIMARY"), (2, "PRIMARY"), (2, "REUSE"))
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def program_word(k: int, family: str) -> dict[str, tuple[tuple[int, ...], ...]]:
    if k not in STRUCTURAL_K or family not in ("PRIMARY", "REUSE"):
        fail("oracle program is outside the declared scope")
    variant = 0 if family == "PRIMARY" else 1
    return {
        "first_characters": tuple(
            (degree, 1 + ((3 * degree + 5 * variant) % 16))
            for degree in range(1, k + 1)
        ),
        "first_shears": tuple(
            (mode, mode + 1, 1 + ((5 * mode + 3 + variant) % 16))
            for mode in range(PRIME - 1)
        ),
        "second_characters": tuple(
            (degree, 1 + ((7 * degree + 2 + 3 * variant) % 16))
            for degree in range(k, 0, -1)
        ),
        "second_shears": tuple(
            (mode + 1, mode, 1 + ((7 * mode + 4 + 2 * variant) % 16))
            for mode in range(PRIME - 2, -1, -1)
        ),
    }


def field_power(alg: backend.Algebra, value: Any, exponent: int) -> Any:
    result = alg.one
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = alg.mul(result, factor)
        remaining >>= 1
        if remaining:
            factor = alg.mul(factor, factor)
    return result


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def apply_character_segment(
    vector: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    steps: tuple[tuple[int, int], ...],
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> None:
    sign = -1 if inverse else 1
    for degree, coefficient in steps:
        for index, counts in enumerate(histograms):
            power_sum = sum(
                count * pow(mode, degree, PRIME)
                for mode, count in enumerate(counts)
            ) % PRIME
            vector[index] = alg.mul(
                vector[index], alg.power(sign * coefficient * power_sum)
            )


def apply_shear(
    vector: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    ranks: dict[tuple[int, ...], int],
    row: int,
    pivot: int,
    coefficient: Any,
    alg: backend.Algebra,
) -> None:
    k = sum(histograms[0])
    powers = [alg.one]
    for _ in range(k):
        powers.append(alg.mul(powers[-1], coefficient))
    for base in histograms:
        if base[row] != 0:
            continue
        total = base[pivot]
        indices = []
        for row_count in range(total + 1):
            member = list(base)
            member[row] = row_count
            member[pivot] = total - row_count
            indices.append(ranks[tuple(member)])
        old = [vector[index] for index in indices]
        updated = [alg.zero for _ in indices]
        for input_row, amplitude in enumerate(old):
            input_pivot = total - input_row
            for moved in range(input_pivot + 1):
                output_row = input_row + moved
                coefficient_value = alg.mul(
                    m127ref.integer(alg, math.comb(input_pivot, moved)),
                    powers[moved],
                )
                updated[output_row] = alg.add(
                    updated[output_row], alg.mul(amplitude, coefficient_value)
                )
        for index, value in zip(indices, updated, strict=True):
            vector[index] = value


def apply_shear_segment(
    u: list[Any],
    w: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    ranks: dict[tuple[int, ...], int],
    steps: tuple[tuple[int, int, int], ...],
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> None:
    ordered = tuple(reversed(steps)) if inverse else steps
    for row, pivot, exponent in ordered:
        coefficient = alg.power(exponent)
        if inverse:
            coefficient = negative(alg, coefficient)
        apply_shear(u, histograms, ranks, row, pivot, coefficient, alg)
        apply_shear(w, histograms, ranks, row, pivot, coefficient, alg)


def apply_module(
    u: list[Any],
    w: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    k: int,
    family: str,
    module: int,
    alg: backend.Algebra,
    cache: dict[tuple[int, tuple[int, ...]], Any],
    *,
    inverse: bool = False,
) -> None:
    for index, counts in enumerate(histograms):
        key = (module, counts)
        if key not in cache:
            cache[key] = m127ref.module_boundary(k, family, module, counts, alg)
        term = alg.mul(cache[key], u[index])
        w[index] = alg.sub(w[index], term) if inverse else alg.add(w[index], term)


def independent_transaction(
    k: int,
    family: str,
    alg: backend.Algebra,
) -> tuple[Any, bool, int]:
    histograms = m127ref.enumerate_histograms(k)
    ranks = {histogram: index for index, histogram in enumerate(histograms)}
    word = program_word(k, family)
    u = [alg.zero for _ in histograms]
    w = [alg.zero for _ in histograms]
    zero = (k, *([0] * (PRIME - 1)))
    u[ranks[zero]] = alg.one
    initial_u = u[:]
    initial_w = w[:]
    cache: dict[tuple[int, tuple[int, ...]], Any] = {}

    apply_shear_segment(u, w, histograms, ranks, word["first_shears"], alg)
    apply_character_segment(u, histograms, word["first_characters"], alg)
    apply_module(u, w, histograms, k, family, 0, alg, cache)
    apply_shear_segment(u, w, histograms, ranks, word["second_shears"], alg)
    apply_character_segment(u, histograms, word["second_characters"], alg)
    apply_module(u, w, histograms, k, family, 1, alg, cache)
    boundary_histogram = (k - 1, 1, *([0] * (PRIME - 2)))
    boundary = w[ranks[boundary_histogram]]

    apply_module(u, w, histograms, k, family, 1, alg, cache, inverse=True)
    apply_character_segment(
        u, histograms, word["second_characters"], alg, inverse=True
    )
    apply_shear_segment(
        u, w, histograms, ranks, word["second_shears"], alg, inverse=True
    )
    apply_module(u, w, histograms, k, family, 0, alg, cache, inverse=True)
    apply_character_segment(
        u, histograms, word["first_characters"], alg, inverse=True
    )
    apply_shear_segment(
        u, w, histograms, ranks, word["first_shears"], alg, inverse=True
    )
    return boundary, u == initial_u and w == initial_w, len(cache)


def certificate(k: int) -> dict[str, Any]:
    histograms = m127ref.enumerate_histograms(k)
    ranks = {histogram: index for index, histogram in enumerate(histograms)}
    signatures = [m127ref.power_sums(histogram, k) for histogram in histograms]
    truncated = {
        m127ref.power_sums(histogram, max(1, k - 1)) for histogram in histograms
    }
    predecessor_count = 0
    predecessor_hasher = hashlib.sha256()
    isolated_transition_coefficients_nonzero = True
    directed_transition_count = 0
    for index, histogram in enumerate(histograms):
        occupied = next(
            (mode for mode in range(1, PRIME) if histogram[mode]), None
        )
        if occupied is not None:
            predecessor = list(histogram)
            predecessor[occupied] -= 1
            predecessor[occupied - 1] += 1
            predecessor_rank = ranks[tuple(predecessor)]
            predecessor_count += 1
            predecessor_hasher.update(
                f"{index}:{predecessor_rank};".encode("ascii")
            )
        for pivot in range(PRIME):
            if not histogram[pivot]:
                continue
            for row in (pivot - 1, pivot + 1):
                if 0 <= row < PRIME:
                    directed_transition_count += 1
                    isolated_transition_coefficients_nonzero &= (
                        histogram[pivot] % PRIME != 0
                    )
    return {
        "k": k,
        "occupation_dimension": len(histograms),
        "stars_and_bars_dimension": math.comb(k + 16, 16),
        "p1_through_pk_signature_count": len(set(signatures)),
        "power_sum_signature_injective": len(set(signatures)) == len(histograms),
        "p1_through_p_k_minus_1_signature_count": len(truncated),
        "omitting_pk_overmerges_when_k_at_least2": (
            k == 1 or len(truncated) < len(histograms)
        ),
        "newton_denominators_invertible_mod17": all(
            math.gcd(degree, PRIME) == 1 for degree in range(1, k + 1)
        ),
        "independent_predecessor_edges_to_zero_mode": predecessor_count,
        "expected_predecessor_edges": len(histograms) - 1,
        "predecessor_certificate_fingerprint": predecessor_hasher.hexdigest(),
        "bidirectional_adjacent_transition_edges": directed_transition_count,
        "isolated_one_particle_transition_coefficients_nonzero": isolated_transition_coefficients_nonzero,
        "full_matrix_algebra_certificate": (
            len(set(signatures)) == len(histograms)
            and predecessor_count == len(histograms) - 1
            and isolated_transition_coefficients_nonzero
        ),
        "minimum_exact_linear_quotient_dimension": len(histograms),
    }


def orthogonality(alg: backend.Algebra) -> dict[str, Any]:
    checks = []
    for difference in range(PRIME):
        value = alg.zero
        for coefficient in range(PRIME):
            value = alg.add(value, alg.power(coefficient * difference))
        expected = m127ref.integer(alg, PRIME) if difference == 0 else alg.zero
        checks.append(value == expected)
    return {
        "algebra": f"F{alg.modulus}" if alg.modulus else "Q_ZETA17",
        "all_17_character_sums_exact": all(checks),
        "zero_difference_nonzero": checks[0],
        "all_nonzero_differences_zero": all(checks[1:]),
    }


def verify(production: dict[str, Any]) -> dict[str, Any]:
    exact_lookup = {
        (item["k"], item["family"]): item
        for item in production["exact_transactions"]
    }
    exact_parity = []
    for k, family in EXACT_CASES:
        alg = backend.Algebra("Q_ZETA17")
        boundary, restored, cache_cells = independent_transaction(k, family, alg)
        exact_parity.append(
            {
                "k": k,
                "family": family,
                "boundary_agreement": alg.serialize(boundary)
                == exact_lookup[(k, family)]["boundary"],
                "independent_forward_inverse_restored": restored,
                "oracle_cached_orbit_boundary_field_cells": cache_cells,
            }
        )

    structural_lookup = {
        (item["field"], item["k"], item["family"]): item
        for item in production["dual_field_structural_transactions"]
    }
    structural_parity = []
    for modulus, root in FINITE_FIELDS:
        for k in STRUCTURAL_K:
            alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            boundary, restored, cache_cells = independent_transaction(
                k, "PRIMARY", alg
            )
            expected = structural_lookup[(f"F{modulus}", k, "PRIMARY")][
                "boundary"
            ]
            structural_parity.append(
                {
                    "field": f"F{modulus}",
                    "k": k,
                    "boundary_agreement": alg.serialize(boundary) == expected,
                    "independent_forward_inverse_restored": restored,
                    "oracle_cached_orbit_boundary_field_cells": cache_cells,
                }
            )

    independent_certificates = [certificate(k) for k in STRUCTURAL_K]
    production_certificates = {
        item["k"]: item for item in production["irreducibility_certificates"]
    }
    certificate_parity = []
    for item in independent_certificates:
        expected = production_certificates[item["k"]]
        certificate_parity.append(
            {
                "k": item["k"],
                "dimension_agreement": item["occupation_dimension"]
                == expected["occupation_dimension"],
                "signature_count_agreement": item["p1_through_pk_signature_count"]
                == expected["p1_through_pk_signature_count"],
                "predecessor_fingerprint_agreement": item[
                    "predecessor_certificate_fingerprint"
                ]
                == expected["predecessor_certificate_fingerprint"],
                "full_matrix_algebra_certificate": item[
                    "full_matrix_algebra_certificate"
                ],
                "minimum_quotient_dimension_agreement": item[
                    "minimum_exact_linear_quotient_dimension"
                ]
                == expected["minimum_exact_linear_quotient_dimension"],
            }
        )

    orthogonality_checks = [
        orthogonality(backend.Algebra("Q_ZETA17")),
        *(
            orthogonality(
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            )
            for modulus, root in FINITE_FIELDS
        ),
    ]
    controls = production["controls"]
    accepted = (
        all(item["boundary_agreement"] and item["independent_forward_inverse_restored"] for item in exact_parity)
        and all(item["boundary_agreement"] and item["independent_forward_inverse_restored"] for item in structural_parity)
        and all(
            item["dimension_agreement"]
            and item["signature_count_agreement"]
            and item["predecessor_fingerprint_agreement"]
            and item["full_matrix_algebra_certificate"]
            and item["minimum_quotient_dimension_agreement"]
            for item in certificate_parity
        )
        and all(item["all_17_character_sums_exact"] for item in orthogonality_checks)
        and all(
            controls[key]
            for key in (
                "missing_inverse_detected",
                "wrong_program_ownership_rejected",
                "premature_projection_rejected",
                "reordered_inverse_detected",
                "null_carrier_rejected",
                "missing_p2_character_rejected",
                "power_sum_character_mutation_changes_boundary",
                "missing_reverse_mode_orientation_rejected",
                "mode_shear_mutation_changes_boundary",
                "p1_only_overmerges_k2",
                "disconnected_mode_graph_preserves_particle_count_components",
                "nonprimitive_root_rejected",
                "k17_newton_applicability_rejected",
            )
        )
    )
    if not accepted:
        fail("independent phase-module irreducibility verification failed")
    return {
        "schema": "CAT_CAS_F17_EXCHANGE_SYMMETRIC_PHASE_MODULE_IRREDUCIBILITY_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_claim": production["claim"],
        "production_result_sha256": digest_json(production),
        "exact_transaction_parity": exact_parity,
        "dual_field_structural_transaction_parity": structural_parity,
        "certificate_parity": certificate_parity,
        "independent_certificates": independent_certificates,
        "phase_character_orthogonality": orthogonality_checks,
        "matched_baseline": "IDENTICAL_H_K_COORDINATE_EXACT_LINEAR_PHASE_MODULE_RECURRENCE",
        "claim_ceiling": "K1_TO_K4_EXCHANGE_SYMMETRIC_F17_POWER_SUM_CHARACTER_AND_BIDIRECTIONAL_ADJACENT_MODE_SHEAR_LINEAR_MODULE_ONLY_DIRECT_PROCESS_SOFTWARE",
        "rejected_interpretations": [
            "THE_NO_GO_ALREADY_APPLIES_TO_UNEXTENDED_M127_SEMANTICS",
            "NONLINEAR_OR_PROGRAM_RESTRICTED_QUOTIENT_NO_GO",
            "GENERAL_COMPLEXITY_LOWER_BOUND",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "CATVM_CUSTODY",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    production = json.loads(Path(arguments.production).read_text(encoding="utf-8"))
    result = verify(production)
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
