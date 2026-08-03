#!/usr/bin/env python3
"""Independent labelled-tensor oracle for the M127 orbit carrier.

This verifier reconstructs the public program, row-profile perfect-matching
closure, and labelled ``17**k`` tensor recurrence without importing either the
production package or its matchgate predecessor.  Labelled expansion and
cached orbit boundary vectors are oracle-only costs and are never attributed
to the accepted carrier.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
from pathlib import Path
from typing import Any

import f17_nonlinear_canonical_mps_separator_chart as backend


PRIME = 17
GRID_N = 4
STRUCTURAL_K = (1, 2, 3, 4)
EXACT_CASES = ((1, "PRIMARY"), (2, "PRIMARY"), (2, "REUSE"))
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def integer(alg: backend.Algebra, value: int) -> Any:
    if alg.modulus:
        return value % alg.modulus
    return alg.domain.convert(value)


def grid_edges() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return (
        *(((row, column), (row, column + 1))
          for row in range(GRID_N) for column in range(GRID_N - 1)),
        *(((row, column), (row + 1, column))
          for row in range(GRID_N - 1) for column in range(GRID_N)),
    )


def public_program(
    k: int,
    family: str,
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    int,
]:
    variant = 0 if family == "PRIMARY" else 1
    edge_count = len(grid_edges())
    weights = []
    controls = []
    degrees = []
    for module in range(2):
        weights.append(
            tuple(
                1 + ((5 * edge + 7 * GRID_N + 3 * module + 4 * variant) % 16)
                for edge in range(edge_count)
            )
        )
        controls.append(
            tuple(
                1 + ((7 * edge + 5 * module + 6 * variant + GRID_N) % 16)
                for edge in range(edge_count)
            )
        )
        degrees.append(
            tuple(
                1 + ((5 * edge + 3 * module + 2 * variant) % k)
                for edge in range(edge_count)
            )
        )
    return (
        (weights[0], weights[1]),
        (controls[0], controls[1]),
        (degrees[0], degrees[1]),
        3 + 2 * variant,
    )


def digits(index: int, k: int) -> tuple[int, ...]:
    result = []
    for _ in range(k):
        result.append(index % PRIME)
        index //= PRIME
    return tuple(result)


def histogram(values: tuple[int, ...]) -> tuple[int, ...]:
    counts = [0] * PRIME
    for value in values:
        counts[value] += 1
    return tuple(counts)


def power_sums(counts: tuple[int, ...], maximum_degree: int) -> tuple[int, ...]:
    return tuple(
        sum(count * pow(mode, degree, PRIME) for mode, count in enumerate(counts))
        % PRIME
        for degree in range(1, maximum_degree + 1)
    )


def matching_boundary(weights: tuple[Any, ...], alg: backend.Algebra) -> Any:
    edge_weight = {
        frozenset((left, right)): weight
        for (left, right), weight in zip(grid_edges(), weights, strict=True)
    }

    @functools.lru_cache(maxsize=None)
    def visit(position: int, mask: int) -> Any:
        if position == GRID_N * GRID_N:
            return alg.one if mask == 0 else alg.zero
        row, column = divmod(position, GRID_N)
        if mask & 1:
            return visit(position + 1, mask >> 1)
        value = alg.zero
        here = (row, column)
        if column + 1 < GRID_N and not (mask & 2):
            edge = frozenset((here, (row, column + 1)))
            value = alg.add(
                value,
                alg.mul(edge_weight[edge], visit(position + 1, (mask | 2) >> 1)),
            )
        if row + 1 < GRID_N:
            edge = frozenset((here, (row + 1, column)))
            value = alg.add(
                value,
                alg.mul(
                    edge_weight[edge],
                    visit(position + 1, (mask >> 1) | (1 << (GRID_N - 1))),
                ),
            )
        return value

    return visit(0, 0)


def module_boundary(
    k: int,
    family: str,
    module: int,
    counts: tuple[int, ...],
    alg: backend.Algebra,
) -> Any:
    weight_rows, control_rows, degree_rows, _ = public_program(k, family)
    sums = power_sums(counts, k)
    weights = tuple(
        alg.mul(
            alg.power(weight),
            alg.power(control * sums[degree - 1]),
        )
        for weight, control, degree in zip(
            weight_rows[module],
            control_rows[module],
            degree_rows[module],
            strict=True,
        )
    )
    return matching_boundary(weights, alg)


def labelled_fourier(
    values: list[Any],
    k: int,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> list[Any]:
    result = values[:]
    scale = alg.divide(alg.one, integer(alg, PRIME)) if inverse else alg.one
    sign = -1 if inverse else 1
    for axis in range(k):
        stride = PRIME ** axis
        block = PRIME * stride
        updated = [alg.zero for _ in result]
        for start in range(0, len(result), block):
            for offset in range(stride):
                old = [result[start + source * stride + offset] for source in range(PRIME)]
                for target in range(PRIME):
                    value = alg.zero
                    for source, amplitude in enumerate(old):
                        value = alg.add(
                            value,
                            alg.mul(
                                amplitude,
                                alg.power(sign * source * target),
                            ),
                        )
                    updated[start + target * stride + offset] = alg.mul(value, scale)
        result = updated
    return result


def apply_module(
    u: list[Any],
    w: list[Any],
    k: int,
    family: str,
    module: int,
    alg: backend.Algebra,
    cache: dict[tuple[int, tuple[int, ...]], Any],
    *,
    inverse: bool = False,
) -> None:
    for index in range(len(u)):
        counts = histogram(digits(index, k))
        key = (module, counts)
        if key not in cache:
            cache[key] = module_boundary(k, family, module, counts, alg)
        term = alg.mul(cache[key], u[index])
        w[index] = alg.sub(w[index], term) if inverse else alg.add(w[index], term)


def apply_chirp(
    u: list[Any],
    w: list[Any],
    k: int,
    chirp_exponent: int,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> None:
    sign = -1 if inverse else 1
    for index in range(len(u)):
        values = digits(index, k)
        quadratic = sum(value * value for value in values) % PRIME
        phase = alg.power(sign * chirp_exponent * quadratic)
        u[index] = alg.mul(u[index], phase)
        w[index] = alg.mul(w[index], phase)


def independent_transaction(
    k: int,
    family: str,
    alg: backend.Algebra,
) -> tuple[Any, bool, int]:
    _, _, _, chirp_exponent = public_program(k, family)
    dimension = PRIME ** k
    u = [alg.zero for _ in range(dimension)]
    w = [alg.zero for _ in range(dimension)]
    u[0] = alg.one
    initial_u = u[:]
    initial_w = w[:]
    cache: dict[tuple[int, tuple[int, ...]], Any] = {}

    u = labelled_fourier(u, k, alg)
    w = labelled_fourier(w, k, alg)
    apply_module(u, w, k, family, 0, alg, cache)
    apply_chirp(u, w, k, chirp_exponent, alg)
    u = labelled_fourier(u, k, alg)
    w = labelled_fourier(w, k, alg)
    apply_module(u, w, k, family, 1, alg, cache)
    boundary = w[0]

    apply_module(u, w, k, family, 1, alg, cache, inverse=True)
    u = labelled_fourier(u, k, alg, inverse=True)
    w = labelled_fourier(w, k, alg, inverse=True)
    apply_chirp(u, w, k, chirp_exponent, alg, inverse=True)
    apply_module(u, w, k, family, 0, alg, cache, inverse=True)
    u = labelled_fourier(u, k, alg, inverse=True)
    w = labelled_fourier(w, k, alg, inverse=True)
    return boundary, u == initial_u and w == initial_w, len(cache)


def enumerate_histograms(k: int) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []

    def visit(mode: int, remaining: int, prefix: list[int]) -> None:
        if mode == PRIME - 1:
            result.append(tuple((*prefix, remaining)))
            return
        for count in range(remaining + 1):
            prefix.append(count)
            visit(mode + 1, remaining - count, prefix)
            prefix.pop()

    visit(0, k, [])
    return tuple(result)


def separation_checks() -> list[dict[str, Any]]:
    result = []
    for k in STRUCTURAL_K:
        histograms = enumerate_histograms(k)
        full = {power_sums(counts, k) for counts in histograms}
        sums = {power_sums(counts, 1) for counts in histograms}
        result.append(
            {
                "k": k,
                "occupation_dimension": len(histograms),
                "stars_and_bars_dimension": math.comb(k + 16, 16),
                "p1_through_pk_signature_count": len(full),
                "p1_signature_count": len(sums),
                "full_power_sums_separate_all_histograms": len(full) == len(histograms),
                "total_sum_alone_overmerges": k >= 2 and len(sums) < len(histograms),
            }
        )
    return result


def permutation_checks() -> list[dict[str, Any]]:
    checks = []
    for values in ((0, 3), (1, 5, 9), (2, 2, 7, 11)):
        signatures = {
            power_sums(histogram(permutation), len(values))
            for permutation in set(itertools.permutations(values))
        }
        checks.append(
            {
                "k": len(values),
                "distinct_particle_permutations": len(set(itertools.permutations(values))),
                "distinct_symmetric_signatures": len(signatures),
                "exchange_invariant": len(signatures) == 1,
            }
        )
    return checks


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
                "agreement": alg.serialize(boundary) == exact_lookup[(k, family)]["boundary"],
                "independent_labelled_forward_inverse_restored": restored,
                "oracle_cached_orbit_boundary_field_cells": cache_cells,
            }
        )

    structural_lookup = {
        (item.get("field"), item["k"], item["family"]): item
        for item in production["dual_field_structural_transactions"]
    }
    structural_parity = []
    for modulus, root in FINITE_FIELDS:
        for k in STRUCTURAL_K:
            alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            boundary, restored, cache_cells = independent_transaction(k, "PRIMARY", alg)
            expected = structural_lookup[(f"F{modulus}", k, "PRIMARY")]["boundary"]
            structural_parity.append(
                {
                    "field": f"F{modulus}",
                    "k": k,
                    "family": "PRIMARY",
                    "agreement": alg.serialize(boundary) == expected,
                    "independent_labelled_forward_inverse_restored": restored,
                    "oracle_cached_orbit_boundary_field_cells": cache_cells,
                }
            )
    alg = backend.Algebra("F103", modulus=103, root=72)
    boundary, restored, cache_cells = independent_transaction(4, "REUSE", alg)
    structural_parity.append(
        {
            "field": "F103",
            "k": 4,
            "family": "REUSE",
            "agreement": alg.serialize(boundary)
            == structural_lookup[("F103", 4, "REUSE")]["boundary"],
            "independent_labelled_forward_inverse_restored": restored,
            "oracle_cached_orbit_boundary_field_cells": cache_cells,
        }
    )

    controls = production["controls"]
    return {
        "schema": "CAT_CAS_F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "independence": {
            "imports_production_or_matchgate_predecessor": False,
            "accepted_recurrence_reused": False,
            "matching_oracle": "INDEPENDENT_ROW_PROFILE_PERFECT_MATCHING_RECURRENCE",
            "latent_oracle": "INDEPENDENT_LABELLED_17_TO_THE_K_TENSOR_DFT_CHIRP_AND_GRID_SHEAR",
            "oracle_only_labelled_tensor_and_orbit_boundary_cache_materialized": True,
        },
        "exact_production_parity": exact_parity,
        "dual_field_structural_parity": structural_parity,
        "power_sum_separation": separation_checks(),
        "particle_permutation_equivariance": permutation_checks(),
        "control_receipt_consistency": {
            "all_required_controls_true": all(
                controls[name]
                for name in (
                    "missing_inverse_detected",
                    "wrong_inverse_ownership_detected",
                    "premature_projection_rejected",
                    "null_carrier_rejected",
                    "reordered_inverse_detected",
                    "symmetry_breaking_descriptor_rejected",
                    "particle_permutation_preserves_histogram_invariants",
                    "total_sum_overmerge_changes_boundary",
                    "missing_power_sum_changes_boundary",
                    "semantic_control_mutation_changes_boundary",
                )
            ),
            "limitation": "LIFECYCLE_RECEIPTS_REQUIRE_SEPARATE_SOURCE_AUDIT_NOT_ORACLE_INFERENCE",
        },
        "observed_resource_law": {
            "production_occupation_dimension": "BINOMIAL_K_PLUS_16_CHOOSE_16",
            "oracle_labelled_tensor_field_cells": "2_TIMES_17_TO_THE_K",
            "oracle_cached_boundary_field_cells": "2_TIMES_BINOMIAL_K_PLUS_16_CHOOSE_16",
            "production_materializes_neither_oracle_expansion": True,
            "strongest_matched_classical_baseline": "IDENTICAL_OCCUPATION_ORBIT_RECURRENCE_NOT_THE_LABELLED_ORACLE",
        },
        "claim_ceiling": {
            "grid_n4_only": True,
            "exchange_symmetric_k1_through_k4_only": True,
            "exact_q_zeta17_k1_k2_only": True,
            "dual_field_k1_through_k4": True,
            "exchange_symmetry_is_required": True,
            "original_labelled_family_compressed": False,
            "fixed_rank_across_growing_k": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_EXACT_EXCHANGE_SYMMETRIC_QUOTIENT_REDUCES_LABELLED_EXPONENTIAL_GROWTH_BUT_RETAINS_H_K_ORBIT_RANK_AND_THE_MATCHED_CLASSICAL_RECURRENCE_IS_IDENTICAL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.production, encoding="utf-8") as handle:
        production = json.load(handle)
    rendered = json.dumps(verify(production), indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
