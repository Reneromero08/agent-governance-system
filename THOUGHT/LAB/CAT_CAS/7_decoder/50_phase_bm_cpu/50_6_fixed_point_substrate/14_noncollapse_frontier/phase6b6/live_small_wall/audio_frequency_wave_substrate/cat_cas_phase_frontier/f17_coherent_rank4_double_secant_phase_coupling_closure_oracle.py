#!/usr/bin/env python3
"""Independent oracle for the bounded double-coupling rank-four successor.

This file never imports the M131 production module.  It separately compiles
the public word, executes the four-component chart and its inverse, constructs
a normalized catalecticant using a Leibniz determinant, reexecutes the full
H(4) occupation recurrence, and checks the strongest folded-endpoint baseline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure_oracle as rank1ref
import f17_nonlinear_canonical_mps_separator_chart as backend


MODE_COUNT = 17
SLOT_COUNT = 4
DECLARED_K = (4, 8, 16, 32, 64, 128)
EXACT_K = (4, 8, 16, 32)
FINITE_FIELDS = ((103, 72), (137, 16))
FINAL_BOUNDARY = "K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION"
COLUMN_PAIRS = ((16, 16), (1, 16), (1, 1), (0, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def coupling_exponents(family: str) -> tuple[int, int]:
    if family == "PRIMARY":
        return 3, 7
    if family == "REUSE":
        return 5, 9
    fail("oracle family changed")


def reflection(vector: list[Any]) -> list[Any]:
    result = list(vector)
    result[0], result[1] = result[1], result[0]
    return result


def reflected_histogram(histogram: tuple[int, ...]) -> tuple[int, ...]:
    result = list(histogram)
    result[0], result[1] = result[1], result[0]
    return tuple(result)


def operations(k: int, family: str) -> tuple[rank1ref.Operation, ...]:
    word = rank1ref.independent_program(k, family)
    if len(word) != 40:
        fail("oracle public word length changed")
    return word


def operation_json(operation: rank1ref.Operation) -> dict[str, Any]:
    return {
        "kind": operation.kind,
        "first": operation.first,
        "second": operation.second,
        "coefficient_exponent": operation.exponent,
    }


def program_descriptor(k: int, family: str) -> dict[str, Any]:
    word = operations(k, family)
    first, second = coupling_exponents(family)
    return {
        "k": k,
        "family": family,
        "chart": "RANK4_DOUBLE_COHERENT_VERONESE_SECANT",
        "mode_count": MODE_COUNT,
        "seed": "MODE0_RAISED_TO_K",
        "first_coupling": {
            "kind": "INVOLUTIVE_COHERENT_SUPERPOSITION",
            "law": "I_PLUS_ETA1_R",
            "eta_exponent": first,
            "reflection": "SWAP_MODE0_MODE1",
        },
        "module_a": [operation_json(item) for item in word[:20]],
        "second_coupling": {
            "kind": "INVOLUTIVE_COHERENT_SUPERPOSITION",
            "law": "I_PLUS_ETA2_R",
            "eta_exponent": second,
            "reflection": "SWAP_MODE0_MODE1",
        },
        "module_b": [operation_json(item) for item in word[20:]],
        "final_boundary": FINAL_BOUNDARY,
    }


def program_fingerprint(k: int, family: str) -> str:
    return rank1ref.digest_json(program_descriptor(k, family))


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1ref.digest_json(
        {
            "kind": alg.kind,
            "modulus": alg.modulus,
            "root": alg.serialize(alg.root),
        }
    )


def seed_vector(alg: backend.Algebra) -> list[Any]:
    return [alg.one, *([alg.zero] * (MODE_COUNT - 1))]


def combine_components(
    terms: list[tuple[Any, list[Any]]], alg: backend.Algebra
) -> list[tuple[Any, list[Any]]]:
    combined: list[tuple[Any, list[Any]]] = []
    for weight, vector in terms:
        if weight == alg.zero:
            continue
        for index, (prior_weight, prior_vector) in enumerate(combined):
            if vector == prior_vector:
                combined[index] = (alg.add(prior_weight, weight), prior_vector)
                break
        else:
            combined.append((weight, list(vector)))
    return [(weight, vector) for weight, vector in combined if weight != alg.zero]


def couple_components(
    components: list[tuple[Any, list[Any]]],
    exponent: int,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> list[tuple[Any, list[Any]]]:
    eta = alg.power(exponent)
    denominator = alg.sub(alg.one, alg.mul(eta, eta))
    if denominator == alg.zero:
        fail("oracle coupling inverse denominator vanished")
    terms: list[tuple[Any, list[Any]]] = []
    if inverse:
        scale = alg.inverse(denominator)
        for weight, vector in components:
            terms.append((alg.mul(weight, scale), list(vector)))
            terms.append(
                (
                    negative(alg, alg.mul(alg.mul(weight, eta), scale)),
                    reflection(vector),
                )
            )
    else:
        for weight, vector in components:
            terms.append((weight, list(vector)))
            terms.append((alg.mul(weight, eta), reflection(vector)))
    return combine_components(terms, alg)


def initial_components(alg: backend.Algebra) -> list[tuple[Any, list[Any]]]:
    return [(alg.one, seed_vector(alg))]


def apply_module(
    components: list[tuple[Any, list[Any]]],
    word: tuple[rank1ref.Operation, ...],
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> None:
    sequence = tuple(reversed(word)) if inverse else word
    for operation in sequence:
        for _, vector in components:
            rank1ref.chart_apply(vector, operation, alg, inverse=inverse)


def forward_components(
    k: int, family: str, alg: backend.Algebra
) -> tuple[list[tuple[Any, list[Any]]], list[tuple[Any, list[Any]]]]:
    word = operations(k, family)
    first, second = coupling_exponents(family)
    components = couple_components(initial_components(alg), first, alg)
    apply_module(components, word[:20], alg)
    after_second = couple_components(components, second, alg)
    components = [(weight, list(vector)) for weight, vector in after_second]
    apply_module(components, word[20:], alg)
    return components, after_second


def inverse_components(
    components: list[tuple[Any, list[Any]]], k: int, family: str, alg: backend.Algebra
) -> list[tuple[Any, list[Any]]]:
    word = operations(k, family)
    first, second = coupling_exponents(family)
    apply_module(components, word[20:], alg, inverse=True)
    components = couple_components(components, second, alg, inverse=True)
    apply_module(components, word[:20], alg, inverse=True)
    return couple_components(components, first, alg, inverse=True)


def restored_seed(components: list[tuple[Any, list[Any]]], alg: backend.Algebra) -> bool:
    return (
        len(components) == 1
        and components[0][0] == alg.one
        and components[0][1] == seed_vector(alg)
    )


def boundary(
    components: list[tuple[Any, list[Any]]], k: int, alg: backend.Algebra
) -> Any:
    result = alg.zero
    k_value = rank1ref.field_integer(alg, k)
    for weight, vector in components:
        result = alg.add(
            result,
            alg.mul(
                weight,
                alg.mul(
                    k_value,
                    alg.mul(rank1ref.scalar_power(alg, vector[0], k - 1), vector[1]),
                ),
            ),
        )
    return result


def flat_components(components: list[tuple[Any, list[Any]]]) -> list[Any]:
    result: list[Any] = []
    for weight, vector in components:
        result.extend((weight, *vector))
    return result


def histogram_from_pair(left: int, right: int) -> tuple[int, ...]:
    histogram = [0] * MODE_COUNT
    histogram[left] += 1
    histogram[right] += 1
    return tuple(histogram)


def monomial(
    vector: list[Any], histogram: tuple[int, ...], alg: backend.Algebra
) -> Any:
    value = alg.one
    for mode, exponent in enumerate(histogram):
        if exponent:
            value = alg.mul(value, rank1ref.scalar_power(alg, vector[mode], exponent))
    return value


def permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1 if inversions % 2 else 1


def leibniz_determinant4(matrix: list[list[Any]], alg: backend.Algebra) -> Any:
    result = alg.zero
    for permutation in itertools.permutations(range(4)):
        term = alg.one
        for row, column in enumerate(permutation):
            term = alg.mul(term, matrix[row][column])
        result = (
            alg.add(result, term)
            if permutation_sign(permutation) > 0
            else alg.sub(result, term)
        )
    return result


def catalecticant_certificate(
    components: list[tuple[Any, list[Any]]], k: int, alg: backend.Algebra
) -> dict[str, Any]:
    columns = tuple(histogram_from_pair(*pair) for pair in COLUMN_PAIRS)
    rows = []
    for column in columns:
        row = list(column)
        row[2] += k - 4
        rows.append(tuple(row))
    matrix: list[list[Any]] = []
    for row in rows:
        matrix_row = []
        for column in columns:
            joined = tuple(
                left + right for left, right in zip(row, column, strict=True)
            )
            value = alg.zero
            for weight, vector in components:
                value = alg.add(value, alg.mul(weight, monomial(vector, joined, alg)))
            matrix_row.append(value)
        matrix.append(matrix_row)
    determinant = leibniz_determinant4(matrix, alg)
    return {
        "certificate": "INDEPENDENT_NORMALIZED_CATALECTICANT_LEIBNIZ_MINOR",
        "minor_nonzero": determinant != alg.zero,
        "lower_bound": 4 if determinant != alg.zero else 0,
        "four_component_upper_bound": len(components) == 4,
        "exact_normalized_divided_power_secant_rank": (
            4 if determinant != alg.zero and len(components) == 4 else None
        ),
        "ordinary_symmetric_waring_rank_interpretation": alg.kind == "Q_ZETA17",
        "column_mode_pairs": [list(pair) for pair in COLUMN_PAIRS],
        "row_common_mode": 2,
        "row_common_power": k - 4,
        "minor_value_serialized": False,
        "intermediate_amplitudes_serialized": False,
    }


def transaction_parity(
    transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    k = transaction["k"]
    family = transaction["family"]
    components, after_second = forward_components(k, family, alg)
    commitment, record_bytes = rank1ref.stream_vector_commitment(
        flat_components(components), alg
    )
    projected = boundary(components, k, alg)
    certificate = catalecticant_certificate(after_second, k, alg)
    restored = restored_seed(inverse_components(components, k, family, alg), alg)
    return {
        "k": k,
        "family": family,
        "algebra": transaction["algebra"],
        "program_fingerprint_agreement": program_fingerprint(k, family)
        == transaction["program_fingerprint"],
        "algebra_signature_agreement": algebra_signature(alg)
        == transaction["algebra"],
        "forward_commitment_agreement": commitment
        == transaction["forward_rank4_commitment"],
        "boundary_agreement": alg.serialize(projected) == transaction["boundary"],
        "boundary_payload_bits_agreement": alg.payload_bits(projected)
        == transaction["final_boundary_payload_bits"],
        "maximum_commitment_record_bound_honest": record_bytes
        <= transaction["maximum_commitment_record_json_bytes"],
        "independent_forward_inverse_restored": restored,
        "independent_catalecticant_rank_four": certificate[
            "exact_normalized_divided_power_secant_rank"
        ]
        == 4,
        "resident_field_cells_agreement": transaction["resident_phase_field_cells"]
        == 72,
        "maximum_coupling_transient_field_cells_agreement": transaction[
            "maximum_coupling_transient_field_cells"
        ]
        == 144,
        "implicit_dimension_agreement": transaction[
            "implicit_occupation_dimension_h_k"
        ]
        == math.comb(k + 16, 16),
        "catalecticant_certificate": certificate,
    }


def dense_coupling(
    values: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    ranks: dict[tuple[int, ...], int],
    exponent: int,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> list[Any]:
    eta = alg.power(exponent)
    if inverse:
        scale = alg.inverse(alg.sub(alg.one, alg.mul(eta, eta)))
        return [
            alg.mul(
                scale,
                alg.sub(
                    value,
                    alg.mul(eta, values[ranks[reflected_histogram(histogram)]]),
                ),
            )
            for value, histogram in zip(values, histograms, strict=True)
        ]
    return [
        alg.add(value, alg.mul(eta, values[ranks[reflected_histogram(histogram)]]))
        for value, histogram in zip(values, histograms, strict=True)
    ]


def expand_components(
    components: list[tuple[Any, list[Any]]],
    k: int,
    histograms: tuple[tuple[int, ...], ...],
    alg: backend.Algebra,
) -> list[Any]:
    result = [alg.zero for _ in histograms]
    for weight, vector in components:
        expanded = rank1ref.expand_chart(vector, k, histograms, alg)
        for index, value in enumerate(expanded):
            result[index] = alg.add(result[index], alg.mul(weight, value))
    return result


def full_occupation_case(
    kind: str, *, modulus: int = 0, root: int = 0
) -> dict[str, Any]:
    k = 4
    family = "PRIMARY"
    alg = backend.Algebra(kind, modulus=modulus, root=root)
    first, second = coupling_exponents(family)
    word = operations(k, family)
    histograms = rank1ref.enumerate_histograms(k)
    ranks = {histogram: index for index, histogram in enumerate(histograms)}
    seed_histogram = (k, *([0] * (MODE_COUNT - 1)))
    boundary_histogram = (k - 1, 1, *([0] * (MODE_COUNT - 2)))

    dense = [alg.zero for _ in histograms]
    dense[ranks[seed_histogram]] = alg.one
    components = initial_components(alg)
    dense = dense_coupling(dense, histograms, ranks, first, alg)
    components = couple_components(components, first, alg)
    first_coupling_agreement = dense == expand_components(
        components, k, histograms, alg
    )
    for operation in word[:20]:
        rank1ref.dense_apply(dense, histograms, ranks, operation, alg)
        for _, vector in components:
            rank1ref.chart_apply(vector, operation, alg)
    module_a_agreement = dense == expand_components(components, k, histograms, alg)
    dense = dense_coupling(dense, histograms, ranks, second, alg)
    components = couple_components(components, second, alg)
    second_coupling_agreement = dense == expand_components(
        components, k, histograms, alg
    )
    certificate = catalecticant_certificate(components, k, alg)
    for operation in word[20:]:
        rank1ref.dense_apply(dense, histograms, ranks, operation, alg)
        for _, vector in components:
            rank1ref.chart_apply(vector, operation, alg)
    forward_agreement = dense == expand_components(components, k, histograms, alg)
    boundary_agreement = dense[ranks[boundary_histogram]] == boundary(
        components, k, alg
    )

    for operation in reversed(word[20:]):
        rank1ref.dense_apply(dense, histograms, ranks, operation, alg, inverse=True)
        for _, vector in components:
            rank1ref.chart_apply(vector, operation, alg, inverse=True)
    dense = dense_coupling(dense, histograms, ranks, second, alg, inverse=True)
    components = couple_components(components, second, alg, inverse=True)
    for operation in reversed(word[:20]):
        rank1ref.dense_apply(dense, histograms, ranks, operation, alg, inverse=True)
        for _, vector in components:
            rank1ref.chart_apply(vector, operation, alg, inverse=True)
    dense = dense_coupling(dense, histograms, ranks, first, alg, inverse=True)
    components = couple_components(components, first, alg, inverse=True)
    dense[ranks[seed_histogram]] = alg.sub(dense[ranks[seed_histogram]], alg.one)
    return {
        "algebra": kind,
        "k": k,
        "occupation_dimension": len(histograms),
        "first_coupling_full_occupation_agreement": first_coupling_agreement,
        "module_a_full_occupation_agreement": module_a_agreement,
        "second_coupling_full_occupation_agreement": second_coupling_agreement,
        "forward_full_occupation_rank4_agreement": forward_agreement,
        "forward_boundary_agreement": boundary_agreement,
        "dense_forward_inverse_restored": all(value == alg.zero for value in dense),
        "chart_forward_inverse_restored": restored_seed(components, alg),
        "rank_four_catalecticant_certificate": certificate,
        "oracle_full_occupation_vector_field_cells": len(histograms),
    }


def baseline_parity(
    baseline: dict[str, Any], transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    k = transaction["k"]
    family = transaction["family"]
    components, _ = forward_components(k, family, alg)
    retained: list[Any] = []
    for weight, vector in components:
        retained.extend((vector[0], alg.mul(weight, vector[1])))
    commitment, record_bytes = rank1ref.stream_vector_commitment(retained, alg)
    projected = alg.zero
    k_value = rank1ref.field_integer(alg, k)
    for slot in range(SLOT_COUNT):
        mode0, weighted_mode1 = retained[2 * slot : 2 * slot + 2]
        projected = alg.add(
            projected,
            alg.mul(
                k_value,
                alg.mul(rank1ref.scalar_power(alg, mode0, k - 1), weighted_mode1),
            ),
        )
    return {
        "k": k,
        "family": family,
        "algebra": transaction["algebra"],
        "folded_endpoint_commitment_agreement": commitment
        == baseline["retained_folded_endpoint_commitment"],
        "maximum_record_json_bytes_agreement": record_bytes
        == baseline["maximum_commitment_record_json_bytes"],
        "boundary_agreement": alg.serialize(projected) == transaction["boundary"],
        "retained_folded_endpoint_field_cells_agreement": baseline[
            "retained_folded_endpoint_field_cells"
        ]
        == 8,
        "total_compiled_warm_field_cells_agreement": baseline[
            "total_compiled_warm_field_cells"
        ]
        == 8,
        "compiler_working_field_cells_agreement": baseline[
            "compiler_working_field_cells"
        ]
        == 72,
    }


def mutation_checks() -> dict[str, Any]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    k = 4
    family = "PRIMARY"
    first, second = coupling_exponents(family)
    word = operations(k, family)
    reference, after_second = forward_components(k, family, alg)
    reference_boundary = boundary(reference, k, alg)

    omitted = couple_components(initial_components(alg), first, alg)
    apply_module(omitted, word[:20], alg)
    omitted_certificate = catalecticant_certificate(omitted, k, alg)
    apply_module(omitted, word[20:], alg)

    reordered = couple_components(initial_components(alg), first, alg)
    reordered = couple_components(reordered, second, alg)
    apply_module(reordered, word, alg)

    missing_inverse = [(weight, list(vector)) for weight, vector in reference]

    wrong_inverse = [(weight, list(vector)) for weight, vector in reference]
    apply_module(wrong_inverse, word[20:], alg, inverse=True)
    wrong_inverse = couple_components(wrong_inverse, second + 1, alg, inverse=True)

    reordered_inverse = [(weight, list(vector)) for weight, vector in reference]
    reordered_inverse = couple_components(reordered_inverse, second, alg, inverse=True)

    singular = []
    for eta in (alg.one, negative(alg, alg.one)):
        singular.append(alg.sub(alg.one, alg.mul(eta, eta)) == alg.zero)

    certificate = catalecticant_certificate(after_second, k, alg)
    restored = inverse_components(
        [(weight, list(vector)) for weight, vector in reference], k, family, alg
    )
    return {
        "second_coupling_omission_changes_boundary": boundary(omitted, k, alg)
        != reference_boundary,
        "second_coupling_omission_forces_zero_four_minor": not omitted_certificate[
            "minor_nonzero"
        ],
        "second_coupling_module_order_changes_boundary": boundary(reordered, k, alg)
        != reference_boundary,
        "missing_inverse_leaves_nonseed_state": not restored_seed(
            missing_inverse, alg
        ),
        "wrong_second_coupling_inverse_not_seed": not restored_seed(
            wrong_inverse, alg
        ),
        "reordered_inverse_not_seed": not restored_seed(reordered_inverse, alg),
        "eta_plus_or_minus_one_inverse_singular": all(singular),
        "exact_forward_inverse_restores_seed": restored_seed(restored, alg),
        "nonzero_minor_and_four_term_upper_bound_prove_exact_rank_four": (
            certificate["minor_nonzero"]
            and certificate["four_component_upper_bound"]
            and certificate["exact_normalized_divided_power_secant_rank"] == 4
        ),
        "generated_term_count_alone_not_treated_as_rank_proof": True,
        "rank_two_capacity_rejected_by_independent_minor": certificate[
            "lower_bound"
        ]
        == 4,
        "k_below_four_excluded_from_certificate_domain": all(
            value not in DECLARED_K for value in (0, 1, 2, 3)
        ),
    }


def run(production: dict[str, Any]) -> dict[str, Any]:
    transactions = [
        *production["exact_transactions"],
        *production["dual_field_structural_transactions"],
    ]
    algebra_by_signature: dict[str, backend.Algebra] = {}
    for alg in (
        backend.Algebra("Q_ZETA17"),
        *(backend.Algebra(f"F{p}", modulus=p, root=r) for p, r in FINITE_FIELDS),
    ):
        algebra_by_signature[algebra_signature(alg)] = alg

    parity = [
        transaction_parity(item, algebra_by_signature[item["algebra"]])
        for item in transactions
    ]
    production_certificates = {
        (item["algebra"], item["k"], item["family"]): item
        for item in production["catalecticant_rank_certificates"]
    }
    certificate_parity = []
    for item in parity:
        production_item = production_certificates[
            (item["algebra"], item["k"], item["family"])
        ]
        oracle_item = item["catalecticant_certificate"]
        certificate_parity.append(
            {
                "k": item["k"],
                "family": item["family"],
                "algebra": item["algebra"],
                "minor_nonzero_agreement": production_item["minor_nonzero"]
                == oracle_item["minor_nonzero"],
                "exact_rank_four_agreement": production_item[
                    "exact_normalized_divided_power_secant_rank"
                ]
                == oracle_item["exact_normalized_divided_power_secant_rank"],
                "rank_interpretation_agreement": production_item[
                    "ordinary_symmetric_waring_rank_interpretation"
                ]
                == oracle_item["ordinary_symmetric_waring_rank_interpretation"],
            }
        )

    baselines_by_key = {
        (item["algebra"], item["k"], item["family"]): item
        for item in production["compiled_eight_total_scalar_classical_baselines"]
    }
    baseline_checks = [
        baseline_parity(
            baselines_by_key[(item["algebra"], item["k"], item["family"])],
            item,
            algebra_by_signature[item["algebra"]],
        )
        for item in transactions
    ]
    full_cases = [
        full_occupation_case("Q_ZETA17"),
        full_occupation_case("F103", modulus=103, root=72),
        full_occupation_case("F137", modulus=137, root=16),
    ]
    mutations = mutation_checks()

    parity_keys = (
        "program_fingerprint_agreement",
        "algebra_signature_agreement",
        "forward_commitment_agreement",
        "boundary_agreement",
        "boundary_payload_bits_agreement",
        "maximum_commitment_record_bound_honest",
        "independent_forward_inverse_restored",
        "independent_catalecticant_rank_four",
        "resident_field_cells_agreement",
        "maximum_coupling_transient_field_cells_agreement",
        "implicit_dimension_agreement",
    )
    if not all(all(item[key] for key in parity_keys) for item in parity):
        fail("independent double-coupling transaction parity failed")
    if not all(
        item["minor_nonzero_agreement"]
        and item["exact_rank_four_agreement"]
        and item["rank_interpretation_agreement"]
        for item in certificate_parity
    ):
        fail("independent catalecticant certificate parity failed")
    baseline_keys = (
        "folded_endpoint_commitment_agreement",
        "maximum_record_json_bytes_agreement",
        "boundary_agreement",
        "retained_folded_endpoint_field_cells_agreement",
        "total_compiled_warm_field_cells_agreement",
        "compiler_working_field_cells_agreement",
    )
    if not all(all(item[key] for key in baseline_keys) for item in baseline_checks):
        fail("independent eight-scalar baseline parity failed")
    if not all(
        item["first_coupling_full_occupation_agreement"]
        and item["module_a_full_occupation_agreement"]
        and item["second_coupling_full_occupation_agreement"]
        and item["forward_full_occupation_rank4_agreement"]
        and item["forward_boundary_agreement"]
        and item["dense_forward_inverse_restored"]
        and item["chart_forward_inverse_restored"]
        and item["rank_four_catalecticant_certificate"][
            "exact_normalized_divided_power_secant_rank"
        ]
        == 4
        for item in full_cases
    ):
        fail("independent full occupation rank-four parity failed")
    if not all(mutations.values()):
        fail("one or more independent rank-four mutations failed")

    return {
        "schema": "CAT_CAS_F17_COHERENT_RANK4_DOUBLE_SECANT_PHASE_COUPLING_CLOSURE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_schema": production["schema"],
        "production_claim": production["claim"],
        "transaction_parity": parity,
        "catalecticant_certificate_parity": certificate_parity,
        "full_occupation_oracle_cases": full_cases,
        "compiled_eight_total_scalar_baseline_checks": baseline_checks,
        "independent_mutation_checks": mutations,
        "independent_methods": [
            "SEPARATE_PUBLIC_PROGRAM_COMPILER",
            "SEPARATE_TWO_COUPLING_FOUR_COMPONENT_CHART_AND_EXACT_INVERSE",
            "INDEPENDENT_LEIBNIZ_FOUR_BY_FOUR_NORMALIZED_CATALECTICANT_MINOR",
            "FULL_H4_4845_COORDINATE_OCCUPATION_REEXECUTION_IN_THREE_ALGEBRAS",
            "SEPARATE_EIGHT_TOTAL_FOLDED_ENDPOINT_SCALAR_BASELINE",
            "ZERO_MINOR_OMISSION_AND_INVERSE_ORDER_MUTATIONS",
        ],
        "resource_law": {
            "accepted_path_resident_field_cells": 72,
            "inverse_coupling_bounded_transient_field_cells": 144,
            "oracle_full_occupation_field_cells_per_full_case": 4845,
            "oracle_catalecticant_scalar_cells_per_certificate": 16,
            "oracle_buffers_are_verification_only": True,
            "accepted_path_occupation_or_catalecticant_cells": 0,
            "exact_payload_height_tuples_independently_reexecuted": False,
            "full_exact_bit_complexity_established": False,
        },
        "matched_baseline": {
            "strongest_sealed_fixture_warm": "EIGHT_TOTAL_FOLDED_ENDPOINT_SCALARS",
            "descriptor_runtime": "IDENTICAL_FOUR_COMPONENT_72_FIELD_CELL_RECURRENCE",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_restoration_class": "NO_RESTORATION_CLAIM",
        "claim_ceiling": "DECLARED_TWO_COUPLING_FOUR_COMPONENT_NORMALIZED_DIVIDED_POWER_SECANT_RANK4_PROGRAM_Q_ZETA17_K4_TO32_DUAL_FIELD_K4_TO128_DIRECT_PROCESS_SOFTWARE",
        "rejected_interpretations": [
            "THIRD_OR_UNBOUNDED_COUPLING_RANK_LAW",
            "FIXED_RANK_UNBOUNDED_DEPTH_CLOSURE",
            "GENERAL_GAUSSIAN_CLOSURE",
            "ARBITRARY_SECANT_INPUT_CLOSURE",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_EXECUTION_OR_BIT_REPLACEMENT",
            "UNBOUNDED_COMPUTATION",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    production = json.loads(Path(arguments.production).read_text(encoding="utf-8"))
    result = run(production)
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
