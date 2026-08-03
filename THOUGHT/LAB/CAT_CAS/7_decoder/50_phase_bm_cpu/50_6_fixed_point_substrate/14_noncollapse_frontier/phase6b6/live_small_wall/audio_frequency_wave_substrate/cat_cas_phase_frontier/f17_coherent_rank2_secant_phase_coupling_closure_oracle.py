#!/usr/bin/env python3
"""Independent oracle for the bounded rank-two coherent-secant successor.

The oracle never imports the M130 production module.  It reconstructs the
public word, involutive coupling, two-component chart, full H(4) occupation
recurrence, inverse, and strongest compiled endpoint baseline separately.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure_oracle as rank1ref
import f17_nonlinear_canonical_mps_separator_chart as backend


MODE_COUNT = 17
DECLARED_K = (4, 8, 16, 32, 64, 128)
EXACT_K = (4, 8, 16, 32)
FINITE_FIELDS = ((103, 72), (137, 16))
FINAL_BOUNDARY = "K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION"


def fail(message: str) -> None:
    raise RuntimeError(message)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def eta_exponent(family: str) -> int:
    if family == "PRIMARY":
        return 3
    if family == "REUSE":
        return 5
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
    result = rank1ref.independent_program(k, family)
    if len(result) != 40:
        fail("oracle public word length changed")
    return result


def operation_json(operation: rank1ref.Operation) -> dict[str, Any]:
    return {
        "kind": operation.kind,
        "first": operation.first,
        "second": operation.second,
        "coefficient_exponent": operation.exponent,
    }


def program_descriptor(k: int, family: str) -> dict[str, Any]:
    word = operations(k, family)
    return {
        "k": k,
        "family": family,
        "chart": "RANK2_COHERENT_VERONESE_SECANT",
        "mode_count": MODE_COUNT,
        "seed": "MODE0_RAISED_TO_K",
        "coupling": {
            "kind": "INVOLUTIVE_COHERENT_SUPERPOSITION",
            "law": "I_PLUS_ETA_R",
            "eta_exponent": eta_exponent(family),
            "reflection": "SWAP_MODE0_MODE1",
        },
        "module_a": [operation_json(item) for item in word[:20]],
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


def coupled_components(
    family: str, alg: backend.Algebra
) -> list[tuple[Any, list[Any]]]:
    seed = seed_vector(alg)
    eta = alg.power(eta_exponent(family))
    return [(alg.one, seed), (eta, reflection(seed))]


def apply_word(
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


def boundary(
    components: list[tuple[Any, list[Any]]], k: int, alg: backend.Algebra
) -> Any:
    result = alg.zero
    for weight, vector in components:
        result = alg.add(
            result, alg.mul(weight, rank1ref.chart_boundary(vector, k, alg))
        )
    return result


def flat_components(components: list[tuple[Any, list[Any]]]) -> list[Any]:
    result: list[Any] = []
    for weight, vector in components:
        result.extend((weight, *vector))
    return result


def merge_components(
    terms: list[tuple[Any, list[Any]]], alg: backend.Algebra
) -> list[tuple[Any, list[Any]]]:
    result: list[tuple[Any, list[Any]]] = []
    for weight, vector in terms:
        if weight == alg.zero:
            continue
        for index, (prior_weight, prior_vector) in enumerate(result):
            if vector == prior_vector:
                result[index] = (alg.add(prior_weight, weight), prior_vector)
                break
        else:
            result.append((weight, list(vector)))
    return [(weight, vector) for weight, vector in result if weight != alg.zero]


def inverse_coupling(
    components: list[tuple[Any, list[Any]]], family: str, alg: backend.Algebra
) -> list[tuple[Any, list[Any]]]:
    eta = alg.power(eta_exponent(family))
    denominator = alg.sub(alg.one, alg.mul(eta, eta))
    if denominator == alg.zero:
        fail("oracle inverse denominator vanished")
    scale = alg.inverse(denominator)
    terms: list[tuple[Any, list[Any]]] = []
    for weight, vector in components:
        scaled = alg.mul(weight, scale)
        terms.append((scaled, list(vector)))
        terms.append(
            (
                negative(alg, alg.mul(alg.mul(weight, eta), scale)),
                reflection(vector),
            )
        )
    return merge_components(terms, alg)


def chart_restore(
    components: list[tuple[Any, list[Any]]],
    k: int,
    family: str,
    alg: backend.Algebra,
) -> bool:
    apply_word(components, operations(k, family), alg, inverse=True)
    restored = inverse_coupling(components, family, alg)
    return (
        len(restored) == 1
        and restored[0][0] == alg.one
        and restored[0][1] == seed_vector(alg)
    )


def rank_two_derivative_certificate(
    k: int, family: str, alg: backend.Algebra
) -> dict[str, Any]:
    eta = alg.power(eta_exponent(family))
    k_value = rank1ref.field_integer(alg, k)
    determinant = alg.mul(alg.mul(k_value, k_value), eta)
    return {
        "certificate": "INDEPENDENT_FIRST_DERIVATIVE_TWO_BY_TWO_MINOR",
        "nonzero": determinant != alg.zero,
        "serialized_minor": alg.serialize(determinant),
    }


def transaction_parity(
    transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    k = transaction["k"]
    family = transaction["family"]
    components = coupled_components(family, alg)
    apply_word(components, operations(k, family), alg)
    commitment, record_bytes = rank1ref.stream_vector_commitment(
        flat_components(components), alg
    )
    projected = boundary(components, k, alg)
    rank_certificate = rank_two_derivative_certificate(k, family, alg)
    restored = chart_restore(components, k, family, alg)
    checks = {
        "program_fingerprint_agreement": program_fingerprint(k, family)
        == transaction["program_fingerprint"],
        "algebra_signature_agreement": algebra_signature(alg)
        == transaction["algebra"],
        "forward_commitment_agreement": commitment
        == transaction["forward_secant_commitment"],
        "boundary_agreement": alg.serialize(projected) == transaction["boundary"],
        "boundary_payload_bits_agreement": alg.payload_bits(projected)
        == transaction["final_boundary_payload_bits"],
        "maximum_commitment_record_bound_honest": record_bytes
        <= transaction["maximum_commitment_record_json_bytes"],
        "independent_rank_two_certificate_nonzero": rank_certificate["nonzero"],
        "independent_forward_inverse_restored": restored,
        "resident_field_cells_agreement": transaction["resident_phase_field_cells"]
        == 36,
        "inverse_transient_field_cells_agreement": transaction[
            "inverse_coupling_transient_field_cells"
        ]
        == 72,
        "implicit_dimension_agreement": transaction[
            "implicit_occupation_dimension_h_k"
        ]
        == math.comb(k + 16, 16),
    }
    return {
        "k": k,
        "family": family,
        "algebra": transaction["algebra"],
        **checks,
        "rank_two_derivative_certificate": rank_certificate,
    }


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


def dense_inverse_coupling(
    values: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    ranks: dict[tuple[int, ...], int],
    family: str,
    alg: backend.Algebra,
) -> list[Any]:
    eta = alg.power(eta_exponent(family))
    scale = alg.inverse(alg.sub(alg.one, alg.mul(eta, eta)))
    return [
        alg.mul(
            scale,
            alg.sub(value, alg.mul(eta, values[ranks[reflected_histogram(histogram)]])),
        )
        for value, histogram in zip(values, histograms, strict=True)
    ]


def full_occupation_case(
    kind: str, *, modulus: int = 0, root: int = 0
) -> dict[str, Any]:
    k = 4
    family = "PRIMARY"
    alg = backend.Algebra(kind, modulus=modulus, root=root)
    histograms = rank1ref.enumerate_histograms(k)
    ranks = {histogram: index for index, histogram in enumerate(histograms)}
    seed_histogram = (k, *([0] * (MODE_COUNT - 1)))
    reflected_seed = (0, k, *([0] * (MODE_COUNT - 2)))
    boundary_histogram = (k - 1, 1, *([0] * (MODE_COUNT - 2)))
    eta = alg.power(eta_exponent(family))

    dense = [alg.zero for _ in histograms]
    dense[ranks[seed_histogram]] = alg.one
    dense[ranks[reflected_seed]] = eta
    components = coupled_components(family, alg)
    initial_expansion_agreement = dense == expand_components(
        components, k, histograms, alg
    )
    for operation in operations(k, family):
        rank1ref.dense_apply(dense, histograms, ranks, operation, alg)
        for _, vector in components:
            rank1ref.chart_apply(vector, operation, alg)
    forward_expansion_agreement = dense == expand_components(
        components, k, histograms, alg
    )
    boundary_agreement = dense[ranks[boundary_histogram]] == boundary(
        components, k, alg
    )
    for operation in reversed(operations(k, family)):
        rank1ref.dense_apply(
            dense, histograms, ranks, operation, alg, inverse=True
        )
        for _, vector in components:
            rank1ref.chart_apply(vector, operation, alg, inverse=True)
    dense = dense_inverse_coupling(dense, histograms, ranks, family, alg)
    dense[ranks[seed_histogram]] = alg.sub(
        dense[ranks[seed_histogram]], alg.one
    )
    restored_components = inverse_coupling(components, family, alg)
    chart_restored = (
        len(restored_components) == 1
        and restored_components[0][0] == alg.one
        and restored_components[0][1] == seed_vector(alg)
    )
    certificate = rank_two_derivative_certificate(k, family, alg)
    return {
        "algebra": kind,
        "k": k,
        "occupation_dimension": len(histograms),
        "initial_coupling_full_occupation_agreement": initial_expansion_agreement,
        "forward_full_occupation_secant_agreement": forward_expansion_agreement,
        "forward_boundary_agreement": boundary_agreement,
        "dense_forward_inverse_restored": all(value == alg.zero for value in dense),
        "chart_forward_inverse_restored": chart_restored,
        "rank_two_derivative_certificate": certificate,
        "oracle_full_occupation_vector_field_cells": len(histograms),
    }


def baseline_parity(
    baseline: dict[str, Any], transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    k = transaction["k"]
    family = transaction["family"]
    components = coupled_components(family, alg)
    apply_word(components, operations(k, family), alg)
    retained: list[Any] = []
    for weight, vector in components:
        retained.extend((vector[0], alg.mul(weight, vector[1])))
    commitment, record_bytes = rank1ref.stream_vector_commitment(
        retained, alg
    )
    projected = alg.zero
    k_value = rank1ref.field_integer(alg, k)
    for slot in range(2):
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
        == 4,
        "total_compiled_warm_field_cells_agreement": baseline[
            "total_compiled_warm_field_cells"
        ] == 4,
        "compiler_working_field_cells_agreement": baseline[
            "compiler_working_field_cells"
        ]
        == 36,
    }


def mutation_checks() -> dict[str, Any]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    k = 4
    family = "PRIMARY"
    word = operations(k, family)
    reference = coupled_components(family, alg)
    apply_word(reference, word, alg)
    reference_boundary = boundary(reference, k, alg)

    omitted = [(alg.one, seed_vector(alg))]
    apply_word(omitted, word, alg)
    omitted_boundary = boundary(omitted, k, alg)

    reordered = coupled_components(family, alg)
    apply_word(reordered, (*word[20:], *word[:20]), alg)
    reordered_boundary = boundary(reordered, k, alg)

    missing_inverse = coupled_components(family, alg)
    apply_word(missing_inverse, word, alg)

    wrong_inverse = coupled_components(family, alg)
    apply_word(wrong_inverse, word, alg)
    apply_word(wrong_inverse, word, alg, inverse=True)
    eta = alg.power(eta_exponent(family))
    wrong_terms: list[tuple[Any, list[Any]]] = []
    for weight, vector in wrong_inverse:
        wrong_terms.extend(
            ((weight, list(vector)), (alg.mul(weight, eta), reflection(vector)))
        )
    wrong_inverse_result = merge_components(wrong_terms, alg)

    repeated = coupled_components(family, alg)
    apply_word(repeated, word, alg)
    repeated_terms: list[tuple[Any, list[Any]]] = []
    for weight, vector in repeated:
        repeated_terms.extend(
            ((weight, list(vector)), (alg.mul(weight, eta), reflection(vector)))
        )
    repeated_rank = len(merge_components(repeated_terms, alg))

    singular_denominators = {}
    for label, value in (("ETA_PLUS_ONE", alg.one), ("ETA_MINUS_ONE", negative(alg, alg.one))):
        singular_denominators[label] = (
            alg.sub(alg.one, alg.mul(value, value)) == alg.zero
        )

    return {
        "coupling_omission_changes_boundary": omitted_boundary != reference_boundary,
        "module_order_changes_boundary": reordered_boundary != reference_boundary,
        "missing_inverse_leaves_nonseed_state": missing_inverse
        != [(alg.one, seed_vector(alg))],
        "wrong_coupling_inverse_rejected_by_seed_equality": wrong_inverse_result
        != [(alg.one, seed_vector(alg))],
        "eta_plus_or_minus_one_inverse_singular": all(singular_denominators.values()),
        "rank_one_seed_to_rank_two_to_rank_one": chart_restore(
            reference, k, family, alg
        ),
        "second_generic_coupling_generates_four_distinct_terms": repeated_rank == 4,
        "second_generic_coupling_observed_generated_term_count": repeated_rank,
        "second_generic_coupling_minimal_secant_rank_established": False,
        "k_one_excluded_from_declared_program_domain": 1 not in DECLARED_K,
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
    baselines_by_key = {
        (item["algebra"], item["k"], item["family"]): item
        for item in production[
            "compiled_four_dynamic_scalar_classical_baselines"
        ]
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
        "independent_rank_two_certificate_nonzero",
        "independent_forward_inverse_restored",
        "resident_field_cells_agreement",
        "inverse_transient_field_cells_agreement",
        "implicit_dimension_agreement",
    )
    if not all(all(item[key] for key in parity_keys) for item in parity):
        fail("independent secant transaction parity failed")
    baseline_keys = (
        "folded_endpoint_commitment_agreement",
        "maximum_record_json_bytes_agreement",
        "boundary_agreement",
        "retained_folded_endpoint_field_cells_agreement",
        "total_compiled_warm_field_cells_agreement",
        "compiler_working_field_cells_agreement",
    )
    if not all(all(item[key] for key in baseline_keys) for item in baseline_checks):
        fail("independent compiled endpoint baseline parity failed")
    if not all(
        item["initial_coupling_full_occupation_agreement"]
        and item["forward_full_occupation_secant_agreement"]
        and item["forward_boundary_agreement"]
        and item["dense_forward_inverse_restored"]
        and item["chart_forward_inverse_restored"]
        and item["rank_two_derivative_certificate"]["nonzero"]
        for item in full_cases
    ):
        fail("independent full occupation secant parity failed")
    required_mutations = (
        "coupling_omission_changes_boundary",
        "module_order_changes_boundary",
        "missing_inverse_leaves_nonseed_state",
        "wrong_coupling_inverse_rejected_by_seed_equality",
        "eta_plus_or_minus_one_inverse_singular",
        "rank_one_seed_to_rank_two_to_rank_one",
        "second_generic_coupling_generates_four_distinct_terms",
        "k_one_excluded_from_declared_program_domain",
    )
    if not all(mutations[key] for key in required_mutations):
        fail("one or more independent secant mutations failed")

    return {
        "schema": "CAT_CAS_F17_COHERENT_RANK2_SECANT_PHASE_COUPLING_CLOSURE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_schema": production["schema"],
        "production_claim": production["claim"],
        "transaction_parity": parity,
        "full_occupation_oracle_cases": full_cases,
        "compiled_four_dynamic_scalar_baseline_checks": baseline_checks,
        "independent_mutation_checks": mutations,
        "independent_methods": [
            "SEPARATE_PUBLIC_PROGRAM_COMPILER",
            "SEPARATE_INVOLUTIVE_COUPLING_AND_TWO_COMPONENT_CHART",
            "FULL_H4_4845_COORDINATE_OCCUPATION_REEXECUTION_IN_THREE_ALGEBRAS",
            "INDEPENDENT_FIRST_DERIVATIVE_RANK_TWO_CERTIFICATE",
            "SEPARATE_EXACT_FORWARD_INVERSE_AND_DUPLICATE_MERGE",
            "SEPARATE_COMPILED_FOUR_TOTAL_FOLDED_ENDPOINT_SCALAR_BASELINE",
            "SECOND_GENERIC_COUPLING_FOUR_GENERATED_TERM_MUTATION_WITHOUT_MINIMAL_RANK_CLAIM",
        ],
        "resource_law": {
            "accepted_path_resident_field_cells": 36,
            "inverse_coupling_bounded_transient_field_cells": 72,
            "oracle_full_occupation_field_cells_per_full_case": 4845,
            "oracle_full_occupation_vectors_are_verification_only": True,
            "accepted_path_occupation_or_assignment_expansion_cells": 0,
            "exact_payload_height_tuples_independently_reexecuted": False,
            "full_exact_bit_complexity_established": False,
        },
        "matched_baseline": {
            "strongest_sealed_fixture_warm": "FOUR_TOTAL_FOLDED_ENDPOINT_SCALARS",
            "descriptor_runtime": "IDENTICAL_TWO_COMPONENT_36_FIELD_CELL_RECURRENCE",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_restoration_class": "NO_RESTORATION_CLAIM",
        "claim_ceiling": "DECLARED_K4_TO128_SINGLE_INVOLUTIVE_RANK1_TO_RANK2_SECANT_COUPLING_WITH_TWO_FIXED_NONCOMMUTING_CONSUMER_MODULES_EXACT_Q_ZETA17_K4_TO32_DUAL_FIELD_K4_TO128_DIRECT_PROCESS_SOFTWARE",
        "rejected_interpretations": [
            "REPEATED_COUPLING_FIXED_RANK_CLOSURE",
            "M127_GRID_ORBIT_SHEAR_CLOSURE",
            "ARBITRARY_SECANT_INPUT_CLOSURE",
            "GENERAL_GAUSSIAN_CLOSURE",
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
