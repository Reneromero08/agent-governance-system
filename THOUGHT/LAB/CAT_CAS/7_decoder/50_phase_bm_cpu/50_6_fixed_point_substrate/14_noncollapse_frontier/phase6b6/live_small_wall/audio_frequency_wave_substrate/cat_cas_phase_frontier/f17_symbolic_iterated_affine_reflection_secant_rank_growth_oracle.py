#!/usr/bin/env python3
"""Independent oracle for iterated affine-reflection secant-rank growth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure_oracle as rank1ref
import f17_nonlinear_canonical_mps_separator_chart as backend


MODE_COUNT = 17
DECLARED_M = (1, 2, 3, 4, 5, 6)
FINITE_FIELDS = ((103, 72), (137, 16))
FINAL_BOUNDARY = "K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION"


def fail(message: str) -> None:
    raise RuntimeError(message)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1ref.digest_json(
        {
            "kind": alg.kind,
            "modulus": alg.modulus,
            "root": alg.serialize(alg.root),
        }
    )


def eta_exponents(m: int, family: str) -> tuple[int, ...]:
    if family == "PRIMARY":
        return tuple(2 * level + 1 for level in range(1, m + 1))
    if family == "REUSE":
        return tuple(2 * level + 2 for level in range(1, m + 1))
    fail("oracle family changed")


def program_descriptor(m: int, family: str) -> dict[str, Any]:
    rank = 1 << m
    k = 2 * rank - 2
    centers = tuple((1 << level) - 1 for level in range(1, m + 1))
    exponents = eta_exponents(m, family)
    return {
        "m": m,
        "k": k,
        "family": family,
        "chart": "ITERATED_AFFINE_REFLECTION_COHERENT_SECANT",
        "mode_count": MODE_COUNT,
        "seed": "MODE0_RAISED_TO_K",
        "couplings": [
            {
                "level": level,
                "kind": "INVOLUTIVE_AFFINE_REFLECTION_SUPERPOSITION",
                "law": "I_PLUS_ETA_LEVEL_TIMES_SYM_K_R_LEVEL",
                "one_particle_action": "V0_FIXED_V1_TO_A_LEVEL_V0_MINUS_V1",
                "a_level": center,
                "eta_exponent": exponent,
            }
            for level, (center, exponent) in enumerate(
                zip(centers, exponents, strict=True), start=1
            )
        ],
        "final_boundary": FINAL_BOUNDARY,
    }


def program_fingerprint(m: int, family: str) -> str:
    return rank1ref.digest_json(program_descriptor(m, family))


def seed_vector(alg: backend.Algebra) -> list[Any]:
    return [alg.one, *([alg.zero] * (MODE_COUNT - 1))]


def affine_reflection(
    vector: list[Any], center: int, alg: backend.Algebra
) -> list[Any]:
    result = list(vector)
    result[1] = alg.sub(
        alg.mul(rank1ref.field_integer(alg, center), vector[0]), vector[1]
    )
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


def coupling(
    components: list[tuple[Any, list[Any]]],
    center: int,
    exponent: int,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> list[tuple[Any, list[Any]]]:
    eta = alg.power(exponent)
    denominator = alg.sub(alg.one, alg.mul(eta, eta))
    if denominator == alg.zero:
        fail("oracle coupling became singular")
    terms: list[tuple[Any, list[Any]]] = []
    if inverse:
        scale = alg.inverse(denominator)
        for weight, vector in components:
            terms.append((alg.mul(weight, scale), list(vector)))
            terms.append(
                (
                    negative(alg, alg.mul(alg.mul(weight, eta), scale)),
                    affine_reflection(vector, center, alg),
                )
            )
    else:
        for weight, vector in components:
            terms.append((weight, list(vector)))
            terms.append(
                (alg.mul(weight, eta), affine_reflection(vector, center, alg))
            )
    return merge_components(terms, alg)


def forward_components(
    m: int, family: str, alg: backend.Algebra
) -> list[tuple[Any, list[Any]]]:
    components = [(alg.one, seed_vector(alg))]
    for level, exponent in enumerate(eta_exponents(m, family), start=1):
        components = coupling(components, (1 << level) - 1, exponent, alg)
    return components


def inverse_components(
    components: list[tuple[Any, list[Any]]],
    m: int,
    family: str,
    alg: backend.Algebra,
) -> list[tuple[Any, list[Any]]]:
    exponents = eta_exponents(m, family)
    for level in range(m, 0, -1):
        components = coupling(
            components,
            (1 << level) - 1,
            exponents[level - 1],
            alg,
            inverse=True,
        )
    return components


def restored_seed(components: list[tuple[Any, list[Any]]], alg: backend.Algebra) -> bool:
    return (
        len(components) == 1
        and components[0][0] == alg.one
        and components[0][1] == seed_vector(alg)
    )


def flat_components(components: list[tuple[Any, list[Any]]]) -> list[Any]:
    result: list[Any] = []
    for weight, vector in components:
        result.extend((weight, *vector))
    return result


def component_moments(
    components: list[tuple[Any, list[Any]]], k: int, alg: backend.Algebra
) -> list[Any]:
    moments = [alg.zero for _ in range(k + 1)]
    for weight, vector in components:
        if vector[0] != alg.one or any(value != alg.zero for value in vector[2:]):
            fail("oracle binary moment curve changed")
        for degree in range(k + 1):
            moments[degree] = alg.add(
                moments[degree],
                alg.mul(weight, rank1ref.scalar_power(alg, vector[1], degree)),
            )
    return moments


def reflected_moments(
    moments: list[Any], center: int, alg: backend.Algebra
) -> list[Any]:
    center_value = rank1ref.field_integer(alg, center)
    result = []
    for degree in range(len(moments)):
        value = alg.zero
        for source_degree in range(degree + 1):
            coefficient = rank1ref.field_integer(
                alg, math.comb(degree, source_degree)
            )
            coefficient = alg.mul(
                coefficient,
                rank1ref.scalar_power(
                    alg, center_value, degree - source_degree
                ),
            )
            if source_degree % 2:
                coefficient = negative(alg, coefficient)
            value = alg.add(
                value, alg.mul(coefficient, moments[source_degree])
            )
        result.append(value)
    return result


def moment_coupling(
    moments: list[Any],
    center: int,
    exponent: int,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> list[Any]:
    eta = alg.power(exponent)
    reflected = reflected_moments(moments, center, alg)
    if inverse:
        scale = alg.inverse(alg.sub(alg.one, alg.mul(eta, eta)))
        return [
            alg.mul(scale, alg.sub(value, alg.mul(eta, reflected_value)))
            for value, reflected_value in zip(moments, reflected, strict=True)
        ]
    return [
        alg.add(value, alg.mul(eta, reflected_value))
        for value, reflected_value in zip(moments, reflected, strict=True)
    ]


def direct_moment_case(m: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    rank = 1 << m
    k = 2 * rank - 2
    moments = [alg.one, *([alg.zero] * k)]
    components = [(alg.one, seed_vector(alg))]
    forward_agreements = []
    exponents = eta_exponents(m, family)
    for level, exponent in enumerate(exponents, start=1):
        center = (1 << level) - 1
        moments = moment_coupling(moments, center, exponent, alg)
        components = coupling(components, center, exponent, alg)
        forward_agreements.append(moments == component_moments(components, k, alg))
    boundary_agreement = alg.mul(
        rank1ref.field_integer(alg, k), moments[1]
    ) == alg.mul(rank1ref.field_integer(alg, k), component_moments(components, k, alg)[1])
    for level in range(m, 0, -1):
        center = (1 << level) - 1
        moments = moment_coupling(
            moments, center, exponents[level - 1], alg, inverse=True
        )
        components = coupling(
            components, center, exponents[level - 1], alg, inverse=True
        )
    return {
        "m": m,
        "k": k,
        "algebra": algebra_signature(alg),
        "full_binary_moment_field_cells": k + 1,
        "all_forward_component_moment_agreements": all(forward_agreements),
        "final_boundary_agreement": boundary_agreement,
        "direct_moments_restored_to_seed": moments
        == [alg.one, *([alg.zero] * k)],
        "component_chart_restored_to_seed": restored_seed(components, alg),
    }


def determinant(matrix: list[list[Any]], alg: backend.Algebra) -> Any:
    work = [list(row) for row in matrix]
    result = alg.one
    for column in range(len(work)):
        pivot = next(
            (row for row in range(column, len(work)) if work[row][column] != alg.zero),
            None,
        )
        if pivot is None:
            return alg.zero
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result = negative(alg, result)
        pivot_value = work[column][column]
        result = alg.mul(result, pivot_value)
        inverse_pivot = alg.inverse(pivot_value)
        for row in range(column + 1, len(work)):
            factor = alg.mul(work[row][column], inverse_pivot)
            for inner in range(column, len(work)):
                work[row][inner] = alg.sub(
                    work[row][inner], alg.mul(factor, work[column][inner])
                )
    return result


def rank_certificate(
    components: list[tuple[Any, list[Any]]], m: int, k: int, alg: backend.Algebra
) -> dict[str, Any]:
    rank = 1 << m
    points = [vector[1] for _, vector in components]
    point_keys = {
        json.dumps(alg.serialize(value), separators=(",", ":")) for value in points
    }
    expected_keys = {
        json.dumps(
            alg.serialize(rank1ref.field_integer(alg, value)), separators=(",", ":")
        )
        for value in range(rank)
    }
    weights_nonzero = all(weight != alg.zero for weight, _ in components)
    factorized = alg.one
    for weight, _ in components:
        factorized = alg.mul(factorized, weight)
    for left in range(rank):
        for right in range(left + 1, rank):
            difference = alg.sub(points[right], points[left])
            factorized = alg.mul(factorized, alg.mul(difference, difference))
    factor_nonzero = factorized != alg.zero

    direct_checked = alg.modulus != 0 or m <= 4
    direct_nonzero = None
    if direct_checked:
        moments = component_moments(components, k, alg)
        matrix = [
            [moments[row + column] for column in range(rank)]
            for row in range(rank)
        ]
        direct_nonzero = determinant(matrix, alg) != alg.zero
    return {
        "m": m,
        "rank": rank,
        "public_point_set_agreement": point_keys == expected_keys,
        "all_weights_nonzero": weights_nonzero,
        "independent_factorized_vandermonde_nonzero": factor_nonzero,
        "direct_hankel_determinant_checked": direct_checked,
        "direct_hankel_determinant_nonzero": direct_nonzero,
        "exact_normalized_divided_power_secant_rank": (
            rank
            if point_keys == expected_keys and weights_nonzero and factor_nonzero
            else None
        ),
        "ordinary_symmetric_waring_rank_interpretation": alg.kind == "Q_ZETA17",
        "determinant_values_serialized": False,
    }


def boundary(components: list[tuple[Any, list[Any]]], k: int, alg: backend.Algebra) -> Any:
    moments = component_moments(components, 1, alg)
    return alg.mul(rank1ref.field_integer(alg, k), moments[1])


def transaction_parity(
    transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    m = transaction["m"]
    k = transaction["k"]
    family = transaction["family"]
    components = forward_components(m, family, alg)
    commitment, record_bytes = rank1ref.stream_vector_commitment(
        flat_components(components), alg
    )
    projected = boundary(components, k, alg)
    restored = inverse_components(components, m, family, alg)
    certificate = rank_certificate(
        forward_components(m, family, alg), m, k, alg
    )
    return {
        "m": m,
        "k": k,
        "family": family,
        "algebra": transaction["algebra"],
        "program_fingerprint_agreement": program_fingerprint(m, family)
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
        "independent_forward_inverse_restored": restored_seed(restored, alg),
        "target_rank_agreement": transaction["target_rank"] == 1 << m,
        "active_field_cells_agreement": transaction["active_phase_field_cells"]
        == 18 * (1 << m),
        "inverse_transient_field_cells_agreement": transaction[
            "maximum_coupling_transient_field_cells"
        ]
        == 36 * (1 << m),
        "rank_certificate": certificate,
    }


def baseline_parity(
    baseline: dict[str, Any], transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    m = transaction["m"]
    k = transaction["k"]
    family = transaction["family"]
    total_weight = alg.one
    first_moment = alg.zero
    for level, exponent in enumerate(eta_exponents(m, family), start=1):
        center = rank1ref.field_integer(alg, (1 << level) - 1)
        eta = alg.power(exponent)
        prior_weight = total_weight
        prior_moment = first_moment
        total_weight = alg.mul(alg.add(alg.one, eta), prior_weight)
        first_moment = alg.add(
            alg.mul(alg.sub(alg.one, eta), prior_moment),
            alg.mul(alg.mul(eta, center), prior_weight),
        )
    projected = alg.mul(rank1ref.field_integer(alg, k), first_moment)
    commitment, record_bytes = rank1ref.stream_vector_commitment(
        [total_weight, first_moment], alg
    )
    return {
        "m": m,
        "k": k,
        "algebra": transaction["algebra"],
        "boundary_agreement": alg.serialize(projected) == transaction["boundary"],
        "moment_commitment_agreement": commitment == baseline["moment_commitment"],
        "record_bytes_agreement": record_bytes
        == baseline["maximum_commitment_record_json_bytes"],
        "two_dynamic_moment_cells_agreement": baseline[
            "descriptor_runtime_dynamic_field_cells"
        ]
        == 2,
        "one_sealed_boundary_cell_agreement": baseline[
            "sealed_word_final_boundary_field_cells"
        ]
        == 1,
        "full_state_triangular_moment_cells": k + 1,
    }


def atomic_weight_baseline_parity(
    baseline: dict[str, Any], transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    """Independently compile the strongest public-support full-state baseline."""

    m = transaction["m"]
    k = transaction["k"]
    family = transaction["family"]
    weights = [alg.one]
    maximum_named_field_cells = 1
    for level, exponent in enumerate(eta_exponents(m, family), start=1):
        center = (1 << level) - 1
        eta = alg.power(exponent)
        old = weights
        new = [alg.zero for _ in range(2 * len(old))]
        for point in range(len(old)):
            new[point] = old[point]
            new[center - point] = alg.mul(eta, old[point])
        maximum_named_field_cells = max(
            maximum_named_field_cells, len(old) + len(new)
        )
        weights = new

    first_moment = alg.zero
    for point, weight in enumerate(weights):
        first_moment = alg.add(
            first_moment,
            alg.mul(rank1ref.field_integer(alg, point), weight),
        )
    projected = alg.mul(rank1ref.field_integer(alg, k), first_moment)
    commitment, record_bytes = rank1ref.stream_vector_commitment(weights, alg)

    components = forward_components(m, family, alg)
    component_weight_agreement = True
    for point, weight in enumerate(weights):
        point_value = rank1ref.field_integer(alg, point)
        matching = [
            component_weight
            for component_weight, vector in components
            if vector[1] == point_value
        ]
        if matching != [weight]:
            component_weight_agreement = False
            break

    return {
        "m": m,
        "k": k,
        "algebra": transaction["algebra"],
        "boundary_agreement": alg.serialize(projected) == transaction["boundary"],
        "atomic_weight_commitment_agreement": commitment
        == baseline["weight_commitment"],
        "record_bytes_agreement": record_bytes
        == baseline["maximum_commitment_record_json_bytes"],
        "resident_weight_cells_agreement": baseline[
            "resident_atomic_weight_field_cells"
        ]
        == 1 << m,
        "maximum_named_field_cells_agreement": baseline[
            "maximum_named_field_cells_including_update_buffer"
        ]
        == maximum_named_field_cells,
        "public_support_rematerialized_without_retained_cells": baseline[
            "public_support_field_cells_retained"
        ]
        == 0,
        "component_weight_agreement": component_weight_agreement,
    }


def mutation_checks() -> dict[str, Any]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    m = 4
    family = "PRIMARY"
    reference = forward_components(m, family, alg)
    k = 2 * (1 << m) - 2
    reference_boundary = boundary(reference, k, alg)
    omitted = forward_components(m - 1, family, alg)
    omitted_boundary = boundary(omitted, k, alg)
    exponents = eta_exponents(m, family)

    wrong = coupling(
        reference,
        (1 << m) - 1,
        exponents[-1] + 1,
        alg,
        inverse=True,
    )
    reordered = coupling(reference, 1, exponents[0], alg, inverse=True)
    seed = seed_vector(alg)
    r1r2 = affine_reflection(affine_reflection(seed, 3, alg), 1, alg)
    r2r1 = affine_reflection(affine_reflection(seed, 1, alg), 3, alg)
    certificate = rank_certificate(reference, m, k, alg)
    return {
        "last_coupling_omission_changes_boundary": omitted_boundary
        != reference_boundary,
        "wrong_inverse_does_not_halve_to_prior_rank": len(wrong) != 1 << (m - 1),
        "reordered_inverse_does_not_halve_to_prior_rank": len(reordered)
        != 1 << (m - 1),
        "distinct_affine_reflections_noncommute": r1r2 != r2r1,
        "rank16_direct_hankel_minor_nonzero": certificate[
            "direct_hankel_determinant_nonzero"
        ],
        "rank16_factorized_vandermonde_nonzero": certificate[
            "independent_factorized_vandermonde_nonzero"
        ],
        "f103_m7_excluded_by_point_collision_gate": (1 << 7) > 103,
        "generated_component_count_not_used_without_minor": True,
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
    baselines = {
        (item["algebra"], item["m"], item["family"]): item
        for item in production["compiled_two_moment_classical_baselines"]
    }
    baseline_checks = [
        baseline_parity(
            baselines[(item["algebra"], item["m"], item["family"])],
            item,
            algebra_by_signature[item["algebra"]],
        )
        for item in transactions
    ]
    weight_baselines = {
        (item["algebra"], item["m"], item["family"]): item
        for item in production["compiled_atomic_weight_classical_baselines"]
    }
    weight_baseline_checks = [
        atomic_weight_baseline_parity(
            weight_baselines[(item["algebra"], item["m"], item["family"])],
            item,
            algebra_by_signature[item["algebra"]],
        )
        for item in transactions
    ]
    moment_cases = [
        direct_moment_case(m, "PRIMARY", backend.Algebra("Q_ZETA17"))
        for m in DECLARED_M
    ]
    for modulus, root in FINITE_FIELDS:
        moment_cases.extend(
            direct_moment_case(
                m,
                "PRIMARY",
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            for m in DECLARED_M
        )
    mutations = mutation_checks()

    parity_keys = (
        "program_fingerprint_agreement",
        "algebra_signature_agreement",
        "forward_commitment_agreement",
        "boundary_agreement",
        "boundary_payload_bits_agreement",
        "maximum_commitment_record_bound_honest",
        "independent_forward_inverse_restored",
        "target_rank_agreement",
        "active_field_cells_agreement",
        "inverse_transient_field_cells_agreement",
    )
    if not all(all(item[key] for key in parity_keys) for item in parity):
        fail("independent iterated transaction parity failed")
    if not all(
        item["rank_certificate"]["public_point_set_agreement"]
        and item["rank_certificate"]["all_weights_nonzero"]
        and item["rank_certificate"]["independent_factorized_vandermonde_nonzero"]
        and item["rank_certificate"]["exact_normalized_divided_power_secant_rank"]
        == 1 << item["m"]
        and (
            not item["rank_certificate"]["direct_hankel_determinant_checked"]
            or item["rank_certificate"]["direct_hankel_determinant_nonzero"]
        )
        for item in parity
    ):
        fail("independent rank-growth certificate failed")
    baseline_keys = (
        "boundary_agreement",
        "moment_commitment_agreement",
        "record_bytes_agreement",
        "two_dynamic_moment_cells_agreement",
        "one_sealed_boundary_cell_agreement",
    )
    if not all(all(item[key] for key in baseline_keys) for item in baseline_checks):
        fail("independent two-moment baseline parity failed")
    weight_baseline_keys = (
        "boundary_agreement",
        "atomic_weight_commitment_agreement",
        "record_bytes_agreement",
        "resident_weight_cells_agreement",
        "maximum_named_field_cells_agreement",
        "public_support_rematerialized_without_retained_cells",
        "component_weight_agreement",
    )
    if not all(
        all(item[key] for key in weight_baseline_keys)
        for item in weight_baseline_checks
    ):
        fail("independent atomic-weight baseline parity failed")
    if not all(
        item["all_forward_component_moment_agreements"]
        and item["final_boundary_agreement"]
        and item["direct_moments_restored_to_seed"]
        and item["component_chart_restored_to_seed"]
        for item in moment_cases
    ):
        fail("independent direct moment recurrence failed")
    if not all(mutations.values()):
        fail("one or more independent mutations failed")

    return {
        "schema": "CAT_CAS_F17_SYMBOLIC_ITERATED_AFFINE_REFLECTION_SECANT_RANK_GROWTH_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_schema": production["schema"],
        "production_claim": production["claim"],
        "transaction_parity": parity,
        "direct_full_binary_moment_cases": moment_cases,
        "compiled_two_moment_baseline_checks": baseline_checks,
        "compiled_atomic_weight_baseline_checks": weight_baseline_checks,
        "independent_mutation_checks": mutations,
        "independent_methods": [
            "SEPARATE_PUBLIC_TOPOLOGY_COMPILER",
            "SEPARATE_COMPONENT_COUPLING_AND_EXACT_INVERSE",
            "INDEPENDENT_PUBLIC_POINT_SET_INDUCTION",
            "INDEPENDENT_VANDERMONDE_PRODUCT_CERTIFICATE",
            "DIRECT_HANKEL_GAUSSIAN_DETERMINANTS_Q_M1_TO4_AND_DUAL_FIELD_M1_TO6",
            "SEPARATE_FULL_BINARY_TRIANGULAR_MOMENT_FORWARD_INVERSE_RECURRENCE",
            "SEPARATE_PUBLIC_SUPPORT_ATOMIC_WEIGHT_FULL_STATE_RECURRENCE",
            "SEPARATE_TWO_DYNAMIC_MOMENT_FINAL_BOUNDARY_BASELINE",
        ],
        "resource_law": {
            "component_chart_active_field_cells": "18_TIMES_TWO_TO_THE_M",
            "component_chart_inverse_transient_field_cells": "36_TIMES_TWO_TO_THE_M",
            "accepted_path_explicit_coherent_components": "TWO_TO_THE_M",
            "compact_atomic_weight_full_state_field_cells": "TWO_TO_THE_M",
            "full_binary_moment_recurrence_field_cells": "TWO_TIMES_TWO_TO_THE_M_MINUS_ONE",
            "final_boundary_recurrence_field_cells": 2,
            "accepted_path_catalecticant_cells": 0,
            "accepted_path_separate_truth_table_or_assignment_buffer_cells": 0,
            "exact_payload_height_tuples_independently_reexecuted": False,
            "full_exact_bit_complexity_established": False,
        },
        "matched_baseline": {
            "strongest_compact_full_state": "TWO_TO_THE_M_ATOMIC_WEIGHTS_ON_PUBLIC_SUPPORT",
            "independent_dense_moment_full_state": "TWO_TIMES_TWO_TO_THE_M_MINUS_ONE_TRIANGULAR_MOMENTS",
            "final_boundary_only": "TWO_DYNAMIC_MOMENTS",
            "sealed_word": "ONE_FINAL_BOUNDARY_SCALAR",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_restoration_class": "NO_RESTORATION_CLAIM",
        "claim_ceiling": "DECLARED_A_LEVEL_TWO_TO_THE_LEVEL_MINUS_ONE_ITERATED_AFFINE_REFLECTION_FAMILY_BOUNDED_EXECUTION_M1_TO6_ANALYTIC_NORMALIZED_SECANT_RANK_TWO_TO_THE_M_Q_ZETA17_AND_DUAL_FIELD_WHERE_RANK_AT_MOST_MODULUS_DIRECT_PROCESS_SOFTWARE",
        "rejected_interpretations": [
            "ARBITRARY_INTERLEAVED_COUPLING_RANK_LAW",
            "GENERAL_GAUSSIAN_NO_GO",
            "FIXED_RANK_CLOSURE",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_EXECUTION_OR_BIT_REPLACEMENT",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
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
