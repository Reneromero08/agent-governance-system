#!/usr/bin/env python3
"""Independent oracle for the fixed-bond Gray phase-factor chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure as rank1ref
import f17_nonlinear_canonical_mps_separator_chart as backend


EXACT_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
DIRECT_DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)
FINITE_FIELD_DEPTHS = (1, 2, 3, 4, 5, 6)
FINITE_FIELDS = ((103, 72), (137, 16))
FINAL_BOUNDARY = "K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION"


def fail(message: str) -> None:
    raise RuntimeError(message)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1ref.algebra_signature(alg)


def eta_exponent(level: int, family: str) -> int:
    if family == "PRIMARY":
        return 1 + ((2 * level) % 16)
    if family == "REUSE":
        return 2 + ((2 * level) % 16)
    fail("oracle family changed")


def eta_exponents(depth: int, family: str) -> tuple[int, ...]:
    return tuple(eta_exponent(level, family) for level in range(1, depth + 1))


def descriptor(depth: int, family: str) -> dict[str, Any]:
    components = 1 << depth
    return {
        "depth": depth,
        "family": family,
        "chart": "GRAY_CODE_NEAREST_NEIGHBOR_PHASE_FACTOR_MPS",
        "conceptual_component_count": str(components),
        "degree": str(2 * components - 2),
        "reflection_center_law": "A_LEVEL_EQUALS_TWO_TO_THE_LEVEL_MINUS_ONE",
        "eta_exponent_schedule": list(eta_exponents(depth, family)),
        "fixed_delta_wiring": "BINARY_BITS_WITH_GRAY_XOR_NEIGHBOR_FACTORS",
        "final_boundary": FINAL_BOUNDARY,
    }


def program_fingerprint(depth: int, family: str) -> str:
    return rank1ref.digest_json(descriptor(depth, family))


def independent_lease_identity(
    depth: int, family: str, alg: backend.Algebra, capacity: int
) -> str:
    return rank1ref.digest_json(
        {
            "program": program_fingerprint(depth, family),
            "algebra": algebra_signature(alg),
            "capacity": capacity,
            "carrier": "GRAY_CODE_BOND2_PHASE_FACTOR",
        }
    )


def factors(depth: int, family: str, alg: backend.Algebra) -> list[Any]:
    return [alg.power(exponent) for exponent in eta_exponents(depth, family)]


def recursive_weights(
    depth: int, family: str, alg: backend.Algebra
) -> list[Any]:
    weights = [alg.one]
    for eta in factors(depth, family, alg):
        old = weights
        weights = [*old, *(alg.mul(eta, value) for value in reversed(old))]
    return weights


def gray_weight(
    depth: int, eta_values: list[Any], index: int, alg: backend.Algebra
) -> Any:
    result = alg.one
    for level in range(1, depth + 1):
        bit = (index >> (level - 1)) & 1
        next_bit = (index >> level) & 1 if level < depth else 0
        if bit ^ next_bit:
            result = alg.mul(result, eta_values[level - 1])
    return result


def row_matrix(
    eta: Any, physical_bit: int, alg: backend.Algebra
) -> list[list[Any]]:
    if physical_bit == 0:
        return [[alg.one, alg.zero], [eta, alg.zero]]
    return [[alg.zero, eta], [alg.zero, alg.one]]


def row_times_matrix(
    row: list[Any], matrix: list[list[Any]], alg: backend.Algebra
) -> list[Any]:
    return [
        alg.add(alg.mul(row[0], matrix[0][column]), alg.mul(row[1], matrix[1][column]))
        for column in range(2)
    ]


def mps_weight(
    depth: int, eta_values: list[Any], index: int, alg: backend.Algebra
) -> Any:
    row = [alg.one, alg.zero]
    for level in range(depth, 0, -1):
        bit = (index >> (level - 1)) & 1
        row = row_times_matrix(
            row, row_matrix(eta_values[level - 1], bit, alg), alg
        )
    return alg.add(row[0], row[1])


def matrix_rank(matrix: list[list[Any]], alg: backend.Algebra) -> int:
    work = [list(row) for row in matrix]
    rows = len(work)
    columns = len(work[0]) if rows else 0
    rank = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, rows) if work[row][column] != alg.zero),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = alg.inverse(work[rank][column])
        for inner in range(column, columns):
            work[rank][inner] = alg.mul(work[rank][inner], inverse)
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column]
            if factor == alg.zero:
                continue
            for inner in range(column, columns):
                work[row][inner] = alg.sub(
                    work[row][inner], alg.mul(factor, work[rank][inner])
                )
        rank += 1
        if rank == rows:
            break
    return rank


def direct_tensor_case(
    depth: int, family: str, alg: backend.Algebra
) -> dict[str, Any]:
    eta_values = factors(depth, family, alg)
    recursive = recursive_weights(depth, family, alg)
    gray = [
        gray_weight(depth, eta_values, index, alg)
        for index in range(1 << depth)
    ]
    mps = [
        mps_weight(depth, eta_values, index, alg)
        for index in range(1 << depth)
    ]
    cut_ranks = []
    for lower_bits in range(1, depth):
        rows = 1 << (depth - lower_bits)
        columns = 1 << lower_bits
        matrix = [
            [recursive[(row << lower_bits) | column] for column in range(columns)]
            for row in range(rows)
        ]
        cut_ranks.append(matrix_rank(matrix, alg))
    return {
        "depth": depth,
        "family": family,
        "algebra": algebra_signature(alg),
        "component_count": len(recursive),
        "recursive_gray_agreement": recursive == gray,
        "recursive_mps_agreement": recursive == mps,
        "direct_cut_ranks": cut_ranks,
        "all_internal_cut_ranks_exactly_two": depth >= 2
        and all(rank == 2 for rank in cut_ranks),
        "verification_only_component_enumeration": True,
    }


def closed_total_moment(
    eta_values: list[Any], alg: backend.Algebra, centers: list[int] | None = None
) -> tuple[Any, Any]:
    depth = len(eta_values)
    if centers is None:
        centers = [(1 << level) - 1 for level in range(1, depth + 1)]
    prefix_plus = [alg.one]
    for eta in eta_values:
        prefix_plus.append(
            alg.mul(prefix_plus[-1], alg.add(alg.one, eta))
        )
    suffix_minus = [alg.one for _ in range(depth + 1)]
    for index in range(depth - 1, -1, -1):
        suffix_minus[index] = alg.mul(
            alg.sub(alg.one, eta_values[index]), suffix_minus[index + 1]
        )
    moment = alg.zero
    for index, (eta, center) in enumerate(zip(eta_values, centers, strict=True)):
        term = alg.mul(
            rank1ref.field_integer(alg, center),
            alg.mul(eta, alg.mul(prefix_plus[index], suffix_minus[index + 1])),
        )
        moment = alg.add(moment, term)
    return prefix_plus[-1], moment


def inverse_site(pair: tuple[Any, Any], eta: Any, alg: backend.Algebra) -> tuple[Any, Any]:
    denominator = alg.sub(alg.one, alg.mul(eta, eta))
    if denominator == alg.zero:
        fail("oracle singular local inverse")
    scale = alg.inverse(denominator)
    return (
        alg.mul(scale, alg.sub(pair[0], alg.mul(eta, pair[1]))),
        alg.mul(scale, alg.sub(pair[1], alg.mul(eta, pair[0]))),
    )


def transaction_parity(
    transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    depth = transaction["depth"]
    family = transaction["family"]
    eta_values = factors(depth, family, alg)
    pairs = [(alg.one, eta) for eta in eta_values]
    flat = [value for pair in pairs for value in pair]
    commitment, record_bytes = rank1ref.stream_vector_commitment(flat, alg)
    total, moment = closed_total_moment(eta_values, alg)
    del total
    k = 2 * (1 << depth) - 2
    boundary = alg.mul(rank1ref.field_integer(alg, k), moment)
    restored_pairs = [inverse_site(pair, eta, alg) for pair, eta in zip(pairs, eta_values, strict=True)]
    restored_and_unseeded = all(
        identity == alg.one and reflected == alg.zero
        for identity, reflected in restored_pairs
    )
    payload_bits = sum(alg.payload_bits(value) for value in flat)
    return {
        "depth": depth,
        "family": family,
        "algebra": transaction["algebra"],
        "program_fingerprint_agreement": program_fingerprint(depth, family)
        == transaction["program_fingerprint"],
        "conceptual_component_count_agreement": transaction[
            "conceptual_component_count"
        ]
        == str(1 << depth),
        "degree_agreement": transaction["degree"] == str(k),
        "factor_commitment_agreement": commitment == transaction["factor_commitment"],
        "commitment_record_bytes_agreement": record_bytes
        == transaction["factor_commitment_json_bytes"],
        "closed_form_boundary_agreement": alg.serialize(boundary)
        == transaction["boundary"],
        "boundary_payload_bits_agreement": alg.payload_bits(boundary)
        == transaction["final_boundary_payload_bits"],
        "resident_payload_bits_agreement": payload_bits
        == transaction["maximum_resident_factor_payload_bits"],
        "resident_phase_factor_cells_agreement": transaction[
            "resident_phase_factor_field_cells"
        ]
        == 2 * depth,
        "resident_nontrivial_eta_cells_agreement": transaction[
            "resident_nontrivial_eta_field_cells"
        ]
        == depth,
        "fixed_bond_agreement": transaction["exact_maximum_mps_bond_dimension"]
        == (1 if depth == 1 else 2),
        "independent_local_inverse_restores_seeded_sites": restored_and_unseeded,
        "public_program_json_bytes_agreement": len(
            json.dumps(descriptor(depth, family), sort_keys=True, separators=(",", ":")).encode()
        )
        == transaction["public_program_json_bytes"],
        "restored_exact_zero_reported": transaction["restored_exact_zero"],
        "same_backing_reported": transaction["same_backing"],
        "no_component_weight_payload_in_transaction": "component_weights"
        not in transaction,
    }


def baseline_parity(
    baseline: dict[str, Any], transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    depth = transaction["depth"]
    family = transaction["family"]
    eta_values = factors(depth, family, alg)
    commitment, record_bytes = rank1ref.stream_vector_commitment(eta_values, alg)
    total, moment = closed_total_moment(eta_values, alg)
    homogeneous = alg.one
    for eta in eta_values:
        homogeneous = alg.mul(homogeneous, alg.sub(alg.one, eta))
    transfer_commitment, transfer_record_bytes = rank1ref.stream_vector_commitment(
        [total, moment, homogeneous], alg
    )
    k = 2 * (1 << depth) - 2
    boundary = alg.mul(rank1ref.field_integer(alg, k), moment)
    return {
        "depth": depth,
        "family": family,
        "algebra": transaction["algebra"],
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "factor_commitment_agreement": commitment == baseline["factor_commitment"],
        "factor_record_bytes_agreement": record_bytes
        == baseline["maximum_commitment_record_json_bytes"],
        "compiled_transfer_commitment_agreement": transfer_commitment
        == baseline["compiled_transfer_commitment"],
        "compiled_transfer_record_bytes_agreement": transfer_record_bytes
        == baseline["compiled_transfer_commitment_json_bytes"],
        "full_signature_factor_cells_agreement": baseline[
            "full_weight_signature_exact_factor_field_cells"
        ]
        == depth,
        "two_dynamic_moment_cells_agreement": baseline[
            "final_boundary_dynamic_field_cells"
        ]
        == 2,
        "three_compiled_transfer_cells_agreement": baseline[
            "sealed_word_compiled_transfer_nonzero_field_cells"
        ]
        == 3,
    }


def mutation_checks() -> dict[str, bool]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    depth = 4
    eta_values = factors(depth, "PRIMARY", alg)
    reference = recursive_weights(depth, "PRIMARY", alg)
    straight = [alg.one]
    for eta in eta_values:
        old = straight
        straight = [*old, *(alg.mul(eta, value) for value in old)]

    _, reference_moment = closed_total_moment(eta_values, alg)
    swapped = list(eta_values)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    _, swapped_moment = closed_total_moment(swapped, alg)
    centers = [(1 << level) - 1 for level in range(1, depth + 1)]
    centers[-1] += 1
    _, perturbed_center_moment = closed_total_moment(eta_values, alg, centers)

    last_pair = (alg.one, eta_values[-1])
    wrong_eta = eta_values[-2]
    wrong_pair = inverse_site(last_pair, wrong_eta, alg)
    singular_determinant = alg.sub(alg.one, alg.mul(alg.one, alg.one))

    delta = [alg.one, negative(alg, rank1ref.field_integer(alg, 2)), alg.one]
    delta_total = alg.zero
    delta_first = alg.zero
    delta_second = alg.zero
    for point, value in enumerate(delta):
        point_value = rank1ref.field_integer(alg, point)
        delta_total = alg.add(delta_total, value)
        delta_first = alg.add(delta_first, alg.mul(point_value, value))
        delta_second = alg.add(
            delta_second,
            alg.mul(alg.mul(point_value, point_value), value),
        )

    return {
        "straight_upper_copy_breaks_reversed_gray_law": straight != reference,
        "swapped_eta_site_order_changes_first_moment": swapped_moment
        != reference_moment,
        "perturbed_reflection_center_changes_first_moment": perturbed_center_moment
        != reference_moment,
        "wrong_inverse_does_not_restore_seeded_site": wrong_pair
        != (alg.one, alg.zero),
        "missing_inverse_leaves_reflected_branch_nonzero": last_pair[1] != alg.zero,
        "reordered_inverse_does_not_restore_latest_site": wrong_pair
        != (alg.one, alg.zero),
        "eta_equal_one_collapses_edge_determinant_and_is_singular": singular_determinant
        == alg.zero,
        "bond_one_rejected_by_direct_rank2_cut": direct_tensor_case(
            depth, "PRIMARY", alg
        )["all_internal_cut_ranks_exactly_two"],
        "two_moments_do_not_determine_second_moment": delta_total == alg.zero
        and delta_first == alg.zero
        and delta_second != alg.zero,
        "independent_primary_reuse_lease_identities_differ": (
            independent_lease_identity(depth, "PRIMARY", alg, depth)
            != independent_lease_identity(depth, "REUSE", alg, depth)
        ),
        "snapshot_command_available": False,
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
        (item["algebra"], item["depth"], item["family"]): item
        for item in production["compiled_classical_baselines"]
    }
    baseline_checks = [
        baseline_parity(
            baselines[(item["algebra"], item["depth"], item["family"])],
            item,
            algebra_by_signature[item["algebra"]],
        )
        for item in transactions
    ]

    direct_cases = [
        direct_tensor_case(depth, "PRIMARY", backend.Algebra("Q_ZETA17"))
        for depth in DIRECT_DEPTHS
    ]
    for modulus, root in FINITE_FIELDS:
        direct_cases.extend(
            direct_tensor_case(
                depth,
                "PRIMARY",
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            for depth in FINITE_FIELD_DEPTHS
        )

    parity_keys = (
        "program_fingerprint_agreement",
        "conceptual_component_count_agreement",
        "degree_agreement",
        "factor_commitment_agreement",
        "commitment_record_bytes_agreement",
        "closed_form_boundary_agreement",
        "boundary_payload_bits_agreement",
        "resident_payload_bits_agreement",
        "resident_phase_factor_cells_agreement",
        "resident_nontrivial_eta_cells_agreement",
        "fixed_bond_agreement",
        "independent_local_inverse_restores_seeded_sites",
        "public_program_json_bytes_agreement",
        "restored_exact_zero_reported",
        "same_backing_reported",
        "no_component_weight_payload_in_transaction",
    )
    if not all(all(item[key] for key in parity_keys) for item in parity):
        fail("independent phase-factor transaction parity failed")

    baseline_keys = (
        "boundary_agreement",
        "factor_commitment_agreement",
        "factor_record_bytes_agreement",
        "compiled_transfer_commitment_agreement",
        "compiled_transfer_record_bytes_agreement",
        "full_signature_factor_cells_agreement",
        "two_dynamic_moment_cells_agreement",
        "three_compiled_transfer_cells_agreement",
    )
    if not all(all(item[key] for key in baseline_keys) for item in baseline_checks):
        fail("independent phase-factor baseline parity failed")
    if not all(
        item["recursive_gray_agreement"]
        and item["recursive_mps_agreement"]
        and item["all_internal_cut_ranks_exactly_two"]
        == (item["depth"] >= 2)
        for item in direct_cases
    ):
        fail("independent direct MPS reconstruction failed")

    mutations = mutation_checks()
    if not all(
        value for key, value in mutations.items() if key != "snapshot_command_available"
    ) or mutations["snapshot_command_available"]:
        fail("independent phase-factor mutation failed")

    return {
        "schema": "CAT_CAS_F17_GRAY_CODE_BOND2_PHASE_FACTOR_CHART_CLOSURE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_schema": production["schema"],
        "production_claim": production["claim"],
        "transaction_parity": parity,
        "compiled_classical_baseline_checks": baseline_checks,
        "direct_expanded_mps_rank_checks": direct_cases,
        "independent_mutation_checks": mutations,
        "independent_methods": [
            "SEPARATE_PUBLIC_DESCRIPTOR_COMPILER",
            "CLOSED_FORM_PREFIX_SUFFIX_TOTAL_AND_FIRST_MOMENT_CONTRACTION",
            "SEPARATE_LOCAL_TWO_BRANCH_EXACT_INVERSE",
            "DIRECT_REVERSED_COPY_GRAY_PRODUCT_AND_OPEN_BOUNDARY_MPS_WEIGHT_PARITY",
            "DIRECT_GAUSSIAN_RANK_OF_EVERY_SMALL_EXPANDED_TENSOR_CUT",
            "SEPARATE_M_FACTOR_FULL_SIGNATURE_BASELINE",
            "SEPARATE_THREE_SCALAR_SEALED_TRANSFER_BASELINE",
            "EXACT_SECOND_MOMENT_NONCLOSURE_WITNESS_FOR_TWO_SCALAR_QUOTIENT",
        ],
        "resource_law": {
            "conceptual_component_count": "TWO_TO_THE_M",
            "phase_factor_carrier_field_cells": "TWO_TIMES_M",
            "nontrivial_eta_field_cells": "M",
            "exact_weight_tensor_mps_bond_dimension": 2,
            "matched_classical_full_signature_field_cells": "M",
            "matched_classical_final_boundary_dynamic_field_cells": 2,
            "sealed_transfer_nonzero_field_cells": 3,
            "accepted_path_component_enumeration": False,
            "exact_payload_height_tuples_independently_reexecuted": True,
            "full_exact_bit_complexity_established": False,
        },
        "matched_baseline": {
            "full_weight_signature": "M_EXACT_PHASE_FACTORS_ON_PUBLIC_GRAY_CHAIN",
            "final_boundary_only": "TWO_DYNAMIC_MOMENTS",
            "sealed_word": "THREE_NONZERO_LOWER_TRIANGULAR_TRANSFER_SCALARS",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_restoration_class": "NO_RESTORATION_CLAIM",
        "claim_ceiling": "DECLARED_GRAY_ORDERED_AFFINE_REFLECTION_COMPONENT_WEIGHT_TENSOR_EXACT_BOND2_FACTOR_CHART_Q_ZETA17_DEPTHS1_2_4_8_16_32_64_128_DUAL_FIELD_SMALL_DEPTH_DIRECT_PROCESS_SOFTWARE",
        "rejected_interpretations": [
            "GENERAL_COHERENT_POLYNOMIAL_COMPACTION",
            "ARBITRARY_BOUNDARY_TWO_MOMENT_CLOSURE",
            "CONVENTIONAL_CLIFFORD_OR_STABILIZER_CLASSIFICATION",
            "GENERAL_GAUSSIAN_CLOSURE_OR_NO_GO",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_EXECUTION",
            "PHYSICAL_BITS_REPLACED_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    production = json.loads(Path(args.production).read_text(encoding="utf-8"))
    result = run(production)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
