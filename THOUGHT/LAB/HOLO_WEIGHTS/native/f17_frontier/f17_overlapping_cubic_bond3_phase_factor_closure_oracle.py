#!/usr/bin/env python3
"""Independent oracle for the overlapping cubic bond-3 factor chart."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure as rank1ref
import f17_nonlinear_canonical_mps_separator_chart as backend


EXACT_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
DIRECT_DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)
STRUCTURAL_DEPTHS = DIRECT_DEPTHS
FINITE_FIELDS = ((103, 72), (137, 16))
FINAL_BOUNDARY = "ALL_BITS_SUMMED_CUBIC_PHASE_PARTITION_SCALAR"


def fail(message: str) -> None:
    raise RuntimeError(message)


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1ref.algebra_signature(alg)


def theta_exponent(level: int, family: str) -> int:
    if family == "PRIMARY":
        return 1 + ((2 * level) % 16)
    if family == "REUSE":
        return 1 + ((2 * level + 1) % 16)
    fail("oracle cubic family changed")


def theta_exponents(depth: int, family: str) -> tuple[int, ...]:
    return tuple(theta_exponent(level, family) for level in range(1, depth + 1))


def descriptor(depth: int, family: str) -> dict[str, Any]:
    return {
        "depth": depth,
        "physical_bits": depth + 2,
        "family": family,
        "factor": "THETA_LEVEL_TO_X_LEVEL_X_LEVEL_PLUS1_X_LEVEL_PLUS2",
        "theta_exponents": list(theta_exponents(depth, family)),
        "chart": "OVERLAPPING_THREE_SITE_BOOLEAN_PHASE_FACTOR_MPS",
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
            "carrier": "OVERLAPPING_CUBIC_BOND3_PHASE_FACTOR",
        }
    )


def factors(depth: int, family: str, alg: backend.Algebra) -> list[Any]:
    return [alg.power(value) for value in theta_exponents(depth, family)]


def field_sum(values: Any, alg: backend.Algebra) -> Any:
    result = alg.zero
    for value in values:
        result = alg.add(result, value)
    return result


def direct_weight(bits: tuple[int, ...], theta_values: list[Any], alg: backend.Algebra) -> Any:
    result = alg.one
    for level, theta in enumerate(theta_values):
        if bits[level] and bits[level + 1] and bits[level + 2]:
            result = alg.mul(result, theta)
    return result


def direct_partition(depth: int, family: str, alg: backend.Algebra) -> Any:
    theta_values = factors(depth, family, alg)
    return field_sum(
        (
            direct_weight(bits, theta_values, alg)
            for bits in itertools.product((0, 1), repeat=depth + 2)
        ),
        alg,
    )


def four_state_partition(theta_values: list[Any], alg: backend.Algebra) -> Any:
    state = [alg.one, alg.one, alg.one, alg.one]
    for theta in theta_values:
        m00, m01, m10, m11 = state
        common = alg.add(m00, m10)
        state = [
            common,
            common,
            alg.add(m01, m11),
            alg.add(m01, alg.mul(theta, m11)),
        ]
    return field_sum(state, alg)


def three_state_boundary(
    theta_values: list[Any], initial: list[Any], alg: backend.Algebra
) -> Any:
    a, b, c = initial
    for theta in theta_values:
        a, b, c = (
            alg.add(a, b),
            alg.add(a, c),
            alg.add(a, alg.mul(theta, c)),
        )
    return field_sum(
        (alg.mul(rank1ref.field_integer(alg, 2), a), b, c), alg
    )


def compiled_final_row(theta_values: list[Any], alg: backend.Algebra) -> list[Any]:
    basis = [
        [alg.one, alg.zero, alg.zero],
        [alg.zero, alg.one, alg.zero],
        [alg.zero, alg.zero, alg.one],
    ]
    return [three_state_boundary(theta_values, vector, alg) for vector in basis]


def inverse_pair(pair: tuple[Any, Any], theta: Any, alg: backend.Algebra) -> tuple[Any, Any]:
    denominator = alg.sub(alg.one, alg.mul(theta, theta))
    if denominator == alg.zero:
        fail("oracle singular cubic factor inverse")
    scale = alg.inverse(denominator)
    return (
        alg.mul(scale, alg.sub(pair[0], alg.mul(theta, pair[1]))),
        alg.mul(scale, alg.sub(pair[1], alg.mul(theta, pair[0]))),
    )


def matrix_rank(matrix: list[list[Any]], alg: backend.Algebra) -> int:
    work = [row[:] for row in matrix]
    rows = len(work)
    cols = len(work[0]) if work else 0
    rank = 0
    for column in range(cols):
        pivot = next(
            (row for row in range(rank, rows) if work[row][column] != alg.zero),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = alg.inverse(work[rank][column])
        work[rank] = [alg.mul(scale, value) for value in work[rank]]
        for row in range(rows):
            if row == rank or work[row][column] == alg.zero:
                continue
            coefficient = work[row][column]
            work[row] = [
                alg.sub(value, alg.mul(coefficient, pivot_value))
                for value, pivot_value in zip(work[row], work[rank], strict=True)
            ]
        rank += 1
        if rank == rows:
            break
    return rank


def local_cross_matrix(left: Any, right: Any, alg: backend.Algebra) -> list[list[Any]]:
    matrix = []
    for a, b in itertools.product((0, 1), repeat=2):
        row = []
        for c, d in itertools.product((0, 1), repeat=2):
            value = alg.one
            if a and b and c:
                value = alg.mul(value, left)
            if b and c and d:
                value = alg.mul(value, right)
            row.append(value)
        matrix.append(row)
    return matrix


def depth_one_flatten_ranks(theta: Any, alg: backend.Algebra) -> tuple[int, int]:
    left = [
        [
            theta if a and b and c else alg.one
            for b, c in itertools.product((0, 1), repeat=2)
        ]
        for a in (0, 1)
    ]
    right = [
        [
            theta if a and b and c else alg.one
            for c in (0, 1)
        ]
        for a, b in itertools.product((0, 1), repeat=2)
    ]
    return matrix_rank(left, alg), matrix_rank(right, alg)


def local_rank_case(depth: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    theta_values = factors(depth, family, alg)
    if depth == 1:
        ranks = depth_one_flatten_ranks(theta_values[0], alg)
        return {
            "depth": depth,
            "family": family,
            "algebra": algebra_signature(alg),
            "direct_cut_ranks": list(ranks),
            "exact_maximum_bond_dimension": max(ranks),
            "all_interior_local_cross_ranks_three": True,
            "verification_only": True,
        }
    ranks = [
        matrix_rank(local_cross_matrix(left, right, alg), alg)
        for left, right in zip(theta_values, theta_values[1:])
    ]
    return {
        "depth": depth,
        "family": family,
        "algebra": algebra_signature(alg),
        "direct_cut_ranks": ranks,
        "exact_maximum_bond_dimension": max(ranks),
        "all_interior_local_cross_ranks_three": all(value == 3 for value in ranks),
        "verification_only": True,
    }


def full_tensor_rank_case(
    depth: int, family: str, alg: backend.Algebra
) -> dict[str, Any]:
    theta_values = factors(depth, family, alg)
    physical_bits = depth + 2
    cut_ranks = []
    for cut in range(1, physical_bits):
        left_words = list(itertools.product((0, 1), repeat=cut))
        right_words = list(
            itertools.product((0, 1), repeat=physical_bits - cut)
        )
        matrix = [
            [direct_weight(left + right, theta_values, alg) for right in right_words]
            for left in left_words
        ]
        cut_ranks.append(matrix_rank(matrix, alg))
    return {
        "depth": depth,
        "family": family,
        "algebra": algebra_signature(alg),
        "direct_full_tensor_cut_ranks": cut_ranks,
        "exact_maximum_bond_dimension": max(cut_ranks),
        "verification_only_assignment_tensor_materialized": True,
        "accepted_path": False,
    }


def transaction_parity(transaction: dict[str, Any], alg: backend.Algebra) -> dict[str, Any]:
    depth = transaction["depth"]
    family = transaction["family"]
    theta_values = factors(depth, family, alg)
    flat = [value for theta in theta_values for value in (alg.one, theta)]
    commitment, record_bytes = rank1ref.stream_vector_commitment(flat, alg)
    boundary = four_state_partition(theta_values, alg)
    restored = [inverse_pair((alg.one, theta), theta, alg) for theta in theta_values]
    return {
        "depth": depth,
        "family": family,
        "algebra": transaction["algebra"],
        "program_fingerprint_agreement": program_fingerprint(depth, family)
        == transaction["program_fingerprint"],
        "physical_bits_agreement": transaction["physical_bits"] == depth + 2,
        "factor_commitment_agreement": commitment == transaction["factor_commitment"],
        "commitment_record_bytes_agreement": record_bytes
        == transaction["factor_commitment_json_bytes"],
        "independent_four_state_boundary_agreement": alg.serialize(boundary)
        == transaction["boundary"],
        "boundary_payload_bits_agreement": alg.payload_bits(boundary)
        == transaction["final_boundary_payload_bits"],
        "resident_payload_bits_agreement": sum(
            alg.payload_bits(value) for value in flat
        )
        == transaction["maximum_resident_factor_payload_bits"],
        "resident_factor_cells_agreement": transaction[
            "resident_phase_factor_field_cells"
        ]
        == 2 * depth,
        "resident_nontrivial_theta_cells_agreement": transaction[
            "resident_nontrivial_theta_field_cells"
        ]
        == depth,
        "bond_agreement": transaction["exact_maximum_mps_bond_dimension"]
        == (2 if depth == 1 else 3),
        "independent_local_inverse_restores_seeded_sites": all(
            identity == alg.one and cubic == alg.zero
            for identity, cubic in restored
        ),
        "public_program_json_bytes_agreement": len(
            json.dumps(descriptor(depth, family), sort_keys=True, separators=(",", ":")).encode()
        )
        == transaction["public_program_json_bytes"],
        "restored_exact_zero_reported": transaction["restored_exact_zero"],
        "same_backing_reported": transaction["same_backing"],
        "no_assignment_or_weight_tensor_payload_in_transaction": all(
            key not in transaction
            for key in ("assignments", "weight_tensor", "component_weights")
        ),
    }


def baseline_parity(
    baseline: dict[str, Any], transaction: dict[str, Any], alg: backend.Algebra
) -> dict[str, Any]:
    depth = transaction["depth"]
    family = transaction["family"]
    theta_values = factors(depth, family, alg)
    factor_commitment, factor_bytes = rank1ref.stream_vector_commitment(theta_values, alg)
    final_row = compiled_final_row(theta_values, alg)
    row_commitment, row_bytes = rank1ref.stream_vector_commitment(final_row, alg)
    boundary = three_state_boundary(theta_values, [alg.one, alg.one, alg.one], alg)
    return {
        "depth": depth,
        "family": family,
        "algebra": transaction["algebra"],
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "factor_commitment_agreement": factor_commitment
        == baseline["factor_commitment"],
        "factor_record_bytes_agreement": factor_bytes
        == baseline["factor_commitment_json_bytes"],
        "compiled_final_row_commitment_agreement": row_commitment
        == baseline["compiled_final_row_commitment"],
        "compiled_final_row_record_bytes_agreement": row_bytes
        == baseline["compiled_final_row_commitment_json_bytes"],
        "full_signature_factor_cells_agreement": baseline[
            "full_weight_signature_exact_factor_field_cells"
        ]
        == depth,
        "three_dynamic_cells_agreement": baseline[
            "final_boundary_dynamic_field_cells"
        ]
        == 3,
        "three_compiled_chart_row_cells_agreement": baseline[
            "sealed_word_three_state_chart_input_final_row_field_cells"
        ]
        == 3,
    }


def mutation_checks() -> dict[str, bool]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    depth = 4
    theta_values = factors(depth, "PRIMARY", alg)
    reference = direct_partition(depth, "PRIMARY", alg)
    omitted = four_state_partition([*theta_values[:-1], alg.one], alg)
    reversed_boundary = four_state_partition(list(reversed(theta_values)), alg)
    perturbed = theta_values[:]
    perturbed[0] = alg.power((theta_exponents(depth, "PRIMARY")[0] % 16) + 1)
    perturbed_boundary = four_state_partition(perturbed, alg)
    all_identity = four_state_partition([alg.one] * depth, alg)
    identity_expected = rank1ref.field_integer(alg, 1 << (depth + 2))
    wrong_pair = inverse_pair((alg.one, theta_values[-1]), theta_values[-2], alg)
    direct = local_rank_case(depth, "PRIMARY", alg)
    return {
        "last_cubic_factor_replaced_by_identity_changes_boundary": omitted
        != reference,
        "reversed_factor_order_preserves_chain_reflection_boundary": (
            reversed_boundary == reference
        ),
        "perturbed_theta_changes_boundary": perturbed_boundary != reference,
        "all_identity_cubic_factors_reduce_to_unweighted_assignment_count": all_identity
        == identity_expected,
        "identity_theta_collapses_rank3_minor": alg.sub(alg.one, alg.one)
        == alg.zero,
        "wrong_inverse_does_not_restore_seeded_site": wrong_pair
        != (alg.one, alg.zero),
        "missing_inverse_leaves_cubic_branch_nonzero": theta_values[-1]
        != alg.zero,
        "reordered_inverse_does_not_restore_latest_site": wrong_pair
        != (alg.one, alg.zero),
        "bond_two_rejected_by_direct_rank3_cross_matrix": direct[
            "all_interior_local_cross_ranks_three"
        ],
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
    algebras = [
        backend.Algebra("Q_ZETA17"),
        *(backend.Algebra(f"F{p}", modulus=p, root=r) for p, r in FINITE_FIELDS),
    ]
    algebra_by_signature = {algebra_signature(alg): alg for alg in algebras}
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

    direct_partitions = []
    for depth in DIRECT_DEPTHS:
        alg = backend.Algebra("Q_ZETA17")
        direct = direct_partition(depth, "PRIMARY", alg)
        four = four_state_partition(factors(depth, "PRIMARY", alg), alg)
        direct_partitions.append(
            {
                "depth": depth,
                "algebra": algebra_signature(alg),
                "direct_assignment_count": 1 << (depth + 2),
                "direct_partition_agrees_with_four_state_recurrence": direct == four,
                "accepted_path": False,
            }
        )

    local_ranks = [
        local_rank_case(depth, "PRIMARY", backend.Algebra("Q_ZETA17"))
        for depth in DIRECT_DEPTHS
    ]
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        local_ranks.extend(
            local_rank_case(depth, "PRIMARY", alg) for depth in DIRECT_DEPTHS
        )

    full_tensor_ranks = [
        full_tensor_rank_case(depth, "PRIMARY", backend.Algebra("Q_ZETA17"))
        for depth in (1, 2, 3, 4, 5)
    ]
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        full_tensor_ranks.extend(
            full_tensor_rank_case(depth, "PRIMARY", alg)
            for depth in (1, 2, 3, 4, 5, 6)
        )

    parity_keys = (
        "program_fingerprint_agreement",
        "physical_bits_agreement",
        "factor_commitment_agreement",
        "commitment_record_bytes_agreement",
        "independent_four_state_boundary_agreement",
        "boundary_payload_bits_agreement",
        "resident_payload_bits_agreement",
        "resident_factor_cells_agreement",
        "resident_nontrivial_theta_cells_agreement",
        "bond_agreement",
        "independent_local_inverse_restores_seeded_sites",
        "public_program_json_bytes_agreement",
        "restored_exact_zero_reported",
        "same_backing_reported",
        "no_assignment_or_weight_tensor_payload_in_transaction",
    )
    if not all(all(item[key] for key in parity_keys) for item in parity):
        fail("one or more cubic transaction parity checks failed")
    baseline_keys = (
        "boundary_agreement",
        "factor_commitment_agreement",
        "factor_record_bytes_agreement",
        "compiled_final_row_commitment_agreement",
        "compiled_final_row_record_bytes_agreement",
        "full_signature_factor_cells_agreement",
        "three_dynamic_cells_agreement",
        "three_compiled_chart_row_cells_agreement",
    )
    if not all(all(item[key] for key in baseline_keys) for item in baseline_checks):
        fail("one or more cubic baseline parity checks failed")
    if not all(
        item["direct_partition_agrees_with_four_state_recurrence"]
        for item in direct_partitions
    ):
        fail("direct cubic assignment oracle disagrees")
    if not all(
        item["exact_maximum_bond_dimension"] == (2 if item["depth"] == 1 else 3)
        for item in local_ranks
    ):
        fail("direct cubic local rank oracle disagrees")
    if not all(
        item["exact_maximum_bond_dimension"] == (2 if item["depth"] == 1 else 3)
        for item in full_tensor_ranks
    ):
        fail("direct cubic full tensor rank oracle disagrees")
    mutations = mutation_checks()
    if not all(
        value for key, value in mutations.items() if key != "snapshot_command_available"
    ) or mutations["snapshot_command_available"]:
        fail("one or more cubic mutation checks failed")

    return {
        "schema": "CAT_CAS_F17_OVERLAPPING_CUBIC_BOND3_PHASE_FACTOR_CLOSURE_ORACLE_V1",
        "production_claim": production["claim"],
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "oracle_independence": {
            "imports_production_module": False,
            "separate_public_descriptor_compiler": True,
            "separate_four_state_classical_recurrence": True,
            "direct_assignment_enumeration_verification_only": True,
            "separate_exact_local_cross_matrix_rank": True,
            "separate_seeded_pair_inverse": True,
            "shared_established_exact_arithmetic_backend_only": True,
        },
        "transaction_parity": parity,
        "compiled_classical_baseline_checks": baseline_checks,
        "direct_assignment_partition_checks": direct_partitions,
        "direct_local_rank_checks": local_ranks,
        "direct_full_tensor_rank_checks": full_tensor_ranks,
        "independent_mutation_checks": mutations,
        "observed_resource_law": {
            "accepted_path_factor_cells": "TWO_TIMES_M",
            "accepted_path_nontrivial_theta_cells": "M",
            "exact_maximum_mps_bond_dimension": 3,
            "matched_classical_full_signature_cells": "M",
            "matched_classical_boundary_dynamic_cells": 3,
            "matched_classical_compiled_final_row_cells": 3,
            "accepted_path_assignment_enumeration": False,
            "independent_direct_assignment_enumeration_maximum": 1024,
            "exact_boundary_payload_bits_reexecuted": True,
            "exact_resident_payload_bits_reexecuted": True,
            "full_exact_bit_complexity_established": False,
        },
        "restoration": {
            "resident_factor_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "projection_compiler_commitment_and_oracle_buffers": "NO_RESTORATION_CLAIM",
        },
        "claim_ceiling": {
            "declared_overlapping_cubic_chain_only": True,
            "arbitrary_cubic_hypergraph_closure": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    production = json.loads(Path(args.production).read_text(encoding="utf-8"))
    Path(args.output).write_text(
        json.dumps(run(production), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
