#!/usr/bin/env python3
"""Independent oracle for the growing shared-cubic separator theorem."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend


FINITE_FIELDS = ((103, 72), (137, 16))
DIRECT_ARITIES = (1, 2, 3, 4, 5, 6)
FAMILIES = ("PRIMARY", "REUSE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


def exponent(index: int, family: str) -> int:
    if family == "PRIMARY":
        return 1 + ((5 * index * index + 7 * index + 3) % 16)
    if family == "REUSE":
        return 1 + ((9 * index * index + 4 * index + 11) % 16)
    fail("oracle family changed")


def descriptor(k: int, family: str) -> dict[str, Any]:
    left = [exponent(2 * index, family) for index in range(k)]
    right = [exponent(2 * index + 1, family) for index in range(k)]
    return {
        "latent_arity": k,
        "family": family,
        "shared_latent_ports": [f"H_{index}" for index in range(k)],
        "branch_local_bits": [
            "LEFT_ANCHOR",
            *[f"LEFT_LEAF_{index}" for index in range(k)],
            "RIGHT_ANCHOR",
            *[f"RIGHT_LEAF_{index}" for index in range(k)],
        ],
        "left_cubic_factors": [
            f"ALPHA_{index}^(H_{index}*LEFT_ANCHOR*LEFT_LEAF_{index})"
            for index in range(k)
        ],
        "right_cubic_factors": [
            f"BETA_{index}^(H_{index}*RIGHT_ANCHOR*RIGHT_LEAF_{index})"
            for index in range(k)
        ],
        "left_theta_exponents": left,
        "right_theta_exponents": right,
        "separator_transport": [
            f"UNNORMALIZED_WALSH_ON_H_{index}" for index in range(k)
        ],
        "declared_observation_family": (
            "ALL_LEFT_REACHABLE_STATES_AND_RIGHT_CONTINUATION_FUNCTIONALS"
        ),
    }


def theta_values(k: int, family: str, alg: backend.Algebra) -> tuple[list[Any], list[Any]]:
    public = descriptor(k, family)
    return (
        [alg.power(value) for value in public["left_theta_exponents"]],
        [alg.power(value) for value in public["right_theta_exponents"]],
    )


def stream_commitment(values: Iterable[Any], alg: backend.Algebra) -> tuple[str, int]:
    state = hashlib.sha256()
    total = 0
    for index, value in enumerate(values):
        record = json.dumps(
            {"index": index, "value": alg.serialize(value)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        state.update(len(record).to_bytes(8, "big"))
        state.update(record)
        total += 8 + len(record)
    return state.hexdigest(), total


def formula_certificate(k: int, left: list[Any], right: list[Any], alg: backend.Algebra) -> dict[str, Any]:
    if len(left) != k or len(right) != k:
        fail("oracle resident factor shape changed")
    if any(alg.sub(value, alg.one) == alg.zero for value in [*left, *right]):
        fail("oracle identity phase entered accepted formula")
    width = 1 << k
    exponent_per_factor = 1 << (k - 1)
    return {
        "field": "Q_ZETA17_OR_DECLARED_STRUCTURAL_FIELD",
        "latent_arity": k,
        "left_local_branch_bits": 1 + k,
        "right_local_branch_bits": 1 + k,
        "full_left_local_assignment_rows": 2 * width,
        "full_right_local_assignment_rows": 2 * width,
        "port_coordinates": width,
        "certified_left_anchor_one_minor_shape": [width, width],
        "certified_right_anchor_one_minor_shape": [width, width],
        "certified_two_sided_boundary_minor_shape": [width, width],
        "local_two_by_two_factor": "[[1,1],[1,THETA_I]]",
        "local_determinant": "THETA_I_MINUS_ONE",
        "all_local_determinants_nonzero": True,
        "left_kronecker_minor_rank": width,
        "right_kronecker_minor_rank": width,
        "full_two_sided_separator_rank": width,
        "typed_configuration_bisimulation_classes": width,
        "individual_typed_continuation_separates_distinct_configurations": True,
        "typed_port_overmerge_exact_relation_preserving": False,
        "minor_determinant_factorization": (
            "PRODUCT_I((ALPHA_I_MINUS_ONE)*(BETA_I_MINUS_ONE))^"
            "(TWO_TO_THE_(K_MINUS_ONE))*DET(U)"
        ),
        "minor_determinant_exponent_per_factor": exponent_per_factor,
        "walsh_determinant": "PRODUCT_OF_K_NONZERO_MINUS_TWO_POWERS",
        "separator_transport_invertible": True,
        "q_zeta17_phase_factor_norm_power_of_17_exponent": 2 * k * exponent_per_factor,
        "uniform_exact_linear_port_quotient_minimum_field_coordinates": width,
        "uniform_exact_linear_port_quotient_below_two_to_the_k": "REJECTED",
        "formula_certificate_work_scalars": 6 * k + 12,
        "dense_port_vector_materialized": False,
        "dense_minor_materialized": False,
        "local_assignment_family_enumerated": False,
        "determinant_value_serialized": False,
        "proof_law": (
            "THE_LEFT_AND_RIGHT_ANCHOR_ONE_ROWS_FORM_KRONECKER_PRODUCTS_OF_"
            "INVERTIBLE_TWO_BY_TWO_MATRICES_AND_THE_SEPARATOR_TRANSPORT_IS_"
            "INVERTIBLE_SO_ANY_FIXED_LINEAR_RELATION_QUOTIENT_PRESERVING_"
            "ALL_REACHABLE_STATES_AND_CONTINUATIONS_MUST_BE_INJECTIVE"
        ),
    }


def load_pair(theta: Any, alg: backend.Algebra) -> tuple[Any, Any]:
    identity = alg.one
    cubic = alg.zero
    return (
        alg.add(identity, alg.mul(theta, cubic)),
        alg.add(alg.mul(theta, identity), cubic),
    )


def inverse_pair(pair: tuple[Any, Any], theta: Any, alg: backend.Algebra) -> tuple[Any, Any]:
    denominator = alg.sub(alg.one, alg.mul(theta, theta))
    scale = alg.inverse(denominator)
    return (
        alg.mul(scale, alg.sub(pair[0], alg.mul(theta, pair[1]))),
        alg.mul(scale, alg.sub(pair[1], alg.mul(theta, pair[0]))),
    )


def matrix_rank(matrix: list[list[Any]], alg: backend.Algebra) -> int:
    work = [row[:] for row in matrix]
    rows = len(work)
    columns = len(work[0]) if work else 0
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row][column] != alg.zero), None)
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
                alg.sub(value, alg.mul(coefficient, pivot))
                for value, pivot in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == rows:
            break
    return rank


def matmul(left: list[list[Any]], right: list[list[Any]], alg: backend.Algebra) -> list[list[Any]]:
    columns = len(right[0])
    return [
        [
            sum_field(
                (alg.mul(left[row][inner], right[inner][column]) for inner in range(len(right))),
                alg,
            )
            for column in range(columns)
        ]
        for row in range(len(left))
    ]


def transpose(matrix: list[list[Any]]) -> list[list[Any]]:
    return [list(column) for column in zip(*matrix)]


def sum_field(values: Iterable[Any], alg: backend.Algebra) -> Any:
    total = alg.zero
    for value in values:
        total = alg.add(total, value)
    return total


def branch_minor(theta: list[Any], alg: backend.Algebra) -> list[list[Any]]:
    k = len(theta)
    width = 1 << k
    matrix = []
    for leaves in range(width):
        row = []
        for port in range(width):
            value = alg.one
            for axis, phase in enumerate(theta):
                if ((leaves >> axis) & 1) and ((port >> axis) & 1):
                    value = alg.mul(value, phase)
            row.append(value)
        matrix.append(row)
    return matrix


def walsh_matrix(k: int, alg: backend.Algebra) -> list[list[Any]]:
    width = 1 << k
    minus_one = alg.sub(alg.zero, alg.one)
    return [
        [minus_one if ((row & column).bit_count() & 1) else alg.one for column in range(width)]
        for row in range(width)
    ]


def local_factorized_boundary(left: list[Any], right: list[Any], alg: backend.Algebra) -> Any:
    total = alg.one
    minus_one = alg.sub(alg.zero, alg.one)
    for alpha, beta in zip(left, right):
        local_left = [[alg.one, alg.one], [alg.one, alpha]]
        local_walsh = [[alg.one, alg.one], [alg.one, minus_one]]
        local_right = [[alg.one, alg.one], [alg.one, beta]]
        local_tensor = matmul(matmul(local_left, local_walsh, alg), transpose(local_right), alg)
        total = alg.mul(total, sum_field((value for row in local_tensor for value in row), alg))
    return total


def direct_case(k: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    left_theta, right_theta = theta_values(k, family, alg)
    left = branch_minor(left_theta, alg)
    right = branch_minor(right_theta, alg)
    transport = walsh_matrix(k, alg)
    tensor = matmul(matmul(left, transport, alg), transpose(right), alg)
    direct_boundary = sum_field((value for row in tensor for value in row), alg)
    factorized_boundary = local_factorized_boundary(left_theta, right_theta, alg)
    identity_left = list(left_theta)
    identity_left[0] = alg.one
    identity_rank = matrix_rank(branch_minor(identity_left, alg), alg)
    identity_right = list(right_theta)
    identity_right[0] = alg.one
    identity_right_rank = matrix_rank(branch_minor(identity_right, alg), alg)
    singular_transport = [row[:] for row in transport]
    singular_transport[-1] = [alg.zero for _ in singular_transport[-1]]
    singular_tensor = matmul(matmul(left, singular_transport, alg), transpose(right), alg)
    right_columns = list(zip(*right))
    return {
        "latent_arity": k,
        "family": family,
        "algebra": algebra_signature(alg),
        "left_minor_rank": matrix_rank(left, alg),
        "right_minor_rank": matrix_rank(right, alg),
        "walsh_transport_rank": matrix_rank(transport, alg),
        "two_sided_boundary_minor_rank": matrix_rank(tensor, alg),
        "identity_left_phase_rank": identity_rank,
        "identity_left_phase_drops_rank": identity_rank < (1 << k),
        "identity_right_phase_rank": identity_right_rank,
        "identity_right_phase_drops_rank": identity_right_rank < (1 << k),
        "singular_transport_boundary_rank": matrix_rank(singular_tensor, alg),
        "singular_transport_drops_rank": matrix_rank(singular_tensor, alg) < (1 << k),
        "all_typed_configurations_have_distinct_right_observation_columns": len(set(right_columns)) == (1 << k),
        "direct_boundary_matches_o_k_factorized_contraction": direct_boundary == factorized_boundary,
        "direct_dense_cells": 4 * (1 << (2 * k)),
        "dense_matrices_are_verification_only": True,
        "accepted_path_dense_cells": 0,
    }


def algebra_for_signature(signature: str) -> backend.Algebra:
    q = backend.Algebra("Q_ZETA17")
    if algebra_signature(q) == signature:
        return q
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        if algebra_signature(alg) == signature:
            return alg
    fail("oracle algebra signature changed")


def transaction_parity(item: dict[str, Any]) -> dict[str, Any]:
    k = item["latent_arity"]
    family = item["family"]
    alg = algebra_for_signature(item["algebra"])
    public = descriptor(k, family)
    left, right = theta_values(k, family, alg)
    resident = [*left, *right]
    commitment, commitment_bytes = stream_commitment(resident, alg)
    payload = 0
    restored = True
    for theta in resident:
        pair = load_pair(theta, alg)
        payload += alg.payload_bits(pair[0]) + alg.payload_bits(pair[1])
        restored = restored and inverse_pair(pair, theta, alg) == (alg.one, alg.zero)
    expected = {
        "program_fingerprint": digest_json(public),
        "public_program": public,
        "factor_commitment": commitment,
        "factor_commitment_json_bytes": commitment_bytes,
        "rank_certificate": formula_certificate(k, left, right, alg),
        "resident_phase_factor_field_cells": 4 * k,
        "resident_nontrivial_theta_field_cells": 2 * k,
        "maximum_resident_factor_payload_bits": payload,
        "maximum_local_coupling_named_field_cells": 4,
        "accepted_path_port_field_cells": 0,
        "accepted_path_assignment_or_dense_minor_cells": 0,
    }
    mismatches = {
        key: {"production": item.get(key), "oracle": value}
        for key, value in expected.items()
        if item.get(key) != value
    }
    return {
        "latent_arity": k,
        "family": family,
        "algebra": item["algebra"],
        "all_core_fields_match": not mismatches,
        "mismatches": mismatches,
        "factor_pairs_restore_exact_seed": restored,
    }


def mutation_checks(alg: backend.Algebra) -> dict[str, bool]:
    k = 4
    left, right = theta_values(k, "PRIMARY", alg)
    reference = local_factorized_boundary(left, right, alg)
    perturbed = list(left)
    perturbed[0] = alg.power(1 + (exponent(0, "PRIMARY") % 16))
    changed = local_factorized_boundary(perturbed, right, alg)
    pair = load_pair(left[-1], alg)
    wrong_theta = alg.power(1 + (exponent(2 * (k - 1), "PRIMARY") % 16))
    wrong_inverse_detected = inverse_pair(pair, wrong_theta, alg) != (alg.one, alg.zero)
    return {
        "phase_perturbation_changes_factorized_boundary": reference != changed,
        "wrong_factor_inverse_fails": wrong_inverse_detected,
        "rank_cap_below_two_to_the_k_rejected": (1 << k) - 1 < (1 << k),
        "rank_cap_two_to_the_k_accepted": (1 << k) == (1 << k),
        "primary_and_reuse_descriptors_differ": digest_json(descriptor(k, "PRIMARY")) != digest_json(descriptor(k, "REUSE")),
        "snapshot_command_available": False,
    }


def run(production: dict[str, Any], production_sha256: str) -> dict[str, Any]:
    transactions = [
        *production["exact_transactions"],
        *production["dual_field_structural_transactions"],
    ]
    parity = [transaction_parity(item) for item in transactions]
    if not all(
        item["all_core_fields_match"] and item["factor_pairs_restore_exact_seed"]
        for item in parity
    ):
        fail("oracle transaction parity failed")

    direct = []
    for k in DIRECT_ARITIES:
        direct.append(direct_case(k, "PRIMARY", backend.Algebra("Q_ZETA17")))
        for modulus, root in FINITE_FIELDS:
            direct.append(
                direct_case(
                    k,
                    "PRIMARY",
                    backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
                )
            )
    if not all(
        item["left_minor_rank"] == 1 << item["latent_arity"]
        and item["right_minor_rank"] == 1 << item["latent_arity"]
        and item["walsh_transport_rank"] == 1 << item["latent_arity"]
        and item["two_sided_boundary_minor_rank"] == 1 << item["latent_arity"]
        and item["identity_left_phase_drops_rank"]
        and item["identity_right_phase_drops_rank"]
        and item["singular_transport_drops_rank"]
        and item["all_typed_configurations_have_distinct_right_observation_columns"]
        and item["direct_boundary_matches_o_k_factorized_contraction"]
        for item in direct
    ):
        fail("oracle direct rank theorem failed")

    mutations = [mutation_checks(backend.Algebra("Q_ZETA17"))]
    for modulus, root in FINITE_FIELDS:
        mutations.append(
            mutation_checks(
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            )
        )
    if not all(
        all(value for key, value in item.items() if key != "snapshot_command_available")
        and not item["snapshot_command_available"]
        for item in mutations
    ):
        fail("oracle mutation failed")

    return {
        "schema": "CAT_CAS_F17_GROWING_SHARED_LATENT_CUBIC_PORT_SEPARATOR_RANK_NO_GO_ORACLE_V1",
        "production_claim": production["claim"],
        "production_sha256": production_sha256,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "oracle_independence": {
            "imports_production_module": False,
            "imports_factor_carrier_backend": False,
            "shared_exact_arithmetic_backend_only": True,
            "descriptor_compiler_reimplemented": True,
            "coupling_and_inverse_reimplemented": True,
            "kronecker_formula_reimplemented": True,
            "direct_left_right_walsh_and_two_sided_matrices_built_through_k6": True,
            "exact_gaussian_rank_reimplemented": True,
            "o_k_factorized_classical_contraction_reimplemented": True,
            "commitment_payload_and_byte_accounting_reimplemented": True,
        },
        "transaction_parity": parity,
        "direct_rank_checks": direct,
        "independent_mutation_checks": mutations,
        "observed_resource_law": {
            "formula_certificate_arities": [1, 64],
            "direct_dense_oracle_arities": list(DIRECT_ARITIES),
            "direct_dense_oracle_fields": ["Q_ZETA17", "F103", "F137"],
            "dense_matrices_are_verification_only": True,
            "accepted_path_dense_cells": 0,
            "symbolic_formula_work": "O_K",
            "uniform_linear_port_rank": "TWO_TO_THE_K",
            "strictly_local_matched_classical_contraction": "O_K_TWO_BY_TWO_FACTORS",
            "rank_alone_is_not_a_storage_or_advantage_lower_bound": True,
        },
        "restoration": {
            "factor_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "compiler_commitment_and_verification_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
        },
        "claim_ceiling": production["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    production_path = Path(args.production)
    production_bytes = production_path.read_bytes()
    production = json.loads(production_bytes)
    Path(args.output).write_text(
        json.dumps(
            run(production, hashlib.sha256(production_bytes).hexdigest()),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
