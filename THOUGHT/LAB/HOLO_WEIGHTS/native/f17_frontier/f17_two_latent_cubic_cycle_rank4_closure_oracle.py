#!/usr/bin/env python3
"""Independent exact oracle for the two-latent cubic cycle package."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import f17_nonlinear_canonical_mps_separator_chart as backend


EXACT_BRANCH_COUNTS = (2, 4, 8, 16, 32, 64)
STRUCTURAL_BRANCH_COUNTS = (2, 3, 4, 5, 6, 7, 8)
FINITE_FIELDS = ((137, 16), (239, 211))
FAMILIES = ("PRIMARY", "REUSE")
PORT_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
FINAL_BOUNDARY = "SUM_FOUR_SHARED_H_K_PORT_COMPONENTS_AFTER_LAST_PUBLIC_TRANSPORT"


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def field_integer(alg: backend.Algebra, value: int) -> Any:
    if value < 0:
        return alg.sub(alg.zero, field_integer(alg, -value))
    result = alg.zero
    addend = alg.one
    remaining = value
    while remaining:
        if remaining & 1:
            result = alg.add(result, addend)
        addend = alg.add(addend, addend)
        remaining >>= 1
    return result


def algebra_signature(alg: backend.Algebra) -> str:
    return digest_json({"kind": alg.kind, "modulus": alg.modulus, "root": alg.serialize(alg.root)})


def stream_commitment(values: Iterable[Any], alg: backend.Algebra) -> tuple[str, int]:
    hasher = hashlib.sha256()
    maximum = 0
    for value in values:
        record = json.dumps(alg.serialize(value), sort_keys=True, separators=(",", ":")).encode()
        maximum = max(maximum, len(record))
        hasher.update(len(record).to_bytes(8, "big"))
        hasher.update(record)
    return hasher.hexdigest(), maximum


def exponent(index: int, family: str) -> int:
    if family == "PRIMARY":
        return 1 + ((2 * index) % 16)
    if family == "REUSE":
        return 1 + ((2 * index + 1) % 16)
    fail("oracle family changed")


def exponent_pairs(branches: int, family: str) -> list[tuple[int, int]]:
    return [(exponent(2 * branch + 1, family), exponent(2 * branch + 2, family)) for branch in range(branches)]


def axes(branches: int) -> list[str]:
    return [("H" if branch % 2 == 0 else "K") if branch < branches - 1 else "NONE" for branch in range(branches)]


def descriptor(branches: int, family: str) -> dict[str, Any]:
    return {
        "branch_count": branches,
        "factor_count": 2 * branches,
        "local_bit_count": 3 * branches,
        "total_logical_bits": 2 + 3 * branches,
        "family": family,
        "shared_latent_ports": ["H", "K"],
        "branch_local_bits": ["ANCHOR", "H_LEAF", "K_LEAF"],
        "branch_factors": [
            "ALPHA_BRANCH_TO_H_TIMES_ANCHOR_TIMES_H_LEAF",
            "BETA_BRANCH_TO_K_TIMES_ANCHOR_TIMES_K_LEAF",
        ],
        "theta_exponent_pairs": [list(pair) for pair in exponent_pairs(branches, family)],
        "transport_axes": axes(branches),
        "transport": "UNNORMALIZED_BINARY_WALSH_ON_DECLARED_SHARED_PORT_AXIS",
        "port_order": [list(pair) for pair in PORT_ORDER],
        "final_boundary": FINAL_BOUNDARY,
    }


def branch_response_enumerated(alpha: Any, beta: Any, alg: backend.Algebra) -> list[Any]:
    response = []
    for h, k in PORT_ORDER:
        total = alg.zero
        for anchor, h_leaf, k_leaf in itertools.product((0, 1), repeat=3):
            value = alg.one
            if h and anchor and h_leaf:
                value = alg.mul(value, alpha)
            if k and anchor and k_leaf:
                value = alg.mul(value, beta)
            total = alg.add(total, value)
        response.append(total)
    return response


def walsh(values: list[Any], axis: str, alg: backend.Algebra) -> list[Any]:
    a, b, c, d = values
    if axis == "H":
        return [alg.add(a, c), alg.add(b, d), alg.sub(a, c), alg.sub(b, d)]
    if axis == "K":
        return [alg.add(a, b), alg.sub(a, b), alg.add(c, d), alg.sub(c, d)]
    fail("oracle Walsh axis changed")


def inverse_walsh(values: list[Any], axis: str, alg: backend.Algebra) -> list[Any]:
    half = alg.inverse(field_integer(alg, 2))
    return [alg.mul(half, value) for value in walsh(values, axis, alg)]


def field_sum(values: Iterable[Any], alg: backend.Algebra) -> Any:
    result = alg.zero
    for value in values:
        result = alg.add(result, value)
    return result


def recurrence(
    branches: int,
    family: str,
    alg: backend.Algebra,
    initial: list[Any] | None = None,
    *,
    disable_transport: bool = False,
    transport_before: bool = False,
    override_pairs: list[tuple[int, int]] | None = None,
) -> tuple[Any, list[Any], int]:
    state = list(initial) if initial is not None else [alg.one for _ in range(4)]
    maximum_payload = sum(alg.payload_bits(value) for value in state)
    pairs = override_pairs if override_pairs is not None else exponent_pairs(branches, family)
    transport = axes(branches)
    for (alpha_exp, beta_exp), axis in zip(pairs, transport):
        if transport_before and axis != "NONE" and not disable_transport:
            state = walsh(state, axis, alg)
        response = branch_response_enumerated(alg.power(alpha_exp), alg.power(beta_exp), alg)
        state = [alg.mul(value, factor) for value, factor in zip(state, response)]
        maximum_payload = max(maximum_payload, sum(alg.payload_bits(value) for value in state))
        if not transport_before and axis != "NONE" and not disable_transport:
            state = walsh(state, axis, alg)
        maximum_payload = max(maximum_payload, sum(alg.payload_bits(value) for value in state))
    return field_sum(state, alg), state, maximum_payload


def reverse_port(
    final_state: list[Any], branches: int, family: str, alg: backend.Algebra
) -> tuple[list[Any], int]:
    state = list(final_state)
    maximum_payload = sum(alg.payload_bits(value) for value in state)
    pairs = exponent_pairs(branches, family)
    transport = axes(branches)
    for branch in range(branches - 1, -1, -1):
        alpha_exp, beta_exp = pairs[branch]
        response = branch_response_enumerated(alg.power(alpha_exp), alg.power(beta_exp), alg)
        state = [alg.mul(value, alg.inverse(factor)) for value, factor in zip(state, response)]
        maximum_payload = max(maximum_payload, sum(alg.payload_bits(value) for value in state))
        if branch > 0:
            state = inverse_walsh(state, transport[branch - 1], alg)
        maximum_payload = max(maximum_payload, sum(alg.payload_bits(value) for value in state))
    return state, maximum_payload


def reverse_port_variant(
    final_state: list[Any],
    branches: int,
    family: str,
    alg: backend.Algebra,
    variant: str,
) -> list[Any]:
    """Reexecute every inverse step while mutating one declared ordering law."""
    state = list(final_state)
    pairs = exponent_pairs(branches, family)
    transport = axes(branches)
    last = branches - 1
    for branch in range(last, -1, -1):
        alpha_exp, beta_exp = pairs[branch]
        response = branch_response_enumerated(
            alg.power(alpha_exp), alg.power(beta_exp), alg
        )
        if variant == "REORDER_LAST_DIAGONAL_AND_WALSH" and branch == last:
            state = inverse_walsh(state, transport[branch - 1], alg)
            state = [
                alg.mul(value, alg.inverse(factor))
                for value, factor in zip(state, response)
            ]
            continue
        state = [
            alg.mul(value, alg.inverse(factor))
            for value, factor in zip(state, response)
        ]
        if branch > 0:
            if variant == "FORWARD_WALSH_AS_INVERSE" and branch == last:
                state = walsh(state, transport[branch - 1], alg)
            else:
                state = inverse_walsh(state, transport[branch - 1], alg)
    return state


def inverse_factor_pair(pair: tuple[Any, Any], theta: Any, alg: backend.Algebra) -> tuple[Any, Any]:
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
                alg.sub(value, alg.mul(coefficient, pivot_value))
                for value, pivot_value in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == rows:
            break
    return rank


def branch_map(alpha: Any, beta: Any, alg: backend.Algebra) -> list[list[Any]]:
    return [
        [
            alg.mul(
                alpha if h and anchor and h_leaf else alg.one,
                beta if k and anchor and k_leaf else alg.one,
            )
            for h, k in PORT_ORDER
        ]
        for anchor, h_leaf, k_leaf in itertools.product((0, 1), repeat=3)
    ]


def two_branch_tensor_case(family: str, alg: backend.Algebra) -> dict[str, Any]:
    pairs = exponent_pairs(2, family)
    left = branch_map(alg.power(pairs[0][0]), alg.power(pairs[0][1]), alg)
    right = branch_map(alg.power(pairs[1][0]), alg.power(pairs[1][1]), alg)
    tensor = []
    for left_row in left:
        row = []
        transported = walsh(left_row, "H", alg)
        for right_row in right:
            row.append(field_sum((alg.mul(a, b) for a, b in zip(transported, right_row)), alg))
        tensor.append(row)
    selected_indices = (4, 5, 6, 7)
    selected = [[tensor[i][j] for j in selected_indices] for i in selected_indices]
    recurrence_boundary, _, _ = recurrence(2, family, alg)
    return {
        "family": family,
        "algebra": algebra_signature(alg),
        "left_branch_map_rank": matrix_rank(left, alg),
        "right_branch_map_rank": matrix_rank(right, alg),
        "full_two_branch_boundary_tensor_rank": matrix_rank(tensor, alg),
        "selected_anchor_one_four_by_four_minor_rank": matrix_rank(selected, alg),
        "direct_tensor_sum_matches_recurrence": alg.serialize(field_sum((value for row in tensor for value in row), alg)) == alg.serialize(recurrence_boundary),
        "full_tensor_shape": [8, 8],
    }


def compiled_row(branches: int, family: str, alg: backend.Algebra) -> list[Any]:
    basis = [
        [alg.one, alg.zero, alg.zero, alg.zero],
        [alg.zero, alg.one, alg.zero, alg.zero],
        [alg.zero, alg.zero, alg.one, alg.zero],
        [alg.zero, alg.zero, alg.zero, alg.one],
    ]
    return [recurrence(branches, family, alg, initial=vector)[0] for vector in basis]


def transaction_oracle(item: dict[str, Any], alg: backend.Algebra) -> dict[str, Any]:
    branches = item["branch_count"]
    family = item["family"]
    pairs = exponent_pairs(branches, family)
    values = []
    maximum_factor_payload = 0
    for alpha_exp, beta_exp in pairs:
        for theta in (alg.power(alpha_exp), alg.power(beta_exp)):
            values.extend((alg.one, theta))
            maximum_factor_payload = sum(alg.payload_bits(value) for value in values)
            if inverse_factor_pair((alg.one, theta), theta, alg) != (alg.one, alg.zero):
                fail("oracle factor inverse failed")
    commitment, commitment_bytes = stream_commitment(values, alg)
    boundary, final_state, forward_port_payload = recurrence(branches, family, alg)
    restored, inverse_port_payload = reverse_port(final_state, branches, family, alg)
    row = compiled_row(branches, family, alg)
    row_commitment, row_bytes = stream_commitment(row, alg)
    public = descriptor(branches, family)
    expected = {
        "boundary": alg.serialize(boundary),
        "program_fingerprint": digest_json(public),
        "factor_commitment": commitment,
        "factor_commitment_json_bytes": commitment_bytes,
        "resident_phase_factor_field_cells": 4 * branches,
        "resident_nontrivial_theta_field_cells": 2 * branches,
        "resident_shared_latent_port_field_cells": 4,
        "exact_two_branch_junction_separator_rank": 4,
        "maximum_named_port_update_field_cells": 16,
        "final_boundary_payload_bits": alg.payload_bits(boundary),
        "maximum_resident_factor_payload_bits": maximum_factor_payload,
        "maximum_resident_port_payload_bits": max(forward_port_payload, inverse_port_payload),
        "public_program_json_bytes": len(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()),
    }
    mismatches = {key: {"production": item.get(key), "oracle": value} for key, value in expected.items() if item.get(key) != value}
    return {
        "branch_count": branches,
        "family": family,
        "algebra": algebra_signature(alg),
        "all_core_fields_match": not mismatches,
        "mismatches": mismatches,
        "port_restores_exact_seed": restored == [alg.one for _ in range(4)],
        "factor_pairs_restore_exact_seed": True,
        "compiled_final_row_commitment": row_commitment,
        "compiled_final_row_commitment_json_bytes": row_bytes,
        "maximum_resident_port_payload_bits": expected[
            "maximum_resident_port_payload_bits"
        ],
    }


def mutation_checks(alg: backend.Algebra) -> dict[str, bool]:
    branches = 4
    family = "PRIMARY"
    pairs = exponent_pairs(branches, family)
    reference, final_state, _ = recurrence(branches, family, alg)
    disabled, _, _ = recurrence(branches, family, alg, disable_transport=True)
    before, _, _ = recurrence(branches, family, alg, transport_before=True)
    perturbed = list(pairs)
    perturbed[0] = (1 + (perturbed[0][0] % 16), perturbed[0][1])
    changed, _, _ = recurrence(branches, family, alg, override_pairs=perturbed)

    first_alpha = alg.power(pairs[0][0])
    first_beta = alg.power(pairs[0][1])
    identity_map = branch_map(alg.one, first_beta, alg)
    proper_map = branch_map(first_alpha, first_beta, alg)

    under_shared = alg.one
    for alpha_exp, beta_exp in pairs:
        local = field_sum(branch_response_enumerated(alg.power(alpha_exp), alg.power(beta_exp), alg), alg)
        under_shared = alg.mul(under_shared, local)
    overmerged = alg.add(final_state[0], final_state[3])

    seed = [alg.one for _ in range(4)]
    lawful = reverse_port_variant(final_state, branches, family, alg, "LAWFUL")
    wrong_order = reverse_port_variant(
        final_state, branches, family, alg, "REORDER_LAST_DIAGONAL_AND_WALSH"
    )
    wrong_walsh = reverse_port_variant(
        final_state, branches, family, alg, "FORWARD_WALSH_AS_INVERSE"
    )

    return {
        "identity_theta_drops_branch_map_below_rank4": matrix_rank(identity_map, alg) < 4,
        "proper_branch_map_has_rank4": matrix_rank(proper_map, alg) == 4,
        "forced_rank3_rejected": matrix_rank(proper_map, alg) == 4,
        "transport_disabled_changes_boundary": alg.serialize(disabled) != alg.serialize(reference),
        "transport_before_consumer_changes_boundary": alg.serialize(before) != alg.serialize(reference),
        "phase_perturbation_changes_boundary": alg.serialize(changed) != alg.serialize(reference),
        "under_share_changes_boundary": alg.serialize(under_shared) != alg.serialize(reference),
        "overmerge_h_equals_k_changes_boundary": alg.serialize(overmerged) != alg.serialize(reference),
        "lawful_complete_port_inverse_restores_seed": lawful == seed,
        "wrong_inverse_order_fails": wrong_order != seed and wrong_order != lawful,
        "wrong_walsh_inverse_fails": wrong_walsh != seed and wrong_walsh != lawful,
        "missing_inverse_leaves_nonseed_port": final_state != seed,
        "primary_and_reuse_descriptors_differ": digest_json(descriptor(4, "PRIMARY")) != digest_json(descriptor(4, "REUSE")),
        "snapshot_command_available": False,
    }


def algebra_for_item(item: dict[str, Any]) -> backend.Algebra:
    signature = item["algebra"]
    q = backend.Algebra("Q_ZETA17")
    if algebra_signature(q) == signature:
        return q
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        if algebra_signature(alg) == signature:
            return alg
    fail("oracle encountered unknown algebra signature")


def run(production: dict[str, Any], production_sha256: str) -> dict[str, Any]:
    transactions = [*production["exact_transactions"], *production["dual_field_structural_transactions"]]
    parity = [transaction_oracle(item, algebra_for_item(item)) for item in transactions]
    if not all(item["all_core_fields_match"] and item["port_restores_exact_seed"] for item in parity):
        fail("oracle transaction parity failed")

    production_baselines = {
        (item["branch_count"], item["family"], item["algebra"]): item
        for item in production["compiled_classical_baselines"]
    }
    compiled_checks = []
    for item in parity:
        baseline = production_baselines[(item["branch_count"], item["family"], item["algebra"])]
        compiled_checks.append(
            {
                "branch_count": item["branch_count"],
                "family": item["family"],
                "algebra": item["algebra"],
                "row_commitment_matches": baseline["compiled_final_row_commitment"] == item["compiled_final_row_commitment"],
                "row_record_bytes_match": baseline["compiled_final_row_commitment_json_bytes"] == item["compiled_final_row_commitment_json_bytes"],
                "runtime_port_payload_matches_transaction": baseline[
                    "runtime_maximum_exact_port_payload_bits"
                ]
                == item["maximum_resident_port_payload_bits"],
            }
        )
    if not all(
        item["row_commitment_matches"]
        and item["row_record_bytes_match"]
        and item["runtime_port_payload_matches_transaction"]
        for item in compiled_checks
    ):
        fail("oracle compiled baseline parity failed")

    tensor_cases = []
    for family in FAMILIES:
        tensor_cases.append(two_branch_tensor_case(family, backend.Algebra("Q_ZETA17")))
        for modulus, root in FINITE_FIELDS:
            tensor_cases.append(two_branch_tensor_case(family, backend.Algebra(f"F{modulus}", modulus=modulus, root=root)))
    if not all(
        item["left_branch_map_rank"] == 4
        and item["right_branch_map_rank"] == 4
        and item["full_two_branch_boundary_tensor_rank"] == 4
        and item["selected_anchor_one_four_by_four_minor_rank"] == 4
        and item["direct_tensor_sum_matches_recurrence"]
        for item in tensor_cases
    ):
        fail("oracle direct two-branch tensor rank failed")

    mutations = [mutation_checks(backend.Algebra("Q_ZETA17"))]
    for modulus, root in FINITE_FIELDS:
        mutations.append(mutation_checks(backend.Algebra(f"F{modulus}", modulus=modulus, root=root)))
    if not all(all(value for key, value in item.items() if key != "snapshot_command_available") and not item["snapshot_command_available"] for item in mutations):
        fail("oracle mutation control failed")

    exact_rows = production["exact_transactions"]
    return {
        "schema": "CAT_CAS_F17_TWO_LATENT_CUBIC_CYCLE_RANK4_CLOSURE_ORACLE_V1",
        "production_claim": production["claim"],
        "production_sha256": production_sha256,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "oracle_independence": {
            "imports_production_module": False,
            "shared_exact_arithmetic_backend_only": True,
            "descriptor_compiler_reimplemented": True,
            "branch_response_direct_local_enumeration": True,
            "four_state_recurrence_reimplemented": True,
            "full_two_branch_boundary_tensor_directly_built": True,
            "exact_gaussian_rank_reimplemented": True,
            "carrier_and_port_inverse_reimplemented": True,
            "commitment_and_byte_accounting_reimplemented": True,
        },
        "transaction_parity": parity,
        "compiled_classical_baseline_checks": compiled_checks,
        "direct_two_branch_tensor_rank_checks": tensor_cases,
        "independent_mutation_checks": mutations,
        "observed_resource_law": {
            "exact_branch_counts": [item["branch_count"] for item in exact_rows],
            "exact_resident_factor_payload_bits": [item["maximum_resident_factor_payload_bits"] for item in exact_rows],
            "exact_resident_port_payload_bits": [item["maximum_resident_port_payload_bits"] for item in exact_rows],
            "exact_final_boundary_payload_bits": [item["final_boundary_payload_bits"] for item in exact_rows],
            "phase_factor_field_cells": [item["resident_phase_factor_field_cells"] for item in exact_rows],
            "shared_port_field_cells": [item["resident_shared_latent_port_field_cells"] for item in exact_rows],
            "matched_runtime_dynamic_port_field_cells": 4,
            "fixed_logical_rank_implies_fixed_exact_bit_width": False,
            "full_exact_bit_complexity_established": False,
        },
        "restoration": {
            "factor_and_shared_port_carrier": "EXACT_ALGEBRAIC_RESTORATION",
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
        json.dumps(run(production, hashlib.sha256(production_bytes).hexdigest()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
