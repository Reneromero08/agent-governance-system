#!/usr/bin/env python3
"""Separate exact oracle for the rank-one cubic/Walsh C17 chart.

The oracle does not import the production package.  It reconstructs public
programs, canonical derivative rank, the 34-coordinate recurrence, a distinct
17-residue multiplicity contraction, direct Boolean assignment sums for small
cases, exact inverse restoration, and coefficient payload measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import f17_nonlinear_canonical_mps_separator_chart as algebra_backend


P = 17
EXACT_CASES = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256))
STRUCTURAL_CASES = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64))
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def integer(alg: algebra_backend.Algebra, value: int) -> Any:
    if value < 0:
        return alg.sub(alg.zero, integer(alg, -value))
    total = alg.zero
    addend = alg.one
    while value:
        if value & 1:
            total = alg.add(total, addend)
        value >>= 1
        if value:
            addend = alg.add(addend, addend)
    return total


def power(alg: algebra_backend.Algebra, value: Any, exponent: int) -> Any:
    result = alg.one
    while exponent:
        if exponent & 1:
            result = alg.mul(result, value)
        exponent >>= 1
        if exponent:
            value = alg.mul(value, value)
    return result


def parameters(index: int, family: str) -> tuple[int, int]:
    if family == "PRIMARY":
        return 1 + ((5 * index * index + 7 * index + 3) % 16), (3 * index + 1) % P
    if family == "REUSE":
        return 1 + ((9 * index * index + 4 * index + 11) % 16), (7 * index + 5) % P
    if family == "ALTERNATE":
        return 1 + ((11 * index * index + 2 * index + 6) % 16), (13 * index + 2) % P
    fail("oracle family changed")


def descriptor(branch_pairs: int, rounds: int, family: str) -> dict[str, Any]:
    terms = [
        {"left": 2 * index, "right": 2 * index + 1, "coefficient_mod17": 1}
        for index in range(branch_pairs)
    ]
    primitives = []
    for index in range(rounds):
        multiplier, constant = parameters(index, family)
        primitives.extend(
            [
                {
                    "kind": "CUBIC_PHASE_ON_UNRESOLVED_X",
                    "signature": 0,
                    "multiplier_mod17": multiplier,
                    "constant_mod17": constant,
                },
                {"kind": "UNNORMALIZED_WALSH_ON_UNRESOLVED_X"},
            ]
        )
    return {
        "branch_pairs": branch_pairs,
        "boolean_branch_bits": 2 * branch_pairs,
        "rounds": rounds,
        "family": family,
        "unresolved_typed_port": "X:BOOLEAN_PHASE_PORT",
        "signatures": [{"name": "PAIR_PRODUCT_SUM", "terms": terms}],
        "primitives": primitives,
        "observation": {
            "kind": "FINAL_X_CHARACTER_ONLY",
            "exponent_mod17": 1 + ((branch_pairs + 3 * rounds + len(family)) % 16),
        },
    }


def canonical_rows(signatures: list[dict[str, Any]], variable_count: int) -> tuple[list[list[int]], list[tuple[int, int]]]:
    maps = []
    for signature in signatures:
        row: dict[tuple[int, int], int] = {}
        for term in signature["terms"]:
            left, right = term["left"], term["right"]
            if not (0 <= left < variable_count and 0 <= right < variable_count):
                fail("oracle topology exceeds public variables")
            key = tuple(sorted((left, right)))
            value = (row.get(key, 0) + term["coefficient_mod17"]) % P
            if value:
                row[key] = value
            elif key in row:
                del row[key]
        maps.append(row)
    columns = sorted({key for row in maps for key in row})
    return [[row.get(key, 0) for key in columns] for row in maps], columns


def rank_mod17(matrix: list[list[int]]) -> int:
    work = [row[:] for row in matrix]
    rank = 0
    columns = len(work[0]) if work else 0
    for column in range(columns):
        pivot = next((row for row in range(rank, len(work)) if work[row][column] % P), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inv = pow(work[rank][column], P - 2, P)
        work[rank] = [(value * inv) % P for value in work[rank]]
        for row in range(len(work)):
            coefficient = work[row][column] % P
            if row != rank and coefficient:
                work[row] = [
                    (value - coefficient * basis) % P
                    for value, basis in zip(work[row], work[rank])
                ]
        rank += 1
    return rank


def serialize_payload_bits(value: Any, alg: algebra_backend.Algebra) -> int:
    serialized = alg.serialize(value)
    if alg.modulus:
        # A declared finite-field cell has fixed modulus width, including zero.
        return max(1, alg.modulus.bit_length())
    total = 0
    for numerator, denominator in serialized:
        signed = 1 if numerator == 0 else abs(int(numerator)).bit_length() + 1
        total += signed + max(1, int(denominator).bit_length())
    return total


def state_payload(rows: list[list[Any]], alg: algebra_backend.Algebra) -> int:
    return sum(serialize_payload_bits(value, alg) for row in rows for value in row)


def forward_rows(branch_pairs: int, rounds: int, family: str, alg: algebra_backend.Algebra) -> tuple[list[list[Any]], int]:
    rows = [[alg.zero for _ in range(P)] for _ in range(2)]
    rows[0][0] = alg.one
    rows[1][0] = alg.one
    maximum = state_payload(rows, alg)
    for index in range(rounds):
        shift, constant = parameters(index, family)
        scalar = alg.power(constant)
        shifted = [alg.zero for _ in range(P)]
        for source, value in enumerate(rows[1]):
            shifted[(source + shift) % P] = alg.mul(scalar, value)
        rows[1] = shifted
        maximum = max(maximum, state_payload(rows, alg))
        left = [alg.add(first, second) for first, second in zip(rows[0], rows[1])]
        right = [alg.sub(first, second) for first, second in zip(rows[0], rows[1])]
        rows = [left, right]
        maximum = max(maximum, state_payload(rows, alg))
    return rows, maximum


def inverse_rows(rows: list[list[Any]], rounds: int, family: str, alg: algebra_backend.Algebra) -> list[list[Any]]:
    half = alg.inverse(integer(alg, 2))
    for index in range(rounds - 1, -1, -1):
        left = [alg.mul(half, alg.add(first, second)) for first, second in zip(rows[0], rows[1])]
        right = [alg.mul(half, alg.sub(first, second)) for first, second in zip(rows[0], rows[1])]
        rows = [left, right]
        shift, constant = parameters(index, family)
        scalar = alg.power(-constant)
        unshifted = [alg.zero for _ in range(P)]
        for source, value in enumerate(rows[1]):
            unshifted[(source - shift) % P] = alg.mul(scalar, value)
        rows[1] = unshifted
    return rows


def project_rows(rows: list[list[Any]], branch_pairs: int, rounds: int, family: str, alg: algebra_backend.Algebra) -> Any:
    observation = alg.power(1 + ((branch_pairs + 3 * rounds + len(family)) % 16))
    boundary = alg.zero
    three = integer(alg, 3)
    for residue in range(P):
        moment = power(alg, alg.add(three, alg.power(residue)), branch_pairs)
        component = alg.add(rows[0][residue], alg.mul(observation, rows[1][residue]))
        boundary = alg.add(boundary, alg.mul(moment, component))
    return boundary


def residue_multiplicities(branch_pairs: int) -> list[int]:
    counts = [0 for _ in range(P)]
    counts[0] = 1
    for _ in range(branch_pairs):
        counts = [3 * counts[q] + counts[(q - 1) % P] for q in range(P)]
    return counts


def residue_class_boundary(branch_pairs: int, rounds: int, family: str, alg: algebra_backend.Algebra) -> Any:
    counts = residue_multiplicities(branch_pairs)
    observation = alg.power(1 + ((branch_pairs + 3 * rounds + len(family)) % 16))
    boundary = alg.zero
    for q, count in enumerate(counts):
        state = [alg.one, alg.one]
        for index in range(rounds):
            multiplier, constant = parameters(index, family)
            state[1] = alg.mul(state[1], alg.power(multiplier * q + constant))
            state = [alg.add(state[0], state[1]), alg.sub(state[0], state[1])]
        observed = alg.add(state[0], alg.mul(observation, state[1]))
        boundary = alg.add(boundary, alg.mul(integer(alg, count), observed))
    return boundary


def direct_assignment_boundary(branch_pairs: int, rounds: int, family: str, alg: algebra_backend.Algebra) -> Any:
    observation = alg.power(1 + ((branch_pairs + 3 * rounds + len(family)) % 16))
    total = alg.zero
    for assignment in range(1 << (2 * branch_pairs)):
        q = sum(
            ((assignment >> (2 * index)) & 1) * ((assignment >> (2 * index + 1)) & 1)
            for index in range(branch_pairs)
        ) % P
        state = [alg.one, alg.one]
        for index in range(rounds):
            multiplier, constant = parameters(index, family)
            state[1] = alg.mul(state[1], alg.power(multiplier * q + constant))
            state = [alg.add(state[0], state[1]), alg.sub(state[0], state[1])]
        total = alg.add(total, alg.add(state[0], alg.mul(observation, state[1])))
    return total


def state_commitment(rows: list[list[Any]], alg: algebra_backend.Algebra) -> str:
    state = hashlib.sha256()
    for row in rows:
        for value in row:
            record = json.dumps(alg.serialize(value), separators=(",", ":")).encode()
            state.update(len(record).to_bytes(8, "big"))
            state.update(record)
    return state.hexdigest()


def reconstruct_transaction(item: dict[str, Any], alg: algebra_backend.Algebra) -> dict[str, Any]:
    branch_pairs, rounds, family = item["branch_pairs"], item["rounds"], item["family"]
    public = descriptor(branch_pairs, rounds, family)
    matrix, columns = canonical_rows(public["signatures"], public["boolean_branch_bits"])
    rows, maximum_payload = forward_rows(branch_pairs, rounds, family, alg)
    boundary = project_rows(rows, branch_pairs, rounds, family, alg)
    residue_boundary = residue_class_boundary(branch_pairs, rounds, family, alg)
    restored = inverse_rows([row[:] for row in rows], rounds, family, alg)
    seed = [alg.one, *[alg.zero for _ in range(P - 1)]]
    return {
        "branch_pairs": branch_pairs,
        "rounds": rounds,
        "family": family,
        "program_fingerprint_equal": digest(public) == item["program_fingerprint"],
        "signature_rank": rank_mod17(matrix),
        "canonical_monomials": len(columns),
        "boundary_equal": alg.serialize(boundary) == item["final_boundary"],
        "residue_class_boundary_equal": boundary == residue_boundary,
        "state_commitment_equal": state_commitment(rows, alg) == item["final_state_commitment"],
        "maximum_resident_payload_bits": maximum_payload,
        "payload_tuple_equal": maximum_payload == item["maximum_resident_payload_bits"],
        "exact_inverse_restores_seed": restored == [seed, seed],
        "logical_cells": 34,
    }


def mutation_checks() -> dict[str, bool]:
    base = descriptor(4, 4, "PRIMARY")
    qprime = {
        "name": "CROSS_PAIR_PRODUCT_SUM",
        "terms": [
            {
                "left": 2 * index + 1,
                "right": 2 * ((index + 1) % 4),
                "coefficient_mod17": 1,
            }
            for index in range(4)
        ],
    }
    rank_two_matrix, _ = canonical_rows([*base["signatures"], qprime], 8)
    duplicate = {
        "name": "DUPLICATE_MOD17",
        "terms": [
            term
            for source in base["signatures"][0]["terms"]
            for term in (
                {**source, "coefficient_mod17": 18},
                {
                    "left": source["right"],
                    "right": source["left"],
                    "coefficient_mod17": -17,
                },
            )
        ],
    }
    duplicate_matrix, _ = canonical_rows([duplicate], 8)
    extra = {
        "name": "EXTRA_UNSAMPLED_MONOMIAL",
        "terms": [{"left": 0, "right": 2, "coefficient_mod17": 1}],
    }
    extra_matrix, _ = canonical_rows([base["signatures"][0], extra], 8)
    alg = algebra_backend.Algebra("F137", modulus=137, root=16)
    rows, _ = forward_rows(4, 4, "PRIMARY", alg)
    wrong = inverse_rows([row[:] for row in rows], 4, "REUSE", alg)
    seed = [alg.one, *[alg.zero for _ in range(P - 1)]]
    one_minus_s = [alg.zero for _ in range(P)]
    one_minus_s[0] = alg.one
    one_minus_s[1] = alg.sub(alg.zero, alg.one)
    geometric = [alg.one for _ in range(P)]
    product = [alg.zero for _ in range(P)]
    for first, left in enumerate(one_minus_s):
        for second, right in enumerate(geometric):
            product[(first + second) % P] = alg.add(
                product[(first + second) % P], alg.mul(left, right)
            )
    return {
        "independent_second_signature_rank_two": rank_mod17(rank_two_matrix) == 2,
        "rank_two_canonical_group_algebra_chart_cells_578": 2 * P ** rank_mod17(rank_two_matrix) == 578,
        "mod17_duplicate_signature_rank_one": rank_mod17(duplicate_matrix) == 1,
        "extra_unsampled_monomial_rank_two": rank_mod17(extra_matrix) == 2,
        "wrong_family_inverse_fails_seed_restoration": wrong != [seed, seed],
        "one_minus_s_is_zero_divisor_in_c17_group_algebra": (
            any(value != alg.zero for value in one_minus_s)
            and any(value != alg.zero for value in geometric)
            and all(value == alg.zero for value in product)
        ),
        "snapshot_or_baseline_reload_used": False,
    }


def algebra_for_item(item: dict[str, Any]) -> algebra_backend.Algebra:
    signature = item["algebra"]
    q = algebra_backend.Algebra("Q_ZETA17")
    if digest({"kind": q.kind, "modulus": q.modulus, "root": q.serialize(q.root)}) == signature:
        return q
    for modulus, root in FINITE_FIELDS:
        alg = algebra_backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        if digest({"kind": alg.kind, "modulus": alg.modulus, "root": alg.serialize(alg.root)}) == signature:
            return alg
    fail("oracle cannot identify production algebra")


def run(production: dict[str, Any]) -> dict[str, Any]:
    items = [*production["exact_transactions"], *production["dual_field_structural_transactions"]]
    parity = [reconstruct_transaction(item, algebra_for_item(item)) for item in items]
    direct_checks = []
    for modulus, root in FINITE_FIELDS:
        for branch_pairs in range(1, 7):
            rounds = 2 * branch_pairs
            alg = algebra_backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            rows, _ = forward_rows(branch_pairs, rounds, "PRIMARY", alg)
            chart = project_rows(rows, branch_pairs, rounds, "PRIMARY", alg)
            direct = direct_assignment_boundary(branch_pairs, rounds, "PRIMARY", alg)
            residue = residue_class_boundary(branch_pairs, rounds, "PRIMARY", alg)
            direct_checks.append(
                {
                    "modulus": modulus,
                    "branch_pairs": branch_pairs,
                    "rounds": rounds,
                    "assignments_enumerated_for_verification_only": 1 << (2 * branch_pairs),
                    "chart_matches_direct_assignments": chart == direct,
                    "chart_matches_residue_class_dynamic_program": chart == residue,
                }
            )
    if not all(
        item["program_fingerprint_equal"]
        and item["signature_rank"] == 1
        and item["boundary_equal"]
        and item["residue_class_boundary_equal"]
        and item["state_commitment_equal"]
        and item["payload_tuple_equal"]
        and item["exact_inverse_restores_seed"]
        for item in parity
    ):
        fail("independent transaction reconstruction failed")
    if not all(
        item["chart_matches_direct_assignments"]
        and item["chart_matches_residue_class_dynamic_program"]
        for item in direct_checks
    ):
        fail("independent direct assignment check failed")
    mutations = mutation_checks()
    if mutations["snapshot_or_baseline_reload_used"] or not all(
        value for key, value in mutations.items() if key != "snapshot_or_baseline_reload_used"
    ):
        fail("independent mutation check failed")
    return {
        "schema": "CAT_CAS_F17_RANK1_CUBIC_WALSH_DERIVATIVE_GROUP_ALGEBRA_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_claim": production["claim"],
        "production_result_sha256": digest(production),
        "transaction_parity": parity,
        "direct_assignment_checks": direct_checks,
        "independent_mutation_checks": mutations,
        "observed_resource_law": {
            "resident_field_cells": 34,
            "payload_tuples_reproduced_independently": True,
            "rank_two_group_algebra_field_cells": 578,
            "accepted_path_assignment_expansion": False,
            "oracle_assignment_expansion_small_cases_only": True,
            "strongest_classical_recurrence": "IDENTICAL_34_COORDINATE_OR_17_RESIDUE_CLASS_RECURRENCE",
        },
        "claim_ceiling": {
            "rank_one_derivative_signature_scope_only": True,
            "rank_two_fixed_34_cell_closure": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "catvm_custody": False,
            "physical_execution": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    production = json.loads(Path(args.production).read_text(encoding="utf-8"))
    Path(args.output).write_text(json.dumps(run(production), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
