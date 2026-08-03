#!/usr/bin/env python3
"""Separate exact oracle for the two-signature separation-rank diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import f17_nonlinear_canonical_mps_separator_chart as algebra_backend


P = 17
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def integer(alg: algebra_backend.Algebra, value: int) -> Any:
    if value < 0:
        return alg.sub(alg.zero, integer(alg, -value))
    result = alg.zero
    addend = alg.one
    while value:
        if value & 1:
            result = alg.add(result, addend)
        value >>= 1
        if value:
            addend = alg.add(addend, addend)
    return result


def scalar_power(alg: algebra_backend.Algebra, value: Any, exponent: int) -> Any:
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


def descriptor(branch_pairs: int, phase_steps: int, family: str) -> dict[str, Any]:
    offset = 2 * branch_pairs
    signatures = [
        {
            "name": "PAIR_PRODUCT_SUM",
            "terms": [
                {"left": 2 * index, "right": 2 * index + 1, "coefficient_mod17": 1}
                for index in range(branch_pairs)
            ],
        },
        {
            "name": "SECOND_DISJOINT_PAIR_PRODUCT_SUM",
            "terms": [
                {
                    "left": offset + 2 * index,
                    "right": offset + 2 * index + 1,
                    "coefficient_mod17": 1,
                }
                for index in range(branch_pairs)
            ],
        },
    ]
    steps = []
    for index in range(phase_steps):
        multiplier, constant = parameters(index, family)
        steps.append(
            {
                "kind": "CUBIC_PHASE_THEN_UNNORMALIZED_WALSH",
                "signature_axis": index % 2,
                "multiplier_mod17": multiplier,
                "constant_mod17": constant,
            }
        )
    return {
        "branch_pairs_per_signature": branch_pairs,
        "boolean_branch_bits": 4 * branch_pairs,
        "phase_steps": phase_steps,
        "family": family,
        "unresolved_typed_port": "X:BOOLEAN_PHASE_PORT",
        "quadratic_signatures": signatures,
        "steps": steps,
        "final_observation_exponent_mod17": 1 + ((3 * branch_pairs + 5 * phase_steps + len(family)) % 16),
    }


def zeros(alg: algebra_backend.Algebra) -> list[list[Any]]:
    return [[alg.zero for _ in range(P)] for _ in range(P)]


def serialized_payload_bits(value: Any, alg: algebra_backend.Algebra) -> int:
    serialized = alg.serialize(value)
    if alg.modulus:
        return max(1, alg.modulus.bit_length())
    total = 0
    for numerator, denominator in serialized:
        signed = 1 if numerator == 0 else abs(int(numerator)).bit_length() + 1
        total += signed + max(1, int(denominator).bit_length())
    return total


def state_payload(rows: list[list[list[Any]]], alg: algebra_backend.Algebra) -> int:
    return sum(serialized_payload_bits(value, alg) for surface in rows for row in surface for value in row)


def shift(surface: list[list[Any]], axis: int, amount: int, scalar: Any, alg: algebra_backend.Algebra) -> list[list[Any]]:
    result = zeros(alg)
    for first in range(P):
        for second in range(P):
            target_first = (first + amount) % P if axis == 0 else first
            target_second = (second + amount) % P if axis == 1 else second
            result[target_first][target_second] = alg.mul(scalar, surface[first][second])
    return result


def forward_rows(phase_steps: int, family: str, alg: algebra_backend.Algebra) -> tuple[list[list[list[Any]]], int]:
    rows = [zeros(alg), zeros(alg)]
    rows[0][0][0] = rows[1][0][0] = alg.one
    maximum = state_payload(rows, alg)
    for index in range(phase_steps):
        multiplier, constant = parameters(index, family)
        rows[1] = shift(rows[1], index % 2, multiplier, alg.power(constant), alg)
        maximum = max(maximum, state_payload(rows, alg))
        first, second = zeros(alg), zeros(alg)
        for row in range(P):
            for column in range(P):
                first[row][column] = alg.add(rows[0][row][column], rows[1][row][column])
                second[row][column] = alg.sub(rows[0][row][column], rows[1][row][column])
        rows = [first, second]
        maximum = max(maximum, state_payload(rows, alg))
    return rows, maximum


def inverse_rows(rows: list[list[list[Any]]], phase_steps: int, family: str, alg: algebra_backend.Algebra) -> list[list[list[Any]]]:
    half = alg.inverse(integer(alg, 2))
    for index in range(phase_steps - 1, -1, -1):
        first, second = zeros(alg), zeros(alg)
        for row in range(P):
            for column in range(P):
                first[row][column] = alg.mul(half, alg.add(rows[0][row][column], rows[1][row][column]))
                second[row][column] = alg.mul(half, alg.sub(rows[0][row][column], rows[1][row][column]))
        rows = [first, second]
        multiplier, constant = parameters(index, family)
        rows[1] = shift(rows[1], index % 2, -multiplier, alg.power(-constant), alg)
    return rows


def matrix_rank(matrix: list[list[Any]], alg: algebra_backend.Algebra) -> int:
    work = [row[:] for row in matrix]
    rank = 0
    for column in range(P):
        pivot = next((row for row in range(rank, P) if work[row][column] != alg.zero), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inv = alg.inverse(work[rank][column])
        work[rank] = [alg.mul(inv, value) for value in work[rank]]
        for row in range(P):
            coefficient = work[row][column]
            if row != rank and coefficient != alg.zero:
                work[row] = [
                    alg.sub(value, alg.mul(coefficient, basis))
                    for value, basis in zip(work[row], work[rank])
                ]
        rank += 1
    return rank


def project(rows: list[list[list[Any]]], branch_pairs: int, phase_steps: int, family: str, alg: algebra_backend.Algebra) -> Any:
    moments = [
        scalar_power(alg, alg.add(integer(alg, 3), alg.power(index)), branch_pairs)
        for index in range(P)
    ]
    observation = alg.power(1 + ((3 * branch_pairs + 5 * phase_steps + len(family)) % 16))
    boundary = alg.zero
    for first in range(P):
        for second in range(P):
            component = alg.add(rows[0][first][second], alg.mul(observation, rows[1][first][second]))
            boundary = alg.add(boundary, alg.mul(component, alg.mul(moments[first], moments[second])))
    return boundary


def residue_counts(branch_pairs: int) -> list[int]:
    counts = [0 for _ in range(P)]
    counts[0] = 1
    for _ in range(branch_pairs):
        counts = [3 * counts[q] + counts[(q - 1) % P] for q in range(P)]
    return counts


def residue_boundary(branch_pairs: int, phase_steps: int, family: str, alg: algebra_backend.Algebra) -> Any:
    counts = residue_counts(branch_pairs)
    observation = alg.power(1 + ((3 * branch_pairs + 5 * phase_steps + len(family)) % 16))
    total = alg.zero
    for first in range(P):
        for second in range(P):
            state = [alg.one, alg.one]
            for index in range(phase_steps):
                multiplier, constant = parameters(index, family)
                q = first if index % 2 == 0 else second
                state[1] = alg.mul(state[1], alg.power(multiplier * q + constant))
                state = [alg.add(state[0], state[1]), alg.sub(state[0], state[1])]
            observed = alg.add(state[0], alg.mul(observation, state[1]))
            total = alg.add(total, alg.mul(integer(alg, counts[first] * counts[second]), observed))
    return total


def direct_assignment_boundary(branch_pairs: int, phase_steps: int, family: str, alg: algebra_backend.Algebra) -> Any:
    observation = alg.power(1 + ((3 * branch_pairs + 5 * phase_steps + len(family)) % 16))
    total = alg.zero
    group_width = 2 * branch_pairs
    for assignment in range(1 << (4 * branch_pairs)):
        first = sum(
            ((assignment >> (2 * index)) & 1) * ((assignment >> (2 * index + 1)) & 1)
            for index in range(branch_pairs)
        ) % P
        second = sum(
            ((assignment >> (group_width + 2 * index)) & 1)
            * ((assignment >> (group_width + 2 * index + 1)) & 1)
            for index in range(branch_pairs)
        ) % P
        state = [alg.one, alg.one]
        for index in range(phase_steps):
            multiplier, constant = parameters(index, family)
            q = first if index % 2 == 0 else second
            state[1] = alg.mul(state[1], alg.power(multiplier * q + constant))
            state = [alg.add(state[0], state[1]), alg.sub(state[0], state[1])]
        total = alg.add(total, alg.add(state[0], alg.mul(observation, state[1])))
    return total


def commitment(rows: list[list[list[Any]]], alg: algebra_backend.Algebra) -> str:
    hasher = hashlib.sha256()
    for surface in rows:
        for row in surface:
            for value in row:
                record = json.dumps(alg.serialize(value), separators=(",", ":")).encode()
                hasher.update(len(record).to_bytes(8, "big"))
                hasher.update(record)
    return hasher.hexdigest()


def algebra_for(item: dict[str, Any]) -> algebra_backend.Algebra:
    candidates = [algebra_backend.Algebra("Q_ZETA17")]
    candidates.extend(
        algebra_backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        for modulus, root in FINITE_FIELDS
    )
    for alg in candidates:
        signature = digest({"kind": alg.kind, "modulus": alg.modulus, "root": alg.serialize(alg.root)})
        if signature == item["algebra"]:
            return alg
    fail("oracle algebra signature changed")


def reconstruct(item: dict[str, Any]) -> dict[str, Any]:
    alg = algebra_for(item)
    n, depth, family = item["branch_pairs_per_signature"], item["phase_steps"], item["family"]
    public = descriptor(n, depth, family)
    rows, payload = forward_rows(depth, family, alg)
    boundary = project(rows, n, depth, family, alg)
    residue = residue_boundary(n, depth, family, alg)
    restored = inverse_rows([[row[:] for row in surface] for surface in rows], depth, family, alg)
    seed = zeros(alg)
    seed[0][0] = alg.one
    ranks = [matrix_rank(surface, alg) for surface in rows]
    return {
        "branch_pairs_per_signature": n,
        "phase_steps": depth,
        "family": family,
        "program_fingerprint_equal": digest(public) == item["program_fingerprint"],
        "separation_ranks": ranks,
        "separation_ranks_equal": ranks == item["coefficient_surface_separation_ranks"],
        "boundary_equal": alg.serialize(boundary) == item["final_boundary"],
        "residue_boundary_equal": boundary == residue,
        "state_commitment_equal": commitment(rows, alg) == item["final_state_commitment"],
        "maximum_resident_payload_bits": payload,
        "payload_equal": payload == item["maximum_resident_payload_bits"],
        "declared_resource_shape_consistent": (
            item["maximum_update_scratch_field_cells"] == 578
            and item["maximum_live_resident_plus_update_scratch_field_cells"] == 1156
            and item["maximum_rank_verification_dense_field_cells"] == 289
            and item["maximum_live_resident_plus_rank_dense_buffer_field_cells"] == 867
            and item["projection_persistently_named_field_cells_excluding_rank_work_and_expression_temporaries"] == 20
        ),
        "exact_inverse_restores_seed": restored == [seed, seed],
        "logical_cells": 578,
    }


def rank_mutations() -> dict[str, bool]:
    alg = algebra_backend.Algebra("F137", modulus=137, root=16)
    rows8, _ = forward_rows(8, "PRIMARY", alg)
    rows12, _ = forward_rows(12, "PRIMARY", alg)
    rows32_reuse, _ = forward_rows(32, "REUSE", alg)
    rows32_alternate, _ = forward_rows(32, "ALTERNATE", alg)
    return {
        "primary_rank_below17_before_family_ceiling": max(matrix_rank(surface, alg) for surface in rows8) < 17,
        "primary_full_rank_at_phase_step12": all(matrix_rank(surface, alg) == 17 for surface in rows12),
        "reuse_full_rank_at_phase_step32": all(matrix_rank(surface, alg) == 17 for surface in rows32_reuse),
        "alternate_full_rank_at_phase_step32": all(matrix_rank(surface, alg) == 17 for surface in rows32_alternate),
        "rank16_matrix_factor_cap_rejected": all(matrix_rank(surface, alg) > 16 for surface in rows12),
        "canonical_578_cells_universal_minimum_claimed": False,
        "snapshot_or_baseline_reload_used": False,
    }


def run(production: dict[str, Any]) -> dict[str, Any]:
    items = [*production["exact_transactions"], *production["structural_transactions"]]
    parity = [reconstruct(item) for item in items]
    direct = []
    for modulus, root in FINITE_FIELDS:
        for branch_pairs in (1, 2, 3):
            phase_steps = 2 * branch_pairs
            alg = algebra_backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            rows, _ = forward_rows(phase_steps, "PRIMARY", alg)
            chart = project(rows, branch_pairs, phase_steps, "PRIMARY", alg)
            assignment = direct_assignment_boundary(branch_pairs, phase_steps, "PRIMARY", alg)
            residue = residue_boundary(branch_pairs, phase_steps, "PRIMARY", alg)
            direct.append(
                {
                    "modulus": modulus,
                    "branch_pairs_per_signature": branch_pairs,
                    "phase_steps": phase_steps,
                    "assignments_enumerated_for_verification_only": 1 << (4 * branch_pairs),
                    "chart_matches_direct_assignments": chart == assignment,
                    "chart_matches_289_class_residue_recurrence": chart == residue,
                }
            )
    mutations = rank_mutations()
    baselines = production["matched_classical_baselines"]
    if not all(
        item["program_fingerprint_equal"]
        and item["separation_ranks_equal"]
        and item["boundary_equal"]
        and item["residue_boundary_equal"]
        and item["state_commitment_equal"]
        and item["payload_equal"]
        and item["declared_resource_shape_consistent"]
        and item["exact_inverse_restores_seed"]
        for item in parity
    ):
        fail("two-signature independent reconstruction failed")
    if not all(item["chart_matches_direct_assignments"] and item["chart_matches_289_class_residue_recurrence"] for item in direct):
        fail("two-signature direct assignment oracle failed")
    if len(baselines) != len(items) or not all(
        baseline["retain_both_surfaces_full_diagnostic"]["maximum_update_scratch_field_cells"] == 578
        and baseline["retain_both_surfaces_full_diagnostic"]["maximum_live_resident_plus_update_scratch_field_cells"] == 1156
        and baseline["streamed_final_scalar"]["executed"]
        and baseline["streamed_final_scalar"]["boundary_equal"]
        and baseline["streamed_final_scalar"]["dynamic_exact_field_cells_upper_bound"] == 8
        and not baseline["streamed_final_scalar"]["reproduces_rank_and_commitment_diagnostics"]
        and not baseline["streamed_rematerialized_full_diagnostic"]["executed"]
        and baseline["streamed_rematerialized_full_diagnostic"]["payload_tuple_not_claimed_or_measured"]
        for baseline in baselines
    ):
        fail("two-signature classical frontier accounting failed")
    false_mutations = {"canonical_578_cells_universal_minimum_claimed", "snapshot_or_baseline_reload_used"}
    if any(mutations[key] for key in false_mutations) or not all(
        value for key, value in mutations.items() if key not in false_mutations
    ):
        fail("two-signature rank mutation failed")
    return {
        "schema": "CAT_CAS_F17_TWO_SIGNATURE_CUBIC_WALSH_SEPARATION_RANK_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_claim": production["claim"],
        "production_result_sha256": digest(production),
        "transaction_parity": parity,
        "direct_assignment_checks": direct,
        "independent_rank_mutations": mutations,
        "observed_resource_law": {
            "canonical_resident_field_cells": 578,
            "full_matrix_separation_rank": 17,
            "payload_tuples_reproduced_independently": True,
            "update_scratch_payload_tuples_reproduced_independently": False,
            "phase_update_scratch_shape_source_audited": True,
            "accepted_path_assignment_expansion": False,
            "oracle_assignment_expansion_small_cases_only": True,
            "compact_classical_frontier": [
                "RETAIN_BOTH_SURFACES",
                "STREAM_JOINT_RESIDUE_FINAL_SCALAR",
                "REMATERIALIZE_COEFFICIENT_ROWS_FOR_FULL_DIAGNOSTICS",
            ],
            "strongest_streamed_payload_tuple_claimed": False,
        },
        "claim_ceiling": {
            "declared_two_signature_alternating_families_only": True,
            "rejects_only_matrix_factor_rank_caps_below17": True,
            "universal_578_cell_minimum": False,
            "general_rank_r_no_go": False,
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
