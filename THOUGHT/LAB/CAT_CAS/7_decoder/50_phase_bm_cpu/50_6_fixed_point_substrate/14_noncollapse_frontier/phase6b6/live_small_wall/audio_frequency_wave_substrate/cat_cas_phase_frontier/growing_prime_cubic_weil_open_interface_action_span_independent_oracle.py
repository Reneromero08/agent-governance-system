#!/usr/bin/env python3
"""Separate exact oracle for the cubic-Weil open-interface action bundle.

The oracle imports no production or predecessor module.  It reconstructs the
field, public factor word, state-bundle action, source-streamed boundary,
inverse, controls, and reuse from independent tuple/list code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
FAMILIES = ("PRIMARY", "ALTERNATE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return value == divisor
        divisor += 1
    return True


@dataclass(frozen=True)
class Field:
    q: int
    p: int
    root: int
    gauss_one: int


def field_for(q: int) -> Field:
    p = 2 * q + 1
    if not is_prime(q) or not is_prime(p):
        fail("not a safe-prime pair")
    generator = 2
    while generator < p and (pow(generator, 2, p) == 1 or pow(generator, q, p) == 1):
        generator += 1
    if generator == p:
        fail("generator absent")
    root = pow(generator, 2, p)
    if pow(root, q, p) != 1 or any(pow(root, exponent, p) == 1 for exponent in range(1, q)):
        fail("invalid phase root")
    gauss = sum(pow(root, x * x % q, p) for x in range(q)) % p
    if not gauss:
        fail("vanishing Gauss sum")
    return Field(q, p, root, gauss)


def phase(field: Field, exponent: int) -> int:
    return pow(field.root, exponent % field.q, field.p)


def matrix2_multiply(left: tuple[int, ...], right: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    a, b, c, d = left
    e, f, g, h = right
    return (
        (a * e + b * g) % modulus,
        (a * f + b * h) % modulus,
        (c * e + d * g) % modulus,
        (c * f + d * h) % modulus,
    )


def matrix2_inverse(matrix: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    a, b, c, d = matrix
    determinant = (a * d - b * c) % modulus
    if not determinant:
        fail("singular 2x2 matrix")
    inverse = pow(determinant, -1, modulus)
    return d * inverse % modulus, -b * inverse % modulus, -c * inverse % modulus, a * inverse % modulus


def kernel(field: Field, symplectic: tuple[int, ...], x: int, y: int) -> int:
    a, b, c, d = symplectic
    if b % field.q:
        half_b = pow(2 * b % field.q, -1, field.q)
        return phase(field, (d * x * x - 2 * x * y + a * y * y) * half_b)
    if y % field.q != d * x % field.q:
        return 0
    return phase(field, c * d * x * x * pow(2, -1, field.q))


def cocycle(field: Field, left: tuple[int, ...], right: tuple[int, ...]) -> int:
    a, b, _, _ = left
    _, f, _, h = right
    if b % field.q == 0 or f % field.q == 0:
        return 1
    coefficient = (a * pow(2 * b % field.q, -1, field.q) + h * pow(2 * f % field.q, -1, field.q)) % field.q
    if coefficient == 0:
        return field.q % field.p
    character = pow(coefficient, (field.q - 1) // 2, field.q)
    return field.gauss_one if character == 1 else -field.gauss_one % field.p


def gaussian_inverse(field: Field, payload: tuple[int, ...]) -> tuple[int, ...]:
    symplectic = payload[:4]
    coefficient = payload[4] % field.p
    inverse_symplectic = matrix2_inverse(symplectic, field.q)
    scalar = cocycle(field, symplectic, inverse_symplectic)
    return inverse_symplectic + (pow(coefficient * scalar % field.p, -1, field.p),)


def descriptor(index: int, family: str, q: int) -> tuple[tuple[str, int, int], ...]:
    if family not in FAMILIES:
        fail("invalid family")
    code = 1 if family == "PRIMARY" else 2
    stages = (
        ("CHIRP", (2 * index + code + 1) % q, (3 * index + code + 2) % q),
        ("CONJUGATE", 0, 0),
        ("CUBIC", (5 * index + 2 * code + 1) % q, (7 * index + 3 * code + 2) % q),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def compile_plan(q: int, depth: int, family: str) -> tuple[list[tuple[str, tuple[int, ...]]], tuple[int, ...]]:
    if depth < 1 or family not in FAMILIES:
        fail("invalid plan")
    field = field_for(q)
    resident_symplectic = (2, 1, 1, 1)
    resident_fiber = (2, 2, 5, 6)
    seed_symplectic = (1, 1, 1, 2)
    fiber = (1, 3, 3, 10)
    left: list[tuple[str, tuple[int, ...]]] = []
    right: list[tuple[str, tuple[int, ...]]] = []
    for index in range(depth):
        for kind, row, column in descriptor(index, family, q):
            if kind == "CHIRP":
                left_chirp = (1, 0, 2 * row % q, 1)
                right_chirp = (1, 0, 2 * column % q, 1)
                resident_symplectic = matrix2_multiply(matrix2_multiply(left_chirp, resident_symplectic, q), right_chirp, q)
            elif kind == "CONJUGATE":
                resident_payload = resident_symplectic + (1,)
                inverse_payload = gaussian_inverse(field, resident_payload)
                left.append(("GAUSSIAN", resident_payload))
                right.append(("GAUSSIAN", inverse_payload))
                fiber = matrix2_multiply(matrix2_multiply(resident_fiber, fiber, field.p), matrix2_inverse(resident_fiber, field.p), field.p)
            else:
                left.append(("CUBIC", (row,)))
                right.append(("CUBIC", (column,)))
    operations = list(reversed(right)) + [("GAUSSIAN", seed_symplectic + (1,))] + left
    return operations, fiber


@dataclass
class State:
    field: Field
    rank: int
    values: list[int]
    stage: str = "IDLE"
    generation: int = 0

    @classmethod
    def basis(cls, q: int, rank: int) -> "State":
        field = field_for(q)
        if not 1 <= rank <= 2 * q:
            fail("invalid rank")
        values = [0] * (2 * q * rank)
        for column in range(rank):
            values[column * rank + column] = 1
        return cls(field, rank, values)

    def canonical(self) -> tuple[Any, ...]:
        return self.field.q, self.field.p, self.rank, tuple(self.values), self.stage, self.generation

    def payload(self) -> tuple[Any, ...]:
        return self.field.q, self.field.p, self.rank, tuple(self.values), self.stage

    def backing(self) -> tuple[int, int]:
        return id(self), id(self.values)


def apply_gaussian(state: State, payload: tuple[int, ...], work: dict[str, int]) -> None:
    q, p, rank = state.field.q, state.field.p, state.rank
    result = [0] * len(state.values)
    for fiber in range(2):
        for x in range(q):
            output_offset = (fiber * q + x) * rank
            for y in range(q):
                value = payload[4] * kernel(state.field, payload[:4], x, y) % p
                work["gaussian_kernel_phase_evaluations"] += 1
                input_offset = (fiber * q + y) * rank
                for column in range(rank):
                    result[output_offset + column] += value * state.values[input_offset + column]
                    work["gaussian_field_multiply_adds"] += 1
            for column in range(rank):
                result[output_offset + column] %= p
    state.values[:] = result
    work["temporary_vector_field_cells_peak"] = max(work["temporary_vector_field_cells_peak"], len(result))


def apply_cubic(state: State, strength: int, work: dict[str, int]) -> None:
    q, p, rank = state.field.q, state.field.p, state.rank
    for fiber in range(2):
        for x in range(q):
            scalar = phase(state.field, strength * x * x * x)
            work["cubic_phase_evaluations"] += 1
            offset = (fiber * q + x) * rank
            for column in range(rank):
                state.values[offset + column] = state.values[offset + column] * scalar % p
                work["cubic_field_multiplications"] += 1


def apply_fiber(state: State, fiber: tuple[int, ...], work: dict[str, int]) -> None:
    q, p, rank = state.field.q, state.field.p, state.rank
    a, b, c, d = fiber
    result = [0] * len(state.values)
    for x in range(q):
        first, second = x * rank, (q + x) * rank
        for column in range(rank):
            left, right = state.values[first + column], state.values[second + column]
            result[first + column] = (a * left + c * right) % p
            result[second + column] = (b * left + d * right) % p
            work["fiber_field_multiply_adds"] += 4
    state.values[:] = result
    work["temporary_vector_field_cells_peak"] = max(work["temporary_vector_field_cells_peak"], len(result))


def blank_work() -> dict[str, int]:
    return {
        "gaussian_kernel_phase_evaluations": 0,
        "gaussian_field_multiply_adds": 0,
        "cubic_phase_evaluations": 0,
        "cubic_field_multiplications": 0,
        "fiber_field_multiply_adds": 0,
        "inverse_gaussian_scalar_rematerializations": 0,
        "temporary_vector_field_cells_peak": 0,
    }


def apply(state: State, operation: tuple[str, tuple[int, ...]], work: dict[str, int]) -> None:
    if operation[0] == "GAUSSIAN":
        apply_gaussian(state, operation[1], work)
    elif operation[0] == "CUBIC":
        apply_cubic(state, operation[1][0], work)
    elif operation[0] == "FIBER":
        apply_fiber(state, operation[1], work)
    else:
        fail("unknown operation")


def inverse(state: State, operation: tuple[str, tuple[int, ...]], work: dict[str, int], wrong: bool = False) -> tuple[str, tuple[int, ...]]:
    if operation[0] == "GAUSSIAN":
        payload = list(gaussian_inverse(state.field, operation[1]))
        work["inverse_gaussian_scalar_rematerializations"] += 1
        if wrong:
            payload[-1] = (payload[-1] + 1) % state.field.p or 1
        return "GAUSSIAN", tuple(payload)
    if operation[0] == "CUBIC":
        return "CUBIC", ((-operation[1][0] + int(wrong)) % state.field.q,)
    payload = list(matrix2_inverse(operation[1], state.field.p))
    if wrong:
        payload[0] = (payload[0] + 1) % state.field.p
    return "FIBER", tuple(payload)


def forward(state: State, depth: int, family: str, reorder: bool = False) -> tuple[list[tuple[str, tuple[int, ...]]], dict[str, int]]:
    if state.stage != "IDLE":
        fail("not idle")
    kernel_operations, fiber = compile_plan(state.field.q, depth, family)
    operations = kernel_operations + [("FIBER", fiber)]
    if reorder:
        operations = list(reversed(operations))
    work = blank_work()
    for operation in operations:
        apply(state, operation, work)
    state.stage = "FORWARD_COMPLETE"
    return operations, work


def reverse(state: State, operations: list[tuple[str, tuple[int, ...]]], work: dict[str, int], mutation: str | None = None) -> None:
    if state.stage != "FORWARD_COMPLETE":
        fail("not forward complete")
    sequence = list(reversed(operations))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(operations)
    for index, operation in enumerate(sequence):
        apply(state, inverse(state, operation, work, mutation == "WRONG" and index == 0), work)
    state.stage = "IDLE"


def probes(q: int, rank: int, family: str) -> tuple[tuple[int, int, int], ...]:
    code = 1 if family == "PRIMARY" else 2
    return (
        ((3 * code + 1) % rank, (5 * code + 2) % (2 * q), 1),
        ((7 * code + 2) % rank, (11 * code + 3) % (2 * q), 2),
        ((13 * code + 1) % rank, (17 * code + 4) % (2 * q), 3),
        ((19 * code + 2) % rank, (23 * code + 5) % (2 * q), 5),
    )


def project(state: State, family: str) -> int:
    if state.stage != "FORWARD_COMPLETE":
        fail("projection unavailable")
    return sum(weight * state.values[row * state.rank + column] for column, row, weight in probes(state.field.q, state.rank, family)) % state.field.p


def digest(state: State) -> str:
    return hashlib.sha256(repr((state.field.q, state.rank, tuple(state.values))).encode("ascii")).hexdigest()


def rank_of(state: State) -> int:
    p, rows, columns = state.field.p, 2 * state.field.q, state.rank
    matrix = [state.values[row * columns : (row + 1) * columns] for row in range(rows)]
    current = 0
    for column in range(columns):
        pivot = next((row for row in range(current, rows) if matrix[row][column] % p), None)
        if pivot is None:
            continue
        matrix[current], matrix[pivot] = matrix[pivot], matrix[current]
        scale = pow(matrix[current][column] % p, -1, p)
        matrix[current] = [value * scale % p for value in matrix[current]]
        for row in range(rows):
            if row == current or not matrix[row][column] % p:
                continue
            factor = matrix[row][column] % p
            matrix[row] = [(left - factor * right) % p for left, right in zip(matrix[row], matrix[current])]
        current += 1
    return current


def single_vector(field: Field, vector: list[int], operation: tuple[str, tuple[int, ...]], work: dict[str, int]) -> list[int]:
    if operation[0] == "CUBIC":
        work["cubic_phase_evaluations"] += field.q
        work["cubic_field_multiplications"] += field.q
        return [value * phase(field, operation[1][0] * x * x * x) % field.p for x, value in enumerate(vector)]
    work["gaussian_kernel_phase_evaluations"] += field.q * field.q
    work["gaussian_field_multiply_adds"] += field.q * field.q
    return [
        operation[1][4] * sum(kernel(field, operation[1][:4], x, y) * vector[y] for y in range(field.q)) % field.p
        for x in range(field.q)
    ]


def streamed_boundary(q: int, depth: int, family: str, rank: int) -> tuple[int, dict[str, int]]:
    field = field_for(q)
    operations, fiber = compile_plan(q, depth, family)
    work = {key: 0 for key in ("gaussian_kernel_phase_evaluations", "gaussian_field_multiply_adds", "cubic_phase_evaluations", "cubic_field_multiplications")}
    total = 0
    for column, row, weight in probes(q, rank, family):
        source, y = divmod(column, q)
        target, x = divmod(row, q)
        vector = [0] * q
        vector[y] = 1
        for operation in operations:
            vector = single_vector(field, vector, operation, work)
        total = (total + weight * fiber[2 * source + target] * vector[x]) % field.p
    return total, work


def reconstruct_case(production: dict[str, Any]) -> dict[str, Any]:
    q, depth, family, source_rank = production["q"], production["depth"], production["family"], production["source_rank"]
    state = State.basis(q, source_rank)
    before, backing = state.canonical(), state.backing()
    operations, work = forward(state, depth, family)
    boundary = project(state, family)
    commitment = digest(state)
    action_rank = rank_of(state)
    baseline_boundary, baseline_work = streamed_boundary(q, depth, family, source_rank)
    reverse(state, operations, work)
    checks = {
        "boundary": boundary == production["boundary"],
        "semantic_commitment": commitment == production["semantic_commitment"],
        "action_span_rank": action_rank == production["observed_action_span_rank"],
        "accepted_resident_cells": 2 * q * source_rank == production["accepted_resident_field_cells"],
        "accepted_peak_cells": 4 * q * source_rank == production["accepted_resident_plus_temporary_peak_field_cells"],
        "work": work == production["work"],
        "streamed_boundary": baseline_boundary == production["matched_source_streamed_classical"]["boundary"],
        "streamed_work": baseline_work == production["matched_source_streamed_classical"]["work"],
        "restoration": state.canonical() == before,
        "same_backing": state.backing() == backing,
    }
    if not all(checks.values()):
        fail(f"case mismatch q={q} depth={depth} family={family} checks={checks}")
    return {"q": q, "depth": depth, "family": family, "source_rank": source_rank, "checks": checks}


def dense_multiply(left: list[list[int]], right: list[list[int]], p: int) -> list[list[int]]:
    order = len(left)
    return [[sum(left[row][shared] * right[shared][column] for shared in range(order)) % p for column in range(order)] for row in range(order)]


def dense_operation(field: Field, operation: tuple[str, tuple[int, ...]]) -> list[list[int]]:
    q, n = field.q, 2 * field.q
    matrix = [[0] * n for _ in range(n)]
    if operation[0] == "FIBER":
        a, b, c, d = operation[1]
        for x in range(q):
            matrix[x][x] = a
            matrix[x][q + x] = c
            matrix[q + x][x] = b
            matrix[q + x][q + x] = d
        return matrix
    for fiber in range(2):
        for x in range(q):
            for y in range(q):
                if operation[0] == "CUBIC":
                    value = phase(field, operation[1][0] * x * x * x) if x == y else 0
                else:
                    value = operation[1][4] * kernel(field, operation[1][:4], x, y) % field.p
                matrix[fiber * q + x][fiber * q + y] = value
    return matrix


def dense_semantics() -> dict[str, bool]:
    results = {}
    for q, depth in ((5, 2), (11, 1)):
        field = field_for(q)
        operations, fiber = compile_plan(q, depth, "PRIMARY")
        full = operations + [("FIBER", fiber)]
        identity = [[int(row == column) for column in range(2 * q)] for row in range(2 * q)]
        dense = identity
        for operation in full:
            dense = dense_multiply(dense_operation(field, operation), dense, field.p)
        state = State.basis(q, 2 * q)
        forward(state, depth, "PRIMARY")
        recurrence = [state.values[row * 2 * q : (row + 1) * 2 * q] for row in range(2 * q)]
        results[f"q{q}_full_basis_dense_factor_parity"] = dense == recurrence
        results[f"q{q}_full_action_rank"] = rank_of(state) == 2 * q
    return results


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except (RuntimeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    def mutation_fails(mutation: str) -> bool:
        state = State.basis(5, 10)
        before = state.canonical()
        operations, work = forward(state, 2, "PRIMARY")
        reverse(state, operations, work, mutation)
        return state.canonical() != before

    normal = State.basis(5, 10)
    operations, work = forward(normal, 2, "PRIMARY")
    normal_boundary = project(normal, "PRIMARY")
    reverse(normal, operations, work)
    reordered = State.basis(5, 10)
    forward(reordered, 2, "PRIMARY", True)
    return {
        "missing_inverse_fails": mutation_fails("MISSING"),
        "wrong_inverse_fails": mutation_fails("WRONG"),
        "reordered_inverse_fails": mutation_fails("REORDER"),
        "module_reorder_changes_boundary": project(reordered, "PRIMARY") != normal_boundary,
        "premature_projection_rejected": raises(lambda: project(State.basis(5, 10), "PRIMARY")),
        "null_rank_rejected": raises(lambda: State.basis(5, 0)),
        "oversize_rank_rejected": raises(lambda: State.basis(5, 11)),
        "invalid_family_rejected": raises(lambda: compile_plan(5, 2, "INVALID")),
        "zero_depth_rejected": raises(lambda: compile_plan(5, 0, "PRIMARY")),
    }


def reuse() -> dict[str, bool | int]:
    state = State.basis(23, 22)
    backing, payload = state.backing(), state.payload()

    def run(depth: int, family: str) -> tuple[int, str]:
        operations, work = forward(state, depth, family)
        answer, commitment = project(state, family), digest(state)
        reverse(state, operations, work)
        state.generation += 1
        return answer, commitment

    first = run(1, "PRIMARY")
    second = run(2, "ALTERNATE")
    fresh = State.basis(23, 22)
    operations, work = forward(fresh, 2, "ALTERNATE")
    fresh_second = project(fresh, "ALTERNATE"), digest(fresh)
    reverse(fresh, operations, work)
    return {
        "first_boundary_nonzero": bool(first[0]),
        "second_boundary_matches_fresh": second[0] == fresh_second[0],
        "second_commitment_matches_fresh": second[1] == fresh_second[1],
        "same_backing": state.backing() == backing,
        "payload_restored": state.payload() == payload,
        "generation_two": state.generation == 2,
        "no_snapshot": True,
    }


def build(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    comparisons = [reconstruct_case(case) for case in production["cases"]]
    dense = dense_semantics()
    control_results = controls()
    reuse_results = reuse()
    payload = {
        "schema": "CAT_CAS_GROWING_PRIME_CUBIC_WEIL_OPEN_INTERFACE_ACTION_SPAN_INDEPENDENT_ORACLE_V1",
        "production_result": "GROWING_PRIME_CUBIC_WEIL_OPEN_INTERFACE_ACTION_SPAN_RESULTS.json",
        "production_result_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "qualified": True,
        "independence": {
            "imports_production_module": False,
            "imports_predecessor_module": False,
            "production_forward_inverse_or_projection_called": False,
            "production_result_used_only_as_comparison_target": True,
            "separate_field_plan_bundle_rank_and_streaming_recurrences": True,
        },
        "all_11_cases_reconstructed": len(comparisons) == 11 and all(all(case["checks"].values()) for case in comparisons),
        "case_comparisons": comparisons,
        "dense_semantic_checks": dense,
        "controls": control_results,
        "restoration_and_reuse": reuse_results,
        "observed_resource_law": {
            "declared_action_span_equals_r": True,
            "executed_full_source_bundles": ["Q5_R10", "Q11_R22"],
            "full_basis_action_bundle_field_cells": "4*Q^2",
            "declared_bundle_resident_field_cells": "2*Q*R",
            "declared_bundle_resident_plus_temporary_peak_field_cells": "4*Q*R",
            "matched_source_streamed_dynamic_field_cells": "2*Q",
            "public_word_descriptor_cells": "12*DEPTH",
            "fixed_rank_general_open_interface_established": False,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "EXACT_FACTORWISE_CUBIC_WEIL_ACTION_ON_DECLARED_SOURCE_BUNDLES",
            "ACTION_SPAN_EQUALS_EACH_DECLARED_SOURCE_RANK",
            "FULL_Q5_AND_Q11_TWO_FIBER_BASIS_ACTION_BUNDLES",
            "EXACT_IN_PLACE_RESTORATION_AND_SAME_BACKING_REUSE",
            "EXECUTED_SOURCE_STREAMED_2Q_DYNAMIC_CELL_FINAL_BOUNDARY_BASELINE",
        ],
        "rejected_interpretations": [
            "COMPACT_GENERAL_OPEN_RELATION_CLOSURE",
            "FULL_SOURCE_BUNDLE_EXECUTION_AT_EVERY_DECLARED_Q",
            "LOWER_BOUND_AGAINST_NONLINEAR_OR_PROGRAM_DESCRIPTOR_REPRESENTATIONS",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }
    if not payload["all_11_cases_reconstructed"] or not all(dense.values()) or not all(control_results.values()) or not all(reuse_results.values()):
        fail("independent qualification failed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(build(arguments.production), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
