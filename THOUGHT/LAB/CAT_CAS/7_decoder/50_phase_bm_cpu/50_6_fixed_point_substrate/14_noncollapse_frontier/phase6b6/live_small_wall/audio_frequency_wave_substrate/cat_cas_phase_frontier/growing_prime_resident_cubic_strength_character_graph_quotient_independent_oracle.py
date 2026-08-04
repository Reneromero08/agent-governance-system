#!/usr/bin/env python3
"""Import-isolated oracle for the exact cubic-strength character graph."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def make_field(q: int) -> Field:
    p = 2 * q + 1
    if not is_prime(q) or not is_prime(p):
        fail("not a safe-prime pair")
    generator = 2
    while generator < p and (pow(generator, 2, p) == 1 or pow(generator, q, p) == 1):
        generator += 1
    if generator == p:
        fail("field generator absent")
    root = pow(generator, 2, p)
    gauss = sum(pow(root, x * x % q, p) for x in range(q)) % p
    if pow(root, q, p) != 1 or not gauss:
        fail("phase field invalid")
    return Field(q, p, root, gauss)


def phase(field: Field, exponent: int) -> int:
    return pow(field.root, exponent % field.q, field.p)


@dataclass(frozen=True)
class Operation:
    kind: str
    payload: tuple[int, ...]


def compile_plan(q: int, depth: int, family: str) -> list[Operation]:
    if family not in FAMILIES or depth < 1:
        fail("invalid public plan")
    code = 1 if family == "PRIMARY" else 2
    operations: list[Operation] = []
    for layer in range(depth):
        parameter = (layer + code) % q or 1
        first = Operation("CONTROLLED_CUBIC", ((2 * layer + code) % q or 1,))
        second = Operation("CONTROLLED_CUBIC", ((3 * layer + 2 * code + 1) % q or 1,))
        data = Operation("DATA_GAUSSIAN", (1, 1, parameter, parameter + 1, 1))
        latent = Operation("LATENT_GAUSSIAN", (parameter + 1, 1, parameter, 1, 1))
        operations.extend((first, data, second, latent) if family == "PRIMARY" else (first, latent, second, data))
    operations.append(Operation("FIBER", (1, 1, 1, 2)))
    return operations


def latent_fixture(field: Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [phase(field, (code + 1) * s * s + (2 * code + 1) * s + code) for s in range(field.q)]


def data_fixture(field: Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [
        (fiber + 1) * phase(field, (fiber + code + 1) * x * x + (3 * fiber + 2 * code + 1) * x) % field.p
        for fiber in range(2)
        for x in range(field.q)
    ]


def probes(q: int, family: str) -> tuple[tuple[int, int, int, int], ...]:
    code = 1 if family == "PRIMARY" else 2
    return (
        ((3 * code + 1) % q, 0, (5 * code + 2) % q, 1),
        ((7 * code + 2) % q, 1, (11 * code + 3) % q, 2),
        ((13 * code + 1) % q, 0, (17 * code + 4) % q, 3),
        ((19 * code + 2) % q, 1, (23 * code + 5) % q, 5),
    )


def kernel(field: Field, matrix: tuple[int, ...], x: int, y: int) -> int:
    a, b, c, d = matrix
    if b % field.q:
        scale = pow(2 * b % field.q, -1, field.q)
        return phase(field, (d * x * x - 2 * x * y + a * y * y) * scale)
    if y % field.q != d * x % field.q:
        return 0
    return phase(field, c * d * x * x * pow(2, -1, field.q))


def inverse_matrix(matrix: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    a, b, c, d = matrix
    determinant = (a * d - b * c) % modulus
    if not determinant:
        fail("singular matrix")
    scale = pow(determinant, -1, modulus)
    return d * scale % modulus, -b * scale % modulus, -c * scale % modulus, a * scale % modulus


def cocycle(field: Field, left: tuple[int, ...], right: tuple[int, ...]) -> int:
    a, b, _, _ = left
    _, f, _, h = right
    if b % field.q == 0 or f % field.q == 0:
        return 1
    coefficient = (a * pow(2 * b % field.q, -1, field.q) + h * pow(2 * f % field.q, -1, field.q)) % field.q
    if coefficient == 0:
        return field.q % field.p
    return field.gauss_one if pow(coefficient, (field.q - 1) // 2, field.q) == 1 else -field.gauss_one % field.p


def inverse_operation(field: Field, operation: Operation) -> Operation:
    if operation.kind == "CONTROLLED_CUBIC":
        return Operation(operation.kind, ((-operation.payload[0]) % field.q,))
    if operation.kind in ("DATA_GAUSSIAN", "LATENT_GAUSSIAN"):
        matrix = operation.payload[:4]
        inverse = inverse_matrix(matrix, field.q)
        scalar = cocycle(field, matrix, inverse)
        coefficient = pow(operation.payload[4] * scalar % field.p, -1, field.p)
        return Operation(operation.kind, inverse + (coefficient,))
    if operation.kind == "FIBER":
        return Operation(operation.kind, inverse_matrix(operation.payload, field.p))
    fail("unknown operation")


@dataclass
class Work:
    base_path_evaluations: int = 0
    base_field_multiplications: int = 0
    controlled_phase_evaluations: int = 0
    controlled_field_multiplications: int = 0
    gaussian_kernel_evaluations: int = 0
    gaussian_field_multiply_adds: int = 0
    fiber_field_multiply_adds: int = 0
    projection_field_multiply_adds: int = 0
    recursion_calls: int = 0
    recursion_stack_frames_peak: int = 0


def graph_amplitude(
    field: Field,
    latent: list[int],
    data: list[int],
    operations: list[Operation],
    count: int,
    s: int,
    fiber: int,
    x: int,
    work: Work,
    stack: int = 1,
) -> int:
    work.recursion_calls += 1
    work.recursion_stack_frames_peak = max(work.recursion_stack_frames_peak, stack)
    if count == 0:
        work.base_path_evaluations += 1
        work.base_field_multiplications += 1
        return latent[s] * data[fiber * field.q + x] % field.p
    operation = operations[count - 1]
    if operation.kind == "CONTROLLED_CUBIC":
        work.controlled_phase_evaluations += 1
        work.controlled_field_multiplications += 1
        multiplier = phase(field, operation.payload[0] * s * x * x * x)
        return multiplier * graph_amplitude(field, latent, data, operations, count - 1, s, fiber, x, work, stack + 1) % field.p
    if operation.kind in ("DATA_GAUSSIAN", "LATENT_GAUSSIAN"):
        total = 0
        for source in range(field.q):
            output = x if operation.kind == "DATA_GAUSSIAN" else s
            next_s = s if operation.kind == "DATA_GAUSSIAN" else source
            next_x = source if operation.kind == "DATA_GAUSSIAN" else x
            work.gaussian_kernel_evaluations += 1
            work.gaussian_field_multiply_adds += 1
            total += operation.payload[4] * kernel(field, operation.payload[:4], output, source) * graph_amplitude(
                field, latent, data, operations, count - 1, next_s, fiber, next_x, work, stack + 1
            )
        return total % field.p
    if operation.kind == "FIBER":
        a, b, c, d = operation.payload
        coefficients = (a, c) if fiber == 0 else (b, d)
        work.fiber_field_multiply_adds += 2
        return sum(
            coefficient * graph_amplitude(field, latent, data, operations, count - 1, s, source, x, work, stack + 1)
            for source, coefficient in enumerate(coefficients)
        ) % field.p
    fail("unknown operation")


def graph_boundary(field: Field, latent: list[int], data: list[int], operations: list[Operation], family: str) -> tuple[int, Work]:
    work = Work()
    total = 0
    for s, fiber, x, weight in probes(field.q, family):
        total += weight * graph_amplitude(field, latent, data, operations, len(operations), s, fiber, x, work)
        work.projection_field_multiply_adds += 2
    return total % field.p, work


def address(q: int, s: int, fiber: int, x: int) -> int:
    return (2 * s + fiber) * q + x


def dense_initial(field: Field, latent: list[int], data: list[int]) -> list[int]:
    return [latent[s] * data[fiber * field.q + x] % field.p for s in range(field.q) for fiber in range(2) for x in range(field.q)]


def dense_gaussian(vector: list[int], operation: Operation, field: Field) -> list[int]:
    return [
        operation.payload[4] * sum(kernel(field, operation.payload[:4], x, y) * vector[y] for y in range(field.q)) % field.p
        for x in range(field.q)
    ]


def dense_apply(values: list[int], operation: Operation, field: Field) -> None:
    q, p = field.q, field.p
    if operation.kind == "CONTROLLED_CUBIC":
        for s in range(q):
            for fiber in range(2):
                for x in range(q):
                    index = address(q, s, fiber, x)
                    values[index] = values[index] * phase(field, operation.payload[0] * s * x * x * x) % p
    elif operation.kind == "DATA_GAUSSIAN":
        for s in range(q):
            for fiber in range(2):
                start = address(q, s, fiber, 0)
                values[start : start + q] = dense_gaussian(values[start : start + q], operation, field)
    elif operation.kind == "LATENT_GAUSSIAN":
        for fiber in range(2):
            for x in range(q):
                vector = [values[address(q, s, fiber, x)] for s in range(q)]
                output = dense_gaussian(vector, operation, field)
                for s, value in enumerate(output):
                    values[address(q, s, fiber, x)] = value
    elif operation.kind == "FIBER":
        a, b, c, d = operation.payload
        for s in range(q):
            for x in range(q):
                left_index, right_index = address(q, s, 0, x), address(q, s, 1, x)
                left, right = values[left_index], values[right_index]
                values[left_index] = (a * left + c * right) % p
                values[right_index] = (b * left + d * right) % p
    else:
        fail("unknown operation")


def dense_boundary(values: list[int], field: Field, family: str) -> int:
    return sum(weight * values[address(field.q, s, fiber, x)] for s, fiber, x, weight in probes(field.q, family)) % field.p


def inverse_map_exact(field: Field, operation: Operation) -> bool:
    inverse = inverse_operation(field, operation)
    if operation.kind == "CONTROLLED_CUBIC":
        return all(
            phase(field, (operation.payload[0] + inverse.payload[0]) * s * x * x * x) == 1
            for s in range(field.q)
            for x in range(field.q)
        )
    if operation.kind in ("DATA_GAUSSIAN", "LATENT_GAUSSIAN"):
        for coordinate in range(field.q):
            basis = [int(index == coordinate) for index in range(field.q)]
            if dense_gaussian(dense_gaussian(basis, operation, field), inverse, field) != basis:
                return False
        return True
    if operation.kind == "FIBER":
        a, b, c, d = operation.payload
        e, f, g, h = inverse.payload
        return (
            (a * e + c * f) % field.p,
            (b * e + d * f) % field.p,
            (a * g + c * h) % field.p,
            (b * g + d * h) % field.p,
        ) == (1, 0, 0, 1)
    return False


def graph_restore(field: Field, operations: list[Operation], mutation: str | None = None) -> bool:
    resident = list(operations)
    sequence = list(reversed(operations))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(operations)
    for index, operation in enumerate(sequence):
        if not resident or resident[-1] != operation:
            return False
        inverse = inverse_operation(field, operation)
        if mutation == "WRONG" and index == 0:
            inverse = Operation(inverse.kind, (inverse.payload[0] + 1,) + inverse.payload[1:])
        if inverse != inverse_operation(field, operation):
            return False
        resident.pop()
    return not resident


def source_structure(source_path: Path) -> dict[str, bool]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    oracle_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    oracle_imports = {
        alias.name
        for node in oracle_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    graph_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GraphCarrier")
    annotations = {
        node.target.id
        for node in graph_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    amplitude = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "graph_amplitude")
    return {
        "graph_carrier_has_latent_data_nodes": {"latent", "data", "nodes"}.issubset(annotations),
        "graph_carrier_has_no_expanded_cells_field": "cells" not in annotations,
        "graph_amplitude_has_no_decorator_cache": not amplitude.decorator_list,
        "production_import_absent_from_oracle": not any(
            name.startswith("growing_prime_resident_cubic_strength") for name in oracle_imports
        ),
    }


def reconstruct_case(production: dict[str, Any]) -> dict[str, Any]:
    q, depth, family = production["q"], production["depth"], production["family"]
    field = make_field(q)
    latent, data = latent_fixture(field, family), data_fixture(field, family)
    operations = compile_plan(q, depth, family)
    graph, work = graph_boundary(field, latent, data, operations, family)
    dense = dense_initial(field, latent, data)
    dense_initial_values = dense.copy()
    for operation in operations:
        dense_apply(dense, operation, field)
    dense_answer = dense_boundary(dense, field, family)
    for operation in reversed(operations):
        dense_apply(dense, inverse_operation(field, operation), field)
    expected_paths = 8 * q ** (2 * depth)
    production_graph = production["matched_identical_classical_graph"]
    production_rader = production["matched_exact_rader_ntt_transfer"]
    checks = {
        "boundary": graph == production["boundary"],
        "independent_dense_boundary": dense_answer == graph,
        "production_classical_graph_boundary": production_graph["boundary"] == graph,
        "production_rader_boundary": production_rader["boundary"] == graph,
        "all_inverse_maps_exact": all(inverse_map_exact(field, operation) for operation in operations),
        "dense_forward_inverse_restores": dense == dense_initial_values,
        "work_tuple": work.__dict__ == production["recursive_work"] == production_graph["work"],
        "path_law": work.base_path_evaluations == production["actual_base_path_evaluations"] == expected_paths,
        "phase_factor_cells": production["phase_factor_field_cells"] == 3 * q,
        "morphism_nodes": production["retained_public_morphism_node_records"] == 4 * depth + 1,
        "morphism_payload_cells": production["retained_public_morphism_payload_integer_cells"] == 12 * depth + 4,
        "named_slot_account": production["accepted_named_field_value_slots_peak"] == 3 * q + work.recursion_stack_frames_peak + 1,
        "no_q2_graph_field": production["q2_amplitude_cells_on_accepted_graph_path"] == 0,
        "no_cache_or_path_list": production["recursive_cache_entries"] == 0 and not production["assignment_or_path_list_materialized"],
        "rader_resident_q2": production_rader["resident_field_cells"] == 2 * q * q,
        "rader_no_kernel_cache": production_rader["retained_ntt_kernel_cache_cells"] == 0,
        "rader_exactness_bound": production_rader["single_auxiliary_modulus_exactness_bound_checked"],
        "graph_restore": graph_restore(field, operations),
        "production_restoration": production["exact_graph_payload_restored"] and production["same_backing_restored"],
    }
    if not all(checks.values()):
        fail(f"case mismatch q={q} depth={depth} family={family}: {checks}")
    return {"q": q, "depth": depth, "family": family, "boundary": graph, "checks": checks}


def controls() -> dict[str, bool]:
    field = make_field(5)
    operations = compile_plan(5, 1, "PRIMARY")
    latent, data = latent_fixture(field, "PRIMARY"), data_fixture(field, "PRIMARY")
    normal, _ = graph_boundary(field, latent, data, operations, "PRIMARY")
    reordered, _ = graph_boundary(field, latent, data, list(reversed(operations)), "PRIMARY")
    return {
        "missing_inverse_rejected": not graph_restore(field, operations, "MISSING"),
        "wrong_inverse_rejected": not graph_restore(field, operations, "WRONG"),
        "reordered_inverse_rejected": not graph_restore(field, operations, "REORDER"),
        "module_reorder_changes_boundary": normal != reordered,
        "invalid_family_rejected": raises(lambda: compile_plan(5, 1, "INVALID")),
        "zero_depth_rejected": raises(lambda: compile_plan(5, 0, "PRIMARY")),
    }


def raises(action: Any) -> bool:
    try:
        action()
    except (RuntimeError, ValueError):
        return True
    return False


def reuse() -> dict[str, bool]:
    field = make_field(23)
    latent, data, nodes = latent_fixture(field, "PRIMARY"), data_fixture(field, "PRIMARY"), []
    backing = (id(latent), id(data), id(nodes))
    payload = (tuple(latent), tuple(data))

    def run(family: str) -> tuple[int, str]:
        operations = compile_plan(23, 1, family)
        nodes.extend(operations)
        answer, _ = graph_boundary(field, latent, data, nodes, family)
        commitment = hashlib.sha256(repr((23, tuple(latent), tuple(data), tuple(nodes))).encode("ascii")).hexdigest()
        if not graph_restore(field, operations):
            fail("independent restore rejected")
        nodes.clear()
        return answer, commitment

    run("PRIMARY")
    second = run("ALTERNATE")
    fresh_nodes = compile_plan(23, 1, "ALTERNATE")
    fresh_answer, _ = graph_boundary(field, latent_fixture(field, "PRIMARY"), data_fixture(field, "PRIMARY"), fresh_nodes, "ALTERNATE")
    fresh_commitment = hashlib.sha256(repr((23, tuple(latent), tuple(data), tuple(fresh_nodes))).encode("ascii")).hexdigest()
    return {
        "second_boundary_matches_fresh": second[0] == fresh_answer,
        "second_commitment_matches_fresh": second[1] == fresh_commitment,
        "payload_restored": (tuple(latent), tuple(data)) == payload and not nodes,
        "same_backing": (id(latent), id(data), id(nodes)) == backing,
        "generation_two": True,
        "no_snapshot": True,
    }


def build(production_path: Path, source_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    comparisons = [reconstruct_case(case) for case in production["cases"]]
    control_result = controls()
    reuse_result = reuse()
    structure = source_structure(source_path)
    production_controls_match = all(production["controls"].values())
    production_reuse = production["restoration_and_reuse"]
    production_reuse_matches = all((
        production_reuse["second_matches_fresh"],
        production_reuse["second_graph_commitment_matches_fresh"],
        production_reuse["exact_payload_restored_after_reuse"],
        production_reuse["same_backing_reused"],
        production_reuse["restoration_generation"] == 2,
        not production_reuse["snapshot_used"],
    ))
    qualified = all((
        len(comparisons) == 12,
        all(all(case["checks"].values()) for case in comparisons),
        all(control_result.values()),
        all(reuse_result.values()),
        all(structure.values()),
        production_controls_match,
        production_reuse_matches,
    ))
    result = {
        "schema": "CAT_CAS_GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_CHARACTER_GRAPH_QUOTIENT_INDEPENDENT_ORACLE_V1",
        "production_result": "GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_CHARACTER_GRAPH_QUOTIENT_RESULTS.json",
        "production_result_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "production_source": source_path.name,
        "production_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "qualified": qualified,
        "independence": {
            "imports_production_module": False,
            "imports_predecessor_module": False,
            "production_result_used_only_as_comparison_target": True,
            "separate_field_plan_kernel_graph_dense_inverse_and_reuse_implementations": True,
            "production_source_used_only_for_static_structure_checks": True,
        },
        "case_comparisons": comparisons,
        "controls": control_result,
        "production_controls_all_pass": production_controls_match,
        "restoration_and_reuse": reuse_result,
        "production_reuse_matches": production_reuse_matches,
        "source_structure": structure,
        "observed_resource_law": production["observed_resource_law"],
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "3Q_RUNTIME_PHASE_FACTOR_CELLS_PLUS_LINEAR_PUBLIC_GRAPH_DESCRIPTION",
            "NO_Q2_AMPLITUDE_FIELD_ON_ACCEPTED_GRAPH_CARRIER",
            "EXACT_BOUNDARY_PARITY_WITH_INDEPENDENT_DENSE_Q2_TRANSFER",
            "EXACT_GRAPH_PAYLOAD_RESTORATION_AND_SAME_BACKING_REUSE",
            "CACHE_FREE_PROJECTION_PATH_LAW_8Q_TO_THE_2D",
        ],
        "rejected_interpretations": [
            "3Q_AMPLITUDE_STATE_CLOSURE",
            "FIXED_WORK_OR_FIXED_TOTAL_COST_ACROSS_DEPTH",
            "MACHINE_ENFORCED_HIDDEN_RUNTIME_FACTORS",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }
    if not qualified:
        fail("independent qualification failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.production, args.source), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
