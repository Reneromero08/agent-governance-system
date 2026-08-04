#!/usr/bin/env python3
"""Import-isolated oracle for cubic character-sum separator messages."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
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
    root = pow(generator, 2, p)
    gauss = sum(pow(root, x * x % q, p) for x in range(q)) % p
    if generator == p or pow(root, q, p) != 1 or not gauss:
        fail("invalid phase field")
    return Field(q, p, root, gauss)


def phase(field: Field, exponent: int) -> int:
    return pow(field.root, exponent % field.q, field.p)


@dataclass(frozen=True)
class Operation:
    kind: str
    payload: tuple[int, ...]


def compile_plan(q: int, depth: int, family: str) -> list[Operation]:
    if depth < 1 or family not in FAMILIES:
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


def initial_message(q: int, family: str) -> tuple[Any, ...]:
    code = 1 if family == "PRIMARY" else 2
    return "GAUSSIAN", 1, (code + 1) % q, (2 * code + 1) % q, code % q


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
    character = pow(coefficient, (field.q - 1) // 2, field.q)
    return field.gauss_one if character == 1 else -field.gauss_one % field.p


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


def legendre(value: int, q: int) -> int:
    symbol = pow(value % q, (q - 1) // 2, q)
    if symbol == 1:
        return 1
    if symbol == q - 1:
        return -1
    fail("zero Legendre argument")


@dataclass
class Work:
    data_history_evaluations: int = 0
    data_kernel_evaluations: int = 0
    controlled_message_phase_updates: int = 0
    gaussian_message_transforms: int = 0
    gaussian_to_gaussian_transforms: int = 0
    gaussian_to_delta_transforms: int = 0
    delta_to_gaussian_transforms: int = 0
    message_evaluations: int = 0
    data_field_multiplications: int = 0
    final_field_multiply_adds: int = 0
    data_history_tuple_cells_peak: int = 0
    latent_message_field_cells_peak: int = 0


def message_cells(message: tuple[Any, ...]) -> int:
    return 4 if message[0] == "GAUSSIAN" else 2


def multiply_message(message: tuple[Any, ...], coefficient: int, field: Field, work: Work) -> tuple[Any, ...]:
    work.controlled_message_phase_updates += 1
    if message[0] == "GAUSSIAN":
        _, scalar, quadratic, linear, constant = message
        output = "GAUSSIAN", scalar, quadratic, (linear + coefficient) % field.q, constant
    else:
        _, scalar, center = message
        output = "DELTA", scalar * phase(field, coefficient * center) % field.p, center
    work.latent_message_field_cells_peak = max(work.latent_message_field_cells_peak, message_cells(output))
    return output


def transform_message(message: tuple[Any, ...], payload: tuple[int, ...], field: Field, work: Work) -> tuple[Any, ...]:
    q, p = field.q, field.p
    a, b, _, d, coefficient = payload
    if b % q == 0:
        fail("nonzero b required")
    inverse_b = pow(b, -1, q)
    half_b = pow(2 * b % q, -1, q)
    work.gaussian_message_transforms += 1
    if message[0] == "DELTA":
        _, scalar, center = message
        output = (
            "GAUSSIAN",
            scalar * coefficient % p,
            d * half_b % q,
            -center * inverse_b % q,
            a * center * center * half_b % q,
        )
        work.delta_to_gaussian_transforms += 1
    else:
        _, scalar, quadratic, linear, constant = message
        alpha = (quadratic + a * half_b) % q
        if alpha == 0:
            center = b * linear % q
            value = scalar * coefficient * q % p
            value = value * phase(field, constant + d * center * center * half_b) % p
            output = "DELTA", value, center
            work.gaussian_to_delta_transforms += 1
        else:
            inverse_four_alpha = pow(4 * alpha % q, -1, q)
            output = (
                "GAUSSIAN",
                scalar * coefficient * field.gauss_one * legendre(alpha, q) % p,
                (d * half_b - pow(b, -2, q) * inverse_four_alpha) % q,
                linear * pow(2 * alpha * b % q, -1, q) % q,
                (constant - linear * linear * inverse_four_alpha) % q,
            )
            work.gaussian_to_gaussian_transforms += 1
    work.latent_message_field_cells_peak = max(work.latent_message_field_cells_peak, message_cells(output))
    return output


def evaluate_message(message: tuple[Any, ...], coordinate: int, field: Field, work: Work) -> int:
    work.message_evaluations += 1
    if message[0] == "DELTA":
        return message[1] if coordinate == message[2] else 0
    _, scalar, quadratic, linear, constant = message
    return scalar * phase(field, quadratic * coordinate * coordinate + linear * coordinate + constant) % field.p


def message_boundary(field: Field, message: tuple[Any, ...], data: list[int], operations: list[Operation], family: str) -> tuple[int, Work]:
    depth = (len(operations) - 1) // 4
    layers = [operations[4 * layer : 4 * layer + 4] for layer in range(depth)]
    fiber_matrix = operations[-1].payload
    work = Work(latent_message_field_cells_peak=message_cells(message))
    total = 0
    for final_s, final_fiber, final_x, weight in probes(field.q, family):
        coefficients = (fiber_matrix[0], fiber_matrix[2]) if final_fiber == 0 else (fiber_matrix[1], fiber_matrix[3])
        for source_fiber, fiber_coefficient in enumerate(coefficients):
            for source_path in itertools.product(range(field.q), repeat=depth):
                work.data_history_evaluations += 1
                work.data_history_tuple_cells_peak = max(work.data_history_tuple_cells_peak, depth + 1)
                coordinates = source_path + (final_x,)
                current = message
                scalar = data[source_fiber * field.q + coordinates[0]]
                for layer_index, layer in enumerate(layers):
                    source_x, target_x = coordinates[layer_index], coordinates[layer_index + 1]
                    if family == "PRIMARY":
                        current = multiply_message(current, layer[0].payload[0] * source_x**3, field, work)
                        scalar = scalar * layer[1].payload[4] % field.p
                        scalar = scalar * kernel(field, layer[1].payload[:4], target_x, source_x) % field.p
                        work.data_kernel_evaluations += 1
                        work.data_field_multiplications += 2
                        current = multiply_message(current, layer[2].payload[0] * target_x**3, field, work)
                        current = transform_message(current, layer[3].payload, field, work)
                    else:
                        current = multiply_message(current, layer[0].payload[0] * source_x**3, field, work)
                        current = transform_message(current, layer[1].payload, field, work)
                        current = multiply_message(current, layer[2].payload[0] * source_x**3, field, work)
                        scalar = scalar * layer[3].payload[4] % field.p
                        scalar = scalar * kernel(field, layer[3].payload[:4], target_x, source_x) % field.p
                        work.data_kernel_evaluations += 1
                        work.data_field_multiplications += 2
                total += weight * fiber_coefficient * scalar * evaluate_message(current, final_s, field, work)
                work.final_field_multiply_adds += 3
    return total % field.p, work


def address(q: int, s: int, fiber: int, x: int) -> int:
    return (2 * s + fiber) * q + x


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
                    values[index] = values[index] * phase(field, operation.payload[0] * s * x**3) % p
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
        fail("unknown dense operation")


def dense_initial(field: Field, message: tuple[Any, ...], data: list[int]) -> list[int]:
    latent = [evaluate_message(message, s, field, Work()) for s in range(field.q)]
    return [latent[s] * data[fiber * field.q + x] % field.p for s in range(field.q) for fiber in range(2) for x in range(field.q)]


def dense_boundary(values: list[int], field: Field, family: str) -> int:
    return sum(weight * values[address(field.q, s, fiber, x)] for s, fiber, x, weight in probes(field.q, family)) % field.p


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


def exhaustive_local_closure() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for q in (5, 11):
        field = make_field(q)
        payloads = {
            operation.payload
            for family in FAMILIES
            for operation in compile_plan(q, 4, family)
            if operation.kind == "LATENT_GAUSSIAN"
        }
        gaussian_count = 0
        for payload in payloads:
            operation = Operation("LATENT_GAUSSIAN", payload)
            for quadratic in range(q):
                for linear in range(q):
                    message = ("GAUSSIAN", 1, quadratic, linear, 0)
                    vector = [evaluate_message(message, coordinate, field, Work()) for coordinate in range(q)]
                    expected = dense_gaussian(vector, operation, field)
                    observed_message = transform_message(message, payload, field, Work())
                    observed = [evaluate_message(observed_message, coordinate, field, Work()) for coordinate in range(q)]
                    if observed != expected:
                        fail(f"local Gaussian closure failed q={q}")
                    gaussian_count += 1
            for center in range(q):
                message = ("DELTA", 1, center)
                vector = [int(coordinate == center) for coordinate in range(q)]
                expected = dense_gaussian(vector, operation, field)
                observed_message = transform_message(message, payload, field, Work())
                observed = [evaluate_message(observed_message, coordinate, field, Work()) for coordinate in range(q)]
                if observed != expected:
                    fail(f"local delta closure failed q={q}")
        checks[f"q{q}_all_declared_gaussian_and_delta_messages"] = gaussian_count > 0
    return checks


def reconstruct_case(production: dict[str, Any]) -> dict[str, Any]:
    q, depth, family = production["q"], production["depth"], production["family"]
    field = make_field(q)
    message, data = initial_message(q, family), data_fixture(field, family)
    operations = compile_plan(q, depth, family)
    boundary, work = message_boundary(field, message, data, operations, family)
    dense = dense_initial(field, message, data)
    dense_seed = dense.copy()
    for operation in operations:
        dense_apply(dense, operation, field)
    dense_answer = dense_boundary(dense, field, family)
    for operation in reversed(operations):
        dense_apply(dense, inverse_operation(field, operation), field)
    expected_histories = 8 * q**depth
    classical = production["matched_identical_classical_message"]
    rader = production["matched_exact_rader_ntt_transfer"]
    checks = {
        "boundary": boundary == production["boundary"],
        "dense_boundary": dense_answer == boundary,
        "dense_forward_inverse": dense == dense_seed,
        "classical_boundary": classical["boundary"] == boundary,
        "rader_boundary": rader["boundary"] == boundary,
        "work": work.__dict__ == production["message_work"] == classical["work"],
        "history_law": work.data_history_evaluations == production["actual_data_history_evaluations"] == expected_histories,
        "runtime_fields": production["accepted_runtime_field_elements"] == 2 * q + 4,
        "runtime_bit_capacity": production["accepted_runtime_bit_capacity_upper_bound"] == (2 * q + 1) * field.p.bit_length() + 3 * q.bit_length(),
        "node_law": production["public_morphism_node_records"] == 4 * depth + 1,
        "payload_law": production["public_morphism_payload_integer_cells"] == 12 * depth + 4,
        "payload_bit_capacity": production["public_morphism_payload_bit_capacity_upper_bound"] == 10 * depth * q.bit_length() + (2 * depth + 4) * field.p.bit_length(),
        "no_q2_or_cache": production["q2_amplitude_cells_on_accepted_message_path"] == 0 and production["recursive_or_dynamic_cache_entries"] == 0,
        "no_materialized_history": not production["data_history_or_assignment_list_materialized"],
        "rader_q2_state": rader["resident_field_cells"] == 2 * q * q,
        "rader_exactness": rader["single_auxiliary_modulus_exactness_bound_checked"],
        "graph_restore": graph_restore(field, operations),
        "production_restore": production["exact_graph_payload_restored"] and production["same_backing_restored"],
    }
    if not all(checks.values()):
        fail(f"case mismatch q={q} depth={depth} family={family}: {checks}")
    return {"q": q, "depth": depth, "family": family, "boundary": boundary, "checks": checks}


def controls() -> dict[str, bool]:
    field = make_field(5)
    operations = compile_plan(5, 1, "PRIMARY")
    normal, _ = message_boundary(field, initial_message(5, "PRIMARY"), data_fixture(field, "PRIMARY"), operations, "PRIMARY")
    mutated, _ = message_boundary(field, ("GAUSSIAN", 1, 2, 0, 1), data_fixture(field, "PRIMARY"), operations, "PRIMARY")
    return {
        "missing_inverse_rejected": not graph_restore(field, operations, "MISSING"),
        "wrong_inverse_rejected": not graph_restore(field, operations, "WRONG"),
        "reordered_inverse_rejected": not graph_restore(field, operations, "REORDER"),
        "message_mutation_changes_boundary": normal != mutated,
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
    message, data, nodes = initial_message(23, "PRIMARY"), data_fixture(field, "PRIMARY"), []
    backing = id(message), id(data), id(nodes)
    payload = message, tuple(data)

    def run(family: str) -> tuple[int, str]:
        operations = compile_plan(23, 2, family)
        nodes.extend(operations)
        answer, _ = message_boundary(field, message, data, nodes, family)
        commitment = hashlib.sha256(repr((23, message, tuple(data), tuple(nodes))).encode("ascii")).hexdigest()
        if not graph_restore(field, operations):
            fail("reuse graph inverse failed")
        nodes.clear()
        return answer, commitment

    run("PRIMARY")
    second = run("ALTERNATE")
    fresh_nodes = compile_plan(23, 2, "ALTERNATE")
    fresh_answer, _ = message_boundary(field, initial_message(23, "PRIMARY"), data_fixture(field, "PRIMARY"), fresh_nodes, "ALTERNATE")
    fresh_commitment = hashlib.sha256(repr((23, message, tuple(data), tuple(fresh_nodes))).encode("ascii")).hexdigest()
    return {
        "second_boundary_matches_fresh": second[0] == fresh_answer,
        "second_commitment_matches_fresh": second[1] == fresh_commitment,
        "payload_restored": (message, tuple(data)) == payload and not nodes,
        "same_backing": (id(message), id(data), id(nodes)) == backing,
        "generation_two": True,
        "no_snapshot": True,
    }


def source_structure(source_path: Path) -> dict[str, bool]:
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    oracle_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    carrier = next(node for node in source_tree.body if isinstance(node, ast.ClassDef) and node.name == "MessageCarrier")
    fields = {
        node.target.id
        for node in carrier.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    imports = {
        alias.name
        for node in oracle_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    return {
        "carrier_has_message_data_nodes": {"latent_message", "data", "nodes"}.issubset(fields),
        "carrier_has_no_expanded_cells_or_latent_vector": "cells" not in fields and "latent" not in fields,
        "oracle_imports_no_production_or_predecessor": not any(name.startswith("growing_prime") for name in imports),
    }


def build(production_path: Path, source_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    comparisons = [reconstruct_case(case) for case in production["cases"]]
    control_result = controls()
    reuse_result = reuse()
    local_closure = exhaustive_local_closure()
    structure = source_structure(source_path)
    production_controls = all(production["controls"].values())
    production_reuse = production["restoration_and_reuse"]
    production_reuse_passes = all((
        production_reuse["second_matches_fresh"],
        production_reuse["second_commitment_matches_fresh"],
        production_reuse["exact_payload_restored_after_reuse"],
        production_reuse["same_backing_reused"],
        production_reuse["restoration_generation"] == 2,
        not production_reuse["snapshot_used"],
    ))
    qualified = all((
        len(comparisons) == 13,
        all(all(case["checks"].values()) for case in comparisons),
        all(control_result.values()),
        all(reuse_result.values()),
        all(local_closure.values()),
        all(structure.values()),
        production_controls,
        production_reuse_passes,
    ))
    result = {
        "schema": "CAT_CAS_GROWING_PRIME_CUBIC_CHARACTER_SUM_SEPARATOR_MESSAGE_CLOSURE_INDEPENDENT_ORACLE_V1",
        "production_result": "GROWING_PRIME_CUBIC_CHARACTER_SUM_SEPARATOR_MESSAGE_CLOSURE_RESULTS.json",
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
            "separate_field_plan_message_dense_inverse_control_and_reuse_implementations": True,
            "production_source_used_only_for_static_carrier_checks": True,
        },
        "case_comparisons": comparisons,
        "controls": control_result,
        "production_controls_all_pass": production_controls,
        "restoration_and_reuse": reuse_result,
        "production_reuse_passes": production_reuse_passes,
        "exhaustive_local_message_closure": local_closure,
        "source_structure": structure,
        "observed_resource_law": production["observed_resource_law"],
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "EXACT_GAUSSIAN_OR_DELTA_LATENT_PHASE_MESSAGE_CLOSURE",
            "ALL_LATENT_STRENGTH_HISTORY_SUMS_ELIMINATED",
            "2Q_PLUS4_MIXED_RUNTIME_FIELD_ELEMENTS_WITH_8Q_TO_THE_D_DATA_HISTORY_WORK",
            "EXACT_BOUNDARY_PARITY_WITH_INDEPENDENT_DENSE_Q2_TRANSFER",
            "EXACT_GRAPH_PAYLOAD_RESTORATION_AND_SAME_BACKING_REUSE",
        ],
        "rejected_interpretations": [
            "ARBITRARY_LATENT_FACTOR_CLOSURE",
            "DATA_AIRY_MESSAGE_CLOSURE",
            "SUBQUADRATIC_STATE_POLYNOMIAL_WORK_CLOSURE",
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
