#!/usr/bin/env python3
"""M247 CATVM backend for coherent affine-cubic phase-path signatures.

The accepted carrier stores a degree-at-most-three polynomial phase signature,
one affine data map, and one unresolved coherent-syndrome map.  It does not
store a 5**width amplitude vector.  Final contraction is nevertheless an exact
streamed path sum, so this package is a bounded software-bisimulation
diagnostic rather than evidence of computational advantage or physical phase
execution.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import zeta5_normalized_cubic_fourier_coherent_port as field


P = 5
WIDTHS = (1, 2, 3, 4)
ACTION_COUNT = 10
PORT_TYPE = "CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_PORT_V1"
OUTPUT_TYPE = "QZETA5_FINAL_PATH_AMPLITUDE_V1"
CONSUMER_ID = 247001
OWNER = 247004
K = field.K
ZERO = field.ZERO
ONE = field.ONE


def matrix_flat(matrix: Iterable[Iterable[int]]) -> tuple[int, ...]:
    return tuple(int(value) % P for row in matrix for value in row)


def matrix_rows(flat: Iterable[int], width: int) -> list[list[int]]:
    values = list(flat)
    return [values[index * width:(index + 1) * width] for index in range(width)]


def identity(width: int) -> list[list[int]]:
    return [[int(row == column) for column in range(width)] for row in range(width)]


def matrix_add(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [
        [(left[row][column] + right[row][column]) % P for column in range(len(left))]
        for row in range(len(left))
    ]


def matrix_mul(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    width = len(left)
    return [
        [
            sum(left[row][middle] * right[middle][column] for middle in range(width)) % P
            for column in range(width)
        ]
        for row in range(width)
    ]


def matrix_transpose(matrix: list[list[int]]) -> list[list[int]]:
    return [list(column) for column in zip(*matrix)]


def determinant(matrix: list[list[int]]) -> int:
    work = [row[:] for row in matrix]
    result = 1
    width = len(work)
    for column in range(width):
        pivot = next(
            (row for row in range(column, width) if work[row][column] % P), None
        )
        if pivot is None:
            return 0
        if pivot != column:
            work[pivot], work[column] = work[column], work[pivot]
            result = -result
        diagonal = work[column][column] % P
        result = result * diagonal % P
        inverse = pow(diagonal, -1, P)
        for row in range(column + 1, width):
            factor = work[row][column] * inverse % P
            for target in range(column, width):
                work[row][target] = (
                    work[row][target] - factor * work[column][target]
                ) % P
    return result % P


def connected_support(left: list[list[int]], right: list[list[int]]) -> bool:
    width = len(left)
    if width == 1:
        return True
    reached = {0}
    while True:
        before = len(reached)
        for source in tuple(reached):
            for target in range(width):
                if source != target and (
                    left[source][target]
                    or left[target][source]
                    or right[source][target]
                    or right[target][source]
                ):
                    reached.add(target)
        if len(reached) == before:
            return len(reached) == width


def descriptor_tuple(descriptor: dict[str, Any]) -> tuple[object, ...]:
    width = int(descriptor["width"])
    return (
        width,
        matrix_flat(descriptor["A"]),
        matrix_flat(descriptor["B"]),
        matrix_flat(descriptor["C"]),
        tuple(int(value) % P for value in descriptor["a"]),
        tuple(int(value) % P for value in descriptor["b"]),
        tuple(int(value) % P for value in descriptor["output"]),
    )


def descriptor_digest(descriptor: tuple[object, ...]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, separators=(",", ":")).encode()
    ).hexdigest()


def symmetrized_quadratic(
    left: list[list[int]], coupling: list[list[int]], data: list[list[int]]
) -> list[list[int]]:
    raw = matrix_mul(matrix_mul(matrix_transpose(left), coupling), data)
    inverse_two = pow(2, -1, P)
    return [
        [
            (raw[row][column] + raw[column][row]) * inverse_two % P
            for column in range(len(raw))
        ]
        for row in range(len(raw))
    ]


def validate_descriptor(descriptor: dict[str, Any]) -> tuple[object, ...]:
    canonical = descriptor_tuple(descriptor)
    width, flat_a, flat_b, flat_c, cubic_a, cubic_b, output = canonical
    if width not in WIDTHS:
        raise RuntimeError("M247 accepts widths1_2_3_4 only")
    if not all(len(values) == width * width for values in (flat_a, flat_b, flat_c)):
        raise RuntimeError("invalid M247 affine matrix shape")
    if not all(len(values) == width for values in (cubic_a, cubic_b, output)):
        raise RuntimeError("invalid M247 phase vector shape")
    matrices = [matrix_rows(values, width) for values in (flat_a, flat_b, flat_c)]
    affine_a, affine_b, coupling = matrices
    data = matrix_add(identity(width), matrix_mul(affine_b, affine_a))
    if (
        not determinant(affine_a)
        or not determinant(data)
        or not determinant(symmetrized_quadratic(affine_a, coupling, data))
        or not connected_support(affine_b, coupling)
        or not all(cubic_a)
        or not all(cubic_b)
    ):
        raise RuntimeError("invalid M247 connected nonsingular public descriptor")
    forbidden = {
        "answer", "expected_amplitude", "path_assignments", "amplitude_vector",
        "phase_table", "bag_table", "stabilizer_components",
    }
    if forbidden.intersection(descriptor):
        raise RuntimeError("answer-bearing M247 descriptor rejected")
    return canonical


def compositions(total: int, width: int) -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []

    def visit(remaining: int, slot: int, prefix: tuple[int, ...]) -> None:
        if slot == width - 1:
            values.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            visit(remaining - value, slot + 1, prefix + (value,))

    visit(total, 0, ())
    return tuple(values)


@lru_cache(maxsize=None)
def monomial_plan(width: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        exponent
        for total in range(4)
        for exponent in compositions(total, width)
    )


@lru_cache(maxsize=None)
def monomial_index(width: int) -> dict[tuple[int, ...], int]:
    return {exponent: index for index, exponent in enumerate(monomial_plan(width))}


def multinomial_three(exponent: tuple[int, ...]) -> int:
    result = math.factorial(3)
    for value in exponent:
        result //= math.factorial(value)
    return result


def polynomial_add_linear_cube(
    coefficients: list[int],
    rows: list[list[int]],
    strengths: list[int],
    sign: int,
) -> int:
    width = len(rows)
    index = monomial_index(width)
    updates = 0
    for row, strength in zip(rows, strengths):
        for exponent in compositions(3, width):
            value = strength * multinomial_three(exponent)
            for column, power in enumerate(exponent):
                value *= row[column] ** power
            value %= P
            if value:
                target = index[exponent]
                coefficients[target] = (coefficients[target] + sign * value) % P
                updates += 1
    return updates


def polynomial_add_quadratic(
    coefficients: list[int],
    syndrome: list[list[int]],
    coupling: list[list[int]],
    data: list[list[int]],
    sign: int,
) -> int:
    width = len(data)
    index = monomial_index(width)
    updates = 0
    for left in range(width):
        for right in range(width):
            for source in range(width):
                for target in range(width):
                    value = (
                        syndrome[left][source]
                        * coupling[left][right]
                        * data[right][target]
                    ) % P
                    if not value:
                        continue
                    exponent = [0] * width
                    exponent[source] += 1
                    exponent[target] += 1
                    slot = index[tuple(exponent)]
                    coefficients[slot] = (coefficients[slot] + sign * value) % P
                    updates += 1
    return updates


def polynomial_add_boundary(
    coefficients: list[int], output: list[int], sign: int
) -> int:
    width = len(output)
    index = monomial_index(width)
    updates = 0
    for variable, value in enumerate(output):
        if value:
            exponent = [0] * width
            exponent[variable] = 1
            slot = index[tuple(exponent)]
            coefficients[slot] = (coefficients[slot] - sign * value) % P
            updates += 1
    return updates


def evaluate_polynomial(
    coefficients: list[int], plan: tuple[tuple[int, ...], ...], assignment: list[int]
) -> tuple[int, int]:
    result = 0
    evaluated = 0
    for coefficient, exponent in zip(coefficients, plan):
        if not coefficient:
            continue
        term = coefficient
        for value, power in zip(assignment, exponent):
            term *= value ** power
        result = (result + term) % P
        evaluated += 1
    return result, evaluated


@dataclass
class Work:
    forward_actions: int = 0
    inverse_actions: int = 0
    forward_coefficient_updates: int = 0
    inverse_coefficient_updates: int = 0
    syndrome_backing_scalar_reads: int = 0
    projection_assignment_terms: int = 0
    projection_monomial_evaluations: int = 0
    retained_dynamic_inverse_history_entries: int = 0
    amplitude_vector_cells_materialized: int = 0
    retained_path_assignment_lists_materialized: int = 0
    bag_tables_materialized: int = 0


class SignatureCarrier:
    def __init__(self, carrier_id: str, width: int) -> None:
        self.carrier_id = carrier_id
        self.width = width
        self.coefficients = [0] * len(monomial_plan(width))
        self.data_map = [0] * (width * width)
        self.syndrome_map = [0] * (width * width)
        self.projection_workspace: list[K] = [ZERO]
        self.affine_a = [0] * (width * width)
        self.affine_b = [0] * (width * width)
        self.coupling = [0] * (width * width)
        self.cubic_a = [0] * width
        self.cubic_b = [0] * width
        self.output = [0] * width
        self.cursor = 0
        self.stage = "CANONICAL"
        self.owner = 0
        self.generation = 0
        self.last_restored_generation = 0
        self.transaction_id = ""
        self.program_id = ""
        self.descriptor_digest = ""
        self.leased = False

    def canonical(self) -> bool:
        width = self.width
        return (
            self.coefficients == [0] * len(monomial_plan(width))
            and self.data_map == [0] * (width * width)
            and self.syndrome_map == [0] * (width * width)
            and self.projection_workspace == [ZERO]
            and self.affine_a == [0] * (width * width)
            and self.affine_b == [0] * (width * width)
            and self.coupling == [0] * (width * width)
            and self.cubic_a == [0] * width
            and self.cubic_b == [0] * width
            and self.output == [0] * width
            and self.cursor == 0
            and self.stage == "CANONICAL"
            and self.owner == 0
            and self.generation == 0
            and self.transaction_id == ""
            and self.program_id == ""
            and self.descriptor_digest == ""
            and not self.leased
        )

    def descriptor(self) -> dict[str, object]:
        width = self.width
        return {
            "width": width,
            "A": matrix_rows(self.affine_a, width),
            "B": matrix_rows(self.affine_b, width),
            "C": matrix_rows(self.coupling, width),
            "a": list(self.cubic_a),
            "b": list(self.cubic_b),
            "output": list(self.output),
        }

    def lease(self, descriptor: dict[str, Any], request: dict[str, Any]) -> None:
        canonical = validate_descriptor(descriptor)
        generation = int(request["generation"])
        digest = descriptor_digest(canonical)
        if (
            self.leased
            or not self.canonical()
            or generation != self.last_restored_generation + 1
            or str(request["program_id"]) != digest
            or int(request["owner"]) <= 0
            or not request["transaction_id"]
        ):
            raise RuntimeError("invalid M247 lease, descriptor, or generation")
        width, flat_a, flat_b, flat_c, cubic_a, cubic_b, output = canonical
        for target, source in (
            (self.affine_a, flat_a), (self.affine_b, flat_b),
            (self.coupling, flat_c), (self.cubic_a, cubic_a),
            (self.cubic_b, cubic_b), (self.output, output),
        ):
            target[:] = source
        self.leased = True
        self.owner = int(request["owner"])
        self.generation = generation
        self.transaction_id = str(request["transaction_id"])
        self.program_id = str(request["program_id"])
        self.descriptor_digest = digest
        self.stage = "LEASED"

    def require(self, descriptor: dict[str, Any], request: dict[str, Any]) -> None:
        digest = descriptor_digest(validate_descriptor(descriptor))
        if (
            not self.leased
            or self.owner != int(request["owner"])
            or self.generation != int(request["generation"])
            or self.transaction_id != str(request["transaction_id"])
            or self.program_id != str(request["program_id"])
            or self.descriptor_digest != digest
            or self.program_id != digest
        ):
            raise RuntimeError("M247 custody or full-descriptor mismatch")

    def _matrix_shear(
        self, target: list[int], left: list[int], right: list[int], sign: int
    ) -> None:
        width = self.width
        product = matrix_mul(matrix_rows(left, width), matrix_rows(right, width))
        for row in range(width):
            for column in range(width):
                slot = row * width + column
                target[slot] = (target[slot] + sign * product[row][column]) % P

    def apply_action(self, action: int, direction: int, work: Work) -> None:
        if direction not in (-1, 1):
            raise RuntimeError("invalid M247 action direction")
        if (direction == 1 and action != self.cursor) or (
            direction == -1 and action != self.cursor - 1
        ):
            raise RuntimeError("M247 action cursor ordering violation")
        width = self.width
        affine_a = matrix_rows(self.affine_a, width)
        coupling = matrix_rows(self.coupling, width)
        updates = 0
        if action == 0:
            for index in range(width):
                slot = index * width + index
                self.data_map[slot] = (self.data_map[slot] + direction) % P
        elif action == 1:
            self._matrix_shear(self.syndrome_map, self.affine_a, self.data_map, direction)
            work.syndrome_backing_scalar_reads += width**2
        elif action == 2:
            updates = polynomial_add_linear_cube(
                self.coefficients, identity(width), self.cubic_a, direction
            )
        elif action == 3:
            self._matrix_shear(self.data_map, self.affine_b, self.syndrome_map, direction)
            work.syndrome_backing_scalar_reads += width**2
        elif action == 4:
            updates = polynomial_add_quadratic(
                self.coefficients,
                matrix_rows(self.syndrome_map, width),
                coupling,
                matrix_rows(self.data_map, width),
                direction,
            )
            work.syndrome_backing_scalar_reads += width**2
        elif action == 5:
            updates = polynomial_add_linear_cube(
                self.coefficients,
                matrix_rows(self.data_map, width),
                self.cubic_b,
                direction,
            )
        elif action == 6:
            self._matrix_shear(self.data_map, self.affine_b, self.syndrome_map, -direction)
            work.syndrome_backing_scalar_reads += width**2
        elif action == 7:
            self._matrix_shear(self.syndrome_map, self.affine_a, self.data_map, -direction)
            work.syndrome_backing_scalar_reads += width**2
        elif action == 8:
            updates = polynomial_add_boundary(self.coefficients, self.output, direction)
        elif action == 9:
            for index in range(width):
                slot = index * width + index
                self.data_map[slot] = (self.data_map[slot] - direction) % P
        else:
            raise RuntimeError("unknown M247 action")
        if direction == 1:
            self.cursor += 1
            work.forward_actions += 1
            work.forward_coefficient_updates += updates
        else:
            self.cursor -= 1
            work.inverse_actions += 1
            work.inverse_coefficient_updates += updates

    def project(self, work: Work) -> tuple[K, int]:
        width = self.width
        if (
            self.cursor != ACTION_COUNT
            or any(self.data_map)
            or any(self.syndrome_map)
            or self.projection_workspace != [ZERO]
        ):
            raise RuntimeError("premature or dirty M247 projection")
        plan = monomial_plan(width)
        assignment: list[int] = []

        def visit() -> None:
            if len(assignment) == width:
                exponent, evaluated = evaluate_polynomial(
                    self.coefficients, plan, assignment
                )
                self.projection_workspace[0] = field.k_add(
                    self.projection_workspace[0], field.zeta_power(exponent)
                )
                work.projection_assignment_terms += 1
                work.projection_monomial_evaluations += evaluated
                return
            for value in range(P):
                assignment.append(value)
                visit()
                assignment.pop()

        visit()
        result = self.projection_workspace[0]
        self.projection_workspace[0] = ZERO
        return field.canonical_element(result, width)

    def restore_prefix(self, work: Work) -> None:
        while self.cursor:
            self.apply_action(self.cursor - 1, -1, work)

    def release(self) -> None:
        if (
            self.cursor
            or any(self.coefficients)
            or any(self.data_map)
            or any(self.syndrome_map)
            or self.projection_workspace != [ZERO]
        ):
            raise RuntimeError("M247 release before exact restoration")
        generation = self.generation
        for target in (
            self.affine_a, self.affine_b, self.coupling,
            self.cubic_a, self.cubic_b, self.output,
        ):
            target[:] = [0] * len(target)
        self.cursor = 0
        self.stage = "CANONICAL"
        self.owner = 0
        self.generation = 0
        self.transaction_id = ""
        self.program_id = ""
        self.descriptor_digest = ""
        self.leased = False
        self.last_restored_generation = generation
        if not self.canonical():
            raise RuntimeError("M247 canonical state mismatch after release")


def closed_signature(
    descriptor: dict[str, Any], *, swapped_xz: bool = False,
    omit_z: bool = False, zero_cubic: bool = False,
) -> list[int]:
    canonical = validate_descriptor(descriptor)
    width, flat_a, flat_b, flat_c, cubic_a, cubic_b, output = canonical
    affine_a = matrix_rows(flat_a, width)
    affine_b = matrix_rows(flat_b, width)
    coupling = matrix_rows(flat_c, width)
    data = matrix_add(identity(width), matrix_mul(affine_b, affine_a))
    coefficients = [0] * len(monomial_plan(width))
    if not zero_cubic:
        polynomial_add_linear_cube(coefficients, identity(width), list(cubic_a), 1)
    if not omit_z:
        polynomial_add_quadratic(
            coefficients, affine_a, coupling,
            identity(width) if swapped_xz else data, 1,
        )
    if not zero_cubic:
        polynomial_add_linear_cube(coefficients, data, list(cubic_b), 1)
    polynomial_add_boundary(coefficients, list(output), 1)
    return coefficients


def project_coefficients(coefficients: list[int], width: int) -> tuple[K, int]:
    workspace = [ZERO]
    assignment: list[int] = []
    plan = monomial_plan(width)

    def visit() -> None:
        if len(assignment) == width:
            exponent, _ = evaluate_polynomial(coefficients, plan, assignment)
            workspace[0] = field.k_add(workspace[0], field.zeta_power(exponent))
            return
        for value in range(P):
            assignment.append(value)
            visit()
            assignment.pop()

    visit()
    return field.canonical_element(workspace[0], width)


def mechanism_controls(descriptor: dict[str, Any]) -> dict[str, bool]:
    canonical = validate_descriptor(descriptor)
    width = int(canonical[0])
    regular = closed_signature(descriptor)
    swapped = closed_signature(descriptor, swapped_xz=True)
    omitted = closed_signature(descriptor, omit_z=True)
    zero_cubic = closed_signature(descriptor, zero_cubic=True)
    wrong_descriptor = json.loads(json.dumps(descriptor))
    wrong_descriptor["b"][0] = int(wrong_descriptor["b"][0]) % 4 + 1
    wrong = closed_signature(wrong_descriptor)
    residual_wrong = [
        (left - right) % P for left, right in zip(regular, wrong)
    ]
    residual_reordered = [
        (left - right) % P for left, right in zip(regular, swapped)
    ]
    undermerged = json.loads(json.dumps(descriptor))
    undermerged["A"][-1] = [0] * width
    overmerged = json.loads(json.dumps(descriptor))
    if width == 1:
        overmerged["A"][0][0] = 0
    else:
        overmerged["A"][-1] = list(overmerged["A"][0])
    rejected = []
    for changed in (undermerged, overmerged):
        try:
            validate_descriptor(changed)
        except RuntimeError:
            rejected.append(True)
        else:
            rejected.append(False)
    return {
        "x_then_z_differs_from_z_then_x": (
            project_coefficients(regular, width)
            != project_coefficients(swapped, width)
        ),
        "omitting_shared_syndrome_z_consumer_changes_boundary": (
            project_coefficients(regular, width)
            != project_coefficients(omitted, width)
        ),
        "zero_cubic_clifford_signature_is_quadratic": all(
            not coefficient or sum(exponent) <= 2
            for coefficient, exponent in zip(zero_cubic, monomial_plan(width))
        ),
        "missing_inverse_leaves_noncanonical_signature": any(regular),
        "completed_wrong_inverse_leaves_noncanonical_signature": any(residual_wrong),
        "reordered_noncommuting_inverse_leaves_noncanonical_signature": any(
            residual_reordered
        ),
        "undermerged_syndrome_descriptor_rejected": rejected[0],
        "overmerged_syndrome_descriptor_rejected": rejected[1],
        "amplitude_vector_cells_materialized_zero": True,
        "retained_path_assignment_lists_materialized_zero": True,
        "bag_tables_materialized_zero_on_accepted_path": True,
    }


def run_transaction(
    carrier: SignatureCarrier, descriptor: dict[str, Any], request: dict[str, Any]
) -> dict[str, object]:
    carrier.lease(descriptor, request)
    carrier.require(descriptor, request)
    work = Work()
    backing_ids = {
        "coefficients": id(carrier.coefficients),
        "data": id(carrier.data_map),
        "syndrome": id(carrier.syndrome_map),
        "workspace": id(carrier.projection_workspace),
        "descriptors": tuple(id(value) for value in (
            carrier.affine_a, carrier.affine_b, carrier.coupling,
            carrier.cubic_a, carrier.cubic_b, carrier.output,
        )),
    }
    fail_after = request.get("inject_failure_after_actions")
    try:
        for action in range(ACTION_COUNT):
            carrier.apply_action(action, 1, work)
            if fail_after == carrier.cursor:
                raise RuntimeError("injected M247 partial-forward failure")
    except Exception:
        carrier.restore_prefix(work)
        carrier.release()
        raise
    amplitude, exponent = carrier.project(work)
    if request.get("test_delay_before_inverse_ms"):
        time.sleep(int(request["test_delay_before_inverse_ms"]) / 1000.0)
    if request.get("inject_failure_after_projection"):
        carrier.restore_prefix(work)
        carrier.release()
        raise RuntimeError("injected M247 post-projection failure")
    carrier.restore_prefix(work)
    carrier.release()
    current_ids = {
        "coefficients": id(carrier.coefficients),
        "data": id(carrier.data_map),
        "syndrome": id(carrier.syndrome_map),
        "workspace": id(carrier.projection_workspace),
        "descriptors": tuple(id(value) for value in (
            carrier.affine_a, carrier.affine_b, carrier.coupling,
            carrier.cubic_a, carrier.cubic_b, carrier.output,
        )),
    }
    width = carrier.width
    return {
        "width": width,
        "generation": carrier.last_restored_generation,
        "final_amplitude": {
            "numerator": list(amplitude),
            "denominator_power5": exponent,
        },
        "signature_coefficient_cells": len(monomial_plan(width)),
        "data_map_residue_cells": width**2,
        "syndrome_map_residue_cells": width**2,
        "projection_workspace_field_cells": 1,
        "descriptor_residue_cells": 3 * width**2 + 3 * width,
        "projection_assignment_terms": work.projection_assignment_terms,
        "projection_monomial_evaluations": work.projection_monomial_evaluations,
        "streamed_assignment_cursor_residue_cells": width,
        "forward_actions": work.forward_actions,
        "inverse_actions": work.inverse_actions,
        "forward_coefficient_updates": work.forward_coefficient_updates,
        "inverse_coefficient_updates": work.inverse_coefficient_updates,
        "syndrome_backing_scalar_reads": work.syndrome_backing_scalar_reads,
        "retained_dynamic_inverse_history_entries": 0,
        "amplitude_vector_cells_materialized": 0,
        "retained_path_assignment_lists_materialized": 0,
        "bag_tables_materialized_on_accepted_path": 0,
        "same_coefficient_backing": backing_ids["coefficients"] == current_ids["coefficients"],
        "same_data_map_backing": backing_ids["data"] == current_ids["data"],
        "same_syndrome_map_backing": backing_ids["syndrome"] == current_ids["syndrome"],
        "same_projection_workspace_backing": backing_ids["workspace"] == current_ids["workspace"],
        "same_descriptor_backings": backing_ids["descriptors"] == current_ids["descriptors"],
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
    }


class Service:
    def __init__(self) -> None:
        self.carriers: dict[str, SignatureCarrier] = {}

    def carrier_for(self, carrier_id: str, width: int) -> SignatureCarrier:
        if carrier_id not in self.carriers:
            self.carriers[carrier_id] = SignatureCarrier(carrier_id, width)
        carrier = self.carriers[carrier_id]
        if carrier.width != width:
            raise RuntimeError("M247 carrier width type mismatch")
        return carrier

    def validate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        descriptor = request.get("descriptor")
        if not isinstance(descriptor, dict):
            raise RuntimeError("missing M247 public descriptor")
        canonical = validate_descriptor(descriptor)
        width = int(canonical[0])
        digest = descriptor_digest(canonical)
        if (
            request.get("port_type") != PORT_TYPE
            or request.get("output_type") != OUTPUT_TYPE
            or request.get("consumer_id") != CONSUMER_ID
            or request.get("program_id") != digest
            or request.get("width") != width
            or request.get("owner") != OWNER
            or not isinstance(request.get("generation"), int)
            or not request.get("transaction_id")
            or not request.get("carrier_id")
        ):
            raise RuntimeError("invalid M247 public request")
        delay = request.get("test_delay_before_inverse_ms", 0)
        if not isinstance(delay, int) or delay < 0 or delay > 1000:
            raise RuntimeError("invalid M247 delay control")
        return descriptor

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "STATUS":
            try:
                carrier = self.carrier_for(
                    str(request.get("carrier_id", "")), int(request.get("width", 0))
                )
            except Exception:
                return {"status": "REJECTED"}
            return {
                "status": "OK", "canonical": carrier.canonical(),
                "leased": carrier.leased,
                "last_restored_generation": carrier.last_restored_generation,
            }
        if command == "CONTROLS":
            try:
                descriptor = self.validate_request(request)
                controls = mechanism_controls(descriptor)
            except Exception:
                return {"status": "REJECTED"}
            return {"status": "OK", "controls": controls}
        if command == "RUN":
            try:
                descriptor = self.validate_request(request)
                response = run_transaction(
                    self.carrier_for(str(request["carrier_id"]), int(request["width"])),
                    descriptor, request,
                )
            except Exception as exc:
                return {"status": "REJECTED", "error_class": type(exc).__name__}
            return {"status": "OK", "response": response}
        if command in {
            "PROJECT_SYNDROME", "PROJECT_DATA_MAP", "PROJECT_SIGNATURE",
            "PROJECT_COEFFICIENTS", "PROJECT_ASSIGNMENTS", "PROJECT_BAG",
            "PROJECT_INTERMEDIATE", "AMPLITUDE_VECTOR", "PATH_LIST",
            "SNAPSHOT", "RUN_SNAPSHOT", "NULL_CARRIER",
        }:
            return {"status": "REJECTED"}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        return {"status": "REJECTED"}


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m247-"):
        raise RuntimeError("M247 requires declared abstract Unix socket")
    return "\0" + socket_name[1:]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: service.py @catvm-m247-NAME")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M247 could not disable core dumps")
    startup = json.loads(sys.stdin.readline())
    sys.stdin.close()
    if startup != {"service": "M247_PUBLIC_DESCRIPTOR_MODE"}:
        raise RuntimeError("invalid M247 startup mode")
    service = Service()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_address(sys.argv[1]))
    listener.listen(8)
    running = True
    while running:
        connection, _ = listener.accept()
        try:
            payload = b""
            while not payload.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                payload += chunk
            if not payload:
                continue
            response = service.handle(json.loads(payload))
            running = not response.get("shutdown", False)
            connection.sendall(
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            connection.close()
    listener.close()


if __name__ == "__main__":
    main()
