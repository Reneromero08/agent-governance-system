#!/usr/bin/env python3
"""Standalone exact oracle for M247 affine-cubic phase signatures.

This file intentionally imports no production, controller, or predecessor
arithmetic.  It reconstructs Q(zeta5), the public polynomial phase, an exact
min-fill variable-elimination comparator, and the quadratic Gauss sham.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Iterable


P = 5
WIDTHS = (1, 2, 3, 4)
K = tuple[int, int, int, int]
ZERO: K = (0, 0, 0, 0)
ONE: K = (1, 0, 0, 0)


def k_add(left: K, right: K) -> K:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def k_neg(value: K) -> K:
    return tuple(-entry for entry in value)  # type: ignore[return-value]


def k_mul(left: K, right: K) -> K:
    raw = [0] * 7
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            raw[i + j] += a * b
    for degree in range(6, 3, -1):
        value = raw[degree]
        if value:
            for shift in range(1, 5):
                raw[degree - shift] -= value
    return tuple(raw[:4])  # type: ignore[return-value]


def k_pow(value: K, exponent: int) -> K:
    result = ONE
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = k_mul(result, factor)
        remaining //= 2
        if remaining:
            factor = k_mul(factor, factor)
    return result


def zeta_power(exponent: int) -> K:
    basis = (ONE, (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    residue = exponent % P
    if residue < 4:
        return basis[residue]
    return (-1, -1, -1, -1)


def canonical(value: K, exponent: int) -> tuple[K, int]:
    result = value
    power = exponent
    while power and all(entry % P == 0 for entry in result):
        result = tuple(entry // P for entry in result)  # type: ignore[assignment]
        power -= 1
    return result, power


def flat(matrix: Iterable[Iterable[int]]) -> tuple[int, ...]:
    return tuple(int(value) % P for row in matrix for value in row)


def rows(values: Iterable[int], width: int) -> list[list[int]]:
    data = list(values)
    return [data[index * width:(index + 1) * width] for index in range(width)]


def identity(width: int) -> list[list[int]]:
    return [[int(row == column) for column in range(width)] for row in range(width)]


def mat_add(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [
        [(left[i][j] + right[i][j]) % P for j in range(len(left))]
        for i in range(len(left))
    ]


def mat_mul(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    width = len(left)
    return [
        [sum(left[i][k] * right[k][j] for k in range(width)) % P for j in range(width)]
        for i in range(width)
    ]


def mat_vec(matrix: list[list[int]], vector: list[int]) -> list[int]:
    return [sum(a * b for a, b in zip(row, vector)) % P for row in matrix]


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    return [list(column) for column in zip(*matrix)]


def determinant(matrix: list[list[int]]) -> int:
    work = [row[:] for row in matrix]
    value = 1
    for column in range(len(work)):
        pivot = next((i for i in range(column, len(work)) if work[i][column]), None)
        if pivot is None:
            return 0
        if pivot != column:
            work[pivot], work[column] = work[column], work[pivot]
            value = -value
        diagonal = work[column][column] % P
        value = value * diagonal % P
        inverse = pow(diagonal, -1, P)
        for row in range(column + 1, len(work)):
            factor = work[row][column] * inverse % P
            for target in range(column, len(work)):
                work[row][target] = (
                    work[row][target] - factor * work[column][target]
                ) % P
    return value % P


def inverse_matrix(matrix: list[list[int]]) -> list[list[int]]:
    width = len(matrix)
    work = [row[:] + identity(width)[i] for i, row in enumerate(matrix)]
    for column in range(width):
        pivot = next((i for i in range(column, width) if work[i][column]), None)
        if pivot is None:
            raise RuntimeError("singular quadratic sham")
        work[pivot], work[column] = work[column], work[pivot]
        scale = pow(work[column][column], -1, P)
        work[column] = [value * scale % P for value in work[column]]
        for row in range(width):
            if row == column:
                continue
            scale = work[row][column]
            work[row] = [
                (left - scale * right) % P
                for left, right in zip(work[row], work[column])
            ]
    return [row[width:] for row in work]


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[object, ...]:
    width = int(descriptor["width"])
    canonical = (
        width,
        flat(descriptor["A"]), flat(descriptor["B"]), flat(descriptor["C"]),
        tuple(int(value) % P for value in descriptor["a"]),
        tuple(int(value) % P for value in descriptor["b"]),
        tuple(int(value) % P for value in descriptor["output"]),
    )
    if width not in WIDTHS:
        raise RuntimeError("reference width rejected")
    if any(len(values) != width * width for values in canonical[1:4]):
        raise RuntimeError("reference matrix shape rejected")
    if any(len(values) != width for values in canonical[4:]):
        raise RuntimeError("reference vector shape rejected")
    return canonical


def digest(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(canonical_descriptor(descriptor), separators=(",", ":")).encode()
    ).hexdigest()


def compositions(total: int, width: int) -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []

    def visit(remaining: int, prefix: tuple[int, ...]) -> None:
        if len(prefix) == width - 1:
            values.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            visit(remaining - value, prefix + (value,))

    visit(total, ())
    return tuple(values)


def plan(width: int) -> tuple[tuple[int, ...], ...]:
    return tuple(exponent for degree in range(4) for exponent in compositions(degree, width))


def cube_add(
    coefficients: list[int], linear_rows: list[list[int]], strengths: list[int], sign: int
) -> None:
    index = {exponent: i for i, exponent in enumerate(plan(len(linear_rows)))}
    width = len(linear_rows)
    for row, strength in zip(linear_rows, strengths):
        for exponent in compositions(3, width):
            multiplier = math.factorial(3)
            for power in exponent:
                multiplier //= math.factorial(power)
            value = strength * multiplier
            for scalar, power in zip(row, exponent):
                value *= scalar**power
            coefficients[index[exponent]] = (
                coefficients[index[exponent]] + sign * value
            ) % P


def quadratic_add(
    coefficients: list[int], syndrome: list[list[int]], coupling: list[list[int]],
    data: list[list[int]], sign: int,
) -> None:
    width = len(data)
    index = {exponent: i for i, exponent in enumerate(plan(width))}
    for i in range(width):
        for j in range(width):
            for source in range(width):
                for target in range(width):
                    value = syndrome[i][source] * coupling[i][j] * data[j][target]
                    exponent = [0] * width
                    exponent[source] += 1
                    exponent[target] += 1
                    slot = index[tuple(exponent)]
                    coefficients[slot] = (coefficients[slot] + sign * value) % P


def boundary_add(coefficients: list[int], output: list[int], sign: int) -> None:
    width = len(output)
    index = {exponent: i for i, exponent in enumerate(plan(width))}
    for variable, value in enumerate(output):
        exponent = [0] * width
        exponent[variable] = 1
        slot = index[tuple(exponent)]
        coefficients[slot] = (coefficients[slot] - sign * value) % P


def compile_coefficients(
    descriptor: dict[str, Any], *, zero_cubic: bool = False, swapped: bool = False,
    omit_z: bool = False,
) -> list[int]:
    canonical = canonical_descriptor(descriptor)
    width = int(canonical[0])
    a_matrix, b_matrix, coupling = [rows(values, width) for values in canonical[1:4]]
    cubic_a, cubic_b, output = [list(values) for values in canonical[4:]]
    data = mat_add(identity(width), mat_mul(b_matrix, a_matrix))
    coefficients = [0] * len(plan(width))
    if not zero_cubic:
        cube_add(coefficients, identity(width), cubic_a, 1)
    if not omit_z:
        quadratic_add(
            coefficients, a_matrix, coupling, identity(width) if swapped else data, 1
        )
    if not zero_cubic:
        cube_add(coefficients, data, cubic_b, 1)
    boundary_add(coefficients, output, 1)
    return coefficients


def exponent_value(coefficients: list[int], assignment: tuple[int, ...]) -> int:
    result = 0
    for coefficient, exponent in zip(coefficients, plan(len(assignment))):
        term = coefficient
        for value, power in zip(assignment, exponent):
            term *= value**power
        result = (result + term) % P
    return result


def stream_assignments(width: int):
    current = [0] * width

    def visit(slot: int):
        if slot == width:
            yield tuple(current)
            return
        for value in range(P):
            current[slot] = value
            yield from visit(slot + 1)

    yield from visit(0)


def direct_polynomial_sum(coefficients: list[int], width: int) -> tuple[K, int]:
    total = ZERO
    for assignment in stream_assignments(width):
        total = k_add(total, zeta_power(exponent_value(coefficients, assignment)))
    return canonical(total, width)


def gate_semantic_sum(descriptor: dict[str, Any], zero_cubic: bool = False) -> tuple[K, int]:
    canonical_value = canonical_descriptor(descriptor)
    width = int(canonical_value[0])
    a_matrix, b_matrix, coupling = [rows(values, width) for values in canonical_value[1:4]]
    cubic_a, cubic_b, output = [list(values) for values in canonical_value[4:]]
    data_matrix = mat_add(identity(width), mat_mul(b_matrix, a_matrix))
    total = ZERO
    for assignment in stream_assignments(width):
        x = list(assignment)
        syndrome = mat_vec(a_matrix, x)
        data = mat_vec(data_matrix, x)
        phase = 0
        if not zero_cubic:
            phase += sum(value * coordinate**3 for value, coordinate in zip(cubic_a, x))
        phase += sum(
            syndrome[i] * coupling[i][j] * data[j]
            for i in range(width) for j in range(width)
        )
        if not zero_cubic:
            phase += sum(value * coordinate**3 for value, coordinate in zip(cubic_b, data))
        phase -= sum(value * coordinate for value, coordinate in zip(output, x))
        total = k_add(total, zeta_power(phase))
    return canonical(total, width)


@dataclass
class Factor:
    scope: tuple[int, ...]
    values: list[K]

    def at(self, assignment: dict[int, int]) -> K:
        index = 0
        for variable in self.scope:
            index = index * P + assignment[variable]
        return self.values[index]


def min_fill_order(coefficients: list[int], width: int) -> tuple[list[int], int]:
    neighbors = {variable: set() for variable in range(width)}
    for coefficient, exponent in zip(coefficients, plan(width)):
        if not coefficient:
            continue
        scope = [variable for variable, power in enumerate(exponent) if power]
        for left in scope:
            neighbors[left].update(right for right in scope if right != left)
    order: list[int] = []
    treewidth = 0
    while neighbors:
        def score(variable: int) -> tuple[int, int, int]:
            adjacent = sorted(neighbors[variable])
            fills = sum(
                int(right not in neighbors[left])
                for i, left in enumerate(adjacent)
                for right in adjacent[i + 1:]
            )
            return fills, len(adjacent), variable
        variable = min(neighbors, key=score)
        adjacent = set(neighbors[variable])
        treewidth = max(treewidth, len(adjacent))
        for left in adjacent:
            neighbors[left].update(adjacent - {left})
            neighbors[left].discard(variable)
        del neighbors[variable]
        order.append(variable)
    return order, treewidth


def variable_elimination(coefficients: list[int], width: int) -> dict[str, object]:
    factors: list[Factor] = []
    for coefficient, exponent in zip(coefficients, plan(width)):
        if not coefficient:
            continue
        scope = tuple(variable for variable, power in enumerate(exponent) if power)
        values: list[K] = []
        for assignment in stream_assignments(len(scope)):
            phase = coefficient
            for variable, value in zip(scope, assignment):
                phase *= value ** exponent[variable]
            values.append(zeta_power(phase))
        factors.append(Factor(scope, values))
    order, predicted_treewidth = min_fill_order(coefficients, width)
    peak_table = max((len(factor.values) for factor in factors), default=1)
    peak_live = sum(len(factor.values) for factor in factors)
    multiplies = 0
    additions = 0
    observed_treewidth = 0
    for variable in order:
        related = [factor for factor in factors if variable in factor.scope]
        retained = [factor for factor in factors if variable not in factor.scope]
        union = tuple(sorted({item for factor in related for item in factor.scope}))
        output_scope = tuple(item for item in union if item != variable)
        observed_treewidth = max(observed_treewidth, len(union) - 1)
        output: list[K] = []
        for assignment_values in stream_assignments(len(output_scope)):
            assignment = dict(zip(output_scope, assignment_values))
            total = ZERO
            for value in range(P):
                assignment[variable] = value
                product = ONE
                for factor in related:
                    product = k_mul(product, factor.at(assignment))
                    multiplies += 1
                total = k_add(total, product)
                additions += 1
            output.append(total)
        generated = Factor(output_scope, output)
        peak_table = max(peak_table, len(output))
        peak_live = max(
            peak_live,
            sum(len(factor.values) for factor in factors) + len(output),
        )
        factors = retained + [generated]
    product = ONE
    for factor in factors:
        product = k_mul(product, factor.values[0])
        multiplies += 1
    value, exponent = canonical(product, width)
    return {
        "final_amplitude": {"numerator": list(value), "denominator_power5": exponent},
        "elimination_order": order,
        "predicted_treewidth": predicted_treewidth,
        "observed_treewidth": observed_treewidth,
        "peak_single_bag_field_cells": peak_table,
        "peak_live_factor_field_cells": peak_live,
        "field_multiplications": multiplies,
        "field_additions": additions,
        "inverse_or_restoration_work": 0,
        "stores_amplitude_vector": False,
        "stores_path_assignment_list": False,
    }


def quadratic_gauss(descriptor: dict[str, Any]) -> tuple[K, int]:
    canonical_value = canonical_descriptor(descriptor)
    width = int(canonical_value[0])
    a_matrix, b_matrix, coupling = [rows(values, width) for values in canonical_value[1:4]]
    output = list(canonical_value[6])
    data = mat_add(identity(width), mat_mul(b_matrix, a_matrix))
    raw = mat_mul(mat_mul(transpose(a_matrix), coupling), data)
    inv2 = pow(2, -1, P)
    quadratic = [
        [(raw[i][j] + raw[j][i]) * inv2 % P for j in range(width)]
        for i in range(width)
    ]
    inverse = inverse_matrix(quadratic)
    linear = [(-value) % P for value in output]
    completed = sum(
        linear[i] * inverse[i][j] * linear[j]
        for i in range(width) for j in range(width)
    ) % P
    phase = (-pow(4, -1, P) * completed) % P
    determinant_value = determinant(quadratic)
    character = 1 if determinant_value in (1, 4) else -1
    gauss_one = ZERO
    for value in range(P):
        gauss_one = k_add(gauss_one, zeta_power(value * value))
    result = k_mul(zeta_power(phase), k_pow(gauss_one, width))
    if character < 0:
        result = k_neg(result)
    return canonical(result, width)


class ReferenceCarrier:
    def __init__(self, width: int) -> None:
        self.width = width
        self.coefficients = [0] * len(plan(width))
        self.data = [0] * (width * width)
        self.syndrome = [0] * (width * width)
        self.workspace = [ZERO]
        self.descriptor_cells = [0] * (3 * width**2 + 3 * width)
        self.cursor = 0
        self.last_generation = 0
        self.leased_digest = ""

    def canonical_state(self) -> bool:
        return (
            not any(self.coefficients) and not any(self.data) and not any(self.syndrome)
            and self.workspace == [ZERO] and not any(self.descriptor_cells)
            and self.cursor == 0 and self.leased_digest == ""
        )

    def lease(self, descriptor: dict[str, Any], generation: int, supplied_digest: str) -> None:
        expected = digest(descriptor)
        if (
            not self.canonical_state() or generation != self.last_generation + 1
            or supplied_digest != expected
        ):
            raise RuntimeError("independent M247 lease rejected")
        canonical_value = canonical_descriptor(descriptor)
        flattened = [value for part in canonical_value[1:] for value in part]
        self.descriptor_cells[:] = flattened
        self.leased_digest = expected

    def require(self, descriptor: dict[str, Any]) -> None:
        if self.leased_digest != digest(descriptor):
            raise RuntimeError("independent M247 full descriptor mismatch")

    def _maps(self, descriptor: dict[str, Any]):
        canonical_value = canonical_descriptor(descriptor)
        return [rows(values, self.width) for values in canonical_value[1:4]]

    def action(self, descriptor: dict[str, Any], action: int, direction: int) -> None:
        if (direction == 1 and action != self.cursor) or (
            direction == -1 and action != self.cursor - 1
        ):
            raise RuntimeError("independent M247 action order rejected")
        width = self.width
        a_matrix, b_matrix, coupling = self._maps(descriptor)
        cubic_a = list(canonical_descriptor(descriptor)[4])
        cubic_b = list(canonical_descriptor(descriptor)[5])
        output = list(canonical_descriptor(descriptor)[6])
        data_rows = rows(self.data, width)
        syndrome_rows = rows(self.syndrome, width)
        if action == 0:
            for i in range(width): self.data[i * width + i] = (self.data[i * width + i] + direction) % P
        elif action == 1:
            product = mat_mul(a_matrix, data_rows)
            self.syndrome[:] = [(x + direction * y) % P for x, y in zip(self.syndrome, flat(product))]
        elif action == 2:
            cube_add(self.coefficients, identity(width), cubic_a, direction)
        elif action == 3:
            product = mat_mul(b_matrix, syndrome_rows)
            self.data[:] = [(x + direction * y) % P for x, y in zip(self.data, flat(product))]
        elif action == 4:
            quadratic_add(self.coefficients, syndrome_rows, coupling, data_rows, direction)
        elif action == 5:
            cube_add(self.coefficients, data_rows, cubic_b, direction)
        elif action == 6:
            product = mat_mul(b_matrix, syndrome_rows)
            self.data[:] = [(x - direction * y) % P for x, y in zip(self.data, flat(product))]
        elif action == 7:
            product = mat_mul(a_matrix, data_rows)
            self.syndrome[:] = [(x - direction * y) % P for x, y in zip(self.syndrome, flat(product))]
        elif action == 8:
            boundary_add(self.coefficients, output, direction)
        elif action == 9:
            for i in range(width): self.data[i * width + i] = (self.data[i * width + i] - direction) % P
        else:
            raise RuntimeError("unknown independent action")
        self.cursor += direction

    def forward(self, descriptor: dict[str, Any]) -> None:
        self.require(descriptor)
        for action in range(10):
            self.action(descriptor, action, 1)

    def reverse(self, descriptor: dict[str, Any]) -> None:
        self.require(descriptor)
        while self.cursor:
            self.action(descriptor, self.cursor - 1, -1)

    def release(self, generation: int) -> None:
        if any(self.coefficients) or any(self.data) or any(self.syndrome) or self.cursor:
            raise RuntimeError("independent M247 release before restoration")
        self.descriptor_cells[:] = [0] * len(self.descriptor_cells)
        self.leased_digest = ""
        self.last_generation = generation
        if not self.canonical_state():
            raise RuntimeError("independent M247 canonical release failure")

    def run(self, descriptor: dict[str, Any], generation: int) -> dict[str, object]:
        supplied = digest(descriptor)
        self.lease(descriptor, generation, supplied)
        ids = tuple(id(value) for value in (
            self.coefficients, self.data, self.syndrome, self.workspace, self.descriptor_cells
        ))
        self.forward(descriptor)
        expected = compile_coefficients(descriptor)
        if self.coefficients != expected or any(self.data) or any(self.syndrome):
            raise RuntimeError("independent forward signature mismatch")
        amplitude, exponent = direct_polynomial_sum(self.coefficients, self.width)
        self.reverse(descriptor)
        self.release(generation)
        current = tuple(id(value) for value in (
            self.coefficients, self.data, self.syndrome, self.workspace, self.descriptor_cells
        ))
        return {
            "width": self.width,
            "generation": generation,
            "final_amplitude": {"numerator": list(amplitude), "denominator_power5": exponent},
            "same_backings": ids == current,
            "canonical_after_restoration": self.canonical_state(),
            "baseline_reload_used": False,
        }


def reference_controls(descriptor: dict[str, Any]) -> dict[str, bool]:
    width = int(descriptor["width"])
    carrier = ReferenceCarrier(width)
    wrong_digest = False
    try:
        carrier.lease(descriptor, 1, "WRONG")
    except RuntimeError:
        wrong_digest = True
    changed = json.loads(json.dumps(descriptor))
    changed["C"][0][0] = (changed["C"][0][0] + 1) % P
    same_id_changed = False
    try:
        carrier.lease(changed, 1, digest(descriptor))
    except RuntimeError:
        same_id_changed = True
    carrier.lease(descriptor, 1, digest(descriptor))
    carrier.forward(descriptor)
    premature_release = False
    try:
        carrier.release(1)
    except RuntimeError:
        premature_release = True
    reordered = False
    try:
        carrier.action(descriptor, 8, -1)
    except RuntimeError:
        reordered = True
    carrier.reverse(descriptor)
    carrier.release(1)
    stale = False
    try:
        carrier.lease(descriptor, 1, digest(descriptor))
    except RuntimeError:
        stale = True
    regular = compile_coefficients(descriptor)
    swapped = compile_coefficients(descriptor, swapped=True)
    omitted = compile_coefficients(descriptor, omit_z=True)
    zero = compile_coefficients(descriptor, zero_cubic=True)
    return {
        "wrong_program_digest_rejected": wrong_digest,
        "same_id_changed_descriptor_rejected": same_id_changed,
        "missing_inverse_release_rejected": premature_release,
        "reordered_inverse_rejected_by_cursor": reordered,
        "stale_generation_rejected": stale,
        "x_then_z_differs_from_z_then_x": direct_polynomial_sum(regular, width) != direct_polynomial_sum(swapped, width),
        "omitting_shared_syndrome_z_consumer_changes_boundary": direct_polynomial_sum(regular, width) != direct_polynomial_sum(omitted, width),
        "zero_cubic_signature_degree_at_most_two": all(
            not coefficient or sum(exponent) <= 2
            for coefficient, exponent in zip(zero, plan(width))
        ),
    }


def main() -> None:
    public = json.load(sys.stdin)
    cases_by_id = public["cases"]
    schedule = [
        ("primary_w1", 1, "PRIMARY_W1"),
        ("primary_w2", 1, "PRIMARY_W2"),
        ("primary_w3", 1, "PRIMARY_W3"),
        ("primary_w4", 1, "PRIMARY_W4"),
    ]
    carriers = {width: ReferenceCarrier(width) for width in WIDTHS}
    cases: list[dict[str, object]] = []
    baselines: dict[str, object] = {}
    streamed_baselines: dict[str, object] = {}
    gate_oracles: dict[str, bool] = {}
    quadratic_shams: dict[str, object] = {}
    for case_id, generation, run_kind in schedule:
        descriptor = cases_by_id[case_id]["descriptor"]
        width = int(descriptor["width"])
        case = carriers[width].run(descriptor, generation)
        case["run_kind"] = run_kind
        cases.append(case)
        coefficients = compile_coefficients(descriptor)
        streamed_baselines[f"W{width}"] = {
            "final_amplitude": case["final_amplitude"],
            "assignment_terms": P**width,
            "nonzero_monomials_per_assignment": sum(bool(value) for value in coefficients),
            "monomial_evaluations": P**width * sum(bool(value) for value in coefficients),
            "peak_field_accumulator_cells": 1,
            "peak_assignment_cursor_residue_cells": width,
            "bag_table_field_cells": 0,
            "amplitude_vector_field_cells": 0,
            "inverse_or_restoration_work": 0,
        }
        baseline = variable_elimination(coefficients, width)
        if baseline["final_amplitude"] != case["final_amplitude"]:
            raise RuntimeError("independent variable-elimination mismatch")
        baselines[f"W{width}"] = baseline
        if width <= 2:
            gate_oracles[f"W{width}"] = gate_semantic_sum(descriptor) == (
                tuple(case["final_amplitude"]["numerator"]),
                case["final_amplitude"]["denominator_power5"],
            )
        zero_direct = gate_semantic_sum(descriptor, zero_cubic=True)
        zero_gauss = quadratic_gauss(descriptor)
        quadratic_shams[f"W{width}"] = {
            "direct_final_amplitude": {
                "numerator": list(zero_direct[0]), "denominator_power5": zero_direct[1]
            },
            "gauss_final_amplitude": {
                "numerator": list(zero_gauss[0]), "denominator_power5": zero_gauss[1]
            },
            "gauss_parity": zero_direct == zero_gauss,
            "cubic_absent": True,
        }
        if zero_direct != zero_gauss:
            raise RuntimeError("independent quadratic Gauss mismatch")

    reuse_descriptor = cases_by_id["reuse_w4"]["descriptor"]
    reuse = carriers[4].run(reuse_descriptor, 2)
    reuse["run_kind"] = "RESTORED_REUSE_W4"
    fresh_carrier = ReferenceCarrier(4)
    fresh = fresh_carrier.run(cases_by_id["reuse_fresh_w4"]["descriptor"], 1)
    fresh["run_kind"] = "FRESH_REUSE_REFERENCE_W4"
    cases.extend((reuse, fresh))
    if reuse["final_amplitude"] != fresh["final_amplitude"]:
        raise RuntimeError("independent restored/fresh mismatch")

    controls = reference_controls(cases_by_id["primary_w4"]["descriptor"])
    controls.update({
        "all_gate_semantic_oracles_match": all(gate_oracles.values()),
        "all_quadratic_gauss_shams_match": all(
            item["gauss_parity"] for item in quadratic_shams.values()
        ),
        "same_backing_restoration_and_reuse": all(case["same_backings"] for case in cases),
        "all_cases_canonical_after_restoration": all(
            case["canonical_after_restoration"] for case in cases
        ),
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in cases),
    })
    if not all(controls.values()):
        raise RuntimeError(f"independent M247 controls failed: {controls}")

    output = {
        "result": "PASS_SEPARATE_REFERENCE_M247_AFFINE_CUBIC_SIGNATURE",
        "cases": cases,
        "variable_elimination_baselines": baselines,
        "streamed_scalar_baselines": streamed_baselines,
        "quadratic_gauss_shams": quadratic_shams,
        "gate_semantic_oracle_parity": gate_oracles,
        "controls": controls,
        "imports_production_service_client_or_m237": False,
        "accepted_path_or_reference_stores5_to_width_amplitude_vector": False,
        "implemented_classical_variable_elimination_uses_width_dependent_factor_tables": True,
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
