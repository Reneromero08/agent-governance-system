#!/usr/bin/env python3
"""Exact resident cubic-strength phase-port separation-rank diagnostic.

The carrier stores a coherent strength port indexed by s in F_q together with
one two-fiber data port indexed by x in F_q.  The same unresolved s coordinate
is consumed by multiple controlled cubic phases separated by noncommuting
local Weil-Gaussian transforms.  This tests a runtime resident operator operand
rather than another operator word compiled entirely from public topology.

The implementation is direct-process exact finite-field software.  It makes no
CATVM custody, physical phase, advantage, or unbounded-computation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import growing_prime_cubic_weil_open_interface_action_span as open_action


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
FAMILIES = ("PRIMARY", "ALTERNATE")
BASE_DEPTH = 1
DEEPEST_ORDER = 113
DEEPEST_DEPTH = 2
ALTERNATE_ORDER = 41
ALTERNATE_DEPTH = 2
CLAIM = (
    "BOUNDED_EXACT_GROWING_SAFE_PRIME_RESIDENT_COHERENT_CUBIC_STRENGTH_PHASE_"
    "PORT_IS_SHARED_ACROSS_MULTIPLE_NONCOMMUTING_WEIL_CONSUMERS_AND_REACHES_"
    "FULL_Q_LATENT_DATA_SEPARATION_RANK_AFTER_THE_FIRST_CONTROLLED_CUBIC_WITH_"
    "FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_MATCHED_DENSE_AND_"
    "EXACT_RADER_NTT_CLASSICAL_RECURRENCES_RETAIN_THE_SAME2Q2_LEADING_STATE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class Operation:
    kind: str
    payload: tuple[int, ...]


@dataclass
class Work:
    controlled_phase_evaluations: int = 0
    controlled_field_multiplications: int = 0
    data_gaussian_kernel_evaluations: int = 0
    data_gaussian_field_multiply_adds: int = 0
    latent_gaussian_kernel_evaluations: int = 0
    latent_gaussian_field_multiply_adds: int = 0
    fiber_field_multiply_adds: int = 0
    projection_field_multiply_adds: int = 0
    inverse_gaussian_scalar_rematerializations: int = 0
    temporary_vector_field_cells_peak: int = 0


@dataclass
class Carrier:
    field: open_action.cubic.gaussian.Field
    cells: list[int]
    fixture_family: str
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(
        cls,
        q: int,
        fixture_family: str,
        latent_override: list[int] | None = None,
    ) -> "Carrier":
        if fixture_family not in FAMILIES:
            fail("invalid fixture family")
        field = open_action.cubic.gaussian.make_field(q)
        latent = latent_override or latent_fixture(field, fixture_family)
        if len(latent) != q or not any(value % field.p for value in latent):
            fail("invalid resident latent port")
        return cls(field, product_cells(field, fixture_family, latent), fixture_family)

    @property
    def q(self) -> int:
        return self.field.q

    def backing_ids(self) -> tuple[int, int]:
        return id(self), id(self.cells)

    def payload_state(self) -> tuple[Any, ...]:
        return self.q, self.field.p, self.fixture_family, tuple(self.cells), self.stage


def phase(field: open_action.cubic.gaussian.Field, exponent: int) -> int:
    return open_action.cubic.gaussian.phase(field, exponent)


def latent_fixture(field: open_action.cubic.gaussian.Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [phase(field, (code + 1) * s * s + (2 * code + 1) * s + code) for s in range(field.q)]


def data_fixture(field: open_action.cubic.gaussian.Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [
        (fiber + 1)
        * phase(
            field,
            (fiber + code + 1) * x * x + (3 * fiber + 2 * code + 1) * x,
        )
        % field.p
        for fiber in range(2)
        for x in range(field.q)
    ]


def product_cells(
    field: open_action.cubic.gaussian.Field,
    family: str,
    latent: list[int],
) -> list[int]:
    data = data_fixture(field, family)
    return [latent[s] * data[fiber * field.q + x] % field.p for s in range(field.q) for fiber in range(2) for x in range(field.q)]


def offset(q: int, s: int, fiber: int, x: int) -> int:
    return (2 * s + fiber) * q + x


def public_plan(q: int, depth: int, family: str) -> list[Operation]:
    if depth < 1 or family not in FAMILIES:
        fail("invalid public plan")
    code = 1 if family == "PRIMARY" else 2
    operations: list[Operation] = []
    for layer in range(depth):
        parameter = (layer + code) % q or 1
        data_symplectic = (1, 1, parameter, parameter + 1)
        latent_symplectic = (parameter + 1, 1, parameter, 1)
        first = Operation("CONTROLLED_CUBIC", ((2 * layer + code) % q or 1,))
        second = Operation("CONTROLLED_CUBIC", ((3 * layer + 2 * code + 1) % q or 1,))
        data = Operation("DATA_GAUSSIAN", data_symplectic + (1,))
        latent = Operation("LATENT_GAUSSIAN", latent_symplectic + (1,))
        if family == "PRIMARY":
            operations.extend((first, data, second, latent))
        else:
            operations.extend((first, latent, second, data))
    operations.append(Operation("FIBER", (1, 1, 1, 2)))
    return operations


def gaussian_vector(
    vector: list[int],
    payload: tuple[int, ...],
    field: open_action.cubic.gaussian.Field,
    work: Work,
    axis: str,
) -> list[int]:
    q, p = field.q, field.p
    coefficient = payload[4] % p
    symplectic = list(payload[:4])
    output = [0] * q
    work.temporary_vector_field_cells_peak = max(work.temporary_vector_field_cells_peak, q)
    for x in range(q):
        total = 0
        for y in range(q):
            total += coefficient * open_action.cubic.gaussian.kernel_value(symplectic, x, y, field) * vector[y]
        output[x] = total % p
    if axis == "DATA":
        work.data_gaussian_kernel_evaluations += q * q
        work.data_gaussian_field_multiply_adds += q * q
    elif axis == "LATENT":
        work.latent_gaussian_kernel_evaluations += q * q
        work.latent_gaussian_field_multiply_adds += q * q
    else:
        fail("invalid Gaussian axis")
    return output


def apply_controlled(carrier: Carrier, strength: int, work: Work) -> None:
    q, p = carrier.q, carrier.field.p
    for s in range(q):
        for x in range(q):
            multiplier = phase(carrier.field, strength * s * x * x * x)
            work.controlled_phase_evaluations += 1
            for fiber in range(2):
                index = offset(q, s, fiber, x)
                carrier.cells[index] = carrier.cells[index] * multiplier % p
                work.controlled_field_multiplications += 1


def apply_data_gaussian(carrier: Carrier, payload: tuple[int, ...], work: Work) -> None:
    q = carrier.q
    for s in range(q):
        for fiber in range(2):
            start = offset(q, s, fiber, 0)
            vector = carrier.cells[start : start + q]
            carrier.cells[start : start + q] = gaussian_vector(vector, payload, carrier.field, work, "DATA")


def apply_latent_gaussian(carrier: Carrier, payload: tuple[int, ...], work: Work) -> None:
    q = carrier.q
    for fiber in range(2):
        for x in range(q):
            vector = [carrier.cells[offset(q, s, fiber, x)] for s in range(q)]
            output = gaussian_vector(vector, payload, carrier.field, work, "LATENT")
            for s, value in enumerate(output):
                carrier.cells[offset(q, s, fiber, x)] = value


def apply_fiber(carrier: Carrier, payload: tuple[int, ...], work: Work) -> None:
    q, p = carrier.q, carrier.field.p
    a, b, c, d = payload
    for s in range(q):
        for x in range(q):
            left_index = offset(q, s, 0, x)
            right_index = offset(q, s, 1, x)
            left, right = carrier.cells[left_index], carrier.cells[right_index]
            carrier.cells[left_index] = (a * left + c * right) % p
            carrier.cells[right_index] = (b * left + d * right) % p
            work.fiber_field_multiply_adds += 4


def apply_operation(carrier: Carrier, operation: Operation, work: Work) -> None:
    if operation.kind == "CONTROLLED_CUBIC":
        apply_controlled(carrier, operation.payload[0], work)
    elif operation.kind == "DATA_GAUSSIAN":
        apply_data_gaussian(carrier, operation.payload, work)
    elif operation.kind == "LATENT_GAUSSIAN":
        apply_latent_gaussian(carrier, operation.payload, work)
    elif operation.kind == "FIBER":
        apply_fiber(carrier, operation.payload, work)
    else:
        fail("unknown operation")


def inverse_operation(carrier: Carrier, operation: Operation, work: Work, wrong: bool = False) -> Operation:
    if operation.kind == "CONTROLLED_CUBIC":
        return Operation(operation.kind, ((-operation.payload[0] + int(wrong)) % carrier.q,))
    if operation.kind in ("DATA_GAUSSIAN", "LATENT_GAUSSIAN"):
        inverse_work = open_action.Work()
        payload = list(open_action.gaussian_payload_inverse(carrier.field, operation.payload, inverse_work))
        work.inverse_gaussian_scalar_rematerializations += inverse_work.inverse_gaussian_scalar_rematerializations
        if wrong:
            payload[-1] = (payload[-1] + 1) % carrier.field.p or 1
        return Operation(operation.kind, tuple(payload))
    if operation.kind == "FIBER":
        payload = open_action.cubic.gaussian.fiber_inverse(list(operation.payload), carrier.field.p)
        if wrong:
            payload[0] = (payload[0] + 1) % carrier.field.p
        return Operation(operation.kind, tuple(payload))
    fail("unknown inverse operation")


def separation_rank(carrier: Carrier) -> int:
    q, p = carrier.q, carrier.field.p
    columns = 2 * q
    matrix = [carrier.cells[offset(q, s, 0, 0) : offset(q, s, 0, 0) + columns] for s in range(q)]
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, q) if matrix[row][column] % p), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = pow(matrix[pivot_row][column], -1, p)
        matrix[pivot_row] = [value * scale % p for value in matrix[pivot_row]]
        for row in range(q):
            if row == pivot_row or not matrix[row][column] % p:
                continue
            factor = matrix[row][column] % p
            matrix[row] = [(left - factor * right) % p for left, right in zip(matrix[row], matrix[pivot_row])]
        pivot_row += 1
        if pivot_row == q:
            break
    return pivot_row


def forward(
    carrier: Carrier,
    depth: int,
    family: str,
    reordered: bool = False,
) -> tuple[list[Operation], Work, list[int]]:
    if carrier.stage != "IDLE":
        fail("carrier is not idle")
    operations = public_plan(carrier.q, depth, family)
    if reordered:
        operations = list(reversed(operations))
    work = Work()
    controlled_ranks: list[int] = []
    for operation in operations:
        apply_operation(carrier, operation, work)
        if operation.kind == "CONTROLLED_CUBIC":
            controlled_ranks.append(separation_rank(carrier))
    carrier.stage = "FORWARD_COMPLETE"
    return operations, work, controlled_ranks


def reverse(
    carrier: Carrier,
    operations: list[Operation],
    work: Work,
    mutation: str | None = None,
) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("carrier lacks forward state")
    sequence = list(reversed(operations))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(operations)
    for index, operation in enumerate(sequence):
        inverse = inverse_operation(carrier, operation, work, mutation == "WRONG" and index == 0)
        apply_operation(carrier, inverse, work)
    carrier.stage = "IDLE"


def probes(q: int, family: str) -> tuple[tuple[int, int, int, int], ...]:
    code = 1 if family == "PRIMARY" else 2
    return (
        ((3 * code + 1) % q, 0, (5 * code + 2) % q, 1),
        ((7 * code + 2) % q, 1, (11 * code + 3) % q, 2),
        ((13 * code + 1) % q, 0, (17 * code + 4) % q, 3),
        ((19 * code + 2) % q, 1, (23 * code + 5) % q, 5),
    )


def boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary unavailable")
    return sum(
        weight * carrier.cells[offset(carrier.q, s, fiber, x)]
        for s, fiber, x, weight in probes(carrier.q, family)
    ) % carrier.field.p


def commitment(carrier: Carrier) -> str:
    return hashlib.sha256(repr((carrier.q, tuple(carrier.cells))).encode("ascii")).hexdigest()


def classical_gaussian_vector(
    vector: list[int],
    payload: tuple[int, ...],
    field: open_action.cubic.gaussian.Field,
    work: dict[str, int],
    axis: str,
) -> list[int]:
    q, p = field.q, field.p
    output = []
    for x in range(q):
        output.append(
            payload[4]
            * sum(
                open_action.cubic.gaussian.kernel_value(list(payload[:4]), x, y, field) * vector[y]
                for y in range(q)
            )
            % p
        )
    work[f"{axis.lower()}_gaussian_kernel_evaluations"] += q * q
    work[f"{axis.lower()}_gaussian_field_multiply_adds"] += q * q
    return output


def matched_classical_boundary(
    restored_cells: list[int],
    field: open_action.cubic.gaussian.Field,
    depth: int,
    family: str,
) -> dict[str, Any]:
    q, p = field.q, field.p
    values = restored_cells.copy()
    work = {
        "controlled_phase_evaluations": 0,
        "controlled_field_multiplications": 0,
        "data_gaussian_kernel_evaluations": 0,
        "data_gaussian_field_multiply_adds": 0,
        "latent_gaussian_kernel_evaluations": 0,
        "latent_gaussian_field_multiply_adds": 0,
        "fiber_field_multiply_adds": 0,
        "projection_field_multiply_adds": 0,
        "inverse_gaussian_scalar_rematerializations": 0,
        "temporary_vector_field_cells_peak": q,
    }
    for operation in public_plan(q, depth, family):
        if operation.kind == "CONTROLLED_CUBIC":
            strength = operation.payload[0]
            for s in range(q):
                for x in range(q):
                    multiplier = phase(field, strength * s * x * x * x)
                    work["controlled_phase_evaluations"] += 1
                    for fiber in range(2):
                        index = offset(q, s, fiber, x)
                        values[index] = values[index] * multiplier % p
                        work["controlled_field_multiplications"] += 1
        elif operation.kind == "DATA_GAUSSIAN":
            for s in range(q):
                for fiber in range(2):
                    start = offset(q, s, fiber, 0)
                    values[start : start + q] = classical_gaussian_vector(
                        values[start : start + q], operation.payload, field, work, "DATA"
                    )
        elif operation.kind == "LATENT_GAUSSIAN":
            for fiber in range(2):
                for x in range(q):
                    vector = [values[offset(q, s, fiber, x)] for s in range(q)]
                    output = classical_gaussian_vector(vector, operation.payload, field, work, "LATENT")
                    for s, value in enumerate(output):
                        values[offset(q, s, fiber, x)] = value
        else:
            a, b, c, d = operation.payload
            for s in range(q):
                for x in range(q):
                    left_index, right_index = offset(q, s, 0, x), offset(q, s, 1, x)
                    left, right = values[left_index], values[right_index]
                    values[left_index] = (a * left + c * right) % p
                    values[right_index] = (b * left + d * right) % p
                    work["fiber_field_multiply_adds"] += 4
    projected = sum(weight * values[offset(q, s, fiber, x)] for s, fiber, x, weight in probes(q, family)) % p
    work["projection_field_multiply_adds"] = 8
    return {
        "boundary": projected,
        "semantic_commitment": hashlib.sha256(repr((q, tuple(values))).encode("ascii")).hexdigest(),
        "resident_field_cells": 2 * q * q,
        "temporary_field_cells_peak": q,
        "resident_plus_temporary_peak_field_cells": 2 * q * q + q,
        "experimental_comparison_concurrent_restored_phase_plus_classical_peak_field_cells": 4 * q * q + q,
        "public_plan_cells": 12 * depth + 4,
        "public_boundary_probe_cells": 16,
        "work": work,
        "cold_start_comparison_used": False,
    }


def matched_rader_ntt_classical_boundary(
    restored_cells: list[int],
    field: open_action.cubic.gaussian.Field,
    depth: int,
    family: str,
) -> dict[str, Any]:
    q, p = field.q, field.p
    values = restored_cells.copy()
    work = {key: 0 for key in (
        "chirp_field_multiplications",
        "controlled_field_multiplications",
        "controlled_phase_evaluations",
        "cubic_field_multiplications",
        "cubic_phase_evaluations",
        "exact_convolution_coefficient_bound_peak",
        "fiber_field_multiply_adds",
        "ntt_butterflies",
        "ntt_max_length",
        "ntt_pointwise_multiplications",
        "ntt_temporary_integer_cells_peak",
        "ntt_transforms",
        "projection_field_multiply_adds",
        "rader_dfts",
        "sparse_gaussian_terms",
    )}
    scratch = open_action.LiveCellLedger()

    def transform(vector: list[int], payload: tuple[int, ...]) -> list[int]:
        scratch.allocate("FIELD", q)
        output = open_action.apply_single_gaussian_rader(vector, payload, field, work, scratch)
        scratch.release("FIELD", q)
        vector.clear()
        return output

    for operation in public_plan(q, depth, family):
        if operation.kind == "CONTROLLED_CUBIC":
            strength = operation.payload[0]
            for s in range(q):
                for x in range(q):
                    multiplier = phase(field, strength * s * x * x * x)
                    work["controlled_phase_evaluations"] += 1
                    for fiber in range(2):
                        index = offset(q, s, fiber, x)
                        values[index] = values[index] * multiplier % p
                        work["controlled_field_multiplications"] += 1
        elif operation.kind == "DATA_GAUSSIAN":
            for s in range(q):
                for fiber in range(2):
                    start = offset(q, s, fiber, 0)
                    output = transform(values[start : start + q], operation.payload)
                    values[start : start + q] = output
                    scratch.release("FIELD", q)
                    output.clear()
        elif operation.kind == "LATENT_GAUSSIAN":
            for fiber in range(2):
                for x in range(q):
                    vector = [values[offset(q, s, fiber, x)] for s in range(q)]
                    output = transform(vector, operation.payload)
                    for s, value in enumerate(output):
                        values[offset(q, s, fiber, x)] = value
                    scratch.release("FIELD", q)
                    output.clear()
        else:
            a, b, c, d = operation.payload
            for s in range(q):
                for x in range(q):
                    left_index, right_index = offset(q, s, 0, x), offset(q, s, 1, x)
                    left, right = values[left_index], values[right_index]
                    values[left_index] = (a * left + c * right) % p
                    values[right_index] = (b * left + d * right) % p
                    work["fiber_field_multiply_adds"] += 4
    scratch.require_empty()
    projected = sum(weight * values[offset(q, s, fiber, x)] for s, fiber, x, weight in probes(q, family)) % p
    work["projection_field_multiply_adds"] = 8
    resident = 2 * q * q
    return {
        "boundary": projected,
        "semantic_commitment": hashlib.sha256(repr((q, tuple(values))).encode("ascii")).hexdigest(),
        "resident_field_cells": resident,
        "scratch_field_cells_peak": scratch.field_cells_peak,
        "scratch_auxiliary_integer_cells_peak": scratch.auxiliary_integer_cells_peak,
        "scratch_logical_payload_cells_peak": scratch.material_cells_peak,
        "resident_plus_scratch_logical_payload_cells_peak": resident + scratch.material_cells_peak,
        "experimental_comparison_concurrent_restored_phase_plus_classical_peak_logical_payload_cells": (
            2 * resident + scratch.material_cells_peak
        ),
        "field_cells_at_combined_peak": resident + scratch.field_cells_at_material_peak,
        "auxiliary_integer_cells_at_combined_peak": scratch.auxiliary_integer_cells_at_material_peak,
        "field_cell_bit_capacity": p.bit_length(),
        "auxiliary_integer_cell_bit_capacity": open_action.NTT_PRIME.bit_length(),
        "combined_payload_bit_capacity_upper_bound_at_peak": (
            (resident + scratch.field_cells_at_material_peak) * p.bit_length()
            + scratch.auxiliary_integer_cells_at_material_peak * open_action.NTT_PRIME.bit_length()
        ),
        "public_plan_cells": 12 * depth + 4,
        "public_boundary_probe_cells": 16,
        "retained_ntt_kernel_cache_cells": 0,
        "auxiliary_ntt_prime": open_action.NTT_PRIME,
        "single_auxiliary_modulus_exactness_bound_checked": (
            0 < work["exact_convolution_coefficient_bound_peak"] < open_action.NTT_PRIME
        ),
        "work": work,
        "cold_start_comparison_used": False,
    }


def transaction(carrier: Carrier, depth: int, family: str) -> dict[str, Any]:
    q, p = carrier.q, carrier.field.p
    expected = product_cells(carrier.field, carrier.fixture_family, latent_fixture(carrier.field, carrier.fixture_family))
    if carrier.cells != expected or carrier.stage != "IDLE":
        fail("carrier does not match declared initial fixture")
    backing = carrier.backing_ids()
    generation = carrier.restoration_generation
    initial_rank = separation_rank(carrier)
    operations, work, controlled_ranks = forward(carrier, depth, family)
    projected = boundary(carrier, family)
    work.projection_field_multiply_adds += 8
    final_rank = separation_rank(carrier)
    semantic_commitment = commitment(carrier)
    forward_work = work.__dict__.copy()
    reverse(carrier, operations, work)
    exact = carrier.cells == expected and carrier.stage == "IDLE"
    same_backing = carrier.backing_ids() == backing
    generation_stable = carrier.restoration_generation == generation
    baseline = matched_classical_boundary(carrier.cells, carrier.field, depth, family)
    rader_baseline = matched_rader_ntt_classical_boundary(carrier.cells, carrier.field, depth, family)
    if not all((
        initial_rank == 1,
        controlled_ranks,
        all(rank == q for rank in controlled_ranks),
        final_rank == q,
        exact,
        same_backing,
        generation_stable,
        projected == baseline["boundary"],
        projected == rader_baseline["boundary"],
        semantic_commitment == baseline["semantic_commitment"],
        semantic_commitment == rader_baseline["semantic_commitment"],
        forward_work == baseline["work"],
    )):
        fail(
            f"resident strength qualification failed q={q} depth={depth} family={family} "
            f"initial={initial_rank} controlled={controlled_ranks} final={final_rank} "
            f"exact={exact} backing={same_backing} generation={generation_stable} "
            f"boundary={projected}/{baseline['boundary']}/{rader_baseline['boundary']}"
        )
    carrier.restoration_generation += 1
    return {
        "q": q,
        "p": p,
        "depth": depth,
        "family": family,
        "fixture_family": carrier.fixture_family,
        "boundary": projected,
        "semantic_commitment": semantic_commitment,
        "initial_latent_data_separation_rank": initial_rank,
        "controlled_cubic_separation_ranks": controlled_ranks,
        "final_latent_data_separation_rank": final_rank,
        "all_controlled_cubic_separation_ranks_full_q": all(rank == q for rank in controlled_ranks),
        "cube_map_bijective_on_fq": math.gcd(3, q - 1) == 1 and len({pow(x, 3, q) for x in range(q)}) == q,
        "shared_latent_port_consumers": 2 * depth,
        "accepted_resident_field_cells": 2 * q * q,
        "accepted_temporary_field_cells_peak": q,
        "accepted_resident_plus_temporary_peak_field_cells": 2 * q * q + q,
        "public_plan_cells": 12 * depth + 4,
        "public_boundary_probe_cells": 16,
        "retained_inverse_history_field_cells": 0,
        "restoration_verification_reference_field_cells": 2 * q * q,
        "separation_rank_verification_copy_field_cells": 2 * q * q,
        "separation_rank_verification_row_temporary_field_cells": 2 * q,
        "forward_work": forward_work,
        "full_lifecycle_work": work.__dict__,
        "matched_identical_matrix_free_classical": {
            **baseline,
            "boundary_matches": True,
            "full_state_commitment_matches": True,
            "forward_work_matches": True,
        },
        "matched_exact_rader_ntt_classical": {
            **rader_baseline,
            "boundary_matches": True,
            "full_state_commitment_matches": True,
        },
        "exact_canonical_payload_restored": exact,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "resident_latent_amplitudes_serialized": False,
        "intermediate_joint_state_serialized": False,
    }


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except (RuntimeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    q, depth, family = 5, 1, "PRIMARY"

    def restoration_mutation(mutation: str) -> bool:
        carrier = Carrier.seal(q, family)
        expected = carrier.cells.copy()
        operations, work, _ = forward(carrier, depth, family)
        reverse(carrier, operations, work, mutation)
        return carrier.cells != expected or carrier.stage != "IDLE"

    normal = Carrier.seal(q, family)
    normal_operations, normal_work, _ = forward(normal, depth, family)
    normal_boundary = boundary(normal, family)
    reverse(normal, normal_operations, normal_work)
    reordered = Carrier.seal(q, family)
    forward(reordered, depth, family, reordered=True)
    reordered_boundary = boundary(reordered, family)
    one_hot = [1] + [0] * (q - 1)
    collapsed = Carrier.seal(q, family, one_hot)
    collapsed_plan, collapsed_work, collapsed_ranks = forward(collapsed, depth, family)
    reverse(collapsed, collapsed_plan, collapsed_work)
    return {
        "missing_inverse_fails_restoration": restoration_mutation("MISSING"),
        "wrong_inverse_fails_restoration": restoration_mutation("WRONG"),
        "reordered_inverse_fails_restoration": restoration_mutation("REORDER"),
        "module_reordering_changes_final_boundary": reordered_boundary != normal_boundary,
        "premature_projection_rejected": raises(lambda: boundary(Carrier.seal(q, family), family)),
        "invalid_family_rejected": raises(lambda: public_plan(q, depth, "INVALID")),
        "zero_depth_rejected": raises(lambda: public_plan(q, 0, family)),
        "null_latent_port_rejected": raises(lambda: Carrier.seal(q, family, [0] * q)),
        "one_hot_latent_semantic_perturbation_does_not_fake_full_rank": bool(collapsed_ranks) and collapsed_ranks[0] < q,
        "public_plan_independent_of_latent_amplitudes": (
            normal.cells != collapsed.cells
            and public_plan(normal.q, depth, family) == public_plan(collapsed.q, depth, family)
        ),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal(23, "PRIMARY")
    backing = carrier.backing_ids()
    initial = carrier.payload_state()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 1, "ALTERNATE")
    fresh = transaction(Carrier.seal(23, "PRIMARY"), 1, "ALTERNATE")
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "fresh_second_boundary": fresh["boundary"],
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_commitment_matches_fresh": second["semantic_commitment"] == fresh["semantic_commitment"],
        "same_backing_reused": carrier.backing_ids() == backing,
        "exact_payload_state_restored_after_reuse": carrier.payload_state() == initial,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def case_specs() -> list[tuple[int, int, str]]:
    cases = [(q, BASE_DEPTH, "PRIMARY") for q in ORDERS]
    cases.append((DEEPEST_ORDER, DEEPEST_DEPTH, "PRIMARY"))
    cases.append((ALTERNATE_ORDER, ALTERNATE_DEPTH, "ALTERNATE"))
    return cases


def build_result() -> dict[str, Any]:
    cases = [transaction(Carrier.seal(q, family), depth, family) for q, depth, family in case_specs()]
    controls_result = controls()
    reuse = reuse_check()
    result = {
        "schema": "CAT_CAS_GROWING_PRIME_RESIDENT_CUBIC_STRENGTH_PHASE_PORT_WEIL_ACTION_RANK_RESULTS_V1",
        "claim_candidate": CLAIM,
        "claim_ceiling": "SAFE_PRIME_PAIRS_Q5_11_23_29_41_53_83_89_113_P11_23_47_59_83_107_167_179_227_PRIMARY_DEPTH1_ALL_PRIMARY_DEPTH2_Q113_ALTERNATE_DEPTH2_Q41_TWO_FIBER_DIRECT_PROCESS_SOFTWARE_AUXILIARY_NTT_MODULUS998244353",
        "cases": cases,
        "controls": controls_result,
        "restoration_and_reuse": reuse,
        "observed_resource_law": {
            "initial_product_separation_rank": 1,
            "all_first_controlled_cubic_separation_ranks": "Q",
            "all_declared_final_separation_ranks": "Q",
            "accepted_resident_field_cells": "2*Q^2",
            "accepted_resident_plus_temporary_peak_field_cells": "2*Q^2+Q",
            "matched_classical_resident_field_cells": "2*Q^2",
            "matched_classical_resident_plus_temporary_peak_field_cells": "2*Q^2+Q",
            "matched_rader_ntt_resident_plus_scratch_peak_logical_payload_cells": "2*Q^2+4*Q-2+2*NEXT_POWER_OF_TWO_AT_LEAST_2Q_MINUS3",
            "retained_inverse_history_field_cells": 0,
            "fixed_rank_across_growing_q_established": False,
        },
        "matched_baseline": {
            "executed_pareto_pair": [
                "IDENTICAL_MINIMUM_SCRATCH_DENSE_MATRIX_FREE_TWO_AXIS_GAUSSIAN_AND_CONTROLLED_CUBIC_CLASSICAL_RECURRENCE",
                "EXACT_WORK_REDUCED_RADER_NTT_MATRIX_FREE_TWO_AXIS_GAUSSIAN_AND_CONTROLLED_CUBIC_CLASSICAL_RECURRENCE_WITH_COUNTED_MIXED_WIDTH_SCRATCH"
            ],
            "all_case_boundaries_match": all(case["matched_identical_matrix_free_classical"]["boundary_matches"] for case in cases),
            "all_case_full_state_commitments_match": all(
                case["matched_identical_matrix_free_classical"]["full_state_commitment_matches"]
                and case["matched_exact_rader_ntt_classical"]["full_state_commitment_matches"]
                for case in cases
            ),
            "all_rader_ntt_boundaries_match": all(case["matched_exact_rader_ntt_classical"]["boundary_matches"] for case in cases),
            "same_leading_2q2_resident_field_cell_law": True,
            "cold_start_comparison_used": False,
        },
        "resource_accounting": {
            "carrier_creation_cells_counted": True,
            "public_plan_and_boundary_probe_cells_counted": True,
            "native_forward_inverse_projection_and_verification_work_counted": True,
            "restoration_verification_reference_and_rank_copy_cells_reported_separately": True,
            "matched_baseline_executed_after_actual_restoration_from_a_copy_of_the_restored_carrier": True,
            "dense_forward_work_tuple_matches_phase_forward_work_tuple": True,
            "rader_ntt_mixed_width_scratch_and_exactness_bound_counted": True,
            "controller_backend_traffic_bytes": 0,
            "snapshot_cells": 0,
            "python_object_headers_list_spare_capacity_allocator_interpreter_native_libraries_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "next_obstruction": "A_GENUINELY_RESIDENT_SHARED_CUBIC_STRENGTH_PORT_REACHES_FULL_Q_LATENT_DATA_SEPARATION_RANK_AFTER_ONE_CONTROLLED_INTERACTION_AND_THE_EXACT_PHASE_CARRIER_HAS_DENSE_AND_WORK_REDUCED_CLASSICAL_RECURRENCES_WITH_THE_SAME2Q2_LEADING_RESIDENT_STATE",
        "claim_boundaries": {
            "runtime_operand_machine_enforced_hidden": False,
            "general_open_relation_closure": False,
            "rank_lower_bound_against_nonlinear_or_program_specific_algorithms": False,
            "fixed_bit_width_across_unbounded_q": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage_or_small_wall_crossing": False,
            "physical_waveform_or_silicon_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
    }
    if not all(controls_result.values()):
        fail("control failure")
    if not all((
        all(case["cube_map_bijective_on_fq"] for case in cases),
        all(case["all_controlled_cubic_separation_ranks_full_q"] for case in cases),
        all(case["exact_canonical_payload_restored"] and case["same_backing_restored"] for case in cases),
        reuse["second_boundary_matches_fresh"],
        reuse["second_commitment_matches_fresh"],
        reuse["same_backing_reused"],
        reuse["exact_payload_state_restored_after_reuse"],
    )):
        fail("qualification failure")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
