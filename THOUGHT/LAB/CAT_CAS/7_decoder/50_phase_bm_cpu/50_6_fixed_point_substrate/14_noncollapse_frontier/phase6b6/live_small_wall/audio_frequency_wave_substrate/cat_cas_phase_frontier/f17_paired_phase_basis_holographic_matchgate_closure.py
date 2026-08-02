#!/usr/bin/env python3
"""Bounded paired-phase-basis holographic matchgate closure over F17.

Compact native exact-one signature generators live in paired zeta_17 bases
on the two sides of an even square grid.  The actual resident bases obey
T S^T = I, so each shared edge index closes without a signature table.  The
surviving exact-one cores form a weighted planar perfect-matching problem,
which is projected by a public Kasteleyn-signed determinant.

The native degree-four signatures have nonzero entries in both fermion
parities and therefore are not parity-preserving matchgate signatures in
their resident basis.
The paired phase transform, rather than native Gaussianity, exposes the
matchgate closure.  This is exact direct-process software.  The identical
holographic transform and determinant are the strongest compact classical
baseline, so no distinct phase resource or computational advantage is
claimed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import f17_nonlinear_canonical_mps_separator_chart as backend


PRIME = 17
EXACT_SIZES = (2, 4, 6)
STRUCTURAL_SIZES = (2, 4, 6, 8, 10, 12)
FAMILIES = ("PRIMARY", "REUSE")
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def integer(alg: backend.Algebra, value: int) -> Any:
    if alg.modulus:
        return value % alg.modulus
    return alg.domain.convert(value)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def grid_vertices(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((row, column) for row in range(n) for column in range(n))


def grid_edges(n: int) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return (
        *(((row, column), (row, column + 1))
          for row in range(n) for column in range(n - 1)),
        *(((row, column), (row + 1, column))
          for row in range(n - 1) for column in range(n)),
    )


def incident_edge_indices(n: int) -> dict[tuple[int, int], tuple[int, ...]]:
    result: dict[tuple[int, int], list[int]] = {
        vertex: [] for vertex in grid_vertices(n)
    }
    for index, (left, right) in enumerate(grid_edges(n)):
        result[left].append(index)
        result[right].append(index)
    return {vertex: tuple(indices) for vertex, indices in result.items()}


@dataclass(frozen=True)
class HolographicProgram:
    n: int
    family: str
    basis_exponent: int
    edge_exponents: tuple[int, ...]

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "n": self.n,
                "family": self.family,
                "basis_exponent": self.basis_exponent,
                "edge_exponents": self.edge_exponents,
            }
        )


def compile_program(n: int, family: str) -> HolographicProgram:
    if n not in STRUCTURAL_SIZES:
        fail("grid size is outside the declared paired-basis scope")
    if family not in FAMILIES:
        fail("unknown paired-basis family")
    variant = 0 if family == "PRIMARY" else 1
    edges = grid_edges(n)
    program = HolographicProgram(
        n=n,
        family=family,
        basis_exponent=3 + 2 * variant,
        edge_exponents=tuple(
            1 + ((7 * index + 3 * n + 5 * variant) % 16)
            for index in range(len(edges))
        ),
    )
    validate_program(program)
    return program


def validate_program(program: HolographicProgram) -> None:
    if program.n not in STRUCTURAL_SIZES or program.n % 2:
        fail("paired-basis closure requires a declared positive even grid")
    if program.family not in FAMILIES:
        fail("paired-basis family changed")
    if not 1 <= program.basis_exponent < PRIME:
        fail("basis exponent must name a nontrivial F17 phase")
    if len(program.edge_exponents) != len(grid_edges(program.n)):
        fail("edge exponent arity changed")
    if not all(1 <= value < PRIME for value in program.edge_exponents):
        fail("edge weights must be nontrivial F17 phases")


@dataclass(frozen=True)
class ResidentFactors:
    left_basis: tuple[Any, Any, Any, Any]
    right_basis: tuple[Any, Any, Any, Any]
    edge_weights: tuple[Any, ...]

    def flattened(self) -> list[Any]:
        return [*self.left_basis, *self.right_basis, *self.edge_weights]


def compile_factors(
    program: HolographicProgram,
    alg: backend.Algebra,
) -> ResidentFactors:
    validate_program(program)
    phase = alg.power(program.basis_exponent)
    inverse_phase = alg.power(-program.basis_exponent)
    half = alg.divide(alg.one, integer(alg, 2))
    left = (alg.one, alg.one, phase, negative(alg, phase))
    right = (
        half,
        half,
        alg.mul(inverse_phase, half),
        negative(alg, alg.mul(inverse_phase, half)),
    )
    return ResidentFactors(
        left,
        right,
        tuple(alg.power(exponent) for exponent in program.edge_exponents),
    )


@dataclass
class DeterminantStats:
    determinant_calls: int = 0
    pivots: int = 0
    swaps: int = 0
    zero_tests: int = 0
    additions: int = 0
    subtractions: int = 0
    multiplications: int = 0
    divisions: int = 0
    maximum_caller_matrix_field_cells: int = 0
    maximum_elimination_copy_field_cells: int = 0
    maximum_named_dense_work_field_cells: int = 0
    maximum_observed_field_value_payload_bits: int = 0
    maximum_named_dense_work_payload_bits_upper_bound: int = 0
    maximum_numerator_signed_bits: int = 0
    maximum_denominator_bits: int = 0

    def declare_work_cells(self, size: int) -> None:
        matrix_cells = size * size
        self.maximum_caller_matrix_field_cells = max(
            self.maximum_caller_matrix_field_cells,
            matrix_cells,
        )
        self.maximum_elimination_copy_field_cells = max(
            self.maximum_elimination_copy_field_cells,
            matrix_cells,
        )
        self.maximum_named_dense_work_field_cells = max(
            self.maximum_named_dense_work_field_cells,
            2 * matrix_cells + 5,
        )

    def observe_values(
        self,
        alg: backend.Algebra,
        values: Iterable[Any],
    ) -> None:
        for value in values:
            self.maximum_observed_field_value_payload_bits = max(
                self.maximum_observed_field_value_payload_bits,
                alg.payload_bits(value),
            )
            numerator, denominator = alg.coefficient_height(value)
            self.maximum_numerator_signed_bits = max(
                self.maximum_numerator_signed_bits,
                numerator,
            )
            self.maximum_denominator_bits = max(
                self.maximum_denominator_bits,
                denominator,
            )
        self.maximum_named_dense_work_payload_bits_upper_bound = (
            self.maximum_named_dense_work_field_cells
            * self.maximum_observed_field_value_payload_bits
        )

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


def determinant(
    matrix: list[list[Any]],
    alg: backend.Algebra,
    stats: DeterminantStats,
) -> Any:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        fail("Kasteleyn matrix must be square")
    work = [row[:] for row in matrix]
    result = alg.one
    stats.determinant_calls += 1
    stats.declare_work_cells(size)
    stats.observe_values(
        alg,
        (*[value for row in matrix for value in row], result),
    )
    for column in range(size):
        pivot_row = None
        for row in range(column, size):
            stats.zero_tests += 1
            if not alg.is_zero(work[row][column]):
                pivot_row = row
                break
        if pivot_row is None:
            return alg.zero
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            result = negative(alg, result)
            stats.swaps += 1
        pivot = work[column][column]
        result = alg.mul(result, pivot)
        stats.multiplications += 1
        pivot_inverse = alg.inverse(pivot)
        stats.divisions += 1
        stats.pivots += 1
        stats.observe_values(alg, (result, pivot, pivot_inverse))
        for row in range(column + 1, size):
            factor = alg.mul(work[row][column], pivot_inverse)
            stats.multiplications += 1
            for target in range(column + 1, size):
                term = alg.mul(factor, work[column][target])
                work[row][target] = alg.sub(work[row][target], term)
                stats.multiplications += 1
                stats.subtractions += 1
                stats.observe_values(
                    alg,
                    (
                        work[row][target],
                        result,
                        pivot,
                        pivot_inverse,
                        factor,
                        term,
                    ),
                )
            work[row][column] = alg.zero
    return result


@dataclass
class HolographicCarrier:
    n: int
    topology_fingerprint: str
    alg: backend.Algebra
    cells: list[Any]
    accumulator: Any
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    factor_load_additions: int = 0
    factor_unload_additions: int = 0
    accumulator_updates: int = 0
    basis_contraction_evaluations: int = 0
    basis_contraction_field_multiplications: int = 0
    edge_identity_substitutions: int = 0
    projection_calls: int = 0
    maximum_resident_payload_bits: int = 0
    determinant_stats: DeterminantStats = field(default_factory=DeterminantStats)

    @classmethod
    def create(cls, n: int, alg: backend.Algebra) -> "HolographicCarrier":
        topology = grid_edges(n)
        carrier = cls(
            n,
            sha256_json({"n": n, "edges": topology}),
            alg,
            [alg.zero for _ in range(8 + len(topology))],
            alg.zero,
        )
        carrier.observe_resident()
        return carrier

    def backing_identity(self) -> tuple[int, int]:
        return id(self), id(self.cells)

    def observe_resident(self) -> None:
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.cells)
            + self.alg.payload_bits(self.accumulator),
        )

    def exact_zero(self) -> bool:
        return (
            all(value == self.alg.zero for value in self.cells)
            and self.accumulator == self.alg.zero
            and self.lease is None
            and self.stage == "RESTORED"
        )

    def digest(self) -> str:
        return sha256_json(
            {
                "n": self.n,
                "topology": self.topology_fingerprint,
                "cells": [self.alg.serialize(value) for value in self.cells],
                "accumulator": self.alg.serialize(self.accumulator),
                "generation": self.generation,
                "lease": self.lease,
                "stage": self.stage,
            }
        )


def load_factors(
    carrier: HolographicCarrier,
    values: list[Any],
    *,
    inverse: bool = False,
) -> None:
    if len(values) != len(carrier.cells):
        fail("resident factor payload does not fit the carrier")
    for index, value in enumerate(values):
        delta = negative(carrier.alg, value) if inverse else value
        carrier.cells[index] = carrier.alg.add(carrier.cells[index], delta)
    if inverse:
        carrier.factor_unload_additions += len(values)
    else:
        carrier.factor_load_additions += len(values)
    carrier.observe_resident()


def factor_views(
    carrier: HolographicCarrier,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    return tuple(carrier.cells[:4]), tuple(carrier.cells[4:8]), tuple(carrier.cells[8:])


def paired_basis_contraction(carrier: HolographicCarrier) -> tuple[Any, ...]:
    left, right, _ = factor_views(carrier)
    result = []
    for left_row in range(2):
        for right_row in range(2):
            value = carrier.alg.zero
            for shared in range(2):
                term = carrier.alg.mul(
                    left[2 * left_row + shared],
                    right[2 * right_row + shared],
                )
                value = carrier.alg.add(value, term)
            result.append(value)
    carrier.basis_contraction_evaluations += 1
    carrier.basis_contraction_field_multiplications += 8
    return tuple(result)


def black_white_vertices(n: int) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    vertices = grid_vertices(n)
    return (
        tuple(vertex for vertex in vertices if sum(vertex) % 2 == 0),
        tuple(vertex for vertex in vertices if sum(vertex) % 2 == 1),
    )


def kasteleyn_edge_sign(
    left: tuple[int, int],
    right: tuple[int, int],
) -> int:
    if left[0] == right[0]:
        return 1
    return -1 if left[1] % 2 else 1


def reference_calibration_sign(n: int) -> int:
    black, white = black_white_vertices(n)
    white_index = {vertex: index for index, vertex in enumerate(white)}
    permutation = []
    edge_sign = 1
    for vertex in black:
        row, column = vertex
        mate = (row, column + 1 if column % 2 == 0 else column - 1)
        permutation.append(white_index[mate])
        edge_sign *= kasteleyn_edge_sign(vertex, mate)
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return edge_sign * (-1 if inversions % 2 else 1)


def compile_kasteleyn_matrix(carrier: HolographicCarrier) -> list[list[Any]]:
    contraction = paired_basis_contraction(carrier)
    identity = (carrier.alg.one, carrier.alg.zero, carrier.alg.zero, carrier.alg.one)
    if contraction != identity:
        fail("resident paired phase bases do not close to the shared-edge identity")
    black, white = black_white_vertices(carrier.n)
    black_index = {vertex: index for index, vertex in enumerate(black)}
    white_index = {vertex: index for index, vertex in enumerate(white)}
    matrix = [[carrier.alg.zero for _ in white] for _ in black]
    _, _, weights = factor_views(carrier)
    for edge, weight in zip(grid_edges(carrier.n), weights, strict=True):
        first, second = edge
        left, right = (first, second) if first in black_index else (second, first)
        value = weight if kasteleyn_edge_sign(first, second) == 1 else negative(carrier.alg, weight)
        row = black_index[left]
        column = white_index[right]
        matrix[row][column] = carrier.alg.add(matrix[row][column], value)
        carrier.edge_identity_substitutions += 1
    return matrix


def contract_resident_hologram(carrier: HolographicCarrier) -> Any:
    matrix = compile_kasteleyn_matrix(carrier)
    value = determinant(matrix, carrier.alg, carrier.determinant_stats)
    if reference_calibration_sign(carrier.n) == -1:
        value = negative(carrier.alg, value)
    return value


def forward(carrier: HolographicCarrier, program: HolographicProgram) -> None:
    if not isinstance(carrier, HolographicCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored paired-basis carrier")
    if carrier.n != program.n:
        fail("program does not own the carrier topology")
    factors = compile_factors(program, carrier.alg)
    carrier.lease = program.fingerprint()
    carrier.stage = "FORWARD_ACTIVE"
    load_factors(carrier, factors.flattened())
    value = contract_resident_hologram(carrier)
    carrier.accumulator = carrier.alg.add(carrier.accumulator, value)
    carrier.accumulator_updates += 1
    carrier.stage = "FORWARD_COMPLETE"
    carrier.observe_resident()


def project_boundary(carrier: HolographicCarrier, program: HolographicProgram) -> Any:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("only the completed owned holographic boundary may be projected")
    carrier.projection_calls += 1
    return carrier.accumulator


def inverse(carrier: HolographicCarrier, program: HolographicProgram) -> None:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("inverse program does not own the live holographic lease")
    factors = compile_factors(program, carrier.alg)
    carrier.stage = "INVERSE_ACTIVE"
    rematerialized = contract_resident_hologram(carrier)
    carrier.accumulator = carrier.alg.sub(carrier.accumulator, rematerialized)
    carrier.accumulator_updates += 1
    load_factors(carrier, factors.flattened(), inverse=True)
    carrier.lease = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact paired-basis carrier restoration")


def execute_transaction(
    carrier: HolographicCarrier,
    program: HolographicProgram,
) -> dict[str, Any]:
    backing = carrier.backing_identity()
    generation = carrier.generation
    initial = carrier.digest()
    forward(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    matrix_size = program.n * program.n // 2
    return {
        "n": program.n,
        "family": program.family,
        "program_fingerprint": program.fingerprint(),
        "boundary": carrier.alg.serialize(boundary),
        "basis_exponent": program.basis_exponent,
        "edge_count": len(grid_edges(program.n)),
        "public_program_integer_cells": len(program.edge_exponents) + 3,
        "compiled_topology_edge_records": len(grid_edges(program.n)),
        "resident_phase_field_cells": len(carrier.cells),
        "shared_basis_field_cells": 8,
        "basis_contraction_evaluations": carrier.basis_contraction_evaluations,
        "basis_contraction_field_multiplications": carrier.basis_contraction_field_multiplications,
        "edge_identity_substitutions": carrier.edge_identity_substitutions,
        "kasteleyn_matrix_dimension": matrix_size,
        "kasteleyn_matrix_field_cells": matrix_size * matrix_size,
        "final_boundary_field_cells": 1,
        "restoration_verification_field_cells": len(carrier.cells) + 1,
        "controller_backend_traffic_bytes": 0,
        "row_transfer_interface_message_baseline": 1 << program.n,
        "native_edge_assignment_baseline": 1 << len(grid_edges(program.n)),
        "accepted_path_native_signature_table_materialized": False,
        "accepted_path_edge_assignment_enumeration": False,
        "intermediate_boundary_projection_calls": 0,
        "final_boundary_projection_calls": 1,
        "factor_load_additions": carrier.factor_load_additions,
        "factor_unload_additions": carrier.factor_unload_additions,
        "accumulator_updates": carrier.accumulator_updates,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "determinant_stats": carrier.determinant_stats.as_json(),
        "generation_before": generation,
        "generation_after": carrier.generation,
        "restoration_generation_increment": carrier.generation == generation + 1,
        "same_backing": carrier.backing_identity() == backing,
        "initial_digest": initial,
        "restored_digest_with_generation": carrier.digest(),
        "exact_phase_carrier_restored": carrier.exact_zero(),
        "response_released_after_restoration": True,
        "snapshot_reload_used": False,
        "inverse_history_retained": False,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_determinant_buffer_restoration_class": "NO_RESTORATION_CLAIM",
    }


def native_local_value(
    alg: backend.Algebra,
    basis: tuple[Any, ...],
    bits: tuple[int, ...],
    weights: tuple[Any, ...],
) -> Any:
    if len(bits) != len(weights):
        fail("native local signature arity mismatch")
    result = alg.zero
    for chosen, chosen_weight in enumerate(weights):
        term = chosen_weight
        for position, bit in enumerate(bits):
            row = 1 if position == chosen else 0
            term = alg.mul(term, basis[2 * row + bit])
        result = alg.add(result, term)
    return result


def bounded_native_holant(program: HolographicProgram, alg: backend.Algebra) -> Any:
    if program.n != 2:
        fail("native edge-assignment control is bounded to n=2")
    factors = compile_factors(program, alg)
    incidents = incident_edge_indices(program.n)
    result = alg.zero
    for assignment in itertools.product((0, 1), repeat=len(program.edge_exponents)):
        term = alg.one
        for vertex in grid_vertices(program.n):
            indices = incidents[vertex]
            bits = tuple(assignment[index] for index in indices)
            if sum(vertex) % 2 == 0:
                weights = tuple(factors.edge_weights[index] for index in indices)
                local = native_local_value(alg, factors.left_basis, bits, weights)
            else:
                weights = tuple(alg.one for _ in indices)
                local = native_local_value(alg, factors.right_basis, bits, weights)
            term = alg.mul(term, local)
        result = alg.add(result, term)
    return result


def exact_case(n: int, family: str) -> dict[str, Any]:
    return execute_transaction(
        HolographicCarrier.create(n, backend.Algebra("Q_ZETA17")),
        compile_program(n, family),
    )


def modular_case(n: int, family: str, modulus: int, root: int) -> dict[str, Any]:
    result = execute_transaction(
        HolographicCarrier.create(
            n,
            backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
        ),
        compile_program(n, family),
    )
    result["field"] = f"F{modulus}"
    return result


def controls() -> dict[str, Any]:
    exact_alg = backend.Algebra("Q_ZETA17")
    n2 = compile_program(2, "PRIMARY")
    native = bounded_native_holant(n2, exact_alg)
    determinant_carrier = HolographicCarrier.create(2, backend.Algebra("Q_ZETA17"))
    determinant_tx = execute_transaction(determinant_carrier, n2)
    bounded_native_holant_agreement = exact_alg.serialize(native) == determinant_tx["boundary"]

    witness_alg = backend.Algebra("Q_ZETA17")
    witness_program = compile_program(4, "PRIMARY")
    witness_factors = compile_factors(witness_program, witness_alg)
    incidents = incident_edge_indices(4)
    left_vertex = (1, 1)
    right_vertex = (1, 2)
    left_indices = incidents[left_vertex]
    right_indices = incidents[right_vertex]
    left_weights = tuple(witness_factors.edge_weights[index] for index in left_indices)
    right_weights = tuple(witness_alg.one for _ in right_indices)
    left_even = native_local_value(witness_alg, witness_factors.left_basis, (0, 0, 0, 0), left_weights)
    left_odd = native_local_value(witness_alg, witness_factors.left_basis, (1, 0, 0, 0), left_weights)
    right_even = native_local_value(witness_alg, witness_factors.right_basis, (0, 0, 0, 0), right_weights)
    right_odd = native_local_value(witness_alg, witness_factors.right_basis, (1, 0, 0, 0), right_weights)
    both_native_parities_nonzero = all(
        value != witness_alg.zero for value in (left_even, left_odd, right_even, right_odd)
    )

    modular_alg = backend.Algebra("F103", modulus=103, root=72)
    program = compile_program(4, "PRIMARY")
    missing = HolographicCarrier.create(4, modular_alg)
    forward(missing, program)
    missing_inverse_detected = not missing.exact_zero()

    wrong = HolographicCarrier.create(
        4,
        backend.Algebra("F103", modulus=103, root=72),
    )
    forward(wrong, program)
    wrong_inverse_detected = False
    try:
        inverse(wrong, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_inverse_detected = True

    premature = HolographicCarrier.create(
        4,
        backend.Algebra("F103", modulus=103, root=72),
    )
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    null_carrier_rejected = False
    try:
        forward(None, program)  # type: ignore[arg-type]
    except (RuntimeError, AttributeError):
        null_carrier_rejected = True

    mutated = HolographicCarrier.create(
        4,
        backend.Algebra("F103", modulus=103, root=72),
    )
    factors = compile_factors(program, mutated.alg)
    mutated.lease = program.fingerprint()
    mutated.stage = "FORWARD_ACTIVE"
    load_factors(mutated, factors.flattened())
    mutated.cells[4] = mutated.alg.add(mutated.cells[4], mutated.alg.one)
    basis_mutation_rejected = False
    try:
        contract_resident_hologram(mutated)
    except RuntimeError:
        basis_mutation_rejected = True

    reordered = HolographicCarrier.create(
        4,
        backend.Algebra("F103", modulus=103, root=72),
    )
    forward(reordered, program)
    reorder_factors = compile_factors(program, reordered.alg)
    load_factors(reordered, reorder_factors.flattened(), inverse=True)
    reordered_inverse_detected = False
    try:
        inverse(reordered, program)
    except RuntimeError:
        reordered_inverse_detected = True

    face_signing_valid = all(
        kasteleyn_edge_sign((row, column), (row, column + 1))
        * kasteleyn_edge_sign((row, column + 1), (row + 1, column + 1))
        * kasteleyn_edge_sign((row + 1, column + 1), (row + 1, column))
        * kasteleyn_edge_sign((row + 1, column), (row, column))
        == -1
        for row in range(3)
        for column in range(3)
    )

    return {
        "bounded_n2_native_holant_matches_holographic_determinant": bounded_native_holant_agreement,
        "degree4_left_and_right_native_witnesses_have_both_parities_nonzero": both_native_parities_nonzero,
        "native_parity_witness_sha256": sha256_json(
            [witness_alg.serialize(value) for value in (left_even, left_odd, right_even, right_odd)]
        ),
        "paired_basis_exact_identity": paired_basis_contraction_from_factors(witness_factors, witness_alg)
        == (witness_alg.one, witness_alg.zero, witness_alg.zero, witness_alg.one),
        "public_grid_face_signing_valid": face_signing_valid,
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_inverse_ownership_detected": wrong_inverse_detected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "basis_mutation_rejected": basis_mutation_rejected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "accepted_path_native_signature_table_materialized": False,
        "accepted_path_edge_assignment_enumeration": False,
        "bounded_control_edge_assignments": 16,
        "snapshot_command_absent": True,
        "catvm_boundary_claimed": False,
    }


def paired_basis_contraction_from_factors(
    factors: ResidentFactors,
    alg: backend.Algebra,
) -> tuple[Any, ...]:
    result = []
    for left_row in range(2):
        for right_row in range(2):
            value = alg.zero
            for shared in range(2):
                value = alg.add(
                    value,
                    alg.mul(
                        factors.left_basis[2 * left_row + shared],
                        factors.right_basis[2 * right_row + shared],
                    ),
                )
            result.append(value)
    return tuple(result)


def run() -> dict[str, Any]:
    exact = [
        exact_case(n, family)
        for family in FAMILIES
        for n in EXACT_SIZES
    ]
    structural = [
        modular_case(n, family, modulus, root)
        for modulus, root in FINITE_FIELDS
        for family in FAMILIES
        for n in STRUCTURAL_SIZES
    ]

    reuse_n = 6
    reuse_carrier = HolographicCarrier.create(reuse_n, backend.Algebra("Q_ZETA17"))
    first = execute_transaction(reuse_carrier, compile_program(reuse_n, "PRIMARY"))
    reused = execute_transaction(reuse_carrier, compile_program(reuse_n, "REUSE"))
    fresh = execute_transaction(
        HolographicCarrier.create(reuse_n, backend.Algebra("Q_ZETA17")),
        compile_program(reuse_n, "REUSE"),
    )
    if reused["boundary"] != fresh["boundary"]:
        fail("restored holographic carrier reuse disagrees with fresh execution")

    return {
        "schema": "CAT_CAS_F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_PAIRED_ZETA17_BASIS_CLOSES_COMPACT_NATIVE_MIXED_PARITY_EXACT_ONE_GENERATORS_ON_GROWING_EVEN_PLANAR_GRIDS_BY_SHARED_EDGE_IDENTITY_AND_KASTELEYN_DETERMINANT_WITH_EXACT_RESTORATION_AND_REUSE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "topology": "EVEN_OPEN_SQUARE_GRIDS_ONLY",
            "exact_q_zeta17_sizes": EXACT_SIZES,
            "dual_field_structural_sizes": STRUCTURAL_SIZES,
            "families": FAMILIES,
            "native_signature_generator": "PAIRED_PHASE_TRANSFORM_OF_WEIGHTED_EXACT_ONE_CORE",
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "reuse": {
            "n": reuse_n,
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "same_actual_backing_across_unrelated_programs": first["same_backing"] and reused["same_backing"],
            "generation_after_two_transactions": reuse_carrier.generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "resource_law": {
            "resident_phase_field_cells": "2N_TIMES_N_MINUS_1_PLUS_8",
            "shared_basis_field_cells": 8,
            "accepted_projection_matrix_dimension": "N_SQUARED_OVER_2",
            "accepted_named_dense_work_field_cells": "N_TO_THE_4_OVER_2_PLUS_5",
            "named_dense_payload": "UPPER_BOUND_FROM_MAXIMUM_OBSERVED_FIELD_VALUE_TIMES_NAMED_LOGICAL_FIELD_CELLS",
            "accepted_field_operation_work": "O_N_TO_THE_6_DETERMINANT_FIELD_OPERATIONS_EXACT_BIT_COMPLEXITY_REPORTED_SEPARATELY",
            "row_transfer_interface_message_baseline": "2_TO_THE_N_NOT_MATERIALIZED_BY_THE_ACCEPTED_OR_MATCHED_DETERMINANT_PATH",
            "native_edge_assignment_baseline": "2_TO_THE_2N_TIMES_N_MINUS_1_NOT_MATERIALIZED_BY_THE_ACCEPTED_OR_MATCHED_DETERMINANT_PATH",
            "native_signature_tables_materialized": 0,
            "inverse_history_retained": 0,
            "public_program_integer_cells": "2N_TIMES_N_MINUS_1_PLUS_3",
            "compiled_topology_edge_records": "2N_TIMES_N_MINUS_1",
            "final_boundary_field_cells": 1,
            "restoration_verification_field_cells": "2N_TIMES_N_MINUS_1_PLUS_9",
            "controller_backend_traffic_bytes": 0,
            "python_container_sympy_native_bigint_and_whole_process_memory_excluded": True,
        },
        "matched_baselines": {
            "strongest_implemented": "IDENTICAL_PAIRED_BASIS_COMPILATION_AND_KASTELEYN_DETERMINANT",
            "independent_planned": "EXACT_MATCHING_RECURSION_N2_N4_N6_AND_DUAL_FIELD_ROW_MATCHING_DP",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_basis_weight_and_accumulator_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_kasteleyn_and_elimination_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "paired_phase_basis_is_primitive_resident_state": True,
            "native_degree4_signature_has_both_fermion_parities": True,
            "native_signature_table_avoided": True,
            "growing_treewidth_polynomial_matchgate_closure": True,
            "arbitrary_native_signature_or_planar_holant_closure": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "THE_PAIRED_PHASE_BASIS_STRICTLY_BROADENS_NATIVE_SIGNATURE_GEOMETRY_AND_AVOIDS_GROWING_SEPARATOR_MESSAGES_BUT_THE_PUBLIC_BASIS_CANCELS_TO_THE_IDENTICAL_CLASSICAL_MATCHGATE_DETERMINANT",
        "next_experiment": "RESIDENTLY_GENERATED_OR_COHERENCE_DEPENDENT_PHASE_BASIS_COUPLING_WITHOUT_A_PUBLIC_CLASSICAL_HOLOGRAPHIC_CANCELLATION_OR_A_MATCHED_NO_GO",
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
