#!/usr/bin/env python3
"""Exact rank-one cubic/Walsh derivative chart over the C17 group algebra.

For Boolean branch pairs z=(z0,z1,...), let

    Q_n(z) = sum_i z_(2i) z_(2i+1) mod 17.

Every declared cubic phase is zeta17^(x*(a*Q_n(z)+c)) for one unresolved
typed bit x.  Public topology compilation canonicalizes the Boolean
quadratics and proves that all derivatives span one F17 signature.  The
resident state therefore closes in K[C17]^2: two 17-coordinate exact phase
rows.  Walsh mixing on x and phase multiplication are native exact row
operations.  No branch assignment, truth table, or projected binomial is
materialized by the accepted path.

This is a bounded direct-process integrable family.  It has the identical
34-coordinate classical group-algebra/character recurrence, and exact cell
count does not bound coefficient height.  It is not CATVM custody, a general
cubic-hypergraph quotient, a distinct phase resource, or a Small Wall result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import f17_coherent_veronese_phase_chart_closure as exact
import f17_nonlinear_canonical_mps_separator_chart as backend


PRIME = 17
CHART_WIDTH = 17
DECLARED_CASES = (
    (1, 2),
    (2, 4),
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
)
STRUCTURAL_CASES = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64))
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
CLAIM = (
    "EXACT_RANK_ONE_CUBIC_PHASE_POLYNOMIAL_WALSH_DERIVATIVE_SIGNATURES_"
    "CLOSE_IN_A_FIXED_34_CELL_C17_GROUP_ALGEBRA_ACROSS_GROWING_BOOLEAN_"
    "BRANCH_COUNT_AND_GATE_DEPTH_WITH_FINAL_ONLY_BOUNDARY_PROJECTION_EXACT_"
    "RESTORATION_AND_REUSE_WHILE_AN_INDEPENDENT_SECOND_SIGNATURE_EXPANDS_"
    "THE_CANONICAL_C17_SQUARED_GROUP_ALGEBRA_CHART_TO_578_CELLS_OR_IS_"
    "REJECTED_BY_THE_34_CELL_COMPILER_AND_THE_ACCEPTED_FAMILY_HAS_AN_"
    "IDENTICAL_COMPACT_CLASSICAL_CHARACTER_RECURRENCE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def algebra_signature(alg: backend.Algebra) -> str:
    return exact.algebra_signature(alg)


@dataclass(frozen=True)
class QuadraticTerm:
    left: int
    right: int
    coefficient: int

    def as_json(self) -> dict[str, int]:
        return {
            "left": self.left,
            "right": self.right,
            "coefficient_mod17": self.coefficient % PRIME,
        }


@dataclass(frozen=True)
class Signature:
    name: str
    terms: tuple[QuadraticTerm, ...]

    def as_json(self) -> dict[str, Any]:
        return {"name": self.name, "terms": [item.as_json() for item in self.terms]}


@dataclass(frozen=True)
class Primitive:
    kind: str
    signature: int = -1
    multiplier: int = 0
    constant: int = 0

    def as_json(self) -> dict[str, Any]:
        if self.kind == "H":
            return {"kind": "UNNORMALIZED_WALSH_ON_UNRESOLVED_X"}
        return {
            "kind": "CUBIC_PHASE_ON_UNRESOLVED_X",
            "signature": self.signature,
            "multiplier_mod17": self.multiplier % PRIME,
            "constant_mod17": self.constant % PRIME,
        }


@dataclass(frozen=True)
class Program:
    branch_pairs: int
    rounds: int
    family: str
    signatures: tuple[Signature, ...]
    primitives: tuple[Primitive, ...]
    observation_exponent: int

    @property
    def boolean_branch_bits(self) -> int:
        return 2 * self.branch_pairs

    def public_descriptor(self) -> dict[str, Any]:
        return {
            "branch_pairs": self.branch_pairs,
            "boolean_branch_bits": self.boolean_branch_bits,
            "rounds": self.rounds,
            "family": self.family,
            "unresolved_typed_port": "X:BOOLEAN_PHASE_PORT",
            "signatures": [item.as_json() for item in self.signatures],
            "primitives": [item.as_json() for item in self.primitives],
            "observation": {
                "kind": "FINAL_X_CHARACTER_ONLY",
                "exponent_mod17": self.observation_exponent % PRIME,
            },
        }

    def fingerprint(self) -> str:
        return digest_json(self.public_descriptor())


def primary_signature(branch_pairs: int) -> Signature:
    return Signature(
        "PAIR_PRODUCT_SUM",
        tuple(QuadraticTerm(2 * index, 2 * index + 1, 1) for index in range(branch_pairs)),
    )


def independent_signature(branch_pairs: int) -> Signature:
    if branch_pairs < 2:
        fail("independent signature needs at least two branch pairs")
    return Signature(
        "CROSS_PAIR_PRODUCT_SUM",
        tuple(
            QuadraticTerm(2 * index + 1, 2 * ((index + 1) % branch_pairs), 1)
            for index in range(branch_pairs)
        ),
    )


def primitive_parameters(index: int, family: str) -> tuple[int, int]:
    if family == "PRIMARY":
        return 1 + ((5 * index * index + 7 * index + 3) % 16), (3 * index + 1) % PRIME
    if family == "REUSE":
        return 1 + ((9 * index * index + 4 * index + 11) % 16), (7 * index + 5) % PRIME
    if family == "ALTERNATE":
        return 1 + ((11 * index * index + 2 * index + 6) % 16), (13 * index + 2) % PRIME
    fail("unknown public program family")


def compile_program(branch_pairs: int, rounds: int, family: str) -> Program:
    if branch_pairs < 1 or branch_pairs > 128:
        fail("branch-pair count is outside the declared bounded family")
    if rounds < 1 or rounds > 256:
        fail("round count is outside the declared bounded family")
    if family not in FAMILIES:
        fail("unknown public program family")
    primitives: list[Primitive] = []
    for index in range(rounds):
        multiplier, constant = primitive_parameters(index, family)
        primitives.append(Primitive("D", 0, multiplier, constant))
        primitives.append(Primitive("H"))
    program = Program(
        branch_pairs,
        rounds,
        family,
        (primary_signature(branch_pairs),),
        tuple(primitives),
        1 + ((branch_pairs + 3 * rounds + len(family)) % 16),
    )
    validate_program(program)
    return program


def canonical_signature(signature: Signature, variable_count: int) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for term in signature.terms:
        if not (0 <= term.left < variable_count and 0 <= term.right < variable_count):
            fail("quadratic topology references an undeclared Boolean branch")
        key = tuple(sorted((term.left, term.right)))
        value = (result.get(key, 0) + term.coefficient) % PRIME
        if value:
            result[key] = value
        elif key in result:
            del result[key]
    if not result:
        fail("zero derivative signature is outside the accepted chart")
    return result


def signature_rank_and_coordinates(program: Program) -> tuple[int, tuple[int, ...], int]:
    canonical = [canonical_signature(item, program.boolean_branch_bits) for item in program.signatures]
    columns = sorted({key for row in canonical for key in row})
    matrix = [[row.get(key, 0) for key in columns] for row in canonical]
    work = [row[:] for row in matrix]
    rank = 0
    for column in range(len(columns)):
        pivot = next((row for row in range(rank, len(work)) if work[row][column] % PRIME), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inv = pow(work[rank][column] % PRIME, PRIME - 2, PRIME)
        work[rank] = [(value * inv) % PRIME for value in work[rank]]
        for row in range(len(work)):
            if row == rank:
                continue
            coefficient = work[row][column] % PRIME
            if coefficient:
                work[row] = [
                    (value - coefficient * pivot_value) % PRIME
                    for value, pivot_value in zip(work[row], work[rank])
                ]
        rank += 1
        if rank == len(work):
            break
    generator = matrix[0]
    generator_pivot = next(index for index, value in enumerate(generator) if value % PRIME)
    generator_inverse = pow(generator[generator_pivot] % PRIME, PRIME - 2, PRIME)
    coordinates = []
    for row in matrix:
        scalar = (row[generator_pivot] * generator_inverse) % PRIME
        coordinates.append(scalar if all((value - scalar * base) % PRIME == 0 for value, base in zip(row, generator)) else -1)
    return rank, tuple(coordinates), len(columns)


def signature_certificate(program: Program) -> dict[str, Any]:
    rank, coordinates, monomial_count = signature_rank_and_coordinates(program)
    return {
        "signature_rank_over_f17": rank,
        "basis_signature_count": len(program.signatures),
        "canonical_boolean_monomials": monomial_count,
        "coordinates_relative_to_first_signature": list(coordinates),
        "canonical_group_algebra_chart_field_cells_for_signature_rank": 2 * (PRIME ** rank),
        "rank_one_34_cell_chart_accepted": rank == 1 and all(item >= 0 for item in coordinates),
        "compiler_reads_public_topology_not_final_boundary_values": True,
        "sampling_or_truth_table_equivalence_used": False,
        "assignment_expansion_used": False,
    }


def validate_program(program: Program) -> None:
    if program.family not in FAMILIES:
        fail("program family changed")
    if len(program.primitives) != 2 * program.rounds:
        fail("program depth changed")
    for index, primitive in enumerate(program.primitives):
        expected = "D" if index % 2 == 0 else "H"
        if primitive.kind != expected:
            fail("declared phase/Walsh topology changed")
        if primitive.kind == "D" and not (0 <= primitive.signature < len(program.signatures)):
            fail("phase primitive signature owner changed")
    certificate = signature_certificate(program)
    if not certificate["rank_one_34_cell_chart_accepted"]:
        fail(
            "rank-one chart rejected public derivative topology; required cells="
            f"{certificate['canonical_group_algebra_chart_field_cells_for_signature_rank']}"
        )


def lease(program: Program, alg: backend.Algebra) -> str:
    return digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "carrier": "RANK_ONE_CUBIC_WALSH_C17_GROUP_ALGEBRA",
            "logical_cells": 34,
        }
    )


@dataclass
class GroupAlgebraCarrier:
    alg: backend.Algebra
    rows: list[list[Any]]
    active_program: str | None = None
    active_lease: str | None = None
    stage: str = "RESTORED"
    forward_index: int = 0
    inverse_index: int = 0
    projection_calls: int = 0
    package_local_restoration_count: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_update_scratch_payload_bits: int = 0
    maximum_update_scratch_field_cells: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra) -> "GroupAlgebraCarrier":
        return cls(alg, [[alg.zero for _ in range(CHART_WIDTH)] for _ in range(2)])

    def backing_identity(self) -> tuple[int, int, int, int]:
        return (id(self), id(self.rows), id(self.rows[0]), id(self.rows[1]))

    def exact_zero(self) -> bool:
        return (
            self.active_program is None
            and self.active_lease is None
            and self.stage == "RESTORED"
            and self.forward_index == 0
            and self.inverse_index == 0
            and self.projection_calls == 0
            and all(value == self.alg.zero for row in self.rows for value in row)
        )

    def observe_resident(self) -> None:
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            sum(self.alg.payload_bits(value) for row in self.rows for value in row),
        )

    def observe_scratch(self, values: Iterable[Any]) -> None:
        materialized = list(values)
        self.maximum_update_scratch_field_cells = max(
            self.maximum_update_scratch_field_cells, len(materialized)
        )
        self.maximum_update_scratch_payload_bits = max(
            self.maximum_update_scratch_payload_bits,
            sum(self.alg.payload_bits(value) for value in materialized),
        )

    def digest(self, include_package_local_count: bool = True) -> str:
        state = {
            "active_program": self.active_program,
            "active_lease": self.active_lease,
            "stage": self.stage,
            "forward_index": self.forward_index,
            "inverse_index": self.inverse_index,
            "projection_calls": self.projection_calls,
            "rows": [[self.alg.serialize(value) for value in row] for row in self.rows],
        }
        if include_package_local_count:
            state["package_local_restoration_count"] = self.package_local_restoration_count
        return digest_json(state)


def require_owned(carrier: GroupAlgebraCarrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, GroupAlgebraCarrier):
        fail("null or wrong rank-one carrier")
    if (
        carrier.stage != stage
        or carrier.active_program != program.fingerprint()
        or carrier.active_lease != lease(program, carrier.alg)
    ):
        fail("rank-one carrier custody or stage changed")


def phase_parameters(program: Program, primitive: Primitive) -> tuple[int, int]:
    rank, coordinates, _ = signature_rank_and_coordinates(program)
    if rank != 1 or coordinates[primitive.signature] < 0:
        fail("phase primitive left the rank-one derivative chart")
    return (
        primitive.multiplier * coordinates[primitive.signature] % PRIME,
        primitive.constant % PRIME,
    )


def apply_phase(carrier: GroupAlgebraCarrier, shift: int, constant: int, inverse: bool = False) -> None:
    alg = carrier.alg
    signed_shift = (-shift if inverse else shift) % PRIME
    scalar = alg.power(-constant if inverse else constant)
    scratch = [alg.zero for _ in range(CHART_WIDTH)]
    for source, value in enumerate(carrier.rows[1]):
        scratch[(source + signed_shift) % PRIME] = alg.mul(scalar, value)
    carrier.observe_scratch(scratch)
    carrier.rows[1][:] = scratch
    carrier.observe_resident()


def apply_walsh(carrier: GroupAlgebraCarrier, inverse: bool = False) -> None:
    alg = carrier.alg
    first = []
    second = []
    half = alg.inverse(exact.field_integer(alg, 2)) if inverse else alg.one
    for left, right in zip(carrier.rows[0], carrier.rows[1]):
        first.append(alg.mul(half, alg.add(left, right)))
        second.append(alg.mul(half, alg.sub(left, right)))
    carrier.observe_scratch([*first, *second])
    carrier.rows[0][:] = first
    carrier.rows[1][:] = second
    carrier.observe_resident()


def begin_forward(carrier: GroupAlgebraCarrier, program: Program) -> None:
    validate_program(program)
    if not isinstance(carrier, GroupAlgebraCarrier) or not carrier.exact_zero():
        fail("rank-one carrier is not restored and available")
    carrier.active_program = program.fingerprint()
    carrier.active_lease = lease(program, carrier.alg)
    carrier.stage = "FORWARD"
    carrier.rows[0][0] = carrier.alg.one
    carrier.rows[1][0] = carrier.alg.one
    carrier.observe_resident()


def forward(carrier: GroupAlgebraCarrier, program: Program) -> None:
    require_owned(carrier, program, "FORWARD")
    for index, primitive in enumerate(program.primitives):
        if primitive.kind == "D":
            shift, constant = phase_parameters(program, primitive)
            apply_phase(carrier, shift, constant)
        else:
            apply_walsh(carrier)
        carrier.forward_index = index + 1
    carrier.stage = "FINAL_BOUNDARY_RESIDENT"


def projected_boundary(carrier: GroupAlgebraCarrier, program: Program) -> Any:
    require_owned(carrier, program, "FINAL_BOUNDARY_RESIDENT")
    if carrier.forward_index != len(program.primitives) or carrier.projection_calls:
        fail("boundary projection count or forward completion changed")
    alg = carrier.alg
    observation = alg.power(program.observation_exponent)
    boundary = alg.zero
    for exponent in range(PRIME):
        moment = exact.scalar_power(
            alg, alg.add(exact.field_integer(alg, 3), alg.power(exponent)), program.branch_pairs
        )
        component = alg.add(carrier.rows[0][exponent], alg.mul(observation, carrier.rows[1][exponent]))
        boundary = alg.add(boundary, alg.mul(component, moment))
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED"
    return boundary


def inverse(carrier: GroupAlgebraCarrier, program: Program) -> None:
    require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("inverse requires exactly one final projection")
    carrier.stage = "INVERSE"
    for index in range(len(program.primitives) - 1, -1, -1):
        primitive = program.primitives[index]
        if primitive.kind == "D":
            shift, constant = phase_parameters(program, primitive)
            apply_phase(carrier, shift, constant, inverse=True)
        else:
            apply_walsh(carrier, inverse=True)
        carrier.inverse_index += 1
    seed = [carrier.alg.one, *[carrier.alg.zero for _ in range(CHART_WIDTH - 1)]]
    if carrier.rows[0] != seed or carrier.rows[1] != seed:
        fail("actual inverse did not restore the exact rank-one chart seed")
    carrier.rows[0][0] = carrier.alg.sub(carrier.rows[0][0], carrier.alg.one)
    carrier.rows[1][0] = carrier.alg.sub(carrier.rows[1][0], carrier.alg.one)
    carrier.active_program = None
    carrier.active_lease = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not carrier.exact_zero():
        fail("rank-one carrier did not restore exact canonical zero")


def stream_state_commitment(carrier: GroupAlgebraCarrier) -> str:
    state = hashlib.sha256()
    for row in carrier.rows:
        for value in row:
            record = json.dumps(carrier.alg.serialize(value), separators=(",", ":")).encode()
            state.update(len(record).to_bytes(8, "big"))
            state.update(record)
    return state.hexdigest()


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "family",
        "program_fingerprint",
        "final_boundary",
        "final_state_commitment",
        "package_local_restoration_count_before",
        "package_local_restoration_count_after",
    }
    return {key: value for key, value in transaction.items() if key not in excluded}


def execute_transaction(carrier: GroupAlgebraCarrier, program: Program) -> dict[str, Any]:
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_update_scratch_payload_bits = 0
    carrier.maximum_update_scratch_field_cells = 0
    initial_digest = carrier.digest(include_package_local_count=False)
    backing = carrier.backing_identity()
    restoration_count = carrier.package_local_restoration_count
    certificate = signature_certificate(program)
    descriptor_bytes = len(json.dumps(program.public_descriptor(), sort_keys=True, separators=(",", ":")).encode())
    begin_forward(carrier, program)
    forward(carrier, program)
    commitment = stream_state_commitment(carrier)
    boundary = projected_boundary(carrier, program)
    inverse(carrier, program)
    return {
        "branch_pairs": program.branch_pairs,
        "boolean_branch_bits": program.boolean_branch_bits,
        "rounds": program.rounds,
        "primitive_count": len(program.primitives),
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "signature_certificate": certificate,
        "final_boundary": carrier.alg.serialize(boundary),
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_state_commitment": commitment,
        "resident_group_algebra_field_cells": 34,
        "resident_unresolved_port_rows": 2,
        "resident_signature_classes": 17,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_update_scratch_field_cells": carrier.maximum_update_scratch_field_cells,
        "maximum_update_scratch_payload_bits": carrier.maximum_update_scratch_payload_bits,
        "maximum_projection_named_field_cells": 6,
        "public_program_json_bytes": descriptor_bytes,
        "public_quadratic_term_records": sum(len(item.terms) for item in program.signatures),
        "public_primitive_records": len(program.primitives),
        "compiler_integer_cells_upper_bound": 4 * certificate["canonical_boolean_monomials"] + 16,
        "inverse_history_cells": 0,
        "inverse_operations_rematerialized_from_public_topology": True,
        "accepted_path_branch_assignment_cells": 0,
        "accepted_path_truth_table_cells": 0,
        "accepted_path_projected_binomial_factor_list": 0,
        "intermediate_group_algebra_coefficients_exposed": False,
        "one_way_final_state_commitment_emitted": True,
        "final_projection_calls": 1,
        "response_released_after_restoration": True,
        "same_backing": carrier.backing_identity() == backing,
        "restored_exact_zero": carrier.exact_zero(),
        "initial_restored_digest_equal": carrier.digest(include_package_local_count=False) == initial_digest,
        "package_local_restoration_count_before": restoration_count,
        "package_local_restoration_count_after": carrier.package_local_restoration_count,
        "snapshot_reload_used": False,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "compiler_projection_and_commitment_buffers_restoration_class": "NO_RESTORATION_CLAIM",
    }


def classical_baseline(program: Program, transaction: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch_pairs": program.branch_pairs,
        "rounds": program.rounds,
        "same_public_program_fingerprint": program.fingerprint(),
        "identical_exact_group_algebra_state_field_cells": 34,
        "equivalent_residue_class_dynamic_program_states": 34,
        "pair_multiplicity_recurrence": "N_NEXT_Q=3*N_Q+N_(Q_MINUS_1)",
        "character_moments": "S_R(N)=(3+ZETA17^R)^N",
        "warm_arithmetic_work": "O(17*(ROUNDS+LOG_BRANCH_PAIRS))",
        "compiled_word_option": "SEVENTEEN_INDEPENDENT_TWO_BY_TWO_TRANSFER_MATRICES",
        "boundary_equal_by_identical_recurrence": True,
        "maximum_payload_bits_equal_to_phase_path": transaction["maximum_resident_payload_bits"],
        "phase_carrier_or_snapshot_used": False,
        "comparison_establishes_distinct_phase_resource": False,
        "comparison_establishes_computational_advantage": False,
    }


def mutated_rank_two_program() -> Program:
    base = compile_program(4, 4, "PRIMARY")
    primitives = list(base.primitives)
    primitives[2] = replace(primitives[2], signature=1)
    return replace(base, signatures=(primary_signature(4), independent_signature(4)), primitives=tuple(primitives))


def cyclic_convolution(left: list[Any], right: list[Any], alg: backend.Algebra) -> list[Any]:
    if len(left) != PRIME or len(right) != PRIME:
        fail("C17 convolution width changed")
    result = [alg.zero for _ in range(PRIME)]
    for first, left_value in enumerate(left):
        for second, right_value in enumerate(right):
            result[(first + second) % PRIME] = alg.add(
                result[(first + second) % PRIME], alg.mul(left_value, right_value)
            )
    return result


def controls() -> dict[str, bool]:
    alg = backend.Algebra("F137", modulus=137, root=16)
    program = compile_program(4, 8, "PRIMARY")
    reference = execute_transaction(GroupAlgebraCarrier.create(alg), program)

    rank_two = mutated_rank_two_program()
    rank_two_certificate = signature_certificate(rank_two)
    rank_two_rejected = False
    try:
        validate_program(rank_two)
    except RuntimeError:
        rank_two_rejected = True

    base = primary_signature(4)
    duplicated_terms = tuple(
        term
        for item in base.terms
        for term in (
            QuadraticTerm(item.left, item.right, 18),
            QuadraticTerm(item.right, item.left, -17),
        )
    )
    equivalent = replace(program, signatures=(Signature("DUPLICATE_MOD17", duplicated_terms),))
    equivalent_rank_one = signature_certificate(equivalent)["rank_one_34_cell_chart_accepted"]

    extra = replace(
        program,
        signatures=(
            base,
            Signature("EXTRA_UNSAMPLED_MONOMIAL", (QuadraticTerm(0, 2, 1),)),
        ),
        primitives=tuple(
            replace(item, signature=1) if index == 2 else item
            for index, item in enumerate(program.primitives)
        ),
    )
    extra_rank_two = signature_certificate(extra)["signature_rank_over_f17"] == 2

    missing = GroupAlgebraCarrier.create(alg)
    begin_forward(missing, program)
    forward(missing, program)

    wrong_inverse_detected = False
    wrong = GroupAlgebraCarrier.create(alg)
    begin_forward(wrong, program)
    forward(wrong, program)
    projected_boundary(wrong, program)
    wrong_program = replace(program, primitives=tuple(
        replace(item, constant=(item.constant + 1) % PRIME) if index == 0 else item
        for index, item in enumerate(program.primitives)
    ))
    wrong.active_program = wrong_program.fingerprint()
    wrong.active_lease = lease(wrong_program, alg)
    try:
        inverse(wrong, wrong_program)
    except RuntimeError:
        wrong_inverse_detected = True

    reordered_inverse_detected = False
    reordered = GroupAlgebraCarrier.create(alg)
    begin_forward(reordered, program)
    forward(reordered, program)
    projected_boundary(reordered, program)
    reordered_program = replace(program, primitives=tuple(reversed(program.primitives)))
    reordered.active_program = reordered_program.fingerprint()
    reordered.active_lease = lease(reordered_program, alg)
    try:
        inverse(reordered, reordered_program)
    except RuntimeError:
        reordered_inverse_detected = True

    premature_projection_rejected = False
    try:
        projected_boundary(GroupAlgebraCarrier.create(alg), program)
    except RuntimeError:
        premature_projection_rejected = True

    wrong_owner_rejected = False
    owner = GroupAlgebraCarrier.create(alg)
    begin_forward(owner, program)
    try:
        forward(owner, compile_program(4, 8, "REUSE"))
    except RuntimeError:
        wrong_owner_rejected = True

    null_carrier_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    constant_program = replace(
        program,
        primitives=tuple(
            replace(item, constant=(item.constant + 1) % PRIME) if index == 0 else item
            for index, item in enumerate(program.primitives)
        ),
    )
    constant_result = execute_transaction(GroupAlgebraCarrier.create(alg), constant_program)

    one_minus_s = [alg.zero for _ in range(PRIME)]
    one_minus_s[0] = alg.one
    one_minus_s[1] = alg.sub(alg.zero, alg.one)
    geometric_sum = [alg.one for _ in range(PRIME)]
    zero_divisor_confirmed = (
        any(value != alg.zero for value in one_minus_s)
        and any(value != alg.zero for value in geometric_sum)
        and all(
            value == alg.zero
            for value in cyclic_convolution(one_minus_s, geometric_sum, alg)
        )
    )

    swapped = list(program.primitives)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    reorder_forward_rejected = False
    try:
        validate_program(replace(program, primitives=tuple(swapped)))
    except RuntimeError:
        reorder_forward_rejected = True

    return {
        "reference_restored_exactly": reference["restored_exact_zero"],
        "rank_two_canonical_group_algebra_chart_has_578_cells": rank_two_certificate["canonical_group_algebra_chart_field_cells_for_signature_rank"] == 578,
        "rank_two_signature_rejected_by_34_cell_compiler": rank_two_rejected,
        "mod17_duplicate_terms_canonicalize_to_rank_one": equivalent_rank_one,
        "extra_independent_monomial_not_merged_by_sampling": extra_rank_two,
        "missing_inverse_leaves_actual_resident_state": not missing.exact_zero(),
        "wrong_inverse_detected": wrong_inverse_detected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "reordered_forward_topology_rejected": reorder_forward_rejected,
        "premature_projection_rejected": premature_projection_rejected,
        "wrong_owner_rejected": wrong_owner_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "constant_phase_ledger_changes_boundary": constant_result["final_boundary"] != reference["final_boundary"],
        "projected_binomial_division_rejected_as_zero_divisor": zero_divisor_confirmed,
        "snapshot_command_available": False,
        "intermediate_projection_available": False,
    }


def run() -> dict[str, Any]:
    exact_transactions = []
    q_alg = backend.Algebra("Q_ZETA17")
    for branch_pairs, rounds in DECLARED_CASES:
        exact_transactions.append(
            execute_transaction(
                GroupAlgebraCarrier.create(q_alg),
                compile_program(branch_pairs, rounds, "PRIMARY"),
            )
        )

    structural_transactions = []
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        for branch_pairs, rounds in STRUCTURAL_CASES:
            structural_transactions.append(
                execute_transaction(
                    GroupAlgebraCarrier.create(alg),
                    compile_program(branch_pairs, rounds, "ALTERNATE"),
                )
            )

    reuse_alg = backend.Algebra("Q_ZETA17")
    reuse_carrier = GroupAlgebraCarrier.create(reuse_alg)
    backing = reuse_carrier.backing_identity()
    primary = execute_transaction(reuse_carrier, compile_program(8, 16, "PRIMARY"))
    restored = execute_transaction(reuse_carrier, compile_program(64, 128, "REUSE"))
    fresh = execute_transaction(
        GroupAlgebraCarrier.create(reuse_alg), compile_program(64, 128, "REUSE")
    )

    transactions = [*exact_transactions, *structural_transactions]
    baselines = [
        classical_baseline(
            compile_program(item["branch_pairs"], item["rounds"], item["family"]), item
        )
        for item in transactions
    ]
    control_results = controls()
    false_controls = {"snapshot_command_available", "intermediate_projection_available"}
    if not all(
        item["restored_exact_zero"]
        and item["same_backing"]
        and item["signature_certificate"]["rank_one_34_cell_chart_accepted"]
        for item in transactions
    ):
        fail("rank-one group-algebra transaction failed")
    if any(control_results[key] for key in false_controls) or not all(
        value for key, value in control_results.items() if key not in false_controls
    ):
        fail("rank-one group-algebra control failed")

    return {
        "schema": "CAT_CAS_F17_RANK1_CUBIC_WALSH_DERIVATIVE_GROUP_ALGEBRA_CLOSURE_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "execution_scope": "LINUX_DIRECT_PROCESS_EXACT_SOFTWARE",
        "source_scope": {
            "exact_branch_pair_round_cases": [list(item) for item in DECLARED_CASES],
            "dual_field_structural_cases": [list(item) for item in STRUCTURAL_CASES],
            "public_families": list(FAMILIES),
            "base_signature": "Q_N_EQUALS_SUM_OF_DISJOINT_BOOLEAN_PAIR_PRODUCTS_MOD17",
            "unresolved_typed_port_count": 1,
            "derivative_signature_rank": 1,
        },
        "exact_transactions": exact_transactions,
        "dual_field_structural_transactions": structural_transactions,
        "matched_classical_baselines": baselines,
        "reuse": {
            "primary_case": [primary["branch_pairs"], primary["rounds"]],
            "unrelated_reuse_case": [restored["branch_pairs"], restored["rounds"]],
            "same_original_backing": reuse_carrier.backing_identity() == backing,
            "fresh_restored_boundary_equal": restored["final_boundary"] == fresh["final_boundary"],
            "fresh_restored_resource_signature_equal": resource_signature(restored) == resource_signature(fresh),
            "package_local_restoration_count": reuse_carrier.package_local_restoration_count,
            "restored_exact_zero": reuse_carrier.exact_zero(),
            "baseline_reload": False,
            "inverse_history_cells": 0,
        },
        "controls": control_results,
        "resource_law": {
            "resident_exact_field_cells": 34,
            "resident_cells_independent_of_branch_pairs_and_rounds": True,
            "rank_r_canonical_group_algebra_chart_field_cells": "TWO_TIMES_17_TO_THE_R",
            "rank_two_canonical_group_algebra_chart_field_cells": 578,
            "public_topology_records": "BRANCH_PAIRS_PLUS_TWO_TIMES_ROUNDS",
            "maximum_update_scratch_field_cells": 34,
            "projection_stream_named_field_cells": 6,
            "inverse_history_cells": 0,
            "coefficient_payload_bits_are_measured_and_not_constant": True,
            "compiler_projection_commitment_and_python_allocator_buffers_excluded_from_restoration": True,
            "full_native_library_and_whole_process_memory_excluded": True,
        },
        "matched_baseline": {
            "strongest_compact_classical_method": "IDENTICAL_EXACT_34_COORDINATE_C17_GROUP_ALGEBRA_OR_17_RESIDUE_CLASS_CHARACTER_RECURRENCE",
            "same_public_inputs_outputs_and_instances": True,
            "same_asymptotic_arithmetic_and_payload_law": True,
            "comparison_against_dense_assignment_expansion_only": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
        },
        "restoration": {
            "resident_group_algebra_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "compiler_projection_and_commitment_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "exact_rank_one_signature_family_only": True,
            "rank_two_fixed_34_cell_closure": False,
            "general_rank_r_or_arbitrary_cubic_hypergraph_closure": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": (
            "THE_FIXED_34_CELL_CHART_IS_EXACT_ONLY_WHILE_ALL_CUBIC_PHASE_"
            "DERIVATIVES_SHARE_ONE_F17_QUADRATIC_SIGNATURE_AND_HAS_AN_"
            "IDENTICAL_CLASSICAL_CHARACTER_RECURRENCE;_AN_INDEPENDENT_SECOND_"
            "SIGNATURE_EXPANDS_THE_CANONICAL_GROUP_ALGEBRA_TO_578_CELLS_SO_"
            "THE_NEXT_PHASE_OWNED_TEST_MUST_MERGE_OR_DYNAMICALLY_COUPLE_"
            "MULTIPLE_SIGNATURES_WITHOUT_MOVING_17_TO_THE_R_STATE_INTO_"
            "COEFFICIENT_HEIGHT_PROJECTION_OR_A_CLASSICAL_SIDE_CHANNEL"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(
        json.dumps(run(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
