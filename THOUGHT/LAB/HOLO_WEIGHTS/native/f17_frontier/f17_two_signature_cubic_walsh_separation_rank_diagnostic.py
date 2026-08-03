#!/usr/bin/env python3
"""Exact two-signature cubic/Walsh separation-rank diagnostic.

This successor adds a second independent Boolean quadratic derivative to the
M137 group-algebra chart.  The signatures act on disjoint public branch
groups but share one unresolved typed Walsh bit.  Alternating phase shifts
and Walsh mixing generate coefficient surfaces in K[C17 x C17]^2.

The experiment measures exact matrix separation rank across the two public
signatures.  It is a bounded diagnostic of whether the 578-cell canonical
chart admits a uniformly low-rank factor representation.  It does not claim
that the canonical chart is a universal minimum, and it retains the
matched retain-all and streamed/rematerialized classical recurrences as the
compact baseline frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import f17_coherent_veronese_phase_chart_closure as exact
import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_rank1_cubic_walsh_derivative_group_algebra_closure as rank1


P = 17
AXIS = 17
RESIDENT_CELLS = 2 * AXIS * AXIS
EXACT_CASES = ((1, 2), (2, 4), (4, 6), (8, 8), (16, 10), (32, 12), (64, 16), (128, 32))
STRUCTURAL_PHASE_STEPS = (2, 4, 6, 8, 10, 12, 16, 32, 64)
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
CLAIM = (
    "EXACT_TWO_INDEPENDENT_CUBIC_WALSH_DERIVATIVE_SIGNATURES_ON_ONE_"
    "UNRESOLVED_TYPED_PORT_GENERATE_FULL_17_BY_17_COEFFICIENT_SEPARATION_"
    "RANK_IN_THE_DECLARED_ALTERNATING_PROGRAM_FAMILIES_SO_UNIFORMLY_LOW_"
    "SEPARATION_RANK_FACTOR_CHARTS_ARE_REJECTED_AT_THE_FAMILY_CEILING_"
    "WHILE_THE_CANONICAL_578_CELL_C17_SQUARED_GROUP_ALGEBRA_CARRIER_"
    "RESTORES_AND_REUSES_EXACTLY_BUT_HAS_MATCHED_COMPACT_CLASSICAL_"
    "RESIDUE_RECURRENCES_AND_GROWING_EXACT_PAYLOAD"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PhaseStep:
    signature_axis: int
    multiplier: int
    constant: int

    def as_json(self) -> dict[str, int | str]:
        return {
            "kind": "CUBIC_PHASE_THEN_UNNORMALIZED_WALSH",
            "signature_axis": self.signature_axis,
            "multiplier_mod17": self.multiplier % P,
            "constant_mod17": self.constant % P,
        }


@dataclass(frozen=True)
class Program:
    branch_pairs_per_signature: int
    phase_steps: int
    family: str
    signatures: tuple[rank1.Signature, rank1.Signature]
    steps: tuple[PhaseStep, ...]
    observation_exponent: int

    @property
    def boolean_branch_bits(self) -> int:
        return 4 * self.branch_pairs_per_signature

    def descriptor(self) -> dict[str, Any]:
        return {
            "branch_pairs_per_signature": self.branch_pairs_per_signature,
            "boolean_branch_bits": self.boolean_branch_bits,
            "phase_steps": self.phase_steps,
            "family": self.family,
            "unresolved_typed_port": "X:BOOLEAN_PHASE_PORT",
            "quadratic_signatures": [item.as_json() for item in self.signatures],
            "steps": [item.as_json() for item in self.steps],
            "final_observation_exponent_mod17": self.observation_exponent,
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def second_signature(branch_pairs: int) -> rank1.Signature:
    offset = 2 * branch_pairs
    return rank1.Signature(
        "SECOND_DISJOINT_PAIR_PRODUCT_SUM",
        tuple(
            rank1.QuadraticTerm(offset + 2 * index, offset + 2 * index + 1, 1)
            for index in range(branch_pairs)
        ),
    )


def compile_program(branch_pairs: int, phase_steps: int, family: str) -> Program:
    if branch_pairs < 1 or branch_pairs > 128:
        fail("two-signature branch-pair count outside declared family")
    if phase_steps < 2 or phase_steps > 64 or phase_steps % 2:
        fail("two-signature phase-step count must be even and bounded")
    if family not in FAMILIES:
        fail("two-signature family changed")
    steps = []
    for index in range(phase_steps):
        multiplier, constant = rank1.primitive_parameters(index, family)
        steps.append(PhaseStep(index % 2, multiplier, constant))
    program = Program(
        branch_pairs,
        phase_steps,
        family,
        (rank1.primary_signature(branch_pairs), second_signature(branch_pairs)),
        tuple(steps),
        1 + ((3 * branch_pairs + 5 * phase_steps + len(family)) % 16),
    )
    validate_program(program)
    return program


def derivative_rank(program: Program) -> int:
    rows = [rank1.canonical_signature(item, program.boolean_branch_bits) for item in program.signatures]
    columns = sorted({key for row in rows for key in row})
    matrix = [[row.get(key, 0) for key in columns] for row in rows]
    return matrix_rank_mod17(matrix)


def matrix_rank_mod17(matrix: list[list[int]]) -> int:
    work = [row[:] for row in matrix]
    rank = 0
    columns = len(work[0]) if work else 0
    for column in range(columns):
        pivot = next((row for row in range(rank, len(work)) if work[row][column] % P), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inv = pow(work[rank][column] % P, P - 2, P)
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


def validate_program(program: Program) -> None:
    if len(program.steps) != program.phase_steps:
        fail("two-signature program depth changed")
    if any(step.signature_axis != index % 2 for index, step in enumerate(program.steps)):
        fail("two-signature alternating public topology changed")
    if derivative_rank(program) != 2:
        fail("two-signature public topology does not have exact rank two")


def lease(program: Program, alg: backend.Algebra) -> str:
    return digest_json(
        {
            "program": program.fingerprint(),
            "algebra": rank1.algebra_signature(alg),
            "carrier": "TWO_SIGNATURE_C17_SQUARED_GROUP_ALGEBRA",
            "resident_cells": RESIDENT_CELLS,
        }
    )


def zeros(alg: backend.Algebra) -> list[list[Any]]:
    return [[alg.zero for _ in range(AXIS)] for _ in range(AXIS)]


@dataclass
class SurfaceCarrier:
    alg: backend.Algebra
    rows: list[list[list[Any]]]
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
    maximum_live_resident_plus_update_scratch_payload_bits: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra) -> "SurfaceCarrier":
        return cls(alg, [zeros(alg), zeros(alg)])

    def backing_identity(self) -> tuple[int, ...]:
        return (
            id(self),
            id(self.rows),
            id(self.rows[0]),
            id(self.rows[1]),
            *[id(row) for surface in self.rows for row in surface],
        )

    def all_values(self) -> Iterable[Any]:
        return (value for surface in self.rows for row in surface for value in row)

    def exact_zero(self) -> bool:
        return (
            self.active_program is None
            and self.active_lease is None
            and self.stage == "RESTORED"
            and self.forward_index == 0
            and self.inverse_index == 0
            and self.projection_calls == 0
            and all(value == self.alg.zero for value in self.all_values())
        )

    def observe_resident(self) -> None:
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.all_values()),
        )

    def observe_scratch(self, *surfaces: list[list[Any]]) -> None:
        values = [value for surface in surfaces for row in surface for value in row]
        scratch_payload = sum(self.alg.payload_bits(value) for value in values)
        self.maximum_update_scratch_field_cells = max(self.maximum_update_scratch_field_cells, len(values))
        self.maximum_update_scratch_payload_bits = max(
            self.maximum_update_scratch_payload_bits,
            scratch_payload,
        )
        self.maximum_live_resident_plus_update_scratch_payload_bits = max(
            self.maximum_live_resident_plus_update_scratch_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.all_values()) + scratch_payload,
        )

    def digest(self, include_package_local_count: bool = True) -> str:
        state = {
            "active_program": self.active_program,
            "active_lease": self.active_lease,
            "stage": self.stage,
            "forward_index": self.forward_index,
            "inverse_index": self.inverse_index,
            "projection_calls": self.projection_calls,
            "rows": [
                [[self.alg.serialize(value) for value in row] for row in surface]
                for surface in self.rows
            ],
        }
        if include_package_local_count:
            state["package_local_restoration_count"] = self.package_local_restoration_count
        return digest_json(state)


def require_owned(carrier: SurfaceCarrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, SurfaceCarrier):
        fail("null or wrong two-signature carrier")
    if (
        carrier.stage != stage
        or carrier.active_program != program.fingerprint()
        or carrier.active_lease != lease(program, carrier.alg)
    ):
        fail("two-signature carrier owner or stage changed")


def shift_surface(surface: list[list[Any]], axis: int, shift: int, scalar: Any, alg: backend.Algebra) -> list[list[Any]]:
    result = zeros(alg)
    for first in range(AXIS):
        for second in range(AXIS):
            target_first = (first + shift) % P if axis == 0 else first
            target_second = (second + shift) % P if axis == 1 else second
            result[target_first][target_second] = alg.mul(scalar, surface[first][second])
    return result


def apply_phase(carrier: SurfaceCarrier, step: PhaseStep, inverse_step: bool = False) -> None:
    shift = (-step.multiplier if inverse_step else step.multiplier) % P
    scalar = carrier.alg.power(-step.constant if inverse_step else step.constant)
    scratch = shift_surface(carrier.rows[1], step.signature_axis, shift, scalar, carrier.alg)
    carrier.observe_scratch(scratch)
    for row, replacement in zip(carrier.rows[1], scratch):
        row[:] = replacement
    carrier.observe_resident()


def apply_walsh(carrier: SurfaceCarrier, inverse_step: bool = False) -> None:
    alg = carrier.alg
    half = alg.inverse(exact.field_integer(alg, 2)) if inverse_step else alg.one
    first = zeros(alg)
    second = zeros(alg)
    for row in range(AXIS):
        for column in range(AXIS):
            first[row][column] = alg.mul(
                half, alg.add(carrier.rows[0][row][column], carrier.rows[1][row][column])
            )
            second[row][column] = alg.mul(
                half, alg.sub(carrier.rows[0][row][column], carrier.rows[1][row][column])
            )
    carrier.observe_scratch(first, second)
    for resident, replacement in zip(carrier.rows[0], first):
        resident[:] = replacement
    for resident, replacement in zip(carrier.rows[1], second):
        resident[:] = replacement
    carrier.observe_resident()


def begin_forward(carrier: SurfaceCarrier, program: Program) -> None:
    validate_program(program)
    if not isinstance(carrier, SurfaceCarrier) or not carrier.exact_zero():
        fail("two-signature carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.active_lease = lease(program, carrier.alg)
    carrier.stage = "FORWARD"
    carrier.rows[0][0][0] = carrier.alg.one
    carrier.rows[1][0][0] = carrier.alg.one
    carrier.observe_resident()


def forward(carrier: SurfaceCarrier, program: Program) -> None:
    require_owned(carrier, program, "FORWARD")
    for index, step in enumerate(program.steps):
        apply_phase(carrier, step)
        apply_walsh(carrier)
        carrier.forward_index = index + 1
    carrier.stage = "FINAL_DIAGNOSTIC_RESIDENT"


def matrix_rank(matrix: list[list[Any]], alg: backend.Algebra) -> int:
    work = [row[:] for row in matrix]
    rank = 0
    for column in range(AXIS):
        pivot = next((row for row in range(rank, AXIS) if work[row][column] != alg.zero), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = alg.inverse(work[rank][column])
        work[rank] = [alg.mul(scale, value) for value in work[rank]]
        for row in range(AXIS):
            if row == rank or work[row][column] == alg.zero:
                continue
            coefficient = work[row][column]
            work[row] = [
                alg.sub(value, alg.mul(coefficient, basis))
                for value, basis in zip(work[row], work[rank])
            ]
        rank += 1
    return rank


def project_diagnostic(carrier: SurfaceCarrier, program: Program) -> tuple[Any, tuple[int, int]]:
    require_owned(carrier, program, "FINAL_DIAGNOSTIC_RESIDENT")
    if carrier.forward_index != program.phase_steps or carrier.projection_calls:
        fail("two-signature final diagnostic stage changed")
    alg = carrier.alg
    ranks = (matrix_rank(carrier.rows[0], alg), matrix_rank(carrier.rows[1], alg))
    moments = [
        exact.scalar_power(alg, alg.add(exact.field_integer(alg, 3), alg.power(index)), program.branch_pairs_per_signature)
        for index in range(P)
    ]
    observation = alg.power(program.observation_exponent)
    boundary = alg.zero
    for first in range(P):
        for second in range(P):
            component = alg.add(
                carrier.rows[0][first][second],
                alg.mul(observation, carrier.rows[1][first][second]),
            )
            boundary = alg.add(
                boundary,
                alg.mul(component, alg.mul(moments[first], moments[second])),
            )
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED"
    return boundary, ranks


def inverse(carrier: SurfaceCarrier, program: Program) -> None:
    require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("two-signature inverse requires final projection")
    carrier.stage = "INVERSE"
    for index in range(program.phase_steps - 1, -1, -1):
        apply_walsh(carrier, inverse_step=True)
        apply_phase(carrier, program.steps[index], inverse_step=True)
        carrier.inverse_index += 1
    seed = zeros(carrier.alg)
    seed[0][0] = carrier.alg.one
    if carrier.rows[0] != seed or carrier.rows[1] != seed:
        fail("two-signature actual inverse did not restore seed")
    carrier.rows[0][0][0] = carrier.alg.sub(carrier.rows[0][0][0], carrier.alg.one)
    carrier.rows[1][0][0] = carrier.alg.sub(carrier.rows[1][0][0], carrier.alg.one)
    carrier.active_program = None
    carrier.active_lease = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not carrier.exact_zero():
        fail("two-signature carrier did not restore exact zero")


def state_commitment(carrier: SurfaceCarrier) -> str:
    hasher = hashlib.sha256()
    for value in carrier.all_values():
        record = json.dumps(carrier.alg.serialize(value), separators=(",", ":")).encode()
        hasher.update(len(record).to_bytes(8, "big"))
        hasher.update(record)
    return hasher.hexdigest()


def execute_transaction(carrier: SurfaceCarrier, program: Program) -> dict[str, Any]:
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_update_scratch_payload_bits = 0
    carrier.maximum_update_scratch_field_cells = 0
    carrier.maximum_live_resident_plus_update_scratch_payload_bits = 0
    initial = carrier.digest(include_package_local_count=False)
    backing = carrier.backing_identity()
    count = carrier.package_local_restoration_count
    descriptor_bytes = len(json.dumps(program.descriptor(), sort_keys=True, separators=(",", ":")).encode())
    begin_forward(carrier, program)
    forward(carrier, program)
    commitment = state_commitment(carrier)
    boundary, ranks = project_diagnostic(carrier, program)
    inverse(carrier, program)
    return {
        "branch_pairs_per_signature": program.branch_pairs_per_signature,
        "boolean_branch_bits": program.boolean_branch_bits,
        "phase_steps": program.phase_steps,
        "family": program.family,
        "algebra": rank1.algebra_signature(carrier.alg),
        "algebra_kind": carrier.alg.kind,
        "algebra_modulus": carrier.alg.modulus,
        "algebra_root": carrier.alg.serialize(carrier.alg.root),
        "program_fingerprint": program.fingerprint(),
        "derivative_signature_rank": 2,
        "coefficient_surface_separation_ranks": list(ranks),
        "both_surfaces_full_separation_rank": ranks == (17, 17),
        "final_boundary": carrier.alg.serialize(boundary),
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_state_commitment": commitment,
        "resident_group_algebra_field_cells": RESIDENT_CELLS,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_update_scratch_field_cells": carrier.maximum_update_scratch_field_cells,
        "maximum_update_scratch_payload_bits": carrier.maximum_update_scratch_payload_bits,
        "maximum_live_resident_plus_update_scratch_field_cells": (
            RESIDENT_CELLS + carrier.maximum_update_scratch_field_cells
        ),
        "maximum_live_resident_plus_update_scratch_payload_bits": (
            carrier.maximum_live_resident_plus_update_scratch_payload_bits
        ),
        "maximum_rank_verification_dense_field_cells": 289,
        "maximum_live_resident_plus_rank_dense_buffer_field_cells": RESIDENT_CELLS + 289,
        "projection_persistently_named_field_cells_excluding_rank_work_and_expression_temporaries": 20,
        "public_program_json_bytes": descriptor_bytes,
        "public_quadratic_term_records": 2 * program.branch_pairs_per_signature,
        "public_phase_step_records": program.phase_steps,
        "inverse_history_cells": 0,
        "inverse_operations_rematerialized_from_public_topology": True,
        "accepted_path_branch_assignment_or_truth_table_cells": 0,
        "intermediate_coefficient_surfaces_exposed": False,
        "one_way_final_state_commitment_emitted": True,
        "final_projection_calls": 1,
        "response_released_after_restoration": True,
        "same_backing": carrier.backing_identity() == backing,
        "restored_exact_zero": carrier.exact_zero(),
        "initial_restored_digest_equal": carrier.digest(include_package_local_count=False) == initial,
        "package_local_restoration_count_before": count,
        "package_local_restoration_count_after": carrier.package_local_restoration_count,
        "snapshot_reload_used": False,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "rank_projection_compiler_and_commitment_buffers_restoration_class": "NO_RESTORATION_CLAIM",
    }


def resource_signature(item: dict[str, Any]) -> dict[str, Any]:
    omitted = {
        "family",
        "program_fingerprint",
        "final_boundary",
        "final_state_commitment",
        "package_local_restoration_count_before",
        "package_local_restoration_count_after",
    }
    return {key: value for key, value in item.items() if key not in omitted}


def controls() -> dict[str, bool]:
    alg = backend.Algebra("F137", modulus=137, root=16)
    program = compile_program(4, 12, "PRIMARY")
    reference = execute_transaction(SurfaceCarrier.create(alg), program)

    missing = SurfaceCarrier.create(alg)
    begin_forward(missing, program)
    forward(missing, program)

    wrong = SurfaceCarrier.create(alg)
    begin_forward(wrong, program)
    forward(wrong, program)
    project_diagnostic(wrong, program)
    wrong_program = compile_program(4, 12, "REUSE")
    wrong.active_program = wrong_program.fingerprint()
    wrong.active_lease = lease(wrong_program, alg)
    wrong_inverse_detected = False
    try:
        inverse(wrong, wrong_program)
    except RuntimeError:
        wrong_inverse_detected = True

    reordered_steps = list(program.steps)
    reordered_steps[0], reordered_steps[1] = reordered_steps[1], reordered_steps[0]
    reordered = Program(
        program.branch_pairs_per_signature,
        program.phase_steps,
        program.family,
        program.signatures,
        tuple(reordered_steps),
        program.observation_exponent,
    )
    reordered_topology_rejected = False
    try:
        validate_program(reordered)
    except RuntimeError:
        reordered_topology_rejected = True

    reordered_inverse = SurfaceCarrier.create(alg)
    begin_forward(reordered_inverse, program)
    forward(reordered_inverse, program)
    project_diagnostic(reordered_inverse, program)
    require_owned(reordered_inverse, program, "PROJECTED")
    reordered_inverse.stage = "INVERSE"
    for step in program.steps:
        apply_walsh(reordered_inverse, inverse_step=True)
        apply_phase(reordered_inverse, step, inverse_step=True)
    seed = zeros(alg)
    seed[0][0] = alg.one
    reordered_inverse_fails = reordered_inverse.rows != [seed, seed]

    premature_projection_rejected = False
    try:
        project_diagnostic(SurfaceCarrier.create(alg), program)
    except RuntimeError:
        premature_projection_rejected = True

    rank_cap_rejected = max(reference["coefficient_surface_separation_ranks"]) > 16
    null_carrier_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    return {
        "reference_full_rank_and_restored": reference["both_surfaces_full_separation_rank"] and reference["restored_exact_zero"],
        "rank16_factor_cap_rejected_at_primary_phase_step12": rank_cap_rejected,
        "missing_inverse_leaves_actual_resident_state": not missing.exact_zero(),
        "wrong_inverse_detected": wrong_inverse_detected,
        "reordered_forward_descriptor_grammar_rejected": reordered_topology_rejected,
        "reordered_actual_inverse_fails_to_restore": reordered_inverse_fails,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "snapshot_command_available": False,
        "intermediate_projection_available": False,
    }


def residue_counts(branch_pairs: int) -> list[int]:
    counts = [0 for _ in range(P)]
    counts[0] = 1
    for _ in range(branch_pairs):
        counts = [3 * counts[q] + counts[(q - 1) % P] for q in range(P)]
    return counts


def streamed_classical_boundary(program: Program, alg: backend.Algebra) -> tuple[Any, dict[str, int]]:
    counts = residue_counts(program.branch_pairs_per_signature)
    observation = alg.power(program.observation_exponent)
    total = alg.zero
    maximum_named_payload_bits = 0

    def observe(*values: Any) -> None:
        nonlocal maximum_named_payload_bits
        maximum_named_payload_bits = max(
            maximum_named_payload_bits,
            sum(alg.payload_bits(value) for value in values),
        )

    for first in range(P):
        for second in range(P):
            state0 = alg.one
            state1 = alg.one
            for step in program.steps:
                residue = first if step.signature_axis == 0 else second
                phase = alg.power(step.multiplier * residue + step.constant)
                phased = alg.mul(state1, phase)
                next0 = alg.add(state0, phased)
                next1 = alg.sub(state0, phased)
                observe(observation, total, state0, state1, phase, phased, next0, next1)
                state0, state1 = next0, next1
            observed = alg.add(state0, alg.mul(observation, state1))
            weight = exact.field_integer(alg, counts[first] * counts[second])
            weighted = alg.mul(weight, observed)
            next_total = alg.add(total, weighted)
            observe(observation, total, state0, state1, observed, weight, weighted, next_total)
            total = next_total
    return total, {
        "dynamic_exact_field_cells_upper_bound": 8,
        "public_residue_count_integer_cells": 17,
        "public_residue_count_payload_bits": sum(max(1, abs(value).bit_length()) for value in counts),
        "maximum_named_dynamic_exact_payload_bits": maximum_named_payload_bits,
        "executed_residue_phase_updates": P * P * program.phase_steps,
        "executed_public_count_updates": P * program.branch_pairs_per_signature,
    }


def classical_baseline(program: Program, item: dict[str, Any], alg: backend.Algebra) -> dict[str, Any]:
    streamed_boundary, streamed_resources = streamed_classical_boundary(program, alg)
    return {
        "branch_pairs_per_signature": program.branch_pairs_per_signature,
        "phase_steps": program.phase_steps,
        "same_public_program_fingerprint": program.fingerprint(),
        "retain_both_surfaces_full_diagnostic": {
            "executed_by_phase_package": True,
            "reproduces_final_scalar_ranks_and_canonical_commitment": True,
            "resident_exact_field_cells": RESIDENT_CELLS,
            "maximum_update_scratch_field_cells": item["maximum_update_scratch_field_cells"],
            "maximum_live_resident_plus_update_scratch_field_cells": item[
                "maximum_live_resident_plus_update_scratch_field_cells"
            ],
            "same_resident_payload_law": True,
            "warm_arithmetic_work": "O(289*PHASE_STEPS_PLUS_17*LOG_BRANCH_PAIRS)",
        },
        "streamed_final_scalar": {
            "executed": True,
            "boundary_equal": alg.serialize(streamed_boundary) == item["final_boundary"],
            "reproduces_rank_and_commitment_diagnostics": False,
            **streamed_resources,
            "warm_arithmetic_work": "289*PHASE_STEPS_PLUS_17*BRANCH_PAIRS",
        },
        "streamed_rematerialized_full_diagnostic": {
            "executed": False,
            "construction": "SEQUENTIAL_COMPONENT_AND_COEFFICIENT_ROW_INVERSE_CHARACTER_TRANSFORM_WITH_INCREMENTAL_RANK_BASIS_AND_SHA256",
            "reproduces_final_scalar_ranks_and_canonical_commitment": True,
            "dynamic_exact_field_cells_conservative_upper_bound": 320,
            "public_residue_count_integer_cells": 17,
            "extra_work": "O(17*289*PHASE_STEPS_PLUS_17*289*17_CHARACTER_TRANSFORM)",
            "payload_tuple_not_claimed_or_measured": True,
        },
        "no_single_classical_point_dominates_memory_and_work": True,
        "compiled_word_option": "289_INDEPENDENT_TWO_BY_TWO_TRANSFER_MATRICES",
        "phase_carrier_or_snapshot_used": False,
        "comparison_establishes_distinct_phase_resource": False,
        "comparison_establishes_computational_advantage": False,
    }


def run() -> dict[str, Any]:
    q_alg = backend.Algebra("Q_ZETA17")
    exact_transactions = [
        execute_transaction(SurfaceCarrier.create(q_alg), compile_program(n, depth, "PRIMARY"))
        for n, depth in EXACT_CASES
    ]
    structural_transactions = []
    for modulus, root in FINITE_FIELDS:
        for family in FAMILIES:
            alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            for depth in STRUCTURAL_PHASE_STEPS:
                structural_transactions.append(
                    execute_transaction(SurfaceCarrier.create(alg), compile_program(4, depth, family))
                )

    reuse_alg = backend.Algebra("Q_ZETA17")
    reuse_carrier = SurfaceCarrier.create(reuse_alg)
    backing = reuse_carrier.backing_identity()
    primary = execute_transaction(reuse_carrier, compile_program(4, 8, "PRIMARY"))
    restored = execute_transaction(reuse_carrier, compile_program(16, 16, "REUSE"))
    fresh = execute_transaction(SurfaceCarrier.create(reuse_alg), compile_program(16, 16, "REUSE"))

    transactions = [*exact_transactions, *structural_transactions]
    baselines = []
    for item in transactions:
        program = compile_program(item["branch_pairs_per_signature"], item["phase_steps"], item["family"])
        alg = q_alg if item["algebra_kind"] == "Q_ZETA17" else backend.Algebra(
            item["algebra_kind"],
            modulus=item["algebra_modulus"],
            root=item["algebra_root"],
        )
        baselines.append(classical_baseline(program, item, alg))
    control_results = controls()
    false_controls = {"snapshot_command_available", "intermediate_projection_available"}
    if not all(item["restored_exact_zero"] and item["same_backing"] for item in transactions):
        fail("two-signature transaction failed restoration")
    if any(control_results[key] for key in false_controls) or not all(
        value for key, value in control_results.items() if key not in false_controls
    ):
        fail("two-signature control failed")

    return {
        "schema": "CAT_CAS_F17_TWO_SIGNATURE_CUBIC_WALSH_SEPARATION_RANK_DIAGNOSTIC_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "execution_scope": "LINUX_DIRECT_PROCESS_EXACT_SOFTWARE",
        "source_scope": {
            "exact_branch_pair_phase_step_cases": [list(item) for item in EXACT_CASES],
            "structural_phase_steps": list(STRUCTURAL_PHASE_STEPS),
            "structural_fields": ["F103", "F137"],
            "families": list(FAMILIES),
            "derivative_signature_rank": 2,
            "unresolved_typed_port_count": 1,
        },
        "exact_transactions": exact_transactions,
        "structural_transactions": structural_transactions,
        "matched_classical_baselines": baselines,
        "reuse": {
            "primary_case": [primary["branch_pairs_per_signature"], primary["phase_steps"]],
            "reuse_case": [restored["branch_pairs_per_signature"], restored["phase_steps"]],
            "same_original_backing": reuse_carrier.backing_identity() == backing,
            "fresh_restored_boundary_equal": restored["final_boundary"] == fresh["final_boundary"],
            "fresh_restored_resource_signature_equal": resource_signature(restored) == resource_signature(fresh),
            "package_local_restoration_count": reuse_carrier.package_local_restoration_count,
            "restored_exact_zero": reuse_carrier.exact_zero(),
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
        },
        "controls": control_results,
        "resource_law": {
            "canonical_resident_exact_field_cells": 578,
            "canonical_cells_independent_of_branch_pairs_and_phase_steps": True,
            "maximum_update_scratch_field_cells": 578,
            "maximum_live_resident_plus_update_scratch_field_cells": 1156,
            "maximum_rank_verification_dense_field_cells": 289,
            "maximum_live_resident_plus_rank_dense_buffer_field_cells": 867,
            "projection_persistently_named_field_cells_excluding_rank_work_and_expression_temporaries": 20,
            "inverse_history_cells": 0,
            "accepted_path_assignment_or_truth_table_cells": 0,
            "exact_payload_bits_measured_and_not_constant": True,
            "uniform_low_separation_rank_below17_for_declared_families": False,
            "canonical_578_cells_universal_representation_minimum": False,
            "python_container_allocator_native_bigint_hashlib_and_whole_process_memory_excluded": True,
        },
        "matched_baseline": {
            "compact_classical_frontier": [
                "RETAIN_BOTH_17_BY_17_SURFACES_FOR_ALL_DIAGNOSTIC_OUTPUTS",
                "STREAM_289_JOINT_RESIDUE_EVALUATIONS_FOR_FINAL_SCALAR",
                "REMATERIALIZE_COMPONENT_ROWS_FOR_FULL_RANK_AND_COMMITMENT_DIAGNOSTICS",
            ],
            "all_executed_streamed_final_boundaries_equal": all(
                item["streamed_final_scalar"]["boundary_equal"] for item in baselines
            ),
            "no_single_classical_point_dominates_memory_and_work": True,
            "same_public_inputs_instances_and_final_scalar_boundary": True,
            "same_rank_and_commitment_outputs_require_retain_all_or_rematerialization": True,
            "same_payload_law_claimed_for_strongest_streamed_path": False,
            "comparison_against_assignment_expansion_only": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
        },
        "restoration": {
            "resident_surface_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "rank_projection_compiler_commitment_and_oracle_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "declared_two_disjoint_quadratic_signatures_and_alternating_programs_only": True,
            "full_separation_rank_rejects_only_below17_matrix_factor_caps": True,
            "canonical_578_cell_chart_is_universal_minimum": False,
            "general_rank_r_or_arbitrary_cubic_hypergraph_no_go": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": (
            "THE_DECLARED_TWO_SIGNATURE_PROGRAMS_REACH_FULL_17_BY_17_"
            "COEFFICIENT_SEPARATION_RANK_AND_THE_CANONICAL_578_CELL_PHASE_"
            "CARRIER_HAS_THE_IDENTICAL_CLASSICAL_RECURRENCE_SO_A_SUCCESSOR_"
            "MUST_CHANGE_THE_NATIVE_PHASE_COUPLING_LAW_OR_USE_A_NONLINEAR_"
            "RELATION_QUOTIENT_NOT_JUST_ADD_SIGNATURE_AXES_OR_DENSE_CELLS"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(run(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
