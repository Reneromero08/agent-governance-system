#!/usr/bin/env python3
"""Exact nonlinear radial phase quotient on the anisotropic F17 plane.

The public coordinate space is F17^2 with anisotropic norm

    Q(x, y) = x^2 - 3 y^2.

Because 3 is a quadratic nonresidue modulo 17, Q=0 contains only the
origin and every nonzero norm shell has 18 points.  Functions constant on
the 17 norm shells form an exact invariant subspace under the normalized
phase Fourier kernel zeta^(u*x-3*v*y)/17.  Quartic radial phase gates
zeta^(a*Q^2+b*Q+c) and that Fourier gate therefore close on 17 unresolved
exact coefficients even though the underlying coordinate function has 289
values.

This is a bounded nonlinear relation-quotient mechanism.  It has an
identical 17-coordinate compact classical recurrence and makes no claim of
computational advantage, CATVM custody, physical execution, or a distinct
phase resource.
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
NONSQUARE = 3
COORDINATE_CELLS = P * P
QUOTIENT_CELLS = P
EXACT_DEPTHS = (1, 2, 4, 8, 16, 32, 64)
STRUCTURAL_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
FINITE_FIELDS = ((103, 72), (137, 16))
CLAIM = (
    "EXACT_ANISOTROPIC_F17_SQUARED_NORM_RELATION_QUOTIENT_CLOSES_"
    "NONCOMMUTING_QUARTIC_RADIAL_PHASE_AND_NORMALIZED_PHASE_FOURIER_"
    "OPERATIONS_ON_17_UNRESOLVED_EXACT_CELLS_ACROSS_GROWING_PROGRAM_"
    "DEPTH_WITH_EXACT_RESTORATION_AND_REUSE_BUT_THE_IDENTICAL_17_"
    "COORDINATE_CLASSICAL_RECURRENCE_AND_GROWING_EXACT_PAYLOAD_REMAIN"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def algebra_signature(alg: backend.Algebra) -> dict[str, Any]:
    return rank1.algebra_signature(alg)


def norm(x: int, y: int, coefficient: int = NONSQUARE) -> int:
    return (x * x - coefficient * y * y) % P


def pairing(u: int, v: int, x: int, y: int) -> int:
    return (u * x - NONSQUARE * v * y) % P


SHELL_COUNTS = (1, *([18] * 16))


def coordinates() -> Iterable[tuple[int, int]]:
    for x in range(P):
        for y in range(P):
            yield x, y


def representatives() -> tuple[tuple[int, int], ...]:
    found: list[tuple[int, int] | None] = [None for _ in range(P)]
    for point in coordinates():
        residue = norm(*point)
        if found[residue] is None:
            found[residue] = point
    if any(point is None for point in found):
        fail("public anisotropic norm does not represent every F17 residue")
    return tuple(point for point in found if point is not None)


@dataclass(frozen=True)
class RadialGate:
    quadratic_multiplier: int
    linear_multiplier: int
    constant: int

    def exponent(self, shell: int) -> int:
        return (
            self.quadratic_multiplier * shell * shell
            + self.linear_multiplier * shell
            + self.constant
        ) % P

    def as_json(self) -> dict[str, int | str]:
        return {
            "kind": "QUARTIC_ANISOTROPIC_NORM_PHASE_THEN_NORMALIZED_PHASE_FOURIER",
            "quadratic_multiplier_mod17": self.quadratic_multiplier % P,
            "linear_multiplier_mod17": self.linear_multiplier % P,
            "constant_mod17": self.constant % P,
        }


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    gates: tuple[RadialGate, ...]
    observation_quadratic: int
    observation_linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "coordinate_field": "F17_SQUARED",
            "typed_relation": "ANISOTROPIC_NORM_Q_EQUALS_X2_MINUS_3Y2",
            "unresolved_port": "Q:F17_ANISOTROPIC_NORM_SHELL",
            "depth": self.depth,
            "family": self.family,
            "gates": [gate.as_json() for gate in self.gates],
            "final_observation": {
                "kind": "RADIAL_QUARTIC_SCALAR",
                "quadratic_multiplier_mod17": self.observation_quadratic,
                "linear_multiplier_mod17": self.observation_linear,
            },
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def gate_parameters(index: int, family: str) -> tuple[int, int, int]:
    bit_weight = index.bit_count()
    gray_weight = (index ^ (index >> 1)).bit_count()
    ternary_weight = 0
    remaining = index
    while remaining:
        ternary_weight += remaining % 3
        remaining //= 3
    if family == "PRIMARY":
        values = (3 * index + 5 * bit_weight + 1, 7 * index + 2 * bit_weight + 2, 11 * index + bit_weight + 4)
    elif family == "REUSE":
        values = (5 * index + 2 * ternary_weight + 3, 4 * index + 3 * ternary_weight + 6, 9 * index + ternary_weight + 8)
    elif family == "ALTERNATE":
        values = (7 * index + 3 * gray_weight + 2, 8 * index + 2 * gray_weight + 5, 6 * index + gray_weight + 1)
    else:
        fail("unknown anisotropic radial program family")
    quadratic = values[0] % P
    if quadratic == 0:
        quadratic = 1
    return quadratic, values[1] % P, values[2] % P


def compile_program(depth: int, family: str) -> Program:
    if depth < 1 or depth > 128:
        fail("anisotropic radial program depth outside declared ceiling")
    gates = tuple(RadialGate(*gate_parameters(index, family)) for index in range(depth))
    program = Program(
        depth,
        family,
        gates,
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )
    validate_program(program)
    return program


def validate_program(program: Program) -> None:
    if program.family not in FAMILIES or len(program.gates) != program.depth:
        fail("anisotropic radial public program descriptor changed")
    if program.gates != tuple(
        RadialGate(*gate_parameters(index, program.family)) for index in range(program.depth)
    ):
        fail("anisotropic radial public topology compiler changed")
    if any(gate.quadratic_multiplier % P == 0 for gate in program.gates):
        fail("quartic radial phase multiplier became zero")


def validate_public_geometry() -> tuple[tuple[int, int], ...]:
    squares = {value * value % P for value in range(P)}
    if NONSQUARE in squares:
        fail("declared anisotropic coefficient became a square")
    observed_counts = [0 for _ in range(P)]
    found: list[tuple[int, int] | None] = [None for _ in range(P)]
    for point in coordinates():
        residue = norm(*point)
        observed_counts[residue] += 1
        if found[residue] is None:
            found[residue] = point
    if tuple(observed_counts) != SHELL_COUNTS:
        fail("anisotropic norm-shell cardinality law changed")
    if sum(observed_counts) != COORDINATE_CELLS:
        fail("public coordinate topology changed")
    if any(point is None for point in found):
        fail("public anisotropic norm does not represent every F17 residue")
    public_representatives = tuple(point for point in found if point is not None)
    if any(norm(*public_representatives[q]) != q for q in range(P)):
        fail("public norm-shell representative compiler changed")
    return public_representatives


@dataclass(frozen=True)
class CompiledGeometry:
    alg: backend.Algebra
    normalized_fourier: tuple[tuple[Any, ...], ...]
    compiled_kernel_payload_bits: int
    public_coordinate_visits: int

    @classmethod
    def compile(cls, alg: backend.Algebra) -> "CompiledGeometry":
        public_representatives = validate_public_geometry()
        inverse17 = alg.inverse(exact.field_integer(alg, P))
        rows: list[tuple[Any, ...]] = []
        for u, v in public_representatives:
            row = [alg.zero for _ in range(P)]
            for x, y in coordinates():
                residue = norm(x, y)
                row[residue] = alg.add(
                    row[residue], alg.power(pairing(u, v, x, y))
                )
            rows.append(tuple(alg.mul(inverse17, value) for value in row))
        normalized = tuple(rows)

        for first in range(P):
            for second in range(P):
                value = alg.zero
                for middle in range(P):
                    value = alg.add(
                        value,
                        alg.mul(normalized[first][middle], normalized[middle][second]),
                    )
                expected = alg.one if first == second else alg.zero
                if value != expected:
                    fail("normalized radial Fourier is not an exact involution")
        return cls(
            alg,
            normalized,
            sum(alg.payload_bits(value) for row in normalized for value in row),
            COORDINATE_CELLS + COORDINATE_CELLS * P,
        )


def verify_public_quotient_invariance(geometry: CompiledGeometry) -> dict[str, int | bool]:
    """Exhaustive verification control, outside the accepted transaction."""
    alg = geometry.alg
    inverse17 = alg.inverse(exact.field_integer(alg, P))
    quotient_row_checks = 0
    target_coordinate_visits = 0
    source_coordinate_visits = 0
    for u, v in coordinates():
        target_coordinate_visits += 1
        representative_row = geometry.normalized_fourier[norm(u, v)]
        row = [alg.zero for _ in range(P)]
        for x, y in coordinates():
            source_coordinate_visits += 1
            residue = norm(x, y)
            row[residue] = alg.add(
                row[residue], alg.power(pairing(u, v, x, y))
            )
        for q, value in enumerate(row):
            quotient_row_checks += 1
            if alg.mul(inverse17, value) != representative_row[q]:
                fail("anisotropic norm quotient is not Fourier invariant")
    return {
        "all_289_target_rows_factor_through_norm": True,
        "quotient_row_checks": quotient_row_checks,
        "target_coordinate_visits": target_coordinate_visits,
        "source_coordinate_visits": source_coordinate_visits,
        "total_coordinate_visits": target_coordinate_visits + source_coordinate_visits,
    }


def lease(program: Program, geometry: CompiledGeometry) -> str:
    return digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(geometry.alg),
            "carrier": "ANISOTROPIC_F17_SQUARED_NORM_RELATION_QUOTIENT",
            "resident_cells": QUOTIENT_CELLS,
        }
    )


@dataclass
class RadialCarrier:
    geometry: CompiledGeometry
    cells: list[Any]
    active_program: str | None = None
    active_lease: str | None = None
    stage: str = "RESTORED"
    forward_index: int = 0
    inverse_index: int = 0
    projection_calls: int = 0
    package_local_restoration_count: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_update_scratch_field_cells: int = 0
    maximum_update_scratch_payload_bits: int = 0
    maximum_live_resident_plus_update_scratch_payload_bits: int = 0
    maximum_projection_scratch_field_cells: int = 0
    maximum_projection_scratch_payload_bits: int = 0
    maximum_live_resident_plus_projection_scratch_payload_bits: int = 0
    maximum_commitment_serialized_record_bytes: int = 0

    @classmethod
    def create(cls, geometry: CompiledGeometry) -> "RadialCarrier":
        return cls(geometry, [geometry.alg.zero for _ in range(P)])

    @property
    def alg(self) -> backend.Algebra:
        return self.geometry.alg

    def backing_identity(self) -> tuple[int, int]:
        return id(self), id(self.cells)

    def exact_zero(self) -> bool:
        return (
            self.cells == [self.alg.zero for _ in range(P)]
            and self.active_program is None
            and self.active_lease is None
            and self.stage == "RESTORED"
            and self.forward_index == 0
            and self.inverse_index == 0
            and self.projection_calls == 0
        )

    def observe_resident(self) -> None:
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.cells),
        )

    def observe_scratch(self, values: Iterable[Any]) -> None:
        materialized = list(values)
        payload = sum(self.alg.payload_bits(value) for value in materialized)
        self.maximum_update_scratch_field_cells = max(
            self.maximum_update_scratch_field_cells, len(materialized)
        )
        self.maximum_update_scratch_payload_bits = max(
            self.maximum_update_scratch_payload_bits, payload
        )
        self.maximum_live_resident_plus_update_scratch_payload_bits = max(
            self.maximum_live_resident_plus_update_scratch_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.cells) + payload,
        )

    def observe_projection_scratch(self, values: Iterable[Any]) -> None:
        materialized = list(values)
        payload = sum(self.alg.payload_bits(value) for value in materialized)
        self.maximum_projection_scratch_field_cells = max(
            self.maximum_projection_scratch_field_cells, len(materialized)
        )
        self.maximum_projection_scratch_payload_bits = max(
            self.maximum_projection_scratch_payload_bits, payload
        )
        self.maximum_live_resident_plus_projection_scratch_payload_bits = max(
            self.maximum_live_resident_plus_projection_scratch_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.cells) + payload,
        )

    def digest(self, include_package_local_count: bool = True) -> str:
        state: dict[str, Any] = {
            "cells": [self.alg.serialize(value) for value in self.cells],
            "active_program": self.active_program,
            "active_lease": self.active_lease,
            "stage": self.stage,
            "forward_index": self.forward_index,
            "inverse_index": self.inverse_index,
            "projection_calls": self.projection_calls,
        }
        if include_package_local_count:
            state["package_local_restoration_count"] = self.package_local_restoration_count
        return digest_json(state)


def require_owned(carrier: RadialCarrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, RadialCarrier):
        fail("null or wrong anisotropic radial carrier")
    if (
        carrier.stage != stage
        or carrier.active_program != program.fingerprint()
        or carrier.active_lease != lease(program, carrier.geometry)
    ):
        fail("anisotropic radial carrier owner or stage changed")


def apply_phase(carrier: RadialCarrier, gate: RadialGate, *, inverse: bool = False) -> None:
    scratch = []
    sign = -1 if inverse else 1
    for shell, value in enumerate(carrier.cells):
        scratch.append(carrier.alg.mul(value, carrier.alg.power(sign * gate.exponent(shell))))
    carrier.observe_scratch(scratch)
    carrier.cells[:] = scratch
    carrier.observe_resident()


def apply_fourier(carrier: RadialCarrier) -> None:
    alg = carrier.alg
    output = [alg.zero for _ in range(P)]
    for target in range(P):
        accumulator = alg.zero
        for source in range(P):
            product = alg.mul(
                carrier.geometry.normalized_fourier[target][source], carrier.cells[source]
            )
            updated = alg.add(accumulator, product)
            carrier.observe_scratch((*output, accumulator, product, updated))
            accumulator = updated
        output[target] = accumulator
    carrier.observe_scratch(output)
    carrier.cells[:] = output
    carrier.observe_resident()


def begin_forward(carrier: RadialCarrier, program: Program) -> None:
    validate_program(program)
    if not isinstance(carrier, RadialCarrier) or not carrier.exact_zero():
        fail("anisotropic radial carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.active_lease = lease(program, carrier.geometry)
    carrier.stage = "FORWARD"
    carrier.cells[:] = [carrier.alg.one for _ in range(P)]
    carrier.observe_resident()


def forward(carrier: RadialCarrier, program: Program) -> None:
    require_owned(carrier, program, "FORWARD")
    for index, gate in enumerate(program.gates):
        apply_phase(carrier, gate)
        apply_fourier(carrier)
        carrier.forward_index = index + 1
    carrier.stage = "FINAL_RADIAL_RELATION_RESIDENT"


def state_commitment(carrier: RadialCarrier) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    total_bytes = 0
    for value in carrier.cells:
        record = json.dumps(carrier.alg.serialize(value), separators=(",", ":")).encode()
        total_bytes += len(record)
        carrier.maximum_commitment_serialized_record_bytes = max(
            carrier.maximum_commitment_serialized_record_bytes, len(record)
        )
        hasher.update(len(record).to_bytes(8, "big"))
        hasher.update(record)
    return hasher.hexdigest(), total_bytes, carrier.maximum_commitment_serialized_record_bytes


def project(carrier: RadialCarrier, program: Program) -> Any:
    require_owned(carrier, program, "FINAL_RADIAL_RELATION_RESIDENT")
    if carrier.forward_index != program.depth or carrier.projection_calls:
        fail("anisotropic radial projection order changed")
    total = carrier.alg.zero
    for shell, value in enumerate(carrier.cells):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        phase = carrier.alg.power(exponent)
        weight = exact.field_integer(carrier.alg, SHELL_COUNTS[shell])
        phased = carrier.alg.mul(phase, value)
        weighted = carrier.alg.mul(weight, phased)
        updated = carrier.alg.add(total, weighted)
        carrier.observe_projection_scratch((total, phase, weight, phased, weighted, updated))
        total = updated
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED"
    return total


def inverse(carrier: RadialCarrier, program: Program) -> None:
    require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("anisotropic radial inverse requires final projection")
    carrier.stage = "INVERSE"
    for index in range(program.depth - 1, -1, -1):
        apply_fourier(carrier)
        apply_phase(carrier, program.gates[index], inverse=True)
        carrier.inverse_index += 1
    if carrier.cells != [carrier.alg.one for _ in range(P)]:
        fail("anisotropic radial actual inverse did not restore seed")
    carrier.cells[:] = [carrier.alg.sub(value, carrier.alg.one) for value in carrier.cells]
    carrier.active_program = None
    carrier.active_lease = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not carrier.exact_zero():
        fail("anisotropic radial carrier did not restore exact zero")


def stats_snapshot(alg: backend.Algebra) -> dict[str, int]:
    return {name: int(value) for name, value in vars(alg.stats).items()}


def stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before.get(key, 0) for key in after}


def execute_transaction(carrier: RadialCarrier, program: Program) -> dict[str, Any]:
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_update_scratch_field_cells = 0
    carrier.maximum_update_scratch_payload_bits = 0
    carrier.maximum_live_resident_plus_update_scratch_payload_bits = 0
    carrier.maximum_projection_scratch_field_cells = 0
    carrier.maximum_projection_scratch_payload_bits = 0
    carrier.maximum_live_resident_plus_projection_scratch_payload_bits = 0
    carrier.maximum_commitment_serialized_record_bytes = 0
    initial = carrier.digest(include_package_local_count=False)
    backing = carrier.backing_identity()
    restoration_count = carrier.package_local_restoration_count
    before_stats = stats_snapshot(carrier.alg)
    descriptor_bytes = len(
        json.dumps(program.descriptor(), sort_keys=True, separators=(",", ":")).encode()
    )
    begin_forward(carrier, program)
    forward(carrier, program)
    commitment, commitment_input_bytes, commitment_record_bytes = state_commitment(carrier)
    boundary = project(carrier, program)
    inverse(carrier, program)
    return {
        "depth": program.depth,
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "algebra_kind": carrier.alg.kind,
        "algebra_modulus": carrier.alg.modulus,
        "algebra_root": carrier.alg.serialize(carrier.alg.root),
        "program_fingerprint": program.fingerprint(),
        "final_boundary": carrier.alg.serialize(boundary),
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_state_commitment": commitment,
        "commitment_serialized_input_bytes": commitment_input_bytes,
        "maximum_commitment_serialized_record_bytes": commitment_record_bytes,
        "commitment_length_prefix_bytes_per_record": 8,
        "commitment_sha256_digest_bytes": 32,
        "commitment_hashlib_internal_state_bytes_measured": False,
        "resident_relation_quotient_field_cells": QUOTIENT_CELLS,
        "represented_coordinate_cells": COORDINATE_CELLS,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_update_scratch_field_cells": carrier.maximum_update_scratch_field_cells,
        "maximum_update_scratch_payload_bits": carrier.maximum_update_scratch_payload_bits,
        "maximum_live_resident_plus_update_scratch_field_cells": (
            QUOTIENT_CELLS + carrier.maximum_update_scratch_field_cells
        ),
        "maximum_live_resident_plus_update_scratch_payload_bits": (
            carrier.maximum_live_resident_plus_update_scratch_payload_bits
        ),
        "compiled_public_fourier_kernel_field_cells": P * P,
        "compiled_public_fourier_kernel_payload_bits": (
            carrier.geometry.compiled_kernel_payload_bits
        ),
        "public_geometry_coordinate_visits_at_compile": carrier.geometry.public_coordinate_visits,
        "public_fourier_involution_check_multiply_accumulates_at_compile": P * P * P,
        "maximum_live_public_compile_exact_field_cells_excluding_expression_temporaries": 306,
        "maximum_projection_scratch_field_cells": carrier.maximum_projection_scratch_field_cells,
        "maximum_projection_scratch_payload_bits": carrier.maximum_projection_scratch_payload_bits,
        "maximum_live_resident_plus_projection_scratch_field_cells": (
            QUOTIENT_CELLS + carrier.maximum_projection_scratch_field_cells
        ),
        "maximum_live_resident_plus_projection_scratch_payload_bits": (
            carrier.maximum_live_resident_plus_projection_scratch_payload_bits
        ),
        "public_program_json_bytes": descriptor_bytes,
        "inverse_history_cells": 0,
        "inverse_operations_rematerialized_from_public_topology": True,
        "accepted_path_assignment_or_truth_table_cells": 0,
        "accepted_path_public_coordinate_expansion_cells": 0,
        "compiled_public_coordinate_topology_not_answer_data": True,
        "intermediate_relation_quotient_exposed": False,
        "final_projection_calls": 1,
        "response_released_after_restoration": True,
        "same_backing": carrier.backing_identity() == backing,
        "restored_exact_zero": carrier.exact_zero(),
        "initial_restored_digest_equal": (
            carrier.digest(include_package_local_count=False) == initial
        ),
        "package_local_restoration_count_before": restoration_count,
        "package_local_restoration_count_after": carrier.package_local_restoration_count,
        "snapshot_reload_used": False,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "compiled_kernel_projection_and_commitment_buffers_restoration_class": (
            "NO_RESTORATION_CLAIM"
        ),
        "native_operation_counts": stats_delta(before_stats, stats_snapshot(carrier.alg)),
    }


def classical_quotient_boundary(
    program: Program, geometry: CompiledGeometry
) -> tuple[Any, dict[str, Any], list[Any]]:
    alg = geometry.alg
    state = [alg.one for _ in range(P)]
    maximum_state_payload = sum(alg.payload_bits(value) for value in state)
    maximum_live_state_plus_scratch_payload = maximum_state_payload
    maximum_projection_scratch_payload = 0
    maximum_live_state_plus_projection_scratch_payload = 0
    for gate in program.gates:
        for shell, value in enumerate(state):
            state[shell] = alg.mul(value, alg.power(gate.exponent(shell)))
        maximum_state_payload = max(
            maximum_state_payload, sum(alg.payload_bits(value) for value in state)
        )
        transformed = [alg.zero for _ in range(P)]
        for target in range(P):
            accumulator = alg.zero
            for source in range(P):
                product = alg.mul(
                    geometry.normalized_fourier[target][source], state[source]
                )
                updated = alg.add(accumulator, product)
                maximum_live_state_plus_scratch_payload = max(
                    maximum_live_state_plus_scratch_payload,
                    sum(alg.payload_bits(value) for value in state)
                    + sum(alg.payload_bits(value) for value in transformed)
                    + alg.payload_bits(accumulator)
                    + alg.payload_bits(product)
                    + alg.payload_bits(updated),
                )
                accumulator = updated
            transformed[target] = accumulator
        state = transformed
        maximum_state_payload = max(
            maximum_state_payload, sum(alg.payload_bits(value) for value in state)
        )
    total = alg.zero
    for shell, value in enumerate(state):
        phase = alg.power(
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        )
        weight = exact.field_integer(alg, SHELL_COUNTS[shell])
        phased = alg.mul(phase, value)
        weighted = alg.mul(weight, phased)
        updated = alg.add(total, weighted)
        scratch_payload = sum(
            alg.payload_bits(item)
            for item in (total, phase, weight, phased, weighted, updated)
        )
        maximum_projection_scratch_payload = max(
            maximum_projection_scratch_payload, scratch_payload
        )
        maximum_live_state_plus_projection_scratch_payload = max(
            maximum_live_state_plus_projection_scratch_payload,
            sum(alg.payload_bits(item) for item in state) + scratch_payload,
        )
        total = updated
    return total, {
        "resident_exact_field_cells": 17,
        "maximum_update_scratch_exact_field_cells": 20,
        "maximum_live_state_plus_scratch_exact_field_cells": 37,
        "compiled_public_kernel_exact_field_cells": 289,
        "maximum_resident_payload_bits": maximum_state_payload,
        "maximum_live_state_plus_scratch_payload_bits": (
            maximum_live_state_plus_scratch_payload
        ),
        "maximum_projection_scratch_exact_field_cells": 6,
        "maximum_live_state_plus_projection_scratch_exact_field_cells": 23,
        "maximum_projection_scratch_payload_bits": maximum_projection_scratch_payload,
        "maximum_live_state_plus_projection_scratch_payload_bits": (
            maximum_live_state_plus_projection_scratch_payload
        ),
        "warm_arithmetic_work": "O(289*DEPTH)",
        "universal_optimality_or_lower_bound_claimed": False,
    }, state


def dense_coordinate_boundary(
    program: Program, alg: backend.Algebra
) -> tuple[Any, bool, list[Any]]:
    """Independent-shaped 289-coordinate control using separable 2D DFTs."""
    inverse17 = alg.inverse(exact.field_integer(alg, P))
    state = [[alg.one for _ in range(P)] for _ in range(P)]
    for gate in program.gates:
        for x, y in coordinates():
            state[x][y] = alg.mul(
                state[x][y], alg.power(gate.exponent(norm(x, y)))
            )
        first = [[alg.zero for _ in range(P)] for _ in range(P)]
        for u in range(P):
            for y in range(P):
                for x in range(P):
                    first[u][y] = alg.add(
                        first[u][y], alg.mul(alg.power(u * x), state[x][y])
                    )
        transformed = [[alg.zero for _ in range(P)] for _ in range(P)]
        for u in range(P):
            for v in range(P):
                for y in range(P):
                    transformed[u][v] = alg.add(
                        transformed[u][v],
                        alg.mul(alg.power(-NONSQUARE * v * y), first[u][y]),
                    )
                transformed[u][v] = alg.mul(inverse17, transformed[u][v])
        state = transformed
    public_representatives = representatives()
    radial = all(
        state[x][y]
        == state[public_representatives[norm(x, y)][0]][public_representatives[norm(x, y)][1]]
        for x, y in coordinates()
    )
    total = alg.zero
    for x, y in coordinates():
        shell = norm(x, y)
        phase = alg.power(
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        )
        total = alg.add(total, alg.mul(phase, state[x][y]))
    shell_values = [
        state[x][y] for x, y in public_representatives
    ]
    return total, radial, shell_values


def controls(geometry: CompiledGeometry) -> dict[str, bool]:
    alg = geometry.alg
    program = compile_program(4, "PRIMARY")
    reference_carrier = RadialCarrier.create(geometry)
    reference = execute_transaction(reference_carrier, program)

    missing = RadialCarrier.create(geometry)
    begin_forward(missing, program)
    forward(missing, program)

    reordered = RadialCarrier.create(geometry)
    begin_forward(reordered, program)
    forward(reordered, program)
    project(reordered, program)
    reordered.stage = "INVERSE"
    for gate in reversed(program.gates):
        apply_phase(reordered, gate, inverse=True)
        apply_fourier(reordered)
    reordered_inverse_fails = reordered.cells != [alg.one for _ in range(P)]

    wrong = RadialCarrier.create(geometry)
    begin_forward(wrong, program)
    forward(wrong, program)
    project(wrong, program)
    wrong.stage = "INVERSE"
    for index in range(program.depth - 1, -1, -1):
        apply_fourier(wrong)
        gate = program.gates[index]
        perturbed = RadialGate(
            (gate.quadratic_multiplier + (1 if index == 0 else 0)) % P,
            gate.linear_multiplier,
            gate.constant,
        )
        apply_phase(wrong, perturbed, inverse=True)
    wrong_inverse_fails = wrong.cells != [alg.one for _ in range(P)]

    premature_projection_rejected = False
    try:
        project(RadialCarrier.create(geometry), program)
    except RuntimeError:
        premature_projection_rejected = True

    null_carrier_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    overmerge_sign_rejected = any(
        program.gates[0].exponent(shell) != program.gates[0].exponent((-shell) % P)
        for shell in range(1, P)
    )

    # The isotropic x^2-y^2 norm has a nonzero zero-shell point.  Its Fourier
    # image differs from the origin for a shell-constant input, so a 17-class
    # quotient that merges those two orbits is not closed.
    isotropic_shells = tuple(
        tuple(point for point in coordinates() if norm(*point, coefficient=1) == q)
        for q in range(P)
    )
    nonzero_zero = next(point for point in isotropic_shells[0] if point != (0, 0))
    origin_row = []
    nonzero_row = []
    for shell in isotropic_shells:
        origin_row.append(sum(alg.power(0) for _ in shell))
        u, v = nonzero_zero
        value = alg.zero
        for x, y in shell:
            value = alg.add(value, alg.power((u * x - v * y) % P))
        nonzero_row.append(value)
    isotropic_zero_orbit_overmerge_rejected = origin_row != nonzero_row

    dense_boundary, dense_radial, dense_shell_values = dense_coordinate_boundary(
        compile_program(2, "ALTERNATE"), alg
    )
    compact_boundary, _, compact_shell_values = classical_quotient_boundary(
        compile_program(2, "ALTERNATE"), geometry
    )

    return {
        "reference_restores_exactly": reference["restored_exact_zero"],
        "missing_inverse_leaves_actual_resident_state": not missing.exact_zero(),
        "reordered_actual_inverse_fails_to_restore": reordered_inverse_fails,
        "wrong_phase_inverse_fails_to_restore": wrong_inverse_fails,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "sign_orbit_overmerge_rejected_by_quartic_phase": overmerge_sign_rejected,
        "isotropic_zero_orbit_overmerge_rejected": isotropic_zero_orbit_overmerge_rejected,
        "dense_289_coordinate_control_remains_radial": dense_radial,
        "dense_289_coordinate_boundary_matches_17_class_quotient": (
            alg.serialize(dense_boundary) == alg.serialize(compact_boundary)
        ),
        "dense_289_coordinate_all_17_shell_values_match_quotient": (
            dense_shell_values == compact_shell_values
        ),
        "snapshot_command_available": False,
        "intermediate_projection_available": False,
        "nonradial_phase_descriptor_available": False,
    }


def resource_signature(item: dict[str, Any]) -> dict[str, Any]:
    omitted = {
        "family",
        "program_fingerprint",
        "final_boundary",
        "final_state_commitment",
        "package_local_restoration_count_before",
        "package_local_restoration_count_after",
        "native_operation_counts",
    }
    return {key: value for key, value in item.items() if key not in omitted}


def run() -> dict[str, Any]:
    exact_alg = backend.Algebra("Q_ZETA17")
    exact_geometry = CompiledGeometry.compile(exact_alg)
    invariance_verification = [
        {"algebra_kind": "Q_ZETA17", **verify_public_quotient_invariance(exact_geometry)}
    ]
    exact_transactions = [
        execute_transaction(
            RadialCarrier.create(exact_geometry), compile_program(depth, "PRIMARY")
        )
        for depth in EXACT_DEPTHS
    ]

    structural_transactions = []
    geometries: dict[str, CompiledGeometry] = {}
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        geometry = CompiledGeometry.compile(alg)
        geometries[f"F{modulus}"] = geometry
        invariance_verification.append(
            {
                "algebra_kind": f"F{modulus}",
                **verify_public_quotient_invariance(geometry),
            }
        )
        for family in FAMILIES:
            for depth in STRUCTURAL_DEPTHS:
                structural_transactions.append(
                    execute_transaction(
                        RadialCarrier.create(geometry), compile_program(depth, family)
                    )
                )

    baselines = []
    for item in [*exact_transactions, *structural_transactions]:
        geometry = (
            exact_geometry
            if item["algebra_kind"] == "Q_ZETA17"
            else geometries[item["algebra_kind"]]
        )
        program = compile_program(item["depth"], item["family"])
        boundary, resources, _ = classical_quotient_boundary(program, geometry)
        baselines.append(
            {
                "depth": program.depth,
                "family": program.family,
                "program_fingerprint": program.fingerprint(),
                "boundary_equal": geometry.alg.serialize(boundary) == item["final_boundary"],
                **resources,
            }
        )

    reuse_carrier = RadialCarrier.create(exact_geometry)
    original_backing = reuse_carrier.backing_identity()
    first = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    restored = execute_transaction(reuse_carrier, compile_program(16, "REUSE"))
    fresh = execute_transaction(
        RadialCarrier.create(exact_geometry), compile_program(16, "REUSE")
    )

    control_results = controls(geometries["F137"])
    false_controls = {
        "snapshot_command_available",
        "intermediate_projection_available",
        "nonradial_phase_descriptor_available",
    }
    if any(control_results[key] for key in false_controls) or not all(
        value for key, value in control_results.items() if key not in false_controls
    ):
        fail("anisotropic radial control failed")
    if not all(item["restored_exact_zero"] and item["same_backing"] for item in exact_transactions + structural_transactions):
        fail("anisotropic radial transaction restoration failed")
    if not all(item["boundary_equal"] for item in baselines):
        fail("matched classical quotient recurrence diverged")

    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_QUARTIC_RADIAL_PHASE_QUOTIENT_CLOSURE_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "execution_scope": "LINUX_DIRECT_PROCESS_EXACT_SOFTWARE",
        "source_scope": {
            "coordinate_field": "F17_SQUARED",
            "anisotropic_norm": "X_SQUARED_MINUS_3_Y_SQUARED",
            "norm_shell_count": 17,
            "norm_shell_sizes": list(SHELL_COUNTS),
            "exact_depths": list(EXACT_DEPTHS),
            "structural_depths": list(STRUCTURAL_DEPTHS),
            "families": list(FAMILIES),
            "operations": [
                "QUARTIC_RADIAL_PHASE",
                "NORMALIZED_ANISOTROPIC_PHASE_FOURIER",
            ],
        },
        "exact_transactions": exact_transactions,
        "structural_transactions": structural_transactions,
        "matched_classical_baselines": baselines,
        "reuse": {
            "first_depth_family": [first["depth"], first["family"]],
            "unrelated_reuse_depth_family": [restored["depth"], restored["family"]],
            "same_original_backing": reuse_carrier.backing_identity() == original_backing,
            "fresh_restored_boundary_equal": restored["final_boundary"] == fresh["final_boundary"],
            "fresh_restored_resource_signature_equal": (
                resource_signature(restored) == resource_signature(fresh)
            ),
            "package_local_restoration_count": reuse_carrier.package_local_restoration_count,
            "restored_exact_zero": reuse_carrier.exact_zero(),
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
        },
        "controls": control_results,
        "quotient_invariance_verification_outside_accepted_transaction": (
            invariance_verification
        ),
        "resource_law": {
            "represented_coordinate_field_cells": 289,
            "resident_relation_quotient_exact_field_cells": 17,
            "resident_cells_independent_of_program_depth": True,
            "maximum_update_scratch_field_cells": max(
                item["maximum_update_scratch_field_cells"] for item in exact_transactions
            ),
            "maximum_live_resident_plus_update_scratch_field_cells": max(
                item["maximum_live_resident_plus_update_scratch_field_cells"]
                for item in exact_transactions
            ),
            "maximum_projection_scratch_field_cells": max(
                item["maximum_projection_scratch_field_cells"]
                for item in exact_transactions
            ),
            "maximum_live_resident_plus_projection_scratch_field_cells": max(
                item["maximum_live_resident_plus_projection_scratch_field_cells"]
                for item in exact_transactions
            ),
            "maximum_commitment_serialized_record_bytes": max(
                item["maximum_commitment_serialized_record_bytes"]
                for item in exact_transactions
            ),
            "commitment_hashlib_internal_state_bytes_measured": False,
            "compiled_public_fourier_kernel_exact_field_cells": 289,
            "compiled_public_kernel_is_shared_across_transactions": True,
            "inverse_history_cells": 0,
            "accepted_path_assignment_or_truth_table_cells": 0,
            "accepted_path_public_coordinate_expansion_cells": 0,
            "retained_public_shell_count_integer_cells": 17,
            "transient_public_representative_coordinate_integer_cells_at_compile": 34,
            "public_topology_compile_coordinate_visits_per_algebra": 5202,
            "public_fourier_involution_check_4913_multiply_accumulates_per_algebra": True,
            "maximum_live_public_compile_exact_field_cells_excluding_expression_temporaries": 306,
            "exhaustive_invariance_verification_target_coordinate_visits_per_algebra": 289,
            "exhaustive_invariance_verification_source_coordinate_visits_per_algebra": 83521,
            "exhaustive_invariance_verification_total_coordinate_visits_per_algebra": 83810,
            "exhaustive_invariance_verification_live_kernel_plus_row_exact_field_cells_excluding_arithmetic_temporaries": 306,
            "verification_only_dense_coordinate_state_and_two_transform_buffers_exact_field_cells_excluding_arithmetic_temporaries": 867,
            "maximum_live_one_compiled_kernel_plus_dense_verification_buffers_exact_field_cells_per_algebra_excluding_arithmetic_temporaries": 1156,
            "maximum_live_all_three_compiled_kernels_plus_dense_verification_buffers_exact_field_cells_excluding_arithmetic_temporaries": 1734,
            "exact_payload_bits_measured_and_not_constant": True,
            "python_container_allocator_sympy_native_bigint_hashlib_and_whole_process_memory_excluded": True,
        },
        "matched_baseline": {
            "executed_matched_compact_classical_method": "IDENTICAL_17_COORDINATE_RADIAL_QUOTIENT_RECURRENCE",
            "resident_exact_field_cells": 17,
            "maximum_update_scratch_exact_field_cells": 20,
            "maximum_live_state_plus_scratch_exact_field_cells": 37,
            "maximum_projection_scratch_exact_field_cells": 6,
            "maximum_live_state_plus_projection_scratch_exact_field_cells": 23,
            "compiled_public_kernel_exact_field_cells": 289,
            "warm_arithmetic_work": "O(289*DEPTH)",
            "fully_fixed_transaction_scalar_cache_is_a_SEPARATE_NONUNIFORM_OPTION": True,
            "universal_strongest_classical_or_lower_bound_claimed": False,
            "dense_289_coordinate_separable_fourier_is_EXECUTED_CONTROL_NOT_MATCHED_COMPACT_BASELINE": True,
            "phase_carrier_or_snapshot_used": False,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
        },
        "restoration": {
            "class": "EXACT_ALGEBRAIC_RESTORATION",
            "actual_reverse_operation_order": "REVERSE_EACH_NORMALIZED_FOURIER_THEN_QUARTIC_PHASE",
            "same_backing": True,
            "fresh_restored_reuse_equal": True,
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "restoration_count_is_package_local_not_catvm_generation": True,
        },
        "claim_boundary": {
            "established": [
                "EXACT_17_CLASS_ANISOTROPIC_NORM_RELATION_QUOTIENT",
                "EXACT_NONCOMMUTING_QUARTIC_RADIAL_PHASE_AND_PHASE_FOURIER_CLOSURE",
                "EXACT_RESTORATION_AND_SAME_BACKING_UNRELATED_REUSE",
                "DENSE_289_COORDINATE_PARITY_AT_BOUNDED_CONTROL_SCOPE",
            ],
            "not_established": [
                "CATVM_MACHINE_ENFORCED_CUSTODY",
                "GENERAL_NONLINEAR_RELATION_QUOTIENTS",
                "ARBITRARY_COORDINATE_DIMENSION_OR_FIELD_SIZE",
                "DISTINCT_PHASE_RESOURCE",
                "COMPUTATIONAL_ADVANTAGE",
                "SMALL_WALL_CROSSING",
                "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
                "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
                "UNBOUNDED_CATALYTIC_COMPUTATION",
            ],
        },
        "next_obstruction": (
            "THE_NONLINEAR_ANISOTROPIC_NORM_QUOTIENT_CLOSES_EXACTLY_ON17_"
            "CELLS_BUT_COMPACT_CLASSICAL_SOFTWARE_EXECUTES_THE_IDENTICAL_"
            "RECURRENCE_THE_PUBLIC_FOURIER_KERNEL_RETAINS289_EXACT_CELLS_"
            "AND_EXACT_COEFFICIENT_PAYLOAD_GROWS_WITH_DEPTH"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
