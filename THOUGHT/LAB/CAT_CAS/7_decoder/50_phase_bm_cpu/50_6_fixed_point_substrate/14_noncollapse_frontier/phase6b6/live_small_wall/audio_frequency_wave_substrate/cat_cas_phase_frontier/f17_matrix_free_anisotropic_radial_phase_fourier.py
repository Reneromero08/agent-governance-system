#!/usr/bin/env python3
"""Exact matrix-free anisotropic radial phase/Fourier closure over F17.

This successor preserves the 17-cell nonlinear radial quotient established by
M139 while removing its retained 17-by-17 public Fourier kernel.  Its action
is factored through 16 streamed character contractions derived from the exact
entry formula

    delta[r=0] - (1/17) * sum_{t != 0} zeta17^(-q*t-r/(4*t)).

The identity follows from the anisotropic quadratic Gauss sum for
Q(x,y)=x^2-3y^2.  The accepted transaction never materializes the matrix or
the 289-coordinate state.  The identical matrix-free 17-coordinate classical
recurrence remains the matched baseline, so no computational advantage or
distinct phase resource is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import f17_anisotropic_quartic_radial_phase_quotient_closure as predecessor
import f17_coherent_veronese_phase_chart_closure as exact
import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_rank1_cubic_walsh_derivative_group_algebra_closure as rank1


P = 17
NONSQUARE = 3
QUOTIENT_CELLS = 17
REPRESENTED_COORDINATE_CELLS = 289
CHARACTER_TERMS_PER_ENTRY = 16
KERNEL_ENTRIES_PER_FOURIER = 289
CHARACTER_PRODUCTS_PER_FACTORED_FOURIER = 2 * P * (P - 1)
EXACT_DEPTHS = (1, 2, 4, 8, 16, 32, 64)
STRUCTURAL_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
FAMILIES = predecessor.FAMILIES
FINITE_FIELDS = predecessor.FINITE_FIELDS
SHELL_COUNTS = predecessor.SHELL_COUNTS
CLAIM = (
    "EXACT_MATRIX_FREE_ANISOTROPIC_F17_RADIAL_PHASE_FOURIER_GENERATED_"
    "FROM_16_TERM_KLOOSTERMAN_CHARACTER_SUM_REMOVES_THE_RETAINED_289_"
    "CELL_PUBLIC_KERNEL_WHILE_PRESERVING_17_CELL_NONCOMMUTING_QUARTIC_"
    "PHASE_CLOSURE_EXACT_RESTORATION_AND_REUSE_BUT_THE_IDENTICAL_"
    "MATRIX_FREE_CLASSICAL_RECURRENCE_AND_GROWING_EXACT_PAYLOAD_REMAIN"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def algebra_signature(alg: backend.Algebra) -> dict[str, Any]:
    return rank1.algebra_signature(alg)


def field_integer(alg: backend.Algebra, value: int) -> Any:
    return exact.field_integer(alg, value)


def norm(x: int, y: int) -> int:
    return (x * x - NONSQUARE * y * y) % P


def pairing(u: int, v: int, x: int, y: int) -> int:
    return (u * x - NONSQUARE * v * y) % P


def coordinates() -> Iterable[tuple[int, int]]:
    for x in range(P):
        for y in range(P):
            yield x, y


@dataclass(frozen=True)
class MatrixFreeGeometry:
    alg: backend.Algebra
    inverse17: Any
    retained_exact_field_cells: int = 1
    retained_public_integer_cells: int = 4

    @classmethod
    def compile(cls, alg: backend.Algebra) -> "MatrixFreeGeometry":
        squares = {value * value % P for value in range(P)}
        if NONSQUARE in squares:
            fail("declared anisotropic coefficient became a square")
        if SHELL_COUNTS != (1, *([18] * 16)):
            fail("declared anisotropic shell law changed")
        return cls(alg, alg.inverse(field_integer(alg, P)))

    def entry(
        self,
        target_shell: int,
        source_shell: int,
        observe: Callable[[tuple[Any, ...]], None] | None = None,
    ) -> Any:
        if not (0 <= target_shell < P and 0 <= source_shell < P):
            fail("matrix-free radial Fourier shell outside F17")
        total = self.alg.zero
        for parameter in range(1, P):
            inverse_four_parameter = pow((4 * parameter) % P, -1, P)
            exponent = (
                -source_shell * parameter
                - target_shell * inverse_four_parameter
            ) % P
            character = self.alg.power(exponent)
            updated = self.alg.add(total, character)
            if observe is not None:
                observe((total, character, updated))
            total = updated
        delta = field_integer(self.alg, P if target_shell == 0 else 0)
        numerator = self.alg.sub(delta, total)
        result = self.alg.mul(self.inverse17, numerator)
        if observe is not None:
            observe((total, delta, numerator, result))
        return result


Program = predecessor.Program
RadialGate = predecessor.RadialGate
compile_program = predecessor.compile_program
validate_program = predecessor.validate_program


def lease(program: Program, geometry: MatrixFreeGeometry) -> str:
    return digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(geometry.alg),
            "carrier": "MATRIX_FREE_ANISOTROPIC_F17_RADIAL_RELATION_QUOTIENT",
            "resident_cells": QUOTIENT_CELLS,
            "generator": "SIXTEEN_TERM_KLOOSTERMAN_CHARACTER_SUM",
        }
    )


@dataclass
class MatrixFreeCarrier:
    geometry: MatrixFreeGeometry
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
    generated_kernel_entries: int = 0
    generated_character_terms: int = 0

    @classmethod
    def create(cls, geometry: MatrixFreeGeometry) -> "MatrixFreeCarrier":
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
        payload = sum(self.alg.payload_bits(value) for value in self.cells)
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, payload
        )

    def observe_update_scratch(self, values: Iterable[Any]) -> None:
        materialized = tuple(values)
        payload = sum(self.alg.payload_bits(value) for value in materialized)
        self.maximum_update_scratch_field_cells = max(
            self.maximum_update_scratch_field_cells, len(materialized)
        )
        self.maximum_update_scratch_payload_bits = max(
            self.maximum_update_scratch_payload_bits, payload
        )
        resident_payload = sum(
            self.alg.payload_bits(value) for value in self.cells
        )
        self.maximum_live_resident_plus_update_scratch_payload_bits = max(
            self.maximum_live_resident_plus_update_scratch_payload_bits,
            resident_payload + payload,
        )

    def observe_projection_scratch(self, values: Iterable[Any]) -> None:
        materialized = tuple(values)
        payload = sum(self.alg.payload_bits(value) for value in materialized)
        self.maximum_projection_scratch_field_cells = max(
            self.maximum_projection_scratch_field_cells, len(materialized)
        )
        self.maximum_projection_scratch_payload_bits = max(
            self.maximum_projection_scratch_payload_bits, payload
        )
        resident_payload = sum(
            self.alg.payload_bits(value) for value in self.cells
        )
        self.maximum_live_resident_plus_projection_scratch_payload_bits = max(
            self.maximum_live_resident_plus_projection_scratch_payload_bits,
            resident_payload + payload,
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
            state["package_local_restoration_count"] = (
                self.package_local_restoration_count
            )
        return digest_json(state)


def require_owned(
    carrier: MatrixFreeCarrier, program: Program, stage: str
) -> None:
    if not isinstance(carrier, MatrixFreeCarrier):
        fail("null or wrong matrix-free anisotropic radial carrier")
    if (
        carrier.stage != stage
        or carrier.active_program != program.fingerprint()
        or carrier.active_lease != lease(program, carrier.geometry)
    ):
        fail("matrix-free anisotropic carrier owner or stage changed")


def apply_phase(
    carrier: MatrixFreeCarrier, gate: RadialGate, *, inverse: bool = False
) -> None:
    sign = -1 if inverse else 1
    scratch = [
        carrier.alg.mul(
            value, carrier.alg.power(sign * gate.exponent(shell))
        )
        for shell, value in enumerate(carrier.cells)
    ]
    carrier.observe_update_scratch(scratch)
    carrier.cells[:] = scratch
    carrier.observe_resident()


def apply_fourier(carrier: MatrixFreeCarrier) -> None:
    alg = carrier.alg
    spectrum = [alg.zero for _ in range(P - 1)]
    for offset, parameter in enumerate(range(1, P)):
        accumulator = alg.zero
        for source, value in enumerate(carrier.cells):
            character = alg.power(-source * parameter)
            product = alg.mul(character, value)
            updated = alg.add(accumulator, product)
            carrier.observe_update_scratch(
                (*spectrum, accumulator, character, product, updated)
            )
            accumulator = updated
        spectrum[offset] = accumulator

    state_sum = alg.zero
    for value in carrier.cells:
        updated = alg.add(state_sum, value)
        carrier.observe_update_scratch((state_sum, value, updated))
        state_sum = updated

    output = [alg.zero for _ in range(P)]
    for target in range(P):
        accumulator = alg.zero
        for offset, parameter in enumerate(range(1, P)):
            inverse_four_parameter = pow((4 * parameter) % P, -1, P)
            character = alg.power(-target * inverse_four_parameter)
            product = alg.mul(character, spectrum[offset])
            updated = alg.add(accumulator, product)
            carrier.observe_update_scratch(
                (
                    *spectrum,
                    *output,
                    state_sum,
                    accumulator,
                    character,
                    product,
                    updated,
                )
            )
            accumulator = updated
        scaled = alg.mul(carrier.geometry.inverse17, accumulator)
        output[target] = alg.sub(state_sum if target == 0 else alg.zero, scaled)
        carrier.observe_update_scratch(
            (
                *spectrum,
                *output,
                state_sum,
                accumulator,
                scaled,
                output[target],
            )
        )
    carrier.generated_character_terms += CHARACTER_PRODUCTS_PER_FACTORED_FOURIER
    carrier.observe_update_scratch(output)
    carrier.cells[:] = output
    carrier.observe_resident()


def begin_forward(carrier: MatrixFreeCarrier, program: Program) -> None:
    validate_program(program)
    if not isinstance(carrier, MatrixFreeCarrier) or not carrier.exact_zero():
        fail("matrix-free anisotropic carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.active_lease = lease(program, carrier.geometry)
    carrier.stage = "FORWARD"
    carrier.cells[:] = [carrier.alg.one for _ in range(P)]
    carrier.observe_resident()


def forward(carrier: MatrixFreeCarrier, program: Program) -> None:
    require_owned(carrier, program, "FORWARD")
    for index, gate in enumerate(program.gates):
        apply_phase(carrier, gate)
        apply_fourier(carrier)
        carrier.forward_index = index + 1
    carrier.stage = "FINAL_RADIAL_RELATION_RESIDENT"


def state_commitment(carrier: MatrixFreeCarrier) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    total_bytes = 0
    for value in carrier.cells:
        record = json.dumps(
            carrier.alg.serialize(value), separators=(",", ":")
        ).encode()
        total_bytes += len(record)
        carrier.maximum_commitment_serialized_record_bytes = max(
            carrier.maximum_commitment_serialized_record_bytes, len(record)
        )
        hasher.update(len(record).to_bytes(8, "big"))
        hasher.update(record)
    return (
        hasher.hexdigest(),
        total_bytes,
        carrier.maximum_commitment_serialized_record_bytes,
    )


def project(carrier: MatrixFreeCarrier, program: Program) -> Any:
    require_owned(carrier, program, "FINAL_RADIAL_RELATION_RESIDENT")
    if carrier.forward_index != program.depth or carrier.projection_calls:
        fail("matrix-free anisotropic projection order changed")
    total = carrier.alg.zero
    for shell, value in enumerate(carrier.cells):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        phase = carrier.alg.power(exponent)
        weight = field_integer(carrier.alg, SHELL_COUNTS[shell])
        phased = carrier.alg.mul(phase, value)
        weighted = carrier.alg.mul(weight, phased)
        updated = carrier.alg.add(total, weighted)
        carrier.observe_projection_scratch(
            (total, phase, weight, phased, weighted, updated)
        )
        total = updated
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED"
    return total


def inverse(carrier: MatrixFreeCarrier, program: Program) -> None:
    require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("matrix-free anisotropic inverse requires final projection")
    carrier.stage = "INVERSE"
    for index in range(program.depth - 1, -1, -1):
        apply_fourier(carrier)
        apply_phase(carrier, program.gates[index], inverse=True)
        carrier.inverse_index += 1
    if carrier.cells != [carrier.alg.one for _ in range(P)]:
        fail("matrix-free anisotropic actual inverse did not restore seed")
    carrier.cells[:] = [
        carrier.alg.sub(value, carrier.alg.one) for value in carrier.cells
    ]
    carrier.active_program = None
    carrier.active_lease = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not carrier.exact_zero():
        fail("matrix-free anisotropic carrier did not restore exact zero")


def stats_snapshot(alg: backend.Algebra) -> dict[str, int]:
    return {name: int(value) for name, value in vars(alg.stats).items()}


def stats_delta(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, int]:
    return {key: after[key] - before.get(key, 0) for key in after}


def execute_transaction(
    carrier: MatrixFreeCarrier, program: Program
) -> dict[str, Any]:
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_update_scratch_field_cells = 0
    carrier.maximum_update_scratch_payload_bits = 0
    carrier.maximum_live_resident_plus_update_scratch_payload_bits = 0
    carrier.maximum_projection_scratch_field_cells = 0
    carrier.maximum_projection_scratch_payload_bits = 0
    carrier.maximum_live_resident_plus_projection_scratch_payload_bits = 0
    carrier.maximum_commitment_serialized_record_bytes = 0
    carrier.generated_kernel_entries = 0
    carrier.generated_character_terms = 0
    initial = carrier.digest(include_package_local_count=False)
    backing = carrier.backing_identity()
    restoration_count = carrier.package_local_restoration_count
    before_stats = stats_snapshot(carrier.alg)
    descriptor_bytes = len(
        json.dumps(
            program.descriptor(), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    begin_forward(carrier, program)
    forward(carrier, program)
    commitment, commitment_input_bytes, commitment_record_bytes = (
        state_commitment(carrier)
    )
    boundary = project(carrier, program)
    inverse(carrier, program)
    expected_fourier_calls = 2 * program.depth
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
        "resident_relation_quotient_exact_field_cells": QUOTIENT_CELLS,
        "represented_coordinate_cells": REPRESENTED_COORDINATE_CELLS,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_update_scratch_field_cells": (
            carrier.maximum_update_scratch_field_cells
        ),
        "maximum_update_scratch_payload_bits": (
            carrier.maximum_update_scratch_payload_bits
        ),
        "maximum_live_resident_plus_update_scratch_field_cells": (
            QUOTIENT_CELLS + carrier.maximum_update_scratch_field_cells
        ),
        "maximum_live_resident_plus_update_scratch_payload_bits": (
            carrier.maximum_live_resident_plus_update_scratch_payload_bits
        ),
        "retained_public_fourier_kernel_field_cells": 0,
        "retained_public_generator_exact_field_cells": (
            carrier.geometry.retained_exact_field_cells
        ),
        "retained_public_generator_payload_bits": carrier.alg.payload_bits(
            carrier.geometry.inverse17
        ),
        "retained_public_generator_integer_cells": (
            carrier.geometry.retained_public_integer_cells
        ),
        "generated_kernel_entries": 0,
        "generated_character_terms": carrier.generated_character_terms,
        "expected_forward_plus_inverse_fourier_calls": expected_fourier_calls,
        "generated_entries_per_fourier": 0,
        "generated_character_products_per_fourier": (
            CHARACTER_PRODUCTS_PER_FACTORED_FOURIER
        ),
        "maximum_projection_scratch_field_cells": (
            carrier.maximum_projection_scratch_field_cells
        ),
        "maximum_projection_scratch_payload_bits": (
            carrier.maximum_projection_scratch_payload_bits
        ),
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
        "accepted_path_dense_kernel_cells": 0,
        "public_generator_not_answer_data": True,
        "intermediate_relation_quotient_exposed": False,
        "final_projection_calls": 1,
        "response_released_after_restoration": True,
        "same_backing": carrier.backing_identity() == backing,
        "restored_exact_zero": carrier.exact_zero(),
        "initial_restored_digest_equal": (
            carrier.digest(include_package_local_count=False) == initial
        ),
        "package_local_restoration_count_before": restoration_count,
        "package_local_restoration_count_after": (
            carrier.package_local_restoration_count
        ),
        "snapshot_reload_used": False,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "generator_projection_and_commitment_buffers_restoration_class": (
            "NO_RESTORATION_CLAIM"
        ),
        "native_operation_counts": stats_delta(
            before_stats, stats_snapshot(carrier.alg)
        ),
    }


def classical_matrix_free_boundary(
    program: Program, geometry: MatrixFreeGeometry
) -> tuple[Any, dict[str, Any], list[Any]]:
    alg = geometry.alg
    state = [alg.one for _ in range(P)]
    maximum_state_payload = sum(alg.payload_bits(value) for value in state)
    maximum_live_update_payload = maximum_state_payload
    maximum_projection_payload = 0
    maximum_live_projection_payload = 0
    generated_character_products = 0
    for gate in program.gates:
        state = [
            alg.mul(value, alg.power(gate.exponent(shell)))
            for shell, value in enumerate(state)
        ]
        maximum_state_payload = max(
            maximum_state_payload,
            sum(alg.payload_bits(value) for value in state),
        )
        spectrum = [alg.zero for _ in range(P - 1)]
        for offset, parameter in enumerate(range(1, P)):
            accumulator = alg.zero
            for source, value in enumerate(state):
                character = alg.power(-source * parameter)
                product = alg.mul(character, value)
                updated = alg.add(accumulator, product)
                scratch = (*spectrum, accumulator, character, product, updated)
                maximum_live_update_payload = max(
                    maximum_live_update_payload,
                    sum(alg.payload_bits(value) for value in state)
                    + sum(alg.payload_bits(value) for value in scratch),
                )
                accumulator = updated
            spectrum[offset] = accumulator

        state_sum = alg.zero
        for value in state:
            state_sum = alg.add(state_sum, value)

        output = [alg.zero for _ in range(P)]
        for target in range(P):
            accumulator = alg.zero
            for offset, parameter in enumerate(range(1, P)):
                inverse_four_parameter = pow((4 * parameter) % P, -1, P)
                character = alg.power(-target * inverse_four_parameter)
                product = alg.mul(character, spectrum[offset])
                updated = alg.add(accumulator, product)
                scratch = (
                    *spectrum,
                    *output,
                    state_sum,
                    accumulator,
                    character,
                    product,
                    updated,
                )
                maximum_live_update_payload = max(
                    maximum_live_update_payload,
                    sum(alg.payload_bits(value) for value in state)
                    + sum(alg.payload_bits(value) for value in scratch),
                )
                accumulator = updated
            scaled = alg.mul(geometry.inverse17, accumulator)
            output[target] = alg.sub(
                state_sum if target == 0 else alg.zero, scaled
            )
        generated_character_products += CHARACTER_PRODUCTS_PER_FACTORED_FOURIER
        state = output
        maximum_state_payload = max(
            maximum_state_payload,
            sum(alg.payload_bits(value) for value in state),
        )
    total = alg.zero
    for shell, value in enumerate(state):
        phase = alg.power(
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        )
        weight = field_integer(alg, SHELL_COUNTS[shell])
        phased = alg.mul(phase, value)
        weighted = alg.mul(weight, phased)
        updated = alg.add(total, weighted)
        scratch = (total, phase, weight, phased, weighted, updated)
        scratch_payload = sum(alg.payload_bits(item) for item in scratch)
        maximum_projection_payload = max(
            maximum_projection_payload, scratch_payload
        )
        maximum_live_projection_payload = max(
            maximum_live_projection_payload,
            sum(alg.payload_bits(item) for item in state) + scratch_payload,
        )
        total = updated
    return total, {
        "resident_exact_field_cells": 17,
        "maximum_update_scratch_exact_field_cells": 38,
        "maximum_live_state_plus_scratch_exact_field_cells": 55,
        "maximum_projection_scratch_exact_field_cells": 6,
        "maximum_live_state_plus_projection_scratch_exact_field_cells": 23,
        "retained_public_kernel_exact_field_cells": 0,
        "retained_public_generator_exact_field_cells": 1,
        "retained_public_generator_payload_bits": alg.payload_bits(
            geometry.inverse17
        ),
        "generated_kernel_entries_forward_only": 0,
        "generated_character_products_forward_only": generated_character_products,
        "maximum_resident_payload_bits": maximum_state_payload,
        "maximum_live_state_plus_scratch_payload_bits": maximum_live_update_payload,
        "maximum_projection_scratch_payload_bits": maximum_projection_payload,
        "maximum_live_state_plus_projection_scratch_payload_bits": (
            maximum_live_projection_payload
        ),
        "warm_arithmetic_work": "O(544*DEPTH)",
        "universal_optimality_or_lower_bound_claimed": False,
    }, state


def generator_parity_control(
    geometry: MatrixFreeGeometry,
) -> dict[str, int | bool]:
    compiled = predecessor.CompiledGeometry.compile(geometry.alg)
    checked = 0
    for target in range(P):
        for source in range(P):
            checked += 1
            if geometry.entry(target, source) != compiled.normalized_fourier[target][source]:
                fail("matrix-free character-sum generator disagrees with coordinate kernel")
    return {
        "all_289_entries_equal_coordinate_compiler": True,
        "entry_checks": checked,
        "verification_only_coordinate_kernel_field_cells": 289,
    }


def matrix_free_involution_control(
    geometry: MatrixFreeGeometry,
) -> dict[str, int | bool]:
    checks = 0
    generated_entries = 0
    for first in range(P):
        for second in range(P):
            total = geometry.alg.zero
            for middle in range(P):
                left = geometry.entry(first, middle)
                right = geometry.entry(middle, second)
                generated_entries += 2
                total = geometry.alg.add(total, geometry.alg.mul(left, right))
            checks += 1
            expected = geometry.alg.one if first == second else geometry.alg.zero
            if total != expected:
                fail("matrix-free radial Fourier is not an exact involution")
    return {
        "matrix_free_fourier_squares_to_identity": True,
        "matrix_product_entry_checks": checks,
        "generated_entries": generated_entries,
        "generated_character_terms": generated_entries * CHARACTER_TERMS_PER_ENTRY,
        "retained_kernel_field_cells": 0,
    }


def controls(geometry: MatrixFreeGeometry) -> dict[str, bool]:
    alg = geometry.alg
    program = compile_program(4, "PRIMARY")
    reference_carrier = MatrixFreeCarrier.create(geometry)
    reference = execute_transaction(reference_carrier, program)

    missing = MatrixFreeCarrier.create(geometry)
    begin_forward(missing, program)
    forward(missing, program)

    reordered = MatrixFreeCarrier.create(geometry)
    begin_forward(reordered, program)
    forward(reordered, program)
    project(reordered, program)
    reordered.stage = "INVERSE"
    for gate in reversed(program.gates):
        apply_phase(reordered, gate, inverse=True)
        apply_fourier(reordered)

    wrong = MatrixFreeCarrier.create(geometry)
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

    premature_projection_rejected = False
    try:
        project(MatrixFreeCarrier.create(geometry), program)
    except RuntimeError:
        premature_projection_rejected = True

    null_carrier_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    return {
        "reference_restores_exactly": reference["restored_exact_zero"],
        "missing_inverse_leaves_actual_resident_state": not missing.exact_zero(),
        "reordered_actual_inverse_fails_to_restore": (
            reordered.cells != [alg.one for _ in range(P)]
        ),
        "wrong_phase_inverse_fails_to_restore": (
            wrong.cells != [alg.one for _ in range(P)]
        ),
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "geometry_has_no_kernel_attribute": not hasattr(geometry, "normalized_fourier"),
        "accepted_carrier_has_no_kernel_cache": not any(
            hasattr(reference_carrier, name)
            for name in ("kernel", "normalized_fourier", "kernel_cache")
        ),
        "snapshot_command_available": False,
        "intermediate_projection_available": False,
        "coordinate_expansion_command_available": False,
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
    exact_geometry = MatrixFreeGeometry.compile(backend.Algebra("Q_ZETA17"))
    exact_transactions = [
        execute_transaction(
            MatrixFreeCarrier.create(exact_geometry),
            compile_program(depth, "PRIMARY"),
        )
        for depth in EXACT_DEPTHS
    ]

    structural_transactions = []
    geometries: dict[str, MatrixFreeGeometry] = {}
    for modulus, root in FINITE_FIELDS:
        geometry = MatrixFreeGeometry.compile(
            backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        )
        geometries[f"F{modulus}"] = geometry
        for family in FAMILIES:
            for depth in STRUCTURAL_DEPTHS:
                structural_transactions.append(
                    execute_transaction(
                        MatrixFreeCarrier.create(geometry),
                        compile_program(depth, family),
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
        boundary, resources, _ = classical_matrix_free_boundary(program, geometry)
        baselines.append(
            {
                "depth": program.depth,
                "family": program.family,
                "program_fingerprint": program.fingerprint(),
                "boundary_equal": (
                    geometry.alg.serialize(boundary) == item["final_boundary"]
                ),
                **resources,
            }
        )

    parity_controls = [
        {
            "algebra_kind": geometry.alg.kind,
            **generator_parity_control(geometry),
            **matrix_free_involution_control(geometry),
        }
        for geometry in (exact_geometry, *geometries.values())
    ]

    reuse_carrier = MatrixFreeCarrier.create(exact_geometry)
    original_backing = reuse_carrier.backing_identity()
    first = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    restored = execute_transaction(reuse_carrier, compile_program(16, "REUSE"))
    fresh = execute_transaction(
        MatrixFreeCarrier.create(exact_geometry), compile_program(16, "REUSE")
    )

    control_results = controls(geometries["F137"])
    false_controls = {
        "snapshot_command_available",
        "intermediate_projection_available",
        "coordinate_expansion_command_available",
    }
    if any(control_results[key] for key in false_controls) or not all(
        value
        for key, value in control_results.items()
        if key not in false_controls
    ):
        fail("matrix-free anisotropic control failed")
    transactions = [*exact_transactions, *structural_transactions]
    if not all(item["restored_exact_zero"] and item["same_backing"] for item in transactions):
        fail("matrix-free anisotropic transaction restoration failed")
    if not all(item["boundary_equal"] for item in baselines):
        fail("matched matrix-free classical recurrence diverged")

    return {
        "schema": "CAT_CAS_F17_MATRIX_FREE_ANISOTROPIC_RADIAL_PHASE_FOURIER_V1",
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
            "matrix_free_generator": (
                "DELTA_R0_MINUS_INV17_SUM_T_NONZERO_ZETA_TO_MINUS_QT_MINUS_R_OVER4T"
            ),
            "character_terms_per_kernel_entry": CHARACTER_TERMS_PER_ENTRY,
            "operations": [
                "QUARTIC_RADIAL_PHASE",
                "MATRIX_FREE_NORMALIZED_ANISOTROPIC_PHASE_FOURIER",
            ],
        },
        "exact_transactions": exact_transactions,
        "structural_transactions": structural_transactions,
        "matched_classical_baselines": baselines,
        "matched_classical_frontier": [
            {
                "method": "IDENTICAL_MATRIX_FREE_17_COORDINATE_RADIAL_QUOTIENT_RECURRENCE",
                "resident_exact_field_cells": 17,
                "retained_public_kernel_exact_field_cells": 0,
                "retained_public_generator_exact_field_cells": 1,
                "maximum_update_scratch_exact_field_cells": 38,
                "maximum_live_state_plus_scratch_exact_field_cells": 55,
                "warm_arithmetic_work": "O(544*DEPTH)",
            },
            {
                "method": "M139_RETAINED_KERNEL_17_COORDINATE_RADIAL_QUOTIENT_RECURRENCE",
                "resident_exact_field_cells": 17,
                "retained_public_kernel_exact_field_cells": 289,
                "maximum_update_scratch_exact_field_cells": 20,
                "maximum_live_state_plus_scratch_exact_field_cells": 37,
                "warm_arithmetic_work": "O(289*DEPTH)",
            },
            {
                "method": "FULLY_FIXED_TRANSACTION_SCALAR_CACHE",
                "classification": "SEPARATE_NONUNIFORM_OPTION",
            },
        ],
        "verification_only_generator_controls": parity_controls,
        "reuse": {
            "first_depth_family": [first["depth"], first["family"]],
            "unrelated_reuse_depth_family": [
                restored["depth"], restored["family"]
            ],
            "same_original_backing": (
                reuse_carrier.backing_identity() == original_backing
            ),
            "fresh_restored_boundary_equal": (
                restored["final_boundary"] == fresh["final_boundary"]
            ),
            "fresh_restored_resource_signature_equal": (
                resource_signature(restored) == resource_signature(fresh)
            ),
            "package_local_restoration_count": (
                reuse_carrier.package_local_restoration_count
            ),
            "restored_exact_zero": reuse_carrier.exact_zero(),
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
        },
        "controls": control_results,
        "resource_law": {
            "represented_coordinate_field_cells": 289,
            "resident_relation_quotient_exact_field_cells": 17,
            "resident_cells_independent_of_program_depth": True,
            "retained_public_fourier_kernel_exact_field_cells": 0,
            "retained_public_generator_exact_field_cells": 1,
            "retained_public_generator_integer_cells": 4,
            "generated_kernel_entries_per_fourier": 0,
            "verification_only_character_terms_per_kernel_entry": (
                CHARACTER_TERMS_PER_ENTRY
            ),
            "generated_character_products_per_factored_fourier": (
                CHARACTER_PRODUCTS_PER_FACTORED_FOURIER
            ),
            "maximum_update_scratch_field_cells": max(
                item["maximum_update_scratch_field_cells"]
                for item in exact_transactions
            ),
            "maximum_live_resident_plus_update_scratch_field_cells": max(
                item["maximum_live_resident_plus_update_scratch_field_cells"]
                for item in exact_transactions
            ),
            "maximum_live_resident_plus_update_plus_generator_exact_field_cells": (
                max(
                    item["maximum_live_resident_plus_update_scratch_field_cells"]
                    for item in exact_transactions
                )
                + 1
            ),
            "maximum_projection_scratch_field_cells": max(
                item["maximum_projection_scratch_field_cells"]
                for item in exact_transactions
            ),
            "maximum_live_resident_plus_projection_scratch_field_cells": max(
                item["maximum_live_resident_plus_projection_scratch_field_cells"]
                for item in exact_transactions
            ),
            "inverse_history_cells": 0,
            "accepted_path_assignment_or_truth_table_cells": 0,
            "accepted_path_public_coordinate_expansion_cells": 0,
            "accepted_path_dense_kernel_cells": 0,
            "verification_only_coordinate_kernel_cells_per_algebra": 289,
            "verification_only_matrix_free_involution_generated_entries_per_algebra": 9826,
            "exact_payload_bits_measured_and_not_constant": True,
            "python_container_allocator_sympy_native_bigint_hashlib_and_whole_process_memory_excluded": True,
        },
        "matched_baseline": {
            "executed_matched_compact_classical_method": (
                "IDENTICAL_MATRIX_FREE_17_COORDINATE_RADIAL_QUOTIENT_RECURRENCE"
            ),
            "resident_exact_field_cells": 17,
            "maximum_update_scratch_exact_field_cells": 38,
            "maximum_live_state_plus_scratch_exact_field_cells": 55,
            "maximum_live_state_plus_scratch_plus_generator_exact_field_cells": 56,
            "maximum_projection_scratch_exact_field_cells": 6,
            "maximum_live_state_plus_projection_scratch_exact_field_cells": 23,
            "retained_public_kernel_exact_field_cells": 0,
            "retained_public_generator_exact_field_cells": 1,
            "warm_arithmetic_work": "O(544*DEPTH)",
            "fully_fixed_transaction_scalar_cache_is_a_SEPARATE_NONUNIFORM_OPTION": True,
            "universal_strongest_classical_or_lower_bound_claimed": False,
            "coordinate_kernel_is_VERIFICATION_ONLY_NOT_MATCHED_BASELINE": True,
            "phase_carrier_or_snapshot_used": False,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
        },
        "restoration": {
            "class": "EXACT_ALGEBRAIC_RESTORATION",
            "actual_reverse_operation_order": (
                "REVERSE_EACH_MATRIX_FREE_FOURIER_THEN_QUARTIC_PHASE"
            ),
            "same_backing": True,
            "fresh_restored_reuse_equal": True,
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "restoration_count_is_package_local_not_catvm_generation": True,
        },
        "claim_boundary": {
            "established": [
                "EXACT_MATRIX_FREE_17_CLASS_ANISOTROPIC_NORM_RELATION_QUOTIENT",
                "RETAINED_289_CELL_PUBLIC_FOURIER_KERNEL_REMOVED",
                "EXACT_16_TERM_KLOOSTERMAN_TYPE_CHARACTER_SUM_ENTRY_GENERATOR",
                "EXACT_NONCOMMUTING_QUARTIC_RADIAL_PHASE_AND_PHASE_FOURIER_CLOSURE",
                "EXACT_RESTORATION_AND_SAME_BACKING_UNRELATED_REUSE",
            ],
            "not_established": [
                "CATVM_MACHINE_ENFORCED_CUSTODY",
                "GENERAL_NONLINEAR_RELATION_QUOTIENTS",
                "ARBITRARY_COORDINATE_DIMENSION_OR_FIELD_SIZE",
                "ASYMPTOTIC_WORK_REDUCTION",
                "DISTINCT_PHASE_RESOURCE",
                "COMPUTATIONAL_ADVANTAGE",
                "SMALL_WALL_CROSSING",
                "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
                "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
                "UNBOUNDED_CATALYTIC_COMPUTATION",
            ],
        },
        "next_obstruction": (
            "THE_289_CELL_RETAINED_KERNEL_IS_REMOVED_BUT_THE_MATRIX_FREE_"
            "FACTORIZATION_USES544_CHARACTER_PRODUCTS_PER_FOURIER_AND38_"
            "TEMPORARY_EXACT_FIELDS_THE_IDENTICAL_CLASSICAL_RECURRENCE_"
            "REMAINS_AND_EXACT_PAYLOAD_STILL_GROWS_WITH_DEPTH"
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
