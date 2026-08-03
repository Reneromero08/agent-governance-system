#!/usr/bin/env python3
"""Exact program-restricted coherent chart for the M128 symmetric phase module.

The M128 selectable character/shear algebra is irreducible on the full H(k)
occupation module.  That does not exclude nonlinear invariant charts.  This
successor restricts the public program family to symmetric powers of one
single-particle vector.  The nonlinear Veronese map

    v -> (sum_j v_j x_j)^k

is closed by every declared power-sum phase character and lifted one-particle
shear.  Production therefore retains 17 exact phase coordinates instead of
H(k)=binomial(k+16,16) occupation coordinates.  It projects one declared
occupation coefficient, reverses the actual operations, restores the same
backing exactly, and reuses it for an unrelated program.

This is a bounded direct-process, program-restricted integrable closure.  It
does not include the M127 grid-orbit shear, sums of coherent components, a
general nonlinear quotient, CATVM custody, or a distinct phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_exchange_symmetric_phase_module_irreducibility as m128


PRIME = 17
MODE_COUNT = 17
DECLARED_K = (4, 8, 16, 32, 64, 128)
EXACT_K = (4, 8, 16, 32)
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE")
FINAL_BOUNDARY = "K_MINUS_1_PARTICLES_IN_MODE0_AND_ONE_PARTICLE_IN_MODE1"


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def algebra_signature(alg: backend.Algebra) -> str:
    return digest_json(
        {
            "kind": alg.kind,
            "modulus": alg.modulus,
            "root": alg.serialize(alg.root),
        }
    )


def stream_vector_commitment(
    values: Iterable[Any], alg: backend.Algebra
) -> tuple[str, int]:
    hasher = hashlib.sha256()
    maximum_record_json_bytes = 0
    for value in values:
        record = json.dumps(
            alg.serialize(value), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        maximum_record_json_bytes = max(maximum_record_json_bytes, len(record))
        hasher.update(len(record).to_bytes(8, "big"))
        hasher.update(record)
    return hasher.hexdigest(), maximum_record_json_bytes


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def field_integer(alg: backend.Algebra, value: int) -> Any:
    if value < 0:
        return negative(alg, field_integer(alg, -value))
    result = alg.zero
    addend = alg.one
    remaining = value
    while remaining:
        if remaining & 1:
            result = alg.add(result, addend)
        remaining >>= 1
        if remaining:
            addend = alg.add(addend, addend)
    return result


def scalar_power(alg: backend.Algebra, value: Any, exponent: int) -> Any:
    if exponent < 0:
        fail("negative scalar exponent is outside the coherent chart")
    result = alg.one
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = alg.mul(result, factor)
        remaining >>= 1
        if remaining:
            factor = alg.mul(factor, factor)
    return result


def stats_snapshot(alg: backend.Algebra) -> dict[str, int]:
    return {name: int(value) for name, value in vars(alg.stats).items()}


def stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: after[name] - before.get(name, 0) for name in after}


@dataclass(frozen=True)
class Primitive:
    kind: str
    first: int
    second: int
    coefficient_exponent: int

    def as_json(self) -> dict[str, int | str]:
        return {
            "kind": self.kind,
            "first": self.first,
            "second": self.second,
            "coefficient_exponent": self.coefficient_exponent,
        }


@dataclass(frozen=True)
class CoherentProgram:
    k: int
    family: str
    chart_rank: int
    primitives: tuple[Primitive, ...]
    final_boundary: str = FINAL_BOUNDARY

    def fingerprint(self) -> str:
        return digest_json(public_program_descriptor(self))


def compile_program(k: int, family: str) -> CoherentProgram:
    if k not in DECLARED_K:
        fail("coherent chart degree is outside the declared growing family")
    if family not in FAMILIES:
        fail("unknown coherent chart program family")
    variant = 0 if family == "PRIMARY" else 1
    first_shears = tuple(
        Primitive("SHEAR", mode + 1, mode, 1 + ((5 * mode + 3 + variant) % 16))
        for mode in range(MODE_COUNT - 1)
    )
    first_characters = tuple(
        Primitive("CHARACTER", degree, 0, 1 + ((3 * degree + 5 * variant) % 16))
        for degree in range(1, 5)
    )
    second_shears = tuple(
        Primitive("SHEAR", mode, mode + 1, 1 + ((7 * mode + 4 + 2 * variant) % 16))
        for mode in range(MODE_COUNT - 2, -1, -1)
    )
    second_characters = tuple(
        Primitive("CHARACTER", degree, 0, 1 + ((7 * degree + 2 + 3 * variant) % 16))
        for degree in range(4, 0, -1)
    )
    program = CoherentProgram(
        k=k,
        family=family,
        chart_rank=1,
        primitives=first_shears + first_characters + second_shears + second_characters,
    )
    validate_program(program)
    return program


def validate_primitive(primitive: Primitive) -> None:
    if not 1 <= primitive.coefficient_exponent < PRIME:
        fail("primitive coefficient is outside the nonzero F17 phase family")
    if primitive.kind == "CHARACTER":
        if not 1 <= primitive.first <= 4 or primitive.second != 0:
            fail("character degree is outside the fixed p1-through-p4 family")
        return
    if primitive.kind == "SHEAR":
        if not (
            0 <= primitive.first < MODE_COUNT
            and 0 <= primitive.second < MODE_COUNT
            and abs(primitive.first - primitive.second) == 1
        ):
            fail("shear is not an adjacent-mode one-particle primitive")
        return
    fail("non-integrable primitive is outside the coherent chart")


def validate_program(program: CoherentProgram) -> None:
    if program.k not in DECLARED_K or program.family not in FAMILIES:
        fail("coherent chart public program identity changed")
    if program.chart_rank != 1:
        fail("only the rank-one coherent Veronese chart is accepted")
    if program.final_boundary != FINAL_BOUNDARY:
        fail("coherent chart boundary type changed")
    if len(program.primitives) != 40:
        fail("coherent chart primitive schedule changed")
    for primitive in program.primitives:
        validate_primitive(primitive)
    characters = [item.first for item in program.primitives if item.kind == "CHARACTER"]
    if characters != [1, 2, 3, 4, 4, 3, 2, 1]:
        fail("coherent chart character schedule changed")
    orientations = {
        (item.first, item.second)
        for item in program.primitives
        if item.kind == "SHEAR"
    }
    if any(
        (mode, mode + 1) not in orientations
        or (mode + 1, mode) not in orientations
        for mode in range(MODE_COUNT - 1)
    ):
        fail("coherent chart shear schedule is not bidirectional")


def public_program_descriptor(program: CoherentProgram) -> dict[str, Any]:
    return {
        "k": program.k,
        "family": program.family,
        "chart": "RANK1_EXCHANGE_SYMMETRIC_COHERENT_VERONESE",
        "chart_rank": program.chart_rank,
        "mode_count": MODE_COUNT,
        "seed": "ONE_PARTICLE_MODE0_RAISED_TO_SYMMETRIC_POWER_K",
        "primitives": [item.as_json() for item in program.primitives],
        "excluded_primitive": "M127_GRID_ORBIT_SHEAR",
        "final_boundary": program.final_boundary,
    }


def lease(program: CoherentProgram, alg: backend.Algebra) -> str:
    return digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "carrier": "17_COORDINATE_RANK1_COHERENT_VERONESE",
        }
    )


@dataclass
class CoherentCarrier:
    alg: backend.Algebra
    cells: list[Any]
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    active_k: int | None = None
    projection_calls: int = 0
    current_resident_payload_bits: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_resident_field_value_payload_bits: int = 0
    maximum_resident_numerator_signed_bits: int = 0
    maximum_resident_denominator_bits: int = 0
    maximum_observed_transient_field_value_payload_bits: int = 0
    maximum_observed_transient_numerator_signed_bits: int = 0
    maximum_observed_transient_denominator_bits: int = 0
    maximum_commitment_record_json_bytes: int = 0
    character_cell_multiplications: int = 0
    shear_cell_updates: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra) -> "CoherentCarrier":
        carrier = cls(alg=alg, cells=[alg.zero for _ in range(MODE_COUNT)])
        carrier.observe_resident()
        return carrier

    def backing_identity(self) -> tuple[int, int]:
        return id(self), id(self.cells)

    def exact_zero(self) -> bool:
        return (
            all(value == self.alg.zero for value in self.cells)
            and self.lease is None
            and self.stage == "RESTORED"
            and self.active_k is None
        )

    def write_cell(self, index: int, value: Any) -> None:
        previous = self.alg.payload_bits(self.cells[index])
        current = self.alg.payload_bits(value)
        self.cells[index] = value
        self.current_resident_payload_bits += current - previous
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, self.current_resident_payload_bits
        )
        self.observe_value(value, resident=True)

    def observe_value(self, value: Any, *, resident: bool = False) -> None:
        payload = self.alg.payload_bits(value)
        numerator, denominator = self.alg.coefficient_height(value)
        if resident:
            self.maximum_resident_field_value_payload_bits = max(
                self.maximum_resident_field_value_payload_bits, payload
            )
            self.maximum_resident_numerator_signed_bits = max(
                self.maximum_resident_numerator_signed_bits, numerator
            )
            self.maximum_resident_denominator_bits = max(
                self.maximum_resident_denominator_bits, denominator
            )
        else:
            self.maximum_observed_transient_field_value_payload_bits = max(
                self.maximum_observed_transient_field_value_payload_bits, payload
            )
            self.maximum_observed_transient_numerator_signed_bits = max(
                self.maximum_observed_transient_numerator_signed_bits, numerator
            )
            self.maximum_observed_transient_denominator_bits = max(
                self.maximum_observed_transient_denominator_bits, denominator
            )

    def observe_resident(self) -> None:
        self.current_resident_payload_bits = sum(
            self.alg.payload_bits(value) for value in self.cells
        )
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, self.current_resident_payload_bits
        )
        for value in self.cells:
            self.observe_value(value, resident=True)

    def digest(self) -> str:
        return digest_json(
            {
                "cells": [self.alg.serialize(value) for value in self.cells],
                "generation": self.generation,
                "lease": self.lease,
                "stage": self.stage,
                "active_k": self.active_k,
            }
        )


def reset_observation(carrier: CoherentCarrier) -> None:
    carrier.projection_calls = 0
    carrier.current_resident_payload_bits = sum(
        carrier.alg.payload_bits(value) for value in carrier.cells
    )
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_resident_field_value_payload_bits = 0
    carrier.maximum_resident_numerator_signed_bits = 0
    carrier.maximum_resident_denominator_bits = 0
    carrier.maximum_observed_transient_field_value_payload_bits = 0
    carrier.maximum_observed_transient_numerator_signed_bits = 0
    carrier.maximum_observed_transient_denominator_bits = 0
    carrier.maximum_commitment_record_json_bytes = 0
    carrier.character_cell_multiplications = 0
    carrier.shear_cell_updates = 0
    carrier.observe_resident()


def apply_primitive(
    carrier: CoherentCarrier,
    primitive: Primitive,
    *,
    inverse: bool = False,
) -> None:
    validate_primitive(primitive)
    sign = -1 if inverse else 1
    if primitive.kind == "CHARACTER":
        degree = primitive.first
        for mode in range(MODE_COUNT):
            phase = carrier.alg.power(
                sign * primitive.coefficient_exponent * pow(mode, degree, PRIME)
            )
            carrier.observe_value(phase)
            carrier.write_cell(mode, carrier.alg.mul(carrier.cells[mode], phase))
            carrier.character_cell_multiplications += 1
    elif primitive.kind == "SHEAR":
        coefficient = carrier.alg.power(primitive.coefficient_exponent)
        if inverse:
            coefficient = negative(carrier.alg, coefficient)
        product = carrier.alg.mul(coefficient, carrier.cells[primitive.second])
        carrier.observe_value(coefficient)
        carrier.observe_value(product)
        carrier.write_cell(
            primitive.first,
            carrier.alg.add(carrier.cells[primitive.first], product),
        )
        carrier.shear_cell_updates += 1
    else:
        fail("non-integrable primitive reached the coherent carrier")
    carrier.observe_resident()


def apply_sequence(
    carrier: CoherentCarrier,
    primitives: Iterable[Primitive],
    *,
    inverse: bool = False,
) -> None:
    sequence = tuple(primitives)
    if inverse:
        sequence = tuple(reversed(sequence))
    for primitive in sequence:
        apply_primitive(carrier, primitive, inverse=inverse)


def load_seed(carrier: CoherentCarrier, *, inverse: bool = False) -> None:
    delta = negative(carrier.alg, carrier.alg.one) if inverse else carrier.alg.one
    carrier.write_cell(0, carrier.alg.add(carrier.cells[0], delta))
    carrier.observe_resident()


def forward(carrier: CoherentCarrier, program: CoherentProgram) -> None:
    if not isinstance(carrier, CoherentCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored coherent carrier")
    validate_program(program)
    carrier.lease = lease(program, carrier.alg)
    carrier.active_k = program.k
    carrier.stage = "FORWARD_ACTIVE"
    load_seed(carrier)
    apply_sequence(carrier, program.primitives)
    carrier.stage = "FORWARD_COMPLETE"


def project_boundary(carrier: CoherentCarrier, program: CoherentProgram) -> Any:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != lease(program, carrier.alg)
        or carrier.active_k != program.k
    ):
        fail("only the final owned coherent boundary may be projected")
    carrier.projection_calls += 1
    mode0_power = scalar_power(carrier.alg, carrier.cells[0], program.k - 1)
    product = carrier.alg.mul(mode0_power, carrier.cells[1])
    boundary = carrier.alg.mul(field_integer(carrier.alg, program.k), product)
    for value in (mode0_power, product, boundary):
        carrier.observe_value(value)
    return boundary


def inverse(carrier: CoherentCarrier, program: CoherentProgram) -> None:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != lease(program, carrier.alg)
        or carrier.active_k != program.k
    ):
        fail("inverse program does not own the coherent lease")
    carrier.stage = "INVERSE_ACTIVE"
    apply_sequence(carrier, program.primitives, inverse=True)
    load_seed(carrier, inverse=True)
    carrier.lease = None
    carrier.active_k = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact coherent-carrier restoration")


def public_program_integer_cells(program: CoherentProgram) -> int:
    return 5 + sum(4 for _ in program.primitives)


def execute_transaction(
    carrier: CoherentCarrier,
    program: CoherentProgram,
) -> dict[str, Any]:
    reset_observation(carrier)
    initial_digest = carrier.digest()
    backing_before = carrier.backing_identity()
    generation_before = carrier.generation
    stats_before = stats_snapshot(carrier.alg)
    forward(carrier, program)
    forward_chart_commitment, commitment_record_bytes = stream_vector_commitment(
        carrier.cells, carrier.alg
    )
    carrier.maximum_commitment_record_json_bytes = max(
        carrier.maximum_commitment_record_json_bytes, commitment_record_bytes
    )
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    stats_after = stats_snapshot(carrier.alg)
    descriptor = public_program_descriptor(program)
    descriptor_bytes = len(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    serialized_boundary = carrier.alg.serialize(boundary)
    implicit_dimension = math.comb(program.k + MODE_COUNT - 1, MODE_COUNT - 1)
    return {
        "k": program.k,
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "boundary": serialized_boundary,
        "forward_chart_commitment": forward_chart_commitment,
        "restored_exact_zero": carrier.exact_zero(),
        "same_backing": backing_before == carrier.backing_identity(),
        "generation_before": generation_before,
        "generation_after": carrier.generation,
        "implicit_occupation_dimension_h_k": implicit_dimension,
        "resident_phase_field_cells": len(carrier.cells),
        "resident_chart_rank": 1,
        "public_mode_topology_integer_cells": 2 * (MODE_COUNT - 1) + 2,
        "public_program_integer_cells": public_program_integer_cells(program),
        "public_program_json_bytes": descriptor_bytes,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_resident_field_value_payload_bits": carrier.maximum_resident_field_value_payload_bits,
        "maximum_resident_numerator_signed_bits": carrier.maximum_resident_numerator_signed_bits,
        "maximum_resident_denominator_bits": carrier.maximum_resident_denominator_bits,
        "maximum_observed_transient_field_value_payload_bits": carrier.maximum_observed_transient_field_value_payload_bits,
        "maximum_observed_transient_numerator_signed_bits": carrier.maximum_observed_transient_numerator_signed_bits,
        "maximum_observed_transient_denominator_bits": carrier.maximum_observed_transient_denominator_bits,
        "maximum_named_transaction_transient_field_cells": 5,
        "maximum_commitment_record_json_bytes": carrier.maximum_commitment_record_json_bytes,
        "field_operation_counts": stats_delta(stats_before, stats_after),
        "character_cell_multiplications": carrier.character_cell_multiplications,
        "shear_cell_updates": carrier.shear_cell_updates,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_boundary_json_bytes": len(
            json.dumps(serialized_boundary, separators=(",", ":")).encode("utf-8")
        ),
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "inverse_history_cells": 0,
        "inverse_history_retained": False,
        "snapshot_reload_used": False,
        "response_released_after_restoration": True,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "initial_digest": initial_digest,
        "restored_digest_with_generation": carrier.digest(),
        "accepted_path_occupation_vector_materialized": False,
        "accepted_path_occupation_topology_materialized": False,
        "accepted_path_character_table_materialized": False,
        "accepted_path_dense_operator_materialized": False,
        "accepted_path_coherent_component_expansion_materialized": False,
        "intermediate_chart_serialized_to_controller": False,
    }


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "implicit_occupation_dimension_h_k",
        "resident_phase_field_cells",
        "resident_chart_rank",
        "public_mode_topology_integer_cells",
        "public_program_integer_cells",
        "public_program_json_bytes",
        "maximum_resident_payload_bits",
        "maximum_resident_field_value_payload_bits",
        "maximum_resident_numerator_signed_bits",
        "maximum_resident_denominator_bits",
        "maximum_observed_transient_field_value_payload_bits",
        "maximum_observed_transient_numerator_signed_bits",
        "maximum_observed_transient_denominator_bits",
        "maximum_named_transaction_transient_field_cells",
        "maximum_commitment_record_json_bytes",
        "field_operation_counts",
        "character_cell_multiplications",
        "shear_cell_updates",
        "final_boundary_field_cells",
        "final_boundary_payload_bits",
        "final_boundary_json_bytes",
    )
    return {key: transaction[key] for key in keys}


def run_case(k: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    return execute_transaction(CoherentCarrier.create(alg), compile_program(k, family))


def compiled_two_scalar_baseline(
    k: int,
    family: str,
    alg: backend.Algebra,
    expected_boundary: Any,
) -> dict[str, Any]:
    program = compile_program(k, family)
    compile_before = stats_snapshot(alg)
    vector = [alg.zero for _ in range(MODE_COUNT)]
    vector[0] = alg.one
    for primitive in program.primitives:
        if primitive.kind == "CHARACTER":
            for mode in range(MODE_COUNT):
                vector[mode] = alg.mul(
                    vector[mode],
                    alg.power(
                        primitive.coefficient_exponent
                        * pow(mode, primitive.first, PRIME)
                    ),
                )
        else:
            vector[primitive.first] = alg.add(
                vector[primitive.first],
                alg.mul(
                    alg.power(primitive.coefficient_exponent),
                    vector[primitive.second],
                ),
            )
    compile_after = stats_snapshot(alg)
    retained_pair = (vector[0], vector[1])
    pair_commitment, maximum_record_json_bytes = stream_vector_commitment(
        retained_pair, alg
    )
    warm_before = stats_snapshot(alg)
    boundary = alg.mul(
        field_integer(alg, k),
        alg.mul(
            scalar_power(alg, retained_pair[0], k - 1), retained_pair[1]
        ),
    )
    warm_after = stats_snapshot(alg)
    return {
        "k": k,
        "family": family,
        "algebra": algebra_signature(alg),
        "program_fingerprint": program.fingerprint(),
        "boundary_agreement": alg.serialize(boundary) == expected_boundary,
        "compiler_working_field_cells": 17,
        "retained_warm_boundary_pair_field_cells": 2,
        "retained_warm_boundary_pair_payload_bits": sum(
            alg.payload_bits(value) for value in retained_pair
        ),
        "retained_warm_boundary_pair_commitment": pair_commitment,
        "maximum_commitment_record_json_bytes": maximum_record_json_bytes,
        "compile_field_operation_counts": stats_delta(compile_before, compile_after),
        "warm_projection_field_operation_counts": stats_delta(
            warm_before, warm_after
        ),
        "warm_named_transient_field_cells": 4,
        "compiled_pair_depends_on_family_and_algebra_but_not_k": True,
        "snapshot_or_phase_carrier_used": False,
    }


def transaction_boundary(program: CoherentProgram) -> Any:
    alg = backend.Algebra("F103", modulus=103, root=72)
    return execute_transaction(CoherentCarrier.create(alg), program)["boundary"]


def noncoherent_rank_two_witness() -> dict[str, Any]:
    # On the binary slice at k=2, a rank-one Veronese vector obeys
    # A20*A02=(A11/2)^2.  x0^2+x1^2 violates it.
    return {
        "k": 2,
        "state": "X0_SQUARED_PLUS_X1_SQUARED",
        "a20": 1,
        "a11": 0,
        "a02": 1,
        "rank_one_catalecticant_minor": 1,
        "rank_one_chart_membership": False,
    }


def actual_m128_grid_exit_witness(family: str) -> dict[str, Any]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    topology = m128.m127.OccupationTopology.compile(2)
    carrier = m128.m127.ExchangeCarrier.create(topology, alg)
    program = m128.compile_program(2, family)
    m128.forward(carrier, program)
    _, _, w_start = m128.m127.carrier_offsets(carrier)
    a20 = carrier.cells[
        w_start + topology.ranks[(2, *([0] * (MODE_COUNT - 1)))]
    ]
    a11 = carrier.cells[
        w_start + topology.ranks[(1, 1, *([0] * (MODE_COUNT - 2)))]
    ]
    a02 = carrier.cells[
        w_start + topology.ranks[(0, 2, *([0] * (MODE_COUNT - 2)))]
    ]
    inverse_two = alg.inverse(field_integer(alg, 2))
    b11 = alg.mul(a11, inverse_two)
    minor = alg.sub(alg.mul(a20, a02), alg.mul(b11, b11))
    m128.inverse(carrier, program)
    return {
        "family": family,
        "de_multinomial_binary_slice_commitment": digest_json(
            [int(a20), int(b11), int(a02)]
        ),
        "rank_one_catalecticant_minor": int(minor),
        "rank_one_chart_membership": minor == alg.zero,
        "prior_carrier_restored_after_control": carrier.exact_zero(),
    }


def controls() -> dict[str, Any]:
    program = compile_program(4, "PRIMARY")
    base_boundary = transaction_boundary(program)

    mutated_primitives = list(program.primitives)
    character_index = next(
        index for index, item in enumerate(mutated_primitives) if item.kind == "CHARACTER"
    )
    item = mutated_primitives[character_index]
    mutated_primitives[character_index] = replace(
        item, coefficient_exponent=1 + (item.coefficient_exponent % 16)
    )
    character_mutation_boundary = transaction_boundary(
        replace(program, primitives=tuple(mutated_primitives))
    )

    mutated_primitives = list(program.primitives)
    shear_index = next(
        index for index, item in enumerate(mutated_primitives) if item.kind == "SHEAR"
    )
    item = mutated_primitives[shear_index]
    mutated_primitives[shear_index] = replace(
        item, coefficient_exponent=1 + (item.coefficient_exponent % 16)
    )
    shear_mutation_boundary = transaction_boundary(
        replace(program, primitives=tuple(mutated_primitives))
    )

    invalid_rank_rejected = False
    try:
        validate_program(replace(program, chart_rank=2))
    except RuntimeError:
        invalid_rank_rejected = True

    nonintegrable_primitive_rejected = False
    try:
        validate_primitive(Primitive("GRID_ORBIT_SHEAR", 0, 0, 1))
    except RuntimeError:
        nonintegrable_primitive_rejected = True

    nonadjacent_shear_rejected = False
    try:
        validate_primitive(Primitive("SHEAR", 0, 2, 1))
    except RuntimeError:
        nonadjacent_shear_rejected = True

    alg = backend.Algebra("F103", modulus=103, root=72)
    carrier = CoherentCarrier.create(alg)
    forward(carrier, program)
    missing_inverse_detected = not carrier.exact_zero()

    wrong_program_rejected = False
    try:
        inverse(carrier, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_program_rejected = True

    premature = CoherentCarrier.create(backend.Algebra("F103", modulus=103, root=72))
    premature.lease = lease(program, premature.alg)
    premature.active_k = program.k
    premature.stage = "FORWARD_ACTIVE"
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    reordered = CoherentCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    forward(reordered, program)
    for primitive in program.primitives:
        apply_primitive(reordered, primitive, inverse=True)
    load_seed(reordered, inverse=True)
    reordered.lease = None
    reordered.active_k = None
    reordered.stage = "RESTORED"
    reordered_inverse_detected = not reordered.exact_zero()

    wrong_inverse = CoherentCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    forward(wrong_inverse, program)
    inverse_sequence = list(reversed(program.primitives))
    bad = inverse_sequence[0]
    inverse_sequence[0] = replace(
        bad, coefficient_exponent=1 + (bad.coefficient_exponent % 16)
    )
    for primitive in inverse_sequence:
        apply_primitive(wrong_inverse, primitive, inverse=True)
    load_seed(wrong_inverse, inverse=True)
    wrong_inverse_detected = any(
        value != wrong_inverse.alg.zero for value in wrong_inverse.cells
    )

    reciprocal_inverse = CoherentCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    forward(reciprocal_inverse, program)
    for primitive in reversed(program.primitives):
        if primitive.kind == "CHARACTER":
            apply_primitive(reciprocal_inverse, primitive, inverse=True)
        else:
            wrong_coefficient = reciprocal_inverse.alg.power(
                -primitive.coefficient_exponent
            )
            product = reciprocal_inverse.alg.mul(
                wrong_coefficient, reciprocal_inverse.cells[primitive.second]
            )
            reciprocal_inverse.write_cell(
                primitive.first,
                reciprocal_inverse.alg.add(
                    reciprocal_inverse.cells[primitive.first], product
                ),
            )
    load_seed(reciprocal_inverse, inverse=True)
    reciprocal_instead_of_additive_inverse_detected = any(
        value != reciprocal_inverse.alg.zero for value in reciprocal_inverse.cells
    )

    null_carrier_rejected = False
    try:
        forward(None, program)  # type: ignore[arg-type]
    except (RuntimeError, AttributeError):
        null_carrier_rejected = True

    invalid_k_rejected = False
    try:
        compile_program(3, "PRIMARY")
    except RuntimeError:
        invalid_k_rejected = True

    witness = noncoherent_rank_two_witness()
    grid_witnesses = [
        actual_m128_grid_exit_witness(family) for family in FAMILIES
    ]
    return {
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_inverse_detected": wrong_inverse_detected,
        "reciprocal_instead_of_additive_shear_inverse_detected": reciprocal_instead_of_additive_inverse_detected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "wrong_program_ownership_rejected": wrong_program_rejected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "invalid_degree_fixture_rejected": invalid_k_rejected,
        "rank_two_chart_descriptor_rejected": invalid_rank_rejected,
        "nonintegrable_grid_orbit_shear_rejected": nonintegrable_primitive_rejected,
        "nonadjacent_shear_rejected": nonadjacent_shear_rejected,
        "power_sum_character_mutation_changes_boundary": (
            base_boundary != character_mutation_boundary
        ),
        "mode_shear_mutation_changes_boundary": base_boundary != shear_mutation_boundary,
        "rank_two_noncoherent_witness_rejected": not witness["rank_one_chart_membership"],
        "rank_two_noncoherent_witness": witness,
        "actual_m128_grid_exit_witnesses": grid_witnesses,
        "actual_m128_grid_injection_leaves_rank_one_chart": all(
            not item["rank_one_chart_membership"]
            and item["prior_carrier_restored_after_control"]
            for item in grid_witnesses
        ),
        "accepted_path_occupation_vector_materialized": False,
        "accepted_path_occupation_topology_materialized": False,
        "accepted_path_character_table_materialized": False,
        "accepted_path_dense_operator_materialized": False,
        "intermediate_chart_serialized_to_controller": False,
        "catvm_boundary_claimed": False,
    }


def run() -> dict[str, Any]:
    exact = [
        run_case(k, "PRIMARY", backend.Algebra("Q_ZETA17")) for k in EXACT_K
    ]
    structural = []
    for modulus, root in FINITE_FIELDS:
        for k in DECLARED_K:
            item = run_case(
                k,
                "PRIMARY",
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            item["field"] = f"F{modulus}"
            structural.append(item)

    exact_baselines = [
        compiled_two_scalar_baseline(
            item["k"],
            item["family"],
            backend.Algebra("Q_ZETA17"),
            item["boundary"],
        )
        for item in exact
    ]
    structural_baselines = []
    for item in structural:
        modulus, root = next(
            pair for pair in FINITE_FIELDS if item["field"] == f"F{pair[0]}"
        )
        structural_baselines.append(
            compiled_two_scalar_baseline(
                item["k"],
                item["family"],
                backend.Algebra(item["field"], modulus=modulus, root=root),
                item["boundary"],
            )
        )
    if not all(
        item["boundary_agreement"]
        for item in (*exact_baselines, *structural_baselines)
    ):
        fail("compiled two-scalar classical baseline disagrees with the boundary")

    reuse_carrier = CoherentCarrier.create(backend.Algebra("Q_ZETA17"))
    first = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(16, "REUSE"))
    fresh = run_case(16, "REUSE", backend.Algebra("Q_ZETA17"))
    if reused["boundary"] != fresh["boundary"]:
        fail("restored coherent carrier disagrees with fresh unrelated program")
    if resource_signature(reused) != resource_signature(fresh):
        fail("restored coherent carrier changed the reported resource signature")

    exact_resident_payloads = {
        item["maximum_resident_payload_bits"] for item in exact
    }
    exact_resident_heights = {
        (
            item["maximum_resident_numerator_signed_bits"],
            item["maximum_resident_denominator_bits"],
        )
        for item in exact
    }
    if len(exact_resident_payloads) != 1 or len(exact_resident_heights) != 1:
        fail("fixed public program moved degree growth into resident chart precision")

    return {
        "schema": "CAT_CAS_F17_COHERENT_VERONESE_PHASE_CHART_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_PROGRAM_RESTRICTED_F17_COHERENT_VERONESE_PHASE_CHART_CLOSES_FIXED_P1_TO_P4_CHARACTERS_AND_BIDIRECTIONAL_MODE_SHEARS_IN_17_RESIDENT_COORDINATES_ACROSS_GROWING_EXCHANGE_SYMMETRIC_DEGREE_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_SEALED_WORDS_HAVE_A_STRONGER_TWO_SCALAR_WARM_CLASSICAL_BASELINE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "base_obstruction": "M128_UNIFORM_LINEAR_QUOTIENT_NO_GO_ON_FULL_H_K_MODULE",
            "chart": "RANK1_EXCHANGE_SYMMETRIC_COHERENT_VERONESE",
            "declared_degrees": DECLARED_K,
            "exact_q_zeta17_degrees": EXACT_K,
            "dual_field_degrees": DECLARED_K,
            "primitive_family": [
                "FIXED_P1_THROUGH_P4_PHASE_CHARACTERS",
                "BIDIRECTIONAL_ADJACENT_MODE_ONE_PARTICLE_SHEARS",
            ],
            "explicitly_excluded": [
                "M127_GRID_ORBIT_SHEAR",
                "SUMS_OF_MULTIPLE_COHERENT_COMPONENTS",
                "ARBITRARY_H_K_INPUT_STATE",
            ],
        },
        "closure_law": {
            "chart_map": "V_TO_SUM_J_V_J_X_J_RAISED_TO_K",
            "occupation_coefficient": "MULTINOMIAL_K_OVER_N_TIMES_PRODUCT_J_V_J_TO_N_J",
            "character_update": "V_J_MAPS_TO_ZETA17_TO_C_J_TO_D_TIMES_V_J",
            "shear_update": "V_ROW_MAPS_TO_V_ROW_PLUS_LAMBDA_V_PIVOT",
            "final_boundary": "K_TIMES_V0_TO_K_MINUS1_TIMES_V1",
            "resident_rank_independent_of_k": True,
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "compiled_two_scalar_warm_classical_baselines": {
            "exact_q_zeta17": exact_baselines,
            "dual_field": structural_baselines,
        },
        "exact_resident_payload_invariant_across_declared_exact_degrees": True,
        "reuse": {
            "first_k": 8,
            "reused_k": 16,
            "first_family": "PRIMARY",
            "reused_family": "REUSE",
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "fresh_restored_resource_signature_agreement": (
                resource_signature(reused) == resource_signature(fresh)
            ),
            "same_actual_backing_across_unrelated_programs": (
                first["same_backing"]
                and reused["same_backing"]
                and reuse_carrier.backing_identity() == backing
            ),
            "generation_after_two_transactions": reuse_carrier.generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "resource_law": {
            "accepted_resident_phase_field_cells": 17,
            "resident_chart_rank": 1,
            "implicit_full_occupation_dimension": "H_K_EQUALS_BINOMIAL_K_PLUS_16_CHOOSE_16",
            "accepted_path_occupation_coordinates": 0,
            "accepted_path_occupation_topology_integer_cells": 0,
            "accepted_path_character_table_cells": 0,
            "accepted_path_dense_operator_cells": 0,
            "inverse_history_cells": 0,
            "maximum_named_transaction_transient_field_cells": 5,
            "maximum_commitment_record_json_bytes_reported": True,
            "hashlib_internal_state_excluded": True,
            "public_program_and_mode_topology_counts_reported": True,
            "exact_resident_payload_height_reported": True,
            "final_boundary_payload_and_projection_work_reported": True,
            "python_container_native_bigint_and_whole_process_memory_excluded": True,
            "full_exact_bit_complexity_established": False,
        },
        "matched_baseline": {
            "strongest_sealed_fixture_warm": "COMPILED_TWO_SCALAR_V0_V1_RETENTION_WITH_CLOSED_FORM_OCCUPATION_PROJECTION",
            "descriptor_runtime_baseline": "IDENTICAL_17_COORDINATE_COHERENT_VECTOR_RECURRENCE_WITH_CLOSED_FORM_OCCUPATION_PROJECTION",
            "compiled_baseline_costs_public_word_compilation_and_reports_WARM_EVALUATION_SEPARATELY": True,
            "full_h_k_occupation_expansion_is_not_the_matched_baseline": True,
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_coherent_phase_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_projection_and_update_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "rank_one_coherent_chart_only": True,
            "fixed_public_primitive_schedule_only": True,
            "declared_k_values_only": DECLARED_K,
            "m127_grid_orbit_shear_closed": False,
            "multiple_coherent_component_closure_established": False,
            "arbitrary_h_k_input_closed": False,
            "general_nonlinear_quotient_established": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "THE_RANK1_COHERENT_VERONESE_CHART_REMOVES_H_K_RESIDENCY_FOR_AN_INTEGRABLE_CHARACTER_AND_ONE_PARTICLE_SHEAR_SUBPROGRAM_BUT_M127_GRID_ORBIT_SHEARS_AND_GENERIC_SUPERPOSITIONS_LEAVE_THE_CHART_AND_EACH_SEALED_PUBLIC_WORD_COMPILES_TO_AN_EVEN_SMALLER_TWO_SCALAR_WARM_CLASSICAL_BOUNDARY_RECURRENCE",
        "next_experiment": "EXACT_BOUNDED_SECANT_OR_GAUSSIAN_COHERENT_PHASE_CHART_FOR_ONE_SHARED_NONINTEGRABLE_ORBIT_COUPLING_WITH_RANK_GROWTH_AND_STRONGEST_COMPACT_CLASSICAL_BASELINE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = run()
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
