#!/usr/bin/env python3
"""Exact growing exchange-symmetric latent geometry over F17.

The accepted family declares ``k`` indistinguishable latent phase coordinates.
Its exact state is the degree-k symmetric power of the 17-mode phase space,
stored as occupation histograms rather than the labelled ``17**k`` tensor.
Two resident orbit vectors undergo exact lifted F17 Fourier transforms,
symmetric chirps, and two M126-derived planar matching shears controlled by
power sums that distinguish the declared multisets for ``k < 17``.

The Fourier lift is executed from a public exact elementary plan.  Every
two-mode shear is streamed through blocks of at most ``k + 1`` coefficients;
neither a dense occupation operator nor the labelled tensor is materialized.
The carrier is restored by reversing the actual operations and is then reused.

This is a bounded direct-process diagnostic.  Exchange symmetry is a premise,
not a compression of the original labelled open-chain family.  The strongest
compact classical baseline is the identical occupation-orbit recurrence, so
no distinct phase resource or computational advantage is assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_paired_phase_basis_holographic_matchgate_closure as matchgate


PRIME = 17
GRID_N = 4
EXACT_CASES = ((1, "PRIMARY"), (2, "PRIMARY"), (2, "REUSE"))
STRUCTURAL_K = (1, 2, 3, 4)
FAMILIES = ("PRIMARY", "REUSE")
FINITE_FIELDS = ((103, 72), (137, 16))
SYMMETRY_CLASS = "S_K_EXCHANGE_SYMMETRIC"


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


def field_power(alg: backend.Algebra, value: Any, exponent: int) -> Any:
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


def enumerate_histograms(k: int) -> tuple[tuple[int, ...], ...]:
    if not 1 <= k < PRIME:
        fail("exchange-symmetric degree must satisfy 1 <= k < 17")
    result: list[tuple[int, ...]] = []

    def visit(mode: int, remaining: int, prefix: list[int]) -> None:
        if mode == PRIME - 1:
            result.append(tuple((*prefix, remaining)))
            return
        for count in range(remaining + 1):
            prefix.append(count)
            visit(mode + 1, remaining - count, prefix)
            prefix.pop()

    visit(0, k, [])
    expected = math.comb(k + PRIME - 1, PRIME - 1)
    if len(result) != expected:
        fail("occupation histogram enumeration violated stars-and-bars")
    return tuple(result)


@dataclass(frozen=True)
class OccupationTopology:
    k: int
    histograms: tuple[tuple[int, ...], ...]
    ranks: dict[tuple[int, ...], int]
    zero_mode_rank: int

    @classmethod
    def compile(cls, k: int) -> "OccupationTopology":
        histograms = enumerate_histograms(k)
        ranks = {histogram: index for index, histogram in enumerate(histograms)}
        zero_mode = (k, *([0] * (PRIME - 1)))
        if zero_mode not in ranks:
            fail("zero-mode occupation is absent")
        return cls(k, histograms, ranks, ranks[zero_mode])

    @property
    def dimension(self) -> int:
        return len(self.histograms)

    @property
    def topology_integer_cells(self) -> int:
        return (PRIME + 1) * self.dimension


@dataclass(frozen=True)
class ExchangeProgram:
    k: int
    family: str
    symmetry_class: str
    module_weight_exponents: tuple[tuple[int, ...], tuple[int, ...]]
    module_control_exponents: tuple[tuple[int, ...], tuple[int, ...]]
    module_control_degrees: tuple[tuple[int, ...], tuple[int, ...]]
    chirp_exponent: int

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "k": self.k,
                "family": self.family,
                "symmetry_class": self.symmetry_class,
                "module_weight_exponents": self.module_weight_exponents,
                "module_control_exponents": self.module_control_exponents,
                "module_control_degrees": self.module_control_degrees,
                "chirp_exponent": self.chirp_exponent,
            }
        )


def compile_program(k: int, family: str) -> ExchangeProgram:
    if k not in STRUCTURAL_K:
        fail("exchange-symmetric program degree is outside the declared scope")
    if family not in FAMILIES:
        fail("unknown exchange-symmetric program family")
    variant = 0 if family == "PRIMARY" else 1
    edge_count = len(matchgate.grid_edges(GRID_N))
    weights: list[tuple[int, ...]] = []
    controls: list[tuple[int, ...]] = []
    degrees: list[tuple[int, ...]] = []
    for module in range(2):
        weights.append(
            tuple(
                1 + ((5 * edge + 7 * GRID_N + 3 * module + 4 * variant) % 16)
                for edge in range(edge_count)
            )
        )
        controls.append(
            tuple(
                1 + ((7 * edge + 5 * module + 6 * variant + GRID_N) % 16)
                for edge in range(edge_count)
            )
        )
        degrees.append(
            tuple(
                1 + ((5 * edge + 3 * module + 2 * variant) % k)
                for edge in range(edge_count)
            )
        )
    program = ExchangeProgram(
        k=k,
        family=family,
        symmetry_class=SYMMETRY_CLASS,
        module_weight_exponents=(weights[0], weights[1]),
        module_control_exponents=(controls[0], controls[1]),
        module_control_degrees=(degrees[0], degrees[1]),
        chirp_exponent=3 + 2 * variant,
    )
    validate_program(program)
    return program


def validate_program(program: ExchangeProgram) -> None:
    if program.k not in STRUCTURAL_K:
        fail("exchange-symmetric program degree changed")
    if program.family not in FAMILIES:
        fail("exchange-symmetric program family changed")
    if program.symmetry_class != SYMMETRY_CLASS:
        fail("symmetry-breaking labelled descriptors are outside this quotient")
    edge_count = len(matchgate.grid_edges(GRID_N))
    for rows in (program.module_weight_exponents, program.module_control_exponents):
        if len(rows) != 2 or any(len(row) != edge_count for row in rows):
            fail("exchange-symmetric module arity changed")
        if not all(1 <= value < PRIME for row in rows for value in row):
            fail("phase exponent is outside F17")
    if len(program.module_control_degrees) != 2 or any(
        len(row) != edge_count for row in program.module_control_degrees
    ):
        fail("power-sum control arity changed")
    if not all(
        1 <= degree <= program.k
        for row in program.module_control_degrees
        for degree in row
    ):
        fail("power-sum degree is outside the separating family")
    if not 1 <= program.chirp_exponent < PRIME:
        fail("chirp exponent is outside F17")


@dataclass(frozen=True)
class ElementaryOperation:
    kind: str
    first: int
    second: int
    coefficient: Any


def inverse_operation(
    operation: ElementaryOperation,
    alg: backend.Algebra,
) -> ElementaryOperation:
    if operation.kind == "SWAP":
        return operation
    if operation.kind == "SCALE":
        return ElementaryOperation(
            "SCALE", operation.first, operation.second, alg.inverse(operation.coefficient)
        )
    if operation.kind == "SHEAR":
        return ElementaryOperation(
            "SHEAR",
            operation.first,
            operation.second,
            negative(alg, operation.coefficient),
        )
    fail("unknown elementary Fourier operation")


def apply_single_particle_operation(
    vector: list[Any],
    operation: ElementaryOperation,
    alg: backend.Algebra,
) -> None:
    if operation.kind == "SWAP":
        vector[operation.first], vector[operation.second] = (
            vector[operation.second],
            vector[operation.first],
        )
    elif operation.kind == "SCALE":
        vector[operation.first] = alg.mul(
            vector[operation.first], operation.coefficient
        )
    elif operation.kind == "SHEAR":
        vector[operation.first] = alg.add(
            vector[operation.first],
            alg.mul(operation.coefficient, vector[operation.second]),
        )
    else:
        fail("unknown elementary Fourier operation")


@dataclass(frozen=True)
class FourierPlan:
    forward: tuple[ElementaryOperation, ...]
    inverse: tuple[ElementaryOperation, ...]
    algebra_signature: str
    compile_stats: dict[str, int]
    coefficient_payload_bits: int
    maximum_coefficient_payload_bits: int
    maximum_coefficient_numerator_signed_bits: int
    maximum_coefficient_denominator_bits: int
    compile_maximum_named_field_cells: int
    fingerprint_maximum_record_json_bytes: int
    fingerprint: str


def compile_fourier_plan(alg: backend.Algebra) -> FourierPlan:
    """Compile DFT17 into exact public row operations and verify reconstruction."""
    before = stats_snapshot(alg)
    direct = [
        [alg.power(row * column) for column in range(PRIME)]
        for row in range(PRIME)
    ]
    work = [row[:] for row in direct]
    elimination: list[ElementaryOperation] = []
    for column in range(PRIME):
        pivot_row = next(
            (row for row in range(column, PRIME) if not alg.is_zero(work[row][column])),
            None,
        )
        if pivot_row is None:
            fail("DFT17 public matrix is singular")
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            elimination.append(ElementaryOperation("SWAP", column, pivot_row, alg.one))
        scale = alg.inverse(work[column][column])
        for target in range(PRIME):
            work[column][target] = alg.mul(work[column][target], scale)
        elimination.append(ElementaryOperation("SCALE", column, column, scale))
        for row in range(PRIME):
            if row == column or alg.is_zero(work[row][column]):
                continue
            coefficient = negative(alg, work[row][column])
            for target in range(PRIME):
                work[row][target] = alg.add(
                    work[row][target],
                    alg.mul(coefficient, work[column][target]),
                )
            elimination.append(
                ElementaryOperation("SHEAR", row, column, coefficient)
            )
    if any(
        work[row][column] != (alg.one if row == column else alg.zero)
        for row in range(PRIME)
        for column in range(PRIME)
    ):
        fail("DFT17 public elimination did not reach identity")

    forward = tuple(
        inverse_operation(operation, alg) for operation in reversed(elimination)
    )
    inverse = tuple(
        inverse_operation(operation, alg) for operation in reversed(forward)
    )
    for source in range(PRIME):
        vector = [alg.one if index == source else alg.zero for index in range(PRIME)]
        for operation in forward:
            apply_single_particle_operation(vector, operation, alg)
        expected = [direct[target][source] for target in range(PRIME)]
        if vector != expected:
            fail("compiled Fourier plan disagrees with the public DFT17 matrix")
        for operation in inverse:
            apply_single_particle_operation(vector, operation, alg)
        identity = [alg.one if index == source else alg.zero for index in range(PRIME)]
        if vector != identity:
            fail("compiled inverse Fourier plan failed exact reconstruction")

    coefficient_payload_bits = 0
    maximum_coefficient_payload_bits = 0
    maximum_coefficient_numerator_signed_bits = 0
    maximum_coefficient_denominator_bits = 0
    fingerprint_maximum_record_json_bytes = 0
    fingerprint_hasher = hashlib.sha256()
    for operation in forward:
        record = json.dumps(
            {
            "kind": operation.kind,
            "first": operation.first,
            "second": operation.second,
            "coefficient": alg.serialize(operation.coefficient),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint_hasher.update(len(record).to_bytes(8, "big"))
        fingerprint_hasher.update(record)
        fingerprint_maximum_record_json_bytes = max(
            fingerprint_maximum_record_json_bytes, len(record)
        )
    for operation in (*forward, *inverse):
        if operation.kind == "SWAP":
            continue
        payload = alg.payload_bits(operation.coefficient)
        numerator, denominator = alg.coefficient_height(operation.coefficient)
        coefficient_payload_bits += payload
        maximum_coefficient_payload_bits = max(
            maximum_coefficient_payload_bits, payload
        )
        maximum_coefficient_numerator_signed_bits = max(
            maximum_coefficient_numerator_signed_bits, numerator
        )
        maximum_coefficient_denominator_bits = max(
            maximum_coefficient_denominator_bits, denominator
        )
    after = stats_snapshot(alg)
    return FourierPlan(
        forward=forward,
        inverse=inverse,
        algebra_signature=sha256_json(
            {
                "kind": alg.kind,
                "modulus": alg.modulus,
                "root": alg.serialize(alg.root),
            }
        ),
        compile_stats=stats_delta(before, after),
        coefficient_payload_bits=coefficient_payload_bits,
        maximum_coefficient_payload_bits=maximum_coefficient_payload_bits,
        maximum_coefficient_numerator_signed_bits=maximum_coefficient_numerator_signed_bits,
        maximum_coefficient_denominator_bits=maximum_coefficient_denominator_bits,
        compile_maximum_named_field_cells=5 * PRIME * PRIME + 3 * PRIME,
        fingerprint_maximum_record_json_bytes=fingerprint_maximum_record_json_bytes,
        fingerprint=fingerprint_hasher.hexdigest(),
    )


def algebra_signature(alg: backend.Algebra) -> str:
    return sha256_json(
        {
            "kind": alg.kind,
            "modulus": alg.modulus,
            "root": alg.serialize(alg.root),
        }
    )


def transaction_lease(program: ExchangeProgram, plan: FourierPlan) -> str:
    return sha256_json(
        {"program": program.fingerprint(), "fourier_plan": plan.fingerprint}
    )


@dataclass
class ExchangeCarrier:
    topology: OccupationTopology
    topology_fingerprint: str
    alg: backend.Algebra
    cells: list[Any]
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    factor_load_additions: int = 0
    factor_unload_additions: int = 0
    lifted_fourier_vector_transforms: int = 0
    lifted_elementary_operations: int = 0
    lifted_shear_blocks: int = 0
    lifted_shear_terms: int = 0
    lifted_scale_cells: int = 0
    lifted_swap_pairs: int = 0
    symmetric_chirp_multiplications: int = 0
    power_sum_evaluations: int = 0
    power_sum_integer_terms: int = 0
    module_boundary_evaluations: int = 0
    basis_mismatch_edge_contractions: int = 0
    orbit_shear_multiplications: int = 0
    orbit_shear_additions: int = 0
    projection_calls: int = 0
    current_resident_payload_bits: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_resident_field_value_payload_bits: int = 0
    maximum_resident_numerator_signed_bits: int = 0
    maximum_resident_denominator_bits: int = 0
    maximum_observed_scratch_field_value_payload_bits: int = 0
    maximum_observed_scratch_numerator_signed_bits: int = 0
    maximum_observed_scratch_denominator_bits: int = 0
    determinant_stats: matchgate.DeterminantStats = field(
        default_factory=matchgate.DeterminantStats
    )

    @classmethod
    def create(
        cls, topology: OccupationTopology, alg: backend.Algebra
    ) -> "ExchangeCarrier":
        edges = matchgate.grid_edges(GRID_N)
        carrier = cls(
            topology=topology,
            topology_fingerprint=sha256_json(
                {
                    "grid_n": GRID_N,
                    "grid_edges": edges,
                    "k": topology.k,
                    "occupation_dimension": topology.dimension,
                    "symmetry": SYMMETRY_CLASS,
                }
            ),
            alg=alg,
            cells=[
                alg.zero
                for _ in range(2 * len(edges) + 2 * topology.dimension)
            ],
        )
        carrier.observe_resident()
        return carrier

    def backing_identity(self) -> tuple[int, int]:
        return id(self), id(self.cells)

    def exact_zero(self) -> bool:
        return (
            all(value == self.alg.zero for value in self.cells)
            and self.lease is None
            and self.stage == "RESTORED"
        )

    def observe_resident(self) -> None:
        self.current_resident_payload_bits = sum(
            self.alg.payload_bits(value) for value in self.cells
        )
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            self.current_resident_payload_bits,
        )
        for value in self.cells:
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
            self.maximum_observed_scratch_field_value_payload_bits = max(
                self.maximum_observed_scratch_field_value_payload_bits, payload
            )
            self.maximum_observed_scratch_numerator_signed_bits = max(
                self.maximum_observed_scratch_numerator_signed_bits, numerator
            )
            self.maximum_observed_scratch_denominator_bits = max(
                self.maximum_observed_scratch_denominator_bits, denominator
            )

    def write_cell(self, index: int, value: Any) -> None:
        previous = self.alg.payload_bits(self.cells[index])
        current = self.alg.payload_bits(value)
        self.cells[index] = value
        self.current_resident_payload_bits += current - previous
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            self.current_resident_payload_bits,
        )
        self.observe_value(value, resident=True)

    def digest(self) -> str:
        return sha256_json(
            {
                "topology": self.topology_fingerprint,
                "cells": [self.alg.serialize(value) for value in self.cells],
                "generation": self.generation,
                "lease": self.lease,
                "stage": self.stage,
            }
        )


def carrier_offsets(carrier: ExchangeCarrier) -> tuple[int, int, int]:
    edge_count = len(matchgate.grid_edges(GRID_N))
    u_start = 2 * edge_count
    w_start = u_start + carrier.topology.dimension
    return edge_count, u_start, w_start


def reset_transaction_observation(carrier: ExchangeCarrier) -> None:
    carrier.factor_load_additions = 0
    carrier.factor_unload_additions = 0
    carrier.lifted_fourier_vector_transforms = 0
    carrier.lifted_elementary_operations = 0
    carrier.lifted_shear_blocks = 0
    carrier.lifted_shear_terms = 0
    carrier.lifted_scale_cells = 0
    carrier.lifted_swap_pairs = 0
    carrier.symmetric_chirp_multiplications = 0
    carrier.power_sum_evaluations = 0
    carrier.power_sum_integer_terms = 0
    carrier.module_boundary_evaluations = 0
    carrier.basis_mismatch_edge_contractions = 0
    carrier.orbit_shear_multiplications = 0
    carrier.orbit_shear_additions = 0
    carrier.projection_calls = 0
    carrier.current_resident_payload_bits = sum(
        carrier.alg.payload_bits(value) for value in carrier.cells
    )
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_resident_field_value_payload_bits = 0
    carrier.maximum_resident_numerator_signed_bits = 0
    carrier.maximum_resident_denominator_bits = 0
    carrier.maximum_observed_scratch_field_value_payload_bits = 0
    carrier.maximum_observed_scratch_numerator_signed_bits = 0
    carrier.maximum_observed_scratch_denominator_bits = 0
    carrier.determinant_stats = matchgate.DeterminantStats()
    carrier.observe_resident()


def load_program(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    *,
    inverse: bool = False,
) -> None:
    index = 0
    for row in program.module_weight_exponents:
        for exponent in row:
            value = carrier.alg.power(exponent)
            delta = negative(carrier.alg, value) if inverse else value
            carrier.write_cell(index, carrier.alg.add(carrier.cells[index], delta))
            index += 1
    _, u_start, _ = carrier_offsets(carrier)
    seed_index = u_start + carrier.topology.zero_mode_rank
    delta = negative(carrier.alg, carrier.alg.one) if inverse else carrier.alg.one
    carrier.write_cell(
        seed_index, carrier.alg.add(carrier.cells[seed_index], delta)
    )
    additions = index + 1
    if inverse:
        carrier.factor_unload_additions += additions
    else:
        carrier.factor_load_additions += additions
    carrier.observe_resident()


def apply_lifted_scale(
    carrier: ExchangeCarrier,
    start: int,
    operation: ElementaryOperation,
) -> None:
    for index, histogram in enumerate(carrier.topology.histograms):
        factor = field_power(
            carrier.alg, operation.coefficient, histogram[operation.first]
        )
        carrier.write_cell(
            start + index,
            carrier.alg.mul(carrier.cells[start + index], factor),
        )
        carrier.lifted_scale_cells += 1


def apply_lifted_swap(
    carrier: ExchangeCarrier,
    start: int,
    operation: ElementaryOperation,
) -> None:
    for index, histogram in enumerate(carrier.topology.histograms):
        target = list(histogram)
        target[operation.first], target[operation.second] = (
            target[operation.second],
            target[operation.first],
        )
        target_index = carrier.topology.ranks[tuple(target)]
        if target_index > index:
            left = start + index
            right = start + target_index
            carrier.cells[left], carrier.cells[right] = carrier.cells[right], carrier.cells[left]
            carrier.lifted_swap_pairs += 1


def apply_lifted_shear(
    carrier: ExchangeCarrier,
    start: int,
    operation: ElementaryOperation,
) -> None:
    row = operation.first
    pivot = operation.second
    powers = [carrier.alg.one]
    for _ in range(carrier.topology.k):
        powers.append(carrier.alg.mul(powers[-1], operation.coefficient))
    for base in carrier.topology.histograms:
        if base[row] != 0:
            continue
        total = base[pivot]
        indices = []
        for row_count in range(total + 1):
            member = list(base)
            member[row] = row_count
            member[pivot] = total - row_count
            indices.append(carrier.topology.ranks[tuple(member)])
        old = [carrier.cells[start + index] for index in indices]
        updated = [carrier.alg.zero for _ in indices]
        for input_row, amplitude in enumerate(old):
            input_pivot = total - input_row
            for moved in range(input_pivot + 1):
                output_row = input_row + moved
                coefficient = carrier.alg.mul(
                    integer(carrier.alg, math.comb(input_pivot, moved)),
                    powers[moved],
                )
                term = carrier.alg.mul(amplitude, coefficient)
                updated[output_row] = carrier.alg.add(updated[output_row], term)
                carrier.observe_value(coefficient)
                carrier.observe_value(term)
                carrier.observe_value(updated[output_row])
                carrier.lifted_shear_terms += 1
        for index, value in zip(indices, updated, strict=True):
            carrier.write_cell(start + index, value)
        carrier.lifted_shear_blocks += 1


def apply_lifted_fourier_segment(
    carrier: ExchangeCarrier,
    start: int,
    plan: FourierPlan,
    *,
    inverse: bool = False,
) -> None:
    operations = plan.inverse if inverse else plan.forward
    for operation in operations:
        if operation.kind == "SCALE":
            apply_lifted_scale(carrier, start, operation)
        elif operation.kind == "SWAP":
            apply_lifted_swap(carrier, start, operation)
        elif operation.kind == "SHEAR":
            apply_lifted_shear(carrier, start, operation)
        else:
            fail("compiled Fourier plan contains an unknown operation")
        carrier.lifted_elementary_operations += 1
    carrier.lifted_fourier_vector_transforms += 1


def apply_lifted_fourier(
    carrier: ExchangeCarrier,
    plan: FourierPlan,
    *,
    inverse: bool = False,
) -> None:
    _, u_start, w_start = carrier_offsets(carrier)
    apply_lifted_fourier_segment(carrier, u_start, plan, inverse=inverse)
    apply_lifted_fourier_segment(carrier, w_start, plan, inverse=inverse)
    carrier.observe_resident()


def power_sums(histogram: tuple[int, ...], maximum_degree: int) -> tuple[int, ...]:
    return tuple(
        sum(count * pow(mode, degree, PRIME) for mode, count in enumerate(histogram))
        % PRIME
        for degree in range(1, maximum_degree + 1)
    )


def apply_symmetric_chirp(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    *,
    inverse: bool = False,
) -> None:
    sign = -1 if inverse else 1
    _, u_start, w_start = carrier_offsets(carrier)
    for index, histogram in enumerate(carrier.topology.histograms):
        quadratic = sum(
            count * mode * mode for mode, count in enumerate(histogram)
        ) % PRIME
        phase = carrier.alg.power(sign * program.chirp_exponent * quadratic)
        carrier.write_cell(
            u_start + index,
            carrier.alg.mul(carrier.cells[u_start + index], phase),
        )
        carrier.write_cell(
            w_start + index,
            carrier.alg.mul(carrier.cells[w_start + index], phase),
        )
        carrier.symmetric_chirp_multiplications += 2
    carrier.observe_resident()


def kasteleyn_matrix(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    module: int,
    histogram: tuple[int, ...],
) -> list[list[Any]]:
    black, white = matchgate.black_white_vertices(GRID_N)
    black_index = {vertex: index for index, vertex in enumerate(black)}
    white_index = {vertex: index for index, vertex in enumerate(white)}
    matrix = [[carrier.alg.zero for _ in white] for _ in black]
    edge_count, _, _ = carrier_offsets(carrier)
    powers = power_sums(histogram, program.k)
    carrier.power_sum_evaluations += program.k
    carrier.power_sum_integer_terms += PRIME * program.k
    for edge_index, edge in enumerate(matchgate.grid_edges(GRID_N)):
        degree = program.module_control_degrees[module][edge_index]
        exponent = (
            program.module_control_exponents[module][edge_index]
            * powers[degree - 1]
        )
        weight = carrier.alg.mul(
            carrier.cells[module * edge_count + edge_index],
            carrier.alg.power(exponent),
        )
        carrier.basis_mismatch_edge_contractions += 1
        first, second = edge
        left, right = (first, second) if first in black_index else (second, first)
        value = (
            weight
            if matchgate.kasteleyn_edge_sign(first, second) == 1
            else negative(carrier.alg, weight)
        )
        row = black_index[left]
        column = white_index[right]
        matrix[row][column] = carrier.alg.add(matrix[row][column], value)
    return matrix


def module_boundary(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    module: int,
    histogram: tuple[int, ...],
) -> Any:
    if module not in (0, 1):
        fail("exchange-symmetric module index changed")
    value = matchgate.determinant(
        kasteleyn_matrix(carrier, program, module, histogram),
        carrier.alg,
        carrier.determinant_stats,
    )
    carrier.module_boundary_evaluations += 1
    calibration = matchgate.reference_calibration_sign(GRID_N)
    return value if calibration == 1 else negative(carrier.alg, value)


def apply_orbit_shear(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    module: int,
    *,
    inverse: bool = False,
) -> None:
    _, u_start, w_start = carrier_offsets(carrier)
    for index, histogram in enumerate(carrier.topology.histograms):
        boundary = module_boundary(carrier, program, module, histogram)
        term = carrier.alg.mul(boundary, carrier.cells[u_start + index])
        carrier.observe_value(term)
        carrier.write_cell(
            w_start + index,
            (
                carrier.alg.sub(carrier.cells[w_start + index], term)
                if inverse
                else carrier.alg.add(carrier.cells[w_start + index], term)
            ),
        )
        carrier.orbit_shear_multiplications += 1
        carrier.orbit_shear_additions += 1
    carrier.observe_resident()


def forward(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    plan: FourierPlan,
) -> None:
    if not isinstance(carrier, ExchangeCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored exchange-symmetric carrier")
    validate_program(program)
    if carrier.topology.k != program.k:
        fail("program degree does not own the occupation topology")
    if plan.algebra_signature != algebra_signature(carrier.alg):
        fail("Fourier plan algebra does not own the carrier")
    carrier.lease = transaction_lease(program, plan)
    carrier.stage = "FORWARD_ACTIVE"
    load_program(carrier, program)
    apply_lifted_fourier(carrier, plan)
    apply_orbit_shear(carrier, program, 0)
    apply_symmetric_chirp(carrier, program)
    apply_lifted_fourier(carrier, plan)
    apply_orbit_shear(carrier, program, 1)
    carrier.stage = "FORWARD_COMPLETE"
    carrier.observe_resident()


def project_boundary(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    plan: FourierPlan,
) -> Any:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != transaction_lease(program, plan)
    ):
        fail("only the completed owned orbit boundary may be projected")
    carrier.projection_calls += 1
    _, _, w_start = carrier_offsets(carrier)
    return carrier.cells[w_start + carrier.topology.zero_mode_rank]


def inverse(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    plan: FourierPlan,
) -> None:
    if (
        carrier.stage != "FORWARD_COMPLETE"
        or carrier.lease != transaction_lease(program, plan)
    ):
        fail("inverse program does not own the exchange-symmetric lease")
    carrier.stage = "INVERSE_ACTIVE"
    apply_orbit_shear(carrier, program, 1, inverse=True)
    apply_lifted_fourier(carrier, plan, inverse=True)
    apply_symmetric_chirp(carrier, program, inverse=True)
    apply_orbit_shear(carrier, program, 0, inverse=True)
    apply_lifted_fourier(carrier, plan, inverse=True)
    load_program(carrier, program, inverse=True)
    carrier.lease = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact exchange-symmetric restoration")


RESOURCE_SIGNATURE_KEYS = (
    "resident_phase_field_cells",
    "resident_grid_weight_field_cells",
    "resident_orbit_field_cells",
    "occupation_dimension",
    "module_boundary_evaluations",
    "basis_mismatch_edge_contractions",
    "lifted_fourier_vector_transforms",
    "lifted_elementary_operations",
    "symmetric_chirp_multiplications",
    "maximum_named_transaction_transient_field_cells",
    "factor_load_additions",
    "factor_unload_additions",
    "resident_carrier_restoration_class",
)


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    signature = {key: transaction[key] for key in RESOURCE_SIGNATURE_KEYS}
    signature["determinant_stats"] = transaction["determinant_stats"]
    return signature


def execute_transaction(
    carrier: ExchangeCarrier,
    program: ExchangeProgram,
    plan: FourierPlan,
) -> dict[str, Any]:
    reset_transaction_observation(carrier)
    initial = carrier.digest()
    backing = carrier.backing_identity()
    generation = carrier.generation
    forward(carrier, program, plan)
    boundary = project_boundary(carrier, program, plan)
    inverse(carrier, program, plan)
    determinant_dimension = GRID_N * GRID_N // 2
    maximum_transient = max(
        2 * determinant_dimension * determinant_dimension + 5,
        3 * (program.k + 1),
    )
    maximum_transient_value_payload = max(
        carrier.maximum_observed_scratch_field_value_payload_bits,
        carrier.determinant_stats.maximum_observed_field_value_payload_bits,
    )
    serialized_boundary = carrier.alg.serialize(boundary)
    edge_count = len(matchgate.grid_edges(GRID_N))
    program_json_bytes = len(
        json.dumps(
            {
                "k": program.k,
                "family": program.family,
                "symmetry_class": program.symmetry_class,
                "module_weight_exponents": program.module_weight_exponents,
                "module_control_exponents": program.module_control_exponents,
                "module_control_degrees": program.module_control_degrees,
                "chirp_exponent": program.chirp_exponent,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "k": program.k,
        "family": program.family,
        "program_fingerprint": program.fingerprint(),
        "fourier_plan_fingerprint": plan.fingerprint,
        "boundary": serialized_boundary,
        "occupation_dimension": carrier.topology.dimension,
        "labelled_tensor_dimension_not_materialized": PRIME ** program.k,
        "resident_phase_field_cells": len(carrier.cells),
        "resident_grid_weight_field_cells": 2 * edge_count,
        "resident_orbit_field_cells": 2 * carrier.topology.dimension,
        "public_occupation_topology_integer_cells": carrier.topology.topology_integer_cells,
        "public_program_integer_cells": 6 * edge_count + 2,
        "public_program_json_bytes": program_json_bytes,
        "public_grid_edge_coordinate_integer_cells": 4 * edge_count,
        "public_grid_vertex_coordinate_integer_cells": 2 * GRID_N * GRID_N,
        "public_fourier_plan_operations": len(plan.forward) + len(plan.inverse),
        "public_fourier_plan_field_cells": sum(
            operation.kind != "SWAP" for operation in (*plan.forward, *plan.inverse)
        ),
        "public_fourier_plan_coefficient_payload_bits": plan.coefficient_payload_bits,
        "public_fourier_plan_maximum_coefficient_payload_bits": plan.maximum_coefficient_payload_bits,
        "public_fourier_plan_maximum_coefficient_numerator_signed_bits": plan.maximum_coefficient_numerator_signed_bits,
        "public_fourier_plan_maximum_coefficient_denominator_bits": plan.maximum_coefficient_denominator_bits,
        "public_fourier_plan_compile_stats": plan.compile_stats,
        "public_fourier_plan_compile_maximum_named_field_cells": plan.compile_maximum_named_field_cells,
        "public_fourier_plan_fingerprint_maximum_record_json_bytes": plan.fingerprint_maximum_record_json_bytes,
        "module_boundary_evaluations": carrier.module_boundary_evaluations,
        "basis_mismatch_edge_contractions": carrier.basis_mismatch_edge_contractions,
        "power_sum_evaluations": carrier.power_sum_evaluations,
        "power_sum_integer_terms": carrier.power_sum_integer_terms,
        "lifted_fourier_vector_transforms": carrier.lifted_fourier_vector_transforms,
        "lifted_elementary_operations": carrier.lifted_elementary_operations,
        "lifted_shear_blocks": carrier.lifted_shear_blocks,
        "lifted_shear_terms": carrier.lifted_shear_terms,
        "lifted_scale_cells": carrier.lifted_scale_cells,
        "lifted_swap_pairs": carrier.lifted_swap_pairs,
        "symmetric_chirp_multiplications": carrier.symmetric_chirp_multiplications,
        "orbit_shear_multiplications": carrier.orbit_shear_multiplications,
        "orbit_shear_additions": carrier.orbit_shear_additions,
        "determinant_matrix_dimension": determinant_dimension,
        "maximum_named_transaction_transient_field_cells": maximum_transient,
        "maximum_named_transaction_transient_integer_cells": 48,
        "maximum_observed_transient_field_value_payload_bits": maximum_transient_value_payload,
        "maximum_named_transaction_transient_payload_bits_upper_bound": maximum_transient * maximum_transient_value_payload,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_resident_field_value_payload_bits": carrier.maximum_resident_field_value_payload_bits,
        "maximum_resident_numerator_signed_bits": carrier.maximum_resident_numerator_signed_bits,
        "maximum_resident_denominator_bits": carrier.maximum_resident_denominator_bits,
        "maximum_observed_scratch_numerator_signed_bits": carrier.maximum_observed_scratch_numerator_signed_bits,
        "maximum_observed_scratch_denominator_bits": carrier.maximum_observed_scratch_denominator_bits,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_boundary_json_bytes": len(
            json.dumps(serialized_boundary, separators=(",", ":")).encode("utf-8")
        ),
        "determinant_stats": carrier.determinant_stats.as_json(),
        "factor_load_additions": carrier.factor_load_additions,
        "factor_unload_additions": carrier.factor_unload_additions,
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "accepted_path_labelled_tensor_materialized": False,
        "accepted_path_dense_occupation_operator_materialized": False,
        "accepted_path_occupation_boundary_vector_materialized": False,
        "accepted_path_assignment_or_relation_table_materialized": False,
        "inverse_history_retained": False,
        "generation_before": generation,
        "generation_after": carrier.generation,
        "restoration_generation_increment": carrier.generation == generation + 1,
        "same_backing": carrier.backing_identity() == backing,
        "initial_digest": initial,
        "restored_digest_with_generation": carrier.digest(),
        "exact_phase_carrier_restored": carrier.exact_zero(),
        "response_released_after_restoration": True,
        "snapshot_reload_used": False,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_buffers_restoration_class": "NO_RESTORATION_CLAIM",
    }


def run_case(k: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    topology = OccupationTopology.compile(k)
    plan = compile_fourier_plan(alg)
    return execute_transaction(
        ExchangeCarrier.create(topology, alg), compile_program(k, family), plan
    )


def program_with_degrees(
    program: ExchangeProgram,
    degrees: tuple[tuple[int, ...], tuple[int, ...]],
) -> ExchangeProgram:
    return ExchangeProgram(
        k=program.k,
        family=program.family,
        symmetry_class=program.symmetry_class,
        module_weight_exponents=program.module_weight_exponents,
        module_control_exponents=program.module_control_exponents,
        module_control_degrees=degrees,
        chirp_exponent=program.chirp_exponent,
    )


def semantic_mutation(program: ExchangeProgram) -> ExchangeProgram:
    controls = [list(row) for row in program.module_control_exponents]
    controls[0][0] = 1 + (controls[0][0] % 16)
    return ExchangeProgram(
        k=program.k,
        family=program.family,
        symmetry_class=program.symmetry_class,
        module_weight_exponents=program.module_weight_exponents,
        module_control_exponents=(tuple(controls[0]), tuple(controls[1])),
        module_control_degrees=program.module_control_degrees,
        chirp_exponent=program.chirp_exponent,
    )


def transaction_boundary(
    topology: OccupationTopology,
    program: ExchangeProgram,
) -> Any:
    alg = backend.Algebra("F103", modulus=103, root=72)
    return execute_transaction(
        ExchangeCarrier.create(topology, alg), program, compile_fourier_plan(alg)
    )["boundary"]


def controls() -> dict[str, Any]:
    topology = OccupationTopology.compile(2)
    program = compile_program(2, "PRIMARY")

    base_boundary = transaction_boundary(topology, program)
    total_sum_rows = tuple(
        tuple(1 for _ in row) for row in program.module_control_degrees
    )
    total_sum_boundary = transaction_boundary(
        topology,
        program_with_degrees(
            program,
            (total_sum_rows[0], total_sum_rows[1]),
        ),
    )
    missing_degree_rows = [list(row) for row in program.module_control_degrees]
    changed_edge = next(
        index for index, degree in enumerate(missing_degree_rows[0]) if degree == 2
    )
    missing_degree_rows[0][changed_edge] = 1
    missing_degree_boundary = transaction_boundary(
        topology,
        program_with_degrees(
            program,
            (tuple(missing_degree_rows[0]), tuple(missing_degree_rows[1])),
        ),
    )
    mutation_boundary = transaction_boundary(topology, semantic_mutation(program))

    alg = backend.Algebra("F103", modulus=103, root=72)
    plan = compile_fourier_plan(alg)
    missing = ExchangeCarrier.create(topology, alg)
    forward(missing, program, plan)
    missing_inverse_detected = not missing.exact_zero()

    wrong_alg = backend.Algebra("F103", modulus=103, root=72)
    wrong = ExchangeCarrier.create(topology, wrong_alg)
    wrong_plan = compile_fourier_plan(wrong_alg)
    forward(wrong, program, wrong_plan)
    wrong_inverse_ownership_detected = False
    try:
        inverse(wrong, compile_program(2, "REUSE"), wrong_plan)
    except RuntimeError:
        wrong_inverse_ownership_detected = True

    premature_alg = backend.Algebra("F103", modulus=103, root=72)
    premature = ExchangeCarrier.create(topology, premature_alg)
    premature_plan = compile_fourier_plan(premature_alg)
    premature.lease = transaction_lease(program, premature_plan)
    premature.stage = "FORWARD_ACTIVE"
    premature_projection_rejected = False
    try:
        project_boundary(premature, program, premature_plan)
    except RuntimeError:
        premature_projection_rejected = True

    null_carrier_rejected = False
    try:
        forward(None, program, plan)  # type: ignore[arg-type]
    except (RuntimeError, AttributeError):
        null_carrier_rejected = True

    reordered_alg = backend.Algebra("F103", modulus=103, root=72)
    reordered = ExchangeCarrier.create(topology, reordered_alg)
    reordered_plan = compile_fourier_plan(reordered_alg)
    forward(reordered, program, reordered_plan)
    apply_orbit_shear(reordered, program, 0, inverse=True)
    apply_orbit_shear(reordered, program, 1, inverse=True)
    apply_lifted_fourier(reordered, reordered_plan, inverse=True)
    apply_symmetric_chirp(reordered, program, inverse=True)
    apply_lifted_fourier(reordered, reordered_plan, inverse=True)
    load_program(reordered, program, inverse=True)
    reordered.lease = None
    reordered.stage = "RESTORED"
    reordered_inverse_detected = not reordered.exact_zero()

    symmetry_breaking_rejected = False
    try:
        invalid = ExchangeProgram(
            k=program.k,
            family=program.family,
            symmetry_class="LABELLED_PARTICLE_CONTROLS",
            module_weight_exponents=program.module_weight_exponents,
            module_control_exponents=program.module_control_exponents,
            module_control_degrees=program.module_control_degrees,
            chirp_exponent=program.chirp_exponent,
        )
        validate_program(invalid)
    except RuntimeError:
        symmetry_breaking_rejected = True

    first_particles = (0, 2)
    second_particles = (2, 0)
    first_histogram = tuple(first_particles.count(mode) for mode in range(PRIME))
    second_histogram = tuple(second_particles.count(mode) for mode in range(PRIME))
    permuted_particles_same_histogram = (
        first_particles != second_particles
        and first_histogram == second_histogram
        and power_sums(first_histogram, 2) == power_sums(second_histogram, 2)
    )

    return {
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_inverse_ownership_detected": wrong_inverse_ownership_detected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "symmetry_breaking_descriptor_rejected": symmetry_breaking_rejected,
        "particle_permutation_preserves_histogram_invariants": permuted_particles_same_histogram,
        "total_sum_overmerge_changes_boundary": base_boundary != total_sum_boundary,
        "missing_power_sum_changes_boundary": base_boundary != missing_degree_boundary,
        "semantic_control_mutation_changes_boundary": base_boundary != mutation_boundary,
        "accepted_path_labelled_tensor_materialized": False,
        "accepted_path_dense_occupation_operator_materialized": False,
        "accepted_path_occupation_boundary_vector_materialized": False,
        "intermediate_orbit_state_serialized": False,
        "catvm_boundary_claimed": False,
    }


def run() -> dict[str, Any]:
    exact = [
        run_case(k, family, backend.Algebra("Q_ZETA17"))
        for k, family in EXACT_CASES
    ]
    structural = []
    for modulus, root in FINITE_FIELDS:
        for k in STRUCTURAL_K:
            item = run_case(
                k,
                "PRIMARY",
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            item["field"] = f"F{modulus}"
            structural.append(item)
    reuse_structural = run_case(
        4,
        "REUSE",
        backend.Algebra("F103", modulus=103, root=72),
    )
    reuse_structural["field"] = "F103"
    structural.append(reuse_structural)

    reuse_topology = OccupationTopology.compile(2)
    reuse_alg = backend.Algebra("Q_ZETA17")
    reuse_plan = compile_fourier_plan(reuse_alg)
    reuse_carrier = ExchangeCarrier.create(reuse_topology, reuse_alg)
    first = execute_transaction(
        reuse_carrier, compile_program(2, "PRIMARY"), reuse_plan
    )
    reuse_backing = reuse_carrier.backing_identity()
    reused = execute_transaction(
        reuse_carrier, compile_program(2, "REUSE"), reuse_plan
    )
    fresh_alg = backend.Algebra("Q_ZETA17")
    fresh = execute_transaction(
        ExchangeCarrier.create(reuse_topology, fresh_alg),
        compile_program(2, "REUSE"),
        compile_fourier_plan(fresh_alg),
    )
    reused_signature = resource_signature(reused)
    fresh_signature = resource_signature(fresh)
    if reused["boundary"] != fresh["boundary"]:
        fail("restored exchange-symmetric carrier reuse disagrees with fresh execution")
    if reused_signature != fresh_signature:
        fail("restored exchange-symmetric reuse changed its resource signature")

    observed_dimensions = {
        str(k): OccupationTopology.compile(k).dimension for k in STRUCTURAL_K
    }
    return {
        "schema": "CAT_CAS_F17_EXCHANGE_SYMMETRIC_LATENT_GEOMETRY_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_GROWING_EXCHANGE_SYMMETRIC_F17_LATENT_GEOMETRY_USES_DEGREE_K_OCCUPATION_ORBITS_WITH_EXACT_LIFTED_FOURIER_NON_SUM_ONLY_POWER_SUM_GRID_CONTROLS_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_THE_IDENTICAL_CLASSICAL_ORBIT_RECURRENCE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "grid_module_topology": "M126_EVEN_OPEN_SQUARE_GRID_N4_ONLY",
            "exchange_symmetric_degrees": STRUCTURAL_K,
            "exact_q_zeta17_cases": EXACT_CASES,
            "dual_field_primary_degrees": STRUCTURAL_K,
            "additional_f103_reuse_degree": 4,
            "symmetry_class": SYMMETRY_CLASS,
            "occupation_dimensions": observed_dimensions,
            "labelled_dimensions_not_materialized": {
                str(k): PRIME ** k for k in STRUCTURAL_K
            },
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "reuse": {
            "k": 2,
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "fresh_restored_resource_signature_agreement": reused_signature == fresh_signature,
            "same_actual_backing_across_unrelated_programs": (
                first["same_backing"]
                and reused["same_backing"]
                and reuse_carrier.backing_identity() == reuse_backing
            ),
            "generation_after_two_transactions": reuse_carrier.generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "resource_law": {
            "occupation_dimension": "H_K_EQUALS_BINOMIAL_K_PLUS_16_CHOOSE_16",
            "observed_occupation_dimensions": observed_dimensions,
            "resident_phase_field_cells": "48_PLUS_2_H_K_AT_GRID_N4",
            "resident_grid_weight_field_cells": 48,
            "resident_orbit_field_cells": "2_H_K",
            "labelled_tensor_field_cells_not_materialized": "17_TO_THE_K",
            "public_occupation_topology_integer_cells": "18_H_K",
            "public_program_integer_cells": 146,
            "public_grid_edge_coordinate_integer_cells": 96,
            "public_grid_vertex_coordinate_integer_cells": 32,
            "public_exact_fourier_plan_field_cells": "REPORTED_PER_TRANSACTION",
            "public_fourier_compile_named_field_cells": 1496,
            "forward_and_inverse_module_determinants": "4_H_K",
            "power_sum_integer_work": "4_H_K_K_17_TERMS",
            "maximum_lifted_fourier_scratch_field_cells": "3_TIMES_K_PLUS_1",
            "accepted_named_transaction_transient_field_cells": "MAX_133_COMMA_3_TIMES_K_PLUS_1",
            "accepted_exact_field_operation_work": "POLYNOMIAL_IN_H_K_FOR_K_AT_MOST_4; FULL_EXACT_BIT_COMPLEXITY_NOT_ESTABLISHED",
            "dense_occupation_operator_cells": 0,
            "labelled_tensor_cells": 0,
            "occupation_boundary_vector_cells": 0,
            "assignment_or_relation_table_cells": 0,
            "inverse_history_cells": 0,
            "final_boundary_field_cells": 1,
            "controller_backend_traffic_bytes": 0,
            "oracle_and_control_runs_are_sequential_and_excluded_from_accepted_peak": True,
            "python_container_native_bigint_and_whole_process_memory_excluded": True,
        },
        "matched_baselines": {
            "strongest_implemented": "IDENTICAL_H_K_COORDINATE_EXCHANGE_SYMMETRIC_ORBIT_RECURRENCE_WITH_THE_SAME_PUBLIC_ELEMENTARY_DFT17_PLAN_AND_STREAMED_KASTELEYN_CLOSURES",
            "phase_advantage_over_matched_classical": False,
            "labelled_17_TO_THE_K_tensor_is_not_the_matched_baseline": True,
        },
        "restoration": {
            "resident_grid_and_orbit_phase_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_determinant_and_lift_blocks": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "exchange_symmetry_is_required_not_inferred": True,
            "original_labelled_open_chain_family_compressed": False,
            "growing_exact_relation_preserving_orbit_geometry": True,
            "non_sum_only_multiset_controls_through_k4": True,
            "power_sums_p1_through_pk_separate_multisets_for_k_less_than_17": True,
            "arbitrary_labelled_or_mixed_symmetry_relations": False,
            "fixed_rank_across_growing_k": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "EXCHANGE_SYMMETRY_REPLACES_LABELLED_17_TO_THE_K_GROWTH_BY_EXACT_H_K_ORBIT_GROWTH_AND_NON_SUM_ONLY_CONTROLS_BUT_THE_RESIDENT_RANK_STILL_GROWS_AS_BINOMIAL_K_PLUS_16_CHOOSE_16_AND_COMPACT_CLASSICAL_SOFTWARE_EXECUTES_THE_IDENTICAL_RECURRENCE",
        "next_experiment": "TEST_EXACT_LOW_RANK_CLOSURE_OR_NATIVE_PHASE_CONVOLUTION_ACROSS_GROWING_EXCHANGE_SYMMETRIC_DEGREE_WITHOUT_MOVING_H_K_GROWTH_INTO_PRECISION_OR_HISTORY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
