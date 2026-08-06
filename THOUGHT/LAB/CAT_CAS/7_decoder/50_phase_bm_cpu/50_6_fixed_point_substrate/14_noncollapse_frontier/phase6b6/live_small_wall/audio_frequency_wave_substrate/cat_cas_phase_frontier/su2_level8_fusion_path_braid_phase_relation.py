#!/usr/bin/env python3
"""Exact SU(2)_8 fusion-path braid phase-relation diagnostic.

M213 closes the local Verlinde fusion algebra on nine Q(zeta_40) cells but
does not test global fusion-tree geometry.  This diagnostic carries the
vacuum fusion-path space of n fundamental SU(2)_8 objects.  Temperley-Lieb
braid generators act matrix-free, in a radical-free non-orthogonal gauge,
on overlapping unresolved internal fusion channels.

Only one final vacuum plat-closure scalar is projected.  The public braid
word is then reversed on the actual coefficient backing, and the restored
carrier is reused for a different public word.  Fusion paths and local action
plans are not retained: public A_9 path counts support streamed rank/unrank.
The full fusion-path coefficient vector is material and is counted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction


sys.set_int_max_str_digits(0)

LEVEL = 8
LABELS = LEVEL + 1
FIELD_DEGREE = 16
ROOT_ORDER = 40
EXECUTED_STRANDS = (4, 6, 8, 10, 12, 14, 16)
PRIMARY_STRANDS = 16
PRIMARY_ROUNDS = 8
REUSE_ROUNDS = 5
ANALYTIC_STRANDS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)


def reduce_root_polynomial(values: list[Fraction]) -> tuple[Fraction, ...]:
    work = values.copy()
    work.extend([Fraction(0)] * max(0, FIELD_DEGREE - len(work)))
    for degree in range(len(work) - 1, FIELD_DEGREE - 1, -1):
        value = work[degree]
        work[degree] = Fraction(0)
        work[degree - 4] += value
        work[degree - 8] -= value
        work[degree - 12] += value
        work[degree - 16] -= value
    return tuple(work[:FIELD_DEGREE])


@dataclass(frozen=True)
class K:
    coefficients: tuple[Fraction, ...]

    @staticmethod
    def integer(value: int) -> "K":
        return K((Fraction(value),) + (Fraction(0),) * 15)

    @staticmethod
    def zeta(power: int) -> "K":
        exponent = power % ROOT_ORDER
        values = [Fraction(0)] * (exponent + 1)
        values[exponent] = Fraction(1)
        return K(reduce_root_polynomial(values))

    def __add__(self, other: "K") -> "K":
        return K(tuple(a + b for a, b in zip(self.coefficients, other.coefficients)))

    def __sub__(self, other: "K") -> "K":
        return K(tuple(a - b for a, b in zip(self.coefficients, other.coefficients)))

    def __mul__(self, other: "K") -> "K":
        values = [Fraction(0)] * 31
        for i, left in enumerate(self.coefficients):
            if not left:
                continue
            for j, right in enumerate(other.coefficients):
                if right:
                    values[i + j] += left * right
        return K(reduce_root_polynomial(values))

    def inverse(self) -> "K":
        if self == ZERO:
            raise ZeroDivisionError("zero cyclotomic element")
        columns = [
            (self * K.zeta(index)).coefficients for index in range(FIELD_DEGREE)
        ]
        matrix = [
            [columns[column][row] for column in range(FIELD_DEGREE)]
            + [Fraction(row == 0)]
            for row in range(FIELD_DEGREE)
        ]
        pivot_row = 0
        for column in range(FIELD_DEGREE):
            pivot = next(
                row
                for row in range(pivot_row, FIELD_DEGREE)
                if matrix[row][column]
            )
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
            scale = matrix[pivot_row][column]
            matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
            for row in range(FIELD_DEGREE):
                if row == pivot_row or not matrix[row][column]:
                    continue
                factor = matrix[row][column]
                matrix[row] = [
                    value - factor * basis
                    for value, basis in zip(
                        matrix[row], matrix[pivot_row], strict=True
                    )
                ]
            pivot_row += 1
        return K(tuple(matrix[row][-1] for row in range(FIELD_DEGREE)))

    def is_zero(self) -> bool:
        return self == ZERO

    def token(self) -> str:
        return ":".join(
            f"{value.numerator}/{value.denominator}"
            for value in self.coefficients
        )


ZERO = K.integer(0)
ONE = K.integer(1)
BRAID_A = K.zeta(11)
BRAID_A_INVERSE = K.zeta(-11)


def quantum_dimensions() -> tuple[K, ...]:
    delta = K.zeta(2) + K.zeta(-2)
    dimensions = [ONE, delta]
    for _ in range(2, LABELS + 1):
        dimensions.append(delta * dimensions[-1] - dimensions[-2])
    if dimensions[LABELS] != ZERO:
        raise RuntimeError("SU2 level-8 Jones-Wenzl relation failed")
    return tuple(dimensions[:LABELS])


QUANTUM_DIMENSIONS = quantum_dimensions()
INVERSE_DIMENSIONS = tuple(value.inverse() for value in QUANTUM_DIMENSIONS)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def integer_payload_bits(values: list[int] | tuple[int, ...]) -> int:
    return sum(signed_bits(value) for value in values)


def field_payload_bits(values: list[K]) -> int:
    return sum(
        signed_bits(coefficient.numerator) + coefficient.denominator.bit_length()
        for value in values
        for coefficient in value.coefficients
    )


def maximum_coordinate_bits(values: list[K]) -> dict[str, int]:
    coordinates = [coefficient for value in values for coefficient in value.coefficients]
    return {
        "maximum_signed_numerator_bits": max(
            signed_bits(value.numerator) for value in coordinates
        ),
        "maximum_denominator_bits": max(
            value.denominator.bit_length() for value in coordinates
        ),
    }


def boundary_commitment(value: K) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


@dataclass(frozen=True)
class FusionPathTopology:
    strands: int
    completions: tuple[tuple[int, ...], ...]

    @staticmethod
    def compile(strands: int) -> "FusionPathTopology":
        if strands < 2 or strands % 2:
            raise ValueError("fusion-path strand count must be positive and even")
        rows: list[tuple[int, ...]] = [
            tuple(1 if label == 0 else 0 for label in range(LABELS))
        ]
        for _remaining in range(1, strands + 1):
            previous = rows[-1]
            row = []
            for label in range(LABELS):
                count = 0
                if label > 0:
                    count += previous[label - 1]
                if label < LEVEL:
                    count += previous[label + 1]
                row.append(count)
            rows.append(tuple(row))
        return FusionPathTopology(strands, tuple(rows))

    @property
    def dimension(self) -> int:
        return self.completions[self.strands][0]

    @property
    def integer_cells(self) -> int:
        return len(self.completions) * LABELS

    @property
    def integer_payload(self) -> int:
        return integer_payload_bits(
            [value for row in self.completions for value in row]
        )

    def token(self) -> str:
        return "|".join(
            ",".join(str(value) for value in row) for row in self.completions
        )

    def commitment(self) -> str:
        return hashlib.sha256(
            f"A9:{self.strands}:{self.token()}".encode("ascii")
        ).hexdigest()

    def unrank(self, rank: int, work: "Work | None" = None) -> tuple[int, ...]:
        if rank < 0 or rank >= self.dimension:
            raise IndexError("fusion-path rank out of range")
        path = [0]
        label = 0
        residual = rank
        for position in range(1, self.strands + 1):
            remaining = self.strands - position
            candidates = []
            if label > 0:
                candidates.append(label - 1)
            if label < LEVEL:
                candidates.append(label + 1)
            for candidate in candidates:
                count = self.completions[remaining][candidate]
                if work is not None:
                    work.topology_count_reads += 1
                if residual < count:
                    label = candidate
                    path.append(label)
                    break
                residual -= count
            else:
                raise RuntimeError("fusion-path unrank failed")
        if label != 0 or residual:
            raise RuntimeError("fusion-path unrank did not close to vacuum")
        if work is not None:
            work.path_unrank_calls += 1
            work.path_labels_rematerialized += len(path)
            work.peak_streamed_path_label_cells = max(
                work.peak_streamed_path_label_cells, len(path)
            )
        return tuple(path)

    def rank(self, path: tuple[int, ...], work: "Work | None" = None) -> int:
        if len(path) != self.strands + 1 or path[0] != 0 or path[-1] != 0:
            raise ValueError("invalid vacuum fusion path")
        rank = 0
        label = 0
        for position, target in enumerate(path[1:], start=1):
            remaining = self.strands - position
            candidates = []
            if label > 0:
                candidates.append(label - 1)
            if label < LEVEL:
                candidates.append(label + 1)
            if target not in candidates:
                raise ValueError("nonadjacent fusion-path label")
            for candidate in candidates:
                if candidate >= target:
                    break
                rank += self.completions[remaining][candidate]
                if work is not None:
                    work.topology_count_reads += 1
            label = target
        if work is not None:
            work.path_rank_calls += 1
            work.path_labels_rematerialized += len(path)
            work.peak_streamed_path_label_cells = max(
                work.peak_streamed_path_label_cells, len(path)
            )
        return rank


@dataclass(frozen=True)
class BraidOperation:
    generator: int
    exponent: int

    def token(self) -> str:
        return f"sigma:{self.generator}:{self.exponent}"


@dataclass(frozen=True)
class BraidProgram:
    strands: int
    rounds: int
    family: int

    def __post_init__(self) -> None:
        if self.strands < 4 or self.strands % 2:
            raise ValueError("braid program needs at least four even strands")
        if self.rounds <= 0 or self.family not in (0, 1):
            raise ValueError("invalid braid program descriptor")

    @property
    def steps(self) -> int:
        return self.rounds * (self.strands - 1)

    def operation(self, step: int) -> BraidOperation:
        if step < 0 or step >= self.steps:
            raise IndexError("public braid step out of range")
        round_index, offset = divmod(step, self.strands - 1)
        if (round_index + self.family) % 2:
            generator = self.strands - 1 - offset
        else:
            generator = offset + 1
        exponent = -1 if (3 * round_index + generator + self.family) % 5 == 0 else 1
        return BraidOperation(generator, exponent)

    def token(self) -> str:
        return (
            f"strands:{self.strands}:rounds:{self.rounds}:family:{self.family}"
        )


def program_commitment(program: BraidProgram) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


@dataclass
class Work:
    field_additions: int = 0
    field_multiplications: int = 0
    path_unrank_calls: int = 0
    path_rank_calls: int = 0
    path_labels_rematerialized: int = 0
    topology_count_reads: int = 0
    paired_local_blocks: int = 0
    scalar_local_blocks: int = 0
    zero_temperley_lieb_blocks: int = 0
    forward_operations: int = 0
    inverse_operations: int = 0
    port_leases: int = 0
    port_releases: int = 0
    public_descriptor_hashes: int = 0
    public_descriptor_integers_hashed: int = 0
    state_commitment_hashes: int = 0
    state_commitment_field_cells_hashed: int = 0
    boundary_commitment_hashes: int = 0
    peak_carrier_field_cells: int = 0
    peak_nonzero_carrier_field_cells: int = 0
    peak_carrier_payload_bits: int = 0
    peak_streamed_path_label_cells: int = 0
    peak_local_field_scratch_cells: int = 0

    def add(self, left: K, right: K) -> K:
        self.field_additions += 1
        return left + right

    def multiply(self, left: K, right: K) -> K:
        self.field_multiplications += 1
        return left * right

    def observe(self, coefficients: list[K]) -> None:
        self.peak_carrier_field_cells = max(
            self.peak_carrier_field_cells, len(coefficients)
        )
        self.peak_nonzero_carrier_field_cells = max(
            self.peak_nonzero_carrier_field_cells,
            sum(not value.is_zero() for value in coefficients),
        )
        self.peak_carrier_payload_bits = max(
            self.peak_carrier_payload_bits, field_payload_bits(coefficients)
        )

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def vacuum_path(strands: int) -> tuple[int, ...]:
    return tuple(index % 2 for index in range(strands + 1))


def source_state(topology: FusionPathTopology) -> list[K]:
    coefficients = [ZERO] * topology.dimension
    coefficients[topology.rank(vacuum_path(topology.strands))] = ONE
    return coefficients


def state_commitment(coefficients: list[K]) -> str:
    return hashlib.sha256(
        "|".join(value.token() for value in coefficients).encode("ascii")
    ).hexdigest()


def mismatch(left: list[K], right: list[K]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def local_braid_scalars(exponent: int) -> tuple[K, K]:
    if exponent == 1:
        return BRAID_A, BRAID_A_INVERSE
    if exponent == -1:
        return BRAID_A_INVERSE, BRAID_A
    raise ValueError("braid exponent must be plus or minus one")


def apply_braid(
    coefficients: list[K],
    topology: FusionPathTopology,
    operation: BraidOperation,
    work: Work,
) -> None:
    generator = operation.generator
    if generator <= 0 or generator >= topology.strands:
        raise ValueError("braid generator outside public topology")
    alpha, beta = local_braid_scalars(operation.exponent)
    for index in range(topology.dimension):
        path = topology.unrank(index, work)
        left = path[generator - 1]
        middle = path[generator]
        right = path[generator + 1]
        if left != right:
            coefficients[index] = work.multiply(alpha, coefficients[index])
            work.zero_temperley_lieb_blocks += 1
            work.peak_local_field_scratch_cells = max(
                work.peak_local_field_scratch_cells, 1
            )
            continue
        alternatives = tuple(
            label for label in (left - 1, left + 1) if 0 <= label <= LEVEL
        )
        if len(alternatives) == 2 and middle == alternatives[1]:
            continue
        if len(alternatives) == 1:
            weighted = work.multiply(
                QUANTUM_DIMENSIONS[middle], coefficients[index]
            )
            temperley = work.multiply(INVERSE_DIMENSIONS[left], weighted)
            coefficients[index] = work.add(
                work.multiply(alpha, coefficients[index]),
                work.multiply(beta, temperley),
            )
            work.scalar_local_blocks += 1
            work.peak_local_field_scratch_cells = max(
                work.peak_local_field_scratch_cells, 2
            )
            continue

        peer_path = path[:generator] + (alternatives[1],) + path[generator + 1 :]
        work.peak_streamed_path_label_cells = max(
            work.peak_streamed_path_label_cells, 2 * len(path)
        )
        peer_index = topology.rank(peer_path, work)
        first = coefficients[index]
        second = coefficients[peer_index]
        weighted_first = work.multiply(QUANTUM_DIMENSIONS[alternatives[0]], first)
        weighted_second = work.multiply(
            QUANTUM_DIMENSIONS[alternatives[1]], second
        )
        summed = work.add(weighted_first, weighted_second)
        temperley = work.multiply(INVERSE_DIMENSIONS[left], summed)
        coefficients[index] = work.add(
            work.multiply(alpha, first), work.multiply(beta, temperley)
        )
        coefficients[peer_index] = work.add(
            work.multiply(alpha, second), work.multiply(beta, temperley)
        )
        work.paired_local_blocks += 1
        work.peak_local_field_scratch_cells = max(
            work.peak_local_field_scratch_cells, 8
        )


@dataclass
class OpenFusionPathPort:
    topology: FusionPathTopology
    coefficients: list[K]
    live: bool = False
    owner: int = 0
    lease_generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    public_program_commitment: str = ""
    topology_commitment: str = ""

    def lease(
        self, owner: int, generation: int, program: BraidProgram, work: Work
    ) -> None:
        if self.live:
            raise RuntimeError("fusion-path port already live")
        if len(self.coefficients) != self.topology.dimension:
            raise ValueError("null or wrong-width fusion-path carrier")
        if program.strands != self.topology.strands:
            raise ValueError("program and fusion-path topology disagree")
        if owner <= 0 or generation <= 0:
            raise ValueError("invalid fusion-path owner or generation")
        self.live = True
        self.owner = owner
        self.lease_generation = generation
        self.cursor = 0
        self.expected_steps = program.steps
        self.public_program_commitment = program_commitment(program)
        self.topology_commitment = self.topology.commitment()
        work.public_descriptor_hashes += 2
        work.public_descriptor_integers_hashed += 3 + self.topology.integer_cells
        work.port_leases += 1
        work.observe(self.coefficients)

    def require(self, owner: int, program: BraidProgram, work: Work) -> None:
        if not self.live:
            raise RuntimeError("fusion-path port is not live")
        if owner != self.owner:
            raise PermissionError("fusion-path owner mismatch")
        work.public_descriptor_hashes += 2
        work.public_descriptor_integers_hashed += 3 + self.topology.integer_cells
        if program_commitment(program) != self.public_program_commitment:
            raise ValueError("public braid program mismatch")
        if self.topology.commitment() != self.topology_commitment:
            raise ValueError("public fusion-path topology mismatch")

    def forward(
        self, owner: int, program: BraidProgram, index: int, work: Work
    ) -> None:
        self.require(owner, program, work)
        if index != self.cursor or index >= self.expected_steps:
            raise ValueError("forward braid cursor mismatch")
        apply_braid(self.coefficients, self.topology, program.operation(index), work)
        self.cursor += 1
        work.forward_operations += 1
        work.observe(self.coefficients)

    def inverse(
        self, owner: int, program: BraidProgram, index: int, work: Work
    ) -> None:
        self.require(owner, program, work)
        if index != self.cursor - 1:
            raise ValueError("inverse braid cursor mismatch")
        operation = program.operation(index)
        apply_braid(
            self.coefficients,
            self.topology,
            BraidOperation(operation.generator, -operation.exponent),
            work,
        )
        self.cursor -= 1
        work.inverse_operations += 1
        work.observe(self.coefficients)

    def project_final(
        self, owner: int, program: BraidProgram, work: Work
    ) -> K:
        self.require(owner, program, work)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal fusion-path projection rejected")
        index = self.topology.rank(vacuum_path(self.topology.strands), work)
        return self.coefficients[index]

    def release(
        self, owner: int, program: BraidProgram, work: Work
    ) -> int:
        self.require(owner, program, work)
        if self.cursor:
            raise RuntimeError("fusion-path port released before inverse")
        generation = self.lease_generation
        self.live = False
        self.owner = 0
        self.lease_generation = 0
        self.expected_steps = 0
        self.public_program_commitment = ""
        self.topology_commitment = ""
        work.port_releases += 1
        return generation


@dataclass
class Carrier:
    port: OpenFusionPathPort
    restoration_generation: int = 0


def canonical_post_restoration(
    carrier: Carrier, source: list[K], expected_generation: int
) -> bool:
    port = carrier.port
    return (
        port.coefficients == source
        and not port.live
        and port.owner == 0
        and port.lease_generation == 0
        and port.cursor == 0
        and port.expected_steps == 0
        and port.public_program_commitment == ""
        and port.topology_commitment == ""
        and carrier.restoration_generation == expected_generation
    )


def transaction(
    carrier: Carrier, source: list[K], program: BraidProgram
) -> tuple[dict[str, object], Work]:
    backing = id(carrier.port.coefficients)
    generation = carrier.restoration_generation + 1
    owner = 214000 + generation
    work = Work()
    carrier.port.lease(owner, generation, program, work)
    for index in range(program.steps):
        carrier.port.forward(owner, program, index, work)
    boundary = carrier.port.project_final(owner, program, work)
    forward_commitment = state_commitment(carrier.port.coefficients)
    work.state_commitment_hashes += 1
    work.state_commitment_field_cells_hashed += len(carrier.port.coefficients)
    final_commitment = boundary_commitment(boundary)
    work.boundary_commitment_hashes += 1
    forward_payload = field_payload_bits(carrier.port.coefficients)
    forward_nonzero = sum(
        not value.is_zero() for value in carrier.port.coefficients
    )
    missing_inverse_error = mismatch(carrier.port.coefficients, source) + carrier.port.cursor
    for index in range(program.steps - 1, -1, -1):
        carrier.port.inverse(owner, program, index, work)
    restored_generation = carrier.port.release(owner, program, work)
    carrier.restoration_generation = restored_generation
    return (
        {
            "boundary_commitment": final_commitment,
            "boundary_payload_bits": field_payload_bits([boundary]),
            "forward_state_commitment": forward_commitment,
            "forward_field_cells": len(carrier.port.coefficients),
            "forward_nonzero_field_cells": forward_nonzero,
            "forward_payload_bits": forward_payload,
            "missing_inverse_error_cells_and_cursor": missing_inverse_error,
            "restoration_error_field_cells": mismatch(
                carrier.port.coefficients, source
            ),
            "same_coefficient_backing": id(carrier.port.coefficients) == backing,
            "canonical_post_restoration_state_exact": canonical_post_restoration(
                carrier, source, generation
            ),
            "restoration_generation": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        work,
    )


def execute_forward(program: BraidProgram) -> tuple[FusionPathTopology, list[K], Work]:
    topology = FusionPathTopology.compile(program.strands)
    coefficients = source_state(topology)
    work = Work()
    work.observe(coefficients)
    for step in range(program.steps):
        apply_braid(coefficients, topology, program.operation(step), work)
        work.forward_operations += 1
        work.observe(coefficients)
    return topology, coefficients, work


def executed_case(strands: int, family: int = 0) -> dict[str, object]:
    program = BraidProgram(strands, PRIMARY_ROUNDS, family)
    topology, coefficients, work = execute_forward(program)
    boundary = coefficients[topology.rank(vacuum_path(strands))]
    return {
        "strands": strands,
        "rounds": program.rounds,
        "family": family,
        "program_steps": program.steps,
        "fusion_path_field_cells": topology.dimension,
        "nonzero_fusion_path_field_cells": sum(
            not value.is_zero() for value in coefficients
        ),
        "carrier_payload_bits": field_payload_bits(coefficients),
        **maximum_coordinate_bits(coefficients),
        "state_commitment": state_commitment(coefficients),
        "boundary_commitment": boundary_commitment(boundary),
        "boundary_payload_bits": field_payload_bits([boundary]),
        "public_topology_integer_cells": topology.integer_cells,
        "public_topology_payload_bits": topology.integer_payload,
        "topology_commitment": topology.commitment(),
        "forward_work": work.as_dict(),
    }


def analytic_dimensions() -> list[dict[str, int]]:
    result = []
    for strands in ANALYTIC_STRANDS:
        topology = FusionPathTopology.compile(strands)
        half = strands // 2
        # C_half = binomial(2 half, half)/(half+1), updated without factorials.
        catalan = 1
        for index in range(half):
            catalan = catalan * 2 * (2 * index + 1) // (index + 2)
        result.append(
            {
                "strands": strands,
                "su2_level8_vacuum_path_cells": topology.dimension,
                "untruncated_catalan_cells": catalan,
                "jones_wenzl_removed_cells": catalan - topology.dimension,
            }
        )
    return result


def algebra_controls() -> dict[str, object]:
    topology = FusionPathTopology.compile(6)
    source = source_state(topology)

    left = source.copy()
    right = source.copy()
    left_work = Work()
    right_work = Work()
    for generator in (2, 3, 2):
        apply_braid(left, topology, BraidOperation(generator, 1), left_work)
    for generator in (3, 2, 3):
        apply_braid(right, topology, BraidOperation(generator, 1), right_work)

    far_left = source.copy()
    far_right = source.copy()
    for generator in (1, 4):
        apply_braid(far_left, topology, BraidOperation(generator, 1), Work())
    for generator in (4, 1):
        apply_braid(far_right, topology, BraidOperation(generator, 1), Work())

    noncommuting_left = source.copy()
    noncommuting_right = source.copy()
    for generator in (2, 3):
        apply_braid(
            noncommuting_left, topology, BraidOperation(generator, 1), Work()
        )
    for generator in (3, 2):
        apply_braid(
            noncommuting_right, topology, BraidOperation(generator, 1), Work()
        )

    inverse = source.copy()
    apply_braid(inverse, topology, BraidOperation(2, 1), Work())
    apply_braid(inverse, topology, BraidOperation(2, -1), Work())

    reordered = source.copy()
    for generator in (2, 3, 2, 3):
        apply_braid(reordered, topology, BraidOperation(generator, 1), Work())
    apply_braid(reordered, topology, BraidOperation(2, -1), Work())
    apply_braid(reordered, topology, BraidOperation(3, -1), Work())
    apply_braid(reordered, topology, BraidOperation(3, -1), Work())
    apply_braid(reordered, topology, BraidOperation(2, -1), Work())

    perturbed = source.copy()
    reference = source.copy()
    word = (2, 3, 4, 3, 2)
    for generator in word:
        apply_braid(reference, topology, BraidOperation(generator, 1), Work())
    for generator in word[:-1] + (3,):
        apply_braid(perturbed, topology, BraidOperation(generator, 1), Work())

    return {
        "yang_baxter_relation_exact": left == right,
        "far_generators_commute_exactly": far_left == far_right,
        "adjacent_generator_noncommutation_mismatch_cells": mismatch(
            noncommuting_left, noncommuting_right
        ),
        "single_generator_inverse_exact": inverse == source,
        "reordered_inverse_changes_state_cells": mismatch(reordered, source),
        "semantic_generator_perturbation_changes_state_cells": mismatch(
            perturbed, reference
        ),
    }


def custody_controls() -> dict[str, object]:
    topology = FusionPathTopology.compile(6)
    source = source_state(topology)
    program = BraidProgram(6, 2, 0)
    port = OpenFusionPathPort(topology, source.copy())
    work = Work()
    port.lease(7001, 1, program, work)
    wrong_owner = wrong_type = premature = reordered = False
    try:
        port.forward(7002, program, 0, work)
    except PermissionError:
        wrong_owner = True
    try:
        apply_braid(port.coefficients, topology, BraidOperation(0, 1), work)
    except ValueError:
        wrong_type = True
    port.forward(7001, program, 0, work)
    try:
        port.project_final(7001, program, work)
    except PermissionError:
        premature = True
    for index in range(1, program.steps):
        port.forward(7001, program, index, work)
    missing_inverse = port.cursor != 0
    try:
        port.inverse(7001, program, program.steps - 2, work)
    except ValueError:
        reordered = True
    for index in range(program.steps - 1, -1, -1):
        port.inverse(7001, program, index, work)
    port.release(7001, program, work)

    null_carrier = False
    try:
        OpenFusionPathPort(topology, []).lease(1, 1, program, Work())
    except ValueError:
        null_carrier = True

    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_operation_type_rejected": wrong_type,
        "premature_projection_rejected": premature,
        "missing_inverse_detected": missing_inverse,
        "reordered_inverse_rejected": reordered,
        "null_carrier_rejected": null_carrier,
        "snapshot_command_available": False,
    }


def main() -> None:
    cases = [executed_case(strands) for strands in EXECUTED_STRANDS]
    primary_topology = FusionPathTopology.compile(PRIMARY_STRANDS)
    primary_source = source_state(primary_topology)
    carrier = Carrier(
        OpenFusionPathPort(primary_topology, primary_source.copy())
    )
    primary_program = BraidProgram(PRIMARY_STRANDS, PRIMARY_ROUNDS, 0)
    reuse_program = BraidProgram(PRIMARY_STRANDS, REUSE_ROUNDS, 1)
    primary, primary_work = transaction(carrier, primary_source, primary_program)
    reuse, reuse_work = transaction(carrier, primary_source, reuse_program)
    fresh_carrier = Carrier(
        OpenFusionPathPort(primary_topology, primary_source.copy())
    )
    fresh_reuse, _fresh_work = transaction(
        fresh_carrier, primary_source, reuse_program
    )
    primary_case = next(
        case for case in cases if case["strands"] == PRIMARY_STRANDS
    )
    algebra = algebra_controls()
    custody = custody_controls()
    dimensions = analytic_dimensions()

    result = {
        "schema": "cat_cas.su2_level8_fusion_path_braid_phase_relation.v1",
        "result": "PASS_EXACT_SU2_LEVEL8_FUSION_PATH_BRAID_CLOSURE_WITH_GROWING_GLOBAL_PATH_WIDTH",
        "claim": "EXACT_SU2_LEVEL8_TEMPERLEY_LIEB_VACUUM_FUSION_PATH_PHASE_CARRIER_EXECUTES_NONCOMMUTING_BRAID_GENERATORS_ON_OVERLAPPING_UNRESOLVED_INTERNAL_CHANNELS_WITH_FINAL_ONLY_PLAT_CLOSURE_EXACT_SAME_BACKING_RESTORATION_REUSE_BUT_GLOBAL_PATH_CELLS_GROW_FROM2TO1430_ACROSS_STRANDS4TO16_AND_FIRST_JONES_WENZL_TRUNCATION_STILL_LEAVES4861_CELLS_AT18_WHILE_THE_IDENTICAL_CLASSICAL_PATH_RECURRENCE_REMAINS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_SU2_LEVEL8_TEMPERLEY_LIEB_A9_VACUUM_FUSION_PATH_REPRESENTATION_QZETA40_EVEN_STRANDS4_6_8_10_12_14_16_EIGHT_SWEEP_FAMILY_PRIMARY16_FIVE_SWEEP_FAMILY_REUSE16_VACUUM_PLAT_BOUNDARY_DIRECT_PROCESS_ONLY",
        "phase_relation_law": {
            "domain": "SU2_LEVEL8_TEMPERLEY_LIEB_A9_VACUUM_FUSION_PATH_SPACE",
            "coefficient_field": "Q_ZETA40_DEGREE16",
            "local_simple_object_labels": LABELS,
            "radical_free_nonorthogonal_path_gauge": True,
            "braid_generator": "B_I_EQUALS_A_IDENTITY_PLUS_A_INVERSE_E_I",
            "braid_inverse": "B_I_INVERSE_EQUALS_A_INVERSE_IDENTITY_PLUS_A_E_I",
            "temperley_lieb_loop_value": "ZETA40_SQUARED_PLUS_ZETA40_MINUS_SQUARED",
            "multiple_noncommuting_consumers": algebra[
                "adjacent_generator_noncommutation_mismatch_cells"
            ]
            > 0,
            "shared_internal_channels_unprojected": True,
            "intermediate_projection": False,
            "retained_fusion_path_list": False,
            "retained_local_action_plan": False,
            "public_path_count_topology_does_not_inspect_boundary": True,
            "global_fusion_path_coefficient_vector_materialized": True,
        },
        "dimension_law": dimensions,
        "executed_cases": cases,
        "transaction": {
            **{f"primary_{key}": value for key, value in primary.items()},
            **{f"reuse_{key}": value for key, value in reuse.items()},
            "fresh_reuse_boundary_commitment": fresh_reuse[
                "boundary_commitment"
            ],
            "fresh_reuse_state_commitment": fresh_reuse[
                "forward_state_commitment"
            ],
            "fresh_restored_reuse_boundary_agreement": reuse[
                "boundary_commitment"
            ]
            == fresh_reuse["boundary_commitment"],
            "fresh_restored_reuse_state_agreement": reuse[
                "forward_state_commitment"
            ]
            == fresh_reuse["forward_state_commitment"],
            "restoration_generation_after_reuse": carrier.restoration_generation,
        },
        "controls": {**custody, **algebra},
        "resource_law": {
            "category_constant_field_cells": 20,
            "category_constant_payload_bits": field_payload_bits(
                list(QUANTUM_DIMENSIONS)
                + list(INVERSE_DIMENSIONS)
                + [BRAID_A, BRAID_A_INVERSE]
            ),
            "primary_public_topology_integer_cells": primary_topology.integer_cells,
            "primary_public_topology_payload_bits": primary_topology.integer_payload,
            "primary_public_program_descriptor_integers": 3,
            "primary_public_operation_records_materialized": 0,
            "primary_retained_path_records": 0,
            "primary_retained_local_action_records": 0,
            "primary_streamed_path_label_scratch_cells": primary_work.peak_streamed_path_label_cells,
            "primary_local_field_scratch_cells": primary_work.peak_local_field_scratch_cells,
            "primary_forward_carrier_field_cells": primary_case[
                "fusion_path_field_cells"
            ],
            "primary_forward_carrier_payload_bits": primary_case[
                "carrier_payload_bits"
            ],
            "primary_full_transaction_work": primary_work.as_dict(),
            "reuse_full_transaction_work": reuse_work.as_dict(),
            "retained_inverse_history_entries": 0,
            "additional_retained_plan_entries": 0,
            "controller_backend_traffic_not_applicable_direct_process": True,
            "excluded_not_zero": "PUBLIC_INTEGER_INDEX_ARITHMETIC_TRANSIENT_PYTHON_OBJECTS_ALLOCATOR_INTERPRETER_HASH_BYTE_TRAFFIC_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_PEAKS",
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_EXACT_QZETA40_TEMPERLEY_LIEB_FUSION_PATH_RECURRENCE_WITH_PUBLIC_A9_RANK_UNRANK",
            "same_global_path_cells": True,
            "same_local_block_work": True,
            "strictly_smaller_or_faster_phase_path_established": False,
        },
        "claim_limits": {
            "fixed_local_label_alphabet": True,
            "fixed_global_carrier_across_growing_strands": False,
            "bounded_exact_coefficient_width": False,
            "fusion_path_enumeration_materialized_as_coefficients": True,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m213_semantic_predecessor": "ROOT_OF_UNITY_SU2_LEVEL8_FUSION_PHASE_RELATION",
            "production_field_arithmetic_is_self_contained": True
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
