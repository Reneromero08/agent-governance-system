#!/usr/bin/env python3
"""Independent oracle for the growing-rotor pair-scattering diagnostic.

This module imports no production implementation.  It reconstructs the
exchange-symmetric global-rotation quotient, the public diagonal phase, and
the momentum-conserving pair-scattering matrix.  SciPy ``expm_multiply`` is a
numerically distinct reference for the production Chebyshev evolution.  An
independent Berlekamp--Massey implementation checks the exact scalar sequence
degrees over F103 and F137.  Sparse matrices retained here are verification
state and are not attributed to the accepted production path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


GRID = 17
PAIR_CHANNELS = 9
PRIMARY_DEPTH = 2
REUSE_DEPTH = 1
REPEATED_CYCLES = 16
CONTROL_FLOOR = 1.0e-6
NUMERICAL_TOLERANCE = 2.0e-10


Histogram = tuple[int, ...]
Control = Literal["correct", "missing", "wrong", "reordered"]


def rotate(histogram: Histogram, shift: int) -> Histogram:
    result = [0] * GRID
    for mode, count in enumerate(histogram):
        result[(mode + shift) % GRID] = count
    return tuple(result)


def canonical(histogram: Histogram) -> Histogram:
    return min(rotate(histogram, shift) for shift in range(GRID))


def generate_histograms(rotors: int) -> list[Histogram]:
    result: list[Histogram] = []
    working = [0] * GRID

    def visit(position: int, remaining: int) -> None:
        if position == GRID - 1:
            working[position] = remaining
            result.append(tuple(working))
            return
        for count in range(remaining + 1):
            working[position] = count
            visit(position + 1, remaining - count)

    visit(0, rotors)
    expected = math.comb(rotors + GRID - 1, rotors)
    if len(result) != expected:
        raise RuntimeError("independent histogram count failed")
    return result


def collision_count(histogram: Histogram) -> int:
    return sum(count * (count - 1) // 2 for count in histogram)


def labelled_weight(histogram: Histogram, rotors: int) -> int:
    denominator = math.prod(math.factorial(count) for count in histogram)
    return GRID * math.factorial(rotors) // denominator


def pair_signature(histogram: Histogram, rotors: int) -> tuple[int, ...]:
    result = [collision_count(histogram)]
    for distance in range(1, PAIR_CHANNELS):
        result.append(
            sum(
                histogram[mode] * histogram[(mode + distance) % GRID]
                for mode in range(GRID)
            )
        )
    if sum(result) != math.comb(rotors, 2):
        raise RuntimeError("independent pair-signature partition failed")
    return tuple(result)


def public_pair_weight(distance: int, step: int, program_tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % GRID % (GRID - 1)


def pair_phase_exponent(
    histogram: Histogram,
    rotors: int,
    step: int,
    program_tag: int,
) -> int:
    signature = pair_signature(histogram, rotors)
    return sum(
        signature[distance]
        * public_pair_weight(distance, step, program_tag)
        for distance in range(PAIR_CHANNELS)
    ) % GRID


def public_scattering_integer(
    signed_shift: int,
    step: int,
    program_tag: int,
) -> int:
    positive = signed_shift % GRID
    if positive == 0:
        raise RuntimeError("zero independent scattering shift")
    distance = min(positive, GRID - positive)
    magnitude = 1 + (
        (distance + 2) * (step + 1)
        + (3 * distance + 1) * (program_tag + 2)
    ) % GRID % 5
    return (
        -magnitude
        if (distance + step + program_tag) % GRID % 3 == 0
        else magnitude
    )


@dataclass(frozen=True)
class Topology:
    rotors: int
    necklaces: tuple[Histogram, ...]
    weights: np.ndarray
    collisions: np.ndarray
    lookup: dict[Histogram, int]
    full_occupation_cells: int


@dataclass(frozen=True)
class Generator:
    matrix: sparse.csr_matrix
    enumerated_terms: int
    weighted_terms: int
    unique_terms: int
    radius_bound: float
    chebyshev_tail_bound: float
    weighted_hermitian_residual: int


def compile_topology(rotors: int) -> Topology:
    if not 0 < rotors < GRID:
        raise RuntimeError("oracle only applies below the stabilizer threshold")
    occupations = generate_histograms(rotors)
    necklaces = tuple(item for item in occupations if canonical(item) == item)
    expected = len(occupations) // GRID
    if len(occupations) % GRID or len(necklaces) != expected:
        raise RuntimeError("independent free-orbit Burnside law failed")
    lookup = {item: index for index, item in enumerate(necklaces)}
    if len(lookup) != len(necklaces):
        raise RuntimeError("independent necklace lookup collision")
    weights = np.asarray(
        [labelled_weight(item, rotors) for item in necklaces],
        dtype=np.int64,
    )
    if int(weights.sum()) != GRID**rotors:
        raise RuntimeError("independent labelled-weight partition failed")
    collisions = np.asarray(
        [collision_count(item) for item in necklaces], dtype=np.int64
    )
    for item in necklaces:
        pair_signature(item, rotors)
    return Topology(
        rotors=rotors,
        necklaces=necklaces,
        weights=weights,
        collisions=collisions,
        lookup=lookup,
        full_occupation_cells=len(occupations),
    )


def compile_generator(
    topology: Topology,
    step: int,
    program_tag: int,
) -> Generator:
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[int] = []
    enumerated_terms = 0
    weighted_terms = 0
    for target, histogram in enumerate(topology.necklaces):
        for first in range(GRID):
            if histogram[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = histogram[first] * (
                    histogram[second] - (1 if first == second else 0)
                )
                if multiplicity == 0:
                    continue
                for shift in range(1, GRID):
                    source = list(histogram)
                    source[first] -= 1
                    source[second] -= 1
                    source[(first - shift) % GRID] += 1
                    source[(second + shift) % GRID] += 1
                    source_index = topology.lookup[canonical(tuple(source))]
                    row_indices.append(target)
                    column_indices.append(source_index)
                    coefficients.append(
                        multiplicity
                        * public_scattering_integer(shift, step, program_tag)
                    )
                    enumerated_terms += 1
                    weighted_terms += multiplicity
    matrix = sparse.coo_matrix(
        (
            np.asarray(coefficients, dtype=np.int64),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(column_indices, dtype=np.int64),
            ),
        ),
        shape=(len(topology.necklaces), len(topology.necklaces)),
        dtype=np.int64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    weighted = sparse.diags(
        topology.weights, dtype=np.int64, format="csr"
    ) @ matrix
    residual = weighted - weighted.transpose()
    residual.eliminate_zeros()
    maximum_residual = (
        int(np.max(np.abs(residual.data))) if residual.nnz else 0
    )
    if maximum_residual != 0:
        raise RuntimeError("independent weighted Hermiticity failed")
    absolute_shift_sum = 0.01 * sum(
        abs(public_scattering_integer(shift, step, program_tag))
        for shift in range(1, GRID)
    )
    radius_bound = (
        topology.rotors * (topology.rotors - 1) / 2 * absolute_shift_sum
    )
    first_omitted = 65
    leading = (
        (0.5 * radius_bound) ** first_omitted
        / math.gamma(first_omitted + 1)
        * math.exp(radius_bound**2 / (4 * (first_omitted + 1)))
    )
    ratio = radius_bound / (2 * (first_omitted + 1))
    tail_bound = 2 * leading / (1 - ratio)
    return Generator(
        matrix=matrix,
        enumerated_terms=enumerated_terms,
        weighted_terms=weighted_terms,
        unique_terms=int(matrix.nnz),
        radius_bound=radius_bound,
        chebyshev_tail_bound=tail_bound,
        weighted_hermitian_residual=maximum_residual,
    )


def make_carrier(topology: Topology, identity: int) -> np.ndarray:
    roots = np.exp(2j * np.pi * np.arange(GRID) / GRID)
    indices = np.arange(len(topology.necklaces), dtype=np.int64)
    exponents = (7 * indices + 3 * topology.collisions + 5 * identity) % GRID
    return roots[exponents] / math.sqrt(GRID**topology.rotors)


def weighted_norm(state: np.ndarray, topology: Topology) -> float:
    return float(np.dot(topology.weights, np.abs(state) ** 2).real)


def weighted_distance(
    left: np.ndarray,
    right: np.ndarray,
    topology: Topology,
) -> float:
    return math.sqrt(
        float(np.dot(topology.weights, np.abs(left - right) ** 2).real)
    )


def project_boundary(state: np.ndarray, topology: Topology) -> list[float]:
    return np.bincount(
        topology.collisions,
        weights=topology.weights * np.abs(state) ** 2,
        minlength=math.comb(topology.rotors, 2) + 1,
    ).astype(float).tolist()


def apply_phase(
    state: np.ndarray,
    topology: Topology,
    step: int,
    program_tag: int,
    adjoint: bool,
) -> np.ndarray:
    sign = -1 if adjoint else 1
    exponents = np.asarray(
        [
            pair_phase_exponent(item, topology.rotors, step, program_tag)
            for item in topology.necklaces
        ],
        dtype=np.int64,
    )
    roots = np.exp(2j * np.pi * np.arange(GRID) / GRID)
    return state * roots[(sign * exponents) % GRID]


def apply_scattering(
    state: np.ndarray,
    generator: Generator,
    adjoint: bool,
) -> np.ndarray:
    sign = -1j if adjoint else 1j
    scaled = generator.matrix.astype(np.float64) * (0.005 * sign)
    return np.asarray(
        expm_multiply(scaled, state, traceA=0.0), dtype=np.complex128
    )


def transaction(
    state: np.ndarray,
    expected: np.ndarray,
    topology: Topology,
    depth: int,
    program_tag: int,
    control: Control,
    generator_cache: dict[tuple[int, int], Generator] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if generator_cache is None:
        generator_cache = {}

    def generator(step: int, tag: int) -> Generator:
        key = (step, tag)
        if key not in generator_cache:
            generator_cache[key] = compile_generator(topology, step, tag)
        return generator_cache[key]

    carrier = state
    for step in range(depth):
        carrier = apply_phase(
            carrier, topology, step, program_tag, adjoint=False
        )
        carrier = apply_scattering(
            carrier,
            generator(step, program_tag),
            adjoint=False,
        )
    boundary = project_boundary(carrier, topology)
    norm_error = abs(weighted_norm(carrier, topology) - 1.0)
    minimum_step = 1 if control == "missing" else 0
    for step in range(depth - 1, minimum_step - 1, -1):
        inverse_tag = (
            program_tag + 1
            if control == "wrong" and step == depth - 1
            else program_tag
        )
        if control == "reordered":
            carrier = apply_phase(
                carrier, topology, step, inverse_tag, adjoint=True
            )
        carrier = apply_scattering(
            carrier,
            generator(step, inverse_tag),
            adjoint=True,
        )
        if control != "reordered":
            carrier = apply_phase(
                carrier, topology, step, inverse_tag, adjoint=True
            )
    return carrier, {
        "boundary": boundary,
        "norm_error": norm_error,
        "restoration_error": weighted_distance(carrier, expected, topology),
    }


def field_power(base: int, exponent: int, prime: int) -> int:
    return pow(base % prime, exponent, prime)


def primitive_seventeenth_root(prime: int) -> int:
    if (prime - 1) % GRID:
        raise RuntimeError("oracle prime lacks a seventeenth root")
    for generator in range(2, prime):
        root = field_power(generator, (prime - 1) // GRID, prime)
        if root != 1 and field_power(root, GRID, prime) == 1:
            return root
    raise RuntimeError("oracle seventeenth-root search failed")


def berlekamp_massey(sequence: list[int], prime: int) -> int:
    connection = [0] * (len(sequence) + 1)
    previous = [0] * (len(sequence) + 1)
    connection[0] = 1
    previous[0] = 1
    degree = 0
    displacement = 1
    prior_discrepancy = 1
    for position in range(len(sequence)):
        discrepancy = sequence[position]
        for offset in range(1, degree + 1):
            discrepancy += connection[offset] * sequence[position - offset]
        discrepancy %= prime
        if discrepancy == 0:
            displacement += 1
            continue
        saved = connection.copy()
        scale = discrepancy * pow(prior_discrepancy, prime - 2, prime) % prime
        for offset in range(len(sequence) + 1 - displacement):
            connection[offset + displacement] = (
                connection[offset + displacement] - scale * previous[offset]
            ) % prime
        if 2 * degree <= position:
            degree = position + 1 - degree
            previous = saved
            prior_discrepancy = discrepancy
            displacement = 1
        else:
            displacement += 1
    for position in range(degree, len(sequence)):
        residual = sum(
            connection[offset] * sequence[position - offset]
            for offset in range(degree + 1)
        ) % prime
        if residual:
            raise RuntimeError("independent recurrence residual is nonzero")
    return degree


def scalar_krylov_degree(
    topology: Topology,
    generator: Generator,
    prime: int,
) -> tuple[int, int]:
    root = primitive_seventeenth_root(prime)
    dimension = len(topology.necklaces)
    indices = np.arange(dimension, dtype=np.int64)
    state = np.asarray(
        [
            field_power(
                root,
                (7 * index + 3 * int(topology.collisions[index])) % GRID,
                prime,
            )
            for index in range(dimension)
        ],
        dtype=np.int64,
    )
    probe = np.asarray(
        [
            field_power(
                root,
                (
                    11 * index
                    + 5 * int(topology.collisions[index])
                    + 1
                )
                % GRID,
                prime,
            )
            for index in range(dimension)
        ],
        dtype=np.int64,
    )
    diagonal = np.asarray(
        [
            field_power(
                root,
                pair_phase_exponent(item, topology.rotors, 0, 0),
                prime,
            )
            for item in topology.necklaces
        ],
        dtype=np.int64,
    )
    integer_matrix = generator.matrix.astype(np.int64)
    sequence: list[int] = []
    for _ in range(2 * dimension + 2):
        sequence.append(int(np.dot(probe, state) % prime))
        phased = diagonal * state % prime
        state = np.asarray(integer_matrix @ phased, dtype=np.int64) % prime
    return berlekamp_massey(sequence, prime), root


def maximum_boundary_error(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def main() -> None:
    cases: list[dict[str, object]] = []
    for rotors in range(2, 6):
        topology = compile_topology(rotors)
        base_generator = compile_generator(topology, 0, 0)
        generator_cache = {(0, 0): base_generator}
        initial = make_carrier(topology, 0)
        restored, primary = transaction(
            initial.copy(),
            initial,
            topology,
            PRIMARY_DEPTH,
            0,
            "correct",
            generator_cache,
        )
        reused, reuse = transaction(
            restored,
            initial,
            topology,
            REUSE_DEPTH,
            3,
            "correct",
            generator_cache,
        )
        del reused
        _, fresh_reuse = transaction(
            initial.copy(),
            initial,
            topology,
            REUSE_DEPTH,
            3,
            "correct",
            generator_cache,
        )
        reuse_boundary_error = maximum_boundary_error(
            reuse["boundary"], fresh_reuse["boundary"]  # type: ignore[arg-type]
        )
        degree_103, root_103 = scalar_krylov_degree(
            topology, base_generator, 103
        )
        degree_137, root_137 = scalar_krylov_degree(
            topology, base_generator, 137
        )
        if (
            primary["restoration_error"] > NUMERICAL_TOLERANCE
            or reuse["restoration_error"] > NUMERICAL_TOLERANCE
            or primary["norm_error"] > NUMERICAL_TOLERANCE
            or reuse_boundary_error > NUMERICAL_TOLERANCE
        ):
            raise RuntimeError("independent growing-rotor numerical gate failed")
        cases.append(
            {
                "rotors": rotors,
                "full_occupation_cells": topology.full_occupation_cells,
                "necklace_cells": len(topology.necklaces),
                "labelled_cells": GRID**rotors,
                "enumerated_generator_terms": base_generator.enumerated_terms,
                "weighted_particle_pair_shift_terms": base_generator.weighted_terms,
                "unique_nonzero_generator_terms": base_generator.unique_terms,
                "weighted_hermitian_residual": base_generator.weighted_hermitian_residual,
                "radius_bound": base_generator.radius_bound,
                "chebyshev_tail_bound": base_generator.chebyshev_tail_bound,
                "f103_primitive_root": root_103,
                "f137_primitive_root": root_137,
                "f103_krylov_degree": degree_103,
                "f137_krylov_degree": degree_137,
                "f103_dimension_deficit": len(topology.necklaces) - degree_103,
                "f137_dimension_deficit": len(topology.necklaces) - degree_137,
                "primary_boundary": primary["boundary"],
                "primary_restoration_error": primary["restoration_error"],
                "reuse_restoration_error": reuse["restoration_error"],
                "fresh_restored_reuse_boundary_error": reuse_boundary_error,
            }
        )

    topology = compile_topology(5)
    generator_cache: dict[tuple[int, int], Generator] = {}
    initial = make_carrier(topology, 0)
    controls: dict[str, float] = {}
    for name in ("missing", "wrong", "reordered"):
        _, result = transaction(
            initial.copy(),
            initial,
            topology,
            PRIMARY_DEPTH,
            0,
            name,  # type: ignore[arg-type]
            generator_cache,
        )
        controls[f"{name}_inverse_error"] = float(result["restoration_error"])
    repeated = initial.copy()
    repeated_max = 0.0
    for generation in range(REPEATED_CYCLES):
        repeated, result = transaction(
            repeated,
            initial,
            topology,
            1,
            2 + generation % 2,
            "correct",
            generator_cache,
        )
        repeated_max = max(repeated_max, float(result["restoration_error"]))
    if (
        min(controls.values()) < CONTROL_FLOOR
        or repeated_max > NUMERICAL_TOLERANCE
    ):
        raise RuntimeError("independent growing-rotor controls failed")

    print(
        json.dumps(
            {
                "oracle": "SEPARATE_PYTHON_NECKLACE_SPARSE_EXPM_AND_EXACT_SCALAR_BERLEKAMP_MASSEY_REEXECUTION",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
                "result": "PASS",
                "cases": cases,
                "controls": {
                    **controls,
                    "repeated_reuse_cycles": REPEATED_CYCLES,
                    "repeated_reuse_max_error": repeated_max,
                },
                "production_source_imported": False,
                "production_projection_called": False,
                "production_inverse_called": False,
                "oracle_sparse_generators_are_verification_only": True,
                "stable_transferable_recurrence_quotient_established": False,
                "matched_classical_recurrence": "IDENTICAL_GROWING_NECKLACE_PAIR_SCATTERING_AND_DIAGONAL_PHASE_RECURRENCE",
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "catvm_custody": False,
                "physical_waveform_execution": False,
                "physical_bit_replacement": False,
                "unbounded_computation_established": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
