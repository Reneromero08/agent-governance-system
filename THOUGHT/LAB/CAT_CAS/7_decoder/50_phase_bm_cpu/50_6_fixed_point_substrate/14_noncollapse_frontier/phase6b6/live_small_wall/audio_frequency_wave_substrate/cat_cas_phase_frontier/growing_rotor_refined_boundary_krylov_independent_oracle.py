#!/usr/bin/env python3
"""Independent exact oracle for the Rotor-6 refined boundary recurrence.

The oracle imports no CAT_CAS production module.  It reconstructs occupation
histograms, cyclic and reflection quotients, pair and triangle signatures,
the public F103 operator, source, probe, scalar sequence, and Berlekamp-Massey
certificates.  Its retained SciPy matrix is verification-only state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy import sparse


GRID = 17
ROTORS = 6
PRIME = 103
ROOT = 72
STEP = 0
PROGRAM_TAG = 0
SOURCE_FAMILY = 0
PAIR_CHANNELS = 9
TRIANGLE_STENCILS = (((1, 3), (2, 3)), ((1, 5), (4, 5)))
TRAINING_TERMS = 4554
HOLDOUT_TERMS = 64
RAW_TRANSITIONS = 684624
Histogram = tuple[int, ...]
Signature = tuple[int, ...]


def integer_commitment(values: list[int] | tuple[int, ...] | np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{int(value)},".encode())
    return digest.hexdigest()


def state_commitment(values: np.ndarray) -> str:
    return hashlib.sha256(",".join(map(str, values.tolist())).encode()).hexdigest()


def rotate(item: Histogram, amount: int) -> Histogram:
    result = [0] * GRID
    for mode, count in enumerate(item):
        result[(mode + amount) % GRID] = count
    return tuple(result)


def cyclic_key(item: Histogram) -> Histogram:
    return min(rotate(item, amount) for amount in range(GRID))


def reflection(item: Histogram) -> Histogram:
    return tuple(item[(-mode) % GRID] for mode in range(GRID))


def bracelet_key(item: Histogram) -> Histogram:
    return min(cyclic_key(item), cyclic_key(reflection(item)))


def histograms(rotors: int) -> Iterator[Histogram]:
    working = [0] * GRID

    def visit(position: int, remaining: int) -> Iterator[Histogram]:
        if position == GRID - 1:
            working[position] = remaining
            yield tuple(working)
            return
        for count in range(remaining + 1):
            working[position] = count
            yield from visit(position + 1, remaining - count)

    yield from visit(0, rotors)


def pair_signature(item: Histogram, rotors: int) -> tuple[int, ...]:
    values = [sum(count * (count - 1) // 2 for count in item)]
    for distance in range(1, PAIR_CHANNELS):
        values.append(
            sum(
                item[mode] * item[(mode + distance) % GRID]
                for mode in range(GRID)
            )
        )
    if sum(values) != math.comb(rotors, 2):
        raise RuntimeError("independent pair-signature partition changed")
    return tuple(values)


def triangle_counts(item: Histogram) -> tuple[int, int]:
    return tuple(
        sum(
            item[anchor]
            * item[(anchor + first) % GRID]
            * item[(anchor + second) % GRID]
            for first, second in orientations
            for anchor in range(GRID)
        )
        for orientations in TRIANGLE_STENCILS
    )  # type: ignore[return-value]


def refined_signature(item: Histogram, rotors: int) -> Signature:
    return pair_signature(item, rotors) + triangle_counts(item)


def pair_weight(distance: int, step: int, tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * tag
    ) % GRID % (GRID - 1)


def phase_exponent(signature: Signature, step: int, tag: int) -> int:
    return sum(
        count * pair_weight(distance, step, tag)
        for distance, count in enumerate(signature[:PAIR_CHANNELS])
    ) % GRID


def scattering_integer(shift: int, step: int, tag: int) -> int:
    distance = min(shift % GRID, GRID - shift % GRID)
    magnitude = 1 + (
        (distance + 2) * (step + 1) + (3 * distance + 1) * (tag + 2)
    ) % GRID % 5
    return -magnitude if (distance + step + tag) % GRID % 3 == 0 else magnitude


@dataclass(frozen=True)
class Topology:
    signatures: tuple[Signature, ...]
    representatives: tuple[Histogram, ...]
    boundary_weights: tuple[int, ...]
    occupations: int
    necklaces: int
    bracelets: int


def compile_topology() -> Topology:
    representatives: dict[Signature, Histogram] = {}
    boundary: dict[Signature, int] = {}
    bracelets: set[Histogram] = set()
    occupation_count = 0
    necklace_count = 0
    for item in histograms(ROTORS):
        occupation_count += 1
        if cyclic_key(item) != item:
            continue
        signature = refined_signature(item, ROTORS)
        representatives.setdefault(signature, item)
        boundary[signature] = (
            boundary.get(signature, 0)
            + pow(ROOT, (11 * necklace_count + 5 * signature[0] + 1) % GRID, PRIME)
        ) % PRIME
        bracelets.add(bracelet_key(item))
        necklace_count += 1
    signatures = tuple(sorted(representatives))
    topology = Topology(
        signatures=signatures,
        representatives=tuple(representatives[value] for value in signatures),
        boundary_weights=tuple(boundary[value] for value in signatures),
        occupations=occupation_count,
        necklaces=necklace_count,
        bracelets=len(bracelets),
    )
    if (
        topology.occupations != math.comb(ROTORS + GRID - 1, ROTORS)
        or topology.necklaces != 4389
        or topology.bracelets != 2277
        or len(topology.signatures) != 2277
    ):
        raise RuntimeError("independent Rotor-6 quotient law changed")
    return topology


def topology_commitment(topology: Topology) -> str:
    digest = hashlib.sha256()
    for signature, representative, weight in zip(
        topology.signatures,
        topology.representatives,
        topology.boundary_weights,
        strict=True,
    ):
        digest.update(
            (
                ",".join(map(str, signature))
                + "|"
                + ",".join(map(str, representative))
                + f"|{weight};"
            ).encode()
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class ExactOperator:
    matrix: sparse.csr_matrix
    diagonal: np.ndarray
    raw_transition_commitment: str
    csr_commitment: str


def compile_operator(topology: Topology) -> ExactOperator:
    lookup = {value: index for index, value in enumerate(topology.signatures)}
    rows = np.empty(RAW_TRANSITIONS, dtype=np.int32)
    columns = np.empty(RAW_TRANSITIONS, dtype=np.int32)
    coefficients = np.empty(RAW_TRANSITIONS, dtype=np.int64)
    digest = hashlib.sha256()
    cursor = 0
    for target, item in enumerate(topology.representatives):
        for first in range(GRID):
            if item[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = item[first] * (
                    item[second] - (1 if first == second else 0)
                )
                if multiplicity == 0:
                    continue
                for shift in range(1, GRID):
                    moved = list(item)
                    moved[first] -= 1
                    moved[second] -= 1
                    moved[(first - shift) % GRID] += 1
                    moved[(second + shift) % GRID] += 1
                    source = lookup[refined_signature(tuple(moved), ROTORS)]
                    rows[cursor] = target
                    columns[cursor] = source
                    coefficients[cursor] = multiplicity * scattering_integer(
                        shift, STEP, PROGRAM_TAG
                    )
                    digest.update(
                        f"{target},{first},{second},{shift},{source},{multiplicity};".encode()
                    )
                    cursor += 1
    if cursor != RAW_TRANSITIONS:
        raise RuntimeError("independent raw transition count changed")
    dimension = len(topology.signatures)
    matrix = sparse.coo_matrix(
        (coefficients, (rows, columns)),
        shape=(dimension, dimension),
        dtype=np.int64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    diagonal = np.asarray(
        [
            pow(ROOT, phase_exponent(signature, STEP, PROGRAM_TAG), PRIME)
            for signature in topology.signatures
        ],
        dtype=np.int64,
    )
    csr_digest = hashlib.sha256()
    for values in (matrix.indptr, matrix.indices, matrix.data):
        csr_digest.update(integer_commitment(values).encode())
    return ExactOperator(
        matrix=matrix,
        diagonal=diagonal,
        raw_transition_commitment=digest.hexdigest(),
        csr_commitment=csr_digest.hexdigest(),
    )


def source_state(signatures: tuple[Signature, ...]) -> np.ndarray:
    return np.asarray(
        [
            (
                1
                + (SOURCE_FAMILY + 3) * (index + 1)
                + sum(
                    (coordinate + 2 + SOURCE_FAMILY) * (count + 1) ** 2
                    for coordinate, count in enumerate(signature)
                )
            )
            % PRIME
            for index, signature in enumerate(signatures)
        ],
        dtype=np.int64,
    )


def apply_operator(state: np.ndarray, operator: ExactOperator) -> np.ndarray:
    if state.ndim != 1 or state.shape[0] != operator.matrix.shape[0]:
        raise ValueError("independent null state")
    return np.asarray(
        operator.matrix.dot(state * operator.diagonal % PRIME) % PRIME,
        dtype=np.int64,
    )


def scalar_sequence(
    topology: Topology, operator: ExactOperator
) -> tuple[list[int], np.ndarray]:
    state = source_state(topology.signatures)
    probe = np.asarray(topology.boundary_weights, dtype=np.int64)
    samples: list[int] = []
    first = np.empty(0, dtype=np.int64)
    for index in range(TRAINING_TERMS + HOLDOUT_TERMS + 1):
        samples.append(int(np.dot(probe, state) % PRIME))
        state = apply_operator(state, operator)
        if index == 0:
            first = state.copy()
    return samples, first


@dataclass(frozen=True)
class Recurrence:
    degree: int
    polynomial: tuple[int, ...]


def minimal_recurrence(samples: list[int]) -> Recurrence:
    polynomial = [1]
    last = [1]
    span = 0
    shift = 1
    last_delta = 1
    for position in range(len(samples)):
        delta = samples[position]
        for lag in range(1, span + 1):
            delta += polynomial[lag] * samples[position - lag]
        delta %= PRIME
        if delta == 0:
            shift += 1
            continue
        old = polynomial.copy()
        ratio = delta * pow(last_delta, PRIME - 2, PRIME) % PRIME
        if len(polynomial) < len(last) + shift:
            polynomial.extend([0] * (len(last) + shift - len(polynomial)))
        for index, value in enumerate(last):
            polynomial[index + shift] = (
                polynomial[index + shift] - ratio * value
            ) % PRIME
        if 2 * span <= position:
            span = position + 1 - span
            last = old
            last_delta = delta
            shift = 1
        else:
            shift += 1
    return Recurrence(span, tuple(polynomial))


def violation_counts(
    sequence: list[int], offset: int, training: int, recurrence: Recurrence
) -> tuple[int, int]:
    shifted = sequence[offset:]
    inside = 0
    outside = 0
    for position in range(recurrence.degree, len(shifted)):
        delta = shifted[position]
        for lag in range(1, recurrence.degree + 1):
            delta += recurrence.polynomial[lag] * shifted[position - lag]
        if delta % PRIME:
            if position < training:
                inside += 1
            else:
                outside += 1
    return inside, outside


def main() -> None:
    if pow(ROOT, GRID, PRIME) != 1 or ROOT == 1:
        raise RuntimeError("independent seventeenth-root law changed")
    topology = compile_topology()
    operator = compile_operator(topology)
    sequence, first = scalar_sequence(topology, operator)
    k0 = minimal_recurrence(sequence[:TRAINING_TERMS])
    k1 = minimal_recurrence(sequence[1 : 1 + TRAINING_TERMS])
    k0_train, k0_holdout = violation_counts(sequence, 0, TRAINING_TERMS, k0)
    k1_train, k1_holdout = violation_counts(sequence, 1, TRAINING_TERMS, k1)
    if (
        operator.raw_transition_commitment
        != "1774f44283dde49154d3aec11bb0dd22179a061290e44448c795402fa2ca0465"
        or operator.matrix.nnz != 172838
        or sequence[1] != 83
        or state_commitment(first)
        != "834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e"
        or k0.degree != 2261
        or k1.degree != 2260
        or any((k0_train, k0_holdout, k1_train, k1_holdout))
    ):
        raise RuntimeError("independent recurrence certificate changed")
    print(
        json.dumps(
            {
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "NO_RESTORATION_CLAIM",
                "result": "PASS_DIAGNOSTIC_NO_COMPACT_RECURRENCE",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_AND_REFLECTION_INVARIANT_TWO_TRIANGLE_REFINED_SIGNATURE_ROTOR6_F103_ROOT72_REPEATED_STEP0_TAG0_SOURCE_FAMILY0_PUBLIC_BOUNDARY_DIRECT_PROCESS_DIAGNOSTIC_ONLY",
                "topology": {
                    "occupation_histograms": topology.occupations,
                    "necklace_cells": topology.necklaces,
                    "bracelet_and_refined_signature_cells": len(
                        topology.signatures
                    ),
                    "topology_commitment": topology_commitment(topology),
                },
                "operator": {
                    "raw_transition_terms": RAW_TRANSITIONS,
                    "raw_transition_commitment": operator.raw_transition_commitment,
                    "aggregated_csr_nonzeros": int(operator.matrix.nnz),
                    "aggregated_csr_commitment": operator.csr_commitment,
                },
                "exact_boundary_recurrence": {
                    "training_terms": TRAINING_TERMS,
                    "holdout_terms": HOLDOUT_TERMS,
                    "sequence_commitment": integer_commitment(sequence),
                    "sequence_prefix": sequence[:8],
                    "first_public_word_boundary": sequence[1],
                    "first_public_word_state_commitment": state_commitment(first),
                    "k0_degree": k0.degree,
                    "k0_connection_slots": len(k0.polynomial),
                    "k0_nonzero_coefficients": sum(
                        value != 0 for value in k0.polynomial
                    ),
                    "k0_last_coefficient": k0.polynomial[-1],
                    "k0_connection_commitment": integer_commitment(k0.polynomial),
                    "k0_training_violations": k0_train,
                    "k0_holdout_violations": k0_holdout,
                    "k1_degree": k1.degree,
                    "k1_dynamic_cell_saving": len(topology.signatures) - k1.degree,
                    "k1_connection_slots": len(k1.polynomial),
                    "k1_nonzero_coefficients": sum(
                        value != 0 for value in k1.polynomial
                    ),
                    "k1_last_coefficient": k1.polynomial[-1],
                    "k1_connection_commitment": integer_commitment(k1.polynomial),
                    "k1_training_violations": k1_train,
                    "k1_holdout_violations": k1_holdout,
                },
                "production_source_imported": False,
                "production_projection_called": False,
                "production_bm_called": False,
                "oracle_sparse_operator_is_verification_only": True,
                "recurrence_does_not_reconstruct_internal_phase_state": True,
                "fixed_rank_or_depth_independent_recurrence_established": False,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "terminal": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
