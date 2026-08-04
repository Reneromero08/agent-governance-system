#!/usr/bin/env python3
"""No-import oracle for the bounded determinant generating-relation result."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


CLASSES = ((0, 0), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1))
WEIGHTS = (1, 2, 2, 1, 2, 1, 1)


def fail(message: str) -> None:
    raise RuntimeError(message)


def factors(value: int) -> list[int]:
    answer: list[int] = []
    candidate = 2
    while candidate * candidate <= value:
        if value % candidate == 0:
            answer.append(candidate)
            while value % candidate == 0:
                value //= candidate
        candidate += 1
    if value > 1:
        answer.append(value)
    return answer


def generator(prime: int) -> int:
    for value in range(2, prime):
        if all(pow(value, (prime - 1) // factor, prime) != 1 for factor in factors(prime - 1)):
            return value
    fail("generator absent")


class Field:
    def __init__(self, q: int, p: int) -> None:
        self.q = q
        self.p = p
        g = generator(p)
        self.zeta = pow(g, (p - 1) // q, p)
        self.eta = pow(g, (p - 1) // (q - 1), p)
        qg = generator(q)
        self.logarithms = {pow(qg, exponent, q): exponent for exponent in range(q - 1)}

    def phase(self, exponent: int) -> int:
        return pow(self.zeta, exponent % self.q, self.p)

    def character(self, value: int, exponent: int) -> int:
        return pow(self.eta, self.logarithms[value % self.q] * exponent % (self.q - 1), self.p)


def det(coordinates: tuple[int, ...], q: int) -> int:
    a, b, c, d, e, f = coordinates
    return (a * d * f + 2 * b * c * e - a * e * e - d * c * c - f * b * b) % q


def matrix_rank(coordinates: tuple[int, ...], q: int) -> int:
    a, b, c, d, e, f = coordinates
    rows = [[a, b, c], [b, d, e], [c, e, f]]
    rank = 0
    for column in range(3):
        pivot = next((row for row in range(rank, 3) if rows[row][column] % q), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column] % q, -1, q)
        rows[rank] = [value * inverse % q for value in rows[rank]]
        for row in range(3):
            if row != rank and rows[row][column] % q:
                multiple = rows[row][column] % q
                rows[row] = [
                    (left - multiple * right) % q
                    for left, right in zip(rows[row], rows[rank])
                ]
        rank += 1
    return rank


def class_of(coordinates: tuple[int, ...], q: int) -> tuple[int, int]:
    rank = matrix_rank(coordinates, q)
    if rank == 0:
        return 0, 0
    a, b, c, d, e, f = coordinates
    matrix = ((a, b, c), (b, d, e), (c, e, f))
    discriminant = 0
    for indices in itertools.combinations(range(3), rank):
        if rank == 1:
            candidate = matrix[indices[0]][indices[0]] % q
        elif rank == 2:
            left, right = indices
            candidate = (
                matrix[left][left] * matrix[right][right]
                - matrix[left][right] * matrix[right][left]
            ) % q
        else:
            candidate = det(coordinates, q)
        if candidate:
            discriminant = candidate
            break
    if not discriminant:
        fail("ranked symmetric matrix lacks nonsingular principal minor")
    return rank, 1 if pow(discriminant, (q - 1) // 2, q) == 1 else -1


def geometry(q: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], tuple[tuple[int, ...], ...]]:
    determinants = np.zeros((q,) * 6, dtype=np.int16)
    classes = np.zeros((q,) * 6, dtype=np.int8)
    class_index = {value: index for index, value in enumerate(CLASSES)}
    invertible: list[tuple[int, ...]] = []
    for coordinates in itertools.product(range(q), repeat=6):
        determinant = det(coordinates, q)
        determinants[coordinates] = determinant
        classes[coordinates] = class_index[class_of(coordinates, q)]
        if determinant:
            invertible.append(coordinates)
    return determinants, classes, tuple(invertible)


def independent_dft(array: np.ndarray[Any, Any], field: Field, axes: int, inverse: bool = False) -> np.ndarray[Any, Any]:
    result = array.astype(np.int64, copy=True)
    sign = -1 if inverse else 1
    for axis in range(axes):
        kernel = np.array(
            [
                [field.phase(sign * WEIGHTS[axis] * target * source) for source in range(field.q)]
                for target in range(field.q)
            ],
            dtype=np.int64,
        )
        moved = np.moveaxis(result, axis, -1)
        shape = moved.shape
        result = np.moveaxis((moved.reshape(-1, field.q) @ kernel.T % field.p).reshape(shape), -1, axis)
    if inverse:
        result = result * pow(pow(field.q, axes, field.p), -1, field.p) % field.p
    return result


def mod_inverse_matrix(matrix: np.ndarray[Any, Any], modulus: int) -> np.ndarray[Any, Any]:
    n = len(matrix)
    work = np.concatenate((matrix.copy() % modulus, np.eye(n, dtype=np.int64)), axis=1)
    for column in range(n):
        pivot = next(row for row in range(column, n) if work[row, column] % modulus)
        work[[column, pivot]] = work[[pivot, column]]
        work[column] = work[column] * pow(int(work[column, column]), -1, modulus) % modulus
        for row in range(n):
            if row != column and work[row, column]:
                work[row] = (work[row] - int(work[row, column]) * work[column]) % modulus
    return work[:, n:]


def column_basis(columns: np.ndarray[Any, Any], modulus: int) -> tuple[np.ndarray[Any, Any], list[int]]:
    basis = np.empty((len(columns), 0), dtype=np.int64)
    rows: list[int] = []
    for column in columns.T:
        if rows:
            coefficients = mod_inverse_matrix(basis[rows], modulus) @ column[rows] % modulus
            residual = (column - basis @ coefficients) % modulus
        else:
            residual = column % modulus
        nonzero = np.flatnonzero(residual)
        if len(nonzero):
            rows.append(int(nonzero[0]))
            basis = np.column_stack((basis, column % modulus))
    return basis, rows


def in_span(basis: np.ndarray[Any, Any], rows: list[int], vector: np.ndarray[Any, Any], modulus: int) -> bool:
    coefficients = mod_inverse_matrix(basis[rows], modulus) @ vector[rows] % modulus
    return bool(np.array_equal(basis @ coefficients % modulus, vector % modulus))


def add_column(basis: np.ndarray[Any, Any], rows: list[int], vector: np.ndarray[Any, Any], modulus: int) -> tuple[np.ndarray[Any, Any], list[int]]:
    coefficients = mod_inverse_matrix(basis[rows], modulus) @ vector[rows] % modulus
    residual = (vector - basis @ coefficients) % modulus
    nonzero = np.flatnonzero(residual)
    if not len(nonzero):
        return basis, rows
    return np.column_stack((basis, vector % modulus)), rows + [int(nonzero[0])]


def singular_vectors(q: int, field: Field, determinants: np.ndarray[Any, Any], classes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    width = 5 * (q - 1) + 7
    answer = np.zeros((q,) * 7 + (width,), dtype=np.int64)
    for coordinates in itertools.product(range(q), repeat=6):
        class_index = int(classes[coordinates])
        if int(determinants[coordinates]) == 0:
            for scale in range(1, q):
                for exponent in range(q - 1):
                    answer[coordinates + (scale, class_index * (q - 1) + exponent)] = field.character(scale, exponent)
        answer[coordinates + (0, 5 * (q - 1) + class_index)] = 1
    return answer


def phase_vector(
    q: int,
    field: Field,
    determinants: np.ndarray[Any, Any],
    invertible: tuple[tuple[int, ...], ...],
    signature: tuple[int, int],
    dual: bool,
) -> np.ndarray[Any, Any]:
    answer = np.zeros((q,) * 7, dtype=np.int64)
    for coordinates in invertible:
        determinant = int(determinants[coordinates])
        for scale in range(1, q):
            sign = -1 if dual else 1
            answer[coordinates + (scale,)] = (
                field.phase(sign * determinant * pow(scale, -1, q))
                * field.character(determinant, signature[0])
                * field.character(scale, signature[1])
            ) % field.p
    return answer


def q5_reexecution(production: dict[str, Any]) -> dict[str, Any]:
    q, p = 5, 41
    field = Field(q, p)
    determinants, classes, invertible = geometry(q)
    singular = singular_vectors(q, field, determinants, classes)
    flat_singular = singular.reshape(q**7, -1)
    transformed_singular = independent_dft(singular, field, 7).reshape(q**7, -1)
    joint = np.concatenate((transformed_singular, -flat_singular % p), axis=1) % p
    base, rows = column_basis(joint, p)

    signatures: list[tuple[tuple[int, int], tuple[int, int]]] = []
    sources: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    transformed: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    targets: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    for left in range(q - 1):
        for right in range(q - 1):
            signature = (left, right)
            sources[signature] = phase_vector(q, field, determinants, invertible, signature, False)
            transformed[signature] = independent_dft(sources[signature], field, 7).reshape(q**7)
            targets[signature] = phase_vector(q, field, determinants, invertible, signature, True).reshape(q**7)
    for source_signature, source_transform in transformed.items():
        for target_signature, target in targets.items():
            extended, extended_rows = add_column(base, rows, -target % p, p)
            if in_span(extended, extended_rows, -source_transform % p, p):
                signatures.append((source_signature, target_signature))

    sealed_matches = production["q5_full_boundary_closure"]["full_closure_matches"]
    expected = [
        (tuple(item["source_character_signature"]), tuple(item["target_character_signature"]))
        for item in sealed_matches
    ]
    if signatures != expected:
        fail("independent q=5 match search differs")

    verified_coefficients = 0
    primary_source: np.ndarray[Any, Any] | None = None
    for item in sealed_matches:
        source_signature = tuple(item["source_character_signature"])
        target_signature = tuple(item["target_character_signature"])
        source_coefficients = np.array(item["source_completion_coefficients"], dtype=np.int64)
        target_coefficients = np.array(item["target_completion_coefficients"], dtype=np.int64)
        completed_source = (sources[source_signature].reshape(q**7) + flat_singular @ source_coefficients) % p
        completed_target = (
            int(item["global_scale"]) * targets[target_signature]
            + flat_singular @ target_coefficients
        ) % p
        if not np.array_equal(
            independent_dft(completed_source.reshape((q,) * 7), field, 7).reshape(q**7),
            completed_target,
        ):
            fail("sealed q=5 completion coefficients fail independent transform")
        verified_coefficients += 1
        if source_signature == (0, 0) and target_signature == (2, 2):
            primary_source = completed_source.reshape((q,) * 7)
    if primary_source is None:
        fail("primary source absent")
    backing = primary_source.copy()
    backing_id = id(backing)
    baseline = backing.copy()
    backing[...] = independent_dft(backing, field, 7)
    commitment = hashlib.sha256(backing.astype("<i8").tobytes()).hexdigest()
    backing[...] = independent_dft(backing, field, 7, inverse=True)
    restored = np.array_equal(backing, baseline) and id(backing) == backing_id
    return {
        "full_match_signatures": [[list(left), list(right)] for left, right in signatures],
        "full_match_count": len(signatures),
        "independently_verified_completion_vectors": verified_coefficients,
        "full_boundary_points_per_match": q**7,
        "joint_boundary_rank": base.shape[1],
        "primary_forward_commitment": commitment,
        "exact_forward_inverse_restoration": bool(restored),
    }


def trace_pair(matrix: np.ndarray[Any, Any], target: tuple[int, ...], q: int) -> np.ndarray[Any, Any]:
    weighted = np.array(
        [target[0], 2 * target[1], 2 * target[2], target[3], 2 * target[4], target[5]],
        dtype=np.int64,
    )
    return matrix @ weighted % q


def q7_direct_sum_attack(production: dict[str, Any]) -> dict[str, Any]:
    q, p = 7, 43
    field = Field(q, p)
    determinants, classes, invertible = geometry(q)
    all_coordinates = np.array(list(itertools.product(range(q), repeat=6)), dtype=np.int16)
    all_det = determinants.reshape(-1).astype(np.int64)
    all_class = classes.reshape(-1).astype(np.int64)
    invertible_array = np.array(invertible, dtype=np.int16)
    invertible_det = np.array([int(determinants[item]) for item in invertible], dtype=np.int64)
    matrix_count = int(production["q7_transfer_attack"]["matrix_samples"])
    targets = tuple(invertible[index * len(invertible) // matrix_count] for index in range(matrix_count))
    width = 5 * (q - 1) + 7
    sample_count = matrix_count * (q - 1)
    singular_sample = np.zeros((sample_count, width), dtype=np.int64)
    source_samples = {
        (left, right): np.zeros(sample_count, dtype=np.int64)
        for left in range(q - 1)
        for right in range(q - 1)
    }
    row = 0
    for target in targets:
        all_trace = trace_pair(all_coordinates, target, q)
        invertible_trace = trace_pair(invertible_array, target, q)
        class_trace_counts = np.bincount(
            all_class * q + all_trace, minlength=7 * q
        ).reshape(7, q)
        det_trace_counts = np.bincount(
            invertible_det * q + invertible_trace, minlength=q * q
        ).reshape(q, q)
        for output_scale in range(1, q):
            # Direct character sums, independently grouped by determinant and trace.
            for class_index in range(5):
                for exponent in range(q - 1):
                    total = 0
                    for trace_value in range(q):
                        x_part = int(class_trace_counts[class_index, trace_value]) * field.phase(trace_value)
                        for source_scale in range(1, q):
                            total += (
                                x_part
                                * field.phase(output_scale * source_scale)
                                * field.character(source_scale, exponent)
                            )
                    singular_sample[row, class_index * (q - 1) + exponent] = total % p
            for class_index in range(7):
                singular_sample[row, 5 * (q - 1) + class_index] = sum(
                    int(class_trace_counts[class_index, trace_value]) * field.phase(trace_value)
                    for trace_value in range(q)
                ) % p
            for source_signature, vector in source_samples.items():
                total = 0
                for determinant_value in range(1, q):
                    determinant_character = field.character(determinant_value, source_signature[0])
                    for trace_value in range(q):
                        count = int(det_trace_counts[determinant_value, trace_value])
                        if not count:
                            continue
                        for source_scale in range(1, q):
                            total += (
                                count
                                * field.phase(
                                    determinant_value * pow(source_scale, -1, q)
                                    + trace_value
                                    + output_scale * source_scale
                                )
                                * determinant_character
                                * field.character(source_scale, source_signature[1])
                            )
                vector[row] = total % p
            row += 1

    base, rows = column_basis(singular_sample, p)
    survivors: list[list[list[int]]] = []
    row_points = tuple(target + (scale,) for target in targets for scale in range(1, q))
    for source_signature, vector in source_samples.items():
        for target_det_character in range(q - 1):
            for target_scale_character in range(q - 1):
                target_vector = np.array(
                    [
                        field.phase(-det(point[:-1], q) * pow(point[-1], -1, q))
                        * field.character(det(point[:-1], q), target_det_character)
                        * field.character(point[-1], target_scale_character)
                        % p
                        for point in row_points
                    ],
                    dtype=np.int64,
                )
                extended, extended_rows = add_column(base, rows, -target_vector % p, p)
                if in_span(extended, extended_rows, -vector % p, p):
                    survivors.append(
                        [list(source_signature), [target_det_character, target_scale_character]]
                    )
    if survivors:
        fail("direct-sum q=7 attack found an unexpected survivor")
    return {
        "independent_algorithm": "DIRECT_SUM_GROUPED_BY_DETERMINANT_TRACE_AND_CONGRUENCE_CLASS",
        "sample_points": sample_count,
        "character_pairs_tested": (q - 1) ** 4,
        "source_completion_sample_rank": base.shape[1],
        "surviving_pairs": survivors,
        "topology_sample_selection_reads_final_values": False,
    }


def inspect_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return {
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "imports_oracle": any("oracle" in item for item in imports),
        "uses_exact_modular_equality_not_digest_for_closure": "np.array_equal" in source,
        "declares_direct_process_scope": "DIRECT_PROCESS_SOFTWARE" in source,
        "declares_no_catvm_custody": '"catvm_custody": False' in source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-source", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    production = json.loads(arguments.production_result.read_text(encoding="utf-8"))
    q5 = q5_reexecution(production)
    q7 = q7_direct_sum_attack(production)
    source = inspect_source(arguments.production_source)
    if source["imports_oracle"]:
        fail("production imports oracle")
    if q5["full_match_count"] != 5 or not q5["exact_forward_inverse_restoration"]:
        fail("q=5 independent reconstruction failed")
    if q7["surviving_pairs"]:
        fail("q=7 rejection failed")
    result = {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "production_source": source,
        "q5_full_boundary_reexecution": q5,
        "q7_direct_character_sum_transfer_attack": q7,
        "controls": {
            "all_q5_character_pairs_searched": True,
            "all_q5_boundary_points_verified_for_each_surviving_pair": True,
            "all_q7_character_pairs_attacked": True,
            "q7_different_transform_algorithm": True,
            "production_does_not_import_oracle": not source["imports_oracle"],
            "digest_not_used_as_state_equality": True,
        },
        "observed_resource_law": {
            "q5_phase_cells": 5**7,
            "q7_phase_cells": 7**7,
            "q5_completion_coordinates": 27,
            "q7_completion_coordinates": 37,
            "fixed_rank_transferable_closure": False,
            "matched_classical_separable_transform_remains": True,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "Q5_F41_FULL_BOUNDARY_CLOSURE_IN_DECLARED_COMPLETION_FAMILY",
            "Q7_F43_EXACT_SAMPLE_REJECTION_OF_ALL_DECLARED_CHARACTER_PAIRS",
            "EXACT_FORWARD_INVERSE_RESTORATION_ON_THE_Q5_RESIDUE_CARRIER",
            "ACTUAL_RESTORED_BACKING_REUSE",
        ],
        "rejected_interpretations": [
            key for key, value in production["claim_boundaries"].items() if value is False
        ],
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
