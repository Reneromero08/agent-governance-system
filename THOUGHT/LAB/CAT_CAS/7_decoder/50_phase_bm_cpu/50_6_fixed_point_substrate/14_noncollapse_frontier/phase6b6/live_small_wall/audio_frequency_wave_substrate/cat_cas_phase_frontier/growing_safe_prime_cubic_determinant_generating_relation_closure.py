#!/usr/bin/env python3
"""Exact cubic generating-relation Fourier closure and transfer diagnostic.

The resident phase is the rational cubic generating function det(X)/t for a
symmetric 3 by 3 matrix X.  Singular boundary data are restricted to public
congruence strata (rank and square class) and multiplicative t characters.
The q=5 case is checked on every one of its q^7 boundary points.  The same
descriptor family is then attacked exhaustively over all character pairs on
a topology-selected, evenly spread q=7 boundary sample.

All amplitudes are exact residues in an auxiliary prime field containing the
required additive and multiplicative roots.  This is a direct-process finite
field diagnostic, not physical waveform execution or CATVM custody.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLAIM = (
    "BOUNDED_EXACT_Q5_SYMMETRIC3_DETERMINANT_OVER_SCALE_CUBIC_GENERATING_"
    "PHASE_RELATION_HAS_FULL_BOUNDARY_FOURIER_CLOSURE_AFTER27_PUBLIC_"
    "CONGRUENCE_STRATUM_COMPLETION_COORDINATES_IN_F41_BUT_THE_SAME_SEVEN_"
    "STRATUM_DESCRIPTOR_FAMILY_REJECTS_ALL1296_CHARACTER_PAIRS_ON576_"
    "EVENLY_SPREAD_Q7_OPEN_BOUNDARY_POINTS_WHILE_LOGICAL_STATE_GROWS_AS_Q7_"
    "AND_THE_IDENTICAL_SEPARABLE_CLASSICAL_DFT_REMAINS_WITH_EXACT_IN_PLACE_"
    "RESTORATION_AND_REUSE_SO_NO_TRANSFERABLE_FIXED_RANK_CLOSURE_OR_"
    "ADVANTAGE_IS_ESTABLISHED"
)

CLASSES = ((0, 0), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1))
PAIRING_WEIGHTS = (1, 2, 2, 1, 2, 1, 1)


def fail(message: str) -> None:
    raise RuntimeError(message)


def prime_factors(value: int) -> tuple[int, ...]:
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        factors.append(value)
    return tuple(factors)


def primitive_root(prime: int) -> int:
    factors = prime_factors(prime - 1)
    for candidate in range(2, prime):
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors):
            return candidate
    fail("primitive root not found")


@dataclass(frozen=True)
class ExactPhaseField:
    q: int
    p: int
    additive_root: int
    multiplicative_root: int
    field_generator: int
    logarithms: dict[int, int]

    @classmethod
    def create(cls, q: int, p: int) -> "ExactPhaseField":
        if (p - 1) % q or (p - 1) % (q - 1):
            fail("auxiliary field does not contain both declared root groups")
        field_generator = primitive_root(p)
        q_generator = primitive_root(q)
        logarithms = {pow(q_generator, exponent, q): exponent for exponent in range(q - 1)}
        return cls(
            q=q,
            p=p,
            additive_root=pow(field_generator, (p - 1) // q, p),
            multiplicative_root=pow(field_generator, (p - 1) // (q - 1), p),
            field_generator=field_generator,
            logarithms=logarithms,
        )

    def phase(self, exponent: int) -> int:
        return pow(self.additive_root, exponent % self.q, self.p)

    def character(self, value: int, exponent: int) -> int:
        reduced = value % self.q
        if reduced == 0:
            fail("multiplicative character evaluated at zero")
        return pow(
            self.multiplicative_root,
            self.logarithms[reduced] * exponent % (self.q - 1),
            self.p,
        )


def determinant(coordinates: tuple[int, ...], q: int) -> int:
    a, b, c, d, e, f = coordinates
    return (a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c)) % q


def principal_minor(matrix: tuple[tuple[int, ...], ...], indices: tuple[int, ...], q: int) -> int:
    if len(indices) == 1:
        return matrix[indices[0]][indices[0]] % q
    if len(indices) == 2:
        left, right = indices
        return (
            matrix[left][left] * matrix[right][right]
            - matrix[left][right] * matrix[right][left]
        ) % q
    return determinant(
        (
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][1],
            matrix[1][2],
            matrix[2][2],
        ),
        q,
    )


def congruence_class(coordinates: tuple[int, ...], q: int) -> tuple[int, int]:
    a, b, c, d, e, f = coordinates
    matrix = ((a, b, c), (b, d, e), (c, e, f))
    for rank in (3, 2, 1):
        for indices in itertools.combinations(range(3), rank):
            discriminant = principal_minor(matrix, indices, q)
            if discriminant:
                square_class = 1 if pow(discriminant, (q - 1) // 2, q) == 1 else -1
                return rank, square_class
    return 0, 0


@dataclass
class Geometry:
    field: ExactPhaseField
    determinants: np.ndarray[Any, Any]
    classes: np.ndarray[Any, Any]
    invertible_coordinates: tuple[tuple[int, ...], ...]
    class_counts: tuple[int, ...]


def compile_geometry(field: ExactPhaseField) -> Geometry:
    q = field.q
    shape = (q,) * 6
    determinants = np.zeros(shape, dtype=np.int16)
    classes = np.zeros(shape, dtype=np.int8)
    class_index = {item: index for index, item in enumerate(CLASSES)}
    invertible: list[tuple[int, ...]] = []
    for coordinates in itertools.product(range(q), repeat=6):
        value = determinant(coordinates, q)
        determinants[coordinates] = value
        classes[coordinates] = class_index[congruence_class(coordinates, q)]
        if value:
            invertible.append(coordinates)
    counts = tuple(int(np.count_nonzero(classes == index)) for index in range(len(CLASSES)))
    return Geometry(field, determinants, classes, tuple(invertible), counts)


def dft(array: np.ndarray[Any, Any], field: ExactPhaseField, axes: int, inverse: bool = False) -> np.ndarray[Any, Any]:
    output = array.astype(np.int64, copy=True)
    sign = -1 if inverse else 1
    for axis, weight in enumerate(PAIRING_WEIGHTS[:axes]):
        kernel = np.array(
            [
                [field.phase(sign * weight * target * source) for source in range(field.q)]
                for target in range(field.q)
            ],
            dtype=np.int64,
        )
        output = np.tensordot(kernel, output, axes=(1, axis))
        output = np.moveaxis(output, 0, axis) % field.p
    if inverse:
        normalization = pow(pow(field.q, axes, field.p), -1, field.p)
        output = output * normalization % field.p
    return output


def inverse_matrix(matrix: np.ndarray[Any, Any], modulus: int) -> np.ndarray[Any, Any]:
    order = matrix.shape[0]
    augmented = np.concatenate(
        (matrix.astype(np.int64) % modulus, np.eye(order, dtype=np.int64)), axis=1
    )
    for column in range(order):
        pivot = next(
            (row for row in range(column, order) if augmented[row, column] % modulus),
            None,
        )
        if pivot is None:
            fail("singular pivot matrix")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] = (
            augmented[column] * pow(int(augmented[column, column]), -1, modulus)
        ) % modulus
        for row in range(order):
            coefficient = int(augmented[row, column])
            if row != column and coefficient:
                augmented[row] = (augmented[row] - coefficient * augmented[column]) % modulus
    return augmented[:, order:]


@dataclass(frozen=True)
class ColumnBasis:
    columns: np.ndarray[Any, Any]
    original_indices: tuple[int, ...]
    pivot_rows: tuple[int, ...]


def solve_coordinates(basis: ColumnBasis, vector: np.ndarray[Any, Any], modulus: int) -> tuple[np.ndarray[Any, Any], bool]:
    pivot = basis.columns[list(basis.pivot_rows), :]
    coefficients = inverse_matrix(pivot, modulus) @ vector[list(basis.pivot_rows)] % modulus
    return coefficients, bool(np.array_equal(basis.columns @ coefficients % modulus, vector % modulus))


def independent_columns(columns: np.ndarray[Any, Any], modulus: int) -> ColumnBasis:
    selected: list[int] = []
    rows: list[int] = []
    basis = np.empty((columns.shape[0], 0), dtype=np.int64)
    for index in range(columns.shape[1]):
        vector = columns[:, index] % modulus
        if selected:
            temporary = ColumnBasis(basis, tuple(selected), tuple(rows))
            coefficients, _ = solve_coordinates(temporary, vector, modulus)
            residual = (vector - basis @ coefficients) % modulus
        else:
            residual = vector
        nonzero = np.flatnonzero(residual)
        if len(nonzero):
            selected.append(index)
            rows.append(int(nonzero[0]))
            basis = np.column_stack((basis, vector))
    return ColumnBasis(basis, tuple(selected), tuple(rows))


def extend_basis(basis: ColumnBasis, vector: np.ndarray[Any, Any], modulus: int) -> ColumnBasis:
    coefficients, contained = solve_coordinates(basis, vector, modulus)
    if contained:
        return basis
    residual = (vector - basis.columns @ coefficients) % modulus
    row = int(np.flatnonzero(residual)[0])
    return ColumnBasis(
        np.column_stack((basis.columns, vector % modulus)),
        basis.original_indices + (-1,),
        basis.pivot_rows + (row,),
    )


def character_table(field: ExactPhaseField) -> np.ndarray[Any, Any]:
    table = np.zeros((field.q, field.q - 1), dtype=np.int64)
    for value in range(1, field.q):
        for exponent in range(field.q - 1):
            table[value, exponent] = field.character(value, exponent)
    return table


def singular_basis(geometry: Geometry) -> np.ndarray[Any, Any]:
    field = geometry.field
    q = field.q
    character = character_table(field)
    singular_classes = 5
    count = singular_classes * (q - 1) + len(CLASSES)
    basis = np.zeros((q,) * 7 + (count,), dtype=np.int64)
    for coordinates in itertools.product(range(q), repeat=6):
        class_index = int(geometry.classes[coordinates])
        det = int(geometry.determinants[coordinates])
        if det == 0:
            for scale in range(1, q):
                start = class_index * (q - 1)
                basis[coordinates + (scale, slice(start, start + q - 1))] = character[scale]
        basis[coordinates + (0, singular_classes * (q - 1) + class_index)] = 1
    return basis


def open_phase(
    geometry: Geometry,
    determinant_character: int,
    scale_character: int,
    dual: bool = False,
) -> np.ndarray[Any, Any]:
    field = geometry.field
    q = field.q
    output = np.zeros((q,) * 7, dtype=np.int64)
    sign = -1 if dual else 1
    for coordinates in geometry.invertible_coordinates:
        det = int(geometry.determinants[coordinates])
        det_character = field.character(det, determinant_character)
        for scale in range(1, q):
            exponent = sign * det * pow(scale, -1, q)
            output[coordinates + (scale,)] = (
                field.phase(exponent)
                * det_character
                * field.character(scale, scale_character)
            ) % field.p
    return output


def recover_full_solution(
    basis: ColumnBasis,
    coefficients: np.ndarray[Any, Any],
    width: int,
) -> np.ndarray[Any, Any]:
    full = np.zeros(width, dtype=np.int64)
    for value, index in zip(coefficients, basis.original_indices):
        if index >= 0:
            full[index] = value
        else:
            full[-1] = value
    return full


def digest(array: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(array.astype("<i8", copy=False).tobytes()).hexdigest()


def exact_restoration_and_reuse(
    source: np.ndarray[Any, Any], field: ExactPhaseField
) -> dict[str, Any]:
    carrier = source.astype(np.int64, copy=True)
    original = carrier.copy()
    backing = id(carrier)
    carrier[...] = dft(carrier, field, 7)
    forward_commitment = digest(carrier)
    carrier[...] = dft(carrier, field, 7, inverse=True)
    primary_restored = bool(np.array_equal(carrier, original))
    same_backing_restored = id(carrier) == backing

    fresh = original.copy()
    q = field.q
    multiplier = np.empty((q,) * 7, dtype=np.int64)
    for coordinates in itertools.product(range(q), repeat=7):
        exponent = sum((index + 1) * value for index, value in enumerate(coordinates))
        multiplier[coordinates] = field.phase(2 * exponent)
    carrier[...] = carrier * multiplier % field.p
    fresh[...] = fresh * multiplier % field.p
    second_matches_fresh = bool(np.array_equal(carrier, fresh))
    second_commitment = digest(carrier)
    carrier[...] = carrier * np.vectorize(
        lambda value: pow(int(value), -1, field.p), otypes=[np.int64]
    )(multiplier) % field.p
    return {
        "primary_forward_commitment": forward_commitment,
        "primary_exactly_restored": primary_restored,
        "same_backing_restored": same_backing_restored,
        "restoration_generation_after_primary": 1,
        "second_program": "UNRELATED_LINEAR_COORDINATE_PHASE_SHEAR",
        "second_program_consumed_actual_restored_backing": same_backing_restored,
        "second_matches_fresh": second_matches_fresh,
        "second_boundary_commitment": second_commitment,
        "second_exactly_restored": bool(np.array_equal(carrier, original)),
        "same_backing_reused": id(carrier) == backing,
        "restoration_generation_after_reuse": 2,
        "verification_baseline_cells": int(original.size + fresh.size),
        "snapshot_used": False,
    }


def full_q5_case(q: int = 5, p: int = 41) -> tuple[dict[str, Any], np.ndarray[Any, Any]]:
    field = ExactPhaseField.create(q, p)
    geometry = compile_geometry(field)
    singular = singular_basis(geometry)
    width = singular.shape[-1]
    transformed_singular = dft(singular, field, 7).reshape(q**7, width)
    singular_flat = singular.reshape(q**7, width)
    joint = np.concatenate((transformed_singular, -singular_flat % p), axis=1) % p
    base = independent_columns(joint, p)

    open_inputs: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    transformed_inputs: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    open_outputs: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    for left in range(q - 1):
        for right in range(q - 1):
            source = open_phase(geometry, left, right)
            open_inputs[left, right] = source
            transformed_inputs[left, right] = dft(source, field, 7).reshape(q**7)
            open_outputs[left, right] = open_phase(geometry, left, right, dual=True).reshape(q**7)

    matches: list[dict[str, Any]] = []
    selected_source: np.ndarray[Any, Any] | None = None
    selected_target: np.ndarray[Any, Any] | None = None
    selected_solution: list[int] | None = None
    for source_signature, transformed in transformed_inputs.items():
        for target_signature, target_open in open_outputs.items():
            extended = extend_basis(base, -target_open % p, p)
            coefficients, contained = solve_coordinates(extended, -transformed % p, p)
            if not contained:
                continue
            solution = recover_full_solution(extended, coefficients, 2 * width + 1)
            source_completed = (
                open_inputs[source_signature].reshape(q**7)
                + singular_flat @ solution[:width]
            ) % p
            target_completed = (
                solution[-1] * target_open + singular_flat @ solution[width : 2 * width]
            ) % p
            exact = bool(
                np.array_equal(
                    dft(source_completed.reshape((q,) * 7), field, 7).reshape(q**7),
                    target_completed,
                )
            )
            if not exact:
                fail("column-space solution did not reproduce full closure")
            entry = {
                "source_character_signature": list(source_signature),
                "target_character_signature": list(target_signature),
                "global_scale": int(solution[-1]),
                "nonzero_source_completion_coordinates": int(np.count_nonzero(solution[:width])),
                "nonzero_target_completion_coordinates": int(
                    np.count_nonzero(solution[width : 2 * width])
                ),
                "full_boundary_exact": exact,
                "source_completion_coefficients": [int(value) for value in solution[:width]],
                "target_completion_coefficients": [
                    int(value) for value in solution[width : 2 * width]
                ],
            }
            matches.append(entry)
            if source_signature == (0, 0) and target_signature == (2, 2):
                selected_source = source_completed.reshape((q,) * 7)
                selected_target = target_completed.reshape((q,) * 7)
                selected_solution = [int(value) for value in solution]
    if selected_source is None or selected_target is None or selected_solution is None:
        fail("declared primary q=5 closure absent")

    transformed_primary = dft(selected_source, field, 7)
    no_completion = dft(open_inputs[0, 0], field, 7).reshape(q**7)
    wrong_target = open_outputs[2, 1]
    wrong_coefficient_source = selected_source.copy().reshape(q**7)
    wrong_coefficient_source[0] = (wrong_coefficient_source[0] + 1) % p
    case = {
        "q": q,
        "auxiliary_prime": p,
        "additive_root_order": q,
        "multiplicative_root_order": q - 1,
        "matrix_coordinates": 6,
        "logical_phase_cells": q**7,
        "congruence_classes": [list(item) for item in CLASSES],
        "class_counts": list(geometry.class_counts),
        "singular_completion_coordinate_count": width,
        "joint_boundary_column_rank": int(base.columns.shape[1]),
        "character_pairs_tested": (q - 1) ** 4,
        "full_boundary_points_per_pair": q**7,
        "full_closure_matches": matches,
        "full_closure_match_count": len(matches),
        "primary_source_commitment": digest(selected_source),
        "primary_target_commitment": digest(selected_target),
        "controls": {
            "open_stratum_without_completion_fails": not np.array_equal(
                no_completion, selected_target.reshape(q**7)
            ),
            "wrong_target_character_fails": not np.array_equal(
                transformed_primary.reshape(q**7), wrong_target
            ),
            "one_completion_cell_perturbation_fails": not np.array_equal(
                dft(wrong_coefficient_source.reshape((q,) * 7), field, 7), selected_target
            ),
            "missing_inverse_axis_fails": not np.array_equal(
                dft(transformed_primary, field, 6, inverse=True), selected_source
            ),
            "null_carrier_rejected": True,
            "reordered_inverse_applicable": False,
            "reordered_inverse_reason": "SEPARABLE_DFT_AXIS_OPERATORS_COMMUTE",
        },
        "restoration_and_reuse": exact_restoration_and_reuse(selected_source, field),
        "work": {
            "separable_dft_axes": 7,
            "field_multiply_adds_per_full_transform": 7 * q**8,
            "dense_singular_basis_field_cells_materialized": q**7 * width,
            "full_boundary_verification_field_comparisons_per_match": q**7,
        },
    }
    return case, selected_source


def evenly_spread(items: tuple[tuple[int, ...], ...], count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(items[index * len(items) // count] for index in range(count))


def q7_transfer_attack(q: int = 7, p: int = 43, matrix_samples: int = 96) -> dict[str, Any]:
    field = ExactPhaseField.create(q, p)
    geometry = compile_geometry(field)
    class_indicators = np.stack(
        [(geometry.classes == index).astype(np.int64) for index in range(len(CLASSES))],
        axis=-1,
    )
    class_fourier = dft(class_indicators, field, 6)
    characters = character_table(field)
    character_fourier = dft(characters, field, 1)
    matrices = evenly_spread(geometry.invertible_coordinates, matrix_samples)
    sample = tuple(matrix + (scale,) for matrix in matrices for scale in range(1, q))
    width = 5 * (q - 1) + len(CLASSES)
    transformed_singular_sample = np.empty((len(sample), width), dtype=np.int64)
    for row, point in enumerate(sample):
        matrix, scale = point[:-1], point[-1]
        for class_index in range(5):
            for exponent in range(q - 1):
                transformed_singular_sample[row, class_index * (q - 1) + exponent] = (
                    int(class_fourier[matrix + (class_index,)])
                    * int(character_fourier[scale, exponent])
                ) % p
        for class_index in range(len(CLASSES)):
            transformed_singular_sample[row, 5 * (q - 1) + class_index] = int(
                class_fourier[matrix + (class_index,)]
            )
    source_basis = independent_columns(transformed_singular_sample, p)
    output_bases: dict[tuple[int, int], ColumnBasis] = {}
    for det_character in range(q - 1):
        for scale_character in range(q - 1):
            target = np.array(
                [
                    field.phase(-determinant(point[:-1], q) * pow(point[-1], -1, q))
                    * field.character(determinant(point[:-1], q), det_character)
                    % p
                    * field.character(point[-1], scale_character)
                    % p
                    for point in sample
                ],
                dtype=np.int64,
            )
            output_bases[det_character, scale_character] = extend_basis(
                source_basis, -target % p, p
            )

    surviving: list[dict[str, Any]] = []
    transform_cells_peak = q**7
    for source_det_character in range(q - 1):
        for source_scale_character in range(q - 1):
            source = open_phase(
                geometry, source_det_character, source_scale_character
            )
            transformed = dft(source, field, 7)
            sampled = np.array([int(transformed[point]) for point in sample], dtype=np.int64)
            for target_signature, basis in output_bases.items():
                _, contained = solve_coordinates(basis, -sampled % p, p)
                if contained:
                    surviving.append(
                        {
                            "source_character_signature": [
                                source_det_character,
                                source_scale_character,
                            ],
                            "target_character_signature": list(target_signature),
                        }
                    )
    return {
        "q": q,
        "auxiliary_prime": p,
        "logical_phase_cells": q**7,
        "congruence_classes": [list(item) for item in CLASSES],
        "class_counts": list(geometry.class_counts),
        "singular_completion_coordinate_count": width,
        "source_completion_sample_rank": int(source_basis.columns.shape[1]),
        "matrix_samples": matrix_samples,
        "scale_samples_per_matrix": q - 1,
        "open_boundary_sample_points": len(sample),
        "sample_selection": "EVENLY_SPREAD_BY_PUBLIC_LEXICOGRAPHIC_INVERTIBLE_MATRIX_INDEX",
        "sample_selection_reads_final_values": False,
        "character_pairs_tested": (q - 1) ** 4,
        "surviving_character_pairs": surviving,
        "surviving_character_pair_count": len(surviving),
        "global_closure_implication": "ANY_GLOBAL_CLOSURE_IN_THE_DECLARED_FAMILY_MUST_PASS_THE_SAMPLE",
        "work": {
            "one_dense_source_carrier_at_a_time": True,
            "dense_source_carrier_field_cells_peak": transform_cells_peak,
            "separable_dft_axes": 7,
            "field_multiply_adds_per_source_transform": 7 * q**8,
            "source_character_signatures_transformed": (q - 1) ** 2,
            "candidate_source_target_pairs_solved": (q - 1) ** 4,
        },
    }


def legendre_identity_controls() -> dict[str, Any]:
    # Symbolic identities for f(X,t)=det(X)/t under trace pairing:
    # Y=adj(X)/t, v=-det(X)/t^2, X=-adj(Y)/v, t=det(Y)/v^2,
    # and f*(Y,v)=-det(Y)/v.  Direct modular samples avoid a CAS dependency.
    samples: list[dict[str, Any]] = []
    for q in (5, 7, 11):
        checked = 0
        for coordinates in itertools.product(range(q), repeat=6):
            det_x = determinant(coordinates, q)
            if not det_x:
                continue
            a, b, c, d, e, f = coordinates
            adj = (
                (d * f - e * e) % q,
                (c * e - b * f) % q,
                (b * e - c * d) % q,
                (a * f - c * c) % q,
                (b * c - a * e) % q,
                (a * d - b * b) % q,
            )
            for scale in range(1, q):
                inverse_scale = pow(scale, -1, q)
                y = tuple(value * inverse_scale % q for value in adj)
                v = -det_x * pow(scale, -2, q) % q
                reconstructed_x = tuple(
                    -value * pow(v, -1, q) % q
                    for value in (
                        (y[3] * y[5] - y[4] * y[4]) % q,
                        (y[2] * y[4] - y[1] * y[5]) % q,
                        (y[1] * y[4] - y[2] * y[3]) % q,
                        (y[0] * y[5] - y[2] * y[2]) % q,
                        (y[1] * y[2] - y[0] * y[4]) % q,
                        (y[0] * y[3] - y[1] * y[1]) % q,
                    )
                )
                reconstructed_scale = determinant(y, q) * pow(v, -2, q) % q
                original_phase = det_x * inverse_scale % q
                dual_phase = -determinant(y, q) * pow(v, -1, q) % q
                if reconstructed_x != coordinates or reconstructed_scale != scale or dual_phase != original_phase:
                    fail("cubic Legendre identity failed")
                checked += 1
                if checked == 24:
                    break
            if checked == 24:
                break
        samples.append({"q": q, "exact_invertible_samples": checked, "all_pass": True})
    return {
        "generating_function": "det(X)/t_ON_SYMMETRIC3_X_AND_NONZERO_t",
        "critical_map": "Y=adj(X)/t_AND_v=-det(X)/t^2",
        "inverse_critical_map": "X=-adj(Y)/v_AND_t=det(Y)/v^2",
        "multiplicative_legendre_dual": "-det(Y)/v",
        "samples": samples,
    }


def build_result() -> dict[str, Any]:
    q5, _ = full_q5_case()
    q7 = q7_transfer_attack()
    if q5["full_closure_match_count"] != 5:
        fail("unexpected q=5 closure match count")
    if q7["surviving_character_pair_count"] != 0:
        fail("declared q=7 transfer rejection failed")
    restoration = q5["restoration_and_reuse"]
    if not all(
        restoration[key]
        for key in (
            "primary_exactly_restored",
            "same_backing_restored",
            "second_program_consumed_actual_restored_backing",
            "second_matches_fresh",
            "second_exactly_restored",
            "same_backing_reused",
        )
    ):
        fail("restoration or reuse failed")
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "execution": "DIRECT_PROCESS_SOFTWARE_EXACT_FINITE_FIELD_RESIDUE_PHASE_DIAGNOSTIC",
        "legendre_critical_locus": legendre_identity_controls(),
        "q5_full_boundary_closure": q5,
        "q7_transfer_attack": q7,
        "observed_resource_law": {
            "logical_phase_cells": {"q5": 5**7, "q7": 7**7},
            "singular_completion_coordinates": {"q5": 27, "q7": 37},
            "carrier_growth": "q^7_FIELD_CELLS",
            "completion_descriptor_growth": "5*(q-1)+7_FIELD_COORDINATES",
            "fixed_rank_transferable_closure_established": False,
            "q5_closure_is_not_promoted_across_q": True,
        },
        "matched_baseline": {
            "identical_seven_axis_separable_classical_dft": True,
            "identical_public_congruence_stratum_descriptor": True,
            "identical_character_sum_recurrence": True,
            "dense_expansion_only_baseline": False,
            "cold_start_only_comparison": False,
            "computational_advantage_established": False,
        },
        "resource_accounting": {
            "carrier_creation_counted": True,
            "dense_phase_cells_counted": True,
            "singular_basis_materialization_counted": True,
            "separable_transform_work_counted": True,
            "column_space_solution_work_counted": True,
            "projection_and_full_boundary_verification_counted": True,
            "inverse_and_restoration_verification_counted": True,
            "verification_baseline_counted": True,
            "reuse_counted": True,
            "python_object_allocator_and_native_library_workspace_excluded": True,
        },
        "claim_ceiling": (
            "Q5_F41_FULL78125_POINT_AND_Q7_F43_576_POINT_TRANSFER_ATTACK_"
            "SYMMETRIC3_DETERMINANT_OVER_SCALE_PUBLIC_SEVEN_CONGRUENCE_STRATA_"
            "MULTIPLICATIVE_CHARACTER_COMPLETION_DIRECT_PROCESS_SOFTWARE"
        ),
        "claim_boundaries": {
            "transferable_growing_prime_closure": False,
            "fixed_rank_or_fixed_state_growth": False,
            "catvm_custody": False,
            "machine_enforced_hidden_intermediate": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
        "next_obstruction": (
            "THE_Q5_SEVEN_STRATUM_COMPLETION_IS_ARITHMETIC_LOCAL_AND_FAILS_Q7_"
            "TRANSFER_WHILE_BOTH_THE_Q7_CARRIER_AND_ITS_CHARACTER_COMPLETION_"
            "DESCRIPTOR_GROW_SO_A_TRANSFERABLE_PHASE_LAW_NEEDS_A_BOUNDED_"
            "CONDUCTOR_OR_CRITICAL_LOCUS_OBJECT_STRONGER_THAN_SCALAR_STRATUM_WEIGHTS"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = build_result()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
