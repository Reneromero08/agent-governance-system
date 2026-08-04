#!/usr/bin/env python3
"""No-import oracle for the fused two-segment Bluestein boundary compiler."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLAIM = (
    "BOUNDED_EXACT_FOURTEEN_DECLARED_PRIME_FUSED_TWO_SEGMENT_BLUESTEIN_"
    "FINAL_BOUNDARY_COMPILER_USES_THE_NONZERO_PUBLIC_KERNEL_SPECTRUM_"
    "TO_REPLACE_THE_QMINUS1_RESIDENT_GAUSS_DESCRIPTOR_AND_THREE_M_"
    "SCRATCH_SEGMENTS_WITH_ONE_RESIDENT_SCALAR_AND_TWO_M_REVERSIBLY_"
    "CLEARED_TRANSFORM_SEGMENTS_WITH_THETA_M_LOG_M_PLUS_Q_WORK_EXACT_"
    "SAME_BACKING_RESTORATION_AND_UNRELATED_REUSE_BUT_ALLOCATED_STATE_"
    "REMAINS_LINEAR_EXACT_WIDTH_GROWS_AND_THE_IDENTICAL_CLASSICAL_"
    "FUSED_BLUESTEIN_COMPILER_REMAINS"
)


CASES = (
    (5, 41, 1, 2),
    (7, 337, 3, 5),
    (11, 5281, 5, 8),
    (13, 1249, 7, 11),
    (17, 5441, 9, 14),
    (19, 32833, 11, 17),
    (23, 16193, 13, 20),
    (29, 38977, 15, 23),
    (31, 29761, 17, 26),
    (37, 127873, 19, 29),
    (41, 78721, 21, 32),
    (43, 231169, 23, 35),
    (47, 691841, 25, 38),
    (53, 264577, 27, 41),
)


PAIRING_WEIGHTS = (1, 2, 2, 1, 2, 1)
PRIMARY_POINT = (2, 0, 0, 1, 0, 1)


def fail(message: str) -> None:
    raise RuntimeError(message)


def factors(number: int) -> list[int]:
    answer: list[int] = []
    divisor = 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            answer.append(divisor)
            while number % divisor == 0:
                number //= divisor
        divisor += 1
    if number > 1:
        answer.append(number)
    return answer


def generator(prime: int) -> int:
    divisors = factors(prime - 1)
    for candidate in range(2, prime):
        if all(
            pow(candidate, (prime - 1) // divisor, prime) != 1
            for divisor in divisors
        ):
            return candidate
    fail("independent primitive-root search failed")


def power_two_ceiling(value: int) -> int:
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class Field:
    q: int
    p: int
    q_generator: int
    additive_root: int
    character_root: int
    chirp_root: int
    convolution_root: int
    convolution_width: int

    @classmethod
    def make(cls, q: int, p: int) -> "Field":
        primitive = generator(p)
        h = q - 1
        width = power_two_ceiling(2 * h - 1)
        return cls(
            q=q,
            p=p,
            q_generator=generator(q),
            additive_root=pow(primitive, (p - 1) // q, p),
            character_root=pow(primitive, (p - 1) // h, p),
            chirp_root=pow(primitive, (p - 1) // (2 * h), p),
            convolution_root=pow(primitive, (p - 1) // width, p),
            convolution_width=width,
        )


def log_map(field: Field) -> dict[int, int]:
    return {
        pow(field.q_generator, exponent, field.q): exponent
        for exponent in range(field.q - 1)
    }


def character(
    field: Field, logs: dict[int, int], value: int, exponent: int
) -> int:
    reduced = value % field.q
    if reduced == 0:
        return 0
    return pow(
        field.character_root,
        logs[reduced] * exponent % (field.q - 1),
        field.p,
    )


def direct_gauss(field: Field, logs: dict[int, int]) -> list[int]:
    return [
        sum(
            pow(field.additive_root, value, field.p)
            * character(field, logs, value, exponent)
            for value in range(1, field.q)
        )
        % field.p
        for exponent in range(field.q - 1)
    ]


def recursive_transform(
    values: list[int], root: int, modulus: int, inverse: bool = False
) -> list[int]:
    """Independent out-of-place recursive radix-two NTT."""
    width = len(values)
    if width == 1:
        return values[:]
    stage_root = pow(root, -1, modulus) if inverse else root

    def recurse(data: list[int], local_root: int) -> list[int]:
        if len(data) == 1:
            return data
        even = recurse(data[0::2], local_root * local_root % modulus)
        odd = recurse(data[1::2], local_root * local_root % modulus)
        twiddle = 1
        half = len(data) // 2
        output = [0] * len(data)
        for index in range(half):
            high = odd[index] * twiddle % modulus
            output[index] = (even[index] + high) % modulus
            output[index + half] = (even[index] - high) % modulus
            twiddle = twiddle * local_root % modulus
        return output

    transformed = recurse(values, stage_root)
    if inverse:
        scale = pow(width, -1, modulus)
        transformed = [value * scale % modulus for value in transformed]
    return transformed


def seed_segments(field: Field) -> tuple[list[int], list[int]]:
    h = field.q - 1
    width = field.convolution_width
    left = [0] * width
    kernel = [0] * width
    orbit = 1
    for index in range(h):
        chirp = pow(field.chirp_root, index * index, field.p)
        left[index] = (
            pow(field.additive_root, orbit, field.p) * chirp % field.p
        )
        anti = pow(field.chirp_root, -index * index, field.p)
        kernel[index] = anti
        if index:
            kernel[width - index] = anti
        orbit = orbit * field.q_generator % field.q
    return left, kernel


def determinant(point: tuple[int, ...], q: int) -> int:
    a, b, c, d, e, f = point
    return (
        a * (d * f - e * e)
        - b * (b * f - e * c)
        + c * (b * e - d * c)
    ) % q


def minor(matrix: tuple[tuple[int, ...], ...], indexes: tuple[int, ...], q: int) -> int:
    if len(indexes) == 1:
        return matrix[indexes[0]][indexes[0]] % q
    if len(indexes) == 2:
        i, j = indexes
        return (matrix[i][i] * matrix[j][j] - matrix[i][j] ** 2) % q
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


def rank_class(point: tuple[int, ...], q: int) -> tuple[int, int]:
    a, b, c, d, e, f = point
    matrix = ((a, b, c), (b, d, e), (c, e, f))
    for rank in (3, 2, 1):
        for indexes in itertools.combinations(range(3), rank):
            value = minor(matrix, indexes, q)
            if value:
                square = 1 if pow(value, (q - 1) // 2, q) == 1 else -1
                return rank, square
    return 0, 0


def boundary(
    field: Field,
    logs: dict[int, int],
    gauss: list[int],
    determinant_index: int,
    scale_index: int,
    point: tuple[int, ...],
    target_scale: int,
) -> int:
    q, p = field.q, field.p
    h = q - 1
    half = h // 2
    matrix_rank, square = rank_class(point, q)
    answer = 0
    for channel in range(h):
        coefficient = (
            pow(h, -1, p) * gauss[(determinant_index - channel) % h] % p
        )
        if matrix_rank == 3:
            gamma = (
                gauss[channel] ** 2
                * gauss[(channel + half) % h]
                * gauss[half] ** 3
            ) % p
            det_factor = gamma * character(
                field, logs, determinant(point, q), -channel
            ) % p
        elif channel == 0:
            det_factor = (
                q**6 - q**5 - q**3 + q**2
                if matrix_rank == 0
                else q**2 - q**3
            ) % p
        elif channel == half and matrix_rank == 1:
            det_factor = q**2 * h * gauss[half] ** 3 * square % p
        else:
            det_factor = 0
        source_index = (determinant_index + scale_index - channel) % h
        scale_factor = (
            gauss[source_index]
            * character(field, logs, target_scale, -source_index)
            % p
            if target_scale % q
            else (h % p if source_index == 0 else 0)
        )
        answer += coefficient * det_factor * scale_factor
    return answer % p


def direct_relation_sum(
    field: Field,
    logs: dict[int, int],
    determinant_index: int,
    scale_index: int,
    target: tuple[int, ...],
    target_scale: int,
) -> tuple[int, int]:
    answer = 0
    terms = 0
    for source in itertools.product(range(field.q), repeat=6):
        det = determinant(source, field.q)
        if det == 0:
            continue
        det_phase = character(field, logs, det, determinant_index)
        pairing = sum(
            weight * left * right
            for weight, left, right in zip(PAIRING_WEIGHTS, target, source)
        )
        for scale in range(1, field.q):
            source_value = (
                pow(
                    field.additive_root,
                    det * pow(scale, -1, field.q) % field.q,
                    field.p,
                )
                * det_phase
                * character(field, logs, scale, scale_index)
            ) % field.p
            final_phase = pow(
                field.additive_root,
                (pairing + target_scale * scale) % field.q,
                field.p,
            )
            answer += source_value * final_phase
            terms += 1
    return answer % field.p, terms


def digest(values: list[int]) -> str:
    hasher = hashlib.sha256()
    for index, value in enumerate(values):
        if index:
            hasher.update(b",")
        hasher.update(str(value).encode("ascii"))
    return hasher.hexdigest()


def fused_update(
    field: Field,
    logs: dict[int, int],
    determinant_index: int,
    scale_index: int,
    point: tuple[int, ...],
    target_scale: int,
    cells: list[int],
    scalar_sign: int,
    omit_frequency: int | None = None,
    force_singular_frequency: int | None = None,
) -> dict[str, Any]:
    width = field.convolution_width
    expected = 1 + 2 * width
    if len(cells) != expected:
        fail("independent carrier size mismatch")
    if any(cells[1:]):
        fail("independent scratch did not enter zero")
    left_offset = 1
    kernel_offset = 1 + width
    left_seed, kernel_seed = seed_segments(field)
    cells[left_offset:kernel_offset] = left_seed
    cells[kernel_offset:] = kernel_seed

    left_hat = recursive_transform(left_seed, field.convolution_root, field.p)
    kernel_hat = recursive_transform(kernel_seed, field.convolution_root, field.p)
    if force_singular_frequency is not None:
        kernel_hat[force_singular_frequency % width] = 0
    if any(value == 0 for value in kernel_hat):
        fail("independent public kernel spectrum is not invertible")
    cells[left_offset:kernel_offset] = left_hat
    cells[kernel_offset:] = kernel_hat
    spectrum_commitment = digest(kernel_hat)

    for index in range(width):
        if omit_frequency is None or index != omit_frequency % width:
            cells[left_offset + index] = (
                cells[left_offset + index] * cells[kernel_offset + index]
            ) % field.p
    convolution = recursive_transform(
        cells[left_offset:kernel_offset],
        field.convolution_root,
        field.p,
        inverse=True,
    )
    cells[left_offset:kernel_offset] = convolution
    h = field.q - 1
    gauss = [
        pow(field.chirp_root, index * index, field.p) * convolution[index] % field.p
        for index in range(h)
    ]
    scalar = boundary(
        field,
        logs,
        gauss,
        determinant_index,
        scale_index,
        point,
        target_scale,
    )
    cells[0] = (cells[0] + scalar_sign * scalar) % field.p

    restored_left_hat = recursive_transform(
        convolution, field.convolution_root, field.p
    )
    for index in range(width):
        if omit_frequency is None or index != omit_frequency % width:
            restored_left_hat[index] = (
                restored_left_hat[index] * pow(kernel_hat[index], -1, field.p)
            ) % field.p
    restored_left = recursive_transform(
        restored_left_hat, field.convolution_root, field.p, inverse=True
    )
    restored_kernel = recursive_transform(
        kernel_hat, field.convolution_root, field.p, inverse=True
    )
    cells[left_offset:kernel_offset] = [
        (left - seed) % field.p
        for left, seed in zip(restored_left, left_seed)
    ]
    cells[kernel_offset:] = [
        (left - seed) % field.p
        for left, seed in zip(restored_kernel, kernel_seed)
    ]
    if any(cells[1:]):
        fail("independent fused scratch did not uncompute")
    return {
        "scalar": scalar,
        "gauss": gauss,
        "kernel_spectrum_commitment": spectrum_commitment,
        "kernel_spectrum_nonzero": all(kernel_hat),
        "ntt_calls": 6,
        "ntt_butterflies": 3 * width * (width.bit_length() - 1),
    }


def oracle_case(q: int, p: int, determinant_index: int, scale_index: int) -> dict[str, Any]:
    field = Field.make(q, p)
    logs = log_map(field)
    expected_gauss = direct_gauss(field, logs)
    width = field.convolution_width
    cells = [0] * (1 + 2 * width)
    backing = id(cells)
    primary = fused_update(
        field,
        logs,
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        cells,
        1,
    )
    if primary["gauss"] != expected_gauss:
        fail("independent fused path differs from direct Gauss values")
    projected = cells[0]
    persisted = projected
    resident_commitment = digest(cells)
    inverse = fused_update(
        field,
        logs,
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        cells,
        -1,
    )
    if any(cells) or id(cells) != backing:
        fail("independent fused inverse failed on the same backing")
    if primary["kernel_spectrum_commitment"] != inverse["kernel_spectrum_commitment"]:
        fail("independent kernel spectrum changed across inverse")

    h = q - 1
    second_det = (determinant_index + 2) % h
    second_scale = (scale_index + 3) % h
    second_point = (3, 0, 0, 2, 0, 1)
    reused = fused_update(
        field, logs, second_det, second_scale, second_point, 3 % q, cells, 1
    )
    reused_boundary = cells[0]
    fused_update(
        field, logs, second_det, second_scale, second_point, 3 % q, cells, -1
    )
    fresh_cells = [0] * (1 + 2 * width)
    fresh = fused_update(
        field,
        logs,
        second_det,
        second_scale,
        second_point,
        3 % q,
        fresh_cells,
        1,
    )
    fresh_boundary = fresh_cells[0]
    fused_update(
        field,
        logs,
        second_det,
        second_scale,
        second_point,
        3 % q,
        fresh_cells,
        -1,
    )
    if (
        reused_boundary != fresh_boundary
        or reused["ntt_butterflies"] != fresh["ntt_butterflies"]
        or any(cells)
        or any(fresh_cells)
        or id(cells) != backing
    ):
        fail("independent restored-carrier reuse differs from fresh")

    missing = [0] * (1 + 2 * width)
    fused_update(
        field,
        logs,
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        missing,
        1,
    )
    missing_inverse_fails = any(missing)
    wrong = missing[:]
    fused_update(
        field,
        logs,
        (determinant_index + 1) % h,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        wrong,
        -1,
    )
    wrong_inverse_fails = any(wrong)
    omitted = [0] * (1 + 2 * width)
    fused_update(
        field,
        logs,
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        omitted,
        1,
        omit_frequency=0,
    )
    omitted_frequency_changes_boundary = omitted[0] != projected
    singular_rejected = False
    try:
        fused_update(
            field,
            logs,
            determinant_index,
            scale_index,
            PRIMARY_POINT,
            2 % q,
            [0] * (1 + 2 * width),
            1,
            force_singular_frequency=0,
        )
    except RuntimeError:
        singular_rejected = True
    null_carrier_rejected = False
    try:
        fused_update(
            field,
            logs,
            determinant_index,
            scale_index,
            PRIMARY_POINT,
            2 % q,
            [],
            1,
        )
    except RuntimeError:
        null_carrier_rejected = True
    if not all(
        (
            missing_inverse_fails,
            wrong_inverse_fails,
            omitted_frequency_changes_boundary,
            singular_rejected,
            null_carrier_rejected,
        )
    ):
        fail("independent fused mutation control failed")

    direct_check: dict[str, Any] | None = None
    if q in (5, 7):
        direct_value, terms = direct_relation_sum(
            field,
            logs,
            determinant_index,
            scale_index,
            PRIMARY_POINT,
            2 % q,
        )
        if direct_value != projected:
            fail("independent original relation sum differs from fused boundary")
        direct_check = {
            "direct_boundary_scalar": direct_value,
            "nonzero_source_terms": terms,
            "matches_fused_boundary": True,
        }
    return {
        "q": q,
        "auxiliary_prime": p,
        "convolution_width": width,
        "logical_carrier_field_cells": 1 + 2 * width,
        "carrier_capacity_bits": (1 + 2 * width) * p.bit_length(),
        "direct_gauss_commitment": digest(expected_gauss),
        "kernel_spectrum_commitment": primary["kernel_spectrum_commitment"],
        "fused_gauss_matches_direct": True,
        "projected_boundary_scalar": projected,
        "projected_result_survives_inverse": persisted == projected,
        "resident_scalar_commitment": resident_commitment,
        "exact_same_backing_restoration": id(cells) == backing and not any(cells),
        "actual_restored_backing_reuse_matches_fresh": reused_boundary
        == fresh_boundary,
        "reused_boundary_scalar": reused_boundary,
        "fresh_boundary_scalar": fresh_boundary,
        "reuse_resource_signature_matches_fresh": reused["ntt_butterflies"]
        == fresh["ntt_butterflies"],
        "missing_inverse_fails": missing_inverse_fails,
        "wrong_program_inverse_fails": wrong_inverse_fails,
        "frequency_zero_omission_changes_boundary": omitted_frequency_changes_boundary,
        "singular_kernel_spectrum_rejected": singular_rejected,
        "singular_rejection_atomic_rollback_claimed": False,
        "null_carrier_rejected": null_carrier_rejected,
        "direct_relation_check": direct_check,
        "expected_ntt_calls_per_compiler": primary["ntt_calls"],
        "expected_butterflies_per_compiler": primary["ntt_butterflies"],
    }


def build_result() -> dict[str, Any]:
    cases = [oracle_case(*case) for case in CASES]
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_independence": {
            "imports_production": False,
            "imports_predecessor": False,
            "field_roots_reconstructed": True,
            "uses_out_of_place_recursive_ntt": True,
            "uses_production_iterative_ntt": False,
            "direct_gauss_values_reconstructed": True,
            "fused_two_segment_transaction_reconstructed": True,
            "direct_original_relation_checks": 2,
        },
        "oracle_cases": cases,
        "observed_resource_law": {
            "production_logical_carrier_field_cells": "ONE_PLUS2M",
            "resident_after_forward_field_cells": 1,
            "retained_inverse_history_cells": 0,
            "ntt_calls_per_compiler": 6,
            "butterflies_per_compiler": "3M_LOG2_M",
            "allocated_state_is_sublinear_in_q": False,
            "exact_bit_width_independent_of_q": False,
            "oracle_uses_out_of_place_recursive_transform_buffers": True,
            "oracle_derives_production_logical_schedule_not_oracle_peak_memory": True,
            "python_integer_allocator_and_native_library_memory_included": False,
        },
        "controls": {
            "all_fused_gauss_values_match_direct": True,
            "all_kernel_spectra_nonzero": True,
            "all_exact_same_backing_restorations_pass": True,
            "all_reuses_match_fresh": True,
            "all_missing_inverses_fail": True,
            "all_wrong_program_inverses_fail": True,
            "frequency_zero_omission_changes_all_declared_boundaries": True,
            "all_forced_singular_kernel_spectra_rejected": True,
            "all_null_carriers_rejected": True,
            "q5_q7_direct_original_relation_boundaries_match": True,
        },
        "claim_ceiling": (
            "FOURTEEN_DECLARED_FIELD_PROGRAM_CASES_INDEPENDENT_RECURSIVE_"
            "NTT_DIRECT_GAUSS_AND_FUSED_TWO_SEGMENT_TRANSACTION_PARITY_"
            "WITH_Q5_Q7_DIRECT_ORIGINAL_RELATION_BOUNDARY_REEXECUTION"
        ),
        "strict_boundaries": {
            "catvm_custody": False,
            "sublinear_allocated_state": False,
            "fixed_exact_bit_width": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    text = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
