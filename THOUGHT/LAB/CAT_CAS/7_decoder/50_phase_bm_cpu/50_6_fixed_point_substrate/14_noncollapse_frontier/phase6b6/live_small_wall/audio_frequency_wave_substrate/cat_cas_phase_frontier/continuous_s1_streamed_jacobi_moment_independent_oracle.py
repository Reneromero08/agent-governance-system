#!/usr/bin/env python3
"""Independent exact oracle for streamed Jacobi moment rematerialization.

This file deliberately imports no CAT_CAS module.  It reconstructs the public
center families and operation words, evaluates the boundary once by a direct
factor recurrence, and evaluates it again by an independently written
reverse-scan Jacobi-log recurrence with one add/subtract scratch value.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction


ORDER = 24
Complex = tuple[Fraction, Fraction]
Series = dict[tuple[int, int], Complex]
Operation = tuple[str, int]
ZERO: Complex = (Fraction(0), Fraction(0))
ONE: Complex = (Fraction(1), Fraction(0))
ROTATIONS: tuple[Complex, ...] = (
    (Fraction(-3, 5), Fraction(4, 5)),
    (Fraction(-4, 5), Fraction(3, 5)),
)


def add(left: Complex, right: Complex) -> Complex:
    return left[0] + right[0], left[1] + right[1]


def negate(value: Complex) -> Complex:
    return -value[0], -value[1]


def multiply(left: Complex, right: Complex) -> Complex:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def scale(value: Complex, scalar: Fraction) -> Complex:
    return value[0] * scalar, value[1] * scalar


def conjugate(value: Complex) -> Complex:
    return value[0], -value[1]


def power(value: Complex, exponent: int) -> Complex:
    if exponent < 0:
        return power(conjugate(value), -exponent)
    result = ONE
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = multiply(result, factor)
        factor = multiply(factor, factor)
        remaining >>= 1
    return result


def unit_phase(parameter: int) -> Complex:
    value = Fraction(parameter)
    denominator = 1 + value * value
    return (1 - value * value) / denominator, 2 * value / denominator


def source_centers(count: int, family: int) -> list[Complex]:
    parameters = (
        [2 * index + 1 for index in range(1, count + 1)]
        if family == 0
        else [index * index + index + 1 for index in range(1, count + 1)]
    )
    return [unit_phase(parameter) for parameter in parameters]


def public_program(count: int, family: int) -> tuple[Operation, ...]:
    operations: list[Operation] = []
    for index in range(count):
        ingest = ("INGEST", index)
        if index % 4 == (family + 1) % 4:
            operations.extend((("ROTATE", (index + family) % 2), ingest))
        elif index % 5 == (family + 2) % 5:
            operations.extend(
                (ingest, ("ROTATE", (index + family + 1) % 2))
            )
        else:
            operations.append(ingest)
    return tuple(operations)


def forward_effective_centers(
    source: list[Complex], operations: tuple[Operation, ...]
) -> list[Complex]:
    active: list[Complex] = []
    for kind, parameter in operations:
        if kind == "ROTATE":
            active = [
                multiply(center, ROTATIONS[parameter]) for center in active
            ]
        elif kind == "INGEST":
            active.append(source[parameter])
        else:
            raise ValueError("unknown operation")
    return active


def series_add_scaled(target: Series, source: Series, scalar: Fraction) -> None:
    for key, value in source.items():
        updated = add(target.get(key, ZERO), scale(value, scalar))
        if updated == ZERO:
            target.pop(key, None)
        else:
            target[key] = updated


def series_multiply(
    left: Series, right: Series, order: int, counter: list[int] | None = None
) -> Series:
    result: Series = {}
    for (left_harmonic, left_exponent), left_value in left.items():
        for (right_harmonic, right_exponent), right_value in right.items():
            exponent = left_exponent + right_exponent
            if exponent > order:
                continue
            key = left_harmonic + right_harmonic, exponent
            result[key] = add(
                result.get(key, ZERO), multiply(left_value, right_value)
            )
            if counter is not None:
                counter[0] += 1
    return {key: value for key, value in result.items() if value != ZERO}


def universal_log(order: int, products: list[int]) -> Series:
    radius = math.isqrt(order)
    augmentation: Series = {
        (mode, mode * mode): ONE
        for mode in range(-radius, radius + 1)
        if mode
    }
    term: Series = {(0, 0): ONE}
    logarithm: Series = {}
    for index in range(1, order + 1):
        term = series_multiply(term, augmentation, order, products)
        series_add_scaled(
            logarithm,
            term,
            Fraction(1 if index % 2 else -1, index),
        )
    return logarithm


def direct_factor_jet(
    centers: list[Complex], order: int
) -> tuple[Complex, ...]:
    table: Series = {(0, 0): ONE}
    for center in centers:
        updated: Series = {}
        for (harmonic, exponent), coefficient in table.items():
            radius = math.isqrt(order - exponent)
            for mode in range(-radius, radius + 1):
                key = harmonic + mode, exponent + mode * mode
                contribution = multiply(coefficient, power(center, mode))
                updated[key] = add(updated.get(key, ZERO), contribution)
        table = {key: value for key, value in updated.items() if value != ZERO}
    return tuple(table.get((1, exponent), ZERO) for exponent in range(order + 1))


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload_bits(values: list[Complex] | tuple[Complex, ...]) -> int:
    return sum(
        signed_bits(component.numerator) + component.denominator.bit_length()
        for value in values
        for component in value
    )


def series_payload_bits(series: Series) -> int:
    return payload_bits(list(series.values()))


def series_key_bits(series: Series) -> int:
    return sum(
        signed_bits(harmonic) + signed_bits(exponent)
        for harmonic, exponent in series
    )


def token(value: Complex) -> str:
    return (
        f"{value[0].numerator}/{value[0].denominator}:"
        f"{value[1].numerator}/{value[1].denominator}"
    )


def commitment(values: tuple[Complex, ...]) -> str:
    return hashlib.sha256(
        "|".join(token(value) for value in values).encode("ascii")
    ).hexdigest()


@dataclass
class ScanCounts:
    scans: int = 0
    records: int = 0
    source_visits: int = 0
    power_evaluations: int = 0


def reverse_scan_moment(
    source: list[Complex],
    operations: tuple[Operation, ...],
    moment: int,
    counts: ScanCounts,
) -> Complex:
    suffix = ONE
    result = ZERO
    counts.scans += 1
    for kind, parameter in reversed(operations):
        counts.records += 1
        if kind == "ROTATE":
            suffix = multiply(suffix, ROTATIONS[parameter])
        elif kind == "INGEST":
            counts.source_visits += 1
            effective = multiply(source[parameter], suffix)
            if moment:
                effective = power(effective, moment)
                counts.power_evaluations += 1
            else:
                effective = ONE
            result = add(result, effective)
        else:
            raise ValueError("unknown operation")
    return result


def streamed_jacobi_jet(
    source: list[Complex], operations: tuple[Operation, ...], order: int
) -> tuple[tuple[Complex, ...], dict[str, int | bool]]:
    counts = ScanCounts()
    products = [0]
    logarithm = universal_log(order, products)
    weighted: Series = {}
    scratch = ZERO
    scratch_writes = 0
    inverse_writes = 0
    for moment in range((order + 1) // 2 + 1):
        materialized = reverse_scan_moment(
            source, operations, moment, counts
        )
        scratch = add(scratch, materialized)
        scratch_writes += 1
        for harmonic in ((0,) if moment == 0 else (moment, -moment)):
            effective = scratch if harmonic >= 0 else conjugate(scratch)
            for (log_harmonic, exponent), coefficient in logarithm.items():
                if log_harmonic == harmonic:
                    product = multiply(coefficient, effective)
                    if product != ZERO:
                        weighted[harmonic, exponent] = product
        inverse = reverse_scan_moment(source, operations, moment, counts)
        scratch = add(scratch, negate(inverse))
        inverse_writes += 1
        if scratch != ZERO:
            raise AssertionError("independent scratch failed to restore")

    exponential: Series = {(0, 0): ONE}
    term: Series = {(0, 0): ONE}
    peak_cells = 0
    peak_payload = 0
    peak_key_bits = 0
    for index in range(1, order + 1):
        term = series_multiply(term, weighted, order, products)
        term = {
            key: scale(value, Fraction(1, index))
            for key, value in term.items()
        }
        series_add_scaled(exponential, term, Fraction(1))
        peak_cells = max(peak_cells, len(exponential), len(term))
        peak_payload = max(
            peak_payload,
            series_payload_bits(exponential),
            series_payload_bits(term),
        )
        peak_key_bits = max(
            peak_key_bits,
            series_key_bits(exponential),
            series_key_bits(term),
        )
    jet = tuple(
        exponential.get((1, exponent), ZERO)
        for exponent in range(order + 1)
    )
    metrics: dict[str, int | bool] = {
        "scratch_restored": scratch == ZERO,
        "public_operation_scans": counts.scans,
        "public_operation_records_visited": counts.records,
        "source_center_visits": counts.source_visits,
        "center_power_evaluations": counts.power_evaluations,
        "scratch_writes": scratch_writes,
        "scratch_inverse_writes": inverse_writes,
        "retained_moment_vector_cells": 0,
        "moment_scratch_cells": 1,
        "universal_log_cells": len(logarithm),
        "weighted_log_cells": len(weighted),
        "weighted_log_payload_bits": series_payload_bits(weighted),
        "weighted_log_key_bits": series_key_bits(weighted),
        "peak_exponential_cells": peak_cells,
        "peak_exponential_payload_bits": peak_payload,
        "peak_exponential_key_bits": peak_key_bits,
        "series_products": products[0],
    }
    return jet, metrics


def verify_case(count: int, family: int) -> dict[str, object]:
    source = source_centers(count, family)
    original = source.copy()
    operations = public_program(count, family)
    effective = forward_effective_centers(source, operations)
    direct = direct_factor_jet(effective, ORDER)
    streamed, metrics = streamed_jacobi_jet(source, operations, ORDER)
    if streamed != direct:
        raise AssertionError("streamed Jacobi and direct-factor boundaries differ")
    return {
        "count": count,
        "family": family,
        "public_operation_records": len(operations),
        "source_payload_bits": payload_bits(source),
        "source_unchanged": source == original,
        "boundary_commitment": commitment(streamed),
        "direct_factor_boundary_commitment": commitment(direct),
        "metrics": metrics,
    }


def main() -> None:
    primary = verify_case(24, 0)
    reuse = verify_case(17, 1)
    expected_primary_metrics = {
        "scratch_restored": True,
        "public_operation_scans": 26,
        "public_operation_records_visited": 884,
        "source_center_visits": 624,
        "center_power_evaluations": 576,
        "scratch_writes": 13,
        "scratch_inverse_writes": 13,
        "retained_moment_vector_cells": 0,
        "moment_scratch_cells": 1,
        "universal_log_cells": 110,
        "weighted_log_cells": 86,
        "weighted_log_payload_bits": 172243,
        "weighted_log_key_bits": 673,
        "peak_exponential_cells": 325,
        "peak_exponential_payload_bits": 2480635,
        "peak_exponential_key_bits": 3143,
        "series_products": 106586,
    }
    if primary["metrics"] != expected_primary_metrics:
        raise AssertionError("independent primary resource tuple drifted")
    result = {
        "schema": "cat_cas.continuous_s1_streamed_jacobi_oracle.v1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_imports_cat_cas_modules": False,
        "independent_algorithms": [
            "FORWARD_EFFECTIVE_CENTER_RECONSTRUCTION",
            "DIRECT_FACTOR_BY_FACTOR_SPARSE_QJET_CONVOLUTION",
            "REVERSE_PUBLIC_WORD_MOMENT_REMATERIALIZATION",
            "INDEPENDENT_JACOBI_LOG_AND_FORMAL_EXPONENTIAL",
        ],
        "primary": primary,
        "reuse": reuse,
        "primary_expected_resource_tuple_reproduced": True,
        "streamed_and_direct_boundaries_match": True,
        "finite_angle_sampling_used": False,
        "full_infinite_theta_scalar_evaluated": False,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
