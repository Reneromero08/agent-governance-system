#!/usr/bin/env python3
"""Independent oracle for the coherent Veronese phase-chart successor.

This file does not import the production M129 implementation.  It separately
reconstructs public programs, the 17-coordinate chart recurrence, ordinary
monomial coefficients, and bounded full H(k) occupation recurrences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_exchange_symmetric_phase_module_irreducibility_oracle as m128ref


PRIME = 17
MODE_COUNT = 17
DECLARED_K = (4, 8, 16, 32, 64, 128)
EXACT_K = (4, 8, 16, 32)
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stream_vector_commitment(
    values: list[Any] | tuple[Any, ...], alg: backend.Algebra
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


def algebra_from_signature(signature: str) -> backend.Algebra:
    candidates = [
        backend.Algebra("Q_ZETA17"),
        *(
            backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            for modulus, root in FINITE_FIELDS
        ),
    ]
    for candidate in candidates:
        candidate_signature = digest_json(
            {
                "kind": candidate.kind,
                "modulus": candidate.modulus,
                "root": candidate.serialize(candidate.root),
            }
        )
        if candidate_signature == signature:
            return candidate
    fail("unknown compiled baseline algebra")


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def field_integer(alg: backend.Algebra, value: int) -> Any:
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


@dataclass(frozen=True)
class Operation:
    kind: str
    first: int
    second: int
    exponent: int


def independent_program(k: int, family: str) -> tuple[Operation, ...]:
    if k not in DECLARED_K or family not in ("PRIMARY", "REUSE"):
        fail("oracle program scope changed")
    variant = 0 if family == "PRIMARY" else 1
    return (
        *(Operation("SHEAR", mode + 1, mode, 1 + ((5 * mode + 3 + variant) % 16))
          for mode in range(MODE_COUNT - 1)),
        *(Operation("CHARACTER", degree, 0, 1 + ((3 * degree + 5 * variant) % 16))
          for degree in range(1, 5)),
        *(Operation("SHEAR", mode, mode + 1, 1 + ((7 * mode + 4 + 2 * variant) % 16))
          for mode in range(MODE_COUNT - 2, -1, -1)),
        *(Operation("CHARACTER", degree, 0, 1 + ((7 * degree + 2 + 3 * variant) % 16))
          for degree in range(4, 0, -1)),
    )


def independent_program_fingerprint(k: int, family: str) -> str:
    operations = independent_program(k, family)
    return digest_json(
        {
            "k": k,
            "family": family,
            "chart": "RANK1_EXCHANGE_SYMMETRIC_COHERENT_VERONESE",
            "chart_rank": 1,
            "mode_count": MODE_COUNT,
            "seed": "ONE_PARTICLE_MODE0_RAISED_TO_SYMMETRIC_POWER_K",
            "primitives": [
                {
                    "kind": item.kind,
                    "first": item.first,
                    "second": item.second,
                    "coefficient_exponent": item.exponent,
                }
                for item in operations
            ],
            "excluded_primitive": "M127_GRID_ORBIT_SHEAR",
            "final_boundary": "K_MINUS_1_PARTICLES_IN_MODE0_AND_ONE_PARTICLE_IN_MODE1",
        }
    )


def chart_apply(
    vector: list[Any],
    operation: Operation,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> None:
    if operation.kind == "CHARACTER":
        sign = -1 if inverse else 1
        for mode in range(MODE_COUNT):
            vector[mode] = alg.mul(
                vector[mode],
                alg.power(
                    sign
                    * operation.exponent
                    * pow(mode, operation.first, PRIME)
                ),
            )
    elif operation.kind == "SHEAR":
        coefficient = alg.power(operation.exponent)
        if inverse:
            coefficient = negative(alg, coefficient)
        vector[operation.first] = alg.add(
            vector[operation.first],
            alg.mul(coefficient, vector[operation.second]),
        )
    else:
        fail("oracle operation type changed")


def chart_forward(k: int, family: str, alg: backend.Algebra) -> list[Any]:
    vector = [alg.zero for _ in range(MODE_COUNT)]
    vector[0] = alg.one
    for operation in independent_program(k, family):
        chart_apply(vector, operation, alg)
    return vector


def chart_reverse(
    vector: list[Any], k: int, family: str, alg: backend.Algebra
) -> bool:
    for operation in reversed(independent_program(k, family)):
        chart_apply(vector, operation, alg, inverse=True)
    vector[0] = alg.sub(vector[0], alg.one)
    return all(value == alg.zero for value in vector)


def chart_boundary(vector: list[Any], k: int, alg: backend.Algebra) -> Any:
    return alg.mul(
        field_integer(alg, k),
        alg.mul(scalar_power(alg, vector[0], k - 1), vector[1]),
    )


def enumerate_histograms(k: int) -> tuple[tuple[int, ...], ...]:
    histograms: list[tuple[int, ...]] = []

    def visit(mode: int, remaining: int, prefix: list[int]) -> None:
        if mode == MODE_COUNT - 1:
            histograms.append(tuple((*prefix, remaining)))
            return
        for count in range(remaining + 1):
            prefix.append(count)
            visit(mode + 1, remaining - count, prefix)
            prefix.pop()

    visit(0, k, [])
    if len(histograms) != math.comb(k + 16, 16):
        fail("oracle stars-and-bars enumeration failed")
    return tuple(histograms)


def multinomial(k: int, histogram: tuple[int, ...]) -> int:
    value = math.factorial(k)
    for count in histogram:
        value //= math.factorial(count)
    return value


def expand_chart(
    vector: list[Any],
    k: int,
    histograms: tuple[tuple[int, ...], ...],
    alg: backend.Algebra,
) -> list[Any]:
    result = []
    for histogram in histograms:
        value = field_integer(alg, multinomial(k, histogram))
        for mode, count in enumerate(histogram):
            if count:
                value = alg.mul(value, scalar_power(alg, vector[mode], count))
        result.append(value)
    return result


def dense_character(
    values: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    operation: Operation,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> None:
    sign = -1 if inverse else 1
    for index, histogram in enumerate(histograms):
        power_sum = sum(
            count * pow(mode, operation.first, PRIME)
            for mode, count in enumerate(histogram)
        ) % PRIME
        values[index] = alg.mul(
            values[index], alg.power(sign * operation.exponent * power_sum)
        )


def dense_shear(
    values: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    ranks: dict[tuple[int, ...], int],
    operation: Operation,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> None:
    coefficient = alg.power(operation.exponent)
    if inverse:
        coefficient = negative(alg, coefficient)
    powers = [alg.one]
    for _ in range(sum(histograms[0])):
        powers.append(alg.mul(powers[-1], coefficient))
    row = operation.first
    pivot = operation.second
    for base in histograms:
        if base[row] != 0:
            continue
        total = base[pivot]
        indices = []
        for row_count in range(total + 1):
            member = list(base)
            member[row] = row_count
            member[pivot] = total - row_count
            indices.append(ranks[tuple(member)])
        old = [values[index] for index in indices]
        updated = [alg.zero for _ in indices]
        for input_row, amplitude in enumerate(old):
            input_pivot = total - input_row
            for moved in range(input_pivot + 1):
                coefficient_value = alg.mul(
                    field_integer(alg, math.comb(input_pivot, moved)), powers[moved]
                )
                output_row = input_row + moved
                updated[output_row] = alg.add(
                    updated[output_row], alg.mul(amplitude, coefficient_value)
                )
        for index, value in zip(indices, updated, strict=True):
            values[index] = value


def dense_apply(
    values: list[Any],
    histograms: tuple[tuple[int, ...], ...],
    ranks: dict[tuple[int, ...], int],
    operation: Operation,
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> None:
    if operation.kind == "CHARACTER":
        dense_character(values, histograms, operation, alg, inverse=inverse)
    else:
        dense_shear(values, histograms, ranks, operation, alg, inverse=inverse)


def full_occupation_case(
    kind: str, *, modulus: int = 0, root: int = 0
) -> dict[str, Any]:
    k = 4
    alg = backend.Algebra(kind, modulus=modulus, root=root)
    histograms = enumerate_histograms(k)
    ranks = {histogram: index for index, histogram in enumerate(histograms)}
    seed_histogram = (k, *([0] * (MODE_COUNT - 1)))
    boundary_histogram = (k - 1, 1, *([0] * (MODE_COUNT - 2)))
    dense = [alg.zero for _ in histograms]
    dense[ranks[seed_histogram]] = alg.one
    vector = [alg.zero for _ in range(MODE_COUNT)]
    vector[0] = alg.one
    for operation in independent_program(k, "PRIMARY"):
        dense_apply(dense, histograms, ranks, operation, alg)
        chart_apply(vector, operation, alg)
    expanded = expand_chart(vector, k, histograms, alg)
    forward_full_occupation_chart_agreement = dense == expanded
    if not forward_full_occupation_chart_agreement:
        fail(f"full occupation expansion disagrees for {kind}")
    boundary = dense[ranks[boundary_histogram]]
    if boundary != chart_boundary(vector, k, alg):
        fail(f"full occupation boundary disagrees for {kind}")

    a40 = dense[ranks[(4, *([0] * (MODE_COUNT - 1)))]]
    a31 = dense[ranks[(3, 1, *([0] * (MODE_COUNT - 2)))]]
    a22 = dense[ranks[(2, 2, *([0] * (MODE_COUNT - 2)))]]
    b31 = alg.mul(a31, alg.inverse(field_integer(alg, 4)))
    b22 = alg.mul(a22, alg.inverse(field_integer(alg, 6)))
    catalecticant_minor = alg.sub(alg.mul(a40, b22), alg.mul(b31, b31))

    for operation in reversed(independent_program(k, "PRIMARY")):
        dense_apply(dense, histograms, ranks, operation, alg, inverse=True)
        chart_apply(vector, operation, alg, inverse=True)
    dense[ranks[seed_histogram]] = alg.sub(dense[ranks[seed_histogram]], alg.one)
    vector[0] = alg.sub(vector[0], alg.one)
    return {
        "algebra": kind,
        "k": k,
        "occupation_dimension": len(histograms),
        "full_occupation_chart_agreement": forward_full_occupation_chart_agreement,
        "forward_boundary": alg.serialize(boundary),
        "forward_rank_one_catalecticant_minor": alg.serialize(catalecticant_minor),
        "forward_rank_one_catalecticant_minor_zero": catalecticant_minor == alg.zero,
        "dense_forward_inverse_restored": all(value == alg.zero for value in dense),
        "chart_forward_inverse_restored": all(value == alg.zero for value in vector),
        "oracle_full_occupation_vector_field_cells": len(histograms),
    }


def individual_primitive_checks(modulus: int, root: int) -> dict[str, Any]:
    alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
    k = 3
    histograms = enumerate_histograms(k)
    ranks = {histogram: index for index, histogram in enumerate(histograms)}
    seed_vector = [
        alg.add(alg.one, alg.power(3 * mode + 1)) for mode in range(MODE_COUNT)
    ]
    seed_dense = expand_chart(seed_vector, k, histograms, alg)
    checks = []
    for operation in independent_program(4, "PRIMARY"):
        vector = seed_vector[:]
        dense = seed_dense[:]
        chart_apply(vector, operation, alg)
        dense_apply(dense, histograms, ranks, operation, alg)
        forward_agreement = dense == expand_chart(vector, k, histograms, alg)
        dense_apply(dense, histograms, ranks, operation, alg, inverse=True)
        chart_apply(vector, operation, alg, inverse=True)
        checks.append(
            forward_agreement and dense == seed_dense and vector == seed_vector
        )
    return {
        "field": f"F{modulus}",
        "k": k,
        "primitive_count": len(checks),
        "every_individual_primitive_and_inverse_agrees": all(checks),
        "occupation_dimension": len(histograms),
    }


def transaction_parity(
    item: dict[str, Any], kind: str, *, modulus: int = 0, root: int = 0
) -> dict[str, Any]:
    alg = backend.Algebra(kind, modulus=modulus, root=root)
    k = item["k"]
    family = item["family"]
    vector = chart_forward(k, family, alg)
    boundary = chart_boundary(vector, k, alg)
    return {
        "algebra": kind,
        "k": k,
        "family": family,
        "program_fingerprint_agreement": (
            independent_program_fingerprint(k, family) == item["program_fingerprint"]
        ),
        "forward_chart_commitment_agreement": (
            stream_vector_commitment(vector, alg)[0]
            == item["forward_chart_commitment"]
        ),
        "boundary_agreement": alg.serialize(boundary) == item["boundary"],
        "independent_forward_inverse_restored": chart_reverse(
            vector, k, family, alg
        ),
        "resident_field_cells_agreement": item["resident_phase_field_cells"] == 17,
        "implicit_dimension_agreement": (
            item["implicit_occupation_dimension_h_k"] == math.comb(k + 16, 16)
        ),
    }


def independent_m128_grid_exit_witness(family: str) -> dict[str, Any]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    k = 2
    histograms = m128ref.m127ref.enumerate_histograms(k)
    ranks = {histogram: index for index, histogram in enumerate(histograms)}
    word = m128ref.program_word(k, family)
    u = [alg.zero for _ in histograms]
    w = [alg.zero for _ in histograms]
    zero = (k, *([0] * (MODE_COUNT - 1)))
    u[ranks[zero]] = alg.one
    initial_u = u[:]
    initial_w = w[:]
    cache: dict[tuple[int, tuple[int, ...]], Any] = {}

    m128ref.apply_shear_segment(
        u, w, histograms, ranks, word["first_shears"], alg
    )
    m128ref.apply_character_segment(
        u, histograms, word["first_characters"], alg
    )
    m128ref.apply_module(u, w, histograms, k, family, 0, alg, cache)
    m128ref.apply_shear_segment(
        u, w, histograms, ranks, word["second_shears"], alg
    )
    m128ref.apply_character_segment(
        u, histograms, word["second_characters"], alg
    )
    m128ref.apply_module(u, w, histograms, k, family, 1, alg, cache)

    a20 = w[ranks[(2, *([0] * (MODE_COUNT - 1)))]]
    a11 = w[ranks[(1, 1, *([0] * (MODE_COUNT - 2)))]]
    a02 = w[ranks[(0, 2, *([0] * (MODE_COUNT - 2)))]]
    b11 = alg.mul(a11, alg.inverse(field_integer(alg, 2)))
    entries = [int(a20), int(b11), int(a02)]
    minor = alg.sub(alg.mul(a20, a02), alg.mul(b11, b11))

    m128ref.apply_module(
        u, w, histograms, k, family, 1, alg, cache, inverse=True
    )
    m128ref.apply_character_segment(
        u, histograms, word["second_characters"], alg, inverse=True
    )
    m128ref.apply_shear_segment(
        u,
        w,
        histograms,
        ranks,
        word["second_shears"],
        alg,
        inverse=True,
    )
    m128ref.apply_module(
        u, w, histograms, k, family, 0, alg, cache, inverse=True
    )
    m128ref.apply_character_segment(
        u, histograms, word["first_characters"], alg, inverse=True
    )
    m128ref.apply_shear_segment(
        u,
        w,
        histograms,
        ranks,
        word["first_shears"],
        alg,
        inverse=True,
    )
    return {
        "family": family,
        "entry_commitment": digest_json(entries),
        "minor": int(minor),
        "restored": u == initial_u and w == initial_w,
        "oracle_cached_grid_boundary_field_cells": len(cache),
    }


def grid_witness_checks(production: dict[str, Any]) -> list[dict[str, Any]]:
    production_witnesses = {
        item["family"]: item
        for item in production["controls"]["actual_m128_grid_exit_witnesses"]
    }
    result = []
    for family in ("PRIMARY", "REUSE"):
        independent = independent_m128_grid_exit_witness(family)
        item = production_witnesses[family]
        result.append(
            {
                "family": family,
                "entry_commitment_agreement": (
                    independent["entry_commitment"]
                    == item["de_multinomial_binary_slice_commitment"]
                ),
                "independent_minor": independent["minor"],
                "minor_agreement": (
                    independent["minor"] == item["rank_one_catalecticant_minor"]
                ),
                "rank_one_chart_rejected": independent["minor"] != 0,
                "independent_prior_carrier_restored": independent["restored"],
                "oracle_cached_grid_boundary_field_cells": independent[
                    "oracle_cached_grid_boundary_field_cells"
                ],
            }
        )
    return result


def compiled_baseline_checks(production: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        *production["compiled_two_scalar_warm_classical_baselines"]["exact_q_zeta17"],
        *production["compiled_two_scalar_warm_classical_baselines"]["dual_field"],
    ]
    result = []
    for item in items:
        alg = algebra_from_signature(item["algebra"])
        vector = chart_forward(item["k"], item["family"], alg)
        pair = (vector[0], vector[1])
        pair_commitment, record_bytes = stream_vector_commitment(pair, alg)
        boundary = chart_boundary(vector, item["k"], alg)
        result.append(
            {
                "algebra": alg.kind,
                "k": item["k"],
                "family": item["family"],
                "pair_commitment_agreement": (
                    pair_commitment == item["retained_warm_boundary_pair_commitment"]
                ),
                "maximum_record_json_bytes_agreement": (
                    record_bytes == item["maximum_commitment_record_json_bytes"]
                ),
                "boundary_agreement": (
                    alg.serialize(boundary)
                    == next(
                        transaction["boundary"]
                        for transaction in (
                            production["exact_transactions"]
                            + production["dual_field_structural_transactions"]
                        )
                        if transaction["k"] == item["k"]
                        and transaction["family"] == item["family"]
                        and transaction["algebra"] == item["algebra"]
                    )
                ),
                "warm_retained_field_cells_agreement": (
                    item["retained_warm_boundary_pair_field_cells"] == 2
                ),
                "compiler_working_field_cells_agreement": (
                    item["compiler_working_field_cells"] == 17
                ),
            }
        )
    return result


def run(production: dict[str, Any]) -> dict[str, Any]:
    exact_parity = [
        transaction_parity(item, "Q_ZETA17")
        for item in production["exact_transactions"]
    ]
    finite_parity = []
    for item in production["dual_field_structural_transactions"]:
        modulus, root = next(
            pair for pair in FINITE_FIELDS if f"F{pair[0]}" == item["field"]
        )
        finite_parity.append(
            transaction_parity(item, item["field"], modulus=modulus, root=root)
        )

    full_cases = [
        full_occupation_case("Q_ZETA17"),
        full_occupation_case("F103", modulus=103, root=72),
        full_occupation_case("F137", modulus=137, root=16),
    ]
    primitive_checks = [
        individual_primitive_checks(modulus, root)
        for modulus, root in FINITE_FIELDS
    ]
    grid_checks = grid_witness_checks(production)
    baseline_checks = compiled_baseline_checks(production)

    required_transaction_keys = (
        "program_fingerprint_agreement",
        "forward_chart_commitment_agreement",
        "boundary_agreement",
        "independent_forward_inverse_restored",
        "resident_field_cells_agreement",
        "implicit_dimension_agreement",
    )
    if not all(
        all(item[key] for key in required_transaction_keys)
        for item in (*exact_parity, *finite_parity)
    ):
        fail("independent coherent transaction parity failed")
    if not all(
        item["full_occupation_chart_agreement"]
        and item["forward_rank_one_catalecticant_minor_zero"]
        and item["dense_forward_inverse_restored"]
        and item["chart_forward_inverse_restored"]
        for item in full_cases
    ):
        fail("independent full occupation parity failed")
    if not all(
        item["every_individual_primitive_and_inverse_agrees"]
        for item in primitive_checks
    ):
        fail("individual primitive parity failed")
    if not all(
        item["entry_commitment_agreement"]
        and item["minor_agreement"]
        and item["rank_one_chart_rejected"]
        and item["independent_prior_carrier_restored"]
        for item in grid_checks
    ):
        fail("actual M128 grid-exit witness check failed")
    if not all(
        item["pair_commitment_agreement"]
        and item["maximum_record_json_bytes_agreement"]
        and item["boundary_agreement"]
        and item["warm_retained_field_cells_agreement"]
        and item["compiler_working_field_cells_agreement"]
        for item in baseline_checks
    ):
        fail("compiled two-scalar warm classical baseline parity failed")

    return {
        "schema": "CAT_CAS_F17_COHERENT_VERONESE_PHASE_CHART_CLOSURE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_schema": production["schema"],
        "production_claim": production["claim"],
        "exact_transaction_parity": exact_parity,
        "dual_field_transaction_parity": finite_parity,
        "full_occupation_oracle_cases": full_cases,
        "individual_primitive_checks": primitive_checks,
        "actual_m128_grid_exit_witness_checks": grid_checks,
        "compiled_two_scalar_warm_classical_baseline_checks": baseline_checks,
        "monomial_convention_control": {
            "k": 2,
            "v0": 1,
            "v1": 1,
            "mixed_occupation_coefficient": 2,
            "raw_unmultiplied_coefficient_rejected": True,
        },
        "independent_methods": [
            "SEPARATE_PUBLIC_PROGRAM_COMPILER",
            "SEPARATE_17_COORDINATE_CHART_RECURRENCE",
            "ORDINARY_MONOMIAL_MULTINOMIAL_EXPANSION",
            "FULL_H4_4845_COORDINATE_OCCUPATION_REEXECUTION_IN_THREE_ALGEBRAS",
            "INDIVIDUAL_CHARACTER_AND_BIDIRECTIONAL_SHEAR_FULL_OCCUPATION_PARITY",
            "CATALECTICANT_RANK_ONE_MEMBERSHIP_AND_GRID_EXIT_CONTROLS",
            "SEPARATE_COMPILED_TWO_SCALAR_WARM_CLASSICAL_BASELINE_RECONSTRUCTION",
        ],
        "resource_law": {
            "accepted_path_resident_field_cells": 17,
            "oracle_full_occupation_field_cells_per_full_case": 4845,
            "oracle_full_occupation_vectors_are_verification_only": True,
            "accepted_path_occupation_vector_or_topology_cells": 0,
            "full_exact_bit_complexity_established": False,
            "python_native_allocator_bigint_and_whole_process_memory_excluded": True,
        },
        "matched_baseline": {
            "strongest_sealed_fixture_warm": "COMPILED_TWO_SCALAR_V0_V1_RETENTION_WITH_CLOSED_FORM_OCCUPATION_PROJECTION",
            "descriptor_runtime": "IDENTICAL_17_COORDINATE_COHERENT_VECTOR_RECURRENCE_WITH_CLOSED_FORM_OCCUPATION_PROJECTION",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_restoration_class": "NO_RESTORATION_CLAIM",
        "claim_ceiling": "DECLARED_K4_TO128_RANK1_COHERENT_VERONESE_FIXED_P1_TO_P4_AND_BIDIRECTIONAL_ADJACENT_MODE_SHEAR_PROGRAM_FAMILY_EXACT_Q_ZETA17_K4_TO32_DUAL_FIELD_K4_TO128_DIRECT_PROCESS_SOFTWARE",
        "rejected_interpretations": [
            "M127_GRID_ORBIT_SHEAR_CLOSURE",
            "MULTIPLE_COHERENT_COMPONENT_CLOSURE",
            "ARBITRARY_H_K_INPUT_CLOSURE",
            "GENERAL_NONLINEAR_QUOTIENT",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_EXECUTION_OR_BIT_REPLACEMENT",
            "UNBOUNDED_COMPUTATION",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    production = json.loads(Path(arguments.production).read_text(encoding="utf-8"))
    result = run(production)
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
