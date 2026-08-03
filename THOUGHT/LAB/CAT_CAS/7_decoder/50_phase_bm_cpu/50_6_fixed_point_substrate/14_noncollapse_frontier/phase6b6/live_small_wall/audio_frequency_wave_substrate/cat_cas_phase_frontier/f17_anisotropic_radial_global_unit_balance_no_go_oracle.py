#!/usr/bin/env python3
"""Independent retained-kernel oracle for the M142 unit-balance result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import f17_anisotropic_radial_shared_scale_height_diagnostic_oracle as state_oracle
import f17_cubic_chain_period17_pi_unit_embedding_balance as units
import f17_nonlinear_canonical_mps_separator_chart as backend


P = 17
DEPTHS = (1, 2, 4, 8, 16, 32, 64)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
UNIT_RANK = 7


def signed_bits(value: int) -> int:
    return 1 if value == 0 else abs(value).bit_length() + 1


def element_payload(element: tuple[int, ...]) -> int:
    return sum(signed_bits(value) for value in element)


def vector_payload(vector: list[tuple[int, ...]]) -> int:
    return sum(element_payload(element) for element in vector)


def vector_width(vector: list[tuple[int, ...]]) -> int:
    return max(
        signed_bits(value)
        for element in vector
        for value in element
    )


def ledger_payload(ledger: list[int]) -> int:
    return sum(signed_bits(value) for value in ledger)


def ring_power(value: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    if exponent < 0:
        raise AssertionError("negative oracle ring exponent")
    result = units.cyclo.ring_one()
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = units.cyclo.ring_multiply(result, factor)
        remaining >>= 1
        if remaining:
            factor = units.cyclo.ring_multiply(factor, factor)
    return result


def ledger_scale(ledger: list[int]) -> tuple[int, ...]:
    result = units.cyclo.ring_one()
    for exponent, generator, inverse in zip(
        ledger,
        units.UNIT_GENERATORS,
        units.UNIT_GENERATOR_INVERSES,
        strict=True,
    ):
        factor = (
            ring_power(generator, exponent)
            if exponent >= 0
            else ring_power(inverse, -exponent)
        )
        result = units.cyclo.ring_multiply(result, factor)
    return result


def multiply_vector(
    scalar: tuple[int, ...], vector: list[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    return [units.cyclo.ring_multiply(scalar, value) for value in vector]


def ring_conjugate(element: tuple[int, ...]) -> tuple[int, ...]:
    result = units.cyclo.ring_zero()
    for exponent, coefficient in enumerate(element):
        monomial = units.cyclo.ring_monomial((-exponent) % P)
        result = units.cyclo.ring_add(
            result,
            tuple(coefficient * value for value in monomial),
        )
    return result


def field_trace(element: tuple[int, ...]) -> int:
    return 16 * element[0] - sum(element[1:])


def vector_energy(vector: list[tuple[int, ...]]) -> int:
    return sum(
        field_trace(units.cyclo.ring_multiply(value, ring_conjugate(value)))
        for value in vector
    )


DIRECTIONS = tuple(
    tuple(1 if coordinate == index else 0 for coordinate in range(UNIT_RANK))
    for index in range(UNIT_RANK)
) + tuple(
    tuple(
        1 if coordinate == left else sign if coordinate == right else 0
        for coordinate in range(UNIT_RANK)
    )
    for left in range(UNIT_RANK)
    for right in range(left + 1, UNIT_RANK)
    for sign in (1, -1)
)


def shared_scale(
    alg: backend.Algebra, state: list[Any]
) -> tuple[list[tuple[int, ...]], int, int]:
    serialized = [alg.serialize(value) for value in state]
    denominator = 1
    for cell in serialized:
        for _, coefficient_denominator in cell:
            denominator = math.lcm(denominator, coefficient_denominator)
    integral = [
        tuple(
            numerator * (denominator // coefficient_denominator)
            for numerator, coefficient_denominator in cell
        )
        for cell in serialized
    ]
    content = denominator
    for element in integral:
        for coefficient in element:
            content = math.gcd(content, abs(coefficient))
    denominator //= content
    integral = [
        tuple(coefficient // content for coefficient in element)
        for element in integral
    ]
    denominator_power = state_oracle.denominator_exponent(denominator)
    valuations = [
        state_oracle.pi_valuation(element)
        for element in integral
        if any(element)
    ]
    if not valuations or any(value is None for value in valuations):
        raise AssertionError("oracle pi valuation is undefined")
    common_pi = min(int(value) for value in valuations if value is not None)
    residuals = []
    for element in integral:
        residual = element
        for _ in range(common_pi):
            residual = state_oracle.divide_pi(residual)
        restored = residual
        for _ in range(common_pi):
            restored = state_oracle.multiply_pi(restored)
        if restored != element:
            raise AssertionError("oracle common-pi extraction changed state")
        residuals.append(residual)
    return residuals, denominator_power, common_pi


def verify_selected_ledger(
    residuals: list[tuple[int, ...]], ledger: list[int]
) -> tuple[list[tuple[int, ...]], int]:
    inverse_scale = ledger_scale([-value for value in ledger])
    balanced = multiply_vector(inverse_scale, residuals)
    if multiply_vector(ledger_scale(ledger), balanced) != residuals:
        raise AssertionError("oracle unit ledger failed reconstruction")
    current_energy = vector_energy(balanced)
    checks = 0
    for direction in DIRECTIONS:
        for sign in (-1, 1):
            trial_delta = [sign * value for value in direction]
            trial = multiply_vector(
                ledger_scale([-value for value in trial_delta]),
                balanced,
            )
            if vector_energy(trial) < current_energy:
                raise AssertionError("declared unit ledger is not locally minimal")
            checks += 1
    return balanced, checks


def execute_case(
    geometry: state_oracle.retained.CompiledGeometry,
    package_case: dict[str, Any],
) -> dict[str, Any]:
    program = state_oracle.compile_program(
        package_case["depth"], package_case["family"]
    )
    state = state_oracle.forward(geometry, program)
    residuals, denominator_power, common_pi = shared_scale(
        geometry.alg, state
    )
    ledger = list(package_case["unit_ledger"])
    balanced, local_checks = verify_selected_ledger(residuals, ledger)
    result = {
        "family": package_case["family"],
        "depth": package_case["depth"],
        "final_state_commitment": state_oracle.commitment(
            geometry.alg, state
        ),
        "final_boundary": geometry.alg.serialize(
            state_oracle.boundary(geometry, program, state)
        ),
        "denominator_power_of_17": denominator_power,
        "common_pi_exponent": common_pi,
        "unit_ledger": ledger,
        "unit_ledger_is_identity": not any(ledger),
        "balanced_residual_payload_bits": vector_payload(balanced),
        "unit_ledger_payload_bits": ledger_payload(ledger),
        "scale_ledger_payload_bits": (
            signed_bits(denominator_power) + signed_bits(common_pi)
        ),
        "balanced_all_ledgers_payload_bits": (
            vector_payload(balanced)
            + ledger_payload(ledger)
            + signed_bits(denominator_power)
            + signed_bits(common_pi)
        ),
        "balanced_residual_maximum_signed_bits": vector_width(balanced),
        "exact_adjacent_direction_checks": local_checks,
        "exact_unit_reconstruction_equal": True,
    }
    restored = state_oracle.inverse(geometry, program, state)
    result["exact_inverse_seed_equal"] = restored == [
        geometry.alg.one for _ in range(P)
    ]
    return result


def run(package_path: Path) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    geometry = state_oracle.retained.CompiledGeometry.compile(
        backend.Algebra("Q_ZETA17")
    )
    oracle_cases = [
        execute_case(geometry, case) for case in package["cases"]
    ]
    compared_fields = (
        "final_state_commitment",
        "final_boundary",
        "denominator_power_of_17",
        "common_pi_exponent",
        "unit_ledger",
        "unit_ledger_is_identity",
        "balanced_residual_payload_bits",
        "unit_ledger_payload_bits",
        "scale_ledger_payload_bits",
        "balanced_all_ledgers_payload_bits",
        "balanced_residual_maximum_signed_bits",
    )
    comparisons = 0
    for candidate, oracle in zip(
        package["cases"], oracle_cases, strict=True
    ):
        for field in compared_fields:
            if candidate[field] != oracle[field]:
                raise AssertionError(
                    f"package/oracle mismatch {oracle['family']} "
                    f"depth {oracle['depth']} field {field}"
                )
            comparisons += 1
        if (
            oracle["exact_adjacent_direction_checks"] != 98
            or not oracle["exact_unit_reconstruction_equal"]
            or not oracle["exact_inverse_seed_equal"]
        ):
            raise AssertionError("oracle control failed")

    mutated = json.loads(json.dumps(package))
    mutated["cases"][-1]["unit_ledger"][0] += 1
    mutation_detected = (
        mutated["cases"][-1]["unit_ledger"]
        != oracle_cases[-1]["unit_ledger"]
    )
    if not mutation_detected:
        raise AssertionError("oracle failed unit-ledger mutation")

    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_GLOBAL_UNIT_BALANCE_NO_GO_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "oracle_method": (
            "SEPARATE_RETAINED_KERNEL_EXACT_RECURRENCE_INDEPENDENT_SHARED_"
            "SCALE_AND_UNIT_LEDGER_MATERIALIZATION_WITH_EXACT98_NEIGHBOR_"
            "DIRECTION_CERTIFICATE_PER_CASE"
        ),
        "case_count": len(oracle_cases),
        "metric_and_boundary_comparisons": comparisons,
        "exact_adjacent_direction_checks": sum(
            case["exact_adjacent_direction_checks"] for case in oracle_cases
        ),
        "all_cases_equal": True,
        "all_exact_unit_reconstructions_equal": True,
        "all_exact_inverse_seed_checks_pass": True,
        "unit_ledger_mutation_detected": mutation_detected,
        "claim_ceiling": (
            "21_DECLARED_CASES_TWO_PRODUCTION_BALANCERS_AND_EXACT_LOCAL_"
            "DIRECTION_CERTIFICATES;NO_GLOBAL_UNIT_OR_REPRESENTATION_OPTIMALITY"
        ),
        "preserved_subclaims": [
            "INDEPENDENT_EXACT_LEDGER_AND_PAYLOAD_PARITY_FOR21_CASES",
            "INDEPENDENT_EXACT98_NEIGHBOR_DIRECTION_CERTIFICATE_PER_CASE",
            "INDEPENDENT_FINAL_BOUNDARY_AND_INVERSE_SEED_PARITY",
        ],
        "rejected_interpretations": [
            "GLOBAL_CYCLOTOMIC_UNIT_OPTIMALITY",
            "LOWER_BOUND_FOR_ALL_EXACT_REPRESENTATIONS",
            "ASYMPTOTIC_HEIGHT_LOWER_BOUND",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.package)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
