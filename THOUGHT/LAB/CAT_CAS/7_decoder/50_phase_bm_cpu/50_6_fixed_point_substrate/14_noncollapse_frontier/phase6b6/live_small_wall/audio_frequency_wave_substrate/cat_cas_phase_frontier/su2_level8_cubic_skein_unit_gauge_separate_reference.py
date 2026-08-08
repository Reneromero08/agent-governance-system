#!/usr/bin/env python3
"""Separate exact reference for M220.

The implementation uses the prior standalone M219 reference substrate, not
M219 or M220 production.  Unit construction, inversion, all-embedding trace,
line search, gauge selection, forward/inverse execution, and reuse are
implemented here independently.
"""

from __future__ import annotations

import json
import hashlib
import math
import sys
from dataclasses import dataclass
from fractions import Fraction

import su2_level8_topology_local_cubic_skein_separate_reference as prior


sys.set_int_max_str_digits(0)
PARAMETERS = (3, 7, 9, 11, 13, 17, 19)
RANK = 7
CASES = (
    *((4, rounds, 0) for rounds in range(1, 5)),
    *((6, rounds, 0) for rounds in range(1, 3)),
    (8, 1, 0),
)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def ledger_bits(ledger: list[int]) -> int:
    return sum(signed_bits(value) for value in ledger)


def integer_content(values: list[prior.E]) -> dict[str, object]:
    global_content = 0
    element_contents = []
    for value in values:
        content = 0
        for coefficient in value.coordinates:
            if coefficient.denominator != 1:
                raise RuntimeError("standalone content input left the integral ring")
            content = math.gcd(content, abs(coefficient.numerator))
            global_content = math.gcd(global_content, abs(coefficient.numerator))
        if content:
            element_contents.append(content)
    return {
        "global_integer_coordinate_content": global_content,
        "all_nonzero_element_integer_coordinate_contents_one": all(
            content == 1 for content in element_contents
        ),
    }


def inverse(value: prior.E) -> prior.E:
    if value == prior.ZERO:
        raise ZeroDivisionError("zero cyclotomic element")
    columns = [(value * prior.E.root(index)).coordinates for index in range(16)]
    matrix = [
        [columns[column][row] for column in range(16)] + [Fraction(row == 0)]
        for row in range(16)
    ]
    for column in range(16):
        pivot = next(row for row in range(column, 16) if matrix[row][column])
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [entry / scale for entry in matrix[column]]
        for row in range(16):
            if row == column or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                entry - factor * basis
                for entry, basis in zip(matrix[row], matrix[column])
            ]
    return prior.E(tuple(matrix[row][-1] for row in range(16)))


def conjugate(value: prior.E) -> prior.E:
    coefficients = [Fraction(0)] * 40
    for exponent, coefficient in enumerate(value.coordinates):
        coefficients[(-exponent) % 40] += coefficient
    return prior.E(prior.reduce_root(coefficients))


def mobius(value: int) -> int:
    parity = 0
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            value //= divisor
            parity += 1
            if value % divisor == 0:
                return 0
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        parity += 1
    return -1 if parity % 2 else 1


def phi(value: int) -> int:
    result = value
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            while value % divisor == 0:
                value //= divisor
            result -= result // divisor
        divisor += 1
    if value > 1:
        result -= result // value
    return result


TRACE = tuple(
    mobius(40 // math.gcd(40, power))
    * phi(40)
    // phi(40 // math.gcd(40, power))
    for power in range(16)
)


def trace(value: prior.E) -> int:
    result = sum(
        coefficient * TRACE[index]
        for index, coefficient in enumerate(value.coordinates)
    )
    if result.denominator != 1:
        raise RuntimeError("standalone trace is not integral")
    return result.numerator


def power(base: prior.E, exponent: int) -> prior.E:
    result = prior.ONE
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = result * factor
        remaining >>= 1
        if remaining:
            factor = factor * factor
    return result


@dataclass(frozen=True)
class Unit:
    value: prior.E
    reciprocal: prior.E
    norm: prior.E
    reciprocal_norm: prior.E


def units() -> tuple[Unit, ...]:
    result = []
    for parameter in PARAMETERS:
        value = sum((prior.E.root(index) for index in range(parameter)), prior.ZERO)
        reciprocal = inverse(value)
        if value * reciprocal != prior.ONE:
            raise RuntimeError("standalone unit inverse failed")
        if any(item.denominator != 1 for item in reciprocal.coordinates):
            raise RuntimeError("standalone unit inverse is not integral")
        result.append(
            Unit(
                value,
                reciprocal,
                value * conjugate(value),
                reciprocal * conjugate(reciprocal),
            )
        )
    return tuple(result)


UNITS = units()


def norm_element(values: list[prior.E]) -> prior.E:
    result = prior.ZERO
    for value in values:
        result = result + value * conjugate(value)
    return result


def norm_factor(unit: Unit, exponent: int) -> prior.E:
    return (
        power(unit.reciprocal_norm, exponent)
        if exponent >= 0
        else power(unit.norm, -exponent)
    )


def residual_factor(unit: Unit, exponent: int) -> prior.E:
    return (
        power(unit.reciprocal, exponent)
        if exponent >= 0
        else power(unit.value, -exponent)
    )


def represented_scale(unit: Unit, exponent: int) -> prior.E:
    return (
        power(unit.value, exponent)
        if exponent >= 0
        else power(unit.reciprocal, -exponent)
    )


def line_minimum(norm: prior.E, unit: Unit) -> tuple[int, int]:
    cache = {0: trace(norm)}

    def energy(exponent: int) -> int:
        if exponent not in cache:
            cache[exponent] = trace(norm_factor(unit, exponent) * norm)
        return cache[exponent]

    if min(energy(-1), energy(1)) >= energy(0):
        return 0, energy(0)
    sign = 1 if energy(1) < energy(-1) else -1
    previous, current = 0, sign
    for _ in range(32):
        following = 2 * current
        if energy(following) >= energy(current):
            low, high = sorted((previous, following))
            break
        previous, current = current, following
    else:
        raise RuntimeError("standalone unit minimum not bracketed")
    while high - low > 8:
        first = low + (high - low) // 3
        second = high - (high - low) // 3
        if energy(first) <= energy(second):
            high = second - 1
        else:
            low = first + 1
    selected = min(range(low, high + 1), key=lambda item: (energy(item), item))
    return selected, energy(selected)


def balance(raw: list[prior.E]) -> tuple[list[prior.E], list[int], dict[str, object]]:
    norm = norm_element(raw)
    raw_payload = prior.payload_bits(raw)
    zero = [0] * RANK
    best_key = (raw_payload + ledger_bits(zero), raw_payload, trace(norm), tuple(zero))
    best = raw.copy()
    best_ledger = zero
    exponents = []
    for index, unit in enumerate(UNITS):
        exponent, energy = line_minimum(norm, unit)
        exponents.append(exponent)
        if not exponent:
            continue
        factor = residual_factor(unit, exponent)
        candidate = [factor * value for value in raw]
        ledger = [0] * RANK
        ledger[index] = exponent
        candidate_payload = prior.payload_bits(candidate)
        key = (
            candidate_payload + ledger_bits(ledger),
            candidate_payload,
            energy,
            tuple(ledger),
        )
        if key < best_key:
            best_key = key
            best = candidate
            best_ledger = ledger
    return best, best_ledger, {
        "raw_payload_bits": raw_payload,
        "balanced_residual_payload_bits": prior.payload_bits(best),
        "unit_ledger_payload_bits": ledger_bits(best_ledger),
        "balanced_residual_plus_ledger_payload_bits": best_key[0],
        "resident_payload_reduction_bits_before_constant_scratch": (
            raw_payload + ledger_bits(zero) - best_key[0]
        ),
        "selected_unit_ledger": best_ledger,
        "per_direction_trace_energy_minimizing_exponents": exponents,
        "selected_exact_embedding_energy_bits": signed_bits(best_key[2]),
        "selected_exact_embedding_energy_sha256": hashlib.sha256(
            str(best_key[2]).encode("ascii")
        ).hexdigest(),
        "identity_selected": not any(best_ledger),
    }


def materialize(residual: list[prior.E], ledger: list[int]) -> list[prior.E]:
    scale = prior.ONE
    for exponent, unit in zip(ledger, UNITS):
        scale = scale * represented_scale(unit, exponent)
    return [scale * value for value in residual]


def source(topology: prior.Topology) -> list[prior.E]:
    result = [prior.ZERO] * topology.dimension
    result[topology.cup_index] = prior.ONE
    return result


def execute(
    strands: int,
    rounds: int,
    family: int,
    *,
    generation: int = 1,
    residual: list[prior.E] | None = None,
    ledger: list[int] | None = None,
    scratch: list[prior.E] | None = None,
) -> tuple[dict[str, object], list[prior.E], list[int], list[prior.E]]:
    topology = prior.Topology.compile(strands)
    expected_source = source(topology)
    residual = expected_source.copy() if residual is None else residual
    ledger = [0] * RANK if ledger is None else ledger
    scratch = [prior.ZERO] * topology.dimension if scratch is None else scratch
    backings = id(residual), id(ledger), id(scratch)
    last_balance: dict[str, object] = {}
    public_operations = prior.operations(strands, rounds, family)
    for operation in public_operations:
        actual = materialize(residual, ledger)
        prior.apply_gate(actual, scratch, topology, operation)
        prior.apply_shear(actual, topology, operation)
        balanced, selected_ledger, last_balance = balance(actual)
        residual[:] = balanced
        ledger[:] = selected_ledger
        scratch[:] = [prior.ZERO] * topology.dimension
    actual = materialize(residual, ledger)
    forward_commitment = prior.state_commitment(actual)
    boundary = prior.markov_boundary(actual, topology)
    raw_payload = prior.payload_bits(actual)
    for operation in reversed(public_operations):
        actual = materialize(residual, ledger)
        prior.apply_shear(actual, topology, operation, inverse=True)
        prior.apply_gate(
            actual,
            scratch,
            topology,
            prior.Operation(operation.generator, -operation.exponent),
        )
        balanced, selected_ledger, _ = balance(actual)
        residual[:] = balanced
        ledger[:] = selected_ledger
        scratch[:] = [prior.ZERO] * topology.dimension
    result = {
        "boundary_commitment": prior.boundary_commitment(boundary),
        "forward_state_commitment": forward_commitment,
        "forward_raw_payload_bits": raw_payload,
        "final_balance": last_balance,
        "same_residual_backing": id(residual) == backings[0],
        "same_unit_ledger_backing": id(ledger) == backings[1],
        "same_scratch_backing": id(scratch) == backings[2],
        "restoration_error_field_cells": sum(
            left != right for left, right in zip(residual, expected_source)
        ),
        "canonical_post_restoration_state_exact": (
            residual == expected_source
            and not any(ledger)
            and all(value == prior.ZERO for value in scratch)
        ),
        "restoration_generation": generation,
        "baseline_reload_used": False,
    }
    return result, residual, ledger, scratch


def case(strands: int, rounds: int, family: int) -> dict[str, object]:
    result, _, _, _ = execute(strands, rounds, family)
    direct_topology, direct, _ = prior.forward(strands, rounds, family)
    if result["forward_state_commitment"] != prior.state_commitment(direct):
        raise RuntimeError("standalone gauge changed M219 state")
    return {
        "strands": strands,
        "rounds": rounds,
        "family": family,
        "link_pattern_cells": direct_topology.dimension,
        **result,
        "direct_m219_state_commitment_agreement": True,
        **integer_content(direct),
    }


def reuse() -> dict[str, object]:
    topology = prior.Topology.compile(4)
    residual = source(topology)
    ledger = [0] * RANK
    scratch = [prior.ZERO] * topology.dimension
    primary, residual, ledger, scratch = execute(
        4, 4, 0, generation=1, residual=residual, ledger=ledger, scratch=scratch
    )
    reused, residual, ledger, scratch = execute(
        4, 2, 1, generation=2, residual=residual, ledger=ledger, scratch=scratch
    )
    fresh, _, _, _ = execute(4, 2, 1)
    return {
        "primary": primary,
        "reuse": reused,
        "fresh_reuse": fresh,
        "fresh_restored_reuse_boundary_agreement": (
            reused["boundary_commitment"] == fresh["boundary_commitment"]
        ),
        "fresh_restored_reuse_state_agreement": (
            reused["forward_state_commitment"] == fresh["forward_state_commitment"]
        ),
        "restoration_generation_after_reuse": 2,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "schema": "cat_cas.su2_level8_cubic_skein_unit_gauge_reference.v1",
                "imports_m220_production": False,
                "uses_prior_standalone_m219_reference_substrate": True,
                "cases": [case(*item) for item in CASES],
                "reuse": reuse(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
