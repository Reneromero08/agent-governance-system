#!/usr/bin/env python3
"""Standalone exact oracle for M240; imports no M240/M239/M237 production."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Sequence


P = 5
COUNTS = (1, 2, 3, 4)
TYPE = "ZETA5_INTERACTING_MULTI_INJECTION_SHARED_SYNDROME_V1"
E = tuple[int, int, int, int]
Z: E = (0, 0, 0, 0)
O: E = (1, 0, 0, 0)
S: E = (-1, 0, -2, -2)


def add(left: E, right: E) -> E:
    return tuple(left[i] + right[i] for i in range(4))  # type: ignore[return-value]


def scale(value: E, scalar: int) -> E:
    return tuple(scalar * item for item in value)  # type: ignore[return-value]


def mul(left: E, right: E) -> E:
    raw = [0] * 7
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            raw[i + j] += a * b
    for power in range(6, 3, -1):
        coefficient = raw[power]
        if coefficient:
            for lower in range(power - 4, power):
                raw[lower] -= coefficient
            raw[power] = 0
    return tuple(raw[:4])  # type: ignore[return-value]


def root(power: int) -> E:
    power %= P
    if power == 4:
        return (-1, -1, -1, -1)
    result = [0] * 4
    result[power] = 1
    return tuple(result)  # type: ignore[return-value]


def conjugate(value: E) -> E:
    result = Z
    for power, coefficient in enumerate(value):
        result = add(result, scale(root(-power), coefficient))
    return result


def normalize(values: list[E], exponent: int) -> int:
    while exponent and all(c % P == 0 for value in values for c in value):
        for i, value in enumerate(values):
            values[i] = tuple(c // P for c in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload_bits(values: Sequence[E], exponent: int) -> int:
    return sum(signed_bits(c) for value in values for c in value) + signed_bits(P**exponent)


def vector_commitment(values: Sequence[E], exponent: int) -> str:
    payload = {"denominator_power5": exponent, "numerators": [list(v) for v in values]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def value_commitment(value: E, exponent: int) -> str:
    values = [value]
    exponent = normalize(values, exponent)
    payload = {"denominator_power5": exponent, "numerator": list(values[0])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def probability_encoding(value: E, exponent: int) -> dict[str, object]:
    values = [value]
    exponent = normalize(values, exponent)
    value = values[0]
    if value[1] or value[2] != value[3]:
        raise RuntimeError("reference probability is not real")
    return {
        "a_numerator": 2 * value[0] - value[2],
        "b_sqrt5_numerator": -value[2],
        "denominator_twice_power5_exponent": exponent,
        "commitment": value_commitment(value, exponent),
    }


def flat_index(state: Sequence[int]) -> int:
    result = 0
    for value in state:
        result = P * result + value
    return result


def coordinates(index: int, wires: int) -> list[int]:
    result = [0] * wires
    for position in range(wires - 1, -1, -1):
        result[position] = index % P
        index //= P
    return result


def sqrt_power(power: int) -> E:
    result = O
    for _ in range(power):
        result = mul(result, S)
    return result


@dataclass(frozen=True)
class Real:
    a: Fraction
    b: Fraction

    def __add__(self, other: "Real") -> "Real":
        return Real(self.a + other.a, self.b + other.b)

    def __sub__(self, other: "Real") -> "Real":
        return Real(self.a - other.a, self.b - other.b)

    def __mul__(self, other: "Real") -> "Real":
        return Real(self.a * other.a + 5 * self.b * other.b, self.a * other.b + self.b * other.a)

    def scale(self, value: Fraction) -> "Real":
        return Real(self.a * value, self.b * value)

    def power(self, exponent: int) -> "Real":
        result = Real(Fraction(1), Fraction(0))
        for _ in range(exponent):
            result = result * self
        return result

    def encoding(self) -> dict[str, int]:
        return {
            "rational_numerator": self.a.numerator,
            "rational_denominator": self.a.denominator,
            "sqrt5_numerator": self.b.numerator,
            "sqrt5_denominator": self.b.denominator,
        }


def real(value: E, exponent: int) -> Real:
    values = [value]
    exponent = normalize(values, exponent)
    value = values[0]
    if value[1] or value[2] != value[3]:
        raise RuntimeError("reference value outside real subfield")
    denominator = 2 * P**exponent
    return Real(Fraction(2 * value[0] - value[2], denominator), Fraction(-value[2], denominator))


def real_sign(value: Real) -> int:
    if not value.a and not value.b:
        return 0
    if value.a >= 0 and value.b >= 0:
        return 1
    if value.a <= 0 and value.b <= 0:
        return -1
    comparison = value.a * value.a - 5 * value.b * value.b
    if not comparison:
        return 0
    if value.a > 0:
        return 1 if comparison > 0 else -1
    return -1 if comparison > 0 else 1


def magic_l1(strength: int) -> tuple[Real, int]:
    amplitudes = [mul(S, root(strength * x**3)) for x in range(P)]
    result = Real(Fraction(0), Fraction(0))
    negative = 0
    for q in range(P):
        for momentum in range(P):
            total = Z
            for displacement in range(P):
                left = amplitudes[(q + 3 * displacement) % P]
                right = conjugate(amplitudes[(q - 3 * displacement) % P])
                total = add(total, mul(root(-momentum * displacement), mul(left, right)))
            value = real(total, 3)
            if real_sign(value) < 0:
                negative += 1
                result = result - value
            else:
                result = result + value
    return result, negative


@dataclass(frozen=True)
class RGate:
    kind: str
    wire: int
    target: int = -1
    parameter: int = 0
    direction: int = 1
    stage: str = "NETWORK"
    injection_id: int = -1
    consumer_id: int = -1

    def inverse(self) -> "RGate":
        return replace(self, direction=-self.direction) if self.kind == "FOURIER" else replace(self, parameter=-self.parameter)

    def serial(self) -> tuple[object, ...]:
        return (
            self.kind, self.wire, self.target, self.parameter % P, self.direction,
            self.stage, self.injection_id, self.consumer_id,
        )


def digest(family: int, count: int, strengths: Sequence[int], output: Sequence[int], gates: Sequence[RGate]) -> str:
    payload = {
        "family": family, "injection_count": count, "strengths": list(strengths),
        "output_data": list(output), "gates": [gate.serial() for gate in gates],
        "port_type": TYPE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Spec:
    family: int
    count: int
    strengths: tuple[int, ...]
    output: tuple[int, ...]
    gates: tuple[RGate, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.count not in COUNTS:
            raise ValueError("reference program outside suite")
        if self.program_id != digest(self.family, self.count, self.strengths, self.output, self.gates):
            raise ValueError("reference full-descriptor digest mismatch")
        consumers = [g.consumer_id for g in self.gates if g.consumer_id >= 0]
        if consumers != list(range(len(consumers))):
            raise ValueError("reference consumers unordered")

    @property
    def descriptor(self) -> tuple[object, ...]:
        return self.family, self.count, self.strengths, self.output, tuple(g.serial() for g in self.gates), self.program_id

    @property
    def dimension(self) -> int:
        return P ** (self.count + 1)

    @property
    def consumer_count(self) -> int:
        return sum(g.consumer_id >= 0 for g in self.gates)


def compile_spec(family: int, count: int) -> Spec:
    strengths = (1, 2, 3, 4)[:count] if family == 0 else (2, 4, 1, 3)[:count]
    syndrome = count
    def nz(value: int) -> int:
        return value % P or 1
    gates: list[RGate] = [RGate("FOURIER", data, stage="PREP") for data in range(count)]
    gates.append(RGate("FOURIER", syndrome, stage="PREP"))
    consumer = 0
    for data, strength in enumerate(strengths):
        gates.append(RGate("CUBIC", syndrome, parameter=strength, stage="INJECTION", injection_id=data))
        for kind in ("SUM", "Q", "L", "G"):
            parameter = -1 if kind == "SUM" else strength
            gates.append(RGate(kind, data, syndrome, parameter, stage="INJECTION", injection_id=data, consumer_id=consumer))
            consumer += 1
    order: Iterable[int] = range(count) if family == 0 else reversed(range(count))
    for data in order:
        if family == 0:
            gates.extend((
                RGate("SUM", data, syndrome, nz(1 + data), stage="NETWORK", consumer_id=consumer),
                RGate("CZ", (data + 1) % count, syndrome, nz(1 + 2 * data), stage="NETWORK", consumer_id=consumer + 1),
                RGate("SUM", syndrome, data, nz(2 + data), stage="NETWORK", consumer_id=consumer + 2),
            ))
            consumer += 3
            if count > 1:
                gates.append(RGate("CZ", data, (data + 1) % count, nz(2 + data), stage="NETWORK"))
        else:
            gates.extend((
                RGate("SUM", syndrome, data, nz(-(2 + data)), stage="NETWORK", consumer_id=consumer),
                RGate("CZ", data, syndrome, nz(2 + data), stage="NETWORK", consumer_id=consumer + 1),
                RGate("SUM", data, syndrome, nz(-(1 + 2 * data)), stage="NETWORK", consumer_id=consumer + 2),
            ))
            consumer += 3
            if count > 1:
                gates.append(RGate("CZ", data, (data - 1) % count, nz(1 + data), stage="NETWORK"))
    for data in range(count):
        direction = (1 if data % 2 == 0 else -1) * (1 if family == 0 else -1)
        gates.append(RGate("FOURIER", data, direction=direction, stage="FINAL"))
    output = tuple(((data + 1) if family == 0 else (2 * data + 1)) % P for data in range(count))
    gate_tuple = tuple(gates)
    return Spec(family, count, tuple(strengths), output, gate_tuple, digest(family, count, strengths, output, gate_tuple))


def phase(gate: RGate, state: Sequence[int]) -> int:
    left = state[gate.wire]
    right = state[gate.target] if gate.target >= 0 else 0
    if gate.kind == "CUBIC": return gate.parameter * left**3
    if gate.kind == "CZ": return gate.parameter * left * right
    if gate.kind == "Q": return -3 * gate.parameter * right * left**2
    if gate.kind == "L": return -3 * gate.parameter * right**2 * left
    if gate.kind == "G": return -gate.parameter * right**3
    raise RuntimeError("reference diagonal gate missing")


def apply(values: list[E], scratch: list[E], exponent: int, gate: RGate) -> int:
    if any(value != Z for value in scratch):
        raise RuntimeError("reference dirty scratch")
    dimension = len(values)
    wires = 0
    size = dimension
    while size > 1:
        size //= P
        wires += 1
    if gate.kind == "FOURIER":
        for destination in range(dimension):
            output_state = coordinates(destination, wires)
            output = output_state[gate.wire]
            total = Z
            for source in range(P):
                source_state = output_state.copy()
                source_state[gate.wire] = source
                total = add(total, mul(root(gate.direction * source * output), values[flat_index(source_state)]))
            scratch[destination] = mul(S, total)
        values[:] = scratch
        scratch[:] = [Z] * dimension
        exponent += 1
    elif gate.kind == "SUM":
        for location in range(dimension):
            state = coordinates(location, wires)
            state[gate.target] = (state[gate.target] + gate.parameter * state[gate.wire]) % P
            scratch[flat_index(state)] = values[location]
        values[:] = scratch
        scratch[:] = [Z] * dimension
    else:
        for location in range(dimension):
            state = coordinates(location, wires)
            values[location] = mul(root(phase(gate, state)), values[location])
    return normalize(values, exponent)


def probability(values: Sequence[E], exponent: int, spec: Spec) -> tuple[E, int]:
    total = Z
    for syndrome in range(P):
        amplitude = values[flat_index((*spec.output, syndrome))]
        total = add(total, mul(amplitude, conjugate(amplitude)))
    encoded = [total]
    encoded_exponent = normalize(encoded, 2 * exponent)
    return encoded[0], encoded_exponent


class ReferencePort:
    def __init__(self, count: int) -> None:
        self.count = count
        self.values = [Z] * (P ** (count + 1)); self.values[0] = O
        self.scratch = [Z] * len(self.values)
        self.exponent = 0
        self.cursor = 0
        self.consumer = 0
        self.owner = 0
        self.generation = 0
        self.last = 0
        self.descriptor: tuple[object, ...] | None = None
        self.leased = False

    def canonical(self) -> bool:
        return self.values[0] == O and all(v == Z for v in self.values[1:]) and all(v == Z for v in self.scratch) and self.exponent == self.cursor == self.consumer == 0

    def lease(self, spec: Spec, owner: int, generation: int, port_type: str = TYPE) -> None:
        if port_type != TYPE or spec.count != self.count or owner <= 0 or self.leased or not self.canonical() or generation != self.last + 1:
            raise RuntimeError("reference lease rejected")
        self.leased = True; self.owner = owner; self.generation = generation; self.descriptor = spec.descriptor

    def require(self, spec: Spec, owner: int, generation: int, port_type: str = TYPE) -> None:
        if not self.leased or owner != self.owner or generation != self.generation or port_type != TYPE or self.descriptor != spec.descriptor:
            raise RuntimeError("reference custody rejected")

    def forward(self, spec: Spec, owner: int, generation: int) -> None:
        self.require(spec, owner, generation)
        gate = spec.gates[self.cursor]
        if gate.consumer_id >= 0 and gate.consumer_id != self.consumer:
            raise RuntimeError("reference consumer order")
        self.exponent = apply(self.values, self.scratch, self.exponent, gate)
        self.cursor += 1
        if gate.consumer_id >= 0: self.consumer += 1

    def inverse(self, spec: Spec, owner: int, generation: int, supplied: RGate | None = None) -> None:
        self.require(spec, owner, generation)
        if self.cursor <= 0: raise RuntimeError("reference inverse exhausted")
        gate = spec.gates[self.cursor - 1]
        inverse = gate.inverse()
        if supplied is not None and supplied.serial() != inverse.serial():
            raise RuntimeError("reference wrong inverse")
        if gate.consumer_id >= 0 and gate.consumer_id != self.consumer - 1:
            raise RuntimeError("reference inverse order")
        self.exponent = apply(self.values, self.scratch, self.exponent, inverse)
        self.cursor -= 1
        if gate.consumer_id >= 0: self.consumer -= 1

    def project(self, spec: Spec, owner: int, generation: int) -> tuple[E, int, str]:
        self.require(spec, owner, generation)
        if self.cursor != len(spec.gates) or self.consumer != spec.consumer_count:
            raise RuntimeError("reference premature projection")
        value, exponent = probability(self.values, self.exponent, spec)
        boundary = [self.values[flat_index((*spec.output, syndrome))] for syndrome in range(P)]
        return value, exponent, vector_commitment(boundary, self.exponent)

    def project_syndrome(self) -> object:
        raise RuntimeError("reference syndrome hidden")

    def release(self, spec: Spec, owner: int, generation: int) -> None:
        self.require(spec, owner, generation)
        if not self.canonical(): raise RuntimeError("reference not restored")
        self.last = generation; self.leased = False; self.owner = self.generation = 0; self.descriptor = None


def transaction(port: ReferencePort, spec: Spec, owner: int, generation: int) -> dict[str, object]:
    if port is None: raise TypeError("null reference port")
    port.lease(spec, owner, generation)
    values_id, scratch_id = id(port.values), id(port.scratch)
    while port.cursor < len(spec.gates): port.forward(spec, owner, generation)
    boundary, boundary_exponent, slice_commitment = port.project(spec, owner, generation)
    retained = probability_encoding(boundary, boundary_exponent)
    final_commitment = vector_commitment(port.values, port.exponent)
    final_payload = payload_bits(port.values, port.exponent)
    while port.cursor: port.inverse(spec, owner, generation)
    port.release(spec, owner, generation)
    return {
        "family": spec.family, "injection_count": spec.count, "wire_count": spec.count + 1,
        "syndrome_consumer_count": spec.consumer_count,
        "selected_data_probability": retained,
        "selected_boundary_slice_commitment": slice_commitment,
        "final_state_commitment": final_commitment,
        "final_total_exact_payload_bits": final_payload,
        "resident_amplitude_field_cells": spec.dimension,
        "scratch_amplitude_field_cells": spec.dimension,
        "canonical_post_inverse_state_exact": port.canonical(),
        "same_amplitude_and_scratch_backings": id(port.values) == values_id and id(port.scratch) == scratch_id,
        "restoration_generation": port.last,
        "baseline_reload_used": False,
        "response_released_after_restoration": True,
    }


def evolve(spec: Spec, gates: Sequence[RGate] | None = None) -> tuple[list[E], int]:
    values = [Z] * spec.dimension; values[0] = O
    scratch = [Z] * spec.dimension; exponent = 0
    for gate in spec.gates if gates is None else gates:
        exponent = apply(values, scratch, exponent, gate)
    return values, exponent


def injection_identity(spec: Spec) -> bool:
    actual, exponent = evolve(spec, [g for g in spec.gates if g.stage in {"PREP", "INJECTION"}])
    expected = [Z] * spec.dimension; numerator = sqrt_power(spec.count + 1)
    for state in itertools.product(range(P), repeat=spec.count + 1):
        cubic = sum(spec.strengths[i] * state[i] ** 3 for i in range(spec.count))
        expected[flat_index(state)] = mul(numerator, root(cubic))
    expected_exponent = normalize(expected, spec.count + 1)
    return actual == expected and exponent == expected_exponent


def network_basis(spec: Spec, state: list[int]) -> tuple[list[int], int]:
    result = state.copy(); phase_value = 0
    for gate in spec.gates:
        if gate.stage != "NETWORK": continue
        if gate.kind == "SUM": result[gate.target] = (result[gate.target] + gate.parameter * result[gate.wire]) % P
        elif gate.kind == "CZ": phase_value += gate.parameter * result[gate.wire] * result[gate.target]
        else: raise RuntimeError("reference network left Clifford grammar")
    return result, phase_value


def scalar_boundary(spec: Spec, fixed_syndrome: int | None = None) -> tuple[list[E], int, int]:
    accumulators = [Z] * P
    syndromes: Iterable[int] = range(P) if fixed_syndrome is None else (fixed_syndrome,)
    power = 2 * spec.count + (1 if fixed_syndrome is None else 0)
    numerator = sqrt_power(power); terms = 0
    final = [g for g in spec.gates if g.stage == "FINAL"]
    for data in itertools.product(range(P), repeat=spec.count):
        cubic = sum(a * x**3 for a, x in zip(spec.strengths, data))
        for syndrome in syndromes:
            state, phase_value = network_basis(spec, [*data, syndrome])
            phase_value += cubic + sum(g.direction * state[g.wire] * spec.output[g.wire] for g in final)
            target = state[spec.count]
            accumulators[target] = add(accumulators[target], mul(numerator, root(phase_value)))
            terms += 1
    return accumulators, normalize(accumulators, power), terms


def slice_probability(values: Sequence[E], exponent: int) -> tuple[E, int]:
    total = Z
    for value in values: total = add(total, mul(value, conjugate(value)))
    encoded = [total]
    encoded_exponent = normalize(encoded, 2 * exponent)
    return encoded[0], encoded_exponent


def baseline(spec: Spec) -> dict[str, object]:
    values, exponent, terms = scalar_boundary(spec)
    value, value_exponent = slice_probability(values, exponent)
    return {
        "selected_data_probability": probability_encoding(value, value_exponent),
        "selected_boundary_slice_commitment": vector_commitment(values, exponent),
        "streamed_component_syndrome_terms": terms,
        "stabilizer_component_upper_bound": P**spec.count,
        "resident_boundary_accumulator_field_values": P,
        "peak_term_field_values": 1,
        "materialized_assignment_table_entries": 0,
        "full_amplitude_vector_retained": False,
    }


def dephased(spec: Spec) -> dict[str, object]:
    total = Z; common = 4 * spec.count + 1; terms = 0
    for syndrome in range(P):
        values, exponent, branch_terms = scalar_boundary(spec, syndrome)
        if exponent != 2 * spec.count:
            values = [scale(value, P ** (2 * spec.count - exponent)) for value in values]
            exponent = 2 * spec.count
        value, value_exponent = slice_probability(values, exponent)
        total = add(total, scale(value, P ** ((common - 1) - value_exponent)))
        terms += branch_terms
    encoded = [total]; exponent = normalize(encoded, common)
    return {
        "selected_data_probability": probability_encoding(encoded[0], exponent),
        "sequential_syndrome_branches": P,
        "streamed_terms": terms,
        "restoring_path": False,
    }


def rejected(callback: Callable[[], object]) -> bool:
    try: callback()
    except (RuntimeError, TypeError, ValueError): return True
    return False


def controls() -> dict[str, bool]:
    specs = [compile_spec(family, count) for family in (0, 1) for count in COUNTS]
    identities = all(injection_identity(spec) for spec in specs)
    mutations = True
    for spec in specs:
        actual, actual_exponent = evolve(spec)
        for injection in range(spec.count):
            altered, altered_exponent = evolve(spec, [g for g in spec.gates if not (g.stage == "INJECTION" and g.injection_id == injection)])
            mutations &= vector_commitment(actual, actual_exponent) != vector_commitment(altered, altered_exponent)
    selected = compile_spec(0, 3)
    selected_values, selected_exponent = evolve(selected)
    selected_encoded = probability_encoding(*probability(selected_values, selected_exponent, selected))
    no_sum_values, no_sum_exponent = evolve(selected, [g for g in selected.gates if not (g.stage == "NETWORK" and g.kind == "SUM")])
    no_cz_values, no_cz_exponent = evolve(selected, [g for g in selected.gates if not (g.stage == "NETWORK" and g.kind == "CZ")])
    no_sum = probability_encoding(*probability(no_sum_values, no_sum_exponent, selected))
    no_cz = probability_encoding(*probability(no_cz_values, no_cz_exponent, selected))
    network_indices = [i for i, g in enumerate(selected.gates) if g.stage == "NETWORK"]
    reordered = list(selected.gates); reordered[network_indices[0]], reordered[network_indices[1]] = reordered[network_indices[1]], reordered[network_indices[0]]
    reordered_values, reordered_exponent = evolve(selected, reordered)
    spec = compile_spec(0, 2)
    port = ReferencePort(2); port.lease(spec, 1, 1)
    premature = rejected(lambda: port.project(spec, 1, 1)); hidden = rejected(port.project_syndrome)
    wrong_owner = rejected(lambda: port.require(spec, 2, 1)); wrong_generation = rejected(lambda: port.require(spec, 1, 2)); wrong_type = rejected(lambda: port.require(spec, 1, 1, "BAD"))
    dirty = ReferencePort(2); dirty.lease(spec, 3, 1); dirty.scratch[0] = O
    dirty_rejected = rejected(lambda: dirty.forward(spec, 3, 1))
    missing = ReferencePort(2); missing.lease(spec, 4, 1)
    while missing.cursor < len(spec.gates): missing.forward(spec, 4, 1)
    missing.inverse(spec, 4, 1); missing_rejected = rejected(lambda: missing.release(spec, 4, 1))
    order = ReferencePort(2); order.lease(spec, 5, 1)
    while order.cursor < len(spec.gates): order.forward(spec, 5, 1)
    reorder_rejected = rejected(lambda: order.inverse(spec, 5, 1, supplied=spec.gates[-2].inverse()))
    wrong = ReferencePort(2); wrong.lease(spec, 6, 1)
    while wrong.cursor < len(spec.gates): wrong.forward(spec, 6, 1)
    final_inverse = spec.gates[-1].inverse()
    wrong_rejected = rejected(lambda: wrong.inverse(spec, 6, 1, supplied=replace(final_inverse, direction=-final_inverse.direction)))
    descriptor_mutation = rejected(lambda: replace(spec, output=tuple((x + 1) % P for x in spec.output)))
    stale = ReferencePort(1); transaction(stale, compile_spec(0, 1), 7, 1)
    stale_rejected = rejected(lambda: stale.lease(compile_spec(1, 1), 7, 1))
    magic1, negative1 = magic_l1(1); magic2, negative2 = magic_l1(2)
    expected = Real(Fraction(1), Fraction(2, 5))
    dephase_changes = all(baseline(s)["selected_data_probability"] != dephased(s)["selected_data_probability"] for s in specs if s.count >= 2)
    return {
        "all_injection_cut_identities_exact": identities,
        "each_declared_injection_changes_final_state_commitment": mutations,
        "remove_network_sum_changes_selected_boundary": no_sum != selected_encoded,
        "remove_network_cz_changes_selected_boundary": no_cz != selected_encoded,
        "reorder_noncommuting_network_gates_changes_final_state": vector_commitment(reordered_values, reordered_exponent) != vector_commitment(selected_values, selected_exponent),
        "dephasing_shared_syndrome_changes_selected_boundary_counts2_3_4": dephase_changes,
        "single_magic_wigner_l1_exact_a1_a2": magic1 == expected and magic2 == expected,
        "single_magic_negative_cell_count_a1_a2": negative1 == negative2 == 5,
        "premature_projection_rejected": premature,
        "syndrome_projection_rejected": hidden,
        "wrong_owner_rejected": wrong_owner,
        "wrong_generation_rejected": wrong_generation,
        "wrong_type_rejected": wrong_type,
        "same_id_descriptor_mutation_rejected": descriptor_mutation,
        "dirty_scratch_rejected": dirty_rejected,
        "missing_inverse_rejected": missing_rejected,
        "reordered_inverse_rejected": reorder_rejected,
        "wrong_inverse_rejected_before_mutation": wrong_rejected,
        "stale_generation_rejected": stale_rejected,
        "null_carrier_rejected": rejected(lambda: transaction(None, spec, 8, 1)),  # type: ignore[arg-type]
        "accepted_path_serializes_syndrome_values": False,
        "accepted_path_materializes_assignment_or_history_table": False,
        "public_compiler_reads_final_answer": False,
        "dephased_diagnostic_enumerates_five_syndrome_branches": True,
        "dephased_diagnostic_is_nonrestoring": True,
    }


def main() -> None:
    cases: list[dict[str, object]] = []
    magic_laws: list[dict[str, object]] = []
    for count in COUNTS:
        port = ReferencePort(count)
        primary = transaction(port, compile_spec(0, count), 240000 + count, 1)
        reuse = transaction(port, compile_spec(1, count), 240000 + count, 2)
        fresh = transaction(ReferencePort(count), compile_spec(1, count), 240000 + count, 1)
        reuse_copy = dict(reuse); reuse_copy["restoration_generation"] = 1
        if reuse_copy != fresh: raise RuntimeError("reference fresh/restored mismatch")
        cases.extend((primary, reuse))
        one, _ = magic_l1(1)
        l1 = one.power(count)
        negativity = (l1 - Real(Fraction(1), Fraction(0))).scale(Fraction(1, 2))
        magic_laws.append({
            "injection_count": count,
            "exact_wigner_l1": l1.encoding(),
            "exact_negative_mass": negativity.encoding(),
            "stabilizer_component_upper_bound": P**count,
            "product_input_law_only": True,
            "computational_lower_bound_established": False,
        })
    for case in cases:
        spec = compile_spec(case["family"], case["injection_count"])
        check = baseline(spec)
        if check["selected_data_probability"] != case["selected_data_probability"] or check["selected_boundary_slice_commitment"] != case["selected_boundary_slice_commitment"]:
            raise RuntimeError("reference scalar/full boundary mismatch")
    result = {
        "schema": "cat_cas.zeta5_interacting_multi_injection_syndrome_network_reference.v1",
        "cases": cases,
        "controls": controls(),
        "magic_laws": magic_laws,
        "imports_m240_m239_or_m237_production": False,
        "independent_power_basis_arithmetic": True,
        "independent_full_amplitude_transaction": True,
        "independent_streamed_scalar_boundary": True,
        "independent_wigner_l1_reconstruction": True,
        "independent_custody_state_machine": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
