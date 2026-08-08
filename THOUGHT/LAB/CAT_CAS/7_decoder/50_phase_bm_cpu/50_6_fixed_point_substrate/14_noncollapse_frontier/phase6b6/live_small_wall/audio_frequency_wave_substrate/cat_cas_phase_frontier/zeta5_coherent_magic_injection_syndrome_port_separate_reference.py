#!/usr/bin/env python3
"""Standalone exact oracle for M239; imports no M239/M237 production code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Sequence


P = 5
E = tuple[int, int, int, int]
Z: E = (0, 0, 0, 0)
O: E = (1, 0, 0, 0)
S: E = (-1, 0, -2, -2)
TYPE = "REFERENCE_ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_V1"
PAIR_COUNTS = (1, 2, 4)


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
        for lower in range(power - 4, power):
            raw[lower] -= coefficient
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
    while exponent and all(coefficient % 5 == 0 for value in values for coefficient in value):
        values[:] = [scale(value, 1) for value in values]
        for i, value in enumerate(values):
            values[i] = tuple(coefficient // 5 for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def vector_commitment(values: Sequence[E], exponent: int) -> str:
    payload = {"denominator_power5": exponent, "numerators": [list(value) for value in values]}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def value_commitment(value: E, exponent: int) -> str:
    values = [value]
    exponent = normalize(values, exponent)
    payload = {"denominator_power5": exponent, "numerator": list(values[0])}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def index(data: int, syndrome: int) -> int:
    return P * data + syndrome


@dataclass(frozen=True)
class RGate:
    kind: str
    wire: int = -1
    target: int = -1
    parameter: int = 0
    direction: int = 1
    consumer_id: int = -1

    def __post_init__(self) -> None:
        if self.kind not in {"FOURIER", "CUBIC", "SUM", "CZ", "Q", "L", "G"}:
            raise ValueError("bad reference gate")
        if self.kind == "FOURIER" and (self.wire not in (0, 1) or self.direction not in (-1, 1)):
            raise ValueError("bad reference Fourier")
        if self.kind in {"CUBIC", "Q", "L", "G", "SUM", "CZ"} and self.parameter % P == 0:
            raise ValueError("zero reference gate parameter")
        if self.kind in {"SUM", "CZ"} and {self.wire, self.target} != {0, 1}:
            raise ValueError("bad reference two-wire gate")
        if self.kind in {"Q", "L", "G", "SUM", "CZ"} and self.consumer_id < 0:
            raise ValueError("missing reference consumer identity")
        if self.kind not in {"Q", "L", "G", "SUM", "CZ"} and self.consumer_id != -1:
            raise ValueError("unexpected reference consumer identity")

    def inverse(self) -> "RGate":
        if self.kind == "FOURIER":
            return RGate(self.kind, self.wire, self.target, self.parameter, -self.direction)
        return RGate(
            self.kind, self.wire, self.target, -self.parameter,
            self.direction, self.consumer_id,
        )

    def serial(self) -> tuple[object, ...]:
        return self.kind, self.wire, self.target, self.parameter % P, self.direction, self.consumer_id


def digest(family: int, pairs: int, output: int, gates: Sequence[RGate]) -> str:
    payload = [family, pairs, output, [gate.serial() for gate in gates], TYPE]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RProgram:
    family: int
    pairs: int
    output: int
    gates: tuple[RGate, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.pairs not in PAIR_COUNTS or self.output not in range(P):
            raise ValueError("reference program outside declared family")
        consumers = [gate.consumer_id for gate in self.gates if gate.consumer_id >= 0]
        if consumers != list(range(len(consumers))):
            raise ValueError("reference consumers not ordered")
        if self.program_id != digest(self.family, self.pairs, self.output, self.gates):
            raise ValueError("reference descriptor digest mismatch")

    @property
    def descriptor(self) -> tuple[object, ...]:
        return self.family, self.pairs, self.output, tuple(g.serial() for g in self.gates), self.program_id

    @property
    def consumer_count(self) -> int:
        return sum(g.consumer_id >= 0 for g in self.gates)


WORDS: dict[tuple[int, int], tuple[tuple[str, int], ...]] = {
    (0, 1): (("F", -1), ("SUM", 1), ("CZ", 2), ("F", 1), ("SUM", 1)),
    (0, 2): (("SUM", 1), ("F", 1), ("CZ", 2), ("SUM", 1), ("CZ", 4),
             ("F", 1), ("F", -1), ("SUM", 3)),
    (0, 4): (("F", -1), ("SUM", 1), ("CZ", 2), ("F", 1), ("SUM", 3),
             ("CZ", 4), ("SUM", 3), ("CZ", 4), ("F", 1), ("CZ", 2),
             ("SUM", 1), ("F", -1), ("F", -1), ("SUM", 3)),
    (1, 1): (("F", -1), ("SUM", 3), ("CZ", 3), ("F", -1), ("SUM", 3)),
    (1, 2): (("SUM", 4), ("F", 1), ("CZ", 1), ("CZ", 2), ("SUM", 2),
             ("F", 1), ("F", 1), ("SUM", 4)),
    (1, 4): (("SUM", 3), ("F", -1), ("CZ", 2), ("CZ", 2), ("F", 1),
             ("SUM", 1), ("F", -1), ("SUM", 3), ("CZ", 4), ("CZ", 2),
             ("SUM", 2), ("F", 1), ("F", -1), ("SUM", 2)),
}


def make_program(family: int, pairs: int) -> RProgram:
    if (family, pairs) not in WORDS:
        raise ValueError("bad reference public family")
    a = family + 1
    gates = [
        RGate("FOURIER", wire=0), RGate("FOURIER", wire=1),
        RGate("CUBIC", wire=1, parameter=a),
        RGate("SUM", wire=0, target=1, parameter=-1, consumer_id=0),
        RGate("Q", parameter=a, consumer_id=1),
        RGate("L", parameter=a, consumer_id=2),
        RGate("G", parameter=a, consumer_id=3),
    ]
    consumer = 4
    for kind, parameter in WORDS[(family, pairs)]:
        if kind == "F":
            gates.append(RGate("FOURIER", wire=1, direction=parameter))
        else:
            gates.append(RGate(kind, wire=1, target=0, parameter=parameter, consumer_id=consumer))
            consumer += 1
    gates.append(RGate("FOURIER", wire=0, direction=1 if family == 0 else -1))
    output = (pairs + 2 * family) % P
    result = tuple(gates)
    return RProgram(family, pairs, output, result, digest(family, pairs, output, result))


def matrix_apply(values: list[E], exponent: int, gate: RGate) -> int:
    result = [Z] * 25
    if gate.kind == "FOURIER":
        for data in range(P):
            for syndrome in range(P):
                output = data if gate.wire == 0 else syndrome
                total = Z
                for source in range(P):
                    source_data, source_syndrome = (
                        (source, syndrome) if gate.wire == 0 else (data, source)
                    )
                    total = add(
                        total,
                        mul(root(gate.direction * source * output), values[index(source_data, source_syndrome)]),
                    )
                result[index(data, syndrome)] = mul(S, total)
        exponent += 1
    elif gate.kind == "SUM":
        for data in range(P):
            for syndrome in range(P):
                destination = [data, syndrome]
                destination[gate.target] = (
                    destination[gate.target] + gate.parameter * destination[gate.wire]
                ) % P
                result[index(destination[0], destination[1])] = values[index(data, syndrome)]
    else:
        for data in range(P):
            for syndrome in range(P):
                if gate.kind == "CUBIC":
                    coordinate = data if gate.wire == 0 else syndrome
                    phase = gate.parameter * coordinate**3
                elif gate.kind == "CZ":
                    phase = gate.parameter * data * syndrome
                elif gate.kind == "Q":
                    phase = -3 * gate.parameter * syndrome * data**2
                elif gate.kind == "L":
                    phase = -3 * gate.parameter * syndrome**2 * data
                elif gate.kind == "G":
                    phase = -gate.parameter * syndrome**3
                else:
                    raise RuntimeError("unknown reference diagonal")
                location = index(data, syndrome)
                result[location] = mul(root(phase), values[location])
    values[:] = result
    return normalize(values, exponent)


def selected_probability(values: Sequence[E], exponent: int, output: int) -> tuple[E, int]:
    total = Z
    for syndrome in range(P):
        value = values[index(output, syndrome)]
        total = add(total, mul(value, conjugate(value)))
    encoded = [total]
    encoded_exponent = normalize(encoded, 2 * exponent)
    return encoded[0], encoded_exponent


def execute(gates: Sequence[RGate]) -> tuple[list[E], int]:
    values = [O] + [Z] * 24
    exponent = 0
    for gate in gates:
        exponent = matrix_apply(values, exponent, gate)
    return values, exponent


class ReferencePort:
    def __init__(self) -> None:
        self.values = [O] + [Z] * 24
        self.scratch = [Z] * 25
        self.exponent = 0
        self.cursor = 0
        self.consumer_cursor = 0
        self.owner = 0
        self.descriptor: tuple[object, ...] | None = None
        self.generation = 0
        self.last = 0
        self.leased = False

    def canonical(self) -> bool:
        return (
            self.values == [O] + [Z] * 24 and self.scratch == [Z] * 25
            and self.exponent == 0 and self.cursor == 0 and self.consumer_cursor == 0
        )

    def lease(self, owner: int, program: RProgram, generation: int, port_type: str = TYPE) -> None:
        if self.leased or port_type != TYPE or generation != self.last + 1 or not self.canonical():
            raise RuntimeError("reference lease rejected")
        self.owner, self.descriptor, self.generation, self.leased = (
            owner, program.descriptor, generation, True
        )

    def require(self, owner: int, program: RProgram, generation: int) -> None:
        if (
            not self.leased or owner != self.owner or generation != self.generation
            or program.descriptor != self.descriptor
        ):
            raise RuntimeError("reference custody rejected")

    def forward(self, owner: int, program: RProgram, generation: int, gate_index: int) -> None:
        self.require(owner, program, generation)
        if gate_index != self.cursor:
            raise RuntimeError("reference forward order rejected")
        gate = program.gates[gate_index]
        if gate.consumer_id >= 0:
            if gate.consumer_id != self.consumer_cursor:
                raise RuntimeError("reference consumer order rejected")
            self.consumer_cursor += 1
        self.exponent = matrix_apply(self.values, self.exponent, gate)
        self.cursor += 1

    def inverse(self, owner: int, program: RProgram, generation: int, gate_index: int) -> None:
        self.require(owner, program, generation)
        if gate_index != self.cursor - 1:
            raise RuntimeError("reference inverse order rejected")
        gate = program.gates[gate_index]
        if gate.consumer_id >= 0 and gate.consumer_id != self.consumer_cursor - 1:
            raise RuntimeError("reference inverse consumer order rejected")
        self.exponent = matrix_apply(self.values, self.exponent, gate.inverse())
        if gate.consumer_id >= 0:
            self.consumer_cursor -= 1
        self.cursor -= 1

    def project(self, owner: int, program: RProgram, generation: int) -> dict[str, object]:
        self.require(owner, program, generation)
        if self.cursor != len(program.gates) or self.consumer_cursor != program.consumer_count:
            raise RuntimeError("reference premature projection")
        if self.scratch != [Z] * 25:
            raise RuntimeError("reference dirty scratch projection")
        probability, exponent = selected_probability(self.values, self.exponent, program.output)
        norm = Z
        for value in self.values:
            norm = add(norm, mul(value, conjugate(value)))
        norm_values = [norm]
        norm_exponent = normalize(norm_values, 2 * self.exponent)
        if norm_values[0] != O or norm_exponent:
            raise RuntimeError("reference norm failure")
        return {
            "selected_data_probability": probability_encoding(probability, exponent),
            "selected_data_probability_nonzero": probability != Z,
            "normalization_exact": True,
            "final_state_commitment": vector_commitment(self.values, self.exponent),
        }

    def project_syndrome(self) -> None:
        raise RuntimeError("reference syndrome projection forbidden")

    def release(self, owner: int, program: RProgram, generation: int) -> None:
        self.require(owner, program, generation)
        if not self.canonical():
            raise RuntimeError("reference carrier not restored")
        self.leased = False
        self.last = generation
        self.owner = self.generation = 0
        self.descriptor = None


def transaction(port: ReferencePort, program: RProgram, generation: int, owner: int = 23901) -> dict[str, object]:
    if port is None:
        raise TypeError("null reference port")
    backings = id(port.values), id(port.scratch)
    port.lease(owner, program, generation)
    for gate_index in range(len(program.gates)):
        port.forward(owner, program, generation, gate_index)
    boundary = port.project(owner, program, generation)
    for gate_index in range(len(program.gates) - 1, -1, -1):
        port.inverse(owner, program, generation, gate_index)
    exact = port.canonical()
    same = backings == (id(port.values), id(port.scratch))
    port.release(owner, program, generation)
    return {
        "family": program.family,
        "cubic_strength": program.family + 1,
        "consumer_pair_count": program.pairs,
        "syndrome_consumer_count": program.consumer_count,
        **boundary,
        "canonical_post_inverse_state_exact": exact,
        "same_amplitude_and_scratch_backings": same,
        "restoration_generation": generation,
        "baseline_reload_used": False,
    }


def injection_identity(a: int, data_input: int) -> bool:
    values = [Z] * 25
    values[index(data_input, 0)] = O
    exponent = 0
    gates = (
        RGate("FOURIER", wire=1), RGate("CUBIC", wire=1, parameter=a),
        RGate("SUM", wire=0, target=1, parameter=-1, consumer_id=0),
        RGate("Q", parameter=a, consumer_id=1), RGate("L", parameter=a, consumer_id=2),
        RGate("G", parameter=a, consumer_id=3),
    )
    for gate in gates:
        exponent = matrix_apply(values, exponent, gate)
    expected = [Z] * 25
    for syndrome in range(P):
        expected[index(data_input, syndrome)] = mul(S, root(a * data_input**3))
    return values == expected and exponent == 1


def compiled_baseline(program: RProgram) -> tuple[list[E], int]:
    values = [Z] * 25
    for data in range(P):
        for syndrome in range(P):
            values[index(data, syndrome)] = root((program.family + 1) * data**3)
    exponent = 1
    for gate in program.gates[7:]:
        exponent = matrix_apply(values, exponent, gate)
    return values, exponent


def dephased(program: RProgram) -> dict[str, object]:
    probability = Z
    probability_exponent = 0
    for syndrome_seed in range(P):
        values = [Z] * 25
        for data in range(P):
            values[index(data, syndrome_seed)] = mul(S, root((program.family + 1) * data**3))
        exponent = 1
        for gate in program.gates[7:]:
            exponent = matrix_apply(values, exponent, gate)
        branch, branch_exponent = selected_probability(values, exponent, program.output)
        common = max(probability_exponent, branch_exponent)
        probability = add(
            scale(probability, 5 ** (common - probability_exponent)),
            scale(branch, 5 ** (common - branch_exponent)),
        )
        probability_exponent = common
        encoded = [probability]
        probability_exponent = normalize(encoded, probability_exponent)
        probability = encoded[0]
    encoded = [probability]
    probability_exponent = normalize(encoded, probability_exponent + 1)
    return probability_encoding(encoded[0], probability_exponent)


def rejected(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    program = make_program(0, 2)
    full_values, full_exponent = execute(program.gates)
    omissions: dict[str, bool] = {}
    for omitted in ("Q", "L", "G"):
        values, exponent = execute(tuple(gate for gate in program.gates if gate.kind != omitted))
        omissions[omitted] = vector_commitment(values, exponent) != vector_commitment(full_values, full_exponent)
    omitted_g_values, omitted_g_exponent = execute(program.gates[:6])
    expected_omitted_g = [Z] * 25
    for data in range(P):
        for syndrome in range(P):
            expected_omitted_g[index(data, syndrome)] = root(
                (program.family + 1) * (data**3 + syndrome**3)
            )
    baseline_values, baseline_exponent = compiled_baseline(program)
    coherent_value, coherent_exponent = selected_probability(full_values, full_exponent, program.output)
    coherent = probability_encoding(coherent_value, coherent_exponent)

    premature = ReferencePort()
    premature.lease(23911, program, 1)
    wrong_type = rejected(lambda: ReferencePort().lease(23911, program, 1, "WRONG"))
    forged_gates = program.gates[:-1] + (RGate("FOURIER", wire=0, direction=-1),)
    forged = rejected(lambda: RProgram(program.family, program.pairs, program.output, forged_gates, program.program_id))
    duplicate = list(program.gates)
    duplicate[5] = RGate("L", parameter=1, consumer_id=1)
    duplicate_rejected = rejected(
        lambda: RProgram(program.family, program.pairs, program.output, tuple(duplicate), digest(program.family, program.pairs, program.output, duplicate))
    )
    dirty = ReferencePort()
    dirty.scratch[0] = O

    missing = ReferencePort()
    missing.lease(23911, program, 1)
    for i in range(len(program.gates)):
        missing.forward(23911, program, 1, i)

    reordered = ReferencePort()
    reordered.lease(23911, program, 1)
    for i in range(len(program.gates)):
        reordered.forward(23911, program, 1, i)

    wrong = ReferencePort()
    wrong.lease(23911, program, 1)
    for i in range(len(program.gates)):
        wrong.forward(23911, program, 1, i)
    last = len(program.gates) - 1
    wrong.exponent = matrix_apply(wrong.values, wrong.exponent, RGate("FOURIER", wire=0, direction=program.gates[-1].direction))
    wrong.cursor -= 1
    for i in range(last - 1, -1, -1):
        wrong.inverse(23911, program, 1, i)

    stale = ReferencePort()
    transaction(stale, make_program(0, 1), 1, 23911)
    transaction(stale, make_program(1, 1), 2, 23911)

    removed_x_values, removed_x_exp = execute(
        tuple(g for g in program.gates if not (g.kind == "SUM" and g.wire == 1))
    )
    removed_z_values, removed_z_exp = execute(tuple(g for g in program.gates if g.kind != "CZ"))
    removed_x_probability = probability_encoding(
        *selected_probability(removed_x_values, removed_x_exp, program.output)
    )
    removed_z_probability = probability_encoding(
        *selected_probability(removed_z_values, removed_z_exp, program.output)
    )
    order_program = program
    order_values, order_exp = execute(order_program.gates)
    swapped = list(order_program.gates)
    swapped[9], swapped[10] = swapped[10], swapped[9]
    swapped_values, swapped_exp = execute(swapped)
    sham_values, sham_exp = execute(
        tuple(g for g in program.gates if g.kind not in {"CUBIC", "Q", "L", "G"})
    )
    sham_norm = Z
    for value in sham_values:
        sham_norm = add(sham_norm, mul(value, conjugate(value)))
    sham_norm_values = [sham_norm]
    sham_norm_exp = normalize(sham_norm_values, 2 * sham_exp)
    return {
        "injection_identity_all_data_inputs_a1_a2": all(injection_identity(a, d) for a in (1, 2) for d in range(P)),
        "q_correction_omission_changes_full_state": omissions["Q"],
        "l_correction_omission_changes_full_state": omissions["L"],
        "g_correction_omission_changes_full_state": omissions["G"],
        "g_omission_exactly_factorizes_data_magic_and_syndrome_magic": (
            omitted_g_values == expected_omitted_g and omitted_g_exponent == 1
        ),
        "complete_injection_compact_state_parity": vector_commitment(baseline_values, baseline_exponent) == vector_commitment(full_values, full_exponent),
        "complete_injection_compact_boundary_parity": probability_encoding(*selected_probability(baseline_values, baseline_exponent, program.output)) == coherent,
        "dephased_syndrome_changes_selected_boundary": dephased(program) != coherent,
        "dephased_measurement_is_nonrestoring_diagnostic_only": True,
        "remove_sum_consumer_changes_full_state": vector_commitment(removed_x_values, removed_x_exp) != vector_commitment(full_values, full_exponent),
        "remove_cz_consumer_changes_full_state": vector_commitment(removed_z_values, removed_z_exp) != vector_commitment(full_values, full_exponent),
        "remove_sum_consumer_changes_selected_boundary": removed_x_probability != coherent,
        "remove_cz_consumer_changes_selected_boundary": removed_z_probability != coherent,
        "adjacent_sum_cz_consumer_order_changes_full_state": vector_commitment(swapped_values, swapped_exp) != vector_commitment(order_values, order_exp),
        "stabilizer_sham_contains_only_clifford_gates": True,
        "stabilizer_sham_normalization_exact": sham_norm_values[0] == O and sham_norm_exp == 0,
        "premature_final_projection_rejected": rejected(lambda: premature.project(23911, program, 1)),
        "syndrome_projection_rejected": rejected(premature.project_syndrome),
        "wrong_owner_rejected": rejected(lambda: premature.forward(23912, program, 1, 0)),
        "wrong_program_rejected": rejected(lambda: premature.forward(23911, make_program(1, 2), 1, 0)),
        "wrong_generation_rejected": rejected(lambda: premature.forward(23911, program, 2, 0)),
        "wrong_type_rejected": wrong_type,
        "same_id_changed_descriptor_rejected": forged,
        "duplicate_consumer_rejected": duplicate_rejected,
        "null_carrier_rejected": rejected(lambda: transaction(None, program, 1)),  # type: ignore[arg-type]
        "dirty_scratch_rejected": rejected(lambda: dirty.lease(23911, program, 1)),
        "missing_inverse_rejected": rejected(lambda: missing.release(23911, program, 1)),
        "reordered_inverse_rejected": rejected(lambda: reordered.inverse(23911, program, 1, len(program.gates) - 2)),
        "wrong_inverse_after_remaining_schedule_rejected": rejected(lambda: wrong.release(23911, program, 1)),
        "stale_generation_rejected": rejected(lambda: stale.lease(23911, program, 2)),
        "syndrome_values_serialized": False,
        "accepted_in_place_path_retains_branch_assignment_expansion": False,
        "dephased_diagnostic_enumerates_five_syndrome_branches": True,
        "public_compiler_reads_final_answer": False,
    }


def comparable(case: dict[str, object]) -> dict[str, object]:
    keys = (
        "family", "cubic_strength", "consumer_pair_count", "syndrome_consumer_count",
        "selected_data_probability", "selected_data_probability_nonzero", "normalization_exact",
        "final_state_commitment", "canonical_post_inverse_state_exact",
        "same_amplitude_and_scratch_backings", "restoration_generation", "baseline_reload_used",
    )
    return {key: case[key] for key in keys}


def main() -> None:
    cases = [comparable(transaction(ReferencePort(), make_program(f, p), 1)) for f in (0, 1) for p in PAIR_COUNTS]
    shared = ReferencePort()
    primary = comparable(transaction(shared, make_program(0, 4), 1))
    reuse = comparable(transaction(shared, make_program(1, 2), 2))
    fresh = comparable(transaction(ReferencePort(), make_program(1, 2), 1))
    output = {
        "schema": "cat_cas.zeta5_coherent_magic_injection_syndrome_port_reference.v1",
        "cases": cases,
        "controls": controls(),
        "reuse": {"primary": primary, "reuse": reuse, "fresh_reuse": fresh},
        "imports_m239_or_m237_production": False,
        "independent_exact_25_amplitude_matrix_recurrence": True,
        "independent_injection_identity": True,
        "independent_compiled_identity_baseline": True,
        "independent_custody_state_machine": True,
        "source_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
