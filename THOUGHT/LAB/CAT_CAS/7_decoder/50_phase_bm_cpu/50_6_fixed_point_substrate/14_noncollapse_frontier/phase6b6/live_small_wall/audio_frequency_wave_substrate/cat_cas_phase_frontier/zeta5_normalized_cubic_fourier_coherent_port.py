#!/usr/bin/env python3
"""M237: exact normalized zeta5 cubic-Fourier coherent port.

The implementation is bounded exact software over Q(zeta_5).  It is not a
physical waveform experiment and does not claim an advantage over the
identical five-amplitude classical recurrence.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


P = 5
WIDTH = 5
K_DEGREE = 4
PORT_TYPE = "ZETA5_NORMALIZED_COHERENT_PORT_V1"
DEPTHS = (2, 4, 8, 16, 32, 64)
ZERO = (0, 0, 0, 0)
ONE = (1, 0, 0, 0)
SQRT5 = (-1, 0, -2, -2)

RESULT = "PASS_EXACT_ZETA5_NORMALIZED_CUBIC_FOURIER_COHERENT_PORT_STRICT_SCOPE"
CLAIM = (
    "EXACT_NORMALIZED_QZETA5_CUBIC_PHASE_AND_FOURIER_MODULES_PRESERVE_A_"
    "FIXED5_AMPLITUDE_COHERENT_SHARED_PORT_WITH_CAUSAL_EXACT_DESTRUCTIVE_"
    "INTERFERENCE_ZERO_VERSUS_DEPHASED_PROBABILITY1_OVER5_FINAL_ONLY_"
    "AMPLITUDE_PROJECTION_EXACT_SAME_BACKING_RESTORATION_AND_DESCRIPTOR_"
    "DISTINCT_REUSE_ACROSS_DEPTHS2_4_8_16_32_64_BUT_THE_IDENTICAL_EXACT_"
    "FIVE_AMPLITUDE_OR_BACKWARD_ROW_CLASSICAL_RECURRENCE_REMAINS"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_QZETA5_NORMALIZED_FOURIER_AND_CUBIC_DIAGONAL_CIRCUITS_"
    "ONE_FIXED_F5_PORT_TWO_PUBLIC_DESCRIPTOR_FAMILIES_DEPTHS2_4_8_16_32_64_"
    "DIRECT_PROCESS_LOGICAL_CUSTODY_ONLY"
)


K = tuple[int, int, int, int]


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def k_add(left: K, right: K) -> K:
    return tuple(left[index] + right[index] for index in range(4))  # type: ignore[return-value]


def k_sub(left: K, right: K) -> K:
    return tuple(left[index] - right[index] for index in range(4))  # type: ignore[return-value]


def k_scale(value: K, scalar: int) -> K:
    return tuple(scalar * item for item in value)  # type: ignore[return-value]


def k_mul(left: K, right: K) -> K:
    raw = [0] * 7
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            raw[i + j] += a * b
    for degree in range(6, 3, -1):
        coefficient = raw[degree]
        if coefficient:
            for target in range(degree - 4, degree):
                raw[target] -= coefficient
            raw[degree] = 0
    return tuple(raw[:4])  # type: ignore[return-value]


def zeta_power(exponent: int) -> K:
    exponent %= 5
    if exponent < 4:
        values = [0] * 4
        values[exponent] = 1
        return tuple(values)  # type: ignore[return-value]
    return (-1, -1, -1, -1)


def k_conjugate(value: K) -> K:
    result = ZERO
    for exponent, coefficient in enumerate(value):
        result = k_add(result, k_scale(zeta_power(-exponent), coefficient))
    return result


def k_commitment(value: K, denominator_exponent: int) -> str:
    payload = json.dumps([list(value), denominator_exponent], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def vector_commitment(numerators: Sequence[K], denominator_exponent: int) -> str:
    payload = json.dumps(
        [[list(value) for value in numerators], denominator_exponent],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_element(value: K, denominator_exponent: int) -> tuple[K, int]:
    result = value
    exponent = denominator_exponent
    while exponent and all(coefficient % 5 == 0 for coefficient in result):
        result = tuple(coefficient // 5 for coefficient in result)  # type: ignore[assignment]
        exponent -= 1
    return result, exponent


def denominator_material_bits(exponent: int) -> int:
    return signed_bits(5**exponent)


def numerator_payload_bits(numerators: Sequence[K]) -> int:
    return sum(signed_bits(coefficient) for value in numerators for coefficient in value)


def vector_payload_bits(numerators: Sequence[K], denominator_exponent: int) -> int:
    return numerator_payload_bits(numerators) + denominator_material_bits(denominator_exponent)


def canonicalize_vector(numerators: list[K], denominator_exponent: int) -> int:
    exponent = denominator_exponent
    while exponent and all(
        coefficient % 5 == 0 for value in numerators for coefficient in value
    ):
        for index, value in enumerate(numerators):
            numerators[index] = tuple(coefficient // 5 for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def probability(value: K, denominator_exponent: int) -> tuple[K, int]:
    product = k_mul(value, k_conjugate(value))
    return canonical_element(product, 2 * denominator_exponent)


def rational_probability(value: K, denominator_exponent: int) -> list[int] | None:
    product, exponent = probability(value, denominator_exponent)
    if any(product[index] for index in range(1, 4)):
        return None
    numerator = product[0]
    denominator = 5**exponent
    while denominator > 1 and numerator % 5 == 0:
        numerator //= 5
        denominator //= 5
    return [numerator, denominator]


def total_norm(numerators: Sequence[K], denominator_exponent: int) -> tuple[int, int]:
    result = ZERO
    for value in numerators:
        result = k_add(result, k_mul(value, k_conjugate(value)))
    if any(result[index] for index in range(1, 4)):
        raise RuntimeError("carrier norm is not rational")
    numerator = result[0]
    denominator = 5 ** (2 * denominator_exponent)
    while denominator > 1 and numerator % 5 == 0:
        numerator //= 5
        denominator //= 5
    return numerator, denominator


@dataclass(frozen=True)
class Gate:
    kind: str
    a: int = 0
    b: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("FOURIER", "CUBIC_PHASE"):
            raise ValueError("invalid coherent gate kind")
        if self.kind == "FOURIER" and (self.a or self.b):
            raise ValueError("Fourier gate has no phase parameters")
        if self.kind == "CUBIC_PHASE" and self.a % 5 == 0:
            raise ValueError("declared cubic coefficient must be nonzero")


def program_digest(family: int, depth: int, output_index: int, gates: Sequence[Gate]) -> str:
    descriptor = [[gate.kind, gate.a, gate.b] for gate in gates]
    encoded = json.dumps([family, depth, output_index, descriptor], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PublicProgram:
    family: int
    depth: int
    output_index: int
    gates: tuple[Gate, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.depth <= 0 or self.output_index not in range(5):
            raise ValueError("invalid coherent public descriptor")
        if self.program_id != program_digest(
            self.family, self.depth, self.output_index, self.gates
        ):
            raise ValueError("coherent program identity is not descriptor-bound")

    @staticmethod
    def build(family: int, depth: int) -> "PublicProgram":
        if family not in (0, 1) or depth <= 0:
            raise ValueError("invalid coherent program family or depth")
        gates = [Gate("FOURIER")]
        for index in range(depth):
            if family == 0:
                a = 1 + (index % 4)
                b = (index * index + index + 1) % 5
            else:
                a = 1 + ((2 * index + 1) % 4)
                b = (3 * index * index + 2 * index + 2) % 5
            gates.extend((Gate("CUBIC_PHASE", a, b), Gate("FOURIER")))
        gate_tuple = tuple(gates)
        output_index = 0 if family == 0 else 2
        return PublicProgram(
            family, depth, output_index, gate_tuple,
            program_digest(family, depth, output_index, gate_tuple),
        )

    @staticmethod
    def compile(family: int, depth: int) -> "PublicProgram":
        if depth not in DEPTHS:
            raise ValueError("depth is outside the declared series")
        return PublicProgram.build(family, depth)


@dataclass
class Work:
    forward_fourier_gates: int = 0
    forward_cubic_phase_gates: int = 0
    inverse_fourier_gates: int = 0
    inverse_cubic_phase_gates: int = 0
    exact_norm_checks: int = 0
    canonical_common_factor_cancellations: int = 0
    retained_dynamic_inverse_history_entries: int = 0
    peak_resident_numerator_payload_bits: int = 0
    peak_denominator_exponent: int = 0
    peak_resident_total_exact_payload_bits: int = 0

    def observe(self, numerators: Sequence[K], denominator_exponent: int) -> None:
        self.peak_resident_numerator_payload_bits = max(
            self.peak_resident_numerator_payload_bits, numerator_payload_bits(numerators)
        )
        self.peak_denominator_exponent = max(
            self.peak_denominator_exponent, denominator_exponent
        )
        self.peak_resident_total_exact_payload_bits = max(
            self.peak_resident_total_exact_payload_bits,
            vector_payload_bits(numerators, denominator_exponent),
        )


def apply_phase(
    numerators: list[K], denominator_exponent: int, a: int, b: int
) -> int:
    for x in range(5):
        exponent = (a * x * x * x + b * x) % 5
        numerators[x] = k_mul(zeta_power(exponent), numerators[x])
    return canonicalize_vector(numerators, denominator_exponent)


def apply_fourier(
    numerators: list[K], scratch: list[K], denominator_exponent: int, sign: int
) -> tuple[int, int]:
    if any(value != ZERO for value in scratch):
        raise RuntimeError("coherent Fourier scratch is dirty")
    for y in range(5):
        accumulator = ZERO
        for x in range(5):
            accumulator = k_add(
                accumulator,
                k_mul(zeta_power(sign * x * y), numerators[x]),
            )
        scratch[y] = k_mul(SQRT5, accumulator)
    next_exponent = denominator_exponent + 1
    cancellations = 0
    while next_exponent and all(
        coefficient % 5 == 0 for value in scratch for coefficient in value
    ):
        for index, value in enumerate(scratch):
            scratch[index] = tuple(coefficient // 5 for coefficient in value)  # type: ignore[assignment]
        next_exponent -= 1
        cancellations += 1
    for index in range(5):
        numerators[index] = scratch[index]
        scratch[index] = ZERO
    return next_exponent, cancellations


def execute_gate(
    numerators: list[K], scratch: list[K], denominator_exponent: int,
    gate: Gate, inverse: bool, work: Work,
) -> int:
    if gate.kind == "FOURIER":
        denominator_exponent, cancellations = apply_fourier(
            numerators, scratch, denominator_exponent, -1 if inverse else 1
        )
        work.canonical_common_factor_cancellations += cancellations
        if inverse:
            work.inverse_fourier_gates += 1
        else:
            work.forward_fourier_gates += 1
    else:
        denominator_exponent = apply_phase(
            numerators, denominator_exponent,
            -gate.a if inverse else gate.a,
            -gate.b if inverse else gate.b,
        )
        if inverse:
            work.inverse_cubic_phase_gates += 1
        else:
            work.forward_cubic_phase_gates += 1
    if total_norm(numerators, denominator_exponent) != (1, 1):
        raise RuntimeError("normalized coherent carrier norm changed")
    work.exact_norm_checks += 1
    work.observe(numerators, denominator_exponent)
    return denominator_exponent


class CoherentPort:
    def __init__(self) -> None:
        self.psi: list[K] = [ONE, ZERO, ZERO, ZERO, ZERO]
        self.scratch: list[K] = [ZERO] * 5
        self.denominator_exponent = 0
        self.owner: int | None = None
        self.program_id: str | None = None
        self.program_descriptor: tuple[object, ...] | None = None
        self.port_type: str | None = None
        self.generation = 0
        self.last_restored_generation = 0
        self.cursor = 0
        self.leased = False

    def canonical(self) -> bool:
        return (
            self.psi == [ONE, ZERO, ZERO, ZERO, ZERO]
            and self.scratch == [ZERO] * 5
            and self.denominator_exponent == 0
            and self.cursor == 0
        )

    @staticmethod
    def descriptor(program: PublicProgram) -> tuple[object, ...]:
        return (
            program.family, program.depth, program.output_index,
            tuple((gate.kind, gate.a, gate.b) for gate in program.gates),
        )

    def lease(
        self, owner: int, program: PublicProgram, generation: int,
        port_type: str = PORT_TYPE,
    ) -> None:
        if self.leased or not self.canonical():
            raise RuntimeError("coherent carrier is not canonical")
        if owner <= 0 or port_type != PORT_TYPE or generation != self.last_restored_generation + 1:
            raise RuntimeError("invalid coherent carrier lease")
        self.owner = owner
        self.program_id = program.program_id
        self.program_descriptor = self.descriptor(program)
        self.port_type = port_type
        self.generation = generation
        self.leased = True

    def require(self, owner: int, program: PublicProgram, generation: int) -> None:
        if not self.leased or (
            self.owner != owner or self.program_id != program.program_id
            or self.program_descriptor != self.descriptor(program)
            or self.port_type != PORT_TYPE or self.generation != generation
        ):
            raise RuntimeError("coherent carrier custody mismatch")

    def forward_gate(
        self, owner: int, program: PublicProgram, generation: int,
        index: int, work: Work,
    ) -> None:
        self.require(owner, program, generation)
        if index != self.cursor:
            raise RuntimeError("coherent forward dependency failure")
        self.denominator_exponent = execute_gate(
            self.psi, self.scratch, self.denominator_exponent,
            program.gates[index], False, work,
        )
        self.cursor += 1

    def inverse_gate(
        self, owner: int, program: PublicProgram, generation: int,
        index: int, work: Work,
    ) -> None:
        self.require(owner, program, generation)
        if index != self.cursor - 1:
            raise RuntimeError("coherent inverse dependency failure")
        self.denominator_exponent = execute_gate(
            self.psi, self.scratch, self.denominator_exponent,
            program.gates[index], True, work,
        )
        self.cursor -= 1

    def project_final(
        self, owner: int, program: PublicProgram, generation: int
    ) -> tuple[K, int]:
        self.require(owner, program, generation)
        if self.cursor != len(program.gates) or self.scratch != [ZERO] * 5:
            raise RuntimeError("premature coherent boundary projection")
        return canonical_element(
            self.psi[program.output_index], self.denominator_exponent
        )

    def project_full_vector(self) -> None:
        raise RuntimeError("full coherent port projection is forbidden")

    def release(self, owner: int, program: PublicProgram, generation: int) -> None:
        self.require(owner, program, generation)
        if not self.canonical():
            raise RuntimeError("coherent carrier is not exactly restored")
        self.last_restored_generation = generation
        self.owner = None
        self.program_id = None
        self.program_descriptor = None
        self.port_type = None
        self.generation = 0
        self.leased = False


def direct_boundary(program: PublicProgram) -> dict[str, object]:
    numerators = [ONE, ZERO, ZERO, ZERO, ZERO]
    scratch = [ZERO] * 5
    exponent = 0
    work = Work()
    work.observe(numerators, exponent)
    for gate in program.gates:
        exponent = execute_gate(numerators, scratch, exponent, gate, False, work)
    amplitude, amplitude_exponent = canonical_element(
        numerators[program.output_index], exponent
    )
    probability_value, probability_exponent = probability(
        amplitude, amplitude_exponent
    )
    return {
        "amplitude_commitment": k_commitment(amplitude, amplitude_exponent),
        "probability_commitment": k_commitment(
            probability_value, probability_exponent
        ),
        "probability_rational": rational_probability(
            amplitude, amplitude_exponent
        ),
        "final_vector_commitment": vector_commitment(numerators, exponent),
        "final_denominator_exponent": exponent,
        "final_numerator_maximum_signed_bits": max(
            signed_bits(coefficient) for value in numerators for coefficient in value
        ),
        "final_total_exact_payload_bits": vector_payload_bits(numerators, exponent),
        "work": asdict(work),
    }


def run_transaction(
    port: CoherentPort, program: PublicProgram, generation: int, owner: int = 23701
) -> dict[str, object]:
    if port is None:
        raise TypeError("null coherent carrier")
    backings = (id(port.psi), id(port.scratch))
    work = Work()
    work.observe(port.psi, port.denominator_exponent)
    port.lease(owner, program, generation)
    for index in range(len(program.gates)):
        port.forward_gate(owner, program, generation, index, work)
    final_vector_commitment = vector_commitment(port.psi, port.denominator_exponent)
    amplitude, amplitude_exponent = port.project_final(owner, program, generation)
    retained_boundary = (amplitude, amplitude_exponent)
    final_denominator_exponent = port.denominator_exponent
    final_maximum_bits = max(
        signed_bits(coefficient) for value in port.psi for coefficient in value
    )
    final_payload = vector_payload_bits(port.psi, port.denominator_exponent)
    for index in range(len(program.gates) - 1, -1, -1):
        port.inverse_gate(owner, program, generation, index, work)
    exact_before_release = port.canonical()
    same_backings = backings == (id(port.psi), id(port.scratch))
    port.release(owner, program, generation)
    if retained_boundary != (amplitude, amplitude_exponent):
        raise RuntimeError("coherent final result was consumed by inverse")
    amplitude_commitment = k_commitment(amplitude, amplitude_exponent)
    probability_value, probability_exponent = probability(
        amplitude, amplitude_exponent
    )
    probability_commitment = k_commitment(
        probability_value, probability_exponent
    )
    probability_rational = rational_probability(amplitude, amplitude_exponent)
    baseline = direct_boundary(program)
    if (
        baseline["amplitude_commitment"] != amplitude_commitment
        or baseline["probability_commitment"] != probability_commitment
        or baseline["probability_rational"] != probability_rational
        or baseline["final_vector_commitment"] != final_vector_commitment
    ):
        raise RuntimeError("post-release identical five-vector baseline mismatch")
    return {
        "family": program.family,
        "depth": program.depth,
        "gate_count": len(program.gates),
        "logical_amplitude_cells": 5,
        "resident_integer_numerator_coordinates": 20,
        "scratch_integer_numerator_coordinates": 20,
        "final_denominator_exponent": final_denominator_exponent,
        "final_denominator_material_bits": denominator_material_bits(final_denominator_exponent),
        "final_numerator_maximum_signed_bits": final_maximum_bits,
        "final_total_exact_payload_bits": final_payload,
        "final_vector_commitment": final_vector_commitment,
        "amplitude_commitment": amplitude_commitment,
        "probability_commitment": probability_commitment,
        "probability_rational": probability_rational,
        "retained_final_amplitude_integer_coordinates_during_inverse": 4,
        "retained_final_amplitude_denominator_exponents_during_inverse": 1,
        "retained_final_amplitude_denominator_material_bits_during_inverse": denominator_material_bits(amplitude_exponent),
        "retained_one_way_vector_commitment_digest_bytes_during_inverse": 32,
        "canonical_post_inverse_state_exact": exact_before_release,
        "same_psi_and_scratch_backings": same_backings,
        "restoration_generation": generation,
        "baseline_reload_used": False,
        "work": asdict(work),
        "matched_classical": baseline,
    }


def expect_rejected(callback) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def dephased_witness_probability() -> Fraction:
    vector = [ONE, ZERO, ZERO, ZERO, ZERO]
    scratch = [ZERO] * 5
    exponent, _ = apply_fourier(vector, scratch, 0, 1)
    weights = []
    for value in vector:
        exact = rational_probability(value, exponent)
        if exact is None:
            raise RuntimeError("dephased input weight is not rational")
        weights.append(Fraction(exact[0], exact[1]))
    transition = rational_probability(SQRT5, 1)
    if transition is None:
        raise RuntimeError("Fourier transition weight is not rational")
    transition_weight = Fraction(transition[0], transition[1])
    return sum((weight * transition_weight for weight in weights), Fraction(0))


def controls() -> dict[str, bool]:
    algebra = {
        "cyclotomic_relation": k_add(k_add(k_add(k_add(ONE, zeta_power(1)), zeta_power(2)), zeta_power(3)), zeta_power(4)) == ZERO,
        "zeta_fifth_power": k_mul(zeta_power(4), zeta_power(1)) == ONE,
        "sqrt5_square": k_mul(SQRT5, SQRT5) == (5, 0, 0, 0),
        "conjugation": k_mul(zeta_power(1), k_conjugate(zeta_power(1))) == ONE,
    }
    # Exact destructive interference: F, D_(1,0), F on delta_0.
    witness_gates = (
        Gate("FOURIER"), Gate("CUBIC_PHASE", 1, 0), Gate("FOURIER")
    )
    witness = PublicProgram(
        0, 1, 0, witness_gates,
        program_digest(0, 1, 0, witness_gates),
    )
    witness_boundary = direct_boundary(witness)
    destructive_zero = witness_boundary["probability_rational"] == [0, 1]
    dephased_probability = dephased_witness_probability()
    # Replace the cubic phase by identity: F^2 delta_0 = delta_0.
    identity_gates = (Gate("FOURIER"), Gate("FOURIER"))
    identity = PublicProgram(
        0, 1, 0, identity_gates,
        program_digest(0, 1, 0, identity_gates),
    )
    identity_probability = direct_boundary(identity)["probability_rational"] == [1, 1]

    # F and D do not commute on a generic basis state.
    generic = [ZERO, ONE, ZERO, ZERO, ZERO]
    scratch = [ZERO] * 5
    left = list(generic)
    exp_left, _ = apply_fourier(left, scratch, 0, 1)
    exp_left = apply_phase(left, exp_left, 1, 0)
    right = list(generic)
    exp_right = apply_phase(right, 0, 1, 0)
    exp_right, _ = apply_fourier(right, scratch, exp_right, 1)
    noncommuting = (left, exp_left) != (right, exp_right)
    third_difference = (6 * 1) % 5 != 0

    program = PublicProgram.compile(0, 2)
    owner = 23711
    premature = CoherentPort()
    premature.lease(owner, program, 1)
    premature_projection = expect_rejected(
        lambda: premature.project_final(owner, program, 1)
    )
    full_vector_projection = expect_rejected(premature.project_full_vector)
    wrong_owner = expect_rejected(
        lambda: premature.forward_gate(owner + 1, program, 1, 0, Work())
    )
    wrong_program = expect_rejected(
        lambda: premature.forward_gate(owner, PublicProgram.compile(1, 2), 1, 0, Work())
    )
    wrong_generation = expect_rejected(
        lambda: premature.forward_gate(owner, program, 2, 0, Work())
    )
    wrong_type = expect_rejected(
        lambda: CoherentPort().lease(owner, program, 1, "WRONG")
    )
    forged_identity = expect_rejected(
        lambda: PublicProgram(
            program.family, program.depth, program.output_index,
            program.gates[:-1] + (Gate("CUBIC_PHASE", 2, 1),),
            program.program_id,
        )
    )
    null_carrier = expect_rejected(
        lambda: run_transaction(None, program, 1)  # type: ignore[arg-type]
    )
    dirty = CoherentPort()
    dirty.scratch[0] = ONE
    dirty_scratch = expect_rejected(lambda: dirty.lease(owner, program, 1))

    missing = CoherentPort()
    missing.lease(owner, program, 1)
    for index in range(len(program.gates)):
        missing.forward_gate(owner, program, 1, index, Work())
    missing_inverse = expect_rejected(lambda: missing.release(owner, program, 1))
    reordered = CoherentPort()
    reordered.lease(owner, program, 1)
    for index in range(len(program.gates)):
        reordered.forward_gate(owner, program, 1, index, Work())
    reordered_inverse = expect_rejected(
        lambda: reordered.inverse_gate(owner, program, 1, len(program.gates) - 2, Work())
    )
    wrong_inverse = CoherentPort()
    wrong_inverse.lease(owner, program, 1)
    for index in range(len(program.gates)):
        wrong_inverse.forward_gate(owner, program, 1, index, Work())
    last = program.gates[-1]
    wrong_inverse.denominator_exponent = execute_gate(
        wrong_inverse.psi, wrong_inverse.scratch,
        wrong_inverse.denominator_exponent, Gate("CUBIC_PHASE", 1, 0), True, Work()
    )
    wrong_inverse.cursor -= 1
    for index in range(len(program.gates) - 2, -1, -1):
        wrong_inverse.inverse_gate(owner, program, 1, index, Work())
    wrong_inverse_rejected = expect_rejected(
        lambda: wrong_inverse.release(owner, program, 1)
    )

    stale = CoherentPort()
    run_transaction(stale, PublicProgram.compile(0, 2), 1, owner)
    run_transaction(stale, PublicProgram.compile(1, 2), 2, owner)
    stale_generation = expect_rejected(
        lambda: stale.lease(owner, program, 2)
    )

    global_phase_program = PublicProgram.build(0, 1)
    global_vector = [k_mul(zeta_power(1), value) for value in [ONE, ZERO, ZERO, ZERO, ZERO]]
    phase_retained = global_vector[0] != ONE
    phase_probability_invariant = rational_probability(global_vector[0], 0) == [1, 1]

    return {
        **algebra,
        "fourier_dagger_fourier_identity": all(
            _basis_fourier_identity(index) for index in range(5)
        ),
        "cubic_phase_unitary_identity": all(
            k_mul(zeta_power(x**3), zeta_power(-(x**3))) == ONE for x in range(5)
        ),
        "cubic_third_finite_difference_nonzero": third_difference,
        "fourier_and_cubic_phase_noncommute": noncommuting,
        "exact_cubic_destructive_interference_zero": destructive_zero,
        "dephased_control_probability_one_over5": dephased_probability == Fraction(1, 5),
        "identity_phase_control_probability_one": identity_probability,
        "global_phase_retained_in_amplitude": phase_retained,
        "global_phase_cancels_only_in_probability": phase_probability_invariant,
        "premature_final_projection_rejected": premature_projection,
        "full_vector_projection_rejected": full_vector_projection,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "wrong_generation_rejected": wrong_generation,
        "wrong_type_rejected": wrong_type,
        "same_id_changed_descriptor_rejected": forged_identity,
        "null_carrier_rejected": null_carrier,
        "dirty_scratch_rejected": dirty_scratch,
        "missing_inverse_rejected": missing_inverse,
        "reordered_inverse_rejected": reordered_inverse,
        "wrong_inverse_rejected": wrong_inverse_rejected,
        "stale_generation_rejected": stale_generation,
        "intermediate_or_full_vectors_serialized": False,
        "path_histories_or_assignments_enumerated": False,
        "public_compiler_reads_final_answer": False,
    }


def _basis_fourier_identity(index: int) -> bool:
    vector = [ZERO] * 5
    vector[index] = ONE
    scratch = [ZERO] * 5
    exponent, _ = apply_fourier(vector, scratch, 0, 1)
    exponent, _ = apply_fourier(vector, scratch, exponent, -1)
    return vector == [ONE if position == index else ZERO for position in range(5)] and exponent == 0


def comparable_case(case: dict[str, object]) -> dict[str, object]:
    keys = (
        "family", "depth", "gate_count", "logical_amplitude_cells",
        "resident_integer_numerator_coordinates", "scratch_integer_numerator_coordinates",
        "final_denominator_exponent", "final_denominator_material_bits",
        "final_numerator_maximum_signed_bits", "final_total_exact_payload_bits",
        "final_vector_commitment", "amplitude_commitment",
        "probability_commitment", "probability_rational",
        "retained_final_amplitude_integer_coordinates_during_inverse",
        "retained_final_amplitude_denominator_exponents_during_inverse",
        "retained_final_amplitude_denominator_material_bits_during_inverse",
        "retained_one_way_vector_commitment_digest_bytes_during_inverse",
        "canonical_post_inverse_state_exact", "same_psi_and_scratch_backings",
        "restoration_generation", "baseline_reload_used",
    )
    return {key: case[key] for key in keys}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text())
    production_controls = controls()
    if production_controls != reference["controls"]:
        raise RuntimeError("independent coherent control parity failed")
    cases = []
    for family in (0, 1):
        for depth in DEPTHS:
            program = PublicProgram.compile(family, depth)
            cases.append(run_transaction(CoherentPort(), program, 1))
    if [comparable_case(case) for case in cases] != reference["cases"]:
        raise RuntimeError("independent coherent case parity failed")

    primary_program = PublicProgram.compile(0, 32)
    reuse_program = PublicProgram.compile(1, 16)
    shared = CoherentPort()
    primary = run_transaction(shared, primary_program, 1)
    reuse = run_transaction(shared, reuse_program, 2)
    fresh_reuse = run_transaction(CoherentPort(), reuse_program, 1)
    if comparable_case(reuse) | {"restoration_generation": 1} != comparable_case(fresh_reuse):
        raise RuntimeError("fresh/restored coherent reuse mismatch")
    if reference["reuse"]["reuse"]["amplitude_commitment"] != reuse["amplitude_commitment"]:
        raise RuntimeError("independent coherent reuse parity failed")

    family0 = [case for case in cases if case["family"] == 0]
    family1 = [case for case in cases if case["family"] == 1]
    here = Path(__file__).resolve()
    output = {
        "schema": "cat_cas.zeta5_normalized_cubic_fourier_coherent_port_result.v1",
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": cases,
        "controls": production_controls,
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh_reuse,
            "restoration_generation_after_reuse": shared.last_restored_generation,
            "fresh_restored_boundary_agreement": reuse["amplitude_commitment"] == fresh_reuse["amplitude_commitment"] and reuse["probability_commitment"] == fresh_reuse["probability_commitment"],
            "fresh_restored_resource_signature_agreement": reuse["final_denominator_exponent"] == fresh_reuse["final_denominator_exponent"] and reuse["final_total_exact_payload_bits"] == fresh_reuse["final_total_exact_payload_bits"],
            "same_backing_across_primary_and_reuse": primary["same_psi_and_scratch_backings"] and reuse["same_psi_and_scratch_backings"],
        },
        "coherent_phase_law": {
            "field": "Q(zeta_5)",
            "physical_embedding": "zeta_5=exp(2*pi*i/5)",
            "sqrt5": "-1-2*zeta_5^2-2*zeta_5^3",
            "normalized_fourier_factor": "sqrt5/5",
            "cubic_phase": "zeta_5^(a*x^3+b*x)",
            "logical_amplitude_cells": 5,
            "global_phase_preserved": True,
            "coherent_destructive_interference_exact": True,
            "stationary_or_path_histories_enumerated": False,
            "final_boundary_only": True,
            "direct_process_logical_custody_only": True,
        },
        "resource_law": {
            "depths": list(DEPTHS),
            "family0_final_denominator_exponents": [case["final_denominator_exponent"] for case in family0],
            "family1_final_denominator_exponents": [case["final_denominator_exponent"] for case in family1],
            "family0_final_total_exact_payload_bits": [case["final_total_exact_payload_bits"] for case in family0],
            "family1_final_total_exact_payload_bits": [case["final_total_exact_payload_bits"] for case in family1],
            "resident_integer_numerator_coordinates": 20,
            "scratch_integer_numerator_coordinates": 20,
            "denominator_material_value_counted": True,
            "retained_dynamic_inverse_history_entries": 0,
            "carrier_initial_state_snapshot_retained": False,
            "whole_transaction_live_cell_and_payload_accounting_complete": False,
            "python_objects_allocator_hash_serialization_rss_excluded_not_zero": True,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "matched_classical": {
            "strongest_implemented": "IDENTICAL_EXACT_FIVE_AMPLITUDE_QZETA5_RECURRENCE",
            "equivalent_single_boundary_form": "BACKWARD_FIVE_COMPONENT_ROW_RECURRENCE",
            "executed_only_after_carrier_release": True,
            "explicit_path_sum_used": False,
            "dephased_density_matrix_used_as_baseline": False,
            "same_exact_boundary": True,
            "computational_advantage": False,
            "distinct_phase_resource": False,
        },
        "separate_reference": {
            "imports_m237_production": False,
            "uses_independent_exact_matrix_recurrence": True,
            "implements_independent_custody_state_machine": True,
            "case_control_reuse_parity": True,
        },
        "source_dependencies": {
            "production_sha256": file_sha256(here),
            "separate_reference_sha256": file_sha256(
                here.with_name("zeta5_normalized_cubic_fourier_coherent_port_separate_reference.py")
            ),
        },
        "claim_limits": {
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "general_tensor_network_contraction": False,
            "machine_enforced_catvm_custody": False,
            "small_wall_crossed": False,
            "unbounded_catalytic_computation_established": False,
            "fixed_bounded_width_exact_state_established": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
        },
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: zeta5_normalized_cubic_fourier_coherent_port.py REFERENCE_JSON")
    main(Path(sys.argv[1]))
