#!/usr/bin/env python3
"""Exact bounded Ising-anyon controlled-holonomy diagnostic (M261).

The implementation evolves exact amplitudes under public adjacent Ising
braids and public Majorana-pair holonomies.  It never reads a stored fusion
charge or expected boundary to implement a transaction.  Coefficients live
in Z[i]/sqrt(2)^h, an exact subring of Q(zeta_16).

This is deterministic software, not QEMU device execution or physical anyon
execution, and it establishes no computational advantage or M257 escape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


CLAIM = (
    "IDEAL_EXACT_ISING_MTC_TRIANGULAR_ADJACENT_BRAID_PREPARATION_"
    "CONTROLLED_TRANSPORTED_MAJORANA_PAIR_HOLONOMY_FINAL_ONLY_BOUNDARY_"
    "FUNCTIONAL_RESTORATION_AND_REUSE_AT_N4_N8_N12_N16_WITH_"
    "CONTIGUOUS_CUT_RANKS2_4_8_AND_COMPACT_SIGNED_PAIRING_RESOURCE_KILL"
)
CLAIM_CEILING = (
    "IDEAL_DETERMINISTIC_EXACT_SOFTWARE_ISING_MTC_CONTROLLED_HOLONOMY_"
    "DIAGNOSTIC_AT_N4_N8_N12_N16_ONLY"
)
RESOURCE_KILL = (
    "TRIANGULAR_ISING_BRAIDS_AND_TRANSPORTED_PAIR_HOLONOMIES_REMAIN_"
    "COMPACTLY_TRACKABLE_BY_AN_O_N_SIZED_SIGNED_MAJORANA_PAIRING"
)


@dataclass(frozen=True)
class GaussianInt:
    real: int = 0
    imag: int = 0

    def __add__(self, other: "GaussianInt") -> "GaussianInt":
        return GaussianInt(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "GaussianInt") -> "GaussianInt":
        return GaussianInt(self.real - other.real, self.imag - other.imag)

    def __neg__(self) -> "GaussianInt":
        return GaussianInt(-self.real, -self.imag)

    def mul_i(self, power: int) -> "GaussianInt":
        power %= 4
        if power == 0:
            return self
        if power == 1:
            return GaussianInt(-self.imag, self.real)
        if power == 2:
            return -self
        return GaussianInt(self.imag, -self.real)

    def divisible_by_two(self) -> bool:
        return self.real % 2 == 0 and self.imag % 2 == 0

    def half(self) -> "GaussianInt":
        return GaussianInt(self.real // 2, self.imag // 2)

    def is_zero(self) -> bool:
        return self.real == 0 and self.imag == 0


ZERO = GaussianInt()
ONE = GaussianInt(1, 0)


@dataclass(frozen=True)
class ExactState:
    qubits: int
    coefficients: tuple[GaussianInt, ...]
    sqrt2_denominator_power: int


def canonical_state(
    qubits: int, coefficients: Sequence[GaussianInt], denominator_power: int
) -> ExactState:
    values = tuple(coefficients)
    while (
        denominator_power >= 2
        and any(not value.is_zero() for value in values)
        and all(value.divisible_by_two() for value in values)
    ):
        values = tuple(value.half() for value in values)
        denominator_power -= 2
    return ExactState(qubits, values, denominator_power)


def basis_state(qubits: int, occupation: int = 0) -> ExactState:
    coefficients = [ZERO] * (1 << qubits)
    coefficients[occupation] = ONE
    return ExactState(qubits, tuple(coefficients), 0)


def scale_i(state: ExactState, power: int) -> ExactState:
    return ExactState(
        state.qubits,
        tuple(value.mul_i(power) for value in state.coefficients),
        state.sqrt2_denominator_power,
    )


def state_commitment(state: ExactState) -> str:
    digest = hashlib.sha256()
    digest.update(f"{state.qubits}:{state.sqrt2_denominator_power}|".encode())
    for value in state.coefficients:
        digest.update(f"{value.real},{value.imag};".encode())
    return digest.hexdigest()


def support_size(state: ExactState) -> int:
    return sum(not value.is_zero() for value in state.coefficients)


def maximum_signed_bits(state: ExactState) -> int:
    return max(
        (
            max(abs(value.real), abs(value.imag)).bit_length() + 1
            for value in state.coefficients
            if not value.is_zero()
        ),
        default=0,
    )


def norm_squared(state: ExactState) -> Fraction:
    numerator = sum(
        value.real * value.real + value.imag * value.imag
        for value in state.coefficients
    )
    return Fraction(numerator, 1 << state.sqrt2_denominator_power)


def overlap_squared(left: ExactState, right: ExactState) -> Fraction:
    if left.qubits != right.qubits:
        raise ValueError("overlap qubit mismatch")
    real = 0
    imag = 0
    for first, second in zip(left.coefficients, right.coefficients):
        real += first.real * second.real + first.imag * second.imag
        imag += first.real * second.imag - first.imag * second.real
    return Fraction(
        real * real + imag * imag,
        1 << (left.sqrt2_denominator_power + right.sqrt2_denominator_power),
    )


def majorana_on_basis(majorana_zero_based: int, basis: int) -> tuple[int, int]:
    """Return destination and i-power for one zero-based JW Majorana."""

    mode = majorana_zero_based // 2
    prefix_parity = (basis & ((1 << mode) - 1)).bit_count() & 1
    occupation = (basis >> mode) & 1
    destination = basis ^ (1 << mode)
    if majorana_zero_based % 2 == 0:
        phase = 2 if prefix_parity else 0
    else:
        phase = 1 + (2 if prefix_parity ^ occupation else 0)
    return destination, phase


def apply_majorana_pair(
    state: ExactState, first_one_based: int, second_one_based: int
) -> ExactState:
    if not (1 <= first_one_based < second_one_based <= 2 * state.qubits):
        raise ValueError("Majorana pair outside state")
    first = first_one_based - 1
    second = second_one_based - 1
    output = [ZERO] * len(state.coefficients)
    for basis, coefficient in enumerate(state.coefficients):
        if coefficient.is_zero():
            continue
        after_second, phase_second = majorana_on_basis(second, basis)
        destination, phase_first = majorana_on_basis(first, after_second)
        output[destination] = output[destination] + coefficient.mul_i(
            phase_second + phase_first
        )
    return ExactState(state.qubits, tuple(output), state.sqrt2_denominator_power)


def apply_pair_rotation(
    state: ExactState,
    first_one_based: int,
    second_one_based: int,
    *,
    inverse: bool = False,
) -> ExactState:
    """Apply (I-gamma_a gamma_b)/sqrt(2), or its public adjoint."""

    coupled = apply_majorana_pair(state, first_one_based, second_one_based)
    output = [
        left + right if inverse else left - right
        for left, right in zip(state.coefficients, coupled.coefficients)
    ]
    result = canonical_state(
        state.qubits, output, state.sqrt2_denominator_power + 1
    )
    if norm_squared(result) != norm_squared(state):
        raise AssertionError("Majorana-pair rotation did not preserve exact norm")
    return result


def apply_clockwise_exchange(
    state: ExactState, generator_one_based: int, *, inverse: bool = False
) -> ExactState:
    if not (1 <= generator_one_based < 2 * state.qubits):
        raise ValueError("adjacent braid generator outside state")
    return apply_pair_rotation(
        state,
        generator_one_based,
        generator_one_based + 1,
        inverse=inverse,
    )


def public_triangular_word(r: int) -> tuple[int, ...]:
    if r < 1:
        raise ValueError("r must be positive")
    return tuple(
        generator
        for t in range(1, 2 * r - 2)
        for generator in range(2 * t + 2, t + 2, -1)
    )


def prepare_carrier(r: int) -> tuple[ExactState, tuple[int, ...]]:
    pair_charge_modes = 2 * r
    word = public_triangular_word(r)
    state = basis_state(pair_charge_modes)
    for generator in word:
        state = apply_clockwise_exchange(state, generator)
    if norm_squared(state) != 1:
        raise AssertionError("prepared carrier is not exactly normalized")
    if any(
        basis.bit_count() % 2
        for basis, value in enumerate(state.coefficients)
        if not value.is_zero()
    ):
        raise AssertionError("preparation left the declared even-charge sector")
    return state, word


@dataclass(frozen=True)
class GaussianFraction:
    real: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)

    def __add__(self, other: "GaussianFraction") -> "GaussianFraction":
        return GaussianFraction(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "GaussianFraction") -> "GaussianFraction":
        return GaussianFraction(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: "GaussianFraction") -> "GaussianFraction":
        return GaussianFraction(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def inverse(self) -> "GaussianFraction":
        denominator = self.real * self.real + self.imag * self.imag
        if denominator == 0:
            raise ZeroDivisionError
        return GaussianFraction(self.real / denominator, -self.imag / denominator)

    def __truediv__(self, other: "GaussianFraction") -> "GaussianFraction":
        return self * other.inverse()

    def is_zero(self) -> bool:
        return self.real == 0 and self.imag == 0


def exact_matrix_rank(matrix: Sequence[Sequence[GaussianInt]]) -> int:
    rows = [
        [GaussianFraction(Fraction(value.real), Fraction(value.imag)) for value in row]
        for row in matrix
    ]
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(pivot_row, row_count) if not rows[row][column].is_zero()),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot_value for value in rows[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or rows[row][column].is_zero():
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * normalized
                for value, normalized in zip(rows[row], rows[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def bipartite_rank(state: ExactState, left_modes: Sequence[int]) -> int:
    left = tuple(left_modes)
    right = tuple(mode for mode in range(state.qubits) if mode not in left)
    matrix = [[ZERO for _ in range(1 << len(right))] for _ in range(1 << len(left))]
    for basis, coefficient in enumerate(state.coefficients):
        row = sum(((basis >> mode) & 1) << index for index, mode in enumerate(left))
        column = sum(
            ((basis >> mode) & 1) << index for index, mode in enumerate(right)
        )
        matrix[row][column] = coefficient
    return exact_matrix_rank(matrix)


def extend_with_zero_qubits(state: ExactState, added: int) -> ExactState:
    coefficients = [ZERO] * (1 << (state.qubits + added))
    for basis, value in enumerate(state.coefficients):
        coefficients[basis] = value
    return ExactState(
        state.qubits + added, tuple(coefficients), state.sqrt2_denominator_power
    )


def apply_hadamard(state: ExactState, qubit: int) -> ExactState:
    output = [ZERO] * len(state.coefficients)
    for basis in range(len(state.coefficients)):
        if (basis >> qubit) & 1:
            continue
        partner = basis | (1 << qubit)
        lower = state.coefficients[basis]
        upper = state.coefficients[partner]
        output[basis] = lower + upper
        output[partner] = lower - upper
    result = canonical_state(
        state.qubits, output, state.sqrt2_denominator_power + 1
    )
    if norm_squared(result) != norm_squared(state):
        raise AssertionError("Hadamard did not preserve exact norm")
    return result


def apply_cnot(state: ExactState, control: int, target: int) -> ExactState:
    if control == target:
        raise ValueError("CNOT control and target must differ")
    output = [ZERO] * len(state.coefficients)
    for basis, value in enumerate(state.coefficients):
        destination = basis ^ (1 << target) if basis & (1 << control) else basis
        output[destination] = output[destination] + value
    return ExactState(state.qubits, tuple(output), state.sqrt2_denominator_power)


def apply_pair_holonomy(
    state: ExactState,
    data_modes: int,
    path_qubit: int,
    loop: tuple[int, int],
    orientation: int,
) -> ExactState:
    """Apply controlled L=-i gamma_a gamma_b to the coherent path branch."""

    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1")
    first, second = loop
    if not (1 <= first < second <= 2 * data_modes):
        raise ValueError("holonomy Majorana pair outside data carrier")
    output = [ZERO] * len(state.coefficients)
    for basis, value in enumerate(state.coefficients):
        if value.is_zero():
            continue
        if not ((basis >> path_qubit) & 1):
            output[basis] = output[basis] + value
            continue
        after_second, phase_second = majorana_on_basis(second - 1, basis)
        destination, phase_first = majorana_on_basis(first - 1, after_second)
        output[destination] = output[destination] + value.mul_i(
            phase_second + phase_first + 3
        )
    result = ExactState(state.qubits, tuple(output), state.sqrt2_denominator_power)
    if norm_squared(result) != norm_squared(state):
        raise AssertionError("pair holonomy did not preserve exact norm")
    # These Z2 pair parities are Hermitian involutions; orientation is retained
    # in the descriptor but has the same exact action.
    return result


def branch_weight(state: ExactState, qubit: int, value: int) -> Fraction:
    numerator = sum(
        coefficient.real * coefficient.real + coefficient.imag * coefficient.imag
        for basis, coefficient in enumerate(state.coefficients)
        if ((basis >> qubit) & 1) == value
    )
    return Fraction(numerator, 1 << state.sqrt2_denominator_power)


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def expected_restored_joint(carrier: ExactState, response_bit: int) -> ExactState:
    joint = extend_with_zero_qubits(carrier, 2)
    if not response_bit:
        return joint
    response_qubit = carrier.qubits + 1
    output = [ZERO] * len(joint.coefficients)
    for basis, value in enumerate(joint.coefficients):
        if not value.is_zero():
            destination = basis | (1 << response_qubit)
            output[destination] = output[destination] + value
    return ExactState(joint.qubits, tuple(output), joint.sqrt2_denominator_power)


def extract_restored_carrier(
    joint: ExactState, data_modes: int, response_bit: int
) -> ExactState:
    path_qubit = data_modes
    response_qubit = data_modes + 1
    coefficients = [ZERO] * (1 << data_modes)
    for basis, value in enumerate(joint.coefficients):
        if ((basis >> path_qubit) & 1) != 0:
            continue
        if ((basis >> response_qubit) & 1) != response_bit:
            continue
        coefficients[basis & ((1 << data_modes) - 1)] = value
    return canonical_state(data_modes, coefficients, joint.sqrt2_denominator_power)


def transaction(
    carrier: ExactState,
    loop: tuple[int, int],
    *,
    orientation: int = 1,
    inverse_loop: tuple[int, int] | None = None,
    omit_inverse: bool = False,
    retain_boundary_copy: bool = True,
    framing: int = 0,
) -> tuple[dict[str, object], ExactState | None]:
    if framing != 0:
        return (
            {
                "descriptor_rejected": True,
                "error": "NONZERO_FRAMING_NOT_IN_ACCEPTED_EXACT_CONTRACT",
                "carrier_unchanged": True,
                "response_released": False,
            },
            None,
        )
    data_modes = carrier.qubits
    path_qubit = data_modes
    response_qubit = data_modes + 1
    joint = extend_with_zero_qubits(carrier, 2)
    joint = apply_hadamard(joint, path_qubit)
    first_branch_weights = (
        branch_weight(joint, path_qubit, 0),
        branch_weight(joint, path_qubit, 1),
    )
    joint = apply_pair_holonomy(joint, data_modes, path_qubit, loop, orientation)
    joint = apply_hadamard(joint, path_qubit)
    forward_weights = (
        branch_weight(joint, path_qubit, 0),
        branch_weight(joint, path_qubit, 1),
    )
    if retain_boundary_copy:
        joint = apply_cnot(joint, path_qubit, response_qubit)
    joint = apply_hadamard(joint, path_qubit)
    if not omit_inverse:
        selected_inverse = loop if inverse_loop is None else inverse_loop
        joint = apply_pair_holonomy(
            joint, data_modes, path_qubit, selected_inverse, -orientation
        )
    joint = apply_hadamard(joint, path_qubit)

    fidelities = {
        str(bit): fraction_text(overlap_squared(joint, expected_restored_joint(carrier, bit)))
        for bit in (0, 1)
    }
    matches = [
        bit for bit in (0, 1) if joint == expected_restored_joint(carrier, bit)
    ]
    if retain_boundary_copy:
        accepted = len(matches) == 1
        boundary = matches[0] if accepted else None
        response_released = accepted
    else:
        accepted = joint == expected_restored_joint(carrier, 0)
        boundary = None
        response_released = False
    restored = (
        extract_restored_carrier(joint, data_modes, matches[0])
        if accepted and matches
        else None
    )
    if restored is not None and restored != carrier:
        raise AssertionError("accepted transaction did not restore exact carrier value")
    record: dict[str, object] = {
        "descriptor_rejected": False,
        "loop_majoranas_one_based": list(loop),
        "loop_semantics": "BRAID_TRANSPORTED_MAJORANA_PAIR_PARITY_HOLONOMY",
        "orientation": orientation,
        "framing": framing,
        "initial_coherent_path_weights": [
            fraction_text(first_branch_weights[0]),
            fraction_text(first_branch_weights[1]),
        ],
        "forward_port_even_weight": fraction_text(forward_weights[0]),
        "forward_port_odd_weight": fraction_text(forward_weights[1]),
        "forward_probe_factorized": 0 in forward_weights,
        "private_boundary_copy_retained": retain_boundary_copy,
        "inverse_derived_from_public_loop": inverse_loop is None,
        "inverse_executed": not omit_inverse,
        "exact_functional_restoration": accepted,
        "response_released_after_restoration_only": response_released,
        "boundary_bit": boundary,
        "restored_form_fidelities": fidelities,
        "same_backing_restoration_established": False,
        "snapshot_or_baseline_reload_used": False,
        "retained_dynamic_inverse_history": 0,
        "joint_final_commitment": state_commitment(joint),
    }
    return record, restored


def loop_eigenvalue(carrier: ExactState, loop: tuple[int, int]) -> int | None:
    joint = extend_with_zero_qubits(carrier, 1)
    # The control path must be one for the holonomy to act.
    path_one = [ZERO] * len(joint.coefficients)
    for basis, value in enumerate(joint.coefficients):
        if not value.is_zero():
            destination = basis | (1 << carrier.qubits)
            path_one[destination] = path_one[destination] + value
    path_state = ExactState(joint.qubits, tuple(path_one), joint.sqrt2_denominator_power)
    looped = apply_pair_holonomy(
        path_state, carrier.qubits, carrier.qubits, loop, 1
    )
    expected_plus = path_state
    expected_minus = scale_i(path_state, 2)
    if looped == expected_plus:
        return 1
    if looped == expected_minus:
        return -1
    return None


def pair_parity_expectation(
    carrier: ExactState, loop: tuple[int, int]
) -> GaussianFraction:
    transformed = scale_i(apply_majorana_pair(carrier, *loop), 3)
    real = 0
    imag = 0
    for first, second in zip(carrier.coefficients, transformed.coefficients):
        real += first.real * second.real + first.imag * second.imag
        imag += first.real * second.imag - first.imag * second.real
    denominator = 1 << carrier.sqrt2_denominator_power
    return GaussianFraction(Fraction(real, denominator), Fraction(imag, denominator))


def signed_pairing_after_word(
    majorana_count: int, word: Sequence[int]
) -> tuple[tuple[int, int, int], ...]:
    """Formula-specific O(n) signed-pairing comparator for the public word."""

    position = list(range(majorana_count + 1))
    occupant = list(range(majorana_count + 1))
    sign = [1] * (majorana_count + 1)
    for generator in word:
        at_left = occupant[generator]
        at_right = occupant[generator + 1]
        position[at_left] = generator + 1
        position[at_right] = generator
        sign[at_right] *= -1
        occupant[generator], occupant[generator + 1] = at_right, at_left
    pairs: list[tuple[int, int, int]] = []
    for original in range(1, majorana_count + 1, 2):
        first = position[original]
        second = position[original + 1]
        pair_sign = sign[original] * sign[original + 1]
        if first > second:
            first, second = second, first
            pair_sign *= -1
        pairs.append((first, second, pair_sign))
    return tuple(sorted(pairs))


def pairing_loop_eigenvalue(
    pairing: Sequence[tuple[int, int, int]], loop: tuple[int, int]
) -> int | None:
    return next(
        (sign for first, second, sign in pairing if (first, second) == loop),
        None,
    )


def public_scramble_word(r: int) -> tuple[int, ...]:
    return tuple(range(2, 4 * r - 2))


def apply_synthesized_scramble(
    carrier: ExactState, r: int
) -> tuple[ExactState, ExactState, tuple[int, ...]]:
    """Apply U=P^dagger C_(4r-2) P, then its public exact inverse."""

    path = public_scramble_word(r)
    state = carrier
    for generator in path:
        state = apply_clockwise_exchange(state, generator)
    state = apply_clockwise_exchange(state, 4 * r - 2)
    for generator in reversed(path):
        state = apply_clockwise_exchange(state, generator, inverse=True)
    scrambled = state
    for generator in path:
        state = apply_clockwise_exchange(state, generator)
    state = apply_clockwise_exchange(state, 4 * r - 2, inverse=True)
    for generator in reversed(path):
        state = apply_clockwise_exchange(state, generator, inverse=True)
    return scrambled, state, path


def scaling_fixture(r: int) -> dict[str, object]:
    anyons = 4 * r
    pair_charge_modes = 2 * r
    carrier, word = prepare_carrier(r)
    rank = bipartite_rank(carrier, tuple(range(r)))
    expected_rank = 1 << (r - 1)
    if rank != expected_rank:
        raise AssertionError(f"n={anyons} contiguous rank {rank} != {expected_rank}")

    loop_zero = (3, 2 * r + 1)
    loop_one = (4, 2 * r + 2)
    mixed_loop = (3, 4)
    eigen_zero = loop_eigenvalue(carrier, loop_zero)
    eigen_one = loop_eigenvalue(carrier, loop_one)
    eigen_mixed = loop_eigenvalue(carrier, mixed_loop)
    if (eigen_zero, eigen_one, eigen_mixed) != (1, -1, None):
        raise AssertionError("declared transported-loop eigensystem changed")

    generation1, restored = transaction(carrier, loop_zero)
    if restored is None:
        raise AssertionError("generation-1 loop failed")
    generation2, restored_twice = transaction(restored, loop_one)
    if restored_twice != carrier:
        raise AssertionError("generation-2 returned-value reuse failed")
    mixed_retained, _ = transaction(carrier, mixed_loop)
    mixed_unlatched, mixed_restored = transaction(
        carrier, mixed_loop, retain_boundary_copy=False
    )
    if (
        mixed_retained["forward_port_even_weight"] != "1/2"
        or mixed_retained["forward_port_odd_weight"] != "1/2"
        or mixed_retained["response_released_after_restoration_only"]
        or sum(Fraction(value) for value in mixed_retained["restored_form_fidelities"].values())
        != Fraction(1, 2)
        or mixed_restored != carrier
        or not mixed_unlatched["exact_functional_restoration"]
    ):
        raise AssertionError("mixed-loop coherent-copy control changed")

    reverse_orientation, reverse_restored = transaction(
        carrier, loop_one, orientation=-1
    )
    missing_inverse, _ = transaction(carrier, loop_one, omit_inverse=True)
    wrong_inverse, _ = transaction(carrier, loop_zero, inverse_loop=loop_one)
    framing_fault, _ = transaction(carrier, loop_zero, framing=1)
    if missing_inverse["response_released_after_restoration_only"]:
        raise AssertionError("missing inverse released a response")
    if wrong_inverse["response_released_after_restoration_only"]:
        raise AssertionError("wrong inverse released a response")

    scramble_loop = (2, 4 * r - 1)
    scramble_expectation = pair_parity_expectation(carrier, scramble_loop)
    if (
        loop_eigenvalue(carrier, scramble_loop) is not None
        or not scramble_expectation.is_zero()
    ):
        raise AssertionError("scramble generator unexpectedly has definite parity")
    direct_scrambled = apply_pair_rotation(carrier, *scramble_loop)
    scrambled, unscrambled, transport_path = apply_synthesized_scramble(carrier, r)
    if not (scrambled == direct_scrambled or scrambled == scale_i(direct_scrambled, 2)):
        raise AssertionError("adjacent synthesis disagrees with direct scramble rotation")
    if unscrambled != carrier:
        raise AssertionError("public scramble inverse did not restore carrier")
    if overlap_squared(carrier, scrambled) != Fraction(1, 2):
        raise AssertionError("scramble overlap changed")
    scrambled_first, scrambled_restored = transaction(scrambled, loop_zero)
    scrambled_second, scrambled_restored_twice = transaction(
        scrambled_restored if scrambled_restored is not None else scrambled,
        loop_one,
    )
    if (
        scrambled_first["boundary_bit"] != generation1["boundary_bit"]
        or scrambled_second["boundary_bit"] != generation2["boundary_bit"]
        or scrambled_restored_twice != scrambled
    ):
        raise AssertionError("same-sector scramble changed commuting loop boundary")

    pairing = signed_pairing_after_word(4 * r, word)
    comparator_values = {
        "L0": pairing_loop_eigenvalue(pairing, loop_zero),
        "L1": pairing_loop_eigenvalue(pairing, loop_one),
        "mixed": pairing_loop_eigenvalue(pairing, mixed_loop),
    }
    if comparator_values != {"L0": 1, "L1": -1, "mixed": None}:
        raise AssertionError("signed-pairing comparator disagrees with amplitudes")

    return {
        "sigma_anyons": anyons,
        "pair_charge_modes": pair_charge_modes,
        "fixed_total_charge": "1",
        "fixed_sector_dimension": 1 << (pair_charge_modes - 1),
        "allocated_full_occupation_cells": 1 << pair_charge_modes,
        "allocated_transaction_joint_coefficient_cells": 1 << (pair_charge_modes + 2),
        "allocated_rank_matrix_gaussian_cells": 1 << pair_charge_modes,
        "public_triangular_execution_word": list(word),
        "public_preparation_braid_count": len(word),
        "carrier_support_cells": support_size(carrier),
        "carrier_commitment": state_commitment(carrier),
        "sqrt2_denominator_power": carrier.sqrt2_denominator_power,
        "maximum_integer_coefficient_signed_bits": maximum_signed_bits(carrier),
        "contiguous_cut": {
            "left_pair_charge_modes_zero_based": list(range(r)),
            "right_pair_charge_modes_zero_based": list(range(r, 2 * r)),
            "exact_amplitude_flattening_rank": rank,
            "expected_triangular_family_rank": expected_rank,
        },
        "transported_holonomies": {
            "semantic_ceiling": "BRAID_TRANSPORTED_MAJORANA_PAIR_PARITY_HOLONOMY",
            "W": "PUBLIC_TRIANGULAR_BRAID_WORD",
            "local_pair_parity": "S_k=-i gamma_(2k-1) gamma_(2k)",
            "L0": {
                "majoranas_one_based": list(loop_zero),
                "exact_eigenvalue": eigen_zero,
                "transport_formula": "L0=W S_2 W_dagger",
            },
            "L1": {
                "majoranas_one_based": list(loop_one),
                "exact_eigenvalue": eigen_one,
                "transport_formula": "L1=-W S_3 W_dagger",
            },
            "literal_enclosed_subset_without_transport_descriptor_claimed": False,
        },
        "generation1_L0": generation1,
        "generation2_distinct_L1": generation2,
        "functional_returned_value_reuse": {
            "same_returned_value_consumed_by_generation2": True,
            "second_prepare_used": False,
            "exact_value_restored_twice": restored_twice == carrier,
            "same_backing_reuse_established": False,
        },
        "mixed_loop_control": {
            "loop_majoranas_one_based": list(mixed_loop),
            "exact_eigenvalue": eigen_mixed,
            "retained_copy_transaction": mixed_retained,
            "restored_prepared_carrier_marginal_weight": "1/2",
            "maximum_factorized_fidelity_allowing_arbitrary_ancilla": "1/2",
            "each_computational_basis_response_restored_form_fidelity": "1/4",
            "result_free_latch_unlatch": mixed_unlatched,
            "result_free_exact_restore": mixed_restored == carrier,
        },
        "same_sector_scramble_control": {
            "K_majoranas_one_based": list(scramble_loop),
            "K_expectation_real": fraction_text(scramble_expectation.real),
            "K_expectation_imag": fraction_text(scramble_expectation.imag),
            "U_K": "(I-gamma_2 gamma_(4r-1))/sqrt(2)",
            "adjacent_transport_path_P": list(transport_path),
            "adjacent_synthesis": "P_dagger C_(4r-2) P",
            "direct_and_synthesized_exact_state_agree_up_to_sign": True,
            "changes_projective_carrier": True,
            "carrier_overlap_squared": "1/2",
            "preserves_L0_boundary": True,
            "preserves_L1_boundary": True,
            "public_adjoint_exactly_restores": unscrambled == carrier,
            "scrambled_value_transactions_restore": scrambled_restored_twice == scrambled,
        },
        "path_dephasing_control": {
            "exact_pre_dephasing_path_weights": ["1/2", "1/2"],
            "exact_post_recombination_port_weights": ["1/2", "1/2"],
            "deterministic_boundary_survives": False,
            "density_matrix_materialized": False,
            "derivation": "REMOVING_THE_EXACT_OFF_DIAGONAL_PATH_BLOCKS_PREVENTS_INTERFERENCE",
        },
        "orientation_control": {
            "reverse_orientation_boundary": reverse_orientation["boundary_bit"],
            "reverse_orientation_restores": reverse_restored == carrier,
            "expected_failure_applicable": False,
            "reason": "HERMITIAN_Z2_PAIR_PARITY_HAS_PHASE_PLUS_OR_MINUS_ONE",
        },
        "framing_control": framing_fault,
        "missing_inverse_control": missing_inverse,
        "wrong_inverse_loop_control": wrong_inverse,
        "signed_pairing_comparator": {
            "state_cells": len(pairing),
            "public_word_updates": len(word),
            "pairing": [list(item) for item in pairing],
            "loop_eigenvalues": comparator_values,
            "matches_exact_amplitude_transactions": True,
        },
    }


def smoke_fixture() -> dict[str, object]:
    carrier, word = prepare_carrier(1)
    local_loop = (1, 2)
    vacuum_copy, vacuum_restored = transaction(carrier, local_loop)
    vacuum_unlatched, vacuum_unlatched_restored = transaction(
        carrier, local_loop, retain_boundary_copy=False
    )
    excited = apply_clockwise_exchange(carrier, 2)
    excited = apply_clockwise_exchange(excited, 2)
    excited_copy, excited_restored = transaction(excited, local_loop)
    excited_unlatched, excited_unlatched_restored = transaction(
        excited, local_loop, retain_boundary_copy=False
    )
    returned = excited_restored if excited_restored is not None else excited
    returned = apply_clockwise_exchange(returned, 2, inverse=True)
    returned = apply_clockwise_exchange(returned, 2, inverse=True)
    if (
        loop_eigenvalue(carrier, local_loop) != 1
        or loop_eigenvalue(excited, local_loop) != -1
        or support_size(excited) != 1
        or excited.coefficients[3].is_zero()
        or vacuum_copy["boundary_bit"] != 0
        or excited_copy["boundary_bit"] != 1
        or vacuum_restored != carrier
        or vacuum_unlatched_restored != carrier
        or excited_restored != excited
        or excited_unlatched_restored != excited
        or returned != carrier
    ):
        raise AssertionError("n=4 lifecycle smoke changed")
    return {
        "sigma_anyons": 4,
        "pair_charge_modes": 2,
        "scope": "ALGEBRA_AND_TWO_GENERATION_LIFECYCLE_SMOKE_ONLY",
        "public_triangular_execution_word": list(word),
        "public_smoke_excitation_word": [2, 2],
        "public_smoke_restoration_word": ["C_2_dagger", "C_2_dagger"],
        "public_smoke_forward_braid_count": 2,
        "carrier_commitment": state_commitment(carrier),
        "excited_commitment": state_commitment(excited),
        "excited_is_occupation_11_up_to_exact_phase": True,
        "allocated_full_occupation_cells": 4,
        "allocated_transaction_joint_coefficient_cells": 16,
        "allocated_rank_matrix_gaussian_cells": 4,
        "fixed_sector_dimension": 2,
        "contiguous_cut_exact_rank": bipartite_rank(carrier, (0,)),
        "local_loop_majoranas_one_based": list(local_loop),
        "vacuum_loop_eigenvalue": 1,
        "excited_loop_eigenvalue": -1,
        "vacuum_retained_copy_transaction": vacuum_copy,
        "vacuum_result_free_latch_unlatch": vacuum_unlatched,
        "excited_retained_copy_transaction": excited_copy,
        "excited_result_free_latch_unlatch": excited_unlatched,
        "public_adjoint_excitation_word_exactly_restores_vacuum": returned == carrier,
        "scaling_cross_cut_loop_formula_claimed": False,
    }


def build_result() -> dict[str, object]:
    smoke = smoke_fixture()
    scaling = [scaling_fixture(r) for r in (2, 3, 4)]
    ranks = [item["contiguous_cut"]["exact_amplitude_flattening_rank"] for item in scaling]
    if ranks != [2, 4, 8]:
        raise AssertionError("contiguous scaling ranks changed")
    explicit_words = [item["public_triangular_execution_word"] for item in scaling]
    if explicit_words != [
        [4],
        [4, 6, 5, 8, 7, 6],
        [4, 6, 5, 8, 7, 6, 10, 9, 8, 7, 12, 11, 10, 9, 8],
    ]:
        raise AssertionError("public triangular word changed")

    return {
        "schema": "phase_qemu_v3_exact_ising_triangular_holonomy_diagnostic_v1",
        "milestone": "M261",
        "experiment": "CONTROLLED_MANY_BODY_EIGENPHASE_HOLONOMY_SCATTERING_PHASE_QEMU_BACKEND",
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "exactness": {
            "coefficient_ring": "Z[i]/sqrt(2)^h_SUBRING_OF_Q(zeta_16)",
            "scientific_floating_point_decisions": 0,
            "braids": "EXACT_JORDAN_WIGNER_MAJORANA_ACTION",
            "ranks": "EXACT_GAUSSIAN_RATIONAL_ELIMINATION_ON_ACTUAL_AMPLITUDE_FLATTENINGS",
            "holonomy": "ACTUAL_CONTROLLED_MINUS_I_GAMMA_A_GAMMA_B_ACTION_ON_EVERY_RESIDENT_AMPLITUDE",
        },
        "verification_scope": {
            "algebra_ranks_holonomy_boundaries": "SEPARATE_REFERENCE_PARITY",
            "production_transaction_response_order": "PACKAGE_SELF_REVIEW_SOURCE_AUDITED",
            "path_dephasing_control": "ANALYTIC_EXACT_NOT_DENSITY_MATRIX_EXECUTED",
        },
        "public_family": {
            "sigma_anyons": "n=4r",
            "pair_charge_modes": "2r_WITH_GLOBAL_EVEN_PARITY",
            "majorana_odd_one_based": "gamma_(2a-1)=Z_<(a) X_a",
            "majorana_even_one_based": "gamma_(2a)=Z_<(a) Y_a",
            "clockwise_adjacent_exchange": "C_j=(I-gamma_j gamma_(j+1))/sqrt(2)",
            "triangular_execution_word": "CONCAT_t=1_TO_2r-3 [2t+2,2t+1,...,t+3]",
            "accepted_loop_semantics": "BRAID_TRANSPORTED_MAJORANA_PAIR_PARITY_HOLONOMY",
            "literal_subset_or_framing_free_Wilson_loop_claimed": False,
        },
        "smoke_n4": smoke,
        "scaling_fixtures": scaling,
        "observed_disposition": {
            "classification": RESOURCE_KILL,
            "contiguous_cut_exact_ranks_n8_n12_n16": ranks,
            "deterministic_holonomies_restore_functional_exact_values": True,
            "mixed_holonomies_fail_after_retained_boundary_copy": True,
            "mixed_holonomies_restore_when_result_copy_is_unlatched": True,
            "commuting_same_sector_scramble_preserves_outputs_and_restores": True,
            "compact_signed_pairing_matches": True,
            "m257_controls_deterministic_software_comparison": True,
            "promote_as_machine_law_calibration_only": True,
            "kill_as_computational_resource": True,
        },
        "restoration": {
            "classification": "EXACT_ALGEBRAIC_RESTORATION",
            "scope": "FUNCTIONAL_EXACT_VALUE_RESTORATION_AND_REUSE_WITHOUT_SAME_BACKING",
            "inverse": "PUBLIC_LOOP_DERIVED_HERMITIAN_PAIR_HOLONOMY",
            "snapshot_reload": False,
            "same_backing_established": False,
            "physical_restoration_established": False,
        },
        "strongest_honest_comparator": {
            "implemented_formula_specific": "O(n)_SIGNED_MAJORANA_PAIRING_WITH_O(n_squared)_PUBLIC_PREPARATION_UPDATES",
            "general_ising_gaussian": "O(n_squared)_COVARIANCE_OR_STABILIZER_STATE_WITH_POLYNOMIAL_UPDATES",
            "declared_pair_holonomy_after_pairing_validation": "O(1)_LOOKUP",
            "dense_exact_occupation_vector": "REFERENCE_ONLY_EXPONENTIAL_REPRESENTATION",
            "equal_access_forward_only_shadow_omits_positive_cost_inverse": True,
            "comparator_optimality_established": False,
            "resource_advantage_comparison_authorized": False,
        },
        "resource_accounting": {
            "preparation_braid_counts_n4_n8_n12_n16": [
                smoke["public_smoke_forward_braid_count"],
                *(item["public_preparation_braid_count"] for item in scaling),
            ],
            "allocated_full_occupation_cells_n4_n8_n12_n16": [
                smoke["allocated_full_occupation_cells"],
                *(item["allocated_full_occupation_cells"] for item in scaling),
            ],
            "allocated_transaction_joint_coefficient_cells_n4_n8_n12_n16": [
                smoke["allocated_transaction_joint_coefficient_cells"],
                *(item["allocated_transaction_joint_coefficient_cells"] for item in scaling),
            ],
            "allocated_rank_matrix_gaussian_cells_n4_n8_n12_n16": [
                smoke["allocated_rank_matrix_gaussian_cells"],
                *(item["allocated_rank_matrix_gaussian_cells"] for item in scaling),
            ],
            "resident_support_cells_n8_n12_n16": [
                item["carrier_support_cells"] for item in scaling
            ],
            "retained_dynamic_inverse_history": 0,
            "functional_immutable_value_allocations_counted_completely": False,
            "python_objects_hashing_serialization_whole_process_peak_complete": False,
            "physical_preparation_energy_gap_noise_precision_bandwidth_latency_modeled": False,
            "verification_level": "PACKAGE_SELF_REVIEW",
        },
        "strict_claim_ceilings": [
            "software Phase-QEMU backend diagnostic only; no QEMU device execution",
            "no CATVM or authenticated custody enforcement",
            "no same-backing restoration or reuse",
            "no physical Ising anyons or coherent anyon interferometer",
            "no physical braid, detector, topological protection, or restoration",
            "no literal enclosed-subset Wilson-loop claim without transport and framing descriptor",
            "no distinct phase-native computational resource",
            "no computational advantage or complexity lower bound",
            "no M257 escape",
            "no Small Wall crossing",
            "no general holonomy or anyon-computation result",
            "no unbounded computation",
            "no replacement of physical bits with pi",
        ],
        "next_obstruction": (
            "A_FACTORING_EIGENPHASE_MUST_DEPEND_ON_A_GROWING_INTERACTING_"
            "RELATIONAL_INVARIANT_NOT_COMPACTLY_TRACKABLE_BY_GAUSSIAN_"
            "SIGNED_PAIRING_OR_STABILIZER_STATE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
