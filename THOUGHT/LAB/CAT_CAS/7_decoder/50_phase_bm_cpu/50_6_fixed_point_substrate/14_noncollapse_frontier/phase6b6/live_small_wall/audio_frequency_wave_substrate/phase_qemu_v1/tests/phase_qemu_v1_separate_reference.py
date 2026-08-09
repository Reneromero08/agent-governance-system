#!/usr/bin/env python3
"""Independent exact oracle for the Phase-QEMU V1 ideal bosonic fixture.

The oracle constructs the two-boson representation from homogeneous creation
polynomials. It does not import a production backend or use expected outputs
to compute the result. All accepted amplitudes are evaluated exactly in
Q(sqrt(2)); known fixture consequences are asserted only after evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class Qsqrt2:
    """a + b*sqrt(2), with exact rational a and b."""

    rational: Fraction = Fraction(0)
    sqrt2: Fraction = Fraction(0)

    @classmethod
    def coerce(cls, value: Qsqrt2 | Fraction | int) -> Qsqrt2:
        if isinstance(value, cls):
            return value
        return cls(Fraction(value), Fraction(0))

    def __add__(self, other: Qsqrt2 | Fraction | int) -> Qsqrt2:
        rhs = self.coerce(other)
        return Qsqrt2(self.rational + rhs.rational, self.sqrt2 + rhs.sqrt2)

    __radd__ = __add__

    def __neg__(self) -> Qsqrt2:
        return Qsqrt2(-self.rational, -self.sqrt2)

    def __sub__(self, other: Qsqrt2 | Fraction | int) -> Qsqrt2:
        return self + (-self.coerce(other))

    def __rsub__(self, other: Qsqrt2 | Fraction | int) -> Qsqrt2:
        return self.coerce(other) - self

    def __mul__(self, other: Qsqrt2 | Fraction | int) -> Qsqrt2:
        rhs = self.coerce(other)
        return Qsqrt2(
            self.rational * rhs.rational + 2 * self.sqrt2 * rhs.sqrt2,
            self.rational * rhs.sqrt2 + self.sqrt2 * rhs.rational,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Qsqrt2 | Fraction | int) -> Qsqrt2:
        rhs = self.coerce(other)
        denominator = rhs.rational * rhs.rational - 2 * rhs.sqrt2 * rhs.sqrt2
        if denominator == 0:
            raise ZeroDivisionError("division by zero in Q(sqrt(2))")
        return Qsqrt2(
            (self.rational * rhs.rational - 2 * self.sqrt2 * rhs.sqrt2)
            / denominator,
            (self.sqrt2 * rhs.rational - self.rational * rhs.sqrt2)
            / denominator,
        )

    def __bool__(self) -> bool:
        return self.rational != 0 or self.sqrt2 != 0

    def text(self) -> str:
        if self.sqrt2 == 0:
            return fraction_text(self.rational)
        sqrt_text = fraction_text(self.sqrt2)
        if self.rational == 0:
            return f"{sqrt_text}*sqrt2"
        sign = "+" if self.sqrt2 > 0 else "-"
        magnitude = fraction_text(abs(self.sqrt2))
        return f"{fraction_text(self.rational)}{sign}{magnitude}*sqrt2"


ZERO = Qsqrt2()
ONE = Qsqrt2(Fraction(1))
SQRT2 = Qsqrt2(Fraction(0), Fraction(1))
INV_SQRT2 = SQRT2 / 2

Occupation = tuple[int, ...]
Vector = list[Qsqrt2]
Matrix = list[list[Qsqrt2]]


def exact_sqrt_integer(value: int) -> Qsqrt2:
    root = math.isqrt(value)
    if root * root == value:
        return Qsqrt2(Fraction(root))
    if value == 2:
        return SQRT2
    raise ValueError(f"sqrt({value}) is outside the required exact field")


def homogeneous_basis(modes: int, particles: int) -> list[Occupation]:
    def generate(prefix: tuple[int, ...], remaining_modes: int, remaining: int):
        if remaining_modes == 1:
            yield prefix + (remaining,)
            return
        for count in range(remaining + 1):
            yield from generate(prefix + (count,), remaining_modes - 1, remaining - count)

    return sorted(generate((), modes, particles), reverse=True)


BASIS = homogeneous_basis(4, 2)
INDEX = {occupation: index for index, occupation in enumerate(BASIS)}


def matrix_identity(size: int) -> Matrix:
    return [
        [Qsqrt2(Fraction(row == column)) for column in range(size)]
        for row in range(size)
    ]


def matrix_transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            sum(
                (left[row][inner] * right[inner][column] for inner in range(len(right))),
                ZERO,
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    return [
        sum((coefficient * value for coefficient, value in zip(row, vector)), ZERO)
        for row in matrix
    ]


def single_particle_exchange(pair: tuple[int, int]) -> Matrix:
    """Public directed R_ij map on four creation operators."""

    first, second = pair
    matrix = matrix_identity(4)
    matrix[first][first] = INV_SQRT2
    matrix[first][second] = INV_SQRT2
    matrix[second][first] = -INV_SQRT2
    matrix[second][second] = INV_SQRT2
    return matrix


def single_particle_matching(pairs: Sequence[tuple[int, int]]) -> Matrix:
    result = matrix_identity(4)
    for pair in pairs:
        result = matrix_multiply(single_particle_exchange(pair), result)
    return result


def fock_representation(single_particle: Matrix) -> Matrix:
    """Generate the normalized homogeneous degree-two representation."""

    result = [[ZERO for _ in BASIS] for _ in BASIS]
    vacuum = (0, 0, 0, 0)
    for column, occupation in enumerate(BASIS):
        terms: dict[Occupation, Qsqrt2] = {vacuum: ONE}
        normalization = math.prod(math.factorial(value) for value in occupation)
        for source_mode, count in enumerate(occupation):
            for _ in range(count):
                next_terms: dict[Occupation, Qsqrt2] = {}
                for state, amplitude in terms.items():
                    for destination_mode in range(4):
                        coefficient = single_particle[destination_mode][source_mode]
                        if not coefficient:
                            continue
                        updated = list(state)
                        creation_factor = exact_sqrt_integer(updated[destination_mode] + 1)
                        updated[destination_mode] += 1
                        updated_state = tuple(updated)
                        next_terms[updated_state] = next_terms.get(updated_state, ZERO) + (
                            amplitude * coefficient * creation_factor
                        )
                terms = next_terms
        normalization_root = exact_sqrt_integer(normalization)
        for state, amplitude in terms.items():
            result[INDEX[state]][column] = amplitude / normalization_root
    return result


def cross_kerr(first: int, second: int) -> Matrix:
    result = [[ZERO for _ in BASIS] for _ in BASIS]
    for index, occupation in enumerate(BASIS):
        result[index][index] = Qsqrt2(Fraction((-1) ** (occupation[first] * occupation[second])))
    return result


def basis_vector(occupation: Occupation) -> Vector:
    result = [ZERO for _ in BASIS]
    result[INDEX[occupation]] = ONE
    return result


def execute(initial: Vector, word: Iterable[Matrix]) -> Vector:
    state = list(initial)
    for operation in word:
        state = matrix_vector(operation, state)
    return state


def inner(left: Vector, right: Vector) -> Qsqrt2:
    return sum((a * b for a, b in zip(left, right)), ZERO)


def norm_squared(vector: Vector) -> Fraction:
    value = inner(vector, vector)
    if value.sqrt2 != 0:
        raise AssertionError(f"norm did not reduce to Q: {value.text()}")
    return value.rational


def fidelity(left: Vector, right: Vector) -> Fraction:
    overlap = inner(left, right)
    squared = overlap * overlap
    if squared.sqrt2 != 0:
        raise AssertionError(f"fidelity did not reduce to Q: {squared.text()}")
    return squared.rational


def ket_text(occupation: Occupation) -> str:
    return "|" + "".join(str(value) for value in occupation) + ">"


def state_record(vector: Vector) -> dict[str, str]:
    return {
        ket_text(occupation): vector[index].text()
        for index, occupation in enumerate(BASIS)
        if vector[index]
    }


def state_commitment(vector: Vector) -> str:
    payload = json.dumps(
        [coefficient.text() for coefficient in vector],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parity_weights(vector: Vector, mode: int) -> tuple[Fraction, Fraction]:
    even = sum(
        (coefficient * coefficient for coefficient, occupation in zip(vector, BASIS) if occupation[mode] % 2 == 0),
        ZERO,
    )
    odd = sum(
        (coefficient * coefficient for coefficient, occupation in zip(vector, BASIS) if occupation[mode] % 2 == 1),
        ZERO,
    )
    if even.sqrt2 != 0 or odd.sqrt2 != 0:
        raise AssertionError("parity weights did not reduce to rational values")
    return even.rational, odd.rational


def pointer_record(vector: Vector, mode: int) -> dict[str, object]:
    even, odd = parity_weights(vector, mode)
    factorized = (even == 1 and odd == 0) or (even == 0 and odd == 1)
    pointer_z = 1 if even == 1 else (-1 if odd == 1 else None)
    return {
        "selector": f"PARITY_MODE_{mode}",
        "even_weight": fraction_text(even),
        "odd_weight": fraction_text(odd),
        "factorized": factorized,
        "pointer_z": pointer_z,
        "reduced_purity": fraction_text(even * even + odd * odd),
        "schmidt_rank": int(even != 0) + int(odd != 0),
    }


def pointer_latch(vector: Vector, mode: int) -> tuple[Vector, Vector]:
    branch_zero = [
        coefficient if occupation[mode] % 2 == 0 else ZERO
        for coefficient, occupation in zip(vector, BASIS)
    ]
    branch_one = [
        coefficient if occupation[mode] % 2 else ZERO
        for coefficient, occupation in zip(vector, BASIS)
    ]
    return branch_zero, branch_one


def pointer_factorization(branch_zero: Vector, branch_one: Vector) -> int | None:
    if not any(branch_one):
        return 0
    if not any(branch_zero):
        return 1
    return None


def pointer_unlatch(
    branch_zero: Vector, branch_one: Vector, mode: int
) -> tuple[Vector, Vector]:
    """Apply the parity-controlled pointer X a second time.

    This is the full two-branch linear action, rather than a shortcut that
    assumes the branches came from ``pointer_latch``.
    """

    restored_zero = [
        zero if occupation[mode] % 2 == 0 else one
        for zero, one, occupation in zip(branch_zero, branch_one, BASIS)
    ]
    restored_one = [
        one if occupation[mode] % 2 == 0 else zero
        for zero, one, occupation in zip(branch_zero, branch_one, BASIS)
    ]
    return restored_zero, restored_one


def zero_vector() -> Vector:
    return [ZERO for _ in BASIS]


def reduced_carrier_fidelity(
    initial: Vector, boundary_zero: Vector, boundary_one: Vector
) -> Fraction:
    """Fidelity with ``initial`` after tracing out a retained boundary bit."""

    overlap_zero = inner(initial, boundary_zero)
    overlap_one = inner(initial, boundary_one)
    value = overlap_zero * overlap_zero + overlap_one * overlap_one
    if value.sqrt2 != 0:
        raise AssertionError("reduced carrier fidelity did not reduce to Q")
    return value.rational


def pointer_transaction(
    final: Vector,
    mode: int,
    inverse: Sequence[Matrix],
    initial: Vector,
) -> dict[str, object]:
    """Audit computational-basis latch, optional copy, unlatch, and inverse.

    The carrier-controlled pointer X first separates even/odd carrier parity
    into pointer ``|0>``/``|1>`` branches.  A definite branch may be copied to
    a distinct boundary bit without entangling it.  The second pointer X then
    clears the interaction pointer before the public adjoint inverse.

    For a mixed-parity carrier, an unlatch *without* copying restores the
    carrier but retains no result.  Copying the pointer retains a boundary,
    but leaves that boundary entangled with the carrier; the public inverse
    consequently cannot restore a pure borrowed carrier.  Both paths are
    evaluated exactly below.
    """

    branch_zero, branch_one = pointer_latch(final, mode)
    boundary_bit = pointer_factorization(branch_zero, branch_one)

    # Path A: do not retain a boundary.  A second controlled X must return the
    # pointer to |0>, after which the public inverse restores the carrier.
    unlatched_zero, unlatched_one = pointer_unlatch(branch_zero, branch_one, mode)
    pointer_cleared = not any(unlatched_one) and unlatched_zero == final
    no_retention_restored_zero = execute(unlatched_zero, inverse)
    no_retention_restored_one = execute(unlatched_one, inverse)
    no_retention_roundtrip_restores = (
        no_retention_restored_zero == initial
        and no_retention_restored_one == zero_vector()
    )

    # Path B: copy the computational pointer to a retained boundary bit before
    # unlatching.  The interaction pointer clears, while the retained boundary
    # branches remain branch_zero/branch_one.  This restores the carrier iff
    # exactly one branch was populated.
    retained_zero = execute(branch_zero, inverse)
    retained_one = execute(branch_one, inverse)
    retained_copy_fidelity = reduced_carrier_fidelity(
        initial, retained_zero, retained_one
    )
    retained_copy_restores = (
        boundary_bit == 0
        and retained_zero == initial
        and retained_one == zero_vector()
    ) or (
        boundary_bit == 1
        and retained_one == initial
        and retained_zero == zero_vector()
    )
    return {
        "boundary_parity_bit": boundary_bit,
        "boundary_pointer_z": None if boundary_bit is None else 1 - 2 * boundary_bit,
        "factorized": boundary_bit is not None,
        "pointer_cleared_before_inverse": pointer_cleared,
        "no_retention_roundtrip_restores": no_retention_roundtrip_restores,
        "no_retention_emits_boundary": False,
        "boundary_retained_through_inverse": retained_copy_restores,
        "retained_copy_carrier_fidelity": fraction_text(retained_copy_fidelity),
        "mixed_parity_copy_prevents_exact_carrier_restoration": (
            boundary_bit is None and retained_copy_fidelity < 1
        ),
        "restored_state_commitment": state_commitment(no_retention_restored_zero),
        "carrier_pointer_gate_count": 2,
        "boundary_copy_gate_count": 1,
    }


def exact_rank(matrix: Matrix) -> int:
    work = [list(row) for row in matrix]
    row_count = len(work)
    column_count = len(work[0]) if work else 0
    rank = 0
    for column in range(column_count):
        pivot = next((row for row in range(rank, row_count) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(row_count):
            if row == rank or not work[row][column]:
                continue
            multiplier = work[row][column]
            work[row] = [
                value - multiplier * pivot_entry
                for value, pivot_entry in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def schmidt_rank(vector: Vector, left_modes: tuple[int, ...] = (0, 1)) -> int:
    right_modes = tuple(mode for mode in range(4) if mode not in left_modes)
    left_states = sorted({tuple(occupation[mode] for mode in left_modes) for occupation in BASIS}, reverse=True)
    right_states = sorted({tuple(occupation[mode] for mode in right_modes) for occupation in BASIS}, reverse=True)
    left_index = {state: index for index, state in enumerate(left_states)}
    right_index = {state: index for index, state in enumerate(right_states)}
    coefficient_matrix = [[ZERO for _ in right_states] for _ in left_states]
    for coefficient, occupation in zip(vector, BASIS):
        left = tuple(occupation[mode] for mode in left_modes)
        right = tuple(occupation[mode] for mode in right_modes)
        coefficient_matrix[left_index[left]][right_index[right]] = coefficient
    return exact_rank(coefficient_matrix)


def word_trace(initial: Vector, names: Sequence[str], operations: Sequence[Matrix]) -> list[dict[str, object]]:
    state = list(initial)
    trace = []
    for name, operation in zip(names, operations):
        state = matrix_vector(operation, state)
        trace.append(
            {
                "operation": name,
                "state": state_record(state),
                "schmidt_rank_01_23": schmidt_rank(state),
            }
        )
    return trace


def main_result() -> dict[str, object]:
    if BASIS != [
        (2, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 0, 0, 1),
        (0, 2, 0, 0),
        (0, 1, 1, 0),
        (0, 1, 0, 1),
        (0, 0, 2, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 2),
    ]:
        raise AssertionError("unexpected generated homogeneous basis")

    a_single = single_particle_matching(((0, 1), (2, 3)))
    b_single = single_particle_matching(((1, 2), (0, 3)))
    if matrix_multiply(a_single, b_single) == matrix_multiply(b_single, a_single):
        raise AssertionError("public exchange matchings unexpectedly commute")

    a = fock_representation(a_single)
    b = fock_representation(b_single)
    a_adjoint = matrix_transpose(a)
    b_adjoint = matrix_transpose(b)
    identity = matrix_identity(len(BASIS))
    if matrix_multiply(a_adjoint, a) != identity or matrix_multiply(b_adjoint, b) != identity:
        raise AssertionError("generated exchange representation is not exactly unitary")

    k01 = cross_kerr(0, 1)
    k03 = cross_kerr(0, 3)
    initial = basis_vector((1, 0, 1, 0))

    f1_names = ("A", "B", "K01", "A", "B")
    f1_word = (a, b, k01, a, b)
    f1_final = execute(initial, f1_word)
    f1_expected = basis_vector((1, 0, 0, 1))
    if f1_final != f1_expected:
        raise AssertionError(f"F1 mismatch: {state_record(f1_final)}")
    f1_inverse_names = ("B_DAG", "A_DAG", "K01", "B_DAG", "A_DAG")
    f1_inverse = (b_adjoint, a_adjoint, k01, b_adjoint, a_adjoint)
    f1_restored = execute(f1_final, f1_inverse)
    if f1_restored != initial:
        raise AssertionError("F1 public adjoint failed exact restoration")
    f1_sham = execute(initial, (a, b, a, b))

    f2_names = ("B", "A", "K03", "B", "A")
    f2_word = (b, a, k03, b, a)
    f2_final = execute(initial, f2_word)
    f2_expected = [-value for value in basis_vector((1, 1, 0, 0))]
    if f2_final != f2_expected:
        raise AssertionError(f"F2 mismatch: {state_record(f2_final)}")
    f2_inverse_names = ("A_DAG", "B_DAG", "K03", "A_DAG", "B_DAG")
    f2_inverse = (a_adjoint, b_adjoint, k03, a_adjoint, b_adjoint)
    f2_restored = execute(f2_final, f2_inverse)
    if f2_restored != initial:
        raise AssertionError("F2 public adjoint failed exact restoration")
    f2_sham = execute(initial, (b, a, b, a))

    held_out_names = ("A", "B", "K01", "A", "B_DAG")
    held_out_word = (a, b, k01, a, b_adjoint)
    held_out_final = execute(initial, held_out_word)
    held_out_inverse_names = ("B", "A_DAG", "K01", "B_DAG", "A_DAG")
    held_out_inverse = (b, a_adjoint, k01, b_adjoint, a_adjoint)
    held_out_restored = execute(held_out_final, held_out_inverse)
    if held_out_restored != initial:
        raise AssertionError("held-out public word failed exact restoration")
    held_out_sham = execute(initial, (a, b, a, b_adjoint))

    f1_pointer = pointer_record(f1_final, 3)
    f1_plus_pointer = pointer_record(f1_final, 2)
    f2_pointer = pointer_record(f2_final, 1)
    f1_sham_pointer = pointer_record(f1_sham, 3)
    f2_sham_pointer = pointer_record(f2_sham, 1)
    f1_pointer_transaction = pointer_transaction(f1_final, 3, f1_inverse, initial)
    f1_plus_pointer_transaction = pointer_transaction(f1_final, 2, f1_inverse, initial)
    f2_pointer_transaction = pointer_transaction(f2_final, 1, f2_inverse, initial)
    held_out_pointer_transaction = pointer_transaction(
        held_out_final, 0, held_out_inverse, initial
    )
    f1_sham_inverse = (b_adjoint, a_adjoint, b_adjoint, a_adjoint)
    f1_sham_pointer_transaction = pointer_transaction(
        f1_sham, 3, f1_sham_inverse, initial
    )
    if f1_pointer["pointer_z"] != -1 or f2_pointer["pointer_z"] != -1:
        raise AssertionError("accepted odd-parity selector did not latch -1")
    if f1_plus_pointer["pointer_z"] != 1 or not f1_plus_pointer["factorized"]:
        raise AssertionError("valid +1 selector did not factorize")
    if f1_sham_pointer["reduced_purity"] != "5/8":
        raise AssertionError("Kerr-disabled pointer purity mismatch")
    accepted_pointer_transactions = (
        f1_pointer_transaction,
        f1_plus_pointer_transaction,
        f2_pointer_transaction,
        held_out_pointer_transaction,
    )
    if not all(
        transaction["factorized"]
        and transaction["pointer_cleared_before_inverse"]
        and transaction["no_retention_roundtrip_restores"]
        and transaction["boundary_retained_through_inverse"]
        and transaction["retained_copy_carrier_fidelity"] == "1"
        for transaction in accepted_pointer_transactions
    ):
        raise AssertionError("factorized pointer transaction law failed")
    if not (
        not f1_sham_pointer_transaction["factorized"]
        and f1_sham_pointer_transaction["pointer_cleared_before_inverse"]
        and f1_sham_pointer_transaction["no_retention_roundtrip_restores"]
        and not f1_sham_pointer_transaction["boundary_retained_through_inverse"]
        and f1_sham_pointer_transaction[
            "mixed_parity_copy_prevents_exact_carrier_restoration"
        ]
        and f1_sham_pointer_transaction["retained_copy_carrier_fidelity"]
        == "5/8"
    ):
        raise AssertionError("independent pointer transaction law failed")

    controls = {
        "missing_entire_inverse": {
            "restoration_fidelity": fraction_text(fidelity(initial, f1_final)),
            "passes": fidelity(initial, f1_final) == 0,
        },
        "omitted_kerr_from_inverse": {},
        "wrong_kerr_edge_in_inverse": {},
        "reordered_inverse_prefix": {},
    }
    omitted_kerr_state = execute(f1_final, (b_adjoint, a_adjoint, b_adjoint, a_adjoint))
    wrong_kerr_state = execute(f1_final, (b_adjoint, a_adjoint, k03, b_adjoint, a_adjoint))
    reordered_state = execute(f1_final, (a_adjoint, b_adjoint, k01, b_adjoint, a_adjoint))
    controls["omitted_kerr_from_inverse"] = {
        "restoration_fidelity": fraction_text(fidelity(initial, omitted_kerr_state)),
        "state_commitment": state_commitment(omitted_kerr_state),
        "passes": fidelity(initial, omitted_kerr_state) == Fraction(1, 4),
    }
    controls["wrong_kerr_edge_in_inverse"] = {
        "restoration_fidelity": fraction_text(fidelity(initial, wrong_kerr_state)),
        "state_commitment": state_commitment(wrong_kerr_state),
        "passes": fidelity(initial, wrong_kerr_state) == Fraction(1, 4),
    }
    controls["reordered_inverse_prefix"] = {
        "restoration_fidelity": fraction_text(fidelity(initial, reordered_state)),
        "state_commitment": state_commitment(reordered_state),
        "passes": fidelity(initial, reordered_state) == 0,
    }
    controls.update(
        {
            "matchings_noncommute": True,
            "f1_exact_restore": f1_restored == initial,
            "f2_exact_restore": f2_restored == initial,
            "f1_kerr_changes_final_state": f1_sham != f1_final,
            "f2_kerr_changes_final_state": f2_sham != f2_final,
            "f1_boundary_factorizes": bool(f1_pointer["factorized"]),
            "f2_boundary_factorizes": bool(f2_pointer["factorized"]),
            "valid_plus_one_boundary_factorizes": bool(f1_plus_pointer["factorized"]),
            "f1_kerr_sham_entangles_pointer": not bool(f1_sham_pointer["factorized"]),
            "f2_kerr_sham_entangles_pointer": not bool(f2_sham_pointer["factorized"]),
        }
    )
    if not all(
        value["passes"] if isinstance(value, dict) and "passes" in value else bool(value)
        for value in controls.values()
    ):
        raise AssertionError(f"one or more independent controls failed: {controls}")

    family_dimensions = {}
    for modes in (4, 6, 8, 10, 12):
        particles = modes // 2
        family_dimensions[str(modes)] = {
            "particles": particles,
            "homogeneous_basis_dimension": math.comb(modes + particles - 1, particles),
            "matching_edges_per_layer": modes // 2,
            "minimum_declared_depth_for_sweep": modes // 2,
        }

    maximum_01_23_rank = sum(
        min(math.comb(2 + particles - 1, particles), math.comb(2 + (2 - particles) - 1, 2 - particles))
        for particles in range(3)
    )

    return {
        "schema": "phase-qemu-v1-separate-reference-v1",
        "field": "Q(sqrt(2))_SUBFIELD_OF_Q(zeta_8)",
        "basis": [ket_text(occupation) for occupation in BASIS],
        "public_maps": {
            "R_ij": {
                "creation_i": "(creation_i-creation_j)/sqrt(2)",
                "creation_j": "(creation_i+creation_j)/sqrt(2)",
            },
            "A": [[0, 1], [2, 3]],
            "B": [[1, 2], [0, 3]],
            "K_ij": "(-1)^(n_i*n_j)",
            "A_B_noncommuting": True,
        },
        "primary": {
            "public_initial": ket_text((1, 0, 1, 0)),
            "forward_execution_order": list(f1_names),
            "final_state_commitment": state_commitment(f1_final),
            "final_support_cells": sum(bool(value) for value in f1_final),
            "inverse_execution_order": list(f1_inverse_names),
            "restored_state_commitment": state_commitment(f1_restored),
            "boundary": f1_pointer,
            "pointer_transaction": f1_pointer_transaction,
            "valid_plus_one_boundary": f1_plus_pointer,
            "valid_plus_one_pointer_transaction": f1_plus_pointer_transaction,
            "kerr_disabled_state_commitment": state_commitment(f1_sham),
            "kerr_disabled_boundary": f1_sham_pointer,
            "kerr_disabled_pointer_transaction": f1_sham_pointer_transaction,
            "accepted_vs_kerr_disabled_fidelity": fraction_text(fidelity(f1_final, f1_sham)),
        },
        "descriptor_distinct_reuse": {
            "public_initial": ket_text((1, 0, 1, 0)),
            "forward_execution_order": list(f2_names),
            "final_state_commitment": state_commitment(f2_final),
            "final_support_cells": sum(bool(value) for value in f2_final),
            "inverse_execution_order": list(f2_inverse_names),
            "restored_state_commitment": state_commitment(f2_restored),
            "boundary": f2_pointer,
            "pointer_transaction": f2_pointer_transaction,
            "kerr_disabled_state_commitment": state_commitment(f2_sham),
            "kerr_disabled_boundary": f2_sham_pointer,
            "accepted_vs_kerr_disabled_fidelity": fraction_text(fidelity(f2_final, f2_sham)),
        },
        "held_out_public_descriptor": {
            "forward_execution_order": list(held_out_names),
            "inverse_execution_order": list(held_out_inverse_names),
            "final_state_commitment": state_commitment(held_out_final),
            "final_support_cells": sum(bool(value) for value in held_out_final),
            "kerr_disabled_state_commitment": state_commitment(held_out_sham),
            "kerr_changes_final_state": held_out_final != held_out_sham,
            "pointer_transaction": held_out_pointer_transaction,
        },
        "controls": controls,
        "qnd_factorization_law": {
            "joint_state": "psi_even*|0> + psi_odd*|1>",
            "retained_pointer_and_carrier_restoration_law": "LAWFUL_IFF_ONE_PARITY_BRANCH_HAS_ZERO_WEIGHT",
            "generic_superposed_boundary": "POINTER_ENTANGLES_AND_CARRIER_CANNOT_RESTORE_WHILE_RESULT_IS_RETAINED",
            "commitment_ceiling": "A_ONE_WAY_RECEIPT_IS_NOT_A_RETAINED_COMPUTATIONAL_BOUNDARY",
        },
        "resource_law": {
            "n4_modes": 4,
            "n4_particles": 2,
            "n4_exact_state_cells": len(BASIS),
            "n4_dense_operator_cells_if_retained": len(BASIS) ** 2,
            "n4_maximum_schmidt_rank_across_01_23_fixed_number_cut": maximum_01_23_rank,
            "family": "n=2m_modes_k=m_bosons",
            "family_dimension_formula": "binomial(3*n/2-1,n/2)",
            "bounded_sweep": family_dimensions,
            "bounded_degree_matching_edges_per_layer": "n/2",
            "constant_depth_warning": "ONE_DIMENSIONAL_CONSTANT_DEPTH_HAS_A_BOUNDED_BOND_TENSOR_NETWORK_SHADOW",
            "exact_reference_retains_dense_matrices": True,
            "whole_python_process_memory_accounted": False,
        },
        "strongest_classical_baselines": [
            "O1_ANALYTIC_CERTIFICATE_AFTER_VALIDATING_EITHER_OF_THE_TWO_PINNED_PUBLIC_FIXTURES",
            "IDENTICAL_EXACT_FIXED_NUMBER_STATE_RECURRENCE",
            "KERR_DISABLED_SINGLE_PARTICLE_LINEAR_OPTICS_PLUS_2X2_PERMANENT_BOUNDARY",
            "ADAPTIVE_MPS_OR_TTN_WITH_MEASURED_BOND_DIMENSION",
        ],
        "baseline_caveats": {
            "frozen_fixture_analytic_certificate_is_not_transferable": True,
            "fixed_number_fock_input_is_not_gaussian": True,
            "gaussian_covariance_is_not_a_sufficient_kerr_disabled_comparator_for_this_input": True,
            "n2_linear_optical_boundary_uses_at_most_2x2_permanents": True,
            "transferable_exact_sparse_state_and_boundary_only_tensor_contraction_both_remain_controlling": True,
        },
        "verification_classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "EXACT_IDEAL_FOUR_MODE_TWO_BOSON_UNITARY_AND_PARITY_POINTER_FIXTURE_ONLY",
        "claim_limits": {
            "physical_bosons_or_phonons": False,
            "physical_qnd_detector": False,
            "physical_restoration": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "m257_escape": False,
            "small_wall_crossing": False,
            "unbounded_compute": False,
            "replace_physical_bit_with_pi": False,
        },
        "kill_conditions": [
            "PRODUCTION_LATCH_READS_OR_PROJECTS_HIDDEN_AMPLITUDES",
            "PARITY_EIGENSTATE_CLOSURE_IS_FIXTURE_ONLY",
            "PRODUCTION_COMPILER_OBTAINS_BOUNDARY_PROMISE_BY_COMPUTING_THE_ANSWER",
            "GROWING_WORD_REDUCES_TO_A_COMPACT_MONOMIAL_OR_BOUNDED_BOND_RECURRENCE",
            "INVERSE_REQUIRES_HISTORY_SNAPSHOT_OR_STATE_DEPENDENT_CONTROLS",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(main_result(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
