#!/usr/bin/env python3
"""Independent exact oracle for the M261 Ising-anyon monodromy family.

The oracle constructs the even-parity fusion basis itself, applies adjacent
Ising exchanges through explicit Jordan-Wigner Majorana actions, and retains
dense Gaussian-integer amplitudes with a shared sqrt(2) denominator.  A
separate signed-Majorana covariance calculation supplies the compact classical
comparator.  Production code and earlier Phase-QEMU packages are not imported.

No floating-point value participates in a scientific decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


GaussianInt = tuple[int, int]
GaussianFraction = tuple[Fraction, Fraction]


def gi_add(left: GaussianInt, right: GaussianInt) -> GaussianInt:
    return left[0] + right[0], left[1] + right[1]


def gi_mul(left: GaussianInt, right: GaussianInt) -> GaussianInt:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def gi_scale(value: GaussianInt, scale: int) -> GaussianInt:
    return value[0] * scale, value[1] * scale


def gi_nonzero(value: GaussianInt) -> bool:
    return value != (0, 0)


def gf_add(left: GaussianFraction,
           right: GaussianFraction) -> GaussianFraction:
    return left[0] + right[0], left[1] + right[1]


def gf_neg(value: GaussianFraction) -> GaussianFraction:
    return -value[0], -value[1]


def gf_sub(left: GaussianFraction,
           right: GaussianFraction) -> GaussianFraction:
    return gf_add(left, gf_neg(right))


def gf_mul(left: GaussianFraction,
           right: GaussianFraction) -> GaussianFraction:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def gf_div(left: GaussianFraction,
           right: GaussianFraction) -> GaussianFraction:
    denominator = right[0] * right[0] + right[1] * right[1]
    if denominator == 0:
        raise ZeroDivisionError("division by zero in Q(i)")
    return (
        (left[0] * right[0] + left[1] * right[1]) / denominator,
        (left[1] * right[0] - left[0] * right[1]) / denominator,
    )


def gf_nonzero(value: GaussianFraction) -> bool:
    return value != (Fraction(0), Fraction(0))


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def gaussian_text(value: GaussianInt) -> str:
    real, imaginary = value
    if imaginary == 0:
        return str(real)
    if real == 0:
        return f"{imaginary}i"
    sign = "+" if imaginary > 0 else "-"
    return f"{real}{sign}{abs(imaginary)}i"


@dataclass(frozen=True)
class FusionSpace:
    sigma_anyons: int
    pair_modes: int
    basis: tuple[int, ...]
    index: dict[int, int]

    @classmethod
    def create(cls, sigma_anyons: int) -> "FusionSpace":
        if sigma_anyons < 4 or sigma_anyons % 4:
            raise ValueError("this M261 family requires n divisible by four")
        pair_modes = sigma_anyons // 2
        basis = tuple(bits for bits in range(1 << pair_modes)
                      if bits.bit_count() % 2 == 0)
        return cls(
            sigma_anyons=sigma_anyons,
            pair_modes=pair_modes,
            basis=basis,
            index={bits: offset for offset, bits in enumerate(basis)},
        )


@dataclass(frozen=True)
class ExactState:
    coefficients: tuple[GaussianInt, ...]
    sqrt2_denominator_exponent: int


def canonical_state(coefficients: Sequence[GaussianInt],
                    exponent: int) -> ExactState:
    values = list(coefficients)
    while exponent >= 2 and values and all(
        real % 2 == 0 and imaginary % 2 == 0
        for real, imaginary in values
    ):
        values = [(real // 2, imaginary // 2)
                  for real, imaginary in values]
        exponent -= 2
    return ExactState(tuple(values), exponent)


def vacuum(space: FusionSpace) -> ExactState:
    coefficients = [(0, 0)] * len(space.basis)
    coefficients[space.index[0]] = (1, 0)
    return ExactState(tuple(coefficients), 0)


def gamma_action(bits: int, majorana: int) -> tuple[int, GaussianInt]:
    """Apply gamma_(2j)=Z^j X_j or gamma_(2j+1)=Z^j Y_j."""

    mode = majorana // 2
    prefix_parity = (bits & ((1 << mode) - 1)).bit_count() & 1
    occupied = (bits >> mode) & 1
    updated = bits ^ (1 << mode)
    if majorana % 2 == 0:
        phase = (-1 if prefix_parity else 1, 0)
    else:
        sign = -1 if prefix_parity ^ occupied else 1
        phase = (0, sign)
    return updated, phase


def adjacent_majorana_product(bits: int,
                              braid_index: int) -> tuple[int, GaussianInt]:
    """Apply gamma_k gamma_(k+1), rightmost Majorana first."""

    middle, right_phase = gamma_action(bits, braid_index + 1)
    output, left_phase = gamma_action(middle, braid_index)
    return output, gi_mul(left_phase, right_phase)


def apply_braid(space: FusionSpace, state: ExactState, braid_index: int,
                inverse: bool = False) -> ExactState:
    if not (0 <= braid_index < space.sigma_anyons - 1):
        raise ValueError("adjacent braid index outside the Majorana chain")
    output = [(0, 0)] * len(space.basis)
    product_sign = -1 if inverse else 1
    for source_index, amplitude in enumerate(state.coefficients):
        if not gi_nonzero(amplitude):
            continue
        bits = space.basis[source_index]
        destination_bits, phase = adjacent_majorana_product(bits, braid_index)
        if destination_bits not in space.index:
            raise AssertionError("adjacent braid escaped the even fusion sector")
        output[source_index] = gi_add(output[source_index], amplitude)
        destination = space.index[destination_bits]
        transformed = gi_scale(gi_mul(phase, amplitude), product_sign)
        output[destination] = gi_add(output[destination], transformed)
    return canonical_state(output, state.sqrt2_denominator_exponent + 1)


def preparation_word(sigma_anyons: int) -> tuple[int, ...]:
    """Generate the public one-based triangular adjacent-exchange word J_r."""

    pair_modes = sigma_anyons // 2
    return tuple(
        braid
        for step in range(1, pair_modes - 2)
        for braid in range(2 * step + 2, step + 2, -1)
    )


def apply_c_word(space: FusionSpace, state: ExactState,
                 one_based_word: Sequence[int],
                 inverse: bool = False) -> ExactState:
    """Apply C_j=(I-gamma_j gamma_(j+1))/sqrt(2), or its adjoint."""

    operations = reversed(one_based_word) if inverse else one_based_word
    result = state
    for one_based_braid in operations:
        result = apply_braid(
            space,
            result,
            one_based_braid - 1,
            inverse=not inverse,
        )
    return result


def norm_squared(state: ExactState) -> Fraction:
    numerator = sum(real * real + imaginary * imaginary
                    for real, imaginary in state.coefficients)
    return Fraction(numerator, 1 << state.sqrt2_denominator_exponent)


def support_size(state: ExactState) -> int:
    return sum(gi_nonzero(value) for value in state.coefficients)


def state_commitment(space: FusionSpace, state: ExactState) -> str:
    payload = json.dumps(
        {
            "basis": space.basis,
            "coefficients": state.coefficients,
            "sqrt2_denominator_exponent": state.sqrt2_denominator_exponent,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def amplitude_record(space: FusionSpace,
                     state: ExactState) -> dict[str, object]:
    return {
        format(bits, f"0{space.pair_modes}b")[::-1]: {
            "gaussian_integer_numerator": gaussian_text(coefficient),
            "sqrt2_denominator_exponent": state.sqrt2_denominator_exponent,
        }
        for bits, coefficient in zip(space.basis, state.coefficients)
        if gi_nonzero(coefficient)
    }


def projective_phase(left: ExactState,
                     right: ExactState) -> GaussianInt | None:
    """Return a Gaussian-unit phase when two normalized states share a ray."""

    if left.sqrt2_denominator_exponent != right.sqrt2_denominator_exponent:
        return None
    units = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for unit in units:
        if all(left_value == gi_mul(unit, right_value)
               for left_value, right_value in zip(left.coefficients,
                                                  right.coefficients)):
            return unit
    return None


def apply_majorana_bilinear(space: FusionSpace, state: ExactState,
                            first_one_based: int,
                            second_one_based: int) -> ExactState:
    """Apply the Hermitian loop -i gamma_a gamma_b exactly."""

    first = first_one_based - 1
    second = second_one_based - 1
    if not (0 <= first < second < space.sigma_anyons):
        raise ValueError("loop Majorana endpoints must obey 1 <= a < b <= n")
    output = [(0, 0)] * len(space.basis)
    minus_i = (0, -1)
    for source_index, amplitude in enumerate(state.coefficients):
        if not gi_nonzero(amplitude):
            continue
        bits = space.basis[source_index]
        middle, right_phase = gamma_action(bits, second)
        destination_bits, left_phase = gamma_action(middle, first)
        destination = space.index[destination_bits]
        phase = gi_mul(minus_i, gi_mul(left_phase, right_phase))
        output[destination] = gi_add(
            output[destination], gi_mul(phase, amplitude)
        )
    return ExactState(tuple(output), state.sqrt2_denominator_exponent)


def apply_nonlocal_exchange(space: FusionSpace, state: ExactState,
                            first_one_based: int,
                            second_one_based: int,
                            inverse: bool = False) -> ExactState:
    """Apply (I-/+ gamma_a gamma_b)/sqrt(2) without expanding a matrix."""

    first = first_one_based - 1
    second = second_one_based - 1
    if not (0 <= first < second < space.sigma_anyons):
        raise ValueError("exchange endpoints must obey 1 <= a < b <= n")
    output = [(0, 0)] * len(space.basis)
    product_sign = 1 if inverse else -1
    for source_index, amplitude in enumerate(state.coefficients):
        if not gi_nonzero(amplitude):
            continue
        bits = space.basis[source_index]
        middle, right_phase = gamma_action(bits, second)
        destination_bits, left_phase = gamma_action(middle, first)
        destination = space.index[destination_bits]
        output[source_index] = gi_add(output[source_index], amplitude)
        transformed = gi_scale(
            gi_mul(gi_mul(left_phase, right_phase), amplitude),
            product_sign,
        )
        output[destination] = gi_add(output[destination], transformed)
    return canonical_state(output, state.sqrt2_denominator_exponent + 1)


def inner_product(left: ExactState,
                  right: ExactState) -> GaussianFraction:
    denominator_exponent = (
        left.sqrt2_denominator_exponent
        + right.sqrt2_denominator_exponent
    )
    real = 0
    imaginary = 0
    for (left_real, left_imaginary), (right_real, right_imaginary) in zip(
        left.coefficients, right.coefficients
    ):
        real += left_real * right_real + left_imaginary * right_imaginary
        imaginary += left_real * right_imaginary - left_imaginary * right_real
    denominator = 1 << (denominator_exponent // 2)
    if denominator_exponent % 2:
        raise AssertionError("oracle inner product unexpectedly retained sqrt(2)")
    return Fraction(real, denominator), Fraction(imaginary, denominator)


def overlap_abs_squared(left: ExactState, right: ExactState) -> Fraction:
    real = 0
    imaginary = 0
    for (left_real, left_imaginary), (right_real, right_imaginary) in zip(
        left.coefficients, right.coefficients
    ):
        real += left_real * right_real + left_imaginary * right_imaginary
        imaginary += left_real * right_imaginary - left_imaginary * right_real
    return Fraction(
        real * real + imaginary * imaginary,
        1 << (
            left.sqrt2_denominator_exponent
            + right.sqrt2_denominator_exponent
        ),
    )


def bilinear_probe_record(space: FusionSpace, state: ExactState,
                          endpoints: tuple[int, int]) -> dict[str, object]:
    looped = apply_majorana_bilinear(space, state, *endpoints)
    expectation = inner_product(state, looped)
    if expectation[1] != 0:
        raise AssertionError("Hermitian bilinear produced imaginary expectation")
    phase = projective_phase(looped, state)
    factorized = phase in ((1, 0), (-1, 0))
    purity = (Fraction(1) + expectation[0] * expectation[0]) / 2
    restored = apply_majorana_bilinear(space, looped, *endpoints)
    joint_commitment = hashlib.sha256(json.dumps(
        {
            "probe_branch_0_target": state_commitment(space, state),
            "probe_branch_1_target": state_commitment(space, looped),
            "shared_denominator": "sqrt(2)",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "majorana_endpoints_one_based": list(endpoints),
        "operator": "-i gamma_a gamma_b",
        "charge_expectation": fraction_text(expectation[0]),
        "definite_topological_charge": factorized,
        "topological_charge": (
            "1" if expectation[0] == 1 else "psi"
            if expectation[0] == -1 else "SUPERPOSITION"
        ),
        "controlled_sigma_probe_phase": (
            "+1" if phase == (1, 0) else "-1"
            if phase == (-1, 0) else None
        ),
        "probe_x_boundary": fraction_text(expectation[0]),
        "probe_reduced_purity": fraction_text(purity),
        "probe_target_schmidt_rank": 1 if factorized else 2,
        "derived_joint_probe_target_state": (
            "(|0> tensor |psi> + |1> tensor L|psi>)/sqrt(2)"
        ),
        "joint_probe_target_commitment": joint_commitment,
        "joint_norm": "1",
        "factorized_boundary_retention_lawful": factorized,
        "target_commitment_before": state_commitment(space, state),
        "target_commitment_after_loop_branch": state_commitment(space, looped),
        "actual_second_monodromy_restores_joint": restored == state,
    }


def gaussian_rank(matrix: Sequence[Sequence[GaussianInt]]) -> int:
    work: list[list[GaussianFraction]] = [
        [(Fraction(real), Fraction(imaginary)) for real, imaginary in row]
        for row in matrix
    ]
    if not work:
        return 0
    rows = len(work)
    columns = len(work[0])
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows)
                      if gf_nonzero(work[row][column])), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for row in range(rank + 1, rows):
            if not gf_nonzero(work[row][column]):
                continue
            factor = gf_div(work[row][column], pivot_value)
            for offset in range(column, columns):
                work[row][offset] = gf_sub(
                    work[row][offset],
                    gf_mul(factor, work[rank][offset]),
                )
        rank += 1
        if rank == rows:
            break
    return rank


def cut_matrix(space: FusionSpace, state: ExactState,
               left_modes: Sequence[int]) -> list[list[GaussianInt]]:
    left_modes = tuple(left_modes)
    right_modes = tuple(
        mode for mode in range(space.pair_modes) if mode not in left_modes
    )
    matrix = [
        [(0, 0)] * (1 << len(right_modes))
        for _ in range(1 << len(left_modes))
    ]

    def project(bits: int, modes: Sequence[int]) -> int:
        return sum(((bits >> mode) & 1) << offset
                   for offset, mode in enumerate(modes))

    for bits, coefficient in zip(space.basis, state.coefficients):
        matrix[project(bits, left_modes)][project(bits, right_modes)] = coefficient
    return matrix


def cut_rank(space: FusionSpace, state: ExactState,
             left_modes: Sequence[int]) -> int:
    return gaussian_rank(cut_matrix(space, state, left_modes))


def integer_rank(matrix: Sequence[Sequence[int]]) -> int:
    work = [[Fraction(value) for value in row] for row in matrix]
    if not work:
        return 0
    rows = len(work)
    columns = len(work[0])
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows)
                      if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for row in range(rank + 1, rows):
            if not work[row][column]:
                continue
            factor = work[row][column] / pivot_value
            for offset in range(column, columns):
                work[row][offset] -= factor * work[rank][offset]
        rank += 1
    return rank


def initial_covariance(majoranas: int) -> list[list[int]]:
    covariance = [[0] * majoranas for _ in range(majoranas)]
    for first in range(0, majoranas, 2):
        covariance[first][first + 1] = 1
        covariance[first + 1][first] = -1
    return covariance


def covariance_braid(covariance: Sequence[Sequence[int]],
                     braid_index: int, inverse: bool = False) -> list[list[int]]:
    """Update the signed Majorana frame for U=(I+/-gamma_k gamma_k+1)/sqrt2."""

    size = len(covariance)
    old_index = list(range(size))
    sign = [1] * size
    if inverse:
        old_index[braid_index] = braid_index + 1
        sign[braid_index] = -1
        old_index[braid_index + 1] = braid_index
    else:
        old_index[braid_index] = braid_index + 1
        old_index[braid_index + 1] = braid_index
        sign[braid_index + 1] = -1
    return [
        [
            sign[row] * sign[column]
            * covariance[old_index[row]][old_index[column]]
            for column in range(size)
        ]
        for row in range(size)
    ]


def covariance_word(majoranas: int,
                    one_based_word: Sequence[int]) -> list[list[int]]:
    covariance = initial_covariance(majoranas)
    for one_based_braid in one_based_word:
        covariance = covariance_braid(
            covariance, one_based_braid - 1, inverse=True
        )
    return covariance


def covariance_cut_rank(covariance: Sequence[Sequence[int]],
                        pair_modes: int,
                        left_modes: Sequence[int]) -> int:
    left_mode_set = frozenset(left_modes)
    left = tuple(index for mode in range(pair_modes)
                 if mode in left_mode_set
                 for index in (2 * mode, 2 * mode + 1))
    right = tuple(index for mode in range(pair_modes)
                  if mode not in left_mode_set
                  for index in (2 * mode, 2 * mode + 1))
    cross = [[covariance[row][column] for column in right] for row in left]
    cross_rank = integer_rank(cross)
    if cross_rank % 2:
        raise AssertionError("pure Gaussian cross-covariance rank is odd")
    return 1 << (cross_rank // 2)


def signed_pairing(covariance: Sequence[Sequence[int]]) -> list[dict[str, int]]:
    """Compress a signed-permutation covariance into its perfect matching."""

    pairs: list[dict[str, int]] = []
    for first, row in enumerate(covariance):
        second = next((index for index, value in enumerate(row) if value), None)
        if second is None:
            raise AssertionError("Majorana covariance lost its unique partner")
        if first < second:
            pairs.append({"first": first + 1, "second": second + 1,
                          "sign": row[second]})
    if len(pairs) * 2 != len(covariance):
        raise AssertionError("Majorana covariance is not a perfect matching")
    return pairs


def n4_orientation_smoke() -> dict[str, object]:
    """Separate two-mode algebra smoke; this is not a triangular-family case."""

    space = FusionSpace.create(4)
    initial = vacuum(space)
    loop = (1, 2)
    initial_probe = bilinear_probe_record(space, initial, loop)
    prepared = apply_c_word(space, initial, (2, 2))

    occupied_coefficients = [(0, 0)] * len(space.basis)
    occupied_coefficients[space.index[0b11]] = (1, 0)
    occupied = ExactState(tuple(occupied_coefficients), 0)
    preparation_phase = projective_phase(prepared, occupied)
    if preparation_phase is None:
        raise AssertionError("C2 twice did not prepare the |11> ray")
    prepared_probe = bilinear_probe_record(space, prepared, loop)
    restored = apply_c_word(space, prepared, (2, 2), inverse=True)

    if not (
        initial_probe["controlled_sigma_probe_phase"] == "+1"
        and prepared_probe["controlled_sigma_probe_phase"] == "-1"
        and restored == initial
    ):
        raise AssertionError("n=4 algebra/orientation smoke failed")

    def copy_unlatch(probe: dict[str, object]) -> dict[str, object]:
        phase = probe["controlled_sigma_probe_phase"]
        return {
            "controlled_probe_phase": phase,
            "definite_charge_before_copy": probe[
                "definite_topological_charge"
            ],
            "coherent_boundary_copy_retains_only_definite_phase": phase,
            "copy_does_not_entangle_target_for_definite_charge": True,
            "second_monodromy_unlatches_target_exactly": probe[
                "actual_second_monodromy_restores_joint"
            ],
            "factorized_boundary_retention_lawful": probe[
                "factorized_boundary_retention_lawful"
            ],
        }

    return {
        "scope": "SEPARATE_N4_ALGEBRA_AND_ORIENTATION_SMOKE_ONLY",
        "included_in_triangular_scaling_family": False,
        "sigma_anyons": 4,
        "pair_charge_modes": 2,
        "even_sector_fusion_dimension": len(space.basis),
        "loop": {
            "majorana_endpoints_one_based": list(loop),
            "operator": "L=-i gamma_1 gamma_2",
        },
        "vacuum": {
            "state": "|00>",
            "loop_phase": initial_probe["controlled_sigma_probe_phase"],
            "probe_copy_unlatch": copy_unlatch(initial_probe),
        },
        "preparation": {
            "public_C_word_one_based": [2, 2],
            "prepared_ray": "|11>",
            "derived_gaussian_unit_phase_relative_to_|11>": gaussian_text(
                preparation_phase
            ),
            "dense_amplitudes": amplitude_record(space, prepared),
        },
        "prepared": {
            "loop_phase": prepared_probe["controlled_sigma_probe_phase"],
            "probe_copy_unlatch": copy_unlatch(prepared_probe),
        },
        "public_adjoint_word": [
            {"C_index_one_based": 2, "orientation": "ADJOINT"},
            {"C_index_one_based": 2, "orientation": "ADJOINT"},
        ],
        "public_adjoint_twice_restores_vacuum_exactly": restored == initial,
        "initial_state_commitment": state_commitment(space, initial),
        "restored_state_commitment": state_commitment(space, restored),
    }


def case_record(sigma_anyons: int) -> dict[str, object]:
    space = FusionSpace.create(sigma_anyons)
    word = preparation_word(sigma_anyons)
    initial = vacuum(space)
    prepared = apply_c_word(space, initial, word)
    if norm_squared(prepared) != 1:
        raise AssertionError("public preparation did not preserve exact norm")
    restored = apply_c_word(space, prepared, word, inverse=True)
    if restored != initial:
        raise AssertionError("reverse public adjacent braids failed exact restoration")

    covariance = covariance_word(space.sigma_anyons, word)
    left_modes = tuple(range(space.pair_modes // 2))
    dense_rank = cut_rank(space, prepared, left_modes)
    compact_rank = covariance_cut_rank(
        covariance, space.pair_modes, left_modes
    )
    if dense_rank != compact_rank:
        raise AssertionError("dense and compact Majorana cut ranks disagree")

    loop_l0 = (3, space.pair_modes + 1)
    loop_l1 = (4, space.pair_modes + 2)
    loops = {
        "L0_charge_1": bilinear_probe_record(space, prepared, loop_l0),
        "L1_charge_psi": bilinear_probe_record(space, prepared, loop_l1),
    }
    for name, endpoints in (("L0_charge_1", loop_l0),
                            ("L1_charge_psi", loop_l1)):
        compact_expectation = covariance[endpoints[0] - 1][endpoints[1] - 1]
        dense_expectation = int(loops[name]["charge_expectation"])
        if compact_expectation != dense_expectation:
            raise AssertionError(f"compact charge comparator disagrees for {name}")
        loops[name]["compact_majorana_charge_expectation"] = compact_expectation

    if not (
        loops["L0_charge_1"]["controlled_sigma_probe_phase"] == "+1"
        and loops["L1_charge_psi"]["controlled_sigma_probe_phase"] == "-1"
    ):
        raise AssertionError("formula-generated definite loop phases changed")

    invalid_endpoints = (3, 4)
    superposition = bilinear_probe_record(space, prepared, invalid_endpoints)
    if (superposition["definite_topological_charge"]
            or superposition["charge_expectation"] != "0"
            or superposition["probe_target_schmidt_rank"] != 2):
        raise AssertionError("invalid bilinear loop did not entangle the probe")
    mixed_expectation = Fraction(superposition["charge_expectation"])
    charge_1_probability = (1 + mixed_expectation) / 2
    charge_psi_probability = (1 - mixed_expectation) / 2
    max_factorized_fidelity = max(charge_1_probability,
                                  charge_psi_probability)
    superposition.update({
        "public_control_name": "M=-i gamma_3 gamma_4",
        "charge_1_probability": fraction_text(charge_1_probability),
        "charge_psi_probability": fraction_text(charge_psi_probability),
        "no_copy_latch_unlatch_exact_restore": superposition[
            "actual_second_monodromy_restores_joint"
        ],
        "retained_coherent_boundary_copy_restored_target_marginal_fidelity": (
            fraction_text(max_factorized_fidelity)
        ),
        "retained_copy_max_factorized_fidelity": fraction_text(
            max_factorized_fidelity
        ),
        "copy_failure_derivation": (
            "EXACT_CHARGE_BRANCH_PROBABILITIES_FROM_(1_PLUS_OR_MINUS_"
            "EXPECTATION_M)_OVER_2"
        ),
        "retained_copy_disposition": (
            "BOUNDARY_COPY_PREVENTS_LAWFUL_FACTORIZED_RETENTION"
        ),
    })

    scramble_endpoints = (2, sigma_anyons - 1)
    scramble_generator = bilinear_probe_record(
        space, prepared, scramble_endpoints
    )
    if scramble_generator["charge_expectation"] != "0":
        raise AssertionError("same-charge scramble generator expectation changed")
    scrambled = apply_nonlocal_exchange(
        space, prepared, *scramble_endpoints
    )
    overlap_squared = overlap_abs_squared(prepared, scrambled)
    if overlap_squared != Fraction(1, 2):
        raise AssertionError("same-charge scramble overlap law changed")
    scramble_l0 = bilinear_probe_record(space, scrambled, loop_l0)
    scramble_l1 = bilinear_probe_record(space, scrambled, loop_l1)
    if not (
        scramble_l0["controlled_sigma_probe_phase"] == "+1"
        and scramble_l1["controlled_sigma_probe_phase"] == "-1"
    ):
        raise AssertionError("disjoint scramble failed to preserve loop charge")
    unscrambled = apply_nonlocal_exchange(
        space, scrambled, *scramble_endpoints, inverse=True
    )
    transport_word = tuple(range(2, sigma_anyons - 2))
    synthesized = apply_c_word(space, prepared, transport_word)
    synthesized = apply_c_word(space, synthesized, (sigma_anyons - 2,))
    synthesized = apply_c_word(space, synthesized, transport_word, inverse=True)
    synthesis_phase = projective_phase(synthesized, scrambled)
    if synthesis_phase is None:
        raise AssertionError("adjacent synthesis disagrees with direct scramble")
    scrambling = {
        "generator": (
            f"K=-i gamma_{scramble_endpoints[0]} gamma_{scramble_endpoints[1]}"
        ),
        "unitary": (
            f"U_K=(I-gamma_{scramble_endpoints[0]}gamma_"
            f"{scramble_endpoints[1]})/sqrt(2)"
        ),
        "commutation_law": "K_IS_DISJOINT_FROM_AND_COMMUTES_WITH_L0_AND_L1",
        "prepared_expectation_of_K": scramble_generator["charge_expectation"],
        "prepared_scrambled_overlap_squared": fraction_text(overlap_squared),
        "projective_state_changed": projective_phase(scrambled, prepared) is None,
        "L0_phase_before_after": [
            loops["L0_charge_1"]["controlled_sigma_probe_phase"],
            scramble_l0["controlled_sigma_probe_phase"],
        ],
        "L1_phase_before_after": [
            loops["L1_charge_psi"]["controlled_sigma_probe_phase"],
            scramble_l1["controlled_sigma_probe_phase"],
        ],
        "both_fixed_charges_invariant": True,
        "adjacent_synthesis": {
            "P_word_one_based": list(transport_word),
            "formula": "U_K=P_DAG C_(4r-2) P",
            "middle_C_index_one_based": sigma_anyons - 2,
            "matches_direct_unitary_on_prepared_ray": True,
            "relative_gaussian_unit": gaussian_text(synthesis_phase),
        },
        "public_adjoint_restores_prepared_state_exactly": unscrambled == prepared,
    }
    if not scrambling["public_adjoint_restores_prepared_state_exactly"]:
        raise AssertionError("public fixed-charge scramble failed exact inverse")

    after_l0 = apply_majorana_bilinear(
        space,
        apply_majorana_bilinear(space, prepared, *loop_l0),
        *loop_l0,
    )
    after_l1 = apply_majorana_bilinear(
        space,
        apply_majorana_bilinear(space, after_l0, *loop_l1),
        *loop_l1,
    )
    distinct_loop_reuse = {
        "first_loop_majoranas": list(loop_l0),
        "first_phase": loops["L0_charge_1"]["controlled_sigma_probe_phase"],
        "second_loop_majoranas": list(loop_l1),
        "second_phase": loops["L1_charge_psi"]["controlled_sigma_probe_phase"],
        "same_prepared_value_consumed_by_both_queries": True,
        "prepared_target_commitment": state_commitment(space, prepared),
        "first_loop_squared_restores_target": after_l0 == prepared,
        "second_distinct_loop_squared_restores_target": after_l1 == prepared,
        "same_backing_or_machine_custody_established": False,
    }

    pairing = signed_pairing(covariance)
    crossing_pair_count = sum(
        (pair["first"] <= space.pair_modes)
        != (pair["second"] <= space.pair_modes)
        for pair in pairing
    )

    return {
        "sigma_anyons": sigma_anyons,
        "pair_charge_modes": space.pair_modes,
        "global_charge_sector": "EVEN_PAIR_CHARGE_PARITY_TOTAL_VACUUM",
        "fusion_dimension": len(space.basis),
        "public_preparation_C_word_one_based": list(word),
        "public_adjoint_word": [
            {"C_index_one_based": braid, "orientation": "ADJOINT"}
            for braid in reversed(word)
        ],
        "prepared_support_cells": support_size(prepared),
        "prepared_dense_amplitudes": amplitude_record(space, prepared),
        "prepared_state_commitment": state_commitment(space, prepared),
        "ordinary_contiguous_central_cut": {
            "left_pair_modes_zero_based": list(left_modes),
            "right_pair_modes_zero_based": list(
                range(space.pair_modes // 2, space.pair_modes)
            ),
            "dense_exact_schmidt_rank": dense_rank,
            "compact_majorana_covariance_rank": compact_rank,
            "signed_pairing_crossing_edges": crossing_pair_count,
        },
        "definite_loop_queries": loops,
        "distinct_loop_reuse": distinct_loop_reuse,
        "invalid_superposed_loop_control": superposition,
        "internal_fixed_charge_scrambling": scrambling,
        "public_inverse_exact_restoration": restored == initial,
        "initial_state_commitment": state_commitment(space, initial),
        "restored_state_commitment": state_commitment(space, restored),
        "compact_comparator": {
            "signed_pairing": pairing,
            "signed_pairing_integer_cells": 3 * len(pairing),
            "dense_covariance_cells_used_only_as_independent_oracle": (
                sigma_anyons * sigma_anyons
            ),
            "signed_majorana_frame_cells": sigma_anyons,
            "adjacent_braid_update": "SIGNED_MAJORANA_SWAP",
            "bilinear_loop_query": "SIGNED_PAIRING_LOOKUP",
            "dense_rank_reproduced": compact_rank == dense_rank,
            "all_declared_loop_expectations_reproduced": True,
        },
    }


def result() -> dict[str, object]:
    family = (8, 12, 16)
    cases = {str(n): case_record(n) for n in family}
    dimensions = [cases[str(n)]["fusion_dimension"] for n in family]
    supports = [cases[str(n)]["prepared_support_cells"] for n in family]
    family_ranks = [cases[str(n)]["ordinary_contiguous_central_cut"][
                        "dense_exact_schmidt_rank"
                    ]
                    for n in (8, 12, 16)]
    if any(
        cases[str(n)]["fusion_dimension"] != 1 << (n // 2 - 1)
        for n in family
    ):
        raise AssertionError("even-sector fusion dimension law failed")
    if family_ranks != sorted(family_ranks) or len(set(family_ranks)) != 3:
        raise AssertionError("two-rail rank family did not grow strictly")
    if not all(cases[str(n)]["public_inverse_exact_restoration"]
               for n in family):
        raise AssertionError("one public inverse failed")
    return {
        "schema": "phase-qemu-v3-ising-anyon-controlled-monodromy-separate-reference-v1",
        "implementation_independence": {
            "imports_production": False,
            "imports_phase_qemu_v1_or_v2": False,
            "dense_state_representation": (
                "EVEN_PARITY_GAUSSIAN_INTEGER_AMPLITUDES_WITH_SHARED_SQRT2_EXPONENT"
            ),
            "operator_construction": "DIRECT_JORDAN_WIGNER_MAJORANA_ACTION",
            "independent_compact_representation": "SIGNED_MAJORANA_COVARIANCE",
            "floating_point_scientific_decisions": False,
            "hardcoded_expected_state_vectors_or_ranks": False,
        },
        "public_algebra": {
            "gamma_2a_minus_1": "Z_1...Z_(a-1) X_a",
            "gamma_2a": "Z_1...Z_(a-1) Y_a",
            "clockwise_adjacent_exchange": "C_j=(I-gamma_j gamma_(j+1))/sqrt(2)",
            "preparation_word": (
                "J_r=concat_(t=1..2r-3)[2t+2,2t+1,...,t+3]"
            ),
            "loop_operator": "L_(a,b)=-i gamma_a gamma_b",
            "loop_interpretation": (
                "BRAID_TRANSPORTED_MAJORANA_PAIR_PARITY_HOLONOMY"
            ),
            "wilson_loop_scope": (
                "NOT_AN_ARBITRARY_ENCLOSED_SUBSET_WILSON_LOOP_WITHOUT_A_"
                "SUPPLIED_TRANSPORT_AND_FRAMING_DESCRIPTOR"
            ),
            "sigma_probe_monodromy": {"charge_1": "+1", "charge_psi": "-1"},
        },
        "n4_algebra_orientation_smoke": n4_orientation_smoke(),
        "cases": cases,
        "measured_family_laws": {
            "minimum_supported_r": 2,
            "fusion_dimensions_n8_n12_n16": dimensions,
            "prepared_support_n8_n12_n16": supports,
            "ordinary_contiguous_central_schmidt_ranks_n8_n12_n16": family_ranks,
            "all_dense_ranks_match_compact_majorana_covariance": all(
                cases[str(n)]["ordinary_contiguous_central_cut"][
                    "dense_exact_schmidt_rank"
                ]
                == cases[str(n)]["ordinary_contiguous_central_cut"][
                    "compact_majorana_covariance_rank"
                ]
                for n in family
            ),
            "all_public_inverses_restore_exactly": True,
            "all_L0_loops_are_charge_1_plus": all(
                cases[str(n)]["definite_loop_queries"]["L0_charge_1"][
                    "controlled_sigma_probe_phase"
                ] == "+1" for n in family
            ),
            "all_L1_loops_are_charge_psi_minus": all(
                cases[str(n)]["definite_loop_queries"]["L1_charge_psi"][
                    "controlled_sigma_probe_phase"
                ] == "-1"
                for n in family
            ),
            "all_invalid_bilinear_controls_entangle": all(
                not cases[str(n)]["invalid_superposed_loop_control"][
                    "factorized_boundary_retention_lawful"
                ] for n in family
            ),
            "all_internal_fixed_charge_scrambles_preserve_both_phases": all(
                cases[str(n)]["internal_fixed_charge_scrambling"][
                    "both_fixed_charges_invariant"
                ] for n in family
            ),
        },
        "strongest_classical_comparator": {
            "name": "SIGNED_MAJORANA_FRAME_AND_SIGNED_PERFECT_PAIRING",
            "state_cells": "O(n) signed permutation/pairing entries",
            "adjacent_braid_work": "O(1) signed-frame swap with inverse replay",
            "bilinear_loop_work": "O(1) signed-pair lookup",
            "cut_rank_work": "O(n) count of pairing edges crossing the declared cut",
            "reproduces_every_accepted_dense_boundary_and_rank": True,
            "route_disposition": (
                "COMPACT_SIGNED_MAJORANA_PAIRING_IS_A_CONTROLLING_NO_GO"
            ),
        },
        "verification_classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "restoration_scope": "FUNCTIONAL_EXACT_STATE_EQUALITY_WITHOUT_SAME_BACKING",
        "claim_ceiling": "EXACT_IDEAL_ISING_ANYON_SOFTWARE_DIAGNOSTIC_ONLY",
        "claim_limits": {
            "qemu_or_catvm_device_execution": False,
            "same_backing_catalytic_reuse": False,
            "physical_anyons_or_majoranas": False,
            "physical_probe_or_monodromy": False,
            "machine_enforced_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "m257_escape": False,
            "small_wall_crossing": False,
            "unbounded_compute": False,
            "replacement_of_physical_bits_with_pi": False,
            "arbitrary_enclosed_subset_wilson_loop": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(result(), indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
