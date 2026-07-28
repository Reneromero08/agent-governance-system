#!/usr/bin/env python3
"""Rematerialized character-phase BCH closure for two Stokes modules."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass

import sympy

import algebraic_stokes_alternating_axis_phase as character
import algebraic_stokes_lie_relation_phase as base


MAX_WORD_GRADE = 6
RESTORE_TOLERANCE = 2.0e-10
Word = tuple[str, ...]
Polynomial = dict[base.Monomial, tuple[int, int]]


@dataclass
class Stats:
    phase_updates: int = 0
    phase_reads: int = 0
    word_rematerializations: int = 0
    maximum_live_scratch_blocks: int = 0
    hidden_decodes: int = 0
    final_decodes: int = 0
    maximum_root_error: float = 0.0


@dataclass
class Carrier:
    finals: list[character.Block]
    scratch: list[character.Block]


def make_block(degree: int) -> character.Block:
    canonical = base.basis(degree, parity_reduced=True)
    return character.Block(
        degree,
        canonical,
        {
            monomial: index
            for index, monomial in enumerate(canonical)
        },
        [
            [
                [1.0 + 0.0j for _ in range(prime)]
                for _ in canonical
            ]
            for prime in base.PRIMES
        ],
    )


def make_carrier() -> Carrier:
    return Carrier(
        finals=[
            make_block(grade + 1)
            for grade in range(1, MAX_WORD_GRADE + 1)
        ],
        scratch=[
            make_block(degree)
            for degree in range(2, MAX_WORD_GRADE + 2)
        ],
    )


def tensor_multiply(
    left: dict[Word, sympy.Rational],
    right: dict[Word, sympy.Rational],
) -> dict[Word, sympy.Rational]:
    result: defaultdict[Word, sympy.Rational] = defaultdict(
        lambda: sympy.Rational(0)
    )
    for left_word, left_value in left.items():
        for right_word, right_value in right.items():
            word = left_word + right_word
            if len(word) <= MAX_WORD_GRADE:
                result[word] += left_value * right_value
    return {
        word: value for word, value in result.items() if value
    }


def bch_words() -> list[tuple[Word, sympy.Rational]]:
    exponential: dict[Word, sympy.Rational] = {
        (): sympy.Rational(1)
    }
    for left_power in range(MAX_WORD_GRADE + 1):
        for right_power in range(
            MAX_WORD_GRADE + 1 - left_power
        ):
            if left_power + right_power:
                exponential[
                    ("A",) * left_power + ("B",) * right_power
                ] = sympy.Rational(
                    1,
                    math.factorial(left_power)
                    * math.factorial(right_power),
                )
    augmentation = {
        word: value
        for word, value in exponential.items()
        if word
    }
    logarithm: defaultdict[Word, sympy.Rational] = defaultdict(
        lambda: sympy.Rational(0)
    )
    power: dict[Word, sympy.Rational] = {
        (): sympy.Rational(1)
    }
    for order in range(1, MAX_WORD_GRADE + 1):
        power = tensor_multiply(power, augmentation)
        sign = 1 if order % 2 else -1
        for word, value in power.items():
            logarithm[word] += (
                sign * value / sympy.Integer(order)
            )
    return sorted(
        (
            word,
            value / sympy.Integer(len(word)),
        )
        for word, value in logarithm.items()
        if value
    )


def generators(program: str) -> dict[str, Polynomial]:
    if program == "PRIMARY":
        tilted = base.mixed_hamiltonian(-1)
        return {"A": base.h0(), "B": tilted}
    elif program == "REUSE":
        tilted = base.mixed_hamiltonian(1)
        return {"A": base.h0(), "B": tilted}
    elif program == "SWAPPED":
        tilted = base.mixed_hamiltonian(-1)
        return {"A": tilted, "B": base.h0()}
    else:
        raise RuntimeError("unknown BCH phase program")


def multiply(
    block: character.Block,
    field: int,
    cell: int,
    harmonic: int,
    factor: complex,
    stats: Stats,
) -> None:
    block.working[field][cell][harmonic] = base.unit(
        block.working[field][cell][harmonic] * factor
    )
    stats.phase_updates += 1


def seal(
    block: character.Block,
    polynomial: Polynomial,
    inverse: bool,
    stats: Stats,
) -> None:
    for monomial, (numerator, denominator) in polynomial.items():
        cell = block.index[monomial]
        for field, prime in enumerate(base.PRIMES):
            value = base.residue(numerator, denominator, prime)
            for harmonic in range(prime):
                phase = base.root(field, value * harmonic)
                multiply(
                    block,
                    field,
                    cell,
                    harmonic,
                    phase.conjugate() if inverse else phase,
                    stats,
                )


def rational_residue(value: sympy.Rational, prime: int) -> int:
    numerator, denominator = sympy.fraction(value)
    return (
        int(numerator)
        * pow(int(denominator), -1, prime)
        % prime
    )


def accumulate_scaled(
    source: character.Block,
    target: character.Block,
    scale: sympy.Rational,
    inverse: bool,
    stats: Stats,
) -> None:
    for field, prime in enumerate(base.PRIMES):
        scalar = rational_residue(scale, prime)
        if scalar == 0:
            continue
        for source_cell, monomial in enumerate(source.basis):
            target_cell = target.index[monomial]
            source_character = source.working[field][source_cell]
            for harmonic in range(prime):
                contribution = source_character[
                    (scalar * harmonic) % prime
                ]
                multiply(
                    target,
                    field,
                    target_cell,
                    harmonic,
                    (
                        contribution.conjugate()
                        if inverse
                        else contribution
                    ),
                    stats,
                )
                stats.phase_reads += 1


def accumulate_bracket(
    source: character.Block,
    target: character.Block,
    generator: Polynomial,
    inverse: bool,
    stats: Stats,
) -> None:
    cyclic = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for left, (numerator, denominator) in generator.items():
        for source_cell, right in enumerate(source.basis):
            for output_variable, first, second in cyclic:
                derivative = (
                    left[first] * right[second]
                    - left[second] * right[first]
                )
                if derivative == 0:
                    continue
                raw = [
                    left[index] + right[index]
                    for index in range(3)
                ]
                raw[first] -= 1
                raw[second] -= 1
                raw[output_variable] += 1
                for reduced, reduction in base.canonical_reduction(
                    tuple(raw)
                ):
                    target_cell = target.index[reduced]
                    for field, prime in enumerate(base.PRIMES):
                        scalar = (
                            base.residue(
                                numerator, denominator, prime
                            )
                            * derivative
                            * reduction
                        ) % prime
                        if scalar == 0:
                            continue
                        source_character = (
                            source.working[field][source_cell]
                        )
                        for harmonic in range(prime):
                            contribution = source_character[
                                (scalar * harmonic) % prime
                            ]
                            multiply(
                                target,
                                field,
                                target_cell,
                                harmonic,
                                (
                                    contribution.conjugate()
                                    if inverse
                                    else contribution
                                ),
                                stats,
                            )
                            stats.phase_reads += 1


def apply_word(
    carrier: Carrier,
    word: Word,
    coefficient: sympy.Rational,
    program_generators: dict[str, Polynomial],
    inverse_final: bool,
    stats: Stats,
) -> None:
    seal(
        carrier.scratch[0],
        program_generators[word[-1]],
        False,
        stats,
    )
    for level, letter in enumerate(reversed(word[:-1]), start=1):
        accumulate_bracket(
            carrier.scratch[level - 1],
            carrier.scratch[level],
            program_generators[letter],
            False,
            stats,
        )
    stats.maximum_live_scratch_blocks = max(
        stats.maximum_live_scratch_blocks, len(word)
    )
    accumulate_scaled(
        carrier.scratch[len(word) - 1],
        carrier.finals[len(word) - 1],
        coefficient,
        inverse_final,
        stats,
    )
    for level, letter in reversed(
        list(enumerate(reversed(word[:-1]), start=1))
    ):
        accumulate_bracket(
            carrier.scratch[level - 1],
            carrier.scratch[level],
            program_generators[letter],
            True,
            stats,
        )
    seal(
        carrier.scratch[0],
        program_generators[word[-1]],
        True,
        stats,
    )
    stats.word_rematerializations += 1


def residual(carrier: Carrier) -> float:
    return max(
        abs(phase - 1.0)
        for block in carrier.finals + carrier.scratch
        for field_coordinates in block.working
        for phase_character in field_coordinates
        for phase in phase_character
    )


def snapshot_reload(carrier: Carrier) -> None:
    for block in carrier.finals + carrier.scratch:
        for field_coordinates in block.working:
            for phase_character in field_coordinates:
                for harmonic in range(len(phase_character)):
                    phase_character[harmonic] = 1.0 + 0.0j


def project(
    carrier: Carrier, stats: Stats
) -> list[dict[str, object]]:
    boundary = []
    for block in carrier.finals:
        record: dict[str, object] = {
            "polynomial_degree": block.degree_limit,
            "quotient_basis_cells": len(block.basis),
        }
        for field, prime in enumerate(base.PRIMES):
            hash_value = character.FNV_OFFSET
            nonzero = 0
            for cell, monomial in enumerate(block.basis):
                for exponent in monomial:
                    hash_value = character.hash_byte(
                        hash_value, exponent
                    )
                phase = block.working[field][cell][1]
                distances = [
                    abs(phase - base.root(field, value))
                    for value in range(prime)
                ]
                value = min(
                    range(prime), key=distances.__getitem__
                )
                stats.maximum_root_error = max(
                    stats.maximum_root_error, distances[value]
                )
                stats.final_decodes += 1
                hash_value = character.hash_byte(
                    hash_value, value
                )
                nonzero += int(value != 0)
            record[f"nonzero_p{prime}"] = nonzero
            record[f"hash_p{prime}"] = (
                f"{hash_value:016x}"
            )
        boundary.append(record)
    return boundary


def execute_on(
    carrier: Carrier,
    program: str,
    omit_last_inverse: bool = False,
    wrong_first_inverse: bool = False,
) -> tuple[list[dict[str, object]], Stats, float]:
    stats = Stats()
    compiled_words = bch_words()
    program_generators = generators(program)
    for word, coefficient in compiled_words:
        apply_word(
            carrier,
            word,
            coefficient,
            program_generators,
            False,
            stats,
        )
    boundary = project(carrier, stats)
    inverse_words = list(reversed(compiled_words))
    if omit_last_inverse:
        inverse_words = inverse_words[:-1]
    for inverse_index, (word, coefficient) in enumerate(
        inverse_words
    ):
        if (
            wrong_first_inverse
            and inverse_index == len(inverse_words) - 1
        ):
            coefficient += sympy.Rational(1)
        apply_word(
            carrier,
            word,
            coefficient,
            program_generators,
            True,
            stats,
        )
    return boundary, stats, residual(carrier)


def main() -> None:
    if sys.argv[1:] == ["--project-scratch"]:
        raise RuntimeError("resident BCH scratch projection denied")
    if sys.argv[1:] == ["--null-carrier"]:
        raise RuntimeError("invalid BCH phase carrier")
    if sys.argv[1:]:
        raise RuntimeError("unsupported BCH phase request")

    carrier = make_carrier()
    primary, primary_stats, primary_residual = execute_on(
        carrier, "PRIMARY"
    )
    reuse, reuse_stats, reuse_residual = execute_on(
        carrier, "REUSE"
    )
    swapped_carrier = make_carrier()
    swapped, _, swapped_residual = execute_on(
        swapped_carrier, "SWAPPED"
    )
    failed_carrier = make_carrier()
    _, _, missing_residual = execute_on(
        failed_carrier, "PRIMARY", omit_last_inverse=True
    )
    wrong_carrier = make_carrier()
    _, _, wrong_residual = execute_on(
        wrong_carrier, "PRIMARY", wrong_first_inverse=True
    )
    snapshot_carrier = make_carrier()
    snapshot_payload_bytes = (
        sum(
            len(block.basis)
            for block in (
                snapshot_carrier.finals
                + snapshot_carrier.scratch
            )
        )
        * sum(base.PRIMES)
        * 16
    )
    snapshot_stats = Stats()
    snapshot_words = bch_words()
    snapshot_generators = generators("PRIMARY")
    for word, coefficient in snapshot_words:
        apply_word(
            snapshot_carrier,
            word,
            coefficient,
            snapshot_generators,
            False,
            snapshot_stats,
        )
    snapshot_boundary = project(
        snapshot_carrier, snapshot_stats
    )
    snapshot_reload(snapshot_carrier)
    snapshot_residual = residual(snapshot_carrier)
    snapshot_reuse, _, snapshot_reuse_residual = execute_on(
        snapshot_carrier, "REUSE"
    )
    if not (
        primary_residual <= RESTORE_TOLERANCE
        and reuse_residual <= RESTORE_TOLERANCE
        and missing_residual >= 1.0e-6
        and wrong_residual >= 1.0e-6
        and swapped_residual <= RESTORE_TOLERANCE
        and swapped != primary
        and snapshot_boundary == primary
        and snapshot_residual == 0.0
        and snapshot_reuse == reuse
        and snapshot_reuse_residual <= RESTORE_TOLERANCE
        and primary_stats.hidden_decodes == 0
        and primary_stats.maximum_root_error <= 1.0e-3
    ):
        raise RuntimeError("rematerialized BCH phase control failed")

    final_cells = sum(len(block.basis) for block in carrier.finals)
    scratch_cells = sum(
        len(block.basis) for block in carrier.scratch
    )
    print(
        json.dumps(
            {
                "result": "PASS",
                "claim": (
                    "BOUNDED_TOPOLOGY_REMATERIALIZED_NONCOMMUTING_"
                    "STOKES_BCH_CHARACTER_PHASE_CLOSURE_WITH_"
                    "RESTORATION_AND_REUSE"
                ),
                "maximum_word_grade": MAX_WORD_GRADE,
                "maximum_polynomial_degree": MAX_WORD_GRADE + 1,
                "declared_boundary": (
                    "FULL_GRADE1_TO6_BCH_COEFFICIENT_SIGNATURE"
                ),
                "primary_components": primary,
                "reuse_components": reuse,
                "swapped_order_boundary_differs": swapped != primary,
                "compiled_nonzero_lie_words": len(bch_words()),
                "compiled_public_word_letter_bytes": sum(
                    len(word) for word, _ in bch_words()
                ),
                "compiled_public_coefficient_logical_bytes": (
                    len(bch_words()) * 16
                ),
                "compiled_public_topology_actual_allocation_measured": (
                    False
                ),
                "word_rematerializations_primary": (
                    primary_stats.word_rematerializations
                ),
                "maximum_live_scratch_blocks": (
                    primary_stats.maximum_live_scratch_blocks
                ),
                "retained_word_history_blocks": 0,
                "final_coefficient_cells": final_cells,
                "reusable_scratch_coefficient_cells": scratch_cells,
                "resident_character_phase_cells": (
                    (final_cells + scratch_cells)
                    * sum(base.PRIMES)
                ),
                "logical_packed_phase_payload_bytes": (
                    (final_cells + scratch_cells)
                    * sum(base.PRIMES)
                    * 16
                ),
                "same_output_dual_prime_classical_semantic_state_bytes": (
                    final_cells * len(base.PRIMES)
                ),
                "same_output_classical_actual_allocation_measured": False,
                "native_phase_updates": primary_stats.phase_updates,
                "resident_phase_reads": primary_stats.phase_reads,
                "hidden_decodes": primary_stats.hidden_decodes,
                "final_decodes": primary_stats.final_decodes,
                "maximum_final_root_error": (
                    primary_stats.maximum_root_error
                ),
                "restoration_max_abs": primary_residual,
                "reuse_restoration_max_abs": reuse_residual,
                "missing_inverse_residual": missing_residual,
                "wrong_inverse_residual": wrong_residual,
                "reordered_inverse_applicable": False,
                "reordered_inverse_reason": (
                    "FINAL_BCH_COMPONENT_ADDITIONS_COMMUTE_WHILE_"
                    "INTRAWORD_DEPENDENCY_ORDER_IS_TOPOLOGICALLY_FIXED"
                ),
                "snapshot_payload_bytes": snapshot_payload_bytes,
                "snapshot_creation_traffic_bytes": snapshot_payload_bytes,
                "snapshot_reload_traffic_bytes": snapshot_payload_bytes,
                "snapshot_sham_loaded": True,
                "snapshot_sham_residual": snapshot_residual,
                "snapshot_sham_reuse_boundary_matches": (
                    snapshot_reuse == reuse
                ),
                "snapshot_sham_reuse_residual": (
                    snapshot_reuse_residual
                ),
                "snapshot_restoration_receipt": 0,
                "actual_inverse_restoration": True,
                "actual_restored_carrier_reuse": True,
                "topology_derived_rematerialization": True,
                "final_boundary_only_projection": True,
                "snapshot_loaded": False,
                "fixed_rank_nonseparable_closure_established": False,
                "genuinely_distinct_phase_resource": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
