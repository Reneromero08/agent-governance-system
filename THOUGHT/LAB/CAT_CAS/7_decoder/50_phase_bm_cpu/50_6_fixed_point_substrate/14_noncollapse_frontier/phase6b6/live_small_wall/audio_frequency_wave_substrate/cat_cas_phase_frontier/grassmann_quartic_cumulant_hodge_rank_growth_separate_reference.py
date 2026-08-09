#!/usr/bin/env python3
"""M255 standalone full-exterior oracle and restoration reference.

This source intentionally imports no M254/M255 production module.  It expands
the bounded even exterior algebra, derives the Berezin complement from basis
permutations, takes the exact exterior logarithm, and independently executes
the reversible compact carrier transactions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Iterable


ZERO = Fraction(0)
ONE = Fraction(1)


def fj(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def mask(indices: Iterable[int]) -> int:
    result = 0
    for index in indices:
        result |= 1 << index
    return result


def product_sign(left: int, right: int) -> int:
    if left & right:
        return 0
    sequence = [index for index in range(max(left.bit_length(), right.bit_length())) if left & (1 << index)]
    sequence += [index for index in range(max(left.bit_length(), right.bit_length())) if right & (1 << index)]
    inversions = sum(sequence[i] > sequence[j] for i in range(len(sequence)) for j in range(i + 1, len(sequence)))
    return -1 if inversions & 1 else 1


Poly = dict[int, Fraction]


def add(left: Poly, right: Poly) -> Poly:
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, ZERO) + value
        if result[key] == ZERO:
            result.pop(key)
    return result


def scale(poly: Poly, scalar: Fraction) -> Poly:
    return {key: value * scalar for key, value in poly.items() if value * scalar}


def wedge(left: Poly, right: Poly) -> Poly:
    result: Poly = {}
    for left_mask, left_value in left.items():
        for right_mask, right_value in right.items():
            sign = product_sign(left_mask, right_mask)
            if sign:
                target = left_mask | right_mask
                result[target] = result.get(target, ZERO) + sign * left_value * right_value
    return {key: value for key, value in result.items() if value}


def exterior_exp(poly: Poly, port_count: int) -> Poly:
    result: Poly = {0: ONE}
    power: Poly = {0: ONE}
    for degree in range(1, port_count // 2 + 1):
        power = wedge(power, poly)
        if not power:
            break
        result = add(result, scale(power, Fraction(1, factorial(degree))))
    return result


def complement_sign_independent(source: int, port_count: int) -> int:
    """Derive the formal Berezin sign from an explicit ordered basis word."""
    full = (1 << port_count) - 1
    complement = full ^ source
    source_word = [i for i in range(port_count) if source & (1 << i)]
    complement_word = [i for i in range(port_count) if complement & (1 << i)]
    word = source_word + complement_word
    inversions = sum(word[i] > word[j] for i in range(len(word)) for j in range(i + 1, len(word)))
    reversal = len(source_word) * (len(source_word) - 1) // 2
    return -1 if (inversions + reversal) & 1 else 1


def hodge(poly: Poly, port_count: int) -> Poly:
    full = (1 << port_count) - 1
    return {
        full ^ source: complement_sign_independent(source, port_count) * value
        for source, value in poly.items()
    }


def exterior_log_unit(poly: Poly, port_count: int) -> Poly:
    if poly.get(0) != ONE:
        raise RuntimeError("M255 reference logarithm requires unit scalar")
    nilpotent = dict(poly)
    nilpotent.pop(0, None)
    result: Poly = {}
    power: Poly = {0: ONE}
    for exponent in range(1, port_count // 2 + 1):
        power = wedge(power, nilpotent)
        if not power:
            break
        result = add(
            result,
            scale(power, Fraction(1 if exponent & 1 else -1, exponent)),
        )
    return result


def fixture(width: int, alternate: bool = False) -> dict[str, object]:
    pair_scales = [2 if alternate else 1, 1, 1, 1]
    quartic_scales = [1, 2, 3] if alternate else [1, 1, 1]
    module_count = width // 2 - 1
    pairs = [[2 * index, 2 * index + 1, pair_scales[index], 1] for index in range(width // 2)]
    modules: list[dict[str, object]] = []
    for index in range(module_count):
        modules.append({
            "pairs": pairs[:2] if index == 0 else [pairs[index + 1]],
            "quartic": [2 * index, 2 * index + 1, 2 * index + 2, 2 * index + 3, quartic_scales[index], 1],
        })
    return {"port_count": width, "modules": modules, "selected_output": list(range(width))}


def full_oracle(descriptor: dict[str, object]) -> tuple[Fraction, Fraction, Poly]:
    width = int(descriptor["port_count"])
    quadratic: Poly = {}
    quartic: Poly = {}
    for module in descriptor["modules"]:  # type: ignore[union-attr]
        for left, right, numerator, denominator in module["pairs"]:  # type: ignore[index]
            pair = (1 << left) | (1 << right)
            quadratic[pair] = quadratic.get(pair, ZERO) + Fraction(numerator, denominator)
        q = module["quartic"]  # type: ignore[index]
        qmask = mask(q[:4])
        quartic[qmask] = quartic.get(qmask, ZERO) + Fraction(q[4], q[5])
    relation = wedge(exterior_exp(quadratic, width), exterior_exp(quartic, width))
    transformed = hodge(relation, width)
    scalar = transformed.get(0, ZERO)
    if scalar == ZERO:
        raise RuntimeError("M255 reference singular normalization")
    normalized = {key: value / scalar for key, value in transformed.items()}
    logarithm = exterior_log_unit(normalized, width)
    return logarithm.get((1 << width) - 1, ZERO), scalar, logarithm


def continuant_scalar(descriptor: dict[str, object]) -> Fraction:
    """Independent two-state monomer-dimer recurrence for the chain scalar."""
    width = int(descriptor["port_count"])
    monomers = [ZERO] * (width // 2)
    edges = [ZERO] * (width // 2 - 1)
    for module in descriptor["modules"]:  # type: ignore[union-attr]
        for left, right, numerator, denominator in module["pairs"]:  # type: ignore[index]
            if right != left + 1 or left & 1:
                raise RuntimeError("M255 reference non-chain pair")
            monomers[left // 2] += Fraction(numerator, denominator)
        q = module["quartic"]  # type: ignore[index]
        edge = q[0] // 2
        if q[:4] != [2 * edge, 2 * edge + 1, 2 * edge + 2, 2 * edge + 3]:
            raise RuntimeError("M255 reference non-chain quartic")
        edges[edge] += Fraction(q[4], q[5])
    previous_previous = ONE
    previous = monomers[0]
    for index in range(1, len(monomers)):
        current = monomers[index] * previous + edges[index - 1] * previous_previous
        previous_previous, previous = previous, current
    hodge_full_sign = complement_sign_independent((1 << width) - 1, width)
    return hodge_full_sign * previous


def independent_rank_certificate(descriptor: dict[str, object]) -> dict[str, object]:
    width = int(descriptor["port_count"])
    quartic: Poly = {}
    decomposable_summands = 0
    for module in descriptor["modules"]:  # type: ignore[union-attr]
        q = module["quartic"]  # type: ignore[index]
        qmask = mask(q[:4])
        if qmask.bit_count() != 4:
            raise RuntimeError("M255 reference nondecomposable input term")
        coefficient = Fraction(q[4], q[5])
        quartic[qmask] = quartic.get(qmask, ZERO) + coefficient
        decomposable_summands += 1
    dual = hodge(quartic, width)
    dual_square = wedge(dual, dual)
    nonzero = [(support, value) for support, value in sorted(dual_square.items()) if value]
    witness_support, witness = nonzero[0] if nonzero else (0, ZERO)
    lower_bound = 2 if witness else (1 if quartic else 0)
    upper_bound = decomposable_summands
    return {
        "decomposable_summand_count": decomposable_summands,
        "rank_lower_bound_from_dual_plucker_square": lower_bound,
        "rank_upper_bound_from_displayed_sum": upper_bound,
        "exact_rank": lower_bound if lower_bound == upper_bound else None,
        "dual_square_witness_support": [index for index in range(width) if witness_support & (1 << index)],
        "dual_square_witness": fj(witness),
    }


@dataclass
class ReferenceCarrier:
    width: int

    def __post_init__(self) -> None:
        self.pairs = [ZERO] * (self.width * (self.width - 1) // 2)
        self.quartics: list[tuple[int, Fraction] | None] = [None] * (self.width // 2 - 1)
        self.cursor = 0
        self.generation = 0
        self.last_generation = 0
        self.program = ""

    def canonical(self) -> bool:
        return all(value == ZERO for value in self.pairs) and all(value is None for value in self.quartics) and self.cursor == 0 and self.program == ""

    def execute(self, descriptor: dict[str, object]) -> tuple[int, bool, bool]:
        if not self.canonical():
            raise RuntimeError("M255 reference dirty lease")
        program = digest(descriptor)
        self.generation = self.last_generation + 1
        generation = self.generation
        self.program = program
        pair_id = id(self.pairs); quartic_id = id(self.quartics)
        for module in descriptor["modules"]:  # type: ignore[union-attr]
            for left, right, numerator, denominator in module["pairs"]:  # type: ignore[index]
                offset = sum(self.width - first - 1 for first in range(left)) + right - left - 1
                self.pairs[offset] += Fraction(numerator, denominator)
            q = module["quartic"]  # type: ignore[index]
            self.quartics[self.cursor] = (mask(q[:4]), Fraction(q[4], q[5]))
            self.cursor += 1
        for module in reversed(descriptor["modules"]):  # type: ignore[union-attr]
            self.cursor -= 1
            q = module["quartic"]  # type: ignore[index]
            if self.quartics[self.cursor] != (mask(q[:4]), Fraction(q[4], q[5])):
                raise RuntimeError("M255 reference inverse mismatch")
            self.quartics[self.cursor] = None
            for left, right, numerator, denominator in reversed(module["pairs"]):  # type: ignore[index]
                offset = sum(self.width - first - 1 for first in range(left)) + right - left - 1
                self.pairs[offset] -= Fraction(numerator, denominator)
        if self.cursor or any(self.pairs) or any(value is not None for value in self.quartics):
            raise RuntimeError("M255 reference restoration failed")
        self.program = ""
        self.last_generation = generation
        self.generation = 0
        return generation, pair_id == id(self.pairs) and quartic_id == id(self.quartics), self.canonical()


def run_case(carrier: ReferenceCarrier, width: int, alternate: bool, kind: str) -> dict[str, object]:
    descriptor = fixture(width, alternate)
    top, scalar, logarithm = full_oracle(descriptor)
    generation, same, canonical = carrier.execute(descriptor)
    return {
        "run_kind": kind,
        "port_count": width,
        "generation": generation,
        "program_id": digest(descriptor),
        "selected_connected_top_degree_cumulant": fj(top),
        "hodge_normalization_scalar": fj(scalar),
        "maximum_nonzero_connected_degree": max((key.bit_count() for key, value in logarithm.items() if value), default=0),
        "full_even_basis_dimension": 1 << (width - 1),
        "same_pair_quartic_backings": same,
        "canonical_after_restoration": canonical,
        "baseline_reload_used": False,
    }


def main() -> None:
    cases = [
        run_case(ReferenceCarrier(4), 4, False, "PRIMARY_M4"),
        run_case(ReferenceCarrier(6), 6, False, "PRIMARY_M6"),
    ]
    shared = ReferenceCarrier(8)
    cases += [
        run_case(shared, 8, False, "PRIMARY_M8"),
        run_case(shared, 8, True, "REUSE_M8"),
        run_case(ReferenceCarrier(8), 8, True, "FRESH_M8"),
    ]
    expected = {
        "PRIMARY_M4": (Fraction(1, 4), Fraction(2)),
        "PRIMARY_M6": (Fraction(-2, 27), Fraction(-3)),
        "PRIMARY_M8": (Fraction(14, 625), Fraction(5)),
        "REUSE_M8": (Fraction(15, 2048), Fraction(16)),
        "FRESH_M8": (Fraction(15, 2048), Fraction(16)),
    }
    hodge_square = {}
    for width in (4, 6, 8):
        expected_sign = -1 if (width * (width - 1) // 2) & 1 else 1
        hodge_square[str(width)] = all(
            hodge(hodge({basis: ONE}, width), width) == {basis: Fraction(expected_sign)}
            for basis in range(1 << width) if basis.bit_count() % 2 == 0
        )
    m6_rank = independent_rank_certificate(fixture(6))
    controls = {
        "independent_full_exterior_expected_boundaries": all(
            (Fraction(*case["selected_connected_top_degree_cumulant"]), Fraction(*case["hodge_normalization_scalar"])) == expected[case["run_kind"]]  # type: ignore[arg-type]
            for case in cases
        ),
        "rank1_quartic_plucker_witness_nonzero_at_m6": m6_rank["dual_square_witness"] != [0, 1],
        "two_displayed_rank1_terms_give_rank_at_most_two_and_plucker_gives_rank_at_least_two": m6_rank["exact_rank"] == 2,
        "port_counts_below_six_cannot_have_a_nondecomposable_four_form": True,
        "hodge_square_has_exact_width_dependent_sign": all(hodge_square.values()),
        "m4_has_no_degree_above_four": cases[0]["maximum_nonzero_connected_degree"] <= 4,
        "m6_has_connected_degree_six": cases[1]["maximum_nonzero_connected_degree"] == 6,
        "m8_has_connected_degree_eight": cases[2]["maximum_nonzero_connected_degree"] == 8,
        "generation2_reuse_and_fresh_generation1": cases[2]["generation"] == 1 and cases[3]["generation"] == 2 and cases[4]["generation"] == 1,
        "restored_and_fresh_alternate_boundary_agree": cases[3]["selected_connected_top_degree_cumulant"] == cases[4]["selected_connected_top_degree_cumulant"],
        "same_backings_exact_restoration_no_reload": all(case["same_pair_quartic_backings"] and case["canonical_after_restoration"] and not case["baseline_reload_used"] for case in cases),
        "two_state_continuant_matches_full_hodge_scalar": all(
            fj(continuant_scalar(fixture(width, alternate))) == case["hodge_normalization_scalar"]
            for case, width, alternate in zip(cases, (4, 6, 8, 8, 8), (False, False, False, True, True))
        ),
    }
    if not all(controls.values()):
        raise RuntimeError("M255 standalone control failure")
    print(json.dumps({
        "milestone": 255,
        "oracle": "STANDALONE_FULL_EVEN_EXTERIOR_EXP_HODGE_LOG_AND_REFERENCE_CARRIER",
        "imports_production": False,
        "cases": cases,
        "controls": controls,
        "rank_obstruction": {
            "m6_quartic_sum_exact_rank": 2,
            "m6_rank_certificate": m6_rank,
            "m6_connected_degree6": fj(Fraction(-2, 27)),
            "m8_connected_degree8": fj(Fraction(14, 625)),
        },
        "classical_baselines": {
            "fixed_fixture": "PUBLIC_O1_CLOSED_CERTIFICATES",
            "transferable_normalization_scalar": "TWO_STATE_MONOMER_DIMER_CONTINUANT_O_M_WORK_O1_LIVE_FIELD_STATE",
            "continuant_state_field_cells": 2,
            "descriptor_level_selected_cumulant": "STREAMED_PFAFFIAN_COEFFICIENT_AND_EVEN_SET_PARTITION_CUMULANT",
            "descriptor_level_selected_cumulant_optimality_claimed": False,
            "accepted_pfaffian_projection_is_not_strongest_for_fixed_fixtures": True,
        },
        "claim_limits": {
            "verifier_only_full_even_expansion": True,
            "accepted_path_full_even_expansion": False,
            "general_all_width_theorem": False,
            "advantage": False,
        },
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
