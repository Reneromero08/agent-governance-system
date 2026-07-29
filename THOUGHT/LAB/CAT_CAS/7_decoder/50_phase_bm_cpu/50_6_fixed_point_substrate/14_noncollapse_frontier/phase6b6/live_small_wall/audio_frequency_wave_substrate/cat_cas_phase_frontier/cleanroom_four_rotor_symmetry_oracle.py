#!/usr/bin/env python3
"""Independent combinatorial oracle for the four-rotor quotient ceilings."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterator


GRID = 17
ROTORS = 4


def compositions(total: int, width: int) -> Iterator[tuple[int, ...]]:
    if width == 1:
        yield (total,)
        return
    for head in range(total + 1):
        for tail in compositions(total - head, width - 1):
            yield (head, *tail)


def rotate(histogram: tuple[int, ...], shift: int) -> tuple[int, ...]:
    return tuple(
        histogram[(index - shift) % GRID] for index in range(GRID)
    )


def canonical(histogram: tuple[int, ...]) -> tuple[int, ...]:
    return min(rotate(histogram, shift) for shift in range(GRID))


def labelled_weight(histogram: tuple[int, ...]) -> int:
    result = math.factorial(sum(histogram))
    for count in histogram:
        result //= math.factorial(count)
    return result


def collision_count(histogram: tuple[int, ...]) -> int:
    return sum(count * (count - 1) // 2 for count in histogram)


def necklace_statistics(rotors: int) -> dict[str, int]:
    representatives: dict[tuple[int, ...], int] = {}
    for histogram in compositions(rotors, GRID):
        key = canonical(histogram)
        representatives.setdefault(key, 0)
        representatives[key] += labelled_weight(histogram)
    return {
        "weak_compositions": math.comb(rotors + GRID - 1, GRID - 1),
        "necklaces": len(representatives),
        "labelled_weight": sum(representatives.values()),
    }


def prime_burnside(rotors: int) -> int:
    fixed_nonidentity = 1 if rotors % GRID == 0 else 0
    return (
        math.comb(rotors + GRID - 1, GRID - 1)
        + (GRID - 1) * fixed_nonidentity
    ) // GRID


def open_chain_energy(labelled: tuple[int, ...]) -> float:
    angles = [2.0 * math.pi * mode / GRID for mode in labelled]
    return sum(
        math.cos(angles[index] - angles[index + 1])
        for index in range(len(angles) - 1)
    )


def pair_difference_energy(labelled: tuple[int, ...]) -> float:
    angles = [2.0 * math.pi * mode / GRID for mode in labelled]
    return sum(
        math.cos(angles[left] - angles[right])
        for left, right in itertools.combinations(range(ROTORS), 2)
    )


def onsite_energy(labelled: tuple[int, ...]) -> float:
    return sum(
        math.cos(2.0 * math.pi * mode / GRID) for mode in labelled
    )


def histogram(labelled: tuple[int, ...]) -> tuple[int, ...]:
    counts = [0] * GRID
    for mode in labelled:
        counts[mode] += 1
    return tuple(counts)


def main() -> int:
    rotor4 = necklace_statistics(4)
    rotor5 = necklace_statistics(5)
    if rotor4 != {
        "weak_compositions": 4845,
        "necklaces": 285,
        "labelled_weight": 83521,
    }:
        raise RuntimeError("four-rotor necklace enumeration failed")
    if rotor5["necklaces"] != 1197:
        raise RuntimeError("five-rotor transfer enumeration failed")

    labelled = (0, 1, 3, 7)
    swapped = (0, 3, 1, 7)
    if histogram(labelled) != histogram(swapped):
        raise RuntimeError("exchange-symmetry mutation changed occupancy")
    if collision_count(histogram(labelled)) != collision_count(
        histogram(swapped)
    ):
        raise RuntimeError("exchange mutation changed collision invariant")
    open_chain_delta = abs(
        open_chain_energy(labelled) - open_chain_energy(swapped)
    )
    if open_chain_delta < 1e-6:
        raise RuntimeError("labelled open-chain mutation did not separate")

    rotated = tuple((mode + 4) % GRID for mode in labelled)
    pair_rotation_delta = abs(
        pair_difference_energy(labelled)
        - pair_difference_energy(rotated)
    )
    onsite_rotation_delta = abs(
        onsite_energy(labelled) - onsite_energy(rotated)
    )
    if pair_rotation_delta > 1e-12 or onsite_rotation_delta < 1e-6:
        raise RuntimeError("global-rotation sector attack failed")

    sector_states = sum(
        1
        for momenta in itertools.product(range(GRID), repeat=4)
        if sum(momenta) % GRID == 0
    )
    if sector_states != GRID**3:
        raise RuntimeError("total-momentum sector dimension failed")
    outside_sector = (0, 0, 0, 1)
    if sum(outside_sector) % GRID == 0:
        raise RuntimeError("outside-sector witness was not outside")

    burnside4 = prime_burnside(4)
    burnside5 = prime_burnside(5)
    burnside17 = prime_burnside(17)
    naive17 = math.comb(33, 16) // GRID
    if (
        burnside4 != rotor4["necklaces"]
        or burnside5 != rotor5["necklaces"]
        or burnside17 - naive17 != 1
    ):
        raise RuntimeError("Burnside/stabilizer law failed")

    result = {
        "source_head": "65be0046ae02c79ab8c3b3356ef68d891de19e53",
        "result": "PASS",
        "production_imports": 0,
        "global_rotation_quotient": {
            "declared_sector": "TOTAL_MOMENTUM_ZERO",
            "sector_cells": sector_states,
            "full_labelled_momentum_cells": GRID**4,
            "reduction_factor": GRID,
            "outside_sector_witness": list(outside_sector),
            "pair_difference_rotation_error": pair_rotation_delta,
            "onsite_rotation_effect": onsite_rotation_delta,
        },
        "necklace": {
            "grid": GRID,
            "rotors": ROTORS,
            **rotor4,
            "rotor5_necklaces": rotor5["necklaces"],
            "exchange_symmetry_required": True,
            "labelled_open_chain_energy_effect": open_chain_delta,
            "labelled_open_chain_compressed": False,
        },
        "burnside": {
            "law": (
                "(C(R+16,16)+16*INDICATOR(17_DIVIDES_R))/17"
            ),
            "rotor4": burnside4,
            "rotor5": burnside5,
            "rotor17": burnside17,
            "rotor17_naive_division": naive17,
            "stabilizer_correction": burnside17 - naive17,
            "simple_division_ceiling": "R_LESS_THAN_17",
        },
        "topology_compilation_uses_final_answer": False,
        "restoration_class": "NO_RESTORATION_CLAIM",
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
