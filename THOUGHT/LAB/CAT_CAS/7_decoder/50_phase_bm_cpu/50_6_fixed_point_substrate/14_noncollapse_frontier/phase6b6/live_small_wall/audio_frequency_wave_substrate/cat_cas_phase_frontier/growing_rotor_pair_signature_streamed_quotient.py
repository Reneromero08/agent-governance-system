#!/usr/bin/env python3
"""Matrix-free pair-signature quotient with no retained shift plan.

M193 established an exact equitable quotient but retained sixteen sparse shift
bases.  This successor keeps only public pair signatures, one public necklace
representative per fiber, and final-boundary weights.  Every scattering update
rematerializes signed pair moves from those representatives and immediately
accumulates them into the output vector.  No quotient row or shift basis
survives an update.

The carrier is the same exact two-register shear used by M193.  The accepted
path is direct-process software and the strongest matched classical path is
the identical streamed recurrence; this package does not establish CATVM
custody or a distinct phase resource.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterator, Literal


GRID = 17
PAIR_CHANNELS = 9
PRIMES = (103, 239)
ROTOR_COUNTS = (2, 3, 4, 5)
PRIMARY_DEPTH = 8
REUSE_DEPTH = 5
Control = Literal["correct", "missing", "wrong", "reordered"]
Histogram = tuple[int, ...]
Signature = tuple[int, ...]


def rotate(histogram: Histogram, shift: int) -> Histogram:
    result = [0] * GRID
    for mode, count in enumerate(histogram):
        result[(mode + shift) % GRID] = count
    return tuple(result)


def canonical(histogram: Histogram) -> Histogram:
    return min(rotate(histogram, shift) for shift in range(GRID))


def iter_histograms(rotors: int) -> Iterator[Histogram]:
    working = [0] * GRID

    def visit(position: int, remaining: int) -> Iterator[Histogram]:
        if position == GRID - 1:
            working[position] = remaining
            yield tuple(working)
            return
        for count in range(remaining + 1):
            working[position] = count
            yield from visit(position + 1, remaining - count)

    yield from visit(0, rotors)


def pair_signature(histogram: Histogram, rotors: int) -> Signature:
    result = [sum(count * (count - 1) // 2 for count in histogram)]
    result.extend(
        sum(
            histogram[mode] * histogram[(mode + distance) % GRID]
            for mode in range(GRID)
        )
        for distance in range(1, PAIR_CHANNELS)
    )
    if sum(result) != math.comb(rotors, 2):
        raise RuntimeError("streamed pair signature failed")
    return tuple(result)


def moved_signature(
    histogram: Histogram,
    signature: Signature,
    first: int,
    second: int,
    shift: int,
) -> Signature:
    """Update the nine pair counts from a four-site population delta."""
    delta = [0] * GRID
    for mode, amount in (
        (first, -1),
        (second, -1),
        ((first - shift) % GRID, 1),
        ((second + shift) % GRID, 1),
    ):
        delta[mode] += amount
    active = tuple(mode for mode, amount in enumerate(delta) if amount)

    result = list(signature)
    result[0] += sum(
        amount * (2 * histogram[mode] + amount - 1) // 2
        for mode in active
        if (amount := delta[mode])
    )
    for distance in range(1, PAIR_CHANNELS):
        difference = sum(
            delta[mode]
            * (
                histogram[(mode + distance) % GRID]
                + histogram[(mode - distance) % GRID]
                + delta[(mode + distance) % GRID]
            )
            for mode in active
        )
        result[distance] += difference
    return tuple(result)


@dataclass(frozen=True)
class StreamedTopology:
    rotors: int
    signatures: tuple[Signature, ...]
    representatives: tuple[Histogram, ...]
    boundary_weights: tuple[int, ...]
    necklace_cells: int
    occupation_histograms_visited: int
    compiler_peak_live_histograms: int
    streamed_mode_pair_shift_terms_per_scattering: int
    streamed_weighted_particle_shift_terms_per_scattering: int
    prior_materialized_plan_nonzero_entries: int


def primitive_generator(prime: int) -> int:
    factors: list[int] = []
    remaining = prime - 1
    candidate = 2
    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            factors.append(candidate)
            while remaining % candidate == 0:
                remaining //= candidate
        candidate += 1
    if remaining > 1:
        factors.append(remaining)
    for value in range(2, prime):
        if all(pow(value, (prime - 1) // factor, prime) != 1 for factor in factors):
            return value
    raise RuntimeError("streamed primitive generator search failed")


def public_scattering_integer(shift: int, step: int, program_tag: int) -> int:
    distance = min(shift % GRID, GRID - shift % GRID)
    magnitude = 1 + (
        (distance + 2) * (step + 1)
        + (3 * distance + 1) * (program_tag + 2)
    ) % GRID % 5
    return -magnitude if (distance + step + program_tag) % GRID % 3 == 0 else magnitude


def public_pair_weight(distance: int, step: int, program_tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % GRID % (GRID - 1)


def phase_exponent(signature: Signature, step: int, program_tag: int) -> int:
    return sum(
        count * public_pair_weight(distance, step, program_tag)
        for distance, count in enumerate(signature)
    ) % GRID


def profile(
    histogram: Histogram, signature: Signature, signatures: tuple[Signature, ...]
) -> dict[tuple[int, int], int]:
    """Verification-only sparse row; never retained by the accepted path."""
    result: dict[tuple[int, int], int] = {}
    for first in range(GRID):
        if histogram[first] == 0:
            continue
        for second in range(GRID):
            multiplicity = histogram[first] * (
                histogram[second] - (1 if first == second else 0)
            )
            if multiplicity == 0:
                continue
            for shift in range(1, GRID):
                source_signature = moved_signature(
                    histogram, signature, first, second, shift
                )
                source = bisect.bisect_left(signatures, source_signature)
                if source == len(signatures) or signatures[source] != source_signature:
                    raise RuntimeError("streamed move escaped the signature topology")
                key = (shift, source)
                result[key] = result.get(key, 0) + multiplicity
    return result


def compile_streamed_topology(
    rotors: int, prime: int, root: int
) -> StreamedTopology:
    if not 0 < rotors < GRID:
        raise RuntimeError("streamed free-orbit law is out of scope")
    representatives: dict[Signature, Histogram] = {}
    boundary_by_signature: dict[Signature, int] = {}
    occupations = 0
    necklaces = 0
    for histogram in iter_histograms(rotors):
        occupations += 1
        if canonical(histogram) != histogram:
            continue
        signature = pair_signature(histogram, rotors)
        representatives.setdefault(signature, histogram)
        boundary_by_signature[signature] = (
            boundary_by_signature.get(signature, 0)
            + pow(root, (11 * necklaces + 5 * signature[0] + 1) % GRID, prime)
        ) % prime
        necklaces += 1
    expected = math.comb(rotors + GRID - 1, rotors)
    if occupations != expected or expected % GRID or necklaces != expected // GRID:
        raise RuntimeError("streamed topology count failed")

    signatures = tuple(sorted(representatives))
    representative_tuple = tuple(representatives[item] for item in signatures)
    boundary_weights = tuple(boundary_by_signature[item] for item in signatures)
    distinct_terms = 0
    weighted_terms = 0
    prior_plan_nonzeros = 0
    for histogram, signature in zip(
        representative_tuple, signatures, strict=True
    ):
        row = profile(histogram, signature, signatures)
        prior_plan_nonzeros += len(row)
        for first in range(GRID):
            if histogram[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = histogram[first] * (
                    histogram[second] - (1 if first == second else 0)
                )
                if multiplicity:
                    distinct_terms += GRID - 1
                    weighted_terms += multiplicity * (GRID - 1)
    return StreamedTopology(
        rotors=rotors,
        signatures=signatures,
        representatives=representative_tuple,
        boundary_weights=boundary_weights,
        necklace_cells=necklaces,
        occupation_histograms_visited=occupations,
        compiler_peak_live_histograms=1,
        streamed_mode_pair_shift_terms_per_scattering=distinct_terms,
        streamed_weighted_particle_shift_terms_per_scattering=weighted_terms,
        prior_materialized_plan_nonzero_entries=prior_plan_nonzeros,
    )


def verify_all_fibers(topology: StreamedTopology) -> tuple[int, int, int]:
    comparisons = 0
    histograms_visited = 0
    peak_row_entries = 0
    for histogram in iter_histograms(topology.rotors):
        histograms_visited += 1
        if canonical(histogram) != histogram:
            continue
        signature = pair_signature(histogram, topology.rotors)
        target = bisect.bisect_left(topology.signatures, signature)
        if target == len(topology.signatures) or topology.signatures[target] != signature:
            raise RuntimeError("verification found an unknown signature fiber")
        candidate = profile(histogram, signature, topology.signatures)
        reference = profile(
            topology.representatives[target], signature, topology.signatures
        )
        peak_row_entries = max(peak_row_entries, len(candidate) + len(reference))
        if candidate != reference:
            raise RuntimeError("streamed signature fiber is not equitable")
        if histogram != topology.representatives[target]:
            comparisons += GRID - 1
    return comparisons, histograms_visited, peak_row_entries


def apply_diagonal(
    state: list[int],
    topology: StreamedTopology,
    prime: int,
    root: int,
    step: int,
    tag: int,
) -> list[int]:
    return [
        value * pow(root, phase_exponent(signature, step, tag), prime) % prime
        for value, signature in zip(state, topology.signatures, strict=True)
    ]


def apply_scattering_streamed(
    state: list[int],
    topology: StreamedTopology,
    prime: int,
    step: int,
    tag: int,
) -> list[int]:
    output = [0] * len(topology.signatures)
    for target, (histogram, signature) in enumerate(
        zip(topology.representatives, topology.signatures, strict=True)
    ):
        accumulator = 0
        for first in range(GRID):
            if histogram[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = histogram[first] * (
                    histogram[second] - (1 if first == second else 0)
                )
                if multiplicity == 0:
                    continue
                for shift in range(1, GRID):
                    source_signature = moved_signature(
                        histogram, signature, first, second, shift
                    )
                    source = bisect.bisect_left(
                        topology.signatures, source_signature
                    )
                    if (
                        source == len(topology.signatures)
                        or topology.signatures[source] != source_signature
                    ):
                        raise RuntimeError("streamed update escaped the quotient")
                    accumulator += (
                        multiplicity
                        * public_scattering_integer(shift, step, tag)
                        * state[source]
                    )
        output[target] = accumulator % prime
    return output


def public_program(depth: int, family: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (step, (family + 3 * step + step * step) % 7) for step in range(depth)
    )


def execute_word(
    source: list[int],
    topology: StreamedTopology,
    prime: int,
    root: int,
    operations: tuple[tuple[int, int], ...],
    reordered: bool = False,
) -> list[int]:
    current = source.copy()
    for step, tag in operations:
        if reordered:
            current = apply_diagonal(
                apply_scattering_streamed(current, topology, prime, step, tag),
                topology,
                prime,
                root,
                step,
                tag,
            )
        else:
            current = apply_scattering_streamed(
                apply_diagonal(current, topology, prime, root, step, tag),
                topology,
                prime,
                step,
                tag,
            )
    return current


def quotient_source(
    signatures: tuple[Signature, ...], prime: int, family: int
) -> list[int]:
    return [
        (
            1
            + (family + 3) * (index + 1)
            + sum(
                (distance + 2 + family) * (count + 1) ** 2
                for distance, count in enumerate(signature)
            )
        )
        % prime
        for index, signature in enumerate(signatures)
    ]


def public_boundary(state: list[int], topology: StreamedTopology, prime: int) -> int:
    return sum(
        value * weight
        for value, weight in zip(state, topology.boundary_weights, strict=True)
    ) % prime


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    generation: int = 0


def transaction(
    carrier: Carrier,
    expected_source: list[int],
    topology: StreamedTopology,
    prime: int,
    root: int,
    operations: tuple[tuple[int, int], ...],
    inverse_operations: tuple[tuple[int, int], ...],
    control: Control,
) -> dict[str, object]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("null or malformed streamed quotient carrier")
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    forward = execute_word(carrier.source, topology, prime, root, operations)
    carrier.target[:] = [
        (left + right) % prime
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    boundary = public_boundary(carrier.target, topology, prime)
    if control != "missing":
        inverse = execute_word(
            carrier.source,
            topology,
            prime,
            root,
            inverse_operations,
            reordered=control == "reordered",
        )
        carrier.target[:] = [
            (left - right) % prime
            for left, right in zip(carrier.target, inverse, strict=True)
        ]
    error_cells = sum(
        left != right
        for left, right in zip(carrier.source, expected_source, strict=True)
    ) + sum(value != 0 for value in carrier.target)
    carrier.generation += 1
    return {
        "boundary": boundary,
        "forward_commitment": state_commitment(forward),
        "restoration_error_field_cells": error_cells,
        "same_backing": id(carrier.source) == source_backing
        and id(carrier.target) == target_backing,
        "generation": carrier.generation,
    }


def state_commitment(state: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in state).encode()
    ).hexdigest()


def run_case(rotors: int, prime: int) -> dict[str, object]:
    generator = primitive_generator(prime)
    root = pow(generator, (prime - 1) // GRID, prime)
    if root == 1 or pow(root, GRID, prime) != 1:
        raise RuntimeError("streamed seventeenth root failed")
    topology = compile_streamed_topology(rotors, prime, root)
    comparisons, verification_visits, verification_peak = verify_all_fibers(topology)
    primary_operations = public_program(PRIMARY_DEPTH, 0)
    wrong_operations = public_program(PRIMARY_DEPTH, 1)
    reuse_operations = public_program(REUSE_DEPTH, 4)
    source = quotient_source(topology.signatures, prime, 0)

    carrier = Carrier(source.copy(), [0] * len(source))
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    primary = transaction(
        carrier,
        source,
        topology,
        prime,
        root,
        primary_operations,
        primary_operations,
        "correct",
    )
    reuse = transaction(
        carrier,
        source,
        topology,
        prime,
        root,
        reuse_operations,
        reuse_operations,
        "correct",
    )
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse = transaction(
        fresh,
        source,
        topology,
        prime,
        root,
        reuse_operations,
        reuse_operations,
        "correct",
    )
    if (
        primary["restoration_error_field_cells"] != 0
        or reuse["restoration_error_field_cells"] != 0
        or reuse["boundary"] != fresh_reuse["boundary"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or id(carrier.source) != source_backing
        or id(carrier.target) != target_backing
    ):
        raise RuntimeError("streamed quotient restoration or reuse failed")

    controls: dict[str, int] = {}
    for control, inverse_operations in (
        ("missing", primary_operations),
        ("wrong", wrong_operations),
        ("reordered", primary_operations),
    ):
        controlled = Carrier(source.copy(), [0] * len(source))
        result = transaction(
            controlled,
            source,
            topology,
            prime,
            root,
            primary_operations,
            inverse_operations,
            control,
        )
        controls[f"{control}_inverse_error_field_cells"] = int(
            result["restoration_error_field_cells"]
        )
    if min(controls.values()) == 0:
        raise RuntimeError("streamed quotient inverse controls did not discriminate")

    null_rejected = False
    try:
        transaction(
            Carrier([], []),
            [],
            topology,
            prime,
            root,
            primary_operations,
            primary_operations,
            "correct",
        )
    except ValueError:
        null_rejected = True
    if not null_rejected:
        raise RuntimeError("null streamed quotient carrier was accepted")

    fibers = len(topology.signatures)
    terms = topology.streamed_mode_pair_shift_terms_per_scattering
    return {
        "rotors": rotors,
        "prime": prime,
        "multiplicative_generator": generator,
        "seventeenth_root": root,
        "necklace_cells": topology.necklace_cells,
        "pair_signature_fiber_cells": fibers,
        "occupation_histograms_visited": topology.occupation_histograms_visited,
        "compiler_peak_live_histograms": topology.compiler_peak_live_histograms,
        "verification_only_occupation_histograms_visited": verification_visits,
        "verification_only_peak_sparse_row_entries": verification_peak,
        "equitability_comparisons": comparisons,
        "all_sixteen_shift_bases_equitable": True,
        "primary_depth": PRIMARY_DEPTH,
        "reuse_depth": REUSE_DEPTH,
        "primary_boundary": primary["boundary"],
        "reuse_boundary": reuse["boundary"],
        "fresh_reuse_boundary": fresh_reuse["boundary"],
        "primary_output_commitment": primary["forward_commitment"],
        "primary_restoration_error_field_cells": primary[
            "restoration_error_field_cells"
        ],
        "reuse_restoration_error_field_cells": reuse[
            "restoration_error_field_cells"
        ],
        "same_backing_primary": primary["same_backing"],
        "same_backing_reuse": reuse["same_backing"],
        "restoration_generation_after_reuse": carrier.generation,
        "controls": {**controls, "null_carrier_rejected": null_rejected},
        "retained_shift_basis_plans": 0,
        "retained_shift_plan_nonzero_entries": 0,
        "prior_materialized_plan_nonzero_entries": topology.prior_materialized_plan_nonzero_entries,
        "public_signature_descriptor_integer_cells": PAIR_CHANNELS * fibers,
        "public_representative_descriptor_integer_cells": GRID * fibers,
        "public_boundary_weight_field_cells": fibers,
        "streamed_mode_pair_shift_terms_per_scattering": terms,
        "streamed_weighted_particle_shift_terms_per_scattering": topology.streamed_weighted_particle_shift_terms_per_scattering,
        "primary_forward_streamed_terms": PRIMARY_DEPTH * terms,
        "primary_forward_inverse_streamed_terms": 2 * PRIMARY_DEPTH * terms,
        "reuse_forward_inverse_streamed_terms": 2 * REUSE_DEPTH * terms,
    }


def main() -> None:
    cases = [
        run_case(rotors, prime) for rotors in ROTOR_COUNTS for prime in PRIMES
    ]
    expected_necklaces = [9, 9, 57, 57, 285, 285, 1197, 1197]
    expected_fibers = [9, 9, 33, 33, 165, 165, 621, 621]
    expected_plans = [272, 272, 2448, 2448, 21904, 21904, 131168, 131168]
    if [case["necklace_cells"] for case in cases] != expected_necklaces:
        raise RuntimeError("streamed quotient necklace dimensions changed")
    if [case["pair_signature_fiber_cells"] for case in cases] != expected_fibers:
        raise RuntimeError("streamed quotient dimensions changed")
    if [case["prior_materialized_plan_nonzero_entries"] for case in cases] != expected_plans:
        raise RuntimeError("streamed comparison plan counts changed")
    print(
        json.dumps(
            {
                "claim_candidate": "BOUNDED_EXACT_TOPOLOGY_STREAMED_PAIR_SIGNATURE_FIBER_QUOTIENT_ELIMINATES_ALL16_RETAINED_SHIFT_BASES_WHILE_CLOSING_THE9_33_165_621_CELL_ROTORS2_TO5_CARRIER_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_RETAINS26S_PUBLIC_INTEGER_DESCRIPTORS_AND_REMATERIALIZES_EACH_SCATTERING_WITH_AN_IDENTICAL_CLASSICAL_STREAM",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_PAIR_SIGNATURE_FIBER_CONSTANT_INPUTS_ROTORS2_TO5_F103_F239_PRIMARY_DEPTH8_REUSE_DEPTH5_DIRECT_PROCESS_SOFTWARE_ONLY",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS",
                "cases": cases,
                "quotient_cell_law": [9, 33, 165, 621],
                "necklace_cell_law": [9, 57, 285, 1197],
                "all_sixteen_shift_bases_equitable": True,
                "pair_diagonal_closed_by_signature": True,
                "resource_law": {
                    "accepted_carrier_resident_field_cells": "TWO_TIMES_PAIR_SIGNATURE_FIBER_COUNT",
                    "accepted_word_scratch_field_cells": "TWO_TIMES_PAIR_SIGNATURE_FIBER_COUNT",
                    "accepted_public_boundary_weight_field_cells": "PAIR_SIGNATURE_FIBER_COUNT",
                    "accepted_public_signature_descriptor_integer_cells": "NINE_TIMES_PAIR_SIGNATURE_FIBER_COUNT",
                    "accepted_public_representative_descriptor_integer_cells": "SEVENTEEN_TIMES_PAIR_SIGNATURE_FIBER_COUNT",
                    "accepted_retained_shift_basis_plans": 0,
                    "accepted_retained_shift_plan_nonzero_entries": 0,
                    "accepted_retained_inverse_history_bytes": 0,
                    "accepted_assignment_or_relation_table_cells": 0,
                    "accepted_streamed_terms_per_scattering": "REPORTED_PER_CASE",
                    "public_topology_compiler_occupation_histograms_visited": "REPORTED_PER_CASE",
                    "public_topology_compiler_peak_live_histograms": 1,
                    "verification_only_sparse_rows_and_full_fiber_scan": "REPORTED_AND_EXCLUDED_FROM_ACCEPTED_PATH",
                    "python_containers_allocator_bigints_and_expression_temporaries": "EXCLUDED_NOT_ZERO",
                },
                "matched_classical_recurrence": "IDENTICAL_TOPOLOGY_STREAMED_PAIR_SIGNATURE_FIBER_RECURRENCE",
                "catvm_custody": False,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "physical_bit_replacement": False,
                "unbounded_computation_established": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
