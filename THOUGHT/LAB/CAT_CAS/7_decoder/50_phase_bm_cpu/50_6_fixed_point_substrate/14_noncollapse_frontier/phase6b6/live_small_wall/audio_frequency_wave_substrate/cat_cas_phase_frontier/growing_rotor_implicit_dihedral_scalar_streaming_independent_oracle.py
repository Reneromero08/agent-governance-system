#!/usr/bin/env python3
"""Independent oracle for implicit-dihedral one-cell Rotor-6 streaming.

No production scalar-stream, implicit-topology, or M199 module is imported.
The oracle separately compiles sorted canonical codes, executes the one-cell
stream from code arithmetic, and compares it with independent direct and
factor references. Full tuple topology and plans remain verification-only.
"""

from __future__ import annotations

import json
import math
import hashlib
from bisect import bisect_left
from dataclasses import dataclass

import growing_rotor_open_momentum_factor_independent_oracle as reference


GRID = reference.GRID
ROTORS = reference.ROTORS
PRIME = reference.PRIME
ROOT = reference.ROOT
Histogram = tuple[int, ...]
Outgoing = tuple[tuple[tuple[int, int], ...], ...]
BASE = ROTORS + 1
HIGH_POWER = BASE ** (GRID - 1)
PLACE_VALUES = tuple(BASE ** (GRID - 1 - mode) for mode in range(GRID))


@dataclass
class ScalarPort:
    values: list[int]
    momentum: int | None = None
    generation: int | None = None
    necklace: int | None = None
    live: bool = False

    def lease(self, momentum: int, generation: int, necklace: int) -> None:
        if self.live or self.values != [0]:
            raise RuntimeError("independent scalar port was not released")
        if not 1 <= momentum <= 8:
            raise ValueError("independent scalar port type is invalid")
        if generation <= 0 or not 0 <= necklace < 4389:
            raise ValueError("independent scalar port owner is invalid")
        self.momentum = momentum
        self.generation = generation
        self.necklace = necklace
        self.live = True

    def require(self, momentum: int, generation: int, necklace: int) -> None:
        if (
            not self.live
            or self.momentum != momentum
            or self.generation != generation
            or self.necklace != necklace
        ):
            raise ValueError("independent scalar port type or owner mismatch")

    def project(self) -> int:
        if self.live:
            raise PermissionError("independent live scalar projection rejected")
        return self.values[0]

    def release(self, momentum: int, generation: int, necklace: int) -> None:
        self.require(momentum, generation, necklace)
        self.values[0] = 0
        self.momentum = None
        self.generation = None
        self.necklace = None
        self.live = False


@dataclass(frozen=True)
class CompactTopology:
    necklace_codes: tuple[int, ...]
    bracelet_codes: tuple[int, ...]
    boundary_weights: tuple[int, ...]
    occupation_count: int


def decode(value: int, work: dict[str, int] | None = None) -> Histogram:
    cells = [0] * GRID
    current = value
    for index in range(GRID - 1, -1, -1):
        cells[index] = current % BASE
        current //= BASE
    if work is not None:
        work["histogram_digit_decodes"] += GRID
    return tuple(cells)


def reflect_code(value: int, work: dict[str, int] | None = None) -> int:
    item = decode(value, work)
    reflected = tuple(item[(-mode) % GRID] for mode in range(GRID))
    if work is not None:
        work["histogram_reflection_cells"] += GRID
    return reference.encode(reflected)


def canonical_code(value: int, work: dict[str, int] | None = None) -> int:
    current = value
    least = value
    for _ in range(1, GRID):
        current = current % BASE * HIGH_POWER + current // BASE
        least = min(least, current)
    if work is not None:
        work["cyclic_code_candidates"] += GRID
        work["cyclic_code_rolling_updates"] += GRID - 1
    return least


def sorted_index(
    codes: tuple[int, ...],
    value: int,
    work: dict[str, int] | None = None,
    *,
    required: bool = True,
) -> int | None:
    index = bisect_left(codes, value)
    if work is not None:
        work["sorted_code_searches"] += 1
        work["sorted_code_comparison_upper_bound"] += len(codes).bit_length()
    if index < len(codes) and codes[index] == value:
        return index
    if required:
        raise KeyError("independent compact code lookup failed")
    return None


def compact_bracelet_index(
    code: int,
    topology: CompactTopology,
    work: dict[str, int] | None = None,
) -> int:
    reflected = canonical_code(reflect_code(code, work), work)
    index = sorted_index(topology.bracelet_codes, min(code, reflected), work)
    if index is None:
        raise RuntimeError("independent compact bracelet lookup failed")
    return index


def compile_compact_topology() -> tuple[CompactTopology, dict[str, int]]:
    occupations = 0
    necklaces: list[int] = []
    for item in reference.histograms(ROTORS):
        occupations += 1
        if reference.cyclic(item) == item:
            necklaces.append(reference.encode(item))
    candidates = [
        min(code, canonical_code(reflect_code(code))) for code in necklaces
    ]
    candidates.sort()
    bracelets = tuple(
        code
        for index, code in enumerate(candidates)
        if index == 0 or code != candidates[index - 1]
    )
    compiler_peak = len(necklaces) + len(candidates) + len(bracelets) + GRID
    candidates.clear()
    necklace_codes = tuple(necklaces)
    necklaces.clear()
    provisional = CompactTopology(necklace_codes, bracelets, (), occupations)
    boundary = [0] * len(bracelets)
    for necklace, code in enumerate(necklace_codes):
        collision = reference.pair_signature(decode(code))[0]
        target = compact_bracelet_index(code, provisional)
        boundary[target] = (
            boundary[target]
            + pow(ROOT, (11 * necklace + 5 * collision + 1) % GRID, PRIME)
        ) % PRIME
    compact = CompactTopology(
        necklace_codes, bracelets, tuple(boundary), occupations
    )
    if (
        occupations != 74613
        or len(necklace_codes) != 4389
        or len(bracelets) != 2277
    ):
        raise RuntimeError("independent compact topology changed")
    return compact, {
        "occupation_histograms_streamed": occupations,
        "retained_necklace_code_integers": len(necklace_codes),
        "retained_bracelet_code_integers": len(bracelets),
        "retained_boundary_weight_field_cells": len(boundary),
        "retained_histogram_cells": 0,
        "retained_hash_map_entries": 0,
        "retained_necklace_to_bracelet_entries": 0,
        "retained_reflection_map_entries": 0,
        "compiler_logical_integer_slot_peak": compiler_peak,
        "transient_histogram_working_cells": GRID,
    }


def transpose_rows(
    rows: tuple[tuple[tuple[int, int], ...], ...],
    source_cells: int,
) -> Outgoing:
    outgoing: list[list[tuple[int, int]]] = [
        [] for _ in range(source_cells)
    ]
    for target, row in enumerate(rows):
        for source, coefficient in row:
            outgoing[source].append((target, coefficient))
    return tuple(tuple(row) for row in outgoing)


def planned_stream_scattering(
    state: list[int],
    topology: reference.Topology,
    plans: reference.OneBodyPlans,
    step: int,
    tag: int,
    port: ScalarPort,
    *,
    wrong_reflection: bool = False,
) -> tuple[list[int], dict[str, int]]:
    output = [0] * len(state)
    if port.live or port.values != [0]:
        raise RuntimeError("independent resident scalar port was not clear")
    leases = 0
    releases = 0
    first_terms = 0
    closure_terms = 0
    verification_plan_entries = 0
    for generation, momentum in enumerate(range(1, 9), 1):
        positive = transpose_rows(
            plans.second[momentum - 1], len(topology.necklaces)
        )
        negative = transpose_rows(
            plans.second[GRID - momentum - 1], len(topology.necklaces)
        )
        verification_plan_entries += sum(map(len, positive)) + sum(
            map(len, negative)
        )
        weight = reference.scattering_weight(momentum, step, tag)
        reflected_weight = reference.scattering_weight(
            GRID - momentum, step, tag
        )
        if weight != reflected_weight:
            raise RuntimeError("independent reflection weight pairing failed")
        for necklace, row in enumerate(plans.first[momentum - 1]):
            port.lease(momentum, generation, necklace)
            leases += 1
            port.values[0] = sum(
                coefficient
                * state[topology.necklace_to_bracelet[source]]
                for source, coefficient in row
            ) % PRIME
            first_terms += len(row)
            port.require(momentum, generation, necklace)
            for target, coefficient in positive[necklace]:
                output[target] = (
                    output[target] + weight * coefficient * port.values[0]
                ) % PRIME
                closure_terms += 1
            reflected = (
                necklace
                if wrong_reflection
                else topology.reflected_necklace[necklace]
            )
            for target, coefficient in negative[reflected]:
                output[target] = (
                    output[target] + weight * coefficient * port.values[0]
                ) % PRIME
                closure_terms += 1
            port.release(momentum, generation, necklace)
            releases += 1
        for target, value in enumerate(state):
            output[target] = (
                output[target] - 2 * weight * ROTORS * value
            ) % PRIME
    if port.live or port.values != [0]:
        raise RuntimeError("independent scalar port failed to restore")
    return output, {
        "scalar_port_leases": leases,
        "scalar_port_releases": releases,
        "scalar_port_clear_field_cells": releases,
        "first_pass_one_body_terms": first_terms,
        "closure_one_body_terms": closure_terms,
        "verification_only_transposed_plan_entries_visited": (
            verification_plan_entries
        ),
    }


def diagonal(
    state: list[int], topology: reference.Topology, step: int, tag: int
) -> list[int]:
    return reference.diagonal(state, topology, step, tag)


def execute_planned(
    source: list[int],
    topology: reference.Topology,
    plans: reference.OneBodyPlans,
    operations: tuple[tuple[int, int], ...],
    port: ScalarPort,
    *,
    reordered: bool = False,
    wrong_reflection: bool = False,
) -> tuple[list[int], dict[str, int]]:
    current = source.copy()
    aggregate: dict[str, int] = {}
    for step, tag in operations:
        if reordered:
            current, work = planned_stream_scattering(
                current,
                topology,
                plans,
                step,
                tag,
                port,
                wrong_reflection=wrong_reflection,
            )
            current = diagonal(current, topology, step, tag)
        else:
            current = diagonal(current, topology, step, tag)
            current, work = planned_stream_scattering(
                current,
                topology,
                plans,
                step,
                tag,
                port,
                wrong_reflection=wrong_reflection,
            )
        for key, value in work.items():
            aggregate[key] = aggregate.get(key, 0) + value
    return current, aggregate


def fresh_work() -> dict[str, int]:
    return {
        "scalar_port_leases": 0,
        "scalar_port_releases": 0,
        "scalar_port_clear_field_cells": 0,
        "first_pass_one_body_terms": 0,
        "closure_one_body_terms": 0,
        "source_orbit_rotations": 0,
        "inverse_candidate_moves": 0,
        "exact_bracelet_lookup_attempts": 0,
        "exact_bracelet_lookup_hits": 0,
        "encoded_move_deltas": 0,
        "cyclic_code_candidates": 0,
        "cyclic_code_rolling_updates": 0,
        "histogram_digit_decodes": 0,
        "histogram_reflection_cells": 0,
        "sorted_code_searches": 0,
        "sorted_code_comparison_upper_bound": 0,
        "diagonal_pair_signature_mode_terms": 0,
    }


def merge_work(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] += value


def moved_bracelet(
    code: int,
    mode: int,
    destination: int,
    topology: CompactTopology,
    work: dict[str, int],
) -> int:
    moved = code - PLACE_VALUES[mode] + PLACE_VALUES[destination]
    canonical = canonical_code(moved, work)
    work["encoded_move_deltas"] += 1
    return compact_bracelet_index(canonical, topology, work)


def scalar_coordinate(
    state: list[int],
    item: Histogram,
    code: int,
    momentum: int,
    topology: CompactTopology,
    work: dict[str, int],
) -> int:
    value = 0
    for mode, count in enumerate(item):
        if count:
            source = moved_bracelet(
                code,
                mode,
                (mode - momentum) % GRID,
                topology,
                work,
            )
            value += count * state[source]
            work["first_pass_one_body_terms"] += 1
    return value % PRIME


def rotate_once(item: Histogram) -> Histogram:
    return (item[-1],) + item[:-1]


def scatter_orbit(
    scalar: int,
    item: Histogram,
    momentum: int,
    sign: int,
    weight: int,
    output: list[int],
    topology: CompactTopology,
    work: dict[str, int],
) -> None:
    rotated = item
    code = reference.encode(item)
    for _ in range(GRID):
        work["source_orbit_rotations"] += 1
        for occupied, count in enumerate(rotated):
            if not count:
                continue
            target_mode = (occupied - sign * momentum) % GRID
            target_code = (
                code - PLACE_VALUES[occupied] + PLACE_VALUES[target_mode]
            )
            work["inverse_candidate_moves"] += 1
            work["exact_bracelet_lookup_attempts"] += 1
            target = sorted_index(
                topology.bracelet_codes,
                target_code,
                work,
                required=False,
            )
            if target is not None:
                output[target] = (
                    output[target]
                    + weight * (rotated[target_mode] + 1) * scalar
                ) % PRIME
                work["closure_one_body_terms"] += 1
                work["exact_bracelet_lookup_hits"] += 1
        rotated = rotate_once(rotated)
        code = code % BASE * HIGH_POWER + code // BASE


def compact_scattering(
    state: list[int],
    topology: CompactTopology,
    step: int,
    tag: int,
    port: ScalarPort,
    *,
    wrong_reflection: bool = False,
) -> tuple[list[int], dict[str, int]]:
    output = [0] * len(state)
    work = fresh_work()
    for generation, momentum in enumerate(range(1, 9), 1):
        weight = reference.scattering_weight(momentum, step, tag)
        if weight != reference.scattering_weight(GRID - momentum, step, tag):
            raise RuntimeError("independent paired weight changed")
        for necklace, code in enumerate(topology.necklace_codes):
            item = decode(code, work)
            port.lease(momentum, generation, necklace)
            work["scalar_port_leases"] += 1
            port.values[0] = scalar_coordinate(
                state, item, code, momentum, topology, work
            )
            port.require(momentum, generation, necklace)
            scatter_orbit(
                port.values[0],
                item,
                momentum,
                1,
                weight,
                output,
                topology,
                work,
            )
            reflected = item if wrong_reflection else reference.reflect(item)
            if not wrong_reflection:
                work["histogram_reflection_cells"] += GRID
            scatter_orbit(
                port.values[0],
                reflected,
                momentum,
                -1,
                weight,
                output,
                topology,
                work,
            )
            port.release(momentum, generation, necklace)
            work["scalar_port_releases"] += 1
            work["scalar_port_clear_field_cells"] += 1
        for target, value in enumerate(state):
            output[target] = (
                output[target] - 2 * weight * ROTORS * value
            ) % PRIME
    return output, work


def compact_diagonal(
    state: list[int],
    topology: CompactTopology,
    step: int,
    tag: int,
) -> tuple[list[int], dict[str, int]]:
    work = fresh_work()
    output: list[int] = []
    for value, code in zip(state, topology.bracelet_codes, strict=True):
        item = decode(code, work)
        output.append(
            value * pow(ROOT, reference.phase_exponent(item, step, tag), PRIME)
            % PRIME
        )
        work["diagonal_pair_signature_mode_terms"] += (
            reference.PAIR_CHANNELS * GRID
        )
    return output, work


def execute(
    source: list[int],
    topology: CompactTopology,
    operations: tuple[tuple[int, int], ...],
    port: ScalarPort,
    *,
    reordered: bool = False,
    wrong_reflection: bool = False,
) -> tuple[list[int], dict[str, int]]:
    current = source.copy()
    total = fresh_work()
    for step, tag in operations:
        if reordered:
            current, scatter = compact_scattering(
                current,
                topology,
                step,
                tag,
                port,
                wrong_reflection=wrong_reflection,
            )
            current, diagonal_work = compact_diagonal(
                current, topology, step, tag
            )
        else:
            current, diagonal_work = compact_diagonal(
                current, topology, step, tag
            )
            current, scatter = compact_scattering(
                current,
                topology,
                step,
                tag,
                port,
                wrong_reflection=wrong_reflection,
            )
        merge_work(total, diagonal_work)
        merge_work(total, scatter)
    return current, total


def transaction(
    carrier: tuple[list[int], list[int], ScalarPort],
    source: list[int],
    topology: CompactTopology,
    operations: tuple[tuple[int, int], ...],
) -> tuple[int, int, bool, str]:
    source_backing = id(carrier[0])
    target_backing = id(carrier[1])
    port_backing = id(carrier[2].values)
    forward, _ = execute(source, topology, operations, carrier[2])
    carrier[1][:] = [
        (left + right) % PRIME
        for left, right in zip(carrier[1], forward, strict=True)
    ]
    forward_commitment = hashlib.sha256(
        ",".join(map(str, forward)).encode()
    ).hexdigest()
    projected = sum(
        value * weight
        for value, weight in zip(
            carrier[1], topology.boundary_weights, strict=True
        )
    ) % PRIME
    forward.clear()
    rematerialized, _ = execute(source, topology, operations, carrier[2])
    carrier[1][:] = [
        (left - right) % PRIME
        for left, right in zip(carrier[1], rematerialized, strict=True)
    ]
    rematerialized.clear()
    error = sum(
        left != right for left, right in zip(carrier[0], source, strict=True)
    ) + sum(value != 0 for value in carrier[1])
    return (
        projected,
        error,
        id(carrier[0]) == source_backing
        and id(carrier[1]) == target_backing
        and id(carrier[2].values) == port_backing
        and not carrier[2].live
        and carrier[2].values == [0],
        forward_commitment,
    )


def mismatch(left: list[int], right: list[int]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def main() -> None:
    reference_topology = reference.compile_topology()
    plans = reference.compile_one_body_plans(reference_topology)
    topology, compiler_work = compile_compact_topology()
    source = reference.source_state(reference_topology, 0)
    primary_word = reference.public_program(1, 0)
    reuse_word = reference.public_program(1, 4)
    wrong_word = reference.public_program(1, 1)
    direct_operator = reference.compile_direct_operator(
        reference_topology, *primary_word[0]
    )

    primary, primary_work = execute(
        source, topology, primary_word, ScalarPort([0])
    )
    reuse, reuse_work = execute(
        source, topology, reuse_word, ScalarPort([0])
    )
    direct = reference.execute_direct(
        source, reference_topology, direct_operator, *primary_word[0]
    )
    factor = reference.execute_factor(
        source, reference_topology, plans, primary_word
    )
    carrier = (source.copy(), [0] * len(source), ScalarPort([0]))
    primary_boundary, primary_error, primary_backing, primary_forward = (
        transaction(carrier, source, topology, primary_word)
    )
    reuse_boundary, reuse_error, reuse_backing, reuse_forward = transaction(
        carrier, source, topology, reuse_word
    )
    fresh = (source.copy(), [0] * len(source), ScalarPort([0]))
    fresh_boundary, fresh_error, fresh_backing, fresh_forward = transaction(
        fresh, source, topology, reuse_word
    )
    wrong, _ = execute(
        source, topology, wrong_word, ScalarPort([0])
    )
    reordered, _ = execute(
        source,
        topology,
        primary_word,
        ScalarPort([0]),
        reordered=True,
    )
    wrong_reflection, _ = execute(
        source,
        topology,
        primary_word,
        ScalarPort([0]),
        wrong_reflection=True,
    )

    typed = ScalarPort([0])
    typed.lease(1, 7, 0)
    wrong_type_rejected = False
    wrong_owner_rejected = False
    premature_projection_rejected = False
    try:
        typed.require(2, 7, 0)
    except ValueError:
        wrong_type_rejected = True
    try:
        typed.require(1, 8, 0)
    except ValueError:
        wrong_owner_rejected = True
    try:
        typed.project()
    except PermissionError:
        premature_projection_rejected = True
    typed.release(1, 7, 0)

    occupation_count = math.comb(ROTORS + GRID - 1, ROTORS)
    nonzero_mode_incidence = GRID * math.comb(
        ROTORS + GRID - 2, ROTORS - 1
    )
    orbit_rotations = 2 * 8 * GRID * len(topology.necklace_codes)
    inverse_candidates = 2 * 8 * nonzero_mode_incidence
    retained_topology_descriptors = (
        len(topology.necklace_codes)
        + len(topology.bracelet_codes)
        + len(topology.boundary_weights)
    )
    active_numeric_cells = 3 * len(topology.bracelet_codes) + 1
    named_slots = active_numeric_cells + retained_topology_descriptors
    necklace_bits = sum(
        max(1, code.bit_length()) for code in topology.necklace_codes
    )
    bracelet_bits = sum(
        max(1, code.bit_length()) for code in topology.bracelet_codes
    )
    source_signature_coordinates = len(
        reference.refined_signature(decode(topology.bracelet_codes[0]))
    )
    source_compiler_peak = len(topology.bracelet_codes) * (
        source_signature_coordinates + 3
    )

    if (
        occupation_count != 74613
        or topology.necklace_codes
        != tuple(map(reference.encode, reference_topology.necklaces))
        or topology.bracelet_codes
        != tuple(map(reference.encode, reference_topology.bracelets))
        or topology.boundary_weights != reference_topology.boundary_weights
        or mismatch(primary, direct)
        or mismatch(primary, factor)
        or primary_forward
        != hashlib.sha256(",".join(map(str, primary)).encode()).hexdigest()
        or reuse_forward
        != hashlib.sha256(",".join(map(str, reuse)).encode()).hexdigest()
        or fresh_forward
        != hashlib.sha256(",".join(map(str, reuse)).encode()).hexdigest()
        or primary_boundary != 83
        or reuse_boundary != 70
        or fresh_boundary != 70
        or reference.signature_commitment(primary, reference_topology)
        != "834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e"
        or any((primary_error, reuse_error, fresh_error))
        or not all((primary_backing, reuse_backing, fresh_backing))
        or primary_work["scalar_port_leases"] != 35112
        or primary_work["scalar_port_releases"] != 35112
        or primary_work["first_pass_one_body_terms"] != 162792
        or primary_work["closure_one_body_terms"] != 168912
        or primary_work["histogram_digit_decodes"] != 3403077
        or primary_work["histogram_reflection_cells"] != 3364368
        or primary_work["sorted_code_searches"] != 5697720
        or primary_work["sorted_code_comparison_upper_bound"] != 68372640
        or orbit_rotations != 1193808
        or inverse_candidates != 5534928
        or mismatch(primary, wrong) == 0
        or mismatch(primary, reordered) == 0
        or mismatch(primary, wrong_reflection) == 0
        or not wrong_type_rejected
        or not wrong_owner_rejected
        or not premature_projection_rejected
        or typed.live
        or typed.values != [0]
    ):
        raise RuntimeError("independent scalar momentum streaming failed")

    print(
        json.dumps(
            {
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS_IMPLICIT_TOPOLOGY_STATE_REDUCTION_WITH_SEARCH_WORK_TRADEOFF",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_REUSE_DIRECT_PROCESS_IMPLICIT_DIHEDRAL_CODE_ONE_CELL_STREAM_ONLY",
                "independent_state": {
                    "occupation_histograms": occupation_count,
                    "necklace_cells": len(topology.necklace_codes),
                    "bracelet_cells": len(topology.bracelet_codes),
                    "direct_two_body_raw_terms": direct_operator.raw_terms,
                    "direct_two_body_csr_nonzeros": int(
                        direct_operator.matrix.nnz
                    ),
                    "streamed_direct_mismatch_cells": mismatch(primary, direct),
                    "streamed_factor_mismatch_cells": mismatch(primary, factor),
                    "primary_boundary": primary_boundary,
                    "reuse_boundary": reuse_boundary,
                    "fresh_reuse_boundary": fresh_boundary,
                    "primary_signature_order_commitment": reference.signature_commitment(
                        primary, reference_topology
                    ),
                    "topology_commitment": reference.topology_commitment(
                        reference_topology
                    ),
                    "compact_codes_and_boundary_match_tuple_reference": True,
                },
                "scalar_port_verification": {
                    "logical_field_cells": 1,
                    "maximum_simultaneously_live": 1,
                    "projected": False,
                    "primary_work": primary_work,
                    "reuse_work": reuse_work,
                    "verification_only_reference_topology_and_plans_retained": True,
                    "verification_only_plan_entries": plans.first_entries
                    + plans.second_entries,
                    "accepted_path_retained_transition_plan_entries": 0,
                    "wrong_type_rejected": wrong_type_rejected,
                    "wrong_owner_rejected": wrong_owner_rejected,
                    "premature_projection_rejected": (
                        premature_projection_rejected
                    ),
                    "control_port_restored": (
                        not typed.live and typed.values == [0]
                    ),
                },
                "transaction": {
                    "primary_restoration_error_field_cells": primary_error,
                    "reuse_restoration_error_field_cells": reuse_error,
                    "fresh_reuse_restoration_error_field_cells": fresh_error,
                    "primary_same_backing": primary_backing,
                    "reuse_same_backing": reuse_backing,
                    "fresh_reuse_same_backing": fresh_backing,
                    "fresh_restored_reuse_state_agreement": reuse_forward
                    == fresh_forward,
                    "restoration_generation_after_reuse": 2,
                    "forward_output_released_before_inverse": True,
                    "baseline_reload_used": False,
                },
                "controls": {
                    "missing_inverse_error_field_cells": sum(
                        value != 0 for value in primary
                    ),
                    "wrong_inverse_error_field_cells": mismatch(primary, wrong),
                    "reordered_noncommuting_error_field_cells": mismatch(
                        primary, reordered
                    ),
                    "wrong_reflection_error_field_cells": mismatch(
                        primary, wrong_reflection
                    ),
                },
                "resource_derivation": {
                    "active_numeric_field_cells": active_numeric_cells,
                    "retained_public_topology_descriptor_integers": (
                        retained_topology_descriptors
                    ),
                    "retained_necklace_code_integers": len(
                        topology.necklace_codes
                    ),
                    "retained_bracelet_code_integers": len(
                        topology.bracelet_codes
                    ),
                    "retained_boundary_weight_field_cells": len(
                        topology.boundary_weights
                    ),
                    "retained_histogram_cells": 0,
                    "retained_hash_map_entries": 0,
                    "retained_necklace_to_bracelet_entries": 0,
                    "retained_reflection_map_entries": 0,
                    "named_algorithm_field_and_descriptor_slots": named_slots,
                    "m199_comparable_named_field_and_descriptor_slots": 107943,
                    "m202_named_field_and_descriptor_slots": 105832,
                    "net_named_slot_saving_against_m199": 107943
                    - named_slots,
                    "net_named_slot_saving_against_m202": 105832
                    - named_slots,
                    "maximum_canonical_code_payload_bits": max(
                        code.bit_length() for code in topology.necklace_codes
                    ),
                    "retained_necklace_code_payload_bits": necklace_bits,
                    "retained_bracelet_code_payload_bits": bracelet_bits,
                    "retained_boundary_fixed_width_bits": len(
                        topology.boundary_weights
                    )
                    * 7,
                    "active_numeric_fixed_width_bits": active_numeric_cells * 7,
                    "accepted_fixed_width_logical_payload_bits": (
                        necklace_bits
                        + bracelet_bits
                        + len(topology.boundary_weights) * 7
                        + active_numeric_cells * 7
                    ),
                    "topology_compiler": compiler_work,
                    "public_source_compiler_signature_coordinates": (
                        source_signature_coordinates
                    ),
                    "public_source_compiler_logical_integer_slot_peak": (
                        source_compiler_peak
                    ),
                    "verification_only_signature_order_indices": len(
                        reference_topology.signature_order
                    ),
                    "public_program_descriptor_integers": 2
                    * len(primary_word),
                    "accepted_full_lifecycle_logical_slot_peak": max(
                        named_slots,
                        compiler_work["compiler_logical_integer_slot_peak"],
                        source_compiler_peak,
                    ),
                    "source_orbit_rotations": orbit_rotations,
                    "inverse_candidate_moves": inverse_candidates,
                    "sorted_code_searches": primary_work[
                        "sorted_code_searches"
                    ],
                    "sorted_code_comparison_upper_bound": primary_work[
                        "sorted_code_comparison_upper_bound"
                    ],
                    "histogram_digit_decodes": primary_work[
                        "histogram_digit_decodes"
                    ],
                    "histogram_reflection_cells": primary_work[
                        "histogram_reflection_cells"
                    ],
                    "accepted_one_body_contributions": primary_work[
                        "first_pass_one_body_terms"
                    ]
                    + primary_work["closure_one_body_terms"],
                    "full_occupation_scratch_cells": 0,
                    "dense_operator_cells": 0,
                    "permanent_assignment_terms": 0,
                },
                "production_implicit_dihedral_module_imported": False,
                "production_scalar_stream_module_imported": False,
                "production_m199_module_imported": False,
                "prior_independent_factor_reference_reused": True,
                "matched_classical_baselines": [
                    "IDENTICAL_IMPLICIT_DIHEDRAL_ONE_CELL_FACTOR_STREAM",
                    "M202_ONE_CELL_RETAINED_TOPOLOGY_FACTOR_STREAM",
                    "M199_REFLECTION_PAIRED4389_CELL_PORT_FACTOR_STREAM",
                ],
                "catvm_custody": False,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "physical_bit_replacement": False,
                "unbounded_computation_established": False,
                "terminal": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
