#!/usr/bin/env python3
"""Implicit-dihedral one-cell open-momentum closure for exact Rotor-6.

M202 closes the wide intermediate to one owner-typed scalar but retains the
M199 necklace histograms and lookup maps.  This successor stores only sorted
canonical necklace and bracelet codes plus final boundary weights.  It
rematerializes histograms, reflection, orbit lookup, and fanout from public
dihedral arithmetic while preserving M202's one-cell resident port.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import dataclass

import growing_rotor_open_momentum_factor_closure as predecessor


GRID = predecessor.GRID
ROTORS = predecessor.ROTORS
PRIME = predecessor.PRIME
ROOT = predecessor.ROOT
Histogram = tuple[int, ...]


@dataclass(frozen=True)
class CompactTopology:
    necklace_codes: tuple[int, ...]
    bracelet_codes: tuple[int, ...]
    boundary_weights: tuple[int, ...]
    occupation_histograms: int


def decode_code(value: int, work: "Work | None" = None) -> Histogram:
    digits = [0] * GRID
    current = value
    for index in range(GRID - 1, -1, -1):
        digits[index] = current % predecessor.BASE
        current //= predecessor.BASE
    if current:
        raise ValueError("encoded histogram exceeds fixed grid")
    if work is not None:
        work.histogram_digit_decodes += GRID
    return tuple(digits)


def reflect_code(value: int, work: "Work | None" = None) -> int:
    item = decode_code(value, work)
    reflected = tuple(item[(-mode) % GRID] for mode in range(GRID))
    if work is not None:
        work.histogram_reflection_cells += GRID
    return predecessor.encode(reflected)


def canonical_code(value: int, work: "Work | None" = None) -> int:
    current = value
    result = value
    for _ in range(1, GRID):
        current = (
            current % predecessor.BASE * predecessor.HIGH_POWER
            + current // predecessor.BASE
        )
        result = min(result, current)
    if work is not None:
        work.cyclic_code_candidates += GRID
        work.cyclic_code_rolling_updates += GRID - 1
    return result


def exact_sorted_index(
    codes: tuple[int, ...],
    value: int,
    work: "Work | None" = None,
    *,
    required: bool = True,
) -> int | None:
    index = bisect_left(codes, value)
    if work is not None:
        work.sorted_code_searches += 1
        work.sorted_code_comparison_upper_bound += len(codes).bit_length()
    if index < len(codes) and codes[index] == value:
        return index
    if required:
        raise KeyError("canonical code is absent from compact topology")
    return None


def bracelet_index_from_necklace_code(
    code: int, topology: CompactTopology, work: "Work | None" = None
) -> int:
    reflected = canonical_code(reflect_code(code, work), work)
    bracelet_code = min(code, reflected)
    index = exact_sorted_index(topology.bracelet_codes, bracelet_code, work)
    if index is None:
        raise RuntimeError("required bracelet code lookup failed")
    return index


def compile_topology() -> tuple[CompactTopology, dict[str, int]]:
    necklace_codes: list[int] = []
    occupation_count = 0
    for item in predecessor.iter_histograms(ROTORS):
        occupation_count += 1
        code = predecessor.encode(item)
        if predecessor.canonical_code(code) == code:
            necklace_codes.append(code)
    if necklace_codes != sorted(necklace_codes):
        raise RuntimeError("necklace canonical codes are not sorted")

    bracelet_candidates = [
        min(code, predecessor.canonical_code(reflect_code(code)))
        for code in necklace_codes
    ]
    bracelet_candidates.sort()
    bracelet_codes = tuple(
        code
        for index, code in enumerate(bracelet_candidates)
        if index == 0 or code != bracelet_candidates[index - 1]
    )
    compiler_peak_integer_slots = (
        len(necklace_codes) + len(bracelet_candidates) + len(bracelet_codes)
        + GRID
    )
    bracelet_candidates.clear()

    necklace_tuple = tuple(necklace_codes)
    necklace_codes.clear()
    boundary = [0] * len(bracelet_codes)
    topology = CompactTopology(
        necklace_codes=necklace_tuple,
        bracelet_codes=bracelet_codes,
        boundary_weights=(),
        occupation_histograms=occupation_count,
    )
    for necklace_index, code in enumerate(necklace_tuple):
        item = decode_code(code)
        collision = sum(count * (count - 1) // 2 for count in item)
        bracelet = bracelet_index_from_necklace_code(code, topology)
        boundary[bracelet] = (
            boundary[bracelet]
            + pow(
                ROOT,
                (11 * necklace_index + 5 * collision + 1) % GRID,
                PRIME,
            )
        ) % PRIME
    topology = CompactTopology(
        necklace_codes=necklace_tuple,
        bracelet_codes=bracelet_codes,
        boundary_weights=tuple(boundary),
        occupation_histograms=occupation_count,
    )
    if (
        occupation_count != math.comb(ROTORS + GRID - 1, ROTORS)
        or len(necklace_tuple) != 4389
        or len(bracelet_codes) != 2277
    ):
        raise RuntimeError("implicit dihedral topology law changed")
    return topology, {
        "occupation_histograms_streamed": occupation_count,
        "retained_necklace_code_integers": len(necklace_tuple),
        "retained_bracelet_code_integers": len(bracelet_codes),
        "retained_boundary_weight_field_cells": len(boundary),
        "retained_histogram_cells": 0,
        "retained_hash_map_entries": 0,
        "retained_necklace_to_bracelet_entries": 0,
        "retained_reflection_map_entries": 0,
        "compiler_logical_integer_slot_peak": compiler_peak_integer_slots,
        "transient_histogram_working_cells": GRID,
    }


def source_and_signature_order(
    topology: CompactTopology, family: int
) -> tuple[list[int], tuple[int, ...]]:
    records = sorted(
        (predecessor.refined_signature(decode_code(code)), bracelet)
        for bracelet, code in enumerate(topology.bracelet_codes)
    )
    source = [0] * len(records)
    order: list[int] = []
    for index, (signature, bracelet) in enumerate(records):
        source[bracelet] = (
            1
            + (family + 3) * (index + 1)
            + sum(
                (coordinate + 2 + family) * (count + 1) ** 2
                for coordinate, count in enumerate(signature)
            )
        ) % PRIME
        order.append(bracelet)
    return source, tuple(order)


def topology_commitment(topology: CompactTopology) -> str:
    digest = hashlib.sha256()
    for index, code in enumerate(topology.necklace_codes):
        item = decode_code(code)
        reflected_code = canonical_code(reflect_code(code))
        bracelet = exact_sorted_index(
            topology.bracelet_codes, min(code, reflected_code)
        )
        reflected = exact_sorted_index(topology.necklace_codes, reflected_code)
        digest.update(
            (
                f"{code}:" + ",".join(map(str, item))
                + f":{bracelet}:{reflected};"
            ).encode()
        )
    digest.update(",".join(map(str, topology.boundary_weights)).encode())
    return digest.hexdigest()


@dataclass
class ScalarMomentumPort:
    values: list[int]
    momentum: int | None = None
    generation: int | None = None
    necklace: int | None = None
    live: bool = False

    def lease(self, momentum: int, generation: int, necklace: int) -> None:
        if self.live or self.values != [0]:
            raise RuntimeError("scalar momentum port was not released")
        if not 1 <= momentum <= 8:
            raise ValueError("scalar momentum type is outside paired range")
        if generation <= 0 or not 0 <= necklace < 4389:
            raise ValueError("scalar momentum owner is malformed")
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
            raise ValueError("scalar momentum type or owner mismatch")

    def release(self, momentum: int, generation: int, necklace: int) -> None:
        self.require(momentum, generation, necklace)
        self.values[0] = 0
        self.momentum = None
        self.generation = None
        self.necklace = None
        self.live = False


@dataclass
class Work:
    scatterings: int = 0
    scalar_port_leases: int = 0
    scalar_port_releases: int = 0
    scalar_port_clear_field_cells: int = 0
    first_pass_one_body_terms: int = 0
    closure_one_body_terms: int = 0
    source_orbit_rotations: int = 0
    inverse_candidate_moves: int = 0
    exact_bracelet_lookup_attempts: int = 0
    exact_bracelet_lookup_hits: int = 0
    encoded_move_deltas: int = 0
    cyclic_code_candidates: int = 0
    cyclic_code_rolling_updates: int = 0
    histogram_digit_decodes: int = 0
    histogram_reflection_cells: int = 0
    sorted_code_searches: int = 0
    sorted_code_comparison_upper_bound: int = 0
    diagonal_pair_signature_mode_terms: int = 0

    def add(self, other: "Work") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name, 0))

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def moved_bracelet_index(
    target_code: int,
    mode: int,
    destination: int,
    topology: CompactTopology,
    work: Work,
) -> int:
    moved = (
        target_code
        - predecessor.PLACE_VALUES[mode]
        + predecessor.PLACE_VALUES[destination]
    )
    key = canonical_code(moved, work)
    work.encoded_move_deltas += 1
    return bracelet_index_from_necklace_code(key, topology, work)


def compute_scalar(
    state: list[int],
    topology: CompactTopology,
    item: Histogram,
    code: int,
    momentum: int,
    work: Work,
) -> int:
    accumulator = 0
    for mode, count in enumerate(item):
        if count:
            source = moved_bracelet_index(
                code,
                mode,
                (mode - momentum) % GRID,
                topology,
                work,
            )
            accumulator += count * state[source]
            work.first_pass_one_body_terms += 1
    return accumulator % PRIME


def rotate_once(item: Histogram) -> Histogram:
    return (item[-1],) + item[:-1]


def scatter_orbit(
    scalar: int,
    source: Histogram,
    momentum: int,
    sign: int,
    weight: int,
    output: list[int],
    bracelet_codes: tuple[int, ...],
    work: Work,
) -> None:
    rotated = source
    source_code = predecessor.encode(source)
    for _ in range(GRID):
        work.source_orbit_rotations += 1
        for occupied_mode, count in enumerate(rotated):
            if count == 0:
                continue
            target_mode = (occupied_mode - sign * momentum) % GRID
            target_code = (
                source_code
                - predecessor.PLACE_VALUES[occupied_mode]
                + predecessor.PLACE_VALUES[target_mode]
            )
            work.inverse_candidate_moves += 1
            work.exact_bracelet_lookup_attempts += 1
            target = exact_sorted_index(
                bracelet_codes, target_code, work, required=False
            )
            if target is not None:
                coefficient = rotated[target_mode] + 1
                output[target] = (
                    output[target] + weight * coefficient * scalar
                ) % PRIME
                work.closure_one_body_terms += 1
                work.exact_bracelet_lookup_hits += 1
        rotated = rotate_once(rotated)
        source_code = (
            source_code % predecessor.BASE * predecessor.HIGH_POWER
            + source_code // predecessor.BASE
        )


def apply_scattering_streamed(
    state: list[int],
    topology: CompactTopology,
    step: int,
    tag: int,
    *,
    wrong_reflection: bool = False,
    resident_port: ScalarMomentumPort | None = None,
) -> tuple[list[int], Work, dict[str, object]]:
    if len(state) != len(topology.bracelet_codes):
        raise ValueError("null or malformed bracelet carrier")
    output = [0] * len(state)
    port = resident_port if resident_port is not None else ScalarMomentumPort([0])
    port_backing = id(port.values)
    work = Work(scatterings=1)
    for generation, momentum in enumerate(range(1, 9), 1):
        weight = predecessor.public_law.public_scattering_integer(
            momentum, step, tag
        )
        reflected_weight = predecessor.public_law.public_scattering_integer(
            GRID - momentum, step, tag
        )
        if weight != reflected_weight:
            raise RuntimeError("public scattering law is not reflection paired")
        for necklace, code in enumerate(topology.necklace_codes):
            item = decode_code(code, work)
            port.lease(momentum, generation, necklace)
            work.scalar_port_leases += 1
            port.values[0] = compute_scalar(
                state, topology, item, code, momentum, work
            )
            port.require(momentum, generation, necklace)
            scatter_orbit(
                port.values[0],
                item,
                momentum,
                1,
                weight,
                output,
                topology.bracelet_codes,
                work,
            )
            reflected = (
                item
                if wrong_reflection
                else predecessor.reflect(item)
            )
            if not wrong_reflection:
                work.histogram_reflection_cells += GRID
            scatter_orbit(
                port.values[0],
                reflected,
                momentum,
                -1,
                weight,
                output,
                topology.bracelet_codes,
                work,
            )
            port.release(momentum, generation, necklace)
            work.scalar_port_releases += 1
            work.scalar_port_clear_field_cells += 1
        for target, value in enumerate(state):
            output[target] = (
                output[target] - 2 * weight * ROTORS * value
            ) % PRIME
    if port.live or port.values != [0] or id(port.values) != port_backing:
        raise RuntimeError("scalar momentum port did not restore on its backing")
    return output, work, {
        "scalar_port_same_backing": id(port.values) == port_backing,
        "scalar_port_live_after_scattering": port.live,
        "scalar_port_value_after_scattering": port.values[0],
        "retained_bracelet_lookup_indices": 0,
    }


def apply_diagonal(
    state: list[int],
    topology: CompactTopology,
    step: int,
    tag: int,
) -> tuple[list[int], Work]:
    work = Work()
    result: list[int] = []
    for value, code in zip(state, topology.bracelet_codes, strict=True):
        item = decode_code(code, work)
        signature = predecessor.public_law.pair_signature(item, ROTORS)
        result.append(
            value
            * pow(
                ROOT,
                predecessor.public_law.phase_exponent(signature, step, tag),
                PRIME,
            )
            % PRIME
        )
        work.diagonal_pair_signature_mode_terms += predecessor.PAIR_CHANNELS * GRID
    return result, work


def execute_word(
    source: list[int],
    topology: CompactTopology,
    operations: tuple[tuple[int, int], ...],
    *,
    reordered: bool = False,
    wrong_reflection: bool = False,
    resident_port: ScalarMomentumPort | None = None,
) -> tuple[list[int], Work, dict[str, object]]:
    current = source.copy()
    total = Work()
    diagnostics: dict[str, object] = {}
    for step, tag in operations:
        if reordered:
            current, scatter, diagnostics = apply_scattering_streamed(
                current,
                topology,
                step,
                tag,
                wrong_reflection=wrong_reflection,
                resident_port=resident_port,
            )
            current, diagonal = apply_diagonal(current, topology, step, tag)
        else:
            current, diagonal = apply_diagonal(current, topology, step, tag)
            current, scatter, diagnostics = apply_scattering_streamed(
                current,
                topology,
                step,
                tag,
                wrong_reflection=wrong_reflection,
                resident_port=resident_port,
            )
        total.add(diagonal)
        total.add(scatter)
    return current, total, diagnostics


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    scalar_port: ScalarMomentumPort
    generation: int = 0


def boundary(state: list[int], topology: CompactTopology) -> int:
    if len(state) != len(topology.bracelet_codes):
        raise ValueError("only closed bracelet state may be projected")
    return sum(
        value * weight
        for value, weight in zip(
            state, topology.boundary_weights, strict=True
        )
    ) % PRIME


def transaction(
    carrier: Carrier,
    expected_source: list[int],
    topology: CompactTopology,
    operations: tuple[tuple[int, int], ...],
) -> tuple[dict[str, object], str, Work, dict[str, object]]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("null or malformed streamed-factor carrier")
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    port_backing = id(carrier.scalar_port.values)
    forward, forward_work, diagnostics = execute_word(
        carrier.source,
        topology,
        operations,
        resident_port=carrier.scalar_port,
    )
    carrier.target[:] = [
        (left + right) % PRIME
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    forward_commitment = hashlib.sha256(
        ",".join(map(str, forward)).encode()
    ).hexdigest()
    projected = boundary(carrier.target, topology)
    missing_error = sum(value != 0 for value in carrier.target)
    forward.clear()
    rematerialized, reverse_work, reverse_diagnostics = execute_word(
        carrier.source,
        topology,
        operations,
        resident_port=carrier.scalar_port,
    )
    carrier.target[:] = [
        (left - right) % PRIME
        for left, right in zip(carrier.target, rematerialized, strict=True)
    ]
    rematerialized.clear()
    restoration_error = sum(
        left != right
        for left, right in zip(carrier.source, expected_source, strict=True)
    ) + sum(value != 0 for value in carrier.target)
    carrier.generation += 1
    total = Work()
    total.add(forward_work)
    total.add(reverse_work)
    if diagnostics != reverse_diagnostics:
        raise RuntimeError("scalar-port rematerialization diagnostics differ")
    return (
        {
            "boundary": projected,
            "missing_inverse_error_field_cells": missing_error,
            "restoration_error_field_cells": restoration_error,
            "same_backing": id(carrier.source) == source_backing
            and id(carrier.target) == target_backing
            and id(carrier.scalar_port.values) == port_backing,
            "scalar_port_restored": not carrier.scalar_port.live
            and carrier.scalar_port.values == [0],
            "generation": carrier.generation,
        },
        forward_commitment,
        total,
        diagnostics,
    )


def mismatch(left: list[int], right: list[int]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def typed_port_controls(topology: CompactTopology) -> dict[str, bool]:
    port = ScalarMomentumPort([0])
    port.lease(1, 7, 0)
    wrong_type = False
    wrong_owner = False
    premature_projection = False
    try:
        port.require(2, 7, 0)
    except ValueError:
        wrong_type = True
    try:
        port.require(1, 8, 0)
    except ValueError:
        wrong_owner = True
    try:
        boundary(port.values, topology)
    except ValueError:
        premature_projection = True
    port.release(1, 7, 0)
    return {
        "wrong_type_rejected": wrong_type,
        "wrong_owner_rejected": wrong_owner,
        "premature_scalar_projection_rejected": premature_projection,
        "control_port_restored": not port.live and port.values == [0],
    }


def source_commitment(state: list[int], order: tuple[int, ...]) -> str:
    return hashlib.sha256(
        ",".join(str(state[index]) for index in order).encode()
    ).hexdigest()


def state_commitment(state: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, state)).encode()).hexdigest()


def main() -> None:
    topology, compiler_work = compile_topology()
    source, signature_order = source_and_signature_order(topology, 0)
    reference_topology = predecessor.compile_topology()
    primary_word = predecessor.public_law.public_program(1, 0)
    reuse_word = predecessor.public_law.public_program(1, 4)
    wrong_word = predecessor.public_law.public_program(1, 1)

    primary_streamed, primary_forward_work, primary_diagnostics = execute_word(
        source, topology, primary_word
    )
    primary_factor, factor_work = predecessor.execute_word(
        source, reference_topology, primary_word
    )
    carrier = Carrier(
        source.copy(), [0] * len(source), ScalarMomentumPort([0])
    )
    resident_port_backing = id(carrier.scalar_port.values)
    primary, primary_forward, primary_work, primary_tx_diagnostics = transaction(
        carrier, source, topology, primary_word
    )
    reuse, reuse_forward, reuse_work, reuse_diagnostics = transaction(
        carrier, source, topology, reuse_word
    )
    fresh = Carrier(source.copy(), [0] * len(source), ScalarMomentumPort([0]))
    fresh_reuse, fresh_forward, fresh_work, fresh_diagnostics = transaction(
        fresh, source, topology, reuse_word
    )
    wrong, wrong_work, _ = execute_word(source, topology, wrong_word)
    reordered, reordered_work, _ = execute_word(
        source, topology, primary_word, reordered=True
    )
    wrong_reflection, wrong_reflection_work, _ = execute_word(
        source, topology, primary_word, wrong_reflection=True
    )
    controls = typed_port_controls(topology)
    null_rejected = False
    try:
        apply_scattering_streamed([], topology, 0, 0)
    except ValueError:
        null_rejected = True

    commitment = source_commitment(primary_streamed, signature_order)
    compact_topology_commitment = topology_commitment(topology)
    if (
        mismatch(primary_streamed, primary_factor)
        or compact_topology_commitment
        != predecessor.topology_commitment(reference_topology)
        or primary_forward != state_commitment(primary_streamed)
        or primary["boundary"] != predecessor.EXPECTED_PRIMARY_BOUNDARY
        or reuse["boundary"] != predecessor.EXPECTED_REUSE_BOUNDARY
        or fresh_reuse["boundary"] != reuse["boundary"]
        or fresh_forward != reuse_forward
        or commitment != predecessor.EXPECTED_PRIMARY_COMMITMENT
        or primary["restoration_error_field_cells"]
        or reuse["restoration_error_field_cells"]
        or fresh_reuse["restoration_error_field_cells"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or not fresh_reuse["same_backing"]
        or not primary["scalar_port_restored"]
        or not reuse["scalar_port_restored"]
        or not fresh_reuse["scalar_port_restored"]
        or id(carrier.scalar_port.values) != resident_port_backing
        or carrier.generation != 2
        or primary_diagnostics != primary_tx_diagnostics
        or reuse_diagnostics != fresh_diagnostics
        or mismatch(primary_streamed, wrong) == 0
        or mismatch(primary_streamed, reordered) == 0
        or mismatch(primary_streamed, wrong_reflection) == 0
        or not all(controls.values())
        or not null_rejected
    ):
        raise RuntimeError("scalar open-momentum transaction or controls failed")

    forward = primary_forward_work.as_dict()
    if (
        forward["scalar_port_leases"] != 35112
        or forward["scalar_port_releases"] != 35112
        or forward["scalar_port_clear_field_cells"] != 35112
        or forward["first_pass_one_body_terms"] != 162792
        or forward["closure_one_body_terms"] != 168912
        or forward["exact_bracelet_lookup_hits"] != 168912
        or forward["source_orbit_rotations"] != 1193808
        or forward["inverse_candidate_moves"] != 5534928
        or forward["histogram_digit_decodes"] != 3403077
        or forward["histogram_reflection_cells"] != 3364368
        or forward["sorted_code_searches"] != 5697720
        or forward["sorted_code_comparison_upper_bound"] != 68372640
    ):
        raise RuntimeError("scalar open-momentum work law changed")

    active_numeric_cells = 3 * len(topology.bracelet_codes) + 1
    retained_topology_descriptors = (
        len(topology.necklace_codes)
        + len(topology.bracelet_codes)
        + len(topology.boundary_weights)
    )
    accepted_named_slots = active_numeric_cells + retained_topology_descriptors
    necklace_code_payload_bits = sum(
        max(1, code.bit_length()) for code in topology.necklace_codes
    )
    bracelet_code_payload_bits = sum(
        max(1, code.bit_length()) for code in topology.bracelet_codes
    )
    boundary_fixed_width_bits = len(topology.boundary_weights) * 7
    active_numeric_fixed_width_bits = active_numeric_cells * 7
    accepted_fixed_width_logical_payload_bits = (
        necklace_code_payload_bits
        + bracelet_code_payload_bits
        + boundary_fixed_width_bits
        + active_numeric_fixed_width_bits
    )
    source_signature_coordinates = len(
        predecessor.refined_signature(decode_code(topology.bracelet_codes[0]))
    )
    public_source_compiler_logical_integer_slot_peak = len(
        topology.bracelet_codes
    ) * (source_signature_coordinates + 3)
    accepted_full_lifecycle_logical_slot_peak = max(
        accepted_named_slots,
        compiler_work["compiler_logical_integer_slot_peak"],
        public_source_compiler_logical_integer_slot_peak,
    )
    result = {
        "claim_candidate": "EXACT_F103_ROTOR6_IMPLICIT_DIHEDRAL_CODED_ONE_CELL_OPEN_MOMENTUM_STREAMING_RETAINS_ONLY8943_PUBLIC_TOPOLOGY_INTEGERS_WITHOUT_NECKLACE_HISTOGRAMS_HASH_MAPS_REFLECTION_MAPS_OR_TRANSITION_PLANS_WITH_EXACT_RESTORATION_AND_REUSE_BUT_PERFORMS5697720_SORTED_SEARCHES_WITH68372640_COMPARISON_UPPER_BOUND_PER_SCATTERING_AND_HAS_AN_IDENTICAL_CLASSICAL_STREAM",
        "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_REUSE_DIRECT_PROCESS_IMPLICIT_DIHEDRAL_CODE_ONE_CELL_STREAM_ONLY",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_IMPLICIT_TOPOLOGY_STATE_REDUCTION_WITH_SEARCH_WORK_TRADEOFF",
        "factor_law": {
            "identity": "K_Q_EQUALS_A_MINUS_Q_COMPOSE_A_Q_MINUS_ROTOR_COUNT_IDENTITY",
            "scalar_port_field_cells": 1,
            "maximum_simultaneously_live_scalar_ports": 1,
            "scalar_port_projected": False,
            "scalar_port_same_backing": primary_diagnostics[
                "scalar_port_same_backing"
            ],
            "scalar_port_live_after_scattering": primary_diagnostics[
                "scalar_port_live_after_scattering"
            ],
            "scalar_port_value_after_scattering": primary_diagnostics[
                "scalar_port_value_after_scattering"
            ],
            "open_momentum_vector_cells": 0,
            "full_occupation_scratch_cells": 0,
            "dense_2277_squared_operator_cells": 0,
            "permanent_assignment_terms": 0,
            "retained_transition_plan_entries": 0,
            "retained_inverse_history_bytes": 0,
            "compact_topology_commitment": compact_topology_commitment,
            "retained_necklace_histogram_cells": 0,
            "retained_hash_map_entries": 0,
            "retained_necklace_to_bracelet_entries": 0,
            "retained_reflection_map_entries": 0,
        },
        "parity": {
            "primary_streamed_factor_mismatch_cells": mismatch(
                primary_streamed, primary_factor
            ),
            "primary_boundary": primary["boundary"],
            "primary_signature_order_commitment": commitment,
            "reuse_boundary": reuse["boundary"],
            "fresh_reuse_boundary": fresh_reuse["boundary"],
            "fresh_restored_reuse_state_agreement": reuse_forward
            == fresh_forward,
        },
        "transaction": {
            "primary_restoration_error_field_cells": primary[
                "restoration_error_field_cells"
            ],
            "reuse_restoration_error_field_cells": reuse[
                "restoration_error_field_cells"
            ],
            "fresh_reuse_restoration_error_field_cells": fresh_reuse[
                "restoration_error_field_cells"
            ],
            "primary_same_backing": primary["same_backing"],
            "reuse_same_backing": reuse["same_backing"],
            "fresh_reuse_same_backing": fresh_reuse["same_backing"],
            "primary_scalar_port_restored": primary[
                "scalar_port_restored"
            ],
            "reuse_scalar_port_restored": reuse["scalar_port_restored"],
            "fresh_reuse_scalar_port_restored": fresh_reuse[
                "scalar_port_restored"
            ],
            "restoration_generation_after_reuse": carrier.generation,
            "restoration_method": "EXACT_TOPOLOGY_REMATERIALIZE_AND_SUBTRACT_ON_SAME_TARGET_BACKING",
            "forward_output_released_before_inverse": True,
            "baseline_reload_used": False,
        },
        "controls": {
            "missing_inverse_error_field_cells": primary[
                "missing_inverse_error_field_cells"
            ],
            "wrong_inverse_error_field_cells": mismatch(
                primary_streamed, wrong
            ),
            "reordered_noncommuting_error_field_cells": mismatch(
                primary_streamed, reordered
            ),
            "wrong_reflection_error_field_cells": mismatch(
                primary_streamed, wrong_reflection
            ),
            "null_carrier_rejected": null_rejected,
            **controls,
        },
        "resource_law": {
            "accepted_source_target_carrier_field_cells": 2
            * len(topology.bracelet_codes),
            "accepted_output_bracelet_field_cells": len(
                topology.bracelet_codes
            ),
            "accepted_scalar_port_field_cells": 1,
            "accepted_active_numeric_field_cells": active_numeric_cells,
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
            "named_algorithm_field_and_descriptor_slots": accepted_named_slots,
            "maximum_canonical_code_payload_bits": max(
                code.bit_length() for code in topology.necklace_codes
            ),
            "retained_necklace_code_payload_bits": necklace_code_payload_bits,
            "retained_bracelet_code_payload_bits": bracelet_code_payload_bits,
            "retained_boundary_fixed_width_bits": boundary_fixed_width_bits,
            "active_numeric_fixed_width_bits": active_numeric_fixed_width_bits,
            "accepted_fixed_width_logical_payload_bits": (
                accepted_fixed_width_logical_payload_bits
            ),
            "topology_compiler": compiler_work,
            "public_source_compiler_signature_coordinates": (
                source_signature_coordinates
            ),
            "public_source_compiler_logical_integer_slot_peak": (
                public_source_compiler_logical_integer_slot_peak
            ),
            "verification_only_signature_order_indices": len(signature_order),
            "public_program_descriptor_integers": 2 * len(primary_word),
            "accepted_full_lifecycle_logical_slot_peak": (
                accepted_full_lifecycle_logical_slot_peak
            ),
            "primary_forward_work": forward,
            "primary_forward_inverse_work": primary_work.as_dict(),
            "reuse_forward_inverse_work": reuse_work.as_dict(),
            "fresh_reuse_verification_work": fresh_work.as_dict(),
            "wrong_control_work": wrong_work.as_dict(),
            "reordered_control_work": reordered_work.as_dict(),
            "wrong_reflection_control_work": wrong_reflection_work.as_dict(),
            "m199_factor_forward_work": factor_work.as_dict(),
            "m199_factor_conservative_named_field_cells": 11220,
            "m199_comparable_named_field_and_descriptor_slots": 107943,
            "m202_named_field_and_descriptor_slots": 105832,
            "net_named_slot_saving_against_m199": 107943
            - accepted_named_slots,
            "net_named_slot_saving_against_m202": 105832
            - accepted_named_slots,
            "m201_position_dual_active_numeric_field_cells": 81444,
            "m201_position_dual_named_field_and_descriptor_slots": 1523509,
            "verification_only_reference_topology_not_in_accepted_path": True,
            "python_tuple_bisect_bigint_allocator_interpreter_timing_and_whole_process_peaks_excluded": True,
        },
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
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
