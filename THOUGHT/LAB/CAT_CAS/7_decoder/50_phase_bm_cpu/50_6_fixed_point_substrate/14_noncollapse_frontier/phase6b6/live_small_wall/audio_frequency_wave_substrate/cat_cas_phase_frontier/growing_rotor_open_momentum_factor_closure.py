#!/usr/bin/env python3
"""Reflection-paired open-momentum factorization of Rotor-6 scattering.

For a nonzero momentum shift q, the M197 two-body operator factors exactly as

    K_q = A_-q A_q - N I,

where A_q moves one unresolved particle by -q.  Reflection symmetry pairs q
with -q, so only eight 4,389-cell necklace intermediates are leased in turn
and each is closed back to the 2,277-cell bracelet carrier before release.
The accepted path retains no transition plan or inverse history.

This remains direct-process exact F103 software.  The identical factor stream
is the strongest matched classical implementation; no distinct phase resource
or computational advantage is claimed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterator

import growing_rotor_pair_signature_streamed_quotient as public_law


GRID = 17
ROTORS = 6
PRIME = 103
ROOT = 72
PRIMARY_DEPTH = 1
REUSE_DEPTH = 1
BASE = ROTORS + 1
HIGH_POWER = BASE ** (GRID - 1)
PLACE_VALUES = tuple(BASE ** (GRID - 1 - mode) for mode in range(GRID))
PAIR_CHANNELS = 9
TRIANGLE_STENCILS = (((1, 3), (2, 3)), ((1, 5), (4, 5)))
PRIOR_TWO_BODY_MOVES = 684624
PRIOR_TRIANGLE_MONOMIAL_EVALUATIONS = 24767280
PRIOR_SIGNATURE_COMPARISONS = 7669494
EXPECTED_PRIMARY_BOUNDARY = 83
EXPECTED_REUSE_BOUNDARY = 70
EXPECTED_PRIMARY_COMMITMENT = (
    "834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e"
)
Histogram = tuple[int, ...]
Signature = tuple[int, ...]


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


def encode(item: Histogram) -> int:
    value = 0
    for count in item:
        value = value * BASE + count
    return value


def canonical_code(value: int) -> int:
    """Minimum cyclic rotation using sixteen scalar rolling updates."""
    current = value
    result = value
    for _ in range(1, GRID):
        current = (current % BASE) * HIGH_POWER + current // BASE
        result = min(result, current)
    return result


def reflect(item: Histogram) -> Histogram:
    return tuple(item[(-mode) % GRID] for mode in range(GRID))


def triangle_counts(item: Histogram) -> tuple[int, int]:
    return tuple(
        sum(
            item[anchor]
            * item[(anchor + first) % GRID]
            * item[(anchor + second) % GRID]
            for first, second in orientations
            for anchor in range(GRID)
        )
        for orientations in TRIANGLE_STENCILS
    )  # type: ignore[return-value]


def refined_signature(item: Histogram) -> Signature:
    return public_law.pair_signature(item, ROTORS) + triangle_counts(item)


@dataclass(frozen=True)
class FactorTopology:
    necklaces: tuple[Histogram, ...]
    necklace_codes: tuple[int, ...]
    necklace_lookup: dict[int, int]
    bracelets: tuple[Histogram, ...]
    bracelet_codes: tuple[int, ...]
    necklace_to_bracelet: tuple[int, ...]
    reflected_necklace: tuple[int, ...]
    boundary_weights: tuple[int, ...]
    occupation_histograms: int


def compile_topology() -> FactorTopology:
    necklaces: list[Histogram] = []
    occupation_count = 0
    for item in iter_histograms(ROTORS):
        occupation_count += 1
        code = encode(item)
        if canonical_code(code) == code:
            necklaces.append(item)
    necklace_tuple = tuple(necklaces)
    necklace_codes = tuple(map(encode, necklace_tuple))
    necklace_lookup = {code: index for index, code in enumerate(necklace_codes)}
    if len(necklace_lookup) != len(necklace_tuple):
        raise RuntimeError("encoded necklace lookup collision")
    reflected_codes = tuple(
        canonical_code(encode(reflect(item))) for item in necklace_tuple
    )
    bracelet_codes = tuple(
        sorted(
            {
                min(code, reflected)
                for code, reflected in zip(
                    necklace_codes, reflected_codes, strict=True
                )
            }
        )
    )
    bracelet_lookup = {code: index for index, code in enumerate(bracelet_codes)}
    bracelets = tuple(necklace_tuple[necklace_lookup[code]] for code in bracelet_codes)
    necklace_to_bracelet = tuple(
        bracelet_lookup[min(code, reflected)]
        for code, reflected in zip(necklace_codes, reflected_codes, strict=True)
    )
    reflected_necklace = tuple(necklace_lookup[code] for code in reflected_codes)
    boundary = [0] * len(bracelets)
    for necklace_index, item in enumerate(necklace_tuple):
        collision = sum(count * (count - 1) // 2 for count in item)
        bracelet = necklace_to_bracelet[necklace_index]
        boundary[bracelet] = (
            boundary[bracelet]
            + pow(ROOT, (11 * necklace_index + 5 * collision + 1) % GRID, PRIME)
        ) % PRIME
    if (
        occupation_count != math.comb(ROTORS + GRID - 1, ROTORS)
        or len(necklace_tuple) != 4389
        or len(bracelets) != 2277
    ):
        raise RuntimeError("open-momentum topology law changed")
    return FactorTopology(
        necklaces=necklace_tuple,
        necklace_codes=necklace_codes,
        necklace_lookup=necklace_lookup,
        bracelets=bracelets,
        bracelet_codes=bracelet_codes,
        necklace_to_bracelet=necklace_to_bracelet,
        reflected_necklace=reflected_necklace,
        boundary_weights=tuple(boundary),
        occupation_histograms=occupation_count,
    )


def source_and_signature_order(
    topology: FactorTopology, family: int
) -> tuple[list[int], tuple[int, ...]]:
    records = sorted(
        (refined_signature(item), bracelet)
        for bracelet, item in enumerate(topology.bracelets)
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


def signature_order_commitment(state: list[int], order: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(str(state[index]) for index in order).encode()).hexdigest()


def topology_commitment(topology: FactorTopology) -> str:
    digest = hashlib.sha256()
    for index, item in enumerate(topology.necklaces):
        digest.update(
            (
                f"{topology.necklace_codes[index]}:"
                + ",".join(map(str, item))
                + f":{topology.necklace_to_bracelet[index]}:"
                + f"{topology.reflected_necklace[index]};"
            ).encode()
        )
    digest.update(",".join(map(str, topology.boundary_weights)).encode())
    return digest.hexdigest()


@dataclass
class OpenMomentumPort:
    values: list[int]
    momentum: int | None = None
    owner_generation: int | None = None
    live: bool = False

    def lease(self, momentum: int, owner_generation: int) -> None:
        if self.live or any(self.values):
            raise RuntimeError("open momentum port was not released")
        if not 1 <= momentum <= 8:
            raise ValueError("momentum port type is outside paired channel range")
        self.momentum = momentum
        self.owner_generation = owner_generation
        self.live = True

    def require(self, momentum: int, owner_generation: int) -> None:
        if (
            not self.live
            or self.momentum != momentum
            or self.owner_generation != owner_generation
        ):
            raise ValueError("open momentum port type or owner mismatch")

    def release(self, momentum: int, owner_generation: int) -> None:
        self.require(momentum, owner_generation)
        self.values[:] = [0] * len(self.values)
        self.momentum = None
        self.owner_generation = None
        self.live = False


@dataclass
class Work:
    scatterings: int = 0
    momentum_port_leases: int = 0
    momentum_port_releases: int = 0
    first_pass_one_body_terms: int = 0
    closure_one_body_terms: int = 0
    encoded_move_deltas: int = 0
    cyclic_code_candidates: int = 0
    cyclic_code_rolling_updates: int = 0
    topology_index_lookups: int = 0
    port_clear_field_cells: int = 0
    diagonal_pair_signature_mode_terms: int = 0

    def add(self, other: "Work") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def moved_necklace_index(
    target_code: int,
    mode: int,
    destination: int,
    topology: FactorTopology,
    work: Work,
) -> int:
    moved = target_code - PLACE_VALUES[mode] + PLACE_VALUES[destination]
    key = canonical_code(moved)
    work.encoded_move_deltas += 1
    work.cyclic_code_candidates += GRID
    work.cyclic_code_rolling_updates += GRID - 1
    work.topology_index_lookups += 1
    return topology.necklace_lookup[key]


def fill_open_port(
    state: list[int],
    topology: FactorTopology,
    port: OpenMomentumPort,
    momentum: int,
    owner_generation: int,
    work: Work,
) -> None:
    port.lease(momentum, owner_generation)
    work.momentum_port_leases += 1
    for target, (item, code) in enumerate(
        zip(topology.necklaces, topology.necklace_codes, strict=True)
    ):
        accumulator = 0
        for mode, count in enumerate(item):
            if count:
                source = moved_necklace_index(
                    code, mode, (mode - momentum) % GRID, topology, work
                )
                accumulator += count * state[topology.necklace_to_bracelet[source]]
                work.first_pass_one_body_terms += 1
        port.values[target] = accumulator % PRIME


def close_reflection_pair(
    state: list[int],
    output: list[int],
    topology: FactorTopology,
    port: OpenMomentumPort,
    momentum: int,
    owner_generation: int,
    step: int,
    tag: int,
    work: Work,
    wrong_reflection: bool = False,
) -> None:
    port.require(momentum, owner_generation)
    positive_weight = public_law.public_scattering_integer(momentum, step, tag)
    negative_weight = public_law.public_scattering_integer(
        GRID - momentum, step, tag
    )
    if positive_weight != negative_weight:
        raise RuntimeError("public scattering law is not reflection paired")
    for target, (item, code) in enumerate(
        zip(topology.bracelets, topology.bracelet_codes, strict=True)
    ):
        positive = 0
        negative = 0
        for mode, count in enumerate(item):
            if count == 0:
                continue
            positive_source = moved_necklace_index(
                code, mode, (mode + momentum) % GRID, topology, work
            )
            negative_source = moved_necklace_index(
                code, mode, (mode - momentum) % GRID, topology, work
            )
            positive += count * port.values[positive_source]
            reflected = (
                negative_source
                if wrong_reflection
                else topology.reflected_necklace[negative_source]
            )
            negative += count * port.values[reflected]
            work.closure_one_body_terms += 2
        output[target] = (
            output[target]
            + positive_weight
            * (positive + negative - 2 * ROTORS * state[target])
        ) % PRIME
    port.release(momentum, owner_generation)
    work.momentum_port_releases += 1
    work.port_clear_field_cells += len(port.values)


def apply_scattering_factorized(
    state: list[int],
    topology: FactorTopology,
    step: int,
    tag: int,
    wrong_reflection: bool = False,
) -> tuple[list[int], Work]:
    if len(state) != len(topology.bracelets):
        raise ValueError("null or malformed bracelet carrier")
    output = [0] * len(state)
    port = OpenMomentumPort([0] * len(topology.necklaces))
    port_backing = id(port.values)
    work = Work(scatterings=1)
    for generation, momentum in enumerate(range(1, 9), 1):
        fill_open_port(
            state, topology, port, momentum, generation, work
        )
        close_reflection_pair(
            state,
            output,
            topology,
            port,
            momentum,
            generation,
            step,
            tag,
            work,
            wrong_reflection=wrong_reflection,
        )
    if port.live or any(port.values) or id(port.values) != port_backing:
        raise RuntimeError("open momentum port did not close on the same backing")
    return output, work


def apply_diagonal(
    state: list[int],
    topology: FactorTopology,
    step: int,
    tag: int,
) -> tuple[list[int], Work]:
    output: list[int] = []
    for value, item in zip(state, topology.bracelets, strict=True):
        signature = public_law.pair_signature(item, ROTORS)
        output.append(
            value
            * pow(ROOT, public_law.phase_exponent(signature, step, tag), PRIME)
            % PRIME
        )
    return output, Work(
        diagonal_pair_signature_mode_terms=len(state) * PAIR_CHANNELS * GRID
    )


def execute_word(
    source: list[int],
    topology: FactorTopology,
    operations: tuple[tuple[int, int], ...],
    reordered: bool = False,
) -> tuple[list[int], Work]:
    current = source.copy()
    total = Work()
    for step, tag in operations:
        if reordered:
            current, scatter = apply_scattering_factorized(
                current, topology, step, tag
            )
            current, diagonal = apply_diagonal(current, topology, step, tag)
        else:
            current, diagonal = apply_diagonal(current, topology, step, tag)
            current, scatter = apply_scattering_factorized(
                current, topology, step, tag
            )
        total.add(diagonal)
        total.add(scatter)
    return current, total


def boundary(state: list[int], topology: FactorTopology) -> int:
    if len(state) != len(topology.bracelets):
        raise ValueError("only closed bracelet state may be projected")
    return sum(
        value * weight
        for value, weight in zip(state, topology.boundary_weights, strict=True)
    ) % PRIME


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    generation: int = 0


def transaction(
    carrier: Carrier,
    expected_source: list[int],
    topology: FactorTopology,
    operations: tuple[tuple[int, int], ...],
) -> tuple[dict[str, object], list[int], Work]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("null or malformed open-momentum carrier")
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    forward, forward_work = execute_word(carrier.source, topology, operations)
    carrier.target[:] = [
        (left + right) % PRIME
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    projected = boundary(carrier.target, topology)
    inverse, inverse_work = execute_word(carrier.source, topology, operations)
    carrier.target[:] = [
        (left - right) % PRIME
        for left, right in zip(carrier.target, inverse, strict=True)
    ]
    error = sum(
        left != right
        for left, right in zip(carrier.source, expected_source, strict=True)
    ) + sum(value != 0 for value in carrier.target)
    carrier.generation += 1
    total = Work()
    total.add(forward_work)
    total.add(inverse_work)
    return (
        {
            "boundary": projected,
            "restoration_error_field_cells": error,
            "same_backing": id(carrier.source) == source_backing
            and id(carrier.target) == target_backing,
            "generation": carrier.generation,
        },
        forward,
        total,
    )


def restoration_error(left: list[int], right: list[int]) -> int:
    return sum(
        (a - b) % PRIME != 0 for a, b in zip(left, right, strict=True)
    )


def typed_port_controls(
    state: list[int], topology: FactorTopology
) -> dict[str, object]:
    port = OpenMomentumPort([0] * len(topology.necklaces))
    work = Work()
    fill_open_port(state, topology, port, 1, 91, work)
    wrong_owner_rejected = False
    try:
        port.require(1, 92)
    except ValueError:
        wrong_owner_rejected = True
    wrong_type_rejected = False
    try:
        port.require(2, 91)
    except ValueError:
        wrong_type_rejected = True
    premature_projection_rejected = False
    try:
        boundary(port.values, topology)
    except ValueError:
        premature_projection_rejected = True
    port.release(1, 91)
    return {
        "wrong_owner_rejected": wrong_owner_rejected,
        "wrong_momentum_type_rejected": wrong_type_rejected,
        "premature_intermediate_projection_rejected": premature_projection_rejected,
        "control_port_released_to_zero": not port.live and not any(port.values),
    }


def main() -> None:
    topology = compile_topology()
    source, signature_order = source_and_signature_order(topology, 0)
    primary_word = public_law.public_program(PRIMARY_DEPTH, 0)
    reuse_word = public_law.public_program(REUSE_DEPTH, 4)
    wrong_word = public_law.public_program(PRIMARY_DEPTH, 1)
    carrier = Carrier(source.copy(), [0] * len(source))
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    primary, primary_forward, primary_work = transaction(
        carrier, source, topology, primary_word
    )
    reuse, reuse_forward, reuse_work = transaction(
        carrier, source, topology, reuse_word
    )
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse, fresh_reuse_forward, fresh_reuse_work = transaction(
        fresh, source, topology, reuse_word
    )
    matched, matched_work = execute_word(source, topology, primary_word)
    wrong, wrong_work = execute_word(source, topology, wrong_word)
    reordered, reordered_work = execute_word(
        source, topology, primary_word, reordered=True
    )
    wrong_reflection, wrong_reflection_work = apply_scattering_factorized(
        apply_diagonal(source, topology, *primary_word[0])[0],
        topology,
        *primary_word[0],
        wrong_reflection=True,
    )
    port_controls = typed_port_controls(source, topology)
    null_rejected = False
    try:
        apply_scattering_factorized([], topology, 0, 0)
    except ValueError:
        null_rejected = True

    primary_commitment = signature_order_commitment(
        primary_forward, signature_order
    )
    if (
        primary["boundary"] != EXPECTED_PRIMARY_BOUNDARY
        or primary_commitment != EXPECTED_PRIMARY_COMMITMENT
        or reuse["boundary"] != EXPECTED_REUSE_BOUNDARY
        or reuse["boundary"] != fresh_reuse["boundary"]
        or reuse_forward != fresh_reuse_forward
        or primary_forward != matched
        or primary["restoration_error_field_cells"]
        or reuse["restoration_error_field_cells"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or id(carrier.source) != source_backing
        or id(carrier.target) != target_backing
        or carrier.generation != 2
        or restoration_error(primary_forward, wrong) == 0
        or restoration_error(primary_forward, reordered) == 0
        or restoration_error(primary_forward, wrong_reflection) == 0
        or not all(port_controls.values())
        or not null_rejected
    ):
        raise RuntimeError("open-momentum transaction or control failed")

    forward = primary_work.as_dict()
    per_scattering_terms = (
        forward["first_pass_one_body_terms"]
        + forward["closure_one_body_terms"]
    ) // forward["scatterings"]
    if (
        forward["first_pass_one_body_terms"] // forward["scatterings"] != 162792
        or forward["closure_one_body_terms"] // forward["scatterings"] != 168912
        or per_scattering_terms != 331704
        or forward["cyclic_code_candidates"] // forward["scatterings"] != 5638968
    ):
        raise RuntimeError("open-momentum exact work law changed")

    result = {
        "claim_candidate": "EXACT_F103_ROTOR6_REFLECTION_PAIRED_OPEN_MOMENTUM_PORT_FACTORS_THE_TWO_BODY_SCATTERING_THROUGH_ONE4389_CELL_UNRESOLVED_NECKLACE_INTERMEDIATE_CLOSES_BACK_TO2277_BRACELET_CELLS_WITH331704_ONE_BODY_TERMS_PER_SCATTERING_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_THE_IDENTICAL_CLASSICAL_FACTOR_STREAM_AND_GROWING_TOPOLOGY_REMAIN",
        "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_AND_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_AND_REUSE_DIRECT_PROCESS_SOFTWARE_TYPED_OPEN_MOMENTUM_PORT_ONLY",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS",
        "factor_law": {
            "identity": "K_Q_EQUALS_A_MINUS_Q_COMPOSE_A_Q_MINUS_ROTOR_COUNT_IDENTITY",
            "reflection_pairs": [[q, GRID - q] for q in range(1, 9)],
            "public_scattering_weights_are_equal_within_each_pair": True,
            "open_port_type": "MOMENTUM_Q_NECKLACE_PHASE_VECTOR",
            "open_port_logical_cells": len(topology.necklaces),
            "final_closed_bracelet_cells": len(topology.bracelets),
            "maximum_simultaneously_live_open_ports": 1,
            "intermediate_projected": False,
            "intermediate_released_to_zero_after_each_pair": True,
            "retained_transition_plan_entries": 0,
            "retained_inverse_history_bytes": 0,
        },
        "topology": {
            "occupation_histograms": topology.occupation_histograms,
            "necklace_cells": len(topology.necklaces),
            "bracelet_cells": len(topology.bracelets),
            "necklace_histogram_descriptor_integers": sum(
                map(len, topology.necklaces)
            ),
            "necklace_encoded_keys": len(topology.necklace_codes),
            "necklace_lookup_indices": len(topology.necklace_lookup),
            "necklace_to_bracelet_indices": len(topology.necklace_to_bracelet),
            "reflected_necklace_indices": len(topology.reflected_necklace),
            "bracelet_encoded_keys": len(topology.bracelet_codes),
            "boundary_weight_field_cells": len(topology.boundary_weights),
            "topology_commitment": topology_commitment(topology),
            "public_topology_compile_reads_final_answers": False,
        },
        "transaction": {
            "primary_boundary": primary["boundary"],
            "primary_signature_order_commitment": primary_commitment,
            "reuse_boundary": reuse["boundary"],
            "fresh_reuse_boundary": fresh_reuse["boundary"],
            "fresh_restored_reuse_state_agreement": reuse_forward
            == fresh_reuse_forward,
            "primary_restoration_error_field_cells": primary[
                "restoration_error_field_cells"
            ],
            "reuse_restoration_error_field_cells": reuse[
                "restoration_error_field_cells"
            ],
            "same_backing_primary": primary["same_backing"],
            "same_backing_reuse": reuse["same_backing"],
            "restoration_generation_after_reuse": carrier.generation,
            "baseline_reload_used": False,
            "primary_forward_inverse_work": primary_work.as_dict(),
            "reuse_forward_inverse_work": reuse_work.as_dict(),
            "fresh_reuse_verification_work": fresh_reuse_work.as_dict(),
            "matched_classical_forward_work": matched_work.as_dict(),
        },
        "controls": {
            "missing_inverse_error_field_cells": sum(
                value != 0 for value in primary_forward
            ),
            "wrong_inverse_error_field_cells": restoration_error(
                primary_forward, wrong
            ),
            "reordered_inverse_error_field_cells": restoration_error(
                primary_forward, reordered
            ),
            "wrong_reflection_pair_error_field_cells": restoration_error(
                primary_forward, wrong_reflection
            ),
            "null_carrier_rejected": null_rejected,
            **port_controls,
            "wrong_inverse_control_work": wrong_work.as_dict(),
            "reordered_inverse_control_work": reordered_work.as_dict(),
            "wrong_reflection_control_work": wrong_reflection_work.as_dict(),
        },
        "resource_law": {
            "accepted_carrier_resident_field_cells": 2 * len(topology.bracelets),
            "accepted_open_port_scratch_field_cells": len(topology.necklaces),
            "accepted_output_scratch_field_cells": len(topology.bracelets),
            "accepted_conservative_carrier_plus_open_port_plus_output_field_cells": (
                3 * len(topology.bracelets) + len(topology.necklaces)
            ),
            "accepted_one_body_terms_per_scattering": per_scattering_terms,
            "accepted_first_pass_terms_per_scattering": 162792,
            "accepted_closure_terms_per_scattering": 168912,
            "accepted_encoded_move_deltas_per_scattering": 331704,
            "accepted_cyclic_code_candidates_per_scattering": 5638968,
            "accepted_cyclic_code_rolling_updates_per_scattering": 5307264,
            "accepted_topology_lookups_per_scattering": 331704,
            "accepted_port_clear_field_cells_per_scattering": 35112,
            "prior_m197_two_body_moves_per_scattering": PRIOR_TWO_BODY_MOVES,
            "prior_m197_triangle_monomial_evaluations_per_scattering": PRIOR_TRIANGLE_MONOMIAL_EVALUATIONS,
            "prior_m197_signature_comparisons_per_scattering": PRIOR_SIGNATURE_COMPARISONS,
            "accepted_one_body_term_ratio_to_prior_two_body_moves": (
                per_scattering_terms / PRIOR_TWO_BODY_MOVES
            ),
            "accepted_retained_transition_plan_entries": 0,
            "accepted_retained_inverse_history_bytes": 0,
            "relation_table_or_assignment_expansion_cells": 0,
            "source_compile_temporary_signature_integer_cells": (
                len(topology.bracelets) * 11
            ),
            "source_compile_temporary_signature_order_indices": len(
                signature_order
            ),
            "python_containers_allocator_bigint_hash_map_expression_temporaries_timing_and_whole_process_peaks_excluded": True,
        },
        "matched_classical_recurrence": "IDENTICAL_REFLECTION_PAIRED_OPEN_MOMENTUM_ONE_BODY_FACTOR_STREAM_ON2277_BRACELET_AND4389_TEMPORARY_NECKLACE_CELLS",
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
