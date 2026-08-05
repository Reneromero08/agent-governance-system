#!/usr/bin/env python3
"""Independent oracle for the Rotor-6 eight-channel momentum-wave stream.

No production wave, M203, M202, or M199 module is imported. The oracle reuses
the prior separate implicit-dihedral oracle, independently implements the
eight-channel port and fused orbit schedule, and compares against separate
direct, factor, and scalar-stream references.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import growing_rotor_implicit_dihedral_scalar_streaming_independent_oracle as ref


GRID = ref.GRID
ROTORS = ref.ROTORS
PRIME = ref.PRIME
ROOT = ref.ROOT
CHANNELS = tuple(range(1, 9))
CHANNEL_TYPE = "REFLECTION_PAIRED_F103_MOMENTUM_WAVE8"
Histogram = tuple[int, ...]
CompactTopology = ref.CompactTopology


@dataclass
class WavePort:
    cells: list[int]
    channel_type: str | None = None
    generation: int | None = None
    necklace: int | None = None
    live: bool = False

    def lease(
        self,
        generation: int,
        necklace: int,
        *,
        channel_type: str = CHANNEL_TYPE,
    ) -> None:
        if self.live or self.cells != [0] * len(CHANNELS):
            raise RuntimeError("independent wave port was not clear")
        if channel_type != CHANNEL_TYPE:
            raise ValueError("independent wave port type is invalid")
        if generation <= 0 or not 0 <= necklace < 4389:
            raise ValueError("independent wave port owner is invalid")
        self.channel_type = channel_type
        self.generation = generation
        self.necklace = necklace
        self.live = True

    def require(
        self,
        generation: int,
        necklace: int,
        *,
        channel_type: str = CHANNEL_TYPE,
    ) -> None:
        if (
            not self.live
            or self.channel_type != channel_type
            or self.generation != generation
            or self.necklace != necklace
        ):
            raise ValueError("independent wave port type or owner mismatch")

    def project(self) -> tuple[int, ...]:
        if self.live:
            raise PermissionError("independent live wave projection rejected")
        return tuple(self.cells)

    def release(self, generation: int, necklace: int) -> None:
        self.require(generation, necklace)
        self.cells[:] = [0] * len(CHANNELS)
        self.channel_type = None
        self.generation = None
        self.necklace = None
        self.live = False


def fresh_work() -> dict[str, int]:
    work = ref.fresh_work()
    work.update(
        {
            "wave_port_leases": 0,
            "wave_port_releases": 0,
            "wave_port_clear_field_cells": 0,
            "wave_channel_values_computed": 0,
            "shared_source_histogram_decodes": 0,
            "shared_reflection_constructions": 0,
        }
    )
    return work


def merge(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def compute_wave(
    state: list[int],
    item: Histogram,
    code: int,
    topology: CompactTopology,
    work: dict[str, int],
) -> list[int]:
    cells = [0] * len(CHANNELS)
    for channel, momentum in enumerate(CHANNELS):
        value = 0
        for mode, count in enumerate(item):
            if count:
                source = ref.moved_bracelet(
                    code,
                    mode,
                    (mode - momentum) % GRID,
                    topology,
                    work,
                )
                value += count * state[source]
                work["first_pass_one_body_terms"] += 1
        cells[channel] = value % PRIME
        work["wave_channel_values_computed"] += 1
    return cells


def scatter_wave_orbit(
    port: WavePort,
    generation: int,
    necklace: int,
    item: Histogram,
    sign: int,
    weights: tuple[int, ...],
    output: list[int],
    topology: CompactTopology,
    work: dict[str, int],
) -> None:
    port.require(generation, necklace)
    rotated = item
    code = ref.reference.encode(item)
    for _ in range(GRID):
        work["source_orbit_rotations"] += 1
        for occupied, count in enumerate(rotated):
            if not count:
                continue
            for channel, momentum in enumerate(CHANNELS):
                target_mode = (occupied - sign * momentum) % GRID
                target_code = (
                    code
                    - ref.PLACE_VALUES[occupied]
                    + ref.PLACE_VALUES[target_mode]
                )
                work["inverse_candidate_moves"] += 1
                work["exact_bracelet_lookup_attempts"] += 1
                target = ref.sorted_index(
                    topology.bracelet_codes,
                    target_code,
                    work,
                    required=False,
                )
                if target is not None:
                    output[target] = (
                        output[target]
                        + weights[channel]
                        * (rotated[target_mode] + 1)
                        * port.cells[channel]
                    ) % PRIME
                    work["closure_one_body_terms"] += 1
                    work["exact_bracelet_lookup_hits"] += 1
        rotated = ref.rotate_once(rotated)
        code = code % ref.BASE * ref.HIGH_POWER + code // ref.BASE


def wave_scattering(
    state: list[int],
    topology: CompactTopology,
    step: int,
    tag: int,
    port: WavePort,
    *,
    wrong_reflection: bool = False,
) -> tuple[list[int], dict[str, int]]:
    if len(state) != len(topology.bracelet_codes):
        raise ValueError("independent null wave carrier rejected")
    output = [0] * len(state)
    work = fresh_work()
    weights = tuple(
        ref.reference.scattering_weight(momentum, step, tag)
        for momentum in CHANNELS
    )
    reflected_weights = tuple(
        ref.reference.scattering_weight(GRID - momentum, step, tag)
        for momentum in CHANNELS
    )
    if weights != reflected_weights:
        raise RuntimeError("independent paired weights changed")
    generation = 1 + step + GRID * tag
    for necklace, code in enumerate(topology.necklace_codes):
        item = ref.decode(code, work)
        work["shared_source_histogram_decodes"] += 1
        port.lease(generation, necklace)
        work["wave_port_leases"] += 1
        port.cells[:] = compute_wave(state, item, code, topology, work)
        scatter_wave_orbit(
            port,
            generation,
            necklace,
            item,
            1,
            weights,
            output,
            topology,
            work,
        )
        reflected = item if wrong_reflection else ref.reference.reflect(item)
        if not wrong_reflection:
            work["histogram_reflection_cells"] += GRID
            work["shared_reflection_constructions"] += 1
        scatter_wave_orbit(
            port,
            generation,
            necklace,
            reflected,
            -1,
            weights,
            output,
            topology,
            work,
        )
        port.release(generation, necklace)
        work["wave_port_releases"] += 1
        work["wave_port_clear_field_cells"] += len(CHANNELS)
    correction = 2 * ROTORS * sum(weights)
    for target, value in enumerate(state):
        output[target] = (output[target] - correction * value) % PRIME
    if port.live or port.cells != [0] * len(CHANNELS):
        raise RuntimeError("independent wave port did not restore")
    return output, work


def execute(
    source: list[int],
    topology: CompactTopology,
    operations: tuple[tuple[int, int], ...],
    port: WavePort,
    *,
    reordered: bool = False,
    wrong_reflection: bool = False,
) -> tuple[list[int], dict[str, int]]:
    current = source.copy()
    total = fresh_work()
    for step, tag in operations:
        if reordered:
            current, scattering = wave_scattering(
                current,
                topology,
                step,
                tag,
                port,
                wrong_reflection=wrong_reflection,
            )
            current, diagonal = ref.compact_diagonal(
                current, topology, step, tag
            )
        else:
            current, diagonal = ref.compact_diagonal(
                current, topology, step, tag
            )
            current, scattering = wave_scattering(
                current,
                topology,
                step,
                tag,
                port,
                wrong_reflection=wrong_reflection,
            )
        merge(total, diagonal)
        merge(total, scattering)
    return current, total


def transaction(
    carrier: tuple[list[int], list[int], WavePort],
    expected_source: list[int],
    topology: CompactTopology,
    operations: tuple[tuple[int, int], ...],
) -> tuple[int, int, bool, str]:
    source_backing = id(carrier[0])
    target_backing = id(carrier[1])
    port_backing = id(carrier[2].cells)
    forward, _ = execute(carrier[0], topology, operations, carrier[2])
    carrier[1][:] = [
        (left + right) % PRIME
        for left, right in zip(carrier[1], forward, strict=True)
    ]
    commitment = hashlib.sha256(
        ",".join(map(str, forward)).encode()
    ).hexdigest()
    boundary = sum(
        value * weight
        for value, weight in zip(
            carrier[1], topology.boundary_weights, strict=True
        )
    ) % PRIME
    forward.clear()
    rematerialized, _ = execute(
        carrier[0], topology, operations, carrier[2]
    )
    carrier[1][:] = [
        (left - right) % PRIME
        for left, right in zip(carrier[1], rematerialized, strict=True)
    ]
    rematerialized.clear()
    error = sum(
        left != right
        for left, right in zip(carrier[0], expected_source, strict=True)
    ) + sum(value != 0 for value in carrier[1])
    backing = (
        id(carrier[0]) == source_backing
        and id(carrier[1]) == target_backing
        and id(carrier[2].cells) == port_backing
        and not carrier[2].live
        and carrier[2].cells == [0] * len(CHANNELS)
    )
    return boundary, error, backing, commitment


def main() -> None:
    reference_topology = ref.reference.compile_topology()
    plans = ref.reference.compile_one_body_plans(reference_topology)
    topology, compiler_work = ref.compile_compact_topology()
    source = ref.reference.source_state(reference_topology, 0)
    primary_word = ref.reference.public_program(1, 0)
    reuse_word = ref.reference.public_program(1, 4)
    wrong_word = ref.reference.public_program(1, 1)
    direct_operator = ref.reference.compile_direct_operator(
        reference_topology, *primary_word[0]
    )

    primary, primary_work = execute(
        source, topology, primary_word, WavePort([0] * len(CHANNELS))
    )
    reuse, reuse_work = execute(
        source, topology, reuse_word, WavePort([0] * len(CHANNELS))
    )
    scalar_primary, scalar_work = ref.execute(
        source, topology, primary_word, ref.ScalarPort([0])
    )
    direct = ref.reference.execute_direct(
        source, reference_topology, direct_operator, *primary_word[0]
    )
    factor = ref.reference.execute_factor(
        source, reference_topology, plans, primary_word
    )
    carrier = (
        source.copy(),
        [0] * len(source),
        WavePort([0] * len(CHANNELS)),
    )
    primary_boundary, primary_error, primary_backing, primary_forward = (
        transaction(carrier, source, topology, primary_word)
    )
    reuse_boundary, reuse_error, reuse_backing, reuse_forward = transaction(
        carrier, source, topology, reuse_word
    )
    fresh = (
        source.copy(),
        [0] * len(source),
        WavePort([0] * len(CHANNELS)),
    )
    fresh_boundary, fresh_error, fresh_backing, fresh_forward = transaction(
        fresh, source, topology, reuse_word
    )
    wrong, _ = execute(
        source, topology, wrong_word, WavePort([0] * len(CHANNELS))
    )
    reordered, _ = execute(
        source,
        topology,
        primary_word,
        WavePort([0] * len(CHANNELS)),
        reordered=True,
    )
    wrong_reflection, _ = execute(
        source,
        topology,
        primary_word,
        WavePort([0] * len(CHANNELS)),
        wrong_reflection=True,
    )

    typed = WavePort([0] * len(CHANNELS))
    typed.lease(7, 0)
    wrong_type = wrong_owner = premature_projection = False
    try:
        typed.require(7, 0, channel_type="WRONG")
    except ValueError:
        wrong_type = True
    try:
        typed.require(8, 0)
    except ValueError:
        wrong_owner = True
    try:
        typed.project()
    except PermissionError:
        premature_projection = True
    typed.release(7, 0)
    null_rejected = False
    try:
        wave_scattering(
            [], topology, 0, 0, WavePort([0] * len(CHANNELS))
        )
    except ValueError:
        null_rejected = True

    active_numeric = 3 * len(topology.bracelet_codes) + len(CHANNELS)
    retained_topology = (
        len(topology.necklace_codes)
        + len(topology.bracelet_codes)
        + len(topology.boundary_weights)
    )
    named_slots = active_numeric + retained_topology
    necklace_bits = sum(
        max(1, code.bit_length()) for code in topology.necklace_codes
    )
    bracelet_bits = sum(
        max(1, code.bit_length()) for code in topology.bracelet_codes
    )
    source_coordinates = len(
        ref.reference.refined_signature(ref.decode(topology.bracelet_codes[0]))
    )
    source_compiler_peak = len(topology.bracelet_codes) * (
        source_coordinates + 3
    )
    fixed_payload = (
        necklace_bits
        + bracelet_bits
        + len(topology.boundary_weights) * 7
        + active_numeric * 7
    )

    if (
        math.comb(ROTORS + GRID - 1, ROTORS) != 74613
        or topology.necklace_codes
        != tuple(map(ref.reference.encode, reference_topology.necklaces))
        or topology.bracelet_codes
        != tuple(map(ref.reference.encode, reference_topology.bracelets))
        or topology.boundary_weights != reference_topology.boundary_weights
        or ref.mismatch(primary, scalar_primary)
        or ref.mismatch(primary, direct)
        or ref.mismatch(primary, factor)
        or primary_boundary != 83
        or reuse_boundary != 70
        or fresh_boundary != 70
        or primary_forward
        != hashlib.sha256(",".join(map(str, primary)).encode()).hexdigest()
        or reuse_forward
        != hashlib.sha256(",".join(map(str, reuse)).encode()).hexdigest()
        or fresh_forward != reuse_forward
        or any((primary_error, reuse_error, fresh_error))
        or not all((primary_backing, reuse_backing, fresh_backing))
        or primary_work["wave_port_leases"] != 4389
        or primary_work["wave_port_releases"] != 4389
        or primary_work["wave_port_clear_field_cells"] != 35112
        or primary_work["wave_channel_values_computed"] != 35112
        or primary_work["source_orbit_rotations"] != 149226
        or primary_work["inverse_candidate_moves"] != 5534928
        or primary_work["histogram_digit_decodes"] != 2880786
        or primary_work["histogram_reflection_cells"] != 2842077
        or primary_work["sorted_code_searches"] != 5697720
        or primary_work["sorted_code_comparison_upper_bound"] != 68372640
        or ref.mismatch(primary, wrong) == 0
        or ref.mismatch(primary, reordered) == 0
        or ref.mismatch(primary, wrong_reflection) == 0
        or not all((wrong_type, wrong_owner, premature_projection))
        or typed.live
        or typed.cells != [0] * len(CHANNELS)
        or not null_rejected
    ):
        raise RuntimeError("independent momentum-wave verification failed")

    print(
        json.dumps(
            {
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS_CHANNEL_FUSION_REDUCES_ORBIT_AND_DECODE_WORK_BUT_NOT_SEARCH_OR_CLASSICAL_EQUIVALENCE",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_REUSE_DIRECT_PROCESS_EIGHT_CHANNEL_IMPLICIT_DIHEDRAL_MOMENTUM_WAVE_STREAM_ONLY",
                "independent_state": {
                    "occupation_histograms": 74613,
                    "necklace_cells": len(topology.necklace_codes),
                    "bracelet_cells": len(topology.bracelet_codes),
                    "direct_two_body_raw_terms": direct_operator.raw_terms,
                    "direct_two_body_csr_nonzeros": int(
                        direct_operator.matrix.nnz
                    ),
                    "wave_scalar_mismatch_cells": ref.mismatch(
                        primary, scalar_primary
                    ),
                    "wave_direct_mismatch_cells": ref.mismatch(primary, direct),
                    "wave_factor_mismatch_cells": ref.mismatch(primary, factor),
                    "primary_boundary": primary_boundary,
                    "reuse_boundary": reuse_boundary,
                    "fresh_reuse_boundary": fresh_boundary,
                    "primary_signature_order_commitment": ref.reference.signature_commitment(
                        primary, reference_topology
                    ),
                    "topology_commitment": ref.reference.topology_commitment(
                        reference_topology
                    ),
                    "compact_codes_and_boundary_match_tuple_reference": True,
                },
                "wave_port_verification": {
                    "channel_type": CHANNEL_TYPE,
                    "channels": list(CHANNELS),
                    "logical_field_cells": len(CHANNELS),
                    "maximum_simultaneously_live": 1,
                    "individual_channels_projected": False,
                    "primary_work": primary_work,
                    "reuse_work": reuse_work,
                    "scalar_reference_work": scalar_work,
                    "verification_only_reference_topology_and_plans_retained": True,
                    "verification_only_plan_entries": plans.first_entries
                    + plans.second_entries,
                    "accepted_path_retained_transition_plan_entries": 0,
                    "wrong_type_rejected": wrong_type,
                    "wrong_owner_rejected": wrong_owner,
                    "premature_projection_rejected": premature_projection,
                    "null_carrier_rejected": null_rejected,
                    "control_port_restored": not typed.live
                    and typed.cells == [0] * len(CHANNELS),
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
                    "wrong_inverse_error_field_cells": ref.mismatch(
                        primary, wrong
                    ),
                    "reordered_noncommuting_error_field_cells": ref.mismatch(
                        primary, reordered
                    ),
                    "wrong_reflection_error_field_cells": ref.mismatch(
                        primary, wrong_reflection
                    ),
                },
                "resource_derivation": {
                    "active_numeric_field_cells": active_numeric,
                    "retained_public_topology_descriptor_integers": retained_topology,
                    "named_algorithm_field_and_descriptor_slots": named_slots,
                    "accepted_fixed_width_logical_payload_bits": fixed_payload,
                    "accepted_full_lifecycle_logical_slot_peak": max(
                        named_slots,
                        compiler_work["compiler_logical_integer_slot_peak"],
                        source_compiler_peak,
                    ),
                    "topology_compiler": compiler_work,
                    "public_source_compiler_logical_integer_slot_peak": source_compiler_peak,
                    "source_orbit_rotations": primary_work[
                        "source_orbit_rotations"
                    ],
                    "inverse_candidate_moves": primary_work[
                        "inverse_candidate_moves"
                    ],
                    "histogram_digit_decodes": primary_work[
                        "histogram_digit_decodes"
                    ],
                    "histogram_reflection_cells": primary_work[
                        "histogram_reflection_cells"
                    ],
                    "sorted_code_searches": primary_work[
                        "sorted_code_searches"
                    ],
                    "sorted_code_comparison_upper_bound": primary_work[
                        "sorted_code_comparison_upper_bound"
                    ],
                    "m203_source_orbit_rotation_reduction": scalar_work[
                        "source_orbit_rotations"
                    ] - primary_work["source_orbit_rotations"],
                    "m203_histogram_digit_decode_reduction": scalar_work[
                        "histogram_digit_decodes"
                    ] - primary_work["histogram_digit_decodes"],
                    "m203_histogram_reflection_cell_reduction": scalar_work[
                        "histogram_reflection_cells"
                    ] - primary_work["histogram_reflection_cells"],
                    "m203_sorted_search_reduction": scalar_work[
                        "sorted_code_searches"
                    ] - primary_work["sorted_code_searches"],
                    "m203_fanout_candidate_reduction": scalar_work[
                        "inverse_candidate_moves"
                    ] - primary_work["inverse_candidate_moves"],
                    "full_occupation_scratch_cells": 0,
                    "dense_operator_cells": 0,
                    "permanent_assignment_terms": 0,
                },
                "production_wave_module_imported": False,
                "production_implicit_dihedral_module_imported": False,
                "production_scalar_stream_module_imported": False,
                "production_m199_module_imported": False,
                "prior_independent_implicit_dihedral_reference_reused": True,
                "matched_classical_baselines": [
                    "IDENTICAL_EIGHT_CHANNEL_IMPLICIT_DIHEDRAL_VECTOR_STREAM",
                    "M203_ONE_CELL_IMPLICIT_DIHEDRAL_SCALAR_STREAM",
                    "M199_REFLECTION_PAIRED4389_CELL_PORT_FACTOR_STREAM",
                ],
                "catvm_custody": False,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "physical_simultaneous_superposition": False,
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
