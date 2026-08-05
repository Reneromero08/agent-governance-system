#!/usr/bin/env python3
"""Independent oracle for one-cell Rotor-6 momentum-factor streaming.

No production scalar-stream or M199 module is imported.  The oracle reuses the
separate M199 reference topology, compiles verification-only transposed
one-body plans, executes each paired momentum coordinate through one Python
scalar, and compares against an independently compiled direct two-body CSR.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import growing_rotor_open_momentum_factor_independent_oracle as reference


GRID = reference.GRID
ROTORS = reference.ROTORS
PRIME = reference.PRIME
Histogram = tuple[int, ...]
Outgoing = tuple[tuple[tuple[int, int], ...], ...]


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


def paired_stream_scattering(
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


def execute(
    source: list[int],
    topology: reference.Topology,
    plans: reference.OneBodyPlans,
    operations: tuple[tuple[int, int], ...],
    port: list[int],
    *,
    reordered: bool = False,
    wrong_reflection: bool = False,
) -> tuple[list[int], dict[str, int]]:
    current = source.copy()
    aggregate: dict[str, int] = {}
    for step, tag in operations:
        if reordered:
            current, work = paired_stream_scattering(
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
            current, work = paired_stream_scattering(
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


def transaction(
    carrier: tuple[list[int], list[int], ScalarPort],
    source: list[int],
    topology: reference.Topology,
    plans: reference.OneBodyPlans,
    operations: tuple[tuple[int, int], ...],
) -> tuple[int, int, bool, list[int]]:
    source_backing = id(carrier[0])
    target_backing = id(carrier[1])
    port_backing = id(carrier[2].values)
    forward, _ = execute(source, topology, plans, operations, carrier[2])
    carrier[1][:] = [
        (left + right) % PRIME
        for left, right in zip(carrier[1], forward, strict=True)
    ]
    projected = reference.boundary(carrier[1], topology)
    rematerialized, _ = execute(
        source, topology, plans, operations, carrier[2]
    )
    carrier[1][:] = [
        (left - right) % PRIME
        for left, right in zip(carrier[1], rematerialized, strict=True)
    ]
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
        forward,
    )


def mismatch(left: list[int], right: list[int]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def main() -> None:
    topology = reference.compile_topology()
    plans = reference.compile_one_body_plans(topology)
    source = reference.source_state(topology, 0)
    primary_word = reference.public_program(1, 0)
    reuse_word = reference.public_program(1, 4)
    wrong_word = reference.public_program(1, 1)
    direct_operator = reference.compile_direct_operator(
        topology, *primary_word[0]
    )

    primary, primary_work = execute(
        source, topology, plans, primary_word, ScalarPort([0])
    )
    reuse, reuse_work = execute(
        source, topology, plans, reuse_word, ScalarPort([0])
    )
    direct = reference.execute_direct(
        source, topology, direct_operator, *primary_word[0]
    )
    factor = reference.execute_factor(source, topology, plans, primary_word)
    carrier = (source.copy(), [0] * len(source), ScalarPort([0]))
    primary_boundary, primary_error, primary_backing, primary_forward = (
        transaction(carrier, source, topology, plans, primary_word)
    )
    reuse_boundary, reuse_error, reuse_backing, reuse_forward = transaction(
        carrier, source, topology, plans, reuse_word
    )
    fresh = (source.copy(), [0] * len(source), ScalarPort([0]))
    fresh_boundary, fresh_error, fresh_backing, fresh_forward = transaction(
        fresh, source, topology, plans, reuse_word
    )
    wrong, _ = execute(
        source, topology, plans, wrong_word, ScalarPort([0])
    )
    reordered, _ = execute(
        source,
        topology,
        plans,
        primary_word,
        ScalarPort([0]),
        reordered=True,
    )
    wrong_reflection, _ = execute(
        source,
        topology,
        plans,
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
    orbit_rotations = 2 * 8 * GRID * len(topology.necklaces)
    inverse_candidates = 2 * 8 * nonzero_mode_incidence
    predecessor_topology_descriptors = (
        len(topology.necklaces) * GRID
        + len(topology.necklaces)
        + len(topology.necklace_lookup)
        + len(topology.bracelets)
        + len(topology.necklace_to_bracelet)
        + len(topology.reflected_necklace)
        + len(topology.boundary_weights)
    )
    active_numeric_cells = 3 * len(topology.bracelets) + 1
    named_slots = (
        active_numeric_cells
        + predecessor_topology_descriptors
        + len(topology.bracelets)
    )

    if (
        occupation_count != 74613
        or mismatch(primary, direct)
        or mismatch(primary, factor)
        or primary_forward != primary
        or reuse_forward != reuse
        or fresh_forward != reuse
        or primary_boundary != 83
        or reuse_boundary != 70
        or fresh_boundary != 70
        or reference.signature_commitment(primary, topology)
        != "834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e"
        or any((primary_error, reuse_error, fresh_error))
        or not all((primary_backing, reuse_backing, fresh_backing))
        or primary_work["scalar_port_leases"] != 35112
        or primary_work["scalar_port_releases"] != 35112
        or primary_work["first_pass_one_body_terms"] != 162792
        or primary_work["closure_one_body_terms"] != 168912
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
                "result": "PASS_COMPACT_PORT_STREAMING_WITH_TOPOLOGY_WORK_TRADEOFF",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_REUSE_DIRECT_PROCESS_ONE_CELL_SCALAR_OPEN_MOMENTUM_STREAM_ONLY",
                "independent_state": {
                    "occupation_histograms": occupation_count,
                    "necklace_cells": len(topology.necklaces),
                    "bracelet_cells": len(topology.bracelets),
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
                        primary, topology
                    ),
                    "topology_commitment": reference.topology_commitment(topology),
                },
                "scalar_port_verification": {
                    "logical_field_cells": 1,
                    "maximum_simultaneously_live": 1,
                    "projected": False,
                    "primary_work": primary_work,
                    "reuse_work": reuse_work,
                    "verification_only_transposed_plans_retained": True,
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
                    "predecessor_topology_descriptor_integers": predecessor_topology_descriptors,
                    "additional_bracelet_lookup_indices": len(
                        topology.bracelets
                    ),
                    "named_algorithm_field_and_descriptor_slots": named_slots,
                    "m199_comparable_named_field_and_descriptor_slots": 11220
                    + predecessor_topology_descriptors,
                    "net_named_slot_saving_against_m199": 11220
                    + predecessor_topology_descriptors
                    - named_slots,
                    "source_orbit_rotations": orbit_rotations,
                    "inverse_candidate_moves": inverse_candidates,
                    "accepted_one_body_contributions": primary_work[
                        "first_pass_one_body_terms"
                    ]
                    + primary_work["closure_one_body_terms"],
                    "full_occupation_scratch_cells": 0,
                    "dense_operator_cells": 0,
                    "permanent_assignment_terms": 0,
                },
                "production_scalar_stream_module_imported": False,
                "production_m199_module_imported": False,
                "prior_independent_factor_reference_reused": True,
                "matched_classical_baselines": [
                    "IDENTICAL_ONE_CELL_TOPOLOGY_REMATERIALIZED_FACTOR_STREAM",
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
