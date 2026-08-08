#!/usr/bin/env python3
"""M231 topology-rematerialized open SU(2)_8 fusion fanout stream.

M230 retained five nine-cell work rails.  This successor retains only the
actual produced nine-cell internal message plus one-cell twist, cubic,
intersection, and final-boundary backings.  Public topology rematerializes the
two branch values at each label, closes them by a reversible scalar Hadamard
product, streams the transposed final fusion into the only projectable scalar,
and immediately clears all three branch scratch cells.  Reverse traversal
rematerializes the same terms, clears the boundary, and then clears the
internal fusion message on its original backing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import root_of_unity_su2_level8_fusion_phase_relation as su2
import su2_level8_open_fusion_signature_fanout_intersection as prior


sys.set_int_max_str_digits(0)

FAMILIES = prior.FAMILIES
PRIMARY_FAMILY = prior.PRIMARY_FAMILY
REUSE_FAMILY = prior.REUSE_FAMILY


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Work(prior.Work):
    scalar_labels_forward: int = 0
    scalar_labels_inverse: int = 0
    twist_scalar_productions: int = 0
    twist_scalar_clears: int = 0
    cubic_scalar_productions: int = 0
    cubic_scalar_clears: int = 0
    intersection_scalar_productions: int = 0
    intersection_scalar_clears: int = 0
    transposed_fusion_boundary_accumulations: int = 0
    transposed_fusion_boundary_clears: int = 0
    cubic_term_rematerializations: int = 0
    scalar_scratch_restore_checks: int = 0
    internal_consumer_mask: int = 0
    retained_result_values: tuple[su2.K, ...] = field(default_factory=tuple, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"retained_topology_integers", "retained_result_values"}
        }

    def observe_relation(
        self,
        rails: Sequence[list[su2.K]],
        *,
        transients: Sequence[su2.K] = (),
        context: str,
    ) -> None:
        descriptor_bits = sum(
            prior.signed_bits(value) for value in self.retained_topology_integers
        )
        field_cells = (
            sum(len(rail) for rail in rails)
            + len(transients)
            + len(self.retained_result_values)
        )
        payload = sum(su2.field_payload_bits(rail) for rail in rails)
        payload += su2.field_payload_bits(tuple(transients))
        payload += su2.field_payload_bits(self.retained_result_values)
        payload += descriptor_bits
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells, field_cells
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context


def require_scalar_zero(cell: list[su2.K], name: str) -> None:
    if len(cell) != 1 or cell[0] != su2.ZERO:
        raise ValueError(f"{name} scratch backing is not exactly zero")


def topology_boundary_weight(
    label: int, topology: prior.CompiledTopology, work: Work
) -> tuple[su2.K, tuple[su2.K, ...]]:
    lower = su2.QUANTUM_DIMENSIONS[label - 1] if label else su2.ZERO
    upper = (
        su2.QUANTUM_DIMENSIONS[label + 1]
        if label + 1 < su2.SIMPLE_OBJECTS
        else su2.ZERO
    )
    dimension_sum = work.add(lower, upper)
    output_phase = su2.K.zeta(topology.output_fusion.parameter)
    weight = work.multiply(output_phase, dimension_sum)
    return weight, (dimension_sum, output_phase, weight)


def produce_twist_scalar(
    twist: list[su2.K],
    internal: list[su2.K],
    label: int,
    topology: prior.CompiledTopology,
    work: Work,
    rails: Sequence[list[su2.K]],
) -> None:
    require_scalar_zero(twist, "twist")
    phase = su2.twist_multiplier(label, topology.twist_branch.parameter, False)
    value = work.multiply(phase, internal[label])
    updated = work.add(twist[0], value)
    work.observe_relation(
        rails,
        transients=(phase, value, updated),
        context="REMATERIALIZE_TWIST_SCALAR",
    )
    twist[0] = updated
    work.twist_scalar_productions += 1
    work.branch_field_multiplications += 1
    work.branch_field_additions += 1
    work.internal_consumer_mask |= 1


def clear_twist_scalar(
    twist: list[su2.K],
    internal: list[su2.K],
    label: int,
    topology: prior.CompiledTopology,
    work: Work,
    rails: Sequence[list[su2.K]],
) -> None:
    phase = su2.twist_multiplier(label, topology.twist_branch.parameter, False)
    value = work.multiply(phase, internal[label])
    updated = work.subtract(twist[0], value)
    work.observe_relation(
        rails,
        transients=(phase, value, updated),
        context="CLEAR_TWIST_SCALAR",
    )
    twist[0] = updated
    require_scalar_zero(twist, "twist")
    work.twist_scalar_clears += 1
    work.branch_field_multiplications += 1
    work.branch_field_subtractions += 1


def cubic_value(
    internal: list[su2.K],
    label: int,
    signature: prior.CubicBranchSignature,
    work: Work,
) -> tuple[su2.K, tuple[su2.K, ...]]:
    if label != signature.target:
        return internal[label], ()
    phase = su2.K.zeta(signature.parameter)
    source_value = internal[signature.source]
    square = work.multiply(source_value, source_value)
    cube = work.multiply(square, source_value)
    term = work.multiply(phase, cube)
    value = work.add(internal[label], term)
    work.cubic_term_rematerializations += 1
    work.branch_field_multiplications += 3
    work.branch_field_additions += 1
    return value, (phase, square, cube, term, value)


def produce_cubic_scalar(
    cubic: list[su2.K],
    internal: list[su2.K],
    label: int,
    topology: prior.CompiledTopology,
    work: Work,
    rails: Sequence[list[su2.K]],
) -> None:
    require_scalar_zero(cubic, "cubic")
    value, transients = cubic_value(internal, label, topology.cubic_branch, work)
    updated = work.add(cubic[0], value)
    work.observe_relation(
        rails,
        transients=transients + (updated,),
        context="REMATERIALIZE_CUBIC_SCALAR",
    )
    cubic[0] = updated
    work.cubic_scalar_productions += 1
    work.branch_field_additions += 1
    work.internal_consumer_mask |= 2


def clear_cubic_scalar(
    cubic: list[su2.K],
    internal: list[su2.K],
    label: int,
    topology: prior.CompiledTopology,
    work: Work,
    rails: Sequence[list[su2.K]],
) -> None:
    value, transients = cubic_value(internal, label, topology.cubic_branch, work)
    updated = work.subtract(cubic[0], value)
    work.observe_relation(
        rails,
        transients=transients + (updated,),
        context="CLEAR_CUBIC_SCALAR",
    )
    cubic[0] = updated
    require_scalar_zero(cubic, "cubic")
    work.cubic_scalar_clears += 1
    work.branch_field_subtractions += 1


def produce_intersection_scalar(
    intersection: list[su2.K],
    twist: list[su2.K],
    cubic: list[su2.K],
    work: Work,
    rails: Sequence[list[su2.K]],
) -> None:
    require_scalar_zero(intersection, "intersection")
    product = work.multiply(twist[0], cubic[0])
    updated = work.add(intersection[0], product)
    work.observe_relation(
        rails,
        transients=(product, updated),
        context="NATIVE_SCALAR_HADAMARD_INTERSECTION",
    )
    intersection[0] = updated
    work.intersection_scalar_productions += 1
    work.branch_field_multiplications += 1
    work.branch_field_additions += 1


def clear_intersection_scalar(
    intersection: list[su2.K],
    twist: list[su2.K],
    cubic: list[su2.K],
    work: Work,
    rails: Sequence[list[su2.K]],
) -> None:
    product = work.multiply(twist[0], cubic[0])
    updated = work.subtract(intersection[0], product)
    work.observe_relation(
        rails,
        transients=(product, updated),
        context="CLEAR_SCALAR_HADAMARD_INTERSECTION",
    )
    intersection[0] = updated
    require_scalar_zero(intersection, "intersection")
    work.intersection_scalar_clears += 1
    work.branch_field_multiplications += 1
    work.branch_field_subtractions += 1


def accumulate_boundary_scalar(
    boundary: list[su2.K],
    intersection: list[su2.K],
    label: int,
    topology: prior.CompiledTopology,
    *,
    subtract: bool,
    work: Work,
    rails: Sequence[list[su2.K]],
) -> None:
    weight, weight_transients = topology_boundary_weight(label, topology, work)
    contribution = work.multiply(weight, intersection[0])
    updated = (
        work.subtract(boundary[0], contribution)
        if subtract
        else work.add(boundary[0], contribution)
    )
    work.observe_relation(
        rails,
        transients=weight_transients + (contribution, updated),
        context=(
            "CLEAR_TRANSPOSED_FINAL_FUSION_BOUNDARY"
            if subtract
            else "STREAM_TRANSPOSED_FINAL_FUSION_BOUNDARY"
        ),
    )
    boundary[0] = updated
    work.boundary_multiplications += 2
    work.boundary_additions += 1
    if subtract:
        work.transposed_fusion_boundary_clears += 1
    else:
        work.transposed_fusion_boundary_accumulations += 1


def close_one_label(
    carrier: "RematerializedCarrier",
    label: int,
    topology: prior.CompiledTopology,
    *,
    subtract_boundary: bool,
    work: Work,
) -> None:
    rails = carrier.rails()
    for name, cell in (
        ("twist", carrier.twist),
        ("cubic", carrier.cubic),
        ("intersection", carrier.intersection),
    ):
        require_scalar_zero(cell, name)
    produce_twist_scalar(
        carrier.twist, carrier.internal, label, topology, work, rails
    )
    produce_cubic_scalar(
        carrier.cubic, carrier.internal, label, topology, work, rails
    )
    produce_intersection_scalar(
        carrier.intersection, carrier.twist, carrier.cubic, work, rails
    )
    accumulate_boundary_scalar(
        carrier.boundary,
        carrier.intersection,
        label,
        topology,
        subtract=subtract_boundary,
        work=work,
        rails=rails,
    )
    clear_intersection_scalar(
        carrier.intersection, carrier.twist, carrier.cubic, work, rails
    )
    clear_cubic_scalar(
        carrier.cubic, carrier.internal, label, topology, work, rails
    )
    clear_twist_scalar(
        carrier.twist, carrier.internal, label, topology, work, rails
    )
    work.scalar_scratch_restore_checks += 1
    if subtract_boundary:
        work.scalar_labels_inverse += 1
    else:
        work.scalar_labels_forward += 1


@dataclass
class RematerializedCarrier:
    source: list[su2.K]
    internal: list[su2.K]
    twist: list[su2.K]
    cubic: list[su2.K]
    intersection: list[su2.K]
    boundary: list[su2.K]
    live: bool = False
    owner: int = 0
    generation: int = 0
    last_restored_generation: int = 0
    stage: int = 0
    sealed_program_commitment: str = ""

    def rails(self) -> tuple[list[su2.K], ...]:
        return (
            self.source,
            self.internal,
            self.twist,
            self.cubic,
            self.intersection,
            self.boundary,
        )

    def lease(
        self, owner: int, generation: int, program: prior.PublicProgram, work: Work
    ) -> prior.CompiledTopology:
        if self.live:
            raise RuntimeError("rematerialized relation carrier already live")
        if len(self.source) != su2.SIMPLE_OBJECTS or len(self.internal) != su2.SIMPLE_OBJECTS:
            raise ValueError("null or wrong-width rematerialized relation carrier")
        prior.require_zero(self.internal, "internal")
        for name, cell in (
            ("twist", self.twist),
            ("cubic", self.cubic),
            ("intersection", self.intersection),
            ("boundary", self.boundary),
        ):
            require_scalar_zero(cell, name)
        if owner <= 0 or generation != self.last_restored_generation + 1:
            raise PermissionError("invalid or stale rematerialized relation lease")
        topology = program.compile()
        self.live = True
        self.owner = owner
        self.generation = generation
        self.stage = 0
        self.sealed_program_commitment = prior.program_commitment(program)
        work.retained_topology_integers = topology.integers()
        work.port_leases += 1
        work.observe_relation(self.rails(), context="REMATERIALIZED_RELATION_LEASE")
        return topology

    def require(
        self, owner: int, generation: int, program: prior.PublicProgram, work: Work
    ) -> prior.CompiledTopology:
        if not self.live:
            raise RuntimeError("rematerialized relation carrier not live")
        work.owner_checks += 1
        if owner != self.owner:
            raise PermissionError("rematerialized relation owner mismatch")
        work.generation_checks += 1
        if generation != self.generation:
            raise PermissionError("rematerialized relation generation mismatch")
        work.program_commitment_checks += 1
        if prior.program_commitment(program) != self.sealed_program_commitment:
            raise ValueError("rematerialized relation program mismatch")
        work.typed_topology_checks += 1
        return program.compile()

    def produce_internal(
        self, owner: int, generation: int, program: prior.PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 0:
            raise ValueError("rematerialized internal production out of order")
        prior.fusion_accumulate(
            self.internal,
            self.source,
            topology.input_fusion,
            subtract=False,
            work=work,
            rails=self.rails(),
            context="PRODUCE_RESIDENT_NONFUNCTIONAL_FUSION_MESSAGE",
        )
        self.stage = 1
        work.internal_message_productions += 1
        work.forward_operations += 1

    def stream_boundary(
        self, owner: int, generation: int, program: prior.PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 1:
            raise ValueError("rematerialized boundary stream out of order")
        require_scalar_zero(self.boundary, "boundary")
        for label in range(su2.SIMPLE_OBJECTS):
            close_one_label(
                self,
                label,
                topology,
                subtract_boundary=False,
                work=work,
            )
        self.stage = 2
        work.forward_operations += 1

    def project(
        self, owner: int, generation: int, program: prior.PublicProgram, work: Work
    ) -> su2.K:
        self.require(owner, generation, program, work)
        if self.stage != 2:
            work.premature_projection_rejections += 1
            raise PermissionError("only final streamed boundary is projectable")
        for name, cell in (
            ("twist", self.twist),
            ("cubic", self.cubic),
            ("intersection", self.intersection),
        ):
            if cell != [su2.ZERO]:
                work.premature_projection_rejections += 1
                raise PermissionError(f"dirty {name} scratch blocks projection")
        return self.boundary[0]

    def project_internal(self) -> None:
        raise PermissionError("resident internal relation message is never projectable")

    def clear_boundary(
        self, owner: int, generation: int, program: prior.PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 2:
            raise ValueError("rematerialized boundary inverse out of order")
        for label in reversed(range(su2.SIMPLE_OBJECTS)):
            close_one_label(
                self,
                label,
                topology,
                subtract_boundary=True,
                work=work,
            )
        require_scalar_zero(self.boundary, "boundary")
        self.stage = 1
        work.inverse_operations += 1

    def clear_internal(
        self, owner: int, generation: int, program: prior.PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 1:
            raise ValueError("rematerialized internal inverse out of order")
        prior.fusion_accumulate(
            self.internal,
            self.source,
            topology.input_fusion,
            subtract=True,
            work=work,
            rails=self.rails(),
            context="CLEAR_RESIDENT_NONFUNCTIONAL_FUSION_MESSAGE",
        )
        prior.require_zero(self.internal, "internal")
        self.stage = 0
        work.internal_message_clears += 1
        work.inverse_operations += 1

    def release(
        self, owner: int, generation: int, program: prior.PublicProgram, work: Work
    ) -> int:
        self.require(owner, generation, program, work)
        if self.stage:
            raise RuntimeError("rematerialized relation released before inverse")
        prior.require_zero(self.internal, "internal")
        for name, cell in (
            ("twist", self.twist),
            ("cubic", self.cubic),
            ("intersection", self.intersection),
            ("boundary", self.boundary),
        ):
            require_scalar_zero(cell, name)
        restored = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.sealed_program_commitment = ""
        self.last_restored_generation = restored
        work.port_releases += 1
        return restored


def make_carrier() -> tuple[RematerializedCarrier, list[su2.K]]:
    source = su2.source_state()
    return (
        RematerializedCarrier(
            source.copy(),
            [su2.ZERO] * su2.SIMPLE_OBJECTS,
            [su2.ZERO],
            [su2.ZERO],
            [su2.ZERO],
            [su2.ZERO],
        ),
        source,
    )


def carrier_restored(
    carrier: RematerializedCarrier, source: list[su2.K], generation: int
) -> bool:
    return (
        carrier.source == source
        and all(value == su2.ZERO for value in carrier.internal)
        and all(cell == [su2.ZERO] for cell in carrier.rails()[2:])
        and not carrier.live
        and carrier.owner == 0
        and carrier.generation == 0
        and carrier.last_restored_generation == generation
        and carrier.stage == 0
        and carrier.sealed_program_commitment == ""
    )


@dataclass
class ClassicalWork:
    field_additions: int = 0
    field_subtractions: int = 0
    field_multiplications: int = 0
    maximum_live_field_cells: int = 0
    maximum_live_payload_bits: int = 0
    maximum_live_context: str = ""
    retained_topology_integers: tuple[int, ...] = field(default_factory=tuple)

    def observe(
        self,
        source: list[su2.K],
        internal: list[su2.K],
        boundary: list[su2.K],
        transients: Sequence[su2.K],
        context: str,
    ) -> None:
        cells = len(source) + len(internal) + len(boundary) + len(transients)
        bits = (
            su2.field_payload_bits(source)
            + su2.field_payload_bits(internal)
            + su2.field_payload_bits(boundary)
            + su2.field_payload_bits(tuple(transients))
            + sum(prior.signed_bits(value) for value in self.retained_topology_integers)
        )
        self.maximum_live_field_cells = max(self.maximum_live_field_cells, cells)
        if bits > self.maximum_live_payload_bits:
            self.maximum_live_payload_bits = bits
            self.maximum_live_context = context


def matched_compact_classical(
    source: list[su2.K],
    program: prior.PublicProgram,
    *,
    cubic_parameter_override: int | None = None,
    output_parameter_override: int | None = None,
    omit_cubic_target: bool = False,
) -> dict[str, Any]:
    topology = program.compile()
    work = ClassicalWork(retained_topology_integers=topology.integers())
    input_phase = su2.K.zeta(topology.input_fusion.parameter)

    def rematerialize_internal(
        label: int, ambient: Sequence[su2.K] = ()
    ) -> su2.K:
        lower = source[label - 1] if label else su2.ZERO
        upper = source[label + 1] if label + 1 < su2.SIMPLE_OBJECTS else su2.ZERO
        neighbor_sum = lower + upper
        value = input_phase * neighbor_sum
        work.field_additions += 1
        work.field_multiplications += 1
        work.observe(
            source,
            [],
            boundary,
            tuple(ambient) + (input_phase, neighbor_sum, value),
            "CLASSICAL_REMATERIALIZE_INTERNAL_SCALAR",
        )
        return value

    boundary = [su2.ZERO]
    for label in range(su2.SIMPLE_OBJECTS):
        internal_value = rematerialize_internal(label)
        twist_phase = su2.twist_multiplier(
            label, topology.twist_branch.parameter, False
        )
        twist_value = twist_phase * internal_value
        work.field_multiplications += 1
        work.observe(
            source,
            [],
            boundary,
            (input_phase, internal_value, twist_phase, twist_value),
            "CLASSICAL_TWIST_SCALAR",
        )
        if label == topology.cubic_branch.target:
            phase = su2.K.zeta(
                topology.cubic_branch.parameter
                if cubic_parameter_override is None
                else cubic_parameter_override
            )
            source_value = rematerialize_internal(
                topology.cubic_branch.source,
                (internal_value, twist_phase, twist_value, phase),
            )
            square = source_value * source_value
            cube = square * source_value
            term = phase * cube
            cubic = (
                internal_value
                if omit_cubic_target
                else internal_value + term
            )
            work.field_multiplications += 3
            work.field_additions += int(not omit_cubic_target)
            work.observe(
                source,
                [],
                boundary,
                (
                    input_phase,
                    internal_value,
                    twist_phase,
                    twist_value,
                    phase,
                    source_value,
                    square,
                    cube,
                    term,
                    cubic,
                ),
                "CLASSICAL_CUBIC_SCALAR",
            )
            del phase, source_value, square, cube, term
        else:
            cubic = internal_value
        product = twist_value * cubic
        work.field_multiplications += 1
        work.observe(
            source,
            [],
            boundary,
            (
                input_phase,
                internal_value,
                twist_phase,
                twist_value,
                cubic,
                product,
            ),
            "CLASSICAL_INTERSECTION_SCALAR",
        )
        del internal_value, twist_phase, twist_value, cubic
        lower_dimension = su2.QUANTUM_DIMENSIONS[label - 1] if label else su2.ZERO
        upper_dimension = (
            su2.QUANTUM_DIMENSIONS[label + 1]
            if label + 1 < su2.SIMPLE_OBJECTS
            else su2.ZERO
        )
        dimension_sum = lower_dimension + upper_dimension
        output_phase = su2.K.zeta(
            topology.output_fusion.parameter
            if output_parameter_override is None
            else output_parameter_override
        )
        weight = output_phase * dimension_sum
        contribution = weight * product
        updated = boundary[0] + contribution
        work.field_additions += 2
        work.field_multiplications += 2
        work.observe(
            source,
            [],
            boundary,
            (
                input_phase,
                product,
                dimension_sum,
                output_phase,
                weight,
                contribution,
                updated,
            ),
            "CLASSICAL_TRANSPOSED_FUSION_BOUNDARY",
        )
        boundary[0] = updated
        del (
            lower_dimension,
            upper_dimension,
            dimension_sum,
            output_phase,
            weight,
            contribution,
            updated,
            product,
        )
    return {
        "boundary_commitment": su2.boundary_commitment(boundary[0]),
        "public_input_field_cells": len(source),
        "working_backing_field_cells": len(boundary),
        "total_backing_field_cells_including_public_input": len(source)
        + len(boundary),
        "maximum_declared_live_field_cells": work.maximum_live_field_cells,
        "maximum_declared_live_payload_bits": work.maximum_live_payload_bits,
        "maximum_declared_live_context": work.maximum_live_context,
        "field_additions": work.field_additions,
        "field_subtractions": work.field_subtractions,
        "field_multiplications": work.field_multiplications,
        "retained_public_topology_descriptor_integers": len(topology.integers()),
        "recurrence": "SOURCE_NEIGHBOR_REMATERIALIZED_SCALAR_BRANCH_INTERSECTION_AND_TRANSPOSED_FUSION_BOUNDARY",
    }


def transaction(
    carrier: RematerializedCarrier,
    source: list[su2.K],
    program: prior.PublicProgram,
) -> dict[str, Any]:
    backings = tuple(id(rail) for rail in carrier.rails())
    internal_backing = id(carrier.internal)
    generation = carrier.last_restored_generation + 1
    owner = 231000 + generation
    work = Work()
    topology = carrier.lease(owner, generation, program, work)
    carrier.produce_internal(owner, generation, program, work)
    carrier.stream_boundary(owner, generation, program, work)
    boundary = carrier.project(owner, generation, program, work)
    boundary_commitment = su2.boundary_commitment(boundary)
    classical = matched_compact_classical(source, program)
    if boundary_commitment != classical["boundary_commitment"]:
        raise RuntimeError("M231 matched compact classical boundary differs")
    work.retained_result_values = (boundary,)
    carrier.clear_boundary(owner, generation, program, work)
    carrier.clear_internal(owner, generation, program, work)
    restored_generation = carrier.release(owner, generation, program, work)
    return {
        "family": program.family,
        "program_commitment": prior.program_commitment(program),
        "topology_descriptor_integers": len(topology.integers()),
        "actual_resident_internal_message_cells": len(carrier.internal),
        "scalar_branch_intersection_backing_cells": 3,
        "final_boundary_backing_cells": len(carrier.boundary),
        "phase_work_backing_cells_excluding_public_input": sum(
            len(rail) for rail in carrier.rails()[1:]
        ),
        "phase_total_backing_cells_including_public_input": sum(
            len(rail) for rail in carrier.rails()
        ),
        "distinct_scalar_backings": len({id(cell) for cell in carrier.rails()[2:]})
        == 4,
        "same_internal_backing_consumed_by_both_branches": work.internal_consumer_mask
        == 3
        and id(carrier.internal) == internal_backing,
        "boundary_commitment": boundary_commitment,
        "same_all_backings": tuple(id(rail) for rail in carrier.rails()) == backings,
        "canonical_post_restoration_state_exact": carrier_restored(
            carrier, source, restored_generation
        ),
        "restoration_generation": restored_generation,
        "baseline_reload_used": False,
        "work": work.as_dict(),
        "matched_compact_classical": classical,
    }


def normalized_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "family",
            "program_commitment",
            "topology_descriptor_integers",
            "actual_resident_internal_message_cells",
            "scalar_branch_intersection_backing_cells",
            "final_boundary_backing_cells",
            "phase_work_backing_cells_excluding_public_input",
            "phase_total_backing_cells_including_public_input",
            "boundary_commitment",
            "canonical_post_restoration_state_exact",
            "restoration_generation",
            "baseline_reload_used",
        )
    }


def controls() -> dict[str, bool]:
    program = prior.PublicProgram(0)
    carrier, source = make_carrier()
    work = Work()
    owner, generation = 231900, 1
    carrier.lease(owner, generation, program, work)
    wrong_owner = wrong_generation = wrong_program = premature = False
    internal_projection = dirty_scratch = dirty_projection = reordered = False
    wrong_type = release_before_inverse = False
    try:
        carrier.produce_internal(owner + 1, generation, program, work)
    except PermissionError:
        wrong_owner = True
    try:
        carrier.produce_internal(owner, generation + 1, program, work)
    except PermissionError:
        wrong_generation = True
    try:
        carrier.produce_internal(owner, generation, prior.PublicProgram(1), work)
    except ValueError:
        wrong_program = True
    try:
        carrier.project(owner, generation, program, work)
    except PermissionError:
        premature = True
    try:
        carrier.project_internal()
    except PermissionError:
        internal_projection = True
    carrier.produce_internal(owner, generation, program, work)
    carrier.twist[0] = su2.ONE
    try:
        carrier.stream_boundary(owner, generation, program, work)
    except ValueError:
        dirty_scratch = True
    carrier.twist[0] = su2.ZERO
    carrier.stream_boundary(owner, generation, program, work)
    carrier.intersection[0] = su2.ONE
    try:
        carrier.project(owner, generation, program, work)
    except PermissionError:
        dirty_projection = True
    carrier.intersection[0] = su2.ZERO
    missing_inverse = carrier.stage == 2 and carrier.boundary[0] != su2.ZERO
    try:
        carrier.release(owner, generation, program, work)
    except RuntimeError:
        release_before_inverse = True
    try:
        carrier.clear_internal(owner, generation, program, work)
    except ValueError:
        reordered = True
    carrier.clear_boundary(owner, generation, program, work)
    carrier.clear_internal(owner, generation, program, work)
    carrier.release(owner, generation, program, work)
    stale = False
    try:
        carrier.lease(owner + 1, generation, program, Work())
    except PermissionError:
        stale = True
    null = False
    try:
        RematerializedCarrier([], [], [], [], [], []).lease(1, 1, program, Work())
    except ValueError:
        null = True
    try:
        prior.FusionSignature(3, prior.SOURCE_TYPE, prior.SOURCE_TYPE)
    except TypeError:
        wrong_type = True
    original = matched_compact_classical(source, program)["boundary_commitment"]
    perturbed_source = source.copy()
    perturbed_source[2] = su2.K.zeta(11)
    perturbed = matched_compact_classical(perturbed_source, program)[
        "boundary_commitment"
    ]
    cubic_parameter_semantic = output_parameter_semantic = omitted_target_semantic = True
    for family in FAMILIES:
        family_program = prior.PublicProgram(family)
        topology = family_program.compile()
        baseline = matched_compact_classical(source, family_program)[
            "boundary_commitment"
        ]
        cubic_parameter_semantic &= baseline != matched_compact_classical(
            source,
            family_program,
            cubic_parameter_override=10 - topology.cubic_branch.parameter,
        )["boundary_commitment"]
        output_parameter_semantic &= baseline != matched_compact_classical(
            source,
            family_program,
            output_parameter_override=10 - topology.output_fusion.parameter,
        )["boundary_commitment"]
        omitted_target_semantic &= baseline != matched_compact_classical(
            source, family_program, omit_cubic_target=True
        )["boundary_commitment"]
    family_results = [
        transaction(*make_carrier(), prior.PublicProgram(family))
        for family in FAMILIES
    ]
    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_generation_rejected": wrong_generation,
        "wrong_public_program_rejected": wrong_program,
        "premature_projection_rejected": premature,
        "internal_projection_rejected": internal_projection,
        "dirty_scalar_scratch_rejected": dirty_scratch,
        "dirty_scalar_scratch_projection_rejected": dirty_projection,
        "missing_inverse_detected": missing_inverse,
        "release_before_inverse_rejected": release_before_inverse,
        "reordered_dependent_inverse_rejected": reordered,
        "stale_generation_rejected": stale,
        "null_carrier_rejected": null,
        "wrong_port_type_rejected": wrong_type,
        "semantic_source_perturbation_changes_boundary": original != perturbed,
        "cubic_parameter_perturbation_changes_boundary_all_families": cubic_parameter_semantic,
        "output_phase_perturbation_changes_boundary_all_families": output_parameter_semantic,
        "omitted_cubic_target_changes_boundary_all_families": omitted_target_semantic,
        "both_public_families_restore": all(
            case["canonical_post_restoration_state_exact"] for case in family_results
        ),
        "public_topology_compilation_reads_final_answer": False,
        "relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_open_fusion_rematerialized_stream.py REFERENCE_JSON"
        )
    here = Path(__file__).resolve().parent
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M231 reference forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_open_fusion_rematerialized_reference.v1":
        raise RuntimeError("M231 reference schema mismatch")
    current_controls = controls()
    if current_controls != reference.get("controls"):
        raise RuntimeError("M231 control parity failed")
    cases = [
        transaction(*make_carrier(), prior.PublicProgram(family))
        for family in FAMILIES
    ]
    if [normalized_case(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M231 independent case parity failed")
    carrier, source = make_carrier()
    primary = transaction(carrier, source, prior.PublicProgram(PRIMARY_FAMILY))
    reuse = transaction(carrier, source, prior.PublicProgram(REUSE_FAMILY))
    fresh, fresh_source = make_carrier()
    fresh_reuse = transaction(
        fresh, fresh_source, prior.PublicProgram(REUSE_FAMILY)
    )
    reuse_result = {
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh_reuse,
        "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"]
        == fresh_reuse["boundary_commitment"],
        "restoration_generation_after_reuse": carrier.last_restored_generation,
    }
    for key in ("primary", "reuse", "fresh_reuse"):
        if normalized_case(reuse_result[key]) != reference["reuse"][key]:
            raise RuntimeError(f"M231 reuse parity failed: {key}")
    for key in (
        "fresh_restored_reuse_boundary_agreement",
        "restoration_generation_after_reuse",
    ):
        if reuse_result[key] != reference["reuse"][key]:
            raise RuntimeError(f"M231 top-level reuse parity failed: {key}")
    primary_case = cases[PRIMARY_FAMILY]
    result = {
        "schema": "cat_cas.su2_level8_open_fusion_rematerialized_stream.v1",
        "result": "PASS_BOUNDED_EXACT_OPEN_FUSION_REMATERIALIZED_STREAM_REDUCES_RETAINED_RAILS_BUT_CLASSICAL_REMAINS_SMALLER",
        "claim": "BOUNDED_EXACT_TOPOLOGY_REMATERIALIZED_SU2_LEVEL8_NONFUNCTIONAL_FUSION_SIGNATURE_RETAINS_ONE_SHARED9_CELL_INTERNAL_RELATION_MESSAGE_AND_USES_THREE_SCALAR_BRANCH_INTERSECTION_BACKINGS_PLUS_ONE_FINAL_BOUNDARY_BACKING_WITHOUT_RELATION_TABLES_WITH_FINAL_ONLY_EXACT_SAME_BACKING_RESTORATION_AND_REUSE_REDUCING_NONINPUT_WORK_BACKINGS_FROM45_TO13_BUT_THE_STRONGEST_SOURCE_REMATERIALIZED_STREAM_USES1_AND_THE_IDENTICAL_CLASSICAL_RECURRENCE_REMAINS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "QZETA40_SU2_LEVEL8_FIXED_FUNDAMENTAL_FUSION_TWO_PUBLIC_TWIST_CUBIC_FAMILIES_ONE9_CELL_INTERNAL_THREE1_CELL_BRANCH_SCRATCH_ONE1_CELL_BOUNDARY_PRIMARY0_REUSE1_DIRECT_PROCESS_ONLY",
        "controls": current_controls,
        "cases": cases,
        "reuse": reuse_result,
        "mechanism_law": {
            "resident_unresolved_internal_message_cells": 9,
            "distinct_twist_cubic_intersection_scalar_backings": 3,
            "final_boundary_backing_cells": 1,
            "same_internal_message_consumed_by_both_branches": True,
            "native_scalar_hadamard_intersection": True,
            "topology_derived_transposed_final_fusion": True,
            "branch_scratch_cleared_after_each_label": True,
            "branch_vectors_materialized": False,
            "output_vector_materialized": False,
            "internal_message_projected": False,
            "final_boundary_only": True,
            "direct_process_logical_custody_only": True,
        },
        "resource_law": {
            "m230_predecessor_work_backing_cells_excluding_public_input": 45,
            "m231_work_backing_cells_excluding_public_input": primary_case[
                "phase_work_backing_cells_excluding_public_input"
            ],
            "m231_total_backing_cells_including_public_input": primary_case[
                "phase_total_backing_cells_including_public_input"
            ],
            "matched_source_rematerialized_classical_work_backing_cells_excluding_public_input": primary_case[
                "matched_compact_classical"
            ]["working_backing_field_cells"],
            "matched_source_rematerialized_classical_total_backing_cells_including_public_input": primary_case[
                "matched_compact_classical"
            ]["total_backing_field_cells_including_public_input"],
            "primary_maximum_declared_live_field_cells": primary_case["work"][
                "maximum_declared_live_field_cells"
            ],
            "primary_maximum_declared_live_payload_bits": primary_case["work"][
                "maximum_declared_live_payload_bits"
            ],
            "matched_classical_maximum_declared_live_field_cells": primary_case[
                "matched_compact_classical"
            ]["maximum_declared_live_field_cells"],
            "matched_classical_maximum_declared_live_payload_bits": primary_case[
                "matched_compact_classical"
            ]["maximum_declared_live_payload_bits"],
            "retained_public_topology_descriptor_integers": primary_case[
                "topology_descriptor_integers"
            ],
            "resource_measurement_verification_level": "PACKAGE_SELF_REVIEW",
            "whole_transaction_live_payload_complete": False,
            "projected_boundary_retention_during_inverse_counted": True,
            "excluded_not_zero": "PYTHON_OBJECT_CONTAINER_ALLOCATOR_INTERPRETER_JSON_SERIALIZATION_TIMING_PROCESS_RSS_AND_IMPORTED_PUBLIC_FIELD_DESCRIPTOR_STORAGE",
        },
        "matched_compact_classical": {
            "strongest": "SOURCE_NEIGHBOR_REMATERIALIZED_SCALAR_BRANCH_INTERSECTION_AND_TRANSPOSED_FUSION_BOUNDARY",
            "boundary_agreement_all_cases": True,
            "classical_work_backings_are_smaller": primary_case[
                "matched_compact_classical"
            ]["working_backing_field_cells"]
            < primary_case["phase_work_backing_cells_excluding_public_input"],
            "classical_total_backings_are_smaller": primary_case[
                "matched_compact_classical"
            ]["total_backing_field_cells_including_public_input"]
            < primary_case["phase_total_backing_cells_including_public_input"],
            "identical_algebraic_recurrence": True,
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "separate_reference": {
            "imports_m231_production": reference.get("imports_m231_production"),
            "imports_m230_production": reference.get("imports_m230_production"),
            "imports_m211_production": reference.get("imports_m211_production"),
            "uses_independent_polynomial_quotient": reference.get(
                "uses_independent_polynomial_quotient"
            ),
            "case_control_restoration_reuse_parity": True,
        },
        "claim_limits": {
            "arbitrary_su2_relation_signatures": False,
            "growing_interface_width": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m211_production_sha256": sha256_file(
                here / "root_of_unity_su2_level8_fusion_phase_relation.py"
            ),
            "m230_production_sha256": sha256_file(
                here / "su2_level8_open_fusion_signature_fanout_intersection.py"
            ),
            "m231_production_sha256": sha256_file(Path(__file__).resolve()),
            "m231_reference_code_sha256": sha256_file(
                here / "su2_level8_open_fusion_rematerialized_stream_separate_reference.py"
            ),
            "m231_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
