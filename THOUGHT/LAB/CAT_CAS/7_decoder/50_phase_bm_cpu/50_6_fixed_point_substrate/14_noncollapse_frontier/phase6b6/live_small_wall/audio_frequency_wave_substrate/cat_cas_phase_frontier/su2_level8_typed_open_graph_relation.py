#!/usr/bin/env python3
"""M229 exact typed open graph-relation composition on SU(2)_8.

The shared port is the actual nine-cell Q(zeta_40) fusion carrier.  Public
fusion, twist, and reversible cubic-shear graph relations consume that same
unresolved carrier without materializing a relation table or assignments.
Only the final quantum-dimension boundary is projected; the exact public word
is then reversed on the same backings and the restored carrier is reused.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import root_of_unity_su2_level8_fusion_phase_relation as su2


sys.set_int_max_str_digits(0)

PORT_TYPE = 22908
MODULE_KINDS = ("FUSE_FUNDAMENTAL", "TWIST_CASIMIR", "CUBIC_PHASE_SHEAR")
CASES = ((1, 0), (2, 0), (2, 1), (3, 0))
PRIMARY = (3, 0)
REUSE = (2, 1)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


@dataclass(frozen=True)
class GraphModule:
    kind: str
    parameter: int
    source: int
    target: int
    consumer: int
    input_type: int = PORT_TYPE
    output_type: int = PORT_TYPE

    def __post_init__(self) -> None:
        if self.kind not in MODULE_KINDS:
            raise ValueError("unknown SU2 graph-relation module")
        if self.input_type != PORT_TYPE or self.output_type != PORT_TYPE:
            raise TypeError("SU2 graph-relation port type mismatch")
        if self.consumer <= 0:
            raise ValueError("invalid graph-relation consumer")
        if self.kind == "CUBIC_PHASE_SHEAR":
            if not (0 <= self.source < su2.SIMPLE_OBJECTS):
                raise ValueError("cubic source outside SU2 carrier")
            if not (0 <= self.target < su2.SIMPLE_OBJECTS):
                raise ValueError("cubic target outside SU2 carrier")
            if self.source == self.target:
                raise ValueError("cubic shear must preserve a distinct source")
        elif self.source != -1 or self.target != -1:
            raise ValueError("linear SU2 modules do not declare cell endpoints")

    def integers(self) -> tuple[int, ...]:
        return (
            MODULE_KINDS.index(self.kind),
            self.parameter,
            self.source,
            self.target,
            self.consumer,
            self.input_type,
            self.output_type,
        )

    def token(self) -> str:
        return ":".join(str(value) for value in self.integers())


@dataclass(frozen=True)
class GraphProgram:
    rounds: int
    family: int

    def __post_init__(self) -> None:
        if self.rounds <= 0 or self.family not in (0, 1):
            raise ValueError("invalid public SU2 graph program")

    @property
    def modules(self) -> tuple[GraphModule, ...]:
        result: list[GraphModule] = []
        for round_index in range(self.rounds):
            fusion = GraphModule(
                "FUSE_FUNDAMENTAL", (round_index + self.family) % 2, -1, -1, 1
            )
            twist = GraphModule(
                "TWIST_CASIMIR", (round_index + self.family) % 2, -1, -1, 2
            )
            source = (2 * round_index + self.family + 1) % su2.SIMPLE_OBJECTS
            target = (source + 2 + self.family) % su2.SIMPLE_OBJECTS
            cubic = GraphModule(
                "CUBIC_PHASE_SHEAR",
                (3, 7)[(round_index + self.family) % 2],
                source,
                target,
                3,
            )
            ordered = (
                (fusion, cubic, twist)
                if (round_index + self.family) % 2 == 0
                else (twist, cubic, fusion)
            )
            result.extend(ordered)
        return tuple(result)

    def token(self) -> str:
        return f"rounds:{self.rounds}:family:{self.family}|" + "|".join(
            module.token() for module in self.modules
        )


def program_commitment(program: GraphProgram) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


@dataclass
class Work(su2.Work):
    graph_relation_module_consumptions: int = 0
    graph_relation_inverse_consumptions: int = 0
    graph_relation_distinct_consumers_mask: int = 0
    cubic_shears: int = 0
    cubic_inverse_shears: int = 0
    cubic_field_squares: int = 0
    cubic_field_cubes: int = 0
    cubic_phase_multiplications: int = 0
    cubic_accumulations: int = 0
    typed_module_checks: int = 0
    generation_checks: int = 0
    ownership_checks: int = 0
    program_commitment_checks: int = 0
    graph_relation_tables_materialized: int = 0
    assignment_expansions_materialized: int = 0
    maximum_graph_descriptor_integer_cells: int = 0
    maximum_graph_descriptor_payload_bits: int = 0
    maximum_declared_live_field_cells: int = 0
    maximum_declared_live_payload_bits: int = 0
    maximum_declared_live_context: str = ""
    retained_graph_descriptor_integers: tuple[int, ...] = field(
        default_factory=tuple, repr=False
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "retained_graph_descriptor_integers"
        }

    def observe_graph(
        self,
        coefficients: list[su2.K],
        scratch: list[su2.K],
        *,
        transients: Sequence[su2.K] = (),
        descriptor_integers: Sequence[int] = (),
        context: str,
    ) -> None:
        logical_descriptors = (
            self.retained_graph_descriptor_integers or tuple(descriptor_integers)
        )
        descriptor_bits = sum(signed_bits(value) for value in logical_descriptors)
        payload = (
            su2.field_payload_bits(coefficients)
            + su2.field_payload_bits(scratch)
            + su2.field_payload_bits(tuple(transients))
            + descriptor_bits
        )
        self.maximum_graph_descriptor_integer_cells = max(
            self.maximum_graph_descriptor_integer_cells, len(logical_descriptors)
        )
        self.maximum_graph_descriptor_payload_bits = max(
            self.maximum_graph_descriptor_payload_bits, descriptor_bits
        )
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells,
            len(coefficients) + len(scratch) + len(transients),
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context
        self.observe(coefficients, scratch)


def apply_module(
    coefficients: list[su2.K],
    scratch: list[su2.K],
    module: GraphModule,
    *,
    inverse: bool,
    work: Work,
) -> None:
    work.typed_module_checks += 1
    if module.input_type != PORT_TYPE or module.output_type != PORT_TYPE:
        raise TypeError("typed SU2 module rejected")
    if module.kind != "CUBIC_PHASE_SHEAR":
        su2.apply_operation(
            coefficients,
            scratch,
            su2.Operation(module.kind, module.parameter),
            inverse,
            work,
        )
        work.observe_graph(
            coefficients,
            scratch,
            descriptor_integers=module.integers(),
            context=f"GRAPH_{module.kind}_{'INVERSE' if inverse else 'FORWARD'}",
        )
        return
    source_value = coefficients[module.source]
    # The reversible shear is y <- y + zeta^p x^3.  Its inverse subtracts
    # the same injected field value; conjugating the public phase would define
    # a different map rather than reverse this one.
    phase = su2.K.zeta(module.parameter)
    square = work.multiply(source_value, source_value)
    cube = work.multiply(square, source_value)
    injected = work.multiply(phase, cube)
    updated = (
        work.subtract(coefficients[module.target], injected)
        if inverse
        else work.add(coefficients[module.target], injected)
    )
    work.cubic_field_squares += 1
    work.cubic_field_cubes += 1
    work.cubic_phase_multiplications += 1
    work.cubic_accumulations += 1
    work.observe_graph(
        coefficients,
        scratch,
        transients=(phase, square, cube, injected, updated),
        descriptor_integers=module.integers(),
        context="GRAPH_CUBIC_PHASE_SHEAR_INVERSE" if inverse else "GRAPH_CUBIC_PHASE_SHEAR_FORWARD",
    )
    coefficients[module.target] = updated
    if inverse:
        work.cubic_inverse_shears += 1
    else:
        work.cubic_shears += 1


@dataclass
class OpenGraphPort:
    coefficients: list[su2.K]
    scratch: list[su2.K]
    live: bool = False
    owner: int = 0
    generation: int = 0
    cursor: int = 0
    expected_modules: int = 0
    sealed_program_commitment: str = ""
    last_restored_generation: int = 0

    def lease(self, owner: int, generation: int, program: GraphProgram, work: Work) -> None:
        if self.live:
            raise RuntimeError("typed graph port already live")
        if len(self.coefficients) != su2.SIMPLE_OBJECTS:
            raise ValueError("null or wrong-width typed graph carrier")
        if len(self.scratch) != su2.SIMPLE_OBJECTS or any(
            value != su2.ZERO for value in self.scratch
        ):
            raise ValueError("dirty typed graph scratch")
        if owner <= 0 or generation <= 0:
            raise ValueError("invalid typed graph lease")
        if generation != self.last_restored_generation + 1:
            raise PermissionError("nonmonotone typed graph generation")
        modules = program.modules
        work.retained_graph_descriptor_integers = tuple(
            value for module in modules for value in module.integers()
        )
        self.live = True
        self.owner = owner
        self.generation = generation
        self.cursor = 0
        self.expected_modules = len(modules)
        self.sealed_program_commitment = program_commitment(program)
        work.port_leases += 1
        work.observe_graph(
            self.coefficients,
            self.scratch,
            descriptor_integers=tuple(value for module in modules for value in module.integers()),
            context="GRAPH_PORT_LEASE",
        )

    def require(
        self,
        owner: int,
        generation: int,
        program: GraphProgram,
        module: GraphModule | None,
        consumer: int | None,
        work: Work,
    ) -> None:
        if not self.live:
            raise RuntimeError("typed graph port is not live")
        work.ownership_checks += 1
        if owner != self.owner:
            raise PermissionError("typed graph owner mismatch")
        work.generation_checks += 1
        if generation != self.generation:
            raise PermissionError("typed graph generation mismatch")
        work.program_commitment_checks += 1
        if program_commitment(program) != self.sealed_program_commitment:
            raise ValueError("typed graph public program mismatch")
        if module is not None:
            work.typed_module_checks += 1
            if consumer != module.consumer:
                raise PermissionError("typed graph consumer mismatch")
            if module.input_type != PORT_TYPE or module.output_type != PORT_TYPE:
                raise TypeError("typed graph module mismatch")

    def consume(
        self,
        owner: int,
        generation: int,
        program: GraphProgram,
        index: int,
        consumer: int,
        work: Work,
        *,
        inverse: bool,
    ) -> None:
        modules = program.modules
        module = modules[index]
        self.require(owner, generation, program, module, consumer, work)
        expected = self.cursor - 1 if inverse else self.cursor
        if index != expected:
            raise ValueError("typed graph module cursor mismatch")
        apply_module(self.coefficients, self.scratch, module, inverse=inverse, work=work)
        self.cursor += -1 if inverse else 1
        work.graph_relation_distinct_consumers_mask |= 1 << module.consumer
        if inverse:
            work.graph_relation_inverse_consumptions += 1
            work.inverse_operations += 1
        else:
            work.graph_relation_module_consumptions += 1
            work.forward_operations += 1

    def project(
        self, owner: int, generation: int, program: GraphProgram, work: Work
    ) -> su2.K:
        self.require(owner, generation, program, None, None, work)
        if self.cursor != self.expected_modules:
            raise PermissionError("nonfinal typed graph projection rejected")
        return su2.project_quantum_dimension(self.coefficients, work)

    def release(
        self, owner: int, generation: int, program: GraphProgram, work: Work
    ) -> int:
        self.require(owner, generation, program, None, None, work)
        if self.cursor:
            raise RuntimeError("typed graph port released before inverse")
        if any(value != su2.ZERO for value in self.scratch):
            raise RuntimeError("typed graph port released with dirty scratch")
        restored = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.expected_modules = 0
        self.sealed_program_commitment = ""
        self.last_restored_generation = restored
        work.port_releases += 1
        return restored


@dataclass
class Carrier:
    port: OpenGraphPort
    restoration_generation: int = 0


def make_carrier() -> tuple[Carrier, list[su2.K]]:
    source = su2.source_state()
    return Carrier(OpenGraphPort(source.copy(), [su2.ZERO] * su2.SIMPLE_OBJECTS)), source


def canonical_restoration(carrier: Carrier, source: list[su2.K], generation: int) -> bool:
    port = carrier.port
    return (
        port.coefficients == source
        and all(value == su2.ZERO for value in port.scratch)
        and not port.live
        and port.owner == 0
        and port.generation == 0
        and port.cursor == 0
        and port.expected_modules == 0
        and port.sealed_program_commitment == ""
        and port.last_restored_generation == generation
        and carrier.restoration_generation == generation
    )


def classical_forward(program: GraphProgram) -> dict[str, Any]:
    coefficients = su2.source_state()
    scratch = [su2.ZERO] * su2.SIMPLE_OBJECTS
    work = Work()
    modules = program.modules
    work.retained_graph_descriptor_integers = tuple(
        value for module in modules for value in module.integers()
    )
    for module in modules:
        apply_module(coefficients, scratch, module, inverse=False, work=work)
    boundary = su2.project_quantum_dimension(coefficients, work)
    return {
        "state_commitment": su2.state_commitment(coefficients),
        "boundary_commitment": su2.boundary_commitment(boundary),
        "resident_field_cells": len(coefficients) + len(scratch),
        "recurrence": "IDENTICAL_NINE_COORDINATE_FUSION_TWIST_CUBIC_SHEAR",
    }


def transaction(carrier: Carrier, source: list[su2.K], program: GraphProgram) -> dict[str, Any]:
    coefficient_backing = id(carrier.port.coefficients)
    scratch_backing = id(carrier.port.scratch)
    generation = carrier.restoration_generation + 1
    owner = 229000 + generation
    work = Work()
    carrier.port.lease(owner, generation, program, work)
    modules = program.modules
    for index, module in enumerate(modules):
        carrier.port.consume(
            owner, generation, program, index, module.consumer, work, inverse=False
        )
    state_commitment = su2.state_commitment(carrier.port.coefficients)
    boundary = carrier.port.project(owner, generation, program, work)
    boundary_commitment = su2.boundary_commitment(boundary)
    forward_payload_bits = su2.field_payload_bits(carrier.port.coefficients)
    for index in range(len(modules) - 1, -1, -1):
        module = modules[index]
        carrier.port.consume(
            owner, generation, program, index, module.consumer, work, inverse=True
        )
    carrier.restoration_generation = carrier.port.release(
        owner, generation, program, work
    )
    classical = classical_forward(program)
    if state_commitment != classical["state_commitment"]:
        raise RuntimeError("typed graph state differs from compact recurrence")
    if boundary_commitment != classical["boundary_commitment"]:
        raise RuntimeError("typed graph boundary differs from compact recurrence")
    return {
        "rounds": program.rounds,
        "family": program.family,
        "module_count": len(modules),
        "distinct_module_kinds": sorted({module.kind for module in modules}),
        "distinct_consumers": sorted({module.consumer for module in modules}),
        "port_type": PORT_TYPE,
        "state_commitment": state_commitment,
        "boundary_commitment": boundary_commitment,
        "forward_payload_bits": forward_payload_bits,
        "same_coefficient_backing": id(carrier.port.coefficients) == coefficient_backing,
        "same_scratch_backing": id(carrier.port.scratch) == scratch_backing,
        "canonical_post_restoration_state_exact": canonical_restoration(
            carrier, source, generation
        ),
        "restoration_generation": carrier.restoration_generation,
        "baseline_reload_used": False,
        "work": work.as_dict(),
        "matched_compact_classical": classical,
    }


def normalized_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "rounds",
            "family",
            "module_count",
            "distinct_module_kinds",
            "distinct_consumers",
            "port_type",
            "state_commitment",
            "boundary_commitment",
            "forward_payload_bits",
            "canonical_post_restoration_state_exact",
            "restoration_generation",
            "baseline_reload_used",
        )
    }


def ordered_state(modules: Sequence[GraphModule]) -> list[su2.K]:
    coefficients = su2.source_state()
    scratch = [su2.ZERO] * su2.SIMPLE_OBJECTS
    work = Work()
    work.retained_graph_descriptor_integers = tuple(
        value for module in modules for value in module.integers()
    )
    for module in modules:
        apply_module(coefficients, scratch, module, inverse=False, work=work)
    return coefficients


def controls() -> dict[str, bool]:
    fusion = GraphModule("FUSE_FUNDAMENTAL", 0, -1, -1, 1)
    twist = GraphModule("TWIST_CASIMIR", 1, -1, -1, 2)
    cubic = GraphModule("CUBIC_PHASE_SHEAR", 7, 1, 3, 3)
    fusion_twist = ordered_state((fusion, twist)) != ordered_state((twist, fusion))
    fusion_cubic = ordered_state((fusion, cubic)) != ordered_state((cubic, fusion))
    twist_cubic = ordered_state((twist, cubic)) != ordered_state((cubic, twist))

    program = GraphProgram(2, 0)
    carrier, source = make_carrier()
    port = carrier.port
    work = Work()
    owner, generation = 229900, 1
    port.lease(owner, generation, program, work)
    wrong_owner = wrong_generation = wrong_consumer = wrong_program = False
    premature = reordered = False
    try:
        port.consume(owner + 1, generation, program, 0, program.modules[0].consumer, work, inverse=False)
    except PermissionError:
        wrong_owner = True
    try:
        port.consume(owner, generation + 1, program, 0, program.modules[0].consumer, work, inverse=False)
    except PermissionError:
        wrong_generation = True
    try:
        port.consume(owner, generation, program, 0, 99, work, inverse=False)
    except PermissionError:
        wrong_consumer = True
    altered_program = GraphProgram(2, 1)
    try:
        port.consume(
            owner,
            generation,
            altered_program,
            0,
            altered_program.modules[0].consumer,
            work,
            inverse=False,
        )
    except ValueError:
        wrong_program = True
    try:
        port.project(owner, generation, program, work)
    except PermissionError:
        premature = True
    for index, module in enumerate(program.modules):
        port.consume(owner, generation, program, index, module.consumer, work, inverse=False)
    missing_inverse = port.cursor != 0 and port.coefficients != source
    try:
        index = len(program.modules) - 2
        port.consume(owner, generation, program, index, program.modules[index].consumer, work, inverse=True)
    except ValueError:
        reordered = True
    for index in range(len(program.modules) - 1, -1, -1):
        module = program.modules[index]
        port.consume(owner, generation, program, index, module.consumer, work, inverse=True)
    restored_generation = port.release(owner, generation, program, work)

    wrong_inverse_coefficients = su2.source_state()
    wrong_inverse_scratch = [su2.ZERO] * su2.SIMPLE_OBJECTS
    wrong_inverse_work = Work()
    modules = program.modules
    for module in modules:
        apply_module(
            wrong_inverse_coefficients,
            wrong_inverse_scratch,
            module,
            inverse=False,
            work=wrong_inverse_work,
        )
    last = modules[-1]
    wrong_last = GraphModule(
        last.kind,
        1 - last.parameter,
        last.source,
        last.target,
        last.consumer,
    )
    apply_module(
        wrong_inverse_coefficients,
        wrong_inverse_scratch,
        wrong_last,
        inverse=True,
        work=wrong_inverse_work,
    )
    for module in reversed(modules[:-1]):
        apply_module(
            wrong_inverse_coefficients,
            wrong_inverse_scratch,
            module,
            inverse=True,
            work=wrong_inverse_work,
        )
    wrong_inverse_changes_state = wrong_inverse_coefficients != source

    stale_generation_rejected = False
    try:
        port.lease(owner + 1, generation, program, Work())
    except PermissionError:
        stale_generation_rejected = True

    null_rejected = wrong_type_rejected = semantic_perturbation = False
    try:
        OpenGraphPort([], [su2.ZERO] * su2.SIMPLE_OBJECTS).lease(
            1, 1, GraphProgram(1, 0), Work()
        )
    except ValueError:
        null_rejected = True
    try:
        GraphModule("CUBIC_PHASE_SHEAR", 3, 1, 2, 3, PORT_TYPE + 1, PORT_TYPE)
    except TypeError:
        wrong_type_rejected = True
    perturbed = GraphModule("CUBIC_PHASE_SHEAR", 3, 1, 3, 3)
    semantic_perturbation = ordered_state((fusion, cubic)) != ordered_state((fusion, perturbed))
    return {
        "fusion_twist_noncommuting": fusion_twist,
        "fusion_cubic_noncommuting": fusion_cubic,
        "twist_cubic_noncommuting": twist_cubic,
        "wrong_owner_rejected": wrong_owner,
        "wrong_generation_rejected": wrong_generation,
        "stale_generation_rejected": stale_generation_rejected,
        "wrong_consumer_rejected": wrong_consumer,
        "wrong_public_program_rejected": wrong_program,
        "premature_projection_rejected": premature,
        "missing_inverse_detected": missing_inverse,
        "reordered_inverse_rejected": reordered,
        "wrong_inverse_parameter_changes_restored_state": wrong_inverse_changes_state,
        "null_carrier_rejected": null_rejected,
        "wrong_port_type_rejected": wrong_type_rejected,
        "semantic_perturbation_changes_state": semantic_perturbation,
        "control_restored_exactly": port.coefficients == source
        and all(value == su2.ZERO for value in port.scratch)
        and restored_generation == generation,
        "public_topology_compilation_reads_final_answer": False,
        "relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: su2_level8_typed_open_graph_relation.py SEPARATE_REFERENCE_JSON")
    here = Path(__file__).resolve().parent
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M229 reference forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_typed_open_graph_relation_reference.v1":
        raise RuntimeError("M229 reference schema mismatch")
    current_controls = controls()
    if reference.get("controls") != current_controls:
        raise RuntimeError("M229 independent controls differ")
    cases: list[dict[str, Any]] = []
    for rounds, family in CASES:
        carrier, source = make_carrier()
        cases.append(transaction(carrier, source, GraphProgram(rounds, family)))
    if [normalized_case(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M229 independent semantic parity failed")

    carrier, source = make_carrier()
    primary = transaction(carrier, source, GraphProgram(*PRIMARY))
    reuse = transaction(carrier, source, GraphProgram(*REUSE))
    fresh, fresh_source = make_carrier()
    fresh_reuse = transaction(fresh, fresh_source, GraphProgram(*REUSE))
    reuse_result = {
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh_reuse,
        "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"]
        == fresh_reuse["boundary_commitment"],
        "fresh_restored_reuse_state_agreement": reuse["state_commitment"]
        == fresh_reuse["state_commitment"],
        "restoration_generation_after_reuse": carrier.restoration_generation,
    }
    reference_reuse = reference["reuse"]
    for section in ("primary", "reuse", "fresh_reuse"):
        if normalized_case(reuse_result[section]) != reference_reuse[section]:
            raise RuntimeError(f"M229 independent reuse parity failed: {section}")
    for key in (
        "fresh_restored_reuse_boundary_agreement",
        "fresh_restored_reuse_state_agreement",
        "restoration_generation_after_reuse",
    ):
        if reuse_result[key] != reference_reuse[key]:
            raise RuntimeError(f"M229 independent top-level reuse parity failed: {key}")

    primary_case = next(
        case for case in cases if (case["rounds"], case["family"]) == PRIMARY
    )
    result = {
        "schema": "cat_cas.su2_level8_typed_open_graph_relation.v1",
        "result": "PASS_BOUNDED_EXACT_TYPED_SHARED_PORT_REVERSIBLE_MAP_EXECUTION_WITH_IDENTICAL_CLASSICAL_RECURRENCE",
        "claim": "BOUNDED_EXACT_TYPED_SHARED_PORT_REVERSIBLE_MAP_CALIBRATION_EXECUTES_NONCOMMUTING_SU2_LEVEL8_FUSION_TWIST_AND_CUBIC_PHASE_SHEAR_ON_ONE_ACTUAL9_CELL_QZETA40_VECTOR_WITHOUT_TABLE_OR_ASSIGNMENT_EXPANSION_WITH_FINAL_STAGE_BOUNDARY_EXACT_SAME_BACKING_RESTORATION_AND_REUSE_BUT_DOES_NOT_CONSTRUCT_OR_CLOSE_OPEN_RELATION_SIGNATURES_AND_COLLAPSES_TO_THE_IDENTICAL_NINE_COORDINATE_CLASSICAL_RECURRENCE",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "QZETA40_SU2_LEVEL8_SIMPLE_OBJECT9_CELL_SEQUENTIAL_REVERSIBLE_MAPS_FUSION_PARAMETERS0_1_TWIST_PARAMETERS0_1_CUBIC_SHEAR_PHASES3_7_PUBLIC_ROUNDS1_2_3_FAMILIES0_1_PRIMARY3_0_REUSE2_1_DIRECT_PROCESS_ONLY_NOT_OPEN_RELATION_SIGNATURE_CLOSURE",
        "controls": current_controls,
        "cases": cases,
        "reuse": reuse_result,
        "execution_law": {
            "actual_shared_port_cells": su2.SIMPLE_OBJECTS,
            "typed_port": PORT_TYPE,
            "sequential_full_vector_execution": True,
            "distinct_public_consumer_labels": 3,
            "actual_independent_consumers": False,
            "nominal_descriptor_consumer_labels": True,
            "noncommuting_module_kinds": list(MODULE_KINDS),
            "modules_are_exact_reversible_maps": True,
            "open_relation_signatures_constructed": False,
            "open_relation_signature_closure": False,
            "shared_unresolved_relational_port_contraction": False,
            "relation_table_cells": 0,
            "assignment_expansions": 0,
            "intermediate_projection": False,
            "final_boundary_only": True,
        },
        "resource_law": {
            "primary_forward_payload_bits": primary_case["forward_payload_bits"],
            "primary_maximum_declared_live_payload_bits": primary_case["work"]["maximum_declared_live_payload_bits"],
            "primary_maximum_declared_live_field_cells": primary_case["work"]["maximum_declared_live_field_cells"],
            "primary_maximum_context": primary_case["work"]["maximum_declared_live_context"],
            "retained_public_module_descriptor_integers_counted": True,
            "primary_retained_public_module_descriptor_integer_cells": primary_case["work"]["maximum_graph_descriptor_integer_cells"],
            "program_commitment_compilation_transients_complete": False,
            "cubic_transients_counted": True,
            "inherited_linear_module_scalar_arithmetic_live_payload_complete": False,
            "whole_transaction_live_payload_complete": False,
            "excluded_not_zero": "PYTHON_OBJECT_CONTAINER_ALLOCATOR_INTERPRETER_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_RSS",
        },
        "matched_classical_baseline": {
            "strongest_compact": "IDENTICAL_NINE_COORDINATE_FUSION_TWIST_CUBIC_SHEAR_RECURRENCE",
            "every_case_state_and_boundary_agreement": True,
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "separate_reference": {
            "imports_m229_production": reference.get("imports_m229_production"),
            "imports_m211_production": reference.get("imports_m211_production"),
            "uses_independent_polynomial_quotient_substrate": reference.get(
                "uses_independent_polynomial_quotient_substrate"
            ),
            "case_and_reuse_semantic_parity": True,
        },
        "claim_limits": {
            "general_open_relation_algebra": False,
            "open_relation_signature_closure": False,
            "nonfunctional_multivalued_relational_closure": False,
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
            "m229_production_sha256": sha256_file(Path(__file__).resolve()),
            "m229_reference_code_sha256": sha256_file(
                here / "su2_level8_typed_open_graph_relation_separate_reference.py"
            ),
            "m229_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
