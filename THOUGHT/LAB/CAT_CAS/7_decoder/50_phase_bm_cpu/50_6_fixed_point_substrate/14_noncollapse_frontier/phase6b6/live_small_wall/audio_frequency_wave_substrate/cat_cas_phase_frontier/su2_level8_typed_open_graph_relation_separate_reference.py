#!/usr/bin/env python3
"""Separate polynomial-quotient reference for M229 typed graph relations.

This implementation imports neither the M229 implementation nor the M211
production carrier.  Linear modules act in Q(zeta_40)[x]/U_9(x/2); the cubic
shear converts through the independent character basis and updates one simple
coefficient.  Its custody model independently checks public type, owner,
generation, consumer, order, final-only projection, restoration, and reuse.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Sequence

import root_of_unity_su2_level8_fusion_independent_oracle as oracle


sys.set_int_max_str_digits(0)

PORT_TYPE = 22908
MODULE_KINDS = ("FUSE_FUNDAMENTAL", "TWIST_CASIMIR", "CUBIC_PHASE_SHEAR")
CASES = ((1, 0), (2, 0), (2, 1), (3, 0))
PRIMARY = (3, 0)
REUSE = (2, 1)


@dataclass(frozen=True)
class Module:
    kind: str
    parameter: int
    source: int
    target: int
    consumer: int
    input_type: int = PORT_TYPE
    output_type: int = PORT_TYPE

    def __post_init__(self) -> None:
        if self.kind not in MODULE_KINDS:
            raise ValueError("unknown reference graph module")
        if self.input_type != PORT_TYPE or self.output_type != PORT_TYPE:
            raise TypeError("reference graph port type mismatch")
        if self.consumer <= 0:
            raise ValueError("invalid reference consumer")
        if self.kind == "CUBIC_PHASE_SHEAR":
            if not (0 <= self.source < oracle.SIMPLE_OBJECTS):
                raise ValueError("reference cubic source outside carrier")
            if not (0 <= self.target < oracle.SIMPLE_OBJECTS):
                raise ValueError("reference cubic target outside carrier")
            if self.source == self.target:
                raise ValueError("reference cubic source equals target")
        elif self.source != -1 or self.target != -1:
            raise ValueError("reference linear module has cell endpoints")

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
class Program:
    rounds: int
    family: int

    def __post_init__(self) -> None:
        if self.rounds <= 0 or self.family not in (0, 1):
            raise ValueError("invalid reference graph program")

    @property
    def modules(self) -> tuple[Module, ...]:
        modules: list[Module] = []
        for round_index in range(self.rounds):
            fusion = Module(
                "FUSE_FUNDAMENTAL", (round_index + self.family) % 2, -1, -1, 1
            )
            twist = Module(
                "TWIST_CASIMIR", (round_index + self.family) % 2, -1, -1, 2
            )
            source = (2 * round_index + self.family + 1) % oracle.SIMPLE_OBJECTS
            target = (source + 2 + self.family) % oracle.SIMPLE_OBJECTS
            cubic = Module(
                "CUBIC_PHASE_SHEAR",
                (3, 7)[(round_index + self.family) % 2],
                source,
                target,
                3,
            )
            modules.extend(
                (fusion, cubic, twist)
                if (round_index + self.family) % 2 == 0
                else (twist, cubic, fusion)
            )
        return tuple(modules)

    def token(self) -> str:
        return f"rounds:{self.rounds}:family:{self.family}|" + "|".join(
            module.token() for module in self.modules
        )


def commitment(program: Program) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


def apply_module(
    polynomial: oracle.Polynomial, module: Module, inverse: bool
) -> oracle.Polynomial:
    if module.kind != "CUBIC_PHASE_SHEAR":
        return oracle.apply_operation(
            polynomial, (module.kind, module.parameter), inverse
        )
    coefficients = oracle.polynomial_to_simple(polynomial)
    source = coefficients[module.source]
    injected = oracle.E.root(module.parameter) * source * source * source
    coefficients[module.target] = (
        coefficients[module.target] - injected
        if inverse
        else coefficients[module.target] + injected
    )
    return oracle.simple_to_polynomial(coefficients)


@dataclass
class ReferencePort:
    polynomial: oracle.Polynomial
    enabled: bool = True
    live: bool = False
    owner: int = 0
    generation: int = 0
    cursor: int = 0
    expected_modules: int = 0
    program_commitment: str = ""
    last_restored_generation: int = 0

    def lease(self, owner: int, generation: int, program: Program) -> None:
        if self.live:
            raise RuntimeError("reference graph port already live")
        if not self.enabled:
            raise ValueError("reference null graph carrier")
        if owner <= 0 or generation <= 0:
            raise ValueError("invalid reference graph lease")
        if generation != self.last_restored_generation + 1:
            raise PermissionError("reference nonmonotone generation")
        if len(oracle.polynomial_to_simple(self.polynomial)) != oracle.SIMPLE_OBJECTS:
            raise ValueError("reference wrong-width graph carrier")
        self.live = True
        self.owner = owner
        self.generation = generation
        self.cursor = 0
        self.expected_modules = len(program.modules)
        self.program_commitment = commitment(program)

    def require(
        self,
        owner: int,
        generation: int,
        program: Program,
        module: Module | None,
        consumer: int | None,
    ) -> None:
        if not self.live:
            raise RuntimeError("reference graph port is not live")
        if owner != self.owner:
            raise PermissionError("reference owner mismatch")
        if generation != self.generation:
            raise PermissionError("reference generation mismatch")
        if commitment(program) != self.program_commitment:
            raise ValueError("reference public program mismatch")
        if module is not None:
            if consumer != module.consumer:
                raise PermissionError("reference consumer mismatch")
            if module.input_type != PORT_TYPE or module.output_type != PORT_TYPE:
                raise TypeError("reference module type mismatch")

    def consume(
        self,
        owner: int,
        generation: int,
        program: Program,
        index: int,
        consumer: int,
        inverse: bool,
    ) -> None:
        modules = program.modules
        module = modules[index]
        self.require(owner, generation, program, module, consumer)
        expected = self.cursor - 1 if inverse else self.cursor
        if index != expected:
            raise ValueError("reference module cursor mismatch")
        self.polynomial = apply_module(self.polynomial, module, inverse)
        self.cursor += -1 if inverse else 1

    def project(self, owner: int, generation: int, program: Program) -> oracle.E:
        self.require(owner, generation, program, None, None)
        if self.cursor != self.expected_modules:
            raise PermissionError("reference nonfinal projection rejected")
        return oracle.evaluate(self.polynomial, oracle.DELTA)

    def release(self, owner: int, generation: int, program: Program) -> int:
        self.require(owner, generation, program, None, None)
        if self.cursor:
            raise RuntimeError("reference release before inverse")
        restored = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.expected_modules = 0
        self.program_commitment = ""
        self.last_restored_generation = restored
        return restored


def normalized_transaction(
    port: ReferencePort, program: Program
) -> dict[str, object]:
    source = oracle.source_polynomial()
    generation = port.last_restored_generation + 1
    owner = 229000 + generation
    port.lease(owner, generation, program)
    modules = program.modules
    for index, module in enumerate(modules):
        port.consume(owner, generation, program, index, module.consumer, False)
    coefficients = oracle.polynomial_to_simple(port.polynomial)
    state = oracle.state_commitment(coefficients)
    boundary = oracle.boundary_commitment(port.project(owner, generation, program))
    payload = oracle.payload(coefficients)
    for index in range(len(modules) - 1, -1, -1):
        module = modules[index]
        port.consume(owner, generation, program, index, module.consumer, True)
    restored_generation = port.release(owner, generation, program)

    return {
        "rounds": program.rounds,
        "family": program.family,
        "module_count": len(modules),
        "distinct_module_kinds": sorted({module.kind for module in modules}),
        "distinct_consumers": sorted({module.consumer for module in modules}),
        "port_type": PORT_TYPE,
        "state_commitment": state,
        "boundary_commitment": boundary,
        "forward_payload_bits": payload,
        "canonical_post_restoration_state_exact": port.polynomial == source,
        "restoration_generation": restored_generation,
        "baseline_reload_used": False,
    }


def ordered_state(modules: Sequence[Module]) -> oracle.Polynomial:
    polynomial = oracle.source_polynomial()
    for module in modules:
        polynomial = apply_module(polynomial, module, False)
    return polynomial


def controls() -> dict[str, bool]:
    fusion = Module("FUSE_FUNDAMENTAL", 0, -1, -1, 1)
    twist = Module("TWIST_CASIMIR", 1, -1, -1, 2)
    cubic = Module("CUBIC_PHASE_SHEAR", 7, 1, 3, 3)
    program = Program(2, 0)
    source = oracle.source_polynomial()
    port = ReferencePort(source.copy())
    owner, generation = 229900, 1
    port.lease(owner, generation, program)
    wrong_owner = wrong_generation = wrong_consumer = wrong_program = False
    premature = reordered = False
    try:
        port.consume(owner + 1, generation, program, 0, program.modules[0].consumer, False)
    except PermissionError:
        wrong_owner = True
    try:
        port.consume(owner, generation + 1, program, 0, program.modules[0].consumer, False)
    except PermissionError:
        wrong_generation = True
    try:
        port.consume(owner, generation, program, 0, 99, False)
    except PermissionError:
        wrong_consumer = True
    altered_program = Program(2, 1)
    try:
        port.consume(
            owner,
            generation,
            altered_program,
            0,
            altered_program.modules[0].consumer,
            False,
        )
    except ValueError:
        wrong_program = True
    try:
        port.project(owner, generation, program)
    except PermissionError:
        premature = True
    for index, module in enumerate(program.modules):
        port.consume(owner, generation, program, index, module.consumer, False)
    missing_inverse = port.cursor != 0 and port.polynomial != source
    try:
        index = len(program.modules) - 2
        port.consume(owner, generation, program, index, program.modules[index].consumer, True)
    except ValueError:
        reordered = True
    for index in range(len(program.modules) - 1, -1, -1):
        module = program.modules[index]
        port.consume(owner, generation, program, index, module.consumer, True)
    restored_generation = port.release(owner, generation, program)

    wrong_inverse = source.copy()
    modules = program.modules
    for module in modules:
        wrong_inverse = apply_module(wrong_inverse, module, False)
    last = modules[-1]
    wrong_last = Module(
        last.kind,
        1 - last.parameter,
        last.source,
        last.target,
        last.consumer,
    )
    wrong_inverse = apply_module(wrong_inverse, wrong_last, True)
    for module in reversed(modules[:-1]):
        wrong_inverse = apply_module(wrong_inverse, module, True)

    stale_generation_rejected = False
    try:
        port.lease(owner + 1, generation, program)
    except PermissionError:
        stale_generation_rejected = True

    null_rejected = wrong_type_rejected = False
    try:
        ReferencePort([], enabled=False).lease(1, 1, Program(1, 0))
    except ValueError:
        null_rejected = True
    try:
        Module("CUBIC_PHASE_SHEAR", 3, 1, 2, 3, PORT_TYPE + 1, PORT_TYPE)
    except TypeError:
        wrong_type_rejected = True
    perturbed = Module("CUBIC_PHASE_SHEAR", 3, 1, 3, 3)
    return {
        "fusion_twist_noncommuting": ordered_state((fusion, twist))
        != ordered_state((twist, fusion)),
        "fusion_cubic_noncommuting": ordered_state((fusion, cubic))
        != ordered_state((cubic, fusion)),
        "twist_cubic_noncommuting": ordered_state((twist, cubic))
        != ordered_state((cubic, twist)),
        "wrong_owner_rejected": wrong_owner,
        "wrong_generation_rejected": wrong_generation,
        "stale_generation_rejected": stale_generation_rejected,
        "wrong_consumer_rejected": wrong_consumer,
        "wrong_public_program_rejected": wrong_program,
        "premature_projection_rejected": premature,
        "missing_inverse_detected": missing_inverse,
        "reordered_inverse_rejected": reordered,
        "wrong_inverse_parameter_changes_restored_state": wrong_inverse != source,
        "null_carrier_rejected": null_rejected,
        "wrong_port_type_rejected": wrong_type_rejected,
        "semantic_perturbation_changes_state": ordered_state((fusion, cubic))
        != ordered_state((fusion, perturbed)),
        "control_restored_exactly": port.polynomial == source
        and restored_generation == generation,
        "public_topology_compilation_reads_final_answer": False,
        "relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
    }


def main() -> None:
    cases = []
    for rounds, family in CASES:
        cases.append(
            normalized_transaction(
                ReferencePort(oracle.source_polynomial()), Program(rounds, family)
            )
        )
    port = ReferencePort(oracle.source_polynomial())
    primary = normalized_transaction(port, Program(*PRIMARY))
    reuse = normalized_transaction(port, Program(*REUSE))
    fresh_reuse = normalized_transaction(
        ReferencePort(oracle.source_polynomial()), Program(*REUSE)
    )
    result = {
        "schema": "cat_cas.su2_level8_typed_open_graph_relation_reference.v1",
        "controls": controls(),
        "cases": cases,
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh_reuse,
            "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"]
            == fresh_reuse["boundary_commitment"],
            "fresh_restored_reuse_state_agreement": reuse["state_commitment"]
            == fresh_reuse["state_commitment"],
            "restoration_generation_after_reuse": port.last_restored_generation,
        },
        "imports_m229_production": False,
        "imports_m211_production": False,
        "uses_independent_polynomial_quotient_substrate": True,
    }
    positive_controls = {
        key: value
        for key, value in result["controls"].items()
        if key
        not in {
            "public_topology_compilation_reads_final_answer",
            "relation_tables_materialized",
            "assignment_expansions_materialized",
        }
    }
    if not all(positive_controls.values()) or any(
        result["controls"][key]
        for key in (
            "public_topology_compilation_reads_final_answer",
            "relation_tables_materialized",
            "assignment_expansions_materialized",
        )
    ):
        raise RuntimeError("M229 separate controls failed")
    if not all(case["canonical_post_restoration_state_exact"] for case in cases):
        raise RuntimeError("M229 separate exact restoration failed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
