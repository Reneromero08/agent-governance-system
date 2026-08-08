#!/usr/bin/env python3
"""Standalone sparse-dictionary oracle for M235.

This file deliberately does not import the production implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


PORT_TYPE = "CUBIC_STATIONARY_CRITICAL_LOCUS_RELATION_V1"
ORIENTATION = "LEFT_TO_RIGHT"
DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)
FAMILY_ROOTS = (
    (1, 3, 5, 7, 11, 13, 17, 19),
    (-2, -3, -5, -7, -11, -13, -17, -19),
)
FAMILY_SHIFTS = (
    (2, 5, 11, 17, 23, 31, 41, 47),
    (1, 4, 8, 14, 22, 32, 44, 58),
)


def bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload(values: Iterable[int]) -> int:
    return sum(bits(value) for value in values)


def dense(poly: dict[int, int]) -> list[int]:
    degree = max(poly, default=0)
    return [poly.get(index, 0) for index in range(degree + 1)]


def clean(poly: dict[int, int]) -> dict[int, int]:
    result = {degree: value for degree, value in poly.items() if value != 0}
    return result or {0: 0}


def poly_commit(poly: dict[int, int]) -> str:
    encoded = json.dumps(dense(poly), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def scalar_commit(value: int) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


@dataclass(frozen=True)
class Descriptor:
    root: int
    shift: int
    input_type: str = PORT_TYPE
    output_type: str = PORT_TYPE
    orientation: str = ORIENTATION


@dataclass(frozen=True)
class Program:
    family: int
    depth: int
    rounds: tuple[Descriptor, ...]
    program_id: str
    probe: int

    @staticmethod
    def compile(family: int, depth: int) -> "Program":
        if family not in (0, 1) or depth not in DEPTHS:
            raise ValueError("invalid public program")
        descriptors = []
        for index in range(depth):
            descriptors.append(Descriptor(FAMILY_ROOTS[family][index], FAMILY_SHIFTS[family][index]))
        public = [(item.root, item.shift) for item in descriptors]
        encoded = json.dumps([family, depth, public], separators=(",", ":")).encode()
        return Program(
            family,
            depth,
            tuple(descriptors),
            hashlib.sha256(encoded).hexdigest(),
            2 if family == 0 else -2,
        )


def adjoin_branch(poly: dict[int, int], root: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for degree, value in poly.items():
        result[degree] = result.get(degree, 0) - root * value
        result[degree + 1] = result.get(degree + 1, 0) + value
    return clean(result)


def remove_branch_exact(poly: dict[int, int], root: int) -> dict[int, int]:
    values = dense(poly)
    if len(values) < 2:
        raise RuntimeError("missing linear factor")
    quotient = [0] * (len(values) - 1)
    quotient[-1] = values[-1]
    for index in range(len(quotient) - 1, 0, -1):
        quotient[index - 1] = values[index] + root * quotient[index]
    if values[0] != -root * quotient[0]:
        raise RuntimeError("missing linear factor")
    return clean({degree: value for degree, value in enumerate(quotient)})


def translate(poly: dict[int, int], amount: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for degree, value in poly.items():
        for target in range(degree + 1):
            term = value * math.comb(degree, target) * amount ** (degree - target)
            result[target] = result.get(target, 0) + term
    return clean(result)


def stationary_compose(poly: dict[int, int], shift: int) -> dict[int, int]:
    translated = translate(poly, -shift)
    return clean({2 * degree: value for degree, value in translated.items()})


def stationary_inverse(poly: dict[int, int], shift: int) -> dict[int, int]:
    if any(degree % 2 and value for degree, value in poly.items()):
        raise RuntimeError("outside stationary image")
    collapsed = {degree // 2: value for degree, value in poly.items()}
    return translate(collapsed, shift)


def evaluate(poly: dict[int, int], point: int) -> int:
    return sum(value * point**degree for degree, value in poly.items())


def expected_degree(depth: int) -> int:
    return 3 * (1 << depth) - 2


def baseline(program: Program) -> dict[str, int | str]:
    point = program.probe
    accumulator = 1
    peak = payload((point, accumulator))
    for descriptor in reversed(program.rounds):
        squared = point * point
        next_point = squared - descriptor.shift
        factor = next_point - descriptor.root
        product = accumulator * factor
        peak = max(peak, payload((point, squared, next_point, factor, accumulator, product)))
        point, accumulator = next_point, product
    boundary = point * accumulator
    peak = max(peak, payload((point, accumulator, boundary)))
    return {
        "boundary_commitment": scalar_commit(boundary),
        "boundary_payload_bits": bits(boundary),
        "persistent_integer_cells": 2,
        "declared_peak_named_integer_cells": 6,
        "declared_peak_named_payload_bits": peak,
        "squarings": program.depth,
        "subtractions": 2 * program.depth,
        "multiplications": program.depth + 1,
    }


class ReferencePort:
    def __init__(self) -> None:
        self.poly = {1: 1}
        self.owner: int | None = None
        self.program_id: str | None = None
        self.port_type: str | None = None
        self.generation = 0
        self.last_restored_generation = 0
        self.cursor = 0
        self.pending: str | None = None
        self.leased = False

    def lease(self, owner: int, program: Program, generation: int, port_type: str = PORT_TYPE) -> None:
        if self.leased or self.poly != {1: 1} or self.cursor != 0:
            raise RuntimeError("not canonical")
        if owner <= 0 or port_type != PORT_TYPE or generation != self.last_restored_generation + 1:
            raise RuntimeError("bad lease")
        self.owner, self.program_id, self.port_type = owner, program.program_id, port_type
        self.generation, self.leased = generation, True

    def require(self, owner: int, program: Program, generation: int) -> None:
        if not self.leased or (owner, program.program_id, PORT_TYPE, generation) != (
            self.owner,
            self.program_id,
            self.port_type,
            self.generation,
        ):
            raise RuntimeError("custody mismatch")

    def forward(self, owner: int, program: Program, generation: int, index: int) -> None:
        self.require(owner, program, generation)
        if index != self.cursor or self.pending is not None:
            raise RuntimeError("ordering")
        descriptor = program.rounds[index]
        if descriptor.input_type != PORT_TYPE or descriptor.output_type != PORT_TYPE:
            raise TypeError("type")
        if descriptor.orientation != ORIENTATION:
            raise TypeError("orientation")
        if (
            evaluate(self.poly, descriptor.root) == 0
            or evaluate(self.poly, -descriptor.shift) == 0
            or -descriptor.shift == descriptor.root
        ):
            raise RuntimeError("stationary branch collision")
        self.pending = "BRANCH_PRODUCT"
        result = adjoin_branch(self.poly, descriptor.root)
        self.pending = "STATIONARY"
        result = stationary_compose(result, descriptor.shift)
        self.poly.clear()
        self.poly.update(result)
        self.cursor += 1
        self.pending = None

    def inverse(self, owner: int, program: Program, generation: int, index: int) -> None:
        self.require(owner, program, generation)
        if index != self.cursor - 1 or self.pending is not None:
            raise RuntimeError("ordering")
        descriptor = program.rounds[index]
        self.pending = "INVERSE_STATIONARY"
        result = stationary_inverse(self.poly, descriptor.shift)
        self.pending = "INVERSE_BRANCH_PRODUCT"
        result = remove_branch_exact(result, descriptor.root)
        self.poly.clear()
        self.poly.update(result)
        self.cursor -= 1
        self.pending = None

    def project(self, owner: int, program: Program, generation: int) -> int:
        self.require(owner, program, generation)
        if self.cursor != program.depth or self.pending is not None:
            raise RuntimeError("premature projection")
        return evaluate(self.poly, program.probe)

    def project_shared(self) -> None:
        raise RuntimeError("shared stationary port projection")

    def release(self, owner: int, program: Program, generation: int) -> None:
        self.require(owner, program, generation)
        if self.cursor or self.pending is not None or self.poly != {1: 1}:
            raise RuntimeError("not restored")
        self.last_restored_generation = generation
        self.owner = self.program_id = self.port_type = None
        self.generation = 0
        self.leased = False


def transaction(port: ReferencePort, program: Program, generation: int, owner: int = 23501) -> dict[str, object]:
    if port is None:
        raise TypeError("null port")
    backing = id(port.poly)
    port.lease(owner, program, generation)
    for index in range(program.depth):
        port.forward(owner, program, generation, index)
    final = dict(port.poly)
    boundary = port.project(owner, program, generation)
    compact = baseline(program)
    if compact["boundary_commitment"] != scalar_commit(boundary):
        raise RuntimeError("baseline mismatch")
    for index in range(program.depth - 1, -1, -1):
        port.inverse(owner, program, generation, index)
    same_backing = id(port.poly) == backing
    restored = port.poly == {1: 1}
    port.release(owner, program, generation)
    values = dense(final)
    return {
        "family": program.family,
        "depth": program.depth,
        "final_relation_degree": max(final),
        "distinct_complex_branch_count": max(final),
        "expected_degree": expected_degree(program.depth),
        "simple_branch_conditions_hold": True,
        "resident_relation_coefficient_cells": len(values),
        "resident_relation_payload_bits": payload(values),
        "retained_final_boundary_integer_cells_during_inverse": 1,
        "retained_final_boundary_payload_bits": bits(boundary),
        "relation_commitment": poly_commit(final),
        "boundary_commitment": scalar_commit(boundary),
        "restoration_error_coefficient_cells": 0 if restored else 1,
        "canonical_post_restoration_state_exact": restored,
        "same_coefficient_backing": same_backing,
        "restoration_generation": generation,
        "baseline_reload_used": False,
        "matched_classical": compact,
    }


def rejected(callback) -> bool:
    try:
        callback()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    program = Program.compile(0, 3)
    owner = 23511
    generic = {0: 3, 1: -2, 2: 5, 3: 1}
    exact_inverse = remove_branch_exact(
        stationary_inverse(stationary_compose(adjoin_branch(generic, 2), -1), -1), 2
    ) == generic
    derivative = all(
        (w * w - (v + 3) == 0) == (v == w * w - 3)
        for w in range(-4, 5)
        for v in range(-5, 20)
    )
    noncommute = stationary_compose(adjoin_branch(generic, 2), 1) != adjoin_branch(
        stationary_compose(generic, 1), 2
    )
    port = ReferencePort()
    port.lease(owner, program, 1)
    premature = rejected(lambda: port.project(owner, program, 1))
    shared = rejected(port.project_shared)
    wrong_owner = rejected(lambda: port.forward(owner + 1, program, 1, 0))
    wrong_generation = rejected(lambda: port.forward(owner, program, 2, 0))
    wrong_program = rejected(lambda: port.forward(owner, Program.compile(1, 3), 1, 0))
    port.pending = "DIRTY"
    dirty = rejected(lambda: port.forward(owner, program, 1, 0))
    wrong_type = rejected(lambda: ReferencePort().lease(owner, program, 1, "WRONG"))
    null = rejected(lambda: transaction(None, program, 1))  # type: ignore[arg-type]

    missing = ReferencePort()
    missing.lease(owner, program, 1)
    for index in range(program.depth):
        missing.forward(owner, program, 1, index)
    missing_inverse = rejected(lambda: missing.release(owner, program, 1))
    reordered = rejected(lambda: remove_branch_exact(missing.poly, program.rounds[-1].root))

    wrong = ReferencePort()
    wrong.lease(owner, program, 1)
    for index in range(program.depth):
        wrong.forward(owner, program, 1, index)
    descriptor = program.rounds[-1]
    changed = stationary_inverse(wrong.poly, descriptor.shift + 1)
    wrong.poly.clear()
    wrong.poly.update(changed)
    wrong.cursor -= 1
    wrong_inverse = rejected(lambda: wrong.release(owner, program, 1))

    base_program = Program.compile(0, 4)
    base = {1: 1}
    for descriptor in base_program.rounds:
        base = stationary_compose(adjoin_branch(base, descriptor.root), descriptor.shift)
    mutated = {1: 1}
    for index, descriptor in enumerate(base_program.rounds):
        root = descriptor.root + 1 if index == 0 else descriptor.root
        mutated = stationary_compose(adjoin_branch(mutated, root), descriptor.shift)
    root_change = poly_commit(mutated) != poly_commit(base) and scalar_commit(
        evaluate(mutated, base_program.probe)
    ) != scalar_commit(evaluate(base, base_program.probe))
    shifted = {1: 1}
    for index, descriptor in enumerate(base_program.rounds):
        amount = descriptor.shift + 1 if index == 0 else descriptor.shift
        shifted = stationary_compose(adjoin_branch(shifted, descriptor.root), amount)
    shift_change = poly_commit(shifted) != poly_commit(base) and scalar_commit(
        evaluate(shifted, base_program.probe)
    ) != scalar_commit(evaluate(base, base_program.probe))

    stale = ReferencePort()
    transaction(stale, Program.compile(0, 1), 1, owner)
    transaction(stale, Program.compile(1, 1), 2, owner)
    stale_rejected = rejected(lambda: stale.lease(owner, Program.compile(0, 1), 2))
    collision_program = Program(0, 1, (Descriptor(0, 2),), "collision-control", 2)
    collision = ReferencePort()
    collision.lease(owner, collision_program, 1)
    collision_rejected = rejected(lambda: collision.forward(owner, collision_program, 1, 0))

    return {
        "stationary_derivative_identity": derivative,
        "generic_forward_inverse_identity": exact_inverse,
        "branch_product_and_composition_noncommute": noncommute,
        "premature_final_projection_rejected": premature,
        "shared_stationary_port_projection_rejected": shared,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "wrong_type_rejected": wrong_type,
        "wrong_generation_rejected": wrong_generation,
        "stale_generation_rejected": stale_rejected,
        "dirty_stage_rejected": dirty,
        "null_carrier_rejected": null,
        "missing_inverse_rejected": missing_inverse,
        "reordered_inverse_rejected": reordered,
        "wrong_inverse_rejected": wrong_inverse,
        "repeated_branch_collision_rejected": collision_rejected,
        "branch_root_perturbation_changes_relation_and_boundary": root_change,
        "stationary_shift_perturbation_changes_relation_and_boundary": shift_change,
        "public_compiler_reads_final_answer": False,
        "relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
        "stationary_branches_enumerated": False,
        "intermediate_relations_serialized": False,
    }


def comparable(case: dict[str, object]) -> dict[str, object]:
    keys = (
        "family",
        "depth",
        "final_relation_degree",
        "distinct_complex_branch_count",
        "expected_degree",
        "simple_branch_conditions_hold",
        "resident_relation_coefficient_cells",
        "resident_relation_payload_bits",
        "retained_final_boundary_integer_cells_during_inverse",
        "retained_final_boundary_payload_bits",
        "relation_commitment",
        "boundary_commitment",
        "restoration_error_coefficient_cells",
        "canonical_post_restoration_state_exact",
        "same_coefficient_backing",
        "restoration_generation",
        "baseline_reload_used",
        "matched_classical",
    )
    return {key: case[key] for key in keys}


def main() -> None:
    cases = [
        transaction(ReferencePort(), Program.compile(family, depth), 1)
        for family in (0, 1)
        for depth in DEPTHS
    ]
    shared = ReferencePort()
    shared_backing = id(shared.poly)
    primary = transaction(shared, Program.compile(0, 8), 1)
    reuse = transaction(shared, Program.compile(1, 7), 2)
    fresh = transaction(ReferencePort(), Program.compile(1, 7), 1)
    output = {
        "schema": "cat_cas.cubic_stationary_critical_relation_reference.v1",
        "cases": cases,
        "controls": controls(),
        "reuse": {
            "primary": comparable(primary),
            "reuse": comparable(reuse),
            "fresh_reuse": comparable(fresh),
            "fresh_restored_boundary_agreement": reuse["boundary_commitment"] == fresh["boundary_commitment"],
            "fresh_restored_relation_agreement": reuse["relation_commitment"] == fresh["relation_commitment"],
            "fresh_restored_degree_and_payload_agreement": (
                reuse["final_relation_degree"], reuse["resident_relation_payload_bits"]
            ) == (fresh["final_relation_degree"], fresh["resident_relation_payload_bits"]),
            "same_backing_across_primary_and_reuse": (
                primary["same_coefficient_backing"]
                and reuse["same_coefficient_backing"]
                and id(shared.poly) == shared_backing
            ),
            "restoration_generation_after_reuse": shared.last_restored_generation,
        },
        "imports_m235_production": False,
        "uses_independent_sparse_exponent_dictionary": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
