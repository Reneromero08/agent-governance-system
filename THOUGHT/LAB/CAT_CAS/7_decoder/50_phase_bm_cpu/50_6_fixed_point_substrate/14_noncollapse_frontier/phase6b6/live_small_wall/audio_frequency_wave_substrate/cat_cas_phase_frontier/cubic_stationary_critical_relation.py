#!/usr/bin/env python3
"""M235: exact cubic stationary critical-locus relation composition.

This is a formal algebraic phase-relation experiment.  It does not evaluate a
convergent oscillatory integral or preserve a Fresnel/Maslov amplitude.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


INITIAL_RELATION = [0, 1]  # v = 0
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

RESULT = (
    "PASS_EXACT_CUBIC_STATIONARY_CRITICAL_RELATION_BRANCH_DEGREE_GROWTH_"
    "WITH_EXACT_RESTORATION_REUSE"
)
CLAIM = (
    "EXACT_PROJECTIVE_CUBIC_PHASE_CRITICAL_LOCUS_RELATIONS_DERIVED_FROM_"
    "W_CUBED_OVER3_MINUS_V_PLUS_C_TIMES_W_COMPOSE_ONE_UNRESOLVED_SHARED_"
    "PORT_BY_NATIVE_SUBSTITUTION_R_OF_W_SQUARED_MINUS_C_AND_ADJOIN_ONE_"
    "PUBLIC_LINEAR_BRANCH_BY_POLYNOMIAL_PRODUCT_WITH_EXACT_INVERSE_"
    "RESTORATION_REUSE_BUT_THE_"
    "EXPANDED_OPEN_RELATION_DEGREE_GROWS_AS3_TIMES2_TO_THE_DEPTH_MINUS2_"
    "THROUGH_DECLARED_DEPTH8_WHILE_A_PUBLIC_DESCRIPTOR_SCALAR_CLASSICAL_"
    "RECURRENCE_RETAINS_CONSTANT_INTEGER_CELL_COUNT_AND_NO_PHASE_RESOURCE_"
    "OR_ADVANTAGE_IS_ESTABLISHED"
)
CLAIM_CEILING = (
    "FORMAL_PROJECTIVE_CUBIC_CRITICAL_LOCUS_POLYNOMIAL_RELATIONS_OVER_Z_"
    "ONE_TYPED_SCALAR_SHARED_PORT_TWO_PUBLIC_PROGRAM_FAMILIES_DEPTHS1_TO8_"
    "PRIMARY_DEPTH8_FAMILY0_REUSE_DEPTH7_FAMILY1_DIRECT_PROCESS_ONLY"
)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload_bits(values: Iterable[int]) -> int:
    return sum(signed_bits(value) for value in values)


def trim(coefficients: Sequence[int]) -> list[int]:
    result = list(coefficients)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def polynomial_commitment(coefficients: Sequence[int]) -> str:
    payload = json.dumps(list(coefficients), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def scalar_commitment(value: int) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


@dataclass(frozen=True)
class RoundDescriptor:
    branch_root: int
    stationary_shift: int
    input_type: str = PORT_TYPE
    output_type: str = PORT_TYPE
    orientation: str = ORIENTATION


@dataclass(frozen=True)
class PublicProgram:
    family: int
    depth: int
    rounds: tuple[RoundDescriptor, ...]
    program_id: str
    probe: int

    @staticmethod
    def compile(family: int, depth: int) -> "PublicProgram":
        if family not in (0, 1) or depth not in DEPTHS:
            raise ValueError("invalid public cubic critical-locus program")
        rounds: list[RoundDescriptor] = []
        for index in range(depth):
            root = FAMILY_ROOTS[family][index]
            shift = FAMILY_SHIFTS[family][index]
            rounds.append(RoundDescriptor(root, shift))
        descriptor = [(item.branch_root, item.stationary_shift) for item in rounds]
        encoded = json.dumps([family, depth, descriptor], separators=(",", ":")).encode()
        return PublicProgram(
            family=family,
            depth=depth,
            rounds=tuple(rounds),
            program_id=hashlib.sha256(encoded).hexdigest(),
            probe=2 if family == 0 else -2,
        )


@dataclass
class Work:
    forward_branch_product_updates: int = 0
    forward_shift_terms: int = 0
    forward_even_embeddings: int = 0
    inverse_even_extractions: int = 0
    inverse_shift_terms: int = 0
    inverse_division_steps: int = 0
    public_rounds_rematerialized: int = 0
    retained_inverse_history_entries: int = 0
    simple_branch_preflight_evaluations: int = 0


def adjoin_linear_branch(coefficients: Sequence[int], root: int, work: Work | None = None) -> list[int]:
    result = [0] * (len(coefficients) + 1)
    for index, value in enumerate(coefficients):
        result[index] -= root * value
        result[index + 1] += value
        if work is not None:
            work.forward_branch_product_updates += 2
    return trim(result)


def remove_linear_branch_exact(coefficients: Sequence[int], root: int, work: Work | None = None) -> list[int]:
    coefficients = trim(coefficients)
    if len(coefficients) < 2:
        raise RuntimeError("linear inverse requires positive degree")
    quotient = [0] * (len(coefficients) - 1)
    quotient[-1] = coefficients[-1]
    for index in range(len(quotient) - 1, 0, -1):
        quotient[index - 1] = coefficients[index] + root * quotient[index]
        if work is not None:
            work.inverse_division_steps += 1
    if coefficients[0] != -root * quotient[0]:
        raise RuntimeError("public branch factor is absent")
    if work is not None:
        work.inverse_division_steps += 1
    return trim(quotient)


def shift_polynomial(
    coefficients: Sequence[int], shift: int, work: Work | None = None, *, inverse: bool = False
) -> list[int]:
    """Return P(x+shift) with exact integer binomial arithmetic."""
    result = [0] * len(coefficients)
    for degree, value in enumerate(coefficients):
        if value == 0:
            continue
        power = 1
        for target in range(degree, -1, -1):
            result[target] += value * math.comb(degree, target) * power
            power *= shift
            if work is not None:
                if inverse:
                    work.inverse_shift_terms += 1
                else:
                    work.forward_shift_terms += 1
    return trim(result)


def compose_cubic_stationary(
    coefficients: Sequence[int], stationary_shift: int, work: Work | None = None
) -> list[int]:
    # d/dw [w^3/3 - (v+c)w] = w^2-v-c, hence v=w^2-c.
    shifted = shift_polynomial(coefficients, -stationary_shift, work)
    result = [0] * (2 * (len(shifted) - 1) + 1)
    for degree, value in enumerate(shifted):
        result[2 * degree] = value
        if work is not None:
            work.forward_even_embeddings += 1
    return trim(result)


def inverse_cubic_stationary(
    coefficients: Sequence[int], stationary_shift: int, work: Work | None = None
) -> list[int]:
    if any(coefficients[index] != 0 for index in range(1, len(coefficients), 2)):
        raise RuntimeError("relation is outside the cubic stationary image")
    collapsed = [coefficients[index] for index in range(0, len(coefficients), 2)]
    if work is not None:
        work.inverse_even_extractions += len(collapsed)
    return shift_polynomial(collapsed, stationary_shift, work, inverse=True)


def evaluate(coefficients: Sequence[int], point: int) -> int:
    accumulator = 0
    for value in reversed(coefficients):
        accumulator = accumulator * point + value
    return accumulator


def expected_degree(depth: int) -> int:
    return 3 * (1 << depth) - 2


def descriptor_scalar_boundary(program: PublicProgram) -> dict[str, int | str]:
    """Strongest implemented final-boundary recurrence, without coefficient expansion."""
    point = program.probe
    accumulator = 1
    peak_cells = 2
    peak_payload = payload_bits((point, accumulator))
    squarings = subtractions = multiplications = 0
    for descriptor in reversed(program.rounds):
        squared = point * point
        next_point = squared - descriptor.stationary_shift
        factor = next_point - descriptor.branch_root
        product = accumulator * factor
        squarings += 1
        subtractions += 2
        multiplications += 1
        peak_cells = max(peak_cells, 6)
        peak_payload = max(
            peak_payload,
            payload_bits((point, squared, next_point, factor, accumulator, product)),
        )
        point = next_point
        accumulator = product
    boundary = point * accumulator  # R0(point)=point
    multiplications += 1
    peak_payload = max(peak_payload, payload_bits((point, accumulator, boundary)))
    return {
        "boundary_commitment": scalar_commitment(boundary),
        "boundary_payload_bits": signed_bits(boundary),
        "persistent_integer_cells": 2,
        "declared_peak_named_integer_cells": peak_cells,
        "declared_peak_named_payload_bits": peak_payload,
        "squarings": squarings,
        "subtractions": subtractions,
        "multiplications": multiplications,
    }


class CubicCriticalPort:
    def __init__(self) -> None:
        self.coefficients = list(INITIAL_RELATION)
        self.owner: int | None = None
        self.program_id: str | None = None
        self.port_type: str | None = None
        self.generation = 0
        self.last_restored_generation = 0
        self.cursor = 0
        self.pending_stage: str | None = None
        self.leased = False

    def canonical_state(self) -> tuple[object, ...]:
        return (
            tuple(self.coefficients),
            self.owner,
            self.program_id,
            self.port_type,
            self.generation,
            self.last_restored_generation,
            self.cursor,
            self.pending_stage,
            self.leased,
        )

    def lease(self, owner: int, program: PublicProgram, generation: int, port_type: str = PORT_TYPE) -> None:
        if self.leased or self.coefficients != INITIAL_RELATION or self.cursor != 0:
            raise RuntimeError("carrier is not canonical")
        if owner <= 0 or port_type != PORT_TYPE or generation != self.last_restored_generation + 1:
            raise RuntimeError("invalid carrier lease")
        self.owner = owner
        self.program_id = program.program_id
        self.port_type = port_type
        self.generation = generation
        self.leased = True

    def require(self, owner: int, program: PublicProgram, generation: int) -> None:
        if not self.leased:
            raise RuntimeError("carrier is not leased")
        if (
            self.owner != owner
            or self.program_id != program.program_id
            or self.port_type != PORT_TYPE
            or self.generation != generation
        ):
            raise RuntimeError("carrier custody mismatch")

    def forward_round(
        self, owner: int, program: PublicProgram, generation: int, index: int, work: Work
    ) -> None:
        self.require(owner, program, generation)
        if index != self.cursor or self.pending_stage is not None:
            raise RuntimeError("forward dependency ordering failure")
        descriptor = program.rounds[index]
        if descriptor.input_type != PORT_TYPE or descriptor.output_type != PORT_TYPE:
            raise TypeError("wrong relation port type")
        if descriptor.orientation != ORIENTATION:
            raise TypeError("wrong relation orientation")
        # If R is square-free, R*(v-h) stays square-free when R(h)!=0.
        # P(w^2-c) is square-free when P(-c)!=0, equivalently
        # R(-c)!=0 and -c-h!=0.  The initial R(v)=v is square-free, so
        # these exact checks inductively certify every accepted branch.
        work.simple_branch_preflight_evaluations += 3
        if (
            evaluate(self.coefficients, descriptor.branch_root) == 0
            or evaluate(self.coefficients, -descriptor.stationary_shift) == 0
            or -descriptor.stationary_shift == descriptor.branch_root
        ):
            raise RuntimeError("stationary branch collision or repeated root")
        self.pending_stage = "BRANCH_PRODUCT"
        intersected = adjoin_linear_branch(self.coefficients, descriptor.branch_root, work)
        self.pending_stage = "CUBIC_STATIONARY_COMPOSITION"
        composed = compose_cubic_stationary(intersected, descriptor.stationary_shift, work)
        self.coefficients[:] = composed
        self.cursor += 1
        self.pending_stage = None
        work.public_rounds_rematerialized += 1

    def inverse_round(
        self, owner: int, program: PublicProgram, generation: int, index: int, work: Work
    ) -> None:
        self.require(owner, program, generation)
        if index != self.cursor - 1 or self.pending_stage is not None:
            raise RuntimeError("inverse dependency ordering failure")
        descriptor = program.rounds[index]
        self.pending_stage = "INVERSE_CUBIC_STATIONARY_COMPOSITION"
        intersected = inverse_cubic_stationary(self.coefficients, descriptor.stationary_shift, work)
        self.pending_stage = "INVERSE_BRANCH_PRODUCT"
        restored = remove_linear_branch_exact(intersected, descriptor.branch_root, work)
        self.coefficients[:] = restored
        self.cursor -= 1
        self.pending_stage = None
        work.public_rounds_rematerialized += 1

    def project_final(self, owner: int, program: PublicProgram, generation: int) -> int:
        self.require(owner, program, generation)
        if self.cursor != program.depth or self.pending_stage is not None:
            raise RuntimeError("final projection before completed closure")
        return evaluate(self.coefficients, program.probe)

    def project_shared_stationary_port(self) -> None:
        raise RuntimeError("shared stationary coordinate projection is forbidden")

    def release(self, owner: int, program: PublicProgram, generation: int) -> None:
        self.require(owner, program, generation)
        if self.cursor != 0 or self.pending_stage is not None or self.coefficients != INITIAL_RELATION:
            raise RuntimeError("carrier is not exactly restored")
        self.last_restored_generation = generation
        self.owner = None
        self.program_id = None
        self.port_type = None
        self.generation = 0
        self.leased = False


def run_transaction(
    port: CubicCriticalPort, program: PublicProgram, generation: int, owner: int = 23501
) -> dict[str, object]:
    if port is None:
        raise TypeError("null cubic critical-locus carrier")
    backing = id(port.coefficients)
    before = tuple(port.coefficients)
    work = Work()
    port.lease(owner, program, generation)
    for index in range(program.depth):
        port.forward_round(owner, program, generation, index, work)
    final_coefficients = tuple(port.coefficients)
    boundary = port.project_final(owner, program, generation)
    retained_boundary = boundary
    baseline = descriptor_scalar_boundary(program)
    if baseline["boundary_commitment"] != scalar_commitment(boundary):
        raise RuntimeError("strong compact baseline mismatch")
    degree = len(final_coefficients) - 1
    if degree != expected_degree(program.depth):
        raise RuntimeError("cubic stationary degree law mismatch")
    for index in range(program.depth - 1, -1, -1):
        port.inverse_round(owner, program, generation, index, work)
    restoration_error = sum(left != right for left, right in zip(port.coefficients, before))
    same_backing = id(port.coefficients) == backing
    port.release(owner, program, generation)
    if retained_boundary != boundary:
        raise RuntimeError("retained result was consumed by inverse")
    return {
        "family": program.family,
        "depth": program.depth,
        "final_relation_degree": degree,
        "distinct_complex_branch_count": degree,
        "expected_degree": expected_degree(program.depth),
        "simple_branch_conditions_hold": True,
        "resident_relation_coefficient_cells": len(final_coefficients),
        "resident_relation_payload_bits": payload_bits(final_coefficients),
        "retained_final_boundary_integer_cells_during_inverse": 1,
        "retained_final_boundary_payload_bits": signed_bits(boundary),
        "relation_commitment": polynomial_commitment(final_coefficients),
        "boundary_commitment": scalar_commitment(boundary),
        "restoration_error_coefficient_cells": restoration_error,
        "canonical_post_restoration_state_exact": tuple(port.coefficients) == before,
        "same_coefficient_backing": same_backing,
        "restoration_generation": generation,
        "baseline_reload_used": False,
        "work": asdict(work),
        "matched_classical": baseline,
    }


def expect_rejected(callback) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    program = PublicProgram.compile(0, 3)
    owner = 23511

    # Exact generic algebraic identities.
    generic = [3, -2, 5, 1]
    intersected = adjoin_linear_branch(generic, 2)
    composed = compose_cubic_stationary(intersected, -1)
    exact_inverse = remove_linear_branch_exact(inverse_cubic_stationary(composed, -1), 2) == generic
    derivative_identity = all(
        (w * w - (v + 3) == 0) == (v == w * w - 3)
        for w in range(-4, 5)
        for v in range(-5, 20)
    )

    left = compose_cubic_stationary(adjoin_linear_branch(generic, 2), 1)
    right = adjoin_linear_branch(compose_cubic_stationary(generic, 1), 2)
    noncommuting = left != right

    premature_port = CubicCriticalPort()
    premature_port.lease(owner, program, 1)
    premature_projection = expect_rejected(
        lambda: premature_port.project_final(owner, program, 1)
    )
    shared_projection = expect_rejected(premature_port.project_shared_stationary_port)

    wrong_owner = expect_rejected(
        lambda: premature_port.forward_round(owner + 1, program, 1, 0, Work())
    )
    wrong_generation = expect_rejected(
        lambda: premature_port.forward_round(owner, program, 2, 0, Work())
    )
    wrong_program = expect_rejected(
        lambda: premature_port.forward_round(owner, PublicProgram.compile(1, 3), 1, 0, Work())
    )
    premature_port.pending_stage = "DIRTY"
    dirty_stage = expect_rejected(
        lambda: premature_port.forward_round(owner, program, 1, 0, Work())
    )

    wrong_type_port = CubicCriticalPort()
    wrong_type = expect_rejected(lambda: wrong_type_port.lease(owner, program, 1, "WRONG"))
    null_carrier = expect_rejected(lambda: run_transaction(None, program, 1))  # type: ignore[arg-type]

    missing_port = CubicCriticalPort()
    missing_port.lease(owner, program, 1)
    for index in range(program.depth):
        missing_port.forward_round(owner, program, 1, index, Work())
    missing_inverse = expect_rejected(lambda: missing_port.release(owner, program, 1))
    reordered_inverse = expect_rejected(
        lambda: remove_linear_branch_exact(
            missing_port.coefficients, program.rounds[-1].branch_root
        )
    )

    wrong_inverse_port = CubicCriticalPort()
    wrong_inverse_port.lease(owner, program, 1)
    for index in range(program.depth):
        wrong_inverse_port.forward_round(owner, program, 1, index, Work())
    last = program.rounds[-1]
    wrong_collapsed = inverse_cubic_stationary(
        wrong_inverse_port.coefficients, last.stationary_shift + 1
    )
    wrong_inverse_port.coefficients[:] = wrong_collapsed
    wrong_inverse_port.cursor -= 1
    wrong_inverse = expect_rejected(lambda: wrong_inverse_port.release(owner, program, 1))

    mutation_program = PublicProgram.compile(0, 4)
    mutation_port = CubicCriticalPort()
    mutation_port.lease(owner, mutation_program, 1)
    for index in range(mutation_program.depth):
        mutation_port.forward_round(owner, mutation_program, 1, index, Work())
    original_commitment = polynomial_commitment(mutation_port.coefficients)
    original_boundary = scalar_commitment(evaluate(mutation_port.coefficients, mutation_program.probe))
    mutated_rounds = list(mutation_program.rounds)
    first = mutated_rounds[0]
    mutated_rounds[0] = RoundDescriptor(first.branch_root + 1, first.stationary_shift)
    encoded = json.dumps([0, 4, [(r.branch_root, r.stationary_shift) for r in mutated_rounds]])
    mutated_program = PublicProgram(0, 4, tuple(mutated_rounds), hashlib.sha256(encoded.encode()).hexdigest(), 2)
    mutated = list(INITIAL_RELATION)
    for descriptor in mutated_program.rounds:
        mutated = compose_cubic_stationary(
            adjoin_linear_branch(mutated, descriptor.branch_root), descriptor.stationary_shift
        )
    root_perturbation = (
        polynomial_commitment(mutated) != original_commitment
        and scalar_commitment(evaluate(mutated, mutated_program.probe)) != original_boundary
    )
    shifted_rounds = list(mutation_program.rounds)
    first = shifted_rounds[0]
    shifted_rounds[0] = RoundDescriptor(first.branch_root, first.stationary_shift + 1)
    mutated_shift = list(INITIAL_RELATION)
    for descriptor in shifted_rounds:
        mutated_shift = compose_cubic_stationary(
            adjoin_linear_branch(mutated_shift, descriptor.branch_root), descriptor.stationary_shift
        )
    shift_perturbation = (
        polynomial_commitment(mutated_shift) != original_commitment
        and scalar_commitment(evaluate(mutated_shift, mutation_program.probe)) != original_boundary
    )

    stale_port = CubicCriticalPort()
    run_transaction(stale_port, PublicProgram.compile(0, 1), 1, owner)
    run_transaction(stale_port, PublicProgram.compile(1, 1), 2, owner)
    stale_generation = expect_rejected(
        lambda: stale_port.lease(owner, PublicProgram.compile(0, 1), 2)
    )
    collision_program = PublicProgram(
        0,
        1,
        (RoundDescriptor(0, 2),),
        "collision-control",
        2,
    )
    collision_port = CubicCriticalPort()
    collision_port.lease(owner, collision_program, 1)
    repeated_branch_collision = expect_rejected(
        lambda: collision_port.forward_round(owner, collision_program, 1, 0, Work())
    )

    return {
        "stationary_derivative_identity": derivative_identity,
        "generic_forward_inverse_identity": exact_inverse,
        "branch_product_and_composition_noncommute": noncommuting,
        "premature_final_projection_rejected": premature_projection,
        "shared_stationary_port_projection_rejected": shared_projection,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "wrong_type_rejected": wrong_type,
        "wrong_generation_rejected": wrong_generation,
        "stale_generation_rejected": stale_generation,
        "dirty_stage_rejected": dirty_stage,
        "null_carrier_rejected": null_carrier,
        "missing_inverse_rejected": missing_inverse,
        "reordered_inverse_rejected": reordered_inverse,
        "wrong_inverse_rejected": wrong_inverse,
        "repeated_branch_collision_rejected": repeated_branch_collision,
        "branch_root_perturbation_changes_relation_and_boundary": root_perturbation,
        "stationary_shift_perturbation_changes_relation_and_boundary": shift_perturbation,
        "public_compiler_reads_final_answer": False,
        "relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
        "stationary_branches_enumerated": False,
        "intermediate_relations_serialized": False,
    }


def comparable_case(case: dict[str, object]) -> dict[str, object]:
    return {
        key: case[key]
        for key in (
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
    }


def main(reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    cases = [
        run_transaction(CubicCriticalPort(), PublicProgram.compile(family, depth), 1)
        for family in (0, 1)
        for depth in DEPTHS
    ]
    production_controls = controls()
    if [comparable_case(case) for case in cases] != [
        comparable_case(case) for case in reference["cases"]
    ]:
        raise RuntimeError("independent case parity failure")
    if production_controls != reference["controls"]:
        raise RuntimeError("independent control parity failure")

    shared = CubicCriticalPort()
    shared_backing = id(shared.coefficients)
    primary = run_transaction(shared, PublicProgram.compile(0, 8), 1)
    reuse = run_transaction(shared, PublicProgram.compile(1, 7), 2)
    fresh = run_transaction(CubicCriticalPort(), PublicProgram.compile(1, 7), 1)
    reuse_summary = {
        "primary": comparable_case(primary),
        "reuse": comparable_case(reuse),
        "fresh_reuse": comparable_case(fresh),
        "fresh_restored_boundary_agreement": reuse["boundary_commitment"] == fresh["boundary_commitment"],
        "fresh_restored_relation_agreement": reuse["relation_commitment"] == fresh["relation_commitment"],
        "fresh_restored_degree_and_payload_agreement": (
            reuse["final_relation_degree"], reuse["resident_relation_payload_bits"]
        ) == (fresh["final_relation_degree"], fresh["resident_relation_payload_bits"]),
        "same_backing_across_primary_and_reuse": (
            primary["same_coefficient_backing"]
            and reuse["same_coefficient_backing"]
            and id(shared.coefficients) == shared_backing
        ),
        "restoration_generation_after_reuse": shared.last_restored_generation,
    }
    if reuse_summary != reference["reuse"]:
        raise RuntimeError("independent reuse parity failure")

    source = Path(__file__)
    reference_code = source.with_name("cubic_stationary_critical_relation_separate_reference.py")
    relation_cells = [expected_degree(depth) + 1 for depth in DEPTHS]
    primary_payload = [case["resident_relation_payload_bits"] for case in cases if case["family"] == 0]
    alternate_payload = [case["resident_relation_payload_bits"] for case in cases if case["family"] == 1]
    output = {
        "schema": "cat_cas.cubic_stationary_critical_relation.v1",
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": cases,
        "controls": production_controls,
        "reuse": reuse_summary,
        "phase_relation_law": {
            "cubic_phase": "w^3/3-(v+c)w",
            "stationary_equation": "w^2-v-c=0",
            "native_shared_port_elimination": "R(v)->R(w^2-c)",
            "native_branch_product": "R(v)->R(v)*(v-h)",
            "branch_product_semantics": "ADJOINS_ZERO_BRANCH_UNION_NOT_SET_THEORETIC_INTERSECTION",
            "set_theoretic_relation_intersection_established": False,
            "exact_public_inverse": "EVEN_POWER_COLLAPSE_PLUS_REVERSE_SHIFT_THEN_EXACT_LINEAR_DIVISION",
            "square_free_induction": "R_H_NONZERO_AND_R_MINUS_C_NONZERO_AND_MINUS_C_NOT_H_EACH_ROUND",
            "declared_complex_branches_are_distinct": True,
            "shared_stationary_port_projected": False,
            "final_boundary_only": True,
            "projective_scalar_amplitude_discarded": True,
            "convergent_wave_integral_claimed": False,
            "direct_process_logical_custody_only": True,
        },
        "resource_law": {
            "depths": list(DEPTHS),
            "exact_degree_law": "3*2^depth-2",
            "resident_relation_coefficient_cells_by_depth": relation_cells,
            "primary_payload_bits_by_depth": primary_payload,
            "alternate_payload_bits_by_depth": alternate_payload,
            "retained_final_boundary_integer_cells_during_inverse": 1,
            "retained_inverse_history_entries": 0,
            "public_round_descriptor_integers": "2*depth",
            "stationary_branches_enumerated": False,
            "all_declared_relations_square_free_by_exact_inductive_preflight": True,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
            "whole_transaction_live_integer_cell_and_payload_accounting_complete": False,
            "excluded_not_zero": (
                "PYTHON_INTEGER_AND_LIST_OBJECT_HEADERS_TEMPORARY_LOCALS_ALLOCATOR_"
                "INTERPRETER_HASH_BYTE_TRAFFIC_JSON_SERIALIZATION_TIMING_PROCESS_RSS"
            ),
        },
        "matched_classical": {
            "strongest_implemented": "PUBLIC_DESCRIPTOR_BACKWARD_SCALAR_BOUNDARY_RECURRENCE",
            "persistent_integer_cells": 2,
            "expanded_relation_cells_at_depth8": relation_cells[-1],
            "same_exact_boundary": True,
            "retains_open_relation_signature": True,
            "public_expression_dag_size": "O(depth)",
            "integer_payload_width_grows": True,
            "complete_integer_bit_operation_cost_fully_accounted": False,
            "computational_advantage": False,
            "distinct_phase_resource": False,
        },
        "separate_reference": {
            "imports_m235_production": False,
            "uses_independent_sparse_exponent_dictionary": True,
            "case_control_reuse_parity": True,
        },
        "source_dependencies": {
            "m235_production_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "m235_reference_code_sha256": hashlib.sha256(reference_code.read_bytes()).hexdigest(),
            "m235_reference_result_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        },
        "claim_limits": {
            "normalized_or_convergent_cubic_integral_established": False,
            "fresnel_or_maslov_scalar_preserved": False,
            "general_cubic_phase_relation_algebra": False,
            "set_theoretic_relation_intersection_established": False,
            "fixed_coefficient_cell_closure": False,
            "all_non_gaussian_relations_require_exponential_state": False,
            "compact_nonlinear_or_expression_quotient_rejected": False,
            "machine_enforced_catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "terminal": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: cubic_stationary_critical_relation.py REFERENCE_JSON")
    main(Path(sys.argv[1]))
