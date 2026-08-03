#!/usr/bin/env python3
"""Exact repeated-root C17 phase-jet open-relation closure over F17.

In characteristic 17 the cyclic group algebra is

    F17[C17] = F17[t]/(t^17 - 1) = F17[epsilon]/(epsilon^17),
    t = 1 + epsilon.

Reduction modulo epsilon**rank is therefore an exact algebra quotient.  This
program executes reversible open-relation convolution and a nonlinear
convolution-square shear directly on the resident quotient coordinates.  It
also executes the identical compact recurrence as the strongest classical
baseline; no computational advantage is inferred from the quotient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIELD = 17
GROUP_ORDER = 17
RANKS = (2, 4, 8)
DEPTHS = (1, 4, 16, 64)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_F17_C17_REPEATED_ROOT_NILPOTENT_PHASE_JET_QUOTIENT_"
    "CLOSES_TRANSLATION_INVARIANT_OPEN_RELATION_COMPOSITION_AND_NONLINEAR_"
    "CONVOLUTION_SHEAR_IN_RANKS2_4_8_THROUGH_DEPTH64_WITH_FINAL_ONLY_"
    "BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_THE_IDENTICAL_RANK_R_"
    "CLASSICAL_JET_RECURRENCE_REMAINS_AND_HIGHER_HASSE_MOMENTS_ARE_"
    "EXPLICITLY_OUTSIDE_THE_QUOTIENT"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def add(left: list[int], right: list[int]) -> list[int]:
    return [(a + b) % FIELD for a, b in zip(left, right, strict=True)]


def subtract(left: list[int], right: list[int]) -> list[int]:
    return [(a - b) % FIELD for a, b in zip(left, right, strict=True)]


def scale(value: list[int], scalar: int) -> list[int]:
    return [(scalar * item) % FIELD for item in value]


def multiply(left: list[int], right: list[int], rank: int) -> list[int]:
    """Multiply in F17[epsilon]/epsilon**rank without a dense table."""
    result = [0] * rank
    for i, a in enumerate(left):
        for j, b in enumerate(right[: rank - i]):
            result[i + j] = (result[i + j] + a * b) % FIELD
    return result


def power(value: list[int], exponent: int, rank: int) -> list[int]:
    if exponent < 0:
        return power(invert_unit(value, rank), -exponent, rank)
    result = [1] + [0] * (rank - 1)
    base = value.copy()
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = multiply(result, base, rank)
        base = multiply(base, base, rank)
        remaining >>= 1
    return result


def invert_unit(value: list[int], rank: int) -> list[int]:
    if len(value) != rank or value[0] % FIELD == 0:
        fail("jet is not a unit")
    inverse = [0] * rank
    inverse[0] = pow(value[0], FIELD - 2, FIELD)
    for degree in range(1, rank):
        subtotal = sum(value[j] * inverse[degree - j] for j in range(1, degree + 1))
        inverse[degree] = (-inverse[0] * subtotal) % FIELD
    return inverse


def phase_shift(rank: int, shift: int) -> list[int]:
    epsilon = [1, 1] + [0] * (rank - 2)
    return power(epsilon, shift % GROUP_ORDER, rank)


def public_kernel(rank: int, index: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    shift = (3 * index + 5 * code + 1) % GROUP_ORDER
    slope = (7 * index + 4 * code + 3) % FIELD
    local = [1, slope] + [0] * (rank - 2)
    return multiply(phase_shift(rank, shift), local, rank)


def step_parameters(index: int, family: str) -> tuple[int, int, int]:
    code = 1 if family == "PRIMARY" else 2
    alpha = (5 * index + 3 * code + 1) % FIELD or 1
    beta = (7 * index + 2 * code + 4) % FIELD or 1
    b_shift = (11 * index + 5 * code + 2) % GROUP_ORDER
    return alpha, beta, b_shift


def public_seed_coefficient(position: int, family: str, register: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    if register == "A":
        atoms = (
            (code, 2 + code),
            ((3 + 2 * code) % GROUP_ORDER, 5),
            ((8 + code) % GROUP_ORDER, 9),
        )
    else:
        atoms = (
            ((2 + code) % GROUP_ORDER, 4),
            ((7 + 2 * code) % GROUP_ORDER, 6 + code),
            ((13 - code) % GROUP_ORDER, 11),
        )
    return sum(weight for location, weight in atoms if location == position) % FIELD


def binomial_mod(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    numerator = 1
    denominator = 1
    for value in range(1, k + 1):
        numerator = numerator * (n - value + 1) % FIELD
        denominator = denominator * value % FIELD
    return numerator * pow(denominator, FIELD - 2, FIELD) % FIELD


def compile_seed_jet(rank: int, family: str, register: str) -> list[int]:
    """Stream the public kernel into Hasse coordinates; retain no 17-vector."""
    return [
        sum(
            public_seed_coefficient(position, family, register)
            * binomial_mod(position, degree)
            for position in range(GROUP_ORDER)
        )
        % FIELD
        for degree in range(rank)
    ]


@dataclass(frozen=True)
class Program:
    rank: int
    depth: int
    family: str
    boundary_weights: tuple[int, ...]

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F17_C17_NILPOTENT_PHASE_JET_PROGRAM_V1",
            "field": FIELD,
            "group": "C17",
            "rank": self.rank,
            "depth": self.depth,
            "family": self.family,
            "port_type": f"F17_C17_TRANSLATION_INVARIANT_OPEN_RELATION_JET_R{self.rank}",
            "state_law": "F17_EPSILON_MOD_EPSILON_TO_RANK",
            "composition": "TRUNCATED_JET_MULTIPLICATION",
            "nonlinear_module": "REVERSIBLE_B_PLUS_BETA_A_CONVOLUTION_SQUARE",
            "projection": "FINAL_ONLY_PUBLIC_LINEAR_B_JET_BOUNDARY",
            "boundary_weights": list(self.boundary_weights),
            "topology_compilation_reads_final_answer": False,
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(rank: int, depth: int, family: str) -> Program:
    if rank not in RANKS:
        fail("rank outside declared set")
    if depth not in DEPTHS:
        fail("depth outside declared set")
    if family not in FAMILIES:
        fail("family outside declared set")
    code = 1 if family == "PRIMARY" else 2
    weights = tuple((3 * degree + 5 * code + rank) % FIELD or 1 for degree in range(rank))
    return Program(rank, depth, family, weights)


@dataclass
class Work:
    jet_multiplications: int = 0
    field_multiply_accumulates: int = 0
    linear_updates: int = 0
    phase_rotations: int = 0
    relation_compositions: int = 0
    nonlinear_shears: int = 0

    def descriptor(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass
class Carrier:
    cells: list[int]
    rank: int
    seed_family: str
    restoration_generation: int = 0
    stage: str = "SEALED"

    @classmethod
    def seal(cls, rank: int, family: str) -> "Carrier":
        if rank not in RANKS or family not in FAMILIES:
            fail("invalid carrier descriptor")
        cells = compile_seed_jet(rank, family, "A") + compile_seed_jet(rank, family, "B")
        return cls(cells, rank, family)

    def backing_id(self) -> int:
        return id(self.cells)

    def state_commitment(self) -> str:
        return digest_json(
            {
                "cells": self.cells,
                "rank": self.rank,
                "seed_family": self.seed_family,
                "stage": self.stage,
            }
        )

    def a(self) -> list[int]:
        return self.cells[: self.rank]

    def b(self) -> list[int]:
        return self.cells[self.rank :]

    def write(self, a: list[int], b: list[int]) -> None:
        self.cells[: self.rank] = a
        self.cells[self.rank :] = b


def counted_multiply(left: list[int], right: list[int], rank: int, work: Work) -> list[int]:
    work.jet_multiplications += 1
    work.field_multiply_accumulates += rank * (rank + 1) // 2
    return multiply(left, right, rank)


def forward_step(carrier: Carrier, index: int, family: str, work: Work) -> None:
    rank = carrier.rank
    a = carrier.a()
    b = carrier.b()
    alpha, beta, b_shift = step_parameters(index, family)
    kernel = public_kernel(rank, index, family)
    if family == "PRIMARY":
        a = add(a, scale(b, alpha))
        work.linear_updates += rank
        a_square = counted_multiply(a, a, rank, work)
        b = add(b, scale(a_square, beta))
        work.nonlinear_shears += 1
        a = counted_multiply(a, kernel, rank, work)
        work.relation_compositions += 1
    else:
        a_square = counted_multiply(a, a, rank, work)
        b = add(b, scale(a_square, beta))
        work.nonlinear_shears += 1
        a = counted_multiply(a, kernel, rank, work)
        work.relation_compositions += 1
        a = add(a, scale(b, alpha))
        work.linear_updates += rank
    b = counted_multiply(b, phase_shift(rank, b_shift), rank, work)
    work.phase_rotations += 1
    carrier.write(a, b)


def inverse_step(
    carrier: Carrier,
    index: int,
    family: str,
    work: Work,
    *,
    mutation: str = "NONE",
) -> None:
    rank = carrier.rank
    a = carrier.a()
    b = carrier.b()
    alpha, beta, b_shift = step_parameters(index, family)
    kernel_inverse = invert_unit(public_kernel(rank, index, family), rank)
    b = counted_multiply(b, phase_shift(rank, -b_shift), rank, work)
    work.phase_rotations += 1
    if family == "PRIMARY":
        if mutation == "REORDER":
            b = subtract(b, scale(counted_multiply(a, a, rank, work), beta))
            a = counted_multiply(a, kernel_inverse, rank, work)
        else:
            a = counted_multiply(a, kernel_inverse, rank, work)
            wrong_beta = (beta + 1) % FIELD if mutation == "WRONG" else beta
            b = subtract(b, scale(counted_multiply(a, a, rank, work), wrong_beta))
        a = subtract(a, scale(b, alpha))
    else:
        a = subtract(a, scale(b, alpha))
        if mutation == "REORDER":
            b = subtract(b, scale(counted_multiply(a, a, rank, work), beta))
            a = counted_multiply(a, kernel_inverse, rank, work)
        else:
            a = counted_multiply(a, kernel_inverse, rank, work)
            wrong_beta = (beta + 1) % FIELD if mutation == "WRONG" else beta
            b = subtract(b, scale(counted_multiply(a, a, rank, work), wrong_beta))
    work.relation_compositions += 1
    work.nonlinear_shears += 1
    work.linear_updates += rank
    carrier.write(a, b)


def forward(carrier: Carrier, program: Program, work: Work, *, enabled: bool = True) -> None:
    if carrier.stage != "SEALED" or carrier.rank != program.rank:
        fail("carrier/program stage or type mismatch")
    carrier.stage = "FORWARD_RUNNING"
    if enabled:
        for index in range(program.depth):
            forward_step(carrier, index, program.family, work)
    carrier.stage = "FORWARD_COMPLETE"


def inverse(carrier: Carrier, program: Program, work: Work, *, mutation: str = "NONE") -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("inverse outside forward-complete stage")
    for index in reversed(range(program.depth)):
        inverse_step(carrier, index, program.family, work, mutation=mutation)
    carrier.stage = "SEALED"


def project_final_boundary(carrier: Carrier, program: Program) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary projection outside final stage")
    return sum(weight * value for weight, value in zip(program.boundary_weights, carrier.b(), strict=True)) % FIELD


def project_hidden_a(_carrier: Carrier) -> None:
    fail("A jet is an unresolved internal relation port")


def classical_multiply(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    rank = len(left)
    values = []
    for degree in range(rank):
        values.append(sum(left[index] * right[degree - index] for index in range(degree + 1)) % FIELD)
    return tuple(values)


def classical_execute(seed: tuple[int, ...], program: Program) -> tuple[int, ...]:
    rank = program.rank
    a = tuple(seed[:rank])
    b = tuple(seed[rank:])
    for index in range(program.depth):
        alpha, beta, b_shift = step_parameters(index, program.family)
        kernel = tuple(public_kernel(rank, index, program.family))
        if program.family == "PRIMARY":
            a = tuple((x + alpha * y) % FIELD for x, y in zip(a, b, strict=True))
            square = classical_multiply(a, a)
            b = tuple((x + beta * y) % FIELD for x, y in zip(b, square, strict=True))
            a = classical_multiply(a, kernel)
        else:
            square = classical_multiply(a, a)
            b = tuple((x + beta * y) % FIELD for x, y in zip(b, square, strict=True))
            a = classical_multiply(a, kernel)
            a = tuple((x + alpha * y) % FIELD for x, y in zip(a, b, strict=True))
        b = classical_multiply(b, tuple(phase_shift(rank, b_shift)))
    return a + b


def boundary_from_cells(cells: tuple[int, ...], program: Program) -> int:
    b = cells[program.rank :]
    return sum(weight * value for weight, value in zip(program.boundary_weights, b, strict=True)) % FIELD


def transaction(carrier: Carrier | None, program: Program) -> dict[str, Any]:
    if carrier is None:
        fail("null carrier")
    before_cells = tuple(carrier.cells)
    before_backing = carrier.backing_id()
    work = Work()
    forward(carrier, program, work)
    final_cells = tuple(carrier.cells)
    boundary = project_final_boundary(carrier, program)
    inverse(carrier, program, work)
    if tuple(carrier.cells) != before_cells or carrier.stage != "SEALED":
        fail("exact carrier restoration failed")
    if carrier.backing_id() != before_backing:
        fail("carrier backing changed")
    carrier.restoration_generation += 1
    return {
        "program_fingerprint": program.fingerprint(),
        "boundary": boundary,
        "final_jet_commitment": digest_json(list(final_cells)),
        "same_backing_restored": True,
        "exact_cells_restored": True,
        "restoration_generation": carrier.restoration_generation,
        "work": work.descriptor(),
    }


def one_case(rank: int, depth: int, family: str) -> dict[str, Any]:
    program = compile_program(rank, depth, family)
    carrier = Carrier.seal(rank, family)
    seed = tuple(carrier.cells)
    backing = carrier.backing_id()
    receipt = transaction(carrier, program)
    classical = classical_execute(seed, program)
    return {
        "rank": rank,
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "boundary": receipt["boundary"],
        "final_jet_commitment": receipt["final_jet_commitment"],
        "matches_identical_compact_classical_recurrence": (
            receipt["boundary"] == boundary_from_cells(classical, program)
            and receipt["final_jet_commitment"] == digest_json(list(classical))
        ),
        "exact_cells_restored": receipt["exact_cells_restored"],
        "same_backing_restored": receipt["same_backing_restored"] and carrier.backing_id() == backing,
        "restoration_generation": receipt["restoration_generation"],
        "accepted_carrier_field_cells": 2 * rank,
        "logical_working_field_cells_peak": 7 * rank,
        "retained_inverse_history_field_cells": 0,
        "work": receipt["work"],
    }


def raises(action: Any) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    program = compile_program(8, 16, "PRIMARY")
    seed = Carrier.seal(8, "ALTERNATE")
    before = tuple(seed.cells)

    missing = Carrier(list(before), 8, "ALTERNATE")
    forward(missing, program, Work())
    missing_inverse_fails = tuple(missing.cells) != before

    wrong = Carrier(list(before), 8, "ALTERNATE")
    forward(wrong, program, Work())
    inverse(wrong, program, Work(), mutation="WRONG")
    wrong_inverse_fails = tuple(wrong.cells) != before

    reordered = Carrier(list(before), 8, "ALTERNATE")
    forward(reordered, program, Work())
    inverse(reordered, program, Work(), mutation="REORDER")
    reordered_inverse_fails = tuple(reordered.cells) != before

    disabled = Carrier(list(before), 8, "ALTERNATE")
    forward(disabled, program, Work(), enabled=False)
    disabled_boundary = project_final_boundary(disabled, program)
    active = Carrier(list(before), 8, "ALTERNATE")
    forward(active, program, Work())
    active_boundary = project_final_boundary(active, program)

    premature = Carrier(list(before), 8, "ALTERNATE")
    premature_projection_rejected = raises(lambda: project_final_boundary(premature, program))

    full = Carrier(list(before), 8, "ALTERNATE")
    forward(full, program, Work())
    full_boundary = project_final_boundary(full, program)
    mass_only_cells = list(before)
    for degree in range(1, 8):
        mass_only_cells[degree] = 0
        mass_only_cells[8 + degree] = 0
    mass_only = Carrier(mass_only_cells, 8, "ALTERNATE")
    forward(mass_only, program, Work())
    mass_only_boundary = project_final_boundary(mass_only, program)

    perturbed_cells = list(before)
    perturbed_cells[7] = (perturbed_cells[7] + 1) % FIELD
    perturbed = Carrier(perturbed_cells, 8, "ALTERNATE")
    forward(perturbed, program, Work())
    perturbed_boundary = project_final_boundary(perturbed, program)

    return {
        "missing_inverse_fails_restoration": missing_inverse_fails,
        "wrong_inverse_fails_restoration": wrong_inverse_fails,
        "reordered_inverse_fails_for_noncommuting_modules": reordered_inverse_fails,
        "premature_projection_rejected": premature_projection_rejected,
        "hidden_a_projection_rejected": raises(lambda: project_hidden_a(seed)),
        "null_carrier_rejected": raises(lambda: transaction(None, program)),
        "wrong_rank_rejected": raises(lambda: compile_program(7, 16, "PRIMARY")),
        "carrier_disabled_path_changes_boundary": disabled_boundary != active_boundary,
        "scalar_mass_only_quotient_changes_boundary": mass_only_boundary != full_boundary,
        "highest_retained_hasse_coordinate_is_boundary_relevant": perturbed_boundary != full_boundary,
    }


def algebra_checks() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for rank in RANKS:
        one = [1] + [0] * (rank - 1)
        epsilon = [0, 1] + [0] * (rank - 2)
        result[f"rank{rank}_t_to_17_is_one"] = power(phase_shift(rank, 1), 17, rank) == one
        result[f"rank{rank}_epsilon_to_rank_is_zero"] = power(epsilon, rank, rank) == [0] * rank
        result[f"rank{rank}_phase_has_order17"] = all(
            phase_shift(rank, exponent) != one for exponent in range(1, 17)
        )
        result[f"rank{rank}_public_kernels_are_units"] = all(
            multiply(
                public_kernel(rank, index, family),
                invert_unit(public_kernel(rank, index, family), rank),
                rank,
            )
            == one
            for family in FAMILIES
            for index in range(max(DEPTHS))
        )
    return result


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal(4, "PRIMARY")
    backing = carrier.backing_id()
    sealed_cells = tuple(carrier.cells)
    first = transaction(carrier, compile_program(4, 1, "PRIMARY"))
    second = transaction(carrier, compile_program(4, 16, "ALTERNATE"))
    fresh = Carrier(list(sealed_cells), 4, "PRIMARY")
    fresh_receipt = transaction(fresh, compile_program(4, 16, "ALTERNATE"))
    return {
        "same_backing_reused": carrier.backing_id() == backing,
        "restoration_generation": carrier.restoration_generation,
        "unrelated_second_boundary_matches_fresh": second["boundary"] == fresh_receipt["boundary"],
        "exact_cells_restored_after_reuse": tuple(carrier.cells) == sealed_cells,
        "snapshot_used": False,
        "first_program_family": "PRIMARY",
        "second_program_family": "ALTERNATE",
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
    }


def build_result() -> dict[str, Any]:
    cases = [
        one_case(rank, depth, family)
        for rank in RANKS
        for family in FAMILIES
        for depth in DEPTHS
    ]
    checks = controls()
    algebra = algebra_checks()
    reuse = reuse_check()
    if not all(case["matches_identical_compact_classical_recurrence"] for case in cases):
        fail("matched compact recurrence mismatch")
    if not all(case["exact_cells_restored"] and case["same_backing_restored"] for case in cases):
        fail("case restoration failure")
    if not all(checks.values()) or not all(algebra.values()):
        fail("control or algebra failure")
    if not all(
        reuse[key]
        for key in (
            "same_backing_reused",
            "unrelated_second_boundary_matches_fresh",
            "exact_cells_restored_after_reuse",
        )
    ):
        fail("reuse failure")
    maximum_rank = max(RANKS)
    return {
        "schema": "CAT_CAS_F17_C17_NILPOTENT_PHASE_JET_RESULTS_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": FIELD,
            "group_order": GROUP_ORDER,
            "ranks": list(RANKS),
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "phase_state_law": "F17_C17_REPEATED_ROOT_EPSILON_JET",
            "open_relation_semantics": "TRANSLATION_INVARIANT_WEIGHTED_RELATION_KERNEL",
            "native_composition": "CYCLIC_CONVOLUTION_DESCENDS_TO_JET_MULTIPLICATION",
            "native_nonlinearity": "B_PLUS_BETA_A_CONVOLUTION_SQUARE",
            "final_boundary_only": True,
            "hidden_a_jet_serialized": False,
            "full17_coefficient_vector_materialized_on_accepted_path": False,
            "dense17_by17_relation_table_materialized": False,
            "topology_compilation_reads_final_answer": False,
        },
        "cases": cases,
        "controls": checks,
        "algebra_checks": algebra,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "maximum_accepted_carrier_field_cells": 2 * maximum_rank,
            "maximum_logical_working_field_cells_peak": 7 * maximum_rank,
            "retained_inverse_history_field_cells": 0,
            "public_kernel_field_cells_materialized_per_step": maximum_rank,
            "full_group_algebra_semantic_reference_field_cells": 2 * GROUP_ORDER,
            "dense_relation_table_cells": 0,
            "matched_compact_classical_carrier_field_cells": 2 * maximum_rank,
            "matched_compact_classical_logical_working_field_cells_peak": 7 * maximum_rank,
            "accepted_over_matched_compact_classical_carrier_ratio": 1,
            "accepted_over_matched_compact_classical_logical_working_ratio": 1,
            "phase_and_matched_classical_field_operation_law_identical": True,
            "logical_cells_independent_of_depth_at_fixed_rank": True,
            "python_allocator_and_whole_process_peaks_excluded": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest": "IDENTICAL_RANK_R_TRUNCATED_POLYNOMIAL_RECURRENCE",
            "strongest_executed": True,
            "semantic_reference": "FULL17_COEFFICIENT_GROUP_ALGEBRA_RESERVED_FOR_INDEPENDENT_ORACLE",
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "F17_C17_TRANSLATION_INVARIANT_OPEN_RELATION_JETS_RANKS2_4_8_TWO_PUBLIC_PROGRAM_FAMILIES_DEPTHS1_4_16_64_FINAL_LINEAR_B_JET_BOUNDARIES",
        "outside_quotient": [
            "HASSE_COORDINATES_AT_DEGREE_RANK_AND_ABOVE",
            "GENERAL_NON_TRANSLATION_INVARIANT_RELATIONS",
            "GENERAL_INTERSECTION_OR_HADAMARD_PRODUCT_OF_COEFFICIENT_KERNELS",
            "ARBITRARY_GRAPH_TOPOLOGY",
        ],
        "not_established": [
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": "THE_NILPOTENT_JET_IS_AN_EXACT_FIXED_WIDTH_PHASE_ALGEBRA_QUOTIENT_BUT_THE_IDENTICAL_TRUNCATED_POLYNOMIAL_RECURRENCE_IS_AN_EQUAL_CLASSICAL_IMPLEMENTATION_AND_HIGHER_HASSE_MOMENTS_ARE_DISCARDED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_result()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
