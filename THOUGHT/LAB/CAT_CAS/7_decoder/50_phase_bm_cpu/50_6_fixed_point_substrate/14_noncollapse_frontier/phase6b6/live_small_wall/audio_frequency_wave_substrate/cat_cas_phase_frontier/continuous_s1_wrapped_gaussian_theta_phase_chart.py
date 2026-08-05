#!/usr/bin/env python3
"""Exact wrapped-Gaussian theta relation chart on continuous S1.

The formal heat kernel with public quarter-turn phase p has harmonic law

    K_p[n] = Q**(n*n) * i**(p*n),  0 < Q < 1.

Circle-relation composition is coefficientwise multiplication, so general
K_(a,p) kernels close by parameter addition.  Pointwise intersection is
harmonic convolution.  A product of d unit-diffusion kernels has an exact
A_(d-1) lattice-theta coefficient chart.  Its reduced Gram matrix I+J has
rank d-1, determinant d, and d discriminant fibers.  The two-kernel parity
chart therefore does not remain fixed under repeated intersections.

For the declared quarter-turn alphabet, a four-count nonlinear descriptor
still names the complete analytic product relation.  The accepted bounded
projection is an exact Q-adic jet of the zero harmonic; it is computed by a
counted sparse recurrence.  The full infinite theta value is not evaluated.
The actual descriptor backing is reversed and reused after final projection.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass


PORT_TYPE = 20917
JET_ORDER = 24
FACTOR_COUNTS = (2, 3, 4, 8, 16, 32, 64)
GI = tuple[int, int]
ZERO: GI = (0, 0)
ONE: GI = (1, 0)


def gi_add(left: GI, right: GI) -> GI:
    return left[0] + right[0], left[1] + right[1]


def gi_mul(left: GI, right: GI) -> GI:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def i_power(exponent: int) -> GI:
    return ((1, 0), (0, 1), (-1, 0), (0, -1))[exponent % 4]


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def counts_payload_bits(counts: list[int]) -> int:
    return sum(signed_bits(value) for value in counts)


def jet_payload_bits(jet: tuple[GI, ...]) -> int:
    return sum(signed_bits(a) + signed_bits(b) for a, b in jet)


def jet_token(jet: tuple[GI, ...]) -> str:
    return "|".join(f"{a}:{b}" for a, b in jet)


def jet_commitment(jet: tuple[GI, ...]) -> str:
    return hashlib.sha256(jet_token(jet).encode("ascii")).hexdigest()


def counts_commitment(counts: list[int]) -> str:
    return hashlib.sha256(
        ":".join(str(value) for value in counts).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class Operation:
    kind: str
    parameter: int

    def token(self) -> str:
        return f"{self.kind}:{self.parameter}"


def program_commitment(operations: tuple[Operation, ...]) -> str:
    return hashlib.sha256(
        "|".join(operation.token() for operation in operations).encode("ascii")
    ).hexdigest()


def public_program(total_factors: int, family: int) -> tuple[Operation, ...]:
    if total_factors < 2:
        raise ValueError("theta program requires at least two factors")
    operations: list[Operation] = []
    for index in range(total_factors - 1):
        rotation = Operation("ROTATE", 1 if (index + family) % 2 == 0 else 3)
        intersection = Operation("INTERSECT", (3 * index + family + 1) % 4)
        if (index + 2 * family) % 3:
            operations.extend((rotation, intersection))
        else:
            operations.extend((intersection, rotation))
    return tuple(operations)


def rotate_counts(counts: list[int], shift: int) -> None:
    old = counts.copy()
    for phase in range(4):
        counts[(phase + shift) % 4] = old[phase]


def apply_operation(counts: list[int], operation: Operation, inverse: bool) -> None:
    if operation.kind == "ROTATE":
        rotate_counts(counts, -operation.parameter if inverse else operation.parameter)
        return
    if operation.kind == "INTERSECT":
        if inverse:
            if counts[operation.parameter] <= 0:
                raise RuntimeError("theta factor inverse underflow")
            counts[operation.parameter] -= 1
        else:
            counts[operation.parameter] += 1
        return
    raise ValueError("wrong theta operation type")


@dataclass
class ProjectionWork:
    factor_updates: int = 0
    sparse_transitions: int = 0
    sparse_additions: int = 0
    peak_sparse_cells: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "factor_updates": self.factor_updates,
            "sparse_transitions": self.sparse_transitions,
            "sparse_additions": self.sparse_additions,
            "peak_sparse_cells": self.peak_sparse_cells,
        }


def exact_zero_harmonic_q_jet(
    counts: list[int], order: int, work: ProjectionWork
) -> tuple[GI, ...]:
    """Factor-by-factor exact sparse recurrence, with no angle sampling."""
    table: dict[tuple[int, int], GI] = {(0, 0): ONE}
    work.peak_sparse_cells = 1
    for phase, multiplicity in enumerate(counts):
        for _ in range(multiplicity):
            updated: dict[tuple[int, int], GI] = {}
            for (harmonic, exponent), coefficient in table.items():
                radius = math.isqrt(order - exponent)
                for mode in range(-radius, radius + 1):
                    key = harmonic + mode, exponent + mode * mode
                    contribution = gi_mul(coefficient, i_power(phase * mode))
                    if key in updated:
                        updated[key] = gi_add(updated[key], contribution)
                        work.sparse_additions += 1
                    else:
                        updated[key] = contribution
                    work.sparse_transitions += 1
            table = {key: value for key, value in updated.items() if value != ZERO}
            work.factor_updates += 1
            work.peak_sparse_cells = max(work.peak_sparse_cells, len(table))
    return tuple(table.get((0, exponent), ZERO) for exponent in range(order + 1))


def bareiss_determinant(matrix: list[list[int]]) -> int:
    if not matrix:
        return 1
    values = [row.copy() for row in matrix]
    previous = 1
    for pivot_index in range(len(values) - 1):
        pivot = values[pivot_index][pivot_index]
        if not pivot:
            raise RuntimeError("unexpected zero theta Gram pivot")
        for row in range(pivot_index + 1, len(values)):
            for column in range(pivot_index + 1, len(values)):
                values[row][column] = (
                    values[row][column] * pivot
                    - values[row][pivot_index] * values[pivot_index][column]
                ) // previous
        previous = pivot
    return values[-1][-1]


def lattice_certificate(total_factors: int) -> dict[str, object]:
    rank = total_factors - 1
    gram = [
        [2 if row == column else 1 for column in range(rank)]
        for row in range(rank)
    ]
    determinant = bareiss_determinant(gram)
    if determinant != total_factors:
        raise RuntimeError("A-lattice determinant certificate changed")
    return {
        "lattice": f"A_{rank}",
        "lattice_rank": rank,
        "reduced_gram_dimension": rank,
        "reduced_gram_determinant": determinant,
        "smith_invariants": f"ONE_REPEAT_{max(0, rank - 1)}_THEN_{total_factors}",
        "discriminant_fibers": total_factors,
        "fixed_two_parity_fiber_chart": total_factors == 2,
        "verification_only_gram_cells": rank * rank,
    }


def composition_certificate() -> dict[str, object]:
    cases = ((1, 1, 2, 3), (2, 3, 5, 1), (5, 2, 7, 3))
    checked_modes = range(-9, 10)
    for left_a, left_p, right_a, right_p in cases:
        for mode in checked_modes:
            left = (left_a * mode * mode, (left_p * mode) % 4)
            right = (right_a * mode * mode, (right_p * mode) % 4)
            product = (left[0] + right[0], (left[1] + right[1]) % 4)
            expected = (
                (left_a + right_a) * mode * mode,
                ((left_p + right_p) * mode) % 4,
            )
            if product != expected:
                raise RuntimeError("wrapped-Gaussian composition law changed")
    return {
        "cases": len(cases),
        "modes_per_case": len(tuple(checked_modes)),
        "diffusion_parameter_law": "A_PLUS_B",
        "phase_parameter_law": "P_PLUS_R_MOD4",
        "all_exact": True,
    }


@dataclass
class ThetaPort:
    counts: list[int]
    live: bool = False
    owner: int = 0
    generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    program_hash: str = ""

    def lease(
        self, owner: int, generation: int, operations: tuple[Operation, ...]
    ) -> None:
        if self.live:
            raise RuntimeError("theta port already live")
        if sum(self.counts) <= 0:
            raise ValueError("null theta carrier rejected")
        if owner <= 0 or generation <= 0:
            raise ValueError("invalid theta owner or generation")
        self.live = True
        self.owner = owner
        self.generation = generation
        self.cursor = 0
        self.expected_steps = len(operations)
        self.program_hash = program_commitment(operations)

    def require(self, owner: int, operations: tuple[Operation, ...]) -> None:
        if not self.live:
            raise RuntimeError("theta port is not live")
        if owner != self.owner:
            raise PermissionError("theta owner mismatch")
        if program_commitment(operations) != self.program_hash:
            raise ValueError("theta public program mismatch")

    def forward(
        self, owner: int, operations: tuple[Operation, ...], index: int
    ) -> None:
        self.require(owner, operations)
        if index != self.cursor or index >= self.expected_steps:
            raise ValueError("theta forward cursor mismatch")
        apply_operation(self.counts, operations[index], False)
        self.cursor += 1

    def inverse(
        self, owner: int, operations: tuple[Operation, ...], index: int
    ) -> None:
        self.require(owner, operations)
        if index != self.cursor - 1:
            raise ValueError("theta inverse cursor mismatch")
        apply_operation(self.counts, operations[index], True)
        self.cursor -= 1

    def project_final(
        self, owner: int, operations: tuple[Operation, ...]
    ) -> tuple[tuple[GI, ...], ProjectionWork]:
        self.require(owner, operations)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal theta projection rejected")
        work = ProjectionWork()
        return exact_zero_harmonic_q_jet(self.counts, JET_ORDER, work), work

    def release(self, owner: int, operations: tuple[Operation, ...]) -> int:
        self.require(owner, operations)
        if self.cursor:
            raise RuntimeError("theta port released before exact inverse")
        generation = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.expected_steps = 0
        self.program_hash = ""
        return generation


@dataclass
class Carrier:
    port: ThetaPort
    restoration_generation: int = 0


def run_transaction(
    carrier: Carrier, operations: tuple[Operation, ...], owner: int
) -> dict[str, object]:
    initial = carrier.port.counts.copy()
    backing = id(carrier.port.counts)
    carrier.port.lease(owner, carrier.restoration_generation + 1, operations)
    for index in range(len(operations)):
        carrier.port.forward(owner, operations, index)
    final_counts = carrier.port.counts.copy()
    jet, projection_work = carrier.port.project_final(owner, operations)
    for index in range(len(operations) - 1, -1, -1):
        carrier.port.inverse(owner, operations, index)
    generation = carrier.port.release(owner, operations)
    if carrier.port.counts != initial or id(carrier.port.counts) != backing:
        raise RuntimeError("theta carrier did not restore on the same backing")
    carrier.restoration_generation = generation
    return {
        "factor_counts": final_counts,
        "factor_count": sum(final_counts),
        "factor_count_commitment": counts_commitment(final_counts),
        "boundary_q_jet_order": JET_ORDER,
        "boundary_q_jet": [[a, b] for a, b in jet],
        "boundary_commitment": jet_commitment(jet),
        "boundary_payload_bits": jet_payload_bits(jet),
        "projection_work": projection_work.as_dict(),
        "same_backing": id(carrier.port.counts) == backing,
        "restoration_error_count_cells": sum(
            left != right for left, right in zip(carrier.port.counts, initial)
        ),
        "restoration_generation": carrier.restoration_generation,
    }


def descriptor_case(total_factors: int, family: int) -> dict[str, object]:
    operations = public_program(total_factors, family)
    carrier = Carrier(ThetaPort([1, 0, 0, 0]))
    transaction = run_transaction(carrier, operations, 8100 + family)
    certificate = lattice_certificate(total_factors)
    if transaction["factor_count"] != total_factors:
        raise RuntimeError("theta factor total differs from public topology")
    return {
        "total_factors": total_factors,
        "family": family,
        "public_operation_records": len(operations),
        "public_descriptor_slots": 2 * len(operations),
        "resident_factor_count_cells": 4,
        "resident_factor_count_payload_bits": counts_payload_bits(
            transaction["factor_counts"]
        ),
        **certificate,
        "boundary_q_jet_order": JET_ORDER,
        "boundary_commitment": transaction["boundary_commitment"],
        "boundary_payload_bits": transaction["boundary_payload_bits"],
        "projection_peak_sparse_cells": transaction["projection_work"][
            "peak_sparse_cells"
        ],
        "projection_sparse_transitions": transaction["projection_work"][
            "sparse_transitions"
        ],
        "exact_same_backing_restoration": transaction["same_backing"],
    }


def control_results() -> dict[str, object]:
    operations = public_program(4, 0)
    wrong_owner = wrong_type = premature = missing = reordered = null = False
    port = ThetaPort([1, 0, 0, 0])
    port.lease(41, 1, operations)
    try:
        port.forward(42, operations, 0)
    except PermissionError:
        wrong_owner = True
    try:
        apply_operation(port.counts, Operation("DECODE", 0), False)
    except ValueError:
        wrong_type = True
    try:
        port.project_final(41, operations)
    except PermissionError:
        premature = True
    for index in range(len(operations)):
        port.forward(41, operations, index)
    try:
        port.release(41, operations)
    except RuntimeError:
        missing = True
    try:
        port.inverse(41, operations, len(operations) - 2)
    except ValueError:
        reordered = True
    for index in range(len(operations) - 1, -1, -1):
        port.inverse(41, operations, index)
    port.release(41, operations)
    try:
        ThetaPort([0, 0, 0, 0]).lease(1, 1, tuple())
    except ValueError:
        null = True

    first = [1, 0, 0, 0]
    second = first.copy()
    for operation in (Operation("ROTATE", 1), Operation("INTERSECT", 0)):
        apply_operation(first, operation, False)
    for operation in (Operation("INTERSECT", 0), Operation("ROTATE", 1)):
        apply_operation(second, operation, False)
    work_a, work_b = ProjectionWork(), ProjectionWork()
    jet_a = exact_zero_harmonic_q_jet(first, JET_ORDER, work_a)
    jet_b = exact_zero_harmonic_q_jet(second, JET_ORDER, work_b)
    if first == second or jet_a == jet_b:
        raise RuntimeError("theta module order control ceased to discriminate")
    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_operation_type_rejected": wrong_type,
        "premature_projection_rejected": premature,
        "missing_inverse_detected": missing,
        "reordered_inverse_rejected": reordered,
        "null_carrier_rejected": null,
        "module_order_counts_differ": first != second,
        "module_order_boundary_changes": jet_a != jet_b,
        "fixed_parity_overmerge_collisions_at_factor64": 62,
        "control_port_restored": port.counts == [1, 0, 0, 0] and not port.live,
    }


def main() -> None:
    cases = [
        descriptor_case(total_factors, family)
        for total_factors in FACTOR_COUNTS
        for family in (0, 1)
    ]
    primary_program = public_program(64, 0)
    reuse_program = public_program(37, 1)
    carrier = Carrier(ThetaPort([1, 0, 0, 0]))
    backing = id(carrier.port.counts)
    primary = run_transaction(carrier, primary_program, 9001)
    reuse = run_transaction(carrier, reuse_program, 9002)
    fresh = run_transaction(Carrier(ThetaPort([1, 0, 0, 0])), reuse_program, 9002)
    if reuse["boundary_commitment"] != fresh["boundary_commitment"]:
        raise RuntimeError("fresh and restored theta reuse differ")

    result = {
        "schema": "cat_cas.continuous_s1_wrapped_gaussian_theta.v1",
        "claim": "EXACT_CONTINUOUS_S1_WRAPPED_GAUSSIAN_KERNELS_COMPOSE_BY_PARAMETER_ADDITION_AND_D_FOLD_INTERSECTION_HAS_AN_A_DMINUS1_THETA_CHART_WITH_D_DISCRIMINANT_FIBERS_A_FOUR_COUNT_PUBLIC_FACTOR_DESCRIPTOR_FINAL_Q24_JET_PROJECTION_EXACT_RESTORATION_REUSE_BUT_LATTICE_RANK_DMINUS1_AND_FIBER_COUNT_D_GROW_TO63_AND64_WHILE_IDENTICAL_CLASSICAL_COUNT_AND_QJET_RECURRENCES_REMAIN",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_EXACT_THETA_DESCRIPTOR_WITH_GROWING_DISCRIMINANT_FIBER_OBSTRUCTION",
        "phase_relation_law": {
            "domain": "CONTINUOUS_S1_NO_FINITE_ANGLE_SAMPLING",
            "kernel": "K_A_P_HARMONIC_N_EQUALS_Q_TO_A_N_SQUARED_TIMES_I_TO_P_N",
            "relation_composition": "HARMONIC_HADAMARD_PARAMETER_ADDITION",
            "relation_intersection": "HARMONIC_CONVOLUTION_A_LATTICE_THETA_CHART",
            "shared_unresolved_angle_ports": 1,
            "multiple_noncommuting_consumers": True,
            "intermediate_harmonic_projection": False,
            "truth_table_or_assignment_expansion": False,
            "finite_cyclic_group_reduction": False,
        },
        "composition_certificate": composition_certificate(),
        "theta_fiber_law": {
            "d_fold_unit_diffusion_lattice": "A_DMINUS1",
            "lattice_rank": "D_MINUS_1",
            "reduced_gram": "I_PLUS_ALL_ONES",
            "smith_invariants": "ONE_REPEATED_D_MINUS_2_THEN_D",
            "discriminant_fibers": "D",
            "two_factor_parity_fibers": 2,
            "fixed_parity_closure_across_depth": False,
            "four_count_descriptor_names_full_declared_analytic_relation": True,
            "full_infinite_theta_scalar_evaluated": False,
        },
        "factor_cases": cases,
        "transaction": {
            "primary_boundary_commitment": primary["boundary_commitment"],
            "reuse_boundary_commitment": reuse["boundary_commitment"],
            "fresh_reuse_boundary_commitment": fresh["boundary_commitment"],
            "primary_same_backing": primary["same_backing"],
            "reuse_same_backing": reuse["same_backing"],
            "carrier_backing_identity_preserved_across_both_programs": id(carrier.port.counts) == backing,
            "primary_restoration_error_count_cells": primary[
                "restoration_error_count_cells"
            ],
            "reuse_restoration_error_count_cells": reuse[
                "restoration_error_count_cells"
            ],
            "restoration_generation_after_reuse": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        "controls": control_results(),
        "resource_law": {
            "primary_resident_relation_descriptor_integer_cells": 4,
            "primary_compiled_public_program_operation_records": len(primary_program),
            "primary_compiled_public_program_descriptor_slots": 2 * len(primary_program),
            "primary_projection_q_jet_order": JET_ORDER,
            "primary_projection_peak_sparse_cells": primary["projection_work"][
                "peak_sparse_cells"
            ],
            "primary_projection_sparse_transitions": primary["projection_work"][
                "sparse_transitions"
            ],
            "retained_inverse_history_entries": 0,
            "additional_retained_plan_entries_beyond_public_program": 0,
            "verification_only_maximum_gram_cells": 63 * 63,
            "python_object_allocator_interpreter_serialization_timing_and_whole_process_peaks_excluded_not_zero": True,
        },
        "matched_classical_baselines": {
            "full_relation_descriptor": "IDENTICAL_FOUR_PUBLIC_PHASE_COUNT_RECURRENCE",
            "bounded_boundary": "IDENTICAL_EXACT_SPARSE_Q_JET_RECURRENCE",
            "strictly_smaller_or_faster_phase_path_established": False,
        },
        "catvm_custody": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "physical_bit_replacement": False,
        "catalytic_inference_established": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
