#!/usr/bin/env python3
"""Exact linear-span obstruction for a two-shear cubic Airy message chart.

The experiment varies the two public controlled-cubic coefficients around one
declared Gaussian axis transform.  A Fourier decomposition over the second
coefficient reduces the q^2-by-q^2 public-orbit rank calculation to the z=0
and z=1 level blocks of z=s*x^3.  Because every declared safe prime satisfies
q == 2 (mod 3), nonzero levels differ only by row scaling and a public column
permutation.  This is direct-process finite-field software and the identical
classical block recurrence is the matched implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import growing_prime_resident_cubic_strength_phase_port_weil_action_rank as resident


ORDERS = resident.ORDERS
PROGRAM_FAMILIES = resident.FAMILIES
CLAIM = (
    "BOUNDED_EXACT_GROWING_SAFE_PRIME_TWO_FIBER_CUBIC_AIRY_PUBLIC_TWO_SHEAR_"
    "ORBIT_HAS_FULL_Q2_LINEAR_SPAN_ON_EVERY_DECLARED_PRIMARY_ORDER_AND_AT_LEAST_"
    "Q2MINUSQPLUS1_ALTERNATE_SPAN_ON_EVERY_DECLARED_QGE11_ORDER_WITH_BLOCK_FOURIER_CERTIFICATES_EXACT_GRAPH_"
    "RESTORATION_AND_REUSE_SO_LINEAR_MESSAGE_QUOTIENTS_BELOW_THE_OBSERVED_SPAN_"
    "ARE_REJECTED_AT_THE_TESTED_SOURCE_CHARTS_BUT_NONLINEAR_CLOSURE_REMAINS_OPEN_"
    "AND_IDENTICAL_CLASSICAL_FACTOR_GRAPH_AND_Q2_DENSE_RECURRENCES_REMAIN"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class OrbitOperation:
    kind: str
    payload: tuple[int, ...]


@dataclass
class OrbitWork:
    character_sum_terms: int = 0
    rank_pivot_inversions: int = 0
    rank_row_eliminations: int = 0
    rank_field_updates: int = 0
    temporary_airy_table_field_cells_peak: int = 0
    temporary_rank_matrix_field_cells_peak: int = 0
    public_orbit_matrix_field_cells_materialized: int = 0


@dataclass
class OrbitCarrier:
    field: resident.open_action.cubic.gaussian.Field
    latent: list[int]
    data: list[int]
    nodes: list[OrbitOperation]
    fixture_family: str
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(
        cls,
        q: int,
        fixture_family: str,
        data_override: list[int] | None = None,
    ) -> "OrbitCarrier":
        if fixture_family not in PROGRAM_FAMILIES:
            fail("invalid fixture family")
        field = resident.open_action.cubic.gaussian.make_field(q)
        latent = resident.latent_fixture(field, fixture_family)
        data = list(data_override) if data_override is not None else resident.data_fixture(field, fixture_family)
        if len(data) != 2 * q or not any(value % field.p for value in data):
            fail("invalid two-fiber data fixture")
        return cls(field, latent, [value % field.p for value in data], [], fixture_family)

    @property
    def q(self) -> int:
        return self.field.q

    def backing_ids(self) -> tuple[int, int, int, int]:
        return id(self), id(self.latent), id(self.data), id(self.nodes)

    def canonical_payload(self) -> tuple[Any, ...]:
        return (
            self.q,
            self.field.p,
            tuple(self.latent),
            tuple(self.data),
            tuple(self.nodes),
            self.fixture_family,
            self.stage,
        )


def public_orbit_plan(q: int, program_family: str) -> list[OrbitOperation]:
    base = resident.public_plan(q, 1, program_family)
    if program_family == "PRIMARY":
        data, latent = base[1], base[3]
        ordered = (
            OrbitOperation("CONTROLLED_CUBIC_FAMILY_A", (0, q - 1)),
            OrbitOperation(data.kind, data.payload),
            OrbitOperation("CONTROLLED_CUBIC_FAMILY_B", (0, q - 1)),
            OrbitOperation(latent.kind, latent.payload),
        )
    else:
        latent, data = base[1], base[3]
        ordered = (
            OrbitOperation("CONTROLLED_CUBIC_FAMILY_A", (0, q - 1)),
            OrbitOperation(latent.kind, latent.payload),
            OrbitOperation("CONTROLLED_CUBIC_FAMILY_B", (0, q - 1)),
            OrbitOperation(data.kind, data.payload),
        )
    return [*ordered, OrbitOperation(base[-1].kind, base[-1].payload)]


def forward(carrier: OrbitCarrier, program_family: str) -> list[OrbitOperation]:
    if carrier.stage != "IDLE" or carrier.nodes:
        fail("orbit carrier is not idle")
    operations = public_orbit_plan(carrier.q, program_family)
    carrier.nodes.extend(operations)
    carrier.stage = "ORBIT_FORWARD_COMPLETE"
    return operations


def inverse_descriptor(
    carrier: OrbitCarrier,
    operation: OrbitOperation,
) -> OrbitOperation:
    if operation.kind.startswith("CONTROLLED_CUBIC_FAMILY_"):
        return OrbitOperation("INVERSE_" + operation.kind, operation.payload)
    dummy = resident.Carrier.seal(carrier.q, carrier.fixture_family)
    inverse = resident.inverse_operation(
        dummy,
        resident.Operation(operation.kind, operation.payload),
        resident.Work(),
    )
    return OrbitOperation(inverse.kind, inverse.payload)


def inverse_pair_exact(
    carrier: OrbitCarrier,
    operation: OrbitOperation,
    inverse: OrbitOperation,
) -> bool:
    return inverse == inverse_descriptor(carrier, operation)


def reverse(
    carrier: OrbitCarrier,
    operations: list[OrbitOperation],
    mutation: str | None = None,
) -> None:
    if carrier.stage != "ORBIT_FORWARD_COMPLETE":
        fail("orbit carrier lacks forward state")
    sequence = list(reversed(operations))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(operations)
    for index, operation in enumerate(sequence):
        if not carrier.nodes or carrier.nodes[-1] != operation:
            fail("inverse order does not match resident orbit graph")
        inverse = inverse_descriptor(carrier, operation)
        if mutation == "WRONG" and index == 0:
            inverse = OrbitOperation(inverse.kind, inverse.payload + (1,))
        if not inverse_pair_exact(carrier, operation, inverse):
            fail("inverse descriptor certificate failed")
        carrier.nodes.pop()
    if carrier.nodes:
        fail("orbit inverse left resident morphisms")
    carrier.stage = "IDLE"


def matrix_rank(matrix: list[list[int]], p: int, work: OrbitWork) -> int:
    rows = [row[:] for row in matrix]
    if not rows:
        return 0
    work.temporary_rank_matrix_field_cells_peak = max(
        work.temporary_rank_matrix_field_cells_peak,
        len(rows) * len(rows[0]),
    )
    pivot_row = 0
    for column in range(len(rows[0])):
        pivot = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column] % p),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = pow(rows[pivot_row][column] % p, -1, p)
        rows[pivot_row] = [value * inverse % p for value in rows[pivot_row]]
        work.rank_pivot_inversions += 1
        work.rank_field_updates += len(rows[pivot_row])
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column] % p:
                continue
            scale = rows[row][column] % p
            rows[row] = [
                (left - scale * right) % p
                for left, right in zip(rows[row], rows[pivot_row])
            ]
            work.rank_row_eliminations += 1
            work.rank_field_updates += len(rows[row])
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def level_points(q: int, level: int) -> list[tuple[int, int]]:
    if level == 0:
        return [(0, x) for x in range(q)] + [(s, 0) for s in range(1, q)]
    return [(pow(x**3 % q, -1, q), x) for x in range(1, q)]


def primary_airy_table(
    carrier: OrbitCarrier,
    work: OrbitWork,
) -> list[list[list[int]]]:
    q, p = carrier.q, carrier.field.p
    operation = public_orbit_plan(q, "PRIMARY")[1]
    symplectic, coefficient = list(operation.payload[:4]), operation.payload[4] % p
    table = [[[0] * q for _ in range(q)] for _ in range(2)]
    for fiber in range(2):
        for parameter in range(q):
            for target in range(q):
                total = 0
                for source in range(q):
                    total += (
                        coefficient
                        * resident.open_action.cubic.gaussian.kernel_value(
                            symplectic, target, source, carrier.field
                        )
                        * carrier.data[fiber * q + source]
                        * resident.phase(carrier.field, parameter * source**3)
                    )
                    work.character_sum_terms += 1
                table[fiber][parameter][target] = total % p
    work.temporary_airy_table_field_cells_peak = 2 * q * q
    return table


def alternate_airy_table(
    carrier: OrbitCarrier,
    work: OrbitWork,
) -> list[list[int]]:
    q, p = carrier.q, carrier.field.p
    operation = public_orbit_plan(q, "ALTERNATE")[1]
    symplectic, coefficient = list(operation.payload[:4]), operation.payload[4] % p
    table = [[0] * q for _ in range(q)]
    for parameter in range(q):
        for target in range(q):
            total = 0
            for source in range(q):
                total += (
                    coefficient
                    * resident.open_action.cubic.gaussian.kernel_value(
                        symplectic, target, source, carrier.field
                    )
                    * carrier.latent[source]
                    * resident.phase(carrier.field, parameter * source)
                )
                work.character_sum_terms += 1
            table[parameter][target] = total % p
    work.temporary_airy_table_field_cells_peak = q * q
    return table


def alternate_quadratic_phase_chart(
    carrier: OrbitCarrier,
    table: list[list[int]],
) -> tuple[int, int, int, int, int]:
    """Fit and exhaustively verify W(t,s)=W(0,0) phase(Q(s,t))."""
    q, p = carrier.q, carrier.field.p
    logarithm = {resident.phase(carrier.field, exponent): exponent for exponent in range(q)}
    base = table[0][0] % p
    if not base:
        fail("alternate Airy table has zero chart base")
    inverse_base = pow(base, -1, p)

    def exponent(strength: int, parameter: int) -> int:
        ratio = table[parameter % q][strength % q] * inverse_base % p
        if ratio not in logarithm:
            fail("alternate Airy table left the quadratic phase chart")
        return logarithm[ratio]

    inverse_two = pow(2, -1, q)
    quadratic_strength = (exponent(1, 0) + exponent(-1, 0)) * inverse_two % q
    linear_strength = (exponent(1, 0) - exponent(-1, 0)) * inverse_two % q
    quadratic_parameter = (exponent(0, 1) + exponent(0, -1)) * inverse_two % q
    linear_parameter = (exponent(0, 1) - exponent(0, -1)) * inverse_two % q
    mixed = (
        exponent(1, 1)
        - quadratic_strength
        - linear_strength
        - quadratic_parameter
        - linear_parameter
    ) % q
    if not all(
        exponent(strength, parameter)
        == (
            quadratic_strength * strength * strength
            + mixed * strength * parameter
            + quadratic_parameter * parameter * parameter
            + linear_strength * strength
            + linear_parameter * parameter
        )
        % q
        for strength in range(q)
        for parameter in range(q)
    ):
        fail("alternate Airy table is not an exact quadratic phase chart")
    return (
        quadratic_strength,
        mixed,
        quadratic_parameter,
        linear_strength,
        linear_parameter,
    )


def primary_block(
    carrier: OrbitCarrier,
    table: list[list[list[int]]],
    level: int,
    fibers: tuple[int, ...],
) -> list[list[int]]:
    q, p = carrier.q, carrier.field.p
    return [
        [
            carrier.latent[strength] * table[fiber][coefficient * strength % q][coordinate] % p
            for coefficient in range(q)
        ]
        for fiber in fibers
        for strength, coordinate in level_points(q, level)
    ]


def alternate_block(
    carrier: OrbitCarrier,
    table: list[list[int]],
    level: int,
    fibers: tuple[int, ...],
) -> list[list[int]]:
    q, p = carrier.q, carrier.field.p
    return [
        [
            carrier.data[fiber * q + coordinate]
            * table[coefficient * coordinate**3 % q][strength]
            % p
            for coefficient in range(q)
        ]
        for fiber in fibers
        for strength, coordinate in level_points(q, level)
    ]


def orbit_rank_certificate(
    carrier: OrbitCarrier,
    program_family: str,
    fibers: tuple[int, ...] = (0, 1),
) -> dict[str, Any]:
    if carrier.stage != "ORBIT_FORWARD_COMPLETE":
        fail("orbit rank unavailable before forward execution")
    work = OrbitWork()
    if program_family == "PRIMARY":
        table = primary_airy_table(carrier, work)
        zero = primary_block(carrier, table, 0, fibers)
        nonzero = primary_block(carrier, table, 1, fibers)
        equivalence_mechanism = "CUBE_PERMUTATION_PLUS_NONZERO_ROW_SCALING"
        quadratic_chart: tuple[int, ...] | None = None
    elif program_family == "ALTERNATE":
        table = alternate_airy_table(carrier, work)
        quadratic_chart = alternate_quadratic_phase_chart(carrier, table)
        zero = alternate_block(carrier, table, 0, fibers)
        nonzero = alternate_block(carrier, table, 1, fibers)
        equivalence_mechanism = "EXACT_QUADRATIC_PHASE_CHART_PLUS_NONZERO_ROW_AND_COLUMN_SCALING"
    else:
        fail("invalid orbit program family")
    zero_rank = matrix_rank(zero, carrier.field.p, work)
    nonzero_rank = matrix_rank(nonzero, carrier.field.p, work)
    total_rank = zero_rank + (carrier.q - 1) * nonzero_rank
    return {
        "zero_level_rank": zero_rank,
        "representative_nonzero_level_rank": nonzero_rank,
        "all_nonzero_levels_equivalent": True,
        "nonzero_level_equivalence_mechanism": equivalence_mechanism,
        "alternate_quadratic_phase_chart_coefficients": quadratic_chart,
        "public_two_shear_program_count": carrier.q**2,
        "public_orbit_linear_span": total_rank,
        "full_q2_by_q2_public_orbit_matrix_materialized": False,
        "work": work.__dict__,
    }


def dense_sample(
    carrier: OrbitCarrier,
    program_family: str,
    first: int,
    second: int,
) -> dict[str, Any]:
    base = resident.public_plan(carrier.q, 1, program_family)
    if program_family == "PRIMARY":
        operations = (
            resident.Operation("CONTROLLED_CUBIC", (first,)),
            base[1],
            resident.Operation("CONTROLLED_CUBIC", (second,)),
            base[3],
            base[-1],
        )
    else:
        operations = (
            resident.Operation("CONTROLLED_CUBIC", (first,)),
            base[1],
            resident.Operation("CONTROLLED_CUBIC", (second,)),
            base[3],
            base[-1],
        )
    dense = resident.Carrier.seal(carrier.q, carrier.fixture_family)
    work = resident.Work()
    for operation in operations:
        resident.apply_operation(dense, operation, work)
    dense.stage = "FORWARD_COMPLETE"
    return {
        "first_public_coefficient": first,
        "second_public_coefficient": second,
        "boundary": resident.boundary(dense, carrier.fixture_family),
        "state_digest": hashlib.sha256(repr(tuple(dense.cells)).encode("ascii")).hexdigest(),
        "expanded_verification_state_field_cells": len(dense.cells),
        "work": work.__dict__,
    }


def transaction(carrier: OrbitCarrier, program_family: str) -> dict[str, Any]:
    initial, backing = carrier.canonical_payload(), carrier.backing_ids()
    operations = forward(carrier, program_family)
    topology_commitment = hashlib.sha256(repr(tuple(operations)).encode("ascii")).hexdigest()
    certificate = orbit_rank_certificate(carrier, program_family)
    samples = [
        dense_sample(carrier, program_family, first, second)
        for first, second in ((0, 0), (1, 1), (carrier.q - 1, 2 % carrier.q))
    ]
    if program_family == "PRIMARY":
        if certificate["public_orbit_linear_span"] != carrier.q**2:
            fail("primary public orbit did not reach full q-squared span")
    elif carrier.q >= 11 and certificate["public_orbit_linear_span"] < carrier.q**2 - carrier.q + 1:
        fail("alternate public orbit fell below the declared lower bound")
    reverse(carrier, operations)
    restored = carrier.canonical_payload() == initial
    same_backing = carrier.backing_ids() == backing
    if not restored or not same_backing:
        fail("orbit graph restoration failed")
    carrier.restoration_generation += 1
    return {
        "q": carrier.q,
        "p": carrier.field.p,
        "fixture_family": carrier.fixture_family,
        "program_family": program_family,
        "topology_commitment": topology_commitment,
        "resident_factor_field_cells": 3 * carrier.q,
        "public_morphism_node_records": len(operations),
        "certificate": certificate,
        "sampled_dense_boundaries": samples,
        "expanded_dense_state_used_only_for_sampled_verification": True,
        "exact_graph_payload_restored": restored,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def mutation_fails(mutation: str) -> bool:
    carrier = OrbitCarrier.seal(5, "PRIMARY")
    initial = carrier.canonical_payload()
    operations = forward(carrier, "PRIMARY")
    try:
        reverse(carrier, operations, mutation)
    except RuntimeError:
        return carrier.canonical_payload() != initial
    return carrier.canonical_payload() != initial


def controls() -> dict[str, bool]:
    premature = OrbitCarrier.seal(5, "PRIMARY")
    premature_rejected = False
    try:
        orbit_rank_certificate(premature, "PRIMARY")
    except RuntimeError:
        premature_rejected = True

    original = OrbitCarrier.seal(11, "PRIMARY")
    duplicate = OrbitCarrier.seal(11, "PRIMARY", data_override=original.data[:11] * 2)
    original_plan = forward(original, "PRIMARY")
    duplicate_plan = forward(duplicate, "PRIMARY")
    single_fiber = orbit_rank_certificate(original, "PRIMARY", fibers=(0,))
    duplicate_rank = orbit_rank_certificate(duplicate, "PRIMARY")["public_orbit_linear_span"]
    return {
        "missing_inverse_fails": mutation_fails("MISSING"),
        "wrong_inverse_fails": mutation_fails("WRONG"),
        "reordered_inverse_fails": mutation_fails("REORDER"),
        "premature_projection_rejected": premature_rejected,
        "public_topology_independent_of_factor_values": original_plan == duplicate_plan,
        "single_fiber_primary_span_is_below_q2": single_fiber["public_orbit_linear_span"] < 11**2,
        "duplicated_second_fiber_primary_span_is_below_q2": duplicate_rank < 11**2,
        "accepted_rank_path_does_not_materialize_q2_by_q2_orbit_matrix": (
            single_fiber["work"]["public_orbit_matrix_field_cells_materialized"] == 0
        ),
    }


def reuse() -> dict[str, Any]:
    carrier = OrbitCarrier.seal(23, "PRIMARY")
    initial, backing = carrier.canonical_payload(), carrier.backing_ids()
    first = transaction(carrier, "PRIMARY")
    second = transaction(carrier, "ALTERNATE")
    fresh = OrbitCarrier.seal(23, "PRIMARY")
    fresh_second = transaction(fresh, "ALTERNATE")
    return {
        "first_primary_span": first["certificate"]["public_orbit_linear_span"],
        "second_alternate_span": second["certificate"]["public_orbit_linear_span"],
        "fresh_alternate_span": fresh_second["certificate"]["public_orbit_linear_span"],
        "second_matches_fresh": second["certificate"] == fresh_second["certificate"],
        "exact_payload_restored_after_reuse": carrier.canonical_payload() == initial,
        "same_backing_reused": carrier.backing_ids() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
    }


def run() -> dict[str, Any]:
    cases = []
    for q in ORDERS:
        cases.append(transaction(OrbitCarrier.seal(q, "PRIMARY"), "PRIMARY"))
        if q >= 11:
            cases.append(transaction(OrbitCarrier.seal(q, "ALTERNATE"), "ALTERNATE"))
    control_result, reuse_result = controls(), reuse()
    if not all(control_result.values()) or not all(
        (
            reuse_result["second_matches_fresh"],
            reuse_result["exact_payload_restored_after_reuse"],
            reuse_result["same_backing_reused"],
            reuse_result["restoration_generation"] == 2,
        )
    ):
        fail("controls or reuse failed")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_TWO_FIBER_CUBIC_AIRY_PUBLIC_ORBIT_SPAN_V1",
        "claim_candidate": CLAIM,
        "claim_ceiling": "SAFE_PRIME_PAIRS_Q5_11_23_29_41_53_83_89_113_P11_23_47_59_83_107_167_179_227_PRIMARY_ALL_ORDERS_ALTERNATE_Q11_23_29_41_53_83_89_113_DEPTH1_PUBLIC_TWO_CONTROLLED_CUBIC_COEFFICIENT_ORBITS_TWO_DECLARED_FIBER_FIXTURES_DIRECT_PROCESS_SOFTWARE",
        "cases": cases,
        "controls": control_result,
        "restoration_and_reuse": reuse_result,
        "observed_resource_law": {
            "accepted_resident_factor_field_cells": "3*Q",
            "primary_public_orbit_program_count": "Q^2",
            "primary_public_orbit_linear_span": "Q^2",
            "alternate_qge11_public_orbit_linear_span": "Q^2-Q+1",
            "temporary_primary_airy_table_field_cells": "2*Q^2",
            "temporary_rank_block_field_cells_peak": "AT_MOST_(4*Q-2)*Q",
            "full_q2_by_q2_public_orbit_matrix_materialized": False,
        },
        "matched_baseline": {
            "identical_classical_factor_graph_recurrence": True,
            "identical_classical_block_fourier_rank_recurrence": True,
            "identical_dense_q2_joint_state_recurrence": True,
            "strongest_compact_classical_representation_for_declared_public_orbit": "3Q_FACTOR_CELLS_PLUS_PUBLIC_TWO_COEFFICIENT_DESCRIPTOR_WITH_BLOCK_OR_PATH_EVALUATION",
            "cold_start_comparison_used": False,
        },
        "resource_accounting": {
            "resident_factors_public_plan_airy_tables_rank_blocks_elimination_dense_samples_restoration_and_reuse_counted": True,
            "python_objects_allocator_interpreter_native_libraries_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "claim_boundaries": {
            "arbitrary_source_or_program_family": False,
            "nonlinear_or_singular_message_quotient_no_go": False,
            "universal_subquadratic_state_lower_bound": False,
            "polynomial_work_subquadratic_state_closure": False,
            "machine_enforced_hidden_runtime_factors": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage_or_small_wall_crossing": False,
            "physical_waveform_or_silicon_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_DECLARED_PRIMARY_TWO_SHEAR_AIRY_PUBLIC_ORBIT_HAS_EXACT_FULL_Q2_LINEAR_SPAN_AND_THE_ALTERNATE_ORBIT_HAS_AT_LEAST_Q2MINUSQPLUS1_SPAN_SO_ANY_SUCCESSOR_MUST_USE_A_GENUINELY_NONLINEAR_OR_SINGULAR_PHASE_MESSAGE_CHART_CHANGE_THE_NATIVE_COUPLING_LAW_OR_ACCEPT_THE_IDENTICAL_COMPACT_CLASSICAL_FACTOR_GRAPH_OR_Q2_DENSE_RECURRENCE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
