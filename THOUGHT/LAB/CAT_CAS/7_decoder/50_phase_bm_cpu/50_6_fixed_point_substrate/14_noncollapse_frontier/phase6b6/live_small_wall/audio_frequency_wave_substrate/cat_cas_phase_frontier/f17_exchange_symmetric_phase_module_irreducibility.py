#!/usr/bin/env python3
"""Exact linear-quotient diagnostic for the M127 symmetric phase carrier.

This successor adds two typed native primitives to the existing occupation
carrier: power-sum phase characters and bidirectional adjacent-mode shears.
The public primitive family is deliberately broader than one fixed program.
For the declared exchange-symmetric degrees, the power sums separate every
occupation orbit and the lifted shear support is connected.  Those two facts
certify that the generated linear phase-module action has no nonzero proper
invariant kernel, hence no smaller exact linear quotient preserving every
declared primitive and the final boundary.

The accepted transaction still uses the M127 carrier and both grid modules,
projects one final scalar, reverses the actual operations, and reuses the same
backing.  This is a bounded direct-process no-go, not a CATVM result or a
distinct phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import f17_exchange_symmetric_latent_geometry_closure as m127
import f17_nonlinear_canonical_mps_separator_chart as backend


PRIME = 17
STRUCTURAL_K = (1, 2, 3, 4)
EXACT_CASES = ((1, "PRIMARY"), (2, "PRIMARY"), (2, "REUSE"))
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PhaseModuleProgram:
    base: m127.ExchangeProgram
    first_characters: tuple[tuple[int, int], ...]
    first_shears: tuple[tuple[int, int, int], ...]
    second_characters: tuple[tuple[int, int], ...]
    second_shears: tuple[tuple[int, int, int], ...]

    def fingerprint(self) -> str:
        return digest_json(
            {
                "base": self.base.fingerprint(),
                "first_characters": self.first_characters,
                "first_shears": self.first_shears,
                "second_characters": self.second_characters,
                "second_shears": self.second_shears,
                "primitive_schema": {
                    "power_sum_phase": "INDEPENDENTLY_SELECTABLE_U_OR_W_MODULE_1_LE_DEGREE_LE_K_AND_NONZERO_F17_COEFFICIENT",
                    "mode_shear": "INDEPENDENTLY_SELECTABLE_U_OR_W_MODULE_BIDIRECTIONAL_ADJACENT_MODE_PAIR_AND_NONZERO_F17_COEFFICIENT",
                },
                "final_boundary": "K_MINUS_1_PARTICLES_IN_MODE0_AND_ONE_PARTICLE_IN_MODE1",
            }
        )


def compile_program(k: int, family: str) -> PhaseModuleProgram:
    base = m127.compile_program(k, family)
    variant = 0 if family == "PRIMARY" else 1
    first_characters = tuple(
        (degree, 1 + ((3 * degree + 5 * variant) % 16))
        for degree in range(1, k + 1)
    )
    second_characters = tuple(
        (degree, 1 + ((7 * degree + 2 + 3 * variant) % 16))
        for degree in range(k, 0, -1)
    )
    first_shears = tuple(
        (mode, mode + 1, 1 + ((5 * mode + 3 + variant) % 16))
        for mode in range(PRIME - 1)
    )
    second_shears = tuple(
        (mode + 1, mode, 1 + ((7 * mode + 4 + 2 * variant) % 16))
        for mode in range(PRIME - 2, -1, -1)
    )
    program = PhaseModuleProgram(
        base=base,
        first_characters=first_characters,
        first_shears=first_shears,
        second_characters=second_characters,
        second_shears=second_shears,
    )
    validate_program(program)
    return program


def validate_character_primitive(k: int, degree: int, coefficient: int) -> None:
    if not 1 <= k < PRIME:
        fail("character primitive degree sector must satisfy 1 <= k < 17")
    if not 1 <= degree <= k or not 1 <= coefficient < PRIME:
        fail("invalid independently selectable power-sum phase primitive")


def validate_mode_shear_primitive(row: int, pivot: int, coefficient: int) -> None:
    if not (
        0 <= row < PRIME
        and 0 <= pivot < PRIME
        and abs(row - pivot) == 1
        and 1 <= coefficient < PRIME
    ):
        fail("invalid independently selectable adjacent-mode shear primitive")


def validate_program(program: PhaseModuleProgram) -> None:
    m127.validate_program(program.base)
    k = program.base.k
    for segment in (program.first_characters, program.second_characters):
        for degree, coefficient in segment:
            validate_character_primitive(k, degree, coefficient)
        if {degree for degree, _ in segment} != set(range(1, k + 1)):
            fail("power-sum phase segment does not cover p1 through pk")
        if not all(1 <= coefficient < PRIME for _, coefficient in segment):
            fail("power-sum character coefficient is outside F17")
    for segment in (program.first_shears, program.second_shears):
        for row, pivot, coefficient in segment:
            validate_mode_shear_primitive(row, pivot, coefficient)
        if len(segment) != PRIME - 1:
            fail("mode-shear segment does not span the adjacent chain")
        if not all(
            abs(row - pivot) == 1
            and 0 <= row < PRIME
            and 0 <= pivot < PRIME
            and 1 <= coefficient < PRIME
            for row, pivot, coefficient in segment
        ):
            fail("invalid adjacent-mode shear descriptor")
    undirected = {
        tuple(sorted((row, pivot)))
        for segment in (program.first_shears, program.second_shears)
        for row, pivot, _ in segment
    }
    if undirected != {(mode, mode + 1) for mode in range(PRIME - 1)}:
        fail("declared mode-shear graph is disconnected")
    orientations = {
        (row, pivot)
        for segment in (program.first_shears, program.second_shears)
        for row, pivot, _ in segment
    }
    if any(
        (mode, mode + 1) not in orientations
        or (mode + 1, mode) not in orientations
        for mode in range(PRIME - 1)
    ):
        fail("declared mode-shear graph is not bidirectional")


def lease(program: PhaseModuleProgram, alg: backend.Algebra) -> str:
    return digest_json(
        {
            "program": program.fingerprint(),
            "algebra": m127.algebra_signature(alg),
            "topology": "M127_EXCHANGE_SYMMETRIC_OCCUPATION_CARRIER",
        }
    )


def public_program_descriptor(program: PhaseModuleProgram) -> dict[str, Any]:
    base = program.base
    return {
        "k": base.k,
        "family": base.family,
        "symmetry_class": base.symmetry_class,
        "module_weight_exponents": base.module_weight_exponents,
        "module_control_exponents": base.module_control_exponents,
        "module_control_degrees": base.module_control_degrees,
        "chirp_exponent": base.chirp_exponent,
        "first_characters": program.first_characters,
        "first_shears": program.first_shears,
        "second_characters": program.second_characters,
        "second_shears": program.second_shears,
        "final_boundary": "K_MINUS_1_PARTICLES_IN_MODE0_AND_ONE_PARTICLE_IN_MODE1",
    }


def validate_orbit_module_start(carrier: m127.ExchangeCarrier, start: int) -> None:
    _, u_start, w_start = m127.carrier_offsets(carrier)
    if start not in (u_start, w_start):
        fail("primitive target is not an owned resident orbit module")


def apply_power_sum_character_primitive(
    carrier: m127.ExchangeCarrier,
    start: int,
    degree: int,
    coefficient: int,
    *,
    inverse: bool = False,
) -> tuple[int, int]:
    validate_orbit_module_start(carrier, start)
    validate_character_primitive(carrier.topology.k, degree, coefficient)
    sign = -1 if inverse else 1
    evaluations = 0
    integer_terms = 0
    for index, histogram in enumerate(carrier.topology.histograms):
        power_sum = sum(
            count * pow(mode, degree, PRIME)
            for mode, count in enumerate(histogram)
        ) % PRIME
        phase = carrier.alg.power(sign * coefficient * power_sum)
        carrier.write_cell(
            start + index,
            carrier.alg.mul(carrier.cells[start + index], phase),
        )
        evaluations += 1
        integer_terms += PRIME
    carrier.observe_resident()
    return evaluations, integer_terms


def apply_character_segment(
    carrier: m127.ExchangeCarrier,
    steps: Iterable[tuple[int, int]],
    *,
    inverse: bool = False,
) -> tuple[int, int]:
    _, u_start, _ = m127.carrier_offsets(carrier)
    evaluations = 0
    integer_terms = 0
    for degree, coefficient in tuple(steps):
        step_evaluations, step_terms = apply_power_sum_character_primitive(
            carrier,
            u_start,
            degree,
            coefficient,
            inverse=inverse,
        )
        evaluations += step_evaluations
        integer_terms += step_terms
    return evaluations, integer_terms


def apply_mode_shear_primitive(
    carrier: m127.ExchangeCarrier,
    start: int,
    row: int,
    pivot: int,
    coefficient_exponent: int,
    *,
    inverse: bool = False,
) -> None:
    validate_orbit_module_start(carrier, start)
    validate_mode_shear_primitive(row, pivot, coefficient_exponent)
    operation = m127.ElementaryOperation(
        "SHEAR", row, pivot, carrier.alg.power(coefficient_exponent)
    )
    if inverse:
        operation = m127.inverse_operation(operation, carrier.alg)
    m127.apply_lifted_shear(carrier, start, operation)


def apply_shear_segment(
    carrier: m127.ExchangeCarrier,
    steps: Iterable[tuple[int, int, int]],
    *,
    inverse: bool = False,
) -> int:
    segment = tuple(steps)
    if inverse:
        segment = tuple(reversed(segment))
    _, u_start, w_start = m127.carrier_offsets(carrier)
    applied = 0
    for row, pivot, coefficient_exponent in segment:
        apply_mode_shear_primitive(
            carrier,
            u_start,
            row,
            pivot,
            coefficient_exponent,
            inverse=inverse,
        )
        apply_mode_shear_primitive(
            carrier,
            w_start,
            row,
            pivot,
            coefficient_exponent,
            inverse=inverse,
        )
        applied += 2
    carrier.observe_resident()
    return applied


def forward(
    carrier: m127.ExchangeCarrier,
    program: PhaseModuleProgram,
) -> dict[str, int]:
    if not isinstance(carrier, m127.ExchangeCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored phase-module carrier")
    validate_program(program)
    if carrier.topology.k != program.base.k:
        fail("program degree does not own the carrier")
    carrier.lease = lease(program, carrier.alg)
    carrier.stage = "FORWARD_ACTIVE"
    m127.load_program(carrier, program.base)
    character_evaluations = 0
    character_integer_terms = 0
    lifted_shear_applications = apply_shear_segment(carrier, program.first_shears)
    evaluations, terms = apply_character_segment(carrier, program.first_characters)
    character_evaluations += evaluations
    character_integer_terms += terms
    m127.apply_orbit_shear(carrier, program.base, 0)
    lifted_shear_applications += apply_shear_segment(carrier, program.second_shears)
    evaluations, terms = apply_character_segment(carrier, program.second_characters)
    character_evaluations += evaluations
    character_integer_terms += terms
    m127.apply_orbit_shear(carrier, program.base, 1)
    carrier.stage = "FORWARD_COMPLETE"
    carrier.observe_resident()
    return {
        "character_evaluations": character_evaluations,
        "character_integer_terms": character_integer_terms,
        "lifted_shear_applications": lifted_shear_applications,
    }


def project_boundary(
    carrier: m127.ExchangeCarrier,
    program: PhaseModuleProgram,
) -> Any:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != lease(program, carrier.alg):
        fail("only the completed owned phase-module boundary may be projected")
    carrier.projection_calls += 1
    _, _, w_start = m127.carrier_offsets(carrier)
    boundary_histogram = (program.base.k - 1, 1, *([0] * (PRIME - 2)))
    return carrier.cells[w_start + carrier.topology.ranks[boundary_histogram]]


def inverse(
    carrier: m127.ExchangeCarrier,
    program: PhaseModuleProgram,
) -> dict[str, int]:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != lease(program, carrier.alg):
        fail("inverse program does not own the phase-module lease")
    carrier.stage = "INVERSE_ACTIVE"
    character_evaluations = 0
    character_integer_terms = 0
    m127.apply_orbit_shear(carrier, program.base, 1, inverse=True)
    evaluations, terms = apply_character_segment(
        carrier, program.second_characters, inverse=True
    )
    character_evaluations += evaluations
    character_integer_terms += terms
    lifted_shear_applications = apply_shear_segment(
        carrier, program.second_shears, inverse=True
    )
    m127.apply_orbit_shear(carrier, program.base, 0, inverse=True)
    evaluations, terms = apply_character_segment(
        carrier, program.first_characters, inverse=True
    )
    character_evaluations += evaluations
    character_integer_terms += terms
    lifted_shear_applications += apply_shear_segment(
        carrier, program.first_shears, inverse=True
    )
    m127.load_program(carrier, program.base, inverse=True)
    carrier.lease = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact phase-module restoration")
    return {
        "character_evaluations": character_evaluations,
        "character_integer_terms": character_integer_terms,
        "lifted_shear_applications": lifted_shear_applications,
    }


def execute_transaction(
    carrier: m127.ExchangeCarrier,
    program: PhaseModuleProgram,
) -> dict[str, Any]:
    m127.reset_transaction_observation(carrier)
    initial_digest = carrier.digest()
    before = carrier.backing_identity()
    generation_before = carrier.generation
    stats_before = m127.stats_snapshot(carrier.alg)
    forward_counts = forward(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse_counts = inverse(carrier, program)
    stats_after = m127.stats_snapshot(carrier.alg)
    dimension = carrier.topology.dimension
    determinant_dimension = m127.GRID_N * m127.GRID_N // 2
    maximum_named_transient = max(
        2 * determinant_dimension * determinant_dimension + 5,
        3 * (program.base.k + 1),
    )
    maximum_transient_value_payload = max(
        carrier.maximum_observed_scratch_field_value_payload_bits,
        carrier.determinant_stats.maximum_observed_field_value_payload_bits,
    )
    descriptor = public_program_descriptor(program)
    descriptor_json_bytes = len(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    serialized_boundary = carrier.alg.serialize(boundary)
    edge_count = len(m127.matchgate.grid_edges(m127.GRID_N))
    return {
        "k": program.base.k,
        "family": program.base.family,
        "algebra": m127.algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "boundary": serialized_boundary,
        "restored_exact_zero": carrier.exact_zero(),
        "same_backing": before == carrier.backing_identity(),
        "generation_before": generation_before,
        "generation_after": carrier.generation,
        "occupation_dimension": dimension,
        "resident_phase_field_cells": len(carrier.cells),
        "resident_orbit_field_cells": 2 * dimension,
        "resident_grid_weight_field_cells": len(carrier.cells) - 2 * dimension,
        "public_occupation_topology_integer_cells": carrier.topology.topology_integer_cells,
        "public_program_integer_cells": 6 * edge_count + 2 + 4 * program.base.k + 6 * (PRIME - 1),
        "public_program_json_bytes": descriptor_json_bytes,
        "public_grid_edge_coordinate_integer_cells": 4 * edge_count,
        "public_grid_vertex_coordinate_integer_cells": 2 * m127.GRID_N * m127.GRID_N,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_resident_field_value_payload_bits": carrier.maximum_resident_field_value_payload_bits,
        "maximum_resident_numerator_signed_bits": carrier.maximum_resident_numerator_signed_bits,
        "maximum_resident_denominator_bits": carrier.maximum_resident_denominator_bits,
        "maximum_observed_scratch_numerator_signed_bits": carrier.maximum_observed_scratch_numerator_signed_bits,
        "maximum_observed_scratch_denominator_bits": carrier.maximum_observed_scratch_denominator_bits,
        "field_operation_counts": m127.stats_delta(stats_before, stats_after),
        "character_evaluations": forward_counts["character_evaluations"] + inverse_counts["character_evaluations"],
        "character_integer_terms": forward_counts["character_integer_terms"] + inverse_counts["character_integer_terms"],
        "lifted_shear_vector_applications": forward_counts["lifted_shear_applications"] + inverse_counts["lifted_shear_applications"],
        "lifted_shear_blocks": carrier.lifted_shear_blocks,
        "lifted_shear_terms": carrier.lifted_shear_terms,
        "module_boundary_evaluations": carrier.module_boundary_evaluations,
        "basis_mismatch_edge_contractions": carrier.basis_mismatch_edge_contractions,
        "determinant_matrix_dimension": determinant_dimension,
        "maximum_named_transaction_transient_field_cells": maximum_named_transient,
        "maximum_named_transaction_transient_integer_cells": 48,
        "maximum_observed_transient_field_value_payload_bits": maximum_transient_value_payload,
        "maximum_named_transaction_transient_payload_bits_upper_bound": maximum_named_transient * maximum_transient_value_payload,
        "determinant_stats": carrier.determinant_stats.as_json(),
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_boundary_json_bytes": len(
            json.dumps(serialized_boundary, separators=(",", ":")).encode("utf-8")
        ),
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "inverse_history_cells": 0,
        "inverse_history_retained": False,
        "snapshot_reload_used": False,
        "initial_digest": initial_digest,
        "restored_digest_with_generation": carrier.digest(),
        "response_released_after_restoration": True,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "accepted_path_labelled_tensor_materialized": False,
        "accepted_path_dense_occupation_operator_materialized": False,
        "accepted_path_character_table_materialized": False,
        "intermediate_orbit_state_serialized": False,
    }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.components = size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root
            self.components -= 1


def irreducibility_certificate(k: int) -> dict[str, Any]:
    topology = m127.OccupationTopology.compile(k)
    signatures = {m127.power_sums(histogram, k) for histogram in topology.histograms}
    signature_count = len(signatures)
    omitted_pk_count = len(
        {
            m127.power_sums(histogram, max(1, k - 1))
            for histogram in topology.histograms
        }
    )
    union = UnionFind(topology.dimension)
    transition_edges = 0
    for index, histogram in enumerate(topology.histograms):
        for mode in range(PRIME - 1):
            if histogram[mode + 1]:
                moved = list(histogram)
                moved[mode] += 1
                moved[mode + 1] -= 1
                union.union(index, topology.ranks[tuple(moved)])
                transition_edges += 1
            if histogram[mode]:
                moved = list(histogram)
                moved[mode] -= 1
                moved[mode + 1] += 1
                union.union(index, topology.ranks[tuple(moved)])
                transition_edges += 1
    predecessor_count = 0
    predecessor_hasher = hashlib.sha256()
    for index, histogram in enumerate(topology.histograms):
        occupied = next(
            (mode for mode in range(1, PRIME) if histogram[mode]), None
        )
        if occupied is None:
            continue
        predecessor = list(histogram)
        predecessor[occupied] -= 1
        predecessor[occupied - 1] += 1
        predecessor_rank = topology.ranks[tuple(predecessor)]
        predecessor_count += 1
        predecessor_hasher.update(f"{index}:{predecessor_rank};".encode("ascii"))
    newton_denominators_invertible = all(math.gcd(degree, PRIME) == 1 for degree in range(1, k + 1))
    certificate = {
        "k": k,
        "occupation_dimension": topology.dimension,
        "stars_and_bars_dimension": math.comb(k + 16, 16),
        "p1_through_pk_signature_count": signature_count,
        "power_sum_signature_injective": signature_count == topology.dimension,
        "p1_through_p_k_minus_1_signature_count": omitted_pk_count,
        "omitting_pk_overmerges_when_k_at_least2": (
            k == 1 or omitted_pk_count < topology.dimension
        ),
        "newton_denominators_invertible_mod17": newton_denominators_invertible,
        "adjacent_mode_graph_vertices": PRIME,
        "adjacent_mode_graph_edges": PRIME - 1,
        "bidirectional_lifted_transition_edges_streamed": transition_edges,
        "lifted_support_components": union.components,
        "lifted_support_connected": union.components == 1,
        "explicit_predecessor_edges_to_zero_mode": predecessor_count,
        "expected_predecessor_edges": topology.dimension - 1,
        "predecessor_certificate_fingerprint": predecessor_hasher.hexdigest(),
        "verification_signature_integer_cells": k * topology.dimension,
        "verification_truncated_signature_integer_cells": max(1, k - 1) * topology.dimension,
        "verification_union_find_integer_cells": topology.dimension,
        "verification_named_logical_integer_cells_conservative_sum": (
            (18 + k + max(1, k - 1) + 1) * topology.dimension
        ),
        "verification_transition_edge_list_materialized": False,
        "verification_predecessor_list_materialized": False,
        "full_diagonal_algebra_from_character_orthogonality": True,
        "coordinate_projector_formula": "PRODUCT_R_OF_ONE_OVER17_SUM_C0_TO16_ZETA_MINUS_C_P_R_ALPHA_TIMES_P_R_TO_POWER_C",
        "only_trivial_coordinate_subset_invariant_under_connected_shears": union.components == 1,
        "nonzero_proper_invariant_kernel_exists": False,
        "minimum_exact_linear_quotient_dimension": topology.dimension,
    }
    if not all(
        (
            certificate["power_sum_signature_injective"],
            certificate["newton_denominators_invertible_mod17"],
            certificate["lifted_support_connected"],
            certificate["explicit_predecessor_edges_to_zero_mode"]
            == certificate["expected_predecessor_edges"],
        )
    ):
        fail(f"irreducibility certificate failed at k={k}")
    return certificate


def character_orthogonality(alg: backend.Algebra) -> dict[str, Any]:
    sums = []
    for difference in range(PRIME):
        value = alg.zero
        for coefficient in range(PRIME):
            value = alg.add(value, alg.power(coefficient * difference))
        expected = m127.integer(alg, PRIME) if difference == 0 else alg.zero
        sums.append(value == expected)
    return {
        "algebra": m127.algebra_signature(alg),
        "all_17_character_sums_exact": all(sums),
        "zero_difference_sum_nonzero": sums[0],
        "nonzero_difference_sums_zero": all(sums[1:]),
    }


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "occupation_dimension",
        "resident_phase_field_cells",
        "resident_orbit_field_cells",
        "resident_grid_weight_field_cells",
        "public_occupation_topology_integer_cells",
        "public_program_integer_cells",
        "public_program_json_bytes",
        "public_grid_edge_coordinate_integer_cells",
        "public_grid_vertex_coordinate_integer_cells",
        "character_evaluations",
        "character_integer_terms",
        "lifted_shear_vector_applications",
        "lifted_shear_blocks",
        "lifted_shear_terms",
        "module_boundary_evaluations",
        "basis_mismatch_edge_contractions",
        "determinant_matrix_dimension",
        "maximum_named_transaction_transient_field_cells",
        "maximum_named_transaction_transient_integer_cells",
        "maximum_observed_transient_field_value_payload_bits",
        "maximum_named_transaction_transient_payload_bits_upper_bound",
        "maximum_resident_payload_bits",
        "maximum_resident_field_value_payload_bits",
        "maximum_resident_numerator_signed_bits",
        "maximum_resident_denominator_bits",
        "maximum_observed_scratch_numerator_signed_bits",
        "maximum_observed_scratch_denominator_bits",
        "field_operation_counts",
        "determinant_stats",
        "final_boundary_field_cells",
        "final_boundary_payload_bits",
        "final_boundary_json_bytes",
    )
    return {key: transaction[key] for key in keys}


def run_case(k: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    topology = m127.OccupationTopology.compile(k)
    return execute_transaction(
        m127.ExchangeCarrier.create(topology, alg), compile_program(k, family)
    )


def transaction_boundary(program: PhaseModuleProgram) -> Any:
    alg = backend.Algebra("F103", modulus=103, root=72)
    topology = m127.OccupationTopology.compile(program.base.k)
    return execute_transaction(m127.ExchangeCarrier.create(topology, alg), program)["boundary"]


def controls() -> dict[str, Any]:
    program = compile_program(2, "PRIMARY")
    base_boundary = transaction_boundary(program)
    missing_character = replace(
        program, first_characters=program.first_characters[:-1]
    )
    missing_character_rejected = False
    try:
        validate_program(missing_character)
    except RuntimeError:
        missing_character_rejected = True
    mutated_characters = list(program.second_characters)
    degree, coefficient = mutated_characters[-1]
    mutated_characters[-1] = (degree, 1 + (coefficient % 16))
    character_mutation = replace(
        program, second_characters=tuple(mutated_characters)
    )
    character_mutation_boundary = transaction_boundary(character_mutation)
    missing_orientation = replace(
        program, second_shears=program.second_shears[:-1]
    )
    missing_orientation_rejected = False
    try:
        validate_program(missing_orientation)
    except RuntimeError:
        missing_orientation_rejected = True
    mutated_shears = list(program.second_shears)
    row, pivot, coefficient = mutated_shears[-1]
    mutated_shears[-1] = (row, pivot, 1 + (coefficient % 16))
    shear_mutation = replace(program, second_shears=tuple(mutated_shears))
    shear_mutation_boundary = transaction_boundary(shear_mutation)

    topology = m127.OccupationTopology.compile(2)
    alg = backend.Algebra("F103", modulus=103, root=72)
    carrier = m127.ExchangeCarrier.create(topology, alg)
    forward(carrier, program)
    missing_inverse_detected = not carrier.exact_zero()

    wrong_program_rejected = False
    try:
        inverse(carrier, compile_program(2, "REUSE"))
    except RuntimeError:
        wrong_program_rejected = True

    premature_alg = backend.Algebra("F103", modulus=103, root=72)
    premature = m127.ExchangeCarrier.create(topology, premature_alg)
    premature.lease = lease(program, premature_alg)
    premature.stage = "FORWARD_ACTIVE"
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    reordered_alg = backend.Algebra("F103", modulus=103, root=72)
    reordered = m127.ExchangeCarrier.create(topology, reordered_alg)
    forward(reordered, program)
    m127.apply_orbit_shear(reordered, program.base, 0, inverse=True)
    m127.apply_orbit_shear(reordered, program.base, 1, inverse=True)
    apply_shear_segment(reordered, program.second_shears, inverse=True)
    apply_character_segment(reordered, program.second_characters, inverse=True)
    apply_shear_segment(reordered, program.first_shears, inverse=True)
    apply_character_segment(reordered, program.first_characters, inverse=True)
    m127.load_program(reordered, program.base, inverse=True)
    reordered.lease = None
    reordered.stage = "RESTORED"
    reordered_inverse_detected = not reordered.exact_zero()

    null_carrier_rejected = False
    try:
        forward(None, program)  # type: ignore[arg-type]
    except (RuntimeError, AttributeError):
        null_carrier_rejected = True

    total_sum_signatures = {
        m127.power_sums(histogram, 1) for histogram in topology.histograms
    }
    disconnected_component_invariants = {
        sum(histogram[:8]) for histogram in topology.histograms
    }
    nonprimitive_root_rejected = False
    try:
        nonprimitive_root = backend.Algebra("F103_BAD_ROOT", modulus=103, root=1)
        nonprimitive_root_rejected = not character_orthogonality(nonprimitive_root)[
            "all_17_character_sums_exact"
        ]
    except RuntimeError:
        nonprimitive_root_rejected = True
    k17_applicability_rejected = False
    try:
        m127.enumerate_histograms(17)
    except RuntimeError:
        k17_applicability_rejected = True
    return {
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_program_ownership_rejected": wrong_program_rejected,
        "premature_projection_rejected": premature_projection_rejected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "null_carrier_rejected": null_carrier_rejected,
        "missing_p2_character_rejected": missing_character_rejected,
        "power_sum_character_mutation_changes_boundary": base_boundary != character_mutation_boundary,
        "missing_reverse_mode_orientation_rejected": missing_orientation_rejected,
        "mode_shear_mutation_changes_boundary": base_boundary != shear_mutation_boundary,
        "p1_only_signature_count_k2": len(total_sum_signatures),
        "p1_only_overmerges_k2": len(total_sum_signatures) < topology.dimension,
        "disconnecting_mode7_from_mode8_component_count_k2": len(disconnected_component_invariants),
        "disconnected_mode_graph_preserves_particle_count_components": len(disconnected_component_invariants) == 3,
        "nonprimitive_root_rejected": nonprimitive_root_rejected,
        "k17_newton_applicability_rejected": k17_applicability_rejected,
        "accepted_path_character_table_materialized": False,
        "accepted_path_dense_operator_materialized": False,
        "accepted_path_labelled_tensor_materialized": False,
        "intermediate_orbit_state_serialized": False,
        "catvm_boundary_claimed": False,
    }


def run() -> dict[str, Any]:
    exact = [
        run_case(k, family, backend.Algebra("Q_ZETA17"))
        for k, family in EXACT_CASES
    ]
    structural = []
    for modulus, root in FINITE_FIELDS:
        for k in STRUCTURAL_K:
            item = run_case(
                k,
                "PRIMARY",
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            item["field"] = f"F{modulus}"
            structural.append(item)

    topology = m127.OccupationTopology.compile(2)
    reuse_alg = backend.Algebra("Q_ZETA17")
    reuse_carrier = m127.ExchangeCarrier.create(topology, reuse_alg)
    first = execute_transaction(reuse_carrier, compile_program(2, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(2, "REUSE"))
    fresh = run_case(2, "REUSE", backend.Algebra("Q_ZETA17"))
    if reused["boundary"] != fresh["boundary"]:
        fail("restored phase-module reuse disagrees with fresh execution")
    if resource_signature(reused) != resource_signature(fresh):
        fail("restored phase-module reuse changed its resource signature")

    certificates = [irreducibility_certificate(k) for k in STRUCTURAL_K]
    orthogonality = [
        character_orthogonality(backend.Algebra("Q_ZETA17")),
        *(
            character_orthogonality(
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
            )
            for modulus, root in FINITE_FIELDS
        ),
    ]
    if not all(item["all_17_character_sums_exact"] for item in orthogonality):
        fail("phase-character orthogonality failed")

    return {
        "schema": "CAT_CAS_F17_EXCHANGE_SYMMETRIC_PHASE_MODULE_IRREDUCIBILITY_V1",
        "claim": "BOUNDED_EXACT_F17_POWER_SUM_CHARACTER_AND_BIDIRECTIONAL_MODE_SHEAR_PHASE_MODULE_HAS_NO_NONTRIVIAL_LINEAR_QUOTIENT_BELOW_H_K_ON_EACH_DECLARED_EXCHANGE_SYMMETRIC_K1_TO_K4_ORBIT_MODULE_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "base_carrier": "M127_EXCHANGE_SYMMETRIC_OCCUPATION_ORBIT_CARRIER",
            "grid_module_topology": "M126_EVEN_OPEN_SQUARE_GRID_N4_ONLY",
            "exchange_symmetric_degrees": STRUCTURAL_K,
            "exact_q_zeta17_cases": EXACT_CASES,
            "dual_field_primary_degrees": STRUCTURAL_K,
            "primitive_algebra": [
                "POWER_SUM_PHASE_CHARACTER_P1_THROUGH_PK",
                "BIDIRECTIONAL_ADJACENT_MODE_SHEAR",
                "M127_STREAMED_GRID_ORBIT_SHEAR",
            ],
        },
        "irreducibility_certificates": certificates,
        "phase_character_orthogonality": orthogonality,
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "reuse": {
            "k": 2,
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "fresh_restored_resource_signature_agreement": resource_signature(reused) == resource_signature(fresh),
            "same_actual_backing_across_unrelated_programs": (
                first["same_backing"]
                and reused["same_backing"]
                and reuse_carrier.backing_identity() == backing
            ),
            "generation_after_two_transactions": reuse_carrier.generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "proof_law": {
            "field_hypotheses": "CHARACTERISTIC_ZERO_OR_PRIME_GREATER_THAN_K_NOT17_WITH_A_PRIMITIVE_17TH_ROOT",
            "power_sum_separation": "NEWTON_IDENTITIES_RECOVER_E1_THROUGH_EK_BECAUSE_1_THROUGH_K_ARE_INVERTIBLE_MOD17",
            "character_span": "F17_CHARACTER_ORTHOGONALITY_ON_THE_INJECTIVE_POWER_SUM_SIGNATURES_GENERATES_THE_FULL_DIAGONAL_COORDINATE_ALGEBRA",
            "kernel_form": "ANY_INVARIANT_KERNEL_OF_THE_FULL_DIAGONAL_ALGEBRA_IS_A_COORDINATE_SUBSET",
            "shear_connectivity": "BIDIRECTIONAL_ADJACENT_MODE_SHEARS_CONNECT_THE_DEGREE_K_OCCUPATION_GRAPH",
            "irreducibility": "ONLY_EMPTY_AND_FULL_COORDINATE_SUBSETS_ARE_INVARIANT",
            "boundary_nontriviality": "FINAL_K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION_PROJECTION_REJECTS_THE_FULL_KERNEL",
            "conclusion": "ANY_EXACT_LINEAR_QUOTIENT_PRESERVING_ALL_DECLARED_PRIMITIVES_AND_FINAL_BOUNDARY_HAS_DIMENSION_AT_LEAST_H_K",
        },
        "resource_law": {
            "minimum_exact_linear_quotient_field_cells": "H_K_EQUALS_BINOMIAL_K_PLUS_16_CHOOSE_16",
            "accepted_resident_phase_field_cells": "48_PLUS_2_H_K",
            "public_occupation_topology_integer_cells": "18_H_K",
            "character_table_cells": 0,
            "dense_operator_cells": 0,
            "labelled_tensor_cells": 0,
            "inverse_history_cells": 0,
            "final_boundary_field_cells": 1,
            "named_exact_field_operation_resident_boundary_and_transient_logical_counts": "REPORTED_PER_TRANSACTION",
            "certificate_verification_named_logical_integer_cells": "REPORTED_PER_K_INCLUDING_PUBLIC_TOPOLOGY_SIGNATURE_SETS_AND_UNION_FIND",
            "certificate_transition_and_predecessor_edge_lists_materialized": False,
            "oracle_cached_boundary_vectors_are_separate_verification_resources": True,
            "python_container_native_bigint_and_whole_process_memory_excluded": True,
            "full_exact_bit_complexity_established": False,
        },
        "matched_baseline": {
            "strongest_implemented": "IDENTICAL_H_K_COORDINATE_EXACT_LINEAR_PHASE_MODULE_RECURRENCE",
            "phase_advantage_over_matched_classical": False,
            "labelled_17_TO_THE_K_tensor_is_not_the_matched_baseline": True,
        },
        "restoration": {
            "resident_grid_and_orbit_phase_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_grid_and_lifted_shear_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "declared_degrees_only": STRUCTURAL_K,
            "uniform_linear_phase_module_quotient_only": True,
            "nonlinear_or_program_restricted_quotient_rejected": False,
            "arbitrary_relation_algebra_rejected": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "THE_NATURAL_POWER_SUM_CHARACTER_AND_CONNECTED_MODE_SHEAR_EXTENSION_MAKES_THE_EXCHANGE_SYMMETRIC_PHASE_MODULE_IRREDUCIBLE_SO_ANY_EXACT_UNIFORM_LINEAR_QUOTIENT_RETAINS_ALL_H_K_COORDINATES_AND_THE_IDENTICAL_CLASSICAL_RECURRENCE",
        "next_experiment": "TEST_A_NONLINEAR_OR_RESTRICTED_INTEGRABLE_PHASE_CONVOLUTION_CHART_WITH_EXACT_COMPOSITION_WITHOUT_H_K_COORDINATE_MATERIALIZATION_OR_HIDDEN_PRECISION_HISTORY_GROWTH",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = run()
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
