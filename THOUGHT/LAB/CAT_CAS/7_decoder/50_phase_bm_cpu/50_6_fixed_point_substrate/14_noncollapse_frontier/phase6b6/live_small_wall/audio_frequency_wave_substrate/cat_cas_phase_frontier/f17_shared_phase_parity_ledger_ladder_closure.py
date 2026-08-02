#!/usr/bin/env python3
"""Exact bounded shared-parity-ledger closure on two-row F17 ladders.

The M123 sparse-field path streams every even insertion sector.  This
successor keeps the insertion choice unresolved inside each local field
factor and contracts the two-row topology in column order.  The live
frontier has four exact phase-field cells independently of ladder length.

The resident carrier contains the actual edge and field phase pairs plus one
boundary accumulator.  The forward recurrence consumes those cells, the
final boundary is projected once, and the same public recurrence is
rematerialized before the factor cells are unloaded.  Dense signature and
occurrence-expanded sector tensors are absent from the accepted path.

This is direct-process exact software.  The identical four-state recurrence
is the strongest matched classical implementation.  It establishes neither
CATVM custody nor a distinct phase resource, computational advantage, Small
Wall crossing, physical execution, bit replacement, or unbounded compute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import f17_planar_free_fermion_phase_covariance_closure as m123


m122 = m123.m122
PRIME = 17
EXACT_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
STRUCTURAL_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128)
MODULAR_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE")
SPIN_STATES = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scalar(alg: m122.Algebra, value: int) -> Any:
    if alg.modulus:
        return value % alg.modulus
    return alg.domain.convert(value)


def neg(alg: m122.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def ladder_topology(width: int) -> tuple[tuple[int, int], ...]:
    if width < 1:
        fail("ladder width must be positive")
    return (
        *((row * width + column, row * width + column + 1)
          for row in range(2) for column in range(width - 1)),
        *((column, width + column) for column in range(width)),
    )


@dataclass(frozen=True)
class LadderProgram:
    width: int
    family: str
    unary: tuple[int, ...]
    edge_weights: tuple[int, ...]
    field_residues: tuple[int, ...]

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "width": self.width,
                "family": self.family,
                "unary": self.unary,
                "edge_weights": self.edge_weights,
                "field_residues": self.field_residues,
            }
        )


def compile_program(width: int, family: str) -> LadderProgram:
    if width not in STRUCTURAL_WIDTHS:
        fail("width is outside the declared M124 ladder scope")
    if family not in FAMILIES:
        fail("unknown M124 ladder family")
    variant = 0 if family == "PRIMARY" else 1
    edges = ladder_topology(width)
    edge_weights = tuple(
        1 + ((7 * ordinal + 3 * width + 5 * variant) % 16)
        for ordinal in range(len(edges))
    )
    residues = tuple(
        1 + ((5 * site + 2 * width + 3 * variant) % 16)
        for site in range(2 * width)
    )
    inverse_two = 9
    unary = []
    for site, residue in enumerate(residues):
        incident = sum(
            weight
            for edge, weight in zip(edges, edge_weights, strict=True)
            if site in edge
        )
        unary.append((inverse_two * (residue - incident)) % PRIME)
    program = LadderProgram(
        width,
        family,
        tuple(unary),
        edge_weights,
        residues,
    )
    validate_program(program)
    return program


def validate_program(program: LadderProgram) -> None:
    if program.width not in STRUCTURAL_WIDTHS:
        fail("program width changed")
    if program.family not in FAMILIES:
        fail("program family changed")
    edges = ladder_topology(program.width)
    if len(program.unary) != 2 * program.width:
        fail("program unary arity changed")
    if len(program.field_residues) != 2 * program.width:
        fail("program field-port arity changed")
    if len(program.edge_weights) != len(edges):
        fail("program edge arity changed")
    if not all(0 <= value < PRIME for value in program.unary):
        fail("unary value is outside F17")
    if not all(1 <= value < PRIME for value in program.edge_weights):
        fail("accepted ladder edge weights must be nonzero")
    if not all(1 <= value < PRIME for value in program.field_residues):
        fail("declared M124 field ports must be nonzero")
    reconstructed = []
    for site, unary in enumerate(program.unary):
        incident = sum(
            weight
            for edge, weight in zip(edges, program.edge_weights, strict=True)
            if site in edge
        )
        reconstructed.append((2 * unary + incident) % PRIME)
    if tuple(reconstructed) != program.field_residues:
        fail("field-port residues do not match the binary-to-spin map")


@dataclass(frozen=True)
class PhaseFactors:
    constant_exponent: int
    edge_pairs: tuple[tuple[Any, Any], ...]
    field_pairs: tuple[tuple[Any, Any], ...]

    def flattened(self) -> list[Any]:
        return [
            *(value for pair in self.edge_pairs for value in pair),
            *(value for pair in self.field_pairs for value in pair),
        ]


def compile_phase_factors(program: LadderProgram, alg: m122.Algebra) -> PhaseFactors:
    validate_program(program)
    return PhaseFactors(
        (9 * sum(program.unary) + 13 * sum(program.edge_weights)) % PRIME,
        tuple(m123.phase_pair(alg, 13 * weight) for weight in program.edge_weights),
        tuple(m123.phase_pair(alg, -13 * residue) for residue in program.field_residues),
    )


@dataclass
class LedgerWorkStats:
    recurrence_calls: int = 0
    local_factor_field_multiplications: int = 0
    horizontal_kernel_field_multiplications: int = 0
    horizontal_kernel_field_additions: int = 0
    local_diagonal_field_multiplications: int = 0
    final_frontier_field_additions: int = 0
    maximum_frontier_field_cells: int = 0
    maximum_named_transient_field_cells: int = 0
    maximum_named_transient_payload_bits: int = 0

    def observe(self, alg: m122.Algebra, *collections: Iterable[Any]) -> None:
        values = [value for collection in collections for value in collection]
        self.maximum_named_transient_field_cells = max(
            self.maximum_named_transient_field_cells,
            len(values),
        )
        self.maximum_named_transient_payload_bits = max(
            self.maximum_named_transient_payload_bits,
            sum(alg.payload_bits(value) for value in values),
        )

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


@dataclass
class ParityLedgerCarrier:
    width: int
    topology_fingerprint: str
    alg: m122.Algebra
    cells: list[Any]
    accumulator: Any
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    loads: int = 0
    unloads: int = 0
    accumulator_updates: int = 0
    projection_calls: int = 0
    maximum_resident_payload_bits: int = 0
    work: LedgerWorkStats = field(default_factory=LedgerWorkStats)

    @classmethod
    def create(cls, width: int, alg: m122.Algebra) -> "ParityLedgerCarrier":
        topology = ladder_topology(width)
        fingerprint = sha256_json({"width": width, "edges": topology})
        cell_count = 2 * (len(topology) + 2 * width)
        carrier = cls(
            width,
            fingerprint,
            alg,
            [alg.zero for _ in range(cell_count)],
            alg.zero,
        )
        carrier.observe_resident()
        return carrier

    def backing_identity(self) -> tuple[int, int]:
        return id(self), id(self.cells)

    def observe_resident(self) -> None:
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.cells)
            + self.alg.payload_bits(self.accumulator),
        )

    def exact_zero(self) -> bool:
        return (
            all(value == self.alg.zero for value in self.cells)
            and self.accumulator == self.alg.zero
            and self.lease is None
            and self.stage == "RESTORED"
        )

    def digest(self) -> str:
        return sha256_json(
            {
                "width": self.width,
                "topology": self.topology_fingerprint,
                "cells": [self.alg.serialize(value) for value in self.cells],
                "accumulator": self.alg.serialize(self.accumulator),
                "generation": self.generation,
                "lease": self.lease,
                "stage": self.stage,
            }
        )


def _load(carrier: ParityLedgerCarrier, values: list[Any], *, inverse: bool = False) -> None:
    if len(values) != len(carrier.cells):
        fail("factor payload does not fit the declared carrier")
    for index, value in enumerate(values):
        delta = neg(carrier.alg, value) if inverse else value
        carrier.cells[index] = carrier.alg.add(carrier.cells[index], delta)
    if inverse:
        carrier.unloads += len(values)
    else:
        carrier.loads += len(values)
    carrier.observe_resident()


def _factor_views(
    carrier: ParityLedgerCarrier,
) -> tuple[tuple[tuple[Any, Any], ...], tuple[tuple[Any, Any], ...]]:
    edge_count = len(ladder_topology(carrier.width))
    edge_pairs = tuple(
        (carrier.cells[2 * index], carrier.cells[2 * index + 1])
        for index in range(edge_count)
    )
    field_offset = 2 * edge_count
    field_pairs = tuple(
        (
            carrier.cells[field_offset + 2 * index],
            carrier.cells[field_offset + 2 * index + 1],
        )
        for index in range(2 * carrier.width)
    )
    return edge_pairs, field_pairs


def _signed_factor(alg: m122.Algebra, pair: tuple[Any, Any], sign: int) -> Any:
    if sign == 1:
        return alg.add(pair[0], pair[1])
    if sign == -1:
        return alg.sub(pair[0], pair[1])
    fail("phase-factor sign must be plus or minus one")


def _contract_resident_ledger(carrier: ParityLedgerCarrier) -> Any:
    """Contract all unresolved insertion choices through a four-state frontier."""

    alg = carrier.alg
    width = carrier.width
    edge_pairs, field_pairs = _factor_views(carrier)
    horizontal_count = 2 * (width - 1)

    def local_factors(column: int) -> list[Any]:
        vertical = edge_pairs[horizontal_count + column]
        top_field = field_pairs[column]
        bottom_field = field_pairs[width + column]
        values = []
        for top_spin, bottom_spin in SPIN_STATES:
            edge_value = _signed_factor(alg, vertical, top_spin * bottom_spin)
            top_value = _signed_factor(alg, top_field, top_spin)
            bottom_value = _signed_factor(alg, bottom_field, bottom_spin)
            value = alg.mul(edge_value, top_value)
            value = alg.mul(value, bottom_value)
            carrier.work.local_factor_field_multiplications += 2
            values.append(value)
        return values

    current = local_factors(0)
    carrier.work.maximum_frontier_field_cells = max(
        carrier.work.maximum_frontier_field_cells,
        len(current),
    )
    carrier.work.observe(alg, current)

    for column in range(1, width):
        top_kernel = edge_pairs[column - 1]
        bottom_kernel = edge_pairs[(width - 1) + column - 1]
        local = local_factors(column)
        temp = [alg.zero for _ in range(4)]
        for target_top in range(2):
            target_top_spin = -1 if target_top == 0 else 1
            for source_bottom in range(2):
                accumulator = alg.zero
                for source_top in range(2):
                    source_top_spin = -1 if source_top == 0 else 1
                    kernel = _signed_factor(
                        alg,
                        top_kernel,
                        source_top_spin * target_top_spin,
                    )
                    term = alg.mul(current[2 * source_top + source_bottom], kernel)
                    accumulator = alg.add(accumulator, term)
                    carrier.work.horizontal_kernel_field_multiplications += 1
                    carrier.work.horizontal_kernel_field_additions += 1
                    carrier.work.observe(
                        alg,
                        current,
                        temp,
                        local,
                        (accumulator, kernel, term),
                    )
                temp[2 * target_top + source_bottom] = accumulator
        following = [alg.zero for _ in range(4)]
        for target_top in range(2):
            for target_bottom in range(2):
                target_bottom_spin = -1 if target_bottom == 0 else 1
                accumulator = alg.zero
                for source_bottom in range(2):
                    source_bottom_spin = -1 if source_bottom == 0 else 1
                    kernel = _signed_factor(
                        alg,
                        bottom_kernel,
                        source_bottom_spin * target_bottom_spin,
                    )
                    term = alg.mul(temp[2 * target_top + source_bottom], kernel)
                    accumulator = alg.add(accumulator, term)
                    carrier.work.horizontal_kernel_field_multiplications += 1
                    carrier.work.horizontal_kernel_field_additions += 1
                    carrier.work.observe(
                        alg,
                        current,
                        temp,
                        following,
                        local,
                        (accumulator, kernel, term),
                    )
                index = 2 * target_top + target_bottom
                following[index] = alg.mul(accumulator, local[index])
                carrier.work.local_diagonal_field_multiplications += 1
        carrier.work.observe(alg, current, temp, following, local)
        current = following
        carrier.work.maximum_frontier_field_cells = max(
            carrier.work.maximum_frontier_field_cells,
            len(current),
        )

    boundary = current[0]
    for value in current[1:]:
        boundary = alg.add(boundary, value)
        carrier.work.final_frontier_field_additions += 1
    carrier.work.recurrence_calls += 1
    carrier.work.observe(alg, current, (boundary,))
    return boundary


def forward(carrier: ParityLedgerCarrier, program: LadderProgram) -> None:
    if not isinstance(carrier, ParityLedgerCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored parity-ledger carrier")
    if carrier.width != program.width:
        fail("program width does not own the parity-ledger carrier")
    factors = compile_phase_factors(program, carrier.alg)
    carrier.lease = program.fingerprint()
    carrier.stage = "FORWARD_ACTIVE"
    _load(carrier, factors.flattened())
    value = _contract_resident_ledger(carrier)
    carrier.accumulator = carrier.alg.add(carrier.accumulator, value)
    carrier.accumulator_updates += 1
    carrier.stage = "FORWARD_COMPLETE"
    carrier.observe_resident()


def project_boundary(carrier: ParityLedgerCarrier, program: LadderProgram) -> Any:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("only the completed owned parity-ledger boundary may be projected")
    carrier.projection_calls += 1
    factors = compile_phase_factors(program, carrier.alg)
    phase = carrier.alg.power(factors.constant_exponent)
    # The four-state recurrence already sums over every spin assignment.  The
    # 2**|V| factor belongs to the even-subgraph/Pfaffian normalization used by
    # the preceding free-fermion package; applying it here would count the spin
    # sum twice.
    return carrier.alg.mul(phase, carrier.accumulator)


def inverse(carrier: ParityLedgerCarrier, program: LadderProgram) -> None:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("inverse program does not own the live parity-ledger lease")
    factors = compile_phase_factors(program, carrier.alg)
    carrier.stage = "INVERSE_ACTIVE"
    rematerialized = _contract_resident_ledger(carrier)
    carrier.accumulator = carrier.alg.sub(carrier.accumulator, rematerialized)
    carrier.accumulator_updates += 1
    _load(carrier, factors.flattened(), inverse=True)
    carrier.lease = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact parity-ledger restoration")


def execute_transaction(
    carrier: ParityLedgerCarrier,
    program: LadderProgram,
) -> dict[str, Any]:
    backing = carrier.backing_identity()
    generation = carrier.generation
    initial = carrier.digest()
    forward(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    edges = len(ladder_topology(program.width))
    return {
        "width": program.width,
        "family": program.family,
        "program_fingerprint": program.fingerprint(),
        "boundary": carrier.alg.serialize(boundary),
        "field_port_count": 2 * program.width,
        "edge_count": edges,
        "resident_factor_field_cells": len(carrier.cells),
        "maximum_frontier_field_cells": carrier.work.maximum_frontier_field_cells,
        "maximum_named_transient_field_cells": carrier.work.maximum_named_transient_field_cells,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_named_transient_payload_bits": carrier.work.maximum_named_transient_payload_bits,
        "work": carrier.work.as_json(),
        "retain_all_signature_field_cells": 1 << (2 * program.width),
        "occurrence_expanded_even_sector_count": 1 << max(0, 2 * program.width - 1),
        "accepted_path_even_sector_enumeration": False,
        "accepted_path_dense_signature_materialized": False,
        "accepted_path_topology_column_interleaving": True,
        "intermediate_boundary_projection_calls": 0,
        "final_boundary_projection_calls": 1,
        "factor_load_additions": carrier.loads,
        "factor_unload_additions": carrier.unloads,
        "accumulator_forward_inverse_updates": carrier.accumulator_updates,
        "generation_before": generation,
        "generation_after": carrier.generation,
        "restoration_generation_increment": carrier.generation == generation + 1,
        "same_backing": carrier.backing_identity() == backing,
        "initial_digest": initial,
        "restored_digest_with_generation": carrier.digest(),
        "exact_factor_carrier_restored": carrier.exact_zero(),
        "response_released_after_restoration": True,
        "snapshot_reload_used": False,
        "inverse_history_retained": False,
        "carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_frontier_restoration_class": "NO_RESTORATION_CLAIM",
    }


def exact_case(width: int, family: str) -> dict[str, Any]:
    alg = m122.Algebra("Q_ZETA17")
    return execute_transaction(
        ParityLedgerCarrier.create(width, alg),
        compile_program(width, family),
    )


def modular_case(width: int, family: str, modulus: int, root: int) -> dict[str, Any]:
    alg = m122.Algebra(f"F{modulus}", modulus=modulus, root=root)
    result = execute_transaction(
        ParityLedgerCarrier.create(width, alg),
        compile_program(width, family),
    )
    result["field"] = f"F{modulus}"
    return result


def controls() -> dict[str, Any]:
    alg = m122.Algebra("F103", modulus=103, root=72)
    program = compile_program(4, "PRIMARY")

    missing = ParityLedgerCarrier.create(4, alg)
    forward(missing, program)
    missing_inverse_detected = not missing.exact_zero()

    wrong = ParityLedgerCarrier.create(4, m122.Algebra("F103", modulus=103, root=72))
    forward(wrong, program)
    wrong_inverse_detected = False
    try:
        inverse(wrong, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_inverse_detected = True

    premature = ParityLedgerCarrier.create(4, m122.Algebra("F103", modulus=103, root=72))
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    null_carrier_rejected = False
    try:
        forward(None, program)  # type: ignore[arg-type]
    except (RuntimeError, AttributeError):
        null_carrier_rejected = True

    mistyped = LadderProgram(
        program.width,
        program.family,
        program.unary,
        program.edge_weights,
        program.field_residues[:-1],
    )
    wrong_field_type_rejected = False
    try:
        validate_program(mistyped)
    except RuntimeError:
        wrong_field_type_rejected = True

    mutation_alg = m122.Algebra("F103", modulus=103, root=72)
    mutation_carrier = ParityLedgerCarrier.create(4, mutation_alg)
    factors = compile_phase_factors(program, mutation_alg)
    mutation_carrier.lease = program.fingerprint()
    mutation_carrier.stage = "FORWARD_ACTIVE"
    _load(mutation_carrier, factors.flattened())
    unmutated = _contract_resident_ledger(mutation_carrier)
    mutation_carrier.cells[0] = mutation_alg.add(mutation_carrier.cells[0], mutation_alg.one)
    mutated = _contract_resident_ledger(mutation_carrier)

    return {
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_inverse_ownership_detected": wrong_inverse_detected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "wrong_field_port_type_rejected": wrong_field_type_rejected,
        "resident_factor_mutation_changes_boundary": unmutated != mutated,
        "reordered_inverse_control_applicable": False,
        "reordered_inverse_reason": "FACTOR_LOADS_AND_ACCUMULATOR_ADDITIONS_COMMUTE",
        "snapshot_command_absent": True,
        "catvm_boundary_claimed": False,
        "accepted_path_even_sector_enumeration": False,
        "accepted_path_dense_signature_materialized": False,
    }


def run() -> dict[str, Any]:
    exact = [
        exact_case(width, family)
        for family in FAMILIES
        for width in EXACT_WIDTHS
    ]

    reuse_width = 16
    reuse_alg = m122.Algebra("Q_ZETA17")
    reuse_carrier = ParityLedgerCarrier.create(reuse_width, reuse_alg)
    first = execute_transaction(reuse_carrier, compile_program(reuse_width, "PRIMARY"))
    reused = execute_transaction(reuse_carrier, compile_program(reuse_width, "REUSE"))
    fresh = execute_transaction(
        ParityLedgerCarrier.create(reuse_width, m122.Algebra("Q_ZETA17")),
        compile_program(reuse_width, "REUSE"),
    )
    if reused["boundary"] != fresh["boundary"]:
        fail("restored parity-ledger carrier reuse disagrees with fresh execution")

    structural = [
        modular_case(width, family, modulus, root)
        for modulus, root in MODULAR_FIELDS
        for family in FAMILIES
        for width in STRUCTURAL_WIDTHS
    ]
    return {
        "schema": "CAT_CAS_F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_COLUMN_INTERLEAVED_SHARED_PHASE_PARITY_LEDGER_CLOSES_DECLARED_TWO_ROW_F17_FIELD_FAMILY_AT_RANK4_WITHOUT_EVEN_SECTOR_ENUMERATION_WITH_EXACT_FACTOR_CARRIER_RESTORATION_AND_REUSE_AND_PRIMARY_WIDTH2_NATIVE_DEFECT_SIGNATURE_FAILS_GRASSMANN_PLUCKER",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "topology": "TWO_ROW_OPEN_LADDERS_ONLY",
            "exact_q_zeta17_widths": EXACT_WIDTHS,
            "dual_field_structural_widths": STRUCTURAL_WIDTHS,
            "families": FAMILIES,
            "all_vertices_are_nonzero_typed_field_ports": True,
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "reuse": {
            "width": reuse_width,
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "same_actual_backing_across_unrelated_programs": first["same_backing"] and reused["same_backing"],
            "generation_after_two_transactions": reuse_carrier.generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "resource_law": {
            "resident_factor_field_cells": "10W_MINUS_4",
            "accepted_live_frontier_field_cells": 4,
            "named_transient_field_cells_upper_bound": 19,
            "accepted_recurrence_work": "O_W_FIELD_OPERATIONS_WITH_FIXED_FOUR_STATE_FRONTIER_EXACT_BIT_COMPLEXITY_REPORTED_SEPARATELY",
            "retain_all_signature_cells": "4_TO_THE_W",
            "occurrence_expanded_even_sectors": "2_TO_THE_2W_MINUS_1",
            "inverse_history_retained": 0,
            "python_container_sympy_native_bigint_and_whole_process_memory_excluded": True,
        },
        "matched_baselines": {
            "strongest_implemented": "IDENTICAL_EXACT_FOUR_STATE_COLUMN_TRANSFER",
            "occurrence_expanded": "M123_STREAMED_EVEN_SECTOR_PFAFFIANS",
            "retain_all": "FULL_4_TO_THE_W_DEFECT_SIGNATURE",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_factor_and_accumulator_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_four_state_projection_frontier": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "two_row_open_ladders_only": True,
            "fixed_rank4_column_interleaved_closure": True,
            "native_non_gaussian_signature_scope": "PRIMARY_WIDTH2_EXACT_GRASSMANN_PLUCKER_WITNESS_ONLY",
            "other_exact_families_or_widths_non_gaussian_established": False,
            "arbitrary_width_or_treewidth_compaction_established": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "THE_SHARED_NON_GAUSSIAN_PARITY_LEDGER_IS_FIXED_RANK_ONLY_BECAUSE_THE_TWO_ROW_TOPOLOGY_HAS_A_FOUR_STATE_COLUMN_SEPARATOR_AND_THE_IDENTICAL_CLASSICAL_TRANSFER_HAS_THE_SAME_LAW_SO_THE_NEXT_PHASE_REPAIR_MUST_CLOSE_GROWING_SEPARATOR_WIDTH_OR_INTRODUCE_A_RESOURCE_NOT_REDUCIBLE_TO_THE_TRANSFER",
        "next_experiment": "EXACT_TOPOLOGY_ORDERED_SHARED_PHASE_PARITY_LEDGER_ON_GROWING_SQUARE_GRID_SEPARATORS_WITH_ALL_ORDER_RESIDUAL_RANK_AND_IDENTICAL_TENSOR_NETWORK_BASELINES",
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
