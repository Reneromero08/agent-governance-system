#!/usr/bin/env python3
"""Exact Kronecker-butterfly repair of the M119 grid closure.

M119 correctly established compact resident phase factors, exact restoration,
and full row-interface ranks for the declared n=2,3,4 grids, but its accepted
row contraction enumerated every source/target pair.  Here the same actual
resident vertical factors are applied as topology-derived 2x2 butterflies.
The repair needs n*2^(n-1) butterflies per row interface and retains 2^n exact
messages.  It therefore repairs the transition-work defect without supplying
a separator quotient, distinct phase resource, or advantage.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_runtime_weighted_grid_phase_factor_closure as m119


pair = m119.pair
base = m119.base
ZERO = m119.ZERO
ONE = m119.ONE
ROOTS = m119.ROOTS
SplitElement = m119.SplitElement
GridPlan = m119.GridPlan
RuntimeProgram = m119.RuntimeProgram
GridCarrier = m119.GridCarrier
SIZES = m119.SIZES
FAMILIES = m119.FAMILIES


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class ButterflyStats:
    row_factor_multiplications: int = 0
    row_diagonal_root_actions: int = 0
    row_diagonal_residue_terms: int = 0
    target_diagonal_multiplications: int = 0
    butterfly_root_actions: int = 0
    butterfly_additions: int = 0
    boundary_additions: int = 0
    maximum_frontier_phase_cells: int = 0
    maximum_live_phase_cells: int = 0
    maximum_live_payload_bits: int = 0
    maximum_frontier_payload_bits: int = 0
    maximum_coordinate_signed_bits: int = 0
    row_interfaces: int = 0
    butterfly_stages: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


@dataclass
class Transaction:
    boundary: tuple[int, ...]
    phase_stats: m119.Stats
    butterfly_stats: ButterflyStats
    restored_exactly: bool
    same_backing: bool


def coordinate_signed_bits(value: SplitElement) -> int:
    return max(base.signed_bits(int(coordinate)) for lane in value for coordinate in lane)


def vector_payload_bits(values: list[SplitElement]) -> int:
    return sum(pair.split_payload_bits(value) for value in values)


def observe_values(stats: ButterflyStats, *groups: list[SplitElement]) -> None:
    flat = [value for group in groups for value in group]
    if not flat:
        return
    stats.maximum_live_phase_cells = max(stats.maximum_live_phase_cells, len(flat))
    stats.maximum_live_payload_bits = max(
        stats.maximum_live_payload_bits,
        sum(pair.split_payload_bits(value) for value in flat),
    )
    stats.maximum_coordinate_signed_bits = max(
        stats.maximum_coordinate_signed_bits,
        *(coordinate_signed_bits(value) for value in flat),
    )


def multiply(left: SplitElement, right: SplitElement, stats: ButterflyStats) -> SplitElement:
    stats.row_factor_multiplications += 1
    result = pair.split_multiply(left, right)
    observe_values(stats, [left, right, result])
    return result


def row_factor(
    carrier: GridCarrier,
    plan: GridPlan,
    row: int,
    assignment: int,
    lookup: dict[tuple[int, int], int],
    stats: ButterflyStats,
) -> SplitElement:
    row_bits = m119.bits(assignment, plan.n)
    value = ONE
    for column, bit in enumerate(row_bits):
        site = m119.vertex_index(plan.n, row, column)
        value = multiply(value, carrier.site_cells[site][bit], stats)
    for column in range(plan.n - 1):
        left = m119.vertex_index(plan.n, row, column)
        right = m119.vertex_index(plan.n, row, column + 1)
        edge = lookup[(left, right)]
        value = multiply(
            value,
            carrier.edge_cells[edge][2 * row_bits[column] + row_bits[column + 1]],
            stats,
        )
    return value


def apply_resident_interface_butterflies(
    carrier: GridCarrier,
    plan: GridPlan,
    lower_row: int,
    values: list[SplitElement],
    lookup: dict[tuple[int, int], int],
    stats: ButterflyStats,
    *,
    skip_column: int | None = None,
) -> list[SplitElement]:
    """Apply tensor_c [[1,1],[1,zeta**J_c]] to one row message.

    The three unit entries are checked on the actual resident edge cell.  The
    fourth entry is consumed directly by the one nontrivial root action.  Each
    stage overwrites a fresh 2^n buffer and retains no inverse history.
    """
    width = 1 << plan.n
    if len(values) != width:
        fail("butterfly frontier width changed")
    current = list(values)
    for column in range(plan.n):
        if skip_column == column:
            continue
        upper = m119.vertex_index(plan.n, lower_row - 1, column)
        lower = m119.vertex_index(plan.n, lower_row, column)
        edge = carrier.edge_cells[lookup[(upper, lower)]]
        if edge[0] != ONE or edge[1] != ONE or edge[2] != ONE:
            fail("resident vertical factor is not a declared K(J) kernel")
        root = edge[3]
        mask = 1 << (plan.n - 1 - column)
        following = list(current)
        for zero_index in range(width):
            if zero_index & mask:
                continue
            one_index = zero_index | mask
            low = current[zero_index]
            high = current[one_index]
            rooted_high = pair.split_multiply(root, high)
            output_zero = pair.split_add(low, high)
            output_one = pair.split_add(low, rooted_high)
            stats.butterfly_root_actions += 1
            stats.butterfly_additions += 2
            observe_values(
                stats,
                current,
                following,
                [root, low, high, rooted_high, output_zero, output_one],
            )
            following[zero_index] = output_zero
            following[one_index] = output_one
        current = following
        stats.butterfly_stages += 1
        stats.maximum_frontier_phase_cells = max(stats.maximum_frontier_phase_cells, width)
        stats.maximum_frontier_payload_bits = max(
            stats.maximum_frontier_payload_bits,
            vector_payload_bits(current),
        )
    return current


def resident_butterfly_contract(
    carrier: GridCarrier,
    plan: GridPlan,
    stats: ButterflyStats,
    *,
    skipped_stage: tuple[int, int] | None = None,
) -> SplitElement:
    lookup = m119.edge_index_map(plan)
    width = 1 << plan.n
    current = [row_factor(carrier, plan, 0, assignment, lookup, stats) for assignment in range(width)]
    stats.maximum_frontier_phase_cells = width
    stats.maximum_frontier_payload_bits = vector_payload_bits(current)
    observe_values(stats, current)
    for row in range(1, plan.n):
        skip_column = None
        if skipped_stage is not None and skipped_stage[0] == row:
            skip_column = skipped_stage[1]
        current = apply_resident_interface_butterflies(
            carrier,
            plan,
            row,
            current,
            lookup,
            stats,
            skip_column=skip_column,
        )
        stats.row_interfaces += 1
        following: list[SplitElement] = []
        for assignment, value in enumerate(current):
            factor = row_factor(carrier, plan, row, assignment, lookup, stats)
            product = pair.split_multiply(value, factor)
            stats.target_diagonal_multiplications += 1
            observe_values(stats, current, following, [value, factor, product])
            following.append(product)
        current = following
        stats.maximum_frontier_payload_bits = max(
            stats.maximum_frontier_payload_bits,
            vector_payload_bits(current),
        )
    boundary = ZERO
    for value in current:
        boundary = pair.split_add(boundary, value)
        stats.boundary_additions += 1
        observe_values(stats, current, [boundary, value])
    return boundary


def project_boundary(
    carrier: GridCarrier,
    plan: GridPlan,
    program: RuntimeProgram,
    phase_stats: m119.Stats,
) -> tuple[tuple[int, ...], ButterflyStats]:
    m119.validate_program(plan, program)
    if carrier.plan_fingerprint != program.plan_fingerprint or carrier.family != program.family:
        fail("projection program does not own the resident grid state")
    if carrier.cursor != len(plan.operations) or not carrier.active:
        fail("only the final grid boundary may be projected")
    if carrier.boundary_projected:
        fail("grid boundary was already projected")
    stats = ButterflyStats()
    boundary_pair = resident_butterfly_contract(carrier, plan, stats)
    full = pair.split_to_full(boundary_pair)
    phase_stats.boundary_full_lifts += 1
    phase_stats.maximum_transfer_live_payload_bits = max(
        phase_stats.maximum_transfer_live_payload_bits,
        stats.maximum_live_payload_bits,
    )
    phase_stats.maximum_transfer_live_phase_cells = max(
        phase_stats.maximum_transfer_live_phase_cells,
        stats.maximum_live_phase_cells,
    )
    phase_stats.maximum_boundary_live_payload_bits = max(
        phase_stats.maximum_boundary_live_payload_bits,
        pair.split_payload_bits(boundary_pair) + base.element_payload_bits(full),
    )
    phase_stats.maximum_accepted_resident_plus_work_payload_bits = max(
        phase_stats.maximum_accepted_resident_plus_work_payload_bits,
        carrier.resident_payload_bits() + stats.maximum_live_payload_bits,
    )
    carrier.boundary_projected = True
    return tuple(int(value) for value in full), stats


def execute_transaction(carrier: GridCarrier, plan: GridPlan, family: str) -> Transaction:
    if not isinstance(carrier, GridCarrier):
        fail("null or invalid butterfly grid carrier")
    program = m119.bind_runtime_program(plan, family)
    backing = carrier.backing_identity()
    phase_stats = m119.Stats()
    m119.load_factor_seed(carrier, plan, program, phase_stats)
    for ordinal in range(len(plan.operations)):
        m119.apply_operation(carrier, plan, program, ordinal, phase_stats)
    boundary, butterfly_stats = project_boundary(carrier, plan, program, phase_stats)
    for ordinal in reversed(range(len(plan.operations))):
        m119.apply_operation(carrier, plan, program, ordinal, phase_stats, inverse=True)
    m119.unload_factor_seed(carrier, plan, phase_stats)
    return Transaction(
        boundary=boundary,
        phase_stats=phase_stats,
        butterfly_stats=butterfly_stats,
        restored_exactly=carrier.all_zero(),
        same_backing=carrier.backing_identity() == backing,
    )


def compact_butterfly_boundary(
    plan: GridPlan,
    program: RuntimeProgram,
) -> tuple[tuple[int, ...], ButterflyStats]:
    """Strong compact descriptor recurrence with the identical interface law."""
    m119.validate_program(plan, program)
    lookup = m119.edge_index_map(plan)
    width = 1 << plan.n
    stats = ButterflyStats()

    def diagonal(row: int, assignment: int) -> SplitElement:
        row_bits = m119.bits(assignment, plan.n)
        exponent = 0
        for column, bit in enumerate(row_bits):
            if bit:
                exponent += program.unary_weights[m119.vertex_index(plan.n, row, column)]
                stats.row_diagonal_residue_terms += 1
        for column in range(plan.n - 1):
            if row_bits[column] and row_bits[column + 1]:
                left = m119.vertex_index(plan.n, row, column)
                right = m119.vertex_index(plan.n, row, column + 1)
                exponent += program.edge_weights[lookup[(left, right)]]
                stats.row_diagonal_residue_terms += 1
        stats.row_diagonal_root_actions += 1
        return ROOTS[exponent % m119.PRIME]

    current = [diagonal(0, assignment) for assignment in range(width)]
    stats.maximum_frontier_phase_cells = width
    stats.maximum_frontier_payload_bits = vector_payload_bits(current)
    observe_values(stats, current)
    for row in range(1, plan.n):
        for column in range(plan.n):
            upper = m119.vertex_index(plan.n, row - 1, column)
            lower = m119.vertex_index(plan.n, row, column)
            root = ROOTS[program.edge_weights[lookup[(upper, lower)]]]
            mask = 1 << (plan.n - 1 - column)
            following = list(current)
            for zero_index in range(width):
                if zero_index & mask:
                    continue
                one_index = zero_index | mask
                low = current[zero_index]
                high = current[one_index]
                rooted_high = pair.split_multiply(root, high)
                following[zero_index] = pair.split_add(low, high)
                following[one_index] = pair.split_add(low, rooted_high)
                stats.butterfly_root_actions += 1
                stats.butterfly_additions += 2
                observe_values(stats, current, following, [root, low, high, rooted_high])
            current = following
            stats.butterfly_stages += 1
        stats.row_interfaces += 1
        following = []
        for assignment, value in enumerate(current):
            factor = diagonal(row, assignment)
            product = pair.split_multiply(value, factor)
            stats.target_diagonal_multiplications += 1
            observe_values(stats, current, following, [value, factor, product])
            following.append(product)
        current = following
        stats.maximum_frontier_payload_bits = max(
            stats.maximum_frontier_payload_bits,
            vector_payload_bits(current),
        )
    boundary = ZERO
    for value in current:
        boundary = pair.split_add(boundary, value)
        stats.boundary_additions += 1
        observe_values(stats, current, [boundary, value])
    return tuple(int(value) for value in pair.split_to_full(boundary)), stats


def resource_summary(
    plan: GridPlan,
    program: RuntimeProgram,
    transaction: Transaction,
) -> dict[str, Any]:
    stats = transaction.butterfly_stats
    return {
        "actual_factor_carrier_cells": 2 * len(plan.vertices) + 4 * len(plan.edges),
        "dense_assignment_phase_cells_not_materialized": 1 << (plan.n * plan.n),
        "row_separator_message_cells": 1 << plan.n,
        "maximum_frontier_phase_cells": stats.maximum_frontier_phase_cells,
        "maximum_live_phase_cells_conservative_alias_inclusive": stats.maximum_live_phase_cells,
        "maximum_frontier_payload_bits": stats.maximum_frontier_payload_bits,
        "maximum_live_payload_bits_conservative_alias_inclusive": stats.maximum_live_payload_bits,
        "maximum_coordinate_signed_bits": stats.maximum_coordinate_signed_bits,
        "expected_butterfly_root_actions": (plan.n - 1) * plan.n * (1 << (plan.n - 1)),
        "expected_butterfly_additions": (plan.n - 1) * plan.n * (1 << plan.n),
        "four_to_the_n_source_target_transitions_materialized": False,
        "dense_transfer_matrix_materialized": False,
        "butterfly_layer_history_retained": False,
        "transient_butterfly_projection_buffers_retained_after_projection": False,
        "transient_butterfly_projection_buffers_restored_by_inverse": False,
        "transient_butterfly_projection_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "factor_program_root_table_and_public_topology_counted_separately": True,
        "plan_logical_payload_bits": plan.payload_bits(),
        "runtime_program_logical_payload_bits": program.payload_bits(),
        "root_table_payload_bits": m119.roots_payload_bits(),
        "phase_stats": transaction.phase_stats.as_json(),
        "python_allocator_bigint_native_library_and_process_rss_excluded": True,
    }


def case(plan: GridPlan, family: str, carrier: GridCarrier) -> dict[str, Any]:
    program = m119.bind_runtime_program(plan, family)
    transaction = execute_transaction(carrier, plan, family)
    compact_boundary, compact_stats = compact_butterfly_boundary(plan, program)
    expected_roots = (plan.n - 1) * plan.n * (1 << (plan.n - 1))
    expected_additions = 2 * expected_roots
    return {
        "n": plan.n,
        "family": family,
        "plan_fingerprint": plan.fingerprint(),
        "unary_weights": list(program.unary_weights),
        "edge_weights": list(program.edge_weights),
        "boundary": list(transaction.boundary),
        "matched_compact_descriptor_butterfly_boundary": list(compact_boundary),
        "boundary_agreement": transaction.boundary == compact_boundary,
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "separator_certificate": m119.separator_certificate(plan, program),
        "butterfly_stats": transaction.butterfly_stats.as_json(),
        "matched_compact_descriptor_butterfly_stats": compact_stats.as_json(),
        "butterfly_root_count_exact": transaction.butterfly_stats.butterfly_root_actions == expected_roots,
        "butterfly_addition_count_exact": transaction.butterfly_stats.butterfly_additions == expected_additions,
        "resources": resource_summary(plan, program, transaction),
    }


def control_results() -> dict[str, Any]:
    plan = m119.compile_topology(3)
    program = m119.bind_runtime_program(plan, "PRIMARY")

    premature = GridCarrier.create(plan)
    premature_stats = m119.Stats()
    m119.load_factor_seed(premature, plan, program, premature_stats)
    try:
        project_boundary(premature, plan, program, premature_stats)
        premature_rejected = False
    except RuntimeError:
        premature_rejected = True

    resident = GridCarrier.create(plan)
    resident_stats = m119.Stats()
    m119.load_factor_seed(resident, plan, program, resident_stats)
    for ordinal in range(len(plan.operations)):
        m119.apply_operation(resident, plan, program, ordinal, resident_stats)
    accepted_pair = resident_butterfly_contract(resident, plan, ButterflyStats())
    skipped_pair = resident_butterfly_contract(
        resident,
        plan,
        ButterflyStats(),
        skipped_stage=(1, 0),
    )
    lookup = m119.edge_index_map(plan)
    vertical = lookup[(m119.vertex_index(3, 0, 0), m119.vertex_index(3, 1, 0))]
    original = resident.edge_cells[vertical][3]
    resident.edge_cells[vertical][3] = pair.split_multiply(original, ROOTS[1])
    mutated_pair = resident_butterfly_contract(resident, plan, ButterflyStats())
    resident.edge_cells[vertical][3] = original
    for ordinal in reversed(range(len(plan.operations))):
        m119.apply_operation(resident, plan, program, ordinal, resident_stats, inverse=True)
    m119.unload_factor_seed(resident, plan, resident_stats)

    try:
        execute_transaction(None, plan, "PRIMARY")  # type: ignore[arg-type]
        null_rejected = False
    except RuntimeError:
        null_rejected = True

    projection_guard = GridCarrier.create(plan)
    guard_stats = m119.Stats()
    m119.load_factor_seed(projection_guard, plan, program, guard_stats)
    for ordinal in range(len(plan.operations)):
        m119.apply_operation(projection_guard, plan, program, ordinal, guard_stats)
    try:
        project_boundary(
            projection_guard,
            plan,
            m119.bind_runtime_program(plan, "REUSE"),
            guard_stats,
        )
        wrong_family_rejected = False
    except RuntimeError:
        wrong_family_rejected = True
    project_boundary(projection_guard, plan, program, guard_stats)
    for ordinal in reversed(range(len(plan.operations))):
        m119.apply_operation(projection_guard, plan, program, ordinal, guard_stats, inverse=True)
    m119.unload_factor_seed(projection_guard, plan, guard_stats)

    inherited = m119.control_results()
    return {
        "premature_projection_rejected": premature_rejected,
        "omitted_butterfly_stage_changes_boundary": skipped_pair != accepted_pair,
        "resident_vertical_factor_mutation_changes_boundary": mutated_pair != accepted_pair,
        "mutated_factor_reverted_before_inverse_and_carrier_restored": resident.all_zero(),
        "wrong_projection_family_rejected": wrong_family_rejected,
        "public_plan_excludes_runtime_weights_and_boundary": not inherited["compiled_plan_contains_runtime_weights"],
        "wrong_plan_fingerprint_rejected": inherited["wrong_plan_fingerprint_rejected"],
        "wrong_projection_fingerprint_rejected": inherited["wrong_projection_fingerprint_rejected"],
        "one_zero_weight_separator_rank_halves": inherited["one_separator_edge_removed_rank_halves"],
        "one_zero_weight_separator_changes_boundary": inherited["one_separator_edge_removed_changes_boundary"],
        "false_separator_rank_cap_rejected": inherited["forced_rank_below_certificate_rejected"],
        "projection_guard_restored": projection_guard.all_zero(),
        "null_carrier_rejected": null_rejected,
        "missing_inverse_leaves_resident_state": inherited["missing_inverse_leaves_resident_state"],
        "wrong_inverse_exponent_fails_restoration": inherited["wrong_inverse_exponent_fails_restoration"],
        "reordered_noncommuting_inverse_fails": inherited["reordered_noncommuting_unary_prepare_inverse_fails"],
        "resident_mutation_detected": inherited["resident_mutation_detected"],
        "snapshot_reload_absent": True,
        "commuting_distinct_bit_butterfly_stage_reorder_failure_required": False,
    }


def reuse_signature(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "boundary": value["boundary"],
        "separator_certificate": value["separator_certificate"],
        "butterfly_stats": value["butterfly_stats"],
        "matched_compact_descriptor_butterfly_stats": value["matched_compact_descriptor_butterfly_stats"],
    }


def main() -> int:
    cases: list[dict[str, Any]] = []
    reuse: list[dict[str, Any]] = []
    for n in SIZES:
        plan = m119.compile_topology(n)
        carrier = GridCarrier.create(plan)
        backing = carrier.backing_identity()
        primary = case(plan, "PRIMARY", carrier)
        restored_reuse = case(plan, "REUSE", carrier)
        fresh_reuse = case(plan, "REUSE", GridCarrier.create(plan))
        cases.extend((primary, restored_reuse))
        reuse.append({
            "n": n,
            "same_original_backing": carrier.backing_identity() == backing,
            "fresh_restored_reuse_equal": reuse_signature(restored_reuse) == reuse_signature(fresh_reuse),
            "generation": carrier.generation,
            "lease": carrier.lease,
            "canonical_restored_state": carrier.canonical_state(),
            "baseline_reload": False,
            "retained_inverse_history_bytes": 0,
        })
    controls = control_results()
    if not all(
        item["boundary_agreement"]
        and item["restored_exactly"]
        and item["same_backing"]
        and item["butterfly_root_count_exact"]
        and item["butterfly_addition_count_exact"]
        for item in cases
    ):
        fail("Kronecker-butterfly cases failed")
    if not all(
        item["same_original_backing"]
        and item["fresh_restored_reuse_equal"]
        and item["canonical_restored_state"]["all_factor_cells_zero"]
        and not item["baseline_reload"]
        for item in reuse
    ):
        fail("Kronecker-butterfly restored reuse failed")
    if not all(value for key, value in controls.items() if key != "commuting_distinct_bit_butterfly_stage_reorder_failure_required"):
        fail("Kronecker-butterfly controls failed")

    result = {
        "experiment": "RUNTIME_WEIGHTED_F17_GRID_PHASE_FACTOR_KRONECKER_BUTTERFLY_CLOSURE",
        "result": "PASS_EXACT_TRANSITION_WORK_REPAIR_WITH_UNCHANGED_FULL_SEPARATOR_RANK",
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "restoration_scope": "ACTUAL_BORROWED_FACTOR_CARRIER_FORWARD_PHASE_ACTIONS_REVERSED_AND_SEED_UNLOADED_TO_ORIGINAL_ZERO_BACKING",
        "transient_butterfly_projection_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "execution_scope": "LINUX_DIRECT_PROCESS_SOFTWARE",
        "historical_m119_evidence_modified": False,
        "accepted_path_uses_actual_resident_factor_cells": True,
        "accepted_path_dense_transfer_matrix_materialized": False,
        "accepted_path_source_target_pair_enumeration": False,
        "accepted_path_butterfly_root_actions": [
            (n - 1) * n * (1 << (n - 1)) for n in SIZES
        ],
        "accepted_path_butterfly_additions": [
            (n - 1) * n * (1 << n) for n in SIZES
        ],
        "accepted_path_row_separator_message_widths": [1 << n for n in SIZES],
        "accepted_path_final_full_lifts_per_transaction": 1,
        "intermediate_factor_or_frontier_values_projected": False,
        "retained_inverse_history_bytes": 0,
        "cases": cases,
        "restoration_reuse": reuse,
        "controls": controls,
        "observed_resource_law": {
            "grid_sizes": list(SIZES),
            "exact_actual_row_interface_ranks": [1 << n for n in SIZES],
            "row_separator_message_cells": [1 << n for n in SIZES],
            "butterfly_root_actions": [(n - 1) * n * (1 << (n - 1)) for n in SIZES],
            "factor_carrier_cells": [2 * n * n + 8 * n * (n - 1) for n in SIZES],
            "dense_assignment_cells_avoided": [1 << (n * n) for n in SIZES],
            "rank_remains_full_after_work_repair": True,
        },
        "matched_classical": {
            "strongest_evaluated_row_recurrence": "IDENTICAL_EXACT_KRONECKER_BUTTERFLY_ON_2_TO_N_Q_ZETA17_MESSAGES",
            "identical_interface_butterfly_boundaries_and_counts_reported": True,
            "resident_phase_and_descriptor_diagonal_generation_are_not_operation_matched": True,
            "gray_delta_17_bin_global_character_histogram_retained_as_lower_memory_higher_work_comparison": True,
            "exact_best_order_treewidth_or_matchgate_reduction_proven_exhausted": False,
            "comparison_establishes_advantage": False,
        },
        "claim_candidate": "BOUNDED_EXACT_RUNTIME_WEIGHTED_F17_GRID_PHASE_FACTOR_KRONECKER_BUTTERFLY_CLOSURE_FOR_N2_N3_N4_WITH_FULL_ROW_INTERFACE_RANKS4_8_16_TOPOLOGY_DERIVED_N_TIMES_TWO_TO_THE_N_TRANSFER_FINAL_ONLY_SCALAR_PROJECTION_EXACT_REVERSE_FACTOR_CARRIER_RESTORATION_AND_HELD_OUT_SAME_BACKING_REUSE_BUT_NO_COMPACTION_BELOW_TWO_TO_THE_N_MESSAGES_OR_DISTINCT_PHASE_RESOURCE",
        "not_established": [
            "SEPARATOR_QUOTIENT_BELOW_TWO_TO_THE_N_MESSAGES",
            "ASYMPTOTIC_LOWER_BOUND_AGAINST_ALL_BOUNDARY_ALGORITHMS",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "MACHINE_ENFORCED_CATVM_CUSTODY",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": "THE_FOUR_TO_THE_N_TRANSFER_WORK_DEFECT_IS_REPAIRED_BUT_EXACT_ROW_INTERFACE_RANK_AND_MESSAGE_STORAGE_REMAIN_TWO_TO_THE_N_AND_THE_IDENTICAL_CLASSICAL_BUTTERFLY_HAS_THE_SAME_RECURRENCE",
        "next_experiment": "OVERLAPPING_CUBIC_PHASE_AND_EXACT_MIXING_COMPOSITION_WITH_THIRD_BOOLEAN_DIFFERENCE_CERTIFICATE_AGAINST_MATCHGATE_TENSOR_NETWORK_AND_DECISION_DIAGRAM_BASELINES",
        "terminal": False,
    }
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
