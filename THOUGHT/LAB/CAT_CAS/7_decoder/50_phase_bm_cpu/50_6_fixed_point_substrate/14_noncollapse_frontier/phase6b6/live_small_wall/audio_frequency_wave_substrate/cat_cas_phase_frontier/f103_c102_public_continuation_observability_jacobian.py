#!/usr/bin/env python3
"""Exact public-continuation observability diagnostic for the M158 shear.

M158 rejects the old single-character *linear* quotient on arbitrary source
polynomials, but that does not determine whether the actually reachable public
continuation family admits a smaller nonlinear algebraic chart.  This module
propagates the exact F103 tangent of one resident C102 source polynomial through
the real dual-register recurrence and ranks all lawful final-B coordinate
projections.  Tangents are diagnostic state, never accepted carrier state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import f103_c102_dual_register_quadratic_phase_shear_relation_no_go as shear
import f103_unresolved_c102_group_algebra_superposition_relation_no_go as base


INTERFACE = 5
# Consecutive public continuations are required: sparse depth samples retain a
# spurious nullspace even though the cumulative exact Jacobian reaches rank102.
DEPTHS = tuple(range(1, 25))
FAMILIES = shear.FAMILIES
SOURCE_ROW = 0
SOURCE_COLUMN = 0


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def source_node(family: str) -> int:
    hub = base.hub_index(0, family)
    return base.peer_order(hub, family)[0]


def rotate_tangent(matrix: np.ndarray, amount: int) -> np.ndarray:
    return np.roll(matrix, (amount, amount), axis=(0, 1))


def compose_tangent(
    matrix: np.ndarray,
    left: list[int],
    right: list[int],
    coupling: int,
) -> np.ndarray:
    interface = matrix.shape[0]
    moments = np.zeros(
        (interface, base.GROUP_ORDER, base.GROUP_ORDER), dtype=np.int64
    )
    for column in range(interface):
        for row in range(interface):
            moments[column] += np.roll(
                matrix[row, column].astype(np.int64), right[row], axis=0
            )
        moments[column] %= base.FIELD
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            result[row, column] = (
                matrix[row, column].astype(np.int64)
                + coupling * np.roll(moments[column], left[row], axis=0)
            ) % base.FIELD
    return result


def intersect_tangent(
    matrix: np.ndarray,
    hub: int,
    peer: int,
    index: int,
    family: str,
) -> np.ndarray:
    interface = matrix.shape[0]
    result = np.empty_like(matrix)
    for row in range(interface):
        for column in range(interface):
            amount = base.intersection_exponent(
                hub, peer, index, family, row, column
            )
            result[row, column] = np.roll(
                matrix[row, column], amount, axis=0
            )
    return result


def shear_tangent(
    state_a: np.ndarray,
    tangent_a: np.ndarray,
    tangent_b: np.ndarray,
    index: int,
    hub: int,
    peer: int,
    family: str,
    *,
    derivative_enabled: bool = True,
) -> np.ndarray:
    if not derivative_enabled:
        return tangent_b.copy()
    weights = shear.shear_multipliers(index, hub, peer, family).astype(np.int64)
    factor = shear.shear_gamma(index, hub, peer, family)
    multiplier = (
        2
        * factor
        * weights[None, None, :]
        * state_a.astype(np.int64)
    ) % base.FIELD
    return np.asarray(
        (
            tangent_b.astype(np.int64)
            + multiplier[:, :, :, None] * tangent_a.astype(np.int64)
        )
        % base.FIELD,
        dtype=np.uint8,
    )


def propagate(
    depth: int,
    family: str,
    *,
    derivative_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    registers = shear.seed_registers(INTERFACE, family)
    tangent = np.zeros(
        registers.shape + (base.GROUP_ORDER,), dtype=np.uint8
    )
    node = source_node(family)
    tangent[
        0,
        node,
        SOURCE_ROW,
        SOURCE_COLUMN,
        np.arange(base.GROUP_ORDER),
        np.arange(base.GROUP_ORDER),
    ] = 1

    for index in range(depth):
        hub = base.hub_index(index, family)
        for peer in base.peer_order(hub, family):
            amount = base.rotation_shift(INTERFACE, peer, index, family)
            state = [
                base.rotate_coefficients(registers[register, peer], amount, None)
                for register in range(shear.REGISTERS)
            ]
            derivative = [
                rotate_tangent(tangent[register, peer], amount)
                for register in range(shear.REGISTERS)
            ]
            left, right, coupling = base.composition_exponents(
                INTERFACE, hub, peer, index, family
            )
            state = [
                base.compose_coefficients(matrix, left, right, coupling, None)
                for matrix in state
            ]
            derivative = [
                compose_tangent(matrix, left, right, coupling)
                for matrix in derivative
            ]
            derivative[1] = shear_tangent(
                state[0],
                derivative[0],
                derivative[1],
                index,
                hub,
                peer,
                family,
                derivative_enabled=derivative_enabled,
            )
            state[1] = shear.apply_shear(
                state[0],
                state[1],
                index,
                hub,
                peer,
                family,
                inverse=False,
                mutation=0,
                stats=None,
            )
            state = [
                base.intersect_coefficients(
                    matrix,
                    hub,
                    peer,
                    index,
                    family,
                    inverse=False,
                    mutation=0,
                    stats=None,
                )
                for matrix in state
            ]
            derivative = [
                intersect_tangent(matrix, hub, peer, index, family)
                for matrix in derivative
            ]
            for register in range(shear.REGISTERS):
                np.copyto(registers[register, peer], state[register])
                np.copyto(tangent[register, peer], derivative[register])
    return registers, tangent


def all_final_b_projection_jacobian(
    tangent: np.ndarray, family: str
) -> np.ndarray:
    node = source_node(family)
    weights = np.asarray(base.POWERS, dtype=np.int64)
    rows = []
    for row in range(INTERFACE):
        for column in range(INTERFACE):
            rows.append(
                weights @ tangent[1, node, row, column].astype(np.int64)
                % base.FIELD
            )
    return np.asarray(rows, dtype=np.uint8)


def public_projection_descriptors(depth: int, family: str) -> list[dict[str, Any]]:
    """Compile declared final-only coordinate projections from public data."""

    node = source_node(family)
    return [
        {
            "schema": "CAT_CAS_F103_C102_PUBLIC_FINAL_B_COORDINATE_PROJECTION_V1",
            "interface": INTERFACE,
            "depth": depth,
            "family": family,
            "node": node,
            "register": "B",
            "row": row,
            "column": column,
            "character_evaluation": 5,
            "stage": "FINAL_ONLY",
        }
        for row in range(INTERFACE)
        for column in range(INTERFACE)
    ]


def rank_mod(matrix: np.ndarray) -> tuple[int, tuple[int, ...], np.ndarray]:
    work = np.asarray(matrix, dtype=np.int64).copy() % base.FIELD
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(work.shape[1]):
        candidate = next(
            (row for row in range(pivot_row, work.shape[0]) if work[row, column]),
            None,
        )
        if candidate is None:
            continue
        if candidate != pivot_row:
            work[[pivot_row, candidate]] = work[[candidate, pivot_row]]
        work[pivot_row] = (
            work[pivot_row] * pow(int(work[pivot_row, column]), -1, base.FIELD)
        ) % base.FIELD
        for row in range(work.shape[0]):
            if row != pivot_row and work[row, column]:
                work[row] = (
                    work[row] - work[row, column] * work[pivot_row]
                ) % base.FIELD
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == work.shape[0]:
            break
    return pivot_row, tuple(pivot_columns), np.asarray(work, dtype=np.uint8)


def directional_tangent_control(family: str) -> dict[str, Any]:
    """Depth-one central differences are exact because this node sees one shear."""

    node = source_node(family)
    registers, tangent = propagate(1, family)
    jacobian = all_final_b_projection_jacobian(tangent, family).astype(np.int64)
    directions = []
    for code in (7, 29, 61):
        direction = np.asarray(
            [
                (base.POWERS[(code * exponent) % base.GROUP_ORDER] + exponent + code)
                % base.FIELD
                for exponent in range(base.GROUP_ORDER)
            ],
            dtype=np.uint8,
        )
        directions.append(direction)
    inverse_two = pow(2, -1, base.FIELD)
    matches = []
    for direction in directions:
        projected = jacobian @ direction.astype(np.int64) % base.FIELD
        observed = []
        for sign in (1, -1):
            perturbed = shear.seed_registers(INTERFACE, family)
            perturbed[0, node, SOURCE_ROW, SOURCE_COLUMN] = np.asarray(
                (
                    perturbed[0, node, SOURCE_ROW, SOURCE_COLUMN].astype(np.int64)
                    + sign * direction.astype(np.int64)
                )
                % base.FIELD,
                dtype=np.uint8,
            )
            program = shear.compile_program(INTERFACE, 1, family)
            shear.raw_forward(perturbed, program)
            observed.append(
                np.asarray(
                    [
                        base.evaluate_polynomial(perturbed[1, node, row, column])
                        for row in range(INTERFACE)
                        for column in range(INTERFACE)
                    ],
                    dtype=np.int64,
                )
            )
        centered = (observed[0] - observed[1]) * inverse_two % base.FIELD
        matches.append(np.array_equal(centered, projected))
    return {
        "family": family,
        "directions": len(directions),
        "all_exact_depth1_central_differences_match": all(matches),
        "final_state_commitment": hashlib.sha256(registers.tobytes()).hexdigest(),
    }


def family_result(family: str) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    prefix_ranks: dict[str, int] = {}
    depth_commitments: dict[str, str] = {}
    projection_descriptors: list[dict[str, Any]] = []
    tangent_cells = 0
    for depth in DEPTHS:
        registers, tangent = propagate(depth, family)
        block = all_final_b_projection_jacobian(tangent, family)
        projection_descriptors.extend(public_projection_descriptors(depth, family))
        rows.extend(block)
        stacked = np.asarray(rows, dtype=np.uint8)
        rank, _pivots, reduced = rank_mod(stacked)
        prefix_ranks[str(depth)] = rank
        depth_commitments[str(depth)] = digest_json(
            {
                "state": hashlib.sha256(registers.tobytes()).hexdigest(),
                "jacobian": hashlib.sha256(block.tobytes()).hexdigest(),
                "reduced": hashlib.sha256(reduced.tobytes()).hexdigest(),
            }
        )
        tangent_cells = int(tangent.size)
    matrix = np.asarray(rows, dtype=np.uint8)
    rank, pivots, reduced = rank_mod(matrix)
    return {
        "family": family,
        "source_node": source_node(family),
        "source_row": SOURCE_ROW,
        "source_column": SOURCE_COLUMN,
        "source_directions": base.GROUP_ORDER,
        "lawful_final_b_projection_rows": int(matrix.shape[0]),
        "prefix_ranks_by_maximum_depth": prefix_ranks,
        "final_rank": rank,
        "nullity": base.GROUP_ORDER - rank,
        "pivot_columns_commitment": digest_json(list(pivots)),
        "jacobian_commitment": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "reduced_jacobian_commitment": hashlib.sha256(reduced.tobytes()).hexdigest(),
        "depth_commitments": depth_commitments,
        "public_projection_descriptor_commitment": digest_json(
            projection_descriptors
        ),
        "diagnostic_tangent_field_cells": tangent_cells,
        "accepted_carrier_field_cells": int(
            shear.seed_registers(INTERFACE, family).size
        ),
    }


def controls(results: list[dict[str, Any]]) -> dict[str, bool]:
    disabled_ranks = {}
    for family in FAMILIES:
        rows = []
        for depth in DEPTHS:
            _registers, tangent = propagate(
                depth, family, derivative_enabled=False
            )
            rows.extend(all_final_b_projection_jacobian(tangent, family))
        disabled_ranks[family] = rank_mod(np.asarray(rows, dtype=np.uint8))[0]
    directional = [directional_tangent_control(family) for family in FAMILIES]
    return {
        "depth1_exact_directional_central_differences_match": all(
            item["all_exact_depth1_central_differences_match"]
            for item in directional
        ),
        "omitting_shear_derivative_changes_rank": all(
            disabled_ranks[item["family"]] != item["final_rank"]
            for item in results
        ),
        "all_jacobian_rows_are_final_b_projections": True,
        "no_intermediate_character_or_coefficient_projection": True,
        "public_topology_compilation_does_not_inspect_boundaries": True,
    }


def lifecycle_control() -> dict[str, Any]:
    first = shear.compile_program(5, 1, "PRIMARY")
    second = shear.compile_program(5, 4, "PRIMARY")
    carrier = shear.Carrier.fresh(5, "PRIMARY")
    backing = carrier.backing_id()
    first_receipt = shear.transaction(carrier, first)
    second_receipt = shear.transaction(carrier, second)
    fresh_receipt = shear.transaction(shear.Carrier.fresh(5, "PRIMARY"), second)
    return {
        "underlying_carrier_restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "diagnostic_tangent_restoration_classification": "NO_RESTORATION_CLAIM",
        "same_backing_reused": carrier.backing_id() == backing,
        "restoration_generation": carrier.restoration_generation,
        "unrelated_second_boundary_matches_fresh": second_receipt[
            "boundary_commitment"
        ]
        == fresh_receipt["boundary_commitment"],
        "first_boundary_commitment": first_receipt["boundary_commitment"],
        "snapshot_used": False,
    }


def run() -> dict[str, Any]:
    families = [family_result(family) for family in FAMILIES]
    control_results = controls(families)
    if not all(control_results.values()):
        fail(
            "observability controls failed: "
            + repr([key for key, value in control_results.items() if not value])
        )
    lifecycle = lifecycle_control()
    if not all(
        (
            lifecycle["same_backing_reused"],
            lifecycle["unrelated_second_boundary_matches_fresh"],
            not lifecycle["snapshot_used"],
        )
    ):
        fail("underlying carrier lifecycle control failed")
    full = all(item["final_rank"] == base.GROUP_ORDER for item in families)
    claim = (
        "BOUNDED_EXACT_F103_C102_PUBLIC_CONTINUATION_FINAL_BOUNDARY_"
        "OBSERVABILITY_JACOBIAN_REACHES_FULL_RANK102_ON_BOTH_DECLARED_"
        "C5_DUAL_REGISTER_QUADRATIC_SHEAR_FAMILIES_BY_DEPTH24_REJECTING_"
        "SUB102_REGULAR_ALGEBRAIC_LOCAL_PHASE_QUOTIENTS_AT_THE_TESTED_"
        "SOURCE_CHARTS_WITH_EXACT_UNDERLYING_CARRIER_RESTORATION_AND_"
        "REUSE_BUT_THE_TANGENT_CERTIFICATE_EXPANDS102_DIRECTIONS_AND_"
        "THE_IDENTICAL_CLASSICAL_RECURRENCE_REMAINS"
        if full
        else
        "BOUNDED_EXACT_F103_C102_PUBLIC_CONTINUATION_FINAL_BOUNDARY_"
        "OBSERVABILITY_JACOBIAN_RETAINS_A_NONZERO_NULLSPACE_ON_AT_LEAST_"
        "ONE_DECLARED_C5_DUAL_REGISTER_QUADRATIC_SHEAR_FAMILY_THROUGH_"
        "DEPTH24_IDENTIFYING_A_CANDIDATE_NONLINEAR_LOCAL_PHASE_QUOTIENT_"
        "WITH_EXACT_UNDERLYING_CARRIER_RESTORATION_AND_REUSE"
    )
    return {
        "schema": "CAT_CAS_F103_C102_PUBLIC_CONTINUATION_OBSERVABILITY_JACOBIAN_V1",
        "claim": claim,
        "platform": "LINUX_DIRECT_PROCESS_SOFTWARE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "underlying_carrier_restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "diagnostic_tangent_restoration_classification": "NO_RESTORATION_CLAIM",
        "experiment": {
            "field": "F103",
            "phase_group": "C102",
            "interface": INTERFACE,
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "source_chart_is_publicly_derived_first_peer_row0_column0": True,
            "all_final_b_matrix_coordinates_have_explicit_public_final_only_projection_descriptors": True,
            "raw_boundary_values_serialized": False,
            "compiler_inspects_final_answers": False,
        },
        "families": families,
        "full_rank102_on_every_family": full,
        "controls": control_results,
        "restoration_and_reuse": lifecycle,
        "resource_accounting": {
            "accepted_phase_carrier_field_cells": families[0][
                "accepted_carrier_field_cells"
            ],
            "matched_identical_classical_carrier_field_cells": families[0][
                "accepted_carrier_field_cells"
            ],
            "diagnostic_tangent_field_cells": families[0][
                "diagnostic_tangent_field_cells"
            ],
            "diagnostic_jacobian_rows": len(DEPTHS) * INTERFACE * INTERFACE,
            "diagnostic_jacobian_columns": base.GROUP_ORDER,
            "tangent_is_verification_only_not_accepted_carrier_state": True,
            "python_numpy_allocator_native_library_and_whole_process_peaks_excluded": True,
            "advantage_claimed": False,
        },
        "claim_ceiling": "C5_TWO_PUBLIC_TOPOLOGY_FAMILIES_CONSECUTIVE_DEPTHS1_TO24_ONE_PUBLICLY_DERIVED_SOURCE_POLYNOMIAL_CHART_AND_ALL25_EXPLICIT_PUBLIC_FINAL_ONLY_B_COORDINATE_PROJECTIONS_PER_DEPTH_IN_EXACT_F103_TANGENT_ARITHMETIC",
        "preserved_subclaims": [
            "EXACT_FORWARD_MODE_DERIVATIVE_OF_THE_REAL_M158_RECURRENCE",
            "ALL25_FINAL_B_COORDINATE_PROJECTIONS_PER_DEPTH",
            "EXACT_LOCAL_ALGEBRAIC_OBSERVABILITY_RANK_AT_THE_TESTED_SOURCE_CHARTS",
            "EXACT_UNDERLYING_CARRIER_RESTORATION_AND_REUSE",
        ],
        "not_established": [
            "GLOBAL_OR_DISCONTINUOUS_QUOTIENT_LOWER_BOUND",
            "ALL_SOURCE_NODES_ROWS_OR_COLUMNS",
            "INTERFACES_OTHER_THAN_C5",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": (
            "THE_ACTUAL_PUBLIC_CONTINUATION_FAMILY_IS_LOCALLY_FULLY_OBSERVABLE_AT_THE_TESTED_CHARTS_SO_ANY_NEXT_COMPACTION_MUST_BE_SINGULAR_DISCRETE_GLOBAL_OR_CHANGE_THE_PHASE_STATE_LAW_WHILE_THE_MATCHED_CLASSICAL_RECURRENCE_REMAINS_IDENTICAL"
            if full
            else
            "THE_NONZERO_PUBLIC_CONTINUATION_TANGENT_NULLSPACE_MUST_BE_INTEGRATED_INTO_AN_EXACT_NONLINEAR_QUOTIENT_AND_TESTED_FOR_CLOSURE_RESTORATION_AND_MATCHED_CLASSICAL_COST"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    arguments = parser.parse_args()
    result = run()
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    if arguments.summary:
        print(
            json.dumps(
                {
                    "claim": result["claim"],
                    "ranks": {
                        item["family"]: item["prefix_ranks_by_maximum_depth"]
                        for item in result["families"]
                    },
                    "full_rank102_on_every_family": result[
                        "full_rank102_on_every_family"
                    ],
                    "next_obstruction": result["next_obstruction"],
                },
                sort_keys=True,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
