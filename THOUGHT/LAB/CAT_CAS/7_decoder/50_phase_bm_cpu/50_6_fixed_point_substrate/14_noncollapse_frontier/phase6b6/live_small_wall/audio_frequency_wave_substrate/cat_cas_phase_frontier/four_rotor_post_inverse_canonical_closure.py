#!/usr/bin/env python3
"""In-place post-inverse TT rounding and fresh/reuse resource parity."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

import four_rotor_kicked_phase_tt as reference
import four_rotor_kicked_phase_tt_matrix_free as matrix_free


STRICT_POST_INVERSE_L2 = 1.0e-7
MATRIX_FREE_POST_INVERSE_L2 = 1.0e-6
REUSE_BOUNDARY_PARITY = 3.0e-8
REUSE_K = 0.9
REUSE_G = 0.22
REUSE_TAU = math.sqrt(5.0)
REUSE_DEPTH = 2


def fail(message: str) -> None:
    raise RuntimeError(message)


def bond_ranks(carrier: reference.Carrier) -> list[int]:
    return [int(tensor.shape[2]) for tensor in carrier.tensors[:-1]]


def carrier_cells(carrier: reference.Carrier) -> int:
    return matrix_free.carrier_backing_cells(carrier)


def logical_carrier_cells(carrier: reference.Carrier) -> int:
    return sum(int(tensor.size) for tensor in carrier.tensors)


@dataclass
class ClosureStats:
    qr_factorizations: int = 0
    svd_factorizations: int = 0
    discarded_l2_squared: float = 0.0
    maximum_workspace_array_cells: int = 0
    maximum_simultaneous_cells: int = 0


def account_workspace(
    carrier: reference.Carrier,
    stats: ClosureStats,
    *arrays: np.ndarray,
) -> None:
    workspace = sum(int(array.size) for array in arrays)
    stats.maximum_workspace_array_cells = max(
        stats.maximum_workspace_array_cells,
        *(int(array.size) for array in arrays),
    )
    stats.maximum_simultaneous_cells = max(
        stats.maximum_simultaneous_cells,
        carrier_cells(carrier) + workspace,
    )


def canonical_round_actual(
    carrier: reference.Carrier,
    total_l2_tolerance: float,
) -> dict[str, object]:
    """Round the actual carrier once without consulting a baseline state."""
    if total_l2_tolerance <= 0.0:
        fail("post-inverse canonical tolerance must be positive")
    carrier_identity = id(carrier)
    ranks_before = bond_ranks(carrier)
    cells_before = carrier_cells(carrier)
    stats = ClosureStats(maximum_simultaneous_cells=cells_before)

    # Left-orthogonalize the actual cores. R factors are absorbed directly
    # into their resident neighbors; no sealed baseline participates.
    for site in range(reference.ROTORS - 1):
        core = carrier.tensors[site]
        neighbor = carrier.tensors[site + 1]
        left, physical, right = core.shape
        q, r_factor = np.linalg.qr(
            core.reshape(left * physical, right), mode="reduced"
        )
        rank = q.shape[1]
        new_core = q.reshape(left, physical, rank)
        new_neighbor = np.einsum(
            "ab,bnc->anc", r_factor, neighbor, optimize=True
        )
        account_workspace(
            carrier,
            stats,
            q,
            r_factor,
            new_core,
            new_neighbor,
        )
        carrier.tensors[site] = new_core
        carrier.tensors[site + 1] = new_neighbor
        stats.qr_factorizations += 1

    bond_budget = total_l2_tolerance / math.sqrt(
        reference.ROTORS - 1
    )
    per_bond: list[dict[str, object]] = []
    # A single right-to-left SVD sweep is ordinary TT rounding. Each cut
    # receives a disjoint share of the declared global L2 budget.
    for site in range(reference.ROTORS - 1, 0, -1):
        core = carrier.tensors[site]
        previous = carrier.tensors[site - 1]
        left, physical, right = core.shape
        u, singular, vh = np.linalg.svd(
            core.reshape(left, physical * right),
            full_matrices=False,
        )
        squared = singular * singular
        keep = len(singular)
        for candidate in range(1, len(singular) + 1):
            discarded = float(np.sum(squared[candidate:]))
            if discarded <= bond_budget**2:
                keep = candidate
                break
        discarded = float(np.sum(squared[keep:]))
        factor = u[:, :keep] * singular[:keep]
        new_core = vh[:keep].reshape(keep, physical, right).copy()
        new_previous = np.einsum(
            "anb,bc->anc", previous, factor, optimize=True
        )
        account_workspace(
            carrier,
            stats,
            u,
            singular,
            vh,
            squared,
            factor,
            new_core,
            new_previous,
        )
        carrier.tensors[site] = new_core
        carrier.tensors[site - 1] = new_previous
        stats.svd_factorizations += 1
        stats.discarded_l2_squared += discarded
        per_bond.append(
            {
                "site": site,
                "input_rank": left,
                "retained_rank": keep,
                "discarded_l2_squared": discarded,
            }
        )

    if id(carrier) != carrier_identity:
        fail("post-inverse closure replaced carrier custody object")
    ranks_after = bond_ranks(carrier)
    cells_after = carrier_cells(carrier)
    if cells_after != logical_carrier_cells(carrier):
        fail("canonical carrier retains oversized NumPy backing allocation")
    return {
        "actual_carrier_object_preserved": True,
        "baseline_state_consulted": False,
        "snapshot_loaded": False,
        "single_rounding_sweep": True,
        "canonical_center": "LEFT_SITE_0",
        "total_l2_tolerance": total_l2_tolerance,
        "per_bond_l2_budget": bond_budget,
        "bond_ranks_before": ranks_before,
        "bond_ranks_after": ranks_after,
        "carrier_cells_before": cells_before,
        "carrier_cells_after": cells_after,
        "retained_backing_matches_logical_cells": True,
        "carrier_payload_bytes_after": cells_after * 16,
        "qr_factorizations": stats.qr_factorizations,
        "svd_factorizations": stats.svd_factorizations,
        "discarded_l2_squared": stats.discarded_l2_squared,
        "maximum_workspace_array_cells": (
            stats.maximum_workspace_array_cells
        ),
        "maximum_simultaneous_cells": stats.maximum_simultaneous_cells,
        "maximum_simultaneous_payload_bytes": (
            stats.maximum_simultaneous_cells * 16
        ),
        "per_bond": per_bond,
    }


def matrix_free_transaction_with_closure(
    carrier: reference.Carrier,
    depth: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    initial = reference.copy_carrier(carrier)
    stats = matrix_free.Stats()
    rank_history: list[int] = []
    central: list[float] = [1.0]
    starting_ranks = bond_ranks(carrier)
    for _ in range(depth):
        central = matrix_free.forward_round(
            carrier,
            kick_strength,
            coupling_strength,
            free_time,
            stats,
        )
        rank_history.append(len(central))
    latched = reference.boundary(carrier, central)
    for _ in range(depth):
        matrix_free.inverse_round(
            carrier,
            kick_strength,
            coupling_strength,
            free_time,
            stats,
        )
    preclosure_error = reference.physical_distance(initial, carrier)
    if preclosure_error > matrix_free.MATRIX_FREE_RESTORATION_TOLERANCE:
        fail("actual inverse failed before post-inverse closure")
    closure = canonical_round_actual(
        carrier, MATRIX_FREE_POST_INVERSE_L2
    )
    postclosure_error = reference.physical_distance(initial, carrier)
    if postclosure_error > (
        preclosure_error + MATRIX_FREE_POST_INVERSE_L2
    ):
        fail("post-inverse canonical closure exceeded error budget")
    carrier.generation += 1
    return {
        "boundary": latched,
        "central_rank_history": rank_history,
        "starting_bond_ranks": starting_ranks,
        "ending_bond_ranks": bond_ranks(carrier),
        "ending_carrier_cells": carrier_cells(carrier),
        "preclosure_inverse_error": preclosure_error,
        "postclosure_restoration_error": postclosure_error,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "actual_carrier_canonical_closure": True,
        "retained_inverse_history_bytes": 0,
        "closure": closure,
        "stats": {
            "maximum_probe_rank": stats.maximum_probe_rank,
            "frobenius_probe_columns": stats.frobenius_probe_columns,
            "matmat_applications": stats.matmat_applications,
            "rmatmat_applications": stats.rmatmat_applications,
            "maximum_carrier_cells": stats.maximum_carrier_cells,
            "maximum_workspace_cells": stats.maximum_workspace_cells,
            "maximum_workspace_array_cells": (
                stats.maximum_workspace_array_cells
            ),
            "maximum_total_live_cells": stats.maximum_total_live_cells,
            "maximum_total_live_payload_bytes": (
                stats.maximum_total_live_cells * 16
            ),
            "maximum_retained_rank": stats.maximum_retained_rank,
        },
    }


def resource_signature(transaction: dict[str, object]) -> dict[str, object]:
    stats = transaction["stats"]
    assert isinstance(stats, dict)
    return {
        key: stats[key]
        for key in (
            "maximum_probe_rank",
            "frobenius_probe_columns",
            "matmat_applications",
            "rmatmat_applications",
            "maximum_carrier_cells",
            "maximum_workspace_cells",
            "maximum_workspace_array_cells",
            "maximum_total_live_cells",
            "maximum_total_live_payload_bytes",
            "maximum_retained_rank",
        )
    }


def main() -> None:
    initial = reference.product_zero_carrier(reference.MODE_RADIUS)
    actual = reference.product_zero_carrier(reference.MODE_RADIUS)
    primary = reference.transaction(
        actual,
        reference.PRIMARY_DEPTH,
        reference.PRIMARY_K,
        reference.PRIMARY_G,
        reference.PRIMARY_TAU,
    )
    missing_closure_ranks = bond_ranks(actual)
    preclosure = reference.copy_carrier(actual)
    strict_closure = canonical_round_actual(
        actual, STRICT_POST_INVERSE_L2
    )
    strict_closure_delta = reference.physical_distance(
        preclosure, actual
    )
    strict_postclosure_error = reference.physical_distance(initial, actual)
    if (
        missing_closure_ranks == [1, 1, 1]
        or bond_ranks(actual) != [1, 1, 1]
        or carrier_cells(actual) != carrier_cells(initial)
        or strict_postclosure_error
        > primary["restoration_error"] + STRICT_POST_INVERSE_L2
    ):
        fail("strict actual-carrier canonical closure gate failed")

    fresh = reference.product_zero_carrier(reference.MODE_RADIUS)
    actual_reuse = matrix_free_transaction_with_closure(
        actual, REUSE_DEPTH, REUSE_K, REUSE_G, REUSE_TAU
    )
    fresh_reuse = matrix_free_transaction_with_closure(
        fresh, REUSE_DEPTH, REUSE_K, REUSE_G, REUSE_TAU
    )
    boundary_error = reference.boundary_distance(
        actual_reuse["boundary"], fresh_reuse["boundary"]
    )
    actual_signature = resource_signature(actual_reuse)
    fresh_signature = resource_signature(fresh_reuse)
    if (
        actual_reuse["central_rank_history"]
        != fresh_reuse["central_rank_history"]
        or actual_signature != fresh_signature
        or boundary_error > REUSE_BOUNDARY_PARITY
        or actual_reuse["ending_bond_ranks"] != [1, 1, 1]
        or fresh_reuse["ending_bond_ranks"] != [1, 1, 1]
        or actual_reuse["ending_carrier_cells"] != carrier_cells(initial)
        or actual_reuse["restoration_generation"] != 2
    ):
        fail("fresh/restored canonical reuse parity gate failed")

    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_ACTUAL_POST_INVERSE_TT_CANONICAL_QUOTIENT_"
            "CLOSURE_WITH_FRESH_RESTORED_MATRIX_FREE_REUSE_"
            "RANK_AND_RESOURCE_PARITY"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTORS_MODE_RADIUS14_STRICT_PRIMARY_"
            "DEPTH3_POST_INVERSE_L2_1E_MINUS7_MATRIX_FREE_REUSE_"
            "DEPTH2_L2_1E_MINUS6_SOFTWARE_FLOAT64"
        ),
        "predeclared_tolerances": {
            "strict_forward_svd_l2": reference.SVD_L2_TOLERANCE,
            "strict_post_inverse_canonical_l2": STRICT_POST_INVERSE_L2,
            "matrix_free_reuse_l2": (
                matrix_free.MATRIX_FREE_L2_TOLERANCE
            ),
            "matrix_free_post_inverse_canonical_l2": (
                MATRIX_FREE_POST_INVERSE_L2
            ),
            "fresh_restored_boundary_parity": REUSE_BOUNDARY_PARITY,
        },
        "primary_actual_inverse": {
            "restoration_error_before_closure": primary[
                "restoration_error"
            ],
            "restoration_generation": primary[
                "restoration_generation"
            ],
            "missing_closure_bond_ranks": missing_closure_ranks,
            "missing_closure_carrier_cells": carrier_cells(preclosure),
            "postclosure_restoration_error": strict_postclosure_error,
            "closure_physical_delta": strict_closure_delta,
            "closure": strict_closure,
        },
        "actual_restored_reuse": actual_reuse,
        "fresh_reuse_baseline": fresh_reuse,
        "fresh_restored_boundary_error": boundary_error,
        "fresh_restored_resource_signature_exact": True,
        "retained_numpy_backing_allocations_counted": True,
        "fresh_restored_resource_signature": actual_signature,
        "actual_carrier_restored_and_reused": True,
        "actual_carrier_generation": actual.generation,
        "snapshot_loaded_on_accepted_path": False,
        "retained_inverse_history_bytes": 0,
        "canonical_rank_accumulation_across_reuse": False,
        "dense_wave_materialized": False,
        "matched_classical_tt_rounding_is_identical": True,
        "fixed_rank_forward_closure": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
        "machine_enforced_hidden_intermediate": False,
        "physical_waveform_execution": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(2) from error
