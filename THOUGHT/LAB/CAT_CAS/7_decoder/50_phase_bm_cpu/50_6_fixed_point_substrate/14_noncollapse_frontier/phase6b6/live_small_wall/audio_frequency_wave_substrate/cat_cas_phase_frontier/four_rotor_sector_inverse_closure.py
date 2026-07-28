#!/usr/bin/env python3
"""Topology-derived total-momentum-sector inverse for four phase rotors."""

from __future__ import annotations

import json
import math
import resource
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.linalg
from scipy.linalg.blas import zherk
from scipy.special import jv

import four_rotor_incremental_schmidt_closure as incremental
import four_rotor_kicked_phase_tt as reference
import four_rotor_kicked_phase_tt_matrix_free as matrix_free
import four_rotor_post_inverse_canonical_closure as canonical


SECTOR_INVERSE_L2_TOLERANCE = 1.0e-6
RESTORATION_L2_TOLERANCE = 5.0e-5
BOUNDARY_PARITY_TOLERANCE = 1.0e-5
SECTOR_INVERSE_RESIDUAL_TOLERANCE = 1.0e-12
SECTOR_CONDITION_CEILING = 2.0


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class Sector:
    total_index: int
    first_indices: np.ndarray
    second_indices: np.ndarray
    lu: np.ndarray
    pivots: np.ndarray
    condition: float
    inverse_residual: float


@dataclass(frozen=True)
class SectorPlan:
    dimension: int
    strength: float
    sectors: tuple[Sector, ...]
    complex_cells: int
    pivot_cells: int
    maximum_condition: float
    maximum_inverse_residual: float


@dataclass
class Stats:
    inverse_sector_closures: int = 0
    sector_rhs_rematerializations: int = 0
    gram_rematerializations: int = 0
    maximum_carrier_cells: int = 0
    maximum_workspace_cells: int = 0
    maximum_workspace_array_cells: int = 0
    maximum_total_live_cells: int = 0
    maximum_context: str = ""
    maximum_retained_rank: int = 1
    discarded_l2_bound: float = 0.0
    public_plan_complex_cells: int = 0
    public_plan_pivot_cells: int = 0
    maximum_sector_condition: float = 0.0
    maximum_sector_inverse_residual: float = 0.0


def process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def account(
    carrier: reference.Carrier,
    stats: Stats,
    *arrays: np.ndarray,
    extra_cells: int = 0,
    context: str,
) -> None:
    allocations: dict[int, np.ndarray] = {}
    for array in arrays:
        allocation = array
        while isinstance(allocation.base, np.ndarray):
            allocation = allocation.base
        allocations[id(allocation)] = allocation
    carrier_cells = matrix_free.carrier_backing_cells(carrier)
    workspace = (
        sum(int(array.size) for array in allocations.values())
        + stats.public_plan_complex_cells
        + (stats.public_plan_pivot_cells + 1) // 2
        + extra_cells
    )
    stats.maximum_carrier_cells = max(
        stats.maximum_carrier_cells, carrier_cells
    )
    stats.maximum_workspace_cells = max(
        stats.maximum_workspace_cells, workspace
    )
    stats.maximum_workspace_array_cells = max(
        [stats.maximum_workspace_array_cells]
        + [int(array.size) for array in arrays]
    )
    total = carrier_cells + workspace
    if total > stats.maximum_total_live_cells:
        stats.maximum_total_live_cells = total
        stats.maximum_context = context


@lru_cache(maxsize=8)
def sector_plan(dimension: int, strength: float) -> SectorPlan:
    sectors: list[Sector] = []
    complex_cells = 0
    pivot_cells = 0
    maximum_condition = 0.0
    maximum_residual = 0.0
    for total_index in range(2 * dimension - 1):
        start = max(0, total_index - (dimension - 1))
        stop = min(dimension - 1, total_index)
        first = np.arange(start, stop + 1, dtype=np.int64)
        second = total_index - first
        shifts = first[:, None] - first[None, :]
        kernel = np.zeros(
            shifts.shape, dtype=np.complex128, order="F"
        )
        admitted = np.abs(shifts) <= reference.COUPLING_KERNEL_RADIUS
        kernel[admitted] = (
            np.power(-1j, shifts[admitted])
            * jv(shifts[admitted], strength)
        )
        condition = float(np.linalg.cond(kernel))
        inverse = scipy.linalg.inv(
            kernel, overwrite_a=False, check_finite=False
        )
        residual = float(
            np.linalg.norm(inverse @ kernel - np.eye(len(first)))
        )
        lu, pivots = scipy.linalg.lu_factor(
            kernel, overwrite_a=True, check_finite=False
        )
        if (
            not math.isfinite(condition)
            or condition > SECTOR_CONDITION_CEILING
            or residual > SECTOR_INVERSE_RESIDUAL_TOLERANCE
        ):
            fail("public sector inverse plan qualification failed")
        sectors.append(
            Sector(
                total_index,
                first,
                second,
                lu,
                pivots,
                condition,
                residual,
            )
        )
        complex_cells += int(lu.size)
        pivot_cells += int(pivots.size)
        maximum_condition = max(maximum_condition, condition)
        maximum_residual = max(maximum_residual, residual)
    return SectorPlan(
        dimension,
        strength,
        tuple(sectors),
        complex_cells,
        pivot_cells,
        maximum_condition,
        maximum_residual,
    )


class SectorInverse:
    def __init__(
        self,
        carrier: reference.Carrier,
        edge: int,
        strength: float,
        stats: Stats,
    ) -> None:
        self.carrier = carrier
        self.edge = edge
        self.left_tensor = carrier.tensors[edge]
        self.right_tensor = carrier.tensors[edge + 1]
        self.left, self.dimension, self.bond = self.left_tensor.shape
        _, _, self.right = self.right_tensor.shape
        self.rows = self.left * self.dimension
        self.columns = self.dimension * self.right
        self.plan = sector_plan(self.dimension, strength)
        stats.public_plan_complex_cells = max(
            stats.public_plan_complex_cells,
            self.plan.complex_cells,
        )
        stats.public_plan_pivot_cells = max(
            stats.public_plan_pivot_cells,
            self.plan.pivot_cells,
        )
        stats.maximum_sector_condition = max(
            stats.maximum_sector_condition,
            self.plan.maximum_condition,
        )
        stats.maximum_sector_inverse_residual = max(
            stats.maximum_sector_inverse_residual,
            self.plan.maximum_inverse_residual,
        )
        self.stats = stats

    def row_block(self, first_output: int) -> np.ndarray:
        block = np.zeros(
            (self.left, self.columns),
            dtype=np.complex128,
            order="F",
        )
        for second_output in range(self.dimension):
            total_index = first_output + second_output
            sector = self.plan.sectors[total_index]
            width = len(sector.first_indices)
            rhs = np.empty(
                (width, self.left * self.right),
                dtype=np.complex128,
                order="F",
            )
            for index, (first, second) in enumerate(
                zip(
                    sector.first_indices,
                    sector.second_indices,
                    strict=True,
                )
            ):
                contraction = (
                    self.left_tensor[:, first, :]
                    @ self.right_tensor[:, second, :]
                )
                rhs[index] = contraction.reshape(-1)
            solved = scipy.linalg.lu_solve(
                (sector.lu, sector.pivots),
                rhs,
                overwrite_b=True,
                check_finite=False,
            )
            position = int(first_output - sector.first_indices[0])
            block[
                :,
                second_output * self.right : (
                    second_output + 1
                )
                * self.right,
            ] = solved[position].reshape(self.left, self.right)
            self.stats.sector_rhs_rematerializations += 1
            account(
                self.carrier,
                self.stats,
                block,
                rhs,
                solved,
                context=f"edge{self.edge}_sector_rhs",
            )
        return block

    def right_gram(self) -> np.ndarray:
        gram = np.zeros(
            (self.columns, self.columns),
            dtype=np.complex128,
            order="F",
        )
        for first in range(self.dimension):
            block = self.row_block(first)
            updated = zherk(
                1.0,
                block,
                beta=1.0,
                c=gram,
                trans=2,
                lower=0,
                overwrite_c=1,
            )
            if updated is not gram:
                fail("sector right Gram lost owned output buffer")
            account(
                self.carrier,
                self.stats,
                gram,
                block,
                context=f"edge{self.edge}_right_gram",
            )
        for column in range(self.columns):
            gram[column + 1 :, column] = gram[
                column, column + 1 :
            ].conj()
        self.stats.gram_rematerializations += 1
        return gram

    def left_gram(self) -> np.ndarray:
        gram = np.zeros(
            (self.rows, self.rows),
            dtype=np.complex128,
            order="F",
        )
        for first in range(self.dimension):
            left_block = self.row_block(first)
            left_rows = np.arange(self.left) * self.dimension + first
            for second_first in range(first, self.dimension):
                right_block = self.row_block(second_first)
                right_rows = (
                    np.arange(self.left) * self.dimension
                    + second_first
                )
                right_adjoint = np.asfortranarray(
                    right_block.conj().T
                )
                product = np.empty(
                    (self.left, self.left),
                    dtype=np.complex128,
                    order="F",
                )
                np.matmul(left_block, right_adjoint, out=product)
                gram[np.ix_(left_rows, right_rows)] = product
                if second_first != first:
                    gram[np.ix_(right_rows, left_rows)] = product.conj().T
                account(
                    self.carrier,
                    self.stats,
                    gram,
                    left_block,
                    right_block,
                    right_adjoint,
                    product,
                    context=f"edge{self.edge}_left_gram",
                )
        self.stats.gram_rematerializations += 1
        return gram


def retained_rank(eigenvalues: np.ndarray, budget: float) -> int:
    clipped = np.maximum(eigenvalues.real, 0.0)
    for keep in range(1, len(clipped) + 1):
        if float(np.sum(clipped[:-keep])) <= budget * budget:
            return keep
    return len(clipped)


def sector_inverse_compress(
    carrier: reference.Carrier,
    edge: int,
    strength: float,
    stats: Stats,
) -> list[float]:
    gauge_stats = reference.Stats()
    reference.left_canonicalize(carrier.tensors, edge, gauge_stats)
    reference.right_canonicalize(
        carrier.tensors, edge + 1, gauge_stats
    )
    operator = SectorInverse(carrier, edge, strength, stats)
    use_right = operator.columns <= operator.rows
    gram = (
        operator.right_gram() if use_right else operator.left_gram()
    )
    gram_size = gram.shape[0]
    eig_workspace_upper = 4 * gram.size + 8 * gram_size
    account(
        carrier,
        stats,
        gram,
        extra_cells=eig_workspace_upper,
        context=f"edge{edge}_eigvalsh",
    )
    eigenvalues = scipy.linalg.eigvalsh(
        gram,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    keep = retained_rank(
        eigenvalues, SECTOR_INVERSE_L2_TOLERANCE
    )
    clipped = np.maximum(eigenvalues.real, 0.0)
    discarded = math.sqrt(float(np.sum(clipped[:-keep])))
    del gram

    gram = (
        operator.right_gram() if use_right else operator.left_gram()
    )
    account(
        carrier,
        stats,
        gram,
        eigenvalues,
        extra_cells=eig_workspace_upper,
        context=f"edge{edge}_eigh",
    )
    selected_values, selected_vectors = scipy.linalg.eigh(
        gram,
        subset_by_index=(gram_size - keep, gram_size - 1),
        overwrite_a=True,
        check_finite=False,
        driver="evr",
    )
    order = np.arange(keep - 1, -1, -1)
    singular = np.sqrt(np.maximum(selected_values[order], 0.0))
    vectors = np.asfortranarray(selected_vectors[:, order])
    del gram, selected_values, selected_vectors

    if use_right:
        new_left = np.empty(
            (operator.left, operator.dimension, keep),
            dtype=np.complex128,
            order="C",
        )
        for first in range(operator.dimension):
            block = operator.row_block(first)
            projected = np.empty(
                (operator.left, keep),
                dtype=np.complex128,
                order="F",
            )
            np.matmul(block, vectors, out=projected)
            new_left[:, first, :] = projected
            account(
                carrier,
                stats,
                eigenvalues,
                singular,
                vectors,
                new_left,
                block,
                projected,
                context=f"edge{edge}_right_reconstruct",
            )
        new_right = np.ascontiguousarray(
            vectors.conj().T.reshape(
                keep, operator.dimension, operator.right
            )
        )
    else:
        left_vectors = vectors
        new_left = np.ascontiguousarray(
            (left_vectors * singular[np.newaxis, :]).reshape(
                operator.left, operator.dimension, keep
            )
        )
        right_matrix = np.zeros(
            (keep, operator.columns),
            dtype=np.complex128,
            order="F",
        )
        for first in range(operator.dimension):
            block = operator.row_block(first)
            rows = np.arange(operator.left) * operator.dimension + first
            left_adjoint = np.asfortranarray(
                left_vectors[rows].conj().T
            )
            contribution = np.empty_like(right_matrix)
            np.matmul(left_adjoint, block, out=contribution)
            np.add(right_matrix, contribution, out=right_matrix)
            account(
                carrier,
                stats,
                eigenvalues,
                singular,
                left_vectors,
                new_left,
                right_matrix,
                block,
                left_adjoint,
                contribution,
                context=f"edge{edge}_left_reconstruct",
            )
        if float(np.min(singular)) <= 0.0:
            fail("sector inverse retained a zero singular value")
        right_matrix /= singular[:, np.newaxis]
        new_right = np.ascontiguousarray(
            right_matrix.reshape(
                keep, operator.dimension, operator.right
            )
        )

    account(
        carrier,
        stats,
        eigenvalues,
        singular,
        vectors,
        new_left,
        new_right,
        extra_cells=eig_workspace_upper,
        context=f"edge{edge}_install",
    )
    carrier.tensors[edge] = new_left
    carrier.tensors[edge + 1] = new_right
    stats.inverse_sector_closures += 1
    stats.maximum_retained_rank = max(
        stats.maximum_retained_rank, keep
    )
    stats.discarded_l2_bound += discarded
    account(carrier, stats, context=f"edge{edge}_installed")
    norm = float(np.linalg.norm(singular))
    return [float(value / norm) for value in singular]


def inverse_round(
    carrier: reference.Carrier,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
    stats: Stats,
) -> None:
    local_stats = reference.Stats()
    reference.apply_free(
        carrier,
        reference.free_phase(carrier.mode_radius, free_time).conj(),
        local_stats,
    )
    for edge in (2, 1, 0):
        sector_inverse_compress(
            carrier, edge, coupling_strength, stats
        )
    reference.apply_local(
        carrier,
        reference.local_kick(carrier.mode_radius, -kick_strength),
        local_stats,
    )
    account(carrier, stats, context="inverse_round")


def transaction(
    carrier: reference.Carrier,
    depth: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    initial = reference.copy_carrier(carrier)
    forward_stats = incremental.Stats()
    inverse_stats = Stats()
    central: list[float] = [1.0]
    ranks: list[int] = []
    start = time.perf_counter_ns()
    for _ in range(depth):
        central = incremental.forward_round(
            carrier,
            kick_strength,
            coupling_strength,
            free_time,
            forward_stats,
        )
        ranks.append(len(central))
    latched = reference.boundary(carrier, central)
    for _ in range(depth):
        inverse_round(
            carrier,
            kick_strength,
            coupling_strength,
            free_time,
            inverse_stats,
        )
    inverse_error = reference.physical_distance(initial, carrier)
    if inverse_error > RESTORATION_L2_TOLERANCE:
        fail("sector inverse restoration failed")
    closure = canonical.canonical_round_actual(
        carrier, incremental.INCREMENTAL_L2_TOLERANCE
    )
    restoration_error = reference.physical_distance(initial, carrier)
    if restoration_error > RESTORATION_L2_TOLERANCE:
        fail("sector inverse canonical restoration failed")
    carrier.generation += 1
    elapsed = time.perf_counter_ns() - start
    peak_cells = max(
        forward_stats.maximum_total_live_cells,
        inverse_stats.maximum_total_live_cells,
        int(closure["maximum_simultaneous_cells"]),
    )
    return {
        "boundary": latched,
        "central_rank_history": ranks,
        "inverse_restoration_error": inverse_error,
        "postclosure_restoration_error": restoration_error,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse_ready": True,
        "closure": closure,
        "resources": {
            "forward_incremental_updates": (
                forward_stats.incremental_updates
            ),
            "inverse_sector_closures": (
                inverse_stats.inverse_sector_closures
            ),
            "total_closure_updates": (
                forward_stats.incremental_updates
                + inverse_stats.inverse_sector_closures
            ),
            "sector_rhs_rematerializations": (
                inverse_stats.sector_rhs_rematerializations
            ),
            "gram_rematerializations": (
                inverse_stats.gram_rematerializations
            ),
            "public_plan_complex_cells": (
                inverse_stats.public_plan_complex_cells
            ),
            "public_plan_pivot_cells": (
                inverse_stats.public_plan_pivot_cells
            ),
            "maximum_sector_condition": (
                inverse_stats.maximum_sector_condition
            ),
            "maximum_sector_inverse_residual": (
                inverse_stats.maximum_sector_inverse_residual
            ),
            "maximum_sector_workspace_cells": (
                inverse_stats.maximum_workspace_cells
            ),
            "maximum_sector_workspace_array_cells": (
                inverse_stats.maximum_workspace_array_cells
            ),
            "maximum_context": inverse_stats.maximum_context,
            "maximum_total_live_cells": peak_cells,
            "maximum_total_live_payload_bytes": peak_cells * 16,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "discarded_inverse_l2_bound": (
                inverse_stats.discarded_l2_bound
            ),
            "retained_inverse_history_bytes": 0,
            "probe_columns": 0,
            "elapsed_ns": elapsed,
        },
    }


def controls() -> dict[str, float]:
    initial = reference.product_zero_carrier(8)
    forward = reference.copy_carrier(initial)
    forward_stats = incremental.Stats()
    incremental.forward_round(
        forward,
        reference.PRIMARY_K,
        reference.PRIMARY_G,
        reference.PRIMARY_TAU,
        forward_stats,
    )
    missing = reference.physical_distance(initial, forward)
    wrong = reference.copy_carrier(forward)
    inverse_round(
        wrong,
        reference.PRIMARY_K,
        reference.PRIMARY_G * 1.1,
        reference.PRIMARY_TAU,
        Stats(),
    )
    reordered = reference.copy_carrier(forward)
    local_stats = reference.Stats()
    reference.apply_local(
        reordered,
        reference.local_kick(8, -reference.PRIMARY_K),
        local_stats,
    )
    reordered_stats = Stats()
    for edge in (2, 1, 0):
        sector_inverse_compress(
            reordered,
            edge,
            reference.PRIMARY_G,
            reordered_stats,
        )
    reference.apply_free(
        reordered,
        reference.free_phase(8, reference.PRIMARY_TAU).conj(),
        local_stats,
    )
    return {
        "missing_inverse_error": missing,
        "wrong_inverse_error": reference.physical_distance(
            initial, wrong
        ),
        "reordered_inverse_error": reference.physical_distance(
            initial, reordered
        ),
    }


def resource_signature(
    transaction_result: dict[str, object],
) -> dict[str, object]:
    resources = transaction_result["resources"]
    assert isinstance(resources, dict)
    return {
        key: resources[key]
        for key in (
            "forward_incremental_updates",
            "inverse_sector_closures",
            "total_closure_updates",
            "sector_rhs_rematerializations",
            "gram_rematerializations",
            "public_plan_complex_cells",
            "public_plan_pivot_cells",
            "maximum_sector_workspace_cells",
            "maximum_sector_workspace_array_cells",
            "maximum_total_live_cells",
            "maximum_total_live_payload_bytes",
            "probe_columns",
        )
    }


def main() -> None:
    carrier = reference.product_zero_carrier(reference.MODE_RADIUS)
    primary = transaction(
        carrier,
        reference.PRIMARY_DEPTH,
        reference.PRIMARY_K,
        reference.PRIMARY_G,
        reference.PRIMARY_TAU,
    )
    reuse = transaction(
        carrier, 2, 0.9, 0.22, math.sqrt(5.0)
    )
    fresh = transaction(
        reference.product_zero_carrier(reference.MODE_RADIUS),
        2,
        0.9,
        0.22,
        math.sqrt(5.0),
    )
    reuse_boundary_error = reference.boundary_distance(
        reuse["boundary"], fresh["boundary"]
    )
    reuse_signature = resource_signature(reuse)
    fresh_signature = resource_signature(fresh)
    control = controls()
    incremental_wrapper_peak_bytes = 10_834_016
    sector_wrapper_peak_bytes = (
        primary["resources"]["maximum_total_live_payload_bytes"]
        + 1_856
    )
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_TOPOLOGY_DERIVED_TOTAL_MOMENTUM_SECTOR_"
            "INVERSE_PHASE_CLOSURE_UPDATE_REDUCTION_WITH_GRAM_"
            "REMATERIALIZATION_OBSTRUCTION_ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTORS_MODE_RADIUS14_PRIMARY_DEPTH3_"
            "REUSE_DEPTH2_SECTOR_INVERSE_L2_1E_MINUS6_FLOAT64"
        ),
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh,
        "fresh_restored_boundary_error": reuse_boundary_error,
        "fresh_restored_resource_signature_exact": (
            reuse_signature == fresh_signature
        ),
        "fresh_restored_resource_signature": reuse_signature,
        "controls": control,
        "incremental_reference_total_updates": 234,
        "sector_primary_total_updates": primary["resources"][
            "total_closure_updates"
        ],
        "tradeoff": {
            "incremental_in_place_wrapper_peak_bytes": (
                incremental_wrapper_peak_bytes
            ),
            "sector_in_place_wrapper_peak_bytes": (
                sector_wrapper_peak_bytes
            ),
            "sector_over_incremental_peak_ratio": (
                sector_wrapper_peak_bytes
                / incremental_wrapper_peak_bytes
            ),
            "incremental_inverse_closure_updates": 117,
            "sector_inverse_closure_updates": primary["resources"][
                "inverse_sector_closures"
            ],
            "sector_rhs_rematerializations": primary["resources"][
                "sector_rhs_rematerializations"
            ],
            "closure_update_reduction_established": True,
            "peak_memory_reduction_established": False,
            "warm_time_reduction_established": False,
            "identified_obstruction": (
                "EXACT_GRAM_AND_SECTOR_RHS_REMATERIALIZATION_COST"
            ),
        },
        "matched_classical_sector_algorithm_identical": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    if (
        primary["resources"]["inverse_sector_closures"] != 9
        or primary["resources"]["forward_incremental_updates"] != 117
        or primary["resources"]["total_closure_updates"] != 126
        or primary["closure"]["bond_ranks_after"] != [1, 1, 1]
        or reuse["restoration_generation"] != 2
        or reuse_boundary_error > BOUNDARY_PARITY_TOLERANCE
        or reuse_signature != fresh_signature
        or primary["resources"]["probe_columns"] != 0
        or min(control.values()) <= 1.0e-4
    ):
        fail("sector inverse closure qualification failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(2) from error
