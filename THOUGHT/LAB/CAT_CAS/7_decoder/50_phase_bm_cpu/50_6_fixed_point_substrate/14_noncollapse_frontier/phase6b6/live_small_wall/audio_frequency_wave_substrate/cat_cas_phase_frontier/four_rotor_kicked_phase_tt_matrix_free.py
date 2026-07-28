#!/usr/bin/env python3
"""Matrix-free streamed Schmidt closure for the four-rotor phase TT."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import jv

import four_rotor_kicked_phase_tt as reference


MATRIX_FREE_L2_TOLERANCE = 1.0e-6
MATRIX_FREE_RESTORATION_TOLERANCE = 2.0e-5
BOUNDARY_PARITY_TOLERANCE = 3.0e-6
POWER_ITERATIONS = 3
FROBENIUS_BLOCK = 32


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class Stats:
    coupling_applications: int = 0
    matmat_applications: int = 0
    rmatmat_applications: int = 0
    frobenius_probe_columns: int = 0
    maximum_probe_rank: int = 0
    maximum_carrier_cells: int = 0
    maximum_single_term_cells: int = 0
    maximum_workspace_cells: int = 0
    maximum_workspace_array_cells: int = 0
    maximum_total_live_cells: int = 0
    maximum_retained_rank: int = 1
    discarded_l2_squared: float = 0.0


class StreamedCoupling:
    def __init__(
        self,
        left_tensor: np.ndarray,
        right_tensor: np.ndarray,
        strength: float,
        stats: Stats,
    ) -> None:
        self.left_tensor = left_tensor
        self.right_tensor = right_tensor
        self.strength = strength
        self.stats = stats
        self.left, self.dimension, self.bond = left_tensor.shape
        _, _, self.right = right_tensor.shape
        self.rows = self.left * self.dimension
        self.columns = self.dimension * self.right

    def term(self, shift: int) -> tuple[np.ndarray, np.ndarray]:
        coefficient = np.power(-1j, shift) * jv(
            shift, self.strength
        )
        left = coefficient * np.einsum(
            "nm,amb->anb",
            reference.shift_matrix(self.dimension, shift),
            self.left_tensor,
            optimize=True,
        ).reshape(self.rows, self.bond)
        right = np.einsum(
            "nm,bmc->bnc",
            reference.shift_matrix(self.dimension, -shift),
            self.right_tensor,
            optimize=True,
        ).reshape(self.bond, self.columns)
        self.stats.maximum_single_term_cells = max(
            self.stats.maximum_single_term_cells,
            left.size + right.size,
        )
        return left, right

    def matmat(self, values: np.ndarray) -> np.ndarray:
        if values.shape[0] != self.columns:
            fail("matrix-free phase TT matmat shape invalid")
        result = np.zeros(
            (self.rows, values.shape[1]), dtype=np.complex128
        )
        for shift in range(
            -reference.COUPLING_KERNEL_RADIUS,
            reference.COUPLING_KERNEL_RADIUS + 1,
        ):
            left, right = self.term(shift)
            result += left @ (right @ values)
        self.stats.matmat_applications += 1
        return result

    def rmatmat(self, values: np.ndarray) -> np.ndarray:
        if values.shape[0] != self.rows:
            fail("matrix-free phase TT rmatmat shape invalid")
        result = np.zeros(
            (self.columns, values.shape[1]), dtype=np.complex128
        )
        for shift in range(
            -reference.COUPLING_KERNEL_RADIUS,
            reference.COUPLING_KERNEL_RADIUS + 1,
        ):
            left, right = self.term(shift)
            result += right.conj().T @ (left.conj().T @ values)
        self.stats.rmatmat_applications += 1
        return result

    def frobenius_squared(self) -> float:
        total = 0.0
        for start in range(0, self.columns, FROBENIUS_BLOCK):
            width = min(FROBENIUS_BLOCK, self.columns - start)
            basis = np.zeros(
                (self.columns, width), dtype=np.complex128
            )
            basis[
                np.arange(start, start + width), np.arange(width)
            ] = 1.0
            image = self.matmat(basis)
            total += float(np.vdot(image, image).real)
            self.stats.frobenius_probe_columns += width
        return total


def deterministic_probes(rows: int, columns: int) -> np.ndarray:
    row = np.arange(rows, dtype=np.float64)[:, None]
    column = np.arange(columns, dtype=np.float64)[None, :]
    return np.exp(
        2j * math.pi * (row + 0.5) * (column + 1.0) / rows
    ) / math.sqrt(rows)


def carrier_backing_cells(carrier: reference.Carrier) -> int:
    """Count unique retained NumPy backing allocations, not view sizes."""
    allocations: dict[int, np.ndarray] = {}
    for tensor in carrier.tensors:
        allocation = tensor
        while isinstance(allocation.base, np.ndarray):
            allocation = allocation.base
        allocations[id(allocation)] = allocation
    return sum(int(allocation.size) for allocation in allocations.values())


def update_live_stats(
    carrier: reference.Carrier,
    stats: Stats,
    workspace_cells: int = 0,
    workspace_array_cells: int = 0,
) -> None:
    carrier_cells = carrier_backing_cells(carrier)
    stats.maximum_carrier_cells = max(
        stats.maximum_carrier_cells, carrier_cells
    )
    stats.maximum_workspace_cells = max(
        stats.maximum_workspace_cells, workspace_cells
    )
    stats.maximum_workspace_array_cells = max(
        stats.maximum_workspace_array_cells,
        workspace_array_cells,
    )
    stats.maximum_total_live_cells = max(
        stats.maximum_total_live_cells,
        carrier_cells
        + workspace_cells
        + stats.maximum_single_term_cells,
    )


def matrix_free_compress(
    carrier: reference.Carrier,
    edge: int,
    strength: float,
    stats: Stats,
) -> list[float]:
    reference_stats = reference.Stats()
    reference.left_canonicalize(
        carrier.tensors, edge, reference_stats
    )
    reference.right_canonicalize(
        carrier.tensors, edge + 1, reference_stats
    )
    operator = StreamedCoupling(
        carrier.tensors[edge],
        carrier.tensors[edge + 1],
        strength,
        stats,
    )
    frobenius_squared = operator.frobenius_squared()
    minimum = min(operator.rows, operator.columns)
    probe_ranks = sorted(
        {
            value
            for value in (
                16,
                32,
                64,
                128,
                256,
                512,
                minimum - 1,
                minimum if minimum <= 64 else 0,
            )
            if 0 < value <= minimum
        }
    )
    selected: tuple[
        np.ndarray, np.ndarray, np.ndarray, float, int
    ] | None = None
    final_projection_residual = frobenius_squared
    for probe_rank in probe_ranks:
        omega = deterministic_probes(operator.columns, probe_rank)
        image = operator.matmat(omega)
        iterations = (
            0 if probe_rank == minimum else POWER_ITERATIONS
        )
        for _ in range(iterations):
            image = operator.matmat(operator.rmatmat(image))
        basis, _ = np.linalg.qr(image, mode="reduced")
        projected = operator.rmatmat(basis).conj().T
        captured = float(np.vdot(projected, projected).real)
        projection_residual = max(
            0.0, frobenius_squared - captured
        )
        final_projection_residual = projection_residual
        workspace = (
            omega.size
            + image.size
            + basis.size
            + projected.size
        )
        nested_contraction_upper = (
            operator.bond * probe_rank
            + operator.rows * probe_rank
            + 2 * operator.dimension * operator.dimension
        )
        qr_factor_upper = probe_rank * probe_rank
        conservative_workspace = (
            workspace
            + nested_contraction_upper
            + qr_factor_upper
        )
        update_live_stats(
            carrier,
            stats,
            conservative_workspace,
            max(
                omega.size,
                image.size,
                basis.size,
                projected.size,
            ),
        )
        stats.maximum_probe_rank = max(
            stats.maximum_probe_rank, probe_rank
        )
        if projection_residual <= MATRIX_FREE_L2_TOLERANCE**2:
            u_small, singular, vh = np.linalg.svd(
                projected, full_matrices=False
            )
            left_vectors = basis @ u_small
            svd_and_output_upper = (
                u_small.size
                + singular.size
                + vh.size
                + left_vectors.size
                + projected.size
                + basis.size
                + omega.size
                + image.size
                + qr_factor_upper
                + nested_contraction_upper
            )
            update_live_stats(
                carrier,
                stats,
                svd_and_output_upper,
                max(
                    u_small.size,
                    vh.size,
                    left_vectors.size,
                    projected.size,
                    basis.size,
                    omega.size,
                    image.size,
                ),
            )
            selected = (
                left_vectors,
                singular,
                vh,
                projection_residual,
                svd_and_output_upper,
            )
            break
    if selected is None:
        fail(
            "matrix-free phase TT certified range not found "
            f"rows={operator.rows} columns={operator.columns} "
            f"residual={final_projection_residual:.17g}"
        )
    (
        left_vectors,
        singular,
        right_vectors,
        projection_residual,
        selected_workspace_cells,
    ) = selected
    squared = singular * singular
    keep = len(singular)
    for candidate in range(1, len(singular) + 1):
        discarded = projection_residual + float(
            np.sum(squared[candidate:])
        )
        if discarded <= MATRIX_FREE_L2_TOLERANCE**2:
            keep = candidate
            break
    total_discarded = projection_residual + float(
        np.sum(squared[keep:])
    )
    compact_right = right_vectors[:keep].reshape(
        keep, carrier.tensors[edge + 1].shape[1], -1
    ).copy()
    new_tensor_cells = (
        left_vectors.shape[0] * keep
        + compact_right.size
    )
    update_live_stats(
        carrier,
        stats,
        selected_workspace_cells + squared.size + new_tensor_cells,
        max(
            stats.maximum_workspace_array_cells,
            left_vectors.shape[0] * keep,
            compact_right.size,
            squared.size,
        ),
    )
    stats.discarded_l2_squared += total_discarded
    stats.maximum_retained_rank = max(
        stats.maximum_retained_rank, keep
    )
    left = carrier.tensors[edge].shape[0]
    dimension = carrier.tensors[edge].shape[1]
    right = carrier.tensors[edge + 1].shape[2]
    carrier.tensors[edge] = (
        left_vectors[:, :keep] * singular[:keep]
    ).reshape(left, dimension, keep)
    carrier.tensors[edge + 1] = compact_right
    stats.coupling_applications += 1
    update_live_stats(carrier, stats)
    norm = float(np.linalg.norm(singular))
    return [float(value / norm) for value in singular[:keep]]


def forward_round(
    carrier: reference.Carrier,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
    stats: Stats,
) -> list[float]:
    local_stats = reference.Stats()
    reference.apply_local(
        carrier,
        reference.local_kick(carrier.mode_radius, kick_strength),
        local_stats,
    )
    central: list[float] = [1.0]
    for edge in (0, 1, 2):
        spectrum = matrix_free_compress(
            carrier, edge, coupling_strength, stats
        )
        if edge == 1:
            central = spectrum
    reference.apply_free(
        carrier,
        reference.free_phase(carrier.mode_radius, free_time),
        local_stats,
    )
    update_live_stats(carrier, stats)
    return central


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
        matrix_free_compress(
            carrier, edge, -coupling_strength, stats
        )
    reference.apply_local(
        carrier,
        reference.local_kick(carrier.mode_radius, -kick_strength),
        local_stats,
    )
    update_live_stats(carrier, stats)


def transaction(
    carrier: reference.Carrier,
    depth: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    initial = reference.copy_carrier(carrier)
    stats = Stats()
    ranks: list[int] = []
    central: list[float] = [1.0]
    for _ in range(depth):
        central = forward_round(
            carrier,
            kick_strength,
            coupling_strength,
            free_time,
            stats,
        )
        ranks.append(len(central))
    latched = reference.boundary(carrier, central)
    for _ in range(depth):
        inverse_round(
            carrier,
            kick_strength,
            coupling_strength,
            free_time,
            stats,
        )
    restoration_error = reference.physical_distance(initial, carrier)
    if restoration_error > MATRIX_FREE_RESTORATION_TOLERANCE:
        fail("matrix-free phase TT restoration failed")
    carrier.generation += 1
    return {
        "boundary": latched,
        "central_rank_history": ranks,
        "restoration_error": restoration_error,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "retained_inverse_history_bytes": 0,
        "dense_interface_core_materialized": False,
        "expanded_mpo_bond_materialized": False,
        "stats": {
            "coupling_applications": stats.coupling_applications,
            "matmat_applications": stats.matmat_applications,
            "rmatmat_applications": stats.rmatmat_applications,
            "frobenius_probe_columns": stats.frobenius_probe_columns,
            "maximum_probe_rank": stats.maximum_probe_rank,
            "maximum_carrier_cells": stats.maximum_carrier_cells,
            "maximum_single_term_cells": (
                stats.maximum_single_term_cells
            ),
            "maximum_workspace_cells": stats.maximum_workspace_cells,
            "maximum_workspace_array_cells": (
                stats.maximum_workspace_array_cells
            ),
            "maximum_total_live_cells": (
                stats.maximum_total_live_cells
            ),
            "maximum_total_live_payload_bytes": (
                stats.maximum_total_live_cells * 16
            ),
            "maximum_retained_rank": stats.maximum_retained_rank,
            "discarded_l2_squared": stats.discarded_l2_squared,
            "resource_accounting_kind": (
                "CONSERVATIVE_SIMULTANEOUS_NUMPY_ARRAY_PAYLOAD_"
                "UPPER_BOUND"
            ),
        },
    }


def controls() -> dict[str, object]:
    initial = reference.product_zero_carrier(8)
    forward = reference.copy_carrier(initial)
    forward_round(
        forward,
        reference.PRIMARY_K,
        reference.PRIMARY_G,
        reference.PRIMARY_TAU,
        Stats(),
    )
    missing_error = reference.physical_distance(initial, forward)
    wrong = reference.copy_carrier(forward)
    inverse_round(
        wrong,
        reference.PRIMARY_K,
        reference.PRIMARY_G * 1.1,
        reference.PRIMARY_TAU,
        Stats(),
    )
    wrong_error = reference.physical_distance(initial, wrong)
    reordered = reference.copy_carrier(forward)
    local_stats = reference.Stats()
    reference.apply_local(
        reordered,
        reference.local_kick(8, -reference.PRIMARY_K),
        local_stats,
    )
    reordered_stats = Stats()
    for edge in (2, 1, 0):
        matrix_free_compress(
            reordered, edge, -reference.PRIMARY_G, reordered_stats
        )
    reference.apply_free(
        reordered,
        reference.free_phase(8, reference.PRIMARY_TAU).conj(),
        local_stats,
    )
    reordered_error = reference.physical_distance(initial, reordered)
    return {
        "missing_inverse_error": missing_error,
        "wrong_inverse_error": wrong_error,
        "reordered_inverse_error": reordered_error,
    }


def main() -> None:
    old_tolerance = reference.SVD_L2_TOLERANCE
    reference.SVD_L2_TOLERANCE = MATRIX_FREE_L2_TOLERANCE
    try:
        dense_reference = reference.forward_diagnostic(
            reference.MODE_RADIUS, reference.PRIMARY_DEPTH
        )
    finally:
        reference.SVD_L2_TOLERANCE = old_tolerance

    carrier = reference.product_zero_carrier(reference.MODE_RADIUS)
    primary = transaction(
        carrier,
        reference.PRIMARY_DEPTH,
        reference.PRIMARY_K,
        reference.PRIMARY_G,
        reference.PRIMARY_TAU,
    )
    boundary_error = reference.boundary_distance(
        primary["boundary"], dense_reference["boundary"]
    )
    if boundary_error > BOUNDARY_PARITY_TOLERANCE:
        fail("matrix-free phase TT boundary parity failed")
    reuse = transaction(
        carrier, 2, 0.9, 0.22, math.sqrt(5.0)
    )
    control = controls()
    if not all(value > 1.0e-4 for value in control.values()):
        fail("matrix-free phase TT control separation failed")

    snapshot_image = reference.product_zero_carrier(
        reference.MODE_RADIUS
    )
    snapshot_working = reference.copy_carrier(snapshot_image)
    snapshot_stats = Stats()
    snapshot_central: list[float] = [1.0]
    for _ in range(reference.PRIMARY_DEPTH):
        snapshot_central = forward_round(
            snapshot_working,
            reference.PRIMARY_K,
            reference.PRIMARY_G,
            reference.PRIMARY_TAU,
            snapshot_stats,
        )
    snapshot_boundary = reference.boundary(
        snapshot_working, snapshot_central
    )
    snapshot_working = reference.copy_carrier(snapshot_image)
    snapshot_reload_error = reference.physical_distance(
        snapshot_working, snapshot_image
    )

    dense_core_cells = (
        (2 * reference.MODE_RADIUS + 1) ** reference.ROTORS
    )
    if (
        primary["stats"]["maximum_total_live_payload_bytes"]
        >= 86091488
        or primary["stats"]["maximum_workspace_array_cells"]
        >= dense_core_cells
    ):
        fail("matrix-free phase TT failed resource repair gate")
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_MATRIX_FREE_STREAMED_BESSEL_SCHMIDT_"
            "CLOSURE_WITHOUT_EXPANDED_MPO_OR_DENSE_INTERFACE_CORE_"
            "WITH_ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTORS_MODE_RADIUS14_DEPTH3_"
            "MATRIX_FREE_L2_1E_MINUS6_SOFTWARE_FLOAT64"
        ),
        "predeclared_tolerances": {
            "matrix_free_l2": MATRIX_FREE_L2_TOLERANCE,
            "restoration_l2": MATRIX_FREE_RESTORATION_TOLERANCE,
            "boundary_parity": BOUNDARY_PARITY_TOLERANCE,
        },
        "primary": primary,
        "reuse": reuse,
        "controls": control,
        "snapshot_sham": {
            "boundary": snapshot_boundary,
            "snapshot_loaded": True,
            "actual_inverse_restoration": False,
            "restoration_generation": 0,
            "reload_error": snapshot_reload_error,
            "image_payload_bytes": reference.payload_bytes(
                snapshot_image
            ),
            "execution_load_payload_copy_bytes": (
                reference.payload_bytes(snapshot_image)
            ),
            "restoration_reload_payload_copy_bytes": (
                reference.payload_bytes(snapshot_image)
            ),
        },
        "dense_reference_same_tolerance": dense_reference,
        "boundary_error_against_dense_reference": boundary_error,
        "resource_repair": {
            "prior_strict_path_peak_payload_bytes": 86091488,
            "prior_dense_interface_core_payload_bytes": 11316496,
            "matrix_free_peak_payload_bytes": primary["stats"][
                "maximum_total_live_payload_bytes"
            ],
            "matrix_free_peak_payload_accounting_kind": primary[
                "stats"
            ]["resource_accounting_kind"],
            "matrix_free_peak_below_dense_equivalent": (
                primary["stats"]["maximum_total_live_cells"]
                < dense_core_cells
            ),
            "largest_workspace_array_cells": primary["stats"][
                "maximum_workspace_array_cells"
            ],
            "dense_interface_core_cells": dense_core_cells,
            "expanded_mpo_bond_eliminated": True,
            "dense_interface_core_eliminated": True,
            "public_bessel_terms_rematerialized": True,
            "frobenius_tail_certified": True,
        },
        "matched_classical_tt_is_identical_representation": True,
        "fixed_central_rank_closure": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
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
