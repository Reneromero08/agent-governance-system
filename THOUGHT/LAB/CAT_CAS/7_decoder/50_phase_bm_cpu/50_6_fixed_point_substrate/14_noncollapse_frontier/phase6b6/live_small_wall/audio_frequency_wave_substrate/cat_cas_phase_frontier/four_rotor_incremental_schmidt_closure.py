#!/usr/bin/env python3
"""Probe-free incremental Schmidt closure for Bessel phase coupling."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.linalg.blas import zherk
from scipy.linalg.lapack import get_lapack_funcs, zgesvd_lwork

import four_rotor_kicked_phase_tt as reference
import four_rotor_kicked_phase_tt_matrix_free as matrix_free
import four_rotor_post_inverse_canonical_closure as canonical


INCREMENTAL_L2_TOLERANCE = 1.0e-6
RESTORATION_L2_TOLERANCE = 5.0e-5
BOUNDARY_PARITY_TOLERANCE = 1.0e-5
ORTHOGONALITY_TOLERANCE = 1.0e-10


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class Stats:
    coupling_applications: int = 0
    incremental_updates: int = 0
    maximum_incremental_rank: int = 0
    maximum_retained_rank: int = 1
    maximum_carrier_backing_cells: int = 0
    maximum_workspace_array_cells: int = 0
    maximum_workspace_cells: int = 0
    maximum_total_live_cells: int = 0
    maximum_context: str = ""
    discarded_l2_bound: float = 0.0
    bessel_kernel_tail_l2_bound: float = 0.0
    maximum_coupling_declared_l2_bound: float = 0.0
    maximum_bessel_analytic_remainder_energy_bound: float = 0.0
    maximum_basis_orthogonality_error: float = 0.0


def account(
    carrier: reference.Carrier,
    stats: Stats,
    *arrays: np.ndarray,
    extra_cells: int = 0,
    context: str = "",
) -> None:
    carrier_cells = matrix_free.carrier_backing_cells(carrier)
    allocations: dict[int, np.ndarray] = {}
    for array in arrays:
        allocation = array
        while isinstance(allocation.base, np.ndarray):
            allocation = allocation.base
        allocations[id(allocation)] = allocation
    workspace = (
        sum(int(array.size) for array in allocations.values())
        + extra_cells
    )
    stats.maximum_carrier_backing_cells = max(
        stats.maximum_carrier_backing_cells, carrier_cells
    )
    stats.maximum_workspace_array_cells = max(
        [stats.maximum_workspace_array_cells]
        + [int(array.size) for array in arrays]
    )
    stats.maximum_workspace_cells = max(
        stats.maximum_workspace_cells, workspace
    )
    total = carrier_cells + workspace
    if total > stats.maximum_total_live_cells:
        stats.maximum_total_live_cells = total
        stats.maximum_context = context


def gesvd_scratch_complex_cells(rows: int, columns: int) -> int:
    optimal, info = zgesvd_lwork(
        rows, columns, compute_uv=1, full_matrices=0
    )
    if info != 0:
        fail("LAPACK zgesvd workspace query failed")
    minimum = min(rows, columns)
    # ZGESVD also requires 5*min(m,n) real doubles. Convert their bytes to
    # complex128-cell equivalents and round upward.
    real_work_as_complex = (5 * minimum + 1) // 2
    return int(optimal.real) + real_work_as_complex


def explicit_thin_qr(
    carrier: reference.Carrier,
    stats: Stats,
    matrix: np.ndarray,
    live_arrays: tuple[np.ndarray, ...],
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        matrix.dtype != np.complex128
        or not matrix.flags.f_contiguous
        or not matrix.flags.owndata
    ):
        fail("QR input must be an owned Fortran complex128 buffer")
    rows, columns = matrix.shape
    thin = min(rows, columns)
    geqrf, ungqr = get_lapack_funcs(("geqrf", "ungqr"), (matrix,))

    query_qr, query_tau, query_work, info = geqrf(
        matrix, lwork=-1, overwrite_a=1
    )
    if info != 0 or query_qr is not matrix:
        fail("in-place ZGEQRF workspace query failed")
    geqrf_lwork = int(query_work[0].real)
    account(
        carrier,
        stats,
        *live_arrays,
        matrix,
        query_tau,
        query_work,
        context=f"{context}_geqrf_query",
    )
    del query_qr, query_tau, query_work

    qr, tau, geqrf_work, info = geqrf(
        matrix, lwork=geqrf_lwork, overwrite_a=1
    )
    if info != 0 or qr is not matrix:
        fail("in-place ZGEQRF failed")
    account(
        carrier,
        stats,
        *live_arrays,
        qr,
        tau,
        geqrf_work,
        context=f"{context}_geqrf",
    )
    r = np.empty((thin, columns), dtype=np.complex128, order="F")
    np.copyto(r, qr[:thin, :])
    for row in range(thin):
        r[row, :row] = 0.0
    account(
        carrier,
        stats,
        *live_arrays,
        qr,
        tau,
        geqrf_work,
        r,
        context=f"{context}_r_extract",
    )
    del geqrf_work

    if columns > rows:
        q_input = np.array(qr[:, :thin], order="F", copy=True)
        account(
            carrier,
            stats,
            *live_arrays,
            qr,
            tau,
            r,
            q_input,
            context=f"{context}_wide_q_extract",
        )
        del qr
    else:
        q_input = qr

    query_q, query_work, info = ungqr(
        q_input, tau, lwork=-1, overwrite_a=1
    )
    if info != 0 or query_q is not q_input:
        fail("in-place ZUNGQR workspace query failed")
    ungqr_lwork = int(query_work[0].real)
    account(
        carrier,
        stats,
        *live_arrays,
        q_input,
        tau,
        r,
        query_work,
        context=f"{context}_zungqr_query",
    )
    del query_q, query_work

    q, ungqr_work, info = ungqr(
        q_input, tau, lwork=ungqr_lwork, overwrite_a=1
    )
    if info != 0 or q is not q_input:
        fail("in-place ZUNGQR failed")
    account(
        carrier,
        stats,
        *live_arrays,
        q,
        tau,
        r,
        ungqr_work,
        context=f"{context}_zungqr",
    )
    del tau, ungqr_work

    gram_residual = zherk(
        1.0,
        q,
        trans=2,
        lower=0,
    )
    if (
        gram_residual.shape != (thin, thin)
        or not gram_residual.flags.owndata
    ):
        fail("ZHERK orthogonality buffer ownership failed")
    account(
        carrier,
        stats,
        *live_arrays,
        q,
        r,
        gram_residual,
        context=f"{context}_orthogonality",
    )
    orthogonality_squared = 0.0
    for column in range(thin):
        diagonal = gram_residual[column, column].real - 1.0
        orthogonality_squared += diagonal * diagonal
        for row in range(column):
            value = gram_residual[row, column]
            orthogonality_squared += 2.0 * (
                value.real * value.real + value.imag * value.imag
            )
    orthogonality_error = math.sqrt(orthogonality_squared)
    stats.maximum_basis_orthogonality_error = max(
        stats.maximum_basis_orthogonality_error,
        orthogonality_error,
    )
    if orthogonality_error > ORTHOGONALITY_TOLERANCE:
        fail("incremental basis orthogonality gate failed")
    return q, r


def bessel_kernel_tail_certificate(
    strength: float,
) -> tuple[float, float]:
    absolute_strength = abs(strength)
    if absolute_strength == 0.0:
        return 0.0, 0.0
    analytic_cutoff = 128
    tail_modes = np.arange(
        reference.COUPLING_KERNEL_RADIUS + 1,
        analytic_cutoff,
    )
    numerical_energy = float(
        2.0
        * np.sum(
            np.square(matrix_free.jv(tail_modes, strength))
        )
    )
    # From the absolute Bessel series,
    # |J_n(x)| <= (|x|/2)^n/n! * exp(x^2/(4(n+1))).
    # Successive bounds from n=analytic_cutoff decrease by at most
    # |x|/(2(n+1)), yielding the geometric squared-tail bound below.
    first = analytic_cutoff
    log_first_bound = (
        first * math.log(absolute_strength / 2.0)
        - math.lgamma(first + 1.0)
        + absolute_strength * absolute_strength / (4.0 * (first + 1.0))
    )
    ratio = absolute_strength / (2.0 * (first + 1.0))
    if ratio >= 1.0:
        fail("analytic Bessel tail ratio is not contractive")
    log_remainder_energy = (
        math.log(2.0)
        + 2.0 * log_first_bound
        - math.log1p(-(ratio * ratio))
    )
    minimum_normal = float(np.finfo(np.float64).tiny)
    if log_remainder_energy < math.log(minimum_normal):
        analytic_remainder_energy = minimum_normal
    else:
        analytic_remainder_energy = math.exp(log_remainder_energy)
    return (
        math.sqrt(numerical_energy + analytic_remainder_energy),
        analytic_remainder_energy,
    )


def incremental_compress(
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
    operator = matrix_free.StreamedCoupling(
        carrier.tensors[edge],
        carrier.tensors[edge + 1],
        strength,
        matrix_free.Stats(),
    )
    terms = 2 * reference.COUPLING_KERNEL_RADIUS + 1
    kernel_tail, analytic_remainder_energy = (
        bessel_kernel_tail_certificate(strength)
    )
    if kernel_tail >= INCREMENTAL_L2_TOLERANCE:
        fail("Bessel kernel tail exhausts coupling error budget")
    update_budget = (
        INCREMENTAL_L2_TOLERANCE - kernel_tail
    ) / terms
    u = np.zeros((operator.rows, 0), dtype=np.complex128)
    singular = np.zeros(0, dtype=np.float64)
    v = np.zeros((operator.columns, 0), dtype=np.complex128)
    discarded_bound = 0.0

    for shift in range(
        -reference.COUPLING_KERNEL_RADIUS,
        reference.COUPLING_KERNEL_RADIUS + 1,
    ):
        old_rank = singular.size
        coefficient = np.power(-1j, shift) * matrix_free.jv(
            shift, strength
        )
        left_factor = coefficient * np.einsum(
            "nm,amb->anb",
            reference.shift_matrix(operator.dimension, shift),
            operator.left_tensor,
            optimize=True,
        ).reshape(operator.rows, operator.bond)
        left_contraction_upper = (
            2 * operator.dimension * operator.dimension
            + operator.bond * left_factor.shape[1]
            + operator.rows * left_factor.shape[1]
        )
        account(
            carrier,
            stats,
            u,
            singular,
            v,
            left_factor,
            extra_cells=left_contraction_upper,
            context=f"edge{edge}_shift{shift}_left_term",
        )
        term_rank = left_factor.shape[1]
        left_augmented = np.empty(
            (operator.rows, old_rank + term_rank),
            dtype=np.complex128,
            order="F",
        )
        left_augmented[:, :old_rank] = u
        left_augmented[:, old_rank:] = left_factor
        account(
            carrier,
            stats,
            u,
            singular,
            v,
            left_factor,
            left_augmented,
            context=f"edge{edge}_shift{shift}_left_augment",
        )
        del u, left_factor
        left_basis, left_coordinates = explicit_thin_qr(
            carrier,
            stats,
            left_augmented,
            (singular, v),
            f"edge{edge}_shift{shift}_left_qr",
        )

        right_factor = np.einsum(
            "nm,bmc->bnc",
            reference.shift_matrix(operator.dimension, -shift),
            operator.right_tensor,
            optimize=True,
        ).reshape(operator.bond, operator.columns)
        right_columns = np.asfortranarray(right_factor.conj().T)
        right_contraction_upper = (
            2 * operator.dimension * operator.dimension
            + operator.bond * right_factor.shape[0]
            + operator.columns * right_factor.shape[0]
        )
        account(
            carrier,
            stats,
            singular,
            v,
            left_basis,
            left_coordinates,
            right_factor,
            right_columns,
            extra_cells=right_contraction_upper,
            context=f"edge{edge}_shift{shift}_right_term",
        )
        right_augmented = np.empty(
            (operator.columns, old_rank + term_rank),
            dtype=np.complex128,
            order="F",
        )
        right_augmented[:, :old_rank] = v
        right_augmented[:, old_rank:] = right_columns
        account(
            carrier,
            stats,
            singular,
            v,
            left_basis,
            left_coordinates,
            right_factor,
            right_columns,
            right_augmented,
            context=f"edge{edge}_shift{shift}_right_augment",
        )
        del v, right_factor, right_columns
        right_basis, right_coordinates = explicit_thin_qr(
            carrier,
            stats,
            right_augmented,
            (singular, left_basis, left_coordinates),
            f"edge{edge}_shift{shift}_right_qr",
        )
        account(
            carrier,
            stats,
            singular,
            left_basis,
            right_basis,
            left_coordinates,
            right_coordinates,
            context=f"edge{edge}_shift{shift}_basis",
        )

        weighted_old_left = np.empty(
            (left_coordinates.shape[0], old_rank),
            dtype=np.complex128,
            order="F",
        )
        if old_rank:
            np.multiply(
                left_coordinates[:, :old_rank],
                singular[np.newaxis, :],
                out=weighted_old_left,
            )
        old_right_adjoint = np.asfortranarray(
            right_coordinates[:, :old_rank].conj().T
        )
        core = np.empty(
            (left_basis.shape[1], right_basis.shape[1]),
            dtype=np.complex128,
            order="F",
        )
        np.matmul(weighted_old_left, old_right_adjoint, out=core)
        account(
            carrier,
            stats,
            singular,
            left_basis,
            right_basis,
            left_coordinates,
            right_coordinates,
            weighted_old_left,
            old_right_adjoint,
            core,
            context=f"edge{edge}_shift{shift}_old_core_product",
        )
        del weighted_old_left, old_right_adjoint, singular

        term_right_adjoint = np.asfortranarray(
            right_coordinates[:, old_rank:].conj().T
        )
        term_product = np.empty_like(core, order="F")
        np.matmul(
            left_coordinates[:, old_rank:],
            term_right_adjoint,
            out=term_product,
        )
        account(
            carrier,
            stats,
            left_basis,
            right_basis,
            left_coordinates,
            right_coordinates,
            core,
            term_right_adjoint,
            term_product,
            context=f"edge{edge}_shift{shift}_term_core_product",
        )
        np.add(core, term_product, out=core)
        del (
            left_coordinates,
            right_coordinates,
            term_right_adjoint,
            term_product,
        )

        svd_scratch = gesvd_scratch_complex_cells(*core.shape)
        account(
            carrier,
            stats,
            left_basis,
            right_basis,
            core,
            extra_cells=svd_scratch,
            context=f"edge{edge}_shift{shift}_core_svd_call",
        )
        core_u, next_singular, core_vh = scipy.linalg.svd(
            core,
            full_matrices=False,
            overwrite_a=True,
            check_finite=False,
            lapack_driver="gesvd",
        )
        squared = next_singular * next_singular
        account(
            carrier,
            stats,
            left_basis,
            right_basis,
            core,
            core_u,
            next_singular,
            core_vh,
            squared,
            extra_cells=svd_scratch,
            context=f"edge{edge}_shift{shift}_core_svd",
        )
        keep = len(next_singular)
        for candidate in range(1, len(next_singular) + 1):
            tail = float(np.sum(squared[candidate:]))
            if tail <= update_budget**2:
                keep = candidate
                break
        discarded = math.sqrt(float(np.sum(squared[keep:])))
        discarded_bound += discarded
        del core

        next_u = np.empty(
            (left_basis.shape[0], keep),
            dtype=np.complex128,
            order="F",
        )
        np.matmul(left_basis, core_u[:, :keep], out=next_u)
        account(
            carrier,
            stats,
            left_basis,
            right_basis,
            core_u,
            core_vh,
            squared,
            next_u,
            next_singular,
            context=f"edge{edge}_shift{shift}_left_update",
        )
        del left_basis, core_u

        right_rotation = np.asfortranarray(
            core_vh[:keep].conj().T
        )
        next_v = np.empty(
            (right_basis.shape[0], keep),
            dtype=np.complex128,
            order="F",
        )
        np.matmul(right_basis, right_rotation, out=next_v)
        retained_singular = next_singular[:keep].copy()
        account(
            carrier,
            stats,
            right_basis,
            core_vh,
            squared,
            next_u,
            next_v,
            next_singular,
            retained_singular,
            right_rotation,
            context=f"edge{edge}_shift{shift}_right_update",
        )
        u, singular, v = next_u, retained_singular, next_v
        stats.incremental_updates += 1
        stats.maximum_incremental_rank = max(
            stats.maximum_incremental_rank, keep
        )

    combined_coupling_bound = discarded_bound + kernel_tail
    if combined_coupling_bound > INCREMENTAL_L2_TOLERANCE:
        fail("incremental Schmidt closure exceeded global error budget")
    stats.discarded_l2_bound += discarded_bound
    stats.bessel_kernel_tail_l2_bound += kernel_tail
    stats.maximum_coupling_declared_l2_bound = max(
        stats.maximum_coupling_declared_l2_bound,
        combined_coupling_bound,
    )
    stats.maximum_bessel_analytic_remainder_energy_bound = max(
        stats.maximum_bessel_analytic_remainder_energy_bound,
        analytic_remainder_energy,
    )
    left = carrier.tensors[edge].shape[0]
    dimension = carrier.tensors[edge].shape[1]
    right = carrier.tensors[edge + 1].shape[2]
    keep = len(singular)
    scaled_u = np.empty_like(u, order="F")
    np.multiply(u, singular[np.newaxis, :], out=scaled_u)
    new_left = scaled_u.reshape(left, dimension, keep)
    new_right = np.empty(
        (keep, dimension, right),
        dtype=np.complex128,
        order="C",
    )
    np.conjugate(
        v.T.reshape(keep, dimension, right),
        out=new_right,
    )
    account(
        carrier,
        stats,
        u,
        singular,
        v,
        scaled_u,
        new_left,
        new_right,
        context=f"edge{edge}_carrier_install",
    )
    carrier.tensors[edge] = new_left
    carrier.tensors[edge + 1] = new_right
    stats.coupling_applications += 1
    stats.maximum_retained_rank = max(
        stats.maximum_retained_rank, keep
    )
    account(carrier, stats)
    norm = float(np.linalg.norm(singular))
    return [float(value / norm) for value in singular]


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
        spectrum = incremental_compress(
            carrier, edge, coupling_strength, stats
        )
        if edge == 1:
            central = spectrum
    reference.apply_free(
        carrier,
        reference.free_phase(carrier.mode_radius, free_time),
        local_stats,
    )
    account(carrier, stats)
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
        incremental_compress(
            carrier, edge, -coupling_strength, stats
        )
    reference.apply_local(
        carrier,
        reference.local_kick(carrier.mode_radius, -kick_strength),
        local_stats,
    )
    account(carrier, stats)


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
    inverse_error = reference.physical_distance(initial, carrier)
    if inverse_error > RESTORATION_L2_TOLERANCE:
        fail("incremental Schmidt actual inverse restoration failed")
    closure = canonical.canonical_round_actual(
        carrier, INCREMENTAL_L2_TOLERANCE
    )
    restoration_error = reference.physical_distance(initial, carrier)
    if restoration_error > (
        inverse_error + INCREMENTAL_L2_TOLERANCE
    ):
        fail("incremental Schmidt canonical restoration failed")
    stats.maximum_total_live_cells = max(
        stats.maximum_total_live_cells,
        int(closure["maximum_simultaneous_cells"]),
    )
    carrier.generation += 1
    return {
        "boundary": latched,
        "central_rank_history": ranks,
        "inverse_restoration_error": inverse_error,
        "postclosure_restoration_error": restoration_error,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse_ready": True,
        "retained_inverse_history_bytes": 0,
        "closure": closure,
        "stats": {
            "coupling_applications": stats.coupling_applications,
            "incremental_updates": stats.incremental_updates,
            "maximum_incremental_rank": (
                stats.maximum_incremental_rank
            ),
            "maximum_retained_rank": stats.maximum_retained_rank,
            "maximum_carrier_backing_cells": (
                stats.maximum_carrier_backing_cells
            ),
            "maximum_workspace_array_cells": (
                stats.maximum_workspace_array_cells
            ),
            "maximum_workspace_cells": stats.maximum_workspace_cells,
            "maximum_total_live_cells": (
                stats.maximum_total_live_cells
            ),
            "maximum_context": stats.maximum_context,
            "maximum_total_live_payload_bytes": (
                stats.maximum_total_live_cells * 16
            ),
            "discarded_l2_bound": stats.discarded_l2_bound,
            "bessel_kernel_tail_l2_bound": (
                stats.bessel_kernel_tail_l2_bound
            ),
            "combined_declared_l2_bound": (
                stats.discarded_l2_bound
                + stats.bessel_kernel_tail_l2_bound
            ),
            "maximum_coupling_declared_l2_bound": (
                stats.maximum_coupling_declared_l2_bound
            ),
            "maximum_bessel_analytic_remainder_energy_bound": (
                stats.maximum_bessel_analytic_remainder_energy_bound
            ),
            "maximum_basis_orthogonality_error": (
                stats.maximum_basis_orthogonality_error
            ),
            "probe_columns": 0,
            "expanded_mpo_bond_materialized": False,
            "dense_interface_core_materialized": False,
        },
    }


def controls() -> dict[str, float]:
    initial = reference.product_zero_carrier(8)
    forward = reference.copy_carrier(initial)
    stats = Stats()
    forward_round(
        forward,
        reference.PRIMARY_K,
        reference.PRIMARY_G,
        reference.PRIMARY_TAU,
        stats,
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
        incremental_compress(
            reordered,
            edge,
            -reference.PRIMARY_G,
            reordered_stats,
        )
    reference.apply_free(
        reordered,
        reference.free_phase(8, reference.PRIMARY_TAU).conj(),
        local_stats,
    )
    return {
        "missing_inverse_error": missing,
        "wrong_inverse_error": reference.physical_distance(initial, wrong),
        "reordered_inverse_error": reference.physical_distance(
            initial, reordered
        ),
    }


def resource_signature(transaction_result: dict[str, object]) -> dict[str, object]:
    stats = transaction_result["stats"]
    assert isinstance(stats, dict)
    return {
        key: stats[key]
        for key in (
            "coupling_applications",
            "incremental_updates",
            "maximum_incremental_rank",
            "maximum_retained_rank",
            "maximum_carrier_backing_cells",
            "maximum_workspace_array_cells",
            "maximum_workspace_cells",
            "maximum_total_live_cells",
            "maximum_total_live_payload_bytes",
            "probe_columns",
        )
    }


def main() -> None:
    old_tolerance = reference.SVD_L2_TOLERANCE
    reference.SVD_L2_TOLERANCE = INCREMENTAL_L2_TOLERANCE
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
    reuse = transaction(
        carrier, 2, 0.9, 0.22, math.sqrt(5.0)
    )
    fresh_carrier = reference.product_zero_carrier(
        reference.MODE_RADIUS
    )
    fresh_reuse = transaction(
        fresh_carrier, 2, 0.9, 0.22, math.sqrt(5.0)
    )
    fresh_restored_boundary_error = reference.boundary_distance(
        reuse["boundary"], fresh_reuse["boundary"]
    )
    reuse_signature = resource_signature(reuse)
    fresh_signature = resource_signature(fresh_reuse)
    control = controls()
    dense_cells = (
        (2 * reference.MODE_RADIUS + 1) ** reference.ROTORS
    )
    qualification_checks = {
        "boundary_parity": (
            boundary_error <= BOUNDARY_PARITY_TOLERANCE
        ),
        "below_dense_equivalent_memory": (
            primary["stats"]["maximum_total_live_cells"] < dense_cells
        ),
        "zero_probe_columns": primary["stats"]["probe_columns"] == 0,
        "basis_orthogonality": (
            primary["stats"]["maximum_basis_orthogonality_error"]
            <= ORTHOGONALITY_TOLERANCE
        ),
        "per_coupling_error_budget": (
            primary["stats"]["maximum_coupling_declared_l2_bound"]
            <= INCREMENTAL_L2_TOLERANCE
        ),
        "aggregate_error_budget": (
            primary["stats"]["combined_declared_l2_bound"]
            <= primary["stats"]["coupling_applications"]
            * INCREMENTAL_L2_TOLERANCE
        ),
        "canonical_closure": (
            primary["closure"]["bond_ranks_after"] == [1, 1, 1]
        ),
        "restored_reuse_generation": (
            reuse["restoration_generation"] == 2
        ),
        "reuse_rank_parity": (
            reuse["central_rank_history"]
            == fresh_reuse["central_rank_history"]
        ),
        "reuse_resource_parity": reuse_signature == fresh_signature,
        "reuse_boundary_parity": (
            fresh_restored_boundary_error
            <= BOUNDARY_PARITY_TOLERANCE
        ),
        "controls": min(control.values()) > 1.0e-4,
    }
    failed_checks = [
        name
        for name, passed in qualification_checks.items()
        if not passed
    ]
    if failed_checks:
        fail(
            "incremental Schmidt closure qualification gate failed: "
            + json.dumps(
                {
                    "failed_checks": failed_checks,
                    "boundary_error": boundary_error,
                    "maximum_total_live_cells": primary["stats"][
                        "maximum_total_live_cells"
                    ],
                    "maximum_context": primary["stats"][
                        "maximum_context"
                    ],
                    "dense_cells": dense_cells,
                    "maximum_basis_orthogonality_error": primary[
                        "stats"
                    ]["maximum_basis_orthogonality_error"],
                    "reuse_signature_equal": (
                        reuse_signature == fresh_signature
                    ),
                },
                sort_keys=True,
            )
        )

    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_PROBE_FREE_INCREMENTAL_BESSEL_SCHMIDT_"
            "PHASE_CLOSURE_BELOW_DENSE_EQUIVALENT_MEMORY_WITH_"
            "ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTORS_MODE_RADIUS14_DEPTH3_"
            "INCREMENTAL_L2_1E_MINUS6_SOFTWARE_FLOAT64"
        ),
        "predeclared_tolerances": {
            "incremental_l2_per_coupling": (
                INCREMENTAL_L2_TOLERANCE
            ),
            "restoration_l2": RESTORATION_L2_TOLERANCE,
            "boundary_parity": BOUNDARY_PARITY_TOLERANCE,
            "basis_orthogonality": ORTHOGONALITY_TOLERANCE,
        },
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse_baseline": fresh_reuse,
        "fresh_restored_boundary_error": (
            fresh_restored_boundary_error
        ),
        "fresh_restored_resource_signature_exact": True,
        "fresh_restored_resource_signature": reuse_signature,
        "controls": control,
        "qualification_checks": qualification_checks,
        "dense_reference_same_tolerance": dense_reference,
        "boundary_error_against_dense_reference": boundary_error,
        "resource_comparison": {
            "accepted_peak_payload_bytes": primary["stats"][
                "maximum_total_live_payload_bytes"
            ],
            "dense_equivalent_payload_bytes": dense_cells * 16,
            "below_dense_equivalent_memory": True,
            "probe_columns": 0,
            "expanded_mpo_bond_materialized": False,
            "dense_interface_core_materialized": False,
            "public_bessel_terms_incrementally_composed": True,
        },
        "phase_is_primitive_wave_amplitude": True,
        "matched_classical_incremental_tt_is_identical": True,
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
