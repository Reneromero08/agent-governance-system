#!/usr/bin/env python3
"""Four-rotor nonseparable continuous kicked-phase Fourier TT test."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import jv


ROTORS = 4
PRIMARY_K = math.sqrt(2.0)
PRIMARY_G = 0.35
PRIMARY_TAU = math.sqrt(3.0)
PRIMARY_DEPTH = 3
MODE_RADIUS = 14
COUPLING_KERNEL_RADIUS = 6
SVD_L2_TOLERANCE = 1.0e-11
RESTORATION_L2_TOLERANCE = 1.0e-7
BOUNDARY_CONVERGENCE_TOLERANCE = 2.0e-10
LOCAL_TAIL_ENERGY = 1.0e-10


def fail(message: str) -> None:
    raise RuntimeError(message)


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


@dataclass
class Stats:
    coupling_applications: int = 0
    local_applications: int = 0
    qr_factorizations: int = 0
    svd_factorizations: int = 0
    maximum_resident_cells: int = 0
    maximum_bond_rank: int = 1
    maximum_retained_bond_rank: int = 1
    maximum_interface_core_cells: int = 0
    discarded_l2_squared: float = 0.0


@dataclass
class Carrier:
    tensors: list[np.ndarray]
    mode_radius: int
    generation: int = 0


def product_zero_carrier(mode_radius: int) -> Carrier:
    dimension = 2 * mode_radius + 1
    tensors: list[np.ndarray] = []
    for _ in range(ROTORS):
        tensor = np.zeros((1, dimension, 1), dtype=np.complex128)
        tensor[0, mode_radius, 0] = 1.0
        tensors.append(tensor)
    return Carrier(tensors, mode_radius)


def copy_carrier(carrier: Carrier) -> Carrier:
    return Carrier(
        [tensor.copy() for tensor in carrier.tensors],
        carrier.mode_radius,
        carrier.generation,
    )


def payload_bytes(carrier: Carrier) -> int:
    return sum(tensor.nbytes for tensor in carrier.tensors)


def update_resident(stats: Stats, tensors: list[np.ndarray]) -> None:
    stats.maximum_resident_cells = max(
        stats.maximum_resident_cells,
        sum(tensor.size for tensor in tensors),
    )
    stats.maximum_bond_rank = max(
        stats.maximum_bond_rank,
        *(tensor.shape[2] for tensor in tensors[:-1]),
    )


def modes(mode_radius: int) -> np.ndarray:
    return np.arange(-mode_radius, mode_radius + 1)


def local_kick(mode_radius: int, strength: float) -> np.ndarray:
    values = modes(mode_radius)
    difference = values[:, None] - values[None, :]
    return np.power(-1j, difference) * jv(difference, strength)


def free_phase(mode_radius: int, free_time: float) -> np.ndarray:
    values = modes(mode_radius)
    return np.exp(-0.5j * free_time * values * values)


def shift_matrix(dimension: int, shift: int) -> np.ndarray:
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    source = np.arange(dimension)
    target = source + shift
    valid = (target >= 0) & (target < dimension)
    result[target[valid], source[valid]] = 1.0
    return result


def apply_local(
    carrier: Carrier, matrix: np.ndarray, stats: Stats
) -> None:
    for index, tensor in enumerate(carrier.tensors):
        carrier.tensors[index] = np.einsum(
            "nm,amb->anb", matrix, tensor, optimize=True
        )
        stats.local_applications += 1
    update_resident(stats, carrier.tensors)


def apply_free(
    carrier: Carrier, diagonal: np.ndarray, stats: Stats
) -> None:
    for tensor in carrier.tensors:
        tensor *= diagonal[None, :, None]
        stats.local_applications += 1


def left_canonicalize(
    tensors: list[np.ndarray], stop: int, stats: Stats
) -> None:
    for index in range(stop):
        left, physical, right = tensors[index].shape
        q, r = np.linalg.qr(
            tensors[index].reshape(left * physical, right),
            mode="reduced",
        )
        rank = q.shape[1]
        tensors[index] = q.reshape(left, physical, rank)
        tensors[index + 1] = np.einsum(
            "ab,bnc->anc", r, tensors[index + 1], optimize=True
        )
        stats.qr_factorizations += 1


def right_canonicalize(
    tensors: list[np.ndarray], start: int, stats: Stats
) -> None:
    for index in range(len(tensors) - 1, start, -1):
        left, physical, right = tensors[index].shape
        q, r = np.linalg.qr(
            tensors[index].reshape(left, physical * right).T,
            mode="reduced",
        )
        rank = q.shape[1]
        tensors[index] = q.T.reshape(rank, physical, right)
        tensors[index - 1] = np.einsum(
            "anb,bc->anc",
            tensors[index - 1],
            r.T,
            optimize=True,
        )
        stats.qr_factorizations += 1


def compress_bond(
    tensors: list[np.ndarray], edge: int, stats: Stats
) -> list[float]:
    left_tensor = tensors[edge]
    right_tensor = tensors[edge + 1]
    left, physical, bond = left_tensor.shape
    _, _, right = right_tensor.shape
    q_left, r_left = np.linalg.qr(
        left_tensor.reshape(left * physical, bond), mode="reduced"
    )
    q_right, r_right = np.linalg.qr(
        right_tensor.reshape(bond, physical * right).T,
        mode="reduced",
    )
    core = r_left @ r_right.T
    stats.maximum_interface_core_cells = max(
        stats.maximum_interface_core_cells, core.size
    )
    u, singular, vh = np.linalg.svd(core, full_matrices=False)
    stats.qr_factorizations += 2
    stats.svd_factorizations += 1
    squared = singular * singular
    keep = len(singular)
    for candidate in range(1, len(singular) + 1):
        discarded = float(np.sum(squared[candidate:]))
        if discarded <= SVD_L2_TOLERANCE**2:
            keep = candidate
            break
    stats.discarded_l2_squared += float(np.sum(squared[keep:]))
    tensors[edge] = (
        q_left @ (u[:, :keep] * singular[:keep])
    ).reshape(left, physical, keep)
    tensors[edge + 1] = (
        vh[:keep] @ q_right.T
    ).reshape(keep, physical, right)
    stats.maximum_retained_bond_rank = max(
        stats.maximum_retained_bond_rank, keep
    )
    update_resident(stats, tensors)
    norm = float(np.linalg.norm(singular))
    return [float(value / norm) for value in singular[:keep]]


def apply_coupling(
    carrier: Carrier, edge: int, strength: float, stats: Stats
) -> list[float]:
    tensors = carrier.tensors
    left_canonicalize(tensors, edge, stats)
    right_canonicalize(tensors, edge + 1, stats)
    left_tensor = tensors[edge]
    right_tensor = tensors[edge + 1]
    left, dimension, bond = left_tensor.shape
    _, _, right = right_tensor.shape
    shifts = range(
        -COUPLING_KERNEL_RADIUS, COUPLING_KERNEL_RADIUS + 1
    )
    operator_rank = len(shifts)
    expanded_left = np.zeros(
        (left, dimension, bond * operator_rank),
        dtype=np.complex128,
    )
    expanded_right = np.zeros(
        (bond * operator_rank, dimension, right),
        dtype=np.complex128,
    )
    for operator_index, shift in enumerate(shifts):
        section = slice(
            operator_index * bond, (operator_index + 1) * bond
        )
        coefficient = np.power(-1j, shift) * jv(shift, strength)
        expanded_left[:, :, section] = coefficient * np.einsum(
            "nm,amb->anb",
            shift_matrix(dimension, shift),
            left_tensor,
            optimize=True,
        )
        expanded_right[section, :, :] = np.einsum(
            "nm,bmc->bnc",
            shift_matrix(dimension, -shift),
            right_tensor,
            optimize=True,
        )
    tensors[edge] = expanded_left
    tensors[edge + 1] = expanded_right
    stats.coupling_applications += 1
    update_resident(stats, tensors)
    return compress_bond(tensors, edge, stats)


def inner(left: Carrier, right: Carrier) -> complex:
    environment = np.ones((1, 1), dtype=np.complex128)
    for left_tensor, right_tensor in zip(
        left.tensors, right.tensors, strict=True
    ):
        environment = np.einsum(
            "xy,xna,ynb->ab",
            environment,
            left_tensor.conj(),
            right_tensor,
            optimize=True,
        )
    return complex(environment[0, 0])


def physical_distance(left: Carrier, right: Carrier) -> float:
    left_norm = np.ones((1, 1), dtype=np.clongdouble)
    right_norm = np.ones((1, 1), dtype=np.clongdouble)
    cross = np.ones((1, 1), dtype=np.clongdouble)
    for left_tensor, right_tensor in zip(
        left.tensors, right.tensors, strict=True
    ):
        left_extended = left_tensor.astype(np.clongdouble)
        right_extended = right_tensor.astype(np.clongdouble)
        left_norm = np.einsum(
            "xy,xna,ynb->ab",
            left_norm,
            left_extended.conj(),
            left_extended,
            optimize=True,
        )
        right_norm = np.einsum(
            "xy,xna,ynb->ab",
            right_norm,
            right_extended.conj(),
            right_extended,
            optimize=True,
        )
        cross = np.einsum(
            "xy,xna,ynb->ab",
            cross,
            left_extended.conj(),
            right_extended,
            optimize=True,
        )
    value = (
        left_norm[0, 0].real
        + right_norm[0, 0].real
        - 2.0 * cross[0, 0].real
    )
    return float(np.sqrt(max(np.longdouble(0.0), value)))


def zero_amplitude(carrier: Carrier) -> complex:
    vector = np.ones(1, dtype=np.complex128)
    for tensor in carrier.tensors:
        vector = vector @ tensor[:, carrier.mode_radius, :]
    return complex(vector[0])


def diagonal_expectation(
    carrier: Carrier, operators: dict[int, np.ndarray]
) -> float:
    environment = np.ones((1, 1), dtype=np.complex128)
    for index, tensor in enumerate(carrier.tensors):
        diagonal = operators.get(index)
        if diagonal is None:
            diagonal = np.ones(tensor.shape[1])
        environment = np.einsum(
            "xy,xna,n,ynb->ab",
            environment,
            tensor.conj(),
            diagonal,
            tensor,
            optimize=True,
        )
    return float(environment[0, 0].real)


def site_probabilities(carrier: Carrier, site: int) -> np.ndarray:
    left = np.ones((1, 1), dtype=np.complex128)
    for tensor in carrier.tensors[:site]:
        left = np.einsum(
            "xy,xna,ynb->ab",
            left,
            tensor.conj(),
            tensor,
            optimize=True,
        )
    right = np.ones((1, 1), dtype=np.complex128)
    for tensor in reversed(carrier.tensors[site + 1 :]):
        right = np.einsum(
            "ab,xna,ynb->xy",
            right,
            tensor.conj(),
            tensor,
            optimize=True,
        )
    tensor = carrier.tensors[site]
    probability = np.einsum(
        "xy,xna,ynb,ab->n",
        left,
        tensor.conj(),
        tensor,
        right,
        optimize=True,
    ).real
    return probability / np.sum(probability)


def local_epsilon_radius(carrier: Carrier) -> int:
    values = modes(carrier.mode_radius)
    maximum = 0
    for site in range(ROTORS):
        probability = site_probabilities(carrier, site)
        for radius in range(carrier.mode_radius + 1):
            if (
                float(np.sum(probability[np.abs(values) > radius]))
                <= LOCAL_TAIL_ENERGY
            ):
                maximum = max(maximum, radius)
                break
    return maximum


def boundary(
    carrier: Carrier, central_singular: list[float]
) -> dict[str, object]:
    values = modes(carrier.mode_radius).astype(np.float64)
    norm = inner(carrier, carrier).real
    central_entropy = -sum(
        value * value * math.log(value * value)
        for value in central_singular
        if value > 0.0
    )
    return {
        "zero_product_amplitude": complex_pair(
            zero_amplitude(carrier)
        ),
        "norm": norm,
        "neighbor_momentum_correlation": diagonal_expectation(
            carrier, {1: values, 2: values}
        ),
        "central_schmidt_rank": len(central_singular),
        "central_schmidt_entropy": central_entropy,
        "maximum_local_epsilon_radius": local_epsilon_radius(carrier),
    }


def boundary_distance(
    left: dict[str, object], right: dict[str, object]
) -> float:
    left_amplitude = complex(*left["zero_product_amplitude"])
    right_amplitude = complex(*right["zero_product_amplitude"])
    return max(
        abs(left_amplitude - right_amplitude),
        abs(float(left["norm"]) - float(right["norm"])),
        abs(
            float(left["neighbor_momentum_correlation"])
            - float(right["neighbor_momentum_correlation"])
        ),
    )


def forward_round(
    carrier: Carrier,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
    stats: Stats,
) -> list[float]:
    apply_local(
        carrier, local_kick(carrier.mode_radius, kick_strength), stats
    )
    central: list[float] = [1.0]
    for edge in (0, 1, 2):
        spectrum = apply_coupling(
            carrier, edge, coupling_strength, stats
        )
        if edge == 1:
            central = spectrum
    apply_free(carrier, free_phase(carrier.mode_radius, free_time), stats)
    return central


def inverse_round(
    carrier: Carrier,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
    stats: Stats,
) -> None:
    apply_free(
        carrier,
        free_phase(carrier.mode_radius, free_time).conj(),
        stats,
    )
    for edge in (2, 1, 0):
        apply_coupling(carrier, edge, -coupling_strength, stats)
    apply_local(
        carrier, local_kick(carrier.mode_radius, -kick_strength), stats
    )


def transaction(
    carrier: Carrier,
    depth: int,
    kick_strength: float,
    coupling_strength: float,
    free_time: float,
) -> dict[str, object]:
    initial = copy_carrier(carrier)
    stats = Stats()
    rank_history: list[int] = []
    central_singular: list[float] = [1.0]
    for _ in range(depth):
        central_singular = forward_round(
            carrier,
            kick_strength,
            coupling_strength,
            free_time,
            stats,
        )
        rank_history.append(len(central_singular))
    latched = boundary(carrier, central_singular)
    latched_copy = json.loads(json.dumps(latched))
    for _ in range(depth):
        inverse_round(
            carrier,
            kick_strength,
            coupling_strength,
            free_time,
            stats,
        )
    restoration_error = physical_distance(initial, carrier)
    if restoration_error > RESTORATION_L2_TOLERANCE:
        fail("four-rotor phase TT actual restoration failed")
    if boundary_distance(latched, latched_copy) != 0.0:
        fail("four-rotor phase TT boundary latch changed")
    carrier.generation += 1
    return {
        "boundary": latched,
        "central_rank_history": rank_history,
        "restoration_error": restoration_error,
        "restoration_generation": carrier.generation,
        "actual_inverse_restoration": True,
        "snapshot_loaded": False,
        "retained_inverse_history_bytes": 0,
        "inverse_topology_rematerialized": True,
        "stats": {
            "coupling_applications": stats.coupling_applications,
            "local_applications": stats.local_applications,
            "qr_factorizations": stats.qr_factorizations,
            "svd_factorizations": stats.svd_factorizations,
            "maximum_resident_cells": stats.maximum_resident_cells,
            "maximum_resident_payload_bytes": (
                stats.maximum_resident_cells * 16
            ),
            "maximum_live_factorized_bond_rank": (
                stats.maximum_bond_rank
            ),
            "maximum_retained_bond_rank": (
                stats.maximum_retained_bond_rank
            ),
            "maximum_interface_core_cells": (
                stats.maximum_interface_core_cells
            ),
            "maximum_interface_core_payload_bytes": (
                stats.maximum_interface_core_cells * 16
            ),
            "discarded_l2_squared": stats.discarded_l2_squared,
        },
    }


def forward_diagnostic(
    mode_radius: int,
    depth: int,
    coupling_strength: float = PRIMARY_G,
) -> dict[str, object]:
    carrier = product_zero_carrier(mode_radius)
    stats = Stats()
    ranks: list[int] = []
    central: list[float] = [1.0]
    for _ in range(depth):
        central = forward_round(
            carrier,
            PRIMARY_K,
            coupling_strength,
            PRIMARY_TAU,
            stats,
        )
        ranks.append(len(central))
    return {
        "mode_radius": mode_radius,
        "local_dimension": 2 * mode_radius + 1,
        "central_rank_history": ranks,
        "boundary": boundary(carrier, central),
        "resident_cells": sum(
            tensor.size for tensor in carrier.tensors
        ),
        "dense_wave_cells_not_materialized": (
            (2 * mode_radius + 1) ** ROTORS
        ),
        "maximum_interface_core_cells": (
            stats.maximum_interface_core_cells
        ),
        "discarded_l2_squared": stats.discarded_l2_squared,
    }


def controls() -> dict[str, object]:
    coupled = forward_diagnostic(8, 1, PRIMARY_G)
    separable = forward_diagnostic(8, 1, 0.0)
    if separable["central_rank_history"] != [1]:
        fail("four-rotor phase TT separable control failed")
    missing = product_zero_carrier(8)
    forward_round(
        missing, PRIMARY_K, PRIMARY_G, PRIMARY_TAU, Stats()
    )
    missing_error = physical_distance(
        product_zero_carrier(8), missing
    )
    wrong = copy_carrier(missing)
    inverse_round(
        wrong, PRIMARY_K, PRIMARY_G * 1.1, PRIMARY_TAU, Stats()
    )
    wrong_error = physical_distance(product_zero_carrier(8), wrong)
    reordered = copy_carrier(missing)
    apply_local(
        reordered, local_kick(8, -PRIMARY_K), Stats()
    )
    for edge in (2, 1, 0):
        apply_coupling(reordered, edge, -PRIMARY_G, Stats())
    apply_free(
        reordered, free_phase(8, PRIMARY_TAU).conj(), Stats()
    )
    reordered_error = physical_distance(
        product_zero_carrier(8), reordered
    )
    return {
        "coupled_central_rank": coupled["central_rank_history"][0],
        "separable_central_rank": separable[
            "central_rank_history"
        ][0],
        "coupling_material": (
            coupled["central_rank_history"][0] > 1
        ),
        "missing_inverse_error": missing_error,
        "wrong_inverse_error": wrong_error,
        "reordered_inverse_error": reordered_error,
    }


def main() -> None:
    guard_runs = [
        forward_diagnostic(radius, PRIMARY_DEPTH)
        for radius in (12, 14, 16)
    ]
    reference_boundary = guard_runs[-1]["boundary"]
    guard_errors = [
        boundary_distance(run["boundary"], reference_boundary)
        for run in guard_runs[:-1]
    ]
    if guard_errors[-1] > BOUNDARY_CONVERGENCE_TOLERANCE:
        fail("four-rotor phase TT mode guard convergence failed")

    accepted = product_zero_carrier(MODE_RADIUS)
    primary = transaction(
        accepted,
        PRIMARY_DEPTH,
        PRIMARY_K,
        PRIMARY_G,
        PRIMARY_TAU,
    )
    reuse = transaction(
        accepted,
        2,
        0.9,
        0.22,
        math.sqrt(5.0),
    )
    control = controls()
    if not (
        control["coupling_material"]
        and control["missing_inverse_error"] > 1.0e-4
        and control["wrong_inverse_error"] > 1.0e-4
        and control["reordered_inverse_error"] > 1.0e-4
    ):
        fail("four-rotor phase TT control separation failed")

    snapshot_image = product_zero_carrier(MODE_RADIUS)
    snapshot_working = copy_carrier(snapshot_image)
    snapshot_stats = Stats()
    snapshot_central: list[float] = [1.0]
    for _ in range(PRIMARY_DEPTH):
        snapshot_central = forward_round(
            snapshot_working,
            PRIMARY_K,
            PRIMARY_G,
            PRIMARY_TAU,
            snapshot_stats,
        )
    snapshot_boundary = boundary(
        snapshot_working, snapshot_central
    )
    snapshot_working = copy_carrier(snapshot_image)
    snapshot_reload_error = physical_distance(
        snapshot_working, snapshot_image
    )
    snapshot_reuse_stats = Stats()
    snapshot_reuse_central: list[float] = [1.0]
    for _ in range(2):
        snapshot_reuse_central = forward_round(
            snapshot_working,
            0.9,
            0.22,
            math.sqrt(5.0),
            snapshot_reuse_stats,
        )
    snapshot_reuse_boundary = boundary(
        snapshot_working, snapshot_reuse_central
    )
    snapshot_working = copy_carrier(snapshot_image)
    snapshot_reuse_reload_error = physical_distance(
        snapshot_working, snapshot_image
    )
    if boundary_distance(snapshot_boundary, primary["boundary"]) > (
        BOUNDARY_CONVERGENCE_TOLERANCE
    ):
        fail("four-rotor phase TT snapshot boundary mismatch")

    dimension = 2 * MODE_RADIUS + 1
    coupling_tail_modes = np.arange(
        COUPLING_KERNEL_RADIUS + 1, 128
    )
    coupling_kernel_tail_energy = float(
        2.0
        * np.sum(
            np.square(jv(coupling_tail_modes, PRIMARY_G))
        )
    )
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_FOUR_ROTOR_NONSEPARABLE_CONTINUOUS_KICKED_"
            "PHASE_FOURIER_TT_CENTRAL_INTERFACE_RANK_GROWTH_"
            "WITH_ACTUAL_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "FOUR_OPEN_CHAIN_ROTORS_MODE_RADIUS14_DEPTH3_"
            "SVD_L2_1E_MINUS11_SOFTWARE_FLOAT64"
        ),
        "law": {
            "onsite_kick_strength": PRIMARY_K,
            "nearest_neighbor_coupling_strength": PRIMARY_G,
            "free_time": PRIMARY_TAU,
            "rotors": ROTORS,
            "open_chain_edges": [[0, 1], [1, 2], [2, 3]],
            "coupling_operator_schmidt_terms": (
                2 * COUPLING_KERNEL_RADIUS + 1
            ),
            "coupling_kernel_tail_energy": (
                coupling_kernel_tail_energy
            ),
        },
        "predeclared_tolerances": {
            "svd_l2": SVD_L2_TOLERANCE,
            "restoration_l2": RESTORATION_L2_TOLERANCE,
            "boundary_guard": BOUNDARY_CONVERGENCE_TOLERANCE,
            "local_tail_energy": LOCAL_TAIL_ENERGY,
        },
        "primary": primary,
        "reuse": reuse,
        "mode_guard_runs": guard_runs,
        "mode_guard_boundary_errors_against_radius14": guard_errors,
        "controls": control,
        "snapshot_sham": {
            "boundary": snapshot_boundary,
            "snapshot_loaded": True,
            "actual_inverse_restoration": False,
            "restoration_generation": 0,
            "reload_error": snapshot_reload_error,
            "reuse_boundary": snapshot_reuse_boundary,
            "reuse_consumed_reloaded_carrier": True,
            "reuse_reload_error": snapshot_reuse_reload_error,
            "image_payload_bytes": payload_bytes(snapshot_image),
            "working_payload_bytes": payload_bytes(snapshot_working),
            "creation_payload_bytes": payload_bytes(snapshot_image),
            "execution_load_payload_copy_bytes": payload_bytes(
                snapshot_image
            ),
            "primary_reload_payload_copy_bytes": payload_bytes(
                snapshot_image
            ),
            "reuse_reload_payload_copy_bytes": payload_bytes(
                snapshot_image
            ),
            "total_creation_and_copy_payload_bytes": (
                4 * payload_bytes(snapshot_image)
            ),
            "primary_maximum_working_payload_bytes": (
                snapshot_stats.maximum_resident_cells * 16
            ),
            "reuse_maximum_working_payload_bytes": (
                snapshot_reuse_stats.maximum_resident_cells * 16
            ),
        },
        "resource_comparison": {
            "local_dimension": dimension,
            "phase_tt_resident_cells": primary["stats"][
                "maximum_resident_cells"
            ],
            "phase_tt_resident_payload_bytes": primary["stats"][
                "maximum_resident_payload_bytes"
            ],
            "dense_wave_cells_not_materialized": dimension**ROTORS,
            "dense_wave_payload_bytes_not_materialized": (
                dimension**ROTORS * 16
            ),
            "maximum_interface_core_payload_bytes": primary["stats"][
                "maximum_interface_core_payload_bytes"
            ],
            "onsite_operator_matrix_payload_bytes": (
                dimension * dimension * 16
            ),
            "free_phase_vector_payload_bytes": dimension * 16,
            "single_shift_matrix_temporary_payload_bytes": (
                dimension * dimension * 16
            ),
            "factorization_counts": {
                "qr": primary["stats"]["qr_factorizations"],
                "svd": primary["stats"]["svd_factorizations"],
            },
            "matched_classical_tt_resident_cells": primary["stats"][
                "maximum_resident_cells"
            ],
            "matched_classical_tt_is_identical_representation": True,
            "controller_backend_traffic_bytes": 0,
        },
        "global_dense_wave_materialized": False,
        "two_site_dense_wave_materialized": False,
        "factorized_coupling_mpo": True,
        "phase_is_primitive_wave_amplitude": True,
        "unresolved_nonseparable_phase_relation": True,
        "local_fourier_compactness_across_unbounded_depth": False,
        "fixed_central_rank_closure": False,
        "machine_enforced_hidden_intermediate": False,
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
