#!/usr/bin/env python3
"""Independent dense oracle for the bounded multi-port TT diagnostic.

This verifier deliberately materializes the 285*2**p state for p=2..6.  That
storage is a counted verification baseline, not an accepted carrier path.  It
does not import or call the tensor-train implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.special import jv


GRID = 17
ROTORS = 4
NECKLACES = 285
CHEBYSHEV_DEGREE = 64
RANK_RELATIVE_TOLERANCE = 2.0e-12
PARITY_TOLERANCE = 4.0e-9
PI = math.pi


def fail(message: str) -> None:
    raise RuntimeError(message)


def compositions(total: int, width: int) -> Iterable[tuple[int, ...]]:
    if width == 1:
        yield (total,)
        return
    for head in range(total + 1):
        for tail in compositions(total - head, width - 1):
            yield (head, *tail)


def rotate(
    histogram: tuple[int, ...],
    shift: int,
) -> tuple[int, ...]:
    return tuple(
        histogram[(index - shift) % GRID] for index in range(GRID)
    )


def canonical(histogram: tuple[int, ...]) -> tuple[int, ...]:
    return min(rotate(histogram, shift) for shift in range(GRID))


def weight(histogram: tuple[int, ...]) -> int:
    denominator = math.prod(math.factorial(value) for value in histogram)
    return GRID * math.factorial(ROTORS) // denominator


def collisions(histogram: tuple[int, ...]) -> int:
    return sum(value * (value - 1) // 2 for value in histogram)


def cyclic(histogram: tuple[int, ...], separation: int) -> int:
    return sum(
        histogram[index] * histogram[(index + separation) % GRID]
        for index in range(GRID)
    )


@dataclass(frozen=True)
class Plan:
    histograms: tuple[tuple[int, ...], ...]
    weights: np.ndarray
    collision: np.ndarray
    index: dict[tuple[int, ...], int]
    roots: np.ndarray


def compile_plan() -> Plan:
    histograms = tuple(
        histogram
        for histogram in compositions(ROTORS, GRID)
        if canonical(histogram) == histogram
    )
    roots = np.exp(2.0j * PI * np.arange(GRID) / GRID)
    result = Plan(
        histograms=histograms,
        weights=np.asarray(
            [weight(histogram) for histogram in histograms],
            dtype=np.float64,
        ),
        collision=np.asarray(
            [collisions(histogram) for histogram in histograms],
            dtype=np.int64,
        ),
        index={
            histogram: index
            for index, histogram in enumerate(histograms)
        },
        roots=roots,
    )
    if (
        len(result.histograms) != NECKLACES
        or int(np.sum(result.weights)) != GRID**ROTORS
    ):
        fail("independent necklace plan mismatch")
    return result


@dataclass(frozen=True)
class Generator:
    matrix: sparse.csr_matrix
    center: float
    radius: float


def compile_generator(plan: Plan, chirp: int) -> Generator:
    angles = np.empty(GRID, dtype=np.float64)
    for momentum in range(GRID):
        eigenvalue = sum(
            plan.roots[
                (chirp * difference * difference + momentum * difference)
                % GRID
            ]
            for difference in range(GRID)
        ) / math.sqrt(GRID)
        if abs(abs(eigenvalue) - 1.0) > 2.0e-12:
            fail("independent one-body phase modulus mismatch")
        angles[momentum] = np.angle(eigenvalue)
    one_body = np.empty((GRID, GRID), dtype=np.complex128)
    for target in range(GRID):
        for source in range(GRID):
            delta = target - source
            one_body[target, source] = sum(
                angles[momentum]
                * plan.roots[(-momentum * delta) % GRID]
                for momentum in range(GRID)
            ) / GRID
    if np.max(np.abs(one_body - np.conjugate(one_body.T))) > 2.0e-12:
        fail("independent one-body generator was non-Hermitian")
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for target_index, histogram in enumerate(plan.histograms):
        accumulated: dict[int, complex] = {}
        for occupied, count in enumerate(histogram):
            if count == 0:
                continue
            for source_mode in range(GRID):
                changed = list(histogram)
                changed[occupied] -= 1
                changed[source_mode] += 1
                source_index = plan.index[canonical(tuple(changed))]
                accumulated[source_index] = (
                    accumulated.get(source_index, 0.0j)
                    + count * one_body[occupied, source_mode]
                )
        for source_index, value in accumulated.items():
            if abs(value) > 1.0e-15:
                rows.append(target_index)
                columns.append(source_index)
                values.append(value)
    lower = ROTORS * float(np.min(angles))
    upper = ROTORS * float(np.max(angles))
    return Generator(
        matrix=sparse.csr_matrix(
            (np.asarray(values), (rows, columns)),
            shape=(NECKLACES, NECKLACES),
        ),
        center=0.5 * (lower + upper),
        radius=0.5 * (upper - lower),
    )


@dataclass(frozen=True)
class Module:
    feature: str
    ports: tuple[int, ...]
    axis: str
    separation: int
    strength: int
    chirp: int


def public_program(ports: int, tag: int) -> list[Module]:
    result: list[Module] = []
    axes = ("x", "y", "z")
    for port in range(1, ports + 1):
        collision = (port + tag) % 2 == 0
        result.append(
            Module(
                feature="collision" if collision else "cyclic_separation",
                ports=(port,),
                axis=axes[(port + tag) % 3],
                separation=0 if collision else 1 + ((2 * port + tag) % 8),
                strength=1 + ((3 * port + 2 * tag) % 16),
                chirp=1 + ((5 * port + tag) % 16),
            )
        )
        if port > 1:
            result.append(
                Module(
                    feature="cyclic_separation",
                    ports=(port - 1, port),
                    axis="controlled_phase",
                    separation=1 + ((port + tag) % 8),
                    strength=1 + ((7 * port + tag) % 16),
                    chirp=1 + ((3 * port + 2 * tag) % 16),
                )
            )
    if ports > 2:
        result.append(
            Module(
                feature="collision",
                ports=(1, ports),
                axis="controlled_phase",
                separation=0,
                strength=1 + ((11 + tag) % 16),
                chirp=1 + ((13 + tag) % 16),
            )
        )
    return result


def descriptor(module: Module) -> dict[str, object]:
    return {
        "feature": module.feature,
        "ports": list(module.ports),
        "axis": module.axis,
        "separation": module.separation,
        "strength": module.strength,
        "chirp": module.chirp,
    }


def program_sha256(program: list[Module]) -> str:
    encoded = json.dumps(
        [descriptor(module) for module in program],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_values(plan: Plan, module: Module) -> np.ndarray:
    if module.feature == "collision":
        return plan.collision
    return np.asarray(
        [
            cyclic(histogram, module.separation)
            for histogram in plan.histograms
        ],
        dtype=np.int64,
    )


def local_matrices(angles: np.ndarray, axis: str) -> np.ndarray:
    result = np.zeros((NECKLACES, 2, 2), dtype=np.complex128)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    if axis == "x":
        result[:, 0, 0] = cosine
        result[:, 1, 1] = cosine
        result[:, 0, 1] = 1.0j * sine
        result[:, 1, 0] = 1.0j * sine
    elif axis == "y":
        result[:, 0, 0] = cosine
        result[:, 1, 1] = cosine
        result[:, 0, 1] = sine
        result[:, 1, 0] = -sine
    elif axis == "z":
        result[:, 0, 0] = np.exp(1.0j * angles)
        result[:, 1, 1] = np.exp(-1.0j * angles)
    else:
        fail("independent local axis mismatch")
    return result


def apply_coupling(
    state: np.ndarray,
    plan: Plan,
    module: Module,
    adjoint: bool,
) -> np.ndarray:
    sign = -1.0 if adjoint else 1.0
    angles = (
        sign
        * 2.0
        * PI
        * ((module.strength * feature_values(plan, module)) % GRID)
        / GRID
    )
    if len(module.ports) == 1:
        axis = module.ports[0]
        moved = np.moveaxis(state, axis, 1)
        transformed = np.einsum(
            "nst,nt...->ns...",
            local_matrices(angles, module.axis),
            moved,
            optimize=True,
        )
        return np.moveaxis(transformed, 1, axis)
    if len(module.ports) != 2 or module.axis != "controlled_phase":
        fail("independent joint descriptor mismatch")
    first, second = module.ports
    moved = np.moveaxis(state, (first, second), (1, 2)).copy()
    phase_shape = (NECKLACES,) + (1,) * (moved.ndim - 3)
    moved[:, 1, 1, ...] *= np.exp(1.0j * angles).reshape(phase_shape)
    return np.moveaxis(moved, (1, 2), (first, second))


def apply_generator(
    state: np.ndarray,
    plan: Plan,
    generator: Generator,
    adjoint: bool,
) -> np.ndarray:
    shape = state.shape
    root_weight = np.sqrt(plan.weights)[:, None]
    previous = state.reshape(NECKLACES, -1) / root_weight

    def scaled(values: np.ndarray) -> np.ndarray:
        return (
            generator.matrix @ values - generator.center * values
        ) / generator.radius

    current = scaled(previous)
    unit = -1.0j if adjoint else 1.0j
    sign = -1.0 if adjoint else 1.0
    result = (
        jv(0, generator.radius) * previous
        + 2.0 * unit * jv(1, generator.radius) * current
    )
    for degree in range(2, CHEBYSHEV_DEGREE + 1):
        following = 2.0 * scaled(current) - previous
        result += (
            2.0
            * (unit**degree)
            * jv(degree, generator.radius)
            * following
        )
        previous, current = current, following
    result *= np.exp(1.0j * sign * generator.center)
    return (result * root_weight).reshape(shape)


def make_state(plan: Plan, ports: int) -> np.ndarray:
    shape = (NECKLACES,) + (2,) * ports
    state = np.empty(shape, dtype=np.complex128)
    necklace = np.asarray(
        [
            math.sqrt(float(plan.weights[index]))
            * plan.roots[
                (7 * index + 3 * int(plan.collision[index])) % GRID
            ]
            / 289.0
            for index in range(NECKLACES)
        ]
    )
    port_state = np.asarray([1.0 + 0.0j])
    for port in range(1, ports + 1):
        port_state = np.kron(
            port_state,
            np.asarray(
                [1.0, plan.roots[(2 * port) % GRID]]
            )
            / math.sqrt(2.0),
        )
    state[:] = (
        necklace[:, None] * port_state[None, :]
    ).reshape(shape)
    return state


def rank_signature(state: np.ndarray) -> list[int]:
    dimensions = state.shape
    result: list[int] = []
    for cut in range(1, len(dimensions)):
        left = math.prod(dimensions[:cut])
        singular = np.linalg.svd(
            state.reshape(left, -1),
            compute_uv=False,
        )
        cutoff = RANK_RELATIVE_TOLERANCE * float(singular[0])
        result.append(int(np.count_nonzero(singular > cutoff)))
    return result


def rank_certificate(state: np.ndarray) -> dict[str, object]:
    dimensions = state.shape
    cuts: list[dict[str, object]] = []
    for cut in range(1, len(dimensions)):
        left = math.prod(dimensions[:cut])
        right = math.prod(dimensions[cut:])
        singular = np.linalg.svd(
            state.reshape(left, right),
            compute_uv=False,
        )
        cutoff = RANK_RELATIVE_TOLERANCE * float(singular[0])
        rank = int(np.count_nonzero(singular > cutoff))
        cuts.append(
            {
                "cut_after_site": cut - 1,
                "left_dimension": left,
                "right_dimension": right,
                "maximum_possible_rank": min(left, right),
                "observed_numerical_rank": rank,
                "minimum_retained_to_cutoff_ratio": (
                    float(singular[rank - 1]) / cutoff
                ),
                "maximum_discarded_to_cutoff_ratio": (
                    float(singular[rank]) / cutoff
                    if rank < singular.size
                    else 0.0
                ),
            }
        )
    return {
        "cuts": cuts,
        "all_cuts_full_numerical_rank": all(
            int(cut["observed_numerical_rank"])
            == int(cut["maximum_possible_rank"])
            for cut in cuts
        ),
        "minimum_retained_to_cutoff_ratio": min(
            float(cut["minimum_retained_to_cutoff_ratio"])
            for cut in cuts
        ),
        "maximum_discarded_to_cutoff_ratio": max(
            float(cut["maximum_discarded_to_cutoff_ratio"])
            for cut in cuts
        ),
    }


def boundary(state: np.ndarray, plan: Plan) -> list[float]:
    probability = np.sum(
        np.abs(state) ** 2,
        axis=tuple(range(1, state.ndim)),
    )
    result = [0.0] * 7
    for index, value in enumerate(probability):
        result[int(plan.collision[index])] += float(value)
    return result


def execute(
    state: np.ndarray,
    plan: Plan,
    program: list[Module],
    generators: dict[int, Generator],
) -> tuple[np.ndarray, list[int]]:
    peak = rank_signature(state)
    for module in program:
        state = apply_coupling(state, plan, module, False)
        observed = rank_signature(state)
        peak = [
            max(old, new)
            for old, new in zip(peak, observed, strict=True)
        ]
        state = apply_generator(
            state,
            plan,
            generators[module.chirp],
            False,
        )
        observed = rank_signature(state)
        peak = [
            max(old, new)
            for old, new in zip(peak, observed, strict=True)
        ]
    return state, peak


def restore(
    state: np.ndarray,
    plan: Plan,
    program: list[Module],
    generators: dict[int, Generator],
) -> np.ndarray:
    for module in reversed(program):
        state = apply_generator(
            state,
            plan,
            generators[module.chirp],
            True,
        )
        state = apply_coupling(state, plan, module, True)
    return state


def max_error(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: four_rotor_necklace_multi_port_dense_oracle.py "
            "PRODUCTION_RESULT"
        )
    production = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    plan = compile_plan()
    chirps = {
        module.chirp
        for ports in range(2, 7)
        for tag in (0, 3)
        for module in public_program(ports, tag)
    }
    generators = {
        chirp: compile_generator(plan, chirp) for chirp in sorted(chirps)
    }
    cases: list[dict[str, object]] = []
    for production_case in production["cases"]:
        ports = int(production_case["ports"])
        primary_program = public_program(ports, 0)
        if (
            program_sha256(primary_program)
            != production_case["primary_program_sha256"]
            or program_sha256(public_program(ports, 3))
            != production_case["reuse_program_sha256"]
        ):
            fail("independent public program disagrees with production")
        baseline = make_state(plan, ports)
        forward, peak = execute(
            baseline.copy(),
            plan,
            primary_program,
            generators,
        )
        final_ranks = rank_signature(forward)
        certificate = rank_certificate(forward)
        boundary_error = max_error(
            boundary(forward, plan),
            list(production_case["primary_boundary"]),
        )
        restored = restore(
            forward.copy(),
            plan,
            primary_program,
            generators,
        )
        restoration_error = float(np.linalg.norm(restored - baseline))
        if (
            peak != production_case["primary_forward_peak_ranks"]
            or final_ranks
            != production_case["primary_forward_final_ranks"]
            or boundary_error > PARITY_TOLERANCE
            or restoration_error > PARITY_TOLERANCE
        ):
            fail(f"independent dense parity failed at {ports} ports")
        reuse_program = public_program(ports, 3)
        reuse_forward, _ = execute(
            restored,
            plan,
            reuse_program,
            generators,
        )
        reuse_boundary_error = max_error(
            boundary(reuse_forward, plan),
            list(production_case["reuse_boundary"]),
        )
        if reuse_boundary_error > PARITY_TOLERANCE:
            fail(f"independent reuse parity failed at {ports} ports")
        cases.append(
            {
                "ports": ports,
                "peak_ranks": peak,
                "final_ranks": final_ranks,
                "final_canonical_matricization_certificate": certificate,
                "boundary_max_abs_error": boundary_error,
                "restoration_l2_error": restoration_error,
                "reuse_boundary_max_abs_error": reuse_boundary_error,
                "verification_dense_complex_cells": int(baseline.size),
            }
        )
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_DIRECT_DENSE_NECKLACE_PORT_RECURRENCE_"
            "AND_NUMERICAL_RANK_PARITY"
        ),
        "production_backend_imported": False,
        "production_projection_called": False,
        "production_program_compiler_called": False,
        "tested_port_counts": [2, 3, 4, 5, 6],
        "cases": cases,
        "all_peak_ranks_match": True,
        "all_final_ranks_match": True,
        "all_final_canonical_cuts_full_numerical_rank": all(
            bool(
                case["final_canonical_matricization_certificate"][
                    "all_cuts_full_numerical_rank"
                ]
            )
            for case in cases
        ),
        "all_boundaries_match": True,
        "all_restorations_within_tolerance": True,
        "verification_dense_assignment_materialized": True,
        "verification_dense_cells_counted": True,
        "maximum_verification_dense_complex_cells": max(
            int(case["verification_dense_complex_cells"])
            for case in cases
        ),
        "dense_operator_materialized": False,
        "accepted_carrier_path": False,
        "matched_identical_tt_reference_exists": True,
        "strongest_compact_classical_established": False,
        "fixed_rank_multi_port_closure_established": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
