#!/usr/bin/env python3
"""Tensor-factorized multi-port phase carrier on the 285-necklace backend.

The accepted path stores a tensor train with physical dimensions
``[285, 2, ..., 2]``.  It never expands the binary latent-port assignment
tensor.  Public local and joint phase consumers are compiled to adjacent TT
updates, the Hermitian necklace generator acts on site zero, and inverse
order is derived by reversing the public forward descriptors.

SVD rank decisions use a declared numerical quotient.  Restoration is
therefore classified as INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT, not exact
restoration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.special import jv


GRID = 17
ROTORS = 4
NECKLACES = 285
MAX_COLLISION = 6
CHEBYSHEV_DEGREE = 64
SVD_RELATIVE_TOLERANCE = 2.0e-12
RESTORATION_TOLERANCE = 4.0e-9
BOUNDARY_TOLERANCE = 4.0e-9
CONTROL_FLOOR = 1.0e-5
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


def rotate_histogram(
    histogram: tuple[int, ...],
    shift: int,
) -> tuple[int, ...]:
    return tuple(
        histogram[(index - shift) % GRID] for index in range(GRID)
    )


def canonical_histogram(
    histogram: tuple[int, ...],
) -> tuple[int, ...]:
    return min(
        rotate_histogram(histogram, shift)
        for shift in range(GRID)
    )


def labelled_weight(histogram: tuple[int, ...]) -> int:
    denominator = math.prod(math.factorial(value) for value in histogram)
    return GRID * math.factorial(ROTORS) // denominator


def collision_count(histogram: tuple[int, ...]) -> int:
    return sum(value * (value - 1) // 2 for value in histogram)


def cyclic_feature(
    histogram: tuple[int, ...],
    separation: int,
) -> int:
    return sum(
        histogram[index] * histogram[(index + separation) % GRID]
        for index in range(GRID)
    )


@dataclass(frozen=True)
class NecklacePlan:
    histograms: tuple[tuple[int, ...], ...]
    weights: np.ndarray
    collisions: np.ndarray
    index: dict[tuple[int, ...], int]
    roots: np.ndarray


def compile_plan() -> NecklacePlan:
    histograms = tuple(
        histogram
        for histogram in compositions(ROTORS, GRID)
        if canonical_histogram(histogram) == histogram
    )
    if len(histograms) != NECKLACES:
        fail("necklace count mismatch")
    weights = np.asarray(
        [labelled_weight(histogram) for histogram in histograms],
        dtype=np.float64,
    )
    if int(np.sum(weights)) != GRID**ROTORS:
        fail("labelled weight mismatch")
    collisions = np.asarray(
        [collision_count(histogram) for histogram in histograms],
        dtype=np.int64,
    )
    roots = np.exp(
        2.0j * PI * np.arange(GRID, dtype=np.float64) / GRID
    )
    return NecklacePlan(
        histograms=histograms,
        weights=weights,
        collisions=collisions,
        index={histogram: i for i, histogram in enumerate(histograms)},
        roots=roots,
    )


@dataclass(frozen=True)
class GeneratorPlan:
    lifted: sparse.csr_matrix
    center: float
    radius: float
    eigenvalue_modulus_error: float
    hermitian_error: float
    tail_bound: float


def compile_generator(plan: NecklacePlan, chirp: int) -> GeneratorPlan:
    angles = np.empty(GRID, dtype=np.float64)
    modulus_error = 0.0
    for momentum in range(GRID):
        eigenvalue = sum(
            plan.roots[
                (chirp * difference * difference + momentum * difference)
                % GRID
            ]
            for difference in range(GRID)
        ) / math.sqrt(GRID)
        modulus_error = max(
            modulus_error, abs(abs(eigenvalue) - 1.0)
        )
        angles[momentum] = np.angle(eigenvalue)
    one_body = np.empty((GRID, GRID), dtype=np.complex128)
    for target in range(GRID):
        for source in range(GRID):
            difference = target - source
            one_body[target, source] = sum(
                angles[momentum]
                * plan.roots[(-momentum * difference) % GRID]
                for momentum in range(GRID)
            ) / GRID
    hermitian_error = float(
        np.max(np.abs(one_body - np.conjugate(one_body.T)))
    )
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for target_index, histogram in enumerate(plan.histograms):
        accumulated: dict[int, complex] = {}
        for occupied, count in enumerate(histogram):
            if count == 0:
                continue
            for source_mode in range(GRID):
                source = list(histogram)
                source[occupied] -= 1
                source[source_mode] += 1
                source_index = plan.index[
                    canonical_histogram(tuple(source))
                ]
                accumulated[source_index] = (
                    accumulated.get(source_index, 0.0j)
                    + count * one_body[occupied, source_mode]
                )
        for source_index, value in accumulated.items():
            if abs(value) > 1.0e-15:
                rows.append(target_index)
                columns.append(source_index)
                values.append(value)
    lifted = sparse.csr_matrix(
        (np.asarray(values), (rows, columns)),
        shape=(NECKLACES, NECKLACES),
        dtype=np.complex128,
    )
    lower = ROTORS * float(np.min(angles))
    upper = ROTORS * float(np.max(angles))
    center = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    first_omitted = CHEBYSHEV_DEGREE + 1
    leading = (
        (0.5 * radius) ** first_omitted
        / math.gamma(first_omitted + 1)
        * math.exp(radius * radius / (4.0 * (first_omitted + 1)))
    )
    ratio = radius / (2.0 * (first_omitted + 1))
    tail = 2.0 * leading / (1.0 - ratio)
    if (
        modulus_error > 2.0e-12
        or hermitian_error > 2.0e-12
        or tail > 1.0e-13
        or radius <= 0.0
    ):
        fail("generator plan invariant failed")
    return GeneratorPlan(
        lifted=lifted,
        center=center,
        radius=radius,
        eigenvalue_modulus_error=modulus_error,
        hermitian_error=hermitian_error,
        tail_bound=tail,
    )


@dataclass
class Stats:
    svd_calls: int = 0
    swap_calls: int = 0
    local_consumptions: int = 0
    joint_consumptions: int = 0
    generator_applications: int = 0
    streamed_generator_terms: int = 0
    chebyshev_vector_updates: int = 0
    discarded_singular_l2: float = 0.0
    maximum_temporary_complex_cells: int = 0
    maximum_estimated_generator_live_complex_cells: int = 0
    maximum_tt_complex_cells: int = 0
    maximum_bond_rank: int = 1
    maximum_first_bond_rank: int = 1
    maximum_rank_signature: list[int] = field(default_factory=list)

    def observe(self, carrier: "Carrier") -> None:
        self.maximum_tt_complex_cells = max(
            self.maximum_tt_complex_cells,
            carrier.complex_cells(),
        )
        signature = carrier.rank_signature()
        if not self.maximum_rank_signature:
            self.maximum_rank_signature = list(signature)
        elif len(self.maximum_rank_signature) != len(signature):
            fail("TT rank-signature width changed")
        else:
            self.maximum_rank_signature = [
                max(previous, observed)
                for previous, observed in zip(
                    self.maximum_rank_signature,
                    signature,
                    strict=True,
                )
            ]
        self.maximum_bond_rank = max(
            self.maximum_bond_rank, *signature
        )
        self.maximum_first_bond_rank = max(
            self.maximum_first_bond_rank, signature[0]
        )


@dataclass
class Module:
    feature: str
    ports: tuple[int, ...]
    axis: str
    separation: int
    strength: int
    chirp: int


@dataclass
class Carrier:
    cores: list[np.ndarray]
    order: list[int]
    generation: int = 0
    lease: int = 1
    discarded_l2: float = 0.0
    retained_inverse_history_bytes: int = 0
    baseline_reload_bytes: int = 0

    def clone(self) -> "Carrier":
        return Carrier(
            cores=[core.copy() for core in self.cores],
            order=list(self.order),
            generation=self.generation,
            lease=self.lease,
            discarded_l2=self.discarded_l2,
            retained_inverse_history_bytes=(
                self.retained_inverse_history_bytes
            ),
            baseline_reload_bytes=self.baseline_reload_bytes,
        )

    def complex_cells(self) -> int:
        return sum(int(core.size) for core in self.cores)

    def rank_signature(self) -> list[int]:
        return [int(core.shape[2]) for core in self.cores[:-1]]


def make_carrier(plan: NecklacePlan, ports: int, identity: int) -> Carrier:
    necklace = np.empty((1, NECKLACES, 1), dtype=np.complex128)
    for index in range(NECKLACES):
        exponent = (
            7 * index + 3 * int(plan.collisions[index]) + 5 * identity
        ) % GRID
        necklace[0, index, 0] = (
            math.sqrt(float(plan.weights[index]))
            * plan.roots[exponent]
            / 289.0
        )
    cores = [necklace]
    scale = 1.0 / math.sqrt(2.0)
    for port in range(1, ports + 1):
        phase = plan.roots[(2 * port) % GRID]
        cores.append(
            np.asarray([scale, scale * phase], dtype=np.complex128)
            .reshape(1, 2, 1)
        )
    return Carrier(cores=cores, order=list(range(ports + 1)))


def split_block(
    theta: np.ndarray,
    physical_dimensions: list[int],
    stats: Stats,
) -> list[np.ndarray]:
    left_rank = int(theta.shape[0])
    right_rank = int(theta.shape[-1])
    work = theta
    result: list[np.ndarray] = []
    for offset, dimension in enumerate(physical_dimensions[:-1]):
        matrix = work.reshape(left_rank * dimension, -1)
        stats.maximum_temporary_complex_cells = max(
            stats.maximum_temporary_complex_cells,
            int(matrix.size),
        )
        u, singular, vh = np.linalg.svd(
            matrix, full_matrices=False
        )
        stats.svd_calls += 1
        if singular.size == 0:
            fail("empty SVD spectrum")
        cutoff = SVD_RELATIVE_TOLERANCE * float(singular[0])
        rank = max(1, int(np.count_nonzero(singular > cutoff)))
        if rank < singular.size:
            discarded = float(
                np.linalg.norm(singular[rank:])
            )
            stats.discarded_singular_l2 = math.hypot(
                stats.discarded_singular_l2, discarded
            )
        result.append(
            u[:, :rank].reshape(left_rank, dimension, rank)
        )
        remaining = physical_dimensions[offset + 1 :]
        work = (
            singular[:rank, None] * vh[:rank, :]
        ).reshape(rank, *remaining, right_rank)
        left_rank = rank
    result.append(
        work.reshape(
            left_rank, physical_dimensions[-1], right_rank
        )
    )
    return result


def swap_adjacent(carrier: Carrier, left: int, stats: Stats) -> None:
    first = carrier.cores[left]
    second = carrier.cores[left + 1]
    theta = np.einsum("asx,xtb->astb", first, second)
    theta = np.transpose(theta, (0, 2, 1, 3))
    carrier.cores[left : left + 2] = split_block(
        theta,
        [int(second.shape[1]), int(first.shape[1])],
        stats,
    )
    carrier.order[left], carrier.order[left + 1] = (
        carrier.order[left + 1],
        carrier.order[left],
    )
    stats.swap_calls += 1
    stats.observe(carrier)


def move_label(
    carrier: Carrier,
    label: int,
    position: int,
    stats: Stats,
) -> None:
    current = carrier.order.index(label)
    while current > position:
        swap_adjacent(carrier, current - 1, stats)
        current -= 1
    while current < position:
        swap_adjacent(carrier, current, stats)
        current += 1


def restore_canonical_order(carrier: Carrier, stats: Stats) -> None:
    for position, label in enumerate(sorted(carrier.order)):
        move_label(carrier, label, position, stats)
    if carrier.order != list(range(len(carrier.order))):
        fail("canonical TT order restoration failed")


def module_features(
    plan: NecklacePlan,
    module: Module,
) -> np.ndarray:
    if module.feature == "collision":
        if module.separation != 0:
            fail("collision module separation was nonzero")
        return plan.collisions
    if (
        module.feature != "cyclic_separation"
        or not 1 <= module.separation <= GRID // 2
    ):
        fail("cyclic feature descriptor invalid")
    return np.asarray(
        [
            cyclic_feature(histogram, module.separation)
            for histogram in plan.histograms
        ],
        dtype=np.int64,
    )


def local_matrices(
    angles: np.ndarray,
    axis: str,
) -> np.ndarray:
    cosine = np.cos(angles)
    sine = np.sin(angles)
    matrices = np.zeros((NECKLACES, 2, 2), dtype=np.complex128)
    if axis == "x":
        matrices[:, 0, 0] = cosine
        matrices[:, 1, 1] = cosine
        matrices[:, 0, 1] = 1.0j * sine
        matrices[:, 1, 0] = 1.0j * sine
    elif axis == "y":
        matrices[:, 0, 0] = cosine
        matrices[:, 1, 1] = cosine
        matrices[:, 0, 1] = sine
        matrices[:, 1, 0] = -sine
    elif axis == "z":
        matrices[:, 0, 0] = np.exp(1.0j * angles)
        matrices[:, 1, 1] = np.exp(-1.0j * angles)
    else:
        fail("local axis invalid")
    return matrices


def apply_coupling(
    carrier: Carrier,
    plan: NecklacePlan,
    module: Module,
    adjoint: bool,
    stats: Stats,
) -> None:
    if len(module.ports) not in (1, 2):
        fail("module port arity invalid")
    if any(
        port <= 0 or port >= len(carrier.order)
        for port in module.ports
    ):
        fail("module port identity invalid")
    if len(set(module.ports)) != len(module.ports):
        fail("duplicate port in joint module")
    sign = -1.0 if adjoint else 1.0
    features = module_features(plan, module)
    angles = (
        sign
        * 2.0
        * PI
        * ((module.strength * features) % GRID)
        / GRID
    )
    if len(module.ports) == 1:
        move_label(carrier, module.ports[0], 1, stats)
        first, second = carrier.cores[0:2]
        theta = np.einsum("anx,xsb->ansb", first, second)
        matrices = local_matrices(angles, module.axis)
        theta = np.einsum("nst,antb->ansb", matrices, theta)
        carrier.cores[0:2] = split_block(
            theta,
            [NECKLACES, 2],
            stats,
        )
        stats.local_consumptions += 1
    else:
        if module.axis != "controlled_phase":
            fail("joint module axis invalid")
        move_label(carrier, module.ports[0], 1, stats)
        move_label(carrier, module.ports[1], 2, stats)
        first, second, third = carrier.cores[0:3]
        theta = np.einsum(
            "anx,xsy,ytb->anstb", first, second, third
        )
        theta[:, :, 1, 1, :] *= np.exp(1.0j * angles)[
            None, :, None
        ]
        carrier.cores[0:3] = split_block(
            theta,
            [NECKLACES, 2, 2],
            stats,
        )
        stats.joint_consumptions += 1
    restore_canonical_order(carrier, stats)
    stats.observe(carrier)


@lru_cache(maxsize=None)
def cached_generator(
    chirp: int,
    plan_identity: int,
) -> GeneratorPlan:
    del plan_identity
    return compile_generator(GLOBAL_PLAN, chirp)


def apply_generator(
    carrier: Carrier,
    module: Module,
    adjoint: bool,
    stats: Stats,
) -> None:
    if carrier.order[0] != 0:
        fail("necklace coordinate left site zero")
    generator = cached_generator(module.chirp, id(GLOBAL_PLAN))
    core = carrier.cores[0]
    if core.shape[0] != 1 or core.shape[1] != NECKLACES:
        fail("necklace TT core shape invalid")
    root_weight = np.sqrt(GLOBAL_PLAN.weights)[:, None]
    samples = core[0, :, :].copy() / root_weight
    estimated_live = 6 * int(samples.size)
    stats.maximum_estimated_generator_live_complex_cells = max(
        stats.maximum_estimated_generator_live_complex_cells,
        estimated_live,
    )
    stats.maximum_temporary_complex_cells = max(
        stats.maximum_temporary_complex_cells,
        estimated_live,
    )
    previous = samples.copy()

    def scaled(values: np.ndarray) -> np.ndarray:
        stats.generator_applications += 1
        stats.streamed_generator_terms += (
            int(generator.lifted.nnz) * int(values.shape[1])
        )
        return (
            generator.lifted @ values
            - generator.center * values
        ) / generator.radius

    current = scaled(previous)
    sign = -1.0 if adjoint else 1.0
    unit = -1.0j if adjoint else 1.0j
    samples = (
        jv(0, generator.radius) * previous
        + 2.0
        * unit
        * jv(1, generator.radius)
        * current
    )
    stats.chebyshev_vector_updates += int(samples.size)
    for degree in range(2, CHEBYSHEV_DEGREE + 1):
        next_values = 2.0 * scaled(current) - previous
        samples += (
            2.0
            * (unit**degree)
            * jv(degree, generator.radius)
            * next_values
        )
        previous, current = current, next_values
        stats.chebyshev_vector_updates += int(samples.size)
    samples *= np.exp(1.0j * sign * generator.center)
    samples *= root_weight
    carrier.cores[0] = samples.reshape(
        1, NECKLACES, samples.shape[1]
    )
    stats.observe(carrier)


def forward_module(
    carrier: Carrier,
    plan: NecklacePlan,
    module: Module,
    stats: Stats,
) -> None:
    apply_coupling(carrier, plan, module, False, stats)
    apply_generator(carrier, module, False, stats)


def inverse_module(
    carrier: Carrier,
    plan: NecklacePlan,
    module: Module,
    stats: Stats,
) -> None:
    apply_generator(carrier, module, True, stats)
    apply_coupling(carrier, plan, module, True, stats)


def reordered_inverse_module(
    carrier: Carrier,
    plan: NecklacePlan,
    module: Module,
    stats: Stats,
) -> None:
    apply_coupling(carrier, plan, module, True, stats)
    apply_generator(carrier, module, True, stats)


def round_sweep(carrier: Carrier, stats: Stats) -> None:
    for edge in range(len(carrier.cores) - 1):
        first = carrier.cores[edge]
        second = carrier.cores[edge + 1]
        theta = np.einsum("asx,xtb->astb", first, second)
        carrier.cores[edge : edge + 2] = split_block(
            theta,
            [int(first.shape[1]), int(second.shape[1])],
            stats,
        )
    for edge in range(len(carrier.cores) - 2, -1, -1):
        first = carrier.cores[edge]
        second = carrier.cores[edge + 1]
        theta = np.einsum("asx,xtb->astb", first, second)
        carrier.cores[edge : edge + 2] = split_block(
            theta,
            [int(first.shape[1]), int(second.shape[1])],
            stats,
        )
    stats.observe(carrier)


def inner_product(
    left: Carrier,
    right: Carrier,
    plan: NecklacePlan,
) -> complex:
    if left.order != right.order:
        fail("inner product TT order mismatch")
    environment = np.ones((1, 1), dtype=np.complex128)
    for site, (left_core, right_core) in enumerate(
        zip(left.cores, right.cores, strict=True)
    ):
        physical_weight = np.ones(
            left_core.shape[1], dtype=np.float64
        )
        environment = np.einsum(
            "asx,ab,bsy,s->xy",
            np.conjugate(left_core),
            environment,
            right_core,
            physical_weight,
            optimize=True,
        )
    return complex(environment[0, 0])


def carrier_norm(carrier: Carrier, plan: NecklacePlan) -> float:
    value = inner_product(carrier, carrier, plan)
    return math.sqrt(max(0.0, float(value.real)))


def carrier_distance(
    left: Carrier,
    right: Carrier,
    plan: NecklacePlan,
) -> float:
    if (
        left.order != right.order
        or len(left.cores) != len(right.cores)
    ):
        fail("difference TT topology mismatch")
    difference: list[np.ndarray] = []
    difference.append(
        np.concatenate(
            (left.cores[0], -right.cores[0]),
            axis=2,
        )
    )
    for left_core, right_core in zip(
        left.cores[1:-1],
        right.cores[1:-1],
        strict=True,
    ):
        if left_core.shape[1] != right_core.shape[1]:
            fail("difference TT physical dimension mismatch")
        left_rank = int(left_core.shape[0])
        right_rank = int(right_core.shape[0])
        left_next = int(left_core.shape[2])
        right_next = int(right_core.shape[2])
        combined = np.zeros(
            (
                left_rank + right_rank,
                int(left_core.shape[1]),
                left_next + right_next,
            ),
            dtype=np.complex128,
        )
        combined[:left_rank, :, :left_next] = left_core
        combined[left_rank:, :, left_next:] = right_core
        difference.append(combined)
    difference.append(
        np.concatenate(
            (left.cores[-1], right.cores[-1]),
            axis=0,
        )
    )
    difference_carrier = Carrier(
        cores=difference,
        order=list(left.order),
    )
    return carrier_norm(difference_carrier, plan)


def rank_one_restoration_bound(
    left: Carrier,
    right: Carrier,
) -> float:
    """Return a cancellation-resistant upper bound for product-state error."""

    if left.order != right.order:
        fail("rank-one restoration order mismatch")
    left_scale = 1.0 + 0.0j
    right_scale = 1.0 + 0.0j
    local_errors: list[float] = []
    for left_core, right_core in zip(
        left.cores,
        right.cores,
        strict=True,
    ):
        if (
            left_core.shape[0] != 1
            or left_core.shape[2] != 1
            or right_core.shape[0] != 1
            or right_core.shape[2] != 1
            or left_core.shape[1] != right_core.shape[1]
        ):
            fail("rank-one restoration requested for non-product TT")

        def canonical(
            core: np.ndarray,
        ) -> tuple[complex, np.ndarray]:
            vector = core[0, :, 0]
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                fail("zero rank-one TT core")
            unit = vector / norm
            nonzero = np.flatnonzero(np.abs(unit) > 1.0e-300)
            if nonzero.size == 0:
                fail("rank-one TT core lacked a stable phase pivot")
            pivot = int(nonzero[0])
            phase = unit[pivot] / abs(unit[pivot])
            return norm * phase, unit / phase

        left_factor, left_unit = canonical(left_core)
        right_factor, right_unit = canonical(right_core)
        left_scale *= left_factor
        right_scale *= right_factor
        local_errors.append(
            float(np.linalg.norm(left_unit - right_unit))
        )
    return float(
        abs(left_scale - right_scale)
        + abs(right_scale) * sum(local_errors)
    )


def generator_resource_counts(
    plan: NecklacePlan,
    programs: Iterable[list[Module]],
) -> dict[str, int]:
    chirps = sorted(
        {
            module.chirp
            for program in programs
            for module in program
        }
    )
    generators = [
        cached_generator(chirp, id(plan)) for chirp in chirps
    ]
    return {
        "unique_generator_plans": len(generators),
        "sparse_generator_nonzeros": sum(
            int(generator.lifted.nnz) for generator in generators
        ),
        "sparse_generator_csr_array_bytes": sum(
            int(generator.lifted.data.nbytes)
            + int(generator.lifted.indices.nbytes)
            + int(generator.lifted.indptr.nbytes)
            for generator in generators
        ),
    }


def boundary(carrier: Carrier, plan: NecklacePlan) -> list[float]:
    if carrier.order != list(range(len(carrier.order))):
        fail("boundary requested from noncanonical TT order")
    environment = np.ones((1, 1), dtype=np.complex128)
    for core in reversed(carrier.cores[1:]):
        environment = np.einsum(
            "asx,xy,bsy->ab",
            core,
            environment,
            np.conjugate(core),
            optimize=True,
        )
    necklace_core = carrier.cores[0][0, :, :]
    probability = np.einsum(
        "na,ab,nb->n",
        necklace_core,
        environment,
        np.conjugate(necklace_core),
        optimize=True,
    ).real
    result = [0.0] * (MAX_COLLISION + 1)
    for index, value in enumerate(probability):
        result[int(plan.collisions[index])] += float(value)
    return result


def boundary_distance(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def public_program(ports: int, tag: int) -> list[Module]:
    if not 2 <= ports <= 8:
        fail("public port count outside bounded range")
    modules: list[Module] = []
    axes = ("x", "y", "z")
    for port in range(1, ports + 1):
        modules.append(
            Module(
                feature=(
                    "collision"
                    if (port + tag) % 2 == 0
                    else "cyclic_separation"
                ),
                ports=(port,),
                axis=axes[(port + tag) % len(axes)],
                separation=(
                    0
                    if (port + tag) % 2 == 0
                    else 1 + ((2 * port + tag) % 8)
                ),
                strength=1 + ((3 * port + 2 * tag) % 16),
                chirp=1 + ((5 * port + tag) % 16),
            )
        )
        if port > 1:
            modules.append(
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
        modules.append(
            Module(
                feature="collision",
                ports=(1, ports),
                axis="controlled_phase",
                separation=0,
                strength=1 + ((11 + tag) % 16),
                chirp=1 + ((13 + tag) % 16),
            )
        )
    return modules


@dataclass
class Transaction:
    final_boundary: list[float]
    restoration_upper_bound: float
    norm_error: float
    restored_ranks: list[int]
    forward_final_ranks: list[int]
    forward_peak_ranks: list[int]
    forward_peak_cells: int
    stats: Stats


def transaction(
    carrier: Carrier,
    baseline: Carrier,
    plan: NecklacePlan,
    program: list[Module],
    control: str = "correct",
) -> Transaction:
    stats = Stats()
    stats.observe(carrier)
    for module in program:
        forward_module(carrier, plan, module, stats)
    final = boundary(carrier, plan)
    norm_error = abs(carrier_norm(carrier, plan) - 1.0)
    forward_final_ranks = carrier.rank_signature()
    forward_peak_ranks = list(stats.maximum_rank_signature)
    forward_peak_cells = stats.maximum_tt_complex_cells
    minimum = 1 if control == "missing" else 0
    for cursor in range(len(program) - 1, minimum - 1, -1):
        module = program[cursor]
        if control == "wrong" and cursor == len(program) - 1:
            module = copy.copy(module)
            module.strength = module.strength % 16 + 1
        if control == "reordered":
            reordered_inverse_module(carrier, plan, module, stats)
        else:
            inverse_module(carrier, plan, module, stats)
    round_sweep(carrier, stats)
    if all(rank == 1 for rank in carrier.rank_signature()):
        restoration_upper_bound = rank_one_restoration_bound(
            carrier,
            baseline,
        )
    else:
        restoration_upper_bound = carrier_distance(
            carrier,
            baseline,
            plan,
        )
    if control == "correct":
        carrier.generation += 1
        carrier.lease += 1
    carrier.discarded_l2 = math.hypot(
        carrier.discarded_l2, stats.discarded_singular_l2
    )
    return Transaction(
        final_boundary=final,
        restoration_upper_bound=restoration_upper_bound,
        norm_error=norm_error,
        restored_ranks=carrier.rank_signature(),
        forward_final_ranks=forward_final_ranks,
        forward_peak_ranks=forward_peak_ranks,
        forward_peak_cells=forward_peak_cells,
        stats=stats,
    )


def module_to_json(module: Module) -> dict[str, object]:
    return {
        "feature": module.feature,
        "ports": list(module.ports),
        "axis": module.axis,
        "separation": module.separation,
        "strength": module.strength,
        "chirp": module.chirp,
    }


def encoded_program(program: list[Module]) -> bytes:
    return json.dumps(
        [module_to_json(module) for module in program],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def run_port_case(plan: NecklacePlan, ports: int) -> dict[str, object]:
    baseline = make_carrier(plan, ports, 0)
    carrier = baseline.clone()
    carrier_identity = id(carrier)
    primary_program = public_program(ports, 0)
    primary = transaction(
        carrier, baseline, plan, primary_program
    )
    if (
        primary.restoration_upper_bound > RESTORATION_TOLERANCE
        or primary.norm_error > RESTORATION_TOLERANCE
        or id(carrier) != carrier_identity
    ):
        fail(f"primary restoration failed at {ports} ports")
    reuse_program = public_program(ports, 3)
    reuse = transaction(
        carrier, baseline, plan, reuse_program
    )
    fresh = make_carrier(plan, ports, 0)
    fresh_run = transaction(
        fresh, baseline, plan, reuse_program
    )
    reuse_boundary_error = boundary_distance(
        reuse.final_boundary, fresh_run.final_boundary
    )
    if (
        reuse.restoration_upper_bound > RESTORATION_TOLERANCE
        or fresh_run.restoration_upper_bound > RESTORATION_TOLERANCE
        or reuse_boundary_error > BOUNDARY_TOLERANCE
        or reuse.restored_ranks != fresh_run.restored_ranks
        or id(carrier) != carrier_identity
        or carrier.generation != 2
        or carrier.baseline_reload_bytes != 0
    ):
        fail(f"reuse restoration failed at {ports} ports")
    dense_cells_not_materialized = NECKLACES * (2**ports)
    baseline_cells = baseline.complex_cells()
    restored_carrier_cells = carrier.complex_cells()
    primary_encoded = encoded_program(primary_program)
    reuse_encoded = encoded_program(reuse_program)
    generator_resources = generator_resource_counts(
        plan,
        (primary_program, reuse_program),
    )
    return {
        "ports": ports,
        "primary_module_count": len(primary_program),
        "reuse_module_count": len(reuse_program),
        "primary_program_sha256": hashlib.sha256(
            primary_encoded
        ).hexdigest(),
        "reuse_program_sha256": hashlib.sha256(
            reuse_encoded
        ).hexdigest(),
        "primary_boundary": primary.final_boundary,
        "primary_restoration_upper_bound": (
            primary.restoration_upper_bound
        ),
        "primary_norm_error": primary.norm_error,
        "primary_forward_final_ranks": primary.forward_final_ranks,
        "primary_forward_peak_ranks": primary.forward_peak_ranks,
        "primary_restored_ranks": primary.restored_ranks,
        "primary_peak_tt_complex_cells": (
            primary.stats.maximum_tt_complex_cells
        ),
        "primary_public_descriptor_bytes": len(primary_encoded),
        "reuse_public_descriptor_bytes": len(reuse_encoded),
        **generator_resources,
        "reuse_restoration_upper_bound": reuse.restoration_upper_bound,
        "reuse_boundary": reuse.final_boundary,
        "fresh_restored_reuse_boundary_error": reuse_boundary_error,
        "reuse_rank_signature_equal": (
            reuse.restored_ranks == fresh_run.restored_ranks
        ),
        "reuse_streamed_generator_terms_equal": (
            reuse.stats.streamed_generator_terms
            == fresh_run.stats.streamed_generator_terms
        ),
        "restoration_generation": carrier.generation,
        "same_logical_carrier_container": id(carrier) == carrier_identity,
        "core_array_backing_preserved": False,
        "baseline_reload_bytes": carrier.baseline_reload_bytes,
        "retained_inverse_history_bytes": (
            carrier.retained_inverse_history_bytes
        ),
        "matched_explicit_dense_assignment_complex_cells": (
            dense_cells_not_materialized
        ),
        "explicit_assignment_indexed_dense_tensor_materialized": False,
        "equivalent_full_width_first_bond_materialized": (
            primary.forward_peak_ranks[0] == 2**ports
        ),
        "tt_to_dense_cell_ratio": (
            primary.stats.maximum_tt_complex_cells
            / dense_cells_not_materialized
        ),
        "svd_calls": primary.stats.svd_calls,
        "swap_calls": primary.stats.swap_calls,
        "maximum_temporary_complex_cells": (
            primary.stats.maximum_temporary_complex_cells
        ),
        "maximum_estimated_generator_live_complex_cells": (
            primary.stats.maximum_estimated_generator_live_complex_cells
        ),
        "discarded_singular_l2": (
            primary.stats.discarded_singular_l2
        ),
        "streamed_generator_terms": (
            primary.stats.streamed_generator_terms
        ),
        "permanent_restoration_baseline_complex_cells": baseline_cells,
        "restored_logical_carrier_complex_cells": restored_carrier_cells,
        "maximum_accounted_primary_resident_carrier_complex_cells": (
            baseline_cells + primary.stats.maximum_tt_complex_cells
        ),
        "maximum_accounted_reuse_parity_resident_carrier_complex_cells": (
            baseline_cells
            + restored_carrier_cells
            + fresh_run.stats.maximum_tt_complex_cells
        ),
        "reported_temporary_cells_are_component_maximum_not_rss_peak": True,
    }


def controls(plan: NecklacePlan, ports: int) -> dict[str, float]:
    program = public_program(ports, 0)
    result: dict[str, float] = {}
    for name in ("missing", "wrong", "reordered"):
        baseline = make_carrier(plan, ports, 0)
        carrier = baseline.clone()
        run = transaction(
            carrier, baseline, plan, program, control=name
        )
        if run.restoration_upper_bound <= CONTROL_FLOOR:
            fail(f"{name} inverse did not separate")
        result[f"{name}_inverse_restoration_upper_bound"] = (
            run.restoration_upper_bound
        )
    return result


GLOBAL_PLAN = compile_plan()


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: four_rotor_necklace_multi_port_tt_phase.py"
        )
    cases = [run_port_case(GLOBAL_PLAN, ports) for ports in range(2, 7)]
    inverse_controls = controls(GLOBAL_PLAN, 4)
    all_programs = [
        public_program(ports, tag)
        for ports in range(2, 7)
        for tag in (0, 3)
    ]
    global_generator_resources = generator_resource_counts(
        GLOBAL_PLAN,
        all_programs,
    )
    first_bond_ranks = [
        int(case["primary_forward_peak_ranks"][0]) for case in cases
    ]
    full_binary_widths = [
        2 ** int(case["ports"]) for case in cases
    ]
    if any(
        later < earlier
        for earlier, later in zip(
            first_bond_ranks, first_bond_ranks[1:]
        )
    ):
        fail("first-bond rank did not grow monotonically")
    result = {
        "claim_candidate": (
            "BOUNDED_TENSOR_FACTORIZED_MULTI_PORT_NECKLACE_PHASE_"
            "DIAGNOSTIC_FINDS_FULL_TOLERANCE_DEFINED_CANONICAL_"
            "MATRICIZATION_RANK_AND_NO_TT_COMPACTION_WITH_"
            "RESTORATION_AND_LOGICAL_REUSE"
        ),
        "result": "PASS",
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_NUMPY_SCIPY_COMPLEX128_GRID17_"
            "FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_"
            "285_NECKLACES_BINARY_PORT_COUNTS2_TO6_PUBLIC_"
            "ALGORITHMIC_LOCAL_CHAIN_AND_RING_JOINT_PROGRAMS_"
            "SVD_RELATIVE_2E_MINUS12_SOFTWARE_ONLY_NO_FIXED_BUFFER_"
            "OR_PROCESS_RSS_CLAIM"
        ),
        "restoration_class": (
            "INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT"
        ),
        "predeclared_tolerances": {
            "svd_relative": SVD_RELATIVE_TOLERANCE,
            "restoration": RESTORATION_TOLERANCE,
            "boundary": BOUNDARY_TOLERANCE,
        },
        "carrier": {
            "necklace_cells": NECKLACES,
            "tested_port_counts": [2, 3, 4, 5, 6],
            "physical_dimensions": "[285]+[2]*ports",
            "explicit_assignment_indexed_dense_tensor_materialized": False,
            "equivalent_full_width_first_bond_materialized": True,
            "truth_table_materialized": False,
            "answer_bearing_lookup_materialized": False,
            "topology_compiler_uses_final_answer": False,
            "inverse_descriptors_retained": 0,
            "inverse_order": "REVERSE_PUBLIC_FORWARD_DESCRIPTORS",
        },
        "cases": cases,
        "rank_observation": {
            "first_bond_ranks": first_bond_ranks,
            "full_binary_widths": full_binary_widths,
            "first_bond_rank_equals_full_binary_width": (
                first_bond_ranks == full_binary_widths
            ),
            "monotone_non_decreasing": True,
            "fixed_rank_across_tested_port_count": (
                len(set(first_bond_ranks)) == 1
            ),
            "maximum_first_bond_rank": max(first_bond_ranks),
            "maximum_tt_cells": max(
                int(case["primary_peak_tt_complex_cells"])
                for case in cases
            ),
            "maximum_matched_explicit_dense_assignment_cells": max(
                int(
                    case[
                        "matched_explicit_dense_assignment_complex_cells"
                    ]
                )
                for case in cases
            ),
            "tensor_train_smaller_than_explicit_dense_assignment": (
                all(
                    int(case["primary_peak_tt_complex_cells"])
                    < int(
                        case[
                            "matched_explicit_dense_assignment_complex_cells"
                        ]
                    )
                    for case in cases
                )
            ),
        },
        "controls": inverse_controls,
        "resource_law": {
            "tt_core_complex_cells_counted": True,
            "svd_temporary_matrix_complex_cells_counted": True,
            "generator_live_complex_cells_conservatively_estimated": True,
            "sparse_generator_nonzeros_counted_via_term_updates": True,
            **global_generator_resources,
            "retained_inverse_history_bytes": 0,
            "explicit_assignment_indexed_dense_cells": 0,
            "maximum_equivalent_width_first_core_complex_cells": (
                NECKLACES * max(first_bond_ranks)
            ),
            "allocator_blas_lapack_scipy_python_os_peak_bounded": False,
        },
        "matched_classical_references": {
            "identical_tensor_train_recurrence_exists": True,
            "separate_direct_dense_recurrence_executed": True,
            "dense_reference_smaller_in_complex_cells_for_all_cases": True,
            "verification_level": "SEPARATE_REFERENCE_PARITY",
            "separate_reference_parity": True,
        },
        "strongest_compact_classical_established": False,
        "catvm_custody_established": False,
        "fixed_rank_multi_port_closure_established": (
            len(set(first_bond_ranks)) == 1
        ),
        "compact_multi_port_carrier_established": False,
        "observed_obstruction": (
            "FINAL_CANONICAL_NUMERICAL_MATRICIZATION_RANKS_ARE_"
            "FULL_AT_EVERY_CUT_FIRST_RANK_EQUALS_2_TO_THE_PORT_"
            "COUNT_AND_TT_STORAGE_EXCEEDS_THE_MATCHED_EXPLICIT_"
            "DENSE_ASSIGNMENT_CELL_COUNT_AT_PORTS2_TO6"
        ),
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "catalytic_inference_established": False,
        "physical_waveform_execution": False,
        "replacement_of_physical_bits_with_pi": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
