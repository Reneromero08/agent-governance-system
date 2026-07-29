#!/usr/bin/env python3
"""Exact F17 interacting-cubic-latent trace-growth diagnostic.

The open phase relation is

    psi(q) = sign * 17**(-(n+k)/2) * sum_x zeta17**P(q, x),

where ``q`` has ``n`` public coordinates, ``x`` has ``k`` unresolved latent
coordinates, and

    P(q,x) = q.T A q / 2 + b.T q + q.T L x + c
             + C3(x) + x.T D x / 2 + e.T x.

Public shears and Fourier transforms close exactly on the coefficient tuple.
The resident signature grows polynomially with ``k``, but the generic final
character trace in this diagnostic streams all ``17**k`` latent assignments.
That streamed work is disclosed as the obstruction; it is not represented as
compact multiple-latent closure.

The identical exact coefficient recurrence is a compact classical
implementation of the forward law. This software diagnostic does not
establish a distinct phase resource or computational advantage.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass

import numpy as np

from f17_gaussian_multi_port_phase_quotient import (
    INV2,
    PRIME,
    LinearTerm,
    Module,
    QuadraticTerm,
    Stats,
    determinant,
    encoded_program,
    inverse,
    legendre,
    matrix_inverse,
    matrix_rank,
    mod,
    module_json,
    validate_module,
)


FIXED_PORTS = 4
TESTED_LATENTS = (1, 2, 3, 4)
DENSE_ORACLE_CASE = (2, 2)
TEST_CASES = tuple((FIXED_PORTS, count) for count in TESTED_LATENTS) + (
    DENSE_ORACLE_CASE,
)
PRIMARY_TAG = 3
REUSE_TAG = 11


def fail(message: str) -> None:
    raise RuntimeError(message)


def quadratic_pairs(latents: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(latents)
        for right in range(left, latents)
    )


def cubic_triples(
    latents: int,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (first, second, third)
        for first in range(latents)
        for second in range(first, latents)
        for third in range(second, latents)
    )


@dataclass
class MultiLatentCarrier:
    public_quadratic: np.ndarray
    public_linear: np.ndarray
    latent_coupling: np.ndarray
    latent_quadratic: np.ndarray
    latent_linear: np.ndarray
    latent_cubic: np.ndarray
    constant: int
    sign: int
    generation: int = 0
    lease: int = 1
    baseline_reload_bytes: int = 0
    retained_inverse_history_bytes: int = 0

    @property
    def ports(self) -> int:
        return int(self.public_linear.size)

    @property
    def latents(self) -> int:
        return int(self.latent_linear.size)

    def clone(self) -> "MultiLatentCarrier":
        return MultiLatentCarrier(
            public_quadratic=self.public_quadratic.copy(),
            public_linear=self.public_linear.copy(),
            latent_coupling=self.latent_coupling.copy(),
            latent_quadratic=self.latent_quadratic.copy(),
            latent_linear=self.latent_linear.copy(),
            latent_cubic=self.latent_cubic.copy(),
            constant=self.constant,
            sign=self.sign,
            generation=self.generation,
            lease=self.lease,
            baseline_reload_bytes=self.baseline_reload_bytes,
            retained_inverse_history_bytes=self.retained_inverse_history_bytes,
        )

    def semantic_state(self) -> tuple[object, ...]:
        return (
            tuple(int(value) for value in self.public_quadratic.flat),
            tuple(int(value) for value in self.public_linear),
            tuple(int(value) for value in self.latent_coupling.flat),
            tuple(int(value) for value in self.latent_quadratic),
            tuple(int(value) for value in self.latent_linear),
            tuple(int(value) for value in self.latent_cubic),
            int(self.constant),
            int(self.sign),
        )

    def f17_field_cells(self) -> int:
        return int(
            self.public_quadratic.size
            + self.public_linear.size
            + self.latent_coupling.size
            + self.latent_quadratic.size
            + self.latent_linear.size
            + self.latent_cubic.size
            + 1
        )

    def algebraic_scalar_slots(self) -> int:
        return self.f17_field_cells() + 1

    def fixed_width_payload_bits(self) -> int:
        return 5 * self.f17_field_cells() + 1


def make_carrier(
    ports: int,
    latents: int,
    identity: int = 0,
) -> MultiLatentCarrier:
    if ports < 2 or latents < 1:
        fail("interacting cubic carrier requires ports>=2 and latents>=1")
    pairs = quadratic_pairs(latents)
    triples = cubic_triples(latents)
    coupling = np.asarray(
        [
            [
                1
                + (
                    (
                        7 * port
                        + 5 * latent
                        + 3 * identity
                    )
                    % (PRIME - 1)
                )
                for latent in range(latents)
            ]
            for port in range(ports)
        ],
        dtype=np.int64,
    )
    return MultiLatentCarrier(
        public_quadratic=np.eye(ports, dtype=np.int64),
        public_linear=np.asarray(
            [
                mod(3 * port + 5 * identity + 1)
                for port in range(ports)
            ],
            dtype=np.int64,
        ),
        latent_coupling=coupling,
        latent_quadratic=np.asarray(
            [
                1
                + (
                    (
                        5 * left
                        + 9 * right
                        + 3 * identity
                    )
                    % (PRIME - 1)
                )
                for left, right in pairs
            ],
            dtype=np.int64,
        ),
        latent_linear=np.asarray(
            [
                mod(6 + 7 * latent + 2 * identity)
                for latent in range(latents)
            ],
            dtype=np.int64,
        ),
        latent_cubic=np.asarray(
            [
                1
                + (
                    (
                        3 * first
                        + 5 * second
                        + 7 * third
                        + 5 * identity
                    )
                    % (PRIME - 1)
                )
                for first, second, third in triples
            ],
            dtype=np.int64,
        ),
        constant=mod(ports + 2 * latents + 7 * identity),
        sign=1,
    )


def apply_shear(
    carrier: MultiLatentCarrier,
    module: Module,
    direction: int,
    stats: Stats,
) -> None:
    for term in module.quadratic:
        value = mod(direction * term.value)
        left = term.left
        right = term.right
        carrier.public_quadratic[left, right] = mod(
            int(carrier.public_quadratic[left, right]) + value
        )
        if left != right:
            carrier.public_quadratic[right, left] = mod(
                int(carrier.public_quadratic[right, left]) + value
            )
        stats.shear_updates += 1
    for term in module.linear:
        carrier.public_linear[term.port] = mod(
            int(carrier.public_linear[term.port])
            + direction * term.value
        )
        stats.shear_updates += 1
    carrier.constant = mod(
        carrier.constant + direction * module.constant
    )
    stats.shear_updates += 1


def apply_fourier(
    carrier: MultiLatentCarrier,
    port: int,
    kernel_sign: int,
    stats: Stats,
) -> None:
    ports = carrier.ports
    latents = carrier.latents
    pivot = int(carrier.public_quadratic[port, port])
    if pivot == 0:
        fail("public Fourier module reached a zero Gaussian pivot")
    pivot_inverse = inverse(pivot)
    stats.field_inversions += 1
    others = [index for index in range(ports) if index != port]
    vector = carrier.public_quadratic[port, others].copy()
    bias = int(carrier.public_linear[port])
    latent_row = carrier.latent_coupling[port, :].copy()
    updated_quadratic = carrier.public_quadratic.copy()
    updated_linear = carrier.public_linear.copy()
    updated_coupling = carrier.latent_coupling.copy()
    updated_latent_quadratic = carrier.latent_quadratic.copy()
    stats.maximum_temporary_field_cells = max(
        stats.maximum_temporary_field_cells,
        int(
            updated_quadratic.size
            + updated_linear.size
            + updated_coupling.size
            + updated_latent_quadratic.size
            + vector.size
            + latent_row.size
        ),
    )
    for left_offset, left in enumerate(others):
        for right_offset, right in enumerate(others):
            updated_quadratic[left, right] = mod(
                int(carrier.public_quadratic[left, right])
                - pivot_inverse
                * int(vector[left_offset])
                * int(vector[right_offset])
            )
            stats.field_multiply_adds += 1
    updated_quadratic[port, port] = mod(-pivot_inverse)
    for offset, other in enumerate(others):
        cross = mod(
            -kernel_sign * pivot_inverse * int(vector[offset])
        )
        updated_quadratic[port, other] = cross
        updated_quadratic[other, port] = cross
        updated_linear[other] = mod(
            int(carrier.public_linear[other])
            - pivot_inverse * bias * int(vector[offset])
        )
        for latent in range(latents):
            updated_coupling[other, latent] = mod(
                int(carrier.latent_coupling[other, latent])
                - pivot_inverse
                * int(vector[offset])
                * int(latent_row[latent])
            )
            stats.field_multiply_adds += 1
        stats.field_multiply_adds += 2
    updated_linear[port] = mod(
        -kernel_sign * pivot_inverse * bias
    )
    for latent in range(latents):
        updated_coupling[port, latent] = mod(
            -kernel_sign
            * pivot_inverse
            * int(latent_row[latent])
        )
        carrier.latent_linear[latent] = mod(
            int(carrier.latent_linear[latent])
            - pivot_inverse * bias * int(latent_row[latent])
        )
        stats.field_multiply_adds += 2
    for index, (left, right) in enumerate(
        quadratic_pairs(latents)
    ):
        updated_latent_quadratic[index] = mod(
            int(carrier.latent_quadratic[index])
            - pivot_inverse
            * int(latent_row[left])
            * int(latent_row[right])
        )
        stats.field_multiply_adds += 1
    carrier.constant = mod(
        carrier.constant - INV2 * pivot_inverse * bias * bias
    )
    carrier.sign *= legendre(mod(pivot * INV2))
    carrier.public_quadratic[:] = updated_quadratic
    carrier.public_linear[:] = updated_linear
    carrier.latent_coupling[:] = updated_coupling
    carrier.latent_quadratic[:] = updated_latent_quadratic
    stats.fourier_updates += 1
    stats.field_multiply_adds += 2


def latent_mask(row: np.ndarray) -> int:
    result = 0
    for latent, value in enumerate(row):
        if int(value) != 0:
            result |= 1 << latent
    return result


def apply_module(
    carrier: MultiLatentCarrier,
    module: Module,
    stats: Stats,
) -> int:
    validate_module(module, carrier.ports)
    apply_shear(carrier, module, 1, stats)
    if module.fourier_port < 0:
        return 0
    consumed = latent_mask(
        carrier.latent_coupling[module.fourier_port, :]
    )
    apply_fourier(carrier, module.fourier_port, 1, stats)
    return consumed


def inverse_module(
    carrier: MultiLatentCarrier,
    module: Module,
    stats: Stats,
) -> None:
    if module.fourier_port >= 0:
        apply_fourier(carrier, module.fourier_port, -1, stats)
    apply_shear(carrier, module, -1, stats)


def reordered_inverse_module(
    carrier: MultiLatentCarrier,
    module: Module,
    stats: Stats,
) -> None:
    apply_shear(carrier, module, -1, stats)
    if module.fourier_port >= 0:
        apply_fourier(carrier, module.fourier_port, -1, stats)


@dataclass
class CompiledProgram:
    modules: list[Module]
    stats: Stats
    latent_consuming_modules: int
    latent_consumer_mask: int
    latent_consuming_port_mask: int
    candidate_row_masks_evaluated: int
    selection_row_masks_evaluated: int
    latent_mask_field_tests: int


def compile_program(
    ports: int,
    latents: int,
    tag: int,
) -> CompiledProgram:
    scratch = make_carrier(ports, latents)
    stats = Stats()
    modules: list[Module] = []
    consuming_modules = 0
    consumer_mask = 0
    port_mask = 0
    candidate_row_masks_evaluated = 0
    selection_row_masks_evaluated = 0
    latent_mask_field_tests = 0
    full_latent_mask = (1 << latents) - 1
    for step in range(ports + 2):
        candidate_count = sum(
            latent_mask(scratch.latent_coupling[port, :])
            == full_latent_mask
            for port in range(ports)
        )
        candidate_row_masks_evaluated += ports
        latent_mask_field_tests += ports * latents
        if candidate_count == 0:
            fail("compiler lost every full-latent public consumer")
        selected_ordinal = (5 * step + tag) % candidate_count
        target = -1
        for port in range(ports):
            selection_row_masks_evaluated += 1
            latent_mask_field_tests += latents
            if (
                latent_mask(scratch.latent_coupling[port, :])
                != full_latent_mask
            ):
                continue
            if selected_ordinal == 0:
                target = port
                break
            selected_ordinal -= 1
        if target < 0:
            fail("compiler failed deterministic full-latent selection")
        desired_pivot = 1 + (
            (7 * step + 3 * tag + latents) % (PRIME - 1)
        )
        diagonal = mod(
            desired_pivot
            - int(scratch.public_quadratic[target, target])
        )
        neighbor = (target + 1 + step) % ports
        if neighbor == target:
            neighbor = (neighbor + 1) % ports
        left, right = sorted((target, neighbor))
        module = Module(
            quadratic=(
                QuadraticTerm(target, target, diagonal),
                QuadraticTerm(
                    left,
                    right,
                    1 + (
                        (11 * step + tag + latents)
                        % (PRIME - 1)
                    ),
                ),
            ),
            linear=(
                LinearTerm(
                    target,
                    1 + (
                        (13 * step + tag) % (PRIME - 1)
                    ),
                ),
                LinearTerm(
                    neighbor,
                    1 + (
                        (3 * step + 2 * tag + latents)
                        % (PRIME - 1)
                    ),
                ),
            ),
            constant=mod(5 * step + tag + latents),
            fourier_port=target,
        )
        consumed = apply_module(scratch, module, stats)
        latent_mask_field_tests += latents
        if consumed != full_latent_mask:
            fail("compiled module did not consume every latent coordinate")
        consuming_modules += 1
        consumer_mask |= consumed
        port_mask |= 1 << target
        modules.append(module)
    diagonal_shift = next(
        (
            shift
            for shift in range(PRIME)
            if determinant(
                (
                    scratch.public_quadratic
                    + shift * np.eye(ports, dtype=np.int64)
                )
                % PRIME,
                stats,
            )
            != 0
        ),
        None,
    )
    if diagonal_shift is None:
        fail("compiler found no nonsingular final public shift")
    final_module = Module(
        quadratic=tuple(
            QuadraticTerm(port, port, diagonal_shift)
            for port in range(ports)
        ),
        linear=(),
        constant=0,
        fourier_port=-1,
    )
    apply_module(scratch, final_module, stats)
    modules.append(final_module)
    stats.compiler_field_multiply_adds = stats.field_multiply_adds
    stats.compiler_field_inversions = stats.field_inversions
    return CompiledProgram(
        modules=modules,
        stats=stats,
        latent_consuming_modules=consuming_modules,
        latent_consumer_mask=consumer_mask,
        latent_consuming_port_mask=port_mask,
        candidate_row_masks_evaluated=(
            candidate_row_masks_evaluated
        ),
        selection_row_masks_evaluated=(
            selection_row_masks_evaluated
        ),
        latent_mask_field_tests=latent_mask_field_tests,
    )


def evaluate_latent_phase(
    values: tuple[int, ...],
    latent_quadratic: list[int],
    latent_linear: list[int],
    latent_cubic: list[int],
) -> int:
    exponent = 0
    for coefficient, (left, right) in zip(
        latent_quadratic,
        quadratic_pairs(len(values)),
        strict=True,
    ):
        factor = INV2 if left == right else 1
        exponent += (
            factor * coefficient * values[left] * values[right]
        )
    for coefficient, value in zip(
        latent_linear,
        values,
        strict=True,
    ):
        exponent += coefficient * value
    for coefficient, (first, second, third) in zip(
        latent_cubic,
        cubic_triples(len(values)),
        strict=True,
    ):
        exponent += (
            coefficient
            * values[first]
            * values[second]
            * values[third]
        )
    return mod(exponent)


def project_boundary(
    carrier: MultiLatentCarrier,
) -> dict[str, object]:
    ports = carrier.ports
    latents = carrier.latents
    rank = matrix_rank(carrier.public_quadratic)
    if rank != ports:
        fail("final public Gaussian block was singular")
    inverse_quadratic = matrix_inverse(carrier.public_quadratic)
    inverse_linear = inverse_quadratic @ carrier.public_linear
    inverse_coupling = (
        inverse_quadratic @ carrier.latent_coupling
    )
    reduced_constant = mod(
        carrier.constant
        - INV2 * int(carrier.public_linear @ inverse_linear)
    )
    reduced_linear = [
        mod(
            int(carrier.latent_linear[latent])
            - int(
                carrier.public_linear
                @ inverse_coupling[:, latent]
            )
        )
        for latent in range(latents)
    ]
    reduced_quadratic = [
        mod(
            int(carrier.latent_quadratic[index])
            - int(
                carrier.latent_coupling[:, left]
                @ inverse_coupling[:, right]
            )
        )
        for index, (left, right) in enumerate(
            quadratic_pairs(latents)
        )
    ]
    cubic = [int(value) for value in carrier.latent_cubic]
    determinant_value = determinant(carrier.public_quadratic)
    gaussian_sign = legendre(
        determinant_value * pow(INV2, ports, PRIME)
    )
    histogram = [0] * PRIME
    trace_assignments = PRIME**latents
    for values in itertools.product(range(PRIME), repeat=latents):
        exponent = mod(
            evaluate_latent_phase(
                values,
                reduced_quadratic,
                reduced_linear,
                cubic,
            )
            + reduced_constant
        )
        histogram[exponent] += 1
    sign = carrier.sign * gaussian_sign
    canonical = [
        sign * (histogram[index] - histogram[PRIME - 1])
        for index in range(PRIME - 1)
    ]
    return {
        "root_order": PRIME,
        "normalization_denominator_base": PRIME,
        "normalization_denominator_sqrt_power": ports + latents,
        "canonical_cyclotomic_coefficients": canonical,
        "canonical_nonzero_coefficients": sum(
            value != 0 for value in canonical
        ),
        "canonical_l1_coefficient_weight": sum(
            abs(value) for value in canonical
        ),
        "canonical_signed_bit_width": max(
            (abs(value).bit_length() + 1 for value in canonical),
            default=1,
        ),
        "latent_trace_assignments_streamed": trace_assignments,
        "latent_trace_histogram_cells": PRIME,
        "latent_trace_histogram_integer_payload_bits": sum(
            max(1, int(value).bit_length()) for value in histogram
        ),
        "canonical_cyclotomic_integer_payload_bits": sum(
            max(1, abs(value).bit_length() + 1)
            for value in canonical
        ),
        "public_gaussian_rank": rank,
    }


@dataclass
class Transaction:
    boundary: dict[str, object]
    restored_exactly: bool
    same_backing: dict[str, bool]
    stats: Stats


def transaction(
    carrier: MultiLatentCarrier,
    baseline: MultiLatentCarrier,
    program: list[Module],
    control: str = "correct",
) -> Transaction:
    stats = Stats()
    backing = {
        "public_quadratic": id(carrier.public_quadratic),
        "public_linear": id(carrier.public_linear),
        "latent_coupling": id(carrier.latent_coupling),
        "latent_quadratic": id(carrier.latent_quadratic),
        "latent_linear": id(carrier.latent_linear),
        "latent_cubic": id(carrier.latent_cubic),
    }
    for module in program:
        apply_module(carrier, module, stats)
    boundary = project_boundary(carrier)
    first = 1 if control == "missing" else 0
    for cursor in range(len(program) - 1, first - 1, -1):
        module = program[cursor]
        if control == "wrong" and cursor == len(program) - 2:
            changed = list(module.quadratic)
            changed[0] = QuadraticTerm(
                changed[0].left,
                changed[0].right,
                mod(changed[0].value + 1),
            )
            module = Module(
                quadratic=tuple(changed),
                linear=module.linear,
                constant=module.constant,
                fourier_port=module.fourier_port,
            )
        if control == "reordered":
            reordered_inverse_module(carrier, module, stats)
        else:
            inverse_module(carrier, module, stats)
    restored = carrier.semantic_state() == baseline.semantic_state()
    if control == "correct":
        carrier.generation += 1
        carrier.lease += 1
    same_backing = {
        name: id(getattr(carrier, name)) == identity
        for name, identity in backing.items()
    }
    return Transaction(
        boundary=boundary,
        restored_exactly=restored,
        same_backing=same_backing,
        stats=stats,
    )


def compiler_metrics(
    compiled: CompiledProgram,
    carrier: MultiLatentCarrier,
) -> dict[str, object]:
    return {
        "program_simulation_field_update_equivalents": (
            compiled.stats.compiler_field_multiply_adds
        ),
        "program_simulation_field_inversions": (
            compiled.stats.compiler_field_inversions
        ),
        "determinant_search_calls": (
            compiled.stats.determinant_search_calls
        ),
        "determinant_search_field_update_equivalents": (
            compiled.stats.determinant_search_field_update_equivalents
        ),
        "determinant_search_field_inversions": (
            compiled.stats.determinant_search_field_inversions
        ),
        "determinant_search_temporary_field_cells_upper_bound": (
            4 * carrier.ports * carrier.ports
        ),
        "compiler_scratch_algebraic_scalar_slots": (
            carrier.algebraic_scalar_slots()
        ),
        "latent_consumer_mask_bits": carrier.latents,
        "public_consumer_port_mask_bits": carrier.ports,
        "candidate_list_materialized": False,
        "consumer_witness_list_materialized": False,
        "candidate_row_masks_evaluated": (
            compiled.candidate_row_masks_evaluated
        ),
        "selection_row_masks_evaluated": (
            compiled.selection_row_masks_evaluated
        ),
        "latent_mask_field_tests": (
            compiled.latent_mask_field_tests
        ),
        "public_module_descriptor_count": len(compiled.modules),
        "python_topology_tuple_allocator_peak_bounded": False,
    }


def run_case(ports: int, latents: int) -> dict[str, object]:
    baseline = make_carrier(ports, latents)
    carrier = baseline.clone()
    carrier_identity = id(carrier)
    primary_compiled = compile_program(
        ports,
        latents,
        PRIMARY_TAG,
    )
    reuse_compiled = compile_program(
        ports,
        latents,
        REUSE_TAG,
    )
    full_mask = (1 << latents) - 1
    if (
        primary_compiled.latent_consumer_mask != full_mask
        or reuse_compiled.latent_consumer_mask != full_mask
    ):
        fail("compiled programs did not consume every latent coordinate")
    primary = transaction(
        carrier,
        baseline,
        primary_compiled.modules,
    )
    if (
        not primary.restored_exactly
        or not all(primary.same_backing.values())
    ):
        fail(
            "primary interacting-latent restoration failed "
            f"at ports={ports}, latents={latents}"
        )
    reuse = transaction(
        carrier,
        baseline,
        reuse_compiled.modules,
    )
    fresh = make_carrier(ports, latents)
    fresh_reuse = transaction(
        fresh,
        baseline,
        reuse_compiled.modules,
    )
    if (
        not reuse.restored_exactly
        or not fresh_reuse.restored_exactly
        or not all(reuse.same_backing.values())
        or reuse.boundary != fresh_reuse.boundary
        or id(carrier) != carrier_identity
        or carrier.generation != 2
        or carrier.lease != 3
        or carrier.baseline_reload_bytes != 0
    ):
        fail(
            "interacting-latent reuse failed "
            f"at ports={ports}, latents={latents}"
        )
    primary_encoded = encoded_program(primary_compiled.modules)
    reuse_encoded = encoded_program(reuse_compiled.modules)
    field_cells = carrier.f17_field_cells()
    algebraic_slots = carrier.algebraic_scalar_slots()
    machine_metadata_slots = 4
    numpy_array_payload_bytes = 8 * (field_cells - 1)
    pair_count = math.comb(latents + 1, 2)
    cubic_count = math.comb(latents + 2, 3)
    interacting_cubic_count = cubic_count - latents
    trace_assignments = PRIME**latents
    projection_temporary_bound = (
        6 * ports * ports
        + 3 * ports * latents
        + 2 * ports
        + pair_count
        + 2 * latents
        + cubic_count
        + 2 * PRIME
        - 1
    )
    projection_f17_temporary_bound = (
        projection_temporary_bound - (2 * PRIME - 1)
    )
    primary_boundary_descriptor_bytes = len(
        json.dumps(
            primary.boundary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    reuse_boundary_descriptor_bytes = len(
        json.dumps(
            reuse.boundary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "ports": ports,
        "latents": latents,
        "primary_modules": len(primary_compiled.modules),
        "reuse_modules": len(reuse_compiled.modules),
        "primary_latent_consuming_modules": (
            primary_compiled.latent_consuming_modules
        ),
        "reuse_latent_consuming_modules": (
            reuse_compiled.latent_consuming_modules
        ),
        "primary_latent_consumer_mask": (
            primary_compiled.latent_consumer_mask
        ),
        "reuse_latent_consumer_mask": (
            reuse_compiled.latent_consumer_mask
        ),
        "primary_latent_consuming_unique_public_ports": (
            primary_compiled.latent_consuming_port_mask.bit_count()
        ),
        "reuse_latent_consuming_unique_public_ports": (
            reuse_compiled.latent_consuming_port_mask.bit_count()
        ),
        "public_primary_program": [
            module_json(module)
            for module in primary_compiled.modules
        ],
        "public_reuse_program": [
            module_json(module)
            for module in reuse_compiled.modules
        ],
        "primary_program_sha256": hashlib.sha256(
            primary_encoded
        ).hexdigest(),
        "reuse_program_sha256": hashlib.sha256(
            reuse_encoded
        ).hexdigest(),
        "primary_program_descriptor_bytes": len(primary_encoded),
        "reuse_program_descriptor_bytes": len(reuse_encoded),
        "primary_boundary_descriptor_bytes": (
            primary_boundary_descriptor_bytes
        ),
        "reuse_boundary_descriptor_bytes": (
            reuse_boundary_descriptor_bytes
        ),
        "primary_boundary": primary.boundary,
        "reuse_boundary": reuse.boundary,
        "primary_restored_exactly": primary.restored_exactly,
        "reuse_restored_exactly": reuse.restored_exactly,
        "fresh_restored_reuse_boundary_equal": (
            reuse.boundary == fresh_reuse.boundary
        ),
        "same_logical_carrier_container": id(carrier) == carrier_identity,
        "primary_same_array_backing": primary.same_backing,
        "reuse_same_array_backing": reuse.same_backing,
        "restoration_generation": carrier.generation,
        "restoration_lease": carrier.lease,
        "latent_quadratic_coefficient_cells": pair_count,
        "latent_cubic_coefficient_cells": cubic_count,
        "interacting_latent_cubic_coefficient_cells": (
            interacting_cubic_count
        ),
        "resident_f17_field_cells": field_cells,
        "resident_sign_bits": 1,
        "resident_algebraic_scalar_slots": algebraic_slots,
        "resident_machine_metadata_scalar_slots": machine_metadata_slots,
        "resident_total_scalar_slots": (
            algebraic_slots + machine_metadata_slots
        ),
        "resident_fixed_width_payload_bits": (
            carrier.fixed_width_payload_bits()
        ),
        "resident_numpy_int64_array_payload_bytes": (
            numpy_array_payload_bytes
        ),
        "accepted_path_baseline_plus_carrier_total_scalar_slots": (
            2 * (algebraic_slots + machine_metadata_slots)
        ),
        "accepted_path_baseline_plus_carrier_numpy_array_payload_bytes": (
            2 * numpy_array_payload_bytes
        ),
        "verification_peak_three_carrier_total_scalar_slots": (
            3 * (algebraic_slots + machine_metadata_slots)
        ),
        "maximum_execution_temporary_field_cells": max(
            primary.stats.maximum_temporary_field_cells,
            reuse.stats.maximum_temporary_field_cells,
        ),
        "maximum_execution_temporary_index_slots": ports - 1,
        "projection_temporary_scalar_slots_upper_bound": (
            projection_temporary_bound
        ),
        "projection_temporary_f17_field_cells_upper_bound": (
            projection_f17_temporary_bound
        ),
        "projection_temporary_integer_count_slots": 2 * PRIME - 1,
        "primary_projection_histogram_integer_payload_bits": (
            primary.boundary[
                "latent_trace_histogram_integer_payload_bits"
            ]
        ),
        "reuse_projection_histogram_integer_payload_bits": (
            reuse.boundary[
                "latent_trace_histogram_integer_payload_bits"
            ]
        ),
        "primary_boundary_cyclotomic_integer_payload_bits": (
            primary.boundary[
                "canonical_cyclotomic_integer_payload_bits"
            ]
        ),
        "reuse_boundary_cyclotomic_integer_payload_bits": (
            reuse.boundary[
                "canonical_cyclotomic_integer_payload_bits"
            ]
        ),
        "latent_trace_assignments_streamed": trace_assignments,
        "latent_trace_assignment_table_materialized": False,
        "latent_trace_ephemeral_index_slots": latents,
        "latent_trace_cubic_term_evaluations": (
            trace_assignments * cubic_count
        ),
        "latent_trace_total_polynomial_term_evaluations": (
            trace_assignments
            * (cubic_count + pair_count + latents)
        ),
        "latent_trace_constant_modulo_histogram_operations": (
            3 * trace_assignments
        ),
        "dense_public_phase_cells_not_materialized": (
            PRIME ** (ports + latents)
        ),
        "primary_compile": compiler_metrics(
            primary_compiled,
            baseline,
        ),
        "reuse_compile": compiler_metrics(
            reuse_compiled,
            baseline,
        ),
        "primary_execution_field_update_equivalents": (
            primary.stats.field_multiply_adds
        ),
        "primary_execution_field_inversions": (
            primary.stats.field_inversions
        ),
        "retained_inverse_history_bytes": (
            carrier.retained_inverse_history_bytes
        ),
        "baseline_reload_bytes": carrier.baseline_reload_bytes,
    }


def controls() -> dict[str, object]:
    ports = FIXED_PORTS
    latents = 3
    compiled = compile_program(ports, latents, PRIMARY_TAG)
    result: dict[str, object] = {}
    for name in ("missing", "wrong", "reordered"):
        baseline = make_carrier(ports, latents)
        carrier = baseline.clone()
        try:
            run = transaction(
                carrier,
                baseline,
                compiled.modules,
                control=name,
            )
        except RuntimeError as error:
            result[f"{name}_inverse_failure_mode"] = str(error)
            result[f"{name}_inverse_rejected"] = True
            continue
        if run.restored_exactly:
            fail(f"{name} inverse control unexpectedly restored")
        result[f"{name}_inverse_failure_mode"] = (
            "EXACT_SEMANTIC_STATE_MISMATCH"
        )
        result[f"{name}_inverse_rejected"] = True
    return result


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_interacting_cubic_latent_trace_diagnostic.py"
        )
    cases = [
        run_case(ports, latents)
        for ports, latents in TEST_CASES
    ]
    fixed_port_series = [
        case for case in cases if case["ports"] == FIXED_PORTS
    ]
    trace_counts = [
        int(case["latent_trace_assignments_streamed"])
        for case in fixed_port_series
    ]
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_EXACT_F17_INTERACTING_MULTI_CUBIC_LATENT_"
            "SIGNATURE_REMAINS_POLYNOMIAL_WHILE_FINAL_CHARACTER_"
            "TRACE_GROWS_AS_17_TO_K_WITH_EXACT_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_NUMPY_EXACT_F17_FIXED_PUBLIC_"
            "PORTS4_LATENTS1_2_3_4_PLUS_DENSE_ORACLE_CASE_PORTS2_"
            "LATENTS2_TWO_PUBLIC_ALGORITHMIC_PROGRAM_FAMILIES_"
            "FULL_MIXED_CUBIC_MONOMIAL_SIGNATURE_NONZERO_GAUSSIAN_"
            "FOURIER_PIVOTS_NONSINGULAR_PUBLIC_FINAL_BLOCK_STREAMED_"
            "17_TO_K_FINAL_TRACE_SOFTWARE_ONLY"
        ),
        "classification_candidate": (
            "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        ),
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "carrier": {
            "field": "F17",
            "phase_root_order": PRIME,
            "fixed_public_ports": FIXED_PORTS,
            "tested_latents": list(TESTED_LATENTS),
            "dense_oracle_case": {
                "ports": DENSE_ORACLE_CASE[0],
                "latents": DENSE_ORACLE_CASE[1],
            },
            "state": (
                "PUBLIC_QUADRATIC_LINEAR_LATENT_COUPLING_PACKED_"
                "LATENT_QUADRATIC_FULL_MIXED_CUBIC_LATENT_"
                "POLYNOMIAL_CONSTANT_SIGN"
            ),
            "latent_projected_during_forward_composition": False,
            "dense_public_latent_phase_tensor_materialized": False,
            "relation_table_materialized": False,
            "inverse_descriptors_retained": 0,
        },
        "cases": cases,
        "controls": controls(),
        "measured_obstruction": {
            "fixed_port_latent_counts": list(TESTED_LATENTS),
            "resident_f17_field_cells": [
                case["resident_f17_field_cells"]
                for case in fixed_port_series
            ],
            "latent_cubic_coefficient_cells": [
                case["latent_cubic_coefficient_cells"]
                for case in fixed_port_series
            ],
            "latent_trace_assignments_streamed": trace_counts,
            "successive_trace_growth_ratio": [
                trace_counts[index + 1] // trace_counts[index]
                for index in range(len(trace_counts) - 1)
            ],
            "resident_signature_polynomial_in_latent_count": True,
            "generic_final_trace_exponential_in_latent_count": True,
            "compact_multiple_latent_final_closure_established": False,
            "interaction_signature_is_current_primary_obstruction": False,
            "final_trace_is_current_primary_obstruction": True,
        },
        "resource_law": {
            "resident_f17_field_cells_formula": (
                "ports*ports+ports+ports*latents+"
                "latents*(latents+1)/2+latents+"
                "latents*(latents+1)*(latents+2)/6+1"
            ),
            "resident_fixed_width_payload_bits_formula": (
                "5*resident_f17_field_cells+1"
            ),
            "compiled_program_modules_formula": "ports+3",
            "execution_work_formula": (
                "O(program_modules*(ports*ports+ports*latents+"
                "latents*latents))"
            ),
            "projection_trace_work_formula": (
                "17**latents*(binomial(latents+2,3)+"
                "binomial(latents+1,2)+latents)"
            ),
            "projection_assignment_table_materialized": False,
            "projection_streams_every_latent_assignment": True,
            "projection_histogram_integer_payload_bits_counted": True,
            "projection_cyclotomic_integer_payload_bits_counted": True,
            "serialized_boundary_descriptor_bytes_counted": True,
            "accepted_path_resident_carriers": 2,
            "verification_peak_resident_carriers": 3,
            "retained_inverse_history_bytes": 0,
            "compiler_candidate_list_materialized": False,
            "compiler_consumer_witness_list_materialized": False,
            "compiler_boundary_inputs": 0,
            "compiler_answer_inputs": 0,
            "compiler_trace_evaluations": 0,
            "compiler_mask_scan_field_tests_counted": True,
            "compiler_python_topology_tuple_allocator_peak_bounded": (
                False
            ),
            "serialized_program_descriptor_bytes_counted": True,
            "python_program_object_allocator_peak_bounded": False,
            "numpy_python_allocator_os_process_peak_bounded": False,
        },
        "no_smuggle_scope": {
            "scope": "PACKAGE_SERIALIZATION_ONLY_NOT_MACHINE_ENFORCED",
            "machine_enforced_custody": False,
            "reduced_latent_polynomial_exposed": False,
            "latent_assignments_serialized": False,
            "intermediate_coefficient_states_serialized": False,
            "public_program_descriptors_serialized": True,
            "final_cyclotomic_boundary_only": True,
        },
        "matched_compact_classical": {
            "identical_exact_multi_latent_coefficient_recurrence_exists": (
                True
            ),
            "identical_generic_final_trace_work": True,
            "strongest_compact_classical_established": False,
        },
        "composition_scope": {
            "public_gaussian_shear_fourier_updates_native": True,
            "mixed_cubic_signature_preinitialized": True,
            "mixed_cubic_coefficients_changed_by_public_modules": False,
            "native_cubic_interaction_updates_established": False,
            "arbitrary_non_gaussian_composition_established": False,
        },
        "next_phase_owned_repair": (
            "TOPOLOGY_FACTORIZED_INTERACTING_CUBIC_LATENT_GRAPH_"
            "WITH_NATIVE_VARIABLE_ELIMINATION_OR_TRANSFER_CLOSURE_"
            "AVOIDING_17_TO_K_FINAL_TRACE"
        ),
        "catvm_custody_established": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "catalytic_inference_established": False,
        "physical_waveform_execution": False,
        "replacement_of_physical_bits_with_pi": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
