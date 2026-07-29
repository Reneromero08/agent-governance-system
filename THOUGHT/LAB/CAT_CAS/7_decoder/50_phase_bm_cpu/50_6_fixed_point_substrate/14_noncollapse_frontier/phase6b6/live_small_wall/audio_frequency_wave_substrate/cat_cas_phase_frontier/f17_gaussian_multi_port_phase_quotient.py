#!/usr/bin/env python3
"""Exact F17 Gaussian quotient for multi-port cyclotomic phase programs.

The carrier represents the normalized phase function

    psi(q) = sign * 17**(-n/2)
             * zeta17**(q.T A q / 2 + b.T q + c)

without materializing any of its 17**n amplitudes. Symmetric phase shears and
single-coordinate Fourier transforms close exactly on ``(A,b,c,sign)`` when
the transformed pivot is nonzero. Public compilation derives pivots and
inverse order from the declared program and initial coefficient law only.

This is an exact compact classical Gaussian recurrence as well as a phase
quotient. It is not evidence of a distinct phase resource.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np


PRIME = 17
INV2 = pow(2, -1, PRIME)
TESTED_PORTS = (2, 4, 8, 16, 32)
PRIMARY_TAG = 1
REUSE_TAG = 6


def fail(message: str) -> None:
    raise RuntimeError(message)


def mod(value: int) -> int:
    return int(value) % PRIME


def inverse(value: int) -> int:
    reduced = mod(value)
    if reduced == 0:
        fail("attempted inversion of zero in F17")
    return pow(reduced, -1, PRIME)


def legendre(value: int) -> int:
    reduced = mod(value)
    if reduced == 0:
        return 0
    symbol = pow(reduced, (PRIME - 1) // 2, PRIME)
    return 1 if symbol == 1 else -1


def matrix_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.int64).copy() % PRIME
    rows, columns = work.shape
    rank = 0
    for column in range(columns):
        pivot = next(
            (
                row
                for row in range(rank, rows)
                if int(work[row, column]) % PRIME != 0
            ),
            None,
        )
        if pivot is None:
            continue
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
        work[rank] = (
            work[rank] * inverse(int(work[rank, column]))
        ) % PRIME
        for row in range(rows):
            if row == rank:
                continue
            factor = int(work[row, column])
            if factor:
                work[row] = (work[row] - factor * work[rank]) % PRIME
        rank += 1
        if rank == rows:
            break
    return rank


def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.int64) % PRIME
    if source.shape[0] != source.shape[1]:
        fail("matrix inverse requested for nonsquare matrix")
    size = int(source.shape[0])
    work = np.concatenate(
        (source.copy(), np.eye(size, dtype=np.int64)),
        axis=1,
    )
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if int(work[row, column]) % PRIME != 0
            ),
            None,
        )
        if pivot is None:
            fail("singular final Gaussian projection matrix")
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
        work[column] = (
            work[column] * inverse(int(work[column, column]))
        ) % PRIME
        for row in range(size):
            if row == column:
                continue
            factor = int(work[row, column])
            if factor:
                work[row] = (
                    work[row] - factor * work[column]
                ) % PRIME
    return work[:, size:] % PRIME


def determinant(
    matrix: np.ndarray,
    stats: "Stats | None" = None,
) -> int:
    if stats is not None:
        stats.determinant_search_calls += 1
    work = np.asarray(matrix, dtype=np.int64).copy() % PRIME
    size = int(work.shape[0])
    result = 1
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if int(work[row, column]) % PRIME != 0
            ),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
            result = -result
        pivot_value = int(work[column, column])
        result = mod(result * pivot_value)
        pivot_inverse = inverse(pivot_value)
        if stats is not None:
            stats.determinant_search_field_inversions += 1
        for row in range(column + 1, size):
            factor = mod(int(work[row, column]) * pivot_inverse)
            if stats is not None:
                stats.determinant_search_field_update_equivalents += 1
            if factor:
                work[row, column:] = (
                    work[row, column:]
                    - factor * work[column, column:]
                ) % PRIME
                if stats is not None:
                    stats.determinant_search_field_update_equivalents += (
                        size - column
                    )
    return mod(result)


@dataclass(frozen=True)
class QuadraticTerm:
    left: int
    right: int
    value: int


@dataclass(frozen=True)
class LinearTerm:
    port: int
    value: int


@dataclass(frozen=True)
class Module:
    quadratic: tuple[QuadraticTerm, ...]
    linear: tuple[LinearTerm, ...]
    constant: int
    fourier_port: int


@dataclass
class Stats:
    shear_updates: int = 0
    fourier_updates: int = 0
    field_multiply_adds: int = 0
    field_inversions: int = 0
    compiler_field_multiply_adds: int = 0
    compiler_field_inversions: int = 0
    determinant_search_calls: int = 0
    determinant_search_field_update_equivalents: int = 0
    determinant_search_field_inversions: int = 0
    maximum_temporary_field_cells: int = 0


@dataclass
class Carrier:
    quadratic: np.ndarray
    linear: np.ndarray
    constant: int
    sign: int
    generation: int = 0
    lease: int = 1
    baseline_reload_bytes: int = 0
    retained_inverse_history_bytes: int = 0

    def clone(self) -> "Carrier":
        return Carrier(
            quadratic=self.quadratic.copy(),
            linear=self.linear.copy(),
            constant=self.constant,
            sign=self.sign,
            generation=self.generation,
            lease=self.lease,
            baseline_reload_bytes=self.baseline_reload_bytes,
            retained_inverse_history_bytes=self.retained_inverse_history_bytes,
        )

    def exact_state(self) -> tuple[object, ...]:
        return (
            tuple(int(value) for value in self.quadratic.flat),
            tuple(int(value) for value in self.linear),
            int(self.constant),
            int(self.sign),
            int(self.generation),
            int(self.lease),
            int(self.baseline_reload_bytes),
            int(self.retained_inverse_history_bytes),
        )

    def semantic_state(self) -> tuple[object, ...]:
        return (
            tuple(int(value) for value in self.quadratic.flat),
            tuple(int(value) for value in self.linear),
            int(self.constant),
            int(self.sign),
        )

    def scalar_slots(self) -> int:
        return int(self.quadratic.size + self.linear.size + 2)

    def f17_field_cells(self) -> int:
        return int(self.quadratic.size + self.linear.size + 1)

    def fixed_width_payload_bits(self) -> int:
        return 5 * self.f17_field_cells() + 1


def make_carrier(ports: int, identity: int = 0) -> Carrier:
    if ports < 2:
        fail("multi-port Gaussian carrier requires at least two ports")
    quadratic = np.eye(ports, dtype=np.int64)
    linear = np.asarray(
        [mod(3 * port + 5 * identity + 1) for port in range(ports)],
        dtype=np.int64,
    )
    return Carrier(
        quadratic=quadratic,
        linear=linear,
        constant=mod(7 * identity + ports),
        sign=1,
    )


def validate_module(module: Module, ports: int) -> None:
    if not -1 <= module.fourier_port < ports:
        fail("Fourier port outside public interface")
    seen_quadratic: set[tuple[int, int]] = set()
    for term in module.quadratic:
        if not (
            0 <= term.left <= term.right < ports
            and 0 <= term.value < PRIME
        ):
            fail("quadratic descriptor outside F17 interface")
        key = (term.left, term.right)
        if key in seen_quadratic:
            fail("duplicate quadratic descriptor")
        seen_quadratic.add(key)
    seen_linear: set[int] = set()
    for term in module.linear:
        if not 0 <= term.port < ports or not 0 <= term.value < PRIME:
            fail("linear descriptor outside F17 interface")
        if term.port in seen_linear:
            fail("duplicate linear descriptor")
        seen_linear.add(term.port)
    if not 0 <= module.constant < PRIME:
        fail("constant descriptor outside F17")


def apply_shear(
    carrier: Carrier,
    module: Module,
    sign: int,
    stats: Stats,
) -> None:
    for term in module.quadratic:
        value = mod(sign * term.value)
        left = term.left
        right = term.right
        carrier.quadratic[left, right] = mod(
            int(carrier.quadratic[left, right]) + value
        )
        if left != right:
            carrier.quadratic[right, left] = mod(
                int(carrier.quadratic[right, left]) + value
            )
        stats.shear_updates += 1
    for term in module.linear:
        carrier.linear[term.port] = mod(
            int(carrier.linear[term.port]) + sign * term.value
        )
        stats.shear_updates += 1
    carrier.constant = mod(carrier.constant + sign * module.constant)
    stats.shear_updates += 1


def apply_fourier(
    carrier: Carrier,
    port: int,
    kernel_sign: int,
    stats: Stats,
) -> None:
    size = int(carrier.linear.size)
    pivot = int(carrier.quadratic[port, port])
    if pivot == 0:
        fail("public Fourier module reached a zero Gaussian pivot")
    pivot_inverse = inverse(pivot)
    stats.field_inversions += 1
    others = [index for index in range(size) if index != port]
    vector = carrier.quadratic[port, others].copy()
    bias = int(carrier.linear[port])
    updated = carrier.quadratic.copy()
    updated_linear = carrier.linear.copy()
    stats.maximum_temporary_field_cells = max(
        stats.maximum_temporary_field_cells,
        int(updated.size + vector.size + updated_linear.size),
    )
    for left_offset, left in enumerate(others):
        for right_offset, right in enumerate(others):
            updated[left, right] = mod(
                int(carrier.quadratic[left, right])
                - pivot_inverse
                * int(vector[left_offset])
                * int(vector[right_offset])
            )
            stats.field_multiply_adds += 1
    updated[port, port] = mod(-pivot_inverse)
    for offset, other in enumerate(others):
        cross = mod(
            -kernel_sign * pivot_inverse * int(vector[offset])
        )
        updated[port, other] = cross
        updated[other, port] = cross
        stats.field_multiply_adds += 1
    for offset, other in enumerate(others):
        updated_linear[other] = mod(
            int(carrier.linear[other])
            - pivot_inverse * bias * int(vector[offset])
        )
        stats.field_multiply_adds += 1
    updated_linear[port] = mod(
        -kernel_sign * pivot_inverse * bias
    )
    carrier.constant = mod(
        carrier.constant - INV2 * pivot_inverse * bias * bias
    )
    carrier.sign *= legendre(mod(pivot * INV2))
    carrier.quadratic[:] = updated
    carrier.linear[:] = updated_linear
    stats.fourier_updates += 1
    stats.field_multiply_adds += 3


def apply_module(
    carrier: Carrier,
    module: Module,
    stats: Stats,
) -> None:
    validate_module(module, int(carrier.linear.size))
    apply_shear(carrier, module, 1, stats)
    if module.fourier_port >= 0:
        apply_fourier(carrier, module.fourier_port, 1, stats)


def inverse_module(
    carrier: Carrier,
    module: Module,
    stats: Stats,
) -> None:
    if module.fourier_port >= 0:
        apply_fourier(carrier, module.fourier_port, -1, stats)
    apply_shear(carrier, module, -1, stats)


def reordered_inverse_module(
    carrier: Carrier,
    module: Module,
    stats: Stats,
) -> None:
    apply_shear(carrier, module, -1, stats)
    if module.fourier_port >= 0:
        apply_fourier(carrier, module.fourier_port, -1, stats)


def module_json(module: Module) -> dict[str, object]:
    return {
        "quadratic": [
            [term.left, term.right, term.value]
            for term in module.quadratic
        ],
        "linear": [
            [term.port, term.value] for term in module.linear
        ],
        "constant": module.constant,
        "fourier_port": module.fourier_port,
    }


def encoded_program(program: list[Module]) -> bytes:
    return json.dumps(
        [module_json(module) for module in program],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compile_program(
    ports: int,
    tag: int,
) -> tuple[list[Module], Stats]:
    scratch = make_carrier(ports)
    stats = Stats()
    program: list[Module] = []
    for step in range(ports + 2):
        target = (3 * step + tag) % ports
        desired_pivot = 1 + ((5 * step + 2 * tag) % (PRIME - 1))
        diagonal = mod(
            desired_pivot - int(scratch.quadratic[target, target])
        )
        neighbor = (target + 1 + (step % (ports - 1))) % ports
        if neighbor == target:
            neighbor = (neighbor + 1) % ports
        left, right = sorted((target, neighbor))
        quadratic = [
            QuadraticTerm(target, target, diagonal),
            QuadraticTerm(
                left,
                right,
                1 + ((7 * step + tag) % (PRIME - 1)),
            ),
        ]
        if ports > 3:
            second = (target + 2 + tag) % ports
            if second not in (target, neighbor):
                left2, right2 = sorted((target, second))
                quadratic.append(
                    QuadraticTerm(
                        left2,
                        right2,
                        1 + ((11 * step + tag) % (PRIME - 1)),
                    )
                )
        module = Module(
            quadratic=tuple(quadratic),
            linear=(
                LinearTerm(
                    target,
                    1 + ((13 * step + tag) % (PRIME - 1)),
                ),
                LinearTerm(
                    neighbor,
                    1 + ((3 * step + 2 * tag) % (PRIME - 1)),
                ),
            ),
            constant=mod(5 * step + tag),
            fourier_port=target,
        )
        apply_module(scratch, module, stats)
        program.append(module)
    diagonal_shift = next(
        (
            shift
            for shift in range(PRIME)
            if determinant(
                (
                    scratch.quadratic
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
        fail("public compiler found no nonsingular final scalar shift")
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
    program.append(final_module)
    stats.compiler_field_multiply_adds = stats.field_multiply_adds
    stats.compiler_field_inversions = stats.field_inversions
    return program, stats


def project_boundary(carrier: Carrier) -> dict[str, int]:
    size = int(carrier.linear.size)
    rank = matrix_rank(carrier.quadratic)
    if rank != size:
        fail("final public Gaussian boundary was singular")
    inverse_quadratic = matrix_inverse(carrier.quadratic)
    completed = int(
        carrier.linear
        @ inverse_quadratic
        @ carrier.linear
    )
    phase_exponent = mod(carrier.constant - INV2 * completed)
    determinant_value = determinant(carrier.quadratic)
    gaussian_sign = legendre(
        determinant_value * pow(INV2, size, PRIME)
    )
    return {
        "overlap_probability_numerator": 1,
        "overlap_probability_denominator_base": PRIME,
        "overlap_probability_denominator_power": size,
        "root_order": PRIME,
        "root_exponent": phase_exponent,
        "real_sign": carrier.sign * gaussian_sign,
        "quadratic_rank": rank,
    }


@dataclass
class Transaction:
    boundary: dict[str, int]
    restored_exactly: bool
    same_quadratic_backing: bool
    same_linear_backing: bool
    stats: Stats


def transaction(
    carrier: Carrier,
    baseline: Carrier,
    program: list[Module],
    control: str = "correct",
) -> Transaction:
    stats = Stats()
    quadratic_backing = id(carrier.quadratic)
    linear_backing = id(carrier.linear)
    for module in program:
        apply_module(carrier, module, stats)
    final = project_boundary(carrier)
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
    return Transaction(
        boundary=final,
        restored_exactly=restored,
        same_quadratic_backing=id(carrier.quadratic) == quadratic_backing,
        same_linear_backing=id(carrier.linear) == linear_backing,
        stats=stats,
    )


def run_case(ports: int) -> dict[str, object]:
    primary_program, primary_compile = compile_program(
        ports,
        PRIMARY_TAG,
    )
    reuse_program, reuse_compile = compile_program(ports, REUSE_TAG)
    baseline = make_carrier(ports)
    carrier = baseline.clone()
    carrier_identity = id(carrier)
    primary = transaction(carrier, baseline, primary_program)
    if (
        not primary.restored_exactly
        or not primary.same_quadratic_backing
        or not primary.same_linear_backing
    ):
        fail(f"primary exact restoration failed at {ports} ports")
    reuse = transaction(carrier, baseline, reuse_program)
    fresh = make_carrier(ports)
    fresh_run = transaction(fresh, baseline, reuse_program)
    if (
        not reuse.restored_exactly
        or not fresh_run.restored_exactly
        or reuse.boundary != fresh_run.boundary
        or id(carrier) != carrier_identity
        or carrier.generation != 2
        or carrier.baseline_reload_bytes != 0
    ):
        fail(f"exact restored reuse failed at {ports} ports")
    primary_encoded = encoded_program(primary_program)
    reuse_encoded = encoded_program(reuse_program)
    scalar_slots = carrier.scalar_slots()
    field_cells = carrier.f17_field_cells()
    machine_metadata_scalar_slots = 4
    return {
        "ports": ports,
        "primary_modules": len(primary_program),
        "reuse_modules": len(reuse_program),
        "primary_program_sha256": hashlib.sha256(
            primary_encoded
        ).hexdigest(),
        "reuse_program_sha256": hashlib.sha256(
            reuse_encoded
        ).hexdigest(),
        "public_primary_program": [
            module_json(module) for module in primary_program
        ],
        "public_reuse_program": [
            module_json(module) for module in reuse_program
        ],
        "primary_program_descriptor_bytes": len(primary_encoded),
        "reuse_program_descriptor_bytes": len(reuse_encoded),
        "primary_boundary": primary.boundary,
        "reuse_boundary": reuse.boundary,
        "fresh_restored_reuse_boundary_equal": (
            reuse.boundary == fresh_run.boundary
        ),
        "primary_restored_exactly": primary.restored_exactly,
        "reuse_restored_exactly": reuse.restored_exactly,
        "same_logical_carrier_container": id(carrier) == carrier_identity,
        "same_quadratic_backing": primary.same_quadratic_backing,
        "same_linear_backing": primary.same_linear_backing,
        "restoration_generation": carrier.generation,
        "resident_f17_field_cells": field_cells,
        "resident_sign_bits": 1,
        "resident_algebraic_scalar_slots": scalar_slots,
        "resident_machine_metadata_scalar_slots": (
            machine_metadata_scalar_slots
        ),
        "resident_total_scalar_slots": (
            scalar_slots + machine_metadata_scalar_slots
        ),
        "resident_fixed_width_payload_bits": (
            carrier.fixed_width_payload_bits()
        ),
        "accepted_path_baseline_plus_carrier_algebraic_scalar_slots": (
            2 * scalar_slots
        ),
        "accepted_path_baseline_plus_carrier_total_scalar_slots": (
            2 * (scalar_slots + machine_metadata_scalar_slots)
        ),
        "verification_peak_three_carrier_total_scalar_slots": (
            3 * (scalar_slots + machine_metadata_scalar_slots)
        ),
        "maximum_execution_temporary_field_cells": max(
            primary.stats.maximum_temporary_field_cells,
            reuse.stats.maximum_temporary_field_cells,
        ),
        "maximum_execution_temporary_index_slots": ports - 1,
        "projection_temporary_field_cells_upper_bound": (
            5 * ports * ports + ports
        ),
        "primary_compile_scratch_algebraic_scalar_slots": scalar_slots,
        "primary_compile_program_simulation_field_update_equivalents": (
            primary_compile.compiler_field_multiply_adds
        ),
        "primary_compile_program_simulation_field_inversions": (
            primary_compile.compiler_field_inversions
        ),
        "primary_compile_determinant_search_calls": (
            primary_compile.determinant_search_calls
        ),
        "primary_compile_determinant_search_field_update_equivalents": (
            primary_compile.determinant_search_field_update_equivalents
        ),
        "primary_compile_determinant_search_field_inversions": (
            primary_compile.determinant_search_field_inversions
        ),
        "primary_compile_determinant_search_temporary_field_cells_upper_bound": (
            4 * ports * ports
        ),
        "reuse_compile_scratch_algebraic_scalar_slots": scalar_slots,
        "reuse_compile_program_simulation_field_update_equivalents": (
            reuse_compile.compiler_field_multiply_adds
        ),
        "reuse_compile_program_simulation_field_inversions": (
            reuse_compile.compiler_field_inversions
        ),
        "reuse_compile_determinant_search_calls": (
            reuse_compile.determinant_search_calls
        ),
        "reuse_compile_determinant_search_field_update_equivalents": (
            reuse_compile.determinant_search_field_update_equivalents
        ),
        "reuse_compile_determinant_search_field_inversions": (
            reuse_compile.determinant_search_field_inversions
        ),
        "reuse_compile_determinant_search_temporary_field_cells_upper_bound": (
            4 * ports * ports
        ),
        "primary_execution_field_update_equivalents": (
            primary.stats.field_multiply_adds
        ),
        "primary_execution_field_inversions": (
            primary.stats.field_inversions
        ),
        "dense_phase_cells_not_materialized": PRIME**ports,
        "retained_inverse_history_bytes": (
            carrier.retained_inverse_history_bytes
        ),
        "baseline_reload_bytes": carrier.baseline_reload_bytes,
    }


def controls(ports: int) -> dict[str, object]:
    program, _ = compile_program(ports, PRIMARY_TAG)
    result: dict[str, object] = {}
    for name in ("missing", "wrong", "reordered"):
        baseline = make_carrier(ports)
        carrier = baseline.clone()
        try:
            run = transaction(
                carrier,
                baseline,
                program,
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
        fail("usage: f17_gaussian_multi_port_phase_quotient.py")
    cases = [run_case(ports) for ports in TESTED_PORTS]
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_EXACT_F17_GAUSSIAN_CYCLOTOMIC_MULTI_PORT_"
            "PHASE_QUOTIENT_WITH_NONCOMMUTING_FOURIER_SHEAR_"
            "CLOSURE_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_NUMPY_EXACT_F17_QUADRATIC_"
            "GAUSSIAN_PHASE_CHART_PORTS2_4_8_16_32_TWO_PUBLIC_"
            "ALGORITHMIC_PROGRAM_FAMILIES_NONZERO_FOURIER_PIVOTS_"
            "NONSINGULAR_FINAL_OVERLAP_BOUNDARY_SOFTWARE_ONLY"
        ),
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "carrier": {
            "field": "F17",
            "phase_root_order": PRIME,
            "state": "SYMMETRIC_QUADRATIC_A_LINEAR_B_CONSTANT_C_SIGN",
            "tested_ports": list(TESTED_PORTS),
            "dense_phase_vector_materialized": False,
            "assignment_enumeration_materialized": False,
            "truth_table_materialized": False,
            "inverse_order": "REVERSE_PUBLIC_MODULES",
            "inverse_descriptors_retained": 0,
        },
        "cases": cases,
        "controls": controls(8),
        "resource_law": {
            "resident_algebraic_scalar_slots_formula": (
                "ports*ports+ports+2"
            ),
            "resident_machine_metadata_scalar_slots": 4,
            "accepted_path_resident_carriers": 2,
            "verification_peak_resident_carriers": 3,
            "resident_f17_field_cells_formula": "ports*ports+ports+1",
            "resident_fixed_width_payload_bits_formula": (
                "5*(ports*ports+ports+1)+1"
            ),
            "compiled_program_modules_formula": "ports+3",
            "execution_work_formula": "O(program_modules*ports*ports)",
            "projection_work_formula": "O(ports_cubed)",
            "projection_scratch_field_cells_formula": "O(ports*ports)",
            "projection_temporary_field_cells_upper_bound_formula": (
                "5*ports*ports+ports"
            ),
            "retained_inverse_history_bytes": 0,
            "dense_phase_cells_materialized": 0,
            "compiler_simulates_public_coefficient_topology": True,
            "compiler_determinant_search_counted": True,
            "compiler_determinant_search_scratch_upper_bound_formula": (
                "4*ports*ports"
            ),
            "compiler_boundary_inputs": 0,
            "compiler_answer_inputs": 0,
            "serialized_program_descriptor_bytes_counted": True,
            "python_program_object_allocator_peak_bounded": False,
            "numpy_python_allocator_os_process_peak_bounded": False,
        },
        "matched_compact_classical": {
            "identical_exact_gaussian_coefficient_recurrence_exists": True,
            "verification_level": "PACKAGE_SELF_REVIEW",
            "strongest_compact_classical_established": False,
        },
        "phase_primitive_is_symbolic_f17_exponent_not_physical_wave": True,
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
