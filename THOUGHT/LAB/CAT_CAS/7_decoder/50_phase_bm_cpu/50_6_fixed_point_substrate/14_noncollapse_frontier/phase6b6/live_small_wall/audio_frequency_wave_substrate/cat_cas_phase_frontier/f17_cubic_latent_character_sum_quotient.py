#!/usr/bin/env python3
"""Exact F17 cubic-latent character-sum phase quotient.

The resident open phase relation is

    psi(q) = sign * 17**(-(n+1)/2) * sum_x zeta17**P(q, x)

where ``P`` is quadratic in the public port coordinates ``q`` and cubic in
one unresolved shared latent coordinate ``x``:

    P(q,x) = q.T A q / 2 + b.T q + c
             + x * l.T q + a*x**3 + d*x**2/2 + e*x.

Phase shears and Fourier transforms on the public ports close exactly on
``(A,b,l,c,a,d,e,sign)`` while the shared ``x`` remains unresolved. Final
projection eliminates the public Gaussian block and performs one fixed
17-term character trace. No ``17**n`` port assignment tensor is materialized.

The same coefficient law is an exact compact classical recurrence. This
software result is not evidence of a distinct physical phase resource.
"""

from __future__ import annotations

import hashlib
import json
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


TESTED_PORTS = (2, 4, 8, 16, 32)
PRIMARY_TAG = 2
REUSE_TAG = 7


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class CubicCarrier:
    quadratic: np.ndarray
    linear: np.ndarray
    latent_coupling: np.ndarray
    constant: int
    latent_cubic: int
    latent_quadratic: int
    latent_linear: int
    sign: int
    generation: int = 0
    lease: int = 1
    baseline_reload_bytes: int = 0
    retained_inverse_history_bytes: int = 0

    def clone(self) -> "CubicCarrier":
        return CubicCarrier(
            quadratic=self.quadratic.copy(),
            linear=self.linear.copy(),
            latent_coupling=self.latent_coupling.copy(),
            constant=self.constant,
            latent_cubic=self.latent_cubic,
            latent_quadratic=self.latent_quadratic,
            latent_linear=self.latent_linear,
            sign=self.sign,
            generation=self.generation,
            lease=self.lease,
            baseline_reload_bytes=self.baseline_reload_bytes,
            retained_inverse_history_bytes=self.retained_inverse_history_bytes,
        )

    def semantic_state(self) -> tuple[object, ...]:
        return (
            tuple(int(value) for value in self.quadratic.flat),
            tuple(int(value) for value in self.linear),
            tuple(int(value) for value in self.latent_coupling),
            int(self.constant),
            int(self.latent_cubic),
            int(self.latent_quadratic),
            int(self.latent_linear),
            int(self.sign),
        )

    def f17_field_cells(self) -> int:
        return int(
            self.quadratic.size
            + self.linear.size
            + self.latent_coupling.size
            + 4
        )

    def algebraic_scalar_slots(self) -> int:
        return self.f17_field_cells() + 1

    def fixed_width_payload_bits(self) -> int:
        return 5 * self.f17_field_cells() + 1


def make_carrier(ports: int, identity: int = 0) -> CubicCarrier:
    if ports < 2:
        fail("cubic-latent carrier requires at least two public ports")
    return CubicCarrier(
        quadratic=np.eye(ports, dtype=np.int64),
        linear=np.asarray(
            [mod(3 * port + 5 * identity + 1) for port in range(ports)],
            dtype=np.int64,
        ),
        latent_coupling=np.asarray(
            [1 + ((7 * port + 3 * identity) % (PRIME - 1))
             for port in range(ports)],
            dtype=np.int64,
        ),
        constant=mod(ports + 7 * identity),
        latent_cubic=1 + ((5 * identity) % (PRIME - 1)),
        latent_quadratic=mod(4 + 3 * identity),
        latent_linear=mod(6 + 2 * identity),
        sign=1,
    )


def apply_shear(
    carrier: CubicCarrier,
    module: Module,
    direction: int,
    stats: Stats,
) -> None:
    for term in module.quadratic:
        value = mod(direction * term.value)
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
            int(carrier.linear[term.port])
            + direction * term.value
        )
        stats.shear_updates += 1
    carrier.constant = mod(
        carrier.constant + direction * module.constant
    )
    stats.shear_updates += 1


def apply_fourier(
    carrier: CubicCarrier,
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
    latent = int(carrier.latent_coupling[port])
    updated = carrier.quadratic.copy()
    updated_linear = carrier.linear.copy()
    updated_latent = carrier.latent_coupling.copy()
    stats.maximum_temporary_field_cells = max(
        stats.maximum_temporary_field_cells,
        int(
            updated.size
            + vector.size
            + updated_linear.size
            + updated_latent.size
        ),
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
        updated_linear[other] = mod(
            int(carrier.linear[other])
            - pivot_inverse * bias * int(vector[offset])
        )
        updated_latent[other] = mod(
            int(carrier.latent_coupling[other])
            - pivot_inverse * latent * int(vector[offset])
        )
        stats.field_multiply_adds += 2
    updated_linear[port] = mod(
        -kernel_sign * pivot_inverse * bias
    )
    updated_latent[port] = mod(
        -kernel_sign * pivot_inverse * latent
    )
    carrier.constant = mod(
        carrier.constant - INV2 * pivot_inverse * bias * bias
    )
    carrier.latent_quadratic = mod(
        carrier.latent_quadratic
        - pivot_inverse * latent * latent
    )
    carrier.latent_linear = mod(
        carrier.latent_linear
        - pivot_inverse * bias * latent
    )
    carrier.sign *= legendre(mod(pivot * INV2))
    carrier.quadratic[:] = updated
    carrier.linear[:] = updated_linear
    carrier.latent_coupling[:] = updated_latent
    stats.fourier_updates += 1
    stats.field_multiply_adds += 6


def apply_module(
    carrier: CubicCarrier,
    module: Module,
    stats: Stats,
) -> bool:
    validate_module(module, int(carrier.linear.size))
    apply_shear(carrier, module, 1, stats)
    if module.fourier_port < 0:
        return False
    consumes_latent = (
        int(carrier.latent_coupling[module.fourier_port]) != 0
    )
    apply_fourier(carrier, module.fourier_port, 1, stats)
    return consumes_latent


def inverse_module(
    carrier: CubicCarrier,
    module: Module,
    stats: Stats,
) -> None:
    if module.fourier_port >= 0:
        apply_fourier(carrier, module.fourier_port, -1, stats)
    apply_shear(carrier, module, -1, stats)


def reordered_inverse_module(
    carrier: CubicCarrier,
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
    latent_consuming_port_mask: int


def compile_program(ports: int, tag: int) -> CompiledProgram:
    scratch = make_carrier(ports)
    stats = Stats()
    modules: list[Module] = []
    latent_consuming_modules = 0
    latent_consuming_port_mask = 0
    for step in range(ports + 2):
        candidate_count = sum(
            int(scratch.latent_coupling[port]) != 0
            for port in range(ports)
        )
        if candidate_count == 0:
            fail("public compiler lost the nonzero shared latent coupling")
        selected_ordinal = (3 * step + tag) % candidate_count
        target = -1
        for port in range(ports):
            if int(scratch.latent_coupling[port]) == 0:
                continue
            if selected_ordinal == 0:
                target = port
                break
            selected_ordinal -= 1
        if target < 0:
            fail("public compiler failed deterministic consumer selection")
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
        if not apply_module(scratch, module, stats):
            fail("compiled Fourier module did not consume shared latent")
        latent_consuming_modules += 1
        latent_consuming_port_mask |= 1 << target
        modules.append(module)
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
    modules.append(final_module)
    stats.compiler_field_multiply_adds = stats.field_multiply_adds
    stats.compiler_field_inversions = stats.field_inversions
    return CompiledProgram(
        modules=modules,
        stats=stats,
        latent_consuming_modules=latent_consuming_modules,
        latent_consuming_port_mask=latent_consuming_port_mask,
    )


def project_boundary(carrier: CubicCarrier) -> dict[str, object]:
    size = int(carrier.linear.size)
    rank = matrix_rank(carrier.quadratic)
    if rank != size:
        fail("final public Gaussian block was singular")
    inverse_quadratic = matrix_inverse(carrier.quadratic)
    b_ab = int(
        carrier.linear @ inverse_quadratic @ carrier.linear
    )
    b_al = int(
        carrier.linear
        @ inverse_quadratic
        @ carrier.latent_coupling
    )
    l_al = int(
        carrier.latent_coupling
        @ inverse_quadratic
        @ carrier.latent_coupling
    )
    reduced_constant = mod(carrier.constant - INV2 * b_ab)
    reduced_quadratic = mod(carrier.latent_quadratic - l_al)
    reduced_linear = mod(carrier.latent_linear - b_al)
    determinant_value = determinant(carrier.quadratic)
    gaussian_sign = legendre(
        determinant_value * pow(INV2, size, PRIME)
    )
    histogram = [0] * PRIME
    for latent in range(PRIME):
        exponent = mod(
            carrier.latent_cubic * latent**3
            + INV2 * reduced_quadratic * latent**2
            + reduced_linear * latent
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
        "normalization_denominator_sqrt_power": size + 1,
        "canonical_cyclotomic_coefficients": canonical,
        "canonical_nonzero_coefficients": sum(
            value != 0 for value in canonical
        ),
        "canonical_l1_coefficient_weight": sum(
            abs(value) for value in canonical
        ),
        "latent_trace_field_evaluations": PRIME,
        "public_gaussian_rank": rank,
    }


@dataclass
class Transaction:
    boundary: dict[str, object]
    restored_exactly: bool
    same_quadratic_backing: bool
    same_linear_backing: bool
    same_latent_coupling_backing: bool
    stats: Stats


def transaction(
    carrier: CubicCarrier,
    baseline: CubicCarrier,
    program: list[Module],
    control: str = "correct",
) -> Transaction:
    stats = Stats()
    quadratic_backing = id(carrier.quadratic)
    linear_backing = id(carrier.linear)
    latent_backing = id(carrier.latent_coupling)
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
    return Transaction(
        boundary=boundary,
        restored_exactly=restored,
        same_quadratic_backing=(
            id(carrier.quadratic) == quadratic_backing
        ),
        same_linear_backing=id(carrier.linear) == linear_backing,
        same_latent_coupling_backing=(
            id(carrier.latent_coupling) == latent_backing
        ),
        stats=stats,
    )


def compiler_metrics(compiled: CompiledProgram, ports: int) -> dict[str, int]:
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
            4 * ports * ports
        ),
        "compiler_scratch_algebraic_scalar_slots": (
            ports * ports + 2 * ports + 5
        ),
        "latent_consumer_mask_bits": ports,
    }


def run_case(ports: int) -> dict[str, object]:
    primary_compiled = compile_program(ports, PRIMARY_TAG)
    reuse_compiled = compile_program(ports, REUSE_TAG)
    baseline = make_carrier(ports)
    carrier = baseline.clone()
    carrier_identity = id(carrier)
    primary = transaction(
        carrier,
        baseline,
        primary_compiled.modules,
    )
    if (
        not primary.restored_exactly
        or not primary.same_quadratic_backing
        or not primary.same_linear_backing
        or not primary.same_latent_coupling_backing
    ):
        fail(f"primary cubic-latent restoration failed at {ports}")
    reuse = transaction(
        carrier,
        baseline,
        reuse_compiled.modules,
    )
    fresh = make_carrier(ports)
    fresh_reuse = transaction(
        fresh,
        baseline,
        reuse_compiled.modules,
    )
    if (
        not reuse.restored_exactly
        or not fresh_reuse.restored_exactly
        or reuse.boundary != fresh_reuse.boundary
        or id(carrier) != carrier_identity
        or not reuse.same_quadratic_backing
        or not reuse.same_linear_backing
        or not reuse.same_latent_coupling_backing
        or carrier.generation != 2
        or carrier.lease != 3
        or carrier.baseline_reload_bytes != 0
    ):
        fail(f"cubic-latent restored reuse failed at {ports}")
    primary_encoded = encoded_program(primary_compiled.modules)
    reuse_encoded = encoded_program(reuse_compiled.modules)
    algebraic_slots = carrier.algebraic_scalar_slots()
    machine_metadata_slots = 4
    numpy_array_payload_bytes = 8 * (ports * ports + 2 * ports)
    primary_boundary_bytes = len(
        json.dumps(
            primary.boundary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    reuse_boundary_bytes = len(
        json.dumps(
            reuse.boundary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "ports": ports,
        "primary_modules": len(primary_compiled.modules),
        "reuse_modules": len(reuse_compiled.modules),
        "primary_latent_consuming_modules": (
            primary_compiled.latent_consuming_modules
        ),
        "primary_latent_consuming_unique_ports": (
            primary_compiled.latent_consuming_port_mask.bit_count()
        ),
        "reuse_latent_consuming_modules": (
            reuse_compiled.latent_consuming_modules
        ),
        "reuse_latent_consuming_unique_ports": (
            reuse_compiled.latent_consuming_port_mask.bit_count()
        ),
        "primary_program_sha256": hashlib.sha256(
            primary_encoded
        ).hexdigest(),
        "reuse_program_sha256": hashlib.sha256(
            reuse_encoded
        ).hexdigest(),
        "public_primary_program": [
            module_json(module)
            for module in primary_compiled.modules
        ],
        "public_reuse_program": [
            module_json(module)
            for module in reuse_compiled.modules
        ],
        "primary_program_descriptor_bytes": len(primary_encoded),
        "reuse_program_descriptor_bytes": len(reuse_encoded),
        "primary_boundary_descriptor_bytes": primary_boundary_bytes,
        "reuse_boundary_descriptor_bytes": reuse_boundary_bytes,
        "boundary_cyclotomic_coefficient_slots": PRIME - 1,
        "boundary_cyclotomic_signed_bit_width_upper_bound": 6,
        "primary_boundary": primary.boundary,
        "reuse_boundary": reuse.boundary,
        "fresh_restored_reuse_boundary_equal": (
            reuse.boundary == fresh_reuse.boundary
        ),
        "primary_restored_exactly": primary.restored_exactly,
        "reuse_restored_exactly": reuse.restored_exactly,
        "same_logical_carrier_container": id(carrier) == carrier_identity,
        "same_quadratic_backing": primary.same_quadratic_backing,
        "same_linear_backing": primary.same_linear_backing,
        "same_latent_coupling_backing": (
            primary.same_latent_coupling_backing
        ),
        "reuse_same_quadratic_backing": (
            reuse.same_quadratic_backing
        ),
        "reuse_same_linear_backing": reuse.same_linear_backing,
        "reuse_same_latent_coupling_backing": (
            reuse.same_latent_coupling_backing
        ),
        "restoration_generation": carrier.generation,
        "restoration_lease": carrier.lease,
        "resident_f17_field_cells": carrier.f17_field_cells(),
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
        "verification_peak_three_carrier_numpy_array_payload_bytes": (
            3 * numpy_array_payload_bytes
        ),
        "maximum_execution_temporary_field_cells": max(
            primary.stats.maximum_temporary_field_cells,
            reuse.stats.maximum_temporary_field_cells,
        ),
        "maximum_execution_temporary_index_slots": ports - 1,
        "projection_temporary_field_cells_upper_bound": (
            5 * ports * ports + 3 * ports + 2 * PRIME - 1
        ),
        "primary_compile": compiler_metrics(primary_compiled, ports),
        "reuse_compile": compiler_metrics(reuse_compiled, ports),
        "primary_execution_field_update_equivalents": (
            primary.stats.field_multiply_adds
        ),
        "primary_execution_field_inversions": (
            primary.stats.field_inversions
        ),
        "dense_public_port_phase_cells_not_materialized": PRIME**ports,
        "latent_trace_field_evaluations": PRIME,
        "retained_inverse_history_bytes": (
            carrier.retained_inverse_history_bytes
        ),
        "baseline_reload_bytes": carrier.baseline_reload_bytes,
    }


def controls(ports: int) -> dict[str, object]:
    compiled = compile_program(ports, PRIMARY_TAG)
    result: dict[str, object] = {}
    for name in ("missing", "wrong", "reordered"):
        baseline = make_carrier(ports)
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
        fail("usage: f17_cubic_latent_character_sum_quotient.py")
    cases = [run_case(ports) for ports in TESTED_PORTS]
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_EXACT_F17_SINGLE_CUBIC_LATENT_CHARACTER_SUM_"
            "MULTI_PORT_PHASE_QUOTIENT_WITH_NATIVE_FOURIER_SHEAR_"
            "CLOSURE_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_NUMPY_EXACT_F17_ONE_UNRESOLVED_"
            "CUBIC_LATENT_PUBLIC_PORTS2_4_8_16_32_TWO_PUBLIC_"
            "ALGORITHMIC_PROGRAM_FAMILIES_NONZERO_GAUSSIAN_"
            "FOURIER_PIVOTS_NONSINGULAR_PUBLIC_GAUSSIAN_FINAL_"
            "BLOCK_FIXED17_TERM_LATENT_TRACE_SOFTWARE_ONLY"
        ),
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "carrier": {
            "field": "F17",
            "phase_root_order": PRIME,
            "state": (
                "PUBLIC_QUADRATIC_A_LINEAR_B_SHARED_LATENT_"
                "COUPLING_L_CUBIC_AUXILIARY_POLYNOMIAL_SIGN"
            ),
            "tested_ports": list(TESTED_PORTS),
            "unresolved_shared_latent_coordinates": 1,
            "latent_projected_during_forward_composition": False,
            "dense_public_port_phase_vector_materialized": False,
            "public_port_assignment_enumeration_materialized": False,
            "relation_table_materialized": False,
            "inverse_order": "REVERSE_PUBLIC_MODULES",
            "inverse_descriptors_retained": 0,
        },
        "cases": cases,
        "controls": controls(8),
        "resource_law": {
            "resident_f17_field_cells_formula": (
                "ports*ports+2*ports+4"
            ),
            "resident_fixed_width_payload_bits_formula": (
                "5*(ports*ports+2*ports+4)+1"
            ),
            "resident_numpy_int64_array_payload_bytes_formula": (
                "8*(ports*ports+2*ports)"
            ),
            "semantic_fixed_width_payload_is_not_process_memory": True,
            "compiled_program_modules_formula": "ports+3",
            "execution_work_formula": (
                "O(program_modules*ports*ports)"
            ),
            "projection_work_formula": "O(ports_cubed+17)",
            "projection_temporary_field_cells_upper_bound_formula": (
                "5*ports*ports+3*ports+33"
            ),
            "latent_trace_field_evaluations": PRIME,
            "latent_trace_scales_with_ports": False,
            "accepted_path_resident_carriers": 2,
            "verification_peak_resident_carriers": 3,
            "retained_inverse_history_bytes": 0,
            "dense_public_port_phase_cells_materialized": 0,
            "compiler_simulates_public_coefficient_state": True,
            "compiler_scratch_algebraic_scalar_slots_formula": (
                "ports*ports+2*ports+5"
            ),
            "compiler_latent_consumer_witness_list_materialized": False,
            "compiler_candidate_list_materialized": False,
            "compiler_latent_consumer_mask_bits": "ports",
            "compiler_boundary_inputs": 0,
            "compiler_answer_inputs": 0,
            "compiler_determinant_search_counted": True,
            "serialized_program_descriptor_bytes_counted": True,
            "serialized_boundary_descriptor_bytes_counted": True,
            "boundary_cyclotomic_coefficient_slots": PRIME - 1,
            "boundary_cyclotomic_signed_bit_width_upper_bound": 6,
            "python_program_object_allocator_peak_bounded": False,
            "numpy_python_allocator_os_process_peak_bounded": False,
        },
        "composition_advance": {
            "strictly_broader_than_quadratic_gaussian_phase_chart": True,
            "latent_cubic_third_finite_difference_mod17": 6,
            "latent_cubic_third_finite_difference_nonzero": True,
            "unresolved_cubic_latent_retained_across_composition": True,
            "multiple_native_fourier_consumers": True,
            "latent_trace_only_at_final_boundary": True,
            "non_gaussian_character_sum_boundary": True,
            "multiple_cubic_latents": False,
            "arbitrary_non_gaussian_closure": False,
            "necklace_carrier_integration": False,
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
            "identical_exact_cubic_latent_coefficient_recurrence_exists": (
                True
            ),
            "verification_level": "PACKAGE_SELF_REVIEW",
            "strongest_compact_classical_established": False,
        },
        "phase_primitive_is_symbolic_f17_character_sum_not_physical_wave": (
            True
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
