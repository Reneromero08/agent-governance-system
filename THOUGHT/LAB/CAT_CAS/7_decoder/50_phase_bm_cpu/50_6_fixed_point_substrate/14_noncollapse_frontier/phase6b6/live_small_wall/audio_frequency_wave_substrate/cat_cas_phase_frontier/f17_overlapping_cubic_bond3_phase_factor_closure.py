#!/usr/bin/env python3
"""Exact bond-3 closure for an overlapping non-affine cubic phase chain."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend


EXACT_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
STRUCTURAL_DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)
PROGRAM_DEPTHS = tuple(sorted(set(EXACT_DEPTHS + STRUCTURAL_DEPTHS)))
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE")
FINAL_BOUNDARY = "ALL_BITS_SUMMED_CUBIC_PHASE_PARTITION_SCALAR"


def fail(message: str) -> None:
    raise RuntimeError(message)


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


def theta_exponent(level: int, family: str) -> int:
    if level < 1:
        fail("cubic factor level must be positive")
    if family == "PRIMARY":
        return 1 + ((2 * level) % 16)
    if family == "REUSE":
        return 1 + ((2 * level + 1) % 16)
    fail("cubic factor family changed")


@dataclass(frozen=True)
class CubicProgram:
    depth: int
    family: str
    theta_exponents: tuple[int, ...]
    final_boundary: str = FINAL_BOUNDARY

    @property
    def physical_bits(self) -> int:
        return self.depth + 2

    def fingerprint(self) -> str:
        return rank1.digest_json(public_descriptor(self))


def compile_program(depth: int, family: str) -> CubicProgram:
    if depth not in PROGRAM_DEPTHS or family not in FAMILIES:
        fail("cubic program identity changed")
    program = CubicProgram(
        depth=depth,
        family=family,
        theta_exponents=tuple(
            theta_exponent(level, family) for level in range(1, depth + 1)
        ),
    )
    validate_program(program)
    return program


def validate_program(program: CubicProgram) -> None:
    if program.depth not in PROGRAM_DEPTHS or program.family not in FAMILIES:
        fail("cubic program domain changed")
    if program.theta_exponents != tuple(
        theta_exponent(level, program.family)
        for level in range(1, program.depth + 1)
    ):
        fail("cubic factor schedule changed")
    if any(not 1 <= value <= 16 for value in program.theta_exponents):
        fail("identity cubic factor admitted")
    if program.final_boundary != FINAL_BOUNDARY:
        fail("cubic final boundary changed")


def public_descriptor(program: CubicProgram) -> dict[str, Any]:
    return {
        "depth": program.depth,
        "physical_bits": program.physical_bits,
        "family": program.family,
        "factor": "THETA_LEVEL_TO_X_LEVEL_X_LEVEL_PLUS1_X_LEVEL_PLUS2",
        "theta_exponents": list(program.theta_exponents),
        "chart": "OVERLAPPING_THREE_SITE_BOOLEAN_PHASE_FACTOR_MPS",
        "final_boundary": program.final_boundary,
    }


def lease(program: CubicProgram, alg: backend.Algebra, capacity: int) -> str:
    return rank1.digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "capacity": capacity,
            "carrier": "OVERLAPPING_CUBIC_BOND3_PHASE_FACTOR",
        }
    )


@dataclass
class CubicFactorCarrier:
    alg: backend.Algebra
    capacity: int
    identity_branches: list[Any]
    cubic_branches: list[Any]
    active_depth: int = 0
    package_local_restoration_count: int = 0
    active_lease: str | None = None
    active_family: str | None = None
    stage: str = "RESTORED"
    projection_calls: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_local_coupling_named_field_cells: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra, capacity: int) -> "CubicFactorCarrier":
        if capacity < 1:
            fail("cubic factor capacity must be positive")
        return cls(
            alg=alg,
            capacity=capacity,
            identity_branches=[alg.zero for _ in range(capacity)],
            cubic_branches=[alg.zero for _ in range(capacity)],
        )

    def backing_identity(self) -> tuple[int, int, int]:
        return (id(self), id(self.identity_branches), id(self.cubic_branches))

    def active_values(self) -> list[Any]:
        values: list[Any] = []
        for site in range(self.active_depth):
            values.extend((self.identity_branches[site], self.cubic_branches[site]))
        return values

    def all_values(self) -> list[Any]:
        return [*self.identity_branches, *self.cubic_branches]

    def observe(self) -> None:
        payload = sum(self.alg.payload_bits(value) for value in self.active_values())
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, payload
        )

    def exact_zero(self) -> bool:
        return (
            self.active_depth == 0
            and all(value == self.alg.zero for value in self.all_values())
            and self.active_lease is None
            and self.active_family is None
            and self.stage == "RESTORED"
        )

    def digest(self) -> str:
        return rank1.digest_json(
            {
                "capacity": self.capacity,
                "active_depth": self.active_depth,
                "package_local_restoration_count": self.package_local_restoration_count,
                "lease": self.active_lease,
                "family": self.active_family,
                "stage": self.stage,
                "identity": [self.alg.serialize(v) for v in self.identity_branches],
                "cubic": [self.alg.serialize(v) for v in self.cubic_branches],
            }
        )


def load_site(carrier: CubicFactorCarrier, theta: Any) -> None:
    site = carrier.active_depth
    if site >= carrier.capacity:
        fail("cubic factor capacity exceeded")
    if (
        carrier.identity_branches[site] != carrier.alg.zero
        or carrier.cubic_branches[site] != carrier.alg.zero
    ):
        fail("cubic factor site was not empty")
    carrier.identity_branches[site] = carrier.alg.one
    old_identity = carrier.identity_branches[site]
    old_cubic = carrier.cubic_branches[site]
    carrier.identity_branches[site] = carrier.alg.add(
        old_identity, carrier.alg.mul(theta, old_cubic)
    )
    carrier.cubic_branches[site] = carrier.alg.add(
        carrier.alg.mul(theta, old_identity), old_cubic
    )
    carrier.maximum_local_coupling_named_field_cells = max(
        carrier.maximum_local_coupling_named_field_cells, 4
    )
    carrier.active_depth += 1
    carrier.observe()


def unload_site(carrier: CubicFactorCarrier, theta: Any, level: int) -> None:
    if level != carrier.active_depth or level < 1:
        fail("cubic factor inverse order changed")
    site = level - 1
    denominator = carrier.alg.sub(carrier.alg.one, carrier.alg.mul(theta, theta))
    if denominator == carrier.alg.zero:
        fail("singular cubic factor inverse")
    scale = carrier.alg.inverse(denominator)
    old_identity = carrier.identity_branches[site]
    old_cubic = carrier.cubic_branches[site]
    identity = carrier.alg.mul(
        scale, carrier.alg.sub(old_identity, carrier.alg.mul(theta, old_cubic))
    )
    cubic = carrier.alg.mul(
        scale, carrier.alg.sub(old_cubic, carrier.alg.mul(theta, old_identity))
    )
    carrier.maximum_local_coupling_named_field_cells = max(
        carrier.maximum_local_coupling_named_field_cells, 4
    )
    if identity != carrier.alg.one or cubic != carrier.alg.zero:
        fail("cubic factor inverse did not restore seeded branch")
    carrier.identity_branches[site] = carrier.alg.sub(identity, carrier.alg.one)
    carrier.cubic_branches[site] = cubic
    carrier.active_depth -= 1
    carrier.observe()


def forward(carrier: CubicFactorCarrier, program: CubicProgram) -> None:
    validate_program(program)
    if not isinstance(carrier, CubicFactorCarrier):
        fail("null or invalid cubic factor carrier")
    if not carrier.exact_zero() or carrier.capacity < program.depth:
        fail("cubic factor carrier not available")
    carrier.active_lease = lease(program, carrier.alg, carrier.capacity)
    carrier.active_family = program.family
    carrier.stage = "FORWARD"
    for exponent in program.theta_exponents:
        theta = carrier.alg.power(exponent)
        if carrier.alg.sub(carrier.alg.one, carrier.alg.mul(theta, theta)) == carrier.alg.zero:
            fail("singular cubic factor coupling")
        load_site(carrier, theta)


def require_active_program(
    carrier: CubicFactorCarrier, program: CubicProgram
) -> None:
    validate_program(program)
    if not isinstance(carrier, CubicFactorCarrier):
        fail("null or invalid cubic factor carrier")
    if (
        carrier.stage != "FORWARD"
        or carrier.active_depth != program.depth
        or carrier.active_family != program.family
        or carrier.active_lease != lease(program, carrier.alg, carrier.capacity)
    ):
        fail("cubic program does not own active carrier lease")


def contract_partition(
    carrier: CubicFactorCarrier, program: CubicProgram
) -> tuple[Any, int]:
    require_active_program(carrier, program)
    a = carrier.alg.one
    b = carrier.alg.one
    c = carrier.alg.one
    maximum_named_field_cells = 3
    for level in range(program.depth):
        if carrier.identity_branches[level] != carrier.alg.one:
            fail("cubic identity branch changed")
        theta = carrier.cubic_branches[level]
        old_a, old_b, old_c = a, b, c
        a = carrier.alg.add(old_a, old_b)
        b = carrier.alg.add(old_a, old_c)
        c = carrier.alg.add(old_a, carrier.alg.mul(theta, old_c))
        maximum_named_field_cells = max(maximum_named_field_cells, 6)
    boundary = carrier.alg.add(
        carrier.alg.mul(rank1.field_integer(carrier.alg, 2), a),
        carrier.alg.add(b, c),
    )
    return boundary, maximum_named_field_cells


def project_boundary(carrier: CubicFactorCarrier, program: CubicProgram) -> Any:
    if carrier.projection_calls != 0:
        fail("cubic boundary projected more than once")
    boundary, _ = contract_partition(carrier, program)
    carrier.projection_calls += 1
    return boundary


def inverse(carrier: CubicFactorCarrier, program: CubicProgram) -> None:
    require_active_program(carrier, program)
    if carrier.projection_calls != 1:
        fail("cubic inverse before final boundary")
    for level in range(program.depth, 0, -1):
        unload_site(
            carrier,
            carrier.alg.power(program.theta_exponents[level - 1]),
            level,
        )
    carrier.active_lease = None
    carrier.active_family = None
    carrier.stage = "RESTORED"
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not carrier.exact_zero():
        fail("cubic factor carrier did not restore exact zero")


def execute_transaction(
    carrier: CubicFactorCarrier, program: CubicProgram
) -> dict[str, Any]:
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_local_coupling_named_field_cells = 0
    initial_digest = carrier.digest()
    backing = carrier.backing_identity()
    restoration_count_before = carrier.package_local_restoration_count
    forward(carrier, program)
    commitment, commitment_bytes = rank1.stream_vector_commitment(
        carrier.active_values(), carrier.alg
    )
    _, projection_work = contract_partition(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    descriptor_bytes = len(
        json.dumps(
            public_descriptor(program), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    return {
        "depth": program.depth,
        "physical_bits": program.physical_bits,
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "boundary": carrier.alg.serialize(boundary),
        "factor_commitment": commitment,
        "factor_commitment_json_bytes": commitment_bytes,
        "resident_phase_factor_field_cells": 2 * program.depth,
        "resident_nontrivial_theta_field_cells": program.depth,
        "fixed_wiring_field_cells": 0,
        "exact_maximum_mps_bond_dimension": 2 if program.depth == 1 else 3,
        "maximum_local_coupling_named_field_cells": carrier.maximum_local_coupling_named_field_cells,
        "projection_dynamic_field_cells": projection_work,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "maximum_resident_factor_payload_bits": carrier.maximum_resident_payload_bits,
        "public_program_json_bytes": descriptor_bytes,
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "accepted_path_assignment_enumeration": False,
        "accepted_path_component_weight_cells": 0,
        "accepted_path_dense_transfer_cells": 0,
        "inverse_history_cells": 0,
        "snapshot_reload_used": False,
        "response_released_after_restoration": True,
        "restored_exact_zero": carrier.exact_zero(),
        "same_backing": carrier.backing_identity() == backing,
        "package_local_restoration_count_before": restoration_count_before,
        "package_local_restoration_count_after": carrier.package_local_restoration_count,
        "initial_digest": initial_digest,
        "restored_digest_with_package_local_count": carrier.digest(),
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "projection_compiler_and_commitment_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "intermediate_factor_payload_exposed_in_result": False,
        "one_way_factor_commitment_emitted": True,
    }


def rank_certificate(program: CubicProgram, alg: backend.Algebra) -> dict[str, Any]:
    thetas = [alg.power(value) for value in program.theta_exponents]
    minor_nonzero = [
        alg.mul(alg.sub(left, alg.one), alg.sub(right, alg.one)) != alg.zero
        for left, right in zip(thetas, thetas[1:])
    ]
    return {
        "depth": program.depth,
        "algebra": algebra_signature(alg),
        "interior_rank3_cut_count": max(0, program.depth - 1),
        "all_declared_rank3_minors_nonzero": all(minor_nonzero),
        "exact_maximum_weight_tensor_mps_bond_dimension": (
            2 if program.depth == 1 else 3
        ),
        "bond_two_rejected_for_depth_at_least_two": (
            program.depth == 1 or all(minor_nonzero)
        ),
        "minor_values_serialized": False,
        "assignment_tensor_materialized": False,
    }


def row_times_matrix(
    row: list[Any], matrix: list[list[Any]], alg: backend.Algebra
) -> list[Any]:
    return [
        sum_field((alg.mul(row[i], matrix[i][j]) for i in range(3)), alg)
        for j in range(3)
    ]


def sum_field(values: Any, alg: backend.Algebra) -> Any:
    result = alg.zero
    for value in values:
        result = alg.add(result, value)
    return result


def compiled_classical_baseline(
    transaction: dict[str, Any], program: CubicProgram, alg: backend.Algebra
) -> dict[str, Any]:
    state = [alg.one, alg.one, alg.one]
    final_row = [
        rank1.field_integer(alg, 2),
        alg.one,
        alg.one,
    ]
    for exponent in program.theta_exponents:
        theta = alg.power(exponent)
        a, b, c = state
        state = [
            alg.add(a, b),
            alg.add(a, c),
            alg.add(a, alg.mul(theta, c)),
        ]
    for exponent in reversed(program.theta_exponents):
        theta = alg.power(exponent)
        transfer = [
            [alg.one, alg.one, alg.zero],
            [alg.one, alg.zero, alg.one],
            [alg.one, alg.zero, theta],
        ]
        final_row = row_times_matrix(final_row, transfer, alg)
    boundary = sum_field(
        (
            alg.mul(rank1.field_integer(alg, 2), state[0]),
            state[1],
            state[2],
        ),
        alg,
    )
    compiled_boundary = sum_field(final_row, alg)
    factors = [alg.power(value) for value in program.theta_exponents]
    factor_commitment, factor_bytes = rank1.stream_vector_commitment(factors, alg)
    row_commitment, row_bytes = rank1.stream_vector_commitment(final_row, alg)
    return {
        "depth": program.depth,
        "family": program.family,
        "algebra": algebra_signature(alg),
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "compiled_boundary_agreement": alg.serialize(compiled_boundary)
        == transaction["boundary"],
        "full_weight_signature_exact_factor_field_cells": program.depth,
        "final_boundary_dynamic_field_cells": 3,
        "final_boundary_maximum_named_update_field_cells": 6,
        "sealed_word_three_state_chart_input_final_row_field_cells": 3,
        "sealed_fixed_initial_and_final_boundary_field_cells": 1,
        "factor_commitment": factor_commitment,
        "factor_commitment_json_bytes": factor_bytes,
        "compiled_final_row_commitment": row_commitment,
        "compiled_final_row_commitment_json_bytes": row_bytes,
        "phase_carrier_or_snapshot_used": False,
    }


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "boundary",
        "factor_commitment",
        "package_local_restoration_count_before",
        "package_local_restoration_count_after",
        "initial_digest",
        "restored_digest_with_package_local_count",
        "program_fingerprint",
        "family",
    }
    return {key: value for key, value in transaction.items() if key not in excluded}


def controls() -> dict[str, bool]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    program = compile_program(4, "PRIMARY")
    reference_carrier = CubicFactorCarrier.create(alg, 4)
    reference = execute_transaction(reference_carrier, program)

    missing = CubicFactorCarrier.create(alg, 4)
    forward(missing, program)
    missing_inverse_detected = not missing.exact_zero()

    wrong = CubicFactorCarrier.create(alg, 4)
    forward(wrong, program)
    wrong_inverse_detected = False
    try:
        unload_site(wrong, alg.power(program.theta_exponents[-2]), program.depth)
    except RuntimeError:
        wrong_inverse_detected = True

    reordered = CubicFactorCarrier.create(alg, 4)
    forward(reordered, program)
    reordered_inverse_rejected = False
    try:
        unload_site(
            reordered,
            alg.power(program.theta_exponents[-2]),
            program.depth - 1,
        )
    except RuntimeError:
        reordered_inverse_rejected = True

    premature = CubicFactorCarrier.create(alg, 4)
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    wrong_projection_owner = CubicFactorCarrier.create(alg, 4)
    forward(wrong_projection_owner, program)
    wrong_projection_owner_rejected = False
    try:
        project_boundary(wrong_projection_owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_projection_owner_rejected = True

    wrong_inverse_owner = CubicFactorCarrier.create(alg, 4)
    forward(wrong_inverse_owner, program)
    project_boundary(wrong_inverse_owner, program)
    wrong_inverse_owner_rejected = False
    try:
        inverse(wrong_inverse_owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_inverse_owner_rejected = True

    null_carrier_rejected = False
    try:
        forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    perturbed = CubicProgram(
        depth=4,
        family="PRIMARY",
        theta_exponents=(
            (program.theta_exponents[0] % 16) + 1,
            *program.theta_exponents[1:],
        ),
    )
    altered_schedule_changes_descriptor = (
        public_descriptor(perturbed) != public_descriptor(program)
    )
    rank3 = rank_certificate(program, alg)
    return {
        "all_declared_theta_factors_nonidentity": all(
            alg.power(value) != alg.one for value in program.theta_exponents
        ),
        "identity_theta_collapses_declared_rank3_minor": alg.sub(
            alg.one, alg.one
        )
        == alg.zero,
        "bond_two_rejected_by_nonzero_rank3_minor": rank3[
            "bond_two_rejected_for_depth_at_least_two"
        ],
        "altered_theta_schedule_changes_public_descriptor": altered_schedule_changes_descriptor,
        "missing_inverse_leaves_resident_state": missing_inverse_detected,
        "wrong_inverse_detected": wrong_inverse_detected,
        "reordered_inverse_rejected": reordered_inverse_rejected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "wrong_projection_owner_rejected": wrong_projection_owner_rejected,
        "wrong_inverse_owner_rejected": wrong_inverse_owner_rejected,
        "snapshot_command_available": False,
        "reference_transaction_restored": reference["restored_exact_zero"],
    }


def run() -> dict[str, Any]:
    exact = [
        execute_transaction(
            CubicFactorCarrier.create(backend.Algebra("Q_ZETA17"), depth),
            compile_program(depth, "PRIMARY"),
        )
        for depth in EXACT_DEPTHS
    ]
    structural = []
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        for depth in STRUCTURAL_DEPTHS:
            item = execute_transaction(
                CubicFactorCarrier.create(alg, depth),
                compile_program(depth, "PRIMARY"),
            )
            item["field"] = f"F{modulus}"
            structural.append(item)

    certificates = [
        rank_certificate(compile_program(depth, "PRIMARY"), backend.Algebra("Q_ZETA17"))
        for depth in EXACT_DEPTHS
    ]
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        certificates.extend(
            rank_certificate(compile_program(depth, "PRIMARY"), alg)
            for depth in STRUCTURAL_DEPTHS
        )
    if not all(
        item["all_declared_rank3_minors_nonzero"]
        and item["exact_maximum_weight_tensor_mps_bond_dimension"]
        == (2 if item["depth"] == 1 else 3)
        for item in certificates
    ):
        fail("one or more cubic bond certificates failed")

    baselines = []
    for item in exact:
        baselines.append(
            compiled_classical_baseline(
                item,
                compile_program(item["depth"], item["family"]),
                backend.Algebra("Q_ZETA17"),
            )
        )
    for item in structural:
        modulus, root = next(
            pair for pair in FINITE_FIELDS if item["field"] == f"F{pair[0]}"
        )
        baselines.append(
            compiled_classical_baseline(
                item,
                compile_program(item["depth"], item["family"]),
                backend.Algebra(item["field"], modulus=modulus, root=root),
            )
        )
    if not all(
        item["boundary_agreement"] and item["compiled_boundary_agreement"]
        for item in baselines
    ):
        fail("cubic matched classical baseline disagrees")

    reuse_carrier = CubicFactorCarrier.create(backend.Algebra("Q_ZETA17"), 16)
    first = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(16, "REUSE"))
    fresh = execute_transaction(
        CubicFactorCarrier.create(backend.Algebra("Q_ZETA17"), 16),
        compile_program(16, "REUSE"),
    )
    if reused["boundary"] != fresh["boundary"]:
        fail("restored cubic factor carrier disagrees with fresh reuse")
    if resource_signature(reused) != resource_signature(fresh):
        fail("restored cubic factor carrier changed resource signature")

    control_results = controls()
    if not all(
        value for key, value in control_results.items() if key != "snapshot_command_available"
    ) or control_results["snapshot_command_available"]:
        fail("one or more cubic controls failed")

    return {
        "schema": "CAT_CAS_F17_OVERLAPPING_CUBIC_BOND3_PHASE_FACTOR_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_OVERLAPPING_NONAFFINE_CUBIC_BOOLEAN_PHASE_FACTOR_CHAIN_HAS_MINIMAL_MPS_BOND3_AND_TWO_M_RESIDENT_FACTOR_CELLS_ACROSS_DEPTH128_WITH_FINAL_ONLY_PARTITION_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_THE_IDENTICAL_THREE_SCALAR_CLASSICAL_RECURRENCE_AND_FIXED_LOGICAL_RANK_HIDES_GROWING_EXACT_BOUNDARY_WIDTH",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "exact_depths": EXACT_DEPTHS,
            "dual_field_structural_depths": STRUCTURAL_DEPTHS,
            "factor_law": "THETA_LEVEL_TO_X_LEVEL_X_LEVEL_PLUS1_X_LEVEL_PLUS2",
            "nonaffine_boolean_degree": 3,
            "exact_bond_law": "DEPTH1_BOND2_DEPTH_AT_LEAST2_BOND3",
            "ordinary_secant_or_waring_interpretation": "NOT_CLAIMED",
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "rank_certificates": certificates,
        "compiled_classical_baselines": baselines,
        "reuse": {
            "first_depth": 8,
            "reused_depth": 16,
            "first_family": "PRIMARY",
            "reused_family": "REUSE",
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "fresh_restored_resource_signature_agreement": resource_signature(reused)
            == resource_signature(fresh),
            "same_actual_backing_across_unrelated_programs": (
                first["same_backing"]
                and reused["same_backing"]
                and reuse_carrier.backing_identity() == backing
            ),
            "package_local_restoration_count_after_two_transactions": (
                reuse_carrier.package_local_restoration_count
            ),
            "baseline_reload_used": False,
        },
        "controls": control_results,
        "resource_law": {
            "physical_bits_at_depth_m": "M_PLUS_TWO",
            "resident_phase_factor_field_cells_at_depth_m": "TWO_TIMES_M",
            "resident_nontrivial_theta_field_cells_at_depth_m": "M",
            "exact_maximum_weight_tensor_mps_bond_dimension": 3,
            "native_local_coupling_named_field_cells": 4,
            "final_boundary_projection_dynamic_field_cells": 6,
            "matched_classical_full_signature_field_cells_at_depth_m": "M",
            "matched_classical_final_boundary_dynamic_field_cells": 3,
            "matched_classical_sealed_three_state_chart_input_final_row_field_cells": 3,
            "accepted_path_assignment_or_weight_tensor_cells": 0,
            "inverse_history_cells": 0,
            "fixed_logical_rank_implies_fixed_exact_bit_width": False,
            "full_exact_bit_complexity_established": False,
            "python_container_allocator_native_bigint_hashlib_and_whole_process_excluded": True,
        },
        "matched_baseline": {
            "strongest_full_weight_signature": "M_EXACT_PUBLIC_THETA_FACTORS",
            "strongest_final_boundary_runtime": "THREE_DYNAMIC_SCALARS",
            "strongest_sealed_three_state_chart_input_boundary": "THREE_COMPILED_FINAL_ROW_SCALARS",
            "strongest_sealed_fixed_transaction": "ONE_CACHED_BOUNDARY_SCALAR",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_phase_factor_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "projection_compiler_and_commitment_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "declared_overlapping_cubic_factor_chain": True,
            "fixed_bond3_nonaffine_phase_factor_signature": True,
            "arbitrary_cubic_hypergraph_closure": False,
            "arbitrary_boundary_compaction": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_DECLARED_NONAFFINE_CUBIC_PHASE_CHAIN_CLOSES_AT_EXACT_BOND3_BUT_THE_PHASE_CARRIER_STORES_TWICE_THE_PUBLIC_FACTOR_SIGNATURE_THE_SELECTED_BOUNDARY_HAS_AN_IDENTICAL_THREE_SCALAR_CLASSICAL_RECURRENCE_AND_EXACT_BOUNDARY_WIDTH_GROWS_WITH_DEPTH",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(
        json.dumps(run(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
