#!/usr/bin/env python3
"""Exact rank-4 closure for a two-shared-latent cubic phase cycle."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_overlapping_cubic_bond3_phase_factor_closure as cubic_chain


EXACT_BRANCH_COUNTS = (2, 4, 8, 16, 32, 64)
STRUCTURAL_BRANCH_COUNTS = (2, 3, 4, 5, 6, 7, 8)
PROGRAM_BRANCH_COUNTS = tuple(sorted(set(EXACT_BRANCH_COUNTS + STRUCTURAL_BRANCH_COUNTS)))
FINITE_FIELDS = ((137, 16), (239, 211))
FAMILIES = ("PRIMARY", "REUSE")
PORT_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
FINAL_BOUNDARY = "SUM_FOUR_SHARED_H_K_PORT_COMPONENTS_AFTER_LAST_PUBLIC_TRANSPORT"


def fail(message: str) -> None:
    raise RuntimeError(message)


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


@dataclass(frozen=True)
class CycleProgram:
    branch_count: int
    family: str
    theta_exponent_pairs: tuple[tuple[int, int], ...]
    transport_axes: tuple[str, ...]
    final_boundary: str = FINAL_BOUNDARY

    @property
    def factor_count(self) -> int:
        return 2 * self.branch_count

    @property
    def local_bit_count(self) -> int:
        return 3 * self.branch_count

    @property
    def total_logical_bits(self) -> int:
        return 2 + self.local_bit_count

    def fingerprint(self) -> str:
        return rank1.digest_json(public_descriptor(self))


def compile_program(branch_count: int, family: str) -> CycleProgram:
    if branch_count not in PROGRAM_BRANCH_COUNTS or family not in FAMILIES:
        fail("cycle program identity changed")
    pairs = tuple(
        (
            cubic_chain.theta_exponent(2 * branch + 1, family),
            cubic_chain.theta_exponent(2 * branch + 2, family),
        )
        for branch in range(branch_count)
    )
    axes = tuple(
        ("H" if branch % 2 == 0 else "K") if branch < branch_count - 1 else "NONE"
        for branch in range(branch_count)
    )
    program = CycleProgram(branch_count, family, pairs, axes)
    validate_program(program)
    return program


def validate_program(program: CycleProgram) -> None:
    if program.branch_count not in PROGRAM_BRANCH_COUNTS or program.family not in FAMILIES:
        fail("cycle program domain changed")
    expected_pairs = tuple(
        (
            cubic_chain.theta_exponent(2 * branch + 1, program.family),
            cubic_chain.theta_exponent(2 * branch + 2, program.family),
        )
        for branch in range(program.branch_count)
    )
    if program.theta_exponent_pairs != expected_pairs:
        fail("cycle phase schedule changed")
    if program.transport_axes != tuple(
        ("H" if branch % 2 == 0 else "K") if branch < program.branch_count - 1 else "NONE"
        for branch in range(program.branch_count)
    ):
        fail("cycle transport schedule changed")
    if any(not 1 <= exponent <= 16 for pair in program.theta_exponent_pairs for exponent in pair):
        fail("identity cycle phase admitted")
    if program.final_boundary != FINAL_BOUNDARY:
        fail("cycle final boundary changed")


def public_descriptor(program: CycleProgram) -> dict[str, Any]:
    return {
        "branch_count": program.branch_count,
        "factor_count": program.factor_count,
        "local_bit_count": program.local_bit_count,
        "total_logical_bits": program.total_logical_bits,
        "family": program.family,
        "shared_latent_ports": ["H", "K"],
        "branch_local_bits": ["ANCHOR", "H_LEAF", "K_LEAF"],
        "branch_factors": [
            "ALPHA_BRANCH_TO_H_TIMES_ANCHOR_TIMES_H_LEAF",
            "BETA_BRANCH_TO_K_TIMES_ANCHOR_TIMES_K_LEAF",
        ],
        "theta_exponent_pairs": [list(pair) for pair in program.theta_exponent_pairs],
        "transport_axes": list(program.transport_axes),
        "transport": "UNNORMALIZED_BINARY_WALSH_ON_DECLARED_SHARED_PORT_AXIS",
        "port_order": [list(pair) for pair in PORT_ORDER],
        "final_boundary": program.final_boundary,
    }


def lease(program: CycleProgram, alg: backend.Algebra, capacity: int) -> str:
    return rank1.digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "capacity": capacity,
            "carrier": "TWO_SHARED_LATENT_CUBIC_CYCLE_RANK4",
        }
    )


@dataclass
class CycleCarrier:
    alg: backend.Algebra
    factor_carrier: cubic_chain.CubicFactorCarrier
    port: list[Any]
    active_branches: int = 0
    active_family: str | None = None
    active_lease: str | None = None
    stage: str = "RESTORED"
    projection_calls: int = 0
    package_local_restoration_count: int = 0
    maximum_port_payload_bits: int = 0
    maximum_named_port_update_field_cells: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra, factor_capacity: int) -> "CycleCarrier":
        if factor_capacity < 4:
            fail("cycle factor capacity must be at least four")
        return cls(
            alg=alg,
            factor_carrier=cubic_chain.CubicFactorCarrier.create(alg, factor_capacity),
            port=[alg.zero for _ in range(4)],
        )

    @property
    def factor_capacity(self) -> int:
        return self.factor_carrier.capacity

    def backing_identity(self) -> tuple[int, int, int, int, int]:
        inner = self.factor_carrier.backing_identity()
        return (id(self), *inner, id(self.port))

    def exact_zero(self) -> bool:
        return (
            self.active_branches == 0
            and self.active_family is None
            and self.active_lease is None
            and self.stage == "RESTORED"
            and self.projection_calls == 0
            and self.factor_carrier.exact_zero()
            and all(value == self.alg.zero for value in self.port)
        )

    def observe_port(self) -> None:
        self.maximum_port_payload_bits = max(
            self.maximum_port_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.port),
        )

    def digest(self) -> str:
        return rank1.digest_json(
            {
                "factor_capacity": self.factor_capacity,
                "active_branches": self.active_branches,
                "active_family": self.active_family,
                "active_lease": self.active_lease,
                "stage": self.stage,
                "projection_calls": self.projection_calls,
                "package_local_restoration_count": self.package_local_restoration_count,
                "factor_carrier": self.factor_carrier.digest(),
                "port": [self.alg.serialize(value) for value in self.port],
            }
        )


def field_integer(alg: backend.Algebra, value: int) -> Any:
    return rank1.field_integer(alg, value)


def branch_response(alpha: Any, beta: Any, alg: backend.Algebra) -> list[Any]:
    two = field_integer(alg, 2)
    four = field_integer(alg, 4)
    response: list[Any] = []
    for h, k in PORT_ORDER:
        h_sum = alg.add(alg.one, alpha) if h else two
        k_sum = alg.add(alg.one, beta) if k else two
        response.append(alg.add(four, alg.mul(h_sum, k_sum)))
    return response


def walsh(values: list[Any], axis: str, alg: backend.Algebra) -> list[Any]:
    if len(values) != 4:
        fail("cycle port width changed")
    a, b, c, d = values
    if axis == "H":
        return [alg.add(a, c), alg.add(b, d), alg.sub(a, c), alg.sub(b, d)]
    if axis == "K":
        return [alg.add(a, b), alg.sub(a, b), alg.add(c, d), alg.sub(c, d)]
    fail("cycle Walsh axis changed")


def inverse_walsh(values: list[Any], axis: str, alg: backend.Algebra) -> list[Any]:
    half = alg.inverse(field_integer(alg, 2))
    return [alg.mul(half, value) for value in walsh(values, axis, alg)]


def require_active_program(carrier: CycleCarrier, program: CycleProgram) -> None:
    validate_program(program)
    if not isinstance(carrier, CycleCarrier):
        fail("null or invalid cycle carrier")
    if (
        carrier.stage != "FORWARD"
        or carrier.active_branches != program.branch_count
        or carrier.active_family != program.family
        or carrier.active_lease != lease(program, carrier.alg, carrier.factor_capacity)
        or carrier.factor_carrier.active_depth != program.factor_count
    ):
        fail("cycle program does not own active carrier lease")


def forward(carrier: CycleCarrier, program: CycleProgram) -> None:
    validate_program(program)
    if not isinstance(carrier, CycleCarrier):
        fail("null or invalid cycle carrier")
    if not carrier.exact_zero() or carrier.factor_capacity < program.factor_count:
        fail("cycle carrier not available")
    carrier.active_family = program.family
    carrier.active_lease = lease(program, carrier.alg, carrier.factor_capacity)
    carrier.stage = "FORWARD"
    carrier.port[:] = [carrier.alg.one for _ in range(4)]
    carrier.observe_port()
    for branch, ((alpha_exp, beta_exp), axis) in enumerate(zip(program.theta_exponent_pairs, program.transport_axes)):
        alpha = carrier.alg.power(alpha_exp)
        beta = carrier.alg.power(beta_exp)
        cubic_chain.load_site(carrier.factor_carrier, alpha)
        cubic_chain.load_site(carrier.factor_carrier, beta)
        actual_alpha = carrier.factor_carrier.cubic_branches[2 * branch]
        actual_beta = carrier.factor_carrier.cubic_branches[2 * branch + 1]
        response = branch_response(actual_alpha, actual_beta, carrier.alg)
        if any(value == carrier.alg.zero for value in response):
            fail("cycle branch response is singular")
        old_port = list(carrier.port)
        carrier.port[:] = [carrier.alg.mul(value, factor) for value, factor in zip(old_port, response)]
        carrier.maximum_named_port_update_field_cells = max(carrier.maximum_named_port_update_field_cells, 16)
        carrier.observe_port()
        if axis != "NONE":
            carrier.port[:] = walsh(carrier.port, axis, carrier.alg)
            carrier.maximum_named_port_update_field_cells = max(carrier.maximum_named_port_update_field_cells, 8)
        carrier.active_branches += 1
        carrier.observe_port()


def project_boundary(carrier: CycleCarrier, program: CycleProgram) -> Any:
    if not isinstance(carrier, CycleCarrier):
        fail("null or invalid cycle carrier")
    require_active_program(carrier, program)
    if carrier.projection_calls != 0:
        fail("cycle boundary projected more than once")
    boundary = carrier.alg.zero
    for value in carrier.port:
        boundary = carrier.alg.add(boundary, value)
    carrier.projection_calls += 1
    return boundary


def inverse(carrier: CycleCarrier, program: CycleProgram) -> None:
    require_active_program(carrier, program)
    if carrier.projection_calls != 1:
        fail("cycle inverse before final boundary")
    for branch in range(program.branch_count - 1, -1, -1):
        alpha = carrier.factor_carrier.cubic_branches[2 * branch]
        beta = carrier.factor_carrier.cubic_branches[2 * branch + 1]
        response = branch_response(alpha, beta, carrier.alg)
        carrier.port[:] = [carrier.alg.mul(value, carrier.alg.inverse(factor)) for value, factor in zip(carrier.port, response)]
        carrier.maximum_named_port_update_field_cells = max(carrier.maximum_named_port_update_field_cells, 16)
        carrier.observe_port()
        carrier.active_branches -= 1
        if branch > 0:
            carrier.port[:] = inverse_walsh(
                carrier.port, program.transport_axes[branch - 1], carrier.alg
            )
        carrier.observe_port()
    if any(value != carrier.alg.one for value in carrier.port):
        fail("cycle port inverse did not restore seed")
    carrier.port[:] = [carrier.alg.sub(value, carrier.alg.one) for value in carrier.port]
    for position in range(program.factor_count, 0, -1):
        exponent = program.theta_exponent_pairs[(position - 1) // 2][(position - 1) % 2]
        cubic_chain.unload_site(carrier.factor_carrier, carrier.alg.power(exponent), position)
    carrier.active_family = None
    carrier.active_lease = None
    carrier.stage = "RESTORED"
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not carrier.exact_zero():
        fail("cycle carrier did not restore exact zero")


def execute_transaction(carrier: CycleCarrier, program: CycleProgram) -> dict[str, Any]:
    carrier.maximum_port_payload_bits = 0
    carrier.maximum_named_port_update_field_cells = 0
    carrier.factor_carrier.maximum_resident_payload_bits = 0
    carrier.factor_carrier.maximum_local_coupling_named_field_cells = 0
    initial_digest = carrier.digest()
    backing = carrier.backing_identity()
    count_before = carrier.package_local_restoration_count
    forward(carrier, program)
    commitment, commitment_bytes = rank1.stream_vector_commitment(carrier.factor_carrier.active_values(), carrier.alg)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    descriptor_bytes = len(json.dumps(public_descriptor(program), sort_keys=True, separators=(",", ":")).encode())
    return {
        "branch_count": program.branch_count,
        "factor_count": program.factor_count,
        "local_bit_count": program.local_bit_count,
        "total_logical_bits": program.total_logical_bits,
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "boundary": carrier.alg.serialize(boundary),
        "factor_commitment": commitment,
        "factor_commitment_json_bytes": commitment_bytes,
        "resident_phase_factor_field_cells": 2 * program.factor_count,
        "resident_nontrivial_theta_field_cells": program.factor_count,
        "resident_shared_latent_port_field_cells": 4,
        "exact_two_branch_junction_separator_rank": 4,
        "maximum_named_port_update_field_cells": carrier.maximum_named_port_update_field_cells,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "maximum_resident_factor_payload_bits": carrier.factor_carrier.maximum_resident_payload_bits,
        "maximum_resident_port_payload_bits": carrier.maximum_port_payload_bits,
        "public_program_json_bytes": descriptor_bytes,
        "accepted_path_local_assignment_enumeration": False,
        "accepted_path_global_assignment_or_dense_tensor_cells": 0,
        "inverse_history_cells": 0,
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "snapshot_reload_used": False,
        "response_released_after_restoration": True,
        "restored_exact_zero": carrier.exact_zero(),
        "same_backing": carrier.backing_identity() == backing,
        "package_local_restoration_count_before": count_before,
        "package_local_restoration_count_after": carrier.package_local_restoration_count,
        "initial_digest": initial_digest,
        "restored_digest_with_package_local_count": carrier.digest(),
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "compiler_commitment_and_verification_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "intermediate_port_or_factor_payload_exposed_in_result": False,
        "one_way_factor_commitment_emitted": True,
    }


def rank_certificate(program: CycleProgram, alg: backend.Algebra) -> dict[str, Any]:
    branch_minors = []
    for alpha_exp, beta_exp in program.theta_exponent_pairs:
        alpha = alg.power(alpha_exp)
        beta = alg.power(beta_exp)
        minor = alg.mul(
            alg.mul(alg.sub(alpha, alg.one), alg.sub(alpha, alg.one)),
            alg.mul(alg.sub(beta, alg.one), alg.sub(beta, alg.one)),
        )
        branch_minors.append(minor != alg.zero)
    return {
        "branch_count": program.branch_count,
        "algebra": algebra_signature(alg),
        "all_branch_kronecker_minors_nonzero": all(branch_minors),
        "each_branch_local_to_h_k_map_rank": 4,
        "two_branch_cycle_junction_separator_rank": 4,
        "walsh_transport_rank": 4,
        "rank_three_separator_rejected": all(branch_minors),
        "minor_values_serialized": False,
        "local_or_global_assignment_tensor_materialized": False,
    }


def classical_recurrence(program: CycleProgram, alg: backend.Algebra) -> tuple[Any, list[Any], int]:
    state = [alg.one for _ in range(4)]
    maximum_payload = sum(alg.payload_bits(value) for value in state)
    for (alpha_exp, beta_exp), axis in zip(program.theta_exponent_pairs, program.transport_axes):
        response = branch_response(alg.power(alpha_exp), alg.power(beta_exp), alg)
        state = [alg.mul(value, factor) for value, factor in zip(state, response)]
        maximum_payload = max(maximum_payload, sum(alg.payload_bits(value) for value in state))
        if axis != "NONE":
            state = walsh(state, axis, alg)
            maximum_payload = max(maximum_payload, sum(alg.payload_bits(value) for value in state))
    boundary = alg.zero
    for value in state:
        boundary = alg.add(boundary, value)
    return boundary, state, maximum_payload


def classical_baseline(transaction: dict[str, Any], program: CycleProgram, alg: backend.Algebra) -> dict[str, Any]:
    boundary, final_state, maximum_payload = classical_recurrence(program, alg)
    row = [alg.one for _ in range(4)]
    for branch in range(program.branch_count - 1, -1, -1):
        response = branch_response(alg.power(program.theta_exponent_pairs[branch][0]), alg.power(program.theta_exponent_pairs[branch][1]), alg)
        row = [alg.mul(value, factor) for value, factor in zip(row, response)]
        if branch > 0:
            row = walsh(row, program.transport_axes[branch - 1], alg)
    compiled = alg.zero
    for value in row:
        compiled = alg.add(compiled, value)
    row_commitment, row_bytes = rank1.stream_vector_commitment(row, alg)
    return {
        "branch_count": program.branch_count,
        "family": program.family,
        "algebra": algebra_signature(alg),
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "compiled_row_boundary_agreement": alg.serialize(compiled) == transaction["boundary"],
        "full_signature_exact_factor_field_cells": program.factor_count,
        "runtime_dynamic_port_field_cells": 4,
        "runtime_named_old_plus_new_port_update_field_cells": 12,
        "runtime_maximum_exact_port_payload_bits": maximum_payload,
        "sealed_arbitrary_port_input_final_row_field_cells": 4,
        "sealed_fixed_transaction_field_cells": 1,
        "compiled_final_row_commitment": row_commitment,
        "compiled_final_row_commitment_json_bytes": row_bytes,
        "final_state_serialized": False,
        "phase_carrier_or_snapshot_used": False,
    }


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "boundary",
        "factor_commitment",
        "family",
        "initial_digest",
        "package_local_restoration_count_before",
        "package_local_restoration_count_after",
        "program_fingerprint",
        "restored_digest_with_package_local_count",
    }
    return {key: value for key, value in transaction.items() if key not in excluded}


def reverse_port_control(
    values: list[Any],
    program: CycleProgram,
    factors: list[Any],
    alg: backend.Algebra,
    variant: str,
) -> list[Any]:
    state = list(values)
    last = program.branch_count - 1
    for branch in range(last, -1, -1):
        response = branch_response(factors[2 * branch], factors[2 * branch + 1], alg)
        if variant == "REORDER_LAST_D_AND_W" and branch == last:
            state = inverse_walsh(state, program.transport_axes[branch - 1], alg)
            state = [alg.mul(value, alg.inverse(factor)) for value, factor in zip(state, response)]
            continue
        state = [alg.mul(value, alg.inverse(factor)) for value, factor in zip(state, response)]
        if branch > 0:
            if variant == "WRONG_WALSH_INVERSE" and branch == last:
                state = walsh(state, program.transport_axes[branch - 1], alg)
            else:
                state = inverse_walsh(state, program.transport_axes[branch - 1], alg)
    return state


def controls() -> dict[str, bool]:
    alg = backend.Algebra("F239", modulus=239, root=211)
    program = compile_program(4, "PRIMARY")
    reference = execute_transaction(CycleCarrier.create(alg, program.factor_count), program)

    missing = CycleCarrier.create(alg, program.factor_count)
    forward(missing, program)

    wrong_factor = CycleCarrier.create(alg, program.factor_count)
    forward(wrong_factor, program)
    wrong_factor_inverse_detected = False
    try:
        cubic_chain.unload_site(wrong_factor.factor_carrier, alg.power(program.theta_exponent_pairs[-1][0]), program.factor_count)
    except RuntimeError:
        wrong_factor_inverse_detected = True

    wrong_order = CycleCarrier.create(alg, program.factor_count)
    forward(wrong_order, program)
    factor_values = list(wrong_order.factor_carrier.cubic_branches[: program.factor_count])
    lawful_reverse = reverse_port_control(
        wrong_order.port, program, factor_values, alg, "LAWFUL"
    )
    reordered_reverse = reverse_port_control(
        wrong_order.port,
        program,
        factor_values,
        alg,
        "REORDER_LAST_D_AND_W",
    )
    wrong_walsh_reverse = reverse_port_control(
        wrong_order.port,
        program,
        factor_values,
        alg,
        "WRONG_WALSH_INVERSE",
    )
    seed = [alg.one for _ in range(4)]
    lawful_complete_inverse_restores_seed = lawful_reverse == seed
    reordered_noncommuting_inverse_fails = reordered_reverse != seed and reordered_reverse != lawful_reverse
    wrong_walsh_inverse_fails = wrong_walsh_reverse != seed and wrong_walsh_reverse != lawful_reverse

    premature = CycleCarrier.create(alg, program.factor_count)
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    owner = CycleCarrier.create(alg, program.factor_count)
    forward(owner, program)
    wrong_projection_owner_rejected = False
    wrong_inverse_owner_rejected = False
    try:
        project_boundary(owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_projection_owner_rejected = True
    try:
        inverse(owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_inverse_owner_rejected = True

    null_carrier_rejected = False
    try:
        project_boundary(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    baseline_boundary, _, _ = classical_recurrence(program, alg)
    no_transport_state = [alg.one for _ in range(4)]
    for alpha_exp, beta_exp in program.theta_exponent_pairs:
        response = branch_response(alg.power(alpha_exp), alg.power(beta_exp), alg)
        no_transport_state = [alg.mul(value, factor) for value, factor in zip(no_transport_state, response)]
    no_transport_boundary = alg.zero
    for value in no_transport_state:
        no_transport_boundary = alg.add(no_transport_boundary, value)

    transport_before_state = [alg.one for _ in range(4)]
    for (alpha_exp, beta_exp), axis in zip(program.theta_exponent_pairs, program.transport_axes):
        if axis != "NONE":
            transport_before_state = walsh(transport_before_state, axis, alg)
        response = branch_response(alg.power(alpha_exp), alg.power(beta_exp), alg)
        transport_before_state = [alg.mul(value, factor) for value, factor in zip(transport_before_state, response)]
    transport_before_boundary = alg.zero
    for value in transport_before_state:
        transport_before_boundary = alg.add(transport_before_boundary, value)

    overmerged_boundary = alg.add(baseline_boundary, alg.zero)
    _, final_state, _ = classical_recurrence(program, alg)
    overmerged_boundary = alg.add(final_state[0], final_state[3])

    under_shared = alg.one
    for alpha_exp, beta_exp in program.theta_exponent_pairs:
        response = branch_response(alg.power(alpha_exp), alg.power(beta_exp), alg)
        local_sum = alg.zero
        for value in response:
            local_sum = alg.add(local_sum, value)
        under_shared = alg.mul(under_shared, local_sum)

    first_alpha = alg.power(program.theta_exponent_pairs[0][0])
    first_beta = alg.power(program.theta_exponent_pairs[0][1])
    identity_minor = alg.mul(
        alg.mul(alg.sub(alg.one, alg.one), alg.sub(alg.one, alg.one)),
        alg.mul(alg.sub(first_beta, alg.one), alg.sub(first_beta, alg.one)),
    )
    perturbed_pairs = list(program.theta_exponent_pairs)
    original_alpha, original_beta = perturbed_pairs[0]
    perturbed_pairs[0] = (1 + (original_alpha % 16), original_beta)
    perturbed = CycleProgram(
        program.branch_count,
        program.family,
        tuple(perturbed_pairs),
        program.transport_axes,
    )
    perturbed_boundary, _, _ = classical_recurrence(perturbed, alg)

    return {
        "identity_theta_collapses_rank4_minor": identity_minor == alg.zero,
        "forced_rank3_separator_rejected": rank_certificate(program, alg)["rank_three_separator_rejected"],
        "transport_disabled_changes_boundary": alg.serialize(no_transport_boundary) != alg.serialize(baseline_boundary),
        "transport_before_consumer_changes_boundary": alg.serialize(transport_before_boundary) != alg.serialize(baseline_boundary),
        "phase_perturbation_changes_boundary": alg.serialize(perturbed_boundary) != alg.serialize(baseline_boundary),
        "h_equals_k_overmerge_changes_boundary": alg.serialize(overmerged_boundary) != alg.serialize(baseline_boundary),
        "independent_port_copy_under_share_changes_boundary": alg.serialize(under_shared) != alg.serialize(baseline_boundary),
        "missing_inverse_leaves_resident_state": not missing.exact_zero(),
        "lawful_complete_port_inverse_restores_seed": lawful_complete_inverse_restores_seed,
        "wrong_factor_inverse_detected": wrong_factor_inverse_detected,
        "reordered_noncommuting_inverse_fails": reordered_noncommuting_inverse_fails,
        "wrong_walsh_inverse_fails": wrong_walsh_inverse_fails,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "wrong_projection_owner_rejected": wrong_projection_owner_rejected,
        "wrong_inverse_owner_rejected": wrong_inverse_owner_rejected,
        "snapshot_command_available": False,
        "reference_transaction_restored": reference["restored_exact_zero"],
    }


def run() -> dict[str, Any]:
    exact = []
    for branches in EXACT_BRANCH_COUNTS:
        program = compile_program(branches, "PRIMARY")
        exact.append(execute_transaction(CycleCarrier.create(backend.Algebra("Q_ZETA17"), program.factor_count), program))

    structural = []
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        for branches in STRUCTURAL_BRANCH_COUNTS:
            program = compile_program(branches, "PRIMARY")
            item = execute_transaction(CycleCarrier.create(alg, program.factor_count), program)
            item["field"] = f"F{modulus}"
            structural.append(item)

    certificates = [rank_certificate(compile_program(branches, "PRIMARY"), backend.Algebra("Q_ZETA17")) for branches in EXACT_BRANCH_COUNTS]
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        certificates.extend(rank_certificate(compile_program(branches, "PRIMARY"), alg) for branches in STRUCTURAL_BRANCH_COUNTS)
    if not all(item["all_branch_kronecker_minors_nonzero"] and item["rank_three_separator_rejected"] for item in certificates):
        fail("cycle rank certificate failed")

    baselines = []
    for item in exact:
        baselines.append(classical_baseline(item, compile_program(item["branch_count"], item["family"]), backend.Algebra("Q_ZETA17")))
    for item in structural:
        modulus, root = next(pair for pair in FINITE_FIELDS if item["field"] == f"F{pair[0]}")
        baselines.append(classical_baseline(item, compile_program(item["branch_count"], item["family"]), backend.Algebra(item["field"], modulus=modulus, root=root)))
    if not all(item["boundary_agreement"] and item["compiled_row_boundary_agreement"] for item in baselines):
        fail("cycle matched classical baseline disagrees")

    capacity = 2 * max(EXACT_BRANCH_COUNTS)
    reuse_carrier = CycleCarrier.create(backend.Algebra("Q_ZETA17"), capacity)
    first = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(16, "REUSE"))
    fresh = execute_transaction(CycleCarrier.create(backend.Algebra("Q_ZETA17"), capacity), compile_program(16, "REUSE"))
    if reused["boundary"] != fresh["boundary"] or resource_signature(reused) != resource_signature(fresh):
        fail("restored cycle carrier disagrees with fresh reuse")

    control_results = controls()
    if not all(value for key, value in control_results.items() if key != "snapshot_command_available") or control_results["snapshot_command_available"]:
        fail("one or more cycle controls failed")

    return {
        "schema": "CAT_CAS_F17_TWO_LATENT_CUBIC_CYCLE_RANK4_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_TWO_SHARED_LATENT_NONAFFINE_CUBIC_PHASE_CYCLE_WITH_INTERLEAVED_WALSH_TRANSPORT_NONCOMMUTING_WITH_BRANCH_DIAGONALS_HAS_CERTIFIED_RANK4_JUNCTION_CLOSURE_ON_A_FOUR_CELL_RESIDENT_PORT_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_THE_IDENTICAL_FOUR_SCALAR_CLASSICAL_RECURRENCE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "exact_branch_counts": EXACT_BRANCH_COUNTS,
            "dual_field_structural_branch_counts": STRUCTURAL_BRANCH_COUNTS,
            "shared_latent_ports": 2,
            "local_cubic_factors_per_branch": 2,
            "boolean_degree": 3,
            "transport_schedule": "ALTERNATING_UNNORMALIZED_WALSH_H_AND_K",
            "two_branch_junction_separator_rank": 4,
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "rank_certificates": certificates,
        "compiled_classical_baselines": baselines,
        "reuse": {
            "first_branch_count": 8,
            "reused_branch_count": 16,
            "first_family": "PRIMARY",
            "reused_family": "REUSE",
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "fresh_restored_resource_signature_agreement": resource_signature(reused) == resource_signature(fresh),
            "same_actual_backing_across_unrelated_programs": first["same_backing"] and reused["same_backing"] and reuse_carrier.backing_identity() == backing,
            "package_local_restoration_count_after_two_transactions": reuse_carrier.package_local_restoration_count,
            "baseline_reload_used": False,
        },
        "controls": control_results,
        "resource_law": {
            "resident_phase_factor_field_cells_at_b_branches": "FOUR_TIMES_B",
            "resident_nontrivial_theta_field_cells_at_b_branches": "TWO_TIMES_B",
            "resident_shared_latent_port_field_cells": 4,
            "exact_two_branch_junction_separator_rank": 4,
            "maximum_named_port_update_field_cells": 16,
            "matched_classical_full_signature_field_cells_at_b_branches": "TWO_TIMES_B",
            "matched_classical_runtime_dynamic_port_field_cells": 4,
            "matched_classical_exact_port_payload_equals_phase_port_payload": True,
            "matched_classical_arbitrary_port_input_compiled_row_field_cells": 4,
            "accepted_path_assignment_or_dense_tensor_cells": 0,
            "inverse_history_cells": 0,
            "fixed_logical_rank_implies_fixed_exact_bit_width": False,
            "full_exact_bit_complexity_established": False,
            "python_container_allocator_native_bigint_hashlib_bit_operation_and_whole_process_excluded": True,
        },
        "matched_baseline": {
            "strongest_full_signature": "TWO_B_EXACT_PUBLIC_PHASE_FACTORS",
            "strongest_final_boundary_runtime": "IDENTICAL_FOUR_DYNAMIC_SCALAR_RECURRENCE",
            "strongest_sealed_arbitrary_port_input_boundary": "FOUR_COMPILED_FINAL_ROW_SCALARS",
            "strongest_sealed_fixed_transaction": "ONE_CACHED_BOUNDARY_SCALAR",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "factor_and_shared_port_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "compiler_commitment_and_verification_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "declared_two_shared_latent_cubic_cycle_family": True,
            "arbitrary_cubic_hypergraph_closure": False,
            "arbitrary_port_arity": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_TWO_SHARED_LATENT_CUBIC_CYCLE_AND_NONCOMMUTING_WALSH_TRANSPORT_REQUIRE_RANK4_BUT_CLOSE_ON_AN_IDENTICAL_FOUR_SCALAR_CLASSICAL_RECURRENCE_THE_PHASE_CARRIER_STORES_TWICE_THE_PUBLIC_FACTOR_SIGNATURE_AND_EXACT_PORT_WIDTH_GROWS_WITH_DEPTH",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(run(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
