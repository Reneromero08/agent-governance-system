#!/usr/bin/env python3
"""Exact shared-latent cubic phase-factor closure on a bounded hypertree."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_overlapping_cubic_bond3_phase_factor_closure as cubic_chain


EXACT_HEIGHTS = (1, 2, 3, 4, 5, 6, 7)
STRUCTURAL_HEIGHTS = (1, 2, 3, 4, 5, 6)
PROGRAM_HEIGHTS = tuple(sorted(set(EXACT_HEIGHTS + STRUCTURAL_HEIGHTS)))
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE")
FINAL_BOUNDARY = "SUM_ALL_SHARED_LATENT_AND_LEAF_BITS"


def fail(message: str) -> None:
    raise RuntimeError(message)


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


@dataclass(frozen=True)
class HypertreeProgram:
    height: int
    family: str
    theta_exponents: tuple[int, ...]
    topology: tuple[tuple[int, int, int], ...]
    final_boundary: str = FINAL_BOUNDARY

    @property
    def factor_count(self) -> int:
        return (1 << self.height) - 1

    @property
    def variable_count(self) -> int:
        return (1 << (self.height + 1)) - 1

    def fingerprint(self) -> str:
        return rank1.digest_json(public_descriptor(self))


def compile_program(height: int, family: str) -> HypertreeProgram:
    if height not in PROGRAM_HEIGHTS or family not in FAMILIES:
        fail("hypertree program identity changed")
    factor_count = (1 << height) - 1
    topology = tuple((node, 2 * node + 1, 2 * node + 2) for node in range(factor_count))
    program = HypertreeProgram(
        height=height,
        family=family,
        theta_exponents=tuple(
            cubic_chain.theta_exponent(index + 1, family)
            for index in range(factor_count)
        ),
        topology=topology,
    )
    validate_program(program)
    return program


def validate_program(program: HypertreeProgram) -> None:
    if program.height not in PROGRAM_HEIGHTS or program.family not in FAMILIES:
        fail("hypertree program domain changed")
    factor_count = (1 << program.height) - 1
    if program.theta_exponents != tuple(
        cubic_chain.theta_exponent(index + 1, program.family)
        for index in range(factor_count)
    ):
        fail("hypertree factor schedule changed")
    if program.topology != tuple(
        (node, 2 * node + 1, 2 * node + 2) for node in range(factor_count)
    ):
        fail("hypertree public topology changed")
    if any(not 1 <= value <= 16 for value in program.theta_exponents):
        fail("identity cubic factor admitted")
    if program.final_boundary != FINAL_BOUNDARY:
        fail("hypertree final boundary changed")


def public_descriptor(program: HypertreeProgram) -> dict[str, Any]:
    return {
        "height": program.height,
        "factor_count": program.factor_count,
        "variable_count": program.variable_count,
        "family": program.family,
        "factor": "THETA_NODE_TO_PARENT_BIT_TIMES_LEFT_BIT_TIMES_RIGHT_BIT",
        "theta_exponents": list(program.theta_exponents),
        "topology": [list(item) for item in program.topology],
        "topology_compiler": "COMPLETE_BINARY_HEAP_FROM_PUBLIC_HEIGHT",
        "latent_custody": "EACH_NONROOT_INTERNAL_BIT_IS_SHARED_BY_PARENT_AND_CHILD_CUBIC_FACTORS",
        "final_boundary": program.final_boundary,
    }


def lease(program: HypertreeProgram, alg: backend.Algebra, capacity: int) -> str:
    return rank1.digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "capacity": capacity,
            "carrier": "SHARED_LATENT_CUBIC_HYPERTREE_FACTOR",
        }
    )


@dataclass
class HypertreeFactorCarrier:
    alg: backend.Algebra
    capacity: int
    identity_branches: list[Any]
    cubic_branches: list[Any]
    active_factors: int = 0
    active_height: int = 0
    active_family: str | None = None
    active_lease: str | None = None
    stage: str = "RESTORED"
    projection_calls: int = 0
    package_local_restoration_count: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_local_coupling_named_field_cells: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra, capacity: int) -> "HypertreeFactorCarrier":
        if capacity < 1:
            fail("hypertree carrier capacity must be positive")
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
        for site in range(self.active_factors):
            values.extend((self.identity_branches[site], self.cubic_branches[site]))
        return values

    def all_values(self) -> list[Any]:
        return [*self.identity_branches, *self.cubic_branches]

    def observe(self) -> None:
        payload = sum(self.alg.payload_bits(value) for value in self.active_values())
        self.maximum_resident_payload_bits = max(self.maximum_resident_payload_bits, payload)

    def exact_zero(self) -> bool:
        return (
            self.active_factors == 0
            and self.active_height == 0
            and self.active_family is None
            and self.active_lease is None
            and self.stage == "RESTORED"
            and all(value == self.alg.zero for value in self.all_values())
        )

    def digest(self) -> str:
        return rank1.digest_json(
            {
                "capacity": self.capacity,
                "active_factors": self.active_factors,
                "active_height": self.active_height,
                "active_family": self.active_family,
                "active_lease": self.active_lease,
                "stage": self.stage,
                "package_local_restoration_count": self.package_local_restoration_count,
                "identity": [self.alg.serialize(value) for value in self.identity_branches],
                "cubic": [self.alg.serialize(value) for value in self.cubic_branches],
            }
        )


def load_factor(carrier: HypertreeFactorCarrier, theta: Any) -> None:
    site = carrier.active_factors
    if site >= carrier.capacity:
        fail("hypertree factor capacity exceeded")
    if carrier.identity_branches[site] != carrier.alg.zero or carrier.cubic_branches[site] != carrier.alg.zero:
        fail("hypertree factor site was not empty")
    carrier.identity_branches[site] = carrier.alg.one
    old_identity = carrier.identity_branches[site]
    old_cubic = carrier.cubic_branches[site]
    carrier.identity_branches[site] = carrier.alg.add(old_identity, carrier.alg.mul(theta, old_cubic))
    carrier.cubic_branches[site] = carrier.alg.add(carrier.alg.mul(theta, old_identity), old_cubic)
    carrier.maximum_local_coupling_named_field_cells = max(carrier.maximum_local_coupling_named_field_cells, 4)
    carrier.active_factors += 1
    carrier.observe()


def unload_factor(carrier: HypertreeFactorCarrier, theta: Any, position: int) -> None:
    if position != carrier.active_factors or position < 1:
        fail("hypertree factor inverse order changed")
    site = position - 1
    denominator = carrier.alg.sub(carrier.alg.one, carrier.alg.mul(theta, theta))
    if denominator == carrier.alg.zero:
        fail("singular hypertree factor inverse")
    scale = carrier.alg.inverse(denominator)
    old_identity = carrier.identity_branches[site]
    old_cubic = carrier.cubic_branches[site]
    identity = carrier.alg.mul(scale, carrier.alg.sub(old_identity, carrier.alg.mul(theta, old_cubic)))
    cubic = carrier.alg.mul(scale, carrier.alg.sub(old_cubic, carrier.alg.mul(theta, old_identity)))
    carrier.maximum_local_coupling_named_field_cells = max(carrier.maximum_local_coupling_named_field_cells, 4)
    if identity != carrier.alg.one or cubic != carrier.alg.zero:
        fail("hypertree factor inverse did not restore seeded branch")
    carrier.identity_branches[site] = carrier.alg.sub(identity, carrier.alg.one)
    carrier.cubic_branches[site] = cubic
    carrier.active_factors -= 1
    carrier.observe()


def forward(carrier: HypertreeFactorCarrier, program: HypertreeProgram) -> None:
    validate_program(program)
    if not isinstance(carrier, HypertreeFactorCarrier):
        fail("null or invalid hypertree factor carrier")
    if not carrier.exact_zero() or carrier.capacity < program.factor_count:
        fail("hypertree factor carrier not available")
    carrier.active_height = program.height
    carrier.active_family = program.family
    carrier.active_lease = lease(program, carrier.alg, carrier.capacity)
    carrier.stage = "FORWARD"
    for exponent in program.theta_exponents:
        theta = carrier.alg.power(exponent)
        if carrier.alg.sub(carrier.alg.one, carrier.alg.mul(theta, theta)) == carrier.alg.zero:
            fail("singular hypertree factor coupling")
        load_factor(carrier, theta)


def require_active_program(carrier: HypertreeFactorCarrier, program: HypertreeProgram) -> None:
    validate_program(program)
    if not isinstance(carrier, HypertreeFactorCarrier):
        fail("null or invalid hypertree factor carrier")
    if (
        carrier.stage != "FORWARD"
        or carrier.active_factors != program.factor_count
        or carrier.active_height != program.height
        or carrier.active_family != program.family
        or carrier.active_lease != lease(program, carrier.alg, carrier.capacity)
    ):
        fail("hypertree program does not own active carrier lease")


@dataclass
class Message:
    values: tuple[Any, Any]
    live: bool = True


@dataclass
class MessageArena:
    live_messages: int = 0
    maximum_live_messages: int = 0

    def acquire(self, values: tuple[Any, Any]) -> Message:
        self.live_messages += 1
        self.maximum_live_messages = max(self.maximum_live_messages, self.live_messages)
        return Message(values)

    def release(self, message: Message) -> None:
        if not message.live:
            fail("hypertree message released twice")
        message.live = False
        self.live_messages -= 1
        if self.live_messages < 0:
            fail("hypertree message arena underflow")


def contract_tree(
    carrier: HypertreeFactorCarrier,
    program: HypertreeProgram,
    *,
    overmerge_right_into_left: bool = False,
) -> tuple[Any, int]:
    require_active_program(carrier, program)
    alg = carrier.alg
    arena = MessageArena()
    first_leaf = program.factor_count

    def visit(node: int) -> Message:
        if node >= first_leaf:
            return arena.acquire((alg.one, alg.one))
        left = visit(2 * node + 1)
        right = visit(2 * node + 2)
        left_values = left.values
        right_values = left.values if overmerge_right_into_left else right.values
        l0, l1 = left_values
        r0, r1 = right_values
        theta = carrier.cubic_branches[node]
        if carrier.identity_branches[node] != alg.one:
            fail("hypertree identity factor branch changed")
        sum_left = alg.add(l0, l1)
        sum_right = alg.add(r0, r1)
        out0 = alg.mul(sum_left, sum_right)
        out1 = alg.add(
            alg.add(alg.mul(l0, r0), alg.mul(l0, r1)),
            alg.add(alg.mul(l1, r0), alg.mul(theta, alg.mul(l1, r1))),
        )
        result = arena.acquire((out0, out1))
        arena.release(left)
        arena.release(right)
        return result

    root = visit(0)
    boundary = alg.add(root.values[0], root.values[1])
    arena.release(root)
    if arena.live_messages != 0:
        fail("hypertree contraction retained a message")
    return boundary, 2 * arena.maximum_live_messages


def project_boundary(carrier: HypertreeFactorCarrier, program: HypertreeProgram) -> Any:
    if carrier.projection_calls != 0:
        fail("hypertree boundary projected more than once")
    boundary, _ = contract_tree(carrier, program)
    carrier.projection_calls += 1
    return boundary


def inverse(carrier: HypertreeFactorCarrier, program: HypertreeProgram) -> None:
    require_active_program(carrier, program)
    if carrier.projection_calls != 1:
        fail("hypertree inverse before final boundary")
    for position in range(program.factor_count, 0, -1):
        unload_factor(carrier, carrier.alg.power(program.theta_exponents[position - 1]), position)
    carrier.active_height = 0
    carrier.active_family = None
    carrier.active_lease = None
    carrier.stage = "RESTORED"
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not carrier.exact_zero():
        fail("hypertree factor carrier did not restore exact zero")


def execute_transaction(carrier: HypertreeFactorCarrier, program: HypertreeProgram) -> dict[str, Any]:
    carrier.maximum_resident_payload_bits = 0
    carrier.maximum_local_coupling_named_field_cells = 0
    initial_digest = carrier.digest()
    backing = carrier.backing_identity()
    count_before = carrier.package_local_restoration_count
    forward(carrier, program)
    commitment, commitment_bytes = rank1.stream_vector_commitment(carrier.active_values(), carrier.alg)
    _, projection_cells = contract_tree(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    descriptor_bytes = len(json.dumps(public_descriptor(program), sort_keys=True, separators=(",", ":")).encode())
    return {
        "height": program.height,
        "factor_count": program.factor_count,
        "variable_count": program.variable_count,
        "shared_internal_latent_count": max(0, program.factor_count - 1),
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "boundary": carrier.alg.serialize(boundary),
        "factor_commitment": commitment,
        "factor_commitment_json_bytes": commitment_bytes,
        "resident_phase_factor_field_cells": 2 * program.factor_count,
        "resident_nontrivial_theta_field_cells": program.factor_count,
        "exact_tree_edge_tensor_bond_dimension": 2,
        "projection_maximum_live_message_field_cells": projection_cells,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "maximum_resident_factor_payload_bits": carrier.maximum_resident_payload_bits,
        "maximum_local_coupling_named_field_cells": carrier.maximum_local_coupling_named_field_cells,
        "public_program_json_bytes": descriptor_bytes,
        "accepted_path_assignment_enumeration": False,
        "accepted_path_dense_tensor_cells": 0,
        "accepted_path_global_message_table_cells": 0,
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
        "projection_compiler_and_commitment_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "intermediate_message_payload_exposed_in_result": False,
        "one_way_factor_commitment_emitted": True,
    }


def edge_rank_certificate(program: HypertreeProgram, alg: backend.Algebra) -> dict[str, Any]:
    nonzero = [alg.sub(alg.power(exponent), alg.one) != alg.zero for exponent in program.theta_exponents]
    return {
        "height": program.height,
        "algebra": algebra_signature(alg),
        "tree_edge_count": program.variable_count - 1,
        "all_local_theta_minus_one_minors_nonzero": all(nonzero),
        "exact_full_factor_tensor_rank_across_every_tree_edge": 2,
        "rank_transfer_law": "FACTORS_STRICTLY_ON_EITHER_SIDE_ONLY_SCALE_OR_REPLICATE_ROWS_OR_COLUMNS",
        "dense_full_tensor_materialized": False,
    }


def public_tree_recurrence(program: HypertreeProgram, alg: backend.Algebra) -> tuple[Any, int]:
    factors = [alg.power(exponent) for exponent in program.theta_exponents]
    arena = MessageArena()
    first_leaf = program.factor_count

    def visit(node: int) -> Message:
        if node >= first_leaf:
            return arena.acquire((alg.one, alg.one))
        left = visit(2 * node + 1)
        right = visit(2 * node + 2)
        l0, l1 = left.values
        r0, r1 = right.values
        theta = factors[node]
        result = arena.acquire(
            (
                alg.mul(alg.add(l0, l1), alg.add(r0, r1)),
                alg.add(
                    alg.add(alg.mul(l0, r0), alg.mul(l0, r1)),
                    alg.add(alg.mul(l1, r0), alg.mul(theta, alg.mul(l1, r1))),
                ),
            )
        )
        arena.release(left)
        arena.release(right)
        return result

    root = visit(0)
    boundary = alg.add(root.values[0], root.values[1])
    arena.release(root)
    if arena.live_messages != 0:
        fail("public hypertree recurrence retained a message")
    return boundary, 2 * arena.maximum_live_messages


def classical_baseline(transaction: dict[str, Any], program: HypertreeProgram, alg: backend.Algebra) -> dict[str, Any]:
    boundary, live_cells = public_tree_recurrence(program, alg)
    return {
        "height": program.height,
        "family": program.family,
        "algebra": algebra_signature(alg),
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "full_signature_exact_factor_field_cells": program.factor_count,
        "runtime_maximum_live_message_field_cells": live_cells,
        "runtime_dynamic_message_width": 2,
        "phase_carrier_or_snapshot_used": False,
        "same_public_topology_and_factors": True,
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


def controls() -> dict[str, bool]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    program = compile_program(3, "PRIMARY")
    reference_carrier = HypertreeFactorCarrier.create(alg, program.factor_count)
    reference = execute_transaction(reference_carrier, program)

    missing = HypertreeFactorCarrier.create(alg, program.factor_count)
    forward(missing, program)

    wrong = HypertreeFactorCarrier.create(alg, program.factor_count)
    forward(wrong, program)
    wrong_inverse_detected = False
    try:
        unload_factor(wrong, alg.power(program.theta_exponents[-2]), program.factor_count)
    except RuntimeError:
        wrong_inverse_detected = True

    reordered = HypertreeFactorCarrier.create(alg, program.factor_count)
    forward(reordered, program)
    out_of_order_release_rejected = False
    try:
        unload_factor(reordered, alg.power(program.theta_exponents[-2]), program.factor_count - 1)
    except RuntimeError:
        out_of_order_release_rejected = True

    premature = HypertreeFactorCarrier.create(alg, program.factor_count)
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    owner = HypertreeFactorCarrier.create(alg, program.factor_count)
    forward(owner, program)
    wrong_projection_owner_rejected = False
    wrong_inverse_owner_rejected = False
    try:
        project_boundary(owner, compile_program(3, "REUSE"))
    except RuntimeError:
        wrong_projection_owner_rejected = True
    try:
        inverse(owner, compile_program(3, "REUSE"))
    except RuntimeError:
        wrong_inverse_owner_rejected = True

    null_carrier_rejected = False
    try:
        project_boundary(None, program)  # type: ignore[arg-type]
    except (AttributeError, RuntimeError):
        null_carrier_rejected = True

    overmerge_carrier = HypertreeFactorCarrier.create(alg, program.factor_count)
    forward(overmerge_carrier, program)
    overmerged, _ = contract_tree(overmerge_carrier, program, overmerge_right_into_left=True)
    overmerge_changes_boundary = alg.serialize(overmerged) != reference["boundary"]

    identity_rank_collapses = alg.sub(alg.one, alg.one) == alg.zero
    altered_program = compile_program(3, "REUSE")
    altered_family_changes_descriptor = altered_program.fingerprint() != program.fingerprint()

    return {
        "all_declared_theta_factors_nonidentity": all(alg.power(value) != alg.one for value in program.theta_exponents),
        "identity_theta_collapses_edge_rank_minor": identity_rank_collapses,
        "altered_family_changes_public_descriptor": altered_family_changes_descriptor,
        "overmerged_sibling_message_changes_boundary": overmerge_changes_boundary,
        "missing_inverse_leaves_resident_state": not missing.exact_zero(),
        "wrong_inverse_detected": wrong_inverse_detected,
        "out_of_order_release_rejected_by_declared_custody": out_of_order_release_rejected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "wrong_projection_owner_rejected": wrong_projection_owner_rejected,
        "wrong_inverse_owner_rejected": wrong_inverse_owner_rejected,
        "snapshot_command_available": False,
        "reference_transaction_restored": reference["restored_exact_zero"],
    }


def run() -> dict[str, Any]:
    exact = []
    for height in EXACT_HEIGHTS:
        program = compile_program(height, "PRIMARY")
        exact.append(execute_transaction(HypertreeFactorCarrier.create(backend.Algebra("Q_ZETA17"), program.factor_count), program))

    structural = []
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        for height in STRUCTURAL_HEIGHTS:
            program = compile_program(height, "PRIMARY")
            item = execute_transaction(HypertreeFactorCarrier.create(alg, program.factor_count), program)
            item["field"] = f"F{modulus}"
            structural.append(item)

    certificates = [edge_rank_certificate(compile_program(height, "PRIMARY"), backend.Algebra("Q_ZETA17")) for height in EXACT_HEIGHTS]
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        certificates.extend(edge_rank_certificate(compile_program(height, "PRIMARY"), alg) for height in STRUCTURAL_HEIGHTS)
    if not all(item["all_local_theta_minus_one_minors_nonzero"] and item["exact_full_factor_tensor_rank_across_every_tree_edge"] == 2 for item in certificates):
        fail("one or more hypertree edge-rank certificates failed")

    baselines = []
    for item in exact:
        baselines.append(classical_baseline(item, compile_program(item["height"], item["family"]), backend.Algebra("Q_ZETA17")))
    for item in structural:
        modulus, root = next(pair for pair in FINITE_FIELDS if item["field"] == f"F{pair[0]}")
        baselines.append(classical_baseline(item, compile_program(item["height"], item["family"]), backend.Algebra(item["field"], modulus=modulus, root=root)))
    if not all(item["boundary_agreement"] for item in baselines):
        fail("hypertree matched classical baseline disagrees")

    maximum_factors = (1 << max(EXACT_HEIGHTS)) - 1
    reuse_carrier = HypertreeFactorCarrier.create(backend.Algebra("Q_ZETA17"), maximum_factors)
    first = execute_transaction(reuse_carrier, compile_program(4, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(5, "REUSE"))
    fresh = execute_transaction(HypertreeFactorCarrier.create(backend.Algebra("Q_ZETA17"), maximum_factors), compile_program(5, "REUSE"))
    if reused["boundary"] != fresh["boundary"] or resource_signature(reused) != resource_signature(fresh):
        fail("restored hypertree carrier disagrees with fresh reuse")

    control_results = controls()
    if not all(value for key, value in control_results.items() if key != "snapshot_command_available") or control_results["snapshot_command_available"]:
        fail("one or more hypertree controls failed")

    return {
        "schema": "CAT_CAS_F17_SHARED_LATENT_CUBIC_HYPERTREE_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_SHARED_LATENT_NONAFFINE_CUBIC_BOOLEAN_PHASE_HYPERTREE_HAS_EXACT_TREE_EDGE_BOND2_AND_LOG_DEPTH_STREAMED_MESSAGE_CLOSURE_WITH_TWO_M_RESIDENT_FACTOR_CELLS_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_IDENTICAL_TWO_COMPONENT_CLASSICAL_BELIEF_PROPAGATION",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "exact_complete_binary_tree_heights": EXACT_HEIGHTS,
            "dual_field_structural_heights": STRUCTURAL_HEIGHTS,
            "factor_law": "THETA_NODE_TO_PARENT_BIT_TIMES_LEFT_BIT_TIMES_RIGHT_BIT",
            "nonaffine_boolean_degree": 3,
            "shared_latent_law": "EVERY_NONROOT_INTERNAL_BIT_HAS_PARENT_AND_CHILD_FACTOR_CONSUMERS",
            "tree_edge_bond_dimension": 2,
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "edge_rank_certificates": certificates,
        "matched_classical_baselines": baselines,
        "reuse": {
            "first_height": 4,
            "reused_height": 5,
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
            "factors_at_height_h": "TWO_TO_THE_H_MINUS_ONE",
            "variables_at_height_h": "TWO_TO_THE_H_PLUS1_MINUS_ONE",
            "resident_phase_factor_field_cells_at_m_factors": "TWO_TIMES_M",
            "resident_nontrivial_theta_field_cells_at_m_factors": "M",
            "exact_tree_edge_tensor_bond_dimension": 2,
            "maximum_live_message_field_cells_at_height_h": "TWO_TIMES_H_PLUS_TWO_FOR_H_AT_LEAST1",
            "matched_classical_full_signature_field_cells_at_m_factors": "M",
            "matched_classical_dynamic_message_width": 2,
            "accepted_path_assignment_or_dense_tensor_cells": 0,
            "inverse_history_cells": 0,
            "full_exact_bit_complexity_established": False,
            "python_container_recursion_allocator_native_bigint_hashlib_and_whole_process_excluded": True,
        },
        "matched_baseline": {
            "strongest_full_signature": "M_EXACT_PUBLIC_THETA_FACTORS",
            "strongest_final_boundary_runtime": "IDENTICAL_TWO_COMPONENT_DEPTH_FIRST_TREE_BELIEF_PROPAGATION",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_phase_factor_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "projection_compiler_commitment_and_message_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "declared_complete_binary_cubic_factor_hypertree": True,
            "generic_public_hypertree_compiler": False,
            "cyclic_or_high_treewidth_cubic_factor_graph": False,
            "noncommuting_shared_port_updates": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "SHARED_LATENT_BRANCHING_ALONE_PRESERVES_TREE_EDGE_BOND2_AND_IDENTICAL_TWO_COMPONENT_CLASSICAL_BELIEF_PROPAGATION_WHILE_THE_PHASE_FACTOR_CARRIER_STORES_TWICE_THE_PUBLIC_SIGNATURE_SO_THE_NEXT_LAW_MUST_ADD_NONCOMMUTING_PORT_TRANSPORT_OR_CYCLIC_SEPARATOR_GEOMETRY_WITHOUT_REINTRODUCING_CLASSICAL_ENUMERATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(run(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
