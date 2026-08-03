#!/usr/bin/env python3
"""Exact fixed-bond Gray phase-factor chart for the M132 coupling family."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend


DECLARED_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
FINITE_FIELD_DEPTHS = (1, 2, 3, 4, 5, 6)
PROGRAM_DEPTHS = tuple(sorted(set(DECLARED_DEPTHS + FINITE_FIELD_DEPTHS)))
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE")
FINAL_BOUNDARY = "K_MINUS_1_MODE0_ONE_MODE1_OCCUPATION"


def fail(message: str) -> None:
    raise RuntimeError(message)


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


def eta_exponent(level: int, family: str) -> int:
    if level < 1:
        fail("coupling level must be positive")
    if family == "PRIMARY":
        return 1 + ((2 * level) % 16)
    if family == "REUSE":
        return 2 + ((2 * level) % 16)
    fail("phase-factor family changed")


@dataclass(frozen=True)
class FactorProgram:
    depth: int
    family: str
    eta_exponents: tuple[int, ...]
    final_boundary: str = FINAL_BOUNDARY

    @property
    def conceptual_components(self) -> int:
        return 1 << self.depth

    @property
    def k(self) -> int:
        return 2 * self.conceptual_components - 2

    def fingerprint(self) -> str:
        return rank1.digest_json(public_program_descriptor(self))


def compile_program(depth: int, family: str) -> FactorProgram:
    if depth not in PROGRAM_DEPTHS or family not in FAMILIES:
        fail("phase-factor program identity changed")
    program = FactorProgram(
        depth=depth,
        family=family,
        eta_exponents=tuple(
            eta_exponent(level, family) for level in range(1, depth + 1)
        ),
    )
    validate_program(program)
    return program


def validate_program(program: FactorProgram) -> None:
    if program.depth not in PROGRAM_DEPTHS or program.family not in FAMILIES:
        fail("phase-factor program domain changed")
    if program.eta_exponents != tuple(
        eta_exponent(level, program.family)
        for level in range(1, program.depth + 1)
    ):
        fail("phase-factor exponent schedule changed")
    if any(not 1 <= exponent <= 16 for exponent in program.eta_exponents):
        fail("singular phase-factor exponent admitted")
    if program.final_boundary != FINAL_BOUNDARY:
        fail("phase-factor final boundary changed")


def public_program_descriptor(program: FactorProgram) -> dict[str, Any]:
    return {
        "depth": program.depth,
        "family": program.family,
        "chart": "GRAY_CODE_NEAREST_NEIGHBOR_PHASE_FACTOR_MPS",
        "conceptual_component_count": str(program.conceptual_components),
        "degree": str(program.k),
        "reflection_center_law": "A_LEVEL_EQUALS_TWO_TO_THE_LEVEL_MINUS_ONE",
        "eta_exponent_schedule": list(program.eta_exponents),
        "fixed_delta_wiring": "BINARY_BITS_WITH_GRAY_XOR_NEIGHBOR_FACTORS",
        "final_boundary": program.final_boundary,
    }


def lease(program: FactorProgram, alg: backend.Algebra, capacity: int) -> str:
    return rank1.digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "capacity": capacity,
            "carrier": "GRAY_CODE_BOND2_PHASE_FACTOR",
        }
    )


@dataclass
class PhaseFactorCarrier:
    alg: backend.Algebra
    capacity: int
    identity_branches: list[Any]
    reflected_branches: list[Any]
    active_depth: int = 0
    package_local_restoration_count: int = 0
    active_lease: str | None = None
    active_family: str | None = None
    stage: str = "RESTORED"
    projection_calls: int = 0
    maximum_resident_payload_bits: int = 0
    maximum_local_coupling_named_field_cells: int = 0

    @classmethod
    def create(cls, alg: backend.Algebra, capacity: int) -> "PhaseFactorCarrier":
        if capacity < 1:
            fail("phase-factor capacity must be positive")
        carrier = cls(
            alg=alg,
            capacity=capacity,
            identity_branches=[alg.zero for _ in range(capacity)],
            reflected_branches=[alg.zero for _ in range(capacity)],
        )
        carrier.observe()
        return carrier

    def backing_identity(self) -> tuple[int, int, int]:
        return (id(self), id(self.identity_branches), id(self.reflected_branches))

    def active_values(self) -> list[Any]:
        values: list[Any] = []
        for site in range(self.active_depth):
            values.extend(
                (self.identity_branches[site], self.reflected_branches[site])
            )
        return values

    def all_values(self) -> list[Any]:
        return [*self.identity_branches, *self.reflected_branches]

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
                "reflected": [self.alg.serialize(v) for v in self.reflected_branches],
            }
        )


def load_seeded_site(carrier: PhaseFactorCarrier, eta: Any) -> None:
    site = carrier.active_depth
    if site >= carrier.capacity:
        fail("phase-factor capacity exceeded")
    if (
        carrier.identity_branches[site] != carrier.alg.zero
        or carrier.reflected_branches[site] != carrier.alg.zero
    ):
        fail("phase-factor site was not empty")
    carrier.identity_branches[site] = carrier.alg.one
    carrier.reflected_branches[site] = carrier.alg.zero
    old_identity = carrier.identity_branches[site]
    old_reflected = carrier.reflected_branches[site]
    carrier.identity_branches[site] = carrier.alg.add(
        old_identity, carrier.alg.mul(eta, old_reflected)
    )
    carrier.reflected_branches[site] = carrier.alg.add(
        carrier.alg.mul(eta, old_identity), old_reflected
    )
    carrier.maximum_local_coupling_named_field_cells = max(
        carrier.maximum_local_coupling_named_field_cells, 4
    )
    carrier.active_depth += 1
    carrier.observe()


def unload_seeded_site(carrier: PhaseFactorCarrier, eta: Any, level: int) -> None:
    if level != carrier.active_depth or level < 1:
        fail("phase-factor inverse order changed")
    site = level - 1
    denominator = carrier.alg.sub(carrier.alg.one, carrier.alg.mul(eta, eta))
    if denominator == carrier.alg.zero:
        fail("singular phase-factor inverse")
    scale = carrier.alg.inverse(denominator)
    old_identity = carrier.identity_branches[site]
    old_reflected = carrier.reflected_branches[site]
    identity = carrier.alg.mul(
        scale, carrier.alg.sub(old_identity, carrier.alg.mul(eta, old_reflected))
    )
    reflected = carrier.alg.mul(
        scale, carrier.alg.sub(old_reflected, carrier.alg.mul(eta, old_identity))
    )
    carrier.maximum_local_coupling_named_field_cells = max(
        carrier.maximum_local_coupling_named_field_cells, 4
    )
    if identity != carrier.alg.one or reflected != carrier.alg.zero:
        fail("phase-factor inverse did not restore seeded empty branch")
    carrier.identity_branches[site] = carrier.alg.sub(identity, carrier.alg.one)
    carrier.reflected_branches[site] = reflected
    carrier.active_depth -= 1
    carrier.observe()


def forward(carrier: PhaseFactorCarrier, program: FactorProgram) -> None:
    validate_program(program)
    if not isinstance(carrier, PhaseFactorCarrier):
        fail("null or invalid phase-factor carrier")
    if not carrier.exact_zero() or carrier.capacity < program.depth:
        fail("phase-factor carrier not available")
    carrier.active_lease = lease(program, carrier.alg, carrier.capacity)
    carrier.active_family = program.family
    carrier.stage = "FORWARD"
    for exponent in program.eta_exponents:
        eta = carrier.alg.power(exponent)
        if carrier.alg.sub(carrier.alg.one, carrier.alg.mul(eta, eta)) == carrier.alg.zero:
            fail("singular phase-factor coupling")
        load_seeded_site(carrier, eta)


def require_active_program(
    carrier: PhaseFactorCarrier, program: FactorProgram
) -> None:
    validate_program(program)
    if not isinstance(carrier, PhaseFactorCarrier):
        fail("null or invalid phase-factor carrier")
    expected_lease = lease(program, carrier.alg, carrier.capacity)
    if (
        carrier.stage != "FORWARD"
        or carrier.active_depth != program.depth
        or carrier.active_family != program.family
        or carrier.active_lease != expected_lease
    ):
        fail("phase-factor program does not own active carrier lease")


def contract_total_first_moment(
    carrier: PhaseFactorCarrier, program: FactorProgram
) -> tuple[Any, Any, int]:
    require_active_program(carrier, program)
    total = carrier.alg.one
    moment = carrier.alg.zero
    maximum_named_field_cells = 2
    for level in range(1, program.depth + 1):
        eta = carrier.reflected_branches[level - 1]
        if carrier.identity_branches[level - 1] != carrier.alg.one:
            fail("phase-factor identity branch changed")
        prior_total = total
        prior_moment = moment
        center = rank1.field_integer(carrier.alg, (1 << level) - 1)
        total = carrier.alg.mul(carrier.alg.add(carrier.alg.one, eta), prior_total)
        moment = carrier.alg.add(
            carrier.alg.mul(carrier.alg.sub(carrier.alg.one, eta), prior_moment),
            carrier.alg.mul(carrier.alg.mul(eta, center), prior_total),
        )
        maximum_named_field_cells = max(maximum_named_field_cells, 4)
    return total, moment, maximum_named_field_cells


def project_boundary(carrier: PhaseFactorCarrier, program: FactorProgram) -> Any:
    if carrier.projection_calls != 0:
        fail("phase-factor boundary projected more than once")
    _, moment, _ = contract_total_first_moment(carrier, program)
    carrier.projection_calls += 1
    return carrier.alg.mul(rank1.field_integer(carrier.alg, program.k), moment)


def inverse(carrier: PhaseFactorCarrier, program: FactorProgram) -> None:
    require_active_program(carrier, program)
    if carrier.projection_calls != 1:
        fail("phase-factor inverse before final boundary")
    for level in range(program.depth, 0, -1):
        unload_seeded_site(
            carrier, carrier.alg.power(program.eta_exponents[level - 1]), level
        )
    carrier.active_lease = None
    carrier.active_family = None
    carrier.stage = "RESTORED"
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not carrier.exact_zero():
        fail("phase-factor carrier did not restore exact zero")


def factor_weight(
    program: FactorProgram, factors: list[Any], index: int, alg: backend.Algebra
) -> Any:
    if not 0 <= index < program.conceptual_components:
        fail("component index outside declared support")
    result = alg.one
    for level in range(1, program.depth + 1):
        bit = (index >> (level - 1)) & 1
        next_bit = (index >> level) & 1 if level < program.depth else 0
        if bit ^ next_bit:
            result = alg.mul(result, factors[level - 1])
    return result


def expanded_weights_verification_only(
    program: FactorProgram, alg: backend.Algebra
) -> list[Any]:
    weights = [alg.one]
    for level, exponent in enumerate(program.eta_exponents, start=1):
        eta = alg.power(exponent)
        weights = [*weights, *(alg.mul(eta, value) for value in reversed(weights))]
        if len(weights) != 1 << level:
            fail("verification-only expanded recurrence changed")
    return weights


def bond_certificate(program: FactorProgram, alg: backend.Algebra) -> dict[str, Any]:
    determinants = []
    for level in range(1, program.depth):
        eta = alg.power(program.eta_exponents[level - 1])
        determinant = alg.sub(alg.one, alg.mul(eta, eta))
        determinants.append(determinant != alg.zero)
    return {
        "depth": program.depth,
        "algebra": algebra_signature(alg),
        "internal_cut_count": max(0, program.depth - 1),
        "all_internal_edge_determinants_nonzero": all(determinants),
        "exact_maximum_weight_tensor_mps_bond_dimension": (
            1 if program.depth == 1 else 2
        ),
        "bond_one_rejected_for_depth_at_least_two": (
            program.depth == 1 or all(determinants)
        ),
        "determinant_values_serialized": False,
        "component_weights_materialized": False,
    }


def execute_transaction(
    carrier: PhaseFactorCarrier, program: FactorProgram
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
    _, _, projection_work = contract_total_first_moment(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    descriptor = public_program_descriptor(program)
    return {
        "depth": program.depth,
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "conceptual_component_count": str(program.conceptual_components),
        "degree": str(program.k),
        "boundary": carrier.alg.serialize(boundary),
        "factor_commitment": commitment,
        "factor_commitment_json_bytes": commitment_bytes,
        "resident_phase_factor_field_cells": 2 * program.depth,
        "resident_nontrivial_eta_field_cells": program.depth,
        "fixed_wiring_field_cells": 0,
        "exact_maximum_mps_bond_dimension": 1 if program.depth == 1 else 2,
        "maximum_local_coupling_named_field_cells": carrier.maximum_local_coupling_named_field_cells,
        "projection_dynamic_field_cells": projection_work,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "maximum_resident_factor_payload_bits": carrier.maximum_resident_payload_bits,
        "public_program_json_bytes": len(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ),
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "accepted_path_explicit_component_enumeration": False,
        "accepted_path_component_weight_cells": 0,
        "accepted_path_catalecticant_cells": 0,
        "accepted_path_dense_operator_cells": 0,
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
        "coupling_projection_and_compiler_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "intermediate_factor_payload_exposed_in_result": False,
        "one_way_factor_commitment_emitted": True,
    }


def run_case(depth: int, family: str, alg: backend.Algebra) -> dict[str, Any]:
    program = compile_program(depth, family)
    return execute_transaction(PhaseFactorCarrier.create(alg, depth), program)


def compiled_classical_baseline(
    transaction: dict[str, Any], program: FactorProgram, alg: backend.Algebra
) -> dict[str, Any]:
    total = alg.one
    moment = alg.zero
    transfer_total = alg.one
    transfer_moment = alg.zero
    transfer_homogeneous = alg.one
    factors = []
    for level, exponent in enumerate(program.eta_exponents, start=1):
        eta = alg.power(exponent)
        factors.append(eta)
        prior_total = total
        prior_moment = moment
        center = rank1.field_integer(alg, (1 << level) - 1)
        total = alg.mul(alg.add(alg.one, eta), prior_total)
        moment = alg.add(
            alg.mul(alg.sub(alg.one, eta), prior_moment),
            alg.mul(alg.mul(eta, center), prior_total),
        )
        prior_transfer_total = transfer_total
        prior_transfer_moment = transfer_moment
        prior_transfer_homogeneous = transfer_homogeneous
        transfer_total = alg.mul(
            alg.add(alg.one, eta), prior_transfer_total
        )
        transfer_moment = alg.add(
            alg.mul(alg.mul(eta, center), prior_transfer_total),
            alg.mul(
                alg.sub(alg.one, eta), prior_transfer_moment
            ),
        )
        transfer_homogeneous = alg.mul(
            alg.sub(alg.one, eta), prior_transfer_homogeneous
        )
    boundary = alg.mul(rank1.field_integer(alg, program.k), moment)
    commitment, record_bytes = rank1.stream_vector_commitment(factors, alg)
    transfer_commitment, transfer_record_bytes = rank1.stream_vector_commitment(
        [transfer_total, transfer_moment, transfer_homogeneous], alg
    )
    return {
        "depth": program.depth,
        "family": program.family,
        "algebra": algebra_signature(alg),
        "boundary_agreement": alg.serialize(boundary) == transaction["boundary"],
        "full_weight_signature_exact_factor_field_cells": program.depth,
        "full_weight_signature_fixed_bond_dimension": 1 if program.depth == 1 else 2,
        "final_boundary_dynamic_field_cells": 2,
        "final_boundary_maximum_named_update_field_cells": 4,
        "sealed_word_compiled_transfer_nonzero_field_cells": 3,
        "compiled_transfer_boundary_agreement": alg.serialize(
            alg.mul(rank1.field_integer(alg, program.k), transfer_moment)
        )
        == transaction["boundary"],
        "factor_commitment": commitment,
        "maximum_commitment_record_json_bytes": record_bytes,
        "compiled_transfer_commitment": transfer_commitment,
        "compiled_transfer_commitment_json_bytes": transfer_record_bytes,
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
    reference = run_case(4, "PRIMARY", alg)
    factors = [alg.power(exponent) for exponent in program.eta_exponents]
    expanded = expanded_weights_verification_only(program, alg)
    gray_agreement = all(
        factor_weight(program, factors, index, alg) == value
        for index, value in enumerate(expanded)
    )
    straight_upper = [alg.one]
    for exponent in program.eta_exponents:
        eta = alg.power(exponent)
        straight_upper = [
            *straight_upper,
            *(alg.mul(eta, value) for value in straight_upper),
        ]

    missing = PhaseFactorCarrier.create(alg, 4)
    forward(missing, program)
    missing_inverse_detected = not missing.exact_zero()

    wrong = PhaseFactorCarrier.create(alg, 4)
    forward(wrong, program)
    wrong_inverse_detected = False
    try:
        unload_seeded_site(
            wrong,
            alg.power((program.eta_exponents[-1] % 16) + 1),
            program.depth,
        )
    except RuntimeError:
        wrong_inverse_detected = True

    reordered = PhaseFactorCarrier.create(alg, 4)
    forward(reordered, program)
    reordered_inverse_rejected = False
    try:
        unload_seeded_site(
            reordered,
            alg.power(program.eta_exponents[-2]),
            program.depth - 1,
        )
    except RuntimeError:
        reordered_inverse_rejected = True

    premature = PhaseFactorCarrier.create(alg, 4)
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    wrong_projection_owner = PhaseFactorCarrier.create(alg, 4)
    forward(wrong_projection_owner, program)
    wrong_projection_owner_rejected = False
    try:
        project_boundary(wrong_projection_owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_projection_owner_rejected = True

    wrong_inverse_owner = PhaseFactorCarrier.create(alg, 4)
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

    singular_rejected = alg.sub(alg.one, alg.mul(alg.one, alg.one)) == alg.zero
    bond_one_rejected = bond_certificate(program, alg)[
        "exact_maximum_weight_tensor_mps_bond_dimension"
    ] == 2
    return {
        "gray_factor_matches_reversed_copy_expansion_m4_verification_only": gray_agreement,
        "straight_upper_copy_differs_from_required_reversed_copy": straight_upper != expanded,
        "all_declared_eta_exponents_nonsingular": all(
            1 <= exponent <= 16 for exponent in program.eta_exponents
        ),
        "eta_equal_one_singular_gate_detected": singular_rejected,
        "bond_one_rejected_by_nonzero_internal_edge_determinant": bond_one_rejected,
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
        run_case(depth, "PRIMARY", backend.Algebra("Q_ZETA17"))
        for depth in DECLARED_DEPTHS
    ]
    structural = []
    for modulus, root in FINITE_FIELDS:
        for depth in FINITE_FIELD_DEPTHS:
            item = run_case(
                depth,
                "PRIMARY",
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            item["field"] = f"F{modulus}"
            structural.append(item)

    certificates = [
        bond_certificate(compile_program(depth, "PRIMARY"), backend.Algebra("Q_ZETA17"))
        for depth in DECLARED_DEPTHS
    ]
    for modulus, root in FINITE_FIELDS:
        certificates.extend(
            bond_certificate(
                compile_program(depth, "PRIMARY"),
                backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
            )
            for depth in FINITE_FIELD_DEPTHS
        )
    if not all(
        item["all_internal_edge_determinants_nonzero"]
        and item["exact_maximum_weight_tensor_mps_bond_dimension"]
        == (1 if item["depth"] == 1 else 2)
        for item in certificates
    ):
        fail("one or more fixed-bond certificates failed")

    verification_only = []
    for depth in (1, 2, 3, 4, 5, 6, 7, 8):
        program = compile_program(depth if depth in DECLARED_DEPTHS else 8, "PRIMARY")
        if depth not in DECLARED_DEPTHS:
            # The m=3,5,6,7 checks use the same public law without becoming
            # accepted program depths.
            program = FactorProgram(
                depth=depth,
                family="PRIMARY",
                eta_exponents=tuple(
                    eta_exponent(level, "PRIMARY") for level in range(1, depth + 1)
                ),
            )
        alg = backend.Algebra("Q_ZETA17")
        factors = [alg.power(exponent) for exponent in program.eta_exponents]
        expanded = expanded_weights_verification_only(program, alg)
        verification_only.append(
            {
                "depth": depth,
                "component_count": len(expanded),
                "all_gray_factor_weights_agree": all(
                    factor_weight(program, factors, index, alg) == value
                    for index, value in enumerate(expanded)
                ),
                "accepted_path": False,
            }
        )
    if not all(item["all_gray_factor_weights_agree"] for item in verification_only):
        fail("verification-only Gray weight comparison failed")

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
                FactorProgram(
                    depth=item["depth"],
                    family=item["family"],
                    eta_exponents=tuple(
                        eta_exponent(level, item["family"])
                        for level in range(1, item["depth"] + 1)
                    ),
                ),
                backend.Algebra(item["field"], modulus=modulus, root=root),
            )
        )
    if not all(
        item["boundary_agreement"] and item["compiled_transfer_boundary_agreement"]
        for item in baselines
    ):
        fail("fixed-bond matched classical baseline disagrees")

    reuse_carrier = PhaseFactorCarrier.create(backend.Algebra("Q_ZETA17"), 16)
    first = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(16, "REUSE"))
    fresh = run_case(16, "REUSE", backend.Algebra("Q_ZETA17"))
    if reused["boundary"] != fresh["boundary"]:
        fail("restored phase-factor carrier disagrees with fresh reuse")
    if resource_signature(reused) != resource_signature(fresh):
        fail("restored phase-factor carrier changed resource signature")

    control_results = controls()
    if not all(
        value for key, value in control_results.items() if key != "snapshot_command_available"
    ) or control_results["snapshot_command_available"]:
        fail("one or more phase-factor controls failed")

    return {
        "schema": "CAT_CAS_F17_GRAY_CODE_BOND2_PHASE_FACTOR_CHART_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_GRAY_CODE_BOND2_PHASE_FACTOR_CHART_COMPRESSES_THE_DECLARED_ITERATED_NONCOMMUTING_AFFINE_REFLECTION_SUPERPOSITION_COMPONENT_WEIGHTS_FROM_TWO_TO_THE_M_EXPLICIT_COMPONENTS_TO_TWO_M_RESIDENT_PHASE_FACTOR_CELLS_ACROSS_DEPTH128_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_IS_AN_IDENTICAL_PUBLIC_MARKOV_FACTOR_CLASSICAL_RECURRENCE_AND_DOES_NOT_COMPACT_THE_GENERAL_COHERENT_POLYNOMIAL_OR_ESTABLISH_A_STABILIZER_RESOURCE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "exact_depths": DECLARED_DEPTHS,
            "dual_field_structural_depths": FINITE_FIELD_DEPTHS,
            "eta_schedule_preserves_m132_primary_depths1_to6": True,
            "weight_law": "PRODUCT_ETA_LEVEL_TO_GRAY_BIT",
            "mps_law": "OPEN_BOUNDARY_NEAREST_NEIGHBOR_BINARY_FACTOR_CHAIN",
            "ordinary_secant_rank_interpretation": "Q_ZETA17_ANALYTIC_ONLY",
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "fixed_bond_certificates": certificates,
        "verification_only_expanded_weight_checks": verification_only,
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
            "conceptual_components_at_depth_m": "TWO_TO_THE_M",
            "resident_phase_factor_field_cells_at_depth_m": "TWO_TIMES_M",
            "resident_nontrivial_eta_field_cells_at_depth_m": "M",
            "exact_maximum_weight_tensor_mps_bond_dimension": 2,
            "native_local_coupling_named_field_cells": 4,
            "final_boundary_projection_dynamic_field_cells": 4,
            "matched_classical_full_signature_field_cells_at_depth_m": "M",
            "matched_classical_final_boundary_dynamic_field_cells": 2,
            "accepted_path_component_weight_cells": 0,
            "accepted_path_catalecticant_cells": 0,
            "inverse_history_cells": 0,
            "full_exact_bit_complexity_established": False,
            "python_container_allocator_native_bigint_hashlib_and_whole_process_excluded": True,
        },
        "matched_baseline": {
            "strongest_full_weight_signature": "M_EXACT_PHASE_FACTORS_ON_PUBLIC_GRAY_CHAIN",
            "strongest_final_boundary_runtime": "TWO_DYNAMIC_MOMENT_SCALARS",
            "strongest_sealed_word": "THREE_NONZERO_LOWER_TRIANGULAR_TRANSFER_SCALARS",
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_phase_factor_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "coupling_projection_and_compiler_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "declared_gray_weight_factor_chart": True,
            "fixed_bond2_weight_signature": True,
            "general_coherent_polynomial_compaction": False,
            "arbitrary_boundary_compaction": False,
            "conventional_clifford_or_stabilizer_classification": False,
            "general_gaussian_closure_or_no_go": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_SPECIAL_GRAY_ORDERED_COMPONENT_WEIGHT_TENSOR_HAS_EXACT_BOND2_AND_LINEAR_FACTOR_STORAGE_BUT_THE_ASSOCIATED_COHERENT_BINARY_FORM_RETAINS_SECANT_RANK_TWO_TO_THE_M_ARBITRARY_HIGHER_MOMENT_BOUNDARIES_DO_NOT_CLOSE_ON_TWO_SCALARS_AND_THE_IDENTICAL_PUBLIC_CLASSICAL_FACTOR_RECURRENCE_REMAINS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run()
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
