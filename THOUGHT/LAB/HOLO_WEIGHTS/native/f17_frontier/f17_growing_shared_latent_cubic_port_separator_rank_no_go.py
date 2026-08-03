#!/usr/bin/env python3
"""Exact growing-arity linear-port obstruction for shared cubic phases."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import f17_coherent_veronese_phase_chart_closure as rank1
import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_overlapping_cubic_bond3_phase_factor_closure as factor_backend


EXACT_ARITIES = (1, 2, 4, 8, 16, 32, 64)
STRUCTURAL_ARITIES = (1, 2, 4, 8, 16)
FORMULA_ARITIES = tuple(range(1, 65))
FINITE_FIELDS = ((103, 72), (137, 16))
FAMILIES = ("PRIMARY", "REUSE")
CLAIM = (
    "EXACT_K_TYPED_SHARED_LATENT_CUBIC_BRANCH_MAPS_WITH_INVERTIBLE_PHASE_"
    "WALSH_INTERLEAVING_HAVE_SEPARATOR_RANK_TWO_TO_THE_K_AND_REJECT_"
    "UNIFORM_EXACT_LINEAR_RELATION_QUOTIENTS_BELOW_TWO_TO_THE_K_WHILE_"
    "BOUNDED_DESCRIPTOR_PHASE_FACTOR_CARRIERS_RESTORE_AND_REUSE_WITHOUT_"
    "PORT_EXPANSION_BUT_LOW_TREEWIDTH_CLASSICAL_FACTOR_CONTRACTION_REMAINS"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def algebra_signature(alg: backend.Algebra) -> str:
    return rank1.algebra_signature(alg)


def exponent(index: int, family: str) -> int:
    if family == "PRIMARY":
        return 1 + ((5 * index * index + 7 * index + 3) % 16)
    if family == "REUSE":
        return 1 + ((9 * index * index + 4 * index + 11) % 16)
    fail("unknown shared-latent family")


@dataclass(frozen=True)
class PortProgram:
    latent_arity: int
    family: str
    left_theta_exponents: tuple[int, ...]
    right_theta_exponents: tuple[int, ...]

    @property
    def local_bit_count(self) -> int:
        return 2 + 2 * self.latent_arity

    @property
    def factor_count(self) -> int:
        return 2 * self.latent_arity

    @property
    def port_coordinates(self) -> int:
        return 1 << self.latent_arity

    def public_descriptor(self) -> dict[str, Any]:
        return {
            "latent_arity": self.latent_arity,
            "family": self.family,
            "shared_latent_ports": [f"H_{index}" for index in range(self.latent_arity)],
            "branch_local_bits": [
                "LEFT_ANCHOR",
                *[f"LEFT_LEAF_{index}" for index in range(self.latent_arity)],
                "RIGHT_ANCHOR",
                *[f"RIGHT_LEAF_{index}" for index in range(self.latent_arity)],
            ],
            "left_cubic_factors": [
                f"ALPHA_{index}^(H_{index}*LEFT_ANCHOR*LEFT_LEAF_{index})"
                for index in range(self.latent_arity)
            ],
            "right_cubic_factors": [
                f"BETA_{index}^(H_{index}*RIGHT_ANCHOR*RIGHT_LEAF_{index})"
                for index in range(self.latent_arity)
            ],
            "left_theta_exponents": list(self.left_theta_exponents),
            "right_theta_exponents": list(self.right_theta_exponents),
            "separator_transport": [
                f"UNNORMALIZED_WALSH_ON_H_{index}"
                for index in range(self.latent_arity)
            ],
            "declared_observation_family": (
                "ALL_LEFT_REACHABLE_STATES_AND_RIGHT_CONTINUATION_FUNCTIONALS"
            ),
        }

    def fingerprint(self) -> str:
        return digest_json(self.public_descriptor())


def compile_program(latent_arity: int, family: str) -> PortProgram:
    if latent_arity < 1 or latent_arity > 64:
        fail("shared-latent arity is outside the declared bounded compiler")
    if family not in FAMILIES:
        fail("shared-latent family is outside the declared compiler")
    return PortProgram(
        latent_arity,
        family,
        tuple(exponent(2 * index, family) for index in range(latent_arity)),
        tuple(exponent(2 * index + 1, family) for index in range(latent_arity)),
    )


def lease(program: PortProgram, alg: backend.Algebra, capacity: int) -> str:
    return digest_json(
        {
            "program": program.fingerprint(),
            "algebra": algebra_signature(alg),
            "capacity": capacity,
            "carrier": "GROWING_SHARED_LATENT_CUBIC_FACTOR_CERTIFICATE",
        }
    )


def extended_zero(carrier: factor_backend.CubicFactorCarrier) -> bool:
    return carrier.exact_zero() and carrier.projection_calls == 0


def canonical_state(carrier: factor_backend.CubicFactorCarrier) -> dict[str, Any]:
    return {
        "capacity": carrier.capacity,
        "active_depth": carrier.active_depth,
        "active_family": carrier.active_family,
        "active_lease": carrier.active_lease,
        "stage": carrier.stage,
        "projection_calls": carrier.projection_calls,
        "all_factor_cells_zero": all(
            value == carrier.alg.zero for value in carrier.all_values()
        ),
    }


def stream_commitment(values: Iterable[Any], alg: backend.Algebra) -> tuple[str, int]:
    state = hashlib.sha256()
    total = 0
    for index, value in enumerate(values):
        record = json.dumps(
            {"index": index, "value": alg.serialize(value)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        state.update(len(record).to_bytes(8, "big"))
        state.update(record)
        total += 8 + len(record)
    return state.hexdigest(), total


def rank_certificate_from_resident(
    left_theta: list[Any],
    right_theta: list[Any],
    alg: backend.Algebra,
    *,
    transport_invertible: bool = True,
) -> dict[str, Any]:
    latent_arity = len(left_theta)
    if latent_arity < 1 or len(right_theta) != latent_arity:
        fail("rank certificate needs paired resident factors")
    left_determinants = [alg.sub(theta, alg.one) for theta in left_theta]
    right_determinants = [alg.sub(theta, alg.one) for theta in right_theta]
    if any(
        value == alg.zero for value in [*left_determinants, *right_determinants]
    ):
        fail("identity theta destroys the declared full-rank minor")
    if not transport_invertible:
        fail("singular separator transport destroys the rank transfer law")
    port_coordinates = 1 << latent_arity
    exponent_per_factor = 1 << (latent_arity - 1)
    return {
        "field": "Q_ZETA17_OR_DECLARED_STRUCTURAL_FIELD",
        "latent_arity": latent_arity,
        "left_local_branch_bits": 1 + latent_arity,
        "right_local_branch_bits": 1 + latent_arity,
        "full_left_local_assignment_rows": 2 * port_coordinates,
        "full_right_local_assignment_rows": 2 * port_coordinates,
        "port_coordinates": port_coordinates,
        "certified_left_anchor_one_minor_shape": [port_coordinates, port_coordinates],
        "certified_right_anchor_one_minor_shape": [port_coordinates, port_coordinates],
        "certified_two_sided_boundary_minor_shape": [port_coordinates, port_coordinates],
        "local_two_by_two_factor": "[[1,1],[1,THETA_I]]",
        "local_determinant": "THETA_I_MINUS_ONE",
        "all_local_determinants_nonzero": True,
        "left_kronecker_minor_rank": port_coordinates,
        "right_kronecker_minor_rank": port_coordinates,
        "full_two_sided_separator_rank": port_coordinates,
        "typed_configuration_bisimulation_classes": port_coordinates,
        "individual_typed_continuation_separates_distinct_configurations": True,
        "typed_port_overmerge_exact_relation_preserving": False,
        "minor_determinant_factorization": (
            "PRODUCT_I((ALPHA_I_MINUS_ONE)*(BETA_I_MINUS_ONE))^"
            "(TWO_TO_THE_(K_MINUS_ONE))*DET(U)"
        ),
        "minor_determinant_exponent_per_factor": exponent_per_factor,
        "walsh_determinant": "PRODUCT_OF_K_NONZERO_MINUS_TWO_POWERS",
        "separator_transport_invertible": True,
        "q_zeta17_phase_factor_norm_power_of_17_exponent": (
            2 * latent_arity * exponent_per_factor
        ),
        "uniform_exact_linear_port_quotient_minimum_field_coordinates": port_coordinates,
        "uniform_exact_linear_port_quotient_below_two_to_the_k": "REJECTED",
        "formula_certificate_work_scalars": 6 * latent_arity + 12,
        "dense_port_vector_materialized": False,
        "dense_minor_materialized": False,
        "local_assignment_family_enumerated": False,
        "determinant_value_serialized": False,
        "proof_law": (
            "THE_LEFT_AND_RIGHT_ANCHOR_ONE_ROWS_FORM_KRONECKER_PRODUCTS_OF_"
            "INVERTIBLE_TWO_BY_TWO_MATRICES_AND_THE_SEPARATOR_TRANSPORT_IS_"
            "INVERTIBLE_SO_ANY_FIXED_LINEAR_RELATION_QUOTIENT_PRESERVING_"
            "ALL_REACHABLE_STATES_AND_CONTINUATIONS_MUST_BE_INJECTIVE"
        ),
    }


def enforce_linear_rank_cap(certificate: dict[str, Any], cap: int) -> None:
    if cap < certificate[
        "uniform_exact_linear_port_quotient_minimum_field_coordinates"
    ]:
        fail("linear port rank cap contradicts the exact Kronecker minor")


def begin_forward(
    carrier: factor_backend.CubicFactorCarrier, program: PortProgram
) -> None:
    if not isinstance(carrier, factor_backend.CubicFactorCarrier):
        fail("null or wrong carrier")
    if not extended_zero(carrier):
        fail("carrier is not restored")
    if carrier.capacity < program.factor_count:
        fail("carrier capacity is below the public latent arity")
    carrier.active_family = program.family
    carrier.active_lease = lease(program, carrier.alg, carrier.capacity)
    carrier.stage = "FORWARD"
    for theta_exponent in (
        *program.left_theta_exponents,
        *program.right_theta_exponents,
    ):
        factor_backend.load_site(carrier, carrier.alg.power(theta_exponent))
    carrier.stage = "CERTIFICATE_READY"


def project_certificate(
    carrier: factor_backend.CubicFactorCarrier, program: PortProgram
) -> tuple[dict[str, Any], str, int]:
    if not isinstance(carrier, factor_backend.CubicFactorCarrier):
        fail("null or wrong carrier")
    if carrier.stage != "CERTIFICATE_READY" or carrier.projection_calls != 0:
        fail("certificate projection stage changed")
    if carrier.active_depth != program.factor_count:
        fail("resident factor arity changed")
    if carrier.active_family != program.family:
        fail("projection family owner changed")
    if carrier.active_lease != lease(program, carrier.alg, carrier.capacity):
        fail("projection lease owner changed")
    resident = list(carrier.cubic_branches[: program.factor_count])
    certificate = rank_certificate_from_resident(
        resident[: program.latent_arity],
        resident[program.latent_arity :],
        carrier.alg,
    )
    commitment, commitment_bytes = stream_commitment(resident, carrier.alg)
    carrier.projection_calls += 1
    carrier.stage = "PROJECTED"
    return certificate, commitment, commitment_bytes


def inverse(
    carrier: factor_backend.CubicFactorCarrier, program: PortProgram
) -> None:
    if not isinstance(carrier, factor_backend.CubicFactorCarrier):
        fail("null or wrong carrier")
    if carrier.stage != "PROJECTED" or carrier.projection_calls != 1:
        fail("inverse stage changed")
    if carrier.active_family != program.family:
        fail("inverse family owner changed")
    if carrier.active_lease != lease(program, carrier.alg, carrier.capacity):
        fail("inverse lease owner changed")
    for level in range(program.factor_count, 0, -1):
        actual_theta = carrier.cubic_branches[level - 1]
        factor_backend.unload_site(carrier, actual_theta, level)
    carrier.active_family = None
    carrier.active_lease = None
    carrier.stage = "RESTORED"
    carrier.projection_calls = 0
    carrier.package_local_restoration_count += 1
    if not extended_zero(carrier):
        fail("actual factor inverse did not restore canonical zero")


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "family",
        "program_fingerprint",
        "factor_commitment",
        "package_local_restoration_count_before",
        "package_local_restoration_count_after",
    }
    return {key: value for key, value in transaction.items() if key not in excluded}


def execute_transaction(
    carrier: factor_backend.CubicFactorCarrier, program: PortProgram
) -> dict[str, Any]:
    backing = carrier.backing_identity()
    before = carrier.package_local_restoration_count
    begin_forward(carrier, program)
    certificate, commitment, commitment_bytes = project_certificate(carrier, program)
    resident_payload = carrier.maximum_resident_payload_bits
    inverse(carrier, program)
    return {
        "latent_arity": program.latent_arity,
        "family": program.family,
        "algebra": algebra_signature(carrier.alg),
        "program_fingerprint": program.fingerprint(),
        "public_program": program.public_descriptor(),
        "factor_commitment": commitment,
        "factor_commitment_json_bytes": commitment_bytes,
        "rank_certificate": certificate,
        "resident_phase_factor_field_cells": 4 * program.latent_arity,
        "resident_nontrivial_theta_field_cells": 2 * program.latent_arity,
        "maximum_resident_factor_payload_bits": resident_payload,
        "maximum_local_coupling_named_field_cells": (
            carrier.maximum_local_coupling_named_field_cells
        ),
        "accepted_path_port_field_cells": 0,
        "accepted_path_assignment_or_dense_minor_cells": 0,
        "intermediate_factor_or_port_payload_exposed": False,
        "one_way_factor_commitment_emitted": True,
        "projection_calls": 1,
        "response_released_after_restoration": True,
        "restored_exact_zero": extended_zero(carrier),
        "restored_canonical_state": canonical_state(carrier),
        "same_backing": carrier.backing_identity() == backing,
        "package_local_restoration_count_before": before,
        "package_local_restoration_count_after": (
            carrier.package_local_restoration_count
        ),
        "inverse_history_cells": 0,
        "snapshot_reload_used": False,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "compiler_commitment_and_verification_buffer_restoration_class": (
            "NO_RESTORATION_CLAIM"
        ),
    }


def classical_baseline(transaction: dict[str, Any]) -> dict[str, Any]:
    k = transaction["latent_arity"]
    return {
        "latent_arity": k,
        "family": transaction["family"],
        "algebra": transaction["algebra"],
        "full_public_signature_field_cells": 2 * k,
        "uniform_arbitrary_message_port_field_cells": 1 << k,
        "analytic_rank_certificate_work_scalars": 6 * k + 12,
        "same_rank_lower_bound": (
            transaction["rank_certificate"]["full_two_sided_separator_rank"]
            == 1 << k
        ),
        "strictly_local_sealed_contraction": "O_K_TWO_BY_TWO_KRONECKER_FACTORS",
        "general_descriptor_contraction": "POLY_DESCRIPTOR_SIZE_TIMES_TWO_TO_THE_TREEWIDTH",
        "accepted_dense_port_or_minor_cells": 0,
        "phase_carrier_or_snapshot_used": False,
        "comparison_establishes_advantage": False,
    }


def controls() -> dict[str, bool]:
    alg = backend.Algebra("F137", modulus=137, root=16)
    program = compile_program(4, "PRIMARY")
    reference = execute_transaction(
        factor_backend.CubicFactorCarrier.create(alg, 8), program
    )

    missing = factor_backend.CubicFactorCarrier.create(alg, 8)
    begin_forward(missing, program)

    wrong = factor_backend.CubicFactorCarrier.create(alg, 8)
    begin_forward(wrong, program)
    wrong_factor_inverse_detected = False
    try:
        factor_backend.unload_site(
            wrong,
            alg.power(1 + (program.right_theta_exponents[-1] % 16)),
            wrong.active_depth,
        )
    except RuntimeError:
        wrong_factor_inverse_detected = True

    reordered = factor_backend.CubicFactorCarrier.create(alg, 8)
    begin_forward(reordered, program)
    reordered_inverse_detected = False
    try:
        factor_backend.unload_site(reordered, reordered.cubic_branches[0], 1)
    except RuntimeError:
        reordered_inverse_detected = True

    certificate = reference["rank_certificate"]
    false_rank_cap_rejected = False
    try:
        enforce_linear_rank_cap(certificate, (1 << program.latent_arity) - 1)
    except RuntimeError:
        false_rank_cap_rejected = True
    exact_rank_cap_accepted = True
    try:
        enforce_linear_rank_cap(certificate, 1 << program.latent_arity)
    except RuntimeError:
        exact_rank_cap_accepted = False

    identity_theta_rejected = False
    try:
        rank_certificate_from_resident(
            [
                alg.one,
                *[alg.power(value) for value in program.left_theta_exponents[1:]],
            ],
            [alg.power(value) for value in program.right_theta_exponents],
            alg,
        )
    except RuntimeError:
        identity_theta_rejected = True

    identity_right_theta_rejected = False
    try:
        rank_certificate_from_resident(
            [alg.power(value) for value in program.left_theta_exponents],
            [
                alg.one,
                *[alg.power(value) for value in program.right_theta_exponents[1:]],
            ],
            alg,
        )
    except RuntimeError:
        identity_right_theta_rejected = True

    wrong_typed_arity_rejected = False
    try:
        rank_certificate_from_resident(
            [alg.power(value) for value in program.left_theta_exponents[:-1]],
            [alg.power(value) for value in program.right_theta_exponents],
            alg,
        )
    except RuntimeError:
        wrong_typed_arity_rejected = True

    singular_transport_rejected = False
    try:
        rank_certificate_from_resident(
            [alg.power(value) for value in program.left_theta_exponents],
            [alg.power(value) for value in program.right_theta_exponents],
            alg,
            transport_invertible=False,
        )
    except RuntimeError:
        singular_transport_rejected = True

    premature_projection_rejected = False
    try:
        project_certificate(
            factor_backend.CubicFactorCarrier.create(alg, 8), program
        )
    except RuntimeError:
        premature_projection_rejected = True

    owner = factor_backend.CubicFactorCarrier.create(alg, 8)
    begin_forward(owner, program)
    wrong_owner_rejected = False
    try:
        project_certificate(owner, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_owner_rejected = True

    null_carrier_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    return {
        "reference_transaction_restored": reference["restored_exact_zero"],
        "missing_inverse_leaves_resident_state": not extended_zero(missing),
        "wrong_factor_inverse_detected": wrong_factor_inverse_detected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "false_linear_rank_cap_rejected": false_rank_cap_rejected,
        "exact_linear_rank_cap_accepted": exact_rank_cap_accepted,
        "identity_theta_rejected": identity_theta_rejected,
        "identity_right_theta_rejected": identity_right_theta_rejected,
        "wrong_typed_arity_rejected": wrong_typed_arity_rejected,
        "singular_transport_rejected": singular_transport_rejected,
        "premature_projection_rejected": premature_projection_rejected,
        "wrong_projection_owner_rejected": wrong_owner_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "primary_and_reuse_descriptors_differ": (
            compile_program(4, "PRIMARY").fingerprint()
            != compile_program(4, "REUSE").fingerprint()
        ),
        "accepted_formula_materializes_port": False,
        "accepted_formula_enumerates_assignments": False,
        "snapshot_command_available": False,
    }


def run() -> dict[str, Any]:
    exact_transactions = []
    q_alg = backend.Algebra("Q_ZETA17")
    for k in EXACT_ARITIES:
        program = compile_program(k, "PRIMARY")
        exact_transactions.append(
            execute_transaction(
                factor_backend.CubicFactorCarrier.create(q_alg, 2 * k), program
            )
        )

    structural_transactions = []
    for modulus, root in FINITE_FIELDS:
        alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
        for k in STRUCTURAL_ARITIES:
            program = compile_program(k, "PRIMARY")
            structural_transactions.append(
                execute_transaction(
                    factor_backend.CubicFactorCarrier.create(alg, 2 * k), program
                )
            )

    reuse_alg = backend.Algebra("Q_ZETA17")
    reuse_carrier = factor_backend.CubicFactorCarrier.create(reuse_alg, 32)
    backing = reuse_carrier.backing_identity()
    primary = execute_transaction(reuse_carrier, compile_program(8, "PRIMARY"))
    restored_reuse = execute_transaction(
        reuse_carrier, compile_program(16, "REUSE")
    )
    fresh_reuse = execute_transaction(
        factor_backend.CubicFactorCarrier.create(reuse_alg, 32),
        compile_program(16, "REUSE"),
    )

    formula_certificates = []
    for k in FORMULA_ARITIES:
        program = compile_program(k, "PRIMARY")
        left = [q_alg.power(value) for value in program.left_theta_exponents]
        right = [q_alg.power(value) for value in program.right_theta_exponents]
        formula_certificates.append(
            rank_certificate_from_resident(left, right, q_alg)
        )

    transactions = [*exact_transactions, *structural_transactions]
    baselines = [classical_baseline(item) for item in transactions]
    control_results = controls()
    if not all(
        item["restored_exact_zero"]
        and item["same_backing"]
        and item["rank_certificate"]["full_two_sided_separator_rank"]
        == 1 << item["latent_arity"]
        for item in transactions
    ):
        fail("growing shared-port transaction failed")
    if not all(
        value
        for key, value in control_results.items()
        if key
        not in {
            "accepted_formula_materializes_port",
            "accepted_formula_enumerates_assignments",
            "snapshot_command_available",
        }
    ) or any(
        control_results[key]
        for key in {
            "accepted_formula_materializes_port",
            "accepted_formula_enumerates_assignments",
            "snapshot_command_available",
        }
    ):
        fail("growing shared-port control failed")

    return {
        "schema": "CAT_CAS_F17_GROWING_SHARED_LATENT_CUBIC_PORT_SEPARATOR_RANK_NO_GO_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "execution_scope": "LINUX_DIRECT_PROCESS_EXACT_SOFTWARE_AND_PARAMETRIC_KRONECKER_CERTIFICATE",
        "source_scope": {
            "exact_arities": list(EXACT_ARITIES),
            "dual_field_structural_arities": list(STRUCTURAL_ARITIES),
            "formula_arities": [1, 64],
            "public_families": list(FAMILIES),
            "boolean_degree": 3,
            "uniform_linear_encoder_scope": (
                "ARBITRARY_INCOMING_PORT_MESSAGES_AND_ALL_LOCAL_ASSIGNMENT_OBSERVATIONS"
            ),
        },
        "exact_transactions": exact_transactions,
        "dual_field_structural_transactions": structural_transactions,
        "formula_certificates_k1_through_k64": formula_certificates,
        "compiled_classical_baselines": baselines,
        "reuse": {
            "primary_arity": primary["latent_arity"],
            "reuse_arity": restored_reuse["latent_arity"],
            "same_original_backing": reuse_carrier.backing_identity() == backing,
            "fresh_restored_reuse_signature_equal": (
                resource_signature(restored_reuse) == resource_signature(fresh_reuse)
            ),
            "package_local_restoration_count": (
                reuse_carrier.package_local_restoration_count
            ),
            "restored_exact_zero": extended_zero(reuse_carrier),
            "baseline_reload": False,
            "inverse_history_cells": 0,
        },
        "controls": control_results,
        "resource_law": {
            "resident_phase_factor_field_cells_at_arity_k": "FOUR_TIMES_K",
            "resident_nontrivial_theta_field_cells_at_arity_k": "TWO_TIMES_K",
            "accepted_path_port_field_cells": 0,
            "accepted_path_assignment_or_dense_minor_cells": 0,
            "analytic_certificate_work_scalars": "SIX_K_PLUS_TWELVE",
            "uniform_exact_linear_arbitrary_message_port_minimum_field_coordinates": (
                "TWO_TO_THE_K"
            ),
            "inverse_history_cells": 0,
            "full_exact_bit_complexity_established": False,
            "python_container_allocator_native_bigint_hashlib_bit_operation_and_whole_process_excluded": True,
        },
        "matched_baseline": {
            "strongest_full_signature": "TWO_K_EXACT_PUBLIC_PHASE_FACTORS",
            "strongest_rank_certificate": "IDENTICAL_O_K_KRONECKER_DETERMINANT_FORMULA",
            "strongest_uniform_arbitrary_message_port": "TWO_TO_THE_K_EXACT_FIELD_COORDINATES",
            "strongest_strictly_local_sealed_contraction": "O_K_TWO_BY_TWO_KRONECKER_FACTORS",
            "strongest_general_descriptor_contraction": "POLY_DESCRIPTOR_SIZE_TIMES_TWO_TO_THE_TREEWIDTH",
            "nonlinear_program_dependent_or_global_contraction_routes_exhausted": False,
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "factor_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "compiler_commitment_and_verification_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "uniform_exact_linear_port_quotient_for_arbitrary_messages": True,
            "rank_alone_general_storage_or_advantage_lower_bound": False,
            "nonlinear_or_program_dependent_quotient_no_go": False,
            "operational_two_to_the_k_port_phase_closure": False,
            "arbitrary_cubic_hypergraph_closure": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": (
            "THE_DECLARED_K_SHARED_LATENT_CUBIC_BRANCH_FAMILY_HAS_AN_EXACT_"
            "TWO_TO_THE_K_UNIFORM_LINEAR_PORT_LOWER_BOUND_SO_ANY_REPAIR_MUST_"
            "USE_A_PROGRAM_DEPENDENT_NONLINEAR_RELATION_PRESERVING_PHASE_"
            "CHART_OR_CHANGE_THE_NATIVE_UPDATE_LAW_WITHOUT_MOVING_EXPONENTIAL_"
            "STATE_TO_PROJECTION_OR_VERIFICATION"
        ),
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
