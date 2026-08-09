#!/usr/bin/env python3
"""M257 source-pinned deterministic software forward-shadow diagnostic.

This package does not execute a catalytic transaction.  It constructs the
ordinary forward-only program that is already present as a prefix of five
independently verified CATVM transactions and records the exact assumptions
under which that prefix is a matched classical shadow.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MILESTONE = 257
RESULT = "PASS_EXACT_SOFTWARE_PHASE_TRANSACTION_FORWARD_SHADOW_STRICT_SCOPE"
CLAIM = (
    "BOUNDED_SOURCE_PINNED_DETERMINISTIC_SOFTWARE_PHASE_TRANSACTION_FORWARD_"
    "SHADOW_CONSTRUCTION_ACROSS_FIVE_ALGEBRAICALLY_DISTINCT_CATVM_PACKAGES_"
    "PROVES_EACH_ACCEPTED_FORWARD_PREFIX_PLUS_FINAL_PROJECTION_IS_AN_ORDINARY_"
    "CLASSICAL_PROGRAM_WITH_NO_GREATER_FORWARD_STATE_OR_WORK_WHILE_THE_CATVM_"
    "PATH_ADDS_A_NONZERO_INVERSE_RESTORATION_STAGE_SO_NO_SAME_DOMAIN_SOFTWARE_"
    "ADVANTAGE_FOLLOWS_WITHOUT_AN_EXTERNAL_RESOURCE_OR_JUSTIFIED_COMPARATOR_"
    "RESTRICTION"
)
CLAIM_CEILING = (
    "FIVE_DECLARED_EXACT_DETERMINISTIC_PYTHON_SOFTWARE_CATVM_PACKAGES_M248_"
    "M250_M253_M254_M256_AT_THEIR_SEALED_PRIMARY_FIXTURES_ONLY"
)


@dataclass(frozen=True)
class Spec:
    name: str
    result_file: str
    service_file: str
    service_hash_key: str
    run_kind: str
    boundary_fields: tuple[str, ...]
    same_backing_field: str
    forward_call: str
    projection_call: str
    inverse_call: str
    state_fields: tuple[tuple[str, str, str], ...]
    inverse_work_keys: tuple[str, ...]
    projection_work_keys: tuple[str, ...]
    baseline_fields: tuple[str, ...]


SPECS = (
    Spec(
        "M248_CUBIC_MAGIC_CATALYST",
        "CATVM_P5_CUBIC_MAGIC_CATALYST_RESULTS.json",
        "catvm_p5_cubic_magic_catalyst_service.py",
        "service_sha256",
        "TWO_SYNDROME_PRIMARY",
        ("final_amplitude",),
        "same_all_backings",
        "forward_use",
        "project",
        "restore_prefix",
        (
            ("catalyst_field_cells", "case", "catalyst_field_cells"),
            ("joint_scratch_field_cells", "case", "joint_interaction_scratch_field_cells"),
            ("phase_signature_field_cells", "case", "phase_signature_field_cells"),
            ("projection_workspace_field_cells", "case", "final_projection_workspace_field_cells"),
        ),
        ("inverse_catalyst_rematerializations",),
        ("final_projection_phase_multiplications", "final_projection_terms"),
        ("strongest_implemented_classical_baselines",),
    ),
    Spec(
        "M250_PROJECTIVE_WEYL_MERMIN",
        "CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_RESULTS.json",
        "catvm_projective_weyl_mermin_square_service.py",
        "catvm_projective_weyl_mermin_square_service.py",
        "PRIMARY",
        ("central_phase_exponent_mod4", "central_phase"),
        "same_carrier_and_custody_backings",
        "forward_action",
        "project_boundary",
        "inverse_action",
        (
            ("carrier_field_cells", "case", "hidden_carrier_field_cells"),
            ("scratch_field_cells", "case", "hidden_scratch_field_cells"),
            ("context_signature_cells", "case", "hidden_context_signature_cells"),
            ("typed_port_receipts", "case", "observable_port_count"),
        ),
        (
            "inverse_pauli_actions",
            "inverse_port_releases",
            "inverse_signature_compositions",
            "inverse_vector_cell_reads",
            "inverse_vector_cell_writes",
        ),
        ("final_overlap_accumulations", "final_overlap_field_multiplications"),
        ("strongest_classical_baseline", "strongest_transferable_descriptor_level_classical_baseline"),
    ),
    Spec(
        "M253_GRASSMANN_GAUSSIAN",
        "CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_RESULTS.json",
        "catvm_grassmann_gaussian_open_relation_service.py",
        "catvm_grassmann_gaussian_open_relation_service.py",
        "PRIMARY",
        ("top_form_boundary",),
        "same_relation_scratch_and_receipt_backings",
        "forward_module",
        "project_boundary",
        "inverse_module",
        (
            ("relation_field_cells", "case", "hidden_relation_field_cells"),
            ("fourier_scratch_field_cells", "case", "hidden_fourier_scratch_field_cells"),
            ("module_receipt_cells", "case", "hidden_module_receipt_cells"),
            ("compiled_intersection_field_cells", "work", "compiled_public_intersection_field_cells"),
            ("compiled_plan_references", "work", "compiled_public_module_plan_references"),
        ),
        (
            "inverse_carrier_field_writes",
            "inverse_field_accumulations",
            "inverse_field_inversions",
            "inverse_field_multiplications",
            "inverse_fourier_closures",
            "inverse_intersections",
            "inverse_scratch_field_writes_and_clears",
        ),
        ("boundary_field_accumulations", "boundary_field_multiplications"),
        (
            "strongest_fixed_fixture_classical_baseline",
            "strongest_implemented_transferable_descriptor_level_classical_baseline",
            "stronger_general_algorithmic_ceiling_not_claimed_as_implemented_baseline",
        ),
    ),
    Spec(
        "M254_GRASSMANN_FULL_EVEN",
        "CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_RESULTS.json",
        "catvm_grassmann_even_exterior_relation_service.py",
        "catvm_grassmann_even_exterior_relation_service.py",
        "PRIMARY",
        ("top_form_boundary",),
        "same_relation_scratch_and_receipt_backings",
        "forward_module",
        "project_boundary",
        "inverse_module",
        (
            ("relation_field_cells", "case", "hidden_even_relation_field_cells"),
            ("hodge_scratch_field_cells", "case", "hidden_hodge_scratch_field_cells"),
            ("module_receipt_cells", "case", "hidden_module_receipt_cells"),
            ("compiled_intersection_field_cells", "work", "compiled_public_intersection_factor_field_cells"),
            ("compiled_plan_references", "work", "compiled_public_module_plan_references"),
        ),
        (
            "inverse_carrier_field_writes",
            "inverse_factor_field_accumulations",
            "inverse_factor_field_inversions",
            "inverse_factor_field_multiplications",
            "inverse_factor_field_negations",
            "inverse_factor_rematerializations",
            "inverse_factor_returned_field_cells_materialized",
            "inverse_field_accumulations",
            "inverse_field_multiplications",
            "inverse_field_negations",
            "inverse_hodge_closures",
            "inverse_hodge_scratch_writes_and_clears",
            "inverse_intersections",
        ),
        (),
        ("strongest_fixed_fixture_classical_baseline", "strongest_implemented_transferable_descriptor_level_classical_baseline"),
    ),
    Spec(
        "M256_SCHUR_ALLPASS_WAVEFORM",
        "CATVM_SCHUR_ALLPASS_WAVEFORM_RESULTS.json",
        "catvm_schur_allpass_waveform_service.py",
        "catvm_schur_allpass_waveform_service.py",
        "PRIMARY",
        ("winding", "evaluation"),
        "same_waveform_scratch_and_receipt_backings",
        "forward",
        "project_boundary",
        "inverse",
        (
            ("resident_waveform_field_cells", "resource_shape", "resident_waveform_field_cells"),
            ("scratch_waveform_field_cells", "resource_shape", "scratch_waveform_field_cells"),
            ("receipt_rational_cells", "resource_shape", "receipt_rational_cells"),
            ("retained_degree_integer_cells", "resource_shape", "retained_final_degree_integer_cells_during_inverse"),
            ("retained_winding_integer_cells", "resource_shape", "retained_final_winding_integer_cells_during_inverse"),
        ),
        (
            "inverse_divisibility_checks",
            "inverse_field_scalar_multiplications",
            "inverse_field_subtractions",
            "inverse_public_rational_divisions",
            "inverse_public_rational_multiplications",
            "inverse_public_rational_subtractions",
            "inverse_scratch_writes",
            "inverse_sections",
        ),
        ("evaluation_field_additions", "evaluation_field_inversions", "evaluation_field_multiplications"),
        (
            "strongest_actual_boundary_classical_baseline",
            "strongest_fixed_fixture_classical_baseline",
            "strongest_full_formal_waveform_classical_baseline",
        ),
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call_tail(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def source_order(path: Path, spec: Spec) -> dict[str, int]:
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_transaction"]
    if len(functions) != 1:
        raise RuntimeError(f"{spec.name}: expected one run_transaction")
    function = functions[0]
    calls = [(node.lineno, call_tail(node)) for node in ast.walk(function) if isinstance(node, ast.Call)]
    forward = min(line for line, name in calls if name == spec.forward_call)
    projection = min(line for line, name in calls if name == spec.projection_call and line > forward)
    inverse = max(line for line, name in calls if name == spec.inverse_call and line > projection)
    release = max(line for line, name in calls if name == "release" and line > inverse)
    response = max(node.lineno for node in ast.walk(function) if isinstance(node, ast.Return))
    if not forward < projection < inverse < release < response:
        raise RuntimeError(f"{spec.name}: forward/projection/inverse/release/response order changed")
    return {
        "forward_line": forward,
        "projection_line": projection,
        "inverse_line": inverse,
        "release_line": release,
        "response_line": response,
    }


def primary_case(result: dict[str, Any], run_kind: str) -> dict[str, Any]:
    matches = [case for case in result["cases"] if case.get("run_kind") == run_kind]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {run_kind} case")
    return matches[0]


def restoration_class(result: dict[str, Any]) -> str:
    return str(result.get("restoration_class") or result.get("restoration_classification"))


def state_shape(case: dict[str, Any], spec: Spec) -> dict[str, int]:
    work = case["work"]
    resource_shape = case.get("resource_shape", {})
    sources = {"case": case, "work": work, "resource_shape": resource_shape}
    shape: dict[str, int] = {}
    for label, source, key in spec.state_fields:
        value = sources[source].get(key)
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{spec.name}: invalid state field {source}.{key}")
        shape[label] = value
    return shape


def analyze(root: Path, spec: Spec) -> dict[str, Any]:
    result_path = root / spec.result_file
    service_path = root / spec.service_file
    result = json.loads(result_path.read_text())
    expected_service_hash = result["source_dependencies"][spec.service_hash_key]
    actual_service_hash = sha256(service_path)
    if expected_service_hash != actual_service_hash:
        raise RuntimeError(f"{spec.name}: service source is not the sealed dependency")
    if result.get("classification") != "INDEPENDENTLY_VERIFIED_STRICT_SCOPE":
        raise RuntimeError(f"{spec.name}: source package is not independently verified")
    if result.get("verification_level") != "SEPARATE_REFERENCE_PARITY":
        raise RuntimeError(f"{spec.name}: source package lacks separate-reference parity")
    if restoration_class(result) != "EXACT_ALGEBRAIC_RESTORATION":
        raise RuntimeError(f"{spec.name}: source package lacks exact algebraic restoration")

    case = primary_case(result, spec.run_kind)
    if not case.get("canonical_after_restoration") or not case.get(spec.same_backing_field):
        raise RuntimeError(f"{spec.name}: restoration/backing precondition failed")
    if case.get("baseline_reload_used") is not False:
        raise RuntimeError(f"{spec.name}: baseline reload is not excluded")
    work = case["work"]
    inverse = {key: work[key] for key in spec.inverse_work_keys}
    if not inverse or not all(isinstance(value, int) and value > 0 for value in inverse.values()):
        raise RuntimeError(f"{spec.name}: nonzero inverse-stage witness missing")
    projection = {key: work[key] for key in spec.projection_work_keys}
    if not all(isinstance(value, int) and value > 0 for value in projection.values()):
        raise RuntimeError(f"{spec.name}: projection work witness invalid")
    forward = {
        key: value for key, value in work.items()
        if key.startswith("forward_") and isinstance(value, int)
    }
    if not forward:
        raise RuntimeError(f"{spec.name}: forward work witness missing")

    boundary = {field: case[field] for field in spec.boundary_fields}
    boundary_digest = hashlib.sha256(
        json.dumps(boundary, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    resources = result["resource_law"]
    baselines = {key: resources[key] for key in spec.baseline_fields}
    shape = state_shape(case, spec)
    order = source_order(service_path, spec)

    return {
        "package": spec.name,
        "sealed_result_file": spec.result_file,
        "sealed_result_sha256": sha256(result_path),
        "sealed_service_file": spec.service_file,
        "sealed_service_sha256": actual_service_hash,
        "primary_run_kind": spec.run_kind,
        "boundary_fields": list(spec.boundary_fields),
        "boundary_sha256": boundary_digest,
        "source_order": order,
        "accepted_forward_state_shape_upper_bound": shape,
        "shadow_forward_state_shape_upper_bound": shape,
        "reported_forward_work_vector_reused_by_shadow": forward,
        "reported_projection_work_vector_reused_by_shadow": projection,
        "reported_transaction_only_inverse_work_vector_omitted_by_shadow": inverse,
        "reported_forward_work_instrumentation_complete": False,
        "same_backing_restoration": True,
        "baseline_reload_used": False,
        "strongest_existing_classical_baselines": baselines,
        "constructive_shadow": "EXECUTE_THE_IDENTICAL_ACCEPTED_FORWARD_PREFIX_AND_FINAL_PROJECTION_THEN_RETURN_WITHOUT_CATVM_INVERSE_RESTORATION_OR_RELEASE_PROTOCOL",
        "exact_boundary_equality_reason": "THE_SHADOW_IS_THE_SAME_DETERMINISTIC_FORWARD_AND_PROJECTION_PREFIX",
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: software_phase_forward_shadow_dominance.py FRONTIER_DIR OUTPUT_JSON")
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2])
    packages = [analyze(root, spec) for spec in SPECS]
    if len({item["boundary_sha256"] for item in packages}) != len(packages):
        raise RuntimeError("boundary digests unexpectedly collide")

    source_dependencies = {
        Path(__file__).name: sha256(Path(__file__)),
        "software_phase_forward_shadow_dominance_separate_reference.py": sha256(root / "software_phase_forward_shadow_dominance_separate_reference.py"),
        "qualify_software_phase_forward_shadow_dominance.sh": sha256(root / "qualify_software_phase_forward_shadow_dominance.sh"),
    }
    for spec in SPECS:
        source_dependencies[spec.result_file] = sha256(root / spec.result_file)
        source_dependencies[spec.service_file] = sha256(root / spec.service_file)

    document = {
        "milestone": MILESTONE,
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "theorem": {
            "name": "DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_SUFFICIENT_CONDITION",
            "assumptions": [
                "THE_ACCEPTED_BACKEND_IS_AN_ORDINARY_DETERMINISTIC_SOFTWARE_PROGRAM",
                "THE_COMPARATOR_MAY_EXECUTE_THE_SAME_FORWARD_ALGEBRA_ON_THE_SAME_CANONICAL_INPUT_STATE",
                "ANY_AUXILIARY_STATE_SECRET_INPUT_OR_ORACLE_ACCESS_IS_GRANTED_EQUALLY_TO_ACCEPTED_AND_SHADOW_PATHS",
                "THE_RELEASED_BOUNDARY_IS_COMPUTED_BY_A_DETERMINISTIC_PROJECTION_OF_THE_FORWARD_STATE",
                "THE_ACCEPTED_FORWARD_STATE_CREATION_AND_PUBLIC_DESCRIPTORS_ARE_AVAILABLE_ON_THE_SAME_ACCOUNTING_BASIS",
                "EXACT_BOUNDARY_EQUALITY_IS_THE_COMPARISON_SEMANTIC",
            ],
            "construction": "SHADOW_T_EQUALS_PROJECT_AFTER_FORWARD_USING_THE_ACCEPTED_PREFIX_AND_RETURNS_BEFORE_INVERSE",
            "boundary_law": "SHADOW_T_X_EQUALS_PROJECT_FORWARD_X_EQUALS_ACCEPTED_BOUNDARY_T_X",
            "forward_state_law": "SHADOW_MAY_REUSE_THE_EXACT_ACCEPTED_FORWARD_ALLOCATION_SO_ITS_FORWARD_STATE_IS_NO_GREATER_COMPONENTWISE",
            "forward_work_law": "SHADOW_MAY_REUSE_THE_EXACT_ACCEPTED_FORWARD_AND_PROJECTION_OPERATIONS_SO_ITS_FORWARD_WORK_IS_NO_GREATER_COMPONENTWISE",
            "strict_transaction_law": "EACH_DECLARED_ACCEPTED_PATH_HAS_AT_LEAST_ONE_POSITIVE_INVERSE_STAGE_COUNTER_THAT_THE_SHADOW_OMITS",
            "does_not_require_summing_heterogeneous_operation_counters": True,
        },
        "packages": packages,
        "aggregate": {
            "algebraically_distinct_packages": len(packages),
            "all_source_hashes_match_sealed_dependencies": True,
            "all_forward_projection_inverse_release_response_orders_verified": True,
            "all_exact_boundaries_have_constructive_forward_shadows": True,
            "all_shadow_forward_state_shapes_no_greater_componentwise": True,
            "all_shadow_forward_work_no_greater_componentwise": True,
            "all_shadow_paths_omit_nonzero_inverse_stage_witnesses": True,
            "same_domain_software_advantage_established": False,
            "distinct_phase_resource_established": False,
            "small_wall_crossed": False,
        },
        "escape_criteria": [
            "AN_EXTERNAL_ORACLE_OR_INTERACTION_NOT_AVAILABLE_AS_THE_SAME_SOFTWARE_FORWARD_UPDATE",
            "A_PHYSICAL_WAVEFORM_OR_ANALOG_RESOURCE_WITH_AN_INDEPENDENTLY_MEASURED_COST_MODEL",
            "A_SECRET_OR_EXOGENOUS_CARRIER_STATE_THAT_THE_COMPARATOR_IS_NOT_LAWFULLY_GIVEN",
            "STOCHASTIC_APPROXIMATE_OR_IRREVERSIBLE_SEMANTICS_THAT_CHANGE_THE_EXACT_FUNCTIONAL_COMPARISON",
            "AN_INTERACTIVE_QUERY_MODEL_WITH_A_JUSTIFIED_BLACK_BOX_BOUNDARY_EXCLUDING_BACKEND_IMPLEMENTATION_COST",
            "A_SCIENTIFICALLY_JUSTIFIED_COMPARATOR_RESTRICTION_NOT_MANUFACTURED_BY_HIDING_THE_ACTUAL_SOFTWARE_BACKEND",
        ],
        "claim_limits": {
            "general_all_software_no_go_theorem": False,
            "physical_phase_computation_ruled_out": False,
            "oracle_advantage_ruled_out": False,
            "stochastic_or_approximate_advantage_ruled_out": False,
            "external_secret_resource_ruled_out": False,
            "computational_complexity_lower_bound": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "resource_law": {
            "comparison_is_componentwise_not_a_sum_of_heterogeneous_counters": True,
            "shadow_can_use_identical_forward_state_and_operations": True,
            "shadow_omits_inverse_restoration_and_atomic_release_requirement": True,
            "existing_stronger_package_specific_baselines_remain_authoritative": True,
            "whole_process_python_object_allocator_socket_hash_serialization_timing_and_rss_costs_complete": False,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "source_dependencies": source_dependencies,
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
