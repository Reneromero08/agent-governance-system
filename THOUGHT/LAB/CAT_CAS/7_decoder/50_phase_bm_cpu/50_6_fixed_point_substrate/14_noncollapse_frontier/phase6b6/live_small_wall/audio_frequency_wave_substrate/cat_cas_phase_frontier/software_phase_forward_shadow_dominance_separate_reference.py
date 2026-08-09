#!/usr/bin/env python3
"""Independent text-order and evidence oracle for the M257 shadow theorem."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RefSpec:
    name: str
    result_file: str
    service_file: str
    service_hash_key: str
    run_kind: str
    boundary_fields: tuple[str, ...]
    same_backing_field: str
    forward_anchor: str
    projection_anchor: str
    inverse_anchor: str
    release_anchor: str
    state_fields: tuple[tuple[str, str, str], ...]
    inverse_keys: tuple[str, ...]


SPECS = (
    RefSpec(
        "M248_CUBIC_MAGIC_CATALYST",
        "CATVM_P5_CUBIC_MAGIC_CATALYST_RESULTS.json",
        "catvm_p5_cubic_magic_catalyst_service.py",
        "service_sha256",
        "TWO_SYNDROME_PRIMARY",
        ("final_amplitude",),
        "same_all_backings",
        "carrier.forward_use(work)",
        "amplitude, exponent = carrier.project(work)",
        "carrier.restore_prefix(work)",
        "carrier.release()",
        (
            ("catalyst_field_cells", "case", "catalyst_field_cells"),
            ("joint_scratch_field_cells", "case", "joint_interaction_scratch_field_cells"),
            ("phase_signature_field_cells", "case", "phase_signature_field_cells"),
            ("projection_workspace_field_cells", "case", "final_projection_workspace_field_cells"),
        ),
        ("inverse_catalyst_rematerializations",),
    ),
    RefSpec(
        "M250_PROJECTIVE_WEYL_MERMIN",
        "CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_RESULTS.json",
        "catvm_projective_weyl_mermin_square_service.py",
        "catvm_projective_weyl_mermin_square_service.py",
        "PRIMARY",
        ("central_phase_exponent_mod4", "central_phase"),
        "same_carrier_and_custody_backings",
        "carrier.forward_action(program, index, work)",
        "boundary = carrier.project_boundary(program, work)",
        "carrier.inverse_action(program, carrier.cursor - 1, work)",
        "carrier.release()",
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
    ),
    RefSpec(
        "M253_GRASSMANN_GAUSSIAN",
        "CATVM_GRASSMANN_GAUSSIAN_OPEN_RELATION_RESULTS.json",
        "catvm_grassmann_gaussian_open_relation_service.py",
        "catvm_grassmann_gaussian_open_relation_service.py",
        "PRIMARY",
        ("top_form_boundary",),
        "same_relation_scratch_and_receipt_backings",
        "carrier.forward_module(index, work)",
        "boundary = carrier.project_boundary(work)",
        "carrier.inverse_module(carrier.cursor - 1, work)",
        "carrier.release()",
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
    ),
    RefSpec(
        "M254_GRASSMANN_FULL_EVEN",
        "CATVM_GRASSMANN_EVEN_EXTERIOR_RELATION_RESULTS.json",
        "catvm_grassmann_even_exterior_relation_service.py",
        "catvm_grassmann_even_exterior_relation_service.py",
        "PRIMARY",
        ("top_form_boundary",),
        "same_relation_scratch_and_receipt_backings",
        "carrier.forward_module(index, work)",
        "boundary = carrier.project_boundary()",
        "carrier.inverse_module(carrier.cursor - 1, work)",
        "carrier.release()",
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
    ),
    RefSpec(
        "M256_SCHUR_ALLPASS_WAVEFORM",
        "CATVM_SCHUR_ALLPASS_WAVEFORM_RESULTS.json",
        "catvm_schur_allpass_waveform_service.py",
        "catvm_schur_allpass_waveform_service.py",
        "PRIMARY",
        ("winding", "evaluation"),
        "same_waveform_scratch_and_receipt_backings",
        "carrier.forward(program, generation, transaction_id, work)",
        "response = project_boundary(",
        "carrier.inverse(program, generation, transaction_id, work)",
        "carrier.release(program, generation, transaction_id)",
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
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_transaction_block(source: str) -> tuple[str, int]:
    marker = "def run_transaction("
    start = source.index(marker)
    candidates = [
        offset for offset in (
            source.find("\ndef ", start + len(marker)),
            source.find("\nclass ", start + len(marker)),
        ) if offset >= 0
    ]
    end = min(candidates) if candidates else len(source)
    return source[start:end], source.count("\n", 0, start) + 1


def absolute_line(block: str, block_line: int, offset: int) -> int:
    return block_line + block.count("\n", 0, offset)


def order_by_distinct_text_anchors(path: Path, spec: RefSpec) -> dict[str, int]:
    block, block_line = run_transaction_block(path.read_text())
    forward_offset = block.find(spec.forward_anchor)
    projection_offset = block.find(spec.projection_anchor, forward_offset + 1)
    inverse_offset = block.rfind(spec.inverse_anchor, projection_offset + 1)
    release_offset = block.rfind(spec.release_anchor, inverse_offset + 1)
    response_offset = block.rfind("return {", release_offset + 1)
    offsets = (forward_offset, projection_offset, inverse_offset, release_offset, response_offset)
    if any(offset < 0 for offset in offsets) or list(offsets) != sorted(offsets):
        raise RuntimeError(f"{spec.name}: independent source-order anchors failed")
    return dict(zip(
        ("forward_line", "projection_line", "inverse_line", "release_line", "response_line"),
        (absolute_line(block, block_line, offset) for offset in offsets),
    ))


def selected_case(result: dict[str, Any], run_kind: str) -> dict[str, Any]:
    cases = [case for case in result["cases"] if case.get("run_kind") == run_kind]
    if len(cases) != 1:
        raise RuntimeError(f"independent oracle expected one {run_kind}")
    return cases[0]


def shape_from_case(case: dict[str, Any], spec: RefSpec) -> dict[str, int]:
    sources = {"case": case, "work": case["work"], "resource_shape": case.get("resource_shape", {})}
    output: dict[str, int] = {}
    for label, source, key in spec.state_fields:
        value = sources[source][key]
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{spec.name}: bad state shape")
        output[label] = value
    return output


def analyze(root: Path, spec: RefSpec) -> dict[str, Any]:
    result_path = root / spec.result_file
    service_path = root / spec.service_file
    result = json.loads(result_path.read_text())
    service_hash = sha256(service_path)
    if service_hash != result["source_dependencies"][spec.service_hash_key]:
        raise RuntimeError(f"{spec.name}: sealed service hash mismatch")
    restoration = result.get("restoration_class") or result.get("restoration_classification")
    if (
        result.get("classification") != "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        or result.get("verification_level") != "SEPARATE_REFERENCE_PARITY"
        or restoration != "EXACT_ALGEBRAIC_RESTORATION"
    ):
        raise RuntimeError(f"{spec.name}: evidence authority mismatch")
    case = selected_case(result, spec.run_kind)
    if not case["canonical_after_restoration"] or not case[spec.same_backing_field] or case["baseline_reload_used"]:
        raise RuntimeError(f"{spec.name}: restoration witness mismatch")
    boundary = {field: case[field] for field in spec.boundary_fields}
    digest = hashlib.sha256(json.dumps(boundary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    inverse = {key: case["work"][key] for key in spec.inverse_keys}
    if not all(isinstance(value, int) and value > 0 for value in inverse.values()):
        raise RuntimeError(f"{spec.name}: inverse witness is not strict")
    shape = shape_from_case(case, spec)
    return {
        "package": spec.name,
        "sealed_result_sha256": sha256(result_path),
        "sealed_service_sha256": service_hash,
        "primary_run_kind": spec.run_kind,
        "boundary_sha256": digest,
        "source_order": order_by_distinct_text_anchors(service_path, spec),
        "accepted_forward_state_shape_upper_bound": shape,
        "shadow_forward_state_shape_upper_bound": shape,
        "reported_transaction_only_inverse_work_vector_omitted_by_shadow": inverse,
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: software_phase_forward_shadow_dominance_separate_reference.py FRONTIER_DIR OUTPUT_JSON")
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2])
    packages = [analyze(root, spec) for spec in SPECS]
    document = {
        "milestone": 257,
        "result": "PASS_INDEPENDENT_SOFTWARE_FORWARD_SHADOW_SOURCE_ORDER_AND_RESOURCE_ORACLE",
        "method": "INDEPENDENT_TEXT_ANCHOR_SOURCE_ORDER_PLUS_EXPLICIT_RESOURCE_VECTOR_RECONSTRUCTION_WITHOUT_IMPORTING_M257_PRODUCTION",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "theorem_certificate": {
            "final_boundary_task_only": True,
            "same_access_model_required": True,
            "full_atomic_transaction_is_a_different_task": True,
            "identical_prefix_gives_exact_boundary_equality": True,
            "identical_prefix_gives_componentwise_no_greater_forward_state_and_work": True,
            "positive_inverse_stage_makes_the_declared_transaction_work_strictly_larger_in_at_least_one_reported_class": True,
            "heterogeneous_operation_counters_are_not_summed": True,
        },
        "packages": packages,
        "applicability_controls": {
            "private_input_denied_to_shadow": "INAPPLICABLE_ACCESS_MISMATCH",
            "restoration_and_reuse_required_as_output": "INAPPLICABLE_FULL_TRANSACTION_TASK",
            "physical_or_opaque_primitive": "INAPPLICABLE_OUTSIDE_EXACT_SOFTWARE_DOMAIN",
            "dense_only_comparator_substitution": "REJECTED_STRONGER_COMPACT_BASELINE_REQUIRED",
            "cell_only_claim_with_unmeasured_coefficient_height": "REJECTED_BIT_COST_NOT_ESTABLISHED",
            "fixed_fixture_o1_certificate_promoted_to_transferable_algorithm": "REJECTED_GENERALIZATION",
        },
        "claim_limits": {
            "physical_phase_computation_ruled_out": False,
            "opaque_oracle_advantage_ruled_out": False,
            "all_catalytic_tasks_dominated": False,
            "all_future_phase_representations_dominated": False,
            "computational_advantage_established": False,
            "small_wall_crossing": False,
        },
        "source_dependencies": {
            Path(__file__).name: sha256(Path(__file__)),
            **{spec.result_file: sha256(root / spec.result_file) for spec in SPECS},
            **{spec.service_file: sha256(root / spec.service_file) for spec in SPECS},
        },
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
