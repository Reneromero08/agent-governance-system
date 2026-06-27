#!/usr/bin/env python3
"""Generate and verify a deterministic Gate R technical-review packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

SCHEMA_VERSION = "1.0.0"
REVIEW_SCHEMA = "l4b5b0_gate_r_technical_review_v1"
BUNDLE_SCHEMA = "l4b5b0_gate_r_effective_bundle_v1"
MANIFEST_SCHEMA = "l4b5b0_gate_r_manifest_v1"
OUTPUTS = (
    "l4b5b0_observability_design.json",
    "gate_r_effective_bundle.json",
    "gate_r_technical_review.json",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_text(path: Path, markers: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise ValueError(f"{path} missing markers: {missing}")


def find_by_id(items: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for item in items:
        if item.get(key) == value:
            return item
    raise ValueError(f"missing {key}={value}")


def inspect_design(design: dict[str, Any]) -> dict[str, Any]:
    if design.get("design_id") != "l4b5b0_observability_operator_v1":
        raise ValueError("unexpected design_id")
    if design.get("status") != "READY_FOR_HUMAN_REVIEW":
        raise ValueError("design is not frozen for review")
    forbidden_true = (
        "implementation_authorized",
        "executed",
        "full_physical_observability_claimed",
        "physical_restoration_claimed",
        "human_reviewed",
    )
    for field in forbidden_true:
        if design.get(field) is not False:
            raise ValueError(f"design field must remain false: {field}")
    if design.get("claim_level") != 1:
        raise ValueError("claim level exceeds Gate R ceiling")

    states = design.get("state_models", [])
    inputs = design.get("input_families", [])
    operators = design.get("operator_candidates", [])
    gates = design.get("acceptance_gates", [])
    falsifications = design.get("falsification_conditions", [])
    artifacts = design.get("artifact_contracts", [])
    if len(states) != 3 or len(inputs) != 11 or len(operators) != 4:
        raise ValueError("unexpected base-design cardinality")
    if len(gates) != 10 or len(falsifications) != 10 or len(artifacts) != 6:
        raise ValueError("incomplete base-design gate graph")

    s1 = find_by_id(states, "model_id", "S1_contextual")
    s1_fields = str(s1.get("fields", ""))
    observed_defects = {
        "state_contains_control": "sender_mode" in s1_fields,
        "state_contains_route_context": "route" in s1_fields,
        "state_contains_time_index": "capture_window" in s1_fields,
        "latent_x_notation": "x(t+1)" in str(design.get("design_scope", {}).get("state_space_model", "")),
        "tone_order_contract_absent_from_base": not any("RND1" in json.dumps(item) for item in inputs),
        "mandatory_drive_off_gate_absent": not any("drive_off" in json.dumps(item).lower() for item in gates),
    }
    return {
        "design_id": design["design_id"],
        "design_version": design.get("design_version"),
        "design_digest": design.get("design_digest"),
        "base_counts": {
            "state_models": len(states),
            "input_families": len(inputs),
            "operator_candidates": len(operators),
            "acceptance_gates": len(gates),
            "falsification_conditions": len(falsifications),
            "artifact_contracts": len(artifacts),
        },
        "base_defects_observed": observed_defects,
        "base_reference_graph_complete": True,
        "base_implementation_authorized": False,
    }


def generate(
    design_path: Path,
    addendum_path: Path,
    review_md_path: Path,
    findings_path: Path,
    phase6b5c_manifest_path: Path,
    phase6b5d_manifest_path: Path,
    tone_order_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    design = load_json(design_path)
    design_summary = inspect_design(design)
    phase6b5c = load_json(phase6b5c_manifest_path)
    phase6b5d = load_json(phase6b5d_manifest_path)

    require_text(addendum_path, (
        "BINDING_REPAIR_ADDENDUM_PENDING_PROJECT_OWNER_RATIFICATION",
        "S1_contextual = gauge_normalize",
        "PERSISTENT_STATE_CANDIDATE",
        "DRIVEN_RELATIONAL_TRANSPORT_ONLY",
        "GR5 governance separation",
    ))
    require_text(review_md_path, (
        "TECHNICAL_ACCEPT_WITH_REQUIRED_REPAIRS_APPLIED",
        "Project-owner ratification: complete",
        "APPROVED_FOR_INTEGRATION",
        "Physical acquisition authorization: FALSE",
    ))
    require_text(findings_path, (
        "F1 — State partition",
        "F3 — Driven transport versus persistence",
        "F4 — Tone/path confounding",
        "F5 — Session gauge",
    ))
    require_text(tone_order_path, (
        "PREREGISTERED_NOT_AUTHORIZED",
        "FWD",
        "REV",
        "RND1",
        "RND2",
        "order-label sham",
    ))

    inherited_time = phase6b5d.get("generated_utc_inherited_from_phase6b5c")
    if not inherited_time:
        inherited_time = phase6b5c.get("generated_utc")
    if not isinstance(inherited_time, str) or not inherited_time:
        raise ValueError("no deterministic provenance time")

    output_dir.mkdir(parents=True, exist_ok=False)
    copied_design = output_dir / "l4b5b0_observability_design.json"
    shutil.copyfile(design_path, copied_design)

    source_bindings = {
        "sealed_design": {"sha256": sha256_file(design_path), "path": str(design_path)},
        "repair_addendum": {"sha256": sha256_file(addendum_path), "path": str(addendum_path)},
        "technical_review_markdown": {"sha256": sha256_file(review_md_path), "path": str(review_md_path)},
        "findings_appendix": {"sha256": sha256_file(findings_path), "path": str(findings_path)},
        "phase6b5c_manifest": {"sha256": sha256_file(phase6b5c_manifest_path), "path": str(phase6b5c_manifest_path)},
        "phase6b5d_manifest": {"sha256": sha256_file(phase6b5d_manifest_path), "path": str(phase6b5d_manifest_path)},
        "tone_order_contract": {"sha256": sha256_file(tone_order_path), "path": str(tone_order_path)},
    }

    bundle = {
        "schema_id": BUNDLE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": inherited_time,
        "design": design_summary,
        "source_bindings": source_bindings,
        "carrier_evidence": {
            "phase6b5c_campaign_id": phase6b5c.get("campaign_id"),
            "phase6b5c_decision": phase6b5c.get("decision"),
            "phase6b5d_decision": phase6b5d.get("decision"),
        },
        "effective_repairs": [
            "separate measured response, executed control, nuisance context, and session gauge",
            "predict measured-state equivalence rather than latent substrate state",
            "mandatory sender-off driven-versus-persistent classification",
            "tone identity/path position disentanglement before path-memory claims",
            "preamble-only session gauge with seed4 retained",
            "diagnostic classification subordinate to held-out trajectory prediction",
        ],
        "effective_gate_additions": ["GR1", "GR2", "GR3", "GR4", "GR5"],
        "implementation_authorized": False,
        "physical_acquisition_authorized": False,
        "restoration_authorized": False,
    }
    write_json(output_dir / "gate_r_effective_bundle.json", bundle)

    review = {
        "schema_id": REVIEW_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": inherited_time,
        "reviewer_role": "independent_ai_technical_reviewer",
        "human_review": False,
        "project_owner_ratification_required": True,
        "project_owner_ratified": False,
        "verdict": "TECHNICAL_ACCEPT_WITH_REQUIRED_REPAIRS_APPLIED",
        "reviewed_design_id": design_summary["design_id"],
        "reviewed_design_version": design_summary["design_version"],
        "reviewed_design_digest": design_summary["design_digest"],
        "effective_bundle_sha256": sha256_file(output_dir / "gate_r_effective_bundle.json"),
        "claim_ceiling": "predictive_observability_of_measured_response_equivalence_class",
        "required_next_record": "PROJECT_OWNER_RATIFICATION",
        "owner_options": [
            "RATIFY_TECHNICAL_REVIEW_NO_ACQUISITION",
            "RATIFY_AND_AUTHORIZE_TONE_ORDER_CONTROL_ONLY",
            "RATIFY_AND_AUTHORIZE_COMBINED_TONE_ORDER_OBSERVABILITY_CAMPAIGN",
            "REJECT_AND_REVISE",
        ],
        "implementation_authorized": False,
        "physical_acquisition_authorized": False,
        "physical_restoration_authorized": False,
        "forbidden_claims": [
            "complete_physical_observability",
            "physical_HoloGeometry",
            "inverse_physical_dynamics",
            "physical_restoration",
            "target_coupling",
            "orientation_recovery",
            "Small_Wall_crossing",
        ],
    }
    write_json(output_dir / "gate_r_technical_review.json", review)

    outputs = {
        name: {
            "size": (output_dir / name).stat().st_size,
            "sha256": sha256_file(output_dir / name),
        }
        for name in OUTPUTS
    }
    manifest = {
        "schema_id": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": inherited_time,
        "outputs": outputs,
        "decision": {
            "technical_review_complete": True,
            "verdict": review["verdict"],
            "repairs_binding": True,
            "project_owner_ratification_required": True,
            "implementation_authorized": False,
            "physical_acquisition_authorized": False,
        },
    }
    write_json(output_dir / "gate_r_manifest.json", manifest)
    return manifest


def verify(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "gate_r_manifest.json"
    manifest = load_json(manifest_path)
    errors: list[str] = []
    for name, expected in manifest.get("outputs", {}).items():
        path = output_dir / name
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        if path.stat().st_size != int(expected["size"]):
            errors.append(f"size mismatch {name}")
        if sha256_file(path) != expected["sha256"]:
            errors.append(f"sha256 mismatch {name}")
    review = load_json(output_dir / "gate_r_technical_review.json")
    if review.get("human_review") is not False:
        errors.append("technical review incorrectly claims human review")
    for field in ("implementation_authorized", "physical_acquisition_authorized", "physical_restoration_authorized"):
        if review.get(field) is not False:
            errors.append(f"review incorrectly authorizes {field}")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_sha256": sha256_file(manifest_path),
        "decision": manifest.get("decision"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--design", type=Path, required=True)
    run.add_argument("--addendum", type=Path, required=True)
    run.add_argument("--review-md", type=Path, required=True)
    run.add_argument("--findings", type=Path, required=True)
    run.add_argument("--phase6b5c", type=Path, required=True)
    run.add_argument("--phase6b5d", type=Path, required=True)
    run.add_argument("--tone-order", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        manifest = generate(
            args.design.resolve(), args.addendum.resolve(), args.review_md.resolve(),
            args.findings.resolve(), args.phase6b5c.resolve(), args.phase6b5d.resolve(),
            args.tone_order.resolve(), args.output.resolve(),
        )
        print(json.dumps(manifest["decision"], indent=2, sort_keys=True))
        return 0
    result = verify(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
