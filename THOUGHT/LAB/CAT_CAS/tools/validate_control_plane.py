#!/usr/bin/env python3
"""Fail-closed validation for the CAT_CAS phase-lock control plane."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


MISSION_ID = "CAT_CAS_UNBOUNDED_COMPUTE_V1"
MODES = {"exploration", "engineering", "verification", "compression"}
TASK_CLASSES = {
    "flagship_compute",
    "enabling_infrastructure",
    "external_product",
    "calibration",
    "evidence_audit",
}
OPERATOR_MODES = {"native_holo", "hybrid", "materialized_fallback"}
PLACEHOLDER = re.compile(r"\b(REPLACE|UNKNOWN|REPLACE_OR_NOT_APPLICABLE)\b")
WINDOWS_ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]")


class ValidationError(Exception):
    """Raised for malformed validation inputs."""


def lab_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Invalid JSON {path}: {exc}") from exc


def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Cannot read {path}: {exc}") from exc


def missing_or_placeholder(value: Any) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return True
    return bool(PLACEHOLDER.search(str(value)))


def require_files(root: Path, errors: list[str]) -> None:
    required = (
        "MISSION.md",
        "AGENTS.md",
        "PRIMER.md",
        "MASTER_REPORT.md",
        "control/mission.json",
        "control/current_state.json",
        "control/canon_manifest.json",
        "control/capability_graph.json",
        "control/branch_registry.json",
        "control/branches/audio_frequency_wave_substrate.json",
        "evals/phase_lock_cases.jsonl",
        "evals/README.md",
        "templates/BRANCH_STATE.json",
        "templates/EXPERIMENT.json",
        "templates/TASK_CONTRACT.json",
        "templates/PHASE_LOCK_RECEIPT.json",
        "tools/phase_lock.py",
        "tools/validate_control_plane.py",
    )
    for relative in required:
        if not (root / relative).exists():
            errors.append(f"missing required file: {relative}")


def validate_mission(root: Path, errors: list[str]) -> None:
    mission = load(root / "control" / "mission.json")
    if mission.get("mission_id") != MISSION_ID:
        errors.append("mission_id mismatch")
    if mission.get("authority_rank") != 0:
        errors.append("mission authority_rank must be 0")
    for field in ("one_sentence", "native_chain", "compute_metric", "fatal_substitutions"):
        if not mission.get(field):
            errors.append(f"mission missing {field}")
    if set(mission.get("task_classes", [])) != TASK_CLASSES:
        errors.append("mission task_classes differ from canonical set")
    if set(mission.get("operator_modes", [])) != OPERATOR_MODES:
        errors.append("mission operator_modes differ from canonical set")

    mission_md = text(root / "MISSION.md")
    mission_search = " ".join(mission_md.split())
    required_phrases = (
        "finite, reusable catalytic substrate",
        "unbounded effective computation",
        "Where is the compute leverage?",
        "REPLACE THE BIT WITH PI" if "REPLACE THE BIT WITH PI" in mission_md else "Pi, phase, and the torus",
    )
    for phrase in required_phrases:
        if phrase not in mission_search:
            errors.append(f"MISSION.md missing required phrase: {phrase}")


def validate_current_state(root: Path, errors: list[str]) -> None:
    state = load(root / "control" / "current_state.json")
    if state.get("mission_id") != MISSION_ID:
        errors.append("current_state mission mismatch")
    if not state.get("active_frontiers"):
        errors.append("current_state has no active_frontiers")
    identifiers = {item.get("id") for item in state.get("active_frontiers", [])}
    for required in ("EXP50_SMALL_WALL", "AUDIO_PHASE_TORUS", "TRACK8_EXTERNAL_FRONTIERS"):
        if required not in identifiers:
            errors.append(f"current_state missing frontier {required}")

    for frontier in state.get("active_frontiers", []):
        source = frontier.get("state_source")
        if not source:
            errors.append(f"{frontier.get('id')}: missing state_source")
            continue
        if frontier.get("branch"):
            continue
        if not (root / source).exists():
            errors.append(f"{frontier.get('id')}: missing state source {source}")


def validate_canon(root: Path, errors: list[str]) -> None:
    manifest = load(root / "control" / "canon_manifest.json")
    records = manifest.get("authority_order", [])
    ranks = [record.get("rank") for record in records]
    identifiers = [record.get("id") for record in records]
    if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
        errors.append("canon authority ranks must be unique and sorted")
    if len(identifiers) != len(set(identifiers)):
        errors.append("canon identifiers are duplicated")
    if not records or records[0].get("id") != "MISSION":
        errors.append("MISSION must be first in canon authority order")
    for record in records:
        relative = str(record.get("path", ""))
        if not relative:
            errors.append(f"canon record {record.get('id')} has no path")
        elif "*" not in relative and not (root / relative).exists():
            errors.append(f"canon path missing: {relative}")


def validate_graph(root: Path, errors: list[str]) -> None:
    graph = load(root / "control" / "capability_graph.json")
    if graph.get("mission_id") != MISSION_ID:
        errors.append("capability graph mission mismatch")
    rung_ids = {rung.get("id") for rung in graph.get("compute_rungs", [])}
    expected_rungs = {f"C{index}_{name}" for index, name in enumerate((
        "CATALYTIC_CLOSURE",
        "BOUNDED_RESIDENCY",
        "REUSE_AND_MULTIPLEX",
        "NONCOLLAPSE_REPRESENTATION",
        "NATIVE_GLOBAL_OPERATOR",
        "FIXED_POINT_ADVANTAGE",
        "EXTERNAL_ACCEPTANCE",
        "CROSS_DOMAIN_TRANSFER",
    ))}
    if rung_ids != expected_rungs:
        errors.append("capability graph compute-rung set is incomplete or changed")

    nodes = graph.get("nodes", [])
    node_ids = [node.get("id") for node in nodes]
    if None in node_ids or len(node_ids) != len(set(node_ids)):
        errors.append("capability node IDs are missing or duplicated")
    known = set(node_ids)
    for node in nodes:
        node_id = node.get("id")
        if node.get("task_class") not in TASK_CLASSES:
            errors.append(f"{node_id}: invalid task_class")
        for rung in node.get("mission_rungs", []):
            if rung not in rung_ids:
                errors.append(f"{node_id}: unknown mission rung {rung}")
        relative = node.get("path")
        if relative and not node.get("branch") and not (root / relative).exists():
            errors.append(f"{node_id}: missing local path {relative}")
        for code_path in node.get("code_entrypoints", []):
            if not (root / code_path).exists():
                errors.append(f"{node_id}: missing code entrypoint {code_path}")

    for edge in graph.get("edges", []):
        if len(edge) != 3:
            errors.append(f"invalid capability edge shape: {edge}")
        elif edge[0] not in known or edge[1] not in known:
            errors.append(f"capability edge references unknown node: {edge}")


def validate_branch_registry(root: Path, errors: list[str]) -> None:
    registry = load(root / "control" / "branch_registry.json")
    patterns = [record.get("pattern") for record in registry.get("branches", [])]
    if len(patterns) != len(set(patterns)):
        errors.append("branch registry contains duplicate patterns")
    if "main" not in patterns:
        errors.append("branch registry does not contain main")
    if "codex/audio-frequency-wave-substrate" not in patterns:
        errors.append("audio branch is not registered")
    for record in registry.get("branches", []):
        source = record.get("context_source")
        if source and not (root / source).exists():
            errors.append(f"branch context source missing: {source}")

    audio = load(root / "control" / "branches" / "audio_frequency_wave_substrate.json")
    if audio.get("branch") != "codex/audio-frequency-wave-substrate":
        errors.append("audio branch context names the wrong branch")
    if audio.get("prime_directive") != "REPLACE THE BIT WITH PI":
        errors.append("audio branch prime directive changed")
    if audio.get("physical_audio_computing_established") is not False:
        errors.append("audio branch context inflates physical audio status")


def validate_evals(root: Path, errors: list[str]) -> None:
    path = root / "evals" / "phase_lock_cases.jsonl"
    cases: list[dict[str, Any]] = []
    for number, line in enumerate(text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"phase-lock case line {number} is invalid JSON: {exc}")
    identifiers = [case.get("case_id") for case in cases]
    if len(identifiers) != len(set(identifiers)):
        errors.append("phase-lock case IDs are duplicated")
    required = {
        "flagship_bounty",
        "classical_product",
        "restoration_only",
        "candidate_container",
        "lawful_classical_boundary",
        "audio_branch",
        "preseeded_fixed_point",
    }
    missing = required.difference(identifiers)
    if missing:
        errors.append(f"phase-lock cases missing: {sorted(missing)}")
    if len(cases) < 7:
        errors.append("phase-lock evaluation set is too small")


def validate_boot_docs(root: Path, errors: list[str]) -> None:
    agents = text(root / "AGENTS.md")
    primer = text(root / "PRIMER.md")
    report = text(root / "MASTER_REPORT.md")
    readme = text(root / "README.md")

    for name, content in (("AGENTS.md", agents), ("PRIMER.md", primer)):
        if "MISSION.md" not in content:
            errors.append(f"{name} does not point to MISSION.md")
        if "tools/phase_lock.py" not in content:
            errors.append(f"{name} does not point to tools/phase_lock.py")
        if WINDOWS_ABSOLUTE.search(content):
            errors.append(f"{name} contains a machine-local absolute path")

    if "does not define the CAT_CAS mission" not in report:
        errors.append("MASTER_REPORT.md lacks the mission-authority banner")
    if "MISSION.md" not in readme or "CAPABILITY_GRAPH.md" not in readme:
        errors.append("README.md does not route through mission and capability views")


def validate_receipt(path: Path, errors: list[str]) -> None:
    receipt = load(path)
    if receipt.get("mission_id") != MISSION_ID:
        errors.append(f"{path}: mission mismatch")
    if receipt.get("mode") not in MODES:
        errors.append(f"{path}: invalid mode")
    task_class = receipt.get("task_class")
    if task_class not in TASK_CLASSES:
        errors.append(f"{path}: invalid task_class")
        return

    common = (
        "mission_summary",
        "infinite_compute_definition",
        "holo_role",
        "pi_torus_role",
        "native_middle",
        "final_classical_boundary",
        "restoration_role",
        "current_frontier",
        "current_claim_ceiling",
    )
    flagship = (
        "classical_explosion",
        "native_operator",
        "invariant_or_fixed_point",
        "no_smuggle_killer_control",
    )
    for field in common + (flagship if task_class == "flagship_compute" else ()):
        if missing_or_placeholder(receipt.get(field)):
            errors.append(f"{path}: incomplete {field}")
    if receipt.get("phase_locked") is not True:
        errors.append(f"{path}: phase_locked must be true")
    if not receipt.get("context_digest"):
        errors.append(f"{path}: context_digest missing")


def validate_task_contract(path: Path, errors: list[str]) -> None:
    contract = load(path)
    if contract.get("mission_id") != MISSION_ID:
        errors.append(f"{path}: mission mismatch")
    if contract.get("mode") not in MODES:
        errors.append(f"{path}: invalid mode")
    task_class = contract.get("task_class")
    if task_class not in TASK_CLASSES:
        errors.append(f"{path}: invalid task_class")
        return
    common = ("requested_operation", "collapse_boundary", "restoration_law", "current_claim_ceiling")
    flagship = (
        "classical_explosion",
        "holo_process_object",
        "native_operator",
        "invariant_or_fixed_point",
        "no_smuggle_killer_control",
        "official_boundary",
    )
    for field in common + (flagship if task_class == "flagship_compute" else ()):
        if missing_or_placeholder(contract.get(field)):
            errors.append(f"{path}: incomplete {field}")


def validate_experiment(path: Path, errors: list[str]) -> None:
    experiment = load(path)
    if experiment.get("mission_id") != MISSION_ID:
        errors.append(f"{path}: mission mismatch")
    task_class = experiment.get("task_class")
    if task_class not in TASK_CLASSES:
        errors.append(f"{path}: invalid task_class")
    process = experiment.get("holo_process_object", {})
    if process.get("operator_mode") not in OPERATOR_MODES:
        errors.append(f"{path}: invalid operator mode")
    if task_class == "flagship_compute":
        for field in (
            "classical_explosion",
            "native_operator",
            "invariant_or_fixed_point",
            "collapse_boundary",
            "external_verifier",
            "restoration_law",
            "no_smuggle_killer_control",
            "scaling_hypothesis",
        ):
            if missing_or_placeholder(experiment.get(field)):
                errors.append(f"{path}: flagship experiment missing {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=lab_root())
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--task-contract", type=Path)
    parser.add_argument("--experiment", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    errors: list[str] = []
    try:
        require_files(root, errors)
        if not errors:
            validate_mission(root, errors)
            validate_current_state(root, errors)
            validate_canon(root, errors)
            validate_graph(root, errors)
            validate_branch_registry(root, errors)
            validate_evals(root, errors)
            validate_boot_docs(root, errors)
        if args.receipt:
            validate_receipt(args.receipt.resolve(), errors)
        if args.task_contract:
            validate_task_contract(args.task_contract.resolve(), errors)
        if args.experiment:
            validate_experiment(args.experiment.resolve(), errors)
    except ValidationError as exc:
        errors.append(str(exc))

    if errors:
        print("CAT_CAS_CONTROL_PLANE_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("CAT_CAS_CONTROL_PLANE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
