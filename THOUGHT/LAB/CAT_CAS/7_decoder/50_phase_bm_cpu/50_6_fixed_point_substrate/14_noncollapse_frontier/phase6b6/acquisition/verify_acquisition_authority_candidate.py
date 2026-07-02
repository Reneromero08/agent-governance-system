#!/usr/bin/env python3
"""Fail-closed verification for the Phase 6B.6 acquisition authority design."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BASE_MAIN = "1db6d5b0b1e0d9b38a0c1709b09c9c11a59217a2"
EVIDENCE_HEAD = "b2b785d064d4704ef2955238593f9f5050425f55"
EVIDENCE_REVIEW = 4614291991
CORRECTED_PAYLOAD_COMMIT = "38eccc2f8377c656b0c21cbd37dd296e81adfad4"
CORRECTED_PAYLOAD_TREE = "159c04abd72ed631e7647249b539b548d8351e4b"
INVENTORY_SHA256 = "7b205dd92482425505e498027a3e96db842297fad949fe22bdbf5542e95573e0"
ARCHIVE_SHA256 = "affbc0b3e9725de62aa946774e3e8830399f9af12414713b1bfbc68547765ca4"
MANIFEST_SHA256 = "59e5c5927cfa7f19bdaafdd740cb350f5819e81741b62821a22f2eb80ecd4676"
TARGET_RESULT_DIGEST = "c2d1bf3c78e2a9318f51e06d27ac39a49fe7a49e3cd49c0c8850cd6c85a07f7f"
SCHEDULE_DIGEST = "c632d59934c2610541e279cac3a48202f2c0a79bb734e995f2cc4f28d19e87d3"
MOCK_CUSTODY_DIGEST = "4c0a58772fd25fe77759d6d09089ad532a09b3a5adcdce01dc099b6b7b00dba1"
PHASE6B6_SUBTREE_DIGEST = "24789f0df9afa2d9f6a243a9050ff8f265cf22ffb42ab33bbe2f67521dbf44b5"
V2_SOURCE_DIGEST = "c95e90c3344a05d67799f44158036f316da66faf0fd66e47336ae045e8b4c976"

HERE = Path(__file__).resolve().parent
PHASE6B6 = HERE.parent
FIXED_POINT_ROOT = PHASE6B6.parents[1]
REPO_ROOT = Path(__file__).resolve()
for parent in REPO_ROOT.parents:
    if (parent / ".git").exists():
        REPO_ROOT = parent
        break
else:
    raise SystemExit("repository root not found")

CANDIDATE_PATH = HERE / "PHASE6B6_ACQUISITION_AUTHORITY_CANDIDATE.json"
SCHEMA_PATH = HERE / "schemas" / "acquisition_authority_candidate.schema.json"
ARCHITECTURE_PATH = HERE / "PHASE6B6_ACQUISITION_AUTHORITY_ARCHITECTURE.md"
ADDENDUM_PATH = FIXED_POINT_ROOT / "PHASE6_ROADMAP_STATUS_ADDENDUM_2026-07-02.md"
EVIDENCE_SEAL_PATH = (
    PHASE6B6
    / "evidence"
    / "nonhardware_qualification_3c6a5dd3_subject_d351a62f"
    / "FINAL_EVIDENCE_SEAL.json"
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot parse JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_ancestor(ancestor: str, descendant: str) -> None:
    result = run_git("merge-base", "--is-ancestor", ancestor, descendant)
    require(
        result.returncode == 0,
        f"commit ancestry failed: {ancestor} !<= {descendant}: {result.stderr.strip()}",
    )


def expected_evidence_bindings() -> dict[str, Any]:
    return {
        "integrated_evidence_head": EVIDENCE_HEAD,
        "evidence_review": EVIDENCE_REVIEW,
        "corrected_payload_commit": CORRECTED_PAYLOAD_COMMIT,
        "corrected_payload_tree": CORRECTED_PAYLOAD_TREE,
        "evidence_inventory_sha256": INVENTORY_SHA256,
        "portable_archive_sha256": ARCHIVE_SHA256,
        "portable_manifest_sha256": MANIFEST_SHA256,
        "target_final_result_digest": TARGET_RESULT_DIGEST,
        "schedule_digest": SCHEDULE_DIGEST,
        "mock_custody_digest": MOCK_CUSTODY_DIGEST,
        "phase6b6_subtree_digest": PHASE6B6_SUBTREE_DIGEST,
        "v2_source_digest": V2_SOURCE_DIGEST,
    }


def expected_gate_state() -> dict[str, Any]:
    closed = {
        "designed": True,
        "reviewed": False,
        "owner_approved": False,
        "authorized": False,
        "executed": False,
    }
    return {
        "gate_a_engineering_smoke": copy.deepcopy(closed),
        "gate_b_calibration_capture_quality": copy.deepcopy(closed),
        "gate_c_scientific_acquisition": copy.deepcopy(closed),
    }


def expected_authority() -> dict[str, Any]:
    return {
        "phase6b6_entry_approved": True,
        "phase6b6_entered": True,
        "implementation_authorized": True,
        "software_qualification_authorized": True,
        "non_hardware_target_qualification_authorized": True,
        "evidence_package_created": True,
        "qualification_evidence_collected": True,
        "hardware_ran": False,
        "authorization_artifact_created": False,
        "engineering_smoke_authorized": False,
        "calibration_authorized": False,
        "scientific_acquisition_authorized": False,
        "restoration_authorized": False,
        "target_coupling_authorized": False,
        "small_wall_authorized": False,
        "automatic_retry": False,
    }


def validate_candidate(candidate: dict[str, Any]) -> None:
    expected_keys = {
        "schema_id",
        "status",
        "base_main_commit",
        "architecture_path",
        "evidence_bindings",
        "gate_state",
        "authority",
        "review_state",
        "next_boundary",
    }
    require(set(candidate) == expected_keys, "candidate top-level key set is not closed")
    require(
        candidate["schema_id"] == "CAT_CAS_PHASE6B6_ACQUISITION_AUTHORITY_CANDIDATE_V1",
        "candidate schema_id mismatch",
    )
    require(candidate["status"] == "DESIGN_ONLY__NO_EXECUTION_AUTHORITY", "candidate status is not fail-closed")
    require(candidate["base_main_commit"] == BASE_MAIN, "candidate base main mismatch")
    require(
        candidate["architecture_path"]
        == "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/acquisition/PHASE6B6_ACQUISITION_AUTHORITY_ARCHITECTURE.md",
        "candidate architecture path mismatch",
    )
    require(candidate["evidence_bindings"] == expected_evidence_bindings(), "candidate evidence binding mismatch")
    require(candidate["gate_state"] == expected_gate_state(), "one or more physical gates are not closed")
    require(candidate["authority"] == expected_authority(), "candidate authority state mismatch")
    require(
        candidate["review_state"]
        == {
            "architecture_review_complete": False,
            "project_owner_execution_approval_recorded": False,
        },
        "candidate review state is not closed",
    )
    require(
        candidate["next_boundary"] == "INDEPENDENT_ACQUISITION_AUTHORITY_ARCHITECTURE_REVIEW",
        "candidate next boundary mismatch",
    )


def validate_schema(schema: dict[str, Any]) -> None:
    require(
        schema.get("$id") == "CAT_CAS_PHASE6B6_ACQUISITION_AUTHORITY_CANDIDATE_SCHEMA_V1",
        "schema ID mismatch",
    )
    require(schema.get("additionalProperties") is False, "schema top level is not closed")
    properties = schema.get("properties")
    require(isinstance(properties, dict), "schema properties missing")
    require(properties["status"].get("const") == "DESIGN_ONLY__NO_EXECUTION_AUTHORITY", "schema status is not closed")
    authority = properties["authority"]["properties"]
    for field in (
        "hardware_ran",
        "authorization_artifact_created",
        "engineering_smoke_authorized",
        "calibration_authorized",
        "scientific_acquisition_authorized",
        "restoration_authorized",
        "target_coupling_authorized",
        "small_wall_authorized",
        "automatic_retry",
    ):
        require(authority[field].get("const") is False, f"schema does not freeze {field}=false")


def validate_documents() -> None:
    architecture = ARCHITECTURE_PATH.read_text(encoding="utf-8")
    addendum = ADDENDUM_PATH.read_text(encoding="utf-8")
    for marker in (
        "DESIGN_ONLY__NO_EXECUTION_AUTHORITY",
        "Gate A: engineering smoke authority",
        "Gate B: calibration and capture-quality authority",
        "Gate C: frozen scientific acquisition authority",
        "engineering_smoke_authorized = false",
        "calibration_authorized = false",
        "scientific_acquisition_authorized = false",
        "No physical gate is approved by this document.",
    ):
        require(marker in architecture, f"architecture marker missing: {marker}")
    for marker in (
        "BINDING_STATUS_RECONCILIATION__NO_NEW_EXECUTION_AUTHORITY",
        "PHASE6B6_SOFTWARE_AND_NONHARDWARE_EVIDENCE_COMPLETE",
        "NO_PHYSICAL_GATE_AUTHORIZED",
        "INDEPENDENT_ACQUISITION_AUTHORITY_ARCHITECTURE_REVIEW",
    ):
        require(marker in addendum, f"roadmap addendum marker missing: {marker}")


def validate_integrated_evidence() -> None:
    seal = load_json(EVIDENCE_SEAL_PATH)
    require(seal.get("corrected_payload_commit") == CORRECTED_PAYLOAD_COMMIT, "evidence payload commit mismatch")
    require(seal.get("corrected_payload_tree") == CORRECTED_PAYLOAD_TREE, "evidence payload tree mismatch")
    require(seal.get("inventory_json_sha256") == INVENTORY_SHA256, "evidence inventory mismatch")
    require(seal.get("inventory_file_count") == 396, "evidence inventory file count mismatch")
    require(seal.get("archive_sha256") == ARCHIVE_SHA256, "portable archive mismatch")
    require(seal.get("portable_manifest_sha256") == MANIFEST_SHA256, "portable manifest mismatch")
    require(seal.get("target_final_result_digest") == TARGET_RESULT_DIGEST, "target final result mismatch")
    authority = seal.get("authority_state")
    require(isinstance(authority, dict), "evidence authority state missing")
    for field in (
        "hardware_ran",
        "authorization_artifact_created",
        "calibration_authorized",
        "scientific_acquisition_authorized",
        "restoration_authorized",
        "target_coupling_authorized",
        "small_wall_authorized",
    ):
        require(authority.get(field) is False, f"integrated evidence does not preserve {field}=false")


def validate_no_other_authority_artifact() -> None:
    for path in HERE.glob("*.json"):
        if path == CANDIDATE_PATH:
            continue
        value = load_json(path)
        status = str(value.get("status", ""))
        require("AUTHORIZED" not in status, f"unexpected acquisition authority artifact: {path.name}")


def validate_repository_state() -> None:
    head = run_git("rev-parse", "HEAD")
    require(head.returncode == 0, f"cannot resolve HEAD: {head.stderr.strip()}")
    head_sha = head.stdout.strip()
    require_ancestor(BASE_MAIN, head_sha)
    require_ancestor(EVIDENCE_HEAD, head_sha)


def self_test(candidate: dict[str, Any]) -> dict[str, bool]:
    mutations: list[tuple[str, dict[str, Any]]] = []

    changed = copy.deepcopy(candidate)
    changed["authority"]["engineering_smoke_authorized"] = True
    mutations.append(("engineering_smoke_authority_escalation", changed))

    changed = copy.deepcopy(candidate)
    changed["gate_state"]["gate_a_engineering_smoke"]["authorized"] = True
    mutations.append(("gate_a_authorized", changed))

    changed = copy.deepcopy(candidate)
    changed["gate_state"]["gate_c_scientific_acquisition"]["executed"] = True
    mutations.append(("gate_c_executed", changed))

    changed = copy.deepcopy(candidate)
    changed["authority"]["automatic_retry"] = True
    mutations.append(("automatic_retry_enabled", changed))

    changed = copy.deepcopy(candidate)
    changed["evidence_bindings"]["integrated_evidence_head"] = "0" * 40
    mutations.append(("evidence_head_changed", changed))

    changed = copy.deepcopy(candidate)
    changed["review_state"]["project_owner_execution_approval_recorded"] = True
    mutations.append(("owner_approval_fabricated", changed))

    changed = copy.deepcopy(candidate)
    changed["unexpected"] = True
    mutations.append(("extra_property", changed))

    results: dict[str, bool] = {}
    for name, mutated in mutations:
        try:
            validate_candidate(mutated)
        except VerificationError:
            results[name] = True
        else:
            results[name] = False
    require(all(results.values()), f"one or more authority mutations were accepted: {results}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    candidate = load_json(CANDIDATE_PATH)
    schema = load_json(SCHEMA_PATH)
    validate_candidate(candidate)
    validate_schema(schema)
    validate_documents()
    validate_integrated_evidence()
    validate_no_other_authority_artifact()
    validate_repository_state()

    mutation_results = self_test(candidate) if args.self_test else {}
    print(
        json.dumps(
            {
                "status": "PHASE6B6_ACQUISITION_AUTHORITY_DESIGN_VALID",
                "base_main_commit": BASE_MAIN,
                "execution_authorized": False,
                "physical_gates_closed": True,
                "mutation_results": mutation_results,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)