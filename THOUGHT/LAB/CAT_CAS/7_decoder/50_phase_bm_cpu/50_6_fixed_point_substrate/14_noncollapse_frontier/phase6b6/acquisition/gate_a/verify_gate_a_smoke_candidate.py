#!/usr/bin/env python3
"""Fail-closed verifier for the Phase 6B.6 Gate A smoke package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BASE_MAIN = "9c41637992536f43d10d152ec176a3577aef1623"
ARCHITECTURE_HEAD = "b3c0ec7cfceeaea95fc9d37fa98266987f418397"
ARCHITECTURE_REVIEW = 4614574719
EVIDENCE_HEAD = "b2b785d064d4704ef2955238593f9f5050425f55"
SMOKE_SCHEDULE_SHA256 = "b2d73afb5b4d3fd351abbb4aa7f7f76cfa532dcff0808f331682b456d3e6e6ed"
TARGET_IDENTITY_STDOUT_SHA256 = "10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4"

HERE = Path(__file__).resolve().parent
PHASE6B6 = HERE.parents[1]
REPO_ROOT = Path(__file__).resolve()
for parent in REPO_ROOT.parents:
    if (parent / ".git").exists():
        REPO_ROOT = parent
        break
else:
    raise SystemExit("repository root not found")

CANDIDATE_PATH = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE.json"
SCHEDULE_PATH = HERE / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"
PLAN_PATH = HERE / "GATE_A_ENGINEERING_SMOKE_RUN_PLAN.md"
SCHEMA_PATH = HERE / "schemas" / "gate_a_engineering_smoke_authority_candidate.schema.json"
ARCHITECTURE_PATH = HERE.parent / "PHASE6B6_ACQUISITION_AUTHORITY_ARCHITECTURE.md"
RUNTIME_PATH = PHASE6B6 / "runtime" / "explicit_slot_runtime.py"
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
    require(path.is_file(), f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
    require(result.returncode == 0, f"ancestry failed: {ancestor} !<= {descendant}: {result.stderr.strip()}")


def expected_sequence() -> list[str]:
    return ["I", "I", "I", "I", "C0", "D0", "S0E", "S0E", "S0E", "S0E", "O0", "O0", "A0P", "A0N", "T", "T"]


def validate_schedule(schedule: dict[str, Any]) -> None:
    require(schedule.get("schema_id") == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_SCHEDULE_V1", "schedule schema mismatch")
    require(schedule.get("status") == "PROPOSED__NO_EXECUTION_AUTHORITY", "schedule status is not closed")
    require(schedule.get("base_main_commit") == BASE_MAIN, "schedule base main mismatch")
    declared = schedule.get("schedule_sha256")
    require(declared == SMOKE_SCHEDULE_SHA256, "schedule declared digest mismatch")
    payload = copy.deepcopy(schedule)
    payload.pop("schedule_sha256", None)
    require(digest(payload) == SMOKE_SCHEDULE_SHA256, "schedule content digest mismatch")

    require(schedule.get("slot_sequence") == expected_sequence(), "slot sequence mismatch")
    definitions = schedule.get("slot_definitions")
    require(isinstance(definitions, dict), "slot definitions missing")
    require(set(definitions) == {"I", "C0", "D0", "S0E", "O0", "A0P", "A0N", "T"}, "slot definition set mismatch")
    require(schedule["timing"] == {
        "automatic_retry": False,
        "maximum_execution_count": 1,
        "nominal_duration_s": 8.0,
        "nominal_samples_per_slot": 4000,
        "read_hz": 8000,
        "slot_count": 16,
        "slot_s": 0.5,
        "temperature_veto_c": 68.0,
    }, "timing geometry mismatch")
    require(schedule["session"] == {
        "boot_state_count": 1,
        "reboot_block": None,
        "receiver_core": 5,
        "route": "v4s5",
        "run_namespace": "phase6b6/gate_a/engineering_smoke_01",
        "scientific_split": None,
        "sender_core": 4,
        "session_count": 1,
        "session_index": 0,
    }, "session geometry mismatch")
    scientific = schedule["scientific_use"]
    require(all(value is False for value in scientific.values()), "scientific use is not fully disabled")
    frequency = schedule["frequency_and_voltage"]
    require(frequency["expected_observed_khz"] == 1600000, "observed frequency predicate mismatch")
    for field in ("frequency_write_authorized", "voltage_write_authorized", "msr_write_authorized"):
        require(frequency[field] is False, f"write authority escalation: {field}")
    require(frequency["mismatch_action"] == "STOP_BEFORE_DRIVE", "frequency mismatch action is not fail closed")
    require(schedule["target"]["target_identity_stdout_sha256"] == TARGET_IDENTITY_STDOUT_SHA256, "target identity binding mismatch")

    step = definitions["S0E"]
    require(step["drive_on"] is True and step["sender_epoch_id"] == "gate-a:step:epoch0", "contiguous sender epoch mismatch")
    for token in ("I", "C0", "D0", "O0", "T"):
        require(definitions[token]["drive_on"] is False, f"off or sham token drives hardware: {token}")
    require(definitions["A0P"]["sign"] == 1 and definitions["A0P"]["phase_action"] == "0", "positive anchor mismatch")
    require(definitions["A0N"]["sign"] == -1 and definitions["A0N"]["phase_action"] == "pi", "negative anchor mismatch")


def validate_candidate(candidate: dict[str, Any]) -> None:
    require(candidate.get("schema_id") == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V1", "candidate schema mismatch")
    require(candidate.get("status") == "CANDIDATE__BLOCKED_PENDING_REVIEW_AND_EXECUTION_BUNDLE", "candidate status is not blocked")
    require(candidate.get("base_main_commit") == BASE_MAIN, "candidate base main mismatch")
    architecture = candidate["architecture_binding"]
    require(architecture["architecture_merge_commit"] == BASE_MAIN, "architecture merge binding mismatch")
    require(architecture["architecture_reviewed_head"] == ARCHITECTURE_HEAD, "architecture head mismatch")
    require(architecture["architecture_review_id"] == ARCHITECTURE_REVIEW, "architecture review mismatch")
    require(candidate["evidence_bindings"]["integrated_evidence_head"] == EVIDENCE_HEAD, "evidence head mismatch")
    require(candidate["evidence_bindings"]["target_identity_stdout_sha256"] == TARGET_IDENTITY_STDOUT_SHA256, "candidate target identity mismatch")
    require(candidate["smoke_schedule"]["schedule_sha256"] == SMOKE_SCHEDULE_SHA256, "candidate schedule mismatch")
    require(candidate["smoke_schedule"]["slot_count"] == 16, "candidate slot count mismatch")
    require(candidate["smoke_schedule"]["maximum_execution_count"] == 1, "candidate execution count mismatch")

    bundle = candidate["execution_bundle"]
    require(bundle == {
        "execution_bundle_ready": False,
        "execution_bundle_sha256": None,
        "hardware_adapter_path": None,
        "hardware_adapter_qualified": False,
        "hardware_adapter_sha256": None,
        "qualified_portable_archive_sha256": "affbc0b3e9725de62aa946774e3e8830399f9af12414713b1bfbc68547765ca4",
        "qualified_portable_manifest_sha256": "59e5c5927cfa7f19bdaafdd740cb350f5819e81741b62821a22f2eb80ecd4676",
        "status": "NOT_IMPLEMENTED__NOT_QUALIFIED",
    }, "execution bundle is not closed")
    require(candidate["review_state"] == {
        "candidate_review_complete": False,
        "execution_bundle_review_complete": False,
        "project_owner_execution_approval_recorded": False,
        "run_plan_review_complete": False,
    }, "review state is not closed")
    require(candidate["gate_state"] == {
        "authorized": False,
        "consumed": False,
        "designed": True,
        "executed": False,
        "owner_approved": False,
        "reviewed": False,
        "run_plan_frozen": True,
    }, "Gate A state is not closed")
    authority = candidate["authority"]
    for field in (
        "authorization_artifact_created",
        "automatic_retry",
        "calibration_authorized",
        "engineering_smoke_authorized",
        "hardware_ran",
        "restoration_authorized",
        "scientific_acquisition_authorized",
        "small_wall_authorized",
        "target_coupling_authorized",
    ):
        require(authority[field] is False, f"authority escalation: {field}")
    require(candidate["next_boundary"] == "INDEPENDENT_GATE_A_PLAN_AND_CANDIDATE_REVIEW", "next boundary mismatch")
    require(candidate["post_review_boundary"] == "GATE_A_HARDWARE_ADAPTER_IMPLEMENTATION_AND_NONEXECUTING_QUALIFICATION", "post-review boundary mismatch")


def validate_integrated_state() -> None:
    seal = load_json(EVIDENCE_SEAL_PATH)
    require(seal["corrected_payload_commit"] == "38eccc2f8377c656b0c21cbd37dd296e81adfad4", "evidence payload mismatch")
    require(seal["inventory_json_sha256"] == "7b205dd92482425505e498027a3e96db842297fad949fe22bdbf5542e95573e0", "evidence inventory mismatch")
    require(seal["archive_sha256"] == "affbc0b3e9725de62aa946774e3e8830399f9af12414713b1bfbc68547765ca4", "archive mismatch")
    require(seal["target_final_result_digest"] == "c2d1bf3c78e2a9318f51e06d27ac39a49fe7a49e3cd49c0c8850cd6c85a07f7f", "target result mismatch")
    for field in ("hardware_ran", "authorization_artifact_created", "calibration_authorized", "scientific_acquisition_authorized", "restoration_authorized", "target_coupling_authorized", "small_wall_authorized"):
        require(seal["authority_state"][field] is False, f"integrated evidence authority escalation: {field}")

    architecture = ARCHITECTURE_PATH.read_text(encoding="utf-8")
    require("Gate A: engineering smoke authority" in architecture, "Gate A architecture missing")
    require("No physical gate is approved by this document." in architecture, "architecture boundary missing")
    runtime = RUNTIME_PATH.read_text(encoding="utf-8")
    require("SOFTWARE_ENTRY_ONLY_AUTHORITY: real hardware execution is not authorized" in runtime, "runtime hardware rejection missing")
    require("def reject_real_hardware" in runtime, "runtime hardware rejection function missing")
    require("if hardware:" in runtime, "runtime hardware guard missing")
    plan = PLAN_PATH.read_text(encoding="utf-8")
    for marker in ("FROZEN_PLAN__NO_EXECUTION_AUTHORITY", "hardware adapter implemented = false", "engineering smoke authorized = false", "GATE_A_HARDWARE_ADAPTER_IMPLEMENTATION_AND_NONEXECUTING_QUALIFICATION"):
        require(marker in plan, f"run-plan marker missing: {marker}")


def validate_schema(schema: dict[str, Any]) -> None:
    require(schema.get("$id") == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_SCHEMA_V1", "schema ID mismatch")
    require(schema.get("additionalProperties") is False, "schema top level is open")
    props = schema["properties"]
    require(props["status"]["const"] == "CANDIDATE__BLOCKED_PENDING_REVIEW_AND_EXECUTION_BUNDLE", "schema status mismatch")
    require(props["smoke_schedule"]["properties"]["schedule_sha256"]["const"] == SMOKE_SCHEDULE_SHA256, "schema schedule digest mismatch")
    require(props["execution_bundle"]["properties"]["execution_bundle_ready"]["const"] is False, "schema bundle can become ready")
    require(props["execution_bundle"]["properties"]["hardware_adapter_qualified"]["const"] is False, "schema adapter can become qualified")


def validate_repository_state() -> None:
    head = run_git("rev-parse", "HEAD")
    require(head.returncode == 0, f"cannot resolve HEAD: {head.stderr.strip()}")
    require_ancestor(BASE_MAIN, head.stdout.strip())


def self_test(candidate: dict[str, Any], schedule: dict[str, Any]) -> dict[str, bool]:
    mutations: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    def add(name: str, candidate_mutation: dict[str, Any] | None = None, schedule_mutation: dict[str, Any] | None = None) -> None:
        mutations.append((name, candidate_mutation or copy.deepcopy(candidate), schedule_mutation or copy.deepcopy(schedule)))

    changed = copy.deepcopy(candidate); changed["authority"]["engineering_smoke_authorized"] = True; add("smoke_authority_escalation", changed)
    changed = copy.deepcopy(candidate); changed["gate_state"]["authorized"] = True; add("gate_authorized", changed)
    changed = copy.deepcopy(candidate); changed["review_state"]["project_owner_execution_approval_recorded"] = True; add("owner_approval_fabricated", changed)
    changed = copy.deepcopy(candidate); changed["execution_bundle"]["execution_bundle_ready"] = True; add("bundle_ready_without_qualification", changed)
    changed = copy.deepcopy(candidate); changed["execution_bundle"]["hardware_adapter_qualified"] = True; add("adapter_qualification_fabricated", changed)
    changed = copy.deepcopy(schedule); changed["timing"]["automatic_retry"] = True; add("automatic_retry_enabled", schedule_mutation=changed)
    changed = copy.deepcopy(schedule); changed["scientific_use"]["scientific_dataset_eligible"] = True; add("scientific_dataset_enabled", schedule_mutation=changed)
    changed = copy.deepcopy(schedule); changed["frequency_and_voltage"]["msr_write_authorized"] = True; add("msr_write_enabled", schedule_mutation=changed)
    changed = copy.deepcopy(schedule); changed["slot_sequence"][6] = "I"; add("slot_sequence_changed", schedule_mutation=changed)
    changed = copy.deepcopy(schedule); changed["timing"]["maximum_execution_count"] = 2; add("execution_count_increased", schedule_mutation=changed)

    results: dict[str, bool] = {}
    for name, mutated_candidate, mutated_schedule in mutations:
        try:
            validate_candidate(mutated_candidate)
            validate_schedule(mutated_schedule)
        except VerificationError:
            results[name] = True
        else:
            results[name] = False
    require(all(results.values()), f"one or more mutations were accepted: {results}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    candidate = load_json(CANDIDATE_PATH)
    schedule = load_json(SCHEDULE_PATH)
    schema = load_json(SCHEMA_PATH)
    validate_candidate(candidate)
    validate_schedule(schedule)
    validate_schema(schema)
    validate_integrated_state()
    validate_repository_state()
    mutations = self_test(candidate, schedule) if args.self_test else {}
    print(json.dumps({
        "status": "PHASE6B6_GATE_A_SMOKE_CANDIDATE_VALID",
        "base_main_commit": BASE_MAIN,
        "schedule_sha256": SMOKE_SCHEDULE_SHA256,
        "slot_count": 16,
        "execution_authorized": False,
        "hardware_adapter_ready": False,
        "physical_gate_closed": True,
        "mutation_results": mutations,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
