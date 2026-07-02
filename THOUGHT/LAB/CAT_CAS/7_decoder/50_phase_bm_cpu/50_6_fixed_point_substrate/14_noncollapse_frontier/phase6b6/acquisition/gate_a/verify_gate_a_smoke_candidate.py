#!/usr/bin/env python3
"""Validate the frozen Gate A smoke plan while preserving zero execution authority."""

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
SCHEDULE_SHA = "418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116"
TARGET_IDENTITY_SHA = "10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4"
HERE = Path(__file__).resolve().parent
PHASE6B6 = HERE.parents[1]
REPO = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)
if REPO is None:
    raise SystemExit("repository root not found")

CANDIDATE = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE.json"
SCHEDULE = HERE / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"
SCHEMA = HERE / "schemas" / "gate_a_engineering_smoke_authority_candidate.schema.json"
PLAN = HERE / "GATE_A_ENGINEERING_SMOKE_RUN_PLAN.md"
RUNTIME = PHASE6B6 / "runtime" / "explicit_slot_runtime.py"
SEAL = PHASE6B6 / "evidence" / "nonhardware_qualification_3c6a5dd3_subject_d351a62f" / "FINAL_EVIDENCE_SEAL.json"


class VerificationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"object required: {path}")
    return value


def digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_schema(candidate: dict[str, Any], schema: dict[str, Any]) -> None:
    require(schema["$id"] == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_SCHEMA_V1", "schema id mismatch")
    require(schema["additionalProperties"] is False, "schema top level open")
    require(set(candidate) == set(schema["required"]), "candidate key set mismatch")
    require(set(schema["properties"]) == set(schema["required"]), "schema property set mismatch")
    for key in schema["required"]:
        definition = schema["properties"][key]
        require("const" in definition, f"field is not const-bound: {key}")
        require(candidate[key] == definition["const"], f"candidate differs from schema: {key}")


def validate_schedule(schedule: dict[str, Any]) -> None:
    require(schedule["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_SCHEDULE_V1", "schedule schema mismatch")
    require(schedule["status"] == "PROPOSED__NO_EXECUTION_AUTHORITY", "schedule authority open")
    require(schedule["base_main_commit"] == BASE_MAIN, "schedule base mismatch")
    require(schedule["schedule_sha256"] == SCHEDULE_SHA, "schedule digest field mismatch")
    unsigned = copy.deepcopy(schedule)
    unsigned.pop("schedule_sha256")
    require(digest(unsigned) == SCHEDULE_SHA, "schedule content digest mismatch")
    require(schedule["slot_sequence"] == ["I", "I", "I", "I", "C0", "D0", "S0E", "S0E", "S0E", "S0E", "O0", "O0", "A0P", "A0N", "T", "T"], "slot sequence mismatch")
    require(schedule["timing"] == {
        "automatic_retry": False,
        "maximum_execution_count": 1,
        "nominal_duration_s": 8.0,
        "nominal_samples_per_slot": 4000,
        "read_hz": 8000,
        "slot_count": 16,
        "slot_s": 0.5,
        "temperature_veto_c": 68.0,
    }, "timing mismatch")
    require(all(v is False for v in schedule["scientific_use"].values()), "scientific use enabled")
    require(schedule["target"]["target_identity_stdout_sha256"] == TARGET_IDENTITY_SHA, "target identity mismatch")
    frequency = schedule["frequency_and_voltage"]
    require(frequency["expected_observed_khz"] == 1600000, "frequency predicate mismatch")
    require(frequency["mismatch_action"] == "STOP_BEFORE_DRIVE", "frequency mismatch action open")
    for key in ("frequency_write_authorized", "voltage_write_authorized", "msr_write_authorized"):
        require(frequency[key] is False, f"write authority enabled: {key}")

    defs = schedule["slot_definitions"]
    require(set(defs) == {"I", "C0", "D0", "S0E", "O0", "A0P", "A0N", "T"}, "token set mismatch")
    for token, definition in defs.items():
        require(set(definition) == {"stage", "declared", "executed"}, f"open token: {token}")
        require(set(definition["declared"]) == {"analysis_tone_index", "declared_amplitude_level", "declared_mode", "declared_phase_action", "declared_physical_tone_index", "declared_sign"}, f"declared set mismatch: {token}")
        require(set(definition["executed"]) == {"amplitude_level", "drive_on", "executed_mode", "phase_action", "physical_tone_index", "sender_epoch_id", "sign"}, f"executed set mismatch: {token}")

    for token in ("I", "C0", "D0", "O0", "T"):
        executed = defs[token]["executed"]
        require(executed["drive_on"] is False, f"off token drives: {token}")
        for key in ("amplitude_level", "phase_action", "physical_tone_index", "sender_epoch_id", "sign"):
            require(executed[key] is None, f"off token carries executed {key}: {token}")
    require(defs["D0"]["declared"]["declared_amplitude_level"] == 2, "sham declaration lost amplitude")
    require(defs["D0"]["executed"]["physical_tone_index"] is None, "sham executes tone")
    require(defs["S0E"]["executed"] == {
        "amplitude_level": 2,
        "drive_on": True,
        "executed_mode": "STEP",
        "phase_action": "0",
        "physical_tone_index": 0,
        "sender_epoch_id": "gate-a:step:epoch0",
        "sign": 1,
    }, "STEP epoch mismatch")
    require(defs["A0P"]["executed"]["sign"] == 1 and defs["A0P"]["executed"]["phase_action"] == "0", "positive anchor mismatch")
    require(defs["A0N"]["executed"]["sign"] == -1 and defs["A0N"]["executed"]["phase_action"] == "pi", "negative anchor mismatch")


def validate_integrated_state() -> None:
    seal = load(SEAL)
    require(seal["corrected_payload_commit"] == "38eccc2f8377c656b0c21cbd37dd296e81adfad4", "evidence payload mismatch")
    require(seal["inventory_json_sha256"] == "7b205dd92482425505e498027a3e96db842297fad949fe22bdbf5542e95573e0", "evidence inventory mismatch")
    require(seal["target_final_result_digest"] == "c2d1bf3c78e2a9318f51e06d27ac39a49fe7a49e3cd49c0c8850cd6c85a07f7f", "target result mismatch")
    for key in ("hardware_ran", "authorization_artifact_created", "calibration_authorized", "scientific_acquisition_authorized", "restoration_authorized", "target_coupling_authorized", "small_wall_authorized"):
        require(seal["authority_state"][key] is False, f"integrated authority enabled: {key}")
    runtime = RUNTIME.read_text(encoding="utf-8")
    require("SOFTWARE_ENTRY_ONLY_AUTHORITY: real hardware execution is not authorized" in runtime, "runtime hardware rejection missing")
    require("def reject_real_hardware" in runtime and "if hardware:" in runtime, "runtime hardware guard missing")
    plan = PLAN.read_text(encoding="utf-8")
    for marker in ("FROZEN_PLAN__NO_EXECUTION_AUTHORITY", SCHEDULE_SHA, "hardware adapter implemented = false", "engineering smoke authorized = false", "GATE_A_HARDWARE_ADAPTER_IMPLEMENTATION_AND_NONEXECUTING_QUALIFICATION"):
        require(marker in plan, f"plan marker missing: {marker}")


def self_test(candidate: dict[str, Any], schedule: dict[str, Any], schema: dict[str, Any]) -> dict[str, bool]:
    cases: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    def add(name: str, c: dict[str, Any] | None = None, s: dict[str, Any] | None = None) -> None:
        cases.append((name, c or copy.deepcopy(candidate), s or copy.deepcopy(schedule)))
    c = copy.deepcopy(candidate); c["authority"]["engineering_smoke_authorized"] = True; add("authority_escalation", c=c)
    c = copy.deepcopy(candidate); c["execution_bundle"]["execution_bundle_ready"] = True; add("bundle_fabricated", c=c)
    c = copy.deepcopy(candidate); c["evidence_bindings"]["portable_archive_sha256"] = "0" * 64; add("predecessor_changed", c=c)
    c = copy.deepcopy(candidate); c["unexpected"] = True; add("extra_property", c=c)
    s = copy.deepcopy(schedule); s["scientific_use"]["scientific_dataset_eligible"] = True; add("scientific_use", s=s)
    s = copy.deepcopy(schedule); s["timing"]["automatic_retry"] = True; add("retry_enabled", s=s)
    s = copy.deepcopy(schedule); s["frequency_and_voltage"]["msr_write_authorized"] = True; add("msr_write", s=s)
    s = copy.deepcopy(schedule); s["slot_sequence"][6] = "I"; add("sequence_changed", s=s)
    s = copy.deepcopy(schedule); s["slot_definitions"]["D0"]["executed"]["physical_tone_index"] = 0; add("sham_executes", s=s)
    s = copy.deepcopy(schedule); s["slot_definitions"]["O0"]["executed"]["amplitude_level"] = 2; add("off_executes", s=s)
    results: dict[str, bool] = {}
    for name, changed_candidate, changed_schedule in cases:
        try:
            validate_schema(changed_candidate, schema)
            validate_schedule(changed_schedule)
        except VerificationError:
            results[name] = True
        else:
            results[name] = False
    require(all(results.values()), f"mutation accepted: {results}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    candidate, schedule, schema = load(CANDIDATE), load(SCHEDULE), load(SCHEMA)
    validate_schema(candidate, schema)
    validate_schedule(schedule)
    validate_integrated_state()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    require(subprocess.run(["git", "merge-base", "--is-ancestor", BASE_MAIN, head], cwd=REPO).returncode == 0, "base main is not ancestor")
    results = self_test(candidate, schedule, schema) if args.self_test else {}
    print(json.dumps({
        "status": "PHASE6B6_GATE_A_SMOKE_CANDIDATE_VALID",
        "schedule_sha256": SCHEDULE_SHA,
        "declared_executed_controls_separated": True,
        "execution_authorized": False,
        "hardware_adapter_ready": False,
        "physical_gate_closed": True,
        "mutation_results": results,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
