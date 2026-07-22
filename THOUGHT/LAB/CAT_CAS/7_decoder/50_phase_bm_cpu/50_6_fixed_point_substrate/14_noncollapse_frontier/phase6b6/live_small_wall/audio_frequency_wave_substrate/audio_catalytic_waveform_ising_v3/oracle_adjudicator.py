from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
MACHINE_SOURCE = PACKAGE_DIR / "v3_machine.py"
FREEZE_FILE = PACKAGE_DIR / "V3_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "V3_BATCH_CUSTODY.json"
EVIDENCE_FILE = PACKAGE_DIR / "V3_PREORACLE_EVIDENCE.json"
TRACE_FILE = PACKAGE_DIR / "V3_PREORACLE_TRACE.json"
SEAL_FILE = PACKAGE_DIR / "V3_PREORACLE_SEAL.json"
CONTROL_RESULTS = PACKAGE_DIR / "CONTROL_RESULTS.json"
RESULT_FILE = PACKAGE_DIR / "V3_BATCH_RESULTS.json"
ORACLE_TRACE_FILE = PACKAGE_DIR / "V3_ORACLE_TRACE.json"
REPORT_FILE = PACKAGE_DIR / "V3_BATCH_REPORT.md"

VERIFIED = "CATALYTIC_WAVEFORM_ISING_V3_VERIFIED"
PARTIAL = "CATALYTIC_WAVEFORM_ISING_V3_PARTIAL"
NOT_ESTABLISHED = "CATALYTIC_WAVEFORM_ISING_V3_NOT_ESTABLISHED"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


machine = load_module(MACHINE_SOURCE, "catcas_waveform_ising_v3_oracle_machine")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problem_sha256(coupling: np.ndarray, field: np.ndarray) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "coupling_matrix_J": np.asarray(
                    coupling, dtype=np.float64
                ).tolist(),
                "field_vector_h": np.asarray(field, dtype=np.float64).tolist(),
            }
        )
    )


def repository_root() -> Path:
    for candidate in PACKAGE_DIR.parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("cannot locate linked worktree root")


def current_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root(),
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


@dataclass
class EventLedger:
    events: list[dict[str, Any]]
    previous_hash: str = "0" * 64

    def add(self, name: str, payload: Any) -> None:
        event = {
            "event_index": len(self.events),
            "name": name,
            "payload_sha256": sha256_bytes(canonical_bytes(payload)),
            "previous_event_sha256": self.previous_hash,
        }
        event["event_sha256"] = sha256_bytes(canonical_bytes(event))
        self.previous_hash = event["event_sha256"]
        self.events.append(event)

    def document(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "events": self.events,
            "final_event_sha256": self.previous_hash,
            "schema": "catalytic_waveform_ising_v3_oracle_trace_v1",
        }


def load_and_verify_custody() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    batch = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
    seal = json.loads(SEAL_FILE.read_text(encoding="utf-8"))
    if machine.machine_fingerprint() != freeze["machine_fingerprint"]:
        raise RuntimeError("machine changed after freeze")
    if freeze["batch_file_sha256"] != sha256_file(BATCH_FILE):
        raise RuntimeError("batch file bytes changed after freeze")
    if batch["ordered_batch_sha256"] != freeze["batch_ordered_sha256"]:
        raise RuntimeError("batch custody differs from freeze")
    if evidence["batch_ordered_sha256"] != freeze["batch_ordered_sha256"]:
        raise RuntimeError("pre-oracle evidence binds a different batch")
    if sha256_file(EVIDENCE_FILE) != seal["evidence_sha256"]:
        raise RuntimeError("pre-oracle seal does not bind evidence")
    if sha256_file(TRACE_FILE) != seal["trace_sha256"]:
        raise RuntimeError("pre-oracle seal does not bind trace")
    if evidence["freeze_commit"] != seal["freeze_commit"]:
        raise RuntimeError("pre-oracle freeze-commit custody mismatch")
    if evidence["oracle_opened"] or seal["oracle_call_count"] != 0:
        raise RuntimeError("oracle-order custody failed")
    if seal["energy_call_count"] != 0:
        raise RuntimeError("energy was reached before the seal")
    if len(evidence["instances"]) != 256 or len(batch["ordered_instances"]) != 256:
        raise RuntimeError("prospective evidence is incomplete")
    identities: list[str] = []
    for expected_index, problem in enumerate(batch["ordered_instances"]):
        if int(problem["index"]) != expected_index:
            raise RuntimeError("prospective batch index mismatch")
        identity = problem_sha256(
            np.asarray(problem["coupling_matrix_J"], dtype=np.float64),
            np.asarray(problem["field_vector_h"], dtype=np.float64),
        )
        if identity != problem["problem_sha256"]:
            raise RuntimeError("prospective problem identity mismatch")
        if evidence["instances"][expected_index]["problem_sha256"] != identity:
            raise RuntimeError("pre-oracle instance identity mismatch")
        identities.append(identity)
    if sha256_bytes(canonical_bytes(identities)) != freeze["batch_ordered_sha256"]:
        raise RuntimeError("prospective ordered identity hash mismatch")
    for name, expected_sha in freeze["execution_source_sha256"].items():
        if sha256_file(PACKAGE_DIR / name) != expected_sha:
            raise RuntimeError(f"frozen execution source drift: {name}")
    # The transform-removed control is the frozen null baseline for every case.
    if not all(
        record["strict_controls"]["checks"]["transform_removal_is_material"]
        for record in evidence["instances"]
    ):
        raise RuntimeError("transform-removed null baseline failed")
    return freeze, batch, evidence, seal


def energy(spins: Sequence[int], coupling: np.ndarray, field: np.ndarray) -> float:
    vector = np.asarray(spins, dtype=np.float64)
    return float(-0.5 * vector @ coupling @ vector - field @ vector)


def exact_oracle(
    coupling: np.ndarray, field: np.ndarray
) -> tuple[float, tuple[tuple[int, ...], ...], tuple[tuple[float, tuple[int, ...]], ...]]:
    rows = tuple(
        sorted(
            (
                energy(state, coupling, field),
                tuple(int(value) for value in state),
            )
            for state in itertools.product((-1, 1), repeat=machine.SITE_COUNT)
        )
    )
    optimum_energy = float(rows[0][0])
    optima = tuple(
        state for row_energy, state in rows if abs(row_energy - optimum_energy) <= 1.0e-12
    )
    return optimum_energy, optima, rows


def expected_mode_penalties(
    coupling: np.ndarray, field: np.ndarray
) -> tuple[float, ...]:
    penalties: list[float] = []
    for column in range(machine.MODE_COUNT):
        mode = np.real(machine.PHASE_MODES[:, column])
        total = 0.0
        for left in range(machine.SITE_COUNT):
            for right in range(left + 1, machine.SITE_COUNT):
                strength = float(coupling[left, right])
                total += abs(strength) - strength * float(mode[left] * mode[right])
        for site in range(machine.SITE_COUNT):
            strength = float(field[site])
            total += abs(strength) - strength * float(mode[site])
        penalties.append(total)
    return tuple(float(value) for value in penalties)


def adjudicate_instance(
    problem: dict[str, Any], evidence_record: dict[str, Any]
) -> dict[str, Any]:
    if int(problem["index"]) != int(evidence_record["index"]):
        raise RuntimeError("instance order changed")
    if problem["problem_sha256"] != evidence_record["problem_sha256"]:
        raise RuntimeError("instance identity changed")
    coupling = np.asarray(problem["coupling_matrix_J"], dtype=np.float64)
    field = np.asarray(problem["field_vector_h"], dtype=np.float64)
    optimum_energy, optima, rows = exact_oracle(coupling, field)
    raw_spins = tuple(int(value) for value in evidence_record["native"]["raw_spins"])
    raw_energy = energy(raw_spins, coupling, field)
    raw_correct = raw_spins in optima
    unique = len(optima) == 1
    accepted = bool(evidence_record["native"]["valid"])
    if unique and accepted and raw_correct:
        classification = "UNIQUE_ACCEPTED_CORRECT"
    elif unique and accepted and not raw_correct:
        classification = "UNIQUE_ACCEPTED_INCORRECT"
    elif unique and not accepted and raw_correct:
        classification = "UNIQUE_REJECTED_CORRECT"
    elif unique:
        classification = "UNIQUE_REJECTED_INCORRECT"
    elif accepted:
        classification = "NON_UNIQUE_ACCEPTED"
    else:
        classification = "NON_UNIQUE_REJECTED"
    expected_penalties = expected_mode_penalties(coupling, field)
    observed_penalties = tuple(
        float(value) for value in evidence_record["native"]["mode_penalties"]
    )
    penalty_delta = float(
        np.max(np.abs(np.asarray(expected_penalties) - np.asarray(observed_penalties)))
    )
    if penalty_delta > 1.0e-9:
        raise RuntimeError("carrier mode penalties do not reproduce Ising relations")
    return {
        "accepted": accepted,
        "classification": classification,
        "index": int(problem["index"]),
        "mode_penalty_max_abs_delta": metric(penalty_delta),
        "optimum_energy": metric(optimum_energy),
        "optimum_state_count": len(optima),
        "optimum_states": [list(state) for state in optima],
        "problem_sha256": str(problem["problem_sha256"]),
        "raw_energy": metric(raw_energy),
        "raw_matches_optimum": raw_correct,
        "raw_spins": list(raw_spins),
        "state_count_enumerated": len(rows),
    }


def build_documents(
    preoracle_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, batch, evidence, seal = load_and_verify_custody()
    evidence_sha256 = seal["evidence_sha256"]
    freeze_commit = seal["freeze_commit"]
    preoracle_trace_sha256 = seal["trace_sha256"]
    expected_batch_sha256 = freeze["batch_ordered_sha256"]
    ledger = EventLedger([])
    ledger.add(
        "remote_preoracle_custody_verified",
        {
            "evidence_sha256": evidence_sha256,
            "freeze_commit": freeze_commit,
            "preoracle_commit": preoracle_commit,
            "preoracle_trace_sha256": preoracle_trace_sha256,
        },
    )
    outcomes = [
        adjudicate_instance(problem, record)
        for problem, record in zip(
            batch["ordered_instances"], evidence["instances"]
        )
    ]
    for outcome in outcomes:
        ledger.add(f"instance_{outcome['index']:03d}_oracle_adjudicated", outcome)
    unique = [outcome for outcome in outcomes if outcome["optimum_state_count"] == 1]
    non_unique = [
        outcome for outcome in outcomes if outcome["optimum_state_count"] != 1
    ]
    counts = {
        name: sum(outcome["classification"] == name for outcome in outcomes)
        for name in (
            "UNIQUE_ACCEPTED_CORRECT",
            "UNIQUE_ACCEPTED_INCORRECT",
            "UNIQUE_REJECTED_CORRECT",
            "UNIQUE_REJECTED_INCORRECT",
            "NON_UNIQUE_ACCEPTED",
            "NON_UNIQUE_REJECTED",
        )
    }
    controls_document = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    criterion = freeze["promotion_criterion"]
    unique_raw_correct = sum(outcome["raw_matches_optimum"] for outcome in unique)
    unique_rate = unique_raw_correct / len(unique) if unique else 0.0
    checks = {
        "accepted_incorrect_maximum": counts["UNIQUE_ACCEPTED_INCORRECT"]
        <= criterion["accepted_incorrect_maximum"],
        "batch_size_required": len(outcomes) == criterion["batch_size_required"],
        "minimum_unique_optimum_cases": len(unique)
        >= criterion["minimum_unique_optimum_cases"],
        "native_no_smuggle_must_pass": controls_document["overall_pass"],
        "non_unique_accepted_maximum": counts["NON_UNIQUE_ACCEPTED"]
        <= criterion["non_unique_accepted_maximum"],
        "oracle_calls_before_preoracle_seal": seal["oracle_call_count"]
        == criterion["oracle_calls_before_preoracle_seal"],
        "rejected_unique_correct_maximum": counts["UNIQUE_REJECTED_CORRECT"]
        <= criterion["rejected_unique_correct_maximum"],
        "restoration_all_cases": evidence["summary"]["restoration_reuse_pass_count"]
        == 256,
        "reuse_all_cases": evidence["summary"]["restoration_reuse_pass_count"] == 256,
        "strict_controls_all_cases": evidence["summary"]["strict_control_pass_count"]
        == 256,
        "unique_raw_correct_rate_minimum": unique_rate
        >= criterion["unique_raw_correct_rate_minimum"],
        "uninterpretable_maximum": evidence["summary"]["uninterpretable_count"]
        <= criterion["uninterpretable_maximum"],
    }
    promotion_pass = all(checks.values())
    if promotion_pass:
        decision = VERIFIED
    elif (
        len(unique) >= criterion["minimum_unique_optimum_cases"]
        and counts["UNIQUE_ACCEPTED_INCORRECT"] == 0
        and evidence["summary"]["restoration_reuse_pass_count"] == 256
    ):
        decision = PARTIAL
    else:
        decision = NOT_ESTABLISHED
    summary = {
        "batch_size": len(outcomes),
        "classification_counts": counts,
        "decision": decision,
        "maximum_mode_penalty_delta": metric(
            max(outcome["mode_penalty_max_abs_delta"] for outcome in outcomes)
        ),
        "non_unique_count": len(non_unique),
        "non_unique_raw_optimum_matches": sum(
            outcome["raw_matches_optimum"] for outcome in non_unique
        ),
        "promotion_checks": checks,
        "promotion_pass": promotion_pass,
        "state_count_enumerated": sum(
            outcome["state_count_enumerated"] for outcome in outcomes
        ),
        "unique_count": len(unique),
        "unique_raw_correct": unique_raw_correct,
        "unique_raw_correct_rate": metric(unique_rate),
    }
    ledger.add("frozen_promotion_law_applied", summary)
    trace = ledger.document()
    results = {
        "batch_ordered_sha256": expected_batch_sha256,
        "claim_ceiling": machine.CLAIM_CEILING,
        "decision": decision,
        "freeze_commit": freeze_commit,
        "machine_fingerprint": freeze["machine_fingerprint"],
        "outcomes": outcomes,
        "preoracle_commit": preoracle_commit,
        "preoracle_evidence_sha256": evidence_sha256,
        "promotion_criterion": criterion,
        "schema": "catalytic_waveform_ising_v3_batch_results_v1",
        "summary": summary,
    }
    return results, trace


def report_bytes(results: dict[str, Any], trace: dict[str, Any]) -> bytes:
    summary = results["summary"]
    counts = summary["classification_counts"]
    lines = [
        "# Catalytic Waveform-Ising V3 Prospective Result",
        "",
        f"Decision: `{results['decision']}`",
        "",
        "```text",
        f"batch size                 {summary['batch_size']}",
        f"unique optima              {summary['unique_count']}",
        f"unique raw correct         {summary['unique_raw_correct']}",
        f"accepted incorrect         {counts['UNIQUE_ACCEPTED_INCORRECT']}",
        f"non-unique rejected        {counts['NON_UNIQUE_REJECTED']}",
        f"promotion pass             {summary['promotion_pass']}",
        "```",
        "",
        f"Oracle trace final hash: `{trace['final_event_sha256']}`",
        "",
        "The result is a bounded 32-mode software spectral reference. It establishes",
        "neither scaling advantage nor physical waveform computation.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    results, trace = build_documents(current_head())
    write_atomic(RESULT_FILE, canonical_bytes(results))
    write_atomic(ORACLE_TRACE_FILE, canonical_bytes(trace))
    write_atomic(REPORT_FILE, report_bytes(results, trace))
    return results, trace


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    committed_results = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    results, trace = build_documents(committed_results["preoracle_commit"])
    if RESULT_FILE.read_bytes() != canonical_bytes(results):
        raise ValueError("V3 adjudication results do not reproduce")
    if ORACLE_TRACE_FILE.read_bytes() != canonical_bytes(trace):
        raise ValueError("V3 oracle trace does not reproduce")
    if REPORT_FILE.read_bytes() != report_bytes(results, trace):
        raise ValueError("V3 result report does not reproduce")
    return results, trace


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    results, trace = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "decision": results["decision"],
                "results_sha256": sha256_file(RESULT_FILE),
                "summary": results["summary"],
                "trace_sha256": sha256_file(ORACLE_TRACE_FILE),
            },
            sort_keys=True,
        )
    )
    return 0 if results["decision"] == VERIFIED else 1


if __name__ == "__main__":
    raise SystemExit(main())
