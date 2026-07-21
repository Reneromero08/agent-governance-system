from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
GATE_SOURCE = PACKAGE_DIR / "stability_gate.py"
FREEZE_FILE = PACKAGE_DIR / "SUCCESSOR_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "STABILITY_BATCH_CUSTODY.json"
EVIDENCE_FILE = PACKAGE_DIR / "PREORACLE_EVIDENCE.json"
TRACE_FILE = PACKAGE_DIR / "PREORACLE_TRACE.json"
SEAL_FILE = PACKAGE_DIR / "PREORACLE_SEAL.json"
GATE_NO_SMUGGLE_FILE = PACKAGE_DIR / "NO_SMUGGLE_PROOF.json"
V2_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v2"
V2_NO_SMUGGLE_FILE = V2_DIR / "NO_SMUGGLE_PROOF.json"
RESULT_FILE = PACKAGE_DIR / "BATCH_RESULTS.json"
ORACLE_TRACE_FILE = PACKAGE_DIR / "ORACLE_TRACE.json"
REPORT_FILE = PACKAGE_DIR / "BATCH_REPORT.md"

FREEZE_COMMIT = "c7ca2059c5bd78b6791bf4fbc2b8d8a04d72c26e"
PREORACLE_COMMIT = "62df1f8f27bedd9a13263f026160e125bc6467a3"
EXPECTED_EVIDENCE_SHA256 = (
    "c50007ede18c2c47d2c2a019dc774509d1c9d96e003dbb21f58eef32ee099224"
)
EXPECTED_TRACE_SHA256 = (
    "3a6f73bd48e2368c99154a4d40c06b2852e4338b1fd94d9ac3588260e078d0fa"
)
VERIFIED = "CATALYTIC_WAVEFORM_ISING_V2_STABILITY_BATCH_VERIFIED"
PARTIAL = "CATALYTIC_WAVEFORM_ISING_V2_STABILITY_BATCH_PARTIAL"
NOT_ESTABLISHED = "CATALYTIC_WAVEFORM_ISING_V2_STABILITY_BATCH_NOT_ESTABLISHED"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


gate = load_module(GATE_SOURCE, "catcas_stability_oracle_gate")
v2 = gate.v2


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("distribution requires at least one value")
    return {
        "max": metric(np.max(array)),
        "mean": metric(np.mean(array)),
        "median": metric(np.median(array)),
        "min": metric(np.min(array)),
    }


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


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
            "schema": "catalytic_waveform_ising_v2_stability_oracle_trace_v1",
        }


def load_and_verify_custody() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    batch = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
    seal = json.loads(SEAL_FILE.read_text(encoding="utf-8"))
    if sha256_file(EVIDENCE_FILE) != EXPECTED_EVIDENCE_SHA256:
        raise RuntimeError("pre-oracle evidence hash mismatch")
    if sha256_file(TRACE_FILE) != EXPECTED_TRACE_SHA256:
        raise RuntimeError("pre-oracle trace hash mismatch")
    if seal["evidence_sha256"] != EXPECTED_EVIDENCE_SHA256:
        raise RuntimeError("pre-oracle seal does not bind evidence")
    if seal["trace_sha256"] != EXPECTED_TRACE_SHA256:
        raise RuntimeError("pre-oracle seal does not bind trace")
    if seal["oracle_call_count"] != 0 or evidence["oracle_opened"]:
        raise RuntimeError("oracle-order custody failed")
    if evidence["freeze_commit"] != FREEZE_COMMIT or seal["freeze_commit"] != FREEZE_COMMIT:
        raise RuntimeError("pre-oracle evidence does not bind the remote freeze commit")
    if evidence["batch_ordered_sha256"] != batch["ordered_batch_sha256"]:
        raise RuntimeError("pre-oracle evidence does not bind frozen batch")
    if len(evidence["instances"]) != 64 or len(batch["ordered_instances"]) != 64:
        raise RuntimeError("batch custody is incomplete")
    for record, problem in zip(evidence["instances"], batch["ordered_instances"]):
        if record["index"] != problem["index"] or record["problem_sha256"] != problem["problem_sha256"]:
            raise RuntimeError("instance order or identity changed")
    null_baseline_pass = all(
        record["stability_controls"][
            "disabling_gate_reproduces_nominal_acceptance"
        ]
        and record["acceptance"]["gate_disabled_acceptance"]
        == record["acceptance"]["nominal_acceptance"]
        for record in evidence["instances"]
    )
    if not null_baseline_pass:
        raise RuntimeError("gate-disabled null baseline failed")
    return freeze, batch, evidence, seal


def nominal_classification(accepted: bool, correct: bool) -> str:
    if accepted and correct:
        return "NOMINAL_ACCEPTED_CORRECT"
    if accepted and not correct:
        return "NOMINAL_ACCEPTED_INCORRECT"
    if not accepted and correct:
        return "NOMINAL_REJECTED_CORRECT"
    return "NOMINAL_REJECTED_INCORRECT"


def stability_classification(accepted: bool, correct: bool) -> str:
    if accepted and correct:
        return "STABILITY_ACCEPTED_CORRECT"
    if accepted and not correct:
        return "STABILITY_ACCEPTED_INCORRECT"
    if not accepted and correct:
        return "STABILITY_REJECTED_CORRECT"
    return "STABILITY_REJECTED_INCORRECT"


def adjudicate_instance(
    evidence_record: dict[str, Any],
    problem: dict[str, Any],
) -> dict[str, Any]:
    if evidence_record["uninterpretable"]:
        return {
            "index": evidence_record["index"],
            "problem_sha256": evidence_record["problem_sha256"],
            "classification": "UNINTERPRETABLE",
        }
    coupling = np.asarray(problem["coupling_matrix_J"], dtype=np.float64)
    field = np.asarray(problem["field_vector_h"], dtype=np.float64)
    raw_spins = tuple(
        int(value) for value in evidence_record["boundary"]["raw_spins"]
    )
    rows = v2.exact_oracle(coupling, field)
    optimum_energy = float(rows[0][0])
    optimum_states = sorted(
        state for energy, state in rows if abs(energy - optimum_energy) <= 1.0e-12
    )
    raw_energy = v2.ising_energy(raw_spins, coupling, field)
    raw_correct = raw_spins in optimum_states
    nominal_accepted = bool(evidence_record["acceptance"]["nominal_acceptance"])
    stability_accepted = bool(
        evidence_record["acceptance"]["stability_gated_acceptance"]
    )
    raw_after = tuple(
        int(value)
        for value in evidence_record["acceptance"]["raw_spins_after_stability_gate"]
    )
    if raw_after != raw_spins:
        raise RuntimeError("stability gate changed the raw result")
    non_unique = len(optimum_states) != 1
    return {
        "index": evidence_record["index"],
        "nominal_accepted": nominal_accepted,
        "nominal_classification": (
            "NON_UNIQUE_OPTIMUM"
            if non_unique
            else nominal_classification(nominal_accepted, raw_correct)
        ),
        "optimum_energy": metric(optimum_energy),
        "optimum_state_count": len(optimum_states),
        "optimum_states": [list(state) for state in optimum_states],
        "problem_sha256": evidence_record["problem_sha256"],
        "raw_energy": metric(raw_energy),
        "raw_matches_any_optimum": raw_correct,
        "raw_spins": list(raw_spins),
        "stability_accepted": stability_accepted,
        "stability_classification": (
            "NON_UNIQUE_OPTIMUM"
            if non_unique
            else stability_classification(stability_accepted, raw_correct)
        ),
        "stability_metrics": {
            "joint_instability_score": evidence_record["diagnostic"]["gate_decision"][
                "joint_instability_score"
            ],
            "late_max_phase_velocity_rad_per_step": evidence_record["diagnostic"][
                "gate_decision"
            ]["late_max_phase_velocity_rad_per_step"],
            "late_mean_response_drift_l2": evidence_record["diagnostic"][
                "gate_decision"
            ]["late_mean_response_drift_l2"],
        },
    }


def count_class(outcomes: Sequence[dict[str, Any]], key: str, value: str) -> int:
    return sum(record.get(key) == value for record in outcomes)


def metric_distributions(
    outcomes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"ALL_INTERPRETABLE": []}
    for record in outcomes:
        if record.get("classification") == "UNINTERPRETABLE":
            continue
        groups["ALL_INTERPRETABLE"].append(record)
        nominal = record["nominal_classification"]
        stability = record["stability_classification"]
        groups.setdefault(nominal, []).append(record)
        if stability != nominal:
            groups.setdefault(stability, []).append(record)
    result: dict[str, Any] = {}
    for name, records in groups.items():
        if not records:
            continue
        result[name] = {
            "count": len(records),
            "joint_instability_score": distribution(
                [record["stability_metrics"]["joint_instability_score"] for record in records]
            ),
            "late_max_phase_velocity_rad_per_step": distribution(
                [
                    record["stability_metrics"]["late_max_phase_velocity_rad_per_step"]
                    for record in records
                ]
            ),
            "late_mean_response_drift_l2": distribution(
                [
                    record["stability_metrics"]["late_mean_response_drift_l2"]
                    for record in records
                ]
            ),
        }
    return result


def build_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, batch, evidence, seal = load_and_verify_custody()
    ledger = EventLedger([])
    custody = {
        "freeze_commit": FREEZE_COMMIT,
        "preoracle_commit": PREORACLE_COMMIT,
        "preoracle_evidence_sha256": EXPECTED_EVIDENCE_SHA256,
        "preoracle_trace_sha256": EXPECTED_TRACE_SHA256,
        "remote_preoracle_required_before_this_source": True,
    }
    ledger.add("remote_preoracle_custody_verified", custody)
    outcomes: list[dict[str, Any]] = []
    for evidence_record, problem in zip(
        evidence["instances"], batch["ordered_instances"]
    ):
        outcome = adjudicate_instance(evidence_record, problem)
        outcomes.append(outcome)
        ledger.add(f"instance_{int(problem['index']):02d}_oracle_adjudicated", outcome)

    interpretable = [
        record for record in outcomes if record.get("classification") != "UNINTERPRETABLE"
    ]
    unique = [record for record in interpretable if record["optimum_state_count"] == 1]
    non_unique = [record for record in interpretable if record["optimum_state_count"] != 1]
    nominal_counts = {
        name: count_class(outcomes, "nominal_classification", name)
        for name in (
            "NOMINAL_ACCEPTED_CORRECT",
            "NOMINAL_ACCEPTED_INCORRECT",
            "NOMINAL_REJECTED_CORRECT",
            "NOMINAL_REJECTED_INCORRECT",
        )
    }
    stability_counts = {
        name: count_class(outcomes, "stability_classification", name)
        for name in (
            "STABILITY_ACCEPTED_CORRECT",
            "STABILITY_ACCEPTED_INCORRECT",
            "STABILITY_REJECTED_CORRECT",
            "STABILITY_REJECTED_INCORRECT",
        )
    }
    nominal_false = nominal_counts["NOMINAL_ACCEPTED_INCORRECT"]
    stability_false = stability_counts["STABILITY_ACCEPTED_INCORRECT"]
    false_reduction_count = nominal_false - stability_false
    false_reduction_rate = (
        false_reduction_count / nominal_false if nominal_false > 0 else None
    )
    nominal_correct = nominal_counts["NOMINAL_ACCEPTED_CORRECT"]
    stability_correct = stability_counts["STABILITY_ACCEPTED_CORRECT"]
    correct_retention = stability_correct / nominal_correct if nominal_correct > 0 else 0.0
    criterion = batch["promotion_criterion"]
    checks = {
        "accepted_correct_count_min": (
            stability_correct >= criterion["accepted_correct_count_min"]
        ),
        "accepted_correct_rate_among_unique_min": (
            stability_correct / len(unique)
            >= criterion["accepted_correct_rate_among_unique_min"]
        ),
        "accepted_incorrect_count_max": (
            stability_false <= criterion["accepted_incorrect_count_max"]
        ),
        "batch_size": len(outcomes) == criterion["batch_size_required"],
        "diagnostic_restoration_all_instances": (
            evidence["summary"]["diagnostic_restoration_success_count"] == 64
        ),
        "native_no_smuggle": json.loads(
            V2_NO_SMUGGLE_FILE.read_text(encoding="utf-8")
        )["pass"],
        "nominal_restoration_all_instances": (
            evidence["summary"]["nominal_restoration_success_count"] == 64
        ),
        "raw_results_unchanged": all(
            record["acceptance"]["raw_spins_before_stability_gate"]
            == record["acceptance"]["raw_spins_after_stability_gate"]
            for record in evidence["instances"]
        ),
        "reuse_all_instances": evidence["summary"]["reuse_success_count"] == 64,
        "stability_controls_all_instances": (
            evidence["summary"]["stability_control_success_count"] == 64
        ),
        "stability_gate_no_smuggle": json.loads(
            GATE_NO_SMUGGLE_FILE.read_text(encoding="utf-8")
        )["pass"],
        "strict_v2_controls_all_instances": (
            evidence["summary"]["strict_v2_control_success_count"] == 64
        ),
        "unique_optimum_instance_count_min": (
            len(unique) >= criterion["unique_optimum_instance_count_min"]
        ),
        "uninterpretable_count_max": (
            len(outcomes) - len(interpretable)
            <= criterion["uninterpretable_count_max"]
        ),
    }
    promotion_pass = all(checks.values())
    mechanism_intact = all(
        checks[name]
        for name in (
            "batch_size",
            "diagnostic_restoration_all_instances",
            "native_no_smuggle",
            "nominal_restoration_all_instances",
            "raw_results_unchanged",
            "reuse_all_instances",
            "stability_controls_all_instances",
            "stability_gate_no_smuggle",
            "strict_v2_controls_all_instances",
            "uninterpretable_count_max",
        )
    )
    meaningful_false_accept_reduction = false_reduction_count > 0
    if promotion_pass:
        decision = VERIFIED
    elif mechanism_intact and meaningful_false_accept_reduction:
        decision = PARTIAL
    else:
        decision = NOT_ESTABLISHED
    summary = {
        "batch_size": len(outcomes),
        "correct_result_retention_rate": metric(correct_retention),
        "decision": decision,
        "false_accept_reduction_count": false_reduction_count,
        "false_accept_reduction_rate": (
            None if false_reduction_rate is None else metric(false_reduction_rate)
        ),
        "meaningful_false_accept_reduction": meaningful_false_accept_reduction,
        "nominal_counts": nominal_counts,
        "non_unique_count": len(non_unique),
        "non_unique_raw_optimum_match_count": sum(
            record["raw_matches_any_optimum"] for record in non_unique
        ),
        "promotion_checks": checks,
        "promotion_pass": promotion_pass,
        "stability_counts": stability_counts,
        "stability_metric_distributions": metric_distributions(outcomes),
        "unique_count": len(unique),
        "uninterpretable_count": len(outcomes) - len(interpretable),
    }
    ledger.add("frozen_promotion_criterion_applied", summary)
    trace = ledger.document()
    results = {
        "claim_ceiling": freeze["claim_ceiling"],
        "custody": custody,
        "decision": decision,
        "discriminator_fingerprint": freeze["discriminator"]["discriminator_fingerprint"],
        "machine_fingerprint": freeze["machine"]["machine_fingerprint"],
        "outcomes": outcomes,
        "preoracle_seal_sha256": sha256_file(SEAL_FILE),
        "promotion_criterion": criterion,
        "schema": "catalytic_waveform_ising_v2_stability_batch_results_v1",
        "summary": summary,
    }
    return results, trace


def report_bytes(results: dict[str, Any], trace: dict[str, Any]) -> bytes:
    summary = results["summary"]
    nominal = summary["nominal_counts"]
    stability = summary["stability_counts"]
    lines = [
        "# V2 stability-gate 64-instance adjudication",
        "",
        f"Decision: `{results['decision']}`",
        "",
        f"Unique instances: {summary['unique_count']}",
        f"Non-unique instances: {summary['non_unique_count']}",
        f"Nominal accepted correct: {nominal['NOMINAL_ACCEPTED_CORRECT']}",
        f"Nominal accepted incorrect: {nominal['NOMINAL_ACCEPTED_INCORRECT']}",
        f"Stability accepted correct: {stability['STABILITY_ACCEPTED_CORRECT']}",
        f"Stability accepted incorrect: {stability['STABILITY_ACCEPTED_INCORRECT']}",
        f"False-accept reduction count: {summary['false_accept_reduction_count']}",
        f"Correct-result retention: {summary['correct_result_retention_rate']}",
        f"Promotion pass: {summary['promotion_pass']}",
        f"Oracle trace SHA-256: {sha256_bytes(canonical_bytes(trace))}",
        "",
        "The discriminator remained reject-only. No raw result was changed or selected.",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build() -> dict[str, Any]:
    results, trace = build_documents()
    write_atomic(RESULT_FILE, canonical_bytes(results))
    write_atomic(ORACLE_TRACE_FILE, canonical_bytes(trace))
    write_atomic(REPORT_FILE, report_bytes(results, trace))
    return {
        "decision": results["decision"],
        "results_sha256": sha256_file(RESULT_FILE),
        "summary": results["summary"],
        "trace_sha256": sha256_file(ORACLE_TRACE_FILE),
    }


def verify() -> dict[str, Any]:
    results, trace = build_documents()
    expected = {
        RESULT_FILE: canonical_bytes(results),
        ORACLE_TRACE_FILE: canonical_bytes(trace),
        REPORT_FILE: report_bytes(results, trace),
    }
    for path, payload in expected.items():
        if path.read_bytes() != payload:
            raise RuntimeError(f"oracle adjudication does not reproduce: {path.name}")
    return {
        "decision": results["decision"],
        "results_sha256": sha256_bytes(expected[RESULT_FILE]),
        "summary": results["summary"],
        "trace_sha256": sha256_bytes(expected[ORACLE_TRACE_FILE]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    result = build() if args.mode == "build" else verify()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
