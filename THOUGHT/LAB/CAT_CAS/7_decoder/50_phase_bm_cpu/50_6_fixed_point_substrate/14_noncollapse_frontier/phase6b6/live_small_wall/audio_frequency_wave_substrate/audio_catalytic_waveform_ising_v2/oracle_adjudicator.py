from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
MACHINE_SOURCE = PACKAGE_DIR / "successor_machine.py"
FREEZE_FILE = PACKAGE_DIR / "SUCCESSOR_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "V2_BATCH_CUSTODY.json"
NO_SMUGGLE_FILE = PACKAGE_DIR / "NO_SMUGGLE_PROOF.json"
PREORACLE_EVIDENCE_FILE = PACKAGE_DIR / "V2_PREORACLE_EVIDENCE.json"
PREORACLE_TRACE_FILE = PACKAGE_DIR / "V2_PREORACLE_TRACE.json"
PREORACLE_SEAL_FILE = PACKAGE_DIR / "V2_PREORACLE_SEAL.json"
RESULT_FILE = PACKAGE_DIR / "V2_BATCH_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "V2_BATCH_REPORT.md"
ORACLE_TRACE_FILE = PACKAGE_DIR / "V2_ORACLE_TRACE.json"

FREEZE_COMMIT = "9b9348064eaaffece01d5a1d7848d613a24857f5"
PREORACLE_REMOTE_COMMIT = "f7249dd256d35ba2ff1e66928f6c04729138eaf9"
EXPECTED_MACHINE_SHA256 = (
    "c20f2cd4068ca32528bc52671793bca04d897456944e33ad8571083428c48930"
)
EXPECTED_BATCH_SHA256 = (
    "4d973f7b6015fa6d9cf201dab249832610334f394ae262797516c3cf27357dbe"
)
EXPECTED_PREORACLE_EVIDENCE_SHA256 = (
    "b698d5ca036242956a395bcdabff06d5b87f115e1fb4a23695be7be04f43353f"
)
EXPECTED_PREORACLE_TRACE_SHA256 = (
    "9b8492b925536fafcad08a48207e4f749152d390d2122fa9936712b7ab9c74eb"
)

VERIFIED = "CATALYTIC_WAVEFORM_ISING_V2_BATCH_GENERALIZATION_VERIFIED"
PARTIAL = "CATALYTIC_WAVEFORM_ISING_V2_BATCH_GENERALIZATION_PARTIAL"
NOT_ESTABLISHED = "CATALYTIC_WAVEFORM_ISING_V2_BATCH_GENERALIZATION_NOT_ESTABLISHED"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v2 = load_module(MACHINE_SOURCE, "catcas_v2_oracle_machine")


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
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires finite values")
    return {
        "max": metric(np.max(array)),
        "mean": metric(np.mean(array)),
        "median": metric(np.median(array)),
        "min": metric(np.min(array)),
    }


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


@dataclass
class EventLedger:
    events: list[dict[str, Any]]
    previous_hash: str = "0" * 64

    def add(self, name: str, payload: Any) -> None:
        record = {
            "name": name,
            "payload_sha256": sha256_bytes(canonical_bytes(payload)),
            "previous_event_sha256": self.previous_hash,
            "sequence": len(self.events),
        }
        record["event_sha256"] = sha256_bytes(canonical_bytes(record))
        self.events.append(record)
        self.previous_hash = record["event_sha256"]

    def document(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "events": self.events,
            "final_event_sha256": self.previous_hash,
            "schema": "catalytic_waveform_ising_v2_oracle_trace_v1",
        }


def load_and_verify_custody() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    batch = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
    no_smuggle = json.loads(NO_SMUGGLE_FILE.read_text(encoding="utf-8"))
    evidence = json.loads(PREORACLE_EVIDENCE_FILE.read_text(encoding="utf-8"))
    seal = json.loads(PREORACLE_SEAL_FILE.read_text(encoding="utf-8"))
    if freeze["machine"]["machine_sha256"] != EXPECTED_MACHINE_SHA256:
        raise ValueError("machine identity mismatch")
    if batch["ordered_batch_sha256"] != EXPECTED_BATCH_SHA256:
        raise ValueError("batch identity mismatch")
    if sha256_file(PREORACLE_EVIDENCE_FILE) != EXPECTED_PREORACLE_EVIDENCE_SHA256:
        raise ValueError("pre-oracle evidence bytes changed")
    if sha256_file(PREORACLE_TRACE_FILE) != EXPECTED_PREORACLE_TRACE_SHA256:
        raise ValueError("pre-oracle trace bytes changed")
    if seal["preoracle_evidence_sha256"] != EXPECTED_PREORACLE_EVIDENCE_SHA256:
        raise ValueError("pre-oracle evidence seal mismatch")
    if seal["preoracle_trace_sha256"] != EXPECTED_PREORACLE_TRACE_SHA256:
        raise ValueError("pre-oracle trace seal mismatch")
    if seal["oracle_call_count"] != 0:
        raise ValueError("pre-oracle seal reports an oracle call")
    if not no_smuggle["pass"]:
        raise ValueError("native no-smuggle proof failed")
    if len(batch["ordered_instances"]) != 32 or len(evidence["instances"]) != 32:
        raise ValueError("batch/evidence cardinality mismatch")
    for frozen, executed in zip(
        batch["ordered_instances"], evidence["instances"], strict=True
    ):
        if frozen["index"] != executed["index"]:
            raise ValueError("instance order changed")
        if frozen["problem_sha256"] != executed["problem_sha256"]:
            raise ValueError("instance identity changed")
        if frozen["coupling_matrix_J"] != executed["coupling_matrix_J"]:
            raise ValueError("executed J differs from freeze")
        if frozen["field_vector_h"] != executed["field_vector_h"]:
            raise ValueError("executed h differs from freeze")
    return freeze, batch, no_smuggle, evidence, seal


def adjudicate_instance(
    record: dict[str, Any], ledger: EventLedger
) -> dict[str, Any]:
    index = int(record["index"])
    coupling, field = v2.validate_problem(
        np.asarray(record["coupling_matrix_J"], dtype=np.float64),
        np.asarray(record["field_vector_h"], dtype=np.float64),
    )
    latch = record["boundary"]["latch"]
    raw_spins = tuple(int(value) for value in latch["raw_spin_shadow"])
    recomputed_raw_energy = v2.ising_energy(raw_spins, coupling, field)
    sealed_raw_energy = float(record["boundary"]["raw_spin_energy"])
    energy_reproduces = abs(recomputed_raw_energy - sealed_raw_energy) <= 1.0e-12

    oracle_rows = v2.exact_oracle(coupling, field)
    optimum_energy = float(oracle_rows[0][0])
    optimum_states = sorted(
        spins
        for energy, spins in oracle_rows
        if abs(energy - optimum_energy) <= 1.0e-12
    )
    oracle_document = {
        "index": index,
        "optimum_energy": metric(optimum_energy),
        "optimum_states": [list(spins) for spins in optimum_states],
        "rows": [
            {"energy": metric(energy), "spins": list(spins)}
            for energy, spins in oracle_rows
        ],
    }
    ledger.add(f"instance_{index:02d}_bounded_32_state_oracle_opened", oracle_document)

    raw_match = raw_spins in set(optimum_states)
    unique = len(optimum_states) == 1
    interpretable = bool(record["interpretable_preoracle"] and energy_reproduces)
    accepted = bool(latch["valid"])
    if not interpretable:
        classification = "UNINTERPRETABLE"
    elif not unique:
        classification = "NON_UNIQUE_OPTIMUM"
    elif accepted and raw_match:
        classification = "ACCEPTED_CORRECT"
    elif accepted and not raw_match:
        classification = "ACCEPTED_INCORRECT"
    elif raw_match:
        classification = "RAW_CORRECT_BELOW_GATE"
    else:
        classification = "RAW_INCORRECT"
    result = {
        "accepted": accepted,
        "classification": classification,
        "coherence": latch["coherence"],
        "energy_reproduces": energy_reproduces,
        "index": index,
        "lock_residual_rad": record["boundary"]["computed_lock_residual_rad"],
        "optimum_energy": metric(optimum_energy),
        "optimum_state_count": len(optimum_states),
        "optimum_states": [list(spins) for spins in optimum_states],
        "problem_sha256": record["problem_sha256"],
        "raw_energy": metric(recomputed_raw_energy),
        "raw_matches_any_optimum": raw_match,
        "raw_spins": list(raw_spins),
        "unique_optimum": unique,
    }
    ledger.add(f"instance_{index:02d}_adjudication_sealed", result)
    return result


def build_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, batch, no_smuggle, evidence, seal = load_and_verify_custody()
    ledger = EventLedger([])
    ledger.add(
        "remote_preoracle_commit_verified_before_oracle",
        {
            "preoracle_remote_commit": PREORACLE_REMOTE_COMMIT,
            "freeze_commit": FREEZE_COMMIT,
        },
    )
    ledger.add(
        "committed_preoracle_roots_verified_before_oracle",
        {
            "preoracle_evidence_sha256": EXPECTED_PREORACLE_EVIDENCE_SHA256,
            "preoracle_trace_sha256": EXPECTED_PREORACLE_TRACE_SHA256,
            "oracle_call_count": seal["oracle_call_count"],
        },
    )
    ledger.add(
        "bounded_oracle_boundary_opened",
        {
            "instance_count": len(evidence["instances"]),
            "states_per_instance": 32,
        },
    )
    outcomes = [
        adjudicate_instance(record, ledger) for record in evidence["instances"]
    ]

    counts = {
        classification: sum(
            record["classification"] == classification for record in outcomes
        )
        for classification in (
            "ACCEPTED_CORRECT",
            "ACCEPTED_INCORRECT",
            "RAW_CORRECT_BELOW_GATE",
            "RAW_INCORRECT",
            "NON_UNIQUE_OPTIMUM",
            "UNINTERPRETABLE",
        )
    }
    unique = [record for record in outcomes if record["unique_optimum"]]
    non_unique = [record for record in outcomes if not record["unique_optimum"]]
    accepted_correct_rate = (
        counts["ACCEPTED_CORRECT"] / len(unique) if unique else 0.0
    )
    criterion = batch["promotion_criterion"]
    promotion_checks = {
        "accepted_correct_count_min": (
            counts["ACCEPTED_CORRECT"] >= criterion["accepted_correct_count_min"]
        ),
        "accepted_correct_rate_min": (
            accepted_correct_rate
            >= criterion["accepted_correct_rate_among_unique_min"]
        ),
        "accepted_incorrect_max": (
            counts["ACCEPTED_INCORRECT"]
            <= criterion["accepted_incorrect_count_max"]
        ),
        "all_other_strict_controls": (
            seal["summary"]["strict_all_controls_pass_count"] == 32
        ),
        "batch_size": len(outcomes) == criterion["batch_size_required"],
        "native_no_smuggle": bool(no_smuggle["pass"]),
        "restoration_all": seal["summary"]["restoration_success_count"] == 32,
        "reuse_all": seal["summary"]["reuse_success_count"] == 32,
        "strict_removed_transform_all": (
            seal["summary"]["strict_removed_transform_pass_count"] == 32
        ),
        "unique_optimum_min": (
            len(unique) >= criterion["unique_optimum_instance_count_min"]
        ),
        "uninterpretable_max": (
            counts["UNINTERPRETABLE"] <= criterion["uninterpretable_count_max"]
        ),
    }
    promotion_pass = all(promotion_checks.values())
    mechanical_core = all(
        promotion_checks[name]
        for name in (
            "all_other_strict_controls",
            "batch_size",
            "native_no_smuggle",
            "restoration_all",
            "reuse_all",
            "strict_removed_transform_all",
            "uninterpretable_max",
        )
    )
    credible_unseen_result = (
        counts["ACCEPTED_CORRECT"] > 0
        or counts["RAW_CORRECT_BELOW_GATE"] > 0
        or any(record["raw_matches_any_optimum"] for record in non_unique)
    )
    if promotion_pass:
        decision = VERIFIED
    elif mechanical_core and credible_unseen_result:
        decision = PARTIAL
    else:
        decision = NOT_ESTABLISHED

    all_coherence = [
        float(value) for record in outcomes for value in record["coherence"]
    ]
    minimum_coherence = [min(record["coherence"]) for record in outcomes]
    residuals = [float(record["lock_residual_rad"]) for record in outcomes]
    restoration = [
        float(record["measurements"]["restoration_max_abs_error"])
        for record in evidence["instances"]
    ]
    reuse_restoration = [
        float(record["measurements"]["reuse_restoration_max_abs_error"])
        for record in evidence["instances"]
    ]
    reuse_response = [
        float(record["reuse"]["boundary_response_delta_l2"])
        for record in evidence["instances"]
    ]
    samplewise = [
        float(record["controls"]["measurements"]["samplewise_non_rank_one_residual"])
        for record in evidence["instances"]
    ]
    transform_history = [
        float(record["controls"]["measurements"]["history_deltas_l2"]["no_transform"])
        for record in evidence["instances"]
    ]
    transform_response = [
        float(record["controls"]["measurements"]["response_deltas_l2"]["no_transform"])
        for record in evidence["instances"]
    ]
    summary = {
        "accepted_correct_rate_among_unique": metric(accepted_correct_rate),
        "batch_size": len(outcomes),
        "classification_counts": counts,
        "coherence_all_sites_distribution": distribution(all_coherence),
        "decision": decision,
        "minimum_coherence_per_instance_distribution": distribution(minimum_coherence),
        "non_unique_raw_match_count": sum(
            record["raw_matches_any_optimum"] for record in non_unique
        ),
        "non_unique_total": len(non_unique),
        "promotion_checks": promotion_checks,
        "promotion_pass": promotion_pass,
        "raw_optimum_agreement_unique": sum(
            record["raw_matches_any_optimum"] for record in unique
        ),
        "residual_rad_distribution": distribution(residuals),
        "restoration_error_distribution": distribution(restoration),
        "reuse_response_delta_distribution": distribution(reuse_response),
        "reuse_restoration_error_distribution": distribution(reuse_restoration),
        "samplewise_non_rank_one_distribution": distribution(samplewise),
        "strict_removed_transform_history_delta_distribution": distribution(
            transform_history
        ),
        "strict_removed_transform_response_delta_distribution": distribution(
            transform_response
        ),
        "unique_optimum_total": len(unique),
    }
    results = {
        "claim_ceiling": v2.CLAIM_CEILING,
        "decision": decision,
        "freeze_commit": FREEZE_COMMIT,
        "machine_sha256": freeze["machine"]["machine_sha256"],
        "oracle_order": {
            "oracle_boundary_open_sequence": 2,
            "preoracle_remote_commit": PREORACLE_REMOTE_COMMIT,
            "preoracle_remote_commit_verified_sequence": 0,
            "preoracle_roots_verified_sequence": 1,
        },
        "ordered_batch_sha256": batch["ordered_batch_sha256"],
        "outcomes": outcomes,
        "preoracle_evidence_sha256": EXPECTED_PREORACLE_EVIDENCE_SHA256,
        "preoracle_trace_sha256": EXPECTED_PREORACLE_TRACE_SHA256,
        "promotion_criterion": criterion,
        "schema": "catalytic_waveform_ising_v2_batch_result_v1",
        "summary": summary,
    }
    results_sha = sha256_bytes(canonical_bytes(results))
    ledger.add(
        "aggregate_adjudication_sealed",
        {
            "decision": decision,
            "results_sha256": results_sha,
            "summary": summary,
        },
    )
    return results, ledger.document()


def report_bytes(results: dict[str, Any], trace: dict[str, Any]) -> bytes:
    summary = results["summary"]
    counts = summary["classification_counts"]
    failed = [name for name, passed in summary["promotion_checks"].items() if not passed]
    lines = [
        "# Catalytic Waveform-Ising V2 Frozen Batch Generalization",
        "",
        f"Decision: `{results['decision']}`",
        "",
        f"Freeze commit: `{results['freeze_commit']}`",
        f"Pre-oracle remote commit: `{results['oracle_order']['preoracle_remote_commit']}`",
        f"Machine SHA-256: `{results['machine_sha256']}`",
        f"Ordered batch SHA-256: `{results['ordered_batch_sha256']}`",
        f"Oracle trace SHA-256: `{sha256_bytes(canonical_bytes(trace))}`",
        "",
        "```text",
        f"batch size                    {summary['batch_size']}",
        f"unique optima                 {summary['unique_optimum_total']}",
        f"accepted correct              {counts['ACCEPTED_CORRECT']}",
        f"accepted incorrect            {counts['ACCEPTED_INCORRECT']}",
        f"raw correct below gate        {counts['RAW_CORRECT_BELOW_GATE']}",
        f"raw incorrect                 {counts['RAW_INCORRECT']}",
        f"non-unique                    {counts['NON_UNIQUE_OPTIMUM']}",
        f"uninterpretable               {counts['UNINTERPRETABLE']}",
        f"raw unique optimum agreement  {summary['raw_optimum_agreement_unique']} / {summary['unique_optimum_total']}",
        f"non-unique raw matches        {summary['non_unique_raw_match_count']} / {summary['non_unique_total']}",
        "```",
        "",
        f"Failed promotion checks: `{', '.join(failed) if failed else 'none'}`.",
        "",
        "This is a bounded deterministic software reference. It does not establish "
        "computational advantage, scale, physical computation, hardware bit replacement, "
        "physical persistence/restoration, or a Wall crossing.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def build() -> dict[str, Any]:
    results, trace = build_documents()
    write_atomic(RESULT_FILE, canonical_bytes(results))
    write_atomic(ORACLE_TRACE_FILE, canonical_bytes(trace))
    write_atomic(REPORT_FILE, report_bytes(results, trace))
    return results


def verify() -> dict[str, Any]:
    results, trace = build_documents()
    expected = {
        RESULT_FILE: canonical_bytes(results),
        ORACLE_TRACE_FILE: canonical_bytes(trace),
        REPORT_FILE: report_bytes(results, trace),
    }
    for path, payload in expected.items():
        if path.read_bytes() != payload:
            raise ValueError(f"adjudication artifact does not reproduce: {path.name}")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    results = build() if args.command == "build" else verify()
    print(json.dumps({
        "decision": results["decision"],
        "results_sha256": sha256_file(RESULT_FILE),
        "summary": results["summary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
