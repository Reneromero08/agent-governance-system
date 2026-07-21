from __future__ import annotations

import argparse
import ast
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
FREEZE_SOURCE = PACKAGE_DIR / "freeze_builder.py"
FREEZE_FILE = PACKAGE_DIR / "SUCCESSOR_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "V2_BATCH_CUSTODY.json"
NO_SMUGGLE_FILE = PACKAGE_DIR / "NO_SMUGGLE_PROOF.json"
CONTRACT_FILE = PACKAGE_DIR / "V2_PREORACLE_CONTRACT.json"
EVIDENCE_FILE = PACKAGE_DIR / "V2_PREORACLE_EVIDENCE.json"
TRACE_FILE = PACKAGE_DIR / "V2_PREORACLE_TRACE.json"
SEAL_FILE = PACKAGE_DIR / "V2_PREORACLE_SEAL.json"

FREEZE_COMMIT = "9b9348064eaaffece01d5a1d7848d613a24857f5"
EXPECTED_MACHINE_SHA256 = (
    "c20f2cd4068ca32528bc52671793bca04d897456944e33ad8571083428c48930"
)
EXPECTED_BATCH_SHA256 = (
    "4d973f7b6015fa6d9cf201dab249832610334f394ae262797516c3cf27357dbe"
)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v2 = load_module(MACHINE_SOURCE, "catcas_v2_preoracle_machine")
freezer = load_module(FREEZE_SOURCE, "catcas_v2_preoracle_freeze")


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


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def source_absence_proof() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {"exact_oracle", "optimum_states", "oracle_document"}
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            findings.append(node.func.id)
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
            findings.append(node.func.attr)
    return {
        "forbidden_oracle_call_findings": sorted(findings),
        "oracle_call_count": len(findings),
        "pass": not findings,
        "source_sha256": sha256_bytes(source.encode("utf-8")),
    }


def frozen_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    batch = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
    no_smuggle = json.loads(NO_SMUGGLE_FILE.read_text(encoding="utf-8"))
    reproduced_freeze = freezer.verify()
    if freeze != reproduced_freeze:
        raise ValueError("frozen successor document does not reproduce")
    if freeze["machine"]["machine_sha256"] != EXPECTED_MACHINE_SHA256:
        raise ValueError("frozen machine identity changed")
    if batch["ordered_batch_sha256"] != EXPECTED_BATCH_SHA256:
        raise ValueError("frozen batch identity changed")
    if len(batch["ordered_instances"]) != 32:
        raise ValueError("frozen batch size changed")
    if not no_smuggle["pass"]:
        raise ValueError("frozen no-smuggle proof did not pass")
    return freeze, batch, no_smuggle


def array_seal(array: np.ndarray) -> dict[str, Any]:
    canonical = np.ascontiguousarray(array, dtype="<c16")
    payload = canonical.tobytes(order="C")
    return {
        "bytes": len(payload),
        "dtype": "little_endian_complex128",
        "sha256": sha256_bytes(payload),
        "shape": list(canonical.shape),
    }


def native_seal(execution: Any) -> dict[str, Any]:
    displaced = array_seal(execution.displaced)
    history = array_seal(execution.operator_history)
    query = array_seal(execution.query_frames)
    root = sha256_bytes(canonical_bytes({
        "displaced": displaced,
        "operator_history": history,
        "query_frames": query,
    }))
    return {
        "displaced_waveform": displaced,
        "native_root_sha256": root,
        "operator_history": history,
        "query_frames": query,
    }


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
            "schema": "catalytic_waveform_ising_v2_preoracle_trace_v1",
        }


def contract_document() -> dict[str, Any]:
    freeze, batch, no_smuggle = frozen_documents()
    absence = source_absence_proof()
    if not absence["pass"]:
        raise ValueError("pre-oracle source contains an oracle call")
    return {
        "batch_size": len(batch["ordered_instances"]),
        "claim_ceiling": v2.CLAIM_CEILING,
        "freeze_commit": FREEZE_COMMIT,
        "machine_sha256": freeze["machine"]["machine_sha256"],
        "native_no_smuggle_proof_sha256": sha256_file(NO_SMUGGLE_FILE),
        "oracle_boundary": (
            "after the exact full pre-oracle evidence and trace roots are committed "
            "and pushed remotely"
        ),
        "ordered_batch_sha256": batch["ordered_batch_sha256"],
        "preoracle_source_absence_proof": absence,
        "promotion_criterion": batch["promotion_criterion"],
        "reuse_law": (
            "rerun the same frozen J,h computation from the exact restored carrier, "
            "compare the complete boundary, and restore again"
        ),
        "schema": "catalytic_waveform_ising_v2_preoracle_contract_v1",
        "strict_control_law": freeze["machine"]["strict_control_law"],
        "thresholds": freeze["machine"]["acceptance_thresholds"],
        "transitive_native_path": no_smuggle["actual_transitive_native_path"],
    }


def execute_instance(
    record: dict[str, Any], ledger: EventLedger
) -> dict[str, Any]:
    index = int(record["index"])
    coupling, field = v2.validate_problem(
        np.asarray(record["coupling_matrix_J"], dtype=np.float64),
        np.asarray(record["field_vector_h"], dtype=np.float64),
    )
    carrier = v2.r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], v2.SITE_COUNT, axis=0)

    execution = v2.execute_native_cycle(borrowed, coupling, field)
    first_native = native_seal(execution)
    ledger.add(f"instance_{index:02d}_native_waveform_and_history_sealed", first_native)

    latch = v2.project_boundary(execution, f"v2_batch_{index:02d}")
    raw_energy = v2.ising_energy(latch.raw_spins, coupling, field)
    computed_residual = v2.lock_residual(latch.phases)
    boundary = {
        "computed_lock_residual_rad": metric(computed_residual),
        "latch": latch.document(),
        "raw_spin_energy": metric(raw_energy),
    }
    boundary_digest = sha256_bytes(canonical_bytes(boundary))
    ledger.add(f"instance_{index:02d}_boundary_sealed_once", boundary)

    restored = v2.restore_carrier(execution, "correct")
    restoration_error = float(np.max(np.abs(restored - borrowed)))
    latch_persisted = sha256_bytes(canonical_bytes(boundary)) == boundary_digest
    restoration = {
        "latch_persisted_outside_reversal": latch_persisted,
        "max_abs_error": metric(restoration_error),
    }
    ledger.add(f"instance_{index:02d}_carrier_restored", restoration)

    reuse_execution = v2.execute_native_cycle(restored, coupling, field)
    reuse_native = native_seal(reuse_execution)
    ledger.add(f"instance_{index:02d}_restored_carrier_reuse_native_sealed", reuse_native)
    reuse_latch = v2.project_boundary(reuse_execution, f"v2_batch_{index:02d}_reuse")
    reuse_boundary = {
        "computed_lock_residual_rad": metric(v2.lock_residual(reuse_latch.phases)),
        "latch": reuse_latch.document(),
        "raw_spin_energy": metric(
            v2.ising_energy(reuse_latch.raw_spins, coupling, field)
        ),
    }
    ledger.add(f"instance_{index:02d}_restored_carrier_reuse_boundary_sealed", reuse_boundary)
    reuse_restored = v2.restore_carrier(reuse_execution, "correct")
    reuse_input_error = float(np.max(np.abs(reuse_execution.borrowed - restored)))
    reuse_restoration_error = float(np.max(np.abs(reuse_restored - restored)))
    reuse_native_match = reuse_native == first_native
    reuse_response_delta = float(
        np.linalg.norm(
            np.asarray(reuse_latch.responses, dtype=np.complex128)
            - np.asarray(latch.responses, dtype=np.complex128)
        )
    )
    reuse_raw_spins_match = reuse_latch.raw_spins == latch.raw_spins
    reuse_validity_match = reuse_latch.valid == latch.valid
    reuse = {
        "boundary_response_delta_l2": metric(reuse_response_delta),
        "input_max_abs_error": metric(reuse_input_error),
        "native_exactly_reproduced": reuse_native_match,
        "raw_spins_reproduced": reuse_raw_spins_match,
        "restoration_max_abs_error": metric(reuse_restoration_error),
        "validity_reproduced": reuse_validity_match,
    }
    ledger.add(f"instance_{index:02d}_restored_carrier_reuse_restored", reuse)

    controls = v2.run_strict_controls(
        borrowed, coupling, field, execution, latch, v2.DEFAULT_LAW
    )
    ledger.add(f"instance_{index:02d}_all_strict_controls_sealed", controls)

    integrity = {
        "all_strict_controls_pass": bool(controls["all_pass"]),
        "boundary_finite": all(
            np.isfinite(
                [
                    *latch.coherence,
                    *latch.phases,
                    raw_energy,
                    computed_residual,
                ]
            )
        ),
        "carrier_displaced": execution.displacement_l2 >= v2.DISPLACEMENT_MIN,
        "carrier_restored": restoration_error <= v2.RESTORATION_MAX,
        "latch_persisted_outside_reversal": latch_persisted,
        "reuse_raw_spins_reproduced": reuse_raw_spins_match,
        "reuse_carrier_restored": reuse_restoration_error <= v2.RESTORATION_MAX,
        "reuse_consumed_exact_restored_carrier": reuse_input_error == 0.0,
        "reuse_validity_reproduced": reuse_validity_match,
    }
    return {
        "boundary": boundary,
        "controls": controls,
        "coupling_matrix_J": coupling.tolist(),
        "field_vector_h": field.tolist(),
        "index": index,
        "integrity": integrity,
        "interpretable_preoracle": all(integrity.values()),
        "measurements": {
            "carrier_displacement_l2": metric(execution.displacement_l2),
            "restoration_max_abs_error": metric(restoration_error),
            "reuse_carrier_displacement_l2": metric(reuse_execution.displacement_l2),
            "reuse_input_max_abs_error": metric(reuse_input_error),
            "reuse_restoration_max_abs_error": metric(reuse_restoration_error),
        },
        "native_seal": first_native,
        "problem_sha256": record["problem_sha256"],
        "reuse": reuse,
    }


def build_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze, batch, no_smuggle = frozen_documents()
    absence = source_absence_proof()
    if not absence["pass"] or absence["oracle_call_count"] != 0:
        raise ValueError("pre-oracle execution source is not oracle-free")
    ledger = EventLedger([])
    ledger.add(
        "remote_freeze_commit_bound",
        {
            "freeze_commit": FREEZE_COMMIT,
            "machine_sha256": EXPECTED_MACHINE_SHA256,
            "ordered_batch_sha256": EXPECTED_BATCH_SHA256,
        },
    )
    ledger.add("preoracle_source_oracle_absence_verified", absence)
    ledger.add("transitive_native_no_smuggle_bound", no_smuggle)

    instances = [
        execute_instance(record, ledger) for record in batch["ordered_instances"]
    ]
    summary = {
        "all_instances_interpretable_preoracle": all(
            record["interpretable_preoracle"] for record in instances
        ),
        "batch_size": len(instances),
        "oracle_call_count": absence["oracle_call_count"],
        "restoration_success_count": sum(
            record["integrity"]["carrier_restored"] for record in instances
        ),
        "reuse_success_count": sum(
            record["integrity"]["reuse_consumed_exact_restored_carrier"]
            and record["integrity"]["reuse_carrier_restored"]
            and record["integrity"]["reuse_raw_spins_reproduced"]
            and record["integrity"]["reuse_validity_reproduced"]
            for record in instances
        ),
        "strict_all_controls_pass_count": sum(
            record["integrity"]["all_strict_controls_pass"] for record in instances
        ),
        "strict_removed_transform_pass_count": sum(
            record["controls"]["outcomes"]["removed_waveform_transform"]
            for record in instances
        ),
    }
    evidence = {
        "claim_ceiling": v2.CLAIM_CEILING,
        "freeze_commit": FREEZE_COMMIT,
        "instances": instances,
        "machine_sha256": freeze["machine"]["machine_sha256"],
        "native_no_smuggle_proof_sha256": sha256_file(NO_SMUGGLE_FILE),
        "ordered_batch_sha256": batch["ordered_batch_sha256"],
        "preoracle_source_absence_proof": absence,
        "schema": "catalytic_waveform_ising_v2_preoracle_evidence_v1",
        "summary": summary,
    }
    evidence_sha = sha256_bytes(canonical_bytes(evidence))
    ledger.add(
        "complete_preoracle_evidence_root_sealed",
        {
            "batch_size": len(instances),
            "preoracle_evidence_sha256": evidence_sha,
        },
    )
    trace = ledger.document()
    seal = {
        "batch_size": len(instances),
        "freeze_commit": FREEZE_COMMIT,
        "machine_sha256": EXPECTED_MACHINE_SHA256,
        "oracle_call_count": absence["oracle_call_count"],
        "ordered_batch_sha256": EXPECTED_BATCH_SHA256,
        "preoracle_evidence_bytes": len(canonical_bytes(evidence)),
        "preoracle_evidence_sha256": evidence_sha,
        "preoracle_trace_sha256": sha256_bytes(canonical_bytes(trace)),
        "schema": "catalytic_waveform_ising_v2_preoracle_seal_v1",
        "summary": summary,
    }
    return evidence, trace, seal


def build() -> dict[str, Any]:
    contract = contract_document()
    evidence, trace, seal = build_documents()
    write_atomic(CONTRACT_FILE, canonical_bytes(contract))
    write_atomic(EVIDENCE_FILE, canonical_bytes(evidence))
    write_atomic(TRACE_FILE, canonical_bytes(trace))
    write_atomic(SEAL_FILE, canonical_bytes(seal))
    return seal


def verify() -> dict[str, Any]:
    contract = contract_document()
    evidence, trace, seal = build_documents()
    expected = {
        CONTRACT_FILE: contract,
        EVIDENCE_FILE: evidence,
        TRACE_FILE: trace,
        SEAL_FILE: seal,
    }
    for path, document in expected.items():
        if path.read_bytes() != canonical_bytes(document):
            raise ValueError(f"pre-oracle artifact does not reproduce: {path.name}")
    return seal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    seal = build() if args.command == "build" else verify()
    print(json.dumps({
        "oracle_call_count": seal["oracle_call_count"],
        "preoracle_evidence_sha256": seal["preoracle_evidence_sha256"],
        "preoracle_trace_sha256": seal["preoracle_trace_sha256"],
        "status": "COMPLETE_PREORACLE_BATCH_ROOT_SEALED",
        "summary": seal["summary"],
    }, sort_keys=True))
    return 0 if seal["summary"]["all_instances_interpretable_preoracle"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
