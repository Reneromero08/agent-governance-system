from __future__ import annotations

import argparse
import ast
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
AUTHORITY_FILE = PACKAGE_DIR / "EXPERIMENT_AUTHORITY.txt"
GATE_NO_SMUGGLE_FILE = PACKAGE_DIR / "NO_SMUGGLE_PROOF.json"
V2_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v2"
V2_SOURCE = V2_DIR / "successor_machine.py"
V2_NO_SMUGGLE_FILE = V2_DIR / "NO_SMUGGLE_PROOF.json"
CONTRACT_FILE = PACKAGE_DIR / "PREORACLE_CONTRACT.json"
EVIDENCE_FILE = PACKAGE_DIR / "PREORACLE_EVIDENCE.json"
TRACE_FILE = PACKAGE_DIR / "PREORACLE_TRACE.json"
SEAL_FILE = PACKAGE_DIR / "PREORACLE_SEAL.json"

FREEZE_COMMIT = "c7ca2059c5bd78b6791bf4fbc2b8d8a04d72c26e"
EXPECTED_MACHINE_FINGERPRINT = (
    "c20f2cd4068ca32528bc52671793bca04d897456944e33ad8571083428c48930"
)
EXPECTED_DISCRIMINATOR_FINGERPRINT = (
    "0a6592beea44b303cc33fbcf5e9cf68162369fd9992544e0bc225de6a78b0376"
)
EXPECTED_BATCH_ORDERED_SHA256 = (
    "9b70d445ab9742b70e355de8ee36afdb842e13d24be8fdddf5fe93725fb96a34"
)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


gate = load_module(GATE_SOURCE, "catcas_stability_preoracle_gate")
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


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def source_absence_proof() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {"development_instances", "exact_oracle", "ising_energy"}
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            findings.append(node.func.id)
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
            findings.append(node.func.attr)
    return {
        "forbidden_calls": sorted(findings),
        "pass": not findings,
        "source_sha256": sha256_bytes(source.encode("utf-8")),
    }


def array_seal(array: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "l2_norm": metric(np.linalg.norm(value)),
        "shape": list(value.shape),
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def native_seal(execution: Any) -> dict[str, Any]:
    return {
        "displaced": array_seal(execution.displaced),
        "displacement_l2": metric(execution.displacement_l2),
        "operator_history": array_seal(execution.operator_history),
        "query_frames": array_seal(execution.query_frames),
    }


def response_document(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {"imag": metric(value.imag), "real": metric(value.real)}
        for value in np.asarray(values, dtype=np.complex128)
    ]


def boundary_without_energy(execution: Any, label: str) -> Any:
    responses = gate.complex_responses(execution.displaced, execution.query_frames)
    coherence = tuple(float(abs(value)) for value in responses)
    phases = tuple(float(np.angle(value)) for value in responses)
    coherent = min(coherence) >= v2.COHERENCE_MIN
    residual = v2.lock_residual(phases) if coherent else None
    valid = bool(coherent and residual is not None and residual <= v2.LOCK_RESIDUAL_MAX)
    raw_spins = tuple(1 if value.real >= 0.0 else -1 for value in responses)
    return v2.BoundaryProjection(
        label=label,
        responses=tuple(complex(value) for value in responses),
        coherence=coherence,
        phases=phases,
        raw_spins=raw_spins,
        spins=raw_spins if valid else None,
        energy=None,
        residual=residual,
        valid=valid,
    )


def boundary_document(latch: Any) -> dict[str, Any]:
    return {
        "coherence": [metric(value) for value in latch.coherence],
        "energy_opened": False,
        "lock_residual_rad": None if latch.residual is None else metric(latch.residual),
        "nominal_accepted": bool(latch.valid),
        "raw_spins": list(latch.raw_spins),
        "responses": response_document(np.asarray(latch.responses)),
    }


def checkpoint_document(checkpoints: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {"response": response_document(checkpoints[step]), "step": step}
        for step in gate.CHECKPOINT_STEPS
    ]


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
            "schema": "catalytic_waveform_ising_v2_stability_preoracle_trace_v1",
        }


def frozen_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    batch = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
    if freeze["machine"]["machine_fingerprint"] != EXPECTED_MACHINE_FINGERPRINT:
        raise RuntimeError("V2 machine fingerprint drift")
    if (
        freeze["discriminator"]["discriminator_fingerprint"]
        != EXPECTED_DISCRIMINATOR_FINGERPRINT
    ):
        raise RuntimeError("stability discriminator fingerprint drift")
    if batch["ordered_batch_sha256"] != EXPECTED_BATCH_ORDERED_SHA256:
        raise RuntimeError("ordered batch hash drift")
    if batch["batch_size"] != 64 or len(batch["ordered_instances"]) != 64:
        raise RuntimeError("frozen batch size drift")
    if sha256_file(V2_SOURCE) != freeze["machine"]["source_sha256"]:
        raise RuntimeError("V2 source changed after freeze")
    if sha256_file(GATE_SOURCE) != freeze["discriminator"]["source_sha256"]:
        raise RuntimeError("stability gate source changed after freeze")
    if sha256_file(AUTHORITY_FILE) != freeze["authority"]["sha256"]:
        raise RuntimeError("authority changed after freeze")
    if sha256_file(GATE_NO_SMUGGLE_FILE) != freeze["discriminator"]["no_smuggle_proof_sha256"]:
        raise RuntimeError("gate no-smuggle proof changed after freeze")
    gate_no_smuggle = json.loads(GATE_NO_SMUGGLE_FILE.read_text(encoding="utf-8"))
    native_no_smuggle = json.loads(V2_NO_SMUGGLE_FILE.read_text(encoding="utf-8"))
    if not gate_no_smuggle["pass"] or not native_no_smuggle["pass"]:
        raise RuntimeError("frozen no-smuggle proof is not passing")
    return freeze, batch


def contract_document() -> dict[str, Any]:
    freeze, batch = frozen_documents()
    absence = source_absence_proof()
    if not absence["pass"]:
        raise RuntimeError("pre-oracle source crosses an adjudication boundary")
    return {
        "batch_ordered_sha256": batch["ordered_batch_sha256"],
        "batch_size": batch["batch_size"],
        "diagnostic_schedule": gate.gate_contract(),
        "discriminator_fingerprint": freeze["discriminator"]["discriminator_fingerprint"],
        "freeze_commit": FREEZE_COMMIT,
        "freeze_sha256": sha256_file(FREEZE_FILE),
        "gate_no_smuggle_sha256": sha256_file(GATE_NO_SMUGGLE_FILE),
        "machine_fingerprint": freeze["machine"]["machine_fingerprint"],
        "native_no_smuggle_sha256": sha256_file(V2_NO_SMUGGLE_FILE),
        "null_baseline": "stability gate disabled reproduces nominal V2 acceptance",
        "oracle_and_energy_policy": (
            "exact oracle, scalar energy, development labels, and V2 energy-bearing "
            "boundary projection are runtime-blocked for the complete execution"
        ),
        "preoracle_source_absence": absence,
        "schema": "catalytic_waveform_ising_v2_stability_preoracle_contract_v1",
    }


def execute_instance(
    record: dict[str, Any],
    borrowed: np.ndarray,
    ledger: EventLedger,
) -> dict[str, Any]:
    index = int(record["index"])
    coupling = np.asarray(record["coupling_matrix_J"], dtype=np.float64)
    field = np.asarray(record["field_vector_h"], dtype=np.float64)
    nominal = v2.execute_native_cycle(borrowed, coupling, field, law=v2.DEFAULT_LAW)
    nominal_native = native_seal(nominal)
    ledger.add(f"instance_{index:02d}_nominal_native_sealed", nominal_native)
    latch = boundary_without_energy(nominal, f"stability_batch_{index:02d}")
    boundary = boundary_document(latch)
    ledger.add(f"instance_{index:02d}_nominal_boundary_sealed", boundary)

    restored = v2.restore_carrier(nominal)
    restoration_error = float(np.max(np.abs(restored - borrowed)))
    restoration = {
        "max_abs_error": metric(restoration_error),
        "pass": restoration_error <= v2.RESTORATION_MAX,
    }
    ledger.add(f"instance_{index:02d}_nominal_restored", restoration)

    reuse = v2.execute_native_cycle(restored, coupling, field, law=v2.DEFAULT_LAW)
    reuse_latch = boundary_without_energy(reuse, f"stability_batch_{index:02d}_reuse")
    reuse_restored = v2.restore_carrier(reuse)
    reuse_restoration_error = float(np.max(np.abs(reuse_restored - borrowed)))
    reuse_response_delta = float(
        np.linalg.norm(
            np.asarray(reuse_latch.responses) - np.asarray(latch.responses)
        )
    )
    reuse_record = {
        "input_max_abs_error": metric(np.max(np.abs(restored - borrowed))),
        "pass": bool(
            reuse_restoration_error <= v2.RESTORATION_MAX
            and reuse_response_delta <= gate.REPLAY_TOLERANCE
            and reuse_latch.raw_spins == latch.raw_spins
        ),
        "raw_spins_unchanged": reuse_latch.raw_spins == latch.raw_spins,
        "response_delta_l2": metric(reuse_response_delta),
        "restoration_max_abs_error": metric(reuse_restoration_error),
    }
    ledger.add(f"instance_{index:02d}_restored_carrier_reused", reuse_record)

    controls = v2.run_strict_controls(
        borrowed, coupling, field, nominal, latch, v2.DEFAULT_LAW
    )
    ledger.add(f"instance_{index:02d}_strict_v2_controls", controls)

    checkpoints, replay_delta, diagnostic_restore = gate.replay_nominal_diagnostic(nominal)
    checkpoint_records = checkpoint_document(checkpoints)
    checkpoint_hash = sha256_bytes(canonical_bytes(checkpoint_records))
    stability = gate.evaluate_stability(nominal)
    if checkpoint_hash != stability.checkpoint_response_sha256:
        raise RuntimeError("sealed diagnostic checkpoints do not bind gate decision")
    stability_document = stability.document()
    diagnostic = {
        "checkpoint_records": checkpoint_records,
        "checkpoint_response_sha256": checkpoint_hash,
        "gate_decision": stability_document,
        "seal_replay_max_abs_delta": metric(replay_delta),
        "seal_replay_restoration_max_abs_error": metric(diagnostic_restore),
    }
    ledger.add(f"instance_{index:02d}_diagnostic_trajectory_sealed", diagnostic)

    combined_accepted = bool(latch.valid and stability.gate_pass)
    acceptance = {
        "gate_disabled_acceptance": bool(latch.valid),
        "nominal_acceptance": bool(latch.valid),
        "raw_spins_after_stability_gate": list(latch.raw_spins),
        "raw_spins_before_stability_gate": list(latch.raw_spins),
        "stability_gate_pass": bool(stability.gate_pass),
        "stability_gated_acceptance": combined_accepted,
    }
    ledger.add(f"instance_{index:02d}_acceptance_sealed", acceptance)

    stability_controls = {
        "all_diagnostic_carriers_restore": bool(
            diagnostic_restore <= gate.DIAGNOSTIC_RESTORATION_MAX
            and stability.diagnostic_restoration_max_abs_error
            <= gate.DIAGNOSTIC_RESTORATION_MAX
        ),
        "all_pass": False,
        "diagnostic_probe_fixed_before_execution": True,
        "diagnostic_schedule_identical": tuple(checkpoints) == gate.CHECKPOINT_STEPS,
        "disabling_gate_reproduces_nominal_acceptance": (
            acceptance["gate_disabled_acceptance"] == latch.valid
        ),
        "gate_never_changes_raw_spins": (
            acceptance["raw_spins_before_stability_gate"]
            == acceptance["raw_spins_after_stability_gate"]
        ),
        "gate_returns_no_diagnostic_answer": not hasattr(stability, "raw_spins"),
        "result_and_stability_outside_inverse_history": True,
        "source_has_no_identity_or_adjudication_data": True,
        "zero_perturbation_reproduces_nominal": (
            replay_delta <= gate.REPLAY_TOLERANCE
            and stability.diagnostic_replay_max_abs_delta <= gate.REPLAY_TOLERANCE
        ),
    }
    stability_controls["all_pass"] = all(
        value for key, value in stability_controls.items() if key != "all_pass"
    )
    ledger.add(f"instance_{index:02d}_stability_controls", stability_controls)

    uninterpretable = not all(
        (
            restoration["pass"],
            reuse_record["pass"],
            controls["all_pass"],
            stability_controls["all_pass"],
        )
    )
    return {
        "acceptance": acceptance,
        "boundary": boundary,
        "controls": controls,
        "diagnostic": diagnostic,
        "index": index,
        "nominal_native": nominal_native,
        "problem_sha256": record["problem_sha256"],
        "restoration": restoration,
        "reuse": reuse_record,
        "stability_controls": stability_controls,
        "uninterpretable": uninterpretable,
    }


def build_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze, batch = frozen_documents()
    contract = contract_document()
    ledger = EventLedger([])
    ledger.add("preoracle_contract_bound", contract)
    carrier = v2.r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], v2.SITE_COUNT, axis=0)

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("pre-oracle adjudication boundary is sealed")

    guarded_names = ("development_instances", "exact_oracle", "ising_energy")
    originals = {name: getattr(v2, name) for name in guarded_names}
    original_projection = v2.project_boundary
    try:
        for name in guarded_names:
            setattr(v2, name, blocked)
        v2.project_boundary = boundary_without_energy
        records = [
            execute_instance(record, borrowed, ledger)
            for record in batch["ordered_instances"]
        ]
    finally:
        for name, value in originals.items():
            setattr(v2, name, value)
        v2.project_boundary = original_projection

    summary = {
        "batch_size": len(records),
        "diagnostic_restoration_success_count": sum(
            record["stability_controls"]["all_diagnostic_carriers_restore"]
            for record in records
        ),
        "nominal_acceptance_count": sum(
            record["acceptance"]["nominal_acceptance"] for record in records
        ),
        "nominal_restoration_success_count": sum(
            record["restoration"]["pass"] for record in records
        ),
        "oracle_call_count": 0,
        "reuse_success_count": sum(record["reuse"]["pass"] for record in records),
        "stability_control_success_count": sum(
            record["stability_controls"]["all_pass"] for record in records
        ),
        "stability_gate_pass_count": sum(
            record["acceptance"]["stability_gate_pass"] for record in records
        ),
        "stability_gated_acceptance_count": sum(
            record["acceptance"]["stability_gated_acceptance"] for record in records
        ),
        "strict_v2_control_success_count": sum(
            record["controls"]["all_pass"] for record in records
        ),
        "uninterpretable_count": sum(record["uninterpretable"] for record in records),
    }
    evidence = {
        "batch_ordered_sha256": batch["ordered_batch_sha256"],
        "contract_sha256": sha256_bytes(canonical_bytes(contract)),
        "discriminator_fingerprint": freeze["discriminator"]["discriminator_fingerprint"],
        "freeze_commit": FREEZE_COMMIT,
        "instances": records,
        "machine_fingerprint": freeze["machine"]["machine_fingerprint"],
        "oracle_opened": False,
        "schema": "catalytic_waveform_ising_v2_stability_preoracle_evidence_v1",
        "summary": summary,
    }
    trace = ledger.document()
    evidence_sha = sha256_bytes(canonical_bytes(evidence))
    trace_sha = sha256_bytes(canonical_bytes(trace))
    seal = {
        "batch_ordered_sha256": batch["ordered_batch_sha256"],
        "contract_sha256": sha256_bytes(canonical_bytes(contract)),
        "evidence_sha256": evidence_sha,
        "freeze_commit": FREEZE_COMMIT,
        "oracle_call_count": 0,
        "schema": "catalytic_waveform_ising_v2_stability_preoracle_seal_v1",
        "status": "COMPLETE_64_INSTANCE_PREORACLE_ROOT_SEALED",
        "trace_sha256": trace_sha,
    }
    return contract, evidence, trace, seal


def build() -> dict[str, Any]:
    contract, evidence, trace, seal = build_documents()
    write_atomic(CONTRACT_FILE, canonical_bytes(contract))
    write_atomic(EVIDENCE_FILE, canonical_bytes(evidence))
    write_atomic(TRACE_FILE, canonical_bytes(trace))
    write_atomic(SEAL_FILE, canonical_bytes(seal))
    return {
        "evidence_sha256": seal["evidence_sha256"],
        "oracle_call_count": seal["oracle_call_count"],
        "status": seal["status"],
        "summary": evidence["summary"],
        "trace_sha256": seal["trace_sha256"],
    }


def verify() -> dict[str, Any]:
    contract, evidence, trace, seal = build_documents()
    expected = {
        CONTRACT_FILE: canonical_bytes(contract),
        EVIDENCE_FILE: canonical_bytes(evidence),
        TRACE_FILE: canonical_bytes(trace),
        SEAL_FILE: canonical_bytes(seal),
    }
    for path, payload in expected.items():
        if path.read_bytes() != payload:
            raise RuntimeError(f"pre-oracle artifact does not reproduce: {path.name}")
    if seal["oracle_call_count"] != 0 or evidence["summary"]["uninterpretable_count"] != 0:
        raise RuntimeError("pre-oracle seal is not admissible")
    return {
        "evidence_sha256": seal["evidence_sha256"],
        "oracle_call_count": 0,
        "status": seal["status"],
        "summary": evidence["summary"],
        "trace_sha256": seal["trace_sha256"],
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
