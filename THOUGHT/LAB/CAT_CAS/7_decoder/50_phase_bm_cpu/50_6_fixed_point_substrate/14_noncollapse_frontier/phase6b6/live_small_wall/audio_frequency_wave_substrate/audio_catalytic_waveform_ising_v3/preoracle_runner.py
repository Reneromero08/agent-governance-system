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
MACHINE_SOURCE = PACKAGE_DIR / "v3_machine.py"
CONTROL_SOURCE = PACKAGE_DIR / "control_qualifier.py"
FREEZE_FILE = PACKAGE_DIR / "V3_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "V3_BATCH_CUSTODY.json"
EVIDENCE_FILE = PACKAGE_DIR / "V3_PREORACLE_EVIDENCE.json"
TRACE_FILE = PACKAGE_DIR / "V3_PREORACLE_TRACE.json"
SEAL_FILE = PACKAGE_DIR / "V3_PREORACLE_SEAL.json"

FREEZE_COMMIT = "854c39c9c7a8321b4ff2ff0556f3a17111536f94"
EXPECTED_MACHINE_FINGERPRINT = (
    "f5945cc9e984da0e18a56002e8a4b664291d48d98adf1719e8038c9538c4a87f"
)
EXPECTED_BATCH_SHA256 = (
    "0e6ee2935dd5472acb94d0fa27b283bb439cc6e002f64059dc5d79c372c11bf7"
)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


machine = load_module(MACHINE_SOURCE, "catcas_waveform_ising_v3_preoracle_machine")
controls = load_module(CONTROL_SOURCE, "catcas_waveform_ising_v3_preoracle_controls")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def complex_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.complex128))
    return sha256_bytes(array.view(np.float64).tobytes(order="C"))


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
            "schema": "catalytic_waveform_ising_v3_preoracle_trace_v1",
        }


def load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    batch = json.loads(BATCH_FILE.read_text(encoding="utf-8"))
    if freeze["machine_fingerprint"] != EXPECTED_MACHINE_FINGERPRINT:
        raise RuntimeError("frozen machine fingerprint mismatch")
    if machine.machine_fingerprint() != EXPECTED_MACHINE_FINGERPRINT:
        raise RuntimeError("executing machine differs from freeze")
    if freeze["machine_source_sha256"] != sha256_file(MACHINE_SOURCE):
        raise RuntimeError("machine source differs from freeze")
    if freeze["batch_ordered_sha256"] != EXPECTED_BATCH_SHA256:
        raise RuntimeError("freeze binds a different batch")
    if batch["ordered_batch_sha256"] != EXPECTED_BATCH_SHA256:
        raise RuntimeError("batch custody hash mismatch")
    if len(batch["ordered_instances"]) != 256:
        raise RuntimeError("prospective batch is incomplete")
    return freeze, batch


def execute_instance(
    record: dict[str, Any], borrowed: np.ndarray, ledger: EventLedger
) -> dict[str, Any]:
    index = int(record["index"])
    coupling = np.asarray(record["coupling_matrix_J"], dtype=np.float64)
    field = np.asarray(record["field_vector_h"], dtype=np.float64)
    execution = machine.execute_native_cycle(borrowed, coupling, field)
    boundary = machine.project_boundary(execution, f"prospective_{index:03d}")
    native = {
        "best_mode_concentration": metric(boundary.best_mode_concentration),
        "best_mode_index": boundary.best_mode_index,
        "displaced_carrier_sha256": complex_array_sha256(execution.displaced),
        "displacement_l2": metric(execution.displacement_l2),
        "minimum_response_coherence": metric(min(boundary.coherence)),
        "minimum_geometry_coherence": metric(boundary.minimum_geometry_coherence),
        "mode_penalties": [metric(value) for value in boundary.mode_penalties],
        "operator_history_sha256": complex_array_sha256(execution.relation_history),
        "raw_spins": list(boundary.raw_spins),
        "second_mode_gap": metric(boundary.second_mode_gap),
        "valid": boundary.valid,
    }
    ledger.add(f"instance_{index:03d}_native_boundary_sealed", native)
    restored = machine.restore_carrier(execution)
    expected = machine.as_carrier_bank(borrowed)
    restoration_error = machine.maximum_abs_error(restored, expected)
    reuse = machine.execute_native_cycle(restored, coupling, field)
    reuse_boundary = machine.project_boundary(reuse, f"prospective_{index:03d}_reuse")
    reuse_restored = machine.restore_carrier(reuse)
    reuse_restoration_error = machine.maximum_abs_error(reuse_restored, expected)
    reuse_record = {
        "raw_spins_reproduced": reuse_boundary.raw_spins == boundary.raw_spins,
        "response_delta_l2": metric(
            controls.development.response_delta(boundary, reuse_boundary)
        ),
        "restoration_max_abs_error": metric(restoration_error),
        "reuse_restoration_max_abs_error": metric(reuse_restoration_error),
        "valid_reproduced": reuse_boundary.valid == boundary.valid,
    }
    reuse_record["pass"] = bool(
        reuse_record["raw_spins_reproduced"]
        and reuse_record["valid_reproduced"]
        and reuse_record["response_delta_l2"] <= 1.0e-12
        and restoration_error <= machine.RESTORATION_MAX
        and reuse_restoration_error <= machine.RESTORATION_MAX
    )
    ledger.add(f"instance_{index:03d}_restoration_reuse_sealed", reuse_record)
    control_input = {
        "coupling": coupling,
        "field": field,
        "label": f"prospective_{index:03d}",
        "problem_sha256": str(record["problem_sha256"]),
        "source_group": "prospective_batch",
    }
    strict = controls.execute_controls(control_input, borrowed)
    ledger.add(f"instance_{index:03d}_strict_controls_sealed", strict)
    return {
        "index": index,
        "native": native,
        "problem_sha256": str(record["problem_sha256"]),
        "restoration_and_reuse": reuse_record,
        "strict_controls": strict,
        "uninterpretable": False,
    }


def build_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze, batch = load_frozen()
    calls = {"energy": 0, "oracle": 0}
    original_energy = machine.v2.ising_energy
    original_oracle = machine.v2.exact_oracle

    def blocked_energy(*args: Any, **kwargs: Any) -> Any:
        calls["energy"] += 1
        raise RuntimeError("energy is forbidden before the pre-oracle seal")

    def blocked_oracle(*args: Any, **kwargs: Any) -> Any:
        calls["oracle"] += 1
        raise RuntimeError("oracle is forbidden before the pre-oracle seal")

    machine.v2.ising_energy = blocked_energy
    machine.v2.exact_oracle = blocked_oracle
    controls.machine.v2.ising_energy = blocked_energy
    controls.machine.v2.exact_oracle = blocked_oracle
    borrowed = machine.v2.r4.borrowed_carrier()
    ledger = EventLedger([])
    ledger.add(
        "remote_freeze_custody",
        {
            "batch_ordered_sha256": EXPECTED_BATCH_SHA256,
            "freeze_commit": FREEZE_COMMIT,
            "machine_fingerprint": EXPECTED_MACHINE_FINGERPRINT,
        },
    )
    try:
        instances = [
            execute_instance(record, borrowed, ledger)
            for record in batch["ordered_instances"]
        ]
    finally:
        machine.v2.ising_energy = original_energy
        machine.v2.exact_oracle = original_oracle
        controls.machine.v2.ising_energy = original_energy
        controls.machine.v2.exact_oracle = original_oracle
    if calls != {"energy": 0, "oracle": 0}:
        raise RuntimeError("forbidden pre-oracle call was attempted")
    summary = {
        "accepted_count": sum(record["native"]["valid"] for record in instances),
        "batch_size": len(instances),
        "maximum_restoration_error": metric(
            max(
                record["restoration_and_reuse"]["restoration_max_abs_error"]
                for record in instances
            )
        ),
        "maximum_reuse_restoration_error": metric(
            max(
                record["restoration_and_reuse"]["reuse_restoration_max_abs_error"]
                for record in instances
            )
        ),
        "minimum_mode_gap": metric(
            min(record["native"]["second_mode_gap"] for record in instances)
        ),
        "minimum_response_coherence": metric(
            min(
                record["native"]["minimum_response_coherence"]
                for record in instances
            )
        ),
        "oracle_call_count": calls["oracle"],
        "energy_call_count": calls["energy"],
        "restoration_reuse_pass_count": sum(
            record["restoration_and_reuse"]["pass"] for record in instances
        ),
        "strict_control_pass_count": sum(
            record["strict_controls"]["all_pass"] for record in instances
        ),
        "uninterpretable_count": sum(record["uninterpretable"] for record in instances),
    }
    evidence = {
        "batch_ordered_sha256": EXPECTED_BATCH_SHA256,
        "freeze_commit": FREEZE_COMMIT,
        "instances": instances,
        "machine_fingerprint": EXPECTED_MACHINE_FINGERPRINT,
        "oracle_opened": False,
        "schema": "catalytic_waveform_ising_v3_preoracle_evidence_v1",
        "summary": summary,
    }
    ledger.add("complete_preoracle_evidence_sealed", summary)
    trace = ledger.document()
    evidence_bytes = canonical_bytes(evidence)
    trace_bytes = canonical_bytes(trace)
    seal = {
        "batch_ordered_sha256": EXPECTED_BATCH_SHA256,
        "energy_call_count": calls["energy"],
        "evidence_sha256": sha256_bytes(evidence_bytes),
        "freeze_commit": FREEZE_COMMIT,
        "machine_fingerprint": EXPECTED_MACHINE_FINGERPRINT,
        "oracle_call_count": calls["oracle"],
        "schema": "catalytic_waveform_ising_v3_preoracle_seal_v1",
        "trace_sha256": sha256_bytes(trace_bytes),
    }
    return evidence, trace, seal


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence, trace, seal = build_documents()
    write_atomic(EVIDENCE_FILE, canonical_bytes(evidence))
    write_atomic(TRACE_FILE, canonical_bytes(trace))
    write_atomic(SEAL_FILE, canonical_bytes(seal))
    return evidence, trace, seal


def verify() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence, trace, seal = build_documents()
    if EVIDENCE_FILE.read_bytes() != canonical_bytes(evidence):
        raise ValueError("V3 pre-oracle evidence does not reproduce")
    if TRACE_FILE.read_bytes() != canonical_bytes(trace):
        raise ValueError("V3 pre-oracle trace does not reproduce")
    if SEAL_FILE.read_bytes() != canonical_bytes(seal):
        raise ValueError("V3 pre-oracle seal does not reproduce")
    return evidence, trace, seal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    evidence, _, seal = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "evidence_sha256": seal["evidence_sha256"],
                "oracle_call_count": seal["oracle_call_count"],
                "summary": evidence["summary"],
                "trace_sha256": seal["trace_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
