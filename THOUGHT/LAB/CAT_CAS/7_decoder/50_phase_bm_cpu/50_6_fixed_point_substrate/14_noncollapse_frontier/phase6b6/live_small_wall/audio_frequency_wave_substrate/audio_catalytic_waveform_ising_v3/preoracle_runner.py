from __future__ import annotations

import argparse
import hashlib
import tempfile
import types
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


sys.dont_write_bytecode = True


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
MACHINE_SOURCE = PACKAGE_DIR / "v3_machine.py"
CONTROL_SOURCE = PACKAGE_DIR / "control_qualifier.py"
FREEZE_FILE = PACKAGE_DIR / "V3_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "V3_BATCH_CUSTODY.json"
EVIDENCE_FILE = PACKAGE_DIR / "V3_PREORACLE_EVIDENCE.json"
TRACE_FILE = PACKAGE_DIR / "V3_PREORACLE_TRACE.json"
SEAL_FILE = PACKAGE_DIR / "V3_PREORACLE_SEAL.json"
SOURCE_EXECUTION_CONTRACT = {
    "bytecode_cache_inputs_forbidden_under_package": True,
    "bytecode_cache_writes_disabled": True,
    "compile_dont_inherit": True,
    "compile_optimization": 0,
    "local_module_loader": "compile_exact_source_bytes",
}
EXECUTABLE_CACHE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dll"}

def load_module(path: Path, name: str) -> Any:
    source = path.read_bytes()
    code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(code, module.__dict__)
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


def unexpected_package_executable_inputs(root: Path = PACKAGE_DIR) -> list[str]:
    unexpected: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_dir() and path.name == "__pycache__":
            unexpected.add(relative + "/")
        elif path.is_file() and path.suffix.lower() in EXECUTABLE_CACHE_SUFFIXES:
            unexpected.add(relative)
    return sorted(unexpected)


def bytecode_rejection_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="catcas_v3_bytecode_gate_") as temporary:
        root = Path(temporary)
        cache = root / "__pycache__"
        cache.mkdir()
        probe = cache / "unsealed_probe.cpython-311.pyc"
        probe.write_bytes(b"not executable fixture bytes")
        found = unexpected_package_executable_inputs(root)
    expected = ["__pycache__/", "__pycache__/unsealed_probe.cpython-311.pyc"]
    return {
        "detected_paths": found,
        "expected_paths": expected,
        "pass": found == expected,
    }

def source_execution_environment() -> dict[str, Any]:
    return {
        **SOURCE_EXECUTION_CONTRACT,
        "observed_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "observed_optimization": int(sys.flags.optimize),
        "unexpected_package_executable_inputs": unexpected_package_executable_inputs(),
    }


def assert_clean_exact_tree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root(),
        capture_output=True,
        check=True,
        text=True,
    )
    dirty = completed.stdout.strip()
    if dirty:
        raise RuntimeError(
            "prospective execution requires a clean exact source tree: " + dirty
        )
    environment = source_execution_environment()
    if not environment["observed_dont_write_bytecode"]:
        raise RuntimeError("prospective execution requires bytecode writes disabled")
    if environment["observed_optimization"] != 0:
        raise RuntimeError("prospective execution requires optimization level zero")
    if environment["unexpected_package_executable_inputs"]:
        raise RuntimeError(
            "prospective execution rejects unsealed executable cache inputs: "
            + ", ".join(environment["unexpected_package_executable_inputs"])
        )

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
    if machine.machine_fingerprint() != freeze["machine_fingerprint"]:
        raise RuntimeError("executing machine differs from freeze")
    if freeze["machine_source_sha256"] != sha256_file(MACHINE_SOURCE):
        raise RuntimeError("machine source differs from freeze")
    if freeze["batch_file_sha256"] != sha256_file(BATCH_FILE):
        raise RuntimeError("batch file bytes differ from freeze")
    if batch["ordered_batch_sha256"] != freeze["batch_ordered_sha256"]:
        raise RuntimeError("batch custody hash mismatch")
    if len(batch["ordered_instances"]) != 256:
        raise RuntimeError("prospective batch is incomplete")
    if freeze["source_execution_contract"] != SOURCE_EXECUTION_CONTRACT:
        raise RuntimeError("source-only execution contract differs from freeze")
    identities: list[str] = []
    for expected_index, record in enumerate(batch["ordered_instances"]):
        if int(record["index"]) != expected_index:
            raise RuntimeError("prospective batch index mismatch")
        identity = problem_sha256(
            np.asarray(record["coupling_matrix_J"], dtype=np.float64),
            np.asarray(record["field_vector_h"], dtype=np.float64),
        )
        if identity != record["problem_sha256"]:
            raise RuntimeError("prospective problem identity mismatch")
        identities.append(identity)
    ordered = sha256_bytes(canonical_bytes(identities))
    if ordered != freeze["batch_ordered_sha256"]:
        raise RuntimeError("prospective ordered identity hash mismatch")
    for name, expected_sha in freeze["execution_source_sha256"].items():
        if sha256_file(PACKAGE_DIR / name) != expected_sha:
            raise RuntimeError(f"frozen execution source drift: {name}")
    for name, expected_sha in freeze["transitive_dependency_sha256"].items():
        if sha256_file(SUBSTRATE_DIR / name) != expected_sha:
            raise RuntimeError(f"frozen transitive dependency drift: {name}")
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
        "accepted_result_reproduced": bool(
            not boundary.valid
            or reuse_boundary.raw_spins == boundary.raw_spins
        ),
        "raw_spins_reproduced": reuse_boundary.raw_spins == boundary.raw_spins,
        "response_delta_l2": metric(
            controls.response_delta(boundary, reuse_boundary)
        ),
        "restoration_max_abs_error": metric(restoration_error),
        "reuse_restoration_max_abs_error": metric(reuse_restoration_error),
        "valid_reproduced": reuse_boundary.valid == boundary.valid,
    }
    reuse_record["pass"] = bool(
        reuse_record["accepted_result_reproduced"]
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


def build_documents(
    freeze_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze, batch = load_frozen()
    expected_machine_fingerprint = freeze["machine_fingerprint"]
    expected_batch_sha256 = freeze["batch_ordered_sha256"]
    calls = {"energy": 0, "oracle": 0}
    static_gate = controls.static_no_smuggle()
    if not static_gate["pass"]:
        raise RuntimeError("transitive no-smuggle source-closure gate failed")
    borrowed = machine.borrowed_carrier()
    runtime_gate = controls.runtime_no_smuggle(
        controls.development_corpus()[0], borrowed
    )
    if not runtime_gate["pass"]:
        raise RuntimeError("runtime no-smuggle gate failed")
    ledger = EventLedger([])
    null_coupling = np.zeros((machine.SITE_COUNT, machine.SITE_COUNT), dtype=np.float64)
    null_field = np.zeros(machine.SITE_COUNT, dtype=np.float64)
    null_execution = machine.execute_native_cycle(
        borrowed, null_coupling, null_field
    )
    null_boundary = machine.project_boundary(null_execution, "null_model_baseline")
    null_restored = machine.restore_carrier(null_execution)
    null_model = {
        "restoration_max_abs_error": metric(
            machine.maximum_abs_error(
                null_restored, machine.as_carrier_bank(borrowed)
            )
        ),
        "second_mode_gap": metric(null_boundary.second_mode_gap),
        "valid": null_boundary.valid,
    }
    null_model["pass"] = bool(
        not null_boundary.valid
        and null_boundary.second_mode_gap < machine.DEFAULT_LAW.unique_gap_min
        and null_model["restoration_max_abs_error"] <= machine.RESTORATION_MAX
    )
    if not null_model["pass"]:
        raise RuntimeError("zero-J/zero-h null model baseline failed")
    ledger.add(
        "remote_freeze_custody",
        {
            "batch_ordered_sha256": expected_batch_sha256,
            "freeze_commit": freeze_commit,
            "machine_fingerprint": expected_machine_fingerprint,
        },
    )
    execution_environment = source_execution_environment()
    if execution_environment["unexpected_package_executable_inputs"]:
        raise RuntimeError("unsealed executable cache appeared during execution")
    ledger.add("source_execution_environment_sealed", execution_environment)
    ledger.add("null_model_baseline_sealed", null_model)
    instances = [
        execute_instance(record, borrowed, ledger)
        for record in batch["ordered_instances"]
    ]
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
        "ignored_executable_negative_fixture_pass": bytecode_negative_control["pass"],
        "null_model_baseline_pass": null_model["pass"],
        "oracle_call_count": calls["oracle"],
        "energy_call_count": calls["energy"],
        "restoration_reuse_pass_count": sum(
            record["restoration_and_reuse"]["pass"] for record in instances
        ),
        "source_closure_no_smuggle_pass": static_gate["pass"] and runtime_gate["pass"],
        "strict_control_pass_count": sum(
            record["strict_controls"]["all_pass"] for record in instances
        ),
        "uninterpretable_count": sum(record["uninterpretable"] for record in instances),
    }
    evidence = {
        "batch_ordered_sha256": expected_batch_sha256,
        "freeze_commit": freeze_commit,
        "instances": instances,
        "machine_fingerprint": expected_machine_fingerprint,
        "oracle_opened": False,
        "schema": "catalytic_waveform_ising_v3_preoracle_evidence_v2",
        "source_execution_environment": execution_environment,
        "summary": summary,
    }
    ledger.add("complete_preoracle_evidence_sealed", summary)
    trace = ledger.document()
    evidence_bytes = canonical_bytes(evidence)
    trace_bytes = canonical_bytes(trace)
    seal = {
        "batch_ordered_sha256": expected_batch_sha256,
        "energy_call_count": calls["energy"],
        "evidence_sha256": sha256_bytes(evidence_bytes),
        "freeze_commit": freeze_commit,
        "machine_fingerprint": expected_machine_fingerprint,
        "oracle_call_count": calls["oracle"],
        "schema": "catalytic_waveform_ising_v3_preoracle_seal_v2",
        "source_execution_environment_sha256": sha256_bytes(
            canonical_bytes(execution_environment)
        ),
        "trace_sha256": sha256_bytes(trace_bytes),
    }
    return evidence, trace, seal


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assert_clean_exact_tree()
    evidence, trace, seal = build_documents(current_head())
    write_atomic(EVIDENCE_FILE, canonical_bytes(evidence))
    write_atomic(TRACE_FILE, canonical_bytes(trace))
    write_atomic(SEAL_FILE, canonical_bytes(seal))
    return evidence, trace, seal


def verify() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    committed_seal = json.loads(SEAL_FILE.read_text(encoding="utf-8"))
    evidence, trace, seal = build_documents(committed_seal["freeze_commit"])
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
