from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
HELDOUT_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_heldout_v1"
HELDOUT_SOURCE = HELDOUT_DIR / "heldout_generalization_reference.py"
FREEZER_SOURCE = PACKAGE_DIR / "batch_instance_freezer.py"
CUSTODY_FILE = PACKAGE_DIR / "BATCH_INSTANCE_CUSTODY.json"
CONTRACT_FILE = PACKAGE_DIR / "BATCH_PREORACLE_CONTRACT.json"
EVIDENCE_FILE = PACKAGE_DIR / "BATCH_PREORACLE_EVIDENCE.json"
TRACE_FILE = PACKAGE_DIR / "BATCH_PREORACLE_TRACE.json"
SEAL_FILE = PACKAGE_DIR / "BATCH_PREORACLE_SEAL.json"

FREEZE_COMMIT = "b6b53493722aeca5cc8cc38bb41f9e9be66afb68"
EXPECTED_BATCH_SHA256 = (
    "4109d430789b8fb3912ad606b78311855e89e40b422fb3ecec9b84f5818c0c12"
)
EXPECTED_HELDOUT_SOURCE_SHA256 = (
    "9599683ed4e178fdc2e644110e143aa99f0cfbb43b3a644379943a1022e947b9"
)
CONTRACT_SCHEMA = "catalytic_waveform_ising_batch_preoracle_contract_v1"
EVIDENCE_SCHEMA = "catalytic_waveform_ising_batch_preoracle_evidence_v1"
TRACE_SCHEMA = "catalytic_waveform_ising_batch_preoracle_trace_v1"
SEAL_SCHEMA = "catalytic_waveform_ising_batch_preoracle_seal_v1"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


held = load_module(HELDOUT_SOURCE, "catcas_batch_heldout_reference")
freezer = load_module(FREEZER_SOURCE, "catcas_batch_freezer")
r4 = held.r4


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


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, canonical_bytes(value))


def custody_document() -> dict[str, Any]:
    expected = freezer.custody_document()
    if CUSTODY_FILE.read_bytes() != canonical_bytes(expected):
        raise ValueError("batch custody bytes do not reproduce")
    if expected["ordered_batch_sha256"] != EXPECTED_BATCH_SHA256:
        raise ValueError("ordered batch identity changed")
    if any(expected["freeze_order"].values()):
        raise ValueError("batch freeze-order declaration is invalid")
    return expected


def frozen_machine_document() -> dict[str, Any]:
    if sha256_file(HELDOUT_SOURCE) != EXPECTED_HELDOUT_SOURCE_SHA256:
        raise ValueError("held-out adapter source identity changed")
    machine = held.frozen_machine_document()
    machine_hash = sha256_bytes(canonical_bytes(machine))
    if machine_hash != freezer.FROZEN_MACHINE_SHA256:
        raise ValueError("frozen machine fingerprint changed")
    return {
        "heldout_adapter_source_sha256": EXPECTED_HELDOUT_SOURCE_SHA256,
        "machine": machine,
        "machine_sha256": machine_hash,
    }


def source_absence_proof() -> dict[str, Any]:
    source = Path(__file__).resolve().read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {"exact_oracle", "oracle_document"}
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            findings.append(node.func.id)
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
            findings.append(node.func.attr)
    return {
        "forbidden_oracle_call_findings": findings,
        "oracle_calls": len(findings),
        "pass": not findings,
        "source_sha256": sha256_bytes(source.encode("utf-8")),
    }


def contract_document() -> dict[str, Any]:
    custody = custody_document()
    machine = frozen_machine_document()
    return {
        "batch_size": freezer.BATCH_SIZE,
        "claim_ceiling": held.CLAIM_CEILING,
        "control_coverage_law": "all_required_controls_on_every_batch_instance",
        "freeze_commit": FREEZE_COMMIT,
        "frozen_machine": machine,
        "intended_variables_only": ["coupling_matrix_J", "field_vector_h"],
        "oracle_boundary": "after_exact_preoracle_evidence_root_is_written_and_sealed",
        "ordered_batch_sha256": EXPECTED_BATCH_SHA256,
        "promotion_criterion": custody["promotion_criterion_frozen_before_execution"],
        "reuse_problem": {
            "coupling_matrix_J": r4.COUPLING.tolist(),
            "field_vector_h": r4.REUSE_FIELD.tolist(),
            "source": "verified_predecessor_reuse_instance",
        },
        "schema": CONTRACT_SCHEMA,
        "thresholds": {
            "coherence_min": r4.QUERY_COHERENCE_MIN,
            "displacement_l2_min": r4.DISPLACEMENT_MIN,
            "lock_residual_max_rad": r4.LOCK_RESIDUAL_MAX,
            "operator_history_change_l2_min": r4.OPERATOR_HISTORY_CHANGE_MIN,
            "restoration_max_abs_error": r4.RESTORE_TOL,
            "samplewise_non_rank_one_min": r4.SAMPLEWISE_DYNAMICS_MIN,
            "wrong_restoration_min_abs_error": r4.WRONG_RESTORE_MIN,
        },
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
            "schema": TRACE_SCHEMA,
        }


def problem_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return r4.validate_problem(
        np.asarray(record["coupling_matrix_J"], dtype=np.float64),
        np.asarray(record["field_vector_h"], dtype=np.float64),
    )


def compact_controls(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "all_pass": controls["all_pass"],
        "measurements": controls["measurements"],
        "outcomes": controls["outcomes"],
    }


def prompt_materiality_controls(controls: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen batch law without altering stricter predecessor outcomes."""
    measurements = controls["measurements"]
    deltas = measurements["response_deltas_l2"]
    threshold = r4.OPERATOR_HISTORY_CHANGE_MIN
    outcomes = {
        "carrier_content_material": (
            measurements["carrier_content_history_change_l2"] >= threshold
            or deltas["uniform_carrier"] >= threshold
        ),
        "flat_geometry_material": (
            measurements["flat_geometry_history_change_l2"] >= threshold
            or deltas["flat_geometry"] >= threshold
        ),
        "missing_phase_operator_material": (
            measurements["missing_phase_operator_history_change_l2"] >= threshold
            or deltas["missing_phase_operator"] >= threshold
        ),
        "no_lock_material": (
            measurements["no_lock_history_change_l2"] >= threshold
            or deltas["no_lock"] >= threshold
        ),
        "no_transform_material": (
            measurements["no_transform_history_change_l2"] >= threshold
            or deltas["no_transform"] >= threshold
        ),
        "omitted_inverse_step_failed": controls["outcomes"][
            "omitted_inverse_step_failed"
        ],
        "omitted_restoration_failed": controls["outcomes"][
            "omitted_restoration_failed"
        ],
        "samplewise_non_rank_one": controls["outcomes"]["samplewise_non_rank_one"],
        "scrambled_geometry_material": (
            measurements["scrambled_geometry_history_change_l2"] >= threshold
            or deltas["scrambled_geometry"] >= threshold
        ),
        "wrong_inverse_failed": controls["outcomes"]["wrong_inverse_failed"],
        "wrong_query_material": deltas["wrong_query"] >= threshold,
    }
    return {
        "all_pass": all(outcomes.values()),
        "law": "material_native_history_or_complex_response_change",
        "outcomes": outcomes,
        "threshold": threshold,
    }


def execute_instance_preoracle(
    record: dict[str, Any], ledger: EventLedger
) -> tuple[dict[str, Any], np.ndarray, Any]:
    index = int(record["index"])
    coupling, field = problem_arrays(record)
    carrier = r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], r4.SITE_COUNT, axis=0)

    execution = held.execute_native_cycle_problem(borrowed, coupling, field)
    native_seal = held.native_seal(execution)
    ledger.add(f"instance_{index:02d}_native_waveform_and_history_sealed", native_seal)

    latch = held.project_boundary_problem(execution, coupling, field, f"batch_{index:02d}")
    raw_energy = r4.ising_energy(latch.raw_spins, coupling, field)
    latch_digest = latch.digest()
    boundary = {"latch": latch.document(), "raw_spin_energy": metric(raw_energy)}
    ledger.add(f"instance_{index:02d}_raw_boundary_sealed", boundary)

    restored = r4.restore_carrier(execution, "correct")
    restoration_error = float(np.max(np.abs(restored - borrowed)))
    latch_persisted = latch.digest() == latch_digest
    restoration = {
        "latch_persisted": latch_persisted,
        "max_abs_error": metric(restoration_error),
    }
    ledger.add(f"instance_{index:02d}_carrier_restored", restoration)

    reuse_execution = held.execute_native_cycle_problem(restored, r4.COUPLING, r4.REUSE_FIELD)
    reuse_native_seal = held.native_seal(reuse_execution)
    ledger.add(f"instance_{index:02d}_reuse_native_waveform_and_history_sealed", reuse_native_seal)
    reuse_latch = held.project_boundary_problem(
        reuse_execution, r4.COUPLING, r4.REUSE_FIELD, f"batch_{index:02d}_reuse"
    )
    ledger.add(f"instance_{index:02d}_reuse_boundary_sealed", reuse_latch.document())
    reuse_restored = r4.restore_carrier(reuse_execution, "correct")
    reuse_input_error = float(np.max(np.abs(reuse_execution.borrowed - restored)))
    reuse_restoration_error = float(np.max(np.abs(reuse_restored - restored)))
    reuse = {
        "input_max_abs_error": metric(reuse_input_error),
        "latch": reuse_latch.document(),
        "native_seal": reuse_native_seal,
        "restoration_max_abs_error": metric(reuse_restoration_error),
    }
    ledger.add(f"instance_{index:02d}_reuse_restored", {
        "input_max_abs_error": metric(reuse_input_error),
        "restoration_max_abs_error": metric(reuse_restoration_error),
    })

    controls = compact_controls(held.run_controls(borrowed, coupling, field, execution, latch))
    materiality_controls = prompt_materiality_controls(controls)
    ledger.add(f"instance_{index:02d}_controls_sealed", controls)
    ledger.add(
        f"instance_{index:02d}_batch_law_control_materiality_sealed",
        materiality_controls,
    )

    integrity = {
        "carrier_displaced": execution.displacement_l2 >= r4.DISPLACEMENT_MIN,
        "carrier_restored": restoration_error <= r4.RESTORE_TOL,
        "control_materiality_pass": materiality_controls["all_pass"],
        "latch_persisted": latch_persisted,
        "reuse_carrier_restored": reuse_restoration_error <= r4.RESTORE_TOL,
        "reuse_consumed_exact_restored_carrier": reuse_input_error == 0.0,
        "reuse_result_valid": reuse_latch.valid,
    }
    result = {
        "batch_law_control_materiality": materiality_controls,
        "controls": controls,
        "coupling_matrix_J": coupling.tolist(),
        "field_vector_h": field.tolist(),
        "index": index,
        "instance_sha256": record["instance_sha256"],
        "integrity": integrity,
        "interpretable_preoracle": all(integrity.values()),
        "measurements": {
            "carrier_displacement_l2": metric(execution.displacement_l2),
            "restoration_max_abs_error": metric(restoration_error),
            "reuse_carrier_displacement_l2": metric(reuse_execution.displacement_l2),
            "reuse_input_max_abs_error": metric(reuse_input_error),
            "reuse_restoration_max_abs_error": metric(reuse_restoration_error),
        },
        "native_seal": native_seal,
        "raw_boundary": boundary,
        "reuse": reuse,
    }
    return result, restored, reuse_execution


def build_preoracle_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    custody = custody_document()
    machine = frozen_machine_document()
    absence = source_absence_proof()
    call_path = held.native_call_path_proof()
    ledger = EventLedger([])
    ledger.add("batch_custody_verified", {
        "freeze_commit": FREEZE_COMMIT,
        "ordered_batch_sha256": EXPECTED_BATCH_SHA256,
    })
    ledger.add("oracle_free_execution_source_verified", absence)
    ledger.add("frozen_machine_verified", machine)

    instances: list[dict[str, Any]] = []
    first_restored: np.ndarray | None = None
    first_reuse_execution: Any = None
    for record in custody["ordered_instances"]:
        result, restored, reuse_execution = execute_instance_preoracle(record, ledger)
        instances.append(result)
        if first_restored is None:
            first_restored = restored
            first_reuse_execution = reuse_execution

    if first_restored is None:
        raise ValueError("frozen batch is empty")
    carrier = r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], r4.SITE_COUNT, axis=0)
    adapter_equivalence = held.exact_adapter_equivalence(
        borrowed, first_restored, first_reuse_execution
    )
    ledger.add("exact_predecessor_adapter_equivalence_sealed", adapter_equivalence)

    evidence = {
        "adapter_equivalence": adapter_equivalence,
        "batch_size": len(instances),
        "claim_ceiling": held.CLAIM_CEILING,
        "freeze_commit": FREEZE_COMMIT,
        "frozen_machine": machine,
        "instances": instances,
        "native_call_path_proof": call_path,
        "oracle_absence_proof": absence,
        "ordered_batch_sha256": EXPECTED_BATCH_SHA256,
        "schema": EVIDENCE_SCHEMA,
        "summary": {
            "all_instances_interpretable_preoracle": all(
                item["interpretable_preoracle"] for item in instances
            ),
            "control_pass_count": sum(item["controls"]["all_pass"] for item in instances),
            "control_materiality_pass_count": sum(
                item["batch_law_control_materiality"]["all_pass"]
                for item in instances
            ),
            "native_call_path_pass": call_path["pass"],
            "oracle_call_count": absence["oracle_calls"],
            "restoration_success_count": sum(
                item["integrity"]["carrier_restored"] for item in instances
            ),
            "reuse_success_count": sum(
                item["integrity"]["reuse_result_valid"]
                and item["integrity"]["reuse_carrier_restored"]
                and item["integrity"]["reuse_consumed_exact_restored_carrier"]
                for item in instances
            ),
        },
    }
    evidence_sha = sha256_bytes(canonical_bytes(evidence))
    ledger.add("full_preoracle_batch_root_sealed", {
        "batch_size": len(instances),
        "preoracle_evidence_sha256": evidence_sha,
    })
    trace = ledger.document()
    seal = {
        "batch_size": len(instances),
        "execution_source_sha256": absence["source_sha256"],
        "freeze_commit": FREEZE_COMMIT,
        "oracle_call_count": absence["oracle_calls"],
        "ordered_batch_sha256": EXPECTED_BATCH_SHA256,
        "preoracle_evidence_bytes": len(canonical_bytes(evidence)),
        "preoracle_evidence_sha256": evidence_sha,
        "preoracle_trace_sha256": sha256_bytes(canonical_bytes(trace)),
        "schema": SEAL_SCHEMA,
    }
    return evidence, trace, seal


def build_package() -> dict[str, Any]:
    write_json(CONTRACT_FILE, contract_document())
    evidence, trace, seal = build_preoracle_documents()
    write_json(EVIDENCE_FILE, evidence)
    write_json(TRACE_FILE, trace)
    write_json(SEAL_FILE, seal)
    return seal


def verify_package() -> dict[str, Any]:
    if CONTRACT_FILE.read_bytes() != canonical_bytes(contract_document()):
        raise ValueError("committed pre-oracle contract does not reproduce")
    evidence, trace, seal = build_preoracle_documents()
    expected = {
        EVIDENCE_FILE: evidence,
        TRACE_FILE: trace,
        SEAL_FILE: seal,
    }
    for path, document in expected.items():
        if path.read_bytes() != canonical_bytes(document):
            raise ValueError(f"committed pre-oracle artifact does not reproduce: {path.name}")
    return seal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    seal = build_package() if args.command == "build" else verify_package()
    print(json.dumps({
        "batch_size": seal["batch_size"],
        "oracle_call_count": seal["oracle_call_count"],
        "ordered_batch_sha256": seal["ordered_batch_sha256"],
        "preoracle_evidence_sha256": seal["preoracle_evidence_sha256"],
        "status": "FULL_PREORACLE_BATCH_ROOT_SEALED",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
