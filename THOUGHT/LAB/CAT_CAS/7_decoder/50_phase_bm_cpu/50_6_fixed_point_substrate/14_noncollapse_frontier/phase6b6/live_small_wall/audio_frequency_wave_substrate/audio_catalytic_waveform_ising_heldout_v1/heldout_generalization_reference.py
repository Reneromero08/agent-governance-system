from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
PREDECESSOR_DIR = SUBSTRATE_DIR / "audio_integrated_catalytic_computation_v1"
PREDECESSOR_SOURCE = PREDECESSOR_DIR / "integrated_catalytic_computation_reference.py"
PREDECESSOR_CONTRACT = (
    PREDECESSOR_DIR / "INTEGRATED_CATALYTIC_COMPUTATION_CONTRACT.json"
)
FREEZER_SOURCE = PACKAGE_DIR / "heldout_instance_freezer.py"
CUSTODY_FILE = PACKAGE_DIR / "HELD_OUT_INSTANCE_CUSTODY.json"
FREEZE_COMMIT = "c5da993afd20649bbf0413dda23b22e8c9c7bb45"
PREDECESSOR_SOURCE_SHA256 = (
    "50b6db77e2602e18356636ddb892f6d51aedb0573c6b2418afc8e5cc174991cc"
)
INSTANCE_SHA256 = "49db989fd525366867cf9c6866ebc7000b531b438b0227d7bb919e0ff3bf2704"

CONTRACT_FILE = "HELD_OUT_GENERALIZATION_CONTRACT.json"
RESULTS_FILE = "HELD_OUT_GENERALIZATION_RESULTS.json"
TRACE_FILE = "HELD_OUT_GENERALIZATION_EXECUTION_TRACE.json"
MANIFEST_FILE = "HELD_OUT_GENERALIZATION_MANIFEST.json"
HELDOUT_LATCH_FILE = "fixtures/heldout_latch.json"
REUSE_LATCH_FILE = "fixtures/reuse_latch.json"

CONTRACT_SCHEMA = "catalytic_waveform_ising_heldout_contract_v1"
RESULT_SCHEMA = "catalytic_waveform_ising_heldout_result_v1"
TRACE_SCHEMA = "catalytic_waveform_ising_heldout_trace_v1"
MANIFEST_SCHEMA = "catalytic_waveform_ising_heldout_manifest_v1"
CLAIM_CEILING = "BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY"
VERIFIED = "CATALYTIC_WAVEFORM_ISING_HELD_OUT_GENERALIZATION_VERIFIED"
PARTIAL = "CATALYTIC_WAVEFORM_ISING_HELD_OUT_GENERALIZATION_PARTIAL"
NOT_ESTABLISHED = "CATALYTIC_WAVEFORM_ISING_HELD_OUT_GENERALIZATION_NOT_ESTABLISHED"


def _load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


r4 = _load_module(PREDECESSOR_SOURCE, "catcas_heldout_r4")
freezer = _load_module(FREEZER_SOURCE, "catcas_heldout_freezer")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, canonical_bytes(value))


def custody_document() -> dict[str, Any]:
    expected = freezer.custody_document()
    if CUSTODY_FILE.read_bytes() != canonical_bytes(expected):
        raise ValueError("held-out custody bytes do not reproduce")
    if expected["held_out_instance_sha256"] != INSTANCE_SHA256:
        raise ValueError("held-out instance identity changed")
    if any(expected["freeze_order"].values()):
        raise ValueError("held-out pre-execution freeze order is invalid")
    return expected


def problem_from_custody() -> tuple[np.ndarray, np.ndarray]:
    instance = custody_document()["held_out_instance"]
    coupling = np.asarray(instance["coupling_matrix_J"], dtype=np.float64)
    field = np.asarray(instance["field_vector_h"], dtype=np.float64)
    coupling, field = r4.validate_problem(coupling, field)
    if np.array_equal(coupling, r4.COUPLING):
        raise ValueError("held-out J duplicates the predecessor J")
    if np.array_equal(field, r4.PRIMARY_FIELD) or np.array_equal(field, r4.REUSE_FIELD):
        raise ValueError("held-out h duplicates a predecessor field")
    return coupling, field


def frozen_machine_document() -> dict[str, Any]:
    if sha256_file(PREDECESSOR_SOURCE) != PREDECESSOR_SOURCE_SHA256:
        raise ValueError("verified predecessor source identity changed")
    committed_contract = json.loads(PREDECESSOR_CONTRACT.read_text(encoding="utf-8"))
    if committed_contract != r4.contract_document():
        raise ValueError("verified predecessor contract no longer matches its source")
    machine = dict(committed_contract)
    machine.pop("coupling")
    machine.pop("primary_field")
    machine.pop("reuse_field")
    return {
        "predecessor_contract_without_J_h": machine,
        "predecessor_source_bytes": PREDECESSOR_SOURCE.stat().st_size,
        "predecessor_source_sha256": PREDECESSOR_SOURCE_SHA256,
    }


def contract_document() -> dict[str, Any]:
    machine = frozen_machine_document()
    return {
        "claim_ceiling": CLAIM_CEILING,
        "freeze_commit": FREEZE_COMMIT,
        "frozen_machine": machine,
        "frozen_machine_sha256": sha256_bytes(canonical_bytes(machine)),
        "held_out_instance_sha256": INSTANCE_SHA256,
        "intended_variables_only": ["coupling_matrix_J", "field_vector_h"],
        "oracle_boundary": "after_heldout_projection_restoration_reuse_and_controls",
        "projection_acceptance_law": (
            "The raw sign shadow is recorded, but VERIFIED requires the unchanged "
            "predecessor coherence and lock-residual gates. An oracle-matching raw "
            "shadow below those gates remains PARTIAL."
        ),
        "reuse_problem": {
            "coupling_matrix_J": r4.COUPLING.tolist(),
            "field_vector_h": r4.REUSE_FIELD.tolist(),
            "source": "verified_predecessor_reuse_instance",
        },
        "schema": CONTRACT_SCHEMA,
        "thresholds": {
            "carrier_history_change_l2_min": r4.OPERATOR_HISTORY_CHANGE_MIN,
            "operator_history_change_l2_min": r4.OPERATOR_HISTORY_CHANGE_MIN,
            "restoration_max_abs_error": r4.RESTORE_TOL,
            "samplewise_dynamics_min": r4.SAMPLEWISE_DYNAMICS_MIN,
            "wrong_restoration_min_abs_error": r4.WRONG_RESTORE_MIN,
        },
    }


def execute_native_cycle_problem(
    borrowed: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    *,
    program_beams: np.ndarray | None = None,
    actual_beams: np.ndarray | None = None,
    calibration_beams: np.ndarray | None = None,
    lock_final: float = r4.LOCK_FINAL,
    relation_enabled: np.ndarray | None = None,
) -> Any:
    """Exact predecessor execution path with J promoted from constant to input."""
    coupling, field = r4.validate_problem(coupling, field)
    borrowed = np.asarray(borrowed, dtype=np.complex128)
    if borrowed.shape == (r4.SAMPLE_COUNT,):
        borrowed = np.repeat(borrowed[np.newaxis, :], r4.SITE_COUNT, axis=0)
    if borrowed.shape != (r4.SITE_COUNT, r4.SAMPLE_COUNT):
        raise ValueError("borrowed carrier bank has the wrong shape")
    if not np.all(np.isfinite(borrowed)) or np.min(np.abs(borrowed)) <= 0.0:
        raise ValueError("borrowed carrier bank must be finite and nonzero")
    canonical_beams = r4.render_trees(r4.canonical_trees())
    program = (
        canonical_beams
        if program_beams is None
        else np.asarray(program_beams, dtype=np.complex128)
    )
    actual = program if actual_beams is None else np.asarray(actual_beams, dtype=np.complex128)
    if program.shape != canonical_beams.shape or np.max(np.abs(np.abs(program) - 1.0)) > 1.0e-12:
        raise ValueError("program geometry bank must be unit modulus")
    if actual.shape != program.shape or np.max(np.abs(np.abs(actual) - 1.0)) > 1.0e-12:
        raise ValueError("actual geometry bank must be unit modulus")
    enabled = (
        np.ones((r4.SITE_COUNT, r4.SITE_COUNT), dtype=np.bool_)
        if relation_enabled is None
        else np.array(relation_enabled, dtype=np.bool_, copy=True)
    )
    np.fill_diagonal(enabled, False)
    masks = r4.transport_mask_bank(program)
    channels = r4.geometry_channel_bank(program)
    calibration_source = (
        canonical_beams
        if calibration_beams is None
        else np.asarray(calibration_beams, dtype=np.complex128)
    )
    calibration = r4.geometry_calibration(calibration_source, borrowed)
    frames = borrowed * program
    states = borrowed * actual * np.exp(1j * r4.INITIAL_PHASES[:, np.newaxis])
    displaced, query_frames, history = r4.evolve_native_waveforms(
        states,
        frames,
        channels,
        masks,
        calibration,
        coupling,
        field,
        lock_final,
        enabled,
    )
    return r4.NativeExecution(
        borrowed=np.array(borrowed, copy=True),
        actual_beams=np.array(actual, copy=True),
        program_beams=np.array(program, copy=True),
        masks=masks,
        displaced=displaced,
        query_frames=query_frames,
        operator_history=history,
        field=np.array(field, copy=True),
        displacement_l2=float(np.linalg.norm(displaced - borrowed)),
    )


def project_boundary_problem(
    execution: Any,
    coupling: np.ndarray,
    field: np.ndarray,
    label: str,
    *,
    query_frames: np.ndarray | None = None,
) -> Any:
    """The only scalar projection boundary; native evolution is already complete."""
    coupling, field = r4.validate_problem(coupling, field)
    queries = (
        execution.query_frames
        if query_frames is None
        else np.asarray(query_frames, dtype=np.complex128)
    )
    responses: list[complex] = []
    for site in range(r4.SITE_COUNT):
        denominator = float(
            np.linalg.norm(queries[site]) * np.linalg.norm(execution.displaced[site])
        )
        responses.append(
            complex(np.vdot(queries[site], execution.displaced[site]) / denominator)
        )
    coherence = tuple(float(abs(value)) for value in responses)
    phases = tuple(float(np.angle(value)) for value in responses)
    coherent = min(coherence) >= r4.QUERY_COHERENCE_MIN
    residual = r4.lock_residual(phases) if coherent else None
    valid = bool(
        coherent and residual is not None and residual <= r4.LOCK_RESIDUAL_MAX
    )
    raw_spins = tuple(1 if value.real >= 0.0 else -1 for value in responses)
    spins = raw_spins if valid else None
    energy = r4.ising_energy(spins, coupling, field) if spins is not None else None
    return r4.ResultLatch(
        label,
        tuple(responses),
        coherence,
        phases,
        raw_spins,
        spins,
        energy,
        residual,
        valid,
    )


def native_seal(execution: Any) -> dict[str, Any]:
    displaced = np.asarray(execution.displaced, dtype="<c16")
    history = np.asarray(execution.operator_history, dtype="<c16")
    return {
        "displaced_bytes_sha256": sha256_bytes(displaced.tobytes(order="C")),
        "displaced_shape": list(displaced.shape),
        "operator_history_bytes_sha256": sha256_bytes(history.tobytes(order="C")),
        "operator_history_shape": list(history.shape),
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


def native_call_path_proof() -> dict[str, Any]:
    functions: list[tuple[str, Callable[..., Any]]] = [
        ("adapter.execute_native_cycle_problem", execute_native_cycle_problem),
        ("r4.canonical_trees", r4.canonical_trees),
        ("r4.site_tree", r4.site_tree),
        ("r4.render_trees", r4.render_trees),
        ("r4.transport_mask_bank", r4.transport_mask_bank),
        ("r4.geometry_channel_bank", r4.geometry_channel_bank),
        ("r4.geometry_calibration", r4.geometry_calibration),
        ("r4.evolve_native_waveforms", r4.evolve_native_waveforms),
        ("r4.native_wave_step", r4.native_wave_step),
        ("r4.native_wave_velocity", r4.native_wave_velocity),
        ("r4.lock_strength", r4.lock_strength),
        ("r4.transport_shift", r4.transport_shift),
        ("r4.validate_problem", r4.validate_problem),
    ]
    forbidden = (
        "decoded",
        "raw_spin",
        "ising_energy",
        "oracle",
        "optimum",
        "winner",
        "expected_result",
        "external_latch",
        "scalar_spin_feedback",
        "argmin",
        "argmax",
    )
    findings: list[dict[str, str]] = []
    matrix_multiply_count = 0
    dynamic_evaluation_calls = 0
    source_hashes: dict[str, str] = {}
    for label, function in functions:
        source = inspect.getsource(function)
        source_hashes[label] = sha256_bytes(source.encode("utf-8"))
        lowered = source.lower()
        for fragment in forbidden:
            if fragment in lowered:
                findings.append({"function": label, "fragment": fragment})
        tree = ast.parse(source)
        matrix_multiply_count += sum(
            isinstance(node, ast.MatMult) for node in ast.walk(tree)
        )
        dynamic_evaluation_calls += sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec"}
            for node in ast.walk(tree)
        )
    return {
        "actual_transitive_path": [label for label, _ in functions],
        "dynamic_evaluation_calls": dynamic_evaluation_calls,
        "findings": findings,
        "matrix_multiply_count": matrix_multiply_count,
        "native_runtime_inputs": [
            "borrowed_complex_carrier",
            "coupling_matrix_J",
            "field_vector_h",
            "recursive_geometry",
            "frozen_machine_constants",
        ],
        "pass": not findings
        and matrix_multiply_count == 0
        and dynamic_evaluation_calls == 0,
        "source_hashes": source_hashes,
    }


def exact_adapter_equivalence(
    borrowed: np.ndarray, restored: np.ndarray, reuse_execution: Any
) -> dict[str, Any]:
    adapter_primary = execute_native_cycle_problem(
        borrowed, r4.COUPLING, r4.PRIMARY_FIELD
    )
    predecessor_primary = r4.execute_native_cycle(borrowed, r4.PRIMARY_FIELD)
    predecessor_reuse = r4.execute_native_cycle(restored, r4.REUSE_FIELD)

    comparisons = {
        "primary_displaced_max_delta": float(
            np.max(np.abs(adapter_primary.displaced - predecessor_primary.displaced))
        ),
        "primary_history_max_delta": float(
            np.max(
                np.abs(
                    adapter_primary.operator_history
                    - predecessor_primary.operator_history
                )
            )
        ),
        "primary_query_max_delta": float(
            np.max(
                np.abs(
                    adapter_primary.query_frames
                    - predecessor_primary.query_frames
                )
            )
        ),
        "reuse_displaced_max_delta": float(
            np.max(np.abs(reuse_execution.displaced - predecessor_reuse.displaced))
        ),
        "reuse_history_max_delta": float(
            np.max(
                np.abs(
                    reuse_execution.operator_history
                    - predecessor_reuse.operator_history
                )
            )
        ),
        "reuse_query_max_delta": float(
            np.max(
                np.abs(
                    reuse_execution.query_frames - predecessor_reuse.query_frames
                )
            )
        ),
    }
    return {
        "comparisons": {name: metric(value) for name, value in comparisons.items()},
        "pass": all(value == 0.0 for value in comparisons.values()),
    }


def response_delta(control: Any, reference: Any) -> float:
    control_values = np.asarray(control.responses, dtype=np.complex128)
    reference_values = np.asarray(reference.responses, dtype=np.complex128)
    return float(np.linalg.norm(control_values - reference_values))


def history_delta(control_execution: Any, reference_execution: Any) -> float:
    return float(
        np.linalg.norm(
            control_execution.operator_history - reference_execution.operator_history
        )
    )


def run_controls(
    borrowed: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    reference_execution: Any,
    reference_latch: Any,
) -> dict[str, Any]:
    canonical = reference_execution.program_beams
    no_transform_execution = execute_native_cycle_problem(
        borrowed,
        coupling,
        field,
        actual_beams=np.ones_like(canonical),
    )
    no_transform = project_boundary_problem(
        no_transform_execution, coupling, field, "no_transform"
    )

    flat = r4.flat_replacement_beams(r4.canonical_trees())
    flat_execution = execute_native_cycle_problem(
        borrowed,
        coupling,
        field,
        program_beams=flat,
        actual_beams=flat,
        calibration_beams=flat,
    )
    flat_latch = project_boundary_problem(flat_execution, coupling, field, "flat")

    scrambled = r4.render_trees(r4.scrambled_trees())
    scrambled_execution = execute_native_cycle_problem(
        borrowed,
        coupling,
        field,
        program_beams=scrambled,
        actual_beams=scrambled,
        calibration_beams=scrambled,
    )
    scrambled_latch = project_boundary_problem(
        scrambled_execution, coupling, field, "scrambled"
    )

    enabled = np.ones((r4.SITE_COUNT, r4.SITE_COUNT), dtype=np.bool_)
    enabled[0, 4] = False
    enabled[4, 0] = False
    missing_execution = execute_native_cycle_problem(
        borrowed, coupling, field, relation_enabled=enabled
    )
    missing_latch = project_boundary_problem(
        missing_execution, coupling, field, "missing_phase_operator"
    )
    no_transform_history_change = history_delta(
        no_transform_execution, reference_execution
    )
    flat_history_change = history_delta(flat_execution, reference_execution)
    scrambled_history_change = history_delta(
        scrambled_execution, reference_execution
    )
    missing_history_change = history_delta(missing_execution, reference_execution)

    no_lock_execution = execute_native_cycle_problem(
        borrowed, coupling, field, lock_final=0.0
    )
    no_lock_latch = project_boundary_problem(
        no_lock_execution, coupling, field, "no_lock"
    )

    wrong_initial = borrowed * scrambled
    wrong_query_frames = r4.transport_query_bank(
        wrong_initial, reference_execution.masks
    )
    wrong_query_latch = project_boundary_problem(
        reference_execution,
        coupling,
        field,
        "wrong_query",
        query_frames=wrong_query_frames,
    )

    uniform = np.ones_like(borrowed)
    uniform_execution = execute_native_cycle_problem(uniform, coupling, field)
    uniform_latch = project_boundary_problem(
        uniform_execution, coupling, field, "uniform_carrier"
    )
    carrier_history_change = history_delta(uniform_execution, reference_execution)
    no_lock_history_change = history_delta(no_lock_execution, reference_execution)

    response_deltas = {
        "flat_geometry": response_delta(flat_latch, reference_latch),
        "missing_phase_operator": response_delta(missing_latch, reference_latch),
        "no_lock": response_delta(no_lock_latch, reference_latch),
        "no_transform": response_delta(no_transform, reference_latch),
        "scrambled_geometry": response_delta(scrambled_latch, reference_latch),
        "uniform_carrier": response_delta(uniform_latch, reference_latch),
        "wrong_query": response_delta(wrong_query_latch, reference_latch),
    }

    rank_one_residual = float(
        np.max(
            np.abs(
                reference_execution.operator_history
                - np.mean(
                    reference_execution.operator_history, axis=2, keepdims=True
                )
            )
        )
    )
    wrong_inverse_error = float(
        np.max(np.abs(r4.restore_carrier(reference_execution, "wrong_order") - borrowed))
    )
    omitted_step_error = float(
        np.max(np.abs(r4.restore_carrier(reference_execution, "omit_middle") - borrowed))
    )
    omitted_restore_error = float(
        np.max(np.abs(r4.restore_carrier(reference_execution, "omitted") - borrowed))
    )

    outcomes = {
        "carrier_content_causal": carrier_history_change
        >= r4.OPERATOR_HISTORY_CHANGE_MIN
        and response_deltas["uniform_carrier"]
        >= r4.OPERATOR_HISTORY_CHANGE_MIN,
        "flat_geometry_changed_or_destroyed": (
            flat_history_change >= r4.OPERATOR_HISTORY_CHANGE_MIN
            and response_deltas["flat_geometry"]
            >= r4.OPERATOR_HISTORY_CHANGE_MIN
        ),
        "missing_phase_operator_changed_or_destroyed": (
            missing_history_change >= r4.OPERATOR_HISTORY_CHANGE_MIN
            and response_deltas["missing_phase_operator"]
            >= r4.OPERATOR_HISTORY_CHANGE_MIN
        ),
        "no_lock_changed_or_destroyed": (
            no_lock_history_change >= r4.OPERATOR_HISTORY_CHANGE_MIN
            and response_deltas["no_lock"] >= r4.OPERATOR_HISTORY_CHANGE_MIN
        ),
        "no_transform_changed_or_destroyed": (
            no_transform_history_change >= r4.OPERATOR_HISTORY_CHANGE_MIN
            and response_deltas["no_transform"]
            >= r4.OPERATOR_HISTORY_CHANGE_MIN
        ),
        "omitted_inverse_step_failed": omitted_step_error >= r4.WRONG_RESTORE_MIN,
        "omitted_restoration_failed": omitted_restore_error >= r4.WRONG_RESTORE_MIN,
        "samplewise_non_rank_one": rank_one_residual >= r4.SAMPLEWISE_DYNAMICS_MIN,
        "scrambled_geometry_changed_or_destroyed": (
            scrambled_history_change >= r4.OPERATOR_HISTORY_CHANGE_MIN
            and response_deltas["scrambled_geometry"]
            >= r4.OPERATOR_HISTORY_CHANGE_MIN
        ),
        "wrong_inverse_failed": wrong_inverse_error >= r4.WRONG_RESTORE_MIN,
        "wrong_query_changed_or_destroyed": (
            response_deltas["wrong_query"] >= r4.OPERATOR_HISTORY_CHANGE_MIN
        ),
    }
    return {
        "all_pass": all(outcomes.values()),
        "measurements": {
            "carrier_content_history_change_l2": metric(carrier_history_change),
            "flat_geometry_history_change_l2": metric(flat_history_change),
            "missing_phase_operator_history_change_l2": metric(
                missing_history_change
            ),
            "no_lock_history_change_l2": metric(no_lock_history_change),
            "no_transform_history_change_l2": metric(
                no_transform_history_change
            ),
            "omitted_inverse_step_error": metric(omitted_step_error),
            "omitted_restoration_error": metric(omitted_restore_error),
            "samplewise_non_rank_one_residual": metric(rank_one_residual),
            "scrambled_geometry_history_change_l2": metric(
                scrambled_history_change
            ),
            "wrong_inverse_error": metric(wrong_inverse_error),
            "response_deltas_l2": {
                name: metric(value) for name, value in response_deltas.items()
            },
        },
        "outcomes": outcomes,
        "latches": {
            "flat_geometry": flat_latch.document(),
            "missing_phase_operator": missing_latch.document(),
            "no_lock": no_lock_latch.document(),
            "no_transform": no_transform.document(),
            "scrambled_geometry": scrambled_latch.document(),
            "uniform_carrier": uniform_latch.document(),
            "wrong_query": wrong_query_latch.document(),
        },
    }


def oracle_document(
    latch: Any, coupling: np.ndarray, field: np.ndarray
) -> dict[str, Any]:
    rows = r4.exact_oracle(coupling, field)
    optimum_energy, optimum_spins = rows[0]
    unique = sum(
        abs(energy - optimum_energy) <= r4.ENERGY_TOL for energy, _ in rows
    ) == 1
    next_energy = next(
        energy for energy, spins in rows[1:] if spins != optimum_spins
    )
    agrees = bool(
        latch.valid
        and latch.spins == optimum_spins
        and latch.energy is not None
        and abs(latch.energy - optimum_energy) <= r4.ENERGY_TOL
    )
    return {
        "agrees": agrees,
        "gap": metric(next_energy - optimum_energy),
        "optimum_energy": metric(optimum_energy),
        "optimum_spins": list(optimum_spins),
        "row_count": len(rows),
        "unique": unique,
    }


def qualification() -> dict[str, Any]:
    custody = custody_document()
    coupling, field = problem_from_custody()
    machine = frozen_machine_document()
    call_path = native_call_path_proof()
    ledger = EventLedger([])
    ledger.add(
        "custody_verified",
        {
            "freeze_commit": FREEZE_COMMIT,
            "instance_sha256": INSTANCE_SHA256,
        },
    )

    carrier = r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], r4.SITE_COUNT, axis=0)
    heldout_execution = execute_native_cycle_problem(
        borrowed, coupling, field
    )
    heldout_native_seal = native_seal(heldout_execution)
    ledger.add("heldout_native_complete_and_raw_waveform_sealed", heldout_native_seal)

    heldout_latch = project_boundary_problem(
        heldout_execution, coupling, field, "held_out"
    )
    heldout_raw_spin_energy = r4.ising_energy(
        heldout_latch.raw_spins, coupling, field
    )
    heldout_latch_digest = heldout_latch.digest()
    ledger.add(
        "heldout_boundary_projected_and_latch_sealed",
        {
            "latch": heldout_latch.document(),
            "raw_spin_energy": metric(heldout_raw_spin_energy),
        },
    )

    restored = r4.restore_carrier(heldout_execution, "correct")
    restoration_error = float(np.max(np.abs(restored - borrowed)))
    ledger.add("heldout_carrier_restored", {"max_abs_error": restoration_error})

    reuse_execution = execute_native_cycle_problem(
        restored, r4.COUPLING, r4.REUSE_FIELD
    )
    reuse_native_seal = native_seal(reuse_execution)
    ledger.add("reuse_native_complete_and_raw_waveform_sealed", reuse_native_seal)
    reuse_latch = project_boundary_problem(
        reuse_execution, r4.COUPLING, r4.REUSE_FIELD, "reuse"
    )
    ledger.add("reuse_boundary_projected_and_latch_sealed", reuse_latch.document())
    reuse_restored = r4.restore_carrier(reuse_execution, "correct")
    reuse_restoration_error = float(np.max(np.abs(reuse_restored - restored)))
    reuse_input_error = float(np.max(np.abs(reuse_execution.borrowed - restored)))
    ledger.add(
        "reuse_carrier_restored",
        {
            "input_max_abs_error": reuse_input_error,
            "restoration_max_abs_error": reuse_restoration_error,
        },
    )

    controls = run_controls(
        borrowed, coupling, field, heldout_execution, heldout_latch
    )
    ledger.add("negative_controls_sealed", controls)

    adapter_equivalence = exact_adapter_equivalence(
        borrowed, restored, reuse_execution
    )
    ledger.add("exact_predecessor_adapter_equivalence_sealed", adapter_equivalence)

    ledger.add("oracle_opened_after_reuse_and_controls", {"instance": INSTANCE_SHA256})
    heldout_oracle = oracle_document(heldout_latch, coupling, field)
    reuse_oracle = oracle_document(reuse_latch, r4.COUPLING, r4.REUSE_FIELD)
    ledger.add(
        "oracle_adjudication_complete",
        {"heldout": heldout_oracle, "reuse": reuse_oracle},
    )

    trace = ledger.document()
    expected_order = [
        "custody_verified",
        "heldout_native_complete_and_raw_waveform_sealed",
        "heldout_boundary_projected_and_latch_sealed",
        "heldout_carrier_restored",
        "reuse_native_complete_and_raw_waveform_sealed",
        "reuse_boundary_projected_and_latch_sealed",
        "reuse_carrier_restored",
        "negative_controls_sealed",
        "exact_predecessor_adapter_equivalence_sealed",
        "oracle_opened_after_reuse_and_controls",
        "oracle_adjudication_complete",
    ]
    oracle_order_pass = [event["name"] for event in trace["events"]] == expected_order
    heldout_result_persisted = heldout_latch.digest() == heldout_latch_digest

    foundational_tests = {
        "custody_frozen_before_execution_and_oracle": (
            custody["held_out_instance_sha256"] == INSTANCE_SHA256
            and not any(custody["freeze_order"].values())
        ),
        "frozen_machine_source_exact": (
            machine["predecessor_source_sha256"] == PREDECESSOR_SOURCE_SHA256
        ),
        "native_transitive_call_path_no_smuggle": call_path["pass"],
        "variable_J_adapter_exact_on_verified_instances": adapter_equivalence["pass"],
        "oracle_order_after_reuse_and_controls": oracle_order_pass,
        "heldout_carrier_displaced": (
            heldout_execution.displacement_l2 >= r4.DISPLACEMENT_MIN
        ),
        "heldout_carrier_restored": restoration_error <= r4.RESTORE_TOL,
        "heldout_latch_persisted_outside_inverse": heldout_result_persisted,
        "reuse_consumed_actual_restored_carrier": reuse_input_error == 0.0,
        "reuse_result_valid": reuse_latch.valid and reuse_oracle["agrees"],
        "reuse_carrier_restored": reuse_restoration_error <= r4.RESTORE_TOL,
        "required_controls_pass": controls["all_pass"],
    }
    generalization_tests = {
        "heldout_projection_valid": heldout_latch.valid,
        "heldout_oracle_unique": heldout_oracle["unique"],
        "heldout_projection_matches_unique_optimum": (
            heldout_oracle["unique"] and heldout_oracle["agrees"]
        ),
    }
    foundational_pass = all(foundational_tests.values())
    generalization_pass = all(generalization_tests.values())
    if foundational_pass and generalization_pass:
        decision = VERIFIED
    elif foundational_pass and heldout_oracle["unique"]:
        decision = PARTIAL
    else:
        decision = NOT_ESTABLISHED

    all_tests = {**foundational_tests, **generalization_tests}
    return {
        "claim_ceiling": CLAIM_CEILING,
        "adapter_equivalence": adapter_equivalence,
        "controls": controls,
        "decision": decision,
        "failures": [name for name, passed in all_tests.items() if not passed],
        "freeze_commit": FREEZE_COMMIT,
        "frozen_machine_sha256": sha256_bytes(canonical_bytes(machine)),
        "held_out_instance": {
            "coupling_matrix_J": coupling.tolist(),
            "field_vector_h": field.tolist(),
            "sha256": INSTANCE_SHA256,
        },
        "heldout_latch": heldout_latch.document(),
        "heldout_native_seal": heldout_native_seal,
        "heldout_oracle": heldout_oracle,
        "measurements": {
            "heldout_carrier_displacement_l2": metric(
                heldout_execution.displacement_l2
            ),
            "heldout_restoration_max_error": metric(restoration_error),
            "raw_spin_energy": metric(heldout_raw_spin_energy),
            "reuse_carrier_displacement_l2": metric(
                reuse_execution.displacement_l2
            ),
            "reuse_input_max_error_from_restored": metric(reuse_input_error),
            "reuse_restoration_max_error": metric(reuse_restoration_error),
        },
        "native_call_path_proof": call_path,
        "reuse_latch": reuse_latch.document(),
        "reuse_native_seal": reuse_native_seal,
        "reuse_oracle": reuse_oracle,
        "schema": RESULT_SCHEMA,
        "test_count": len(all_tests),
        "test_pass_count": sum(all_tests.values()),
        "tests": all_tests,
        "trace": trace,
    }


def manifest_document() -> dict[str, Any]:
    paths = [
        PACKAGE_DIR / CONTRACT_FILE,
        PACKAGE_DIR / TRACE_FILE,
        PACKAGE_DIR / HELDOUT_LATCH_FILE,
        PACKAGE_DIR / REUSE_LATCH_FILE,
    ]
    records = []
    for path in paths:
        records.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(PACKAGE_DIR).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    root_payload = "".join(
        f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n"
        for row in sorted(records, key=lambda item: item["path"])
    ).encode("utf-8")
    return {
        "file_count": len(records),
        "files": records,
        "fixture_root_sha256": sha256_bytes(root_payload),
        "schema": MANIFEST_SCHEMA,
        "total_bytes": sum(record["bytes"] for record in records),
    }


def source_binding() -> dict[str, Any]:
    payload = Path(__file__).resolve().read_bytes()
    return {"source_bytes": len(payload), "source_sha256": sha256_bytes(payload)}


def build_package() -> dict[str, Any]:
    write_json(PACKAGE_DIR / CONTRACT_FILE, contract_document())
    result = qualification()
    write_json(PACKAGE_DIR / TRACE_FILE, result["trace"])
    write_json(PACKAGE_DIR / HELDOUT_LATCH_FILE, result["heldout_latch"])
    write_json(PACKAGE_DIR / REUSE_LATCH_FILE, result["reuse_latch"])
    manifest = manifest_document()
    write_json(PACKAGE_DIR / MANIFEST_FILE, manifest)
    result.pop("trace")
    result["execution_trace_sha256"] = sha256_file(PACKAGE_DIR / TRACE_FILE)
    result["fixture_manifest_sha256"] = sha256_file(PACKAGE_DIR / MANIFEST_FILE)
    result["source"] = source_binding()
    write_json(PACKAGE_DIR / RESULTS_FILE, result)
    return result


def verify_package() -> dict[str, Any]:
    if (PACKAGE_DIR / CONTRACT_FILE).read_bytes() != canonical_bytes(contract_document()):
        raise ValueError("committed held-out contract does not reproduce")
    result = qualification()
    if (PACKAGE_DIR / TRACE_FILE).read_bytes() != canonical_bytes(result["trace"]):
        raise ValueError("committed execution trace does not reproduce")
    if (PACKAGE_DIR / HELDOUT_LATCH_FILE).read_bytes() != canonical_bytes(
        result["heldout_latch"]
    ):
        raise ValueError("committed held-out latch does not reproduce")
    if (PACKAGE_DIR / REUSE_LATCH_FILE).read_bytes() != canonical_bytes(
        result["reuse_latch"]
    ):
        raise ValueError("committed reuse latch does not reproduce")
    manifest = manifest_document()
    if (PACKAGE_DIR / MANIFEST_FILE).read_bytes() != canonical_bytes(manifest):
        raise ValueError("committed manifest does not reproduce")
    result.pop("trace")
    result["execution_trace_sha256"] = sha256_file(PACKAGE_DIR / TRACE_FILE)
    result["fixture_manifest_sha256"] = sha256_file(PACKAGE_DIR / MANIFEST_FILE)
    result["source"] = source_binding()
    if (PACKAGE_DIR / RESULTS_FILE).read_bytes() != canonical_bytes(result):
        raise ValueError("committed held-out results do not reproduce")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify", "self-test"))
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_package()
    elif args.command == "verify":
        result = verify_package()
    else:
        result = qualification()
        result.pop("trace")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "failures": result["failures"],
                "heldout_raw_spins": result["heldout_latch"]["raw_spin_shadow"],
                "test_count": result["test_count"],
                "test_pass_count": result["test_pass_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["decision"] != NOT_ESTABLISHED else 1


if __name__ == "__main__":
    raise SystemExit(main())
