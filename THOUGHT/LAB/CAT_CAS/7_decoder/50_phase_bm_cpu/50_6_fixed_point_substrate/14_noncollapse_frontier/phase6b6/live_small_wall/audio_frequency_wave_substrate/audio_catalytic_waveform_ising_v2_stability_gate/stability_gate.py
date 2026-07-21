from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
V2_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v2"
V2_SOURCE = V2_DIR / "successor_machine.py"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v2 = load_module(V2_SOURCE, "catcas_waveform_ising_v2_stability_gate_frozen")


CHECKPOINT_STEPS = (980, 985, 990, 995, 1000, 1001, 1002, 1003, 1004)
MAX_LATE_PHASE_VELOCITY_RAD_PER_STEP = 0.008
MAX_LATE_MEAN_RESPONSE_DRIFT_L2 = 0.08
REPLAY_TOLERANCE = 1.0e-12
DIAGNOSTIC_RESTORATION_MAX = 2.0e-12


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def complex_responses(states: np.ndarray, frames: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.complex128)
    frames = np.asarray(frames, dtype=np.complex128)
    if states.shape != (v2.SITE_COUNT, v2.SAMPLE_COUNT) or frames.shape != states.shape:
        raise ValueError("waveform response bank has the wrong shape")
    responses: list[complex] = []
    for site in range(v2.SITE_COUNT):
        denominator = float(np.linalg.norm(states[site]) * np.linalg.norm(frames[site]))
        if denominator <= 0.0:
            raise ValueError("waveform response denominator must be positive")
        responses.append(complex(np.vdot(frames[site], states[site]) / denominator))
    values = np.asarray(responses, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError("waveform response must be finite")
    return values


def wrapped_phase_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    return np.asarray(v2.wrap_phase(np.angle(left) - np.angle(right)), dtype=np.float64)


def replay_nominal_diagnostic(execution: Any) -> tuple[dict[int, np.ndarray], float, float]:
    """Replay sealed complex operators, sample fixed late checkpoints, then reverse."""
    if execution.law.step_count != 1005:
        raise ValueError("diagnostic requires the unchanged frozen V2 step count")
    if CHECKPOINT_STEPS[-1] != execution.law.step_count - 1:
        raise ValueError("diagnostic final checkpoint does not bind the V2 boundary")
    states = (
        np.asarray(execution.borrowed, dtype=np.complex128)
        * np.asarray(execution.actual_beams, dtype=np.complex128)
        * np.exp(1j * v2.INITIAL_PHASES[:, np.newaxis])
    )
    frames = (
        np.asarray(execution.borrowed, dtype=np.complex128)
        * np.asarray(execution.program_beams, dtype=np.complex128)
    )
    checkpoints: dict[int, np.ndarray] = {}
    for step in range(execution.law.step_count):
        arm = step % execution.masks.shape[0]
        next_states = np.empty_like(states)
        next_frames = np.empty_like(frames)
        for site in range(v2.SITE_COUNT):
            shift = v2.transport_shift(step, site)
            next_states[site] = np.roll(
                states[site]
                * execution.operator_history[step, site]
                * execution.masks[arm, site],
                shift,
            )
            next_frames[site] = np.roll(
                frames[site] * execution.masks[arm, site], shift
            )
        states, frames = next_states, next_frames
        if step in CHECKPOINT_STEPS:
            checkpoints[step] = complex_responses(states, frames)
    if tuple(checkpoints) != CHECKPOINT_STEPS:
        raise RuntimeError("diagnostic checkpoint schedule was not executed exactly")
    displaced_delta = float(np.max(np.abs(states - execution.displaced)))
    query_delta = float(np.max(np.abs(frames - execution.query_frames)))
    if max(displaced_delta, query_delta) > REPLAY_TOLERANCE:
        raise RuntimeError("zero-perturbation diagnostic did not reproduce nominal V2")
    diagnostic_execution = replace(
        execution,
        displaced=np.array(states, copy=True),
        query_frames=np.array(frames, copy=True),
    )
    restored = v2.restore_carrier(diagnostic_execution)
    restoration_error = float(
        np.max(np.abs(restored - np.asarray(execution.borrowed, dtype=np.complex128)))
    )
    if restoration_error > DIAGNOSTIC_RESTORATION_MAX:
        raise RuntimeError("diagnostic carrier restoration exceeded the frozen gate")
    return checkpoints, max(displaced_delta, query_delta), restoration_error


@dataclass(frozen=True)
class StabilityDecision:
    gate_pass: bool
    joint_instability_score: float
    late_max_phase_velocity_rad_per_step: float
    late_mean_response_drift_l2: float
    checkpoint_response_sha256: str
    diagnostic_replay_max_abs_delta: float
    diagnostic_restoration_max_abs_error: float

    def document(self) -> dict[str, Any]:
        return {
            "checkpoint_response_sha256": self.checkpoint_response_sha256,
            "diagnostic_replay_max_abs_delta": metric(
                self.diagnostic_replay_max_abs_delta
            ),
            "diagnostic_restoration_max_abs_error": metric(
                self.diagnostic_restoration_max_abs_error
            ),
            "gate_pass": self.gate_pass,
            "joint_instability_score": metric(self.joint_instability_score),
            "late_max_phase_velocity_rad_per_step": metric(
                self.late_max_phase_velocity_rad_per_step
            ),
            "late_mean_response_drift_l2": metric(
                self.late_mean_response_drift_l2
            ),
        }


def evaluate_stability(execution: Any) -> StabilityDecision:
    checkpoints, replay_delta, restoration_error = replay_nominal_diagnostic(execution)
    ordered = [(step, checkpoints[step]) for step in CHECKPOINT_STEPS]
    phase_velocities = [
        np.abs(wrapped_phase_delta(right, left)) / float(right_step - left_step)
        for (left_step, left), (right_step, right) in zip(ordered, ordered[1:])
    ]
    max_velocity = float(max(np.max(values) for values in phase_velocities))
    final_response = ordered[-1][1]
    drifts = [float(np.linalg.norm(values - final_response)) for _, values in ordered[:-1]]
    mean_drift = float(np.mean(drifts))
    score = min(
        max_velocity / MAX_LATE_PHASE_VELOCITY_RAD_PER_STEP,
        mean_drift / MAX_LATE_MEAN_RESPONSE_DRIFT_L2,
    )
    gate_pass = bool(score <= 1.0)
    checkpoint_document = [
        {
            "response": [
                {"imag": metric(value.imag), "real": metric(value.real)}
                for value in responses
            ],
            "step": step,
        }
        for step, responses in ordered
    ]
    return StabilityDecision(
        gate_pass=gate_pass,
        joint_instability_score=score,
        late_max_phase_velocity_rad_per_step=max_velocity,
        late_mean_response_drift_l2=mean_drift,
        checkpoint_response_sha256=sha256_bytes(canonical_bytes(checkpoint_document)),
        diagnostic_replay_max_abs_delta=replay_delta,
        diagnostic_restoration_max_abs_error=restoration_error,
    )


def gate_contract() -> dict[str, Any]:
    return {
        "acceptance_law": (
            "PASS when min(late_max_phase_velocity/0.008, "
            "late_mean_response_drift/0.08) <= 1; otherwise REJECT"
        ),
        "checkpoint_steps_zero_based_after_update": list(CHECKPOINT_STEPS),
        "diagnostic_restoration_max": DIAGNOSTIC_RESTORATION_MAX,
        "max_late_mean_response_drift_l2": MAX_LATE_MEAN_RESPONSE_DRIFT_L2,
        "max_late_phase_velocity_rad_per_step": MAX_LATE_PHASE_VELOCITY_RAD_PER_STEP,
        "null_baseline": (
            "disabling this reject-only gate reproduces the unchanged nominal V2 "
            "acceptance decision and raw result"
        ),
        "raw_result_effect": "REJECT_ONLY_NEVER_ALTERS_OR_SELECTS_A_RESULT",
        "replay_tolerance": REPLAY_TOLERANCE,
    }
