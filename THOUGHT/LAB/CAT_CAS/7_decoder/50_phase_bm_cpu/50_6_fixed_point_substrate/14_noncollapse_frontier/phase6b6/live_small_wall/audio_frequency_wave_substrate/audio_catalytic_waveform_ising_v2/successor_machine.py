from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
PREDECESSOR_SOURCE = (
    SUBSTRATE_DIR
    / "audio_integrated_catalytic_computation_v1"
    / "integrated_catalytic_computation_reference.py"
)
HELDOUT_CUSTODY = (
    SUBSTRATE_DIR
    / "audio_catalytic_waveform_ising_heldout_v1"
    / "HELD_OUT_INSTANCE_CUSTODY.json"
)
BATCH_CUSTODY = (
    SUBSTRATE_DIR
    / "audio_catalytic_waveform_ising_batch_v1"
    / "BATCH_INSTANCE_CUSTODY.json"
)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


r4 = load_module(PREDECESSOR_SOURCE, "catcas_waveform_ising_v2_predecessor")

SITE_COUNT = r4.SITE_COUNT
SAMPLE_COUNT = r4.SAMPLE_COUNT
INITIAL_PHASES = np.array(r4.INITIAL_PHASES, dtype=np.float64, copy=True)

COHERENCE_MIN = 0.90
LOCK_RESIDUAL_MAX = 0.15
RESTORATION_MAX = 2.0e-12
WRONG_RESTORATION_MIN = 1.0e-3
MATERIALITY_MIN = 1.0e-3
SAMPLEWISE_MIN = 1.0e-3
DISPLACEMENT_MIN = 1.0
CLAIM_CEILING = "BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY"


@dataclass(frozen=True)
class MachineLaw:
    step_count: int = 1005
    solve_step_count: int = 1000
    time_step: float = 0.03
    interaction_floor: float = 0.0
    lock_initial: float = 0.0
    lock_solve: float = 1.2
    lock_schedule_power: float = 1.0
    lock_final: float = 1.3
    coherence_initial: float = 4.0
    coherence_final: float = 4.2
    spatial_initial: float = 0.6
    spatial_final: float = 0.65
    carrier_power_exponent: float = 0.90
    geometry_phase_scale: float = 0.50
    transform_channel_weight: float = 1.0

    def validate(self) -> None:
        if self.step_count < 3:
            raise ValueError("step_count must be at least three")
        if not 2 <= self.solve_step_count < self.step_count:
            raise ValueError("solve_step_count must leave a consolidation interval")
        scalars = asdict(self)
        if not all(math.isfinite(float(value)) for value in scalars.values()):
            raise ValueError("machine law contains a nonfinite value")
        if self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        if not 0.0 <= self.interaction_floor <= 1.0:
            raise ValueError("interaction_floor outside [0, 1]")
        if min(
            self.lock_initial,
            self.lock_solve,
            self.lock_final,
            self.lock_schedule_power,
            self.coherence_initial,
            self.coherence_final,
            self.spatial_initial,
            self.spatial_final,
            self.carrier_power_exponent,
            self.geometry_phase_scale,
            self.transform_channel_weight,
        ) < 0.0:
            raise ValueError("machine law strengths must be nonnegative")


DEFAULT_LAW = MachineLaw()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def wrap_phase(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value, dtype=np.float64)
    wrapped = (array + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped) if np.ndim(value) == 0 else wrapped


def lock_residual(phases: Sequence[float]) -> float:
    values = np.asarray(phases, dtype=np.float64)
    zero = np.abs(wrap_phase(values))
    pi = np.abs(wrap_phase(values - math.pi))
    return float(np.max(np.minimum(zero, pi)))


def smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, float(value)))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def schedule(
    step: int, law: MachineLaw
) -> tuple[float, float, float, float, float, float]:
    if not 0 <= step < law.step_count:
        raise ValueError("step outside frozen schedule")
    if step < law.solve_step_count:
        interaction = 1.0
        solve_fraction = step / (law.solve_step_count - 1)
        lock = law.lock_initial + (law.lock_solve - law.lock_initial) * (
            solve_fraction ** law.lock_schedule_power
        )
        coherence = law.coherence_initial
        spatial = law.spatial_initial
        power_exponent = 1.0
        weighted_coherence_fraction = 0.0
    else:
        settle_step = step - law.solve_step_count
        settle_count = law.step_count - law.solve_step_count
        settle = smoothstep(
            settle_step / (settle_count - 1)
        )
        interaction = 1.0 + (law.interaction_floor - 1.0) * settle
        lock = law.lock_solve + (law.lock_final - law.lock_solve) * settle
        coherence = law.coherence_initial + (
            law.coherence_final - law.coherence_initial
        ) * settle
        spatial = law.spatial_initial + (law.spatial_final - law.spatial_initial) * settle
        power_exponent = 1.0 + (law.carrier_power_exponent - 1.0) * settle
        weighted_coherence_fraction = settle
    return (
        float(interaction),
        float(lock),
        float(coherence),
        float(spatial),
        float(power_exponent),
        float(weighted_coherence_fraction),
    )


@dataclass(frozen=True)
class NativeExecution:
    borrowed: np.ndarray
    coupling: np.ndarray
    field: np.ndarray
    actual_beams: np.ndarray
    program_beams: np.ndarray
    masks: np.ndarray
    displaced: np.ndarray
    query_frames: np.ndarray
    operator_history: np.ndarray
    displacement_l2: float
    law: MachineLaw


@dataclass(frozen=True)
class BoundaryProjection:
    label: str
    responses: tuple[complex, ...]
    coherence: tuple[float, ...]
    phases: tuple[float, ...]
    raw_spins: tuple[int, ...]
    spins: tuple[int, ...] | None
    energy: float | None
    residual: float | None
    valid: bool

    def document(self) -> dict[str, Any]:
        return {
            "coherence": [metric(value) for value in self.coherence],
            "energy": None if self.energy is None else metric(self.energy),
            "label": self.label,
            "phases_rad": [metric(value) for value in self.phases],
            "raw_spin_shadow": list(self.raw_spins),
            "residual_rad": None if self.residual is None else metric(self.residual),
            "responses": [
                {"imag": metric(value.imag), "real": metric(value.real)}
                for value in self.responses
            ],
            "spins": None if self.spins is None else list(self.spins),
            "valid": self.valid,
        }


def validate_problem(
    coupling: np.ndarray, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return r4.validate_problem(coupling, field)


def transport_shift(step: int, site: int) -> int:
    if not 0 <= site < SITE_COUNT:
        raise ValueError("site outside frozen transport schedule")
    return int(r4.TRANSPORT_SHIFTS[step % len(r4.TRANSPORT_SHIFTS)])


def transform_channel_bank(
    program_beams: np.ndarray, actual_beams: np.ndarray, law: MachineLaw
) -> np.ndarray:
    """Bind recurrent pair interactions to the waveform transform itself."""
    program_channels = np.exp(
        1j * law.geometry_phase_scale * np.angle(r4.geometry_channel_bank(program_beams))
    )
    actual_channels = np.exp(
        1j * law.geometry_phase_scale * np.angle(r4.geometry_channel_bank(actual_beams))
    )
    if law.transform_channel_weight == 1.0:
        return actual_channels
    phase = (
        (1.0 - law.transform_channel_weight) * np.angle(program_channels)
        + law.transform_channel_weight * np.angle(actual_channels)
    )
    return np.exp(1j * phase)


def channel_calibration(channels: np.ndarray, borrowed: np.ndarray) -> np.ndarray:
    power = np.abs(borrowed[0]) ** 2
    denominator = float(np.sum(power))
    calibration = np.ones((SITE_COUNT, SITE_COUNT), dtype=np.complex128)
    for site in range(SITE_COUNT):
        for neighbor in range(SITE_COUNT):
            if site == neighbor:
                continue
            mean_channel = complex(
                np.sum(power * channels[site, neighbor]) / denominator
            )
            if abs(mean_channel) < 0.05:
                raise ValueError("transform channel mean is ill-conditioned")
            calibration[site, neighbor] = 1.0 / mean_channel
    return calibration


def native_wave_velocity(
    states: np.ndarray,
    frames: np.ndarray,
    transform_channels: np.ndarray,
    calibration: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    relation_enabled: np.ndarray,
    interaction_gain: float,
    lock_gain: float,
    coherence_gain: float,
    spatial_gain: float,
    carrier_power_exponent: float,
    weighted_coherence_fraction: float,
) -> np.ndarray:
    """Waveform-only phase velocity; no scalar result is decoded here."""
    coupling, field = validate_problem(coupling, field)
    states = np.asarray(states, dtype=np.complex128)
    frames = np.asarray(frames, dtype=np.complex128)
    channels = np.asarray(transform_channels, dtype=np.complex128)
    enabled = np.asarray(relation_enabled, dtype=np.bool_)
    if states.shape != (SITE_COUNT, SAMPLE_COUNT) or frames.shape != states.shape:
        raise ValueError("native waveform state shape mismatch")
    if channels.shape != (SITE_COUNT, SITE_COUNT, SAMPLE_COUNT):
        raise ValueError("transform channel shape mismatch")
    if calibration.shape != (SITE_COUNT, SITE_COUNT):
        raise ValueError("transform calibration shape mismatch")
    if enabled.shape != (SITE_COUNT, SITE_COUNT):
        raise ValueError("phase-operator enable map shape mismatch")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(frames)):
        raise ValueError("native waveform state must be finite")
    frame_power = np.abs(frames) ** 2
    if np.min(frame_power) <= 0.0:
        raise ValueError("native reference frame has a zero coordinate")
    velocity = np.zeros((SITE_COUNT, SAMPLE_COUNT), dtype=np.float64)
    for site in range(SITE_COUNT):
        drive = field[site] * frames[site]
        for neighbor in range(SITE_COUNT):
            if site == neighbor:
                continue
            if enabled[site, neighbor]:
                frame_relation = (
                    frames[site]
                    * np.conjugate(frames[neighbor])
                    / frame_power[neighbor]
                )
                incoming = (
                    calibration[site, neighbor]
                    * channels[site, neighbor]
                    * frame_relation
                    * states[neighbor]
                )
            else:
                incoming = states[neighbor]
            drive = drive + coupling[site, neighbor] * incoming

        orientation = states[site] / frames[site]
        mean_power = float(np.mean(frame_power[site]))
        power_condition = (frame_power[site] / mean_power) ** carrier_power_exponent
        local_flow = (
            np.imag(np.conjugate(states[site]) * drive)
            / frame_power[site]
            * power_condition
        )
        left = np.roll(orientation, 1)
        right = np.roll(orientation, -1)
        spatial_flow = np.imag(np.conjugate(orientation) * (left + right))
        unweighted_mean = complex(np.mean(orientation))
        weighted_mean = complex(
            np.sum(frame_power[site] * orientation) / np.sum(frame_power[site])
        )
        coherence_target = (
            (1.0 - weighted_coherence_fraction) * unweighted_mean
            + weighted_coherence_fraction * weighted_mean
        )
        coherence_flow = np.imag(np.conjugate(orientation) * coherence_target)
        lock_flow = -np.sin(2.0 * np.angle(orientation))
        velocity[site] = (
            interaction_gain * local_flow
            + spatial_gain * spatial_flow
            + coherence_gain * coherence_flow
            + lock_gain * lock_flow * power_condition
        )
    return velocity


def evolve_native_waveforms(
    states: np.ndarray,
    frames: np.ndarray,
    transform_channels: np.ndarray,
    masks: np.ndarray,
    calibration: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    relation_enabled: np.ndarray,
    law: MachineLaw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete native evolution; the scalar boundary is unreachable here."""
    law.validate()
    current_states = np.array(states, dtype=np.complex128, copy=True)
    current_frames = np.array(frames, dtype=np.complex128, copy=True)
    current_channels = np.array(transform_channels, dtype=np.complex128, copy=True)
    history = np.empty(
        (law.step_count, SITE_COUNT, SAMPLE_COUNT), dtype=np.complex128
    )
    for step in range(law.step_count):
        (
            interaction,
            lock,
            coherence,
            spatial,
            power_exponent,
            weighted_coherence_fraction,
        ) = schedule(step, law)
        velocity = native_wave_velocity(
            current_states,
            current_frames,
            current_channels,
            calibration,
            coupling,
            field,
            relation_enabled,
            interaction,
            lock,
            coherence,
            spatial,
            power_exponent,
            weighted_coherence_fraction,
        )
        updates = np.exp(1j * law.time_step * velocity)
        arm = step % masks.shape[0]
        next_states = np.empty_like(current_states)
        next_frames = np.empty_like(current_frames)
        next_channels = np.empty_like(current_channels)
        for site in range(SITE_COUNT):
            shift = transport_shift(step, site)
            next_states[site] = np.roll(
                current_states[site] * updates[site] * masks[arm, site], shift
            )
            next_frames[site] = np.roll(
                current_frames[site] * masks[arm, site], shift
            )
            for neighbor in range(SITE_COUNT):
                next_channels[site, neighbor] = np.roll(
                    current_channels[site, neighbor], shift
                )
        current_states = next_states
        current_frames = next_frames
        current_channels = next_channels
        history[step] = updates
    return current_states, current_frames, history


def execute_native_cycle(
    borrowed: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    *,
    law: MachineLaw = DEFAULT_LAW,
    program_beams: np.ndarray | None = None,
    actual_beams: np.ndarray | None = None,
    relation_enabled: np.ndarray | None = None,
    lock_enabled: bool = True,
) -> NativeExecution:
    law.validate()
    coupling, field = validate_problem(coupling, field)
    borrowed = np.asarray(borrowed, dtype=np.complex128)
    if borrowed.shape == (SAMPLE_COUNT,):
        borrowed = np.repeat(borrowed[np.newaxis, :], SITE_COUNT, axis=0)
    if borrowed.shape != (SITE_COUNT, SAMPLE_COUNT):
        raise ValueError("borrowed carrier bank has the wrong shape")
    if not np.all(np.isfinite(borrowed)) or np.min(np.abs(borrowed)) <= 0.0:
        raise ValueError("borrowed carrier bank must be finite and nonzero")
    canonical = r4.render_trees(r4.canonical_trees())
    program = canonical if program_beams is None else np.asarray(program_beams, dtype=np.complex128)
    actual = program if actual_beams is None else np.asarray(actual_beams, dtype=np.complex128)
    if program.shape != canonical.shape or actual.shape != canonical.shape:
        raise ValueError("waveform transform bank has the wrong shape")
    if max(
        float(np.max(np.abs(np.abs(program) - 1.0))),
        float(np.max(np.abs(np.abs(actual) - 1.0))),
    ) > 1.0e-12:
        raise ValueError("waveform transform banks must be unit modulus")
    enabled = (
        np.ones((SITE_COUNT, SITE_COUNT), dtype=np.bool_)
        if relation_enabled is None
        else np.array(relation_enabled, dtype=np.bool_, copy=True)
    )
    np.fill_diagonal(enabled, False)
    masks = r4.transport_mask_bank(program)
    channels = transform_channel_bank(program, actual, law)
    calibration = channel_calibration(channels, borrowed)
    frames = borrowed * program
    states = borrowed * actual * np.exp(1j * INITIAL_PHASES[:, np.newaxis])
    execution_law = law
    if not lock_enabled:
        execution_law = MachineLaw(**{
            **asdict(law),
            "lock_initial": 0.0,
            "lock_solve": 0.0,
            "lock_final": 0.0,
        })
    displaced, query_frames, history = evolve_native_waveforms(
        states,
        frames,
        channels,
        masks,
        calibration,
        coupling,
        field,
        enabled,
        execution_law,
    )
    return NativeExecution(
        borrowed=np.array(borrowed, copy=True),
        coupling=np.array(coupling, copy=True),
        field=np.array(field, copy=True),
        actual_beams=np.array(actual, copy=True),
        program_beams=np.array(program, copy=True),
        masks=np.array(masks, copy=True),
        displaced=displaced,
        query_frames=query_frames,
        operator_history=history,
        displacement_l2=float(np.linalg.norm(displaced - borrowed)),
        law=execution_law,
    )


def ising_energy(
    spins: Sequence[int], coupling: np.ndarray, field: np.ndarray
) -> float:
    coupling, field = validate_problem(coupling, field)
    if len(spins) != SITE_COUNT or any(value not in (-1, 1) for value in spins):
        raise ValueError("energy boundary requires exact antipodal spins")
    vector = np.asarray(spins, dtype=np.float64)
    return float(-0.5 * vector @ coupling @ vector - field @ vector)


def project_boundary(execution: NativeExecution, label: str) -> BoundaryProjection:
    responses: list[complex] = []
    for site in range(SITE_COUNT):
        denominator = float(
            np.linalg.norm(execution.query_frames[site])
            * np.linalg.norm(execution.displaced[site])
        )
        responses.append(
            complex(
                np.vdot(execution.query_frames[site], execution.displaced[site])
                / denominator
            )
        )
    coherence = tuple(float(abs(value)) for value in responses)
    phases = tuple(float(np.angle(value)) for value in responses)
    coherent = min(coherence) >= COHERENCE_MIN
    residual = lock_residual(phases) if coherent else None
    valid = bool(coherent and residual is not None and residual <= LOCK_RESIDUAL_MAX)
    raw_spins = tuple(1 if value.real >= 0.0 else -1 for value in responses)
    spins = raw_spins if valid else None
    energy = (
        ising_energy(spins, execution.coupling, execution.field)
        if spins is not None
        else None
    )
    return BoundaryProjection(
        label=label,
        responses=tuple(responses),
        coherence=coherence,
        phases=phases,
        raw_spins=raw_spins,
        spins=spins,
        energy=energy,
        residual=residual,
        valid=valid,
    )


def restore_carrier(execution: NativeExecution, mode: str = "correct") -> np.ndarray:
    restored = np.array(execution.displaced, copy=True)
    if mode == "omitted":
        return restored
    for step in range(execution.law.step_count - 1, -1, -1):
        arm = step % execution.masks.shape[0]
        for site in range(SITE_COUNT):
            shift = transport_shift(step, site)
            if mode == "correct":
                value = np.roll(restored[site], -shift)
                value = value * np.conjugate(execution.masks[arm, site])
            elif mode == "wrong_order":
                value = np.roll(
                    restored[site] * np.conjugate(execution.masks[arm, site]),
                    -shift,
                )
            elif mode == "omit_middle" and step == execution.law.step_count // 2:
                continue
            elif mode == "omit_middle":
                value = np.roll(restored[site], -shift)
                value = value * np.conjugate(execution.masks[arm, site])
            else:
                raise ValueError(f"unsupported inverse mode: {mode}")
            restored[site] = value * np.conjugate(
                execution.operator_history[step, site]
            )
    restored = restored * np.conjugate(execution.actual_beams)
    restored = restored * np.exp(-1j * INITIAL_PHASES[:, np.newaxis])
    return restored


def transport_query_bank(
    initial_frames: np.ndarray, masks: np.ndarray, law: MachineLaw
) -> np.ndarray:
    frames = np.array(initial_frames, dtype=np.complex128, copy=True)
    for step in range(law.step_count):
        arm = step % masks.shape[0]
        for site in range(SITE_COUNT):
            frames[site] = np.roll(
                frames[site] * masks[arm, site], transport_shift(step, site)
            )
    return frames


def response_delta(
    control: BoundaryProjection, reference: BoundaryProjection
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(control.responses, dtype=np.complex128)
            - np.asarray(reference.responses, dtype=np.complex128)
        )
    )


def history_delta(control: NativeExecution, reference: NativeExecution) -> float:
    return float(
        np.linalg.norm(control.operator_history - reference.operator_history)
    )


def run_strict_controls(
    borrowed: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    reference_execution: NativeExecution,
    reference_latch: BoundaryProjection,
    law: MachineLaw,
) -> dict[str, Any]:
    canonical = reference_execution.program_beams
    no_transform_execution = execute_native_cycle(
        borrowed,
        coupling,
        field,
        law=law,
        actual_beams=np.ones_like(canonical),
    )
    no_transform_latch = project_boundary(no_transform_execution, "no_transform")

    flat = r4.flat_replacement_beams(r4.canonical_trees())
    flat_execution = execute_native_cycle(
        borrowed,
        coupling,
        field,
        law=law,
        program_beams=flat,
        actual_beams=flat,
    )
    flat_latch = project_boundary(flat_execution, "flat_geometry")

    scrambled = r4.render_trees(r4.scrambled_trees())
    scrambled_execution = execute_native_cycle(
        borrowed,
        coupling,
        field,
        law=law,
        program_beams=scrambled,
        actual_beams=scrambled,
    )
    scrambled_latch = project_boundary(scrambled_execution, "scrambled_geometry")

    relation_enabled = np.ones((SITE_COUNT, SITE_COUNT), dtype=np.bool_)
    relation_enabled[0, 4] = False
    relation_enabled[4, 0] = False
    missing_execution = execute_native_cycle(
        borrowed,
        coupling,
        field,
        law=law,
        relation_enabled=relation_enabled,
    )
    missing_latch = project_boundary(missing_execution, "missing_pair_operator")

    no_lock_execution = execute_native_cycle(
        borrowed,
        coupling,
        field,
        law=law,
        lock_enabled=False,
    )
    no_lock_latch = project_boundary(no_lock_execution, "no_lock")

    uniform_execution = execute_native_cycle(
        np.ones_like(borrowed), coupling, field, law=law
    )
    uniform_latch = project_boundary(uniform_execution, "uniform_carrier")

    wrong_query_frames = transport_query_bank(
        borrowed * scrambled, reference_execution.masks, law
    )
    wrong_query_responses: list[complex] = []
    for site in range(SITE_COUNT):
        denominator = float(
            np.linalg.norm(wrong_query_frames[site])
            * np.linalg.norm(reference_execution.displaced[site])
        )
        wrong_query_responses.append(
            complex(
                np.vdot(
                    wrong_query_frames[site], reference_execution.displaced[site]
                )
                / denominator
            )
        )
    wrong_query_latch = BoundaryProjection(
        label="wrong_query",
        responses=tuple(wrong_query_responses),
        coherence=tuple(abs(value) for value in wrong_query_responses),
        phases=tuple(float(np.angle(value)) for value in wrong_query_responses),
        raw_spins=tuple(1 if value.real >= 0.0 else -1 for value in wrong_query_responses),
        spins=None,
        energy=None,
        residual=None,
        valid=False,
    )

    executions = {
        "flat_geometry": flat_execution,
        "missing_pair_operator": missing_execution,
        "no_lock": no_lock_execution,
        "no_transform": no_transform_execution,
        "scrambled_geometry": scrambled_execution,
        "uniform_carrier": uniform_execution,
    }
    latches = {
        "flat_geometry": flat_latch,
        "missing_pair_operator": missing_latch,
        "no_lock": no_lock_latch,
        "no_transform": no_transform_latch,
        "scrambled_geometry": scrambled_latch,
        "uniform_carrier": uniform_latch,
        "wrong_query": wrong_query_latch,
    }
    history_deltas = {
        name: history_delta(execution, reference_execution)
        for name, execution in executions.items()
    }
    response_deltas = {
        name: response_delta(latch, reference_latch)
        for name, latch in latches.items()
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
        np.max(np.abs(restore_carrier(reference_execution, "wrong_order") - borrowed))
    )
    omitted_step_error = float(
        np.max(np.abs(restore_carrier(reference_execution, "omit_middle") - borrowed))
    )
    omitted_restoration_error = float(
        np.max(np.abs(restore_carrier(reference_execution, "omitted") - borrowed))
    )
    outcomes = {
        "uniform_carrier_replacement": (
            history_deltas["uniform_carrier"] >= MATERIALITY_MIN
            and response_deltas["uniform_carrier"] >= MATERIALITY_MIN
        ),
        "flat_geometry": (
            history_deltas["flat_geometry"] >= MATERIALITY_MIN
            and response_deltas["flat_geometry"] >= MATERIALITY_MIN
        ),
        "parent_child_geometry_scramble": (
            history_deltas["scrambled_geometry"] >= MATERIALITY_MIN
            and response_deltas["scrambled_geometry"] >= MATERIALITY_MIN
        ),
        "removed_waveform_transform": (
            history_deltas["no_transform"] >= MATERIALITY_MIN
            and response_deltas["no_transform"] >= MATERIALITY_MIN
        ),
        "removed_pair_operator": (
            history_deltas["missing_pair_operator"] >= MATERIALITY_MIN
            and response_deltas["missing_pair_operator"] >= MATERIALITY_MIN
        ),
        "no_lock": (
            history_deltas["no_lock"] >= MATERIALITY_MIN
            and response_deltas["no_lock"] >= MATERIALITY_MIN
        ),
        "wrong_query": response_deltas["wrong_query"] >= MATERIALITY_MIN,
        "wrong_inverse": wrong_inverse_error >= WRONG_RESTORATION_MIN,
        "omitted_inverse_step": omitted_step_error >= WRONG_RESTORATION_MIN,
        "omitted_restoration": (
            omitted_restoration_error >= WRONG_RESTORATION_MIN
        ),
        "samplewise_non_rank_one": rank_one_residual >= SAMPLEWISE_MIN,
    }
    return {
        "all_pass": all(outcomes.values()),
        "measurements": {
            "history_deltas_l2": {
                name: metric(value) for name, value in history_deltas.items()
            },
            "omitted_inverse_step_error": metric(omitted_step_error),
            "omitted_restoration_error": metric(omitted_restoration_error),
            "response_deltas_l2": {
                name: metric(value) for name, value in response_deltas.items()
            },
            "samplewise_non_rank_one_residual": metric(rank_one_residual),
            "wrong_inverse_error": metric(wrong_inverse_error),
        },
        "outcomes": outcomes,
    }


def exact_oracle(
    coupling: np.ndarray, field: np.ndarray
) -> list[tuple[float, tuple[int, ...]]]:
    rows = [
        (ising_energy(spins, coupling, field), tuple(int(value) for value in spins))
        for spins in itertools.product((-1, 1), repeat=SITE_COUNT)
    ]
    return sorted(rows, key=lambda item: (item[0], item[1]))


def development_instances() -> list[dict[str, Any]]:
    heldout = json.loads(HELDOUT_CUSTODY.read_text(encoding="utf-8"))
    batch = json.loads(BATCH_CUSTODY.read_text(encoding="utf-8"))
    records = [
        {
            "label": "verified_primary",
            "coupling": np.array(r4.COUPLING, dtype=np.float64),
            "field": np.array(r4.PRIMARY_FIELD, dtype=np.float64),
        },
        {
            "label": "verified_reuse",
            "coupling": np.array(r4.COUPLING, dtype=np.float64),
            "field": np.array(r4.REUSE_FIELD, dtype=np.float64),
        },
        {
            "label": "first_heldout",
            "coupling": np.asarray(
                heldout["held_out_instance"]["coupling_matrix_J"],
                dtype=np.float64,
            ),
            "field": np.asarray(
                heldout["held_out_instance"]["field_vector_h"],
                dtype=np.float64,
            ),
        },
    ]
    for record in batch["ordered_instances"]:
        records.append(
            {
                "label": f"batch_{int(record['index']):02d}",
                "coupling": np.asarray(record["coupling_matrix_J"], dtype=np.float64),
                "field": np.asarray(record["field_vector_h"], dtype=np.float64),
            }
        )
    return records


def development_probe(
    law: MachineLaw,
    *,
    include_controls: bool = False,
    labels: set[str] | None = None,
) -> dict[str, Any]:
    carrier = r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], SITE_COUNT, axis=0)
    outcomes: list[dict[str, Any]] = []
    records = development_instances()
    if labels is not None:
        records = [record for record in records if record["label"] in labels]
        missing = labels - {record["label"] for record in records}
        if missing:
            raise ValueError(f"unknown development labels: {sorted(missing)}")
    for record in records:
        coupling = record["coupling"]
        field = record["field"]
        execution = execute_native_cycle(borrowed, coupling, field, law=law)
        latch = project_boundary(execution, record["label"])
        rows = exact_oracle(coupling, field)
        optimum_energy = rows[0][0]
        optimum_states = {
            spins for energy, spins in rows if abs(energy - optimum_energy) <= 1.0e-12
        }
        raw_match = latch.raw_spins in optimum_states
        restored = restore_carrier(execution)
        restore_error = float(np.max(np.abs(restored - borrowed)))
        controls = (
            run_strict_controls(
                borrowed, coupling, field, execution, latch, law
            )
            if include_controls
            else None
        )
        outcomes.append(
            {
                "accepted": latch.valid,
                "accepted_correct": latch.valid and raw_match,
                "accepted_incorrect": latch.valid and not raw_match,
                "label": record["label"],
                "minimum_coherence": metric(min(latch.coherence)),
                "optimum_count": len(optimum_states),
                "raw_match": raw_match,
                "raw_spins": list(latch.raw_spins),
                "residual_rad": None if latch.residual is None else metric(latch.residual),
                "restoration_max_abs_error": metric(restore_error),
                "strict_controls": controls,
            }
        )
    unique = [item for item in outcomes if item["optimum_count"] == 1]
    coherence = np.asarray([item["minimum_coherence"] for item in outcomes])
    return {
        "law": asdict(law),
        "machine_law_sha256": sha256_bytes(canonical_bytes(asdict(law))),
        "outcomes": outcomes,
        "summary": {
            "accepted_correct": sum(item["accepted_correct"] for item in unique),
            "accepted_incorrect": sum(item["accepted_incorrect"] for item in unique),
            "accepted_total": sum(item["accepted"] for item in unique),
            "development_instance_count": len(outcomes),
            "minimum_coherence_distribution": {
                "max": metric(np.max(coherence)),
                "mean": metric(np.mean(coherence)),
                "median": metric(np.median(coherence)),
                "min": metric(np.min(coherence)),
            },
            "raw_correct_unique": sum(item["raw_match"] for item in unique),
            "strict_all_controls": (
                sum(bool(item["strict_controls"]["all_pass"]) for item in outcomes)
                if include_controls
                else None
            ),
            "unique_instance_count": len(unique),
        },
    }


def law_from_arguments(args: argparse.Namespace) -> MachineLaw:
    values = asdict(DEFAULT_LAW)
    for name in values:
        replacement = getattr(args, name)
        if replacement is not None:
            values[name] = replacement
    return MachineLaw(**values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--labels")
    for name, value in asdict(DEFAULT_LAW).items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int if isinstance(value, int) else float,
        )
    args = parser.parse_args(argv)
    labels = None if args.labels is None else set(args.labels.split(","))
    document = development_probe(
        law_from_arguments(args), include_controls=args.controls, labels=labels
    )
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
