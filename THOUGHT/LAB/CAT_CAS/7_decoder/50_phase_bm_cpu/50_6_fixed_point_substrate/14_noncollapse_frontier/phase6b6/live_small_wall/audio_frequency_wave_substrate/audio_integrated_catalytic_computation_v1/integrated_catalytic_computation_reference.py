#!/usr/bin/env python3
"""Bounded carrier-causal recursive-wave Ising computation reference.

The native state is a bank of complete complex waveforms.  Couplings are applied as
pointwise phase-relation operators between recursive carrier geometries.  No decoded
spin, Ising energy, oracle row, winner, or expected answer enters native evolution.

This is an ordinary-software reference.  It establishes no physical computation,
performance advantage, general Ising solver, hardware bit replacement, or Wall claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
R0_SOURCE = ROOT / "audio_recursive_phase_tree_v1" / "recursive_phase_tree_reference.py"
R2_SOURCE = ROOT / "audio_catalytic_wave_loop_v1" / "catalytic_wave_loop_reference.py"
R3_SOURCE = ROOT / "audio_recursive_catalytic_ising_v1" / "recursive_catalytic_ising_reference.py"


def _load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"unable to load predecessor module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


r0 = _load_module(R0_SOURCE, "catcas_integrated_r0")

GENERATOR_ID = "integrated_catalytic_computation_reference_v1"
RESULT_SCHEMA = "integrated_catalytic_computation_result_v1"
CONTRACT_SCHEMA = "integrated_catalytic_computation_contract_v1"
LATCH_SCHEMA = "integrated_catalytic_computation_latch_v1"
MANIFEST_SCHEMA = "integrated_catalytic_computation_manifest_v1"
TEST_SCHEMA = "integrated_catalytic_computation_tests_v1"
CLAIM_CEILING = "BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY"
VERIFIED_TOKEN = "CATALYTIC_WAVEFORM_ISING_COMPUTATION_VERIFIED"

SITE_COUNT = 5
SAMPLE_COUNT = 256
STEP_COUNT = 1000
TIME_STEP = 0.03
LOCK_FINAL = 1.2
INITIAL_PHASES = np.array([0.31, 1.27, -2.11, 2.53, -0.83], dtype=np.float64)
COUPLING = np.array(
    [
        [0.0, 0.0, 1.0, 2.0, -2.0],
        [0.0, 0.0, 2.0, -1.0, -1.0],
        [1.0, 2.0, 0.0, 2.0, -1.0],
        [2.0, -1.0, 2.0, 0.0, -2.0],
        [-2.0, -1.0, -1.0, -2.0, 0.0],
    ],
    dtype=np.float64,
)
PRIMARY_FIELD = np.array([-2.0, 1.0, -2.0, -2.0, -2.0], dtype=np.float64)
REUSE_FIELD = np.array([1.0, -1.0, 0.5, 0.5, -1.0], dtype=np.float64)

# Frozen before final fixture generation.
RESTORE_TOL = 2.0e-12
LOCK_RESIDUAL_MAX = 0.15
QUERY_COHERENCE_MIN = 0.90
DISPLACEMENT_MIN = 1.0
WRONG_RESTORE_MIN = 1.0e-3
OPERATOR_HISTORY_CHANGE_MIN = 1.0e-3
SAMPLEWISE_DYNAMICS_MIN = 1.0e-3
ENERGY_TOL = 1.0e-12
TRANSPORT_PHASE_AMPLITUDE = 0.19
TRANSPORT_SECOND_AMPLITUDE = 0.07
TRANSPORT_OFFSETS = (0.0, 0.73, -0.41)
TRANSPORT_SHIFTS = (3, -5, 7)
GEOMETRY_GATE_DEPTH = 1.8
SPATIAL_COUPLING = 0.6
GLOBAL_COHERENCE_COUPLING = 4.0

FIXTURE_DIR = PACKAGE_DIR / "fixtures"
CONTRACT_FILE = "INTEGRATED_CATALYTIC_COMPUTATION_CONTRACT.json"
TESTS_FILE = "INTEGRATED_CATALYTIC_COMPUTATION_REFERENCE_TESTS.json"
MANIFEST_FILE = "INTEGRATED_CATALYTIC_COMPUTATION_FIXTURE_MANIFEST.json"
RESULTS_FILE = "INTEGRATED_CATALYTIC_COMPUTATION_REFERENCE_RESULTS.json"
BORROWED_FILE = "fixtures/borrowed_carrier.c128le"
PRIMARY_LATCH_FILE = "fixtures/primary_latch.json"
REUSE_LATCH_FILE = "fixtures/reuse_latch.json"

PARENT_EXPECTED = {
    "audio_catalytic_wave_loop_v1": {
        "source_bytes": 117873,
        "source_sha256": "6c55861da950caf0738bb5ffb676f0c458a593a805ddd49419d6b2b427f6c33c",
    },
    "audio_recursive_catalytic_ising_v1": {
        "source_bytes": 146825,
        "source_sha256": "076fa3f392a9a0f1307e222deeabef38d558bf93db10c317af481ee40bf17b48",
    },
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_bytes(value))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def validate_problem(coupling: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coupling = np.asarray(coupling, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    if coupling.shape != (SITE_COUNT, SITE_COUNT) or field.shape != (SITE_COUNT,):
        raise ValueError("problem shape mismatch")
    if not np.all(np.isfinite(coupling)) or not np.all(np.isfinite(field)):
        raise ValueError("problem coefficients must be finite")
    if not np.array_equal(coupling, coupling.T):
        raise ValueError("coupling must be exactly symmetric")
    if not np.array_equal(np.diag(coupling), np.zeros(SITE_COUNT)):
        raise ValueError("coupling diagonal must be exactly zero")
    return coupling, field


def wrap_phase(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value, dtype=np.float64)
    wrapped = (array + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped) if np.ndim(value) == 0 else wrapped


def lock_residual(phases: Sequence[float]) -> float:
    values = np.asarray(phases, dtype=np.float64)
    zero = np.abs(wrap_phase(values))
    pi = np.abs(wrap_phase(values - math.pi))
    return float(np.max(np.minimum(zero, pi)))


def borrowed_carrier() -> np.ndarray:
    times = r0.sample_times()[:SAMPLE_COUNT]
    carrier = np.asarray(r0.borrowed_tape(times), dtype=np.complex128)
    if carrier.shape != (SAMPLE_COUNT,) or np.min(np.abs(carrier)) <= 0.0:
        raise ValueError("borrowed carrier is outside the frozen envelope")
    return carrier


def site_tree(site: int, *, scrambled: bool = False) -> Any:
    if isinstance(site, bool) or not 0 <= site < SITE_COUNT:
        raise ValueError("site index outside frozen range")
    leaf = r0.PhaseNode(
        f"site{site}.leaf", 47.0 + 5.0 * site, phase_rad=0.31 + 0.07 * site
    )
    middle_plain = r0.PhaseNode(
        f"site{site}.middle", 233.0 + 13.0 * site, phase_rad=-0.22 + 0.05 * site
    )
    inner = 0.31 + 0.035 * site
    outer = 0.62 + 0.045 * site
    if scrambled:
        leaf_with_child = r0.PhaseNode(
            leaf.node_id,
            leaf.frequency_hz,
            phase_rad=leaf.phase_rad,
            children=(r0.PhaseEdge(inner, middle_plain),),
        )
        root = r0.PhaseNode(
            f"site{site}.root",
            1237.0 + 71.0 * site,
            phase_rad=0.17 - 0.04 * site,
            children=(r0.PhaseEdge(outer, leaf_with_child),),
        )
    else:
        middle = r0.PhaseNode(
            middle_plain.node_id,
            middle_plain.frequency_hz,
            phase_rad=middle_plain.phase_rad,
            children=(r0.PhaseEdge(inner, leaf),),
        )
        root = r0.PhaseNode(
            f"site{site}.root",
            1237.0 + 71.0 * site,
            phase_rad=0.17 - 0.04 * site,
            children=(r0.PhaseEdge(outer, middle),),
        )
    return r0.RecursivePhaseBeam(root=root, global_spin_phase_rad=0.0)


def canonical_trees() -> tuple[Any, ...]:
    return tuple(site_tree(site) for site in range(SITE_COUNT))


def scrambled_trees() -> tuple[Any, ...]:
    return tuple(site_tree(site, scrambled=True) for site in range(SITE_COUNT))


def render_trees(trees: Sequence[Any]) -> np.ndarray:
    if len(trees) != SITE_COUNT:
        raise ValueError("tree bank must contain five complete trees")
    times = r0.sample_times()[:SAMPLE_COUNT]
    beams = np.stack([np.asarray(tree.render(times), dtype=np.complex128) for tree in trees])
    if beams.shape != (SITE_COUNT, SAMPLE_COUNT):
        raise ValueError("rendered tree bank has the wrong shape")
    if np.max(np.abs(np.abs(beams) - 1.0)) > 1.0e-12:
        raise ValueError("recursive tree beams must be unit modulus")
    return beams


def flat_replacement_beams(trees: Sequence[Any]) -> np.ndarray:
    times = r0.sample_times()[:SAMPLE_COUNT]
    return np.stack([r0.flat_multitone_replacement(tree, times) for tree in trees]).astype(np.complex128)


def transport_mask_bank(program_beams: np.ndarray) -> np.ndarray:
    phases = np.angle(program_beams)
    masks = np.empty((len(TRANSPORT_OFFSETS), SITE_COUNT, SAMPLE_COUNT), dtype=np.complex128)
    for arm, offset in enumerate(TRANSPORT_OFFSETS):
        phase = (
            TRANSPORT_PHASE_AMPLITUDE * np.sin(phases + offset)
            + TRANSPORT_SECOND_AMPLITUDE * np.sin(2.0 * phases - offset)
        )
        masks[arm] = np.exp(1j * phase)
    return masks


def geometry_channel_bank(program_beams: np.ndarray) -> np.ndarray:
    """Complex pair channels made from recursive parent-child phase geometry."""
    channels = np.empty(
        (SITE_COUNT, SITE_COUNT, SAMPLE_COUNT), dtype=np.complex128
    )
    for site in range(SITE_COUNT):
        for neighbor in range(SITE_COUNT):
            phase_relation = np.angle(
                program_beams[site] * np.conjugate(program_beams[neighbor])
            )
            channels[site, neighbor] = np.exp(
                1j * GEOMETRY_GATE_DEPTH * np.sin(phase_relation)
            )
    return channels


def geometry_calibration(beams: np.ndarray, borrowed: np.ndarray) -> np.ndarray:
    """Normalize only each channel mean; pointwise geometry remains native."""
    channels = geometry_channel_bank(beams)
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
                raise ValueError("geometry channel mean is ill-conditioned")
            calibration[site, neighbor] = 1.0 / mean_channel
    return calibration


def transport_shift(step: int, site: int) -> int:
    if not 0 <= site < SITE_COUNT:
        raise ValueError("site outside frozen transport schedule")
    return int(TRANSPORT_SHIFTS[step % len(TRANSPORT_SHIFTS)])


def lock_strength(step: int, final_value: float) -> float:
    if not 0 <= step < STEP_COUNT:
        raise ValueError("step outside frozen lock schedule")
    if not math.isfinite(final_value) or final_value < 0.0:
        raise ValueError("lock final value must be finite and nonnegative")
    return float(final_value * step / (STEP_COUNT - 1))


def native_wave_velocity(
    states: np.ndarray,
    frames: np.ndarray,
    geometry_channels: np.ndarray,
    calibration: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    lock_value: float,
    relation_enabled: np.ndarray,
) -> np.ndarray:
    """Compute phase velocity only from complex waveform interactions."""
    coupling, field = validate_problem(coupling, field)
    states = np.asarray(states, dtype=np.complex128)
    frames = np.asarray(frames, dtype=np.complex128)
    geometry_channels = np.asarray(geometry_channels, dtype=np.complex128)
    calibration = np.asarray(calibration, dtype=np.complex128)
    relation_enabled = np.asarray(relation_enabled, dtype=np.bool_)
    if states.shape != (SITE_COUNT, SAMPLE_COUNT) or frames.shape != states.shape:
        raise ValueError("native waveform state shape mismatch")
    if relation_enabled.shape != (SITE_COUNT, SITE_COUNT):
        raise ValueError("phase-operator enable map shape mismatch")
    if geometry_channels.shape != (SITE_COUNT, SITE_COUNT, SAMPLE_COUNT):
        raise ValueError("recursive geometry channel shape mismatch")
    if calibration.shape != (SITE_COUNT, SITE_COUNT):
        raise ValueError("recursive geometry calibration shape mismatch")
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
            if relation_enabled[site, neighbor]:
                relation = (
                    frames[site]
                    * np.conjugate(frames[neighbor])
                    / frame_power[neighbor]
                )
                incoming = (
                    calibration[site, neighbor]
                    * geometry_channels[site, neighbor]
                    * relation
                    * states[neighbor]
                )
            else:
                incoming = states[neighbor]
            drive = drive + coupling[site, neighbor] * incoming
        lock_wave = (
            lock_value
            * frames[site]
            * frames[site]
            * np.conjugate(states[site])
            / frame_power[site]
        )
        drive = drive + lock_wave
        mean_power = float(np.mean(frame_power[site]))
        local_flow = np.imag(np.conjugate(states[site]) * drive) / mean_power
        orientation_field = states[site] / frames[site]
        left = np.roll(orientation_field, 1)
        right = np.roll(orientation_field, -1)
        spatial_flow = np.imag(
            np.conjugate(orientation_field) * (left + right)
        )
        mean_orientation = complex(np.mean(orientation_field))
        coherence_flow = np.imag(
            np.conjugate(orientation_field) * mean_orientation
        )
        velocity[site] = (
            local_flow
            + SPATIAL_COUPLING * spatial_flow
            + GLOBAL_COHERENCE_COUPLING * coherence_flow
        )
    return velocity


def native_wave_step(
    states: np.ndarray,
    frames: np.ndarray,
    geometry_channels: np.ndarray,
    step: int,
    masks: np.ndarray,
    calibration: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    lock_final: float,
    relation_enabled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    velocity = native_wave_velocity(
        states,
        frames,
        geometry_channels,
        calibration,
        coupling,
        field,
        lock_strength(step, lock_final),
        relation_enabled,
    )
    updates = np.exp(1j * TIME_STEP * velocity)
    next_states = np.empty_like(states)
    next_frames = np.empty_like(frames)
    next_channels = np.empty_like(geometry_channels)
    arm = step % masks.shape[0]
    for site in range(SITE_COUNT):
        shift = transport_shift(step, site)
        next_states[site] = np.roll(
            states[site] * updates[site] * masks[arm, site], shift
        )
        next_frames[site] = np.roll(frames[site] * masks[arm, site], shift)
        for neighbor in range(SITE_COUNT):
            next_channels[site, neighbor] = np.roll(
                geometry_channels[site, neighbor], shift
            )
    return next_states, next_frames, next_channels, updates


def evolve_native_waveforms(
    states: np.ndarray,
    frames: np.ndarray,
    geometry_channels: np.ndarray,
    masks: np.ndarray,
    calibration: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    lock_final: float,
    relation_enabled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete native evolution; no scalar boundary projection occurs here."""
    current_states = np.array(states, dtype=np.complex128, copy=True)
    current_frames = np.array(frames, dtype=np.complex128, copy=True)
    current_channels = np.array(geometry_channels, dtype=np.complex128, copy=True)
    history = np.empty(
        (STEP_COUNT, SITE_COUNT, SAMPLE_COUNT), dtype=np.complex128
    )
    for step in range(STEP_COUNT):
        current_states, current_frames, current_channels, history[step] = native_wave_step(
            current_states,
            current_frames,
            current_channels,
            step,
            masks,
            calibration,
            coupling,
            field,
            lock_final,
            relation_enabled,
        )
    return current_states, current_frames, history


@dataclass(frozen=True)
class NativeExecution:
    borrowed: np.ndarray
    actual_beams: np.ndarray
    program_beams: np.ndarray
    masks: np.ndarray
    displaced: np.ndarray
    query_frames: np.ndarray
    operator_history: np.ndarray
    field: np.ndarray
    displacement_l2: float


@dataclass(frozen=True)
class ResultLatch:
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
            "responses": [
                {"imag": metric(value.imag), "real": metric(value.real)}
                for value in self.responses
            ],
            "raw_spin_shadow": list(self.raw_spins),
            "residual_rad": None if self.residual is None else metric(self.residual),
            "schema": LATCH_SCHEMA,
            "spins": None if self.spins is None else list(self.spins),
            "valid": self.valid,
        }

    def digest(self) -> str:
        return sha256_bytes(canonical_bytes(self.document()))


def execute_native_cycle(
    borrowed: np.ndarray,
    field: np.ndarray,
    *,
    program_beams: np.ndarray | None = None,
    actual_beams: np.ndarray | None = None,
    calibration_beams: np.ndarray | None = None,
    lock_final: float = LOCK_FINAL,
    relation_enabled: np.ndarray | None = None,
) -> NativeExecution:
    coupling, field = validate_problem(COUPLING, field)
    borrowed = np.asarray(borrowed, dtype=np.complex128)
    if borrowed.shape == (SAMPLE_COUNT,):
        borrowed = np.repeat(borrowed[np.newaxis, :], SITE_COUNT, axis=0)
    if borrowed.shape != (SITE_COUNT, SAMPLE_COUNT):
        raise ValueError("borrowed carrier bank has the wrong shape")
    if not np.all(np.isfinite(borrowed)) or np.min(np.abs(borrowed)) <= 0.0:
        raise ValueError("borrowed carrier bank must be finite and nonzero")
    canonical_beams = render_trees(canonical_trees())
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
        np.ones((SITE_COUNT, SITE_COUNT), dtype=np.bool_)
        if relation_enabled is None
        else np.asarray(relation_enabled, dtype=np.bool_)
    )
    np.fill_diagonal(enabled, False)
    masks = transport_mask_bank(program)
    channels = geometry_channel_bank(program)
    calibration_source = (
        canonical_beams
        if calibration_beams is None
        else np.asarray(calibration_beams, dtype=np.complex128)
    )
    calibration = geometry_calibration(calibration_source, borrowed)
    frames = borrowed * program
    states = borrowed * actual * np.exp(1j * INITIAL_PHASES[:, np.newaxis])
    displaced, query_frames, history = evolve_native_waveforms(
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
    displacement = float(np.linalg.norm(displaced - borrowed))
    return NativeExecution(
        borrowed=np.array(borrowed, copy=True),
        actual_beams=np.array(actual, copy=True),
        program_beams=program,
        masks=masks,
        displaced=displaced,
        query_frames=query_frames,
        operator_history=history,
        field=np.array(field, copy=True),
        displacement_l2=displacement,
    )


def extract_boundary(
    execution: NativeExecution,
    label: str,
    *,
    query_frames: np.ndarray | None = None,
) -> ResultLatch:
    """The sole scalar projection boundary, called only after native evolution."""
    queries = execution.query_frames if query_frames is None else np.asarray(query_frames, dtype=np.complex128)
    responses: list[complex] = []
    for site in range(SITE_COUNT):
        denominator = float(np.linalg.norm(queries[site]) * np.linalg.norm(execution.displaced[site]))
        responses.append(complex(np.vdot(queries[site], execution.displaced[site]) / denominator))
    coherence = tuple(float(abs(value)) for value in responses)
    phases = tuple(float(np.angle(value)) for value in responses)
    coherent = min(coherence) >= QUERY_COHERENCE_MIN
    residual = lock_residual(phases) if coherent else None
    valid = bool(coherent and residual is not None and residual <= LOCK_RESIDUAL_MAX)
    raw_spins = tuple(1 if value.real >= 0.0 else -1 for value in responses)
    spins = raw_spins if valid else None
    energy = ising_energy(spins, COUPLING, execution.field) if spins is not None else None
    return ResultLatch(
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


def ising_energy(spins: Sequence[int], coupling: np.ndarray, field: np.ndarray) -> float:
    coupling, field = validate_problem(coupling, field)
    if len(spins) != SITE_COUNT or any(value not in (-1, 1) for value in spins):
        raise ValueError("energy boundary requires exact antipodal spins")
    vector = np.asarray(spins, dtype=np.float64)
    return float(-0.5 * vector @ coupling @ vector - field @ vector)


def exact_oracle(coupling: np.ndarray, field: np.ndarray) -> list[tuple[float, tuple[int, ...]]]:
    """Adjudication only; this function is unreachable from native evolution."""
    rows = [
        (ising_energy(spins, coupling, field), tuple(int(value) for value in spins))
        for spins in itertools.product((-1, 1), repeat=SITE_COUNT)
    ]
    return sorted(rows, key=lambda item: (item[0], item[1]))


def oracle_agreement(latch: ResultLatch, field: np.ndarray) -> dict[str, Any]:
    if not latch.valid or latch.spins is None or latch.energy is None:
        return {"agrees": False, "gap": None, "unique": False}
    rows = exact_oracle(COUPLING, field)
    optimum_energy, optimum_spins = rows[0]
    next_energy = next(energy for energy, spins in rows[1:] if spins != optimum_spins)
    return {
        "agrees": latch.spins == optimum_spins and abs(latch.energy - optimum_energy) <= ENERGY_TOL,
        "gap": metric(next_energy - optimum_energy),
        "unique": sum(abs(energy - optimum_energy) <= ENERGY_TOL for energy, _ in rows) == 1,
    }


def restore_carrier(execution: NativeExecution, mode: str = "correct") -> np.ndarray:
    restored = np.array(execution.displaced, copy=True)
    if mode == "omitted":
        return restored
    for step in range(STEP_COUNT - 1, -1, -1):
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
            elif mode == "omit_middle" and step == STEP_COUNT // 2:
                continue
            elif mode == "omit_middle":
                value = np.roll(restored[site], -shift)
                value = value * np.conjugate(execution.masks[arm, site])
            else:
                raise ValueError(f"unsupported inverse mode: {mode}")
            restored[site] = value * np.conjugate(execution.operator_history[step, site])
    restored = restored * np.conjugate(execution.actual_beams)
    restored = restored * np.exp(-1j * INITIAL_PHASES[:, np.newaxis])
    return restored


def transport_query_bank(initial_frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    frames = np.array(initial_frames, dtype=np.complex128, copy=True)
    for step in range(STEP_COUNT):
        arm = step % masks.shape[0]
        for site in range(SITE_COUNT):
            frames[site] = np.roll(
                frames[site] * masks[arm, site], transport_shift(step, site)
            )
    return frames


def scalar_spin_feedback_baseline(coupling: np.ndarray, field: np.ndarray) -> tuple[int, ...]:
    """Forbidden control: decoded scalar J@s feedback, never a native path."""
    phases = INITIAL_PHASES.copy()
    for step in range(STEP_COUNT):
        spins = np.where(np.cos(phases) >= 0.0, 1.0, -1.0)
        scalar_drive = coupling @ spins + field
        phases = wrap_phase(phases + TIME_STEP * scalar_drive - lock_strength(step, LOCK_FINAL) * np.sin(2.0 * phases))
    return tuple(int(value) for value in np.where(np.cos(phases) >= 0.0, 1, -1))


def native_ast_proof() -> dict[str, Any]:
    roots = {
        "native_wave_velocity",
        "native_wave_step",
        "evolve_native_waveforms",
    }
    text = Path(__file__).resolve().read_text(encoding="utf-8")
    module = ast.parse(text)
    functions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    reachable = set(roots)
    pending = list(roots)
    while pending:
        name = pending.pop()
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called = node.func.id
                if called in functions and called not in reachable:
                    reachable.add(called)
                    pending.append(called)
    reachable_nodes = [functions[name] for name in sorted(reachable)]
    reachable_text = "\n".join(
        ast.get_source_segment(text, node) or "" for node in reachable_nodes
    )
    forbidden_fragments = (
        "decode",
        "oracle",
        "ising_energy",
        "winner",
        "expected",
        "argmin",
        "argmax",
        "np.sign",
        "scalar_spin_feedback_baseline",
    )
    lowered = reachable_text.lower()
    hits = [fragment for fragment in forbidden_fragments if fragment in lowered]
    matrix_multiply_count = sum(
        isinstance(node, ast.MatMult)
        for function in reachable_nodes
        for node in ast.walk(function)
    )
    return {
        "forbidden_hits": hits,
        "matrix_multiply_count": matrix_multiply_count,
        "native_function_count": len(reachable),
        "pass": not hits and matrix_multiply_count == 0,
        "reachable_functions": sorted(reachable),
    }


def scalar_baseline_rejected() -> bool:
    text = inspect.getsource(scalar_spin_feedback_baseline)
    tree = ast.parse(text)
    has_projection = "np.where" in text and "np.cos" in text
    has_matrix_multiply = any(isinstance(node, ast.MatMult) for node in ast.walk(tree))
    return has_projection and has_matrix_multiply


def input_contract_has_no_answer_fields() -> bool:
    forbidden = {
        "answer",
        "candidate",
        "energy",
        "expected",
        "optimum",
        "result",
        "score",
        "spin",
        "winner",
    }

    def walk(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in forbidden):
                    return False
                if not walk(item):
                    return False
        elif isinstance(value, list):
            return all(walk(item) for item in value)
        return True

    document = contract_document()
    declarations = document.pop("no_smuggle")
    document.pop("thresholds")
    return all(value is False for value in declarations.values()) and walk(document)


def oracle_after_boundary_proof() -> bool:
    text = inspect.getsource(qualification)
    primary_boundary = text.index('extract_boundary(primary, "primary")')
    primary_oracle = text.index("oracle_agreement(primary_latch")
    reuse_boundary = text.index('extract_boundary(reuse, "reuse")')
    reuse_oracle = text.index("oracle_agreement(reuse_latch")
    return primary_boundary < primary_oracle and reuse_boundary < reuse_oracle


def parent_custody() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, path in (
        ("audio_catalytic_wave_loop_v1", R2_SOURCE),
        ("audio_recursive_catalytic_ising_v1", R3_SOURCE),
    ):
        payload = path.read_bytes()
        expected = PARENT_EXPECTED[name]
        record = {"source_bytes": len(payload), "source_sha256": sha256_bytes(payload)}
        record["pass"] = record == expected
        observed[name] = record
    return observed


def contract_document() -> dict[str, Any]:
    trees = canonical_trees()
    return {
        "claim_ceiling": CLAIM_CEILING,
        "coupling": COUPLING.tolist(),
        "initial_phases_rad": INITIAL_PHASES.tolist(),
        "mechanism": "pointwise_complex_recursive_geometry_interference",
        "geometry_calibration": "carrier_weighted_mean_only_per_program_geometry",
        "geometry_gate_depth": GEOMETRY_GATE_DEPTH,
        "global_coherence_coupling": GLOBAL_COHERENCE_COUPLING,
        "spatial_coupling": SPATIAL_COUPLING,
        "native_boundary": "after_complete_waveform_evolution_only",
        "no_smuggle": {
            "answer_fields": False,
            "decoded_spin_feedback": False,
            "energy_feedback": False,
            "exact_oracle_feedback": False,
        },
        "primary_field": PRIMARY_FIELD.tolist(),
        "reuse_field": REUSE_FIELD.tolist(),
        "sample_count": SAMPLE_COUNT,
        "sample_dtype": "little_endian_complex128",
        "schema": CONTRACT_SCHEMA,
        "site_tree_sha256": [sha256_bytes(tree.canonical_bytes()) for tree in trees],
        "step_count": STEP_COUNT,
        "thresholds": {
            "displacement_l2_min": DISPLACEMENT_MIN,
            "energy_abs_max": ENERGY_TOL,
            "lock_residual_rad_max": LOCK_RESIDUAL_MAX,
            "operator_history_change_l2_min": OPERATOR_HISTORY_CHANGE_MIN,
            "query_coherence_min": QUERY_COHERENCE_MIN,
            "restoration_max_abs_error": RESTORE_TOL,
            "samplewise_dynamics_min": SAMPLEWISE_DYNAMICS_MIN,
            "wrong_restoration_min_abs_error": WRONG_RESTORE_MIN,
        },
        "time_step": TIME_STEP,
        "transport": {
            "mask_first_amplitude": TRANSPORT_PHASE_AMPLITUDE,
            "mask_offsets": list(TRANSPORT_OFFSETS),
            "mask_second_amplitude": TRANSPORT_SECOND_AMPLITUDE,
            "shift_bases": list(TRANSPORT_SHIFTS),
        },
    }


def test_ids() -> list[str]:
    return [
        "parent_r2_source_custody",
        "parent_r3_source_custody",
        "native_ast_no_scalar_feedback",
        "native_waveform_has_samplewise_dynamics",
        "carrier_content_changes_native_history",
        "zero_carrier_rejected",
        "scalar_js_baseline_rejected",
        "input_contract_contains_no_answer_fields",
        "oracle_is_after_boundary_only",
        "primary_waveform_result_valid",
        "primary_oracle_agreement_after_boundary",
        "primary_unique_optimum",
        "primary_nonzero_displacement",
        "primary_correct_inverse_restores",
        "primary_result_persists_outside_inverse",
        "reuse_uses_restored_carrier",
        "reuse_waveform_result_valid",
        "reuse_oracle_agreement_after_boundary",
        "reuse_result_differs_from_primary",
        "reuse_correct_inverse_restores",
        "remove_waveform_transform_destroys_result",
        "replace_recursive_geometry_destroys_result",
        "scramble_parent_child_geometry_destroys_result",
        "remove_one_phase_operator_changes_native_history",
        "no_lock_destroys_result",
        "wrong_query_destroys_result",
        "wrong_inverse_fails",
        "omitted_inverse_step_fails",
        "omitted_restoration_fails",
        "jh_change_changes_extracted_result",
        "result_latch_contains_no_reversed_history",
    ]


def test_spec_document() -> dict[str, Any]:
    return {
        "count": len(test_ids()),
        "schema": TEST_SCHEMA,
        "tests": test_ids(),
        "thresholds_sha256": sha256_bytes(canonical_bytes(contract_document()["thresholds"])),
    }


def qualification() -> dict[str, Any]:
    carrier = borrowed_carrier()
    borrowed_bank = np.repeat(carrier[np.newaxis, :], SITE_COUNT, axis=0)
    primary = execute_native_cycle(borrowed_bank, PRIMARY_FIELD)
    primary_latch = extract_boundary(primary, "primary")
    primary_latch_before = primary_latch.digest()
    primary_oracle = oracle_agreement(primary_latch, PRIMARY_FIELD)
    restored = restore_carrier(primary, "correct")
    restore_error = float(np.max(np.abs(restored - borrowed_bank)))
    primary_latch_after = primary_latch.digest()

    reuse = execute_native_cycle(restored, REUSE_FIELD)
    reuse_latch = extract_boundary(reuse, "reuse")
    reuse_oracle = oracle_agreement(reuse_latch, REUSE_FIELD)
    reuse_restored = restore_carrier(reuse, "correct")
    reuse_restore_error = float(np.max(np.abs(reuse_restored - restored)))

    canonical_beams = primary.program_beams
    no_transform = execute_native_cycle(
        borrowed_bank,
        PRIMARY_FIELD,
        actual_beams=np.ones_like(canonical_beams),
    )
    no_transform_latch = extract_boundary(no_transform, "no_transform")

    flat = flat_replacement_beams(canonical_trees())
    flat_execution = execute_native_cycle(
        borrowed_bank,
        PRIMARY_FIELD,
        program_beams=flat,
        actual_beams=flat,
        calibration_beams=flat,
    )
    flat_latch = extract_boundary(flat_execution, "flat_geometry")

    scrambled = render_trees(scrambled_trees())
    scrambled_execution = execute_native_cycle(
        borrowed_bank,
        PRIMARY_FIELD,
        program_beams=scrambled,
        actual_beams=scrambled,
        calibration_beams=scrambled,
    )
    scrambled_latch = extract_boundary(scrambled_execution, "scrambled_geometry")

    enabled = np.ones((SITE_COUNT, SITE_COUNT), dtype=np.bool_)
    enabled[0, 4] = False
    enabled[4, 0] = False
    missing_operator = execute_native_cycle(
        borrowed_bank,
        PRIMARY_FIELD,
        relation_enabled=enabled,
    )
    operator_history_change = float(
        np.linalg.norm(missing_operator.operator_history - primary.operator_history)
    )
    canonical_rank_one_residual = float(
        np.max(
            np.abs(
                primary.operator_history
                - np.mean(primary.operator_history, axis=2, keepdims=True)
            )
        )
    )
    uniform_carrier = np.ones_like(borrowed_bank)
    uniform_execution = execute_native_cycle(uniform_carrier, PRIMARY_FIELD)
    carrier_content_history_change = float(
        np.linalg.norm(uniform_execution.operator_history - primary.operator_history)
    )
    try:
        execute_native_cycle(np.zeros_like(borrowed_bank), PRIMARY_FIELD)
        zero_carrier_rejected = False
    except ValueError:
        zero_carrier_rejected = True
    missing_operator_latch = extract_boundary(missing_operator, "missing_phase_operator")

    no_lock = execute_native_cycle(borrowed_bank, PRIMARY_FIELD, lock_final=0.0)
    no_lock_latch = extract_boundary(no_lock, "no_lock")

    wrong_query_initial = borrowed_bank * scrambled
    wrong_query_frames = transport_query_bank(wrong_query_initial, primary.masks)
    wrong_query_latch = extract_boundary(primary, "wrong_query", query_frames=wrong_query_frames)

    wrong_inverse_error = float(
        np.max(np.abs(restore_carrier(primary, "wrong_order") - borrowed_bank))
    )
    omitted_step_error = float(
        np.max(np.abs(restore_carrier(primary, "omit_middle") - borrowed_bank))
    )
    omitted_restore_error = float(
        np.max(np.abs(restore_carrier(primary, "omitted") - borrowed_bank))
    )

    parent = parent_custody()
    ast_proof = native_ast_proof()
    tests = {
        "parent_r2_source_custody": parent["audio_catalytic_wave_loop_v1"]["pass"],
        "parent_r3_source_custody": parent["audio_recursive_catalytic_ising_v1"]["pass"],
        "native_ast_no_scalar_feedback": ast_proof["pass"],
        "native_waveform_has_samplewise_dynamics": (
            canonical_rank_one_residual >= SAMPLEWISE_DYNAMICS_MIN
        ),
        "carrier_content_changes_native_history": (
            carrier_content_history_change >= OPERATOR_HISTORY_CHANGE_MIN
        ),
        "zero_carrier_rejected": zero_carrier_rejected,
        "scalar_js_baseline_rejected": scalar_baseline_rejected(),
        "input_contract_contains_no_answer_fields": input_contract_has_no_answer_fields(),
        "oracle_is_after_boundary_only": oracle_after_boundary_proof(),
        "primary_waveform_result_valid": primary_latch.valid,
        "primary_oracle_agreement_after_boundary": primary_oracle["agrees"],
        "primary_unique_optimum": primary_oracle["unique"],
        "primary_nonzero_displacement": primary.displacement_l2 >= DISPLACEMENT_MIN,
        "primary_correct_inverse_restores": restore_error <= RESTORE_TOL,
        "primary_result_persists_outside_inverse": primary_latch_before == primary_latch_after,
        "reuse_uses_restored_carrier": float(np.max(np.abs(reuse.borrowed - restored))) == 0.0,
        "reuse_waveform_result_valid": reuse_latch.valid,
        "reuse_oracle_agreement_after_boundary": reuse_oracle["agrees"],
        "reuse_result_differs_from_primary": primary_latch.spins != reuse_latch.spins,
        "reuse_correct_inverse_restores": reuse_restore_error <= RESTORE_TOL,
        "remove_waveform_transform_destroys_result": not no_transform_latch.valid,
        "replace_recursive_geometry_destroys_result": (
            not flat_latch.valid and flat_latch.raw_spins != primary_latch.raw_spins
        ),
        "scramble_parent_child_geometry_destroys_result": (
            not scrambled_latch.valid
            and scrambled_latch.raw_spins != primary_latch.raw_spins
        ),
        "remove_one_phase_operator_changes_native_history": (
            operator_history_change >= OPERATOR_HISTORY_CHANGE_MIN
            and not missing_operator_latch.valid
        ),
        "no_lock_destroys_result": not no_lock_latch.valid,
        "wrong_query_destroys_result": not wrong_query_latch.valid,
        "wrong_inverse_fails": wrong_inverse_error >= WRONG_RESTORE_MIN,
        "omitted_inverse_step_fails": omitted_step_error >= WRONG_RESTORE_MIN,
        "omitted_restoration_fails": omitted_restore_error >= WRONG_RESTORE_MIN,
        "jh_change_changes_extracted_result": primary_latch.spins != reuse_latch.spins,
        "result_latch_contains_no_reversed_history": set(primary_latch.document()) == {
            "coherence", "energy", "label", "phases_rad", "raw_spin_shadow", "responses", "residual_rad", "schema", "spins", "valid"
        },
    }
    failures = [name for name in test_ids() if not tests[name]]
    controls = {
        "no_lock": no_lock_latch.document(),
        "no_transform": no_transform_latch.document(),
        "missing_phase_operator": missing_operator_latch.document(),
        "carrier_content_history_change_l2": metric(carrier_content_history_change),
        "geometry_replacement_calibration": "self_consistent_per_replacement",
        "operator_history_change_l2": metric(operator_history_change),
        "rank_one_operator_residual_max": metric(canonical_rank_one_residual),
        "recursive_geometry_replacement": flat_latch.document(),
        "scrambled_parent_child_geometry": scrambled_latch.document(),
        "wrong_query": wrong_query_latch.document(),
    }
    return {
        "ast_proof": ast_proof,
        "controls": controls,
        "decision": VERIFIED_TOKEN if not failures else "CATALYTIC_WAVEFORM_ISING_COMPUTATION_NOT_ESTABLISHED",
        "failures": failures,
        "measurements": {
            "omitted_inverse_step_error": metric(omitted_step_error),
            "omitted_restoration_error": metric(omitted_restore_error),
            "primary_displacement_l2": metric(primary.displacement_l2),
            "primary_restore_max_error": metric(restore_error),
            "reuse_displacement_l2": metric(reuse.displacement_l2),
            "reuse_input_max_error_from_restored": 0.0,
            "reuse_restore_max_error": metric(reuse_restore_error),
            "wrong_inverse_error": metric(wrong_inverse_error),
        },
        "parent_custody": parent,
        "primary_latch": primary_latch.document(),
        "primary_oracle": primary_oracle,
        "reuse_latch": reuse_latch.document(),
        "reuse_oracle": reuse_oracle,
        "schema": RESULT_SCHEMA,
        "test_count": len(tests),
        "test_pass_count": sum(bool(value) for value in tests.values()),
        "tests": tests,
    }


def fixture_records() -> list[dict[str, Any]]:
    paths = [CONTRACT_FILE, TESTS_FILE, BORROWED_FILE, PRIMARY_LATCH_FILE, REUSE_LATCH_FILE]
    return [
        {
            "bytes": (PACKAGE_DIR / path).stat().st_size,
            "path": path,
            "sha256": sha256_file(PACKAGE_DIR / path),
        }
        for path in paths
    ]


def manifest_document() -> dict[str, Any]:
    records = fixture_records()
    root = sha256_bytes(canonical_bytes(records))
    return {
        "fixture_count": len(records),
        "fixture_root_sha256": root,
        "fixtures": records,
        "generator": GENERATOR_ID,
        "schema": MANIFEST_SCHEMA,
        "total_bytes": sum(record["bytes"] for record in records),
    }


def source_binding() -> dict[str, Any]:
    payload = Path(__file__).resolve().read_bytes()
    return {"source_bytes": len(payload), "source_sha256": sha256_bytes(payload)}


def build_package() -> dict[str, Any]:
    write_json_atomic(PACKAGE_DIR / CONTRACT_FILE, contract_document())
    write_json_atomic(PACKAGE_DIR / TESTS_FILE, test_spec_document())
    carrier = borrowed_carrier().astype("<c16", copy=False)
    write_bytes_atomic(PACKAGE_DIR / BORROWED_FILE, carrier.tobytes(order="C"))
    result = qualification()
    write_json_atomic(PACKAGE_DIR / PRIMARY_LATCH_FILE, result["primary_latch"])
    write_json_atomic(PACKAGE_DIR / REUSE_LATCH_FILE, result["reuse_latch"])
    manifest = manifest_document()
    write_json_atomic(PACKAGE_DIR / MANIFEST_FILE, manifest)
    result["claim_ceiling"] = CLAIM_CEILING
    result["fixture_manifest_sha256"] = sha256_file(PACKAGE_DIR / MANIFEST_FILE)
    result["source"] = source_binding()
    write_json_atomic(PACKAGE_DIR / RESULTS_FILE, result)
    return result


def verify_package() -> dict[str, Any]:
    expected_contract = canonical_bytes(contract_document())
    expected_tests = canonical_bytes(test_spec_document())
    expected_carrier = borrowed_carrier().astype("<c16", copy=False).tobytes(order="C")
    if (PACKAGE_DIR / CONTRACT_FILE).read_bytes() != expected_contract:
        raise ValueError("committed contract bytes do not match frozen source constants")
    if (PACKAGE_DIR / TESTS_FILE).read_bytes() != expected_tests:
        raise ValueError("committed test specification bytes do not match")
    if (PACKAGE_DIR / BORROWED_FILE).read_bytes() != expected_carrier:
        raise ValueError("committed borrowed carrier bytes do not match")
    expected = qualification()
    if (PACKAGE_DIR / PRIMARY_LATCH_FILE).read_bytes() != canonical_bytes(expected["primary_latch"]):
        raise ValueError("committed primary latch bytes do not reproduce")
    if (PACKAGE_DIR / REUSE_LATCH_FILE).read_bytes() != canonical_bytes(expected["reuse_latch"]):
        raise ValueError("committed reuse latch bytes do not reproduce")
    manifest = manifest_document()
    if (PACKAGE_DIR / MANIFEST_FILE).read_bytes() != canonical_bytes(manifest):
        raise ValueError("committed manifest bytes do not reproduce")
    expected["claim_ceiling"] = CLAIM_CEILING
    expected["fixture_manifest_sha256"] = sha256_file(PACKAGE_DIR / MANIFEST_FILE)
    expected["source"] = source_binding()
    if (PACKAGE_DIR / RESULTS_FILE).read_bytes() != canonical_bytes(expected):
        raise ValueError("committed reference results do not reproduce")
    if expected["failures"]:
        raise ValueError(f"reference qualification failures: {expected['failures']}")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify", "self-test"))
    arguments = parser.parse_args(argv)
    if arguments.command == "build":
        result = build_package()
    elif arguments.command == "verify":
        result = verify_package()
    else:
        result = qualification()
    print(json.dumps({
        "decision": result["decision"],
        "failures": result["failures"],
        "test_count": result["test_count"],
        "test_pass_count": result["test_pass_count"],
    }, sort_keys=True))
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
