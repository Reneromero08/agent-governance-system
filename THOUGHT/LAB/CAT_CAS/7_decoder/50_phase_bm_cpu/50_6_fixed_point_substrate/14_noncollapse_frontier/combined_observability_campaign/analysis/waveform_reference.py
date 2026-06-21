#!/usr/bin/env python3
"""Canonical reference for the Phase 6 combined-observability sender waveform."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODE_NAMES = ("basis", "rotation", "residual", "mini")
MASK64 = (1 << 64) - 1


def tone_hz(index: int) -> float:
    """Return the frozen physical tone for an index in [0, 11]."""
    if not 0 <= index < 12:
        raise ValueError(f"tone index out of range: {index}")
    low = math.log(20.0)
    high = math.log(1500.0)
    x = index / 11.0
    return math.exp(low + (high - low) * x) * (
        1.0 + 0.013 * math.sin(2.399963 * (index + 1))
    )


def make_codebook() -> np.ndarray:
    """Reproduce the exact C codebook in combined_pdn_hardware.c."""
    state = (0x243F6A8885A308D3 ^ 7) & MASK64

    def code_rand() -> int:
        nonlocal state
        value = state
        value ^= (value << 13) & MASK64
        value ^= value >> 7
        value ^= (value << 17) & MASK64
        state = value & MASK64
        return state

    weights = (4, 5, 6, 7)
    best = None
    best_distance = -1
    for _ in range(4000):
        candidate = np.ones((4, 12), dtype=np.int8)
        for mode, weight in enumerate(weights):
            pool = list(range(12))
            for i in range(weight):
                j = i + code_rand() % (12 - i)
                pool[i], pool[j] = pool[j], pool[i]
                candidate[mode, pool[i]] = -1
        distance = 99
        for left in range(4):
            for right in range(left + 1, 4):
                distance = min(
                    distance,
                    int(np.count_nonzero(candidate[left] != candidate[right])),
                )
        if distance > best_distance:
            best_distance = distance
            best = candidate.copy()
    if best is None:
        raise RuntimeError("codebook generation failed")
    return best


CODEBOOK = make_codebook()


def phase_index(mode: int, source_index: int, theta_index: int) -> int:
    """Return the exact phase index used by the acquisition executor."""
    if not 0 <= mode < 4:
        raise ValueError(f"mode out of range: {mode}")
    if not 0 <= source_index < 12:
        raise ValueError(f"source index out of range: {source_index}")
    if not 0 <= theta_index < 8:
        raise ValueError(f"theta index out of range: {theta_index}")
    sign_offset = 4 if CODEBOOK[mode, source_index] < 0 else 0
    return (theta_index + sign_offset) % 8


def acquisition_gate(
    timestamps_tsc: np.ndarray,
    *,
    origin_tsc: int,
    tsc_hz: float,
    tone_index: int,
    phase_index_value: int,
    amplitude_level: int,
) -> np.ndarray:
    """Reproduce the exact gate executed by commit 81ea84f3."""
    if amplitude_level not in (1, 2, 3):
        raise ValueError(f"amplitude level out of range: {amplitude_level}")
    if not 0 <= phase_index_value < 8:
        raise ValueError(f"phase index out of range: {phase_index_value}")
    frequency = tone_hz(tone_index)
    half_ticks = 0.5 * tsc_hz / frequency
    requested_period_ticks = 2.0 * half_ticks
    offset = (
        timestamps_tsc.astype(np.float64)
        - float(origin_tsc)
        - (phase_index_value / 8.0) * requested_period_ticks
    )
    half = np.floor(offset / half_ticks).astype(np.int64)
    quadrant = np.mod(half, 8)
    return (quadrant < amplitude_level * 2).astype(np.float64)


def intended_v2_gate(
    timestamps_tsc: np.ndarray,
    *,
    origin_tsc: int,
    tsc_hz: float,
    tone_index: int,
    phase_index_value: int,
    amplitude_level: int,
) -> np.ndarray:
    """Reference waveform for a future corrected executor.

    The complete eight-state cycle lasts one requested tone period. One phase
    index is pi/4 and four indices are pi.
    """
    if amplitude_level not in (1, 2, 3):
        raise ValueError(f"amplitude level out of range: {amplitude_level}")
    if not 0 <= phase_index_value < 8:
        raise ValueError(f"phase index out of range: {phase_index_value}")
    frequency = tone_hz(tone_index)
    step_ticks = tsc_hz / (8.0 * frequency)
    offset = (
        timestamps_tsc.astype(np.float64)
        - float(origin_tsc)
        - phase_index_value * step_ticks
    )
    state = np.mod(np.floor(offset / step_ticks).astype(np.int64), 8)
    return (state < amplitude_level * 2).astype(np.float64)


def lockin(
    timestamps_tsc: np.ndarray,
    samples: np.ndarray,
    *,
    origin_tsc: int,
    tsc_hz: float,
    frequency_hz: float,
) -> complex:
    """Reproduce the executor lock-in convention."""
    if len(samples) < 4 or len(samples) != len(timestamps_tsc):
        raise ValueError("lock-in requires matching arrays of at least four samples")
    count = len(samples)
    window = 0.5 * (
        1.0 - np.cos(2.0 * np.pi * np.arange(count) / (count - 1))
    )
    centered = samples - float(np.mean(samples))
    seconds = (timestamps_tsc.astype(np.float64) - float(origin_tsc)) / tsc_hz
    phase = 2.0 * np.pi * frequency_hz * seconds
    weighted = centered * window
    weight = float(np.sum(window))
    i_value = 2.0 * float(np.sum(weighted * np.cos(phase))) / weight
    q_value = 2.0 * float(np.sum(weighted * np.sin(phase))) / weight
    return complex(i_value, q_value)


def matched_gate_correlation(samples: np.ndarray, gate: np.ndarray) -> tuple[float, float]:
    """Return weighted gate regression coefficient and weighted correlation."""
    if len(samples) < 4 or len(samples) != len(gate):
        raise ValueError("matched gate requires matching arrays")
    count = len(samples)
    window = 0.5 * (
        1.0 - np.cos(2.0 * np.pi * np.arange(count) / (count - 1))
    )
    weight = float(np.sum(window))
    centered_samples = samples - float(np.sum(samples * window) / weight)
    centered_gate = gate - float(np.sum(gate * window) / weight)
    gate_energy = float(np.sum(centered_gate * centered_gate * window))
    sample_energy = float(np.sum(centered_samples * centered_samples * window))
    if gate_energy <= 0.0 or sample_energy <= 0.0:
        return math.nan, math.nan
    cross = float(np.sum(centered_samples * centered_gate * window))
    beta = cross / gate_energy
    correlation = cross / math.sqrt(gate_energy * sample_energy)
    return beta, correlation


@dataclass(frozen=True)
class HarmonicDefinition:
    label: str
    ratio: float


RECOVERY_HARMONICS = (
    HarmonicDefinition("f_over_4", 0.25),
    HarmonicDefinition("f_over_2", 0.50),
    HarmonicDefinition("three_f_over_4", 0.75),
    HarmonicDefinition("requested_f", 1.00),
    HarmonicDefinition("five_f_over_4", 1.25),
    HarmonicDefinition("three_f_over_2", 1.50),
    HarmonicDefinition("two_f", 2.00),
)
