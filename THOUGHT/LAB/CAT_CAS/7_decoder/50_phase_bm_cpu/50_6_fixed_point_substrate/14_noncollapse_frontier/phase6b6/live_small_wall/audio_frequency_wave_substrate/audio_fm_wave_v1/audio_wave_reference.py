#!/usr/bin/env python3
"""Deterministic offline reference for the audio-frequency wave substrate lane.

This module intentionally performs every operation in ordinary software.  It is an
algebra, custody, and adversary reference; it is not a physical-carrier controller.
It imports no playback, recording, device, network, or live-authority interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = ROOT / "fixtures"

SAMPLE_RATE = 48_000
DURATION_SECONDS = 2.0
SAMPLE_COUNT = SAMPLE_RATE * 2
PRIMARY_CARRIER_HZ = 8_000.0
BASEBAND_LIMIT_HZ = 1_000.0
ABSOLUTE_SAMPLE_CEILING = 0.95
WAV_ENCODING = "IEEE_FLOAT32_LE"
GENERATOR_ID = "audio_wave_reference_v1"
CLAIM_CEILING = "AUDIO_FM_WAVE_ALGEBRA_ESTABLISHED"
FROZEN_PYTHON_VERSION = "3.11.6"
FROZEN_NUMPY_VERSION = "1.26.4"
REPORTED_METRIC_SIGNIFICANT_DIGITS = 12

FM_K_HZ_PER_UNIT = 420.0
PM_K_RAD_PER_UNIT = 1.15
CARRIER_AMPLITUDE = 0.90
EDGE_SAMPLES = 4_096


def _q(value: float) -> float:
    """Quantize reported metrics, not computed arrays, to stable JSON decimals."""

    return float(f"{float(value):.{REPORTED_METRIC_SIGNIFICANT_DIGITS}g}")


def reference_runtime() -> dict[str, str | int]:
    return {
        "python": FROZEN_PYTHON_VERSION,
        "numpy": FROZEN_NUMPY_VERSION,
        "reported_metric_significant_digits": REPORTED_METRIC_SIGNIFICANT_DIGITS,
    }


def assert_frozen_runtime() -> None:
    observed_python = ".".join(str(part) for part in sys.version_info[:3])
    observed_numpy = np.__version__
    if observed_python != FROZEN_PYTHON_VERSION or observed_numpy != FROZEN_NUMPY_VERSION:
        raise RuntimeError(
            "reference runtime mismatch: "
            f"expected Python {FROZEN_PYTHON_VERSION}/NumPy {FROZEN_NUMPY_VERSION}, "
            f"observed Python {observed_python}/NumPy {observed_numpy}"
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _riff_chunk(kind: bytes, payload: bytes) -> bytes:
    if len(kind) != 4:
        raise ValueError("RIFF chunk identifiers must have four bytes")
    padding = b"\x00" if len(payload) % 2 else b""
    return kind + struct.pack("<I", len(payload)) + payload + padding


def float32_wav_bytes(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    list_metadata: bytes | None = None,
) -> bytes:
    """Encode a minimal deterministic WAVE_FORMAT_IEEE_FLOAT little-endian WAV."""

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] not in (1, 2):
        raise ValueError("fixtures must be mono or explicit two-axis stereo")
    if array.shape[0] != SAMPLE_COUNT:
        raise ValueError(f"fixture sample count must be {SAMPLE_COUNT}")
    if not np.all(np.isfinite(array)):
        raise ValueError("fixture contains non-finite samples")
    peak = float(np.max(np.abs(array)))
    if peak > ABSOLUTE_SAMPLE_CEILING + 1e-12:
        raise ValueError(f"fixture peak {peak} exceeds {ABSOLUTE_SAMPLE_CEILING}")

    channels = int(array.shape[1])
    bits_per_sample = 32
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, bits_per_sample
    )
    chunks = [_riff_chunk(b"fmt ", fmt)]
    if list_metadata is not None:
        chunks.append(_riff_chunk(b"LIST", list_metadata))
    data = np.asarray(array, dtype="<f4", order="C").tobytes(order="C")
    chunks.append(_riff_chunk(b"data", data))
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write_float32_wav(path: Path, samples: np.ndarray) -> None:
    payload = float32_wav_bytes(samples)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def parse_float32_wav_bytes(payload: bytes) -> tuple[int, np.ndarray, list[str]]:
    """Parse IEEE float32 WAV and return sample rate, samples, and chunk IDs."""

    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    if struct.unpack("<I", payload[4:8])[0] + 8 != len(payload):
        raise ValueError("RIFF size does not match file length")

    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    data: bytes | None = None
    chunk_ids: list[str] = []
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("truncated WAV chunk header")
        kind = payload[offset : offset + 4]
        size = struct.unpack("<I", payload[offset + 4 : offset + 8])[0]
        start = offset + 8
        end = start + size
        if end > len(payload):
            raise ValueError("truncated WAV chunk payload")
        chunk_ids.append(kind.decode("ascii", errors="strict"))
        if kind == b"fmt ":
            if size < 16:
                raise ValueError("short fmt chunk")
            fmt = struct.unpack("<HHIIHH", payload[start : start + 16])
        elif kind == b"data":
            if data is not None:
                raise ValueError("multiple data chunks are not allowed")
            data = payload[start:end]
        offset = end + (size % 2)

    if fmt is None or data is None:
        raise ValueError("WAV requires fmt and data chunks")
    format_tag, channels, rate, byte_rate, block_align, bits = fmt
    if format_tag != 3 or bits != 32 or channels not in (1, 2):
        raise ValueError("only mono/stereo IEEE float32 WAV is accepted")
    if block_align != channels * 4 or byte_rate != rate * block_align:
        raise ValueError("inconsistent WAV alignment or byte rate")
    if len(data) % block_align:
        raise ValueError("data chunk is not frame-aligned")
    samples = np.frombuffer(data, dtype="<f4").astype(np.float64)
    samples = samples.reshape((-1, channels))
    if channels == 1:
        samples = samples[:, 0]
    return rate, samples, chunk_ids


def read_float32_wav(path: Path) -> tuple[int, np.ndarray, list[str]]:
    return parse_float32_wav_bytes(path.read_bytes())


def time_axis(sample_count: int = SAMPLE_COUNT, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.arange(sample_count, dtype=np.float64) / float(sample_rate)


def left_integral(values: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    result[1:] = np.cumsum(values[:-1], dtype=np.float64) / float(sample_rate)
    return result


def fm_encode(
    message: np.ndarray,
    carrier_hz: float = PRIMARY_CARRIER_HZ,
    k_hz_per_unit: float = FM_K_HZ_PER_UNIT,
    amplitude: float = CARRIER_AMPLITUDE,
) -> np.ndarray:
    t = time_axis(len(message))
    phase = 2.0 * np.pi * carrier_hz * t
    phase += 2.0 * np.pi * k_hz_per_unit * left_integral(message)
    return amplitude * np.cos(phase)


def pm_encode(
    message: np.ndarray,
    carrier_hz: float = PRIMARY_CARRIER_HZ,
    k_rad_per_unit: float = PM_K_RAD_PER_UNIT,
    amplitude: float = CARRIER_AMPLITUDE,
) -> np.ndarray:
    t = time_axis(len(message))
    phase = 2.0 * np.pi * carrier_hz * t + k_rad_per_unit * message
    return amplitude * np.cos(phase)


def analytic_signal(real_signal: np.ndarray) -> np.ndarray:
    """FFT-domain analytic signal with even/odd and Nyquist conventions frozen."""

    values = np.asarray(real_signal, dtype=np.float64)
    spectrum = np.fft.fft(values)
    multiplier = np.zeros(values.size, dtype=np.float64)
    multiplier[0] = 1.0
    if values.size % 2 == 0:
        multiplier[1 : values.size // 2] = 2.0
        multiplier[values.size // 2] = 1.0
    else:
        multiplier[1 : (values.size + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier)


def recover_fm(
    signal: np.ndarray,
    carrier_hz: float = PRIMARY_CARRIER_HZ,
    k_hz_per_unit: float = FM_K_HZ_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    phase = np.unwrap(np.angle(analytic_signal(signal)))
    instantaneous_frequency = np.gradient(phase) * SAMPLE_RATE / (2.0 * np.pi)
    return (instantaneous_frequency - carrier_hz) / k_hz_per_unit, instantaneous_frequency


def recover_pm(
    signal: np.ndarray,
    carrier_hz: float = PRIMARY_CARRIER_HZ,
    k_rad_per_unit: float = PM_K_RAD_PER_UNIT,
    edge_samples: int = EDGE_SAMPLES,
) -> np.ndarray:
    t = time_axis(len(signal))
    phase = np.unwrap(np.angle(analytic_signal(signal)))
    residual = phase - 2.0 * np.pi * carrier_hz * t
    interior = residual[edge_samples:-edge_samples]
    residual -= float(np.mean(interior))
    return residual / k_rad_per_unit


def conjugate_phase_subtract(state: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.complex128) * np.conjugate(
        np.asarray(query, dtype=np.complex128)
    )


def ordinary_phase_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.complex128) * np.asarray(b, dtype=np.complex128)


def multitone_complex_state(
    frequencies_hz: Iterable[float], coefficients: Iterable[complex]
) -> np.ndarray:
    t = time_axis()
    result = np.zeros(SAMPLE_COUNT, dtype=np.complex128)
    for frequency, coefficient in zip(frequencies_hz, coefficients, strict=True):
        result += complex(coefficient) * np.exp(2j * np.pi * float(frequency) * t)
    return result


def multitone_real(
    frequencies_hz: Iterable[float], coefficients: Iterable[complex]
) -> np.ndarray:
    return np.real(multitone_complex_state(frequencies_hz, coefficients))


def sample_delay(signal: np.ndarray, samples: int, *, mode: str) -> np.ndarray:
    values = np.asarray(signal)
    if samples < 0:
        raise ValueError("delay must be nonnegative")
    if mode == "circular":
        return np.roll(values, samples)
    if mode == "zero":
        result = np.zeros_like(values)
        if samples < len(values):
            result[samples:] = values[: len(values) - samples]
        return result
    raise ValueError("delay mode must be circular or zero")


def phase_rotate(signal: np.ndarray, radians: float) -> np.ndarray:
    return np.asarray(signal, dtype=np.complex128) * np.exp(1j * radians)


def filter_bank_projection(
    signal: np.ndarray, frequencies_hz: Iterable[float], *, analytic: bool = False
) -> dict[str, complex]:
    values = np.asarray(signal)
    t = time_axis(len(values))
    scale = 1.0 / len(values) if analytic or np.iscomplexobj(values) else 2.0 / len(values)
    return {
        f"{float(frequency):.6f}": complex(
            scale
            * np.sum(values * np.exp(-2j * np.pi * float(frequency) * t), dtype=np.complex128)
        )
        for frequency in frequencies_hz
    }


def correlation(a: np.ndarray, b: np.ndarray, *, normalized: bool) -> complex:
    left = np.asarray(a, dtype=np.complex128)
    right = np.asarray(b, dtype=np.complex128)
    value = np.vdot(left, right)
    if not normalized:
        return complex(value)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0 + 0.0j
    return complex(value / denominator)


def matched_filter(signal: np.ndarray, template: np.ndarray, *, normalized: bool) -> complex:
    return correlation(template, signal, normalized=normalized)


def fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full linear convolution; values outside the finite inputs are exactly zero."""

    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    output_size = len(left) + len(right) - 1
    fft_size = 1 << (output_size - 1).bit_length()
    result = np.fft.irfft(np.fft.rfft(left, fft_size) * np.fft.rfft(right, fft_size), fft_size)
    return result[:output_size]


def polynomial_nonlinear_mix(signal: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    return values + alpha * values * values


def normalized_vector_error(observed: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(expected))
    if denominator == 0.0:
        return 0.0 if float(np.linalg.norm(observed)) == 0.0 else math.inf
    return float(np.linalg.norm(observed - expected) / denominator)


def sample_rmse(observed: np.ndarray, expected: np.ndarray) -> float:
    delta = np.asarray(observed, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def angular_error(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.angle(np.asarray(observed) * np.conjugate(np.asarray(expected)))


def _baseband_messages() -> tuple[np.ndarray, np.ndarray]:
    t = time_axis()
    fm_message = 0.52 * np.sin(2.0 * np.pi * 137.0 * t)
    fm_message += 0.21 * np.cos(2.0 * np.pi * 311.0 * t)
    pm_message = 0.46 * np.sin(2.0 * np.pi * 173.0 * t)
    pm_message += 0.19 * np.sin(2.0 * np.pi * 421.0 * t)
    return fm_message, pm_message


def _matched_pair() -> tuple[np.ndarray, np.ndarray]:
    t = time_axis()
    sweep_rate = (4_000.0 - 1_000.0) / DURATION_SECONDS
    phase = 2.0 * np.pi * (1_000.0 * t + 0.5 * sweep_rate * t * t) + 0.2
    envelope = np.sin(np.pi * np.arange(SAMPLE_COUNT) / (SAMPLE_COUNT - 1)) ** 2
    target = 0.75 * envelope * np.sin(phase)
    return target, target[::-1].copy()


def generate_fixture_arrays() -> dict[str, dict[str, Any]]:
    t = time_axis()
    fm_message, pm_message = _baseband_messages()
    fm_wave = fm_encode(fm_message)
    pm_wave = pm_encode(pm_message)

    relation_phase = 0.28 * np.sin(2.0 * np.pi * 223.0 * t)
    state_iq = 0.70 * np.exp(1j * (2.0 * np.pi * PRIMARY_CARRIER_HZ * t + relation_phase))
    query_iq = 0.70 * np.exp(1j * 2.0 * np.pi * PRIMARY_CARRIER_HZ * t)

    multitone_frequencies = [6_300.0, 8_000.0, 9_700.0]
    multitone_coefficients = [
        0.22 * np.exp(0.31j),
        0.29 * np.exp(-0.72j),
        0.17 * np.exp(1.11j),
    ]
    multitone = multitone_real(multitone_frequencies, multitone_coefficients)
    multitone_delayed = sample_delay(multitone, 37, mode="circular")

    matched_target, matched_sham = _matched_pair()
    nonlinear_input = 0.30 * np.cos(2.0 * np.pi * 7_000.0 * t)
    nonlinear_input += 0.30 * np.cos(2.0 * np.pi * 8_200.0 * t)
    nonlinear_output = polynomial_nonlinear_mix(nonlinear_input)

    return {
        "analytic_query_iq": {
            "samples": np.column_stack((query_iq.real, query_iq.imag)),
            "file": "analytic_query_iq.wav",
            "semantic_role": "two_axis_analytic_query",
            "parameters": {"carrier_hz": PRIMARY_CARRIER_HZ, "amplitude_per_axis": 0.70},
        },
        "analytic_state_iq": {
            "samples": np.column_stack((state_iq.real, state_iq.imag)),
            "file": "analytic_state_iq.wav",
            "semantic_role": "two_axis_analytic_state",
            "parameters": {
                "carrier_hz": PRIMARY_CARRIER_HZ,
                "phase_modulation_hz": 223.0,
                "phase_modulation_rad": 0.28,
                "amplitude_per_axis": 0.70,
            },
        },
        "fm_encoded": {
            "samples": fm_wave,
            "file": "fm_encoded.wav",
            "semantic_role": "fm_carrier",
            "parameters": {
                "carrier_hz": PRIMARY_CARRIER_HZ,
                "k_hz_per_unit": FM_K_HZ_PER_UNIT,
                "amplitude": CARRIER_AMPLITUDE,
                "integral": "left_riemann_integral_zero_at_sample_0",
            },
        },
        "fm_message": {
            "samples": fm_message,
            "file": "fm_message.wav",
            "semantic_role": "fm_baseband_message",
            "parameters": {"tones_hz": [137.0, 311.0], "coefficients": [0.52, 0.21]},
        },
        "matched_pair_b": {
            "samples": matched_sham,
            "file": "wave_pair_b.wav",
            "semantic_role": "matched_filter_pair_member_b",
            "parameters": {"construction": "time_reverse_of_pair_member_a"},
        },
        "matched_pair_a": {
            "samples": matched_target,
            "file": "wave_pair_a.wav",
            "semantic_role": "matched_filter_pair_member_a",
            "parameters": {
                "chirp_start_hz": 1_000.0,
                "chirp_end_hz": 4_000.0,
                "envelope": "sin_squared_end_zero",
            },
        },
        "multitone_delayed": {
            "samples": multitone_delayed,
            "file": "multitone_delayed.wav",
            "semantic_role": "circular_exact_sample_delay_control",
            "parameters": {"delay_samples": 37, "boundary": "circular"},
        },
        "multitone_state": {
            "samples": multitone,
            "file": "multitone_state.wav",
            "semantic_role": "complex_coefficient_multitone_state",
            "parameters": {
                "frequencies_hz": multitone_frequencies,
                "complex_coefficients": [
                    {"magnitude": 0.22, "phase_rad": 0.31},
                    {"magnitude": 0.29, "phase_rad": -0.72},
                    {"magnitude": 0.17, "phase_rad": 1.11},
                ],
            },
        },
        "nonlinear_input": {
            "samples": nonlinear_input,
            "file": "nonlinear_input.wav",
            "semantic_role": "two_tone_nonlinear_input",
            "parameters": {"frequencies_hz": [7_000.0, 8_200.0], "amplitudes": [0.30, 0.30]},
        },
        "nonlinear_output": {
            "samples": nonlinear_output,
            "file": "nonlinear_output.wav",
            "semantic_role": "ordinary_polynomial_intermodulation_output",
            "parameters": {"law": "y=x+alpha*x^2", "alpha": 0.8},
        },
        "pm_encoded": {
            "samples": pm_wave,
            "file": "pm_encoded.wav",
            "semantic_role": "pm_carrier",
            "parameters": {
                "carrier_hz": PRIMARY_CARRIER_HZ,
                "k_rad_per_unit": PM_K_RAD_PER_UNIT,
                "amplitude": CARRIER_AMPLITUDE,
            },
        },
        "pm_message": {
            "samples": pm_message,
            "file": "pm_message.wav",
            "semantic_role": "pm_baseband_message",
            "parameters": {"tones_hz": [173.0, 421.0], "coefficients": [0.46, 0.19]},
        },
    }


TEST_SPECS: list[dict[str, Any]] = [
    {"id": "fm_round_trip_recovery", "inputs": ["fm_message", "fm_encoded"], "expected": "recovered baseband", "metric": "sample_rmse", "comparator": "<=", "tolerance": 0.012, "edge": "exclude first and last 4096 samples"},
    {"id": "pm_round_trip_recovery", "inputs": ["pm_message", "pm_encoded"], "expected": "recovered zero-mean baseband", "metric": "sample_rmse", "comparator": "<=", "tolerance": 0.012, "edge": "exclude first and last 4096 samples; PM DC phase is removed"},
    {"id": "analytic_signal", "inputs": ["multitone_state"], "expected": "real part preserved and negative-frequency energy suppressed", "metric": "max_reconstruction_or_negative_energy_ratio", "comparator": "<=", "tolerance": 1e-12, "edge": "FFT periodic boundary"},
    {"id": "conjugate_phase_subtraction", "inputs": ["analytic_state_iq", "analytic_query_iq"], "expected": "state phase minus query phase", "metric": "max_angular_error_rad", "comparator": "<=", "tolerance": 1e-6, "edge": "parsed float32 stereo I/Q; modulo 2*pi"},
    {"id": "ordinary_phase_addition", "inputs": ["analytic_state_iq", "analytic_query_iq"], "expected": "state phase plus query phase", "metric": "max_angular_error_rad", "comparator": "<=", "tolerance": 1e-6, "edge": "parsed float32 stereo I/Q; modulo 2*pi"},
    {"id": "multitone_complex_projection", "inputs": ["multitone_state"], "expected": "three frozen complex coefficients", "metric": "normalized_vector_error", "comparator": "<=", "tolerance": 1e-7, "edge": "parsed float32 integer-cycle two-second record"},
    {"id": "delay_phase_rotation", "inputs": ["multitone_state", "multitone_delayed"], "expected": "coefficient rotation exp(-j*2*pi*f*37/fs)", "metric": "normalized_vector_error", "comparator": "<=", "tolerance": 1e-7, "edge": "parsed float32 circular exact-sample delay"},
    {"id": "zero_filled_delay", "inputs": ["multitone_state"], "expected": "first 37 outputs zero and remaining samples equal the shifted input", "metric": "max_absolute_sample_error", "comparator": "<=", "tolerance": 0.0, "edge": "causal zero-filled exact-sample delay"},
    {"id": "fractional_phase_rotation_inverse", "inputs": ["analytic_state_iq"], "expected": "rotation followed by exact inverse", "metric": "normalized_vector_error", "comparator": "<=", "tolerance": 1e-14, "edge": "no time-boundary operation"},
    {"id": "filter_bank_projection", "inputs": ["multitone_state"], "expected": "frozen bins recover coefficients and null bin remains small", "metric": "max_coefficient_or_null_error", "comparator": "<=", "tolerance": 1e-7, "edge": "parsed float32 complex inner product over full record"},
    {"id": "matched_filter_energy_sham", "inputs": ["matched_pair_a", "matched_pair_b"], "expected": "A-to-A score exceeds B-to-A score by at least 0.80", "metric": "pair_a_minus_pair_b_normalized_score", "comparator": ">=", "tolerance": 0.80, "edge": "full parsed float32 records; zero norm returns zero"},
    {"id": "correlation_zero_denominator", "inputs": [], "expected": "normalized correlation with zero vector equals complex zero", "metric": "absolute_value", "comparator": "<=", "tolerance": 0.0, "edge": "explicit denominator-zero convention"},
    {"id": "unnormalized_correlation", "inputs": ["matched_pair_a", "matched_pair_b"], "expected": "unnormalized result equals explicit complex inner product", "metric": "absolute_error", "comparator": "<=", "tolerance": 0.0, "edge": "full parsed float32 records"},
    {"id": "convolution_full_boundary", "inputs": ["matched_pair_a"], "expected": "FFT full convolution equals direct convolution", "metric": "sample_rmse", "comparator": "<=", "tolerance": 1e-13, "edge": "finite inputs zero outside support; full N+M-1 output"},
    {"id": "nonlinear_intermodulation", "inputs": ["nonlinear_input", "nonlinear_output"], "expected": "predeclared 1200 and 15200 Hz products each have amplitude 0.072", "metric": "max_complex_coefficient_error", "comparator": "<=", "tolerance": 1e-7, "edge": "parsed float32 integer-cycle record; y=x+0.8*x^2"},
    {"id": "label_only_renaming_invariance", "inputs": ["multitone_state"], "expected": "renaming public labels changes neither samples nor query result", "metric": "max_sample_or_result_error", "comparator": "<=", "tolerance": 0.0, "edge": "labels are external aliases, not generator inputs"},
    {"id": "query_scramble_breaks_recovery", "inputs": ["analytic_state_iq", "analytic_query_iq"], "expected": "37 Hz wrong query causes at least 0.50 rad RMS phase error", "metric": "phase_rmse_rad", "comparator": ">=", "tolerance": 0.50, "edge": "exclude first and last 4096 samples"},
    {"id": "finite_answer_cache_replay", "inputs": ["analytic_state_iq", "analytic_query_iq"], "expected": "finite four-query answer vector exactly replays delayed selection", "metric": "max_absolute_error", "comparator": "<=", "tolerance": 1e-14, "edge": "coefficient derived from parsed I/Q; closed query set {0,pi/2,pi,3pi/2}"},
    {"id": "compressed_generator_held_out_query", "inputs": ["analytic_state_iq", "analytic_query_iq"], "expected": "two-parameter sinusoidal generator replays held-out continuous queries", "metric": "max_absolute_error", "comparator": "<=", "tolerance": 1e-14, "edge": "coefficient derived from parsed I/Q; held-out q values absent from finite cache"},
    {"id": "public_manifest_parameter_replay", "inputs": ["AUDIO_WAVE_FIXTURE_MANIFEST.json"], "expected": "public manifest parameters reconstruct all multitone complex coefficients", "metric": "normalized_vector_error", "comparator": "<=", "tolerance": 0.0, "edge": "manifest-only non-sample side channel is deliberately public"},
    {"id": "ordinary_dsp_replay", "inputs": ["all committed fixtures"], "expected": "public DSP reproduces every applicable scored algebra operation", "metric": "max_fraction_of_frozen_tolerance", "comparator": "<=", "tolerance": 1.0, "edge": "inherits each operation-specific parsed-fixture convention"},
    {"id": "ordinary_linear_filter_replay", "inputs": ["matched_pair_a"], "expected": "public FFT filter exactly replays linear response", "metric": "sample_rmse", "comparator": "<=", "tolerance": 1e-13, "edge": "full linear convolution"},
    {"id": "ordinary_nonlinear_filter_replay", "inputs": ["nonlinear_input", "nonlinear_output"], "expected": "public polynomial law replays float32 output within serialization error", "metric": "sample_rmse", "comparator": "<=", "tolerance": 5e-8, "edge": "parsed float32 pointwise polynomial"},
    {"id": "spectral_energy_leakage_control", "inputs": ["matched_pair_a", "matched_pair_b"], "expected": "magnitude spectra match while matched responses remain separated", "metric": "max_spectral_error_or_gap_deficit", "comparator": "<=", "tolerance": 1e-12, "edge": "parsed float32 real-signal time reversal and full FFT"},
    {"id": "phase_label_leakage_control", "inputs": ["multitone_state"], "expected": "phase truth is not encoded in external label names", "metric": "max_sample_or_projection_error", "comparator": "<=", "tolerance": 0.0, "edge": "alias permutation only"},
    {"id": "file_metadata_stripping_invariance", "inputs": ["multitone_state"], "expected": "LIST metadata addition and stripping leave samples and projection invariant", "metric": "max_sample_or_projection_error", "comparator": "<=", "tolerance": 0.0, "edge": "unknown even-padded RIFF chunks ignored"},
    {"id": "query_preselection_attack", "inputs": ["analytic_state_iq", "analytic_query_iq"], "expected": "source-visible preselected query admits exact answer precomputation", "metric": "absolute_error", "comparator": "<=", "tolerance": 1e-14, "edge": "coefficient derived from parsed I/Q; query known before closure"},
    {"id": "file_storage_fake_persistence", "inputs": ["multitone_state"], "expected": "reopened serialized WAV exactly reproduces the observable", "metric": "normalized_vector_error", "comparator": "<=", "tolerance": 1e-7, "edge": "float32 serialization; classified as file persistence only"},
    {"id": "digital_restoration_r0_only", "inputs": ["analytic_state_iq"], "expected": "digital inverse returns samples but supports only R0-like algebraic return", "metric": "normalized_vector_error", "comparator": "<=", "tolerance": 1e-14, "edge": "no physical carrier or baseline distribution"},
]


def _test_record(
    test_id: str,
    observed: float,
    passed: bool,
    *,
    measurements: dict[str, Any] | None = None,
    conclusion: str,
) -> dict[str, Any]:
    spec = next(item for item in TEST_SPECS if item["id"] == test_id)
    return {
        "id": test_id,
        "status": "PASS" if passed else "FAIL",
        "metric": spec["metric"],
        "observed": _q(observed),
        "comparator": spec["comparator"],
        "tolerance": spec["tolerance"],
        "edge": spec["edge"],
        "measurements": measurements or {},
        "conclusion": conclusion,
    }


def run_reference_tests(fixtures: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    arrays = {key: np.asarray(value["samples"]) for key, value in fixtures.items()}
    results: list[dict[str, Any]] = []
    interior = slice(EDGE_SAMPLES, -EDGE_SAMPLES)
    state = arrays["analytic_state_iq"][:, 0] + 1j * arrays["analytic_state_iq"][:, 1]
    query = arrays["analytic_query_iq"][:, 0] + 1j * arrays["analytic_query_iq"][:, 1]

    fm_recovered, fm_frequency = recover_fm(arrays["fm_encoded"])
    fm_error = sample_rmse(fm_recovered[interior], arrays["fm_message"][interior])
    results.append(_test_record("fm_round_trip_recovery", fm_error, fm_error <= 0.012, measurements={"frequency_rmse_hz": _q(fm_error * FM_K_HZ_PER_UNIT), "carrier_mean_hz": _q(np.mean(fm_frequency[interior]))}, conclusion="Ordinary analytic-signal FM recovery meets the frozen envelope."))

    pm_recovered = recover_pm(arrays["pm_encoded"])
    pm_error = sample_rmse(pm_recovered[interior], arrays["pm_message"][interior])
    results.append(_test_record("pm_round_trip_recovery", pm_error, pm_error <= 0.012, measurements={"phase_rmse_rad": _q(pm_error * PM_K_RAD_PER_UNIT)}, conclusion="Ordinary analytic-signal PM recovery meets the frozen envelope."))

    analytic = analytic_signal(arrays["multitone_state"])
    reconstruction = sample_rmse(analytic.real, arrays["multitone_state"])
    spectrum = np.fft.fft(analytic)
    negative_ratio = float(np.sum(np.abs(spectrum[SAMPLE_COUNT // 2 + 1 :]) ** 2) / np.sum(np.abs(spectrum[1 : SAMPLE_COUNT // 2]) ** 2))
    analytic_error = max(reconstruction, negative_ratio)
    results.append(_test_record("analytic_signal", analytic_error, analytic_error <= 1e-12, measurements={"real_reconstruction_rmse": _q(reconstruction), "negative_to_positive_energy_ratio": _q(negative_ratio)}, conclusion="The frozen FFT Hilbert convention constructs an analytic complex signal."))

    t = time_axis()
    state_phase = 2.0 * np.pi * PRIMARY_CARRIER_HZ * t + 0.28 * np.sin(2.0 * np.pi * 223.0 * t)
    query_phase = 2.0 * np.pi * PRIMARY_CARRIER_HZ * t
    relative = conjugate_phase_subtract(state, query)
    expected_relative = (0.70**2) * np.exp(1j * (state_phase - query_phase))
    conjugate_error = float(np.max(np.abs(angular_error(relative, expected_relative))))
    results.append(_test_record("conjugate_phase_subtraction", conjugate_error, conjugate_error <= 1e-6, conclusion="Conjugate mixing of parsed stereo I/Q fixtures performs phase subtraction."))

    added = ordinary_phase_add(state, query)
    expected_added = (0.70**2) * np.exp(1j * (state_phase + query_phase))
    addition_error = float(np.max(np.abs(angular_error(added, expected_added))))
    results.append(_test_record("ordinary_phase_addition", addition_error, addition_error <= 1e-6, conclusion="Ordinary multiplication of parsed stereo I/Q fixtures performs phase addition."))

    frequencies = np.array([6_300.0, 8_000.0, 9_700.0])
    expected_coefficients = np.array([0.22 * np.exp(0.31j), 0.29 * np.exp(-0.72j), 0.17 * np.exp(1.11j)])
    projected_map = filter_bank_projection(arrays["multitone_state"], frequencies)
    projected = np.array([projected_map[f"{frequency:.6f}"] for frequency in frequencies])
    projection_error = normalized_vector_error(projected, expected_coefficients)
    results.append(_test_record("multitone_complex_projection", projection_error, projection_error <= 1e-7, measurements={"frequencies_hz": frequencies.tolist()}, conclusion="Complex coefficients, including phase, are recovered from the committed float32 fixture."))

    delayed_map = filter_bank_projection(arrays["multitone_delayed"], frequencies)
    delayed_coefficients = np.array([delayed_map[f"{frequency:.6f}"] for frequency in frequencies])
    predicted_delayed = expected_coefficients * np.exp(-2j * np.pi * frequencies * 37.0 / SAMPLE_RATE)
    delay_error = normalized_vector_error(delayed_coefficients, predicted_delayed)
    results.append(_test_record("delay_phase_rotation", delay_error, delay_error <= 1e-7, measurements={"delay_samples": 37}, conclusion="The committed circular-delay fixture produces the predeclared per-bin phase rotation."))

    zero_delayed = sample_delay(arrays["multitone_state"], 37, mode="zero")
    zero_delay_error = max(
        float(np.max(np.abs(zero_delayed[:37]))),
        float(np.max(np.abs(zero_delayed[37:] - arrays["multitone_state"][:-37]))),
    )
    results.append(_test_record("zero_filled_delay", zero_delay_error, zero_delay_error == 0.0, measurements={"delay_samples": 37}, conclusion="The zero-filled delay convention is exercised exactly."))

    rotated = phase_rotate(state, 0.731)
    restored = phase_rotate(rotated, -0.731)
    rotation_error = normalized_vector_error(restored, state)
    results.append(_test_record("fractional_phase_rotation_inverse", rotation_error, rotation_error <= 1e-14, conclusion="Complex phase rotation has an exact algebraic inverse."))

    bank_map = filter_bank_projection(arrays["multitone_state"], [6_300.0, 8_000.0, 9_700.0, 7_111.0])
    bank_coefficients = np.array([bank_map[f"{frequency:.6f}"] for frequency in frequencies])
    bank_error = max(normalized_vector_error(bank_coefficients, expected_coefficients), abs(bank_map["7111.000000"]))
    results.append(_test_record("filter_bank_projection", bank_error, bank_error <= 1e-7, measurements={"null_bin_absolute_amplitude": _q(abs(bank_map["7111.000000"]))}, conclusion="The frozen complex filter bank recovers occupied bins from the committed float32 fixture and rejects a declared null bin."))

    target_score = abs(matched_filter(arrays["matched_pair_a"], arrays["matched_pair_a"], normalized=True))
    sham_score = abs(matched_filter(arrays["matched_pair_b"], arrays["matched_pair_a"], normalized=True))
    matched_gap = float(target_score - sham_score)
    results.append(_test_record("matched_filter_energy_sham", matched_gap, matched_gap >= 0.80, measurements={"target_score": _q(target_score), "sham_score": _q(sham_score)}, conclusion="Matched filtering uses phase/time structure rather than energy alone."))

    zero_correlation = abs(correlation(np.zeros(8), np.ones(8), normalized=True))
    results.append(_test_record("correlation_zero_denominator", zero_correlation, zero_correlation == 0.0, conclusion="The zero-denominator convention is frozen to complex zero."))

    unnormalized_observed = correlation(arrays["matched_pair_a"], arrays["matched_pair_b"], normalized=False)
    unnormalized_expected = complex(
        np.vdot(
            arrays["matched_pair_a"].astype(np.complex128),
            arrays["matched_pair_b"].astype(np.complex128),
        )
    )
    unnormalized_error = abs(unnormalized_observed - unnormalized_expected)
    results.append(_test_record("unnormalized_correlation", unnormalized_error, unnormalized_error == 0.0, conclusion="The unnormalized correlation path equals the explicit complex inner product."))

    convolution_input = arrays["matched_pair_a"][:4_096]
    kernel = np.hanning(257)
    kernel /= np.sum(kernel)
    fft_result = fft_convolve(convolution_input, kernel)
    direct_result = np.convolve(convolution_input, kernel, mode="full")
    convolution_error = sample_rmse(fft_result, direct_result)
    results.append(_test_record("convolution_full_boundary", convolution_error, convolution_error <= 1e-13, conclusion="FFT convolution matches the direct full-boundary reference."))

    intermodulation_map = filter_bank_projection(arrays["nonlinear_output"], [1_200.0, 15_200.0])
    intermodulation = np.array([intermodulation_map["1200.000000"], intermodulation_map["15200.000000"]])
    expected_intermodulation = np.array([0.072 + 0.0j, 0.072 + 0.0j])
    nonlinear_error = float(np.max(np.abs(intermodulation - expected_intermodulation)))
    results.append(_test_record("nonlinear_intermodulation", nonlinear_error, nonlinear_error <= 1e-7, measurements={"difference_hz": 1_200.0, "sum_hz": 15_200.0, "expected_amplitude": 0.072}, conclusion="The committed float32 output contains the declared ordinary polynomial intermodulation products."))

    aliases_before = {"state_alias": "multitone_state", "control_alias": "matched_pair_a"}
    aliases_after = {"renamed_control": "matched_pair_a", "renamed_state": "multitone_state"}
    state_before = arrays[aliases_before["state_alias"]]
    state_after = arrays[aliases_after["renamed_state"]]
    projection_after_map = filter_bank_projection(state_after, frequencies)
    projection_after = np.array(
        [projection_after_map[f"{frequency:.6f}"] for frequency in frequencies]
    )
    label_sample_error = float(np.max(np.abs(state_before - state_after)))
    label_projection_error = normalized_vector_error(projected, projection_after)
    label_error = max(label_sample_error, label_projection_error)
    results.append(_test_record("label_only_renaming_invariance", label_error, label_error == 0.0, measurements={"aliases_before": aliases_before, "aliases_after": aliases_after}, conclusion="A bijective external alias rename changes neither content resolution nor the scored result."))

    wrong_query = 0.70 * np.exp(1j * 2.0 * np.pi * (PRIMARY_CARRIER_HZ + 37.0) * t)
    wrong_relative = conjugate_phase_subtract(state, wrong_query)
    wrong_phase = np.angle(wrong_relative * np.conjugate(relative))
    wrong_rmse = float(np.sqrt(np.mean(wrong_phase[interior] ** 2)))
    results.append(_test_record("query_scramble_breaks_recovery", wrong_rmse, wrong_rmse >= 0.50, conclusion="A frequency-scrambled query destroys the intended relative-phase recovery."))

    coefficient = complex(np.mean(relative))
    finite_queries = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    cache = np.real(coefficient * np.exp(-1j * finite_queries))
    cache_replay = np.array([cache[index] for index in range(len(finite_queries))])
    direct_finite = np.real(coefficient * np.exp(-1j * finite_queries))
    cache_error = float(np.max(np.abs(cache_replay - direct_finite)))
    results.append(_test_record("finite_answer_cache_replay", cache_error, cache_error <= 1e-14, conclusion="The finite-query theorem is constructively reproduced: the literal cache survives."))

    held_out_queries = np.array([0.37, 0.731, 1.13, 2.77])
    amplitude, phase = abs(coefficient), np.angle(coefficient)
    generated = amplitude * np.cos(phase - held_out_queries)
    direct_held_out = np.real(coefficient * np.exp(-1j * held_out_queries))
    generator_error = float(np.max(np.abs(generated - direct_held_out)))
    results.append(_test_record("compressed_generator_held_out_query", generator_error, generator_error <= 1e-14, conclusion="A compact formula survives held-out continuous queries; continuity is not capacity separation."))

    manifest = json.loads((ROOT / "AUDIO_WAVE_FIXTURE_MANIFEST.json").read_text(encoding="utf-8"))
    multitone_record = next(
        record for record in manifest["fixtures"] if record["fixture_id"] == "multitone_state"
    )
    manifest_coefficients = np.array(
        [
            float(item["magnitude"]) * np.exp(1j * float(item["phase_rad"]))
            for item in multitone_record["generator_parameters"]["complex_coefficients"]
        ]
    )
    manifest_replay_error = normalized_vector_error(manifest_coefficients, expected_coefficients)
    results.append(_test_record("public_manifest_parameter_replay", manifest_replay_error, manifest_replay_error == 0.0, conclusion="Public non-sample manifest parameters exactly replay the declared multitone coefficients and remain an admissible ordinary side channel."))

    linear_replay_error = sample_rmse(fft_convolve(convolution_input, kernel), direct_result)
    nonlinear_replayed = polynomial_nonlinear_mix(arrays["nonlinear_input"])
    nonlinear_replay_error = sample_rmse(nonlinear_replayed, arrays["nonlinear_output"])
    dsp_ratios = {
        "fm": fm_error / 0.012,
        "pm": pm_error / 0.012,
        "analytic": analytic_error / 1e-12,
        "conjugate": conjugate_error / 1e-6,
        "addition": addition_error / 1e-6,
        "multitone": projection_error / 1e-7,
        "circular_delay": delay_error / 1e-7,
        "rotation_inverse": rotation_error / 1e-14,
        "filter_bank": bank_error / 1e-7,
        "convolution": convolution_error / 1e-13,
        "nonlinear_intermodulation": nonlinear_error / 1e-7,
        "linear_replay": linear_replay_error / 1e-13,
        "nonlinear_replay": nonlinear_replay_error / 5e-8,
        "matched_filter": 0.0 if matched_gap >= 0.80 else 1.0 + (0.80 - matched_gap) / 0.80,
        "zero_filled_delay": 0.0 if zero_delay_error == 0.0 else math.inf,
        "zero_denominator_correlation": 0.0 if zero_correlation == 0.0 else math.inf,
        "unnormalized_correlation": 0.0 if unnormalized_error == 0.0 else math.inf,
    }
    dsp_ratio = float(max(dsp_ratios.values()))
    results.append(_test_record("ordinary_dsp_replay", dsp_ratio, dsp_ratio <= 1.0, measurements={key: _q(value) for key, value in sorted(dsp_ratios.items())}, conclusion="The public ordinary-DSP adversary reproduces every applicable committed-fixture algebra result within its own frozen tolerance."))
    results.append(_test_record("ordinary_linear_filter_replay", linear_replay_error, linear_replay_error <= 1e-13, conclusion="An ordinary linear-filter model remains sufficient."))
    results.append(_test_record("ordinary_nonlinear_filter_replay", nonlinear_replay_error, nonlinear_replay_error <= 5e-8, conclusion="The public polynomial law explains the committed float32 output within the frozen serialization tolerance."))

    target_spectrum = np.abs(np.fft.fft(arrays["matched_pair_a"]))
    sham_spectrum = np.abs(np.fft.fft(arrays["matched_pair_b"]))
    spectral_error = normalized_vector_error(sham_spectrum, target_spectrum)
    gap_deficit = max(0.0, 0.80 - matched_gap)
    spectral_control = max(spectral_error, gap_deficit)
    results.append(_test_record("spectral_energy_leakage_control", spectral_control, spectral_control <= 1e-12, measurements={"magnitude_spectrum_error": _q(spectral_error), "matched_score_gap": _q(matched_gap)}, conclusion="Energy and magnitude spectrum do not encode the matched-filter answer."))

    phase_labels_before = {"alpha": 6_300.0, "beta": 8_000.0, "gamma": 9_700.0}
    phase_labels_after = {"renamed_gamma": 9_700.0, "renamed_alpha": 6_300.0, "renamed_beta": 8_000.0}
    before_by_frequency = {
        frequency: projected_map[f"{frequency:.6f}"] for frequency in phase_labels_before.values()
    }
    after_by_frequency = {
        frequency: projected_map[f"{frequency:.6f}"] for frequency in phase_labels_after.values()
    }
    phase_label_error = max(
        abs(before_by_frequency[frequency] - after_by_frequency[frequency])
        for frequency in sorted(before_by_frequency)
    )
    results.append(_test_record("phase_label_leakage_control", phase_label_error, phase_label_error == 0.0, measurements={"labels_before": phase_labels_before, "labels_after": phase_labels_after}, conclusion="A bijective phase-label rename leaves frequency-bound complex coefficients unchanged; public numeric parameters remain separately replayable."))

    metadata_payload = float32_wav_bytes(arrays["multitone_state"], list_metadata=b"INFOpublic-note")
    rate_with_metadata, samples_with_metadata, chunks = parse_float32_wav_bytes(metadata_payload)
    stripped_payload = float32_wav_bytes(samples_with_metadata, sample_rate=rate_with_metadata)
    _, stripped_samples, stripped_chunks = parse_float32_wav_bytes(stripped_payload)
    metadata_sample_error = float(np.max(np.abs(samples_with_metadata - stripped_samples)))
    meta_projection = filter_bank_projection(samples_with_metadata, frequencies)
    stripped_projection = filter_bank_projection(stripped_samples, frequencies)
    meta_vector = np.array([meta_projection[f"{frequency:.6f}"] for frequency in frequencies])
    stripped_vector = np.array([stripped_projection[f"{frequency:.6f}"] for frequency in frequencies])
    metadata_projection_error = normalized_vector_error(meta_vector, stripped_vector)
    metadata_error = max(metadata_sample_error, metadata_projection_error)
    metadata_passed = metadata_error == 0.0 and "LIST" in chunks and "LIST" not in stripped_chunks
    results.append(_test_record("file_metadata_stripping_invariance", metadata_error, metadata_passed, measurements={"chunks_before": chunks, "chunks_after": stripped_chunks}, conclusion="A nonessential RIFF LIST chunk is neither required nor answer-bearing; manifest/path side channels are tested separately."))

    preselected_query = 0.731
    precomputed_answer = float(np.real(coefficient * np.exp(-1j * preselected_query)))
    realized_answer = float(amplitude * np.cos(phase - preselected_query))
    preselection_error = abs(precomputed_answer - realized_answer)
    results.append(_test_record("query_preselection_attack", preselection_error, preselection_error <= 1e-14, conclusion="A preselected query is exactly answer-smuggleable and cannot support query separation."))

    _, reopened_multitone, reopened_chunks = read_float32_wav(FIXTURE_ROOT / "multitone_state.wav")
    reopened_map = filter_bank_projection(reopened_multitone, frequencies)
    reopened_coefficients = np.array([reopened_map[f"{frequency:.6f}"] for frequency in frequencies])
    persistence_error = normalized_vector_error(reopened_coefficients, projected)
    results.append(_test_record("file_storage_fake_persistence", persistence_error, persistence_error <= 1e-7, measurements={"wav_chunks": reopened_chunks, "interface_buffer_exclusion_tested": False}, conclusion="The durable state exercised here is serialized file persistence only; interface-buffer persistence remains an untested future physical adversary."))

    digital_restored = phase_rotate(phase_rotate(state, 0.913), -0.913)
    restoration_error = normalized_vector_error(digital_restored, state)
    results.append(_test_record("digital_restoration_r0_only", restoration_error, restoration_error <= 1e-14, measurements={"restoration_tier": "R0_LIKE_DIGITAL_ALGEBRA_ONLY", "physical_R2_established": False}, conclusion="Exact digital inversion must not be promoted to physical R2 restoration."))

    expected_ids = [item["id"] for item in TEST_SPECS]
    observed_ids = [item["id"] for item in results]
    if observed_ids != expected_ids:
        raise AssertionError("reference result order diverged from frozen test order")
    return results


def build_manifest(fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for fixture_id in sorted(fixtures):
        definition = fixtures[fixture_id]
        path = FIXTURE_ROOT / definition["file"]
        rate, samples, chunks = read_float32_wav(path)
        if rate != SAMPLE_RATE or samples.shape[0] != SAMPLE_COUNT:
            raise AssertionError("fixture parser did not recover frozen dimensions")
        channels = 1 if samples.ndim == 1 else int(samples.shape[1])
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "fixture_id": fixture_id,
                "path": f"fixtures/{definition['file']}",
                "semantic_role": definition["semantic_role"],
                "generator_parameters": definition["parameters"],
                "sample_rate_hz": rate,
                "sample_count": int(samples.shape[0]),
                "duration_seconds": _q(samples.shape[0] / rate),
                "channel_count": channels,
                "dtype": "float32_le",
                "peak_amplitude": _q(np.max(np.abs(samples))),
                "rms_amplitude": _q(np.sqrt(np.mean(samples * samples))),
                "byte_count": size,
                "sha256": sha256_file(path),
                "riff_chunks": chunks,
            }
        )
    return {
        "schema": "audio_wave_fixture_manifest_v1",
        "generator": GENERATOR_ID,
        "reference_runtime": reference_runtime(),
        "numerical_envelope": {
            "sample_rate_hz": SAMPLE_RATE,
            "duration_seconds": DURATION_SECONDS,
            "sample_count": SAMPLE_COUNT,
            "primary_carrier_hz": PRIMARY_CARRIER_HZ,
            "baseband_bandwidth_max_hz": BASEBAND_LIMIT_HZ,
            "absolute_sample_ceiling": ABSOLUTE_SAMPLE_CEILING,
            "wav_encoding": WAV_ENCODING,
            "scalar_channels": "mono",
            "two_axis_channels": "stereo_I_then_Q",
        },
        "metadata_law": {
            "committed_chunks": ["fmt ", "data"],
            "answer_bearing_metadata_forbidden": True,
            "forbidden_manifest_keys": ["expected_answer", "winner_label", "query_result", "truth_label"],
        },
        "fixture_count": len(records),
        "total_bytes": total_bytes,
        "fixtures": records,
    }


def load_committed_fixtures(
    fixture_definitions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    committed: dict[str, dict[str, Any]] = {}
    for fixture_id, definition in fixture_definitions.items():
        rate, samples, _ = read_float32_wav(FIXTURE_ROOT / definition["file"])
        if rate != SAMPLE_RATE:
            raise AssertionError(f"unexpected sample rate for {fixture_id}")
        committed[fixture_id] = {**definition, "samples": samples}
    return committed


def reference_tests_document() -> dict[str, Any]:
    return {
        "schema": "audio_wave_reference_tests_v1",
        "generator": GENERATOR_ID,
        "reference_runtime": reference_runtime(),
        "freeze_law": "This file is emitted before scoring and binds every metric, comparator, tolerance, and edge convention.",
        "representation_law": "All named WAV inputs are parsed from committed IEEE float32 bytes before scoring.",
        "numerical_envelope": {
            "sample_rate_hz": SAMPLE_RATE,
            "duration_seconds": DURATION_SECONDS,
            "sample_count": SAMPLE_COUNT,
            "carrier_hz": PRIMARY_CARRIER_HZ,
            "baseband_limit_hz": BASEBAND_LIMIT_HZ,
            "absolute_sample_ceiling": ABSOLUTE_SAMPLE_CEILING,
            "fm_k_hz_per_unit": FM_K_HZ_PER_UNIT,
            "pm_k_rad_per_unit": PM_K_RAD_PER_UNIT,
            "default_edge_samples": EDGE_SAMPLES,
            "fixture_dtype": "float32_le",
            "compute_dtype": "float64_or_complex128_after_parse",
        },
        "tests": TEST_SPECS,
    }


def reference_results_document(
    test_results: list[dict[str, Any]], manifest_sha: str, tests_sha: str
) -> dict[str, Any]:
    passed = sum(item["status"] == "PASS" for item in test_results)
    return {
        "schema": "audio_wave_reference_results_v1",
        "generator": GENERATOR_ID,
        "reference_runtime": reference_runtime(),
        "fixture_manifest_sha256": manifest_sha,
        "reference_tests_sha256": tests_sha,
        "scored_representation": "parsed_committed_ieee_float32_wav",
        "claim_ceiling": CLAIM_CEILING if passed == len(test_results) else "NO_POSITIVE_OFFLINE_CLAIM",
        "physical_claims_established": [],
        "ordinary_software_replay_expected": True,
        "ordinary_software_replay_survived": all(
            next(item for item in test_results if item["id"] == test_id)["status"] == "PASS"
            for test_id in (
                "finite_answer_cache_replay",
                "compressed_generator_held_out_query",
                "public_manifest_parameter_replay",
                "ordinary_dsp_replay",
                "ordinary_linear_filter_replay",
                "ordinary_nonlinear_filter_replay",
            )
        ),
        "summary": {
            "test_count": len(test_results),
            "passed": passed,
            "failed": len(test_results) - passed,
        },
        "tests": test_results,
        "attestations": {
            "audio_playback_count": 0,
            "audio_recording_count": 0,
            "audio_hardware_contact_count": 0,
            "target_contact_count": 0,
            "network_contact_count": 0,
            "live_authority_used": False,
        },
    }


def build_package() -> dict[str, Any]:
    assert_frozen_runtime()
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    fixtures = generate_fixture_arrays()
    for fixture_id in sorted(fixtures):
        write_float32_wav(FIXTURE_ROOT / fixtures[fixture_id]["file"], fixtures[fixture_id]["samples"])

    tests_document = reference_tests_document()
    write_json(ROOT / "AUDIO_WAVE_REFERENCE_TESTS.json", tests_document)

    manifest = build_manifest(fixtures)
    write_json(ROOT / "AUDIO_WAVE_FIXTURE_MANIFEST.json", manifest)
    manifest_sha = sha256_file(ROOT / "AUDIO_WAVE_FIXTURE_MANIFEST.json")
    tests_sha = sha256_file(ROOT / "AUDIO_WAVE_REFERENCE_TESTS.json")

    committed_fixtures = load_committed_fixtures(fixtures)
    test_results = run_reference_tests(committed_fixtures)
    results_document = reference_results_document(test_results, manifest_sha, tests_sha)
    write_json(ROOT / "AUDIO_WAVE_REFERENCE_RESULTS.json", results_document)
    return results_document


def verify_package() -> dict[str, Any]:
    assert_frozen_runtime()
    manifest_path = ROOT / "AUDIO_WAVE_FIXTURE_MANIFEST.json"
    tests_path = ROOT / "AUDIO_WAVE_REFERENCE_TESTS.json"
    results_path = ROOT / "AUDIO_WAVE_REFERENCE_RESULTS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    definitions = generate_fixture_arrays()
    expected_paths = sorted(definition["file"] for definition in definitions.values())
    actual_paths = sorted(path.name for path in FIXTURE_ROOT.glob("*.wav"))
    if actual_paths != expected_paths:
        failures.append("fixture_file_set")

    expected_manifest = build_manifest(definitions)
    if manifest != expected_manifest:
        failures.append("manifest_recompute")
    expected_tests = reference_tests_document()
    if tests != expected_tests:
        failures.append("test_freeze_recompute")

    byte_total = 0
    for record in manifest["fixtures"]:
        path = ROOT / record["path"]
        if not path.is_file():
            failures.append(f"missing:{record['path']}")
            continue
        byte_total += path.stat().st_size
        if sha256_file(path) != record["sha256"]:
            failures.append(f"hash:{record['path']}")
        try:
            rate, samples, chunks = read_float32_wav(path)
        except ValueError as error:
            failures.append(f"parse:{record['path']}:{error}")
            continue
        if rate != record["sample_rate_hz"] or samples.shape[0] != record["sample_count"]:
            failures.append(f"identity:{record['path']}")
        if chunks != record["riff_chunks"] or chunks != ["fmt ", "data"]:
            failures.append(f"chunks:{record['path']}")
        if float(np.max(np.abs(samples))) > ABSOLUTE_SAMPLE_CEILING + 1e-7:
            failures.append(f"ceiling:{record['path']}")
        expected_peak = _q(np.max(np.abs(samples)))
        expected_rms = _q(np.sqrt(np.mean(samples * samples)))
        expected_channels = 1 if samples.ndim == 1 else int(samples.shape[1])
        if record["peak_amplitude"] != expected_peak or record["rms_amplitude"] != expected_rms:
            failures.append(f"statistics:{record['path']}")
        if record["channel_count"] != expected_channels or record["byte_count"] != path.stat().st_size:
            failures.append(f"shape_or_bytes:{record['path']}")

    if byte_total != manifest["total_bytes"]:
        failures.append("fixture_total_bytes")
    if len(manifest["fixtures"]) != manifest["fixture_count"]:
        failures.append("fixture_count")
    if sha256_file(manifest_path) != results["fixture_manifest_sha256"]:
        failures.append("manifest_result_binding")
    if sha256_file(tests_path) != results["reference_tests_sha256"]:
        failures.append("tests_result_binding")

    committed_fixtures = load_committed_fixtures(definitions)
    recomputed_tests = run_reference_tests(committed_fixtures)
    expected_results = reference_results_document(
        recomputed_tests, sha256_file(manifest_path), sha256_file(tests_path)
    )
    if results != expected_results:
        failures.append("result_recompute")
    if any(item["status"] != "PASS" for item in recomputed_tests):
        failures.append("recomputed_reference_test_failure")
    if expected_results["claim_ceiling"] != CLAIM_CEILING:
        failures.append("recomputed_claim_ceiling")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "fixture_count": manifest["fixture_count"],
        "fixture_total_bytes": manifest["total_bytes"],
        "fixture_manifest_sha256": sha256_file(manifest_path),
        "reference_tests_sha256": sha256_file(tests_path),
        "reference_results_sha256": sha256_file(results_path),
        "test_count": results["summary"]["test_count"],
        "tests_passed": sum(item["status"] == "PASS" for item in recomputed_tests),
        "claim_ceiling": expected_results["claim_ceiling"],
        "recomputed_results_match": results == expected_results,
        "audio_hardware_contact_count": 0,
        "target_contact_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "verify", "self-test"), nargs="?", default="self-test")
    arguments = parser.parse_args()

    if arguments.operation in ("build", "self-test"):
        build_result = build_package()
        if build_result["summary"]["failed"]:
            print(json.dumps(build_result["summary"], sort_keys=True))
            return 1
    verification = verify_package()
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
