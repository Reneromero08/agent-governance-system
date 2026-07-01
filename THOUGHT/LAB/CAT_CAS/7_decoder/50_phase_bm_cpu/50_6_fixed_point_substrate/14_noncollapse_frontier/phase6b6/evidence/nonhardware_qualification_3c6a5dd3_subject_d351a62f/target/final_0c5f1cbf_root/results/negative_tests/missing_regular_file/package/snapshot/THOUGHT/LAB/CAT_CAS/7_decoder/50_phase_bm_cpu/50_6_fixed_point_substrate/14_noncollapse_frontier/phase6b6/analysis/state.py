"""State construction with explicit measured/input/context separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from contracts.contract import DELAY_CANDIDATES, PROHIBITED_MEASURED_STATE_FIELDS


@dataclass(frozen=True)
class Gauge:
    complex_anchor_alpha: tuple[complex, ...]
    amplitude_floor: tuple[float, ...]
    preamble_drift_estimate: complex
    local_idle_covariance: tuple[tuple[float, float], tuple[float, float]]


def validate_measured_state_fields(fields: Iterable[str]) -> None:
    forbidden = sorted(set(fields).intersection(PROHIBITED_MEASURED_STATE_FIELDS))
    if forbidden:
        raise ValueError("measured state contains prohibited fields: " + ",".join(forbidden))


def complex_response(row: dict[str, Any]) -> complex:
    validate_measured_state_fields(row["r_t"].keys())
    return complex(row["r_t"]["lockin_I"], row["r_t"]["lockin_Q"])


def s0(row: dict[str, Any]) -> tuple[complex, float]:
    return (complex_response(row), float(row["r_t"]["ring_osc_period"]))


def estimate_preamble_gauge(rows: list[dict[str, Any]]) -> Gauge:
    if any(row.get("stage") != "preamble" for row in rows):
        raise ValueError("g_s must be estimated from preamble rows only")
    anchors = [[] for _ in range(12)]
    idle_values: list[complex] = []
    for row in rows:
        z = complex_response(row)
        if row["u_t"]["drive_on"] and row["u_t"]["executed_mode"] == "ANCHOR":
            tone = row["u_t"]["physical_tone_index"]
            if tone is None:
                raise ValueError("driven anchor lacks physical tone")
            anchors[int(tone)].append(z)
        if row["u_t"]["executed_mode"] == "SENDER_OFF_IDLE":
            idle_values.append(z)
    if any(not values for values in anchors):
        raise ValueError("missing per-tone driven preamble anchor")
    alpha = tuple(sum(values) / len(values) for values in anchors)
    floors = tuple(min(abs(value) for value in values) for values in anchors)
    drift = (idle_values[-1] - idle_values[0]) if len(idle_values) >= 2 else 0j
    if len(idle_values) >= 2:
        arr = np.array([[value.real, value.imag] for value in idle_values], dtype=float)
        cov = np.cov(arr.T, bias=True) + np.eye(2) * 1e-9
    else:
        cov = np.eye(2)
    return Gauge(alpha, floors, drift, ((float(cov[0, 0]), float(cov[0, 1])), (float(cov[1, 0]), float(cov[1, 1]))))


def estimate_session_gauges(rows: list[dict[str, Any]]) -> dict[int, Gauge]:
    gauges: dict[int, Gauge] = {}
    for session_index in sorted({int(row["session_index"]) for row in rows}):
        preamble = [row for row in rows if row["session_index"] == session_index and row["stage"] == "preamble"]
        gauges[session_index] = estimate_preamble_gauge(preamble)
    return gauges


def training_global_covariance(rows: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    bad = [row for row in rows if row.get("split") != "train" or row.get("stage") != "preamble"]
    if bad:
        raise ValueError("global whitening covariance may use training preambles only")
    values = np.array([[complex_response(row).real, complex_response(row).imag] for row in rows], dtype=float)
    cov = np.cov(values.T, bias=True) + np.eye(2) * 1e-6
    return ((float(cov[0, 0]), float(cov[0, 1])), (float(cov[1, 0]), float(cov[1, 1])))


def symmetric_inverse_sqrt(
    sigma_train: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    cov = np.array(sigma_train, dtype=float)
    values, vectors = np.linalg.eigh(cov)
    clipped = np.maximum(values, 1e-9)
    return vectors @ np.diag(1.0 / np.sqrt(clipped)) @ vectors.T


def assert_training_only_global_covariance(rows: list[dict[str, Any]]) -> None:
    training_global_covariance(rows)


def gauge_normalize(
    row: dict[str, Any],
    gauge: Gauge,
    sigma_train: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[complex, float]:
    tone = row["u_t"].get("physical_tone_index")
    if tone is None:
        tone = row["declared"].get("analysis_tone_index")
    if tone is None:
        tone = 0
    z, period = s0(row)
    centered = z - gauge.complex_anchor_alpha[int(tone)]
    whitened = symmetric_inverse_sqrt(sigma_train) @ np.array([centered.real, centered.imag], dtype=float)
    return (complex(float(whitened[0]), float(whitened[1])), period)


def executed_control_vector(row: dict[str, Any]) -> np.ndarray:
    u = row["u_t"]
    phase_map = {None: 0.0, "none": 0.0, "0": 0.0, "pi": np.pi, "pi/2": np.pi / 2.0, "-pi/2": -np.pi / 2.0}
    return np.array(
        [
            1.0 if u["drive_on"] else 0.0,
            -1.0 if u.get("amplitude_level") is None else float(u["amplitude_level"]),
            -1.0 if u.get("physical_tone_index") is None else float(u["physical_tone_index"]),
            float(phase_map.get(u.get("phase_action"), 0.0)),
            0.0 if u.get("codeword_sign") is None else float(u["codeword_sign"]),
        ],
        dtype=float,
    )


def state_vector(
    rows: list[dict[str, Any]],
    index: int,
    state_level: str,
    gauge: Gauge,
    sigma_train: tuple[tuple[float, float], tuple[float, float]],
    delay: int | None = None,
) -> np.ndarray:
    if state_level == "S0":
        z, period = s0(rows[index])
        return np.array([z.real, z.imag, period], dtype=float)
    if state_level == "S1":
        z, period = gauge_normalize(rows[index], gauge, sigma_train)
        return np.array([z.real, z.imag, period], dtype=float)
    if state_level == "S2":
        if delay not in DELAY_CANDIDATES:
            raise ValueError("delay outside frozen candidate set")
        if index - delay + 1 < 0:
            raise ValueError("insufficient history for delay state")
        parts: list[np.ndarray] = []
        for i in range(index, index - delay, -1):
            z, period = gauge_normalize(rows[i], gauge, sigma_train)
            parts.append(np.array([z.real, z.imag, period], dtype=float))
        for i in range(index - 1, index - delay, -1):
            parts.append(executed_control_vector(rows[i]))
        return np.concatenate(parts)
    raise ValueError(f"unknown state level: {state_level}")


def s2_delayed(rows: list[dict[str, Any]], index: int, delay: int, gauge: Gauge) -> dict[str, Any]:
    if delay not in DELAY_CANDIDATES:
        raise ValueError("delay outside frozen candidate set")
    if index - delay + 1 < 0:
        raise ValueError("insufficient history for delay state")
    states = [gauge_normalize(rows[i], gauge, ((1.0, 0.0), (0.0, 1.0))) for i in range(index, index - delay, -1)]
    prior_controls = [rows[i]["u_t"] for i in range(index - 1, index - delay, -1)]
    return {"S1_history": states, "prior_executed_controls": prior_controls}
