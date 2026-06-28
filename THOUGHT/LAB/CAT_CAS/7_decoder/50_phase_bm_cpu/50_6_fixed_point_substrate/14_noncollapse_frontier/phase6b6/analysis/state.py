"""State construction with explicit measured/input/context separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from contracts.contract import PROHIBITED_MEASURED_STATE_FIELDS


@dataclass(frozen=True)
class Gauge:
    complex_anchor_alpha: complex
    amplitude_floor: float
    preamble_drift_estimate: complex
    local_idle_covariance: tuple[tuple[float, float], tuple[float, float]]


def validate_measured_state_fields(fields: Iterable[str]) -> None:
    forbidden = sorted(set(fields).intersection(PROHIBITED_MEASURED_STATE_FIELDS))
    if forbidden:
        raise ValueError("measured state contains prohibited fields: " + ",".join(forbidden))


def s0(row: dict[str, Any]) -> tuple[complex, float]:
    validate_measured_state_fields(row["r_t"].keys())
    return (complex(row["r_t"]["lockin_I"], row["r_t"]["lockin_Q"]), float(row["r_t"]["ring_osc_period"]))


def estimate_preamble_gauge(rows: list[dict[str, Any]]) -> Gauge:
    if any(row.get("stage") != "preamble" for row in rows):
        raise ValueError("g_s must be estimated from preamble rows only")
    complex_values = [complex(row["r_t"]["lockin_I"], row["r_t"]["lockin_Q"]) for row in rows]
    if not complex_values:
        raise ValueError("empty preamble gauge")
    anchor = sum(complex_values) / len(complex_values)
    floor = min(abs(value) for value in complex_values)
    drift = complex_values[-1] - complex_values[0]
    return Gauge(anchor, floor, drift, ((1.0, 0.0), (0.0, 1.0)))


def gauge_normalize(row: dict[str, Any], gauge: Gauge, sigma_train: tuple[tuple[float, float], tuple[float, float]]) -> tuple[complex, float]:
    if sigma_train != ((1.0, 0.0), (0.0, 1.0)):
        raise ValueError("only frozen real-block identity whitening is implemented for software entry")
    z, period = s0(row)
    return (z - gauge.complex_anchor_alpha, period)


def s2_delayed(rows: list[dict[str, Any]], index: int, delay: int, gauge: Gauge) -> dict[str, Any]:
    if delay not in (2, 4, 8, 16):
        raise ValueError("delay outside frozen candidate set")
    if index - delay + 1 < 0:
        raise ValueError("insufficient history for delay state")
    states = [gauge_normalize(rows[i], gauge, ((1.0, 0.0), (0.0, 1.0))) for i in range(index, index - delay, -1)]
    prior_controls = [rows[i]["u_t"] for i in range(index - 1, index - delay, -1)]
    return {"S1_history": states, "prior_executed_controls": prior_controls}


def assert_training_only_global_covariance(rows: list[dict[str, Any]]) -> None:
    bad = [row for row in rows if row.get("split") != "train" or row.get("stage") != "preamble"]
    if bad:
        raise ValueError("global whitening covariance may use training preambles only")
