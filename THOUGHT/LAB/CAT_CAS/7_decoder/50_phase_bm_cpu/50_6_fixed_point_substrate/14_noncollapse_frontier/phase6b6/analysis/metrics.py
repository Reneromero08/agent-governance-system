"""Phase-native metrics and deterministic bootstrap."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.std(y_true)) or 1.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)


def complex_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    zt = y_true[:, 0] + 1j * y_true[:, 1]
    zp = y_pred[:, 0] + 1j * y_pred[:, 1]
    denom = np.linalg.norm(zt) * np.linalg.norm(zp)
    if denom == 0:
        return 0.0
    return float(abs(np.vdot(zt, zp)) / denom)


def amplitude_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.hypot(y_true[:, 0], y_true[:, 1]) - np.hypot(y_pred[:, 0], y_pred[:, 1]))))


def wrapped_phase_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    phase_true = np.arctan2(y_true[:, 1], y_true[:, 0])
    phase_pred = np.arctan2(y_pred[:, 1], y_pred[:, 0])
    diff = np.angle(np.exp(1j * (phase_true - phase_pred)))
    return float(np.mean(np.abs(diff)))


def summarize(rows: list[dict[str, Any]], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "complex_nrmse": nrmse(y_true, y_pred),
        "complex_correlation": complex_corr(y_true, y_pred),
        "amplitude_error": amplitude_error(y_true, y_pred),
        "wrapped_phase_error": wrapped_phase_error(y_true, y_pred),
        "per_route": {},
        "per_session": {},
        "sender_on": {},
        "sender_off": {},
    }
    for key, selector in (
        ("sender_on", lambda row: row["u_t"]["drive_on"]),
        ("sender_off", lambda row: not row["u_t"]["drive_on"]),
    ):
        idx = [i for i, row in enumerate(rows) if selector(row)]
        if idx:
            payload[key] = {"complex_nrmse": nrmse(y_true[idx], y_pred[idx]), "complex_correlation": complex_corr(y_true[idx], y_pred[idx])}
    for route in sorted({row["route"] for row in rows}):
        idx = [i for i, row in enumerate(rows) if row["route"] == route]
        payload["per_route"][route] = {"complex_nrmse": nrmse(y_true[idx], y_pred[idx]), "complex_correlation": complex_corr(y_true[idx], y_pred[idx])}
    for session in sorted({row["session_index"] for row in rows}):
        idx = [i for i, row in enumerate(rows) if row["session_index"] == session]
        payload["per_session"][str(session)] = {"complex_nrmse": nrmse(y_true[idx], y_pred[idx])}
    return payload


def bootstrap_gain_lower(gains: list[float], seed: int, iterations: int = 200) -> float:
    if not gains:
        return 0.0
    rng = np.random.default_rng(seed)
    means = []
    arr = np.array(gains, dtype=float)
    for _ in range(iterations):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.quantile(means, 0.025))


def packet_groups(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[f"{row['session_index']}:{row.get('packet_id')}"].append(i)
    return groups
