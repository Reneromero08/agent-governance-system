"""Executable frozen operator ladder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from contracts.contract import O4_FIXED_LIFTS, REGULARIZATION_LADDER, digest


OPERATOR_LADDER = (
    "O0_TRAINING_MEAN",
    "O0_LAST_VALUE",
    "O0_RETURN_TO_BASELINE",
    "O0_INPUT_ONLY",
    "O0_TIME_INDEX",
    "O0_SESSION_LOOKUP_DIAGNOSTIC",
    "O1_SHARED_COMPLEX_AFFINE",
    "O2_ROUTE_CONDITIONED_COMPLEX_AFFINE",
    "O3_COMPLEX_BILINEAR_STATE_CONTROL",
    "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR",
)
FORBIDDEN_MODEL_TERMS = ("neural", "backprop", "auc_first", "scalar_candidate_verifier", "learned_representation")


@dataclass
class FittedOperator:
    operator_class: str
    regularization: float
    coefficients: Any
    baseline: np.ndarray
    route_vocabulary: tuple[str, ...] = ("v2s3", "v4s5")

    def predict(self, x: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
        if self.operator_class == "O0_TRAINING_MEAN":
            return np.repeat(self.baseline.reshape(1, -1), len(x), axis=0)
        if self.operator_class == "O0_LAST_VALUE":
            return x[:, :3]
        if self.operator_class == "O0_RETURN_TO_BASELINE":
            return np.repeat(self.baseline.reshape(1, -1), len(x), axis=0)
        if self.operator_class == "O0_INPUT_ONLY":
            return _design_input_only(x, rows) @ self.coefficients
        if self.operator_class == "O0_TIME_INDEX":
            return _design_time(rows) @ self.coefficients
        if self.operator_class == "O0_SESSION_LOOKUP_DIAGNOSTIC":
            return np.vstack([self.coefficients.get(row["session_index"], self.baseline) for row in rows])
        if self.operator_class == "O1_SHARED_COMPLEX_AFFINE":
            return _design_affine(x) @ self.coefficients
        if self.operator_class == "O2_ROUTE_CONDITIONED_COMPLEX_AFFINE":
            return _design_route_affine(x, rows, self.route_vocabulary) @ self.coefficients
        if self.operator_class == "O3_COMPLEX_BILINEAR_STATE_CONTROL":
            return _design_bilinear(x, rows) @ self.coefficients
        if self.operator_class == "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR":
            return _design_o4(x, rows) @ self.coefficients
        raise ValueError(f"unknown operator {self.operator_class}")


def validate_operator_manifest(manifest: dict[str, object]) -> None:
    text = repr(manifest).lower()
    blocked = [term for term in FORBIDDEN_MODEL_TERMS if term in text]
    if blocked:
        raise ValueError("forbidden operator term present: " + ",".join(blocked))
    if tuple(manifest.get("operator_ladder", ())) != OPERATOR_LADDER:
        raise ValueError("operator ladder is not frozen")
    if tuple(manifest.get("regularization_ladder", ())) != REGULARIZATION_LADDER:
        raise ValueError("regularization ladder is not frozen")
    if tuple(manifest.get("o4_fixed_lifts", ())) != O4_FIXED_LIFTS:
        raise ValueError("O4 lift family is not frozen")


def deterministic_seed(analysis_contract_digest: str, label: str) -> int:
    import hashlib

    payload = f"{analysis_contract_digest}:{label}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def choose_simplest_within_two_percent(candidates: Iterable[tuple[str, float]]) -> str:
    ordered = list(candidates)
    if not ordered:
        raise ValueError("empty candidate set")
    best_score = min(score for _, score in ordered)
    for name, score in ordered:
        if score <= best_score * 1.02:
            return name
    raise AssertionError("unreachable")


def fit_operator(operator_class: str, x: np.ndarray, y: np.ndarray, rows: list[dict[str, Any]], regularization: float = 0.0) -> FittedOperator:
    baseline = np.mean(y, axis=0)
    route_vocabulary = ("v2s3", "v4s5")
    if operator_class in ("O0_TRAINING_MEAN", "O0_LAST_VALUE", "O0_RETURN_TO_BASELINE"):
        return FittedOperator(operator_class, regularization, None, baseline)
    if operator_class == "O0_SESSION_LOOKUP_DIAGNOSTIC":
        means = {}
        for session in sorted({row["session_index"] for row in rows}):
            idx = [i for i, row in enumerate(rows) if row["session_index"] == session]
            means[session] = np.mean(y[idx], axis=0)
        return FittedOperator(operator_class, regularization, means, baseline)
    design = {
        "O0_INPUT_ONLY": _design_input_only,
        "O0_TIME_INDEX": lambda _x, r: _design_time(r),
        "O1_SHARED_COMPLEX_AFFINE": lambda _x, _r: _design_affine(_x),
        "O2_ROUTE_CONDITIONED_COMPLEX_AFFINE": lambda _x, r: _design_route_affine(_x, r, route_vocabulary),
        "O3_COMPLEX_BILINEAR_STATE_CONTROL": _design_bilinear,
        "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR": _design_o4,
    }[operator_class](x, rows)
    coefs = _ridge(design, y, regularization)
    return FittedOperator(operator_class, regularization, coefs, baseline, route_vocabulary)


def _ridge(design: np.ndarray, y: np.ndarray, regularization: float) -> np.ndarray:
    gram = design.T @ design
    penalty = regularization * np.eye(gram.shape[0])
    return np.linalg.pinv(gram + penalty) @ design.T @ y


def _design_affine(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def _design_input_only(x: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    phase_map = {None: 0.0, "none": 0.0, "0": 0.0, "pi": np.pi, "pi/2": np.pi / 2.0, "-pi/2": -np.pi / 2.0}
    controls = np.array([
        [
            1.0,
            1.0 if row["u_t"]["drive_on"] else 0.0,
            float(phase_map.get(row["u_t"].get("phase_action"), 0.0)),
            -1.0 if row["u_t"].get("physical_tone_index") is None else float(row["u_t"]["physical_tone_index"]),
            0.0 if row["u_t"].get("codeword_sign") is None else float(row["u_t"]["codeword_sign"]),
        ]
        for row in rows
    ])
    return controls


def _design_time(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[1.0, float(row["slot_index"])] for row in rows], dtype=float)


def _design_session(rows: list[dict[str, Any]]) -> np.ndarray:
    sessions = sorted({row["session_index"] for row in rows})
    index = {session: i for i, session in enumerate(sessions)}
    design = np.zeros((len(rows), len(sessions)))
    for row_i, row in enumerate(rows):
        design[row_i, index[row["session_index"]]] = 1.0
    return design


def _design_route_affine(x: np.ndarray, rows: list[dict[str, Any]], route_vocabulary: tuple[str, ...] = ("v2s3", "v4s5")) -> np.ndarray:
    parts = [np.ones((len(x), 1)), x]
    for route in route_vocabulary:
        mask = np.array([1.0 if row["route"] == route else 0.0 for row in rows]).reshape(-1, 1)
        parts.append(x * mask)
    return np.column_stack(parts)


def _design_bilinear(x: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    u = _design_input_only(x, rows)[:, 1:]
    terms = [np.ones((len(x), 1)), x, u]
    for col in range(u.shape[1]):
        terms.append(x * u[:, col : col + 1])
    return np.column_stack(terms)


def _design_o4(x: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    z = x[:, 0] + 1j * x[:, 1]
    u = _design_input_only(x, rows)[:, 1:]
    phase = np.exp(1j * u[:, 1])
    features = [
        np.ones(len(x)),
        z.real,
        z.imag,
        np.conjugate(z).real,
        np.conjugate(z).imag,
        np.abs(z) ** 2,
        (z * u[:, 0]).real,
        (z * u[:, 0]).imag,
        phase.real,
        phase.imag,
        x[:, 0] * x[:, 1],
    ]
    if x.shape[1] > 3:
        features.extend([x[:, i] for i in range(3, x.shape[1])])
    return np.column_stack(features)


def analysis_contract_digest() -> str:
    return digest({"operator_ladder": OPERATOR_LADDER, "regularization_ladder": REGULARIZATION_LADDER, "o4_fixed_lifts": O4_FIXED_LIFTS})
