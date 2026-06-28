"""Frozen operator ladder for software-entry analysis tests."""

from __future__ import annotations

from typing import Iterable

from contracts.contract import O4_FIXED_LIFTS, REGULARIZATION_LADDER


OPERATOR_LADDER = (
    "O0_BASELINES",
    "O1_SHARED_COMPLEX_AFFINE",
    "O2_ROUTE_CONDITIONED_COMPLEX_AFFINE",
    "O3_COMPLEX_BILINEAR_STATE_CONTROL",
    "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR",
)
FORBIDDEN_MODEL_TERMS = ("neural", "backprop", "auc_first", "scalar_candidate_verifier", "learned_representation")


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
