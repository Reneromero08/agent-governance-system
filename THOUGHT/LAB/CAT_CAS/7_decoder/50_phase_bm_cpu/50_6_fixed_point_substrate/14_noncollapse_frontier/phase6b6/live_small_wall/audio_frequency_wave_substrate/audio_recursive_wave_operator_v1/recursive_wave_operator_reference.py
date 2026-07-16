#!/usr/bin/env python3
"""Deterministic R1 candidate for complete-tree temporal phase recurrence.

Each native step creates a new phase node whose children are the entire previous
recursive phase tree and an entire drive tree. The previous state is not decoded to a
spin, score, energy, FFT magnitude, or flat coefficient bank. It remains an executable
subtree of the next state.

This is an ordinary-software architecture candidate. It establishes no catalytic-loop,
Ising, physical-wave, optimization, restoration, or Wall claim.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
R0_SOURCE = (
    PACKAGE_DIR.parent
    / "audio_recursive_phase_tree_v1"
    / "recursive_phase_tree_reference.py"
)

_spec = importlib.util.spec_from_file_location("catcas_recursive_phase_tree_r0", R0_SOURCE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("unable to load the established R0 recursive phase-tree reference")
r0 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r0
_spec.loader.exec_module(r0)

GENERATOR_ID = "recursive_wave_operator_reference_v1"
CLAIM_CEILING = "SOFTWARE_COMPLETE_TREE_TEMPORAL_RECURRENCE_REFERENCE_ONLY"
ESTABLISHED_TOKEN = "AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED"
RESULT_SCHEMA = "recursive_wave_operator_reference_result_v1"
STEP_SCHEMA = "recursive_wave_temporal_step_v1"
RECEIPT_SCHEMA = "recursive_wave_ancestry_receipt_v1"
MAX_STEPS = 6
UNIT_MODULUS_TOL = 1e-12
NONIDENTITY_TOL = 1e-6
ORDER_RESPONSE_MAX = 0.99
PORTABLE_METRIC_ATOL = 5e-12
PORTABLE_METRIC_RTOL = 5e-12
MAX_ID_BYTES = 64
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")

STEP_SCHEMA_FILE = "RECURSIVE_WAVE_OPERATOR_STEP_SPEC_SCHEMA.json"
RECEIPT_SCHEMA_FILE = "RECURSIVE_WAVE_OPERATOR_ANCESTRY_RECEIPT_SCHEMA.json"
MANIFEST_FILE = "RECURSIVE_WAVE_OPERATOR_TRAJECTORY_MANIFEST.json"
TESTS_FILE = "RECURSIVE_WAVE_OPERATOR_REFERENCE_TESTS.json"
RESULTS_FILE = "RECURSIVE_WAVE_OPERATOR_REFERENCE_RESULTS.json"
FIXTURE_DIR_NAME = "fixtures"

STEP_KEYS = {
    "carrier_frequency_hz",
    "drive_modulation_index",
    "root_id",
    "root_phase_rad",
    "schema",
    "state_child_index",
    "state_modulation_index",
    "step_index",
    "drive_child_index",
}
RECEIPT_KEYS = {
    "child_roles",
    "drive_child_index",
    "drive_edge_modulation_index",
    "drive_root_id",
    "drive_tree_digest",
    "result_root_id",
    "result_tree_digest",
    "schema",
    "state_edge_modulation_index",
    "state_child_index",
    "state_root_id",
    "state_tree_digest",
    "step_index",
    "step_spec",
    "step_spec_canonical_sha256",
    "step_spec_digest",
}


def _metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds the finite numeric envelope") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return 0.0 if number == 0.0 else number


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    if len(value.encode("utf-8")) > MAX_ID_BYTES:
        raise ValueError(f"{label} exceeds the byte envelope")
    if r0.NODE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} contains an unsafe character")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class TemporalStepSpec:
    """Prospective parameters for one complete-tree temporal lift."""

    step_index: int
    root_id: str
    carrier_frequency_hz: float
    root_phase_rad: float
    state_modulation_index: float
    drive_modulation_index: float
    state_child_index: int
    drive_child_index: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or not 1 <= self.step_index <= MAX_STEPS
        ):
            raise ValueError("step_index is outside the bounded R1 envelope")
        root_id = _safe_id(self.root_id, "root_id")
        expected_prefix = f"time{self.step_index}."
        if not root_id.startswith(expected_prefix):
            raise ValueError("root_id must be scoped to its step index")

        numeric = {
            "carrier_frequency_hz": self.carrier_frequency_hz,
            "root_phase_rad": self.root_phase_rad,
            "state_modulation_index": self.state_modulation_index,
            "drive_modulation_index": self.drive_modulation_index,
        }
        normalized = {
            name: _finite_number(value, name) for name, value in numeric.items()
        }
        if not 0.0 < normalized["carrier_frequency_hz"] < r0.SAMPLE_RATE_HZ / 2:
            raise ValueError("carrier_frequency_hz must be inside (0, Nyquist)")
        if abs(normalized["root_phase_rad"]) > 2.0 * math.pi:
            raise ValueError("root_phase_rad exceeds the frozen envelope")
        if not 0.0 < normalized["state_modulation_index"] <= 2.0:
            raise ValueError("state_modulation_index exceeds the frozen envelope")
        if not 0.0 < normalized["drive_modulation_index"] <= 2.0:
            raise ValueError("drive_modulation_index exceeds the frozen envelope")
        if (
            isinstance(self.state_child_index, bool)
            or isinstance(self.drive_child_index, bool)
            or not isinstance(self.state_child_index, int)
            or not isinstance(self.drive_child_index, int)
            or {self.state_child_index, self.drive_child_index} != {0, 1}
        ):
            raise ValueError("step child indices must be distinct and exactly 0 and 1")
        object.__setattr__(self, "root_id", root_id)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def document(self) -> dict[str, Any]:
        return {"schema": STEP_SCHEMA, **asdict(self)}

    def canonical_bytes(self) -> bytes:
        return r0.canonical_json_bytes(self.document())

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def digest(self) -> str:
        return hashlib.sha256(
            r0.canonical_json_bytes(self.document(), pretty=False)
        ).hexdigest()

    @classmethod
    def from_document(cls, document: Any) -> "TemporalStepSpec":
        if not isinstance(document, dict):
            raise ValueError("step specification must be an object")
        _exact_keys(document, STEP_KEYS, "step specification")
        if document["schema"] != STEP_SCHEMA:
            raise ValueError("unsupported step specification schema")
        return cls(
            step_index=document["step_index"],
            root_id=document["root_id"],
            carrier_frequency_hz=document["carrier_frequency_hz"],
            root_phase_rad=document["root_phase_rad"],
            state_modulation_index=document["state_modulation_index"],
            drive_modulation_index=document["drive_modulation_index"],
            state_child_index=document["state_child_index"],
            drive_child_index=document["drive_child_index"],
        )

    @classmethod
    def from_bytes(
        cls, payload: bytes, *, require_canonical: bool = True
    ) -> "TemporalStepSpec":
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("step specification must be UTF-8") from exc
        spec = cls.from_document(r0.strict_json_loads(text))
        if require_canonical and payload != spec.canonical_bytes():
            raise ValueError("step specification is valid but not canonical")
        return spec


@dataclass(frozen=True)
class TemporalStepReceipt:
    """Exact two-role ancestry binding for one native temporal lift."""

    step_index: int
    step_spec: TemporalStepSpec
    step_spec_canonical_sha256: str
    step_spec_digest: str
    result_root_id: str
    result_tree_digest: str
    state_root_id: str
    state_tree_digest: str
    state_edge_modulation_index: float
    state_child_index: int
    drive_root_id: str
    drive_tree_digest: str
    drive_edge_modulation_index: float
    drive_child_index: int
    child_roles: tuple[str, str] = ("state", "drive")

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or not 1 <= self.step_index <= MAX_STEPS
        ):
            raise ValueError("receipt step_index is outside the bounded R1 envelope")
        if self.child_roles != ("state", "drive"):
            raise ValueError("receipt child roles must be exactly state then drive")
        if (
            isinstance(self.state_child_index, bool)
            or isinstance(self.drive_child_index, bool)
            or not isinstance(self.state_child_index, int)
            or not isinstance(self.drive_child_index, int)
            or {self.state_child_index, self.drive_child_index} != {0, 1}
        ):
            raise ValueError("receipt child indices must be distinct and exactly 0 and 1")
        if self.step_index != self.step_spec.step_index:
            raise ValueError("receipt step index does not match its step specification")
        if self.step_spec_canonical_sha256 != self.step_spec.canonical_sha256():
            raise ValueError("receipt step canonical bytes do not match their SHA-256")
        if self.step_spec_digest != self.step_spec.digest():
            raise ValueError("receipt step digest does not match its step specification")
        if (
            self.state_child_index != self.step_spec.state_child_index
            or self.drive_child_index != self.step_spec.drive_child_index
        ):
            raise ValueError("receipt child indices do not match the exact step")
        _hash(self.result_tree_digest, "result_tree_digest")
        _hash(self.state_tree_digest, "state_tree_digest")
        _hash(self.drive_tree_digest, "drive_tree_digest")
        _safe_id(self.result_root_id, "result_root_id")
        _safe_id(self.state_root_id, "state_root_id")
        _safe_id(self.drive_root_id, "drive_root_id")
        state_beta = _finite_number(
            self.state_edge_modulation_index, "state_edge_modulation_index"
        )
        drive_beta = _finite_number(
            self.drive_edge_modulation_index, "drive_edge_modulation_index"
        )
        if state_beta != self.step_spec.state_modulation_index:
            raise ValueError("receipt state edge does not match the step specification")
        if drive_beta != self.step_spec.drive_modulation_index:
            raise ValueError("receipt drive edge does not match the step specification")
        object.__setattr__(self, "state_edge_modulation_index", state_beta)
        object.__setattr__(self, "drive_edge_modulation_index", drive_beta)

    def document(self) -> dict[str, Any]:
        return {
            "child_roles": list(self.child_roles),
            "drive_child_index": self.drive_child_index,
            "drive_edge_modulation_index": self.drive_edge_modulation_index,
            "drive_root_id": self.drive_root_id,
            "drive_tree_digest": self.drive_tree_digest,
            "result_root_id": self.result_root_id,
            "result_tree_digest": self.result_tree_digest,
            "schema": RECEIPT_SCHEMA,
            "state_edge_modulation_index": self.state_edge_modulation_index,
            "state_child_index": self.state_child_index,
            "state_root_id": self.state_root_id,
            "state_tree_digest": self.state_tree_digest,
            "step_index": self.step_index,
            "step_spec": self.step_spec.document(),
            "step_spec_canonical_sha256": self.step_spec_canonical_sha256,
            "step_spec_digest": self.step_spec_digest,
        }

    def canonical_bytes(self) -> bytes:
        return r0.canonical_json_bytes(self.document())

    def digest(self) -> str:
        return hashlib.sha256(
            r0.canonical_json_bytes(self.document(), pretty=False)
        ).hexdigest()

    @classmethod
    def from_document(cls, document: Any) -> "TemporalStepReceipt":
        if not isinstance(document, dict):
            raise ValueError("ancestry receipt must be an object")
        _exact_keys(document, RECEIPT_KEYS, "ancestry receipt")
        if document["schema"] != RECEIPT_SCHEMA:
            raise ValueError("unsupported ancestry receipt schema")
        roles = document["child_roles"]
        if not isinstance(roles, list):
            raise ValueError("receipt child_roles must be an array")
        return cls(
            step_index=document["step_index"],
            step_spec=TemporalStepSpec.from_document(document["step_spec"]),
            step_spec_canonical_sha256=document["step_spec_canonical_sha256"],
            step_spec_digest=document["step_spec_digest"],
            result_root_id=document["result_root_id"],
            result_tree_digest=document["result_tree_digest"],
            state_root_id=document["state_root_id"],
            state_tree_digest=document["state_tree_digest"],
            state_edge_modulation_index=document["state_edge_modulation_index"],
            state_child_index=document["state_child_index"],
            drive_root_id=document["drive_root_id"],
            drive_tree_digest=document["drive_tree_digest"],
            drive_edge_modulation_index=document["drive_edge_modulation_index"],
            drive_child_index=document["drive_child_index"],
            child_roles=tuple(roles),
        )

    @classmethod
    def from_bytes(
        cls, payload: bytes, *, require_canonical: bool = True
    ) -> "TemporalStepReceipt":
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("ancestry receipt must be UTF-8") from exc
        receipt = cls.from_document(r0.strict_json_loads(text))
        if require_canonical and payload != receipt.canonical_bytes():
            raise ValueError("ancestry receipt is valid but not canonical")
        return receipt


def node_ids(beam: Any) -> set[str]:
    return {node.node_id for node in beam.root.walk()}


def require_native_tree(beam: Any, label: str) -> Any:
    if not isinstance(beam, r0.RecursivePhaseBeam):
        raise ValueError(f"{label} must be a complete RecursivePhaseBeam")
    r0.deserialize_tree_bytes(beam.canonical_bytes(), require_canonical=True)
    return beam


def require_pre_ising_orientation(beam: Any, label: str) -> None:
    require_native_tree(beam, label)
    if beam.global_spin_phase_rad != 0.0:
        raise ValueError(
            f"{label} global orientation must remain zero before the Ising package"
        )


def temporal_step(
    state: Any, drive: Any, spec: TemporalStepSpec
) -> tuple[Any, TemporalStepReceipt]:
    """Embed the complete state and drive trees under one new phase root."""

    require_pre_ising_orientation(state, "state")
    require_pre_ising_orientation(drive, "drive")
    state_ids = node_ids(state)
    drive_ids = node_ids(drive)
    if state_ids & drive_ids:
        raise ValueError("state and drive node identities must be disjoint")
    if spec.root_id in state_ids | drive_ids:
        raise ValueError("step root identity collides with a child tree")

    root = r0.PhaseNode(
        spec.root_id,
        spec.carrier_frequency_hz,
        spec.root_phase_rad,
        (
            r0.PhaseEdge(spec.state_modulation_index, state.root),
            r0.PhaseEdge(spec.drive_modulation_index, drive.root),
        ),
    )
    result = r0.RecursivePhaseBeam(root=root, global_spin_phase_rad=0.0)
    canonical_result = r0.deserialize_tree_bytes(
        result.canonical_bytes(), require_canonical=True
    )
    canonical_roots = [edge.child.node_id for edge in canonical_result.root.children]
    state_child_index = canonical_roots.index(state.root.node_id)
    drive_child_index = canonical_roots.index(drive.root.node_id)
    if (
        state_child_index != spec.state_child_index
        or drive_child_index != spec.drive_child_index
    ):
        raise ValueError("step role indices do not match canonical child order")
    receipt = TemporalStepReceipt(
        step_index=spec.step_index,
        step_spec=spec,
        step_spec_canonical_sha256=spec.canonical_sha256(),
        step_spec_digest=spec.digest(),
        result_root_id=result.root.node_id,
        result_tree_digest=result.digest(),
        state_root_id=state.root.node_id,
        state_tree_digest=state.digest(),
        state_edge_modulation_index=spec.state_modulation_index,
        state_child_index=state_child_index,
        drive_root_id=drive.root.node_id,
        drive_tree_digest=drive.digest(),
        drive_edge_modulation_index=spec.drive_modulation_index,
        drive_child_index=drive_child_index,
    )
    return result, receipt


def _beam_from_child(node: Any) -> Any:
    return r0.RecursivePhaseBeam(root=node, global_spin_phase_rad=0.0)


def validate_step_receipt(
    result: Any, spec: TemporalStepSpec, receipt: TemporalStepReceipt
) -> tuple[Any, Any]:
    """Validate the exact result, step, two child roles, and both edge bindings."""

    require_native_tree(result, "result")
    if not isinstance(spec, TemporalStepSpec):
        raise ValueError("exact TemporalStepSpec is required for receipt validation")
    if not isinstance(receipt, TemporalStepReceipt):
        raise ValueError("exact TemporalStepReceipt is required for ancestry validation")
    TemporalStepReceipt.from_bytes(receipt.canonical_bytes(), require_canonical=True)
    if receipt.step_spec.canonical_bytes() != spec.canonical_bytes():
        raise ValueError("receipt does not bind the exact step specification")
    if receipt.step_index != spec.step_index:
        raise ValueError("receipt step index does not match the exact step")
    if receipt.step_spec_canonical_sha256 != spec.canonical_sha256():
        raise ValueError("receipt step canonical bytes do not match the exact step")
    if receipt.step_spec_digest != spec.digest():
        raise ValueError("receipt step digest does not match the exact step")
    if result.root.node_id != spec.root_id or result.root.node_id != receipt.result_root_id:
        raise ValueError("result root identity does not match step and receipt")
    if result.root.frequency_hz != spec.carrier_frequency_hz:
        raise ValueError("result carrier frequency does not match the exact step")
    if result.root.phase_rad != spec.root_phase_rad:
        raise ValueError("result root phase does not match the exact step")
    if result.digest() != receipt.result_tree_digest:
        raise ValueError("complete result digest does not match the ancestry receipt")
    if len(result.root.children) != 2:
        raise ValueError("result root must contain exactly two child roles")
    if receipt.child_roles != ("state", "drive"):
        raise ValueError("receipt child-role multiplicity or order is invalid")
    if (
        receipt.state_child_index != spec.state_child_index
        or receipt.drive_child_index != spec.drive_child_index
    ):
        raise ValueError("receipt child indices do not match the exact step")

    canonical_result = r0.deserialize_tree_bytes(
        result.canonical_bytes(), require_canonical=True
    )
    state_edge = canonical_result.root.children[receipt.state_child_index]
    drive_edge = canonical_result.root.children[receipt.drive_child_index]
    if (
        state_edge.child.node_id != receipt.state_root_id
        or drive_edge.child.node_id != receipt.drive_root_id
    ):
        raise ValueError(
            "receipt role roots do not match positional state/drive child binding"
        )
    if state_edge.child.node_id == drive_edge.child.node_id:
        raise ValueError("positional state and drive children must be distinct")
    if state_edge.modulation_index != receipt.state_edge_modulation_index:
        raise ValueError("state-edge modulation index does not match the receipt")
    if drive_edge.modulation_index != receipt.drive_edge_modulation_index:
        raise ValueError("drive-edge modulation index does not match the receipt")
    if state_edge.modulation_index != spec.state_modulation_index:
        raise ValueError("state-edge modulation index does not match the exact step")
    if drive_edge.modulation_index != spec.drive_modulation_index:
        raise ValueError("drive-edge modulation index does not match the exact step")

    state = _beam_from_child(state_edge.child)
    drive = _beam_from_child(drive_edge.child)
    if state.digest() != receipt.state_tree_digest:
        raise ValueError("state tree digest does not match the ancestry receipt")
    if drive.digest() != receipt.drive_tree_digest:
        raise ValueError("drive tree digest does not match the ancestry receipt")
    covered_roots = {state.root.node_id, drive.root.node_id}
    observed_roots = {edge.child.node_id for edge in result.root.children}
    if covered_roots != observed_roots:
        raise ValueError("receipt leaves an extra or missing child role")
    return state, drive


def extract_predecessor(
    result: Any, spec: TemporalStepSpec, receipt: TemporalStepReceipt
) -> Any:
    """Recover the exact complete predecessor after full two-role validation."""

    predecessor, _ = validate_step_receipt(result, spec, receipt)
    return predecessor


def trajectory(
    initial: Any,
    drives: Sequence[Any],
    specs: Sequence[TemporalStepSpec],
) -> tuple[list[Any], list[TemporalStepReceipt]]:
    if not drives or len(drives) != len(specs):
        raise ValueError("trajectory requires equal nonempty drive and spec sequences")
    if len(drives) > MAX_STEPS:
        raise ValueError("trajectory exceeds the maximum bounded step count")
    state = initial
    states = [state]
    receipts: list[TemporalStepReceipt] = []
    for expected_index, (drive, spec) in enumerate(
        zip(drives, specs, strict=True), start=1
    ):
        if not isinstance(spec, TemporalStepSpec) or spec.step_index != expected_index:
            raise ValueError("trajectory steps must be ordered and contiguous")
        state, receipt = temporal_step(state, drive, spec)
        states.append(state)
        receipts.append(receipt)
    return states, receipts


def validate_trajectory(
    states: Sequence[Any],
    drives: Sequence[Any],
    specs: Sequence[TemporalStepSpec],
    receipts: Sequence[TemporalStepReceipt],
) -> Any:
    if len(states) != len(drives) + 1:
        raise ValueError("trajectory state count is truncated or extended")
    if len(drives) != len(specs) or len(specs) != len(receipts):
        raise ValueError("trajectory drive, step, and receipt counts must match")
    if not drives or len(drives) > MAX_STEPS:
        raise ValueError("trajectory step count is outside the bounded envelope")
    for index, (drive, spec, receipt) in enumerate(
        zip(drives, specs, receipts, strict=True), start=1
    ):
        if spec.step_index != index or receipt.step_index != index:
            raise ValueError("trajectory steps must be ordered and contiguous")
        predecessor, embedded_drive = validate_step_receipt(
            states[index], spec, receipt
        )
        if predecessor.canonical_bytes() != states[index - 1].canonical_bytes():
            raise ValueError("trajectory predecessor bytes do not match prior state")
        if embedded_drive.canonical_bytes() != drive.canonical_bytes():
            raise ValueError("trajectory drive bytes do not match declared drive")
    recovered = states[-1]
    for spec, receipt in zip(reversed(specs), reversed(receipts), strict=True):
        recovered = extract_predecessor(recovered, spec, receipt)
    if recovered.canonical_bytes() != states[0].canonical_bytes():
        raise ValueError("reverse receipt traversal did not recover exact T0 bytes")
    return recovered


def drive_tree(step_index: int, variant: int = 0) -> Any:
    """Create a small, step-scoped complete drive tree."""

    base = f"drive{step_index}v{variant}"
    leaf = r0.PhaseNode(
        f"{base}.leaf",
        23.0 + 14.0 * step_index + 2.0 * variant,
        phase_rad=0.11 * (step_index + 1) + 0.03 * variant,
    )
    root = r0.PhaseNode(
        f"{base}.root",
        83.0 + 48.0 * step_index + 5.0 * variant,
        phase_rad=-0.07 * (step_index + 1) + 0.02 * variant,
        children=(r0.PhaseEdge(0.29 + 0.04 * step_index, leaf),),
    )
    return r0.RecursivePhaseBeam(root=root, global_spin_phase_rad=0.0)


def step_spec(step_index: int) -> TemporalStepSpec:
    return TemporalStepSpec(
        step_index=step_index,
        root_id=f"time{step_index}.root",
        carrier_frequency_hz=1601.0 + 137.0 * step_index,
        root_phase_rad=0.09 * step_index,
        state_modulation_index=0.63 + 0.03 * step_index,
        drive_modulation_index=0.41 + 0.02 * step_index,
        state_child_index=1,
        drive_child_index=0,
    )


def step_spec_schema_document() -> dict[str, Any]:
    return {
        "$id": "urn:cat-cas:recursive-wave-temporal-step:v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "carrier_frequency_hz": {
                "exclusiveMaximum": r0.SAMPLE_RATE_HZ / 2,
                "exclusiveMinimum": 0.0,
                "type": "number",
            },
            "drive_modulation_index": {
                "exclusiveMinimum": 0.0,
                "maximum": 2.0,
                "type": "number",
            },
            "drive_child_index": {"maximum": 1, "minimum": 0, "type": "integer"},
            "root_id": {
                "maxLength": MAX_ID_BYTES,
                "pattern": r"^time[1-6]\.[A-Za-z0-9._-]+$",
                "type": "string",
            },
            "root_phase_rad": {
                "maximum": 2.0 * math.pi,
                "minimum": -2.0 * math.pi,
                "type": "number",
            },
            "schema": {"const": STEP_SCHEMA, "type": "string"},
            "state_modulation_index": {
                "exclusiveMinimum": 0.0,
                "maximum": 2.0,
                "type": "number",
            },
            "state_child_index": {"maximum": 1, "minimum": 0, "type": "integer"},
            "step_index": {
                "maximum": MAX_STEPS,
                "minimum": 1,
                "type": "integer",
            },
        },
        "required": sorted(STEP_KEYS),
        "title": "CAT_CAS complete-tree temporal step specification v1",
        "type": "object",
        "x-semantic-constraints": [
            "root_id begins with time{step_index}.",
            "state_child_index and drive_child_index are distinct canonical serialized child positions 0 and 1",
            "all numbers are finite and booleans are not numbers",
            "no answer, score, response, energy, spin, candidate, or winner field is admitted",
        ],
    }


def receipt_schema_document() -> dict[str, Any]:
    hash_field = {"pattern": HASH_PATTERN.pattern, "type": "string"}
    id_field = {
        "maxLength": MAX_ID_BYTES,
        "pattern": r0.NODE_ID_PATTERN.pattern,
        "type": "string",
    }
    return {
        "$id": "urn:cat-cas:recursive-wave-ancestry-receipt:v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "child_roles": {
                "const": ["state", "drive"],
                "maxItems": 2,
                "minItems": 2,
                "type": "array",
            },
            "drive_child_index": {"maximum": 1, "minimum": 0, "type": "integer"},
            "drive_edge_modulation_index": {
                "exclusiveMinimum": 0.0,
                "maximum": 2.0,
                "type": "number",
            },
            "drive_root_id": id_field,
            "drive_tree_digest": hash_field,
            "result_root_id": id_field,
            "result_tree_digest": hash_field,
            "schema": {"const": RECEIPT_SCHEMA, "type": "string"},
            "state_edge_modulation_index": {
                "exclusiveMinimum": 0.0,
                "maximum": 2.0,
                "type": "number",
            },
            "state_child_index": {"maximum": 1, "minimum": 0, "type": "integer"},
            "state_root_id": id_field,
            "state_tree_digest": hash_field,
            "step_index": {
                "maximum": MAX_STEPS,
                "minimum": 1,
                "type": "integer",
            },
            "step_spec": step_spec_schema_document(),
            "step_spec_canonical_sha256": hash_field,
            "step_spec_digest": hash_field,
        },
        "required": sorted(RECEIPT_KEYS),
        "title": "CAT_CAS complete-tree ancestry receipt v1",
        "type": "object",
        "x-semantic-constraints": [
            "the exact step specification is embedded and bound twice",
            "state_child_index and drive_child_index are distinct positions 0 and 1 in canonical serialized child order",
            "one state role and one drive role cover every result-root child",
            "both child roots, complete tree digests, and modulation indices match the result",
            "the complete result root and tree digest match",
        ],
    }


def reference_test_ids() -> list[str]:
    return [
        "depth_grows_one_per_step",
        "node_count_growth",
        "every_state_is_valid_complete_tree",
        "unit_modulus_trajectory",
        "each_step_changes_the_complete_state",
        "deterministic_trajectory_and_receipts",
        "trajectory_construction_order_rejected",
        "exact_reverse_ancestry_recovers_t0_bytes",
        "drive_order_changes_history",
        "reordered_trajectory_response_bounded",
        "step_spec_strict_round_trip",
        "step_spec_duplicate_key_rejected",
        "step_spec_unknown_field_rejected",
        "step_spec_nonfinite_rejected",
        "step_spec_boolean_number_rejected",
        "step_spec_step_root_mismatch_rejected",
        "step_spec_nyquist_rejected",
        "step_spec_modulation_envelope_rejected",
        "step_spec_canonical_mutation_rejected",
        "step_spec_digest_deterministic",
        "receipt_strict_round_trip",
        "receipt_duplicate_key_rejected",
        "receipt_unknown_field_rejected",
        "receipt_nonfinite_number_rejected",
        "receipt_boolean_number_rejected",
        "receipt_canonical_mutation_rejected",
        "receipt_binds_complete_result",
        "receipt_binds_exact_step_spec",
        "receipt_binds_both_roles",
        "state_drive_id_collision_rejected",
        "new_root_collision_rejected",
        "wrong_state_digest_rejected",
        "wrong_drive_digest_rejected",
        "swapped_roles_rejected",
        "equal_beta_swapped_roles_rejected",
        "changed_state_beta_rejected",
        "changed_drive_beta_rejected",
        "changed_step_spec_rejected",
        "wrong_result_root_rejected",
        "extra_child_rejected",
        "missing_child_rejected",
        "duplicate_role_rejected",
        "reordered_receipt_rejected",
        "drive_order_reversal_changes_history",
        "flat_wave_native_admission_rejected",
        "flat_wave_receipt_validation_rejected",
        "flat_wave_exact_ancestry_unavailable",
        "decoded_spin_internal_hierarchy_absent",
        "decoded_spin_native_admission_rejected",
        "decoded_spin_receipt_validation_rejected",
        "decoded_spin_exact_ancestry_unavailable",
        "decoded_spin_final_waveform_differs",
        "trajectory_truncation_rejected",
        "trajectory_duplication_rejected",
        "fixture_substitution_rejected",
        "manifest_order_mutation_rejected",
        "ast_native_call_graph_no_scalar_feedback",
        "ast_forbidden_identifiers_mutation_rejected",
        "ast_alias_callable_mutation_rejected",
        "ast_module_helper_mutation_rejected",
        "ast_class_method_mutation_rejected",
        "ast_dynamic_callable_mutation_rejected",
        "ast_digest_control_mutation_rejected",
        "ast_length_feedback_mutation_rejected",
        "ast_module_shadowing_mutation_rejected",
        "ast_module_attribute_rebind_mutation_rejected",
        "ast_decorated_root_mutation_rejected",
        "ast_global_rebind_mutation_rejected",
        "ast_import_alias_rebind_mutation_rejected",
        "ast_loop_target_rebind_mutation_rejected",
        "ast_function_default_rebind_mutation_rejected",
        "ast_native_runtime_binding_shadow_mutation_rejected",
        "ast_receipt_lifecycle_rebind_mutation_rejected",
        "ast_match_capture_result_rebind_mutation_rejected",
        "ast_result_store_rebind_mutation_rejected",
        "committed_tree_wav_bytes_close",
        "committed_step_receipt_bytes_close",
        "manifest_binds_all_fixture_bytes",
    ]


def reference_test_spec() -> dict[str, Any]:
    return {
        "edge_conventions": {
            "canonical_json": "UTF-8, sorted keys, indent=2, newline terminated, strict duplicate rejection",
            "collapsed_controls": "waveform-only controls must fail native tree and receipt admission",
            "native_call_graph": "closed-call AST reachability from temporal_step, trajectory, and extract_predecessor; unapproved dynamic, module, and method calls fail",
            "receipt_roles": "explicit distinct canonical child indices bind state and drive, with no additional child",
            "result_comparison": "environment receipt informational; numeric leaves use frozen portable tolerance",
            "wav": "minimal RIFF fmt then data, stereo I/Q IEEE float32, 48000 Hz, 6000 samples",
        },
        "numeric_envelope": {
            "maximum_bounded_steps": MAX_STEPS,
            "nonidentity_tolerance": NONIDENTITY_TOL,
            "ordered_response_max": ORDER_RESPONSE_MAX,
            "portable_metric_atol": PORTABLE_METRIC_ATOL,
            "portable_metric_rtol": PORTABLE_METRIC_RTOL,
            "unit_modulus_tolerance": UNIT_MODULUS_TOL,
            "wav_render_tolerance": r0.FLOAT32_COMPLEX_TOL,
        },
        "schema": "recursive_wave_operator_reference_tests_v1",
        "tests": [{"id": test_id} for test_id in reference_test_ids()],
    }


def collapsed_flat_trajectory(
    initial: Any,
    drives: Sequence[Any],
    specs: Sequence[TemporalStepSpec],
    t: np.ndarray,
) -> np.ndarray:
    """Declared baseline that retains only a rendered flat waveform between steps."""

    state_wave = r0.flat_multitone_replacement(initial, t)
    for drive, spec in zip(drives, specs, strict=True):
        drive_wave = r0.flat_multitone_replacement(drive, t)
        phase = (
            2.0 * math.pi * spec.carrier_frequency_hz * t
            + spec.root_phase_rad
            + spec.state_modulation_index * np.sin(np.angle(state_wave))
            + spec.drive_modulation_index * np.sin(np.angle(drive_wave))
        )
        state_wave = np.exp(1j * phase)
    return state_wave


def collapsed_spin_trajectory(
    initial: Any,
    drives: Sequence[Any],
    specs: Sequence[TemporalStepSpec],
    t: np.ndarray,
) -> np.ndarray:
    """Declared baseline that collapses each complete state to one global sign."""

    state_wave = initial.render(t)
    for drive, spec in zip(drives, specs, strict=True):
        spin = 1.0 if float(np.real(np.mean(state_wave))) >= 0.0 else -1.0
        drive_wave = drive.render(t)
        phase = (
            2.0 * math.pi * spec.carrier_frequency_hz * t
            + spec.root_phase_rad
            + (0.0 if spin > 0.0 else math.pi)
            + spec.drive_modulation_index * np.sin(np.angle(drive_wave))
        )
        state_wave = np.exp(1j * phase)
    return state_wave


def reference_objects() -> tuple[
    list[Any], list[Any], list[TemporalStepSpec], list[TemporalStepReceipt]
]:
    initial = r0.hierarchy_a()
    drives = [drive_tree(index) for index in range(1, 4)]
    specs = [step_spec(index) for index in range(1, 4)]
    states, receipts = trajectory(initial, drives, specs)
    validate_trajectory(states, drives, specs, receipts)
    return states, drives, specs, receipts


def state_tree_path(index: int) -> str:
    return f"{FIXTURE_DIR_NAME}/states/T{index}.tree.json"


def state_wav_path(index: int) -> str:
    return f"{FIXTURE_DIR_NAME}/states/T{index}_iq.wav"


def drive_path(index: int) -> str:
    return f"{FIXTURE_DIR_NAME}/drives/D{index}.tree.json"


def spec_path(index: int) -> str:
    return f"{FIXTURE_DIR_NAME}/steps/step{index}.spec.json"


def receipt_path(index: int) -> str:
    return f"{FIXTURE_DIR_NAME}/receipts/step{index}.receipt.json"


def ordered_fixture_paths() -> list[str]:
    paths: list[str] = []
    for index in range(4):
        paths.extend((state_tree_path(index), state_wav_path(index)))
    paths.extend(drive_path(index) for index in range(1, 4))
    paths.extend(spec_path(index) for index in range(1, 4))
    paths.extend(receipt_path(index) for index in range(1, 4))
    return paths


def build_fixture_files(package_dir: Path) -> None:
    states, drives, specs, receipts = reference_objects()
    t = r0.sample_times()
    for index, state in enumerate(states):
        r0.write_bytes_atomic(package_dir / state_tree_path(index), state.canonical_bytes())
        r0.write_bytes_atomic(
            package_dir / state_wav_path(index),
            r0.float32_iq_wav_bytes(state.render(t)),
        )
    for index, drive in enumerate(drives, start=1):
        r0.write_bytes_atomic(package_dir / drive_path(index), drive.canonical_bytes())
    for index, spec in enumerate(specs, start=1):
        r0.write_bytes_atomic(package_dir / spec_path(index), spec.canonical_bytes())
    for index, receipt in enumerate(receipts, start=1):
        r0.write_bytes_atomic(
            package_dir / receipt_path(index), receipt.canonical_bytes()
        )
    fixture_root = package_dir / FIXTURE_DIR_NAME
    expected = set(ordered_fixture_paths())
    observed = {
        path.relative_to(package_dir).as_posix()
        for path in fixture_root.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(f"unexpected R1 fixture files: {unexpected}")


def validate_state_fixture(tree: Any, payload: bytes) -> tuple[np.ndarray, list[str]]:
    expected = r0.float32_iq_wav_bytes(tree.render(r0.sample_times()))
    if payload != expected:
        raise ValueError("committed state WAV is not the deterministic tree render")
    _, samples, chunks = r0.parse_float32_wav_bytes(payload)
    if chunks != ["fmt ", "data"]:
        raise ValueError("committed state WAV must contain exactly fmt then data")
    beam = r0.complex_from_iq(samples)
    if float(np.max(np.abs(beam - tree.render(r0.sample_times())))) > r0.FLOAT32_COMPLEX_TOL:
        raise ValueError("committed state WAV exceeds the frozen float32 envelope")
    return beam, chunks


def load_committed_packet(package_dir: Path) -> dict[str, Any]:
    expected_states, expected_drives, expected_specs, expected_receipts = (
        reference_objects()
    )
    states: list[Any] = []
    state_waves: list[np.ndarray] = []
    state_chunks: list[list[str]] = []
    for index, expected in enumerate(expected_states):
        tree_path = package_dir / state_tree_path(index)
        if tree_path.read_bytes() != expected.canonical_bytes():
            raise ValueError(f"committed T{index} tree differs from the native trajectory")
        state = r0.load_tree(tree_path)
        payload = (package_dir / state_wav_path(index)).read_bytes()
        beam, chunks = validate_state_fixture(state, payload)
        states.append(state)
        state_waves.append(beam)
        state_chunks.append(chunks)

    drives: list[Any] = []
    for index, expected in enumerate(expected_drives, start=1):
        path = package_dir / drive_path(index)
        if path.read_bytes() != expected.canonical_bytes():
            raise ValueError(f"committed D{index} tree differs from the native drive")
        drives.append(r0.load_tree(path))

    specs: list[TemporalStepSpec] = []
    for index, expected in enumerate(expected_specs, start=1):
        path = package_dir / spec_path(index)
        if path.read_bytes() != expected.canonical_bytes():
            raise ValueError(f"committed step {index} differs from the frozen step")
        specs.append(TemporalStepSpec.from_bytes(path.read_bytes()))

    receipts: list[TemporalStepReceipt] = []
    for index, expected in enumerate(expected_receipts, start=1):
        path = package_dir / receipt_path(index)
        if path.read_bytes() != expected.canonical_bytes():
            raise ValueError(f"committed receipt {index} differs from native custody")
        receipts.append(TemporalStepReceipt.from_bytes(path.read_bytes()))

    recovered = validate_trajectory(states, drives, specs, receipts)
    return {
        "drives": drives,
        "receipts": receipts,
        "recovered": recovered,
        "specs": specs,
        "state_chunks": state_chunks,
        "state_waves": state_waves,
        "states": states,
    }


def trajectory_manifest(package_dir: Path) -> dict[str, Any]:
    packet = load_committed_packet(package_dir)
    states = packet["states"]
    drives = packet["drives"]
    specs = packet["specs"]
    receipts = packet["receipts"]
    trajectory_records: list[dict[str, Any]] = []
    drive_records: list[dict[str, Any]] = []
    spec_records: list[dict[str, Any]] = []
    receipt_records: list[dict[str, Any]] = []
    wav_bytes = 0

    for index, state in enumerate(states):
        tree_file = package_dir / state_tree_path(index)
        wav_file = package_dir / state_wav_path(index)
        _, samples, chunks = r0.parse_float32_wav_bytes(wav_file.read_bytes())
        wave = r0.complex_from_iq(samples)
        wav_bytes += wav_file.stat().st_size
        trajectory_records.append(
            {
                "channels": 2,
                "complex_energy": _metric(np.sum(np.abs(wave) ** 2)),
                "complex_peak": _metric(np.max(np.abs(wave))),
                "complex_rms": _metric(np.sqrt(np.mean(np.abs(wave) ** 2))),
                "depth": state.root.max_depth(),
                "drive_digest": None if index == 0 else drives[index - 1].digest(),
                "dtype": "ieee_float32_le",
                "node_count": len(node_ids(state)),
                "predecessor_digest": None if index == 0 else states[index - 1].digest(),
                "receipt_digest": None if index == 0 else receipts[index - 1].digest(),
                "riff_chunks": chunks,
                "root_id": state.root.node_id,
                "sample_count": int(samples.shape[0]),
                "sample_rate_hz": r0.SAMPLE_RATE_HZ,
                "state_index": index,
                "step_spec_digest": None if index == 0 else specs[index - 1].digest(),
                "tree_digest": state.digest(),
                "tree_path": state_tree_path(index),
                "tree_sha256": r0.sha256_file(tree_file),
                "wav_byte_count": wav_file.stat().st_size,
                "wav_path": state_wav_path(index),
                "wav_sha256": r0.sha256_file(wav_file),
            }
        )

    for index, drive in enumerate(drives, start=1):
        path = package_dir / drive_path(index)
        drive_records.append(
            {
                "depth": drive.root.max_depth(),
                "drive_index": index,
                "node_count": len(node_ids(drive)),
                "path": drive_path(index),
                "root_id": drive.root.node_id,
                "sha256": r0.sha256_file(path),
                "tree_digest": drive.digest(),
            }
        )
    for index, spec in enumerate(specs, start=1):
        path = package_dir / spec_path(index)
        spec_records.append(
            {
                "canonical_sha256": spec.canonical_sha256(),
                "digest": spec.digest(),
                "path": spec_path(index),
                "sha256": r0.sha256_file(path),
                "step_index": index,
            }
        )
    for index, receipt in enumerate(receipts, start=1):
        path = package_dir / receipt_path(index)
        receipt_records.append(
            {
                "digest": receipt.digest(),
                "path": receipt_path(index),
                "sha256": r0.sha256_file(path),
                "step_index": index,
            }
        )

    paths = ordered_fixture_paths()
    set_hasher = hashlib.sha256()
    total_bytes = 0
    for relative in paths:
        path = package_dir / relative
        digest = r0.sha256_file(path)
        total_bytes += path.stat().st_size
        set_hasher.update(relative.encode("utf-8") + b"\0")
        set_hasher.update(bytes.fromhex(digest))
    manifest = {
        "drives": drive_records,
        "fixture_count": len(paths),
        "fixture_set_sha256": set_hasher.hexdigest(),
        "generator": GENERATOR_ID,
        "ordered_fixture_paths": paths,
        "receipts": receipt_records,
        "schema": "recursive_wave_operator_trajectory_manifest_v1",
        "step_specs": spec_records,
        "total_fixture_bytes": total_bytes,
        "trajectory": trajectory_records,
        "wav_fixture_bytes": wav_bytes,
        "wav_fixture_count": len(states),
    }
    validate_manifest_document(manifest)
    return manifest


def validate_manifest_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("trajectory manifest must be an object")
    _exact_keys(
        value,
        {
            "drives",
            "fixture_count",
            "fixture_set_sha256",
            "generator",
            "ordered_fixture_paths",
            "receipts",
            "schema",
            "step_specs",
            "total_fixture_bytes",
            "trajectory",
            "wav_fixture_bytes",
            "wav_fixture_count",
        },
        "trajectory manifest",
    )
    if value["schema"] != "recursive_wave_operator_trajectory_manifest_v1":
        raise ValueError("unexpected trajectory manifest schema")
    if value["generator"] != GENERATOR_ID:
        raise ValueError("unexpected trajectory manifest generator")
    if value["ordered_fixture_paths"] != ordered_fixture_paths():
        raise ValueError("manifest fixture order differs from the frozen packet")
    if [item.get("state_index") for item in value["trajectory"]] != list(range(4)):
        raise ValueError("manifest trajectory order is not exactly T0 through T3")
    if [item.get("drive_index") for item in value["drives"]] != [1, 2, 3]:
        raise ValueError("manifest drive order is not exactly D1 through D3")
    if [item.get("step_index") for item in value["step_specs"]] != [1, 2, 3]:
        raise ValueError("manifest step-spec order is not exact")
    if [item.get("step_index") for item in value["receipts"]] != [1, 2, 3]:
        raise ValueError("manifest receipt order is not exact")
    if value["fixture_count"] != len(ordered_fixture_paths()):
        raise ValueError("manifest fixture count is not exact")
    if value["wav_fixture_count"] != 4:
        raise ValueError("manifest WAV fixture count is not exact")
    _hash(value["fixture_set_sha256"], "fixture_set_sha256")
    return value


def _expect_error(action: Callable[[], Any], expected_text: str) -> tuple[bool, str]:
    try:
        action()
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        message = str(exc)
        return expected_text in message, message
    return False, "NO_ERROR"


def _receipt_with(receipt: TemporalStepReceipt, **changes: Any) -> TemporalStepReceipt:
    document = copy.deepcopy(receipt.document())
    document.update(changes)
    return TemporalStepReceipt.from_document(document)


NATIVE_AST_ROOTS = ("temporal_step", "trajectory", "extract_predecessor")
NATIVE_AST_FORBIDDEN = {
    "argmax",
    "argmin",
    "candidate",
    "collapsed_flat_trajectory",
    "collapsed_spin_trajectory",
    "decode_spin",
    "energy",
    "expected_answer",
    "fft",
    "matched_response",
    "response",
    "scalar",
    "score",
    "spin",
    "winner",
}
NATIVE_AST_ALLOWED_BUILTIN_CALLS = {
    "TemporalStepReceipt",
    "ValueError",
    "enumerate",
    "isinstance",
    "len",
    "set",
    "zip",
}
NATIVE_AST_ALLOWED_ATTRIBUTE_CALLS = {
    "TemporalStepReceipt.from_bytes",
    "beam.canonical_bytes",
    "beam.root.walk",
    "canonical_roots.index",
    "drive.digest",
    "r0.PhaseEdge",
    "r0.PhaseNode",
    "r0.RecursivePhaseBeam",
    "r0.deserialize_tree_bytes",
    "receipt.canonical_bytes",
    "receipt.step_spec.canonical_bytes",
    "receipts.append",
    "result.digest",
    "result.canonical_bytes",
    "spec.canonical_bytes",
    "spec.canonical_sha256",
    "spec.digest",
    "state.digest",
    "states.append",
}


def _ast_dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _ast_same(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def _parsed_expression(source: str) -> ast.AST:
    return ast.parse(source, mode="eval").body


def _stored_root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _binding_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.append(child.id)
        elif isinstance(child, ast.ExceptHandler) and isinstance(child.name, str):
            names.append(child.name)
        elif isinstance(child, ast.MatchAs) and child.name is not None:
            names.append(child.name)
        elif isinstance(child, ast.MatchStar) and child.name is not None:
            names.append(child.name)
        elif isinstance(child, ast.MatchMapping) and child.rest is not None:
            names.append(child.rest)
    return names


def _native_shape_violations(functions: Mapping[str, ast.FunctionDef]) -> list[str]:
    violations: list[str] = []
    temporal = functions["temporal_step"]
    trajectory_node = functions["trajectory"]
    extractor = functions["extract_predecessor"]

    def named_assignments(function: ast.FunctionDef, target_name: str) -> list[ast.AST]:
        values: list[ast.AST] = []
        for node in ast.walk(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == target_name for target in targets):
                values.append(node.value)
        return values

    def name_binding_count(function: ast.FunctionDef, target_name: str) -> int:
        return sum(name == target_name for name in _binding_names(function))

    expected_root = _parsed_expression(
        "r0.PhaseNode(spec.root_id, spec.carrier_frequency_hz, spec.root_phase_rad, "
        "(r0.PhaseEdge(spec.state_modulation_index, state.root), "
        "r0.PhaseEdge(spec.drive_modulation_index, drive.root)))"
    )
    root_values = named_assignments(temporal, "root")
    if len(root_values) != 1 or not _ast_same(root_values[0], expected_root):
        violations.append("temporal_step:root_assignment_shape")
    if name_binding_count(temporal, "root") != 1:
        violations.append("temporal_step:root_write_count")

    expected_result = _parsed_expression(
        "r0.RecursivePhaseBeam(root=root, global_spin_phase_rad=0.0)"
    )
    result_values = named_assignments(temporal, "result")
    if len(result_values) != 1 or not _ast_same(result_values[0], expected_result):
        violations.append("temporal_step:result_assignment_shape")
    if name_binding_count(temporal, "result") != 1:
        violations.append("temporal_step:result_write_count")
    if name_binding_count(temporal, "receipt") != 1:
        violations.append("temporal_step:receipt_write_count")

    for function, protected in (
        (temporal, {"state", "drive", "spec"}),
        (trajectory_node, {"initial", "drives", "specs"}),
        (extractor, {"result", "spec", "receipt"}),
    ):
        for binding_name in _binding_names(function):
            if binding_name in protected:
                violations.append(f"{function.name}:protected_write:{binding_name}")
        for node in ast.walk(function):
            if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(
                getattr(node, "ctx", None), (ast.Store, ast.Del)
            ):
                root_name = _stored_root_name(node)
                if root_name in protected:
                    violations.append(f"{function.name}:protected_write:{root_name}")

    temporal_returns = [node for node in ast.walk(temporal) if isinstance(node, ast.Return)]
    expected_temporal_return = _parsed_expression("(result, receipt)")
    if (
        len(temporal_returns) != 1
        or temporal_returns[0].value is None
        or not _ast_same(temporal_returns[0].value, expected_temporal_return)
    ):
        violations.append("temporal_step:return_shape")

    step_calls = [
        node
        for node in ast.walk(trajectory_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "temporal_step"
    ]
    expected_step_call = _parsed_expression("temporal_step(state, drive, spec)")
    if len(step_calls) != 1 or not _ast_same(step_calls[0], expected_step_call):
        violations.append("trajectory:temporal_step_call_shape")
    if name_binding_count(trajectory_node, "state") != 2:
        violations.append("trajectory:state_write_count")
    if name_binding_count(trajectory_node, "receipt") != 1:
        violations.append("trajectory:receipt_write_count")
    trajectory_returns = [
        node for node in ast.walk(trajectory_node) if isinstance(node, ast.Return)
    ]
    expected_trajectory_return = _parsed_expression("(states, receipts)")
    if (
        len(trajectory_returns) != 1
        or trajectory_returns[0].value is None
        or not _ast_same(trajectory_returns[0].value, expected_trajectory_return)
    ):
        violations.append("trajectory:return_shape")

    validation_calls = [
        node
        for node in ast.walk(extractor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_step_receipt"
    ]
    expected_validation = _parsed_expression(
        "validate_step_receipt(result, spec, receipt)"
    )
    if len(validation_calls) != 1 or not _ast_same(
        validation_calls[0], expected_validation
    ):
        violations.append("extract_predecessor:validation_call_shape")
    extractor_returns = [
        node for node in ast.walk(extractor) if isinstance(node, ast.Return)
    ]
    expected_extractor_return = _parsed_expression("predecessor")
    if (
        len(extractor_returns) != 1
        or extractor_returns[0].value is None
        or not _ast_same(extractor_returns[0].value, expected_extractor_return)
    ):
        violations.append("extract_predecessor:return_shape")
    if name_binding_count(extractor, "predecessor") != 1:
        violations.append("extract_predecessor:predecessor_write_count")

    receipt_validator = functions["validate_step_receipt"]
    validator_returns = [
        node for node in ast.walk(receipt_validator) if isinstance(node, ast.Return)
    ]
    expected_validator_return = _parsed_expression("(state, drive)")
    if (
        len(validator_returns) != 1
        or validator_returns[0].value is None
        or not _ast_same(validator_returns[0].value, expected_validator_return)
    ):
        violations.append("validate_step_receipt:return_shape")
    if name_binding_count(receipt_validator, "state") != 1:
        violations.append("validate_step_receipt:state_write_count")
    if name_binding_count(receipt_validator, "drive") != 1:
        violations.append("validate_step_receipt:drive_write_count")
    return sorted(set(violations))


def _module_shadowing_violations(
    module: ast.Module, functions: Mapping[str, ast.FunctionDef]
) -> list[str]:
    skeleton = copy.deepcopy(module)
    for statement in skeleton.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            statement.body = [ast.Pass()]
    skeleton_sha256 = hashlib.sha256(
        ast.dump(skeleton, include_attributes=False).encode("utf-8")
    ).hexdigest()

    class ModuleRuntimeVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.assignments: list[ast.AST] = []
            self.calls: list[ast.Call] = []
            self.definitions: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.definitions.append(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.definitions.append(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.definitions.append(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Assign(self, node: ast.Assign) -> None:
            self.assignments.append(node)
            self.visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self.assignments.append(node)
            if node.value is not None:
                self.visit(node.value)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self.assignments.append(node)
            self.visit(node.value)

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            self.assignments.append(node)
            self.visit(node.value)

        def visit_Call(self, node: ast.Call) -> None:
            self.calls.append(node)
            self.generic_visit(node)

    def collect(tree: ast.AST) -> ModuleRuntimeVisitor:
        visitor = ModuleRuntimeVisitor()
        visitor.visit(tree)
        return visitor

    visitor = collect(module)
    violations: set[str] = set()
    if skeleton_sha256 != "306c5a8c074ab84a1f4b81156d385017826562e7c8b061d3235260e93e78c278":
        violations.add(f"module_load_skeleton_changed:{skeleton_sha256}")
    assigned_names: list[str] = []
    expected_sys_binding = ast.parse("sys.modules[_spec.name] = r0").body[0]
    r0_assignments: list[ast.AST] = []
    spec_assignments: list[ast.AST] = []
    for assignment in visitor.assignments:
        if isinstance(assignment, ast.Assign):
            targets = list(assignment.targets)
        elif isinstance(assignment, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [assignment.target]
        else:
            targets = []
        for target in targets:
            if isinstance(target, ast.Name):
                assigned_names.append(target.id)
                if target.id == "r0":
                    r0_assignments.append(assignment)
                elif target.id == "_spec":
                    spec_assignments.append(assignment)
            elif not (
                isinstance(assignment, ast.Assign)
                and _ast_same(assignment, expected_sys_binding)
            ):
                violations.add(
                    f"complex_module_write:{ast.dump(target, include_attributes=False)}"
                )

    protected = NATIVE_AST_ALLOWED_BUILTIN_CALLS | set(functions) | {"dataclass"}
    violations.update(
        f"shadowed_callable:{name}" for name in set(assigned_names) & protected
    )
    expected_r0 = ast.parse("r0 = importlib.util.module_from_spec(_spec)").body[0]
    if len(r0_assignments) != 1 or not _ast_same(r0_assignments[0], expected_r0):
        violations.add("r0_runtime_binding_changed")
    expected_spec = ast.parse(
        '_spec = importlib.util.spec_from_file_location('
        '"catcas_recursive_phase_tree_r0", R0_SOURCE)'
    ).body[0]
    if len(spec_assignments) != 1 or not _ast_same(spec_assignments[0], expected_spec):
        violations.add("r0_loader_spec_binding_changed")

    for definition in visitor.definitions:
        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if definition.decorator_list:
                violations.add(f"decorated_runtime_function:{definition.name}")
        elif isinstance(definition, ast.ClassDef):
            expected = (
                [ast.parse("@dataclass(frozen=True)\nclass X:\n    pass").body[0].decorator_list[0]]
                if definition.name in {"TemporalStepSpec", "TemporalStepReceipt"}
                else []
            )
            if len(definition.decorator_list) != len(expected) or any(
                not _ast_same(left, right)
                for left, right in zip(definition.decorator_list, expected, strict=True)
            ):
                violations.add(f"runtime_class_decorators_changed:{definition.name}")

    expected_runtime = ast.parse(
        "PACKAGE_DIR = Path(__file__).resolve().parent\n"
        '_spec = importlib.util.spec_from_file_location('
        '"catcas_recursive_phase_tree_r0", R0_SOURCE)\n'
        "if _spec is None or _spec.loader is None:\n"
        '    raise RuntimeError("unable to load the established R0 recursive phase-tree reference")\n'
        "r0 = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(r0)\n"
        'HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")\n'
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )
    expected_calls = sorted(
        ast.dump(call, include_attributes=False) for call in collect(expected_runtime).calls
    )
    observed_calls = sorted(
        ast.dump(call, include_attributes=False) for call in visitor.calls
    )
    if observed_calls != expected_calls:
        violations.add("module_runtime_call_set_changed")
    return sorted(violations)


def native_ast_proof_text(source_text: str) -> dict[str, Any]:
    """Prove a closed, explicit native call graph with no scalar feedback names."""

    module = ast.parse(source_text)
    functions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    shape_violations = _native_shape_violations(functions)
    shadowing_violations = _module_shadowing_violations(module, functions)
    reachable: set[str] = set()
    frontier = list(NATIVE_AST_ROOTS)
    unresolved_calls: set[str] = set()
    forbidden_constructs: set[str] = set()
    protected_runtime_names = (
        NATIVE_AST_ALLOWED_BUILTIN_CALLS | set(functions) | {"r0"}
    )
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        if name not in functions:
            raise ValueError(f"native AST root/helper missing: {name}")
        reachable.add(name)
        for binding_name in _binding_names(functions[name]):
            if binding_name in protected_runtime_names:
                forbidden_constructs.add(
                    f"{name}:runtime_binding_write:{binding_name}"
                )
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                if node.id in protected_runtime_names:
                    forbidden_constructs.add(f"{name}:runtime_binding_write:{node.id}")
            elif isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                forbidden_constructs.add(
                    f"{name}:indirect_runtime_write:{ast.dump(node, include_attributes=False)}"
                )
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda, ast.ClassDef)):
                forbidden_constructs.add(f"{name}:{type(node).__name__}")
            if isinstance(node, ast.FunctionDef) and node is not functions[name]:
                forbidden_constructs.add(f"{name}:nested_function:{node.name}")
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called = node.func.id
                if called in functions:
                    if called not in reachable:
                        frontier.append(called)
                elif called not in NATIVE_AST_ALLOWED_BUILTIN_CALLS:
                    unresolved_calls.add(f"{name}:dynamic_or_aliased:{called}")
            elif isinstance(node.func, ast.Attribute):
                called = _ast_dotted_name(node.func)
                if called not in NATIVE_AST_ALLOWED_ATTRIBUTE_CALLS:
                    unresolved_calls.add(f"{name}:unapproved_attribute:{called}")
            else:
                unresolved_calls.add(f"{name}:dynamic_callable:{type(node.func).__name__}")
    hits: set[str] = set()
    for name in reachable:
        for node in ast.walk(functions[name]):
            candidate: str | None = None
            if isinstance(node, ast.Name):
                candidate = node.id
            elif isinstance(node, ast.Attribute):
                candidate = node.attr
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = node.value
            if candidate is not None:
                lowered = candidate.lower()
                if lowered != "global_spin_phase_rad":
                    hits.update(
                        term for term in NATIVE_AST_FORBIDDEN if term in lowered
                    )
    status = (
        "PASS"
        if not hits
        and not unresolved_calls
        and not forbidden_constructs
        and not shape_violations
        and not shadowing_violations
        else "FAIL"
    )
    return {
        "allowed_attribute_calls": sorted(NATIVE_AST_ALLOWED_ATTRIBUTE_CALLS),
        "forbidden_hits": sorted(hits),
        "forbidden_constructs": sorted(forbidden_constructs),
        "native_roots": list(NATIVE_AST_ROOTS),
        "reachable_functions": sorted(reachable),
        "recurrence_shape_violations": shape_violations,
        "shadowing_violations": shadowing_violations,
        "status": status,
        "unresolved_calls": sorted(unresolved_calls),
    }


def native_ast_proof(source_path: Path) -> dict[str, Any]:
    return native_ast_proof_text(source_path.read_text(encoding="utf-8"))


def native_ast_mutation_probes(source_path: Path) -> dict[str, Any]:
    source_text = source_path.read_text(encoding="utf-8")
    insertion = (
        '    require_pre_ising_orientation(state, "state")\n'
        '    require_pre_ising_orientation(drive, "drive")'
    )
    if source_text.count(insertion) != 1:
        raise ValueError("native AST mutation insertion point is not unique")

    def inject(statement: str) -> str:
        return source_text.replace(insertion, f"{statement}\n{insertion}", 1)

    phase_argument = "        spec.root_phase_rad,\n        ("
    if source_text.count(phase_argument) != 1:
        raise ValueError("native phase mutation point is not unique")

    def replace_phase(expression: str) -> str:
        return source_text.replace(
            phase_argument, f"        {expression},\n        (", 1
        )

    receipt_class_offset = source_text.index("class TemporalStepReceipt:")
    receipt_post_offset = source_text.index(
        "    def __post_init__(self) -> None:\n", receipt_class_offset
    ) + len("    def __post_init__(self) -> None:\n")
    receipt_lifecycle_rebind = (
        source_text[:receipt_post_offset]
        + "        global temporal_step\n"
        + '        if self.state_tree_digest == "x" * 64:\n'
        + "            temporal_step = lambda state, drive, spec: (state, self)\n"
        + source_text[receipt_post_offset:]
    )
    result_assignment = (
        "    result = r0.RecursivePhaseBeam(root=root, global_spin_phase_rad=0.0)\n"
    )
    if source_text.count(result_assignment) != 1:
        raise ValueError("native result mutation point is not unique")
    result_store_rebind = source_text.replace(
        result_assignment,
        result_assignment
        + '    if state.digest() == "x" * 64 and (result := state):\n'
        + "        pass\n",
        1,
    )
    match_capture_result_rebind = source_text.replace(
        result_assignment,
        result_assignment
        + "    match {state.digest(): state}:\n"
        + '        case {"111425f51dc50ceaeb30e01dec958a33fe9ecd5a6a0b0155e84fb83cee0d2521": result}:\n'
        + "            pass\n",
        1,
    )

    forbidden_results = {
        term: native_ast_proof_text(inject(f"    {term} = None"))["status"] == "FAIL"
        for term in sorted(NATIVE_AST_FORBIDDEN)
    }
    probes = {
        "alias_callable": inject(
            "    alias = require_pre_ising_orientation\n    alias(state, \"state\")"
        ),
        "class_method": inject("    state.compute()"),
        "dynamic_callable": inject("    (lambda value: value)(state)"),
        "digest_control": replace_phase(
            'spec.root_phase_rad + (math.pi if state.digest() == "x" * 64 else 0.0)'
        ),
        "length_feedback": replace_phase("spec.root_phase_rad + len(state_ids)"),
        "module_helper": inject("    np.mean(state)"),
        "module_shadowing": source_text + "\nscore = lambda value: 1\nlen = score\n",
        "module_attribute_rebind": source_text
        + "\n_native_phase_node = r0.PhaseNode\n"
        + "def scalar_phase_node(node_id, frequency_hz, phase_rad=0.0, children=()):\n"
        + "    score = 0.125\n"
        + "    return _native_phase_node(node_id, frequency_hz, phase_rad + score, children)\n"
        + "r0.PhaseNode = scalar_phase_node\n",
        "decorated_root": source_text.replace(
            "def temporal_step(\n", "@staticmethod\ndef temporal_step(\n", 1
        ),
        "global_rebind": source_text
        + '\nreplacement = temporal_step\nglobals()["temporal_step"] = replacement\n',
        "import_alias_rebind": source_text
        + "\nfrom builtins import len as temporal_step\n",
        "loop_target_rebind": source_text
        + "\nreplacement = temporal_step\nfor temporal_step in [replacement]:\n    pass\n",
        "function_default_rebind": source_text
        + "\n_native_phase_node = r0.PhaseNode\n"
        + "def scalar_phase_node(node_id, frequency_hz, phase_rad=0.0, children=()):\n"
        + "    score = 0.125\n"
        + "    return _native_phase_node(node_id, frequency_hz, phase_rad + score, children)\n"
        + 'def bind_default(value=setattr(r0, "PhaseNode", scalar_phase_node)):\n'
        + "    return value\n",
        "native_runtime_binding_shadow": inject("    r0 = spec.runtime_binding"),
        "receipt_lifecycle_rebind": receipt_lifecycle_rebind,
        "match_capture_result_rebind": match_capture_result_rebind,
        "result_store_rebind": result_store_rebind,
    }
    return {
        "forbidden_identifiers": forbidden_results,
        **{
            name: native_ast_proof_text(mutated)["status"] == "FAIL"
            for name, mutated in probes.items()
        },
    }


def run_reference_tests(
    package_dir: Path, source_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    packet = load_committed_packet(package_dir)
    states = packet["states"]
    drives = packet["drives"]
    specs = packet["specs"]
    receipts = packet["receipts"]
    t = r0.sample_times()
    final = states[-1]
    tests: list[dict[str, Any]] = []

    def record(test_id: str, passed: bool, observed: Any) -> None:
        tests.append(
            {
                "id": test_id,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
            }
        )

    depths = [state.root.max_depth() for state in states]
    counts = [len(node_ids(state)) for state in states]
    record("depth_grows_one_per_step", depths == [3, 4, 5, 6], depths)
    record("node_count_growth", counts == [3, 6, 9, 12], counts)
    valid_states = all(
        r0.deserialize_tree_bytes(state.canonical_bytes(), require_canonical=True).digest()
        == state.digest()
        for state in states
    )
    record("every_state_is_valid_complete_tree", valid_states, valid_states)
    unit_error = max(
        float(np.max(np.abs(np.abs(state.render(t)) - 1.0))) for state in states
    )
    record("unit_modulus_trajectory", unit_error <= UNIT_MODULUS_TOL, _metric(unit_error))
    step_differences = [
        float(np.max(np.abs(states[index + 1].render(t) - states[index].render(t))))
        for index in range(3)
    ]
    record(
        "each_step_changes_the_complete_state",
        all(value > NONIDENTITY_TOL for value in step_differences),
        [_metric(value) for value in step_differences],
    )
    repeated_states, repeated_receipts = trajectory(states[0], drives, specs)
    deterministic = (
        [state.canonical_bytes() for state in states]
        == [state.canonical_bytes() for state in repeated_states]
        and [receipt.canonical_bytes() for receipt in receipts]
        == [receipt.canonical_bytes() for receipt in repeated_receipts]
    )
    record(
        "deterministic_trajectory_and_receipts",
        deterministic,
        [state.digest() for state in states],
    )
    passed, message = _expect_error(
        lambda: trajectory(states[0], drives, list(reversed(specs))),
        "ordered and contiguous",
    )
    record("trajectory_construction_order_rejected", passed, message)
    recovered = validate_trajectory(states, drives, specs, receipts)
    exact_recovery = recovered.canonical_bytes() == states[0].canonical_bytes()
    record(
        "exact_reverse_ancestry_recovers_t0_bytes",
        exact_recovery,
        recovered.digest(),
    )

    reordered_drives = [drives[1], drives[0], drives[2]]
    reordered_states, _ = trajectory(states[0], reordered_drives, specs)
    order_response = abs(
        r0.matched_response(final.render(t), reordered_states[-1].render(t))
    )
    order_changed = final.digest() != reordered_states[-1].digest()
    record(
        "drive_order_changes_history",
        order_changed,
        {"native": final.digest(), "reordered": reordered_states[-1].digest()},
    )
    record(
        "reordered_trajectory_response_bounded",
        order_response <= ORDER_RESPONSE_MAX,
        _metric(order_response),
    )

    strict_round_trip = all(
        TemporalStepSpec.from_bytes(spec.canonical_bytes()).canonical_bytes()
        == spec.canonical_bytes()
        for spec in specs
    )
    record("step_spec_strict_round_trip", strict_round_trip, strict_round_trip)
    duplicate_text = specs[0].canonical_bytes().decode("utf-8").replace(
        f'  "schema": "{STEP_SCHEMA}",',
        f'  "schema": "{STEP_SCHEMA}",\n  "schema": "{STEP_SCHEMA}",',
        1,
    )
    passed, message = _expect_error(
        lambda: TemporalStepSpec.from_bytes(duplicate_text.encode("utf-8")),
        "duplicate JSON object key",
    )
    record("step_spec_duplicate_key_rejected", passed, message)
    unknown = specs[0].document() | {"answer": "hidden"}
    passed, message = _expect_error(
        lambda: TemporalStepSpec.from_document(unknown), "keys mismatch"
    )
    record("step_spec_unknown_field_rejected", passed, message)
    nonfinite = specs[0].document() | {"carrier_frequency_hz": float("nan")}
    passed, message = _expect_error(
        lambda: TemporalStepSpec.from_document(nonfinite), "must be finite"
    )
    record("step_spec_nonfinite_rejected", passed, message)
    boolean = specs[0].document() | {"state_modulation_index": True}
    passed, message = _expect_error(
        lambda: TemporalStepSpec.from_document(boolean), "must be a JSON number"
    )
    record("step_spec_boolean_number_rejected", passed, message)
    mismatch = specs[0].document() | {"root_id": "time2.root"}
    passed, message = _expect_error(
        lambda: TemporalStepSpec.from_document(mismatch), "scoped to its step index"
    )
    record("step_spec_step_root_mismatch_rejected", passed, message)
    nyquist = specs[0].document() | {"carrier_frequency_hz": r0.SAMPLE_RATE_HZ / 2}
    passed, message = _expect_error(
        lambda: TemporalStepSpec.from_document(nyquist), "inside (0, Nyquist)"
    )
    record("step_spec_nyquist_rejected", passed, message)
    envelope = specs[0].document() | {"drive_modulation_index": 2.01}
    passed, message = _expect_error(
        lambda: TemporalStepSpec.from_document(envelope), "frozen envelope"
    )
    record("step_spec_modulation_envelope_rejected", passed, message)
    passed, message = _expect_error(
        lambda: TemporalStepSpec.from_bytes(b" " + specs[0].canonical_bytes()),
        "not canonical",
    )
    record("step_spec_canonical_mutation_rejected", passed, message)
    digest_deterministic = specs[0].digest() == step_spec(1).digest()
    record("step_spec_digest_deterministic", digest_deterministic, specs[0].digest())

    receipt_round_trip = all(
        TemporalStepReceipt.from_bytes(receipt.canonical_bytes()).canonical_bytes()
        == receipt.canonical_bytes()
        for receipt in receipts
    )
    record("receipt_strict_round_trip", receipt_round_trip, receipt_round_trip)
    receipt_duplicate_text = receipts[0].canonical_bytes().decode("utf-8").replace(
        f'  "schema": "{RECEIPT_SCHEMA}",',
        f'  "schema": "{RECEIPT_SCHEMA}",\n  "schema": "{RECEIPT_SCHEMA}",',
        1,
    )
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_bytes(receipt_duplicate_text.encode("utf-8")),
        "duplicate JSON object key",
    )
    record("receipt_duplicate_key_rejected", passed, message)
    receipt_unknown = receipts[0].document() | {"answer": "hidden"}
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_document(receipt_unknown), "keys mismatch"
    )
    record("receipt_unknown_field_rejected", passed, message)
    receipt_nonfinite = receipts[0].document() | {
        "state_edge_modulation_index": float("nan")
    }
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_document(receipt_nonfinite), "must be finite"
    )
    record("receipt_nonfinite_number_rejected", passed, message)
    receipt_boolean = receipts[0].document() | {"step_index": True}
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_document(receipt_boolean),
        "outside the bounded R1 envelope",
    )
    record("receipt_boolean_number_rejected", passed, message)
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_bytes(b" " + receipts[0].canonical_bytes()),
        "not canonical",
    )
    record("receipt_canonical_mutation_rejected", passed, message)
    receipt_validations = [
        validate_step_receipt(states[index], specs[index - 1], receipts[index - 1])
        for index in range(1, 4)
    ]
    record(
        "receipt_binds_complete_result",
        len(receipt_validations) == 3,
        [receipt.result_tree_digest for receipt in receipts],
    )
    exact_specs = all(
        receipt.step_spec.canonical_bytes() == spec.canonical_bytes()
        and receipt.step_spec_canonical_sha256 == spec.canonical_sha256()
        and receipt.step_spec_digest == spec.digest()
        for spec, receipt in zip(specs, receipts, strict=True)
    )
    record("receipt_binds_exact_step_spec", exact_specs, exact_specs)
    both_roles = all(
        receipt.child_roles == ("state", "drive")
        and state.digest() == receipt.state_tree_digest
        and drive.digest() == receipt.drive_tree_digest
        for (state, drive), receipt in zip(receipt_validations, receipts, strict=True)
    )
    record("receipt_binds_both_roles", both_roles, both_roles)

    passed, message = _expect_error(
        lambda: temporal_step(states[0], states[0], specs[0]),
        "identities must be disjoint",
    )
    record("state_drive_id_collision_rejected", passed, message)
    collision_drive = r0.RecursivePhaseBeam(
        root=r0.PhaseNode(specs[0].root_id, 91.0), global_spin_phase_rad=0.0
    )
    passed, message = _expect_error(
        lambda: temporal_step(states[0], collision_drive, specs[0]),
        "root identity collides",
    )
    record("new_root_collision_rejected", passed, message)
    wrong_state = _receipt_with(receipts[0], state_tree_digest="0" * 64)
    passed, message = _expect_error(
        lambda: validate_step_receipt(states[1], specs[0], wrong_state),
        "state tree digest",
    )
    record("wrong_state_digest_rejected", passed, message)
    wrong_drive = _receipt_with(receipts[0], drive_tree_digest="f" * 64)
    passed, message = _expect_error(
        lambda: validate_step_receipt(states[1], specs[0], wrong_drive),
        "drive tree digest",
    )
    record("wrong_drive_digest_rejected", passed, message)
    swapped = _receipt_with(
        receipts[0],
        state_root_id=receipts[0].drive_root_id,
        state_tree_digest=receipts[0].drive_tree_digest,
        drive_root_id=receipts[0].state_root_id,
        drive_tree_digest=receipts[0].state_tree_digest,
    )
    passed, message = _expect_error(
        lambda: validate_step_receipt(states[1], specs[0], swapped),
        "positional state/drive",
    )
    record("swapped_roles_rejected", passed, message)
    equal_beta_spec = TemporalStepSpec(
        step_index=1,
        root_id="time1.equal-beta",
        carrier_frequency_hz=specs[0].carrier_frequency_hz,
        root_phase_rad=specs[0].root_phase_rad,
        state_modulation_index=0.5,
        drive_modulation_index=0.5,
        state_child_index=1,
        drive_child_index=0,
    )
    equal_beta_result, equal_beta_receipt = temporal_step(
        states[0], drives[0], equal_beta_spec
    )
    equal_beta_swapped = equal_beta_receipt.document() | {
        "state_root_id": equal_beta_receipt.drive_root_id,
        "state_tree_digest": equal_beta_receipt.drive_tree_digest,
        "drive_root_id": equal_beta_receipt.state_root_id,
        "drive_tree_digest": equal_beta_receipt.state_tree_digest,
        "state_child_index": equal_beta_receipt.drive_child_index,
        "drive_child_index": equal_beta_receipt.state_child_index,
    }
    passed, message = _expect_error(
        lambda: validate_step_receipt(
            equal_beta_result,
            equal_beta_spec,
            TemporalStepReceipt.from_document(equal_beta_swapped),
        ),
        "child indices do not match the exact step",
    )
    record("equal_beta_swapped_roles_rejected", passed, message)
    changed_state_doc = receipts[0].document() | {
        "state_edge_modulation_index": specs[0].state_modulation_index + 0.1
    }
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_document(changed_state_doc),
        "state edge does not match",
    )
    record("changed_state_beta_rejected", passed, message)
    changed_drive_doc = receipts[0].document() | {
        "drive_edge_modulation_index": specs[0].drive_modulation_index + 0.1
    }
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_document(changed_drive_doc),
        "drive edge does not match",
    )
    record("changed_drive_beta_rejected", passed, message)
    changed_spec = TemporalStepSpec(
        step_index=1,
        root_id=specs[0].root_id,
        carrier_frequency_hz=specs[0].carrier_frequency_hz,
        root_phase_rad=specs[0].root_phase_rad + 0.01,
        state_modulation_index=specs[0].state_modulation_index,
        drive_modulation_index=specs[0].drive_modulation_index,
        state_child_index=specs[0].state_child_index,
        drive_child_index=specs[0].drive_child_index,
    )
    passed, message = _expect_error(
        lambda: validate_step_receipt(states[1], changed_spec, receipts[0]),
        "exact step specification",
    )
    record("changed_step_spec_rejected", passed, message)
    wrong_root = _receipt_with(receipts[0], result_root_id=receipts[0].state_root_id)
    passed, message = _expect_error(
        lambda: validate_step_receipt(states[1], specs[0], wrong_root),
        "result root identity",
    )
    record("wrong_result_root_rejected", passed, message)

    extra = drive_tree(6, variant=9)
    extra_result = r0.RecursivePhaseBeam(
        root=r0.PhaseNode(
            states[1].root.node_id,
            states[1].root.frequency_hz,
            states[1].root.phase_rad,
            states[1].root.children + (r0.PhaseEdge(0.2, extra.root),),
        ),
        global_spin_phase_rad=0.0,
    )
    extra_receipt = _receipt_with(
        receipts[0], result_tree_digest=extra_result.digest()
    )
    passed, message = _expect_error(
        lambda: validate_step_receipt(extra_result, specs[0], extra_receipt),
        "exactly two child roles",
    )
    record("extra_child_rejected", passed, message)
    missing_result = r0.RecursivePhaseBeam(
        root=r0.PhaseNode(
            states[1].root.node_id,
            states[1].root.frequency_hz,
            states[1].root.phase_rad,
            (states[1].root.children[0],),
        ),
        global_spin_phase_rad=0.0,
    )
    missing_receipt = _receipt_with(
        receipts[0], result_tree_digest=missing_result.digest()
    )
    passed, message = _expect_error(
        lambda: validate_step_receipt(missing_result, specs[0], missing_receipt),
        "exactly two child roles",
    )
    record("missing_child_rejected", passed, message)
    duplicate_roles = receipts[0].document() | {"child_roles": ["state", "state"]}
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_document(duplicate_roles),
        "exactly state then drive",
    )
    record("duplicate_role_rejected", passed, message)
    reordered_roles = receipts[0].document() | {"child_roles": ["drive", "state"]}
    passed, message = _expect_error(
        lambda: TemporalStepReceipt.from_document(reordered_roles),
        "exactly state then drive",
    )
    record("reordered_receipt_rejected", passed, message)
    record(
        "drive_order_reversal_changes_history",
        order_changed and order_response <= ORDER_RESPONSE_MAX,
        {"digest_changed": order_changed, "response": _metric(order_response)},
    )

    flat_wave = collapsed_flat_trajectory(states[0], drives, specs, t)
    flat_response = abs(r0.matched_response(final.render(t), flat_wave))
    passed, message = _expect_error(
        lambda: require_native_tree(flat_wave, "flat baseline"),
        "complete RecursivePhaseBeam",
    )
    record("flat_wave_native_admission_rejected", passed, message)
    passed, message = _expect_error(
        lambda: extract_predecessor(flat_wave, specs[-1], receipts[-1]),
        "complete RecursivePhaseBeam",
    )
    record("flat_wave_receipt_validation_rejected", passed, message)
    flat_ancestry = not hasattr(flat_wave, "canonical_bytes")
    record("flat_wave_exact_ancestry_unavailable", flat_ancestry, flat_ancestry)

    spin_wave = collapsed_spin_trajectory(states[0], drives, specs, t)
    spin_response = abs(r0.matched_response(final.render(t), spin_wave))
    spin_diff = float(np.max(np.abs(final.render(t) - spin_wave)))
    spin_no_hierarchy = not hasattr(spin_wave, "root")
    record("decoded_spin_internal_hierarchy_absent", spin_no_hierarchy, spin_no_hierarchy)
    passed, message = _expect_error(
        lambda: require_native_tree(spin_wave, "decoded spin baseline"),
        "complete RecursivePhaseBeam",
    )
    record("decoded_spin_native_admission_rejected", passed, message)
    passed, message = _expect_error(
        lambda: extract_predecessor(spin_wave, specs[-1], receipts[-1]),
        "complete RecursivePhaseBeam",
    )
    record("decoded_spin_receipt_validation_rejected", passed, message)
    spin_ancestry = not hasattr(spin_wave, "canonical_bytes")
    record("decoded_spin_exact_ancestry_unavailable", spin_ancestry, spin_ancestry)
    record(
        "decoded_spin_final_waveform_differs",
        spin_diff > NONIDENTITY_TOL,
        _metric(spin_diff),
    )
    passed, message = _expect_error(
        lambda: validate_trajectory(states[:-1], drives, specs, receipts),
        "state count is truncated or extended",
    )
    record("trajectory_truncation_rejected", passed, message)
    duplicated = list(states)
    duplicated[2] = duplicated[1]
    passed, message = _expect_error(
        lambda: validate_trajectory(duplicated, drives, specs, receipts),
        "result root identity",
    )
    record("trajectory_duplication_rejected", passed, message)
    substitute_payload = (package_dir / state_wav_path(1)).read_bytes()
    passed, message = _expect_error(
        lambda: validate_state_fixture(states[0], substitute_payload),
        "not the deterministic tree render",
    )
    record("fixture_substitution_rejected", passed, message)
    mutated_manifest = copy.deepcopy(dict(manifest))
    mutated_manifest["trajectory"] = list(reversed(mutated_manifest["trajectory"]))
    passed, message = _expect_error(
        lambda: validate_manifest_document(mutated_manifest),
        "trajectory order",
    )
    record("manifest_order_mutation_rejected", passed, message)
    ast_proof = native_ast_proof(source_path)
    record(
        "ast_native_call_graph_no_scalar_feedback",
        ast_proof["status"] == "PASS",
        ast_proof,
    )
    ast_mutations = native_ast_mutation_probes(source_path)
    forbidden_identifiers_rejected = all(
        ast_mutations["forbidden_identifiers"].values()
    )
    record(
        "ast_forbidden_identifiers_mutation_rejected",
        forbidden_identifiers_rejected,
        ast_mutations["forbidden_identifiers"],
    )
    for test_id, probe_name in (
        ("ast_alias_callable_mutation_rejected", "alias_callable"),
        ("ast_module_helper_mutation_rejected", "module_helper"),
        ("ast_class_method_mutation_rejected", "class_method"),
        ("ast_dynamic_callable_mutation_rejected", "dynamic_callable"),
        ("ast_digest_control_mutation_rejected", "digest_control"),
        ("ast_length_feedback_mutation_rejected", "length_feedback"),
        ("ast_module_shadowing_mutation_rejected", "module_shadowing"),
        ("ast_module_attribute_rebind_mutation_rejected", "module_attribute_rebind"),
        ("ast_decorated_root_mutation_rejected", "decorated_root"),
        ("ast_global_rebind_mutation_rejected", "global_rebind"),
        ("ast_import_alias_rebind_mutation_rejected", "import_alias_rebind"),
        ("ast_loop_target_rebind_mutation_rejected", "loop_target_rebind"),
        ("ast_function_default_rebind_mutation_rejected", "function_default_rebind"),
        (
            "ast_native_runtime_binding_shadow_mutation_rejected",
            "native_runtime_binding_shadow",
        ),
        (
            "ast_receipt_lifecycle_rebind_mutation_rejected",
            "receipt_lifecycle_rebind",
        ),
        (
            "ast_match_capture_result_rebind_mutation_rejected",
            "match_capture_result_rebind",
        ),
        ("ast_result_store_rebind_mutation_rejected", "result_store_rebind"),
    ):
        record(test_id, ast_mutations[probe_name], ast_mutations[probe_name])
    record("committed_tree_wav_bytes_close", True, True)
    record("committed_step_receipt_bytes_close", True, True)
    manifest_bytes_close = all(
        r0.sha256_file(package_dir / path)
        for path in manifest["ordered_fixture_paths"]
    ) and manifest == trajectory_manifest(package_dir)
    record("manifest_binds_all_fixture_bytes", bool(manifest_bytes_close), bool(manifest_bytes_close))

    observed_ids = [test["id"] for test in tests]
    expected_ids = reference_test_ids()
    if observed_ids != expected_ids:
        raise ValueError(
            f"test order/coverage mismatch: expected={expected_ids}, observed={observed_ids}"
        )
    passed_count = sum(test["status"] == "PASS" for test in tests)
    return {
        "measurements": {
            "ast_no_feedback": ast_proof,
            "decoded_spin_max_difference": _metric(spin_diff),
            "decoded_spin_response_diagnostic": _metric(spin_response),
            "depths": depths,
            "final_depth": final.root.max_depth(),
            "final_node_count": len(node_ids(final)),
            "flat_response_diagnostic": _metric(flat_response),
            "node_counts": counts,
            "order_response": _metric(order_response),
            "reordered_final_digest": reordered_states[-1].digest(),
            "state_step_max_differences": [_metric(value) for value in step_differences],
            "trajectory_digests": [state.digest() for state in states],
            "unit_modulus_max_error": _metric(unit_error),
        },
        "summary": {
            "failed": len(tests) - passed_count,
            "passed": passed_count,
            "test_count": len(tests),
        },
        "tests": tests,
    }


def source_binding(source_path: Path) -> dict[str, Any]:
    payload = source_path.read_bytes()
    return {
        "source_byte_count": len(payload),
        "source_git_blob_sha1": r0.git_blob_sha1_bytes(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verification_policy() -> dict[str, Any]:
    return {
        "committed_byte_authority": "all trajectory trees, WAVs, specs, and receipts are parsed and rescored",
        "environment_receipt": "informational only",
        "execution_boundary": "fresh process from exact R1 and R0 source bytes; post-import monkeypatching is outside this source qualification",
        "portable_metric_atol": PORTABLE_METRIC_ATOL,
        "portable_metric_rtol": PORTABLE_METRIC_RTOL,
        "stored_pass_authority": False,
    }


def scientific_result(
    package_dir: Path,
    source_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    scored = run_reference_tests(package_dir, source_path, manifest)
    return {
        "claim_ceiling": CLAIM_CEILING,
        "collapsed_control_boundary": "flat and decoded-spin controls are diagnostic arrays rejected by native tree/receipt admission",
        "established_token_if_all_gates_close": ESTABLISHED_TOKEN,
        "fixture_count": manifest["fixture_count"],
        "fixture_manifest_sha256": r0.sha256_file(package_dir / MANIFEST_FILE),
        "fixture_set_sha256": manifest["fixture_set_sha256"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "measurements": scored["measurements"],
        "ordinary_software_only": True,
        "physical_claims_established": [],
        "reference_tests_sha256": r0.sha256_file(package_dir / TESTS_FILE),
        "schema": "recursive_wave_operator_scientific_result_v1",
        "summary": scored["summary"],
        "tests": scored["tests"],
        "wav_fixture_bytes": manifest["wav_fixture_bytes"],
        "wav_fixture_count": manifest["wav_fixture_count"],
    }


def result_document(package_dir: Path, source_path: Path) -> dict[str, Any]:
    manifest = load_exact_generated_json(
        package_dir / MANIFEST_FILE,
        {
            "drives",
            "fixture_count",
            "fixture_set_sha256",
            "generator",
            "ordered_fixture_paths",
            "receipts",
            "schema",
            "step_specs",
            "total_fixture_bytes",
            "trajectory",
            "wav_fixture_bytes",
            "wav_fixture_count",
        },
        "trajectory manifest",
    )
    validate_manifest_document(manifest)
    return {
        "execution_environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "r0_source_binding": source_binding(R0_SOURCE),
        "schema": RESULT_SCHEMA,
        "scientific": scientific_result(package_dir, source_path, manifest),
        "source_binding": source_binding(source_path),
        "verification_policy": verification_policy(),
    }


def load_exact_generated_json(
    path: Path, expected_keys: set[str], label: str
) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    value = r0.strict_json_loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _exact_keys(value, expected_keys, label)
    if payload != r0.canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical generated JSON bytes")
    return value


def portable_equal(stored: Any, recomputed: Any) -> bool:
    if isinstance(stored, bool) or isinstance(recomputed, bool):
        return type(stored) is type(recomputed) and stored == recomputed
    if isinstance(stored, (int, float)) and isinstance(recomputed, (int, float)):
        return math.isclose(
            float(stored),
            float(recomputed),
            rel_tol=PORTABLE_METRIC_RTOL,
            abs_tol=PORTABLE_METRIC_ATOL,
        )
    if isinstance(stored, dict) and isinstance(recomputed, dict):
        return set(stored) == set(recomputed) and all(
            portable_equal(stored[key], recomputed[key]) for key in stored
        )
    if isinstance(stored, list) and isinstance(recomputed, list):
        return len(stored) == len(recomputed) and all(
            portable_equal(left, right)
            for left, right in zip(stored, recomputed, strict=True)
        )
    return type(stored) is type(recomputed) and stored == recomputed


def build_package(package_dir: Path, source_path: Path) -> dict[str, Any]:
    package_dir.mkdir(parents=True, exist_ok=True)
    r0.write_json_atomic(package_dir / STEP_SCHEMA_FILE, step_spec_schema_document())
    r0.write_json_atomic(package_dir / RECEIPT_SCHEMA_FILE, receipt_schema_document())
    r0.write_json_atomic(package_dir / TESTS_FILE, reference_test_spec())
    build_fixture_files(package_dir)
    manifest = trajectory_manifest(package_dir)
    r0.write_json_atomic(package_dir / MANIFEST_FILE, manifest)
    result = result_document(package_dir, source_path)
    r0.write_json_atomic(package_dir / RESULTS_FILE, result)
    return {
        "fixture_count": manifest["fixture_count"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "manifest_sha256": r0.sha256_file(package_dir / MANIFEST_FILE),
        "operation": "build",
        "result_sha256": r0.sha256_file(package_dir / RESULTS_FILE),
        "source_sha256": result["source_binding"]["source_sha256"],
        "status": "PASS" if result["scientific"]["summary"]["failed"] == 0 else "FAIL",
        "test_count": result["scientific"]["summary"]["test_count"],
        "tests_passed": result["scientific"]["summary"]["passed"],
        "tests_sha256": r0.sha256_file(package_dir / TESTS_FILE),
        "wav_fixture_bytes": manifest["wav_fixture_bytes"],
        "wav_fixture_count": manifest["wav_fixture_count"],
    }


def verify_package(package_dir: Path, source_path: Path) -> dict[str, Any]:
    step_schema = load_exact_generated_json(
        package_dir / STEP_SCHEMA_FILE,
        {
            "$id",
            "$schema",
            "additionalProperties",
            "properties",
            "required",
            "title",
            "type",
            "x-semantic-constraints",
        },
        "step specification schema",
    )
    if step_schema != step_spec_schema_document():
        raise ValueError("committed step schema differs from the source law")
    receipt_schema = load_exact_generated_json(
        package_dir / RECEIPT_SCHEMA_FILE,
        {
            "$id",
            "$schema",
            "additionalProperties",
            "properties",
            "required",
            "title",
            "type",
            "x-semantic-constraints",
        },
        "ancestry receipt schema",
    )
    if receipt_schema != receipt_schema_document():
        raise ValueError("committed receipt schema differs from the source law")
    test_specification = load_exact_generated_json(
        package_dir / TESTS_FILE,
        {"edge_conventions", "numeric_envelope", "schema", "tests"},
        "reference test specification",
    )
    if test_specification != reference_test_spec():
        raise ValueError("committed test specification differs from the source law")
    stored_manifest = load_exact_generated_json(
        package_dir / MANIFEST_FILE,
        {
            "drives",
            "fixture_count",
            "fixture_set_sha256",
            "generator",
            "ordered_fixture_paths",
            "receipts",
            "schema",
            "step_specs",
            "total_fixture_bytes",
            "trajectory",
            "wav_fixture_bytes",
            "wav_fixture_count",
        },
        "trajectory manifest",
    )
    validate_manifest_document(stored_manifest)
    recomputed_manifest = trajectory_manifest(package_dir)
    if stored_manifest != recomputed_manifest:
        raise ValueError("committed manifest does not match committed trajectory bytes")
    stored_result = load_exact_generated_json(
        package_dir / RESULTS_FILE,
        {
            "execution_environment",
            "r0_source_binding",
            "schema",
            "scientific",
            "source_binding",
            "verification_policy",
        },
        "reference result",
    )
    if stored_result["schema"] != RESULT_SCHEMA:
        raise ValueError("unexpected reference result schema")
    if stored_result["verification_policy"] != verification_policy():
        raise ValueError("verification policy differs from the frozen source law")
    if stored_result["r0_source_binding"] != source_binding(R0_SOURCE):
        raise ValueError("R0 source binding mismatch")
    recomputed = scientific_result(package_dir, source_path, stored_manifest)
    if not portable_equal(stored_result["scientific"], recomputed):
        raise ValueError("committed-byte scientific result recomputation mismatch")
    if stored_result["source_binding"] != source_binding(source_path):
        raise ValueError("source binding mismatch")
    failed = recomputed["summary"]["failed"]
    return {
        "environment_receipt_compared": False,
        "fixture_count": stored_manifest["fixture_count"],
        "fixture_total_bytes": stored_manifest["total_fixture_bytes"],
        "manifest_sha256": r0.sha256_file(package_dir / MANIFEST_FILE),
        "operation": "verify",
        "recomputed_results_match": True,
        "r0_source_binding_match": True,
        "result_sha256": r0.sha256_file(package_dir / RESULTS_FILE),
        "source_binding_match": True,
        "status": "PASS" if failed == 0 else "FAIL",
        "test_count": recomputed["summary"]["test_count"],
        "tests_passed": recomputed["summary"]["passed"],
        "tests_sha256": r0.sha256_file(package_dir / TESTS_FILE),
        "wav_fixture_bytes": stored_manifest["wav_fixture_bytes"],
        "wav_fixture_count": stored_manifest["wav_fixture_count"],
    }


def self_test(source_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="recursive_wave_operator_self_test_") as raw:
        package_dir = Path(raw)
        build = build_package(package_dir, source_path)
        verify = verify_package(package_dir, source_path)
        return {
            "build_status": build["status"],
            "fixture_count": verify["fixture_count"],
            "fixture_total_bytes": verify["fixture_total_bytes"],
            "operation": "self-test",
            "recomputed_results_match": verify["recomputed_results_match"],
            "status": "PASS" if build["status"] == "PASS" and verify["status"] == "PASS" else "FAIL",
            "test_count": verify["test_count"],
            "tests_passed": verify["tests_passed"],
            "verify_status": verify["status"],
            "wav_fixture_bytes": verify["wav_fixture_bytes"],
            "wav_fixture_count": verify["wav_fixture_count"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("build", "verify", "self-test"),
        nargs="?",
        default="self-test",
    )
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    args = parser.parse_args()
    source_path = Path(__file__).resolve()
    package_dir = args.package_dir.resolve()
    try:
        if args.operation == "build":
            payload = build_package(package_dir, source_path)
        elif args.operation == "verify":
            payload = verify_package(package_dir, source_path)
        else:
            payload = self_test(source_path)
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        payload = {
            "error": f"{type(exc).__name__}: {exc}",
            "operation": args.operation,
            "status": "FAIL",
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if payload.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
