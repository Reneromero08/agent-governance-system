#!/usr/bin/env python3
"""Deterministic R2S reference for a bounded software catalytic wave loop.

The reference borrows the established R0 complex carrier, displaces it through the
committed R1 complete-tree trajectory, applies one public query selected before the
trajectory, copies a complex relational latch outside the histories, restores the
carrier to a prospectively frozen numerical equivalence region, and then recovers the
exact committed T0 ancestry bytes.

This is an ordinary-software reference.  It establishes no byte-exact carrier claim
when hashes differ, physical carrier, Ising computation, optimization advantage,
hardware bit replacement, silicon-phononic result, or Wall crossing.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
R1_PACKAGE_DIR = PACKAGE_DIR.parent / "audio_recursive_wave_operator_v1"
R1_SOURCE = R1_PACKAGE_DIR / "recursive_wave_operator_reference.py"

_spec = importlib.util.spec_from_file_location(
    "catcas_recursive_wave_operator_r1", R1_SOURCE
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("unable to load the established R1 recursive-wave reference")
r1 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r1
_spec.loader.exec_module(r1)
r0 = r1.r0
R0_PACKAGE_DIR = r0.PACKAGE_DIR
R0_SOURCE = r1.R0_SOURCE

GENERATOR_ID = "catalytic_wave_loop_reference_v1"
CLAIM_CEILING = "SOFTWARE_CATALYTIC_WAVE_LOOP_REFERENCE_ONLY"
ESTABLISHED_TOKEN = "AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED"
NEXT_BOUNDARY = "AUDIO_RECURSIVE_CATALYTIC_ISING_V1_CONTRACT"

CONTRACT_SCHEMA = "catalytic_wave_loop_contract_v1"
LATCH_SCHEMA = "catalytic_wave_relational_latch_v1"
CLOSURE_SCHEMA = "catalytic_wave_closure_v1"
MANIFEST_SCHEMA = "catalytic_wave_loop_fixture_manifest_v1"
RESULT_SCHEMA = "catalytic_wave_loop_reference_result_v1"

SHIFT_SCHEDULE = (17, -29, 43)
TRAJECTORY_ORDER = ("T1", "T2", "T3")
OPERATOR_ORDER = ("multiply_tree_beam", "circular_roll")
INVERSE_OPERATOR_ORDER = (
    "circular_unroll",
    "multiply_conjugate_tree_beam",
)
QUERY_ID = "hierarchy_a_public_preselected"
QUERY_SELECTION_STAGE = "before_trajectory_execution"
LATCH_CREATION_STAGE = "after_complete_forward_displacement"
CARRIER_SOURCE_ID = "r0_borrowed_tape_v1"
CARRIER_FORMAT = "raw_little_endian_interleaved_complex128"
CARRIER_DTYPE = "<c16"
CARRIER_BYTE_ORDER = "little"
CARRIER_BYTE_COUNT = r0.SAMPLE_COUNT * 16
RESTORATION_METRIC = "max_abs_complex_sample_error"
CARRIER_RESTORE_TOL = 1e-12
MIN_FORWARD_DISPLACEMENT_L2 = 1.0
MIN_WRONG_RESTORE_ERROR = 0.05
MIN_QUERY_CHANGE = 1e-6
PORTABLE_METRIC_ATOL = 5e-12
PORTABLE_METRIC_RTOL = 5e-12

CONTRACT_SCHEMA_FILE = "CATALYTIC_WAVE_LOOP_CONTRACT_SCHEMA.json"
LATCH_SCHEMA_FILE = "CATALYTIC_WAVE_LOOP_LATCH_SCHEMA.json"
CLOSURE_SCHEMA_FILE = "CATALYTIC_WAVE_LOOP_CLOSURE_SCHEMA.json"
MANIFEST_FILE = "CATALYTIC_WAVE_LOOP_FIXTURE_MANIFEST.json"
TESTS_FILE = "CATALYTIC_WAVE_LOOP_REFERENCE_TESTS.json"
RESULTS_FILE = "CATALYTIC_WAVE_LOOP_REFERENCE_RESULTS.json"
FIXTURE_DIR_NAME = "fixtures"

CARRIER_BEFORE_PATH = f"{FIXTURE_DIR_NAME}/carrier_before.c128le"
CARRIER_DISPLACED_PATH = f"{FIXTURE_DIR_NAME}/carrier_displaced.c128le"
CARRIER_RESTORED_PATH = f"{FIXTURE_DIR_NAME}/carrier_restored.c128le"
CONTRACT_FIXTURE_PATH = f"{FIXTURE_DIR_NAME}/loop_contract.json"
LATCH_FIXTURE_PATH = f"{FIXTURE_DIR_NAME}/relational_latch.json"
CLOSURE_FIXTURE_PATH = f"{FIXTURE_DIR_NAME}/catalytic_closure.json"

ORDERED_FIXTURE_PATHS = (
    CARRIER_BEFORE_PATH,
    CARRIER_DISPLACED_PATH,
    CARRIER_RESTORED_PATH,
    CONTRACT_FIXTURE_PATH,
    LATCH_FIXTURE_PATH,
    CLOSURE_FIXTURE_PATH,
)

R0_EXPECTED = {
    "source_git_blob_sha1": "956adb0ae8e84c091c1dc1e3de650be374fa96d1",
    "source_byte_count": 77043,
    "source_sha256": "e5911cb868f244ac69f3f8f8c4cfa83440385347be2d4526d5f25376de736887",
    "manifest_sha256": "7112307fa4406cf4880736545a88e56c45fafc6f27cd0a6518a1b40963fb62fa",
    "fixture_set_sha256": "6afb8adb0d14ab2e5a750df519ced073475fbf1554ee8be0732a2ebde5e15925",
    "tests_sha256": "3cecfa9f0d79babc4f9d76d7b463a1b8f825e209f2af592e590c52686dc95b2c",
    "result_sha256": "46e2cc7cb72217c647f8653ebe61a0dbf2060a222de0eec6624fbb7fbcb94eab",
}
R1_EXPECTED = {
    "source_git_blob_sha1": "3685be9ae63dcd213b2155c8cd66f6f81e45c071",
    "source_byte_count": 107055,
    "source_sha256": "26b2cfaa63f5fe6bfa97f6d9f64b97d0ee944bc39ac45d406092aea257b2179e",
    "manifest_sha256": "28cbcec8997f6f5eb49dc13e6bf919342af0863a5ba6cb1a70f10dea6fcdbc4e",
    "fixture_set_sha256": "da62112c0459c49673675182e67011899d8ee1e841df3650c0c4a0aeecd137dd",
    "tests_sha256": "5bf39db581fbc4f5cc290d1ad0ba34bc87315c2d1cf4777acf12d1d8a35023b5",
    "result_sha256": "37cb46f6806555cfaec60910f9b5b92fbcac5bf1d0e976fb67e7f2d2c0ec4139",
}

FORBIDDEN_CUSTODY_FIELDS = {
    "answer",
    "candidate",
    "energy",
    "expected",
    "expected_result",
    "score",
    "spin",
    "winner",
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


def _exact_integer(value: Any, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value != expected:
        raise ValueError(f"{label} must equal {expected}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha1(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-1 digest")
    return value


def _reject_forbidden_keys(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if lowered in FORBIDDEN_CUSTODY_FIELDS:
                raise ValueError(f"{label} contains forbidden field: {key}")
            _reject_forbidden_keys(item, label)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item, label)


def _strict_document(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    return r0.strict_json_loads(text)


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    value = _strict_document(payload, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if payload != r0.canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical generated JSON bytes")
    return value


def _complex_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1 or array.shape != (r0.SAMPLE_COUNT,):
        raise ValueError("carrier must be one complex vector with exactly 6000 samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("carrier contains non-finite values")
    return np.asarray(array, dtype="<c16", order="C").tobytes(order="C")


def parse_complex_carrier_bytes(payload: bytes) -> np.ndarray:
    if len(payload) != CARRIER_BYTE_COUNT:
        raise ValueError(
            f"raw carrier byte count must equal {CARRIER_BYTE_COUNT}, got {len(payload)}"
        )
    values = np.frombuffer(payload, dtype="<c16")
    if values.shape != (r0.SAMPLE_COUNT,):
        raise ValueError("raw carrier sample count is not exact")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw carrier contains non-finite values")
    return values.astype(np.complex128, copy=True)


def carrier_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(_complex_bytes(values)).hexdigest()


def _line_ending_identity(payload: bytes) -> str:
    crlf = payload.count(b"\r\n")
    lone_lf = payload.count(b"\n") - crlf
    lone_cr = payload.count(b"\r") - crlf
    if crlf and not lone_lf and not lone_cr:
        return "CRLF"
    if lone_lf and not crlf and not lone_cr:
        return "LF"
    if not crlf and not lone_lf and not lone_cr:
        return "NONE"
    return "MIXED"


def source_binding(source_path: Path) -> dict[str, Any]:
    payload = source_path.read_bytes()
    compile(payload, str(source_path), "exec")
    return {
        "line_ending_identity": _line_ending_identity(payload),
        "python_syntax": "PASS",
        "source_byte_count": len(payload),
        "source_git_blob_sha1": r0.git_blob_sha1_bytes(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }


CONTRACT_KEYS = {
    "carrier_before_sha256",
    "carrier_byte_count",
    "carrier_byte_order",
    "carrier_dtype",
    "carrier_format",
    "carrier_sample_count",
    "carrier_source_id",
    "claim_ceiling",
    "forward_displacement_l2_min",
    "inverse_operator_order",
    "latch_creation_stage",
    "operator_order",
    "query_id",
    "query_selection_stage",
    "query_tree_canonical_sha256",
    "query_tree_digest",
    "r0_fixture_manifest_sha256",
    "r0_fixture_set_sha256",
    "r0_reference_result_sha256",
    "r0_reference_tests_sha256",
    "r0_source_byte_count",
    "r0_source_git_blob_sha1",
    "r0_source_sha256",
    "r1_fixture_manifest_sha256",
    "r1_fixture_set_sha256",
    "r1_initial_tree_canonical_sha256",
    "r1_initial_tree_digest",
    "r1_reference_result_sha256",
    "r1_reference_tests_sha256",
    "r1_source_byte_count",
    "r1_source_git_blob_sha1",
    "r1_source_sha256",
    "r1_trajectory_digests",
    "restoration_metric",
    "restoration_tolerance",
    "schema",
    "shift_schedule",
    "trajectory_order",
    "wrong_query_separation_min",
    "wrong_restoration_min",
}


@dataclass(frozen=True)
class LoopContract:
    schema: str
    r0_source_git_blob_sha1: str
    r0_source_byte_count: int
    r0_source_sha256: str
    r0_fixture_manifest_sha256: str
    r0_fixture_set_sha256: str
    r0_reference_tests_sha256: str
    r0_reference_result_sha256: str
    r1_source_git_blob_sha1: str
    r1_source_byte_count: int
    r1_source_sha256: str
    r1_fixture_manifest_sha256: str
    r1_fixture_set_sha256: str
    r1_reference_tests_sha256: str
    r1_reference_result_sha256: str
    r1_initial_tree_digest: str
    r1_initial_tree_canonical_sha256: str
    r1_trajectory_digests: tuple[str, str, str, str]
    query_id: str
    query_tree_digest: str
    query_tree_canonical_sha256: str
    query_selection_stage: str
    shift_schedule: tuple[int, int, int]
    trajectory_order: tuple[str, str, str]
    carrier_source_id: str
    carrier_format: str
    carrier_dtype: str
    carrier_byte_order: str
    carrier_sample_count: int
    carrier_byte_count: int
    carrier_before_sha256: str
    forward_displacement_l2_min: float
    restoration_metric: str
    restoration_tolerance: float
    wrong_restoration_min: float
    wrong_query_separation_min: float
    operator_order: tuple[str, str]
    inverse_operator_order: tuple[str, str]
    latch_creation_stage: str
    claim_ceiling: str

    def __post_init__(self) -> None:
        _sha1(self.r0_source_git_blob_sha1, "r0_source_git_blob_sha1")
        _sha1(self.r1_source_git_blob_sha1, "r1_source_git_blob_sha1")
        for name in (
            "r0_source_sha256",
            "r0_fixture_manifest_sha256",
            "r0_fixture_set_sha256",
            "r0_reference_tests_sha256",
            "r0_reference_result_sha256",
            "r1_source_sha256",
            "r1_fixture_manifest_sha256",
            "r1_fixture_set_sha256",
            "r1_reference_tests_sha256",
            "r1_reference_result_sha256",
            "r1_initial_tree_digest",
            "r1_initial_tree_canonical_sha256",
            "query_tree_digest",
            "query_tree_canonical_sha256",
            "carrier_before_sha256",
        ):
            _sha256(getattr(self, name), name)
        if len(self.r1_trajectory_digests) != 4:
            raise ValueError("r1_trajectory_digests must contain exactly T0 through T3")
        for digest in self.r1_trajectory_digests:
            _sha256(digest, "r1_trajectory_digest")
        _exact_integer(self.r0_source_byte_count, R0_EXPECTED["source_byte_count"], "r0_source_byte_count")
        _exact_integer(self.r1_source_byte_count, R1_EXPECTED["source_byte_count"], "r1_source_byte_count")
        _exact_integer(self.carrier_sample_count, r0.SAMPLE_COUNT, "carrier_sample_count")
        _exact_integer(self.carrier_byte_count, CARRIER_BYTE_COUNT, "carrier_byte_count")
        for name in (
            "forward_displacement_l2_min",
            "restoration_tolerance",
            "wrong_restoration_min",
            "wrong_query_separation_min",
        ):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))

    def document(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "r1_trajectory_digests",
            "shift_schedule",
            "trajectory_order",
            "operator_order",
            "inverse_operator_order",
        ):
            value[key] = list(value[key])
        return value

    def canonical_bytes(self) -> bytes:
        return r0.canonical_json_bytes(self.document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_document(cls, document: Any) -> "LoopContract":
        if not isinstance(document, dict):
            raise ValueError("loop contract must be an object")
        _reject_forbidden_keys(document, "loop contract")
        _exact_keys(document, CONTRACT_KEYS, "loop contract")
        value = dict(document)
        for key in (
            "r1_trajectory_digests",
            "shift_schedule",
            "trajectory_order",
            "operator_order",
            "inverse_operator_order",
        ):
            if not isinstance(value[key], list):
                raise ValueError(f"{key} must be an array")
            value[key] = tuple(value[key])
        contract = cls(**value)
        validate_contract_semantics(contract)
        return contract

    @classmethod
    def from_bytes(cls, payload: bytes, *, require_canonical: bool = True) -> "LoopContract":
        contract = cls.from_document(_strict_document(payload, "loop contract"))
        if require_canonical and payload != contract.canonical_bytes():
            raise ValueError("loop contract is valid but not canonical")
        return contract


LATCH_KEYS = {
    "carrier_before_sha256",
    "carrier_displaced_sha256",
    "final_tree_digest",
    "forward_displacement_l2",
    "latch_stage",
    "query_id",
    "query_tree_canonical_sha256",
    "query_tree_digest",
    "response_imag",
    "response_real",
    "schema",
}


@dataclass(frozen=True)
class RelationalLatch:
    schema: str
    query_id: str
    query_tree_digest: str
    query_tree_canonical_sha256: str
    final_tree_digest: str
    carrier_before_sha256: str
    carrier_displaced_sha256: str
    forward_displacement_l2: float
    response_real: float
    response_imag: float
    latch_stage: str

    def __post_init__(self) -> None:
        if self.schema != LATCH_SCHEMA:
            raise ValueError("relational latch schema mismatch")
        if self.query_id != QUERY_ID:
            raise ValueError("relational latch query_id mismatch")
        if self.latch_stage != LATCH_CREATION_STAGE:
            raise ValueError("relational latch creation stage mismatch")
        for name in (
            "query_tree_digest",
            "query_tree_canonical_sha256",
            "final_tree_digest",
            "carrier_before_sha256",
            "carrier_displaced_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in ("forward_displacement_l2", "response_real", "response_imag"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))
        if self.forward_displacement_l2 <= 0.0:
            raise ValueError("forward_displacement_l2 must be positive")

    def response(self) -> complex:
        return complex(self.response_real, self.response_imag)

    def document(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        return r0.canonical_json_bytes(self.document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_document(cls, document: Any) -> "RelationalLatch":
        if not isinstance(document, dict):
            raise ValueError("relational latch must be an object")
        _reject_forbidden_keys(document, "relational latch")
        _exact_keys(document, LATCH_KEYS, "relational latch")
        return cls(**document)

    @classmethod
    def from_bytes(cls, payload: bytes, *, require_canonical: bool = True) -> "RelationalLatch":
        latch = cls.from_document(_strict_document(payload, "relational latch"))
        if require_canonical and payload != latch.canonical_bytes():
            raise ValueError("relational latch is valid but not canonical")
        return latch


CLOSURE_KEYS = {
    "ancestry_byte_exact",
    "carrier_before_sha256",
    "carrier_byte_exact",
    "carrier_equivalence_restored",
    "carrier_restored_sha256",
    "claim_ceiling",
    "forward_displacement_l2",
    "latch",
    "latch_sha256",
    "recovered_initial_tree_canonical_sha256",
    "recovered_initial_tree_digest",
    "restore_max_error",
    "schema",
}


@dataclass(frozen=True)
class CatalyticClosure:
    schema: str
    latch: RelationalLatch
    latch_sha256: str
    carrier_before_sha256: str
    carrier_restored_sha256: str
    carrier_byte_exact: bool
    carrier_equivalence_restored: bool
    recovered_initial_tree_digest: str
    recovered_initial_tree_canonical_sha256: str
    ancestry_byte_exact: bool
    forward_displacement_l2: float
    restore_max_error: float
    claim_ceiling: str

    def __post_init__(self) -> None:
        if self.schema != CLOSURE_SCHEMA:
            raise ValueError("catalytic closure schema mismatch")
        if not isinstance(self.latch, RelationalLatch):
            raise ValueError("closure latch must be an exact RelationalLatch")
        for name in (
            "latch_sha256",
            "carrier_before_sha256",
            "carrier_restored_sha256",
            "recovered_initial_tree_digest",
            "recovered_initial_tree_canonical_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in (
            "carrier_byte_exact",
            "carrier_equivalence_restored",
            "ancestry_byte_exact",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("forward_displacement_l2", "restore_max_error"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))
        if self.forward_displacement_l2 <= 0.0 or self.restore_max_error < 0.0:
            raise ValueError("closure displacement/error envelope is invalid")
        if self.forward_displacement_l2 < MIN_FORWARD_DISPLACEMENT_L2:
            raise ValueError("closure forward displacement is below the frozen minimum")
        if self.restore_max_error > CARRIER_RESTORE_TOL:
            raise ValueError("closure restoration error exceeds the frozen tolerance")
        if self.latch_sha256 != self.latch.digest():
            raise ValueError("closure latch digest mismatch")
        if self.carrier_equivalence_restored is not True:
            raise ValueError("closure equivalence-restoration verdict must be true")
        if self.ancestry_byte_exact is not True:
            raise ValueError("closure ancestry_byte_exact verdict must be true")
        if self.claim_ceiling != CLAIM_CEILING:
            raise ValueError("closure claim ceiling mismatch")

    def document(self) -> dict[str, Any]:
        value = asdict(self)
        value["latch"] = self.latch.document()
        return value

    def canonical_bytes(self) -> bytes:
        return r0.canonical_json_bytes(self.document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_document(cls, document: Any) -> "CatalyticClosure":
        if not isinstance(document, dict):
            raise ValueError("catalytic closure must be an object")
        _reject_forbidden_keys(document, "catalytic closure")
        _exact_keys(document, CLOSURE_KEYS, "catalytic closure")
        value = dict(document)
        value["latch"] = RelationalLatch.from_document(value["latch"])
        return cls(**value)

    @classmethod
    def from_bytes(cls, payload: bytes, *, require_canonical: bool = True) -> "CatalyticClosure":
        closure = cls.from_document(_strict_document(payload, "catalytic closure"))
        if require_canonical and payload != closure.canonical_bytes():
            raise ValueError("catalytic closure is valid but not canonical")
        return closure


@dataclass(frozen=True)
class LifecycleExecution:
    contract: LoopContract
    carrier_before: np.ndarray
    carrier_displaced: np.ndarray
    carrier_restored: np.ndarray
    states: tuple[Any, ...]
    drives: tuple[Any, ...]
    specs: tuple[Any, ...]
    receipts: tuple[Any, ...]
    latch: RelationalLatch
    closure: CatalyticClosure
    exact_query_response: complex
    wrong_query_response: complex


def _parent_packet() -> dict[str, Any]:
    return r1.load_committed_packet(R1_PACKAGE_DIR)


def _expected_contract_document() -> dict[str, Any]:
    packet = _parent_packet()
    states = packet["states"]
    query = r0.hierarchy_a()
    carrier_before = r0.borrowed_tape(r0.sample_times())
    return {
        "carrier_before_sha256": carrier_sha256(carrier_before),
        "carrier_byte_count": CARRIER_BYTE_COUNT,
        "carrier_byte_order": CARRIER_BYTE_ORDER,
        "carrier_dtype": CARRIER_DTYPE,
        "carrier_format": CARRIER_FORMAT,
        "carrier_sample_count": r0.SAMPLE_COUNT,
        "carrier_source_id": CARRIER_SOURCE_ID,
        "claim_ceiling": CLAIM_CEILING,
        "forward_displacement_l2_min": MIN_FORWARD_DISPLACEMENT_L2,
        "inverse_operator_order": list(INVERSE_OPERATOR_ORDER),
        "latch_creation_stage": LATCH_CREATION_STAGE,
        "operator_order": list(OPERATOR_ORDER),
        "query_id": QUERY_ID,
        "query_selection_stage": QUERY_SELECTION_STAGE,
        "query_tree_canonical_sha256": hashlib.sha256(query.canonical_bytes()).hexdigest(),
        "query_tree_digest": query.digest(),
        "r0_fixture_manifest_sha256": R0_EXPECTED["manifest_sha256"],
        "r0_fixture_set_sha256": R0_EXPECTED["fixture_set_sha256"],
        "r0_reference_result_sha256": R0_EXPECTED["result_sha256"],
        "r0_reference_tests_sha256": R0_EXPECTED["tests_sha256"],
        "r0_source_byte_count": R0_EXPECTED["source_byte_count"],
        "r0_source_git_blob_sha1": R0_EXPECTED["source_git_blob_sha1"],
        "r0_source_sha256": R0_EXPECTED["source_sha256"],
        "r1_fixture_manifest_sha256": R1_EXPECTED["manifest_sha256"],
        "r1_fixture_set_sha256": R1_EXPECTED["fixture_set_sha256"],
        "r1_initial_tree_canonical_sha256": hashlib.sha256(states[0].canonical_bytes()).hexdigest(),
        "r1_initial_tree_digest": states[0].digest(),
        "r1_reference_result_sha256": R1_EXPECTED["result_sha256"],
        "r1_reference_tests_sha256": R1_EXPECTED["tests_sha256"],
        "r1_source_byte_count": R1_EXPECTED["source_byte_count"],
        "r1_source_git_blob_sha1": R1_EXPECTED["source_git_blob_sha1"],
        "r1_source_sha256": R1_EXPECTED["source_sha256"],
        "r1_trajectory_digests": [state.digest() for state in states],
        "restoration_metric": RESTORATION_METRIC,
        "restoration_tolerance": CARRIER_RESTORE_TOL,
        "schema": CONTRACT_SCHEMA,
        "shift_schedule": list(SHIFT_SCHEDULE),
        "trajectory_order": list(TRAJECTORY_ORDER),
        "wrong_query_separation_min": MIN_QUERY_CHANGE,
        "wrong_restoration_min": MIN_WRONG_RESTORE_ERROR,
    }


def reference_contract() -> LoopContract:
    return LoopContract.from_document(_expected_contract_document())


def validate_contract_semantics(contract: LoopContract) -> None:
    observed = contract.document()
    expected = _expected_contract_document()
    if observed["shift_schedule"] != expected["shift_schedule"]:
        raise ValueError("loop contract shift schedule must be exactly [17, -29, 43]")
    if len(observed["shift_schedule"]) != 3:
        raise ValueError("loop contract shift schedule length must be exactly three")
    if observed["query_id"] != expected["query_id"] or observed["query_tree_digest"] != expected["query_tree_digest"]:
        raise ValueError("loop contract query identity does not match the frozen public query")
    if observed["r0_source_sha256"] != expected["r0_source_sha256"] or observed["r0_source_git_blob_sha1"] != expected["r0_source_git_blob_sha1"]:
        raise ValueError("loop contract R0 source identity mismatch")
    if observed["r1_source_sha256"] != expected["r1_source_sha256"] or observed["r1_source_git_blob_sha1"] != expected["r1_source_git_blob_sha1"]:
        raise ValueError("loop contract R1 source identity mismatch")
    if observed != expected:
        differing = sorted(key for key in expected if observed.get(key) != expected[key])
        raise ValueError(f"loop contract differs from frozen semantics: {differing}")


def _native_carrier_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != (r0.SAMPLE_COUNT,):
        raise ValueError("carrier shape does not match the frozen sample count")
    if not np.all(np.isfinite(array)):
        raise ValueError("carrier contains non-finite values")
    return array


def _native_shift_sequence(shifts: Sequence[int]) -> tuple[int, ...]:
    if len(shifts) != len(SHIFT_SCHEDULE):
        raise ValueError("carrier operation requires exactly three shifts")
    normalized: list[int] = []
    for shift in shifts:
        if isinstance(shift, bool) or not isinstance(shift, int):
            raise ValueError("carrier shifts must be integers")
        normalized.append(shift)
    return tuple(normalized)


def forward_carrier_step(carrier: np.ndarray, state: Any, shift_samples: int) -> np.ndarray:
    """Multiply by one complete-tree beam and then circularly transport."""

    r1.require_native_tree(state, "carrier state")
    if isinstance(shift_samples, bool) or not isinstance(shift_samples, int):
        raise ValueError("shift_samples must be an integer")
    carrier_array = _native_carrier_array(carrier)
    beam = state.render(r0.sample_times())
    return np.roll(r0.apply_phase_operator(carrier_array, beam), shift_samples)


def inverse_carrier_step(carrier: np.ndarray, state: Any, shift_samples: int) -> np.ndarray:
    """Circularly unroll and multiply by the conjugate complete-tree beam."""

    r1.require_native_tree(state, "inverse carrier state")
    if isinstance(shift_samples, bool) or not isinstance(shift_samples, int):
        raise ValueError("shift_samples must be an integer")
    carrier_array = _native_carrier_array(carrier)
    unshifted = np.roll(carrier_array, -shift_samples)
    return r0.uncompute_phase_operator(unshifted, state.render(r0.sample_times()))


def forward_carrier(carrier: np.ndarray, trajectory_states: Sequence[Any], shifts: Sequence[int] = SHIFT_SCHEDULE) -> np.ndarray:
    if not trajectory_states or len(trajectory_states) != len(shifts):
        raise ValueError("forward carrier requires equal nonempty state and shift sequences")
    normalized_shifts = _native_shift_sequence(shifts)
    current = _native_carrier_array(carrier)
    current = current.copy()
    for state, shift in zip(trajectory_states, normalized_shifts, strict=True):
        current = forward_carrier_step(current, state, shift)
    return current


def restore_carrier(displaced: np.ndarray, trajectory_states: Sequence[Any], shifts: Sequence[int] = SHIFT_SCHEDULE) -> np.ndarray:
    if not trajectory_states or len(trajectory_states) != len(shifts):
        raise ValueError("restore carrier requires equal nonempty state and shift sequences")
    normalized_shifts = _native_shift_sequence(shifts)
    current = _native_carrier_array(displaced)
    current = current.copy()
    for state, shift in reversed(list(zip(trajectory_states, normalized_shifts, strict=True))):
        current = inverse_carrier_step(current, state, shift)
    return current


def validate_borrowed_carrier(carrier: np.ndarray, contract: LoopContract) -> None:
    array = _native_carrier_array(carrier)
    if np.any(np.abs(array) == 0.0):
        raise ValueError("borrowed carrier contains a zero-amplitude coordinate")
    if carrier_sha256(array) != contract.carrier_before_sha256:
        raise ValueError("borrowed carrier identity does not match the loop contract")


def validate_forward_displacement(carrier_before: np.ndarray, carrier_displaced: np.ndarray, contract: LoopContract) -> float:
    displacement = float(np.linalg.norm(_native_carrier_array(carrier_displaced) - _native_carrier_array(carrier_before)))
    if displacement < contract.forward_displacement_l2_min:
        raise ValueError("forward carrier displacement is below the frozen minimum")
    return displacement


def validate_query_preselection(contract: LoopContract, query_tree: Any, selection_stage: str) -> None:
    r1.require_native_tree(query_tree, "query tree")
    if selection_stage != QUERY_SELECTION_STAGE:
        raise ValueError("query must be selected before trajectory execution")
    if contract.query_selection_stage != QUERY_SELECTION_STAGE:
        raise ValueError("loop contract query-selection stage is not prospective")
    if query_tree.digest() != contract.query_tree_digest or hashlib.sha256(query_tree.canonical_bytes()).hexdigest() != contract.query_tree_canonical_sha256:
        raise ValueError("selected query does not match the frozen public query identity")


def carrier_query_response(carrier_before: np.ndarray, carrier_displaced: np.ndarray, query_tree: Any, shifts: Sequence[int] = SHIFT_SCHEDULE) -> complex:
    r1.require_native_tree(query_tree, "query tree")
    before = _native_carrier_array(carrier_before)
    displaced = _native_carrier_array(carrier_displaced)
    normalized_shifts = _native_shift_sequence(shifts)
    query_carrier = r0.apply_phase_operator(before, query_tree.render(r0.sample_times()))
    for shift in normalized_shifts:
        query_carrier = np.roll(query_carrier, shift)
    return r0.matched_response(displaced, query_carrier)


def latch_relational_observable(carrier_before: np.ndarray, carrier_displaced: np.ndarray, final_tree: Any, query_tree: Any, contract: LoopContract, forward_displacement: float) -> RelationalLatch:
    response = carrier_query_response(carrier_before, carrier_displaced, query_tree, contract.shift_schedule)
    return RelationalLatch(
        schema=LATCH_SCHEMA,
        query_id=contract.query_id,
        query_tree_digest=query_tree.digest(),
        query_tree_canonical_sha256=hashlib.sha256(query_tree.canonical_bytes()).hexdigest(),
        final_tree_digest=final_tree.digest(),
        carrier_before_sha256=carrier_sha256(carrier_before),
        carrier_displaced_sha256=carrier_sha256(carrier_displaced),
        forward_displacement_l2=_metric(forward_displacement),
        response_real=_metric(response.real),
        response_imag=_metric(response.imag),
        latch_stage=LATCH_CREATION_STAGE,
    )


def validate_latch_semantics(latch: RelationalLatch, contract: LoopContract, final_tree: Any, carrier_before: np.ndarray, carrier_displaced: np.ndarray, query_tree: Any) -> None:
    expected = latch_relational_observable(
        carrier_before,
        carrier_displaced,
        final_tree,
        query_tree,
        contract,
        float(np.linalg.norm(carrier_displaced - carrier_before)),
    )
    if latch.canonical_bytes() != expected.canonical_bytes():
        raise ValueError("relational latch does not match the committed displaced carrier")


def validate_closure_semantics(closure: CatalyticClosure, lifecycle: "LifecycleExecution") -> None:
    if closure.latch.canonical_bytes() != lifecycle.latch.canonical_bytes():
        raise ValueError("closure latch differs from the external latch")
    if closure.latch_sha256 != lifecycle.latch.digest():
        raise ValueError("closure latch digest mismatch")
    before_sha = carrier_sha256(lifecycle.carrier_before)
    restored_sha = carrier_sha256(lifecycle.carrier_restored)
    if closure.carrier_before_sha256 != before_sha or closure.carrier_restored_sha256 != restored_sha:
        raise ValueError("closure carrier hash binding mismatch")
    if closure.carrier_byte_exact != (before_sha == restored_sha):
        raise ValueError("closure byte-exact diagnostic is not honest")
    if closure.carrier_equivalence_restored != (closure.restore_max_error <= CARRIER_RESTORE_TOL):
        raise ValueError("closure equivalence-restoration verdict mismatch")
    expected_initial_digest = lifecycle.states[0].digest()
    expected_initial_canonical_sha256 = hashlib.sha256(
        lifecycle.states[0].canonical_bytes()
    ).hexdigest()
    if (
        closure.recovered_initial_tree_digest != expected_initial_digest
        or closure.recovered_initial_tree_canonical_sha256
        != expected_initial_canonical_sha256
        or closure.ancestry_byte_exact is not True
    ):
        raise ValueError("closure recovered T0 ancestry identity mismatch")
    expected_displacement = _metric(
        np.linalg.norm(lifecycle.carrier_displaced - lifecycle.carrier_before)
    )
    if closure.forward_displacement_l2 != expected_displacement:
        raise ValueError("closure forward displacement does not match committed carriers")
    expected_restore_error = _metric(
        np.max(np.abs(lifecycle.carrier_restored - lifecycle.carrier_before))
    )
    if closure.restore_max_error != expected_restore_error:
        raise ValueError("closure restoration error does not match committed carriers")
    if closure.claim_ceiling != CLAIM_CEILING:
        raise ValueError("closure claim ceiling mismatch")


def execute_reference_lifecycle(contract: LoopContract | None = None) -> LifecycleExecution:
    loop_contract = reference_contract() if contract is None else contract
    validate_contract_semantics(loop_contract)
    query = r0.hierarchy_a()
    validate_query_preselection(loop_contract, query, QUERY_SELECTION_STAGE)
    packet = _parent_packet()
    states = tuple(packet["states"])
    drives = tuple(packet["drives"])
    specs = tuple(packet["specs"])
    receipts = tuple(packet["receipts"])
    carrier_before = r0.borrowed_tape(r0.sample_times())
    validate_borrowed_carrier(carrier_before, loop_contract)
    carrier_displaced = forward_carrier(carrier_before, states[1:], loop_contract.shift_schedule)
    forward_displacement = validate_forward_displacement(carrier_before, carrier_displaced, loop_contract)
    latch = latch_relational_observable(carrier_before, carrier_displaced, states[-1], query, loop_contract, forward_displacement)
    latch_before_carrier_restore = latch.canonical_bytes()
    carrier_restored = restore_carrier(carrier_displaced, states[1:], loop_contract.shift_schedule)
    restore_error = float(np.max(np.abs(carrier_restored - carrier_before)))
    if restore_error > loop_contract.restoration_tolerance:
        raise ValueError("correct inverse did not enter the frozen carrier equivalence region")
    if latch.canonical_bytes() != latch_before_carrier_restore:
        raise ValueError("external latch changed during carrier restoration")
    recovered = r1.validate_trajectory(states, drives, specs, receipts)
    ancestry_byte_exact = recovered.canonical_bytes() == states[0].canonical_bytes()
    if not ancestry_byte_exact:
        raise ValueError("exact T0 ancestry bytes were not recovered")
    if latch.canonical_bytes() != latch_before_carrier_restore:
        raise ValueError("external latch changed during ancestry restoration")
    closure = CatalyticClosure(
        schema=CLOSURE_SCHEMA,
        latch=latch,
        latch_sha256=latch.digest(),
        carrier_before_sha256=carrier_sha256(carrier_before),
        carrier_restored_sha256=carrier_sha256(carrier_restored),
        carrier_byte_exact=carrier_sha256(carrier_before) == carrier_sha256(carrier_restored),
        carrier_equivalence_restored=restore_error <= loop_contract.restoration_tolerance,
        recovered_initial_tree_digest=recovered.digest(),
        recovered_initial_tree_canonical_sha256=hashlib.sha256(recovered.canonical_bytes()).hexdigest(),
        ancestry_byte_exact=ancestry_byte_exact,
        forward_displacement_l2=_metric(forward_displacement),
        restore_max_error=_metric(restore_error),
        claim_ceiling=CLAIM_CEILING,
    )
    wrong_query = r0.hierarchy_b()
    wrong_query_response = carrier_query_response(carrier_before, carrier_displaced, wrong_query, loop_contract.shift_schedule)
    return LifecycleExecution(
        contract=loop_contract,
        carrier_before=carrier_before,
        carrier_displaced=carrier_displaced,
        carrier_restored=carrier_restored,
        states=states,
        drives=drives,
        specs=specs,
        receipts=receipts,
        latch=latch,
        closure=closure,
        exact_query_response=latch.response(),
        wrong_query_response=wrong_query_response,
    )


def parent_custody() -> dict[str, Any]:
    r0_source = r1.source_binding(R0_SOURCE)
    r1_source = r1.source_binding(R1_SOURCE)
    r0_manifest = _load_canonical_json(R0_PACKAGE_DIR / r0.MANIFEST_FILE, "R0 manifest")
    r1_manifest = _load_canonical_json(R1_PACKAGE_DIR / r1.MANIFEST_FILE, "R1 manifest")
    r0_observed = {
        **r0_source,
        "manifest_sha256": r0.sha256_file(R0_PACKAGE_DIR / r0.MANIFEST_FILE),
        "fixture_set_sha256": r0_manifest["fixture_set_sha256"],
        "tests_sha256": r0.sha256_file(R0_PACKAGE_DIR / r0.TESTS_FILE),
        "result_sha256": r0.sha256_file(R0_PACKAGE_DIR / r0.RESULTS_FILE),
    }
    r1_observed = {
        **r1_source,
        "manifest_sha256": r0.sha256_file(R1_PACKAGE_DIR / r1.MANIFEST_FILE),
        "fixture_set_sha256": r1_manifest["fixture_set_sha256"],
        "tests_sha256": r0.sha256_file(R1_PACKAGE_DIR / r1.TESTS_FILE),
        "result_sha256": r0.sha256_file(R1_PACKAGE_DIR / r1.RESULTS_FILE),
    }
    return {
        "r0": r0_observed,
        "r0_exact": r0_observed == R0_EXPECTED,
        "r1": r1_observed,
        "r1_exact": r1_observed == R1_EXPECTED,
    }


def parent_verification() -> dict[str, Any]:
    r0_verify = r0.verify_package(R0_PACKAGE_DIR, R0_SOURCE)
    r1_verify = r1.verify_package(R1_PACKAGE_DIR, R1_SOURCE)
    custody = parent_custody()
    return {
        "custody": custody,
        "r0_verify": r0_verify,
        "r1_verify": r1_verify,
        "status": "PASS" if custody["r0_exact"] and custody["r1_exact"] and r0_verify["status"] == "PASS" and r1_verify["status"] == "PASS" else "FAIL",
    }


def _wrong_restore_measurements(lifecycle: LifecycleExecution) -> dict[str, float]:
    states = lifecycle.states
    before = lifecycle.carrier_before
    displaced = lifecycle.carrier_displaced
    shifts = lifecycle.contract.shift_schedule

    forward_order = displaced.copy()
    for state, shift in zip(states[1:], shifts, strict=True):
        forward_order = inverse_carrier_step(forward_order, state, shift)

    reordered_states, _ = r1.trajectory(
        states[0],
        [r1.drive_tree(2), r1.drive_tree(1), r1.drive_tree(3)],
        [r1.step_spec(index) for index in range(1, 4)],
    )
    wrong_trajectory = restore_carrier(displaced, reordered_states[1:], shifts)

    omitted = displaced.copy()
    for state, shift in reversed(list(zip(states[2:], shifts[1:], strict=True))):
        omitted = inverse_carrier_step(omitted, state, shift)

    duplicated = displaced.copy()
    reverse_pairs = list(reversed(list(zip(states[1:], shifts, strict=True))))
    duplicate_pairs = reverse_pairs[:2] + [reverse_pairs[1]] + reverse_pairs[2:]
    for state, shift in duplicate_pairs:
        duplicated = inverse_carrier_step(duplicated, state, shift)

    wrong_sign = displaced.copy()
    for state, shift in reversed(list(zip(states[1:], shifts, strict=True))):
        wrong_sign = inverse_carrier_step(wrong_sign, state, -shift)

    wrong_magnitude = restore_carrier(displaced, states[1:], (18, -29, 43))

    wrong_state = displaced.copy()
    wrong_states = (states[1], r1.drive_tree(2, variant=7), states[3])
    for state, shift in reversed(list(zip(wrong_states, shifts, strict=True))):
        wrong_state = inverse_carrier_step(wrong_state, state, shift)

    def error(values: np.ndarray) -> float:
        return float(np.max(np.abs(values - before)))

    return {
        "duplicated_inverse_error": _metric(error(duplicated)),
        "forward_order_inverse_error": _metric(error(forward_order)),
        "no_restore_error": _metric(error(displaced)),
        "omitted_inverse_error": _metric(error(omitted)),
        "wrong_shift_magnitude_error": _metric(error(wrong_magnitude)),
        "wrong_shift_sign_error": _metric(error(wrong_sign)),
        "wrong_state_one_leg_error": _metric(error(wrong_state)),
        "wrong_trajectory_inverse_error": _metric(error(wrong_trajectory)),
    }


def loop_contract_schema_document() -> dict[str, Any]:
    digest = {"pattern": "^[0-9a-f]{64}$", "type": "string"}
    sha1 = {"pattern": "^[0-9a-f]{40}$", "type": "string"}
    properties: dict[str, Any] = {key: {"type": "string"} for key in CONTRACT_KEYS}
    for key in CONTRACT_KEYS:
        if key.endswith("sha256") or key.endswith("digest"):
            properties[key] = digest
    properties["r0_source_git_blob_sha1"] = sha1
    properties["r1_source_git_blob_sha1"] = sha1
    properties["r0_source_byte_count"] = {"const": R0_EXPECTED["source_byte_count"], "type": "integer"}
    properties["r1_source_byte_count"] = {"const": R1_EXPECTED["source_byte_count"], "type": "integer"}
    properties["carrier_sample_count"] = {"const": r0.SAMPLE_COUNT, "type": "integer"}
    properties["carrier_byte_count"] = {"const": CARRIER_BYTE_COUNT, "type": "integer"}
    for key in ("forward_displacement_l2_min", "restoration_tolerance", "wrong_restoration_min", "wrong_query_separation_min"):
        properties[key] = {"type": "number"}
    properties["r1_trajectory_digests"] = {"items": digest, "maxItems": 4, "minItems": 4, "type": "array"}
    properties["shift_schedule"] = {"const": list(SHIFT_SCHEDULE), "maxItems": 3, "minItems": 3, "type": "array"}
    properties["trajectory_order"] = {"const": list(TRAJECTORY_ORDER), "type": "array"}
    properties["operator_order"] = {"const": list(OPERATOR_ORDER), "type": "array"}
    properties["inverse_operator_order"] = {"const": list(INVERSE_OPERATOR_ORDER), "type": "array"}
    properties["schema"] = {"const": CONTRACT_SCHEMA, "type": "string"}
    return {
        "$id": "urn:cat-cas:catalytic-wave-loop-contract:v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(CONTRACT_KEYS),
        "title": "CAT_CAS catalytic wave loop contract v1",
        "type": "object",
        "x-semantic-constraints": [
            "strict canonical JSON with duplicate-key rejection",
            "exact R0 and R1 source and result identities",
            "query identity selected before trajectory execution",
            "shift schedule is exactly [17,-29,43]",
            "answer, expected-result, spin, energy, winner, candidate, and score fields are forbidden",
        ],
    }


def latch_schema_document() -> dict[str, Any]:
    digest = {"pattern": "^[0-9a-f]{64}$", "type": "string"}
    return {
        "$id": "urn:cat-cas:catalytic-wave-relational-latch:v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "carrier_before_sha256": digest,
            "carrier_displaced_sha256": digest,
            "final_tree_digest": digest,
            "forward_displacement_l2": {"minimum": MIN_FORWARD_DISPLACEMENT_L2, "type": "number"},
            "latch_stage": {"const": LATCH_CREATION_STAGE, "type": "string"},
            "query_id": {"const": QUERY_ID, "type": "string"},
            "query_tree_canonical_sha256": digest,
            "query_tree_digest": digest,
            "response_imag": {"type": "number"},
            "response_real": {"type": "number"},
            "schema": {"const": LATCH_SCHEMA, "type": "string"},
        },
        "required": sorted(LATCH_KEYS),
        "title": "CAT_CAS external relational latch v1",
        "type": "object",
        "x-semantic-constraints": [
            "created only after complete forward displacement",
            "immutable through carrier and ancestry restoration",
            "contains no expected value or selection instruction",
        ],
    }


def closure_schema_document() -> dict[str, Any]:
    digest = {"pattern": "^[0-9a-f]{64}$", "type": "string"}
    return {
        "$id": "urn:cat-cas:catalytic-wave-closure:v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "ancestry_byte_exact": {"const": True, "type": "boolean"},
            "carrier_before_sha256": digest,
            "carrier_byte_exact": {"type": "boolean"},
            "carrier_equivalence_restored": {"const": True, "type": "boolean"},
            "carrier_restored_sha256": digest,
            "claim_ceiling": {"const": CLAIM_CEILING, "type": "string"},
            "forward_displacement_l2": {"minimum": MIN_FORWARD_DISPLACEMENT_L2, "type": "number"},
            "latch": latch_schema_document(),
            "latch_sha256": digest,
            "recovered_initial_tree_canonical_sha256": digest,
            "recovered_initial_tree_digest": digest,
            "restore_max_error": {"maximum": CARRIER_RESTORE_TOL, "minimum": 0.0, "type": "number"},
            "schema": {"const": CLOSURE_SCHEMA, "type": "string"},
        },
        "required": sorted(CLOSURE_KEYS),
        "title": "CAT_CAS catalytic closure v1",
        "type": "object",
        "x-semantic-constraints": [
            "closure retains no states, drives, specs, receipts, or displaced carrier",
            "carrier restoration means numerical equivalence unless hashes match",
            "T0 ancestry recovery remains byte exact",
        ],
    }


def reference_test_ids() -> list[str]:
    return [
        "parent_r0_source_and_fixture_custody_exact",
        "parent_r0_committed_result_recomputed",
        "parent_r1_source_and_fixture_custody_exact",
        "parent_r1_committed_result_recomputed",
        "loop_contract_strict_round_trip",
        "loop_contract_duplicate_key_rejected",
        "loop_contract_unknown_field_rejected",
        "loop_contract_nonfinite_rejected",
        "loop_contract_boolean_number_rejected",
        "loop_contract_wrong_schedule_length_rejected",
        "loop_contract_wrong_shifts_rejected",
        "loop_contract_wrong_query_rejected",
        "loop_contract_wrong_source_identity_rejected",
        "loop_contract_forbidden_field_rejected",
        "latch_strict_round_trip",
        "latch_duplicate_key_rejected",
        "latch_unknown_field_rejected",
        "latch_nonfinite_rejected",
        "latch_boolean_number_rejected",
        "latch_forbidden_field_rejected",
        "latch_wrong_schema_rejected",
        "latch_wrong_query_id_rejected",
        "latch_wrong_stage_rejected",
        "closure_strict_round_trip",
        "closure_duplicate_key_rejected",
        "closure_unknown_field_rejected",
        "closure_nonfinite_rejected",
        "closure_boolean_rejected",
        "closure_forbidden_history_field_rejected",
        "closure_wrong_schema_rejected",
        "closure_false_ancestry_rejected",
        "closure_wrong_recovered_digest_rejected",
        "closure_wrong_recovered_canonical_sha_rejected",
        "closure_wrong_displacement_rejected",
        "closure_wrong_restore_error_rejected",
        "closure_wrong_equivalence_verdict_rejected",
        "closure_wrong_byte_exact_verdict_rejected",
        "closure_wrong_latch_digest_rejected",
        "closure_wrong_carrier_before_hash_rejected",
        "closure_wrong_carrier_restored_hash_rejected",
        "closure_wrong_claim_ceiling_rejected",
        "carrier_before_strict_binary_parse",
        "carrier_displaced_strict_binary_parse",
        "carrier_restored_strict_binary_parse",
        "carrier_truncation_rejected",
        "carrier_extension_rejected",
        "carrier_nonfinite_rejected",
        "carrier_endian_mutation_rejected",
        "zero_amplitude_carrier_coordinate_rejected",
        "forward_carrier_displacement_nonzero",
        "correct_carrier_equivalence_restoration",
        "carrier_byte_exact_diagnostic_honest",
        "exact_t0_ancestry_bytes_recovered",
        "external_latch_survives_both_restoration_stages",
        "closure_excludes_reversed_histories",
        "query_frozen_before_trajectory_execution",
        "wrong_query_changes_complex_latch",
        "forward_order_inverse_rejected",
        "wrong_trajectory_inverse_rejected",
        "omitted_inverse_step_rejected",
        "duplicated_inverse_step_rejected",
        "wrong_shift_sign_rejected",
        "wrong_shift_magnitude_rejected",
        "wrong_state_one_inverse_leg_rejected",
        "no_restore_control_rejected",
        "identity_untouched_carrier_rejected",
        "query_selected_after_result_rejected",
        "mutated_displaced_carrier_fixture_rejected",
        "mutated_restored_carrier_fixture_rejected",
        "mutated_latch_fixture_rejected",
        "mutated_loop_contract_rejected",
        "manifest_role_substitution_rejected",
        "committed_fixture_bytes_match_reference_lifecycle",
        "manifest_binds_all_fixture_bytes",
        "ast_native_call_graph_no_scalar_feedback",
        "ast_query_and_latch_downstream_of_forward_displacement",
        "ast_lifecycle_order_and_external_closure",
        "ast_no_feedback_mutation_probes_rejected",
    ]


def reference_test_spec() -> dict[str, Any]:
    return {
        "edge_conventions": {
            "binary_carrier": "raw little-endian <c16, 6000 samples, no header or metadata",
            "canonical_json": "UTF-8, sorted keys, indent=2, newline terminated, duplicate keys rejected",
            "carrier_restoration": "max absolute complex sample error <= 1e-12; byte identity is diagnostic only",
            "committed_byte_scoring": "all carriers and JSON receipts are parsed from committed fixture bytes",
            "query": "hierarchy A selected before trajectory; both A and B complex responses reported",
            "t0_restoration": "exact canonical committed T0 bytes",
        },
        "numeric_envelope": {
            "forward_displacement_l2_min": MIN_FORWARD_DISPLACEMENT_L2,
            "portable_metric_atol": PORTABLE_METRIC_ATOL,
            "portable_metric_rtol": PORTABLE_METRIC_RTOL,
            "restoration_tolerance": CARRIER_RESTORE_TOL,
            "wrong_query_separation_min": MIN_QUERY_CHANGE,
            "wrong_restoration_error_min": MIN_WRONG_RESTORE_ERROR,
        },
        "schema": "catalytic_wave_loop_reference_tests_v1",
        "tests": [{"id": test_id} for test_id in reference_test_ids()],
    }


def _carrier_metrics(values: np.ndarray) -> dict[str, float]:
    array = _native_carrier_array(values)
    magnitude = np.abs(array)
    return {
        "complex_energy": _metric(np.sum(magnitude**2)),
        "complex_min_abs": _metric(np.min(magnitude)),
        "complex_norm": _metric(np.linalg.norm(array)),
        "complex_peak": _metric(np.max(magnitude)),
        "complex_rms": _metric(np.sqrt(np.mean(magnitude**2))),
    }


FIXTURE_RECORD_KEYS = {
    "byte_count",
    "byte_order",
    "complex_energy",
    "complex_min_abs",
    "complex_norm",
    "complex_peak",
    "complex_rms",
    "dtype",
    "media_type",
    "path",
    "role",
    "sample_count",
    "schema",
    "sha256",
}
MANIFEST_CONTEXT_KEYS = {
    "carrier_before_sha256",
    "carrier_byte_exact",
    "carrier_equivalence_restored",
    "carrier_source_id",
    "forward_displacement_l2",
    "query_id",
    "query_tree_canonical_sha256",
    "query_tree_digest",
    "r0_source_sha256",
    "r1_trajectory_digests",
    "restoration_metric",
    "restoration_tolerance",
    "restore_max_error",
    "shift_schedule",
}
MANIFEST_KEYS = {
    "binary_carrier_bytes",
    "binary_carrier_count",
    "carrier_context",
    "fixture_count",
    "fixture_set_sha256",
    "fixtures",
    "generator",
    "ordered_fixture_paths",
    "schema",
    "total_fixture_bytes",
}


def build_fixture_files(package_dir: Path) -> None:
    contract = reference_contract()
    lifecycle = execute_reference_lifecycle(contract)
    payloads = {
        CARRIER_BEFORE_PATH: _complex_bytes(lifecycle.carrier_before),
        CARRIER_DISPLACED_PATH: _complex_bytes(lifecycle.carrier_displaced),
        CARRIER_RESTORED_PATH: _complex_bytes(lifecycle.carrier_restored),
        CONTRACT_FIXTURE_PATH: contract.canonical_bytes(),
        LATCH_FIXTURE_PATH: lifecycle.latch.canonical_bytes(),
        CLOSURE_FIXTURE_PATH: lifecycle.closure.canonical_bytes(),
    }
    for relative, payload in payloads.items():
        r0.write_bytes_atomic(package_dir / relative, payload)
    fixture_root = package_dir / FIXTURE_DIR_NAME
    observed = {
        path.relative_to(package_dir).as_posix()
        for path in fixture_root.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(observed - set(ORDERED_FIXTURE_PATHS))
    if unexpected:
        raise ValueError(f"unexpected R2S fixture files: {unexpected}")


def load_committed_fixture_packet(package_dir: Path) -> dict[str, Any]:
    contract = LoopContract.from_bytes((package_dir / CONTRACT_FIXTURE_PATH).read_bytes())
    before_payload = (package_dir / CARRIER_BEFORE_PATH).read_bytes()
    displaced_payload = (package_dir / CARRIER_DISPLACED_PATH).read_bytes()
    restored_payload = (package_dir / CARRIER_RESTORED_PATH).read_bytes()
    before = parse_complex_carrier_bytes(before_payload)
    displaced = parse_complex_carrier_bytes(displaced_payload)
    restored = parse_complex_carrier_bytes(restored_payload)
    latch = RelationalLatch.from_bytes((package_dir / LATCH_FIXTURE_PATH).read_bytes())
    closure = CatalyticClosure.from_bytes((package_dir / CLOSURE_FIXTURE_PATH).read_bytes())
    expected = execute_reference_lifecycle(contract)
    expected_payloads = {
        CARRIER_BEFORE_PATH: _complex_bytes(expected.carrier_before),
        CARRIER_DISPLACED_PATH: _complex_bytes(expected.carrier_displaced),
        CARRIER_RESTORED_PATH: _complex_bytes(expected.carrier_restored),
        CONTRACT_FIXTURE_PATH: expected.contract.canonical_bytes(),
        LATCH_FIXTURE_PATH: expected.latch.canonical_bytes(),
        CLOSURE_FIXTURE_PATH: expected.closure.canonical_bytes(),
    }
    for relative, expected_payload in expected_payloads.items():
        if (package_dir / relative).read_bytes() != expected_payload:
            raise ValueError(f"committed fixture bytes differ from reference role: {relative}")
    validate_latch_semantics(latch, contract, expected.states[-1], before, displaced, r0.hierarchy_a())
    proxy = LifecycleExecution(
        contract=contract,
        carrier_before=before,
        carrier_displaced=displaced,
        carrier_restored=restored,
        states=expected.states,
        drives=expected.drives,
        specs=expected.specs,
        receipts=expected.receipts,
        latch=latch,
        closure=closure,
        exact_query_response=latch.response(),
        wrong_query_response=expected.wrong_query_response,
    )
    validate_closure_semantics(closure, proxy)
    return {
        "before": before,
        "closure": closure,
        "contract": contract,
        "displaced": displaced,
        "expected": expected,
        "latch": latch,
        "restored": restored,
    }


def _fixture_record(path: Path, relative: str, role: str, schema: str | None, carrier: np.ndarray | None) -> dict[str, Any]:
    payload = path.read_bytes()
    if carrier is None:
        metrics: dict[str, Any] = {
            "complex_energy": None,
            "complex_min_abs": None,
            "complex_norm": None,
            "complex_peak": None,
            "complex_rms": None,
        }
        dtype = "canonical_json"
        byte_order = "utf-8"
        media_type = "application/json"
        sample_count = None
    else:
        metrics = _carrier_metrics(carrier)
        dtype = CARRIER_DTYPE
        byte_order = CARRIER_BYTE_ORDER
        media_type = "application/vnd.cat-cas.raw-complex128"
        sample_count = r0.SAMPLE_COUNT
    return {
        "byte_count": len(payload),
        "byte_order": byte_order,
        **metrics,
        "dtype": dtype,
        "media_type": media_type,
        "path": relative,
        "role": role,
        "sample_count": sample_count,
        "schema": schema,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def fixture_manifest(package_dir: Path) -> dict[str, Any]:
    packet = load_committed_fixture_packet(package_dir)
    lifecycle = packet["expected"]
    role_data = {
        CARRIER_BEFORE_PATH: ("carrier_before", None, packet["before"]),
        CARRIER_DISPLACED_PATH: ("carrier_displaced", None, packet["displaced"]),
        CARRIER_RESTORED_PATH: ("carrier_restored", None, packet["restored"]),
        CONTRACT_FIXTURE_PATH: ("loop_contract", CONTRACT_SCHEMA, None),
        LATCH_FIXTURE_PATH: ("relational_latch", LATCH_SCHEMA, None),
        CLOSURE_FIXTURE_PATH: ("catalytic_closure", CLOSURE_SCHEMA, None),
    }
    records: list[dict[str, Any]] = []
    total_bytes = 0
    binary_bytes = 0
    set_hasher = hashlib.sha256()
    for relative in ORDERED_FIXTURE_PATHS:
        role, schema, carrier = role_data[relative]
        path = package_dir / relative
        record = _fixture_record(path, relative, role, schema, carrier)
        records.append(record)
        total_bytes += record["byte_count"]
        if carrier is not None:
            binary_bytes += record["byte_count"]
        set_hasher.update(relative.encode("utf-8") + b"\0")
        set_hasher.update(bytes.fromhex(record["sha256"]))
    context = {
        "carrier_before_sha256": lifecycle.closure.carrier_before_sha256,
        "carrier_byte_exact": lifecycle.closure.carrier_byte_exact,
        "carrier_equivalence_restored": lifecycle.closure.carrier_equivalence_restored,
        "carrier_source_id": CARRIER_SOURCE_ID,
        "forward_displacement_l2": lifecycle.closure.forward_displacement_l2,
        "query_id": QUERY_ID,
        "query_tree_canonical_sha256": lifecycle.contract.query_tree_canonical_sha256,
        "query_tree_digest": lifecycle.contract.query_tree_digest,
        "r0_source_sha256": R0_EXPECTED["source_sha256"],
        "r1_trajectory_digests": list(lifecycle.contract.r1_trajectory_digests),
        "restoration_metric": RESTORATION_METRIC,
        "restoration_tolerance": CARRIER_RESTORE_TOL,
        "restore_max_error": lifecycle.closure.restore_max_error,
        "shift_schedule": list(SHIFT_SCHEDULE),
    }
    manifest = {
        "binary_carrier_bytes": binary_bytes,
        "binary_carrier_count": 3,
        "carrier_context": context,
        "fixture_count": len(records),
        "fixture_set_sha256": set_hasher.hexdigest(),
        "fixtures": records,
        "generator": GENERATOR_ID,
        "ordered_fixture_paths": list(ORDERED_FIXTURE_PATHS),
        "schema": MANIFEST_SCHEMA,
        "total_fixture_bytes": total_bytes,
    }
    validate_manifest_document(manifest)
    return manifest


def validate_manifest_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("fixture manifest must be an object")
    _exact_keys(value, MANIFEST_KEYS, "fixture manifest")
    if value["schema"] != MANIFEST_SCHEMA or value["generator"] != GENERATOR_ID:
        raise ValueError("fixture manifest schema or generator mismatch")
    if value["ordered_fixture_paths"] != list(ORDERED_FIXTURE_PATHS):
        raise ValueError("fixture manifest path order or role order is invalid")
    if value["fixture_count"] != 6 or value["binary_carrier_count"] != 3:
        raise ValueError("fixture manifest count is not exact")
    if not isinstance(value["fixtures"], list) or len(value["fixtures"]) != 6:
        raise ValueError("fixture manifest record count is not exact")
    roles = ["carrier_before", "carrier_displaced", "carrier_restored", "loop_contract", "relational_latch", "catalytic_closure"]
    for index, record in enumerate(value["fixtures"]):
        if not isinstance(record, dict):
            raise ValueError("fixture manifest record must be an object")
        _exact_keys(record, FIXTURE_RECORD_KEYS, f"fixture[{index}]")
        if record["path"] != ORDERED_FIXTURE_PATHS[index] or record["role"] != roles[index]:
            raise ValueError("fixture manifest role substitution detected")
        _sha256(record["sha256"], f"fixture[{index}].sha256")
    if not isinstance(value["carrier_context"], dict):
        raise ValueError("carrier context must be an object")
    _exact_keys(value["carrier_context"], MANIFEST_CONTEXT_KEYS, "carrier context")
    if value["carrier_context"]["shift_schedule"] != list(SHIFT_SCHEDULE):
        raise ValueError("carrier context shift schedule mismatch")
    _sha256(value["fixture_set_sha256"], "fixture_set_sha256")
    return value


def validate_fixture_payload(record: Mapping[str, Any], payload: bytes) -> None:
    if len(payload) != record["byte_count"]:
        raise ValueError(f"fixture byte-count mismatch for role {record['role']}")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"fixture SHA-256 mismatch for role {record['role']}")


def _expect_error(action: Callable[[], Any], expected_text: str) -> tuple[bool, str]:
    try:
        action()
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        message = str(exc)
        return expected_text in message, message
    return False, "NO_ERROR"


NATIVE_AST_ROOTS = (
    "forward_carrier_step",
    "inverse_carrier_step",
    "forward_carrier",
    "restore_carrier",
)
NATIVE_AST_FORBIDDEN = {
    "answer",
    "argmax",
    "argmin",
    "candidate",
    "energy",
    "expected",
    "ising",
    "latch",
    "matched_response",
    "response",
    "score",
    "spin",
    "verdict",
    "winner",
}
NATIVE_AST_ALLOWED_NAME_CALLS = {
    "ValueError",
    "isinstance",
    "len",
    "list",
    "reversed",
    "tuple",
    "zip",
}
NATIVE_AST_ALLOWED_ATTRIBUTE_CALLS = {
    "current.copy",
    "np.all",
    "np.asarray",
    "np.isfinite",
    "np.roll",
    "normalized.append",
    "r0.apply_phase_operator",
    "r0.sample_times",
    "r0.uncompute_phase_operator",
    "r1.require_native_tree",
    "state.render",
}
NATIVE_AST_FUNCTION_SHA256 = {
    "_native_carrier_array": "744c67b22d8ed48c1fd3f6c02f01ab6aefb0247545de79400d4d90a3c0391f91",
    "_native_shift_sequence": "35d61bb488296820df6e586f2f5e3c12abacbc37e72c55604675499121230d08",
    "forward_carrier": "bcc88616df5528f915852b7fbae7a8217abfaf3bd13f0722941f8f3119a7e461",
    "forward_carrier_step": "5f6f303db9334e28e6ab7bebbacfd7558e8499dd249d1595313b2cef3ae10765",
    "inverse_carrier_step": "858355daad96c754729d0fa6edffc1f60395088bacd20c4da60e06b8a7e6faab",
    "restore_carrier": "a154d1d8eaefb44568f1719f7527c21396408a121552ec36ab83843bf2679e2f",
}
LIFECYCLE_AST_SHA256 = "878a9687a6090ebaa9a8747a451df2be1c675ca2a1b58fd34fbe4c9022785785"
NATIVE_AST_EXPECTED_BINDING_COUNTS = {
    "forward_carrier_step": {"beam": 1, "carrier_array": 1},
    "inverse_carrier_step": {"carrier_array": 1, "unshifted": 1},
    "forward_carrier": {"current": 3, "normalized_shifts": 1},
    "restore_carrier": {"current": 3, "normalized_shifts": 1},
}
LIFECYCLE_AST_EXPECTED_BINDING_COUNTS = {
    "query": 1,
    "carrier_displaced": 1,
    "latch": 1,
    "carrier_restored": 1,
    "recovered": 1,
    "closure": 1,
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


def _assigned_name(statement: ast.stmt) -> str | None:
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


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


def _ast_shape_sha256(node: ast.AST) -> str:
    return hashlib.sha256(
        ast.dump(node, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _module_load_skeleton(module: ast.Module) -> ast.Module:
    skeleton = copy.deepcopy(module)

    class StripDeferredBodies(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.decorator_list = [self.visit(item) for item in node.decorator_list]
            node.args = self.visit(node.args)
            node.returns = self.visit(node.returns) if node.returns is not None else None
            node.body = [ast.Pass()]
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            node.decorator_list = [self.visit(item) for item in node.decorator_list]
            node.args = self.visit(node.args)
            node.returns = self.visit(node.returns) if node.returns is not None else None
            node.body = [ast.Pass()]
            return node

        def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
            node.args = self.visit(node.args)
            node.body = ast.Constant(value=None)
            return node

    stripped = StripDeferredBodies().visit(skeleton)
    if not isinstance(stripped, ast.Module):
        raise ValueError("module skeleton transformation failed")
    return stripped


def _module_binding_inventory(skeleton: ast.Module) -> list[str]:
    inventory: list[str] = []
    for node in ast.walk(skeleton):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            inventory.append(f"name:{type(node.ctx).__name__}:{node.id}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inventory.append(f"function:{node.name}")
        elif isinstance(node, ast.ClassDef):
            inventory.append(f"class:{node.name}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                inventory.append(f"import:{bound}")
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            inventory.append(f"except:{node.name}")
        elif isinstance(node, ast.MatchAs) and node.name is not None:
            inventory.append(f"match_as:{node.name}")
        elif isinstance(node, ast.MatchStar) and node.name is not None:
            inventory.append(f"match_star:{node.name}")
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            inventory.append(f"match_rest:{node.rest}")
    return sorted(inventory)


def _ast_dump_set_sha256(nodes: Sequence[ast.AST]) -> str:
    dumps = sorted(ast.dump(node, include_attributes=False) for node in nodes)
    return hashlib.sha256(
        r0.canonical_json_bytes(dumps, pretty=False)
    ).hexdigest()


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _ast_dotted_name(node.func)
    return None


def _lifecycle_latch_taint_violations(
    lifecycle: ast.FunctionDef,
) -> list[str]:
    tainted = {"latch"}
    violations: set[str] = set()
    native_calls = set(NATIVE_AST_ROOTS)
    for statement in lifecycle.body:
        for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
            called = _call_name(call)
            if called not in native_calls:
                continue
            argument_names = {
                child.id
                for argument in (*call.args, *(item.value for item in call.keywords))
                for child in ast.walk(argument)
                if isinstance(child, ast.Name)
            }
            contaminated = sorted(argument_names & tainted)
            if contaminated:
                violations.add(
                    f"{called}:latch_tainted_input:{','.join(contaminated)}"
                )

        assignments: list[tuple[ast.AST, ast.AST | None]] = []
        if isinstance(statement, ast.Assign):
            assignments.extend((target, statement.value) for target in statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            assignments.append((statement.target, statement.value))
        elif isinstance(statement, ast.AugAssign):
            assignments.append((statement.target, statement.value))
        for target, value in assignments:
            if value is None:
                continue
            dependencies = {
                child.id
                for child in ast.walk(value)
                if isinstance(child, ast.Name)
            }
            target_names = set(_binding_names(target))
            if dependencies & tainted or target_names & tainted:
                tainted.update(target_names)
    return sorted(violations)


def native_ast_proof_text(source_text: str) -> dict[str, Any]:
    module = ast.parse(source_text)
    expected_module_skeleton_sha256 = "12d4b9aa30f135af3b5e31f86bb8c6f04eb46f7b86c1a688f0565ebf21ce77a3"
    expected_module_runtime_calls_sha256 = "8d7e495615e46a2d691b37dcbaeb97f4b1d8fd67292e5e595947dd87adb5e6bc"
    expected_module_binding_inventory_sha256 = "9c7f749c0aa78f2d9c1037a2316601f692bf1597990500c61d4eb6b9667580cf"
    expected_module_indirect_writes_sha256 = "883f925d48b3e732632137741c8fb7336f6a3b56c7e5d6b7e5f09b7299106749"
    skeleton = _module_load_skeleton(module)
    observed_module_skeleton_sha256 = _ast_shape_sha256(skeleton)
    module_runtime_calls = [
        node for node in ast.walk(skeleton) if isinstance(node, ast.Call)
    ]
    observed_module_runtime_calls_sha256 = _ast_dump_set_sha256(
        module_runtime_calls
    )
    module_binding_inventory = _module_binding_inventory(skeleton)
    observed_module_binding_inventory_sha256 = hashlib.sha256(
        r0.canonical_json_bytes(module_binding_inventory, pretty=False)
    ).hexdigest()
    module_indirect_writes = [
        node
        for node in ast.walk(skeleton)
        if isinstance(node, (ast.Attribute, ast.Subscript))
        and isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del))
    ]
    observed_module_indirect_writes_sha256 = _ast_dump_set_sha256(
        module_indirect_writes
    )
    module_load_violations: set[str] = set()
    if observed_module_skeleton_sha256 != expected_module_skeleton_sha256:
        module_load_violations.add(
            f"module_skeleton:{observed_module_skeleton_sha256}"
        )
    if observed_module_runtime_calls_sha256 != expected_module_runtime_calls_sha256:
        module_load_violations.add(
            f"module_runtime_calls:{observed_module_runtime_calls_sha256}"
        )
    if (
        observed_module_binding_inventory_sha256
        != expected_module_binding_inventory_sha256
    ):
        module_load_violations.add(
            f"module_binding_inventory:{observed_module_binding_inventory_sha256}"
        )
    if (
        observed_module_indirect_writes_sha256
        != expected_module_indirect_writes_sha256
    ):
        module_load_violations.add(
            f"module_indirect_writes:{observed_module_indirect_writes_sha256}"
        )
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    reachable: set[str] = set()
    frontier = list(NATIVE_AST_ROOTS)
    unresolved: set[str] = set()
    forbidden_hits: set[str] = set()
    forbidden_constructs: set[str] = set()
    function_shape_violations: set[str] = set()
    runtime_binding_violations: set[str] = set()
    protected_runtime_names = set(functions) | {"np", "r0", "r1"}
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        if name not in functions:
            raise ValueError(f"native AST root/helper missing: {name}")
        reachable.add(name)
        function = functions[name]
        expected_shape = NATIVE_AST_FUNCTION_SHA256.get(name)
        observed_shape = _ast_shape_sha256(function)
        if expected_shape is None or observed_shape != expected_shape:
            function_shape_violations.add(
                f"{name}:shape_sha256:{observed_shape}"
            )
        expected_counts = NATIVE_AST_EXPECTED_BINDING_COUNTS.get(name, {})
        observed_bindings = _binding_names(function)
        for binding, expected_count in expected_counts.items():
            observed_count = observed_bindings.count(binding)
            if observed_count != expected_count:
                function_shape_violations.add(
                    f"{name}:binding_count:{binding}:{observed_count}"
                )
        parameter_names = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        for binding_name in observed_bindings:
            if binding_name in parameter_names or binding_name in protected_runtime_names:
                runtime_binding_violations.add(
                    f"{name}:runtime_binding_write:{binding_name}"
                )
        for node in ast.walk(function):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda, ast.ClassDef)):
                forbidden_constructs.add(f"{name}:{type(node).__name__}")
            if isinstance(node, ast.FunctionDef) and node is not function:
                forbidden_constructs.add(f"{name}:nested_function:{node.name}")
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in protected_runtime_names:
                runtime_binding_violations.add(f"{name}:runtime_binding_write:{node.id}")
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
                forbidden_constructs.add(f"{name}:indirect_runtime_write")
            candidate: str | None = None
            if isinstance(node, ast.Name):
                candidate = node.id
            elif isinstance(node, ast.Attribute):
                candidate = node.attr
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = node.value
            if candidate is not None:
                lowered = candidate.lower()
                forbidden_hits.update(term for term in NATIVE_AST_FORBIDDEN if term in lowered)
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called = node.func.id
                if called in functions:
                    frontier.append(called)
                elif called not in NATIVE_AST_ALLOWED_NAME_CALLS:
                    unresolved.add(f"{name}:dynamic_or_unapproved:{called}")
            elif isinstance(node.func, ast.Attribute):
                called = _ast_dotted_name(node.func)
                if called not in NATIVE_AST_ALLOWED_ATTRIBUTE_CALLS:
                    unresolved.add(f"{name}:unapproved_attribute:{called}")
            else:
                unresolved.add(f"{name}:dynamic_callable:{type(node.func).__name__}")

    module_binding_violations: set[str] = set()
    for statement in module.body:
        if isinstance(statement, ast.FunctionDef):
            if statement.name in reachable and statement.decorator_list:
                module_binding_violations.add(f"decorated_native_function:{statement.name}")
            continue
        if isinstance(statement, ast.ClassDef):
            continue
        for binding in _binding_names(statement):
            if binding in reachable:
                module_binding_violations.add(f"module_rebind:{binding}")

    lifecycle = functions.get("execute_reference_lifecycle")
    if lifecycle is None:
        raise ValueError("execute_reference_lifecycle is missing")
    lifecycle_shape_violations: set[str] = set()
    observed_lifecycle_shape = _ast_shape_sha256(lifecycle)
    if observed_lifecycle_shape != LIFECYCLE_AST_SHA256:
        lifecycle_shape_violations.add(
            f"execute_reference_lifecycle:shape_sha256:{observed_lifecycle_shape}"
        )
    lifecycle_bindings = _binding_names(lifecycle)
    for binding, expected_count in LIFECYCLE_AST_EXPECTED_BINDING_COUNTS.items():
        observed_count = lifecycle_bindings.count(binding)
        if observed_count != expected_count:
            lifecycle_shape_violations.add(
                f"execute_reference_lifecycle:binding_count:{binding}:{observed_count}"
            )

    positions: dict[str, int] = {}
    query_validation_position: int | None = None
    for index, statement in enumerate(lifecycle.body):
        name = _assigned_name(statement)
        if name is not None and name in {
            "query",
            "carrier_displaced",
            "latch",
            "carrier_restored",
            "recovered",
            "closure",
        }:
            positions[name] = index
        for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
            if _call_name(call) == "validate_query_preselection":
                query_validation_position = index
    expected_order = ["query", "carrier_displaced", "latch", "carrier_restored", "recovered", "closure"]
    lifecycle_order = all(key in positions for key in expected_order) and [positions[key] for key in expected_order] == sorted(positions[key] for key in expected_order)
    query_preselection_order = (
        "query" in positions
        and "carrier_displaced" in positions
        and query_validation_position is not None
        and positions["query"] < query_validation_position < positions["carrier_displaced"]
    )
    latch_feedback_calls: list[str] = []
    for node in ast.walk(lifecycle):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.id if isinstance(node.func, ast.Name) else _ast_dotted_name(node.func)
        if called in {"forward_carrier", "restore_carrier", "forward_carrier_step", "inverse_carrier_step"}:
            if any(isinstance(child, ast.Name) and child.id == "latch" for arg in node.args for child in ast.walk(arg)):
                latch_feedback_calls.append(str(called))
    query_downstream = (
        "carrier_displaced" in positions
        and "latch" in positions
        and positions["latch"] > positions["carrier_displaced"]
    )
    latch_taint_violations = _lifecycle_latch_taint_violations(lifecycle)
    status = "PASS" if not unresolved and not forbidden_hits and not forbidden_constructs and not function_shape_violations and not runtime_binding_violations and not module_binding_violations and not module_load_violations and not lifecycle_shape_violations and lifecycle_order and query_preselection_order and query_downstream and not latch_feedback_calls and not latch_taint_violations else "FAIL"
    return {
        "forbidden_constructs": sorted(forbidden_constructs),
        "forbidden_hits": sorted(forbidden_hits),
        "function_shape_violations": sorted(function_shape_violations),
        "latch_feedback_calls": latch_feedback_calls,
        "latch_taint_violations": latch_taint_violations,
        "lifecycle_order": lifecycle_order,
        "lifecycle_positions": positions,
        "lifecycle_shape_violations": sorted(lifecycle_shape_violations),
        "module_binding_violations": sorted(module_binding_violations),
        "module_binding_inventory_sha256": observed_module_binding_inventory_sha256,
        "module_indirect_writes_sha256": observed_module_indirect_writes_sha256,
        "module_load_skeleton_sha256": observed_module_skeleton_sha256,
        "module_load_violations": sorted(module_load_violations),
        "module_runtime_call_count": len(module_runtime_calls),
        "module_runtime_calls_sha256": observed_module_runtime_calls_sha256,
        "native_roots": list(NATIVE_AST_ROOTS),
        "query_and_latch_downstream": query_downstream,
        "query_preselection_order": query_preselection_order,
        "query_validation_position": query_validation_position,
        "reachable_functions": sorted(reachable),
        "runtime_binding_violations": sorted(runtime_binding_violations),
        "status": status,
        "unresolved_calls": sorted(unresolved),
    }


def native_ast_proof(source_path: Path) -> dict[str, Any]:
    return native_ast_proof_text(source_path.read_text(encoding="utf-8"))


def native_ast_mutation_probes(source_path: Path) -> dict[str, bool]:
    source_text = source_path.read_text(encoding="utf-8")
    anchor = "    carrier_array = _native_carrier_array(carrier)\n    beam = state.render(r0.sample_times())\n"
    if source_text.count(anchor) != 1:
        raise ValueError("native AST mutation anchor is not unique")

    def inject(statement: str) -> str:
        return source_text.replace(anchor, anchor + statement + "\n", 1)

    probes = {
        "direct_matched_response": inject("    response = r0.matched_response(carrier_array, beam)"),
        "latch_feedback": inject("    carrier_array = carrier_array * latch.response()"),
        "score_feedback": inject("    carrier_array = carrier_array * score"),
        "spin_feedback": inject("    carrier_array = carrier_array * spin"),
        "winner_feedback": inject("    carrier_array = carrier_array * winner"),
        "dynamic_callable": inject("    operator = r0.matched_response\n    operator(carrier_array, beam)"),
        "module_rebind": source_text + "\nforward_carrier = carrier_query_response\n",
        "decorated_native": source_text.replace(
            "def forward_carrier(carrier:",
            "@staticmethod\ndef forward_carrier(carrier:",
            1,
        ),
        "tree_scalar_feedback": source_text.replace(
            "    return np.roll(r0.apply_phase_operator(carrier_array, beam), shift_samples)\n",
            "    scalar = len(state.root.children)\n"
            "    return np.roll(r0.apply_phase_operator(carrier_array, beam) * np.exp(1j * scalar), shift_samples)\n",
            1,
        ),
        "runtime_match_capture": source_text.replace(
            "    carrier_array = _native_carrier_array(carrier)\n    beam = state.render(r0.sample_times())\n",
            "    carrier_array = _native_carrier_array(carrier)\n"
            "    match {\"runtime\": r0}:\n"
            "        case {\"runtime\": r0}:\n"
            "            pass\n"
            "    beam = state.render(r0.sample_times())\n",
            1,
        ),
        "swapped_inverse_order": source_text.replace(
            "    unshifted = np.roll(carrier_array, -shift_samples)\n"
            "    return r0.uncompute_phase_operator(unshifted, state.render(r0.sample_times()))\n",
            "    uncomputed = r0.uncompute_phase_operator(carrier_array, state.render(r0.sample_times()))\n"
            "    return np.roll(uncomputed, -shift_samples)\n",
            1,
        ),
        "late_query_selection": source_text.replace(
            "    query = r0.hierarchy_a()\n"
            "    validate_query_preselection(loop_contract, query, QUERY_SELECTION_STAGE)\n",
            "",
            1,
        ).replace(
            "    forward_displacement = validate_forward_displacement(carrier_before, carrier_displaced, loop_contract)\n",
            "    forward_displacement = validate_forward_displacement(carrier_before, carrier_displaced, loop_contract)\n"
            "    query = r0.hierarchy_a()\n"
            "    validate_query_preselection(loop_contract, query, QUERY_SELECTION_STAGE)\n",
            1,
        ),
        "aliased_latch_feedback": source_text.replace(
            "    latch_before_carrier_restore = latch.canonical_bytes()\n",
            "    latch_before_carrier_restore = latch.canonical_bytes()\n"
            "    feedback = latch.response_real\n"
            "    carrier_displaced = carrier_displaced * np.exp(1j * feedback)\n",
            1,
        ),
        "module_import_alias_replacement": source_text
        + "\nfrom builtins import len as forward_carrier\n",
        "module_globals_replacement": source_text
        + "\n_original_forward = forward_carrier\n"
        + 'globals().__setitem__("forward_carrier", lambda carrier, trajectory_states, shifts=SHIFT_SCHEDULE: carrier)\n',
        "module_runtime_attribute_replacement": source_text
        + "\n_original_phase_operator = r0.apply_phase_operator\n"
        + "r0.apply_phase_operator = lambda tape, beam: tape\n",
        "module_definition_default_side_effect": source_text
        + "\ndef replacement_forward(carrier, trajectory_states, shifts=SHIFT_SCHEDULE):\n"
        + "    return carrier\n"
        + 'def install(value=globals().__setitem__("forward_carrier", replacement_forward)):\n'
        + "    return value\n",
        "module_class_body_side_effect": source_text
        + "\nclass InstallForwardReplacement:\n"
        + '    globals().__setitem__("forward_carrier", lambda carrier, trajectory_states, shifts=SHIFT_SCHEDULE: carrier)\n',
    }
    return {name: native_ast_proof_text(mutated)["status"] == "FAIL" for name, mutated in probes.items()}


def _mutated_bytes(payload: bytes) -> bytes:
    if not payload:
        raise ValueError("cannot mutate empty fixture")
    mutable = bytearray(payload)
    mutable[len(mutable) // 2] ^= 0x01
    return bytes(mutable)


def run_reference_tests(package_dir: Path, source_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    packet = load_committed_fixture_packet(package_dir)
    lifecycle: LifecycleExecution = packet["expected"]
    contract = packet["contract"]
    latch = packet["latch"]
    closure = packet["closure"]
    parents = parent_verification()
    wrong = _wrong_restore_measurements(lifecycle)
    tests: list[dict[str, Any]] = []

    def record(test_id: str, passed: bool, observed: Any) -> None:
        tests.append({"id": test_id, "observed": observed, "status": "PASS" if passed else "FAIL"})

    record("parent_r0_source_and_fixture_custody_exact", parents["custody"]["r0_exact"], parents["custody"]["r0"])
    record("parent_r0_committed_result_recomputed", parents["r0_verify"]["status"] == "PASS" and parents["r0_verify"]["recomputed_results_match"], parents["r0_verify"])
    record("parent_r1_source_and_fixture_custody_exact", parents["custody"]["r1_exact"], parents["custody"]["r1"])
    record("parent_r1_committed_result_recomputed", parents["r1_verify"]["status"] == "PASS" and parents["r1_verify"]["recomputed_results_match"], parents["r1_verify"])

    record("loop_contract_strict_round_trip", LoopContract.from_bytes(contract.canonical_bytes()).canonical_bytes() == contract.canonical_bytes(), contract.digest())
    duplicate_contract = contract.canonical_bytes().decode("utf-8").replace(f'  "schema": "{CONTRACT_SCHEMA}",', f'  "schema": "{CONTRACT_SCHEMA}",\n  "schema": "{CONTRACT_SCHEMA}",', 1).encode("utf-8")
    passed, message = _expect_error(lambda: LoopContract.from_bytes(duplicate_contract), "duplicate JSON object key")
    record("loop_contract_duplicate_key_rejected", passed, message)
    passed, message = _expect_error(lambda: LoopContract.from_document(contract.document() | {"extra": "hidden"}), "keys mismatch")
    record("loop_contract_unknown_field_rejected", passed, message)
    nonfinite = contract.document() | {"restoration_tolerance": float("nan")}
    passed, message = _expect_error(lambda: LoopContract.from_document(nonfinite), "must be finite")
    record("loop_contract_nonfinite_rejected", passed, message)
    boolean = contract.document() | {"carrier_sample_count": True}
    passed, message = _expect_error(lambda: LoopContract.from_document(boolean), "must be an integer")
    record("loop_contract_boolean_number_rejected", passed, message)
    short_schedule = contract.document() | {"shift_schedule": [17, -29]}
    passed, message = _expect_error(lambda: LoopContract.from_document(short_schedule), "exactly T0 through T3" if False else "shift schedule")
    record("loop_contract_wrong_schedule_length_rejected", passed, message)
    wrong_shifts = contract.document() | {"shift_schedule": [17, -29, 44]}
    passed, message = _expect_error(lambda: LoopContract.from_document(wrong_shifts), "shift schedule")
    record("loop_contract_wrong_shifts_rejected", passed, message)
    wrong_query_doc = contract.document() | {"query_tree_digest": "0" * 64}
    passed, message = _expect_error(lambda: LoopContract.from_document(wrong_query_doc), "query identity")
    record("loop_contract_wrong_query_rejected", passed, message)
    wrong_source_doc = contract.document() | {"r1_source_sha256": "0" * 64}
    passed, message = _expect_error(lambda: LoopContract.from_document(wrong_source_doc), "R1 source identity")
    record("loop_contract_wrong_source_identity_rejected", passed, message)
    forbidden_doc = contract.document() | {"expected_result": 1}
    passed, message = _expect_error(lambda: LoopContract.from_document(forbidden_doc), "forbidden field")
    record("loop_contract_forbidden_field_rejected", passed, message)

    record("latch_strict_round_trip", RelationalLatch.from_bytes(latch.canonical_bytes()).canonical_bytes() == latch.canonical_bytes(), latch.digest())
    duplicate_latch = latch.canonical_bytes().decode("utf-8").replace(f'  "schema": "{LATCH_SCHEMA}"', f'  "schema": "{LATCH_SCHEMA}",\n  "schema": "{LATCH_SCHEMA}"', 1).encode("utf-8")
    passed, message = _expect_error(lambda: RelationalLatch.from_bytes(duplicate_latch), "duplicate JSON object key")
    record("latch_duplicate_key_rejected", passed, message)
    passed, message = _expect_error(lambda: RelationalLatch.from_document(latch.document() | {"extra": 1}), "keys mismatch")
    record("latch_unknown_field_rejected", passed, message)
    passed, message = _expect_error(lambda: RelationalLatch.from_document(latch.document() | {"response_real": float("inf")}), "must be finite")
    record("latch_nonfinite_rejected", passed, message)
    passed, message = _expect_error(lambda: RelationalLatch.from_document(latch.document() | {"response_real": True}), "must be a JSON number")
    record("latch_boolean_number_rejected", passed, message)
    passed, message = _expect_error(lambda: RelationalLatch.from_document(latch.document() | {"score": 0.5}), "forbidden field")
    record("latch_forbidden_field_rejected", passed, message)
    passed, message = _expect_error(lambda: RelationalLatch.from_document(latch.document() | {"schema": "wrong"}), "schema mismatch")
    record("latch_wrong_schema_rejected", passed, message)
    passed, message = _expect_error(lambda: RelationalLatch.from_document(latch.document() | {"query_id": "hierarchy_b"}), "query_id mismatch")
    record("latch_wrong_query_id_rejected", passed, message)
    passed, message = _expect_error(lambda: RelationalLatch.from_document(latch.document() | {"latch_stage": "before_forward"}), "creation stage mismatch")
    record("latch_wrong_stage_rejected", passed, message)

    record("closure_strict_round_trip", CatalyticClosure.from_bytes(closure.canonical_bytes()).canonical_bytes() == closure.canonical_bytes(), closure.digest())
    duplicate_closure = closure.canonical_bytes().decode("utf-8").replace(f'  "schema": "{CLOSURE_SCHEMA}"', f'  "schema": "{CLOSURE_SCHEMA}",\n  "schema": "{CLOSURE_SCHEMA}"', 1).encode("utf-8")
    passed, message = _expect_error(lambda: CatalyticClosure.from_bytes(duplicate_closure), "duplicate JSON object key")
    record("closure_duplicate_key_rejected", passed, message)
    passed, message = _expect_error(lambda: CatalyticClosure.from_document(closure.document() | {"extra": 1}), "keys mismatch")
    record("closure_unknown_field_rejected", passed, message)
    passed, message = _expect_error(lambda: CatalyticClosure.from_document(closure.document() | {"restore_max_error": float("nan")}), "must be finite")
    record("closure_nonfinite_rejected", passed, message)
    passed, message = _expect_error(lambda: CatalyticClosure.from_document(closure.document() | {"ancestry_byte_exact": 1}), "must be a boolean")
    record("closure_boolean_rejected", passed, message)
    passed, message = _expect_error(lambda: CatalyticClosure.from_document(closure.document() | {"states": []}), "keys mismatch")
    record("closure_forbidden_history_field_rejected", passed, message)

    def validate_mutated_closure(**changes: Any) -> None:
        document = copy.deepcopy(closure.document())
        document.update(changes)
        mutated = CatalyticClosure.from_document(document)
        validate_closure_semantics(mutated, lifecycle)

    passed, message = _expect_error(lambda: validate_mutated_closure(schema="wrong"), "schema mismatch")
    record("closure_wrong_schema_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(ancestry_byte_exact=False), "must be true")
    record("closure_false_ancestry_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(recovered_initial_tree_digest="0" * 64), "recovered T0 ancestry identity")
    record("closure_wrong_recovered_digest_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(recovered_initial_tree_canonical_sha256="0" * 64), "recovered T0 ancestry identity")
    record("closure_wrong_recovered_canonical_sha_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(forward_displacement_l2=closure.forward_displacement_l2 + 0.125), "forward displacement does not match")
    record("closure_wrong_displacement_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(restore_max_error=1e-13), "restoration error does not match")
    record("closure_wrong_restore_error_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(carrier_equivalence_restored=False), "must be true")
    record("closure_wrong_equivalence_verdict_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(carrier_byte_exact=not closure.carrier_byte_exact), "byte-exact diagnostic")
    record("closure_wrong_byte_exact_verdict_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(latch_sha256="0" * 64), "latch digest mismatch")
    record("closure_wrong_latch_digest_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(carrier_before_sha256="0" * 64), "carrier hash binding mismatch")
    record("closure_wrong_carrier_before_hash_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(carrier_restored_sha256="0" * 64), "carrier hash binding mismatch")
    record("closure_wrong_carrier_restored_hash_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_mutated_closure(claim_ceiling="PHYSICAL"), "claim ceiling mismatch")
    record("closure_wrong_claim_ceiling_rejected", passed, message)

    for test_id, relative in (
        ("carrier_before_strict_binary_parse", CARRIER_BEFORE_PATH),
        ("carrier_displaced_strict_binary_parse", CARRIER_DISPLACED_PATH),
        ("carrier_restored_strict_binary_parse", CARRIER_RESTORED_PATH),
    ):
        values = parse_complex_carrier_bytes((package_dir / relative).read_bytes())
        record(test_id, values.shape == (r0.SAMPLE_COUNT,) and np.all(np.isfinite(values)), {"samples": int(values.size), "sha256": r0.sha256_file(package_dir / relative)})
    before_payload = (package_dir / CARRIER_BEFORE_PATH).read_bytes()
    passed, message = _expect_error(lambda: parse_complex_carrier_bytes(before_payload[:-1]), "byte count")
    record("carrier_truncation_rejected", passed, message)
    passed, message = _expect_error(lambda: parse_complex_carrier_bytes(before_payload + b"\0"), "byte count")
    record("carrier_extension_rejected", passed, message)
    nonfinite_carrier = lifecycle.carrier_before.copy()
    nonfinite_carrier[0] = complex(float("nan"), 0.0)
    raw_nonfinite = np.asarray(nonfinite_carrier, dtype="<c16").tobytes()
    passed, message = _expect_error(lambda: parse_complex_carrier_bytes(raw_nonfinite), "non-finite")
    record("carrier_nonfinite_rejected", passed, message)
    endian_mutated = np.asarray(lifecycle.carrier_before, dtype=">c16").tobytes()
    before_record = manifest["fixtures"][0]
    passed, message = _expect_error(lambda: validate_fixture_payload(before_record, endian_mutated), "SHA-256 mismatch")
    record("carrier_endian_mutation_rejected", passed, message)
    zero_carrier = lifecycle.carrier_before.copy()
    zero_carrier[17] = 0.0j
    passed, message = _expect_error(lambda: validate_borrowed_carrier(zero_carrier, contract), "zero-amplitude coordinate")
    record("zero_amplitude_carrier_coordinate_rejected", passed, message)

    displacement = lifecycle.closure.forward_displacement_l2
    record("forward_carrier_displacement_nonzero", displacement >= MIN_FORWARD_DISPLACEMENT_L2, displacement)
    record("correct_carrier_equivalence_restoration", lifecycle.closure.carrier_equivalence_restored and lifecycle.closure.restore_max_error <= CARRIER_RESTORE_TOL, {"metric": RESTORATION_METRIC, "max_error": lifecycle.closure.restore_max_error, "tolerance": CARRIER_RESTORE_TOL})
    byte_honest = lifecycle.closure.carrier_byte_exact == (lifecycle.closure.carrier_before_sha256 == lifecycle.closure.carrier_restored_sha256)
    record("carrier_byte_exact_diagnostic_honest", byte_honest, {"before": lifecycle.closure.carrier_before_sha256, "restored": lifecycle.closure.carrier_restored_sha256, "byte_exact": lifecycle.closure.carrier_byte_exact})
    record("exact_t0_ancestry_bytes_recovered", lifecycle.closure.ancestry_byte_exact and lifecycle.closure.recovered_initial_tree_canonical_sha256 == contract.r1_initial_tree_canonical_sha256, {"byte_exact": lifecycle.closure.ancestry_byte_exact, "tree_digest": lifecycle.closure.recovered_initial_tree_digest})
    record("external_latch_survives_both_restoration_stages", lifecycle.latch.canonical_bytes() == lifecycle.closure.latch.canonical_bytes(), lifecycle.latch.digest())
    forbidden_closure_keys = {"states", "drives", "specs", "receipts", "carrier_displaced", "displaced_carrier"}
    record("closure_excludes_reversed_histories", not (set(lifecycle.closure.document()) & forbidden_closure_keys), sorted(lifecycle.closure.document()))
    record("query_frozen_before_trajectory_execution", contract.query_selection_stage == QUERY_SELECTION_STAGE and contract.query_tree_digest == r0.hierarchy_a().digest(), {"query_id": contract.query_id, "stage": contract.query_selection_stage})
    query_change = abs(lifecycle.exact_query_response - lifecycle.wrong_query_response)
    record("wrong_query_changes_complex_latch", query_change >= MIN_QUERY_CHANGE, {"exact": {"real": _metric(lifecycle.exact_query_response.real), "imag": _metric(lifecycle.exact_query_response.imag)}, "wrong": {"real": _metric(lifecycle.wrong_query_response.real), "imag": _metric(lifecycle.wrong_query_response.imag)}, "difference": _metric(query_change)})

    for test_id, key in (
        ("forward_order_inverse_rejected", "forward_order_inverse_error"),
        ("wrong_trajectory_inverse_rejected", "wrong_trajectory_inverse_error"),
        ("omitted_inverse_step_rejected", "omitted_inverse_error"),
        ("duplicated_inverse_step_rejected", "duplicated_inverse_error"),
        ("wrong_shift_sign_rejected", "wrong_shift_sign_error"),
        ("wrong_shift_magnitude_rejected", "wrong_shift_magnitude_error"),
        ("wrong_state_one_inverse_leg_rejected", "wrong_state_one_leg_error"),
        ("no_restore_control_rejected", "no_restore_error"),
    ):
        record(test_id, wrong[key] >= MIN_WRONG_RESTORE_ERROR, wrong[key])
    passed, message = _expect_error(lambda: validate_forward_displacement(lifecycle.carrier_before, lifecycle.carrier_before, contract), "below the frozen minimum")
    record("identity_untouched_carrier_rejected", passed, message)
    passed, message = _expect_error(lambda: validate_query_preselection(contract, r0.hierarchy_a(), "after_result_observation"), "before trajectory execution")
    record("query_selected_after_result_rejected", passed, message)

    by_role = {record["role"]: record for record in manifest["fixtures"]}
    for test_id, role, relative in (
        ("mutated_displaced_carrier_fixture_rejected", "carrier_displaced", CARRIER_DISPLACED_PATH),
        ("mutated_restored_carrier_fixture_rejected", "carrier_restored", CARRIER_RESTORED_PATH),
        ("mutated_latch_fixture_rejected", "relational_latch", LATCH_FIXTURE_PATH),
    ):
        passed, message = _expect_error(lambda role=role, relative=relative: validate_fixture_payload(by_role[role], _mutated_bytes((package_dir / relative).read_bytes())), "SHA-256 mismatch")
        record(test_id, passed, message)
    mutated_contract = contract.document() | {"shift_schedule": [17, -29, 44]}
    passed, message = _expect_error(lambda: LoopContract.from_document(mutated_contract), "shift schedule")
    record("mutated_loop_contract_rejected", passed, message)
    substituted_manifest = copy.deepcopy(dict(manifest))
    substituted_manifest["fixtures"][0]["role"], substituted_manifest["fixtures"][1]["role"] = substituted_manifest["fixtures"][1]["role"], substituted_manifest["fixtures"][0]["role"]
    passed, message = _expect_error(lambda: validate_manifest_document(substituted_manifest), "role substitution")
    record("manifest_role_substitution_rejected", passed, message)
    record("committed_fixture_bytes_match_reference_lifecycle", True, list(ORDERED_FIXTURE_PATHS))
    manifest_close = manifest == fixture_manifest(package_dir) and all(r0.sha256_file(package_dir / record["path"]) == record["sha256"] for record in manifest["fixtures"])
    record("manifest_binds_all_fixture_bytes", manifest_close, manifest["fixture_set_sha256"])

    ast_proof = native_ast_proof(source_path)
    record("ast_native_call_graph_no_scalar_feedback", ast_proof["status"] == "PASS", ast_proof)
    record("ast_query_and_latch_downstream_of_forward_displacement", ast_proof["query_and_latch_downstream"] and not ast_proof["latch_feedback_calls"], {"downstream": ast_proof["query_and_latch_downstream"], "feedback_calls": ast_proof["latch_feedback_calls"]})
    record("ast_lifecycle_order_and_external_closure", ast_proof["lifecycle_order"] and not (set(lifecycle.closure.document()) & forbidden_closure_keys), ast_proof["lifecycle_positions"])
    mutations = native_ast_mutation_probes(source_path)
    record("ast_no_feedback_mutation_probes_rejected", all(mutations.values()), mutations)

    observed_ids = [test["id"] for test in tests]
    expected_ids = reference_test_ids()
    if observed_ids != expected_ids:
        raise ValueError(f"test order/coverage mismatch: expected={expected_ids}, observed={observed_ids}")
    passed_count = sum(test["status"] == "PASS" for test in tests)
    measurements = {
        "ast_no_feedback": ast_proof,
        "carrier_before_sha256": lifecycle.closure.carrier_before_sha256,
        "carrier_byte_exact": lifecycle.closure.carrier_byte_exact,
        "carrier_displaced_sha256": lifecycle.latch.carrier_displaced_sha256,
        "carrier_restored_sha256": lifecycle.closure.carrier_restored_sha256,
        "equivalence_restoration": lifecycle.closure.carrier_equivalence_restored,
        "exact_query_response": {"imag": _metric(lifecycle.exact_query_response.imag), "real": _metric(lifecycle.exact_query_response.real)},
        "forward_displacement_l2": lifecycle.closure.forward_displacement_l2,
        "query_response_difference": _metric(query_change),
        "restore_max_error": lifecycle.closure.restore_max_error,
        "t0_ancestry_byte_exact": lifecycle.closure.ancestry_byte_exact,
        "wrong_query_response": {"imag": _metric(lifecycle.wrong_query_response.imag), "real": _metric(lifecycle.wrong_query_response.real)},
        **wrong,
    }
    return {
        "measurements": measurements,
        "summary": {"failed": len(tests) - passed_count, "passed": passed_count, "test_count": len(tests)},
        "tests": tests,
    }


def verification_policy() -> dict[str, Any]:
    return {
        "carrier_restoration": "numerical equivalence under max_abs_complex_sample_error <= 1e-12; hashes reported separately",
        "committed_byte_authority": "all six fixtures are parsed and recomputed from committed bytes",
        "environment_receipt": "informational only",
        "parent_authority": "exact R0 and R1 source, manifest, fixture-set, tests, and result identities plus committed-result recomputation",
        "portable_metric_atol": PORTABLE_METRIC_ATOL,
        "portable_metric_rtol": PORTABLE_METRIC_RTOL,
        "stored_pass_authority": False,
        "t0_restoration": "exact canonical bytes",
    }


def scientific_result(package_dir: Path, source_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    scored = run_reference_tests(package_dir, source_path, manifest)
    return {
        "binary_carrier_bytes": manifest["binary_carrier_bytes"],
        "binary_carrier_count": manifest["binary_carrier_count"],
        "claim_ceiling": CLAIM_CEILING,
        "contact_counts": {"adc_dac": 0, "audio_playback": 0, "audio_recording": 0, "hardware": 0, "ssh_scp": 0, "target": 0, "transducer": 0},
        "established_token_if_all_gates_close": ESTABLISHED_TOKEN,
        "fixture_count": manifest["fixture_count"],
        "fixture_manifest_sha256": r0.sha256_file(package_dir / MANIFEST_FILE),
        "fixture_set_sha256": manifest["fixture_set_sha256"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "measurements": scored["measurements"],
        "next_boundary": NEXT_BOUNDARY,
        "ordinary_software_only": True,
        "physical_claims_established": [],
        "reference_tests_sha256": r0.sha256_file(package_dir / TESTS_FILE),
        "schema": "catalytic_wave_loop_scientific_result_v1",
        "summary": scored["summary"],
        "tests": scored["tests"],
    }


def result_document(package_dir: Path, source_path: Path) -> dict[str, Any]:
    manifest = _load_canonical_json(package_dir / MANIFEST_FILE, "fixture manifest")
    validate_manifest_document(manifest)
    return {
        "execution_environment": {"numpy": np.__version__, "platform": platform.platform(), "python": ".".join(str(part) for part in sys.version_info[:3])},
        "parent_custody": parent_custody(),
        "schema": RESULT_SCHEMA,
        "schema_bindings": {
            "closure_schema_sha256": r0.sha256_file(package_dir / CLOSURE_SCHEMA_FILE),
            "contract_schema_sha256": r0.sha256_file(package_dir / CONTRACT_SCHEMA_FILE),
            "latch_schema_sha256": r0.sha256_file(package_dir / LATCH_SCHEMA_FILE),
        },
        "scientific": scientific_result(package_dir, source_path, manifest),
        "source_binding": source_binding(source_path),
        "verification_policy": verification_policy(),
    }


def portable_equal(stored: Any, recomputed: Any) -> bool:
    if isinstance(stored, bool) or isinstance(recomputed, bool):
        return type(stored) is type(recomputed) and stored == recomputed
    if isinstance(stored, (int, float)) and isinstance(recomputed, (int, float)):
        return math.isclose(float(stored), float(recomputed), rel_tol=PORTABLE_METRIC_RTOL, abs_tol=PORTABLE_METRIC_ATOL)
    if isinstance(stored, dict) and isinstance(recomputed, dict):
        return set(stored) == set(recomputed) and all(portable_equal(stored[key], recomputed[key]) for key in stored)
    if isinstance(stored, list) and isinstance(recomputed, list):
        return len(stored) == len(recomputed) and all(portable_equal(left, right) for left, right in zip(stored, recomputed, strict=True))
    return type(stored) is type(recomputed) and stored == recomputed


def build_package(package_dir: Path, source_path: Path) -> dict[str, Any]:
    package_dir.mkdir(parents=True, exist_ok=True)
    r0.write_json_atomic(package_dir / CONTRACT_SCHEMA_FILE, loop_contract_schema_document())
    r0.write_json_atomic(package_dir / LATCH_SCHEMA_FILE, latch_schema_document())
    r0.write_json_atomic(package_dir / CLOSURE_SCHEMA_FILE, closure_schema_document())
    r0.write_json_atomic(package_dir / TESTS_FILE, reference_test_spec())
    build_fixture_files(package_dir)
    manifest = fixture_manifest(package_dir)
    r0.write_json_atomic(package_dir / MANIFEST_FILE, manifest)
    result = result_document(package_dir, source_path)
    r0.write_json_atomic(package_dir / RESULTS_FILE, result)
    return {
        "binary_carrier_bytes": manifest["binary_carrier_bytes"],
        "binary_carrier_count": manifest["binary_carrier_count"],
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
    }


def verify_package(package_dir: Path, source_path: Path) -> dict[str, Any]:
    schemas = (
        (CONTRACT_SCHEMA_FILE, loop_contract_schema_document(), "contract schema"),
        (LATCH_SCHEMA_FILE, latch_schema_document(), "latch schema"),
        (CLOSURE_SCHEMA_FILE, closure_schema_document(), "closure schema"),
    )
    for filename, expected, label in schemas:
        observed = _load_canonical_json(package_dir / filename, label)
        if observed != expected:
            raise ValueError(f"committed {label} differs from source law")
    test_spec = _load_canonical_json(package_dir / TESTS_FILE, "reference test specification")
    if test_spec != reference_test_spec():
        raise ValueError("committed reference tests differ from source law")
    stored_manifest = _load_canonical_json(package_dir / MANIFEST_FILE, "fixture manifest")
    validate_manifest_document(stored_manifest)
    recomputed_manifest = fixture_manifest(package_dir)
    if stored_manifest != recomputed_manifest:
        raise ValueError("committed manifest does not match committed fixture bytes")
    stored_result = _load_canonical_json(package_dir / RESULTS_FILE, "reference result")
    _exact_keys(stored_result, {"execution_environment", "parent_custody", "schema", "schema_bindings", "scientific", "source_binding", "verification_policy"}, "reference result")
    if stored_result["schema"] != RESULT_SCHEMA:
        raise ValueError("reference result schema mismatch")
    if stored_result["verification_policy"] != verification_policy():
        raise ValueError("verification policy differs from source law")
    if stored_result["source_binding"] != source_binding(source_path):
        raise ValueError("R2S source binding mismatch")
    if stored_result["parent_custody"] != parent_custody():
        raise ValueError("parent custody binding mismatch")
    recomputed_scientific = scientific_result(package_dir, source_path, stored_manifest)
    if not portable_equal(stored_result["scientific"], recomputed_scientific):
        raise ValueError("committed-byte scientific result recomputation mismatch")
    failed = recomputed_scientific["summary"]["failed"]
    return {
        "binary_carrier_bytes": stored_manifest["binary_carrier_bytes"],
        "binary_carrier_count": stored_manifest["binary_carrier_count"],
        "fixture_count": stored_manifest["fixture_count"],
        "fixture_total_bytes": stored_manifest["total_fixture_bytes"],
        "manifest_sha256": r0.sha256_file(package_dir / MANIFEST_FILE),
        "operation": "verify",
        "parent_binding_match": True,
        "recomputed_results_match": True,
        "result_sha256": r0.sha256_file(package_dir / RESULTS_FILE),
        "source_binding_match": True,
        "status": "PASS" if failed == 0 else "FAIL",
        "test_count": recomputed_scientific["summary"]["test_count"],
        "tests_passed": recomputed_scientific["summary"]["passed"],
        "tests_sha256": r0.sha256_file(package_dir / TESTS_FILE),
    }


def self_test(source_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="catalytic_wave_loop_self_test_") as raw:
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
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "verify", "self-test"), nargs="?", default="self-test")
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
        payload = {"error": f"{type(exc).__name__}: {exc}", "operation": args.operation, "status": "FAIL"}
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if payload.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
