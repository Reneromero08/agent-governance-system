#!/usr/bin/env python3
"""Bounded, dependency-injected Gate A engineering-smoke execution gate.

This module contains no transport and opens no network connection.  The host
adapter validates authority before transport; the target runner validates the
same authority again and calls :func:`execute_once` with local surfaces.  Tests
replace every filesystem, process, temperature, frequency, evidence, claim and
runtime surface, so the live backend is never reached during qualification.
"""

from __future__ import annotations

import enum
import json
import math
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gate_a_process_custody as process_custody

SCHEDULE_SHA256 = "418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116"
SEQUENCE = ("I", "I", "I", "I", "C0", "D0", "S0E", "S0E", "S0E", "S0E", "O0", "O0", "A0P", "A0N", "T", "T")
OFF_TOKENS = frozenset({"I", "C0", "D0", "O0", "T"})
SENDER_CORE = 4
RECEIVER_CORE = 5
READ_HZ = 8000
SLOT_S = 0.5
SLOT_COUNT = 16
NOMINAL_DURATION_S = 8.0
NOMINAL_SAMPLES_PER_SLOT = 4000
TEMPERATURE_VETO_C = 68.0
REQUIRED_FREQUENCY_KHZ = 1_600_000
FORBIDDEN_PROCESSES = process_custody.FORBIDDEN_MARKERS


class ExecutorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutorError(message)


class NamespaceState(enum.Enum):
    ABSENT = "absent"
    PRESENT = "present"
    UNOBSERVABLE = "unobservable"


@dataclass(frozen=True)
class FrozenPlan:
    authority_sha256: str
    execution_bundle_sha256: str
    output_root: Path
    sequence: tuple[str, ...] = SEQUENCE
    sender_core: int = SENDER_CORE
    receiver_core: int = RECEIVER_CORE
    read_hz: int = READ_HZ
    slot_s: float = SLOT_S
    slot_count: int = SLOT_COUNT
    nominal_duration_s: float = NOMINAL_DURATION_S
    temperature_veto_c: float = TEMPERATURE_VETO_C
    required_frequency_khz: int = REQUIRED_FREQUENCY_KHZ
    maximum_execution_count: int = 1
    automatic_retry: bool = False


class PreflightSurface(Protocol):
    def inspect_namespace(self, path: Path) -> NamespaceState: ...
    def process_snapshot(self, phase: str) -> dict[str, Any]: ...
    def temperature_c(self) -> float: ...
    def frequency_khz(self, core: int) -> int: ...


class ClaimStore(Protocol):
    def claim(self, authority_sha256: str, plan: FrozenPlan) -> None: ...


class EvidenceStore(Protocol):
    def begin(self, plan: FrozenPlan, preflight: dict[str, Any]) -> None: ...
    def event(self, value: dict[str, Any]) -> None: ...
    def process_receipt(self, phase: str, receipt: dict[str, Any]) -> None: ...
    def complete(self, result: dict[str, Any]) -> None: ...
    def fail(self, reason: str) -> None: ...


class RuntimeSurface(Protocol):
    def execute(self, plan: FrozenPlan) -> dict[str, Any]: ...
    def verify_evidence(self, plan: FrozenPlan, result: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class ExecutionSurfaces:
    preflight: PreflightSurface
    claims: ClaimStore
    evidence: EvidenceStore
    runtime: RuntimeSurface


def validate_frozen_schedule(schedule: dict[str, Any]) -> None:
    require(schedule.get("schedule_sha256") == SCHEDULE_SHA256, "schedule digest mismatch")
    require(tuple(schedule.get("slot_sequence", ())) == SEQUENCE, "slot sequence mismatch")
    require(schedule.get("timing") == {
        "automatic_retry": False,
        "maximum_execution_count": 1,
        "nominal_duration_s": 8.0,
        "nominal_samples_per_slot": 4000,
        "read_hz": 8000,
        "slot_count": 16,
        "slot_s": 0.5,
        "temperature_veto_c": 68.0,
    }, "frozen timing mismatch")
    require(schedule.get("session", {}).get("sender_core") == SENDER_CORE, "sender core mismatch")
    require(schedule.get("session", {}).get("receiver_core") == RECEIVER_CORE, "receiver core mismatch")
    require(schedule.get("frequency_and_voltage") == {
        "expected_observed_khz": REQUIRED_FREQUENCY_KHZ,
        "frequency_write_authorized": False,
        "mismatch_action": "STOP_BEFORE_DRIVE",
        "msr_write_authorized": False,
        "voltage_write_authorized": False,
    }, "frequency boundary mismatch")
    definitions = schedule.get("slot_definitions", {})
    require(set(definitions) == {"I", "C0", "D0", "S0E", "O0", "A0P", "A0N", "T"}, "slot definition set mismatch")
    for token in OFF_TOKENS:
        executed = definitions[token].get("executed", {})
        require(executed.get("drive_on") is False, f"{token} must remain physically off")
        for field in ("amplitude_level", "phase_action", "physical_tone_index", "sender_epoch_id", "sign"):
            require(executed.get(field) is None, f"{token} executed {field} must remain null")
    step = definitions["S0E"]["executed"]
    require(step == {
        "amplitude_level": 2,
        "drive_on": True,
        "executed_mode": "STEP",
        "phase_action": "0",
        "physical_tone_index": 0,
        "sender_epoch_id": "gate-a:step:epoch0",
        "sign": 1,
    }, "S0E physical mapping mismatch")
    positive = definitions["A0P"]["executed"]
    negative = definitions["A0N"]["executed"]
    require(positive["drive_on"] is True and positive["sign"] == 1 and positive["phase_action"] == "0", "A0P mapping mismatch")
    require(negative["drive_on"] is True and negative["sign"] == -1 and negative["phase_action"] == "pi", "A0N mapping mismatch")


def validate_runtime_result(result: dict[str, Any], plan: FrozenPlan) -> None:
    required = {
        "status", "automatic_retry", "runtime_execution_count", "slot_records",
        "capture", "frequency_writes", "voltage_writes", "msr_reads", "msr_writes",
        "step_sender_epoch_count", "hardware_executed",
    }
    require(set(result) == required, "runtime result key set mismatch")
    require(result["status"] == "GATE_A_ENGINEERING_SMOKE_COMPLETE", "runtime did not complete")
    require(result["automatic_retry"] is False, "runtime retry flag changed")
    require(result["runtime_execution_count"] == 1, "runtime execution count mismatch")
    require(result["frequency_writes"] == 0 and result["voltage_writes"] == 0, "control write reported")
    require(result["msr_reads"] == 0 and result["msr_writes"] == 0, "MSR access reported")
    require(result["step_sender_epoch_count"] == 1, "S0E must use exactly one sender epoch")
    require(result["hardware_executed"] is True, "authorized runtime did not report hardware execution")
    capture = result["capture"]
    require(set(capture) == {
        "continuous", "covers_complete_sequence", "sample_count", "slot_sample_counts",
        "origin_tsc", "deadline_tsc", "first_sample_tsc", "last_sample_tsc", "tsc_hz",
    }, "capture key set mismatch")
    require(capture["continuous"] is True and capture["covers_complete_sequence"] is True, "capture does not cover complete sequence")
    require(isinstance(capture["sample_count"], int) and capture["sample_count"] >= int(0.9 * READ_HZ * NOMINAL_DURATION_S), "capture sample count too short")
    require(capture["slot_sample_counts"] and len(capture["slot_sample_counts"]) == SLOT_COUNT, "slot capture count mismatch")
    require(all(isinstance(count, int) and count >= int(0.9 * NOMINAL_SAMPLES_PER_SLOT) for count in capture["slot_sample_counts"]), "partial slot capture")
    for field in ("origin_tsc", "deadline_tsc", "first_sample_tsc", "last_sample_tsc"):
        require(isinstance(capture[field], int) and capture[field] > 0, f"capture {field} malformed")
    tsc_hz = capture["tsc_hz"]
    require(isinstance(tsc_hz, (int, float)) and math.isfinite(tsc_hz) and tsc_hz >= 100_000_000.0, "capture TSC calibration malformed")
    origin = capture["origin_tsc"]
    deadline = capture["deadline_tsc"]
    first = capture["first_sample_tsc"]
    last = capture["last_sample_tsc"]
    period = tsc_hz / READ_HZ
    require(deadline == origin + int(NOMINAL_DURATION_S * tsc_hz), "capture deadline drift")
    require(origin <= first <= origin + int(4 * period), "capture does not begin at the sequence boundary")
    require(deadline - int(4 * period) <= last <= deadline + int(0.02 * tsc_hz), "capture does not reach the sequence boundary")

    records = result["slot_records"]
    require(isinstance(records, list) and len(records) == SLOT_COUNT, "slot record count mismatch")
    step_epochs: set[str] = set()
    for index, (token, record) in enumerate(zip(plan.sequence, records, strict=True)):
        require(set(record) == {"index", "token", "requested_start_s", "requested_end_s", "drive_on", "amplitude_level", "phase_index", "sign", "sender_epoch_id"}, "slot record key set mismatch")
        require(record["index"] == index and record["token"] == token, "slot sequence drift")
        require(record["requested_start_s"] == index * SLOT_S, "slot start drift")
        require(record["requested_end_s"] == (index + 1) * SLOT_S, "slot end drift")
        if token in OFF_TOKENS:
            require(record["drive_on"] is False, f"{token} drove hardware")
            require(all(record[field] is None for field in ("amplitude_level", "phase_index", "sign", "sender_epoch_id")), f"{token} physical controls not null")
        elif token == "S0E":
            require(record["drive_on"] is True and record["amplitude_level"] == 2 and record["phase_index"] == 0 and record["sign"] == 1, "S0E mapping drift")
            require(record["sender_epoch_id"] == "gate-a:step:epoch0", "S0E sender epoch drift")
            step_epochs.add(record["sender_epoch_id"])
        elif token == "A0P":
            require(record["drive_on"] is True and record["amplitude_level"] == 2 and record["phase_index"] == 0 and record["sign"] == 1, "A0P mapping drift")
        elif token == "A0N":
            require(record["drive_on"] is True and record["amplitude_level"] == 2 and record["phase_index"] == 4 and record["sign"] == -1, "A0N mapping drift")
    require(step_epochs == {"gate-a:step:epoch0"}, "S0E used multiple epochs")


LOCKIN_RECORD_KEYS = frozenset({
    "schema_id", "slot_index", "token", "raw_sample_start_index",
    "raw_sample_end_index", "sample_count", "tone_frequency_hz", "lockin_i",
    "lockin_q", "magnitude", "off_frequency_floor", "origin_tsc",
    "slot_start_tsc", "slot_end_tsc",
})
LIFECYCLE_RECORD_KEYS = frozenset({
    "schema_id", "record_type", "event_tsc", "slot_index", "token",
    "sender_state", "sender_epoch_id", "phase_index", "sign",
    "requested_start_tsc", "requested_end_tsc", "sender_transition_tsc",
    "thread_create_tsc",
    "thread_ready_tsc", "epoch_start_tsc", "first_drive_tsc",
    "stop_requested_tsc", "thread_exit_tsc", "thread_join_start_tsc",
    "thread_join_tsc",
})
LIFECYCLE_STATES = frozenset({"not_created", "starting", "active", "stopping", "joined"})


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"retained evidence missing: {path.name}")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            require(line != "", f"blank JSONL record: {path.name}:{line_number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL object required: {path.name}:{line_number}")
            records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutorError(f"retained JSONL unreadable: {path.name}") from exc
    require(records, f"retained JSONL is empty: {path.name}")
    return records


def _tone_zero_hz() -> float:
    return math.exp(math.log(20.0)) * (1.0 + 0.013 * math.sin(2.399963))


def _lockin_from_raw(
    raw: list[tuple[int, float]],
    *,
    frequency: float,
    origin_tsc: int,
    tsc_hz: float,
) -> tuple[float, float, float, float]:
    require(len(raw) >= 2, "lock-in sample range too short")
    mean = sum(sample for _timestamp, sample in raw) / len(raw)
    i_acc = q_acc = f_i = f_q = weight = 0.0
    off_frequency = frequency * 1.37 + 0.071
    for index, (timestamp, observed) in enumerate(raw):
        window = 0.5 * (1.0 - math.cos(2.0 * math.pi * index / (len(raw) - 1)))
        sample = (observed - mean) * window
        seconds = (timestamp - origin_tsc) / tsc_hz
        i_acc += sample * math.cos(2.0 * math.pi * frequency * seconds)
        q_acc += sample * math.sin(2.0 * math.pi * frequency * seconds)
        f_i += sample * math.cos(2.0 * math.pi * off_frequency * seconds)
        f_q += sample * math.sin(2.0 * math.pi * off_frequency * seconds)
        weight += window
    require(weight > 0.0, "lock-in window has zero weight")
    i_value = 2.0 * i_acc / weight
    q_value = 2.0 * q_acc / weight
    return i_value, q_value, math.hypot(i_value, q_value), 2.0 * math.hypot(f_i, f_q) / weight


def _verify_lockin_custody(runtime_root: Path, result: dict[str, Any], plan: FrozenPlan) -> None:
    raw_path = runtime_root / "raw_samples.bin"
    require(raw_path.is_file() and not raw_path.is_symlink(), "raw sample custody missing")
    raw_bytes = raw_path.read_bytes()
    require(len(raw_bytes) % 16 == 0 and len(raw_bytes) > 0, "raw sample custody size malformed")
    raw = list(struct.iter_unpack("<Qd", raw_bytes))
    require(len(raw) == result["capture"]["sample_count"], "raw sample count mismatch")
    require(all(raw[index][0] < raw[index + 1][0] for index in range(len(raw) - 1)), "raw timestamps are not strictly increasing")

    records = _load_jsonl(runtime_root / "LOCKIN_IQ.jsonl")
    require(len(records) == SLOT_COUNT, "lock-in slot record count mismatch")
    expected_frequency = _tone_zero_hz()
    cursor = 0
    capture = result["capture"]
    anchor_vectors: dict[str, tuple[float, float]] = {}
    for slot, (token, record) in enumerate(zip(plan.sequence, records)):
        require(set(record) == LOCKIN_RECORD_KEYS, "lock-in record key set mismatch")
        require(record["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_LOCKIN_IQ_V1", "lock-in schema mismatch")
        require(record["slot_index"] == slot and record["token"] == token, "lock-in slot mapping mismatch")
        start = record["raw_sample_start_index"]
        end = record["raw_sample_end_index"]
        require(isinstance(start, int) and isinstance(end, int) and start == cursor and start < end <= len(raw), "lock-in sample range mismatch")
        require(record["sample_count"] == end - start == capture["slot_sample_counts"][slot], "lock-in sample count mismatch")
        slot_start = capture["origin_tsc"] + int(slot * SLOT_S * capture["tsc_hz"])
        slot_end = capture["origin_tsc"] + int((slot + 1) * SLOT_S * capture["tsc_hz"])
        expected_origin = capture["origin_tsc"] + int((6 if token == "S0E" else slot) * SLOT_S * capture["tsc_hz"])
        require(record["slot_start_tsc"] == slot_start and record["slot_end_tsc"] == slot_end, "lock-in slot timing mismatch")
        require(record["origin_tsc"] == expected_origin, "lock-in origin mismatch")
        require(math.isclose(record["tone_frequency_hz"], expected_frequency, rel_tol=1e-14, abs_tol=1e-14), "lock-in tone identity mismatch")
        selected = raw[start:end]
        require(selected[0][0] >= slot_start and selected[-1][0] < slot_end, "lock-in range crosses slot boundary")
        recomputed = _lockin_from_raw(selected, frequency=expected_frequency, origin_tsc=expected_origin, tsc_hz=capture["tsc_hz"])
        for field, value in zip(("lockin_i", "lockin_q", "magnitude", "off_frequency_floor"), recomputed):
            observed = record[field]
            require(isinstance(observed, (int, float)) and math.isfinite(observed), f"lock-in {field} malformed")
            require(math.isclose(observed, value, rel_tol=1e-10, abs_tol=1e-10), f"lock-in {field} does not recompute from raw samples")
        if token in {"A0P", "A0N"}:
            anchor_vectors[token] = (recomputed[0], recomputed[1])
        cursor = end
    require(cursor == len(raw), "lock-in ranges do not close over raw custody")
    require(set(anchor_vectors) == {"A0P", "A0N"}, "anchor lock-in custody missing")
    positive = anchor_vectors["A0P"]
    negative = anchor_vectors["A0N"]
    positive_norm = math.hypot(*positive)
    negative_norm = math.hypot(*negative)
    require(positive_norm > 1e-12 and negative_norm > 1e-12, "anchor phase is unobservable")
    phase_cosine = (positive[0] * negative[0] + positive[1] * negative[1]) / (positive_norm * negative_norm)
    require(phase_cosine < -0.5, "A0P and A0N raw lock-in phases are not opposite")


def _verify_lifecycle_custody(runtime_root: Path, result: dict[str, Any], plan: FrozenPlan) -> None:
    records = _load_jsonl(runtime_root / "SENDER_LIFECYCLE.jsonl")
    for record in records:
        require(set(record) == LIFECYCLE_RECORD_KEYS, "sender lifecycle key set mismatch")
        require(record["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_SENDER_LIFECYCLE_V1", "sender lifecycle schema mismatch")
        require(record["sender_state"] in LIFECYCLE_STATES, "sender lifecycle state malformed")
        require(isinstance(record["event_tsc"], int) and record["event_tsc"] > 0, "sender lifecycle event TSC malformed")

    transitions = [record for record in records if record["record_type"] == "slot_transition"]
    require(len(transitions) == SLOT_COUNT, "sender lifecycle slot transition count mismatch")
    transitions.sort(key=lambda item: item["slot_index"])
    capture = result["capture"]
    period = capture["tsc_hz"] / READ_HZ
    expected_epochs = {
        "gate-a:step:epoch0": (6, 10, 0, 1),
        "gate-a:anchor:positive": (12, 13, 0, 1),
        "gate-a:anchor:negative": (13, 14, 4, -1),
    }
    epoch_timings: dict[str, tuple[int, ...]] = {}
    for slot, (token, record) in enumerate(zip(plan.sequence, transitions)):
        require(record["slot_index"] == slot and record["token"] == token, "sender lifecycle slot mapping mismatch")
        requested_start = capture["origin_tsc"] + int(slot * SLOT_S * capture["tsc_hz"])
        requested_end = capture["origin_tsc"] + int((slot + 1) * SLOT_S * capture["tsc_hz"])
        require(record["requested_start_tsc"] == requested_start and record["requested_end_tsc"] == requested_end, "sender lifecycle requested timing mismatch")
        require(requested_start <= record["event_tsc"] <= requested_start + int(4 * period), "sender lifecycle transition drift")
        sender_transition = record["sender_transition_tsc"]
        if token in {"S0E", "A0P", "A0N"}:
            require(isinstance(sender_transition, int) and requested_start <= sender_transition <= requested_start + int(0.005 * capture["tsc_hz"]), "actual sender transition TSC missing or drifted")
        else:
            require(sender_transition is None, "sender-absent slot carries a sender transition TSC")
        if slot <= 5:
            require(record["sender_state"] == "not_created", "sender existed during initial off/sham slot")
            require(all(record[field] is None for field in (
                "sender_epoch_id", "phase_index", "sign", "thread_create_tsc",
                "thread_ready_tsc", "epoch_start_tsc", "first_drive_tsc",
                "stop_requested_tsc", "thread_exit_tsc", "thread_join_start_tsc",
                "thread_join_tsc", "sender_transition_tsc",
            )), "initial off/sham slot carries sender lifecycle")
            continue
        if token in {"O0", "T"}:
            require(record["sender_state"] == "joined", f"sender not joined during {token}")
            require(isinstance(record["thread_join_tsc"], int) and record["thread_join_tsc"] <= requested_start, f"sender joined too late for {token}")
            continue
        require(record["sender_state"] == "active", "driven slot lacks active sender epoch")
        epoch = record["sender_epoch_id"]
        require(epoch in expected_epochs, "unknown sender epoch")
        first_slot, end_slot, phase, sign = expected_epochs[epoch]
        require(record["phase_index"] == phase and record["sign"] == sign, "sender epoch phase/sign mismatch")
        values = tuple(record[field] for field in (
            "thread_create_tsc", "thread_ready_tsc", "epoch_start_tsc",
            "first_drive_tsc", "stop_requested_tsc", "thread_exit_tsc",
            "thread_join_start_tsc", "thread_join_tsc",
        ))
        require(all(isinstance(value, int) and value > 0 for value in values), "sender epoch TSC custody incomplete")
        create, ready, start, first_drive, stop, exited, join_start, joined = values
        epoch_start = capture["origin_tsc"] + int(first_slot * SLOT_S * capture["tsc_hz"])
        epoch_end = capture["origin_tsc"] + int(end_slot * SLOT_S * capture["tsc_hz"])
        require(epoch_start <= create <= ready <= start <= first_drive < stop <= exited <= joined, "sender epoch lifecycle ordering invalid")
        require(join_start <= joined and stop <= join_start, "sender join ordering invalid")
        require(ready <= epoch_start + int(0.005 * capture["tsc_hz"]), "sender startup missed bounded skew")
        observed_phase_offset = first_drive - start
        expected_phase_offset = 0 if phase == 0 else int(0.5 / _tone_zero_hz() * capture["tsc_hz"])
        require(abs(observed_phase_offset - expected_phase_offset) <= int(0.005 * capture["tsc_hz"]), "sender first-drive TSC does not prove declared phase")
        require(joined <= epoch_end, "sender was not joined before sender-absent boundary")
        require(capture["origin_tsc"] <= create and joined <= capture["deadline_tsc"], "capture does not span sender lifecycle")
        if epoch in epoch_timings:
            require(epoch_timings[epoch] == values, "sender epoch timing changed between slots")
        else:
            epoch_timings[epoch] = values
    require(set(epoch_timings) == set(expected_epochs), "sender epoch set mismatch")
    require(len({epoch_timings[epoch] for epoch in expected_epochs}) == 3, "anchor and step epochs are not distinct")
    for epoch in expected_epochs:
        states = [record["sender_state"] for record in records if record["sender_epoch_id"] == epoch and record["record_type"] != "slot_transition"]
        require(states.count("starting") == 1 and states.count("active") == 1 and states.count("stopping") == 1 and states.count("joined") == 1, "sender epoch state custody incomplete")


def verify_retained_runtime_evidence(plan: FrozenPlan, result: dict[str, Any]) -> None:
    runtime_root = plan.output_root / "runtime"
    require(runtime_root.is_dir() and not runtime_root.is_symlink(), "runtime evidence root missing")
    _verify_lockin_custody(runtime_root, result, plan)
    _verify_lifecycle_custody(runtime_root, result, plan)


def _retain_process_receipt(
    surfaces: ExecutionSurfaces,
    *,
    phase: str,
) -> dict[str, Any]:
    receipt = surfaces.preflight.process_snapshot(phase)
    surfaces.evidence.process_receipt(phase, receipt)
    surfaces.evidence.event({
        "event": phase + "_process_scan",
        "scan_complete": receipt.get("scan_complete") if isinstance(receipt, dict) else False,
        "return_code": receipt.get("return_code") if isinstance(receipt, dict) else None,
        "stdout_sha256": receipt.get("stdout_sha256") if isinstance(receipt, dict) else None,
        "stderr_sha256": receipt.get("stderr_sha256") if isinstance(receipt, dict) else None,
        "forbidden_process_hits": receipt.get("parsed_forbidden_hits") if isinstance(receipt, dict) else None,
    })
    process_custody.validate_process_receipt(receipt, expected_phase=phase)
    return receipt


def execute_once(
    *,
    authority_validation: dict[str, Any],
    authority_sha256: str,
    execution_bundle_sha256: str,
    schedule: dict[str, Any],
    output_root: Path,
    surfaces: ExecutionSurfaces,
) -> dict[str, Any]:
    """Consume one exact authority and execute the frozen runtime at most once."""

    require(set(authority_validation) == {
        "status", "reviewed_adapter_head", "independent_review_id", "execution_bundle_sha256",
    }, "authority validation result is not closed")
    require(authority_validation["status"] == "GATE_A_EXECUTION_AUTHORITY_EXACT", "exact authority validation required")
    require(isinstance(authority_validation["reviewed_adapter_head"], str) and len(authority_validation["reviewed_adapter_head"]) == 40, "reviewed adapter head missing")
    require(isinstance(authority_validation["independent_review_id"], int) and authority_validation["independent_review_id"] > 0, "independent review binding missing")
    require(authority_validation["execution_bundle_sha256"] == execution_bundle_sha256, "authority validation bundle binding mismatch")
    require(len(authority_sha256) == 64 and all(c in "0123456789abcdef" for c in authority_sha256), "authority digest malformed")
    require(len(execution_bundle_sha256) == 64 and all(c in "0123456789abcdef" for c in execution_bundle_sha256), "bundle digest malformed")
    validate_frozen_schedule(schedule)
    state = surfaces.preflight.inspect_namespace(output_root)
    require(state is NamespaceState.ABSENT, f"output namespace not provably absent (state={state.value})")

    plan = FrozenPlan(
        authority_sha256=authority_sha256,
        execution_bundle_sha256=execution_bundle_sha256,
        output_root=output_root,
    )
    # The durable claim is intentionally before every veto and runtime call.
    # Any authorized attempt, including a veto or failure, consumes the one shot.
    surfaces.claims.claim(authority_sha256, plan)

    initial_preflight = {
        "namespace_state": state.value,
        "preflight_complete": False,
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
    }
    evidence_started = False
    try:
        surfaces.evidence.begin(plan, initial_preflight)
        evidence_started = True
        _retain_process_receipt(surfaces, phase="pre_runtime")
        temperature = surfaces.preflight.temperature_c()
        surfaces.evidence.event({"event": "temperature_preflight", "temperature_c": temperature})
        require(math.isfinite(temperature) and temperature < TEMPERATURE_VETO_C, "temperature veto")
        frequencies = {str(core): surfaces.preflight.frequency_khz(core) for core in (SENDER_CORE, RECEIVER_CORE)}
        surfaces.evidence.event({"event": "frequency_preflight", "frequency_khz": frequencies})
        require(all(value == REQUIRED_FREQUENCY_KHZ for value in frequencies.values()), "frequency veto")
        surfaces.evidence.event({"event": "preflight_complete", "preflight_complete": True})
        surfaces.evidence.event({"event": "runtime_start", "runtime_execution_count": 1, "automatic_retry": False})
        result: dict[str, Any] | None = None
        runtime_error: Exception | None = None
        try:
            result = surfaces.runtime.execute(plan)
        except Exception as exc:
            runtime_error = exc
        post_runtime_error: Exception | None = None
        try:
            _retain_process_receipt(surfaces, phase="post_runtime")
        except Exception as exc:
            post_runtime_error = exc
        if runtime_error is not None:
            if post_runtime_error is not None:
                raise ExecutorError(
                    f"runtime failed ({runtime_error}); post-runtime process custody failed ({post_runtime_error})"
                ) from runtime_error
            raise runtime_error
        if post_runtime_error is not None:
            raise post_runtime_error
        require(result is not None, "runtime returned no result")
        validate_runtime_result(result, plan)
        surfaces.runtime.verify_evidence(plan, result)
        surfaces.evidence.complete(result)
        return result
    except Exception as exc:
        if evidence_started:
            surfaces.evidence.fail(str(exc))
        raise


class LocalPreflight:
    """Read-only target preflight.  It contains no control-write or MSR API."""

    def inspect_namespace(self, path: Path) -> NamespaceState:
        try:
            os.lstat(path)
        except FileNotFoundError:
            return NamespaceState.ABSENT
        except OSError:
            return NamespaceState.UNOBSERVABLE
        return NamespaceState.PRESENT

    def process_snapshot(self, phase: str) -> dict[str, Any]:
        return process_custody.scan_processes(phase)

    def temperature_c(self) -> float:
        try:
            paths = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
        except OSError as exc:
            raise ExecutorError("temperature namespace unobservable") from exc
        require(paths, "temperature unobservable")
        values: list[float] = []
        for path in paths:
            try:
                raw = float(path.read_text(encoding="ascii").strip())
            except (OSError, ValueError) as exc:
                raise ExecutorError(f"temperature unobservable: {path}") from exc
            values.append(raw / 1000.0 if raw > 1000 else raw)
        return max(values)

    def frequency_khz(self, core: int) -> int:
        path = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq")
        try:
            value = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError) as exc:
            raise ExecutorError(f"frequency unobservable for core {core}") from exc
        return value


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        require(written > 0, "short evidence write")
        view = view[written:]


class FileClaimStore:
    """Durable O_EXCL one-attempt claim retained outside the execution root."""

    def __init__(self, root: Path):
        self.root = root

    def claim(self, authority_sha256: str, plan: FrozenPlan) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(self.root.is_dir() and not self.root.is_symlink(), "claim root is not a real directory")
        path = self.root / f"{authority_sha256}.claim.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise ExecutorError("authority execution already claimed") from exc
        try:
            payload = json.dumps({
                "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_CLAIM_V1",
                "authority_sha256": authority_sha256,
                "execution_bundle_sha256": plan.execution_bundle_sha256,
                "maximum_execution_count": 1,
                "automatic_retry": False,
            }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


class JsonEvidenceStore:
    """Exclusive, fsynced evidence writes that preserve partial failure state."""

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.events: Any = None

    def _exclusive_json(self, name: str, value: dict[str, Any]) -> None:
        path = self.output_root / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            _write_all(fd, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def begin(self, plan: FrozenPlan, preflight: dict[str, Any]) -> None:
        self.output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._exclusive_json("ATTEMPT.json", {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_ATTEMPT_V1",
            "authority_sha256": plan.authority_sha256,
            "execution_bundle_sha256": plan.execution_bundle_sha256,
            "sequence": list(plan.sequence),
            "maximum_execution_count": 1,
            "automatic_retry": False,
            "preflight": preflight,
        })
        events_path = self.output_root / "EVENTS.jsonl"
        self.events = events_path.open("x", encoding="utf-8", newline="\n")

    def event(self, value: dict[str, Any]) -> None:
        require(self.events is not None, "evidence event stream not open")
        self.events.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        self.events.flush()
        os.fsync(self.events.fileno())

    def process_receipt(self, phase: str, receipt: dict[str, Any]) -> None:
        names = {
            "pre_runtime": "PRE_RUNTIME_PROCESS_RECEIPT.json",
            "post_runtime": "POST_RUNTIME_PROCESS_RECEIPT.json",
        }
        require(phase in names, "unsupported target process-receipt phase")
        self._exclusive_json(names[phase], receipt)

    def _close_events(self) -> None:
        if self.events is not None:
            self.events.flush()
            os.fsync(self.events.fileno())
            self.events.close()
            self.events = None

    def complete(self, result: dict[str, Any]) -> None:
        self.event({"event": "runtime_complete", "status": result["status"]})
        self._close_events()
        self._exclusive_json("RESULT.json", result)

    def fail(self, reason: str) -> None:
        if self.events is not None:
            self.event({"event": "runtime_failed", "reason": reason})
        self._close_events()
        self._exclusive_json("FAILURE.json", {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FAILURE_V1",
            "reason": reason,
            "partial_evidence_preserved": True,
            "automatic_retry": False,
        })


class WorkerRuntime:
    """Single-call bridge to the packaged C worker's real Gate A backend."""

    def __init__(self, executable: Path, *, timeout_s: float = 20.0):
        self.executable = executable
        self.timeout_s = timeout_s
        self.calls = 0

    def execute(self, plan: FrozenPlan) -> dict[str, Any]:
        require(self.calls == 0, "runtime may be called at most once")
        self.calls += 1
        runtime_output = plan.output_root / "runtime"
        try:
            completed = subprocess.run([
                str(self.executable),
                "--execute-authorized",
                "--authority-sha256", plan.authority_sha256,
                "--schedule-sha256", SCHEDULE_SHA256,
                "--execution-bundle-sha256", plan.execution_bundle_sha256,
                "--output-root", str(runtime_output),
                "--sender-core", str(SENDER_CORE),
                "--receiver-core", str(RECEIVER_CORE),
                "--read-hz", str(READ_HZ),
                "--slot-s", str(SLOT_S),
                "--temperature-veto-c", str(TEMPERATURE_VETO_C),
                "--required-frequency-khz", str(REQUIRED_FREQUENCY_KHZ),
            ], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ExecutorError("worker exceeded the bounded target-local timeout") from exc
        require(completed.returncode == 0, f"worker failed: {completed.stderr.strip()}")
        value = json.loads(completed.stdout)
        require(isinstance(value, dict), "worker result must be an object")
        return value

    def verify_evidence(self, plan: FrozenPlan, result: dict[str, Any]) -> None:
        verify_retained_runtime_evidence(plan, result)
