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
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
FORBIDDEN_PROCESSES = (
    "combined_pdn_runner",
    "run_combined_campaign",
    "explicit_slot_runtime",
    "wrmsr",
    "rdmsr",
    "cpupower",
    "turbostat",
    "gate_a_worker",
)


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
class ProcessSnapshot:
    complete: bool
    return_code: int
    raw_stdout: str
    raw_stderr: str
    forbidden_hits: tuple[str, ...]


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
    def process_snapshot(self) -> ProcessSnapshot: ...
    def temperature_c(self) -> float: ...
    def frequency_khz(self, core: int) -> int: ...


class ClaimStore(Protocol):
    def claim(self, authority_sha256: str, plan: FrozenPlan) -> None: ...


class EvidenceStore(Protocol):
    def begin(self, plan: FrozenPlan, preflight: dict[str, Any]) -> None: ...
    def event(self, value: dict[str, Any]) -> None: ...
    def complete(self, result: dict[str, Any]) -> None: ...
    def fail(self, reason: str) -> None: ...


class RuntimeSurface(Protocol):
    def execute(self, plan: FrozenPlan) -> dict[str, Any]: ...


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
        snapshot = surfaces.preflight.process_snapshot()
        surfaces.evidence.event({
            "event": "process_preflight",
            "scan_complete": snapshot.complete,
            "return_code": snapshot.return_code,
            "raw_process_listing": snapshot.raw_stdout,
            "raw_process_stderr": snapshot.raw_stderr,
            "forbidden_process_hits": list(snapshot.forbidden_hits),
        })
        require(snapshot.complete and snapshot.return_code == 0, "process state unobservable")
        require(not snapshot.forbidden_hits, f"forbidden process present: {snapshot.forbidden_hits}")
        temperature = surfaces.preflight.temperature_c()
        surfaces.evidence.event({"event": "temperature_preflight", "temperature_c": temperature})
        require(math.isfinite(temperature) and temperature < TEMPERATURE_VETO_C, "temperature veto")
        frequencies = {str(core): surfaces.preflight.frequency_khz(core) for core in (SENDER_CORE, RECEIVER_CORE)}
        surfaces.evidence.event({"event": "frequency_preflight", "frequency_khz": frequencies})
        require(all(value == REQUIRED_FREQUENCY_KHZ for value in frequencies.values()), "frequency veto")
        surfaces.evidence.event({"event": "preflight_complete", "preflight_complete": True})
        surfaces.evidence.event({"event": "runtime_start", "runtime_execution_count": 1, "automatic_retry": False})
        result = surfaces.runtime.execute(plan)
        validate_runtime_result(result, plan)
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

    def process_snapshot(self) -> ProcessSnapshot:
        try:
            completed = subprocess.run(
                ["ps", "-eo", "pid,comm,args"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            return ProcessSnapshot(False, -1, "", str(exc), ())
        hits = tuple(marker for marker in FORBIDDEN_PROCESSES if marker in completed.stdout)
        return ProcessSnapshot(completed.returncode == 0, completed.returncode, completed.stdout, completed.stderr, hits)

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
