#!/usr/bin/env python3
"""Non-driving tests for the authority-gated Gate A engineering-smoke executor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import gate_a_hardware_adapter as host_adapter
import gate_a_engineering_smoke_executor as executor
import gate_a_engineering_smoke_transport as smoke_transport
import gate_a_process_custody as process_custody
import gate_a_target_runner

HERE = Path(__file__).resolve().parent
SCHEDULE_PATH = HERE.parent / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"
NAMESPACE_PATH = HERE.parent / "GATE_A_TARGET_NAMESPACE_BINDING.json"


def schedule() -> dict[str, Any]:
    return json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))


def authority_validation() -> dict[str, Any]:
    return {
        "status": "GATE_A_EXECUTION_AUTHORITY_EXACT",
        "reviewed_adapter_head": "1" * 40,
        "independent_review_id": 1,
        "execution_bundle_sha256": "b" * 64,
    }


def slot_record(index: int, token: str) -> dict[str, Any]:
    driven = token in {"S0E", "A0P", "A0N"}
    if token == "S0E":
        phase, sign, epoch = 0, 1, "gate-a:step:epoch0"
    elif token == "A0P":
        phase, sign, epoch = 0, 1, "gate-a:anchor:positive"
    elif token == "A0N":
        phase, sign, epoch = 4, -1, "gate-a:anchor:negative"
    else:
        phase = sign = epoch = None
    return {
        "index": index,
        "token": token,
        "requested_start_s": index * 0.5,
        "requested_end_s": (index + 1) * 0.5,
        "drive_on": driven,
        "amplitude_level": 2 if driven else None,
        "phase_index": phase,
        "sign": sign,
        "sender_epoch_id": epoch,
    }


def valid_runtime_result() -> dict[str, Any]:
    return {
        "status": "GATE_A_ENGINEERING_SMOKE_COMPLETE",
        "automatic_retry": False,
        "runtime_execution_count": 1,
        "slot_records": [slot_record(i, token) for i, token in enumerate(executor.SEQUENCE)],
        "capture": {
            "continuous": True,
            "covers_complete_sequence": True,
            "sample_count": 64000,
            "slot_sample_counts": [4000] * 16,
            "origin_tsc": 1_000_000_000,
            "deadline_tsc": 26_600_000_000,
            "first_sample_tsc": 1_000_000_000,
            "last_sample_tsc": 26_599_600_000,
            "tsc_hz": 3_200_000_000.0,
        },
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "step_sender_epoch_count": 1,
        "hardware_executed": True,
    }


def closed_process_receipt(phase: str) -> dict[str, Any]:
    completed = subprocess.CompletedProcess(
        list(process_custody.PROCESS_COMMAND),
        0,
        stdout=b"1 init /sbin/init\n2 kthreadd [kthreadd]\n",
        stderr=b"",
    )
    return process_custody.scan_processes(
        phase,
        runner=lambda *_args, **_kwargs: completed,
    )


def _lifecycle_record(
    *,
    record_type: str,
    event_tsc: int,
    slot: int,
    state: str,
    sender: dict[str, Any] | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    capture = result["capture"]
    start = capture["origin_tsc"] + int(slot * 0.5 * capture["tsc_hz"])
    end = capture["origin_tsc"] + int((slot + 1) * 0.5 * capture["tsc_hz"])
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_SENDER_LIFECYCLE_V1",
        "record_type": record_type,
        "event_tsc": event_tsc,
        "slot_index": slot,
        "token": executor.SEQUENCE[slot],
        "sender_state": state,
        "sender_epoch_id": sender["epoch"] if sender else None,
        "phase_index": sender["phase"] if sender else None,
        "sign": sender["sign"] if sender else None,
        "requested_start_tsc": start,
        "requested_end_tsc": end,
        "sender_transition_tsc": start + 3 if record_type == "slot_transition" and state == "active" else None,
        "thread_create_tsc": sender["create"] if sender else None,
        "thread_ready_tsc": sender["ready"] if sender else None,
        "epoch_start_tsc": sender["start"] if sender else None,
        "first_drive_tsc": sender["first_drive"] if sender else None,
        "stop_requested_tsc": sender["stop"] if sender else None,
        "thread_exit_tsc": sender["exit"] if sender else None,
        "thread_join_start_tsc": sender["join_start"] if sender else None,
        "thread_join_tsc": sender["join"] if sender else None,
    }


def write_synthetic_runtime_evidence(root: Path, result: dict[str, Any]) -> executor.FrozenPlan:
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    capture = result["capture"]
    spacing = int(capture["tsc_hz"] / executor.READ_HZ)
    frequency = executor._tone_zero_hz()
    raw: list[tuple[int, float]] = []
    for index in range(capture["sample_count"]):
        timestamp = capture["origin_tsc"] + index * spacing
        slot = index // executor.NOMINAL_SAMPLES_PER_SLOT
        value = 100.0 + (index % 17) * 0.001
        if executor.SEQUENCE[slot] in {"S0E", "A0P", "A0N"}:
            epoch_slot = 6 if executor.SEQUENCE[slot] == "S0E" else slot
            epoch_origin = capture["origin_tsc"] + int(epoch_slot * 0.5 * capture["tsc_hz"])
            phase = math.pi if executor.SEQUENCE[slot] == "A0N" else 0.0
            value += 0.25 * math.cos(2 * math.pi * frequency * ((timestamp - epoch_origin) / capture["tsc_hz"]) + phase)
        raw.append((timestamp, value))
    (runtime / "raw_samples.bin").write_bytes(b"".join(struct.pack("<Qd", *item) for item in raw))
    lockin_records = []
    for slot, token in enumerate(executor.SEQUENCE):
        start = slot * executor.NOMINAL_SAMPLES_PER_SLOT
        end = (slot + 1) * executor.NOMINAL_SAMPLES_PER_SLOT
        slot_start = capture["origin_tsc"] + int(slot * 0.5 * capture["tsc_hz"])
        slot_end = capture["origin_tsc"] + int((slot + 1) * 0.5 * capture["tsc_hz"])
        origin = capture["origin_tsc"] + int((6 if token == "S0E" else slot) * 0.5 * capture["tsc_hz"])
        values = executor._lockin_from_raw(raw[start:end], frequency=frequency, origin_tsc=origin, tsc_hz=capture["tsc_hz"])
        lockin_records.append({
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_LOCKIN_IQ_V1",
            "slot_index": slot,
            "token": token,
            "raw_sample_start_index": start,
            "raw_sample_end_index": end,
            "sample_count": end - start,
            "tone_frequency_hz": frequency,
            "lockin_i": values[0],
            "lockin_q": values[1],
            "magnitude": values[2],
            "off_frequency_floor": values[3],
            "origin_tsc": origin,
            "slot_start_tsc": slot_start,
            "slot_end_tsc": slot_end,
        })
    (runtime / "LOCKIN_IQ.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in lockin_records),
        encoding="utf-8",
    )
    def epoch(first: int, end: int, name: str, phase: int, sign: int) -> dict[str, Any]:
        start = capture["origin_tsc"] + int(first * 0.5 * capture["tsc_hz"])
        finish = capture["origin_tsc"] + int(end * 0.5 * capture["tsc_hz"])
        stop = finish - int(0.002 * capture["tsc_hz"])
        first_drive = start + 4
        if phase == 4:
            first_drive += int(0.5 / frequency * capture["tsc_hz"])
        return {
            "first": first, "end": end, "epoch": name, "phase": phase, "sign": sign,
            "create": start + 1, "ready": start + 2, "start": start + 3,
            "first_drive": first_drive, "stop": stop, "join_start": stop + 1,
            "exit": stop + 2, "join": stop + 3,
        }
    epochs = [
        epoch(6, 10, "gate-a:step:epoch0", 0, 1),
        epoch(12, 13, "gate-a:anchor:positive", 0, 1),
        epoch(13, 14, "gate-a:anchor:negative", 4, -1),
    ]
    lifecycle: list[dict[str, Any]] = []
    for value in epochs:
        for state, event in (
            ("starting", value["create"]),
            ("active", value["ready"]),
            ("stopping", value["stop"]),
            ("joined", value["join"]),
        ):
            lifecycle.append(_lifecycle_record(
                record_type="sender_state",
                event_tsc=event,
                slot=value["first"] if state in {"starting", "active"} else value["end"] - 1,
                state=state,
                sender=value,
                result=result,
            ))
    for slot, token in enumerate(executor.SEQUENCE):
        sender = None
        state = "not_created"
        if 6 <= slot <= 9:
            sender, state = epochs[0], "active"
        elif slot == 12:
            sender, state = epochs[1], "active"
        elif slot == 13:
            sender, state = epochs[2], "active"
        elif slot in {10, 11}:
            sender, state = epochs[0], "joined"
        elif slot in {14, 15}:
            sender, state = epochs[2], "joined"
        event = capture["origin_tsc"] + int(slot * 0.5 * capture["tsc_hz"])
        lifecycle.append(_lifecycle_record(
            record_type="slot_transition",
            event_tsc=event,
            slot=slot,
            state=state,
            sender=sender,
            result=result,
        ))
    (runtime / "SENDER_LIFECYCLE.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in lifecycle),
        encoding="utf-8",
    )
    return executor.FrozenPlan(
        authority_sha256="a" * 64,
        execution_bundle_sha256="b" * 64,
        output_root=root,
    )


def make_transport_request(root: Path) -> smoke_transport.HostExecutionRequest:
    authority_bytes = b"{}\n"
    authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    source_identity = {
        "role": "process_custody",
        "package_path": "adapter/gate_a_process_custody.py",
        "source_repository_path": "reviewed/adapter/gate_a_process_custody.py",
        "git_blob_sha1": "5" * 40,
        "git_mode": "100644",
        "sha256": "6" * 64,
        "byte_size": 123,
    }
    manifest = {
        "execution_bundle_sha256": "b" * 64,
        "target_namespace_sha256": "c" * 64,
        "deterministic_archive_sha256": "d" * 64,
        "target_identity_stdout_sha256": "e" * 64,
        "files": [source_identity],
    }
    remote_execution_root = "/root/catcas_phase6b6_gate_a_smoke_9c416379"
    remote_output_root = f"{remote_execution_root}/evidence"
    source_review_binding = {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_SOURCE_REVIEW_BINDING_V1",
        "reviewed_source_commit": "1" * 40,
        "reviewed_source_tree": "3" * 40,
        "independent_review_id": 123,
        "authority_bearing_execution_commit": "2" * 40,
        "authority_bearing_execution_tree": "4" * 40,
        "authority_sha256": authority_sha256,
        "authority_git_blob_sha1": "5" * 40,
        "source_identities": [source_identity],
        "schedule_sha256": executor.SCHEDULE_SHA256,
        "target_namespace_sha256": "c" * 64,
        "execution_bundle_sha256": "b" * 64,
        "deterministic_archive_sha256": "d" * 64,
        "target_identity_sha256": "e" * 64,
        "target": "root@192.168.137.100",
        "remote_execution_root": remote_execution_root,
        "remote_output_root": remote_output_root,
    }
    return smoke_transport.HostExecutionRequest(
        target="root@192.168.137.100",
        authority_path=root / "future-authority.json",
        authority_sha256=authority_sha256,
        reviewed_adapter_head="1" * 40,
        independent_review_id=123,
        execution_bundle_sha256="b" * 64,
        schedule_sha256=executor.SCHEDULE_SHA256,
        namespace_sha256="c" * 64,
        remote_execution_root=remote_execution_root,
        remote_output_root=remote_output_root,
        local_evidence_root=root / "host-evidence",
        authority_bytes=authority_bytes,
        schedule_bytes=(json.dumps({"schedule_sha256": executor.SCHEDULE_SHA256}) + "\n").encode(),
        manifest_bytes=(json.dumps(manifest) + "\n").encode(),
        source_review_binding=source_review_binding,
        authority_bearing_execution_commit="2" * 40,
        reviewed_source_tree="3" * 40,
        authority_bearing_execution_tree="4" * 40,
        authority_git_blob_sha1="5" * 40,
    )


class FakeTransportBackend:
    def __init__(
        self,
        request: smoke_transport.HostExecutionRequest,
        *,
        mismatch_inventory: bool = False,
        runner_return_code: int = 0,
        target_timeout: bool = False,
        archive_created: bool = True,
        runner_command_timeout: bool = False,
        recovery_output_available: bool = True,
        cleanup_return_code: int | None = None,
        cleanup_overrides: dict[str, Any] | None = None,
    ):
        self.request = request
        self.mismatch_inventory = mismatch_inventory
        self.runner_return_code = runner_return_code
        self.target_timeout = target_timeout
        self.archive_created = archive_created
        self.runner_command_timeout = runner_command_timeout
        self.recovery_output_available = recovery_output_available
        self.cleanup_return_code = cleanup_return_code
        self.cleanup_overrides = cleanup_overrides or {}
        self.calls: list[list[str]] = []
        self.runner_starts = 0
        self.claim_created = False
        self.copyback_receipt_uploaded = False
        self.network_connections = 0
        self.target_files = {
            "ATTEMPT.json": (json.dumps({
                "authority_sha256": request.authority_sha256,
                "execution_bundle_sha256": request.execution_bundle_sha256,
            }, sort_keys=True) + "\n").encode(),
            "POST_RUNTIME_PROCESS_RECEIPT.json": (json.dumps(closed_process_receipt("post_runtime"), sort_keys=True, indent=2) + "\n").encode(),
            "runtime/partial-or-complete.bin": b"retained target evidence",
        }

    def _target_inventory(self) -> tuple[dict[str, Any], str]:
        files = [
            {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in sorted(self.target_files.items())
        ]
        value = {"schema_id": "CAT_CAS_PHASE6B6_GATE_A_EVIDENCE_INVENTORY_V1", "files": files}
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if self.mismatch_inventory:
            value = copy.deepcopy(value)
            value["files"][0]["sha256"] = "0" * 64
        return value, digest

    def _write_archive(self, destination: Path) -> None:
        with tarfile.open(destination, "w") as archive:
            for name, data in sorted(self.target_files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                import io
                archive.addfile(info, io.BytesIO(data))

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        input_text = kwargs.get("input_text") or ""
        if argv[0] == "scp":
            if argv[1].startswith(self.request.target + ":"):
                self._write_archive(Path(argv[2]))
            elif argv[-1].endswith(".copy_back.json"):
                self.copyback_receipt_uploaded = True
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "inspection_complete" in input_text:
            value = {
                "inspection_complete": True,
                "execution_root": "absent", "output_root": "absent",
                "stage": "absent", "authority": "absent", "archive": "absent",
                "receipt": "absent", "target_inventory": "absent", "claim": "absent",
                "prefix_matches": [],
            }
        elif "claim_created" in input_text:
            self.claim_created = True
            value = {
                "claim_created": True,
                "claim_root": f"/root/.catcas_gate_a_claim_{self.request.authority_sha256}",
                "claim_sha256": smoke_transport._canonical_line_sha256(smoke_transport._expected_claim(self.request)),
            }
        elif "runner_command" in input_text:
            self.runner_starts += 1
            if self.runner_command_timeout:
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 180), output=b"", stderr=b"lost wrapper response")
            inventory, digest = self._target_inventory()
            runner_stdout = json.dumps(valid_runtime_result())
            value = {
                "runner_command": ["python3", "gate_a_target_runner.py"],
                "runner_return_code": self.runner_return_code,
                "runner_stdout": runner_stdout,
                "runner_stderr": "",
                "runner_stdout_sha256": hashlib.sha256(runner_stdout.encode()).hexdigest(),
                "runner_stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "target_timeout": self.target_timeout,
                "evidence_archive_created": self.archive_created,
                "target_evidence_inventory": inventory,
                "target_evidence_inventory_sha256": digest,
                "post_runtime_process_receipt": closed_process_receipt("post_runtime"),
            }
        elif "created_by_recovery" in input_text:
            inventory, digest = self._target_inventory()
            recovered = self.runner_starts > 0 and self.recovery_output_available
            value = {
                "evidence_archive_created": recovered,
                "created_by_recovery": recovered,
                "target_evidence_inventory": inventory if recovered else None,
                "target_evidence_inventory_sha256": digest if recovered else None,
                "post_runtime_process_receipt": closed_process_receipt("post_runtime") if recovered else None,
            }
        elif "cleanup_return_code" in input_text:
            expected_claim = smoke_transport._canonical_line_sha256(smoke_transport._expected_claim(self.request))
            expected_started = smoke_transport._canonical_line_sha256(smoke_transport._expected_execution_started(self.request))
            return_code = self.cleanup_return_code
            if return_code is None:
                allow_no_output = "allow_no_output_cleanup=True" in input_text
                if not self.claim_created:
                    return_code = 72
                elif not allow_no_output and not self.copyback_receipt_uploaded and (self.runner_starts == 0 or not self.recovery_output_available):
                    return_code = 72
                elif self.runner_starts and not self.copyback_receipt_uploaded:
                    return_code = 70
                else:
                    return_code = 0
            if return_code == 0:
                mode = "verified_copyback" if self.runner_starts else "no_output_created"
            elif return_code == 72:
                mode = "blocked_runner_start_unresolved"
            elif return_code == 71:
                mode = "blocked_available_evidence"
            else:
                mode = "blocked_unverified_copyback"
            inner = {
                "status": "GATE_A_CLEANUP_COMPLETE_AFTER_VERIFIED_COPY_BACK",
                "remote_output_root": self.request.remote_output_root,
                "claim_retained": True,
                "claim_sha256": expected_claim,
                "execution_started_sha256": expected_started,
            }
            value = {
                "cleanup_return_code": return_code, "cleanup_mode": mode,
                "cleanup_runner_stdout": json.dumps(inner) if mode == "verified_copyback" else "", "cleanup_runner_stderr": "",
                "execution_root_absent": return_code == 0, "output_root_absent": return_code == 0,
                "stage_absent": return_code == 0, "authority_absent": return_code == 0, "archive_absent": return_code == 0,
                "receipt_absent": return_code == 0, "target_inventory_absent": return_code == 0,
                "claim_retained": self.claim_created, "claim_sha256": expected_claim if self.claim_created else None,
                "execution_started_sha256": expected_started if self.runner_starts else None,
            }
            value.update(self.cleanup_overrides)
        elif "CAT_CAS_PHASE6B6_GATE_A_PROCESS_RECEIPT_V1" in input_text:
            value = closed_process_receipt("post_cleanup")
        else:
            value = {}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(value), stderr="")


class FakePreflight:
    def __init__(
        self,
        *,
        namespace: executor.NamespaceState = executor.NamespaceState.ABSENT,
        complete: bool = True,
        return_code: int = 0,
        hits: tuple[str, ...] = (),
        temperature: float = 42.0,
        frequencies: dict[int, int] | None = None,
        frequency_error: bool = False,
    ):
        self.namespace = namespace
        self.complete = complete
        self.return_code = return_code
        self.hits = hits
        self.temperature = temperature
        self.frequencies = frequencies or {4: 1600000, 5: 1600000}
        self.frequency_error = frequency_error
        self.calls: list[str] = []

    def inspect_namespace(self, path: Path) -> executor.NamespaceState:
        self.calls.append("namespace")
        return self.namespace

    def process_snapshot(self, phase: str) -> dict[str, Any]:
        self.calls.append(f"process:{phase}")
        lines = ["1 init /sbin/init"]
        for index, marker in enumerate(self.hits, 100):
            lines.append(f"{index} {marker} {marker} --test")
        completed = subprocess.CompletedProcess(
            list(process_custody.PROCESS_COMMAND),
            self.return_code,
            stdout=("\n".join(lines) + "\n").encode("utf-8"),
            stderr=b"simulated scanner failure" if self.return_code else b"",
        )
        receipt = process_custody.scan_processes(
            phase,
            runner=lambda *_args, **_kwargs: completed,
        )
        if not self.complete and self.return_code == 0:
            receipt["scan_complete"] = False
            receipt["forbidden_filter_evaluated"] = False
            receipt["failure"] = "SIMULATED_UNOBSERVABLE"
        return receipt

    def temperature_c(self) -> float:
        self.calls.append("temperature")
        return self.temperature

    def frequency_khz(self, core: int) -> int:
        self.calls.append(f"frequency:{core}")
        if self.frequency_error:
            raise executor.ExecutorError("frequency unobservable")
        return self.frequencies[core]


class FakeClaims:
    def __init__(self):
        self.claimed: set[str] = set()
        self.calls = 0

    def claim(self, authority_sha256: str, plan: executor.FrozenPlan) -> None:
        self.calls += 1
        if authority_sha256 in self.claimed:
            raise executor.ExecutorError("authority execution already claimed")
        self.claimed.add(authority_sha256)


class FakeEvidence:
    def __init__(self):
        self.begun = 0
        self.events: list[dict[str, Any]] = []
        self.completed = 0
        self.failures: list[str] = []
        self.process_receipts: dict[str, dict[str, Any]] = {}

    def begin(self, plan: executor.FrozenPlan, preflight: dict[str, Any]) -> None:
        self.begun += 1

    def event(self, value: dict[str, Any]) -> None:
        self.events.append(value)

    def process_receipt(self, phase: str, receipt: dict[str, Any]) -> None:
        self.process_receipts[phase] = copy.deepcopy(receipt)

    def complete(self, result: dict[str, Any]) -> None:
        self.completed += 1

    def fail(self, reason: str) -> None:
        self.failures.append(reason)


class FakeRuntime:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None):
        self.result = result or valid_runtime_result()
        self.error = error
        self.calls = 0
        self.network_calls = 0
        self.real_hardware_calls = 0
        self.msr_calls = 0
        self.control_write_calls = 0
        self.evidence_verifications = 0

    def execute(self, plan: executor.FrozenPlan) -> dict[str, Any]:
        self.calls += 1
        if self.error:
            raise self.error
        return copy.deepcopy(self.result)

    def verify_evidence(self, plan: executor.FrozenPlan, result: dict[str, Any]) -> None:
        self.evidence_verifications += 1


def run_once(
    *,
    preflight: FakePreflight | None = None,
    claims: FakeClaims | None = None,
    evidence: Any | None = None,
    runtime: FakeRuntime | None = None,
    output_root: Path = Path("/fake/evidence"),
    validation: dict[str, Any] | None = None,
    authority_sha256: str = "a" * 64,
) -> tuple[dict[str, Any], FakeClaims, Any, FakeRuntime]:
    use_claims = claims or FakeClaims()
    use_evidence = evidence or FakeEvidence()
    use_runtime = runtime or FakeRuntime()
    result = executor.execute_once(
        authority_validation=validation or authority_validation(),
        authority_sha256=authority_sha256,
        execution_bundle_sha256="b" * 64,
        schedule=schedule(),
        output_root=output_root,
        surfaces=executor.ExecutionSurfaces(
            preflight=preflight or FakePreflight(),
            claims=use_claims,
            evidence=use_evidence,
            runtime=use_runtime,
        ),
    )
    return result, use_claims, use_evidence, use_runtime


class GateAExecutorTests(unittest.TestCase):
    def assert_executor_rejects(self, **kwargs: Any) -> None:
        with self.assertRaises(Exception):
            run_once(**kwargs)

    def test_frozen_schedule_and_geometry_validate(self) -> None:
        executor.validate_frozen_schedule(schedule())

    def test_no_authority_rejects_before_any_preflight_or_runtime(self) -> None:
        preflight, claims, evidence, runtime = FakePreflight(), FakeClaims(), FakeEvidence(), FakeRuntime()
        self.assert_executor_rejects(
            preflight=preflight,
            claims=claims,
            evidence=evidence,
            runtime=runtime,
            validation={"status": "NOT_AUTHORIZED"},
        )
        self.assertEqual(preflight.calls, [])
        self.assertEqual(claims.calls, 0)
        self.assertEqual(runtime.calls, 0)

    def test_incorrect_digest_binding_rejects(self) -> None:
        self.assert_executor_rejects(authority_sha256="x" * 64)

    def test_present_or_unobservable_namespace_rejects_before_claim(self) -> None:
        for state in (executor.NamespaceState.PRESENT, executor.NamespaceState.UNOBSERVABLE):
            with self.subTest(state=state):
                claims, runtime = FakeClaims(), FakeRuntime()
                self.assert_executor_rejects(preflight=FakePreflight(namespace=state), claims=claims, runtime=runtime)
                self.assertEqual(claims.calls, 0)
                self.assertEqual(runtime.calls, 0)

    def test_temperature_and_frequency_vetoes_consume_without_runtime(self) -> None:
        cases = (
            FakePreflight(temperature=68.0),
            FakePreflight(temperature=math.nan),
            FakePreflight(frequencies={4: 1600000, 5: 1599999}),
            FakePreflight(frequency_error=True),
        )
        for preflight in cases:
            with self.subTest(calls=preflight.calls):
                claims, runtime = FakeClaims(), FakeRuntime()
                self.assert_executor_rejects(preflight=preflight, claims=claims, runtime=runtime)
                self.assertEqual(claims.calls, 1)
                self.assertEqual(runtime.calls, 0)

    def test_forbidden_or_unobservable_process_state_rejects(self) -> None:
        cases = (
            FakePreflight(hits=("cpupower",)),
            FakePreflight(complete=False, return_code=7),
        )
        for preflight in cases:
            claims, runtime = FakeClaims(), FakeRuntime()
            self.assert_executor_rejects(preflight=preflight, claims=claims, runtime=runtime)
            self.assertEqual(claims.calls, 1)
            self.assertEqual(runtime.calls, 0)

    def test_second_execution_and_retry_are_impossible(self) -> None:
        claims, runtime = FakeClaims(), FakeRuntime()
        run_once(claims=claims, runtime=runtime)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(runtime.evidence_verifications, 1)
        self.assert_executor_rejects(claims=claims, runtime=runtime, output_root=Path("/fake/evidence2"))
        self.assertEqual(runtime.calls, 1)

    def test_runtime_failure_has_no_retry(self) -> None:
        claims, evidence = FakeClaims(), FakeEvidence()
        runtime = FakeRuntime(error=RuntimeError("simulated capture failure"))
        self.assert_executor_rejects(claims=claims, evidence=evidence, runtime=runtime)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(len(evidence.failures), 1)
        self.assertIn("post_runtime", evidence.process_receipts)
        self.assert_executor_rejects(claims=claims, evidence=evidence, runtime=runtime, output_root=Path("/fake/evidence2"))
        self.assertEqual(runtime.calls, 1)

    def test_off_sham_step_anchor_sequence_timing_and_capture_invariants(self) -> None:
        result, _, _, runtime = run_once()
        self.assertEqual(runtime.calls, 1)
        self.assertEqual([r["token"] for r in result["slot_records"]], list(executor.SEQUENCE))
        self.assertTrue(all(not result["slot_records"][i]["drive_on"] for i in (0, 1, 2, 3, 4, 5, 10, 11, 14, 15)))
        self.assertEqual({result["slot_records"][i]["sender_epoch_id"] for i in range(6, 10)}, {"gate-a:step:epoch0"})
        self.assertEqual((result["slot_records"][12]["sign"], result["slot_records"][12]["phase_index"]), (1, 0))
        self.assertEqual((result["slot_records"][13]["sign"], result["slot_records"][13]["phase_index"]), (-1, 4))
        self.assertTrue(result["capture"]["covers_complete_sequence"])
        self.assertEqual(runtime.network_calls + runtime.real_hardware_calls + runtime.msr_calls + runtime.control_write_calls, 0)

    def test_shared_process_custody_success_failures_and_mutations(self) -> None:
        receipt = closed_process_receipt("pre_runtime")
        process_custody.validate_process_receipt(receipt, expected_phase="pre_runtime")
        self.assertEqual(receipt["exact_command"], list(process_custody.PROCESS_COMMAND))
        self.assertEqual(receipt["stdout_sha256"], hashlib.sha256(receipt["raw_stdout"].encode()).hexdigest())
        mutations = []
        changed = copy.deepcopy(receipt); changed["exact_command"] = ["ps"]; mutations.append(changed)
        changed = copy.deepcopy(receipt); changed["command_sha256"] = "0" * 64; mutations.append(changed)
        changed = copy.deepcopy(receipt); changed["stdout_sha256"] = "0" * 64; mutations.append(changed)
        changed = copy.deepcopy(receipt); changed["scan_complete"] = False; mutations.append(changed)
        changed = copy.deepcopy(receipt); changed["parsed_forbidden_hits"] = [{"marker": "forged"}]; mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(process_custody.ProcessCustodyError):
                    process_custody.validate_process_receipt(changed, expected_phase="pre_runtime")

        def timeout(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
            raise subprocess.TimeoutExpired(process_custody.PROCESS_COMMAND, 1, output=b"partial", stderr=b"late")
        def unavailable(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
            raise PermissionError("ps unavailable")
        cases = [
            process_custody.scan_processes("pre_runtime", runner=timeout),
            process_custody.scan_processes("pre_runtime", runner=unavailable),
            process_custody.scan_processes("pre_runtime", runner=lambda *_a, **_k: subprocess.CompletedProcess([], 7, stdout=b"", stderr=b"fail")),
            process_custody.scan_processes("pre_runtime", runner=lambda *_a, **_k: subprocess.CompletedProcess([], 0, stdout=b"malformed\n", stderr=b"")),
            process_custody.scan_processes("pre_runtime", runner=lambda *_a, **_k: subprocess.CompletedProcess([], 0, stdout=b"99 gate_a_worker gate_a_worker --execute-authorized\n", stderr=b"")),
        ]
        for failed in cases:
            with self.subTest(failed=failed):
                with self.assertRaises(process_custody.ProcessCustodyError):
                    process_custody.validate_process_receipt(failed, expected_phase="pre_runtime")

    def test_raw_lockin_recomputes_and_mutations_reject(self) -> None:
        result = valid_runtime_result()
        with tempfile.TemporaryDirectory(prefix="gate_a_lockin_custody_") as tmp:
            root = Path(tmp) / "evidence"
            plan = write_synthetic_runtime_evidence(root, result)
            executor.verify_retained_runtime_evidence(plan, result)
            records_path = root / "runtime/LOCKIN_IQ.jsonl"
            original = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
            self.assertGreater(original[12]["lockin_i"], 0)
            self.assertLess(original[13]["lockin_i"], 0)
            mutations = (
                (12, "lockin_i", original[12]["lockin_i"] + 1),
                (12, "lockin_q", original[12]["lockin_q"] + 1),
                (12, "magnitude", original[12]["magnitude"] + 1),
                (12, "off_frequency_floor", original[12]["off_frequency_floor"] + 1),
                (7, "raw_sample_start_index", original[7]["raw_sample_start_index"] + 1),
                (7, "tone_frequency_hz", original[7]["tone_frequency_hz"] + 1),
                (7, "token", "O0"),
                (7, "slot_index", 99),
                (7, "origin_tsc", original[7]["origin_tsc"] + 1),
            )
            for slot, field, value in mutations:
                with self.subTest(slot=slot, field=field):
                    changed = copy.deepcopy(original)
                    changed[slot][field] = value
                    records_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in changed), encoding="utf-8")
                    with self.assertRaises(executor.ExecutorError):
                        executor.verify_retained_runtime_evidence(plan, result)
            records_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in original), encoding="utf-8")
            raw_path = root / "runtime/raw_samples.bin"
            original_raw = raw_path.read_bytes()
            raw_records = list(struct.iter_unpack("<Qd", original_raw))
            positive_start = 12 * executor.NOMINAL_SAMPLES_PER_SLOT
            negative_start = 13 * executor.NOMINAL_SAMPLES_PER_SLOT
            for offset in range(executor.NOMINAL_SAMPLES_PER_SLOT):
                timestamp = raw_records[negative_start + offset][0]
                raw_records[negative_start + offset] = (timestamp, raw_records[positive_start + offset][1])
            raw_path.write_bytes(b"".join(struct.pack("<Qd", *item) for item in raw_records))
            coherent = copy.deepcopy(original)
            selected = raw_records[negative_start:negative_start + executor.NOMINAL_SAMPLES_PER_SLOT]
            values = executor._lockin_from_raw(
                selected,
                frequency=coherent[13]["tone_frequency_hz"],
                origin_tsc=coherent[13]["origin_tsc"],
                tsc_hz=result["capture"]["tsc_hz"],
            )
            for field, value in zip(("lockin_i", "lockin_q", "magnitude", "off_frequency_floor"), values):
                coherent[13][field] = value
            records_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in coherent), encoding="utf-8")
            with self.assertRaises(executor.ExecutorError):
                executor.verify_retained_runtime_evidence(plan, result)
            records_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in original), encoding="utf-8")
            raw_path.write_bytes(original_raw)
            raw = bytearray(original_raw)
            raw[-1] ^= 1
            raw_path.write_bytes(raw)
            with self.assertRaises(executor.ExecutorError):
                executor.verify_retained_runtime_evidence(plan, result)

    def test_sender_lifecycle_proves_absence_epochs_and_continuous_capture(self) -> None:
        result = valid_runtime_result()
        with tempfile.TemporaryDirectory(prefix="gate_a_lifecycle_custody_") as tmp:
            root = Path(tmp) / "evidence"
            plan = write_synthetic_runtime_evidence(root, result)
            executor.verify_retained_runtime_evidence(plan, result)
            path = root / "runtime/SENDER_LIFECYCLE.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            transitions = {record["slot_index"]: record for record in records if record["record_type"] == "slot_transition"}
            self.assertTrue(all(transitions[index]["sender_state"] == "not_created" for index in range(6)))
            self.assertTrue(all(transitions[index]["sender_state"] == "joined" for index in (10, 11, 14, 15)))
            self.assertEqual({transitions[index]["sender_epoch_id"] for index in range(6, 10)}, {"gate-a:step:epoch0"})
            self.assertNotEqual(transitions[12]["sender_epoch_id"], transitions[13]["sender_epoch_id"])
            self.assertEqual((transitions[12]["phase_index"], transitions[12]["sign"]), (0, 1))
            self.assertEqual((transitions[13]["phase_index"], transitions[13]["sign"]), (4, -1))
            self.assertTrue(result["capture"]["continuous"] and result["capture"]["covers_complete_sequence"])
            mutations = []
            changed = copy.deepcopy(records); next(item for item in changed if item["record_type"] == "slot_transition" and item["slot_index"] == 4)["sender_state"] = "active"; mutations.append(changed)
            changed = copy.deepcopy(records); next(item for item in changed if item["record_type"] == "slot_transition" and item["slot_index"] == 8)["sender_epoch_id"] = "gate-a:step:epoch1"; mutations.append(changed)
            changed = copy.deepcopy(records); next(item for item in changed if item["record_type"] == "slot_transition" and item["slot_index"] == 10)["thread_join_tsc"] += int(result["capture"]["tsc_hz"]); mutations.append(changed)
            changed = copy.deepcopy(records); next(item for item in changed if item["record_type"] == "slot_transition" and item["slot_index"] == 13)["phase_index"] = 0; mutations.append(changed)
            changed = copy.deepcopy(records); next(item for item in changed if item["record_type"] == "slot_transition" and item["slot_index"] == 8)["sender_transition_tsc"] += int(result["capture"]["tsc_hz"]); mutations.append(changed)
            for changed in mutations:
                path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in changed), encoding="utf-8")
                with self.assertRaises(executor.ExecutorError):
                    executor.verify_retained_runtime_evidence(plan, result)

    def test_pre_and_post_runtime_process_receipts_are_mandatory_on_success_and_failure(self) -> None:
        evidence = FakeEvidence()
        result, _, evidence, _ = run_once(evidence=evidence)
        self.assertEqual(set(evidence.process_receipts), {"pre_runtime", "post_runtime"})
        self.assertEqual(result["status"], "GATE_A_ENGINEERING_SMOKE_COMPLETE")
        failing_evidence = FakeEvidence()
        with self.assertRaises(RuntimeError):
            run_once(evidence=failing_evidence, runtime=FakeRuntime(error=RuntimeError("worker failed")))
        self.assertEqual(set(failing_evidence.process_receipts), {"pre_runtime", "post_runtime"})

    def test_complete_host_packet_and_inventory_are_retained(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_host_packet_") as tmp:
            request = make_transport_request(Path(tmp))
            backend = FakeTransportBackend(request)
            transport = smoke_transport.SshScpTransport(command_runner=backend)
            with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                result = transport.execute(request)
            self.assertEqual(result["status"], "GATE_A_AUTHORIZED_TRANSPORT_COMPLETE")
            self.assertEqual(result["retry_count"], 0)
            self.assertEqual(backend.runner_starts, 1)
            self.assertEqual(backend.network_connections, 0)
            packet = request.local_evidence_root
            self.assertTrue(smoke_transport.HostEvidencePacket.REQUIRED_SUCCESS_FILES <= {path.name for path in packet.iterdir() if path.is_file()})
            self.assertTrue((packet / "TARGET_OUTPUT/runtime/partial-or-complete.bin").is_file())
            smoke_transport.validate_final_packet(packet)
            commands = [json.loads(line) for line in (packet / "HOST_COMMANDS.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["sequence"] for item in commands], list(range(1, len(commands) + 1)))
            self.assertEqual(len({item["operation"] for item in commands}), len(commands))
            for item in commands:
                self.assertEqual(item["stdout_sha256"], hashlib.sha256(item["raw_stdout"].encode()).hexdigest())
                self.assertEqual(item["stderr_sha256"], hashlib.sha256(item["raw_stderr"].encode()).hexdigest())
            self.assertLess(
                next(item["sequence"] for item in commands if item["operation"] == "copy_back_receipt_upload"),
                next(item["sequence"] for item in commands if item["operation"] == "remote_cleanup_attempted"),
            )
            (packet / "TARGET_OUTPUT/runtime/partial-or-complete.bin").write_bytes(b"altered")
            with self.assertRaises(smoke_transport.TransportError):
                smoke_transport.validate_final_packet(packet)

    def test_source_review_binding_rejects_every_bound_category_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_source_binding_") as tmp:
            request = make_transport_request(Path(tmp))
            manifest = json.loads(request.manifest_bytes)
            smoke_transport.validate_source_review_binding(
                request.source_review_binding,
                request=request,
                manifest=manifest,
            )
            cases: list[tuple[str, Any]] = [
                ("schema_id", "WRONG"),
                ("reviewed_source_commit", "0" * 40),
                ("reviewed_source_tree", "0" * 40),
                ("independent_review_id", 124),
                ("authority_bearing_execution_commit", "0" * 40),
                ("authority_bearing_execution_tree", "0" * 40),
                ("authority_sha256", "0" * 64),
                ("authority_git_blob_sha1", "0" * 40),
                ("schedule_sha256", "0" * 64),
                ("target_namespace_sha256", "0" * 64),
                ("execution_bundle_sha256", "0" * 64),
                ("deterministic_archive_sha256", "0" * 64),
                ("target_identity_sha256", "0" * 64),
                ("target", "root@127.0.0.1"),
                ("remote_execution_root", "/root/wrong"),
                ("remote_output_root", "/root/wrong/evidence"),
            ]
            changed = copy.deepcopy(request.source_review_binding)
            changed["source_identities"][0]["sha256"] = "0" * 64
            cases.append(("source_identities", changed["source_identities"]))
            for field, replacement in cases:
                with self.subTest(field=field):
                    changed = copy.deepcopy(request.source_review_binding)
                    changed[field] = replacement
                    with self.assertRaises(smoke_transport.TransportError):
                        smoke_transport.validate_source_review_binding(
                            changed,
                            request=request,
                            manifest=manifest,
                        )

    def test_cleanup_requires_every_closed_absence_and_claim_proof(self) -> None:
        cases = (
            {"execution_root_absent": False},
            {"output_root_absent": False},
            {"stage_absent": False},
            {"authority_absent": False},
            {"archive_absent": False},
            {"receipt_absent": False},
            {"target_inventory_absent": False},
            {"claim_retained": False},
            {"claim_sha256": "0" * 64},
            {"execution_started_sha256": "0" * 64},
            {"cleanup_runner_stdout": "not-json"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory(prefix="gate_a_cleanup_closed_") as tmp:
                request = make_transport_request(Path(tmp))
                backend = FakeTransportBackend(request, cleanup_overrides=overrides)
                transport = smoke_transport.SshScpTransport(command_runner=backend)
                with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                    with self.assertRaises(smoke_transport.TransportError):
                        transport.execute(request)
                receipt = json.loads((request.local_evidence_root / "CLEANUP_RECEIPT.json").read_text(encoding="utf-8"))
                self.assertEqual(receipt["parsed"].get(next(iter(overrides))), next(iter(overrides.values())))
                failure = json.loads((request.local_evidence_root / "TRANSPORT_FAILURE_RECEIPT.json").read_text(encoding="utf-8"))
                self.assertEqual(failure["failed_stage"], "remote_cleanup_attempted")
                self.assertFalse(failure["authority_claim_preserved"])

    def test_lost_target_wrapper_response_recovers_evidence_and_post_runtime_receipt_without_retry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_lost_wrapper_") as tmp:
            request = make_transport_request(Path(tmp))
            backend = FakeTransportBackend(request, runner_command_timeout=True)
            transport = smoke_transport.SshScpTransport(command_runner=backend)
            with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                with self.assertRaises(smoke_transport.TransportError):
                    transport.execute(request)
            packet = request.local_evidence_root
            failure = json.loads((packet / "TRANSPORT_FAILURE_RECEIPT.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["failed_stage"], "target_runner_started")
            self.assertTrue(failure["copy_back_verified"])
            self.assertEqual(failure["runner_start_count"], 1)
            post = json.loads((packet / "POST_RUNTIME_PROCESS_RECEIPT.json").read_text(encoding="utf-8"))
            process_custody.validate_process_receipt(post, expected_phase="post_runtime")
            commands = [json.loads(line) for line in (packet / "HOST_COMMANDS.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(item["operation"] == "target_runner_started" for item in commands), 1)
            self.assertIn("recovery_evidence_download", {item["operation"] for item in commands})

    def test_lost_wrapper_with_no_output_yet_blocks_cleanup_deletion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_lost_wrapper_no_output_") as tmp:
            request = make_transport_request(Path(tmp))
            backend = FakeTransportBackend(
                request,
                runner_command_timeout=True,
                recovery_output_available=False,
            )
            transport = smoke_transport.SshScpTransport(command_runner=backend)
            with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                with self.assertRaises(smoke_transport.TransportError):
                    transport.execute(request)
            packet = request.local_evidence_root
            cleanup = json.loads((packet / "CLEANUP_RECEIPT.json").read_text(encoding="utf-8"))["parsed"]
            self.assertEqual(cleanup["cleanup_mode"], "blocked_runner_start_unresolved")
            self.assertFalse(cleanup["execution_root_absent"])
            self.assertFalse(cleanup["stage_absent"])
            failure = json.loads((packet / "TRANSPORT_FAILURE_RECEIPT.json").read_text(encoding="utf-8"))
            self.assertTrue(failure["cleanup_attempted"])
            self.assertFalse(failure["copy_back_verified"])
            self.assertEqual(failure["runner_start_count"], 1)
            self.assertTrue(any(item.startswith("cleanup:") for item in failure["secondary_errors"]))

    def test_available_remote_archive_is_not_deleted_without_verified_copyback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_available_archive_") as tmp:
            request = make_transport_request(Path(tmp))
            backend = FakeTransportBackend(request, runner_command_timeout=True, cleanup_return_code=71)
            transport = smoke_transport.SshScpTransport(command_runner=backend)
            self.assertIn("blocked_available_evidence", transport._cleanup_script(request, transport._paths(request)))
            with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                with self.assertRaises(smoke_transport.TransportError):
                    transport.execute(request)
            cleanup = json.loads((request.local_evidence_root / "CLEANUP_RECEIPT.json").read_text(encoding="utf-8"))["parsed"]
            self.assertEqual(cleanup["cleanup_mode"], "blocked_available_evidence")
            self.assertFalse(cleanup["archive_absent"])
            failure = json.loads((request.local_evidence_root / "TRANSPORT_FAILURE_RECEIPT.json").read_text(encoding="utf-8"))
            self.assertTrue(any(item.startswith("cleanup:") for item in failure["secondary_errors"]))

    def test_transport_failure_injection_seals_without_retry(self) -> None:
        points = (
            "deployment_archive_build", "remote_namespace_inspected",
            "authority_claimed", "bundle_staged", "authority_staged",
            "target_runner_started",
            "evidence_archive_creation", "evidence_download", "safe_extract",
            "target_inventory_verification", "copy_back_receipt_persist",
            "copy_back_receipt_upload",
            "remote_cleanup_attempted", "post_cleanup_process_scan",
            "cleanup_verification", "target_result_verification",
            "final_local_seal",
        )
        for point in points:
            with self.subTest(point=point), tempfile.TemporaryDirectory(prefix="gate_a_transport_failure_") as tmp:
                request = make_transport_request(Path(tmp))
                backend = FakeTransportBackend(request)
                fired = False
                def inject(name: str) -> None:
                    nonlocal fired
                    if name == point and not fired:
                        fired = True
                        raise RuntimeError(f"injected:{point}")
                transport = smoke_transport.SshScpTransport(command_runner=backend, failure_injector=inject)
                with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                    with self.assertRaises(smoke_transport.TransportError):
                        transport.execute(request)
                self.assertTrue(fired)
                packet = request.local_evidence_root
                failure = json.loads((packet / "TRANSPORT_FAILURE_RECEIPT.json").read_text(encoding="utf-8"))
                self.assertEqual(failure["failed_stage"], point)
                self.assertEqual(failure["retry_count"], 0)
                self.assertFalse(failure["automatic_retry"])
                self.assertLessEqual(failure["runner_start_count"], 1)
                self.assertIn(failure["authority_claim_state"], {"not_attempted", "uncertain", "confirmed"})
                if point == "authority_claimed":
                    self.assertFalse(failure["authority_claim_preserved"])
                self.assertTrue((packet / "FINAL_EVIDENCE_INVENTORY.json").is_file())
                smoke_transport.validate_final_packet(packet)
                commands = [json.loads(line) for line in (packet / "HOST_COMMANDS.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len({item["operation"] for item in commands}), len(commands))
                self.assertLessEqual(backend.runner_starts, 1)

    def test_transport_timeout_nonzero_archive_and_inventory_fail_closed_after_custody(self) -> None:
        cases = (
            {"target_timeout": True},
            {"runner_return_code": 9},
            {"archive_created": False},
            {"mismatch_inventory": True},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory(prefix="gate_a_transport_result_failure_") as tmp:
                request = make_transport_request(Path(tmp))
                backend = FakeTransportBackend(request, **kwargs)
                transport = smoke_transport.SshScpTransport(command_runner=backend)
                with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                    with self.assertRaises(smoke_transport.TransportError):
                        transport.execute(request)
                packet = request.local_evidence_root
                self.assertTrue((packet / "TRANSPORT_FAILURE_RECEIPT.json").is_file())
                self.assertTrue((packet / "COPY_BACK_RECEIPT.json").is_file())
                self.assertTrue((packet / "CLEANUP_RECEIPT.json").is_file())
                self.assertTrue((packet / "POST_CLEANUP_PROCESS_RECEIPT.json").is_file())
                self.assertEqual(backend.runner_starts, 1)

    def test_runtime_result_mutations_fail_closed(self) -> None:
        mutations = []
        wrong = valid_runtime_result(); wrong["slot_records"][5]["drive_on"] = True; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["slot_records"][8]["sender_epoch_id"] = "epoch1"; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["slot_records"][13]["phase_index"] = 0; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["slot_records"][7]["requested_start_s"] = 99.0; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["capture"]["covers_complete_sequence"] = False; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["capture"]["slot_sample_counts"][15] = 1; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["capture"]["first_sample_tsc"] += 10_000_000; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["capture"]["last_sample_tsc"] -= 10_000_000; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["frequency_writes"] = 1; mutations.append(wrong)
        wrong = valid_runtime_result(); wrong["automatic_retry"] = True; mutations.append(wrong)
        for result in mutations:
            with self.subTest(result=result):
                self.assert_executor_rejects(runtime=FakeRuntime(result=result))

    def test_partial_evidence_survives_simulated_runtime_failure(self) -> None:
        class PartialRuntime(FakeRuntime):
            def execute(self, plan: executor.FrozenPlan) -> dict[str, Any]:
                self.calls += 1
                runtime_root = plan.output_root / "runtime"
                runtime_root.mkdir()
                (runtime_root / "partial-slot-00.bin").write_bytes(b"partial")
                raise RuntimeError("simulated slot failure")

        with tempfile.TemporaryDirectory(prefix="gate_a_partial_test_") as tmp:
            output = Path(tmp) / "evidence"
            evidence = executor.JsonEvidenceStore(output)
            runtime = PartialRuntime()
            self.assert_executor_rejects(evidence=evidence, runtime=runtime, output_root=output)
            self.assertTrue((output / "ATTEMPT.json").is_file())
            self.assertTrue((output / "EVENTS.jsonl").is_file())
            self.assertTrue((output / "FAILURE.json").is_file())
            self.assertEqual((output / "runtime/partial-slot-00.bin").read_bytes(), b"partial")

    def test_consumed_preflight_failure_preserves_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_preflight_evidence_") as tmp:
            output = Path(tmp) / "evidence"
            evidence = executor.JsonEvidenceStore(output)
            preflight = FakePreflight(complete=False, return_code=7)
            with self.assertRaises(Exception):
                run_once(preflight=preflight, evidence=evidence, output_root=output)
            self.assertTrue((output / "ATTEMPT.json").is_file())
            self.assertTrue((output / "FAILURE.json").is_file())
            receipt = json.loads((output / "PRE_RUNTIME_PROCESS_RECEIPT.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["return_code"], 7)
            self.assertIn("simulated scanner failure", receipt["raw_stderr"])
            self.assertIn("runtime_failed", (output / "EVENTS.jsonl").read_text(encoding="utf-8"))

    def test_worker_timeout_is_one_shot_and_evidenced(self) -> None:
        runtime = executor.WorkerRuntime(Path("/fake/gate_a_worker"), timeout_s=0.01)
        evidence = FakeEvidence()
        with mock.patch.object(
            executor.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["gate_a_worker"], 0.01),
        ):
            with self.assertRaises(executor.ExecutorError):
                run_once(runtime=runtime, evidence=evidence)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(len(evidence.failures), 1)

    def test_cleanup_requires_closed_verified_copy_back(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_cleanup_test_") as tmp:
            root = Path(tmp) / "evidence"
            root.mkdir()
            args = argparse.Namespace(copy_back_receipt=None, output_root=str(root))
            with self.assertRaises(Exception):
                gate_a_target_runner.cleanup_after_verified_copy(args, expected_output_root=str(root))
            self.assertTrue(root.exists())
            (root / "evidence.bin").write_bytes(b"exact evidence")
            args.authority_sha256 = "a" * 64
            args.execution_bundle_sha256 = "b" * 64
            claim_root = Path(tmp) / "claim"
            claim_root.mkdir()
            claim_value = gate_a_target_runner.expected_transport_claim(
                args.authority_sha256, args.execution_bundle_sha256,
            )
            marker_value = gate_a_target_runner.expected_execution_started(
                args.authority_sha256, args.execution_bundle_sha256,
            )
            (claim_root / "CLAIM.json").write_text(json.dumps(claim_value), encoding="utf-8")
            (claim_root / "EXECUTION_STARTED.json").write_text(json.dumps(marker_value), encoding="utf-8")
            args.transport_claim_root = str(claim_root)
            (root / "ATTEMPT.json").write_text(json.dumps({
                "authority_sha256": args.authority_sha256,
                "execution_bundle_sha256": args.execution_bundle_sha256,
            }), encoding="utf-8")
            receipt = Path(tmp) / "receipt.json"
            receipt.write_text(json.dumps({
                "schema_id": "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1",
                "remote_output_root": str(root),
                "authority_sha256": args.authority_sha256,
                "execution_bundle_sha256": args.execution_bundle_sha256,
                "retained_evidence_custody_verified": False,
                "evidence_inventory_sha256": "c" * 64,
                "target_evidence_inventory_sha256": "c" * 64,
                "downloaded_evidence_inventory_sha256": "c" * 64,
                "archive_sha256": "d" * 64,
                "copy_back_complete": True,
            }), encoding="utf-8")
            args.copy_back_receipt = str(receipt)
            with self.assertRaises(Exception):
                gate_a_target_runner.cleanup_after_verified_copy(args, expected_output_root=str(root))
            self.assertTrue(root.exists())
            value = json.loads(receipt.read_text(encoding="utf-8"))
            value["retained_evidence_custody_verified"] = True
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(Exception):
                gate_a_target_runner.cleanup_after_verified_copy(args, expected_output_root=str(root))
            self.assertTrue(root.exists())
            value["evidence_inventory_sha256"] = gate_a_target_runner.evidence_inventory_sha256(root)
            value["target_evidence_inventory_sha256"] = value["evidence_inventory_sha256"]
            value["downloaded_evidence_inventory_sha256"] = value["evidence_inventory_sha256"]
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(gate_a_target_runner, "expected_transport_claim_root", return_value=claim_root):
                changed_claim = dict(claim_value)
                changed_claim["schema_id"] = "MUTATED"
                (claim_root / "CLAIM.json").write_text(json.dumps(changed_claim), encoding="utf-8")
                with self.assertRaises(Exception):
                    gate_a_target_runner.cleanup_after_verified_copy(args, expected_output_root=str(root))
                self.assertTrue(root.exists())
                (claim_root / "CLAIM.json").write_text(json.dumps(claim_value), encoding="utf-8")
                result = gate_a_target_runner.cleanup_after_verified_copy(args, expected_output_root=str(root))
            self.assertEqual(result["status"], "GATE_A_CLEANUP_COMPLETE_AFTER_VERIFIED_COPY_BACK")
            self.assertFalse(root.exists())
            self.assertTrue(claim_root.exists())

    def test_host_rejects_before_transport_construction(self) -> None:
        context = host_adapter.AdapterContext(
            schedule=schedule(),
            namespace=json.loads(NAMESPACE_PATH.read_text(encoding="utf-8")),
            manifest={"execution_bundle_sha256": "b" * 64},
        )
        calls = {"factory": 0, "execute": 0, "custody": 0}

        class FakeTransport:
            def execute(self, request: Any) -> dict[str, Any]:
                calls["execute"] += 1
                return {"transport_execution_count": 1, "automatic_retry": False}

        def factory() -> FakeTransport:
            calls["factory"] += 1
            return FakeTransport()

        def custody(_path: Path, _authority: dict[str, Any]) -> dict[str, Any]:
            calls["custody"] += 1
            raise host_adapter.AdapterError("bad committed custody")

        args = argparse.Namespace(
            authority_artifact=None,
            authority_sha256=None,
            reviewed_adapter_head=None,
            independent_review_id=None,
            local_evidence_root=None,
        )
        with self.assertRaises(Exception):
            host_adapter.execute_authorized(args, context, transport_factory=factory)
        self.assertEqual(calls, {"factory": 0, "execute": 0, "custody": 0})

        args = argparse.Namespace(
            authority_artifact="future.json",
            authority_sha256=hashlib.sha256(b"{}\n").hexdigest(),
            reviewed_adapter_head="1" * 40,
            independent_review_id=1,
            local_evidence_root="future-evidence",
        )
        with mock.patch.object(host_adapter, "load_authority", side_effect=host_adapter.AdapterError("bad authority")):
            with self.assertRaises(Exception):
                host_adapter.execute_authorized(args, context, transport_factory=factory)
        self.assertEqual(calls, {"factory": 0, "execute": 0, "custody": 0})

        valid = {
            "target": "root@192.168.137.100",
            "reviewed_adapter_head": "1" * 40,
            "independent_review_id": 1,
            "maximum_execution_count": 1,
            "authority_state": {"automatic_retry": False},
        }
        with mock.patch.object(host_adapter, "load_authority", return_value=valid):
            with self.assertRaises(Exception):
                host_adapter.execute_authorized(
                    args,
                    context,
                    transport_factory=factory,
                    authority_custody_validator=custody,
                )
        self.assertEqual(calls, {"factory": 0, "execute": 0, "custody": 1})

    def test_host_validated_authority_invokes_one_fake_transport_only(self) -> None:
        context = host_adapter.AdapterContext(
            schedule=schedule(),
            namespace=json.loads(NAMESPACE_PATH.read_text(encoding="utf-8")),
            manifest={"execution_bundle_sha256": "b" * 64},
        )
        calls = {"factory": 0, "execute": 0, "custody": 0}

        class FakeTransport:
            def execute(self, request: Any) -> dict[str, Any]:
                calls["execute"] += 1
                return {"transport_execution_count": 1, "automatic_retry": False}

        def factory() -> FakeTransport:
            calls["factory"] += 1
            return FakeTransport()

        args = argparse.Namespace(
            authority_artifact="future.json",
            authority_sha256=hashlib.sha256(b"{}\n").hexdigest(),
            reviewed_adapter_head="1" * 40,
            independent_review_id=1,
            local_evidence_root="future-evidence",
        )
        valid = {
            "target": "root@192.168.137.100",
            "reviewed_adapter_head": "1" * 40,
            "independent_review_id": 1,
            "maximum_execution_count": 1,
            "authority_state": {"automatic_retry": False},
        }
        def custody(_path: Path, authority: dict[str, Any]) -> dict[str, Any]:
            calls["custody"] += 1
            return {
                "status": "GATE_A_EXECUTION_AUTHORITY_GIT_CUSTODY_EXACT",
                "reviewed_adapter_head": authority["reviewed_adapter_head"],
                "reviewed_source_tree": "2" * 40,
                "independent_review_id": authority["independent_review_id"],
                "authority_bearing_head": "3" * 40,
                "authority_bearing_tree": "4" * 40,
                "authority_git_blob_sha1": "5" * 40,
            }
        def committed_bytes(path: Path, *, head: str) -> bytes:
            self.assertEqual(head, "3" * 40)
            if path.name == "future.json":
                return b"{}\n"
            return b"{}\n"
        with mock.patch.object(host_adapter, "load_authority", return_value=valid), \
             mock.patch.object(host_adapter, "_committed_path_bytes", side_effect=committed_bytes), \
             mock.patch.object(host_adapter, "build_source_review_binding", return_value={"schema_id": "TEST"}), \
             mock.patch.object(host_adapter, "_git", return_value=subprocess.CompletedProcess([], 0, stdout="3" * 40 + "\n", stderr="")):
            result = host_adapter.execute_authorized(
                args,
                context,
                transport_factory=factory,
                authority_custody_validator=custody,
            )
        self.assertEqual(result["transport_execution_count"], 1)
        self.assertEqual(calls, {"factory": 1, "execute": 1, "custody": 1})

    def test_committed_authority_two_commit_custody(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_authority_git_") as tmp:
            root = Path(tmp)
            def git(*args: str) -> str:
                completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return completed.stdout.strip()
            git("init", "-q")
            git("config", "user.name", "Gate A Test")
            git("config", "user.email", "gate-a-test@example.invalid")
            git("config", "core.autocrlf", "false")
            protected = root / "reviewed_runner.py"
            protected.write_bytes(b"reviewed = True\n")
            git("add", protected.name)
            git("commit", "-q", "-m", "reviewed source")
            reviewed = git("rev-parse", "HEAD")
            authority_path = root / "GATE_A_EXECUTION_AUTHORITY.json"
            authority = {"reviewed_adapter_head": reviewed, "independent_review_id": 123}
            authority_path.write_bytes((json.dumps(authority, sort_keys=True) + "\n").encode("utf-8"))
            git("add", authority_path.name)
            git("commit", "-q", "-m", "owner authority")
            custody = host_adapter.validate_authority_git_custody(
                authority_path,
                authority,
                root=root,
                protected_paths=(protected.name,),
                expected_authority_rel=authority_path.name,
            )
            self.assertEqual(custody["reviewed_adapter_head"], reviewed)
            self.assertEqual(custody["reviewed_source_tree"], git("rev-parse", f"{reviewed}^{{tree}}"))
            self.assertEqual(custody["authority_bearing_head"], git("rev-parse", "HEAD"))
            self.assertEqual(custody["authority_bearing_tree"], git("rev-parse", "HEAD^{tree}"))
            self.assertEqual(custody["authority_git_blob_sha1"], git("rev-parse", f"HEAD:{authority_path.name}"))
            authority_path.write_bytes(b"{}\n")
            with self.assertRaises(Exception):
                host_adapter.validate_authority_git_custody(
                    authority_path,
                    authority,
                    root=root,
                    protected_paths=(protected.name,),
                    expected_authority_rel=authority_path.name,
                )

    def test_remote_namespace_preflight_rejects_before_any_scp(self) -> None:
        base = {
            "inspection_complete": True,
            "execution_root": "absent",
            "output_root": "absent",
            "stage": "absent",
            "authority": "absent",
            "archive": "absent",
            "receipt": "absent",
            "target_inventory": "absent",
            "claim": "absent",
            "prefix_matches": [],
        }
        cases: list[dict[str, Any] | str] = []
        for key in ("execution_root", "output_root", "stage", "authority", "archive", "receipt", "target_inventory", "claim"):
            value = copy.deepcopy(base); value[key] = "present"; cases.append(value)
            value = copy.deepcopy(base); value[key] = "unobservable:PermissionError"; cases.append(value)
        value = copy.deepcopy(base); value["prefix_matches"] = ["/root/.catcas_gate_a_collision"]; cases.append(value)
        value = copy.deepcopy(base); value["inspection_complete"] = False; cases.append(value)
        cases.append("not-json")

        for index, response in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory(prefix="gate_a_transport_preflight_") as tmp:
                root = Path(tmp)
                calls: list[list[str]] = []
                def command_runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
                    calls.append(argv)
                    stdout = response if isinstance(response, str) else json.dumps(response)
                    return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
                request = make_transport_request(root)
                transport = smoke_transport.SshScpTransport(command_runner=command_runner)
                with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                    with self.assertRaises(Exception):
                        transport.execute(request)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], "ssh")
                self.assertFalse(any(call[0] == "scp" for call in calls))


if __name__ == "__main__":
    unittest.main()
