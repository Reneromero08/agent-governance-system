#!/usr/bin/env python3
"""Non-driving tests for the authority-gated Gate A engineering-smoke executor."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import gate_a_hardware_adapter as host_adapter
import gate_a_engineering_smoke_executor as executor
import gate_a_engineering_smoke_transport as smoke_transport
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

    def process_snapshot(self) -> executor.ProcessSnapshot:
        self.calls.append("process")
        return executor.ProcessSnapshot(self.complete, self.return_code, "raw ps", "", self.hits)

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

    def begin(self, plan: executor.FrozenPlan, preflight: dict[str, Any]) -> None:
        self.begun += 1

    def event(self, value: dict[str, Any]) -> None:
        self.events.append(value)

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

    def execute(self, plan: executor.FrozenPlan) -> dict[str, Any]:
        self.calls += 1
        if self.error:
            raise self.error
        return copy.deepcopy(self.result)


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
        self.assert_executor_rejects(claims=claims, runtime=runtime, output_root=Path("/fake/evidence2"))
        self.assertEqual(runtime.calls, 1)

    def test_runtime_failure_has_no_retry(self) -> None:
        claims, evidence = FakeClaims(), FakeEvidence()
        runtime = FakeRuntime(error=RuntimeError("simulated capture failure"))
        self.assert_executor_rejects(claims=claims, evidence=evidence, runtime=runtime)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(len(evidence.failures), 1)
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
            events = (output / "EVENTS.jsonl").read_text(encoding="utf-8")
            self.assertIn('"raw_process_listing":"raw ps"', events)
            self.assertIn('"return_code":7', events)
            self.assertIn("runtime_failed", events)

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
            receipt.write_text(json.dumps(value), encoding="utf-8")
            result = gate_a_target_runner.cleanup_after_verified_copy(args, expected_output_root=str(root))
            self.assertEqual(result["status"], "GATE_A_CLEANUP_COMPLETE_AFTER_VERIFIED_COPY_BACK")
            self.assertFalse(root.exists())

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
            authority_sha256="a" * 64,
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
            authority_sha256="a" * 64,
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
                "independent_review_id": authority["independent_review_id"],
            }
        with mock.patch.object(host_adapter, "load_authority", return_value=valid):
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
            "archive": "absent",
            "prefix_matches": [],
        }
        cases: list[dict[str, Any] | str] = []
        for key in ("execution_root", "output_root", "stage", "archive"):
            value = copy.deepcopy(base); value[key] = "present"; cases.append(value)
            value = copy.deepcopy(base); value[key] = "unobservable:PermissionError"; cases.append(value)
        value = copy.deepcopy(base); value["prefix_matches"] = ["/root/.catcas_gate_a_collision"]; cases.append(value)
        value = copy.deepcopy(base); value["inspection_complete"] = False; cases.append(value)
        cases.append("not-json")

        for index, response in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory(prefix="gate_a_transport_preflight_") as tmp:
                root = Path(tmp)
                authority = root / "authority.json"
                authority.write_text("{}\n", encoding="utf-8")
                calls: list[list[str]] = []
                def command_runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
                    calls.append(argv)
                    stdout = response if isinstance(response, str) else json.dumps(response)
                    return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
                request = smoke_transport.HostExecutionRequest(
                    target="root@192.168.137.100",
                    authority_path=authority,
                    authority_sha256="a" * 64,
                    reviewed_adapter_head="1" * 40,
                    independent_review_id=1,
                    execution_bundle_sha256="b" * 64,
                    schedule_sha256=executor.SCHEDULE_SHA256,
                    namespace_sha256="c" * 64,
                    remote_execution_root="/root/catcas_phase6b6_gate_a_smoke_9c416379",
                    remote_output_root="/root/catcas_phase6b6_gate_a_smoke_9c416379/evidence",
                    local_evidence_root=root / "evidence",
                )
                transport = smoke_transport.SshScpTransport(command_runner=command_runner)
                with mock.patch.object(smoke_transport.bundle, "write_deployment_archive", side_effect=lambda path, _treeish: path.write_bytes(b"bundle")):
                    with self.assertRaises(Exception):
                        transport.execute(request)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], "ssh")
                self.assertFalse(any(call[0] == "scp" for call in calls))


if __name__ == "__main__":
    unittest.main()
