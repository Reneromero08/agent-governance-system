#!/usr/bin/env python3
"""Executor-level proof that temperature receipts precede every thermal stop."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import gate_a_engineering_smoke_executor as base
import gate_a_process_custody as process_custody
import gate_a_temperature_custody as temperature

# The wrapper imports this package name inside the extracted target bundle.  Bind
# the reviewed repository module to the same name for non-contact source tests.
sys.modules.setdefault("gate_a_engineering_smoke_executor_base", base)
import gate_a_engineering_smoke_executor_k10temp as repaired  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE.parents[3] / "holo_runtime_v2"
SCHEDULE = json.loads((HERE.parent / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json").read_text(encoding="utf-8"))
AUTHORITY_SHA256 = "a" * 64
BUNDLE_SHA256 = "b" * 64
AUTHORITY_VALIDATION = {
    "status": "GATE_A_EXECUTION_AUTHORITY_EXACT",
    "reviewed_adapter_head": "c" * 40,
    "independent_review_id": 1,
    "execution_bundle_sha256": BUNDLE_SHA256,
}


def make_hwmon(root: Path, raw: str | None) -> None:
    entry = root / "hwmon0"
    entry.mkdir()
    (entry / "name").write_text("k10temp\n", encoding="ascii")
    if raw is not None:
        (entry / "temp1_input").write_text(raw, encoding="ascii")


def native_receipt(
    phase: str,
    *,
    millidegrees: int = 42000,
    observation_tsc: int,
) -> dict[str, Any]:
    raw = f"{millidegrees}\n"
    normalized = millidegrees / temperature.MILLIDEGREES_PER_C
    passed = normalized < temperature.VETO_C
    root = repaired.NATIVE_TEMPERATURE_HWMON_ROOT
    selected = f"{root}/hwmon0"
    return {
        "schema_id": repaired.NATIVE_TEMPERATURE_SCHEMA_ID,
        "phase": phase,
        "hwmon_root": root,
        "required_driver_name": temperature.DRIVER_NAME,
        "required_temperature_input": temperature.TEMPERATURE_INPUT,
        "millidegrees_per_c": temperature.MILLIDEGREES_PER_C,
        "enumerated_hwmon_count": 1,
        "k10temp_candidate_count": 1,
        "selected_hwmon_entry": selected,
        "selected_driver_name": temperature.DRIVER_NAME,
        "selected_temperature_path": f"{selected}/{temperature.TEMPERATURE_INPUT}",
        "raw_temperature_text": raw,
        "raw_temperature_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
        "raw_millidegrees_c": millidegrees,
        "normalized_temperature_c": normalized,
        "veto_temperature_c": temperature.VETO_C,
        "observation_complete": True,
        "veto_passed": passed,
        "failure": None if passed else "TEMPERATURE_VETO",
        "observation_tsc": observation_tsc,
    }


class ClaimStore:
    def __init__(self) -> None:
        self.calls = 0

    def claim(self, _authority_sha256: str, _plan: base.FrozenPlan) -> None:
        self.calls += 1


class EvidenceStore:
    def __init__(self) -> None:
        self.started = False
        self.receipt: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.failure: str | None = None

    def begin(self, _plan: base.FrozenPlan, _preflight: dict[str, Any]) -> None:
        self.started = True

    def event(self, value: dict[str, Any]) -> None:
        self.events.append(value)

    def process_receipt(self, _phase: str, _receipt: dict[str, Any]) -> None:
        pass

    def temperature_receipt(self, receipt: dict[str, Any]) -> None:
        self.receipt = dict(receipt)

    def complete(self, _result: dict[str, Any]) -> None:
        raise AssertionError("thermal stop must not complete")

    def fail(self, reason: str) -> None:
        self.failure = reason


class Runtime:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _plan: base.FrozenPlan) -> dict[str, Any]:
        self.calls += 1
        raise AssertionError("thermal stop reached runtime")

    def verify_evidence(self, _plan: base.FrozenPlan, _result: dict[str, Any]) -> None:
        raise AssertionError("thermal stop verified runtime evidence")


class Preflight(repaired.LocalPreflight):
    def __init__(self, root: Path) -> None:
        super().__init__(hwmon_root=root)
        self.frequency_calls = 0

    def inspect_namespace(self, _path: Path) -> base.NamespaceState:
        return base.NamespaceState.ABSENT

    def process_snapshot(self, phase: str) -> dict[str, Any]:
        return process_custody._receipt(
            phase,
            return_code=0,
            stdout=b"1 init /sbin/init\n",
            stderr=b"",
            timed_out=False,
            failure=None,
        )

    def frequency_khz(self, _core: int) -> int:
        self.frequency_calls += 1
        raise AssertionError("thermal stop reached frequency preflight")


class K10tempExecutorTests(unittest.TestCase):
    def run_thermal_stop(self, raw: str | None) -> tuple[Preflight, ClaimStore, EvidenceStore, Runtime]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hwmon"
            root.mkdir()
            make_hwmon(root, raw)
            preflight = Preflight(root)
            claims = ClaimStore()
            evidence = EvidenceStore()
            runtime = Runtime()
            with self.assertRaises(base.ExecutorError):
                repaired.execute_once(
                    authority_validation=AUTHORITY_VALIDATION,
                    authority_sha256=AUTHORITY_SHA256,
                    execution_bundle_sha256=BUNDLE_SHA256,
                    schedule=SCHEDULE,
                    output_root=Path(tmp) / "output",
                    surfaces=base.ExecutionSurfaces(
                        preflight=preflight,
                        claims=claims,
                        evidence=evidence,
                        runtime=runtime,
                    ),
                )
            return preflight, claims, evidence, runtime

    def test_unobservable_temperature_receipt_precedes_stop(self) -> None:
        preflight, claims, evidence, runtime = self.run_thermal_stop(None)
        self.assertEqual(claims.calls, 1)
        self.assertTrue(evidence.started)
        self.assertIsNotNone(evidence.receipt)
        self.assertFalse(evidence.receipt["observation_complete"])
        self.assertFalse(evidence.receipt["veto_passed"])
        self.assertIn("TEMPERATURE_UNOBSERVABLE", evidence.receipt["failure"])
        self.assertIsNotNone(evidence.failure)
        self.assertEqual(preflight.frequency_calls, 0)
        self.assertEqual(runtime.calls, 0)

    def test_68c_veto_receipt_precedes_stop(self) -> None:
        preflight, claims, evidence, runtime = self.run_thermal_stop("68000\n")
        self.assertEqual(claims.calls, 1)
        self.assertTrue(evidence.started)
        self.assertIsNotNone(evidence.receipt)
        self.assertTrue(evidence.receipt["observation_complete"])
        self.assertFalse(evidence.receipt["veto_passed"])
        self.assertEqual(evidence.receipt["failure"], "TEMPERATURE_VETO")
        self.assertEqual(evidence.receipt["normalized_temperature_c"], 68.0)
        self.assertIsNotNone(evidence.failure)
        self.assertEqual(preflight.frequency_calls, 0)
        self.assertEqual(runtime.calls, 0)


class NativeTemperatureReceiptTests(unittest.TestCase):
    def test_python_and_c_temperature_constants_agree(self) -> None:
        header = (RUNTIME_ROOT / "gate_a_engineering_smoke_runtime.h").read_text(encoding="utf-8")
        self.assertRegex(header, r'#define GATE_A_TEMPERATURE_HWMON_ROOT "/sys/class/hwmon"')
        self.assertRegex(header, r'#define GATE_A_TEMPERATURE_DRIVER_NAME "k10temp"')
        self.assertRegex(header, r'#define GATE_A_TEMPERATURE_INPUT "temp1_input"')
        values = {
            name: int(re.search(rf"#define {name} \((-?[0-9]+)L\)|#define {name} (-?[0-9]+)L", header).group(1) or
                      re.search(rf"#define {name} \((-?[0-9]+)L\)|#define {name} (-?[0-9]+)L", header).group(2))
            for name in (
                "GATE_A_TEMPERATURE_MILLIDEGREES_PER_C",
                "GATE_A_TEMPERATURE_VETO_MILLIDEGREES",
                "GATE_A_TEMPERATURE_MIN_MILLIDEGREES",
                "GATE_A_TEMPERATURE_MAX_MILLIDEGREES",
            )
        }
        self.assertEqual(values["GATE_A_TEMPERATURE_MILLIDEGREES_PER_C"], temperature.MILLIDEGREES_PER_C)
        self.assertEqual(values["GATE_A_TEMPERATURE_VETO_MILLIDEGREES"], int(temperature.VETO_C * temperature.MILLIDEGREES_PER_C))
        self.assertEqual(values["GATE_A_TEMPERATURE_MIN_MILLIDEGREES"], int(temperature.MIN_PLAUSIBLE_C * temperature.MILLIDEGREES_PER_C))
        self.assertEqual(values["GATE_A_TEMPERATURE_MAX_MILLIDEGREES"], int(temperature.MAX_PLAUSIBLE_C * temperature.MILLIDEGREES_PER_C))

    def test_native_temperature_closed_receipt_mutation_baseline(self) -> None:
        baseline = native_receipt("pre_capture", observation_tsc=900_000_000)
        repaired.validate_native_temperature_receipt(
            baseline,
            expected_phase="pre_capture",
            require_pass=True,
        )
        mutations: list[dict[str, Any]] = []
        for field, value in (
            ("phase", "post_capture"),
            ("selected_temperature_path", "/sys/class/hwmon/hwmon0/temp2_input"),
            ("millidegrees_per_c", 999),
            ("normalized_temperature_c", 41.0),
            ("veto_passed", False),
            ("observation_tsc", 0),
        ):
            altered = copy.deepcopy(baseline)
            altered[field] = value
            mutations.append(altered)
        altered_raw = copy.deepcopy(baseline)
        altered_raw["raw_temperature_text"] = "43000\n"
        altered_raw["raw_temperature_sha256"] = hashlib.sha256(b"43000\n").hexdigest()
        mutations.append(altered_raw)
        missing = copy.deepcopy(baseline)
        del missing["phase"]
        mutations.append(missing)
        extra = copy.deepcopy(baseline)
        extra["unexpected"] = None
        mutations.append(extra)
        for altered in mutations:
            with self.subTest(altered=altered):
                with self.assertRaises(base.ExecutorError):
                    repaired.validate_native_temperature_receipt(
                        altered,
                        expected_phase="pre_capture",
                        require_pass=True,
                    )

    def test_native_receipt_sequence_and_canonical_wrapper_surface(self) -> None:
        self.assertIs(repaired._write_all, base._write_all)
        self.assertTrue(issubclass(repaired.WorkerRuntime, base.WorkerRuntime))
        pre = native_receipt("pre_capture", observation_tsc=900_000_000)
        post = native_receipt("post_capture", observation_tsc=26_600_000_001)
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp) / "runtime"
            runtime_root.mkdir()
            receipt_path = runtime_root / repaired.NATIVE_TEMPERATURE_RECEIPT_FILE
            receipt_path.write_text(
                "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in (pre, post)),
                encoding="ascii",
            )
            result = {
                "capture": {
                    "origin_tsc": 1_000_000_000,
                    "last_sample_tsc": 26_599_600_000,
                }
            }
            observed = repaired.verify_native_temperature_receipts(runtime_root, result)
            self.assertEqual(observed, (pre, post))

    @unittest.skipUnless(os.name == "posix" and shutil.which("cc"), "native compiler required")
    def test_compiled_worker_receipts_parse_under_closed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "gate_a_worker"
            subprocess.run(
                [
                    "cc", "-std=c11", "-O2", "-pthread",
                    "-Wall", "-Wextra", "-Werror", "-pedantic",
                    str(HERE / "gate_a_worker.c"),
                    str(RUNTIME_ROOT / "gate_a_engineering_smoke_runtime.c"),
                    str(RUNTIME_ROOT / "captured_file.c"),
                    f"-I{RUNTIME_ROOT}", "-lm", "-o", str(executable),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output = Path(tmp) / "retained"
            completed = subprocess.run(
                [str(executable), "--self-test-retain", str(output)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["temperature_receipt_count"], 2)
            retained = json.loads((output / "runtime_result.json").read_text(encoding="utf-8"))
            result = {
                "capture": {
                    "origin_tsc": retained["capture_origin_tsc"],
                    "last_sample_tsc": retained["capture_last_sample_tsc"],
                }
            }
            repaired.verify_native_temperature_receipts(output, result)

if __name__ == "__main__":
    unittest.main()
