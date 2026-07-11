#!/usr/bin/env python3
"""Executor-level proof that temperature receipts precede every thermal stop."""

from __future__ import annotations

import json
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
            with self.assertRaises(temperature.TemperatureCustodyError):
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


if __name__ == "__main__":
    unittest.main()
