#!/usr/bin/env python3
"""Non-contact tests for exact Gate A k10temp custody."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import gate_a_temperature_custody as temperature


def make_hwmon(root: Path, index: int, name: str, raw: str | None = None) -> Path:
    entry = root / f"hwmon{index}"
    entry.mkdir()
    (entry / "name").write_text(name, encoding="ascii")
    if raw is not None:
        (entry / "temp1_input").write_text(raw, encoding="ascii")
    return entry


class TemperatureCustodyTests(unittest.TestCase):
    def observe(self, root: Path, **kwargs: Any) -> dict[str, Any]:
        return temperature.observe_temperature("pre_runtime", hwmon_root=root, **kwargs)

    def assert_rejects(self, receipt: dict[str, Any]) -> None:
        with self.assertRaises(temperature.TemperatureCustodyError):
            temperature.validate_temperature_receipt(
                receipt,
                expected_phase="pre_runtime",
                require_pass=True,
            )

    def test_exact_k10temp_24600_normalizes_to_24_6(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_hwmon(root, 0, "k10temp\n", "24600\n")
            receipt = self.observe(root)
        temperature.validate_temperature_receipt(receipt, expected_phase="pre_runtime", require_pass=True)
        self.assertEqual(receipt["selected_driver_name"], "k10temp")
        self.assertEqual(receipt["raw_millidegrees_c"], 24600)
        self.assertEqual(receipt["normalized_temperature_c"], 24.6)
        self.assertTrue(receipt["veto_passed"])
        self.assertEqual(
            receipt["raw_temperature_sha256"],
            hashlib.sha256(b"24600\n").hexdigest(),
        )

    def test_no_root_or_empty_enumeration_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            self.assert_rejects(self.observe(missing))
            empty = Path(tmp) / "empty"
            empty.mkdir()
            self.assert_rejects(self.observe(empty))

    def test_nouveau_alone_rejects_and_mixed_selects_only_k10temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_hwmon(root, 0, "nouveau\n", "53000\n")
            self.assert_rejects(self.observe(root))
            make_hwmon(root, 1, "k10temp\n", "25000\n")
            receipt = self.observe(root)
        temperature.validate_temperature_receipt(receipt, expected_phase="pre_runtime", require_pass=True)
        self.assertTrue(receipt["selected_hwmon_entry"].endswith("hwmon1"))
        self.assertEqual([entry["driver_name"] for entry in receipt["enumerated_entries"]], ["nouveau", "k10temp"])

    def test_multiple_k10temp_entries_reject_as_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_hwmon(root, 0, "k10temp\n", "24000\n")
            make_hwmon(root, 1, "k10temp\n", "25000\n")
            receipt = self.observe(root)
        self.assert_rejects(receipt)
        self.assertIn("found 2", receipt["failure"])

    def test_missing_empty_malformed_and_overflow_input_reject(self) -> None:
        values = (None, "", "24.6\n", "1000001\n")
        for raw in values:
            with self.subTest(raw=raw):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    make_hwmon(root, 0, "k10temp\n", raw)
                    self.assert_rejects(self.observe(root))

    def test_io_error_and_inconsistent_repeated_reads_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = make_hwmon(root, 0, "k10temp\n", "24600\n")
            temperature_path = entry / "temp1_input"

            def unreadable(path: Path) -> str:
                if path == temperature_path:
                    raise PermissionError("denied")
                return path.read_text(encoding="ascii")

            self.assert_rejects(self.observe(root, read_text=unreadable))

            calls = 0

            def unstable(path: Path) -> str:
                nonlocal calls
                if path == temperature_path:
                    calls += 1
                    return "24600\n" if calls == 1 else "24700\n"
                return path.read_text(encoding="ascii")

            self.assert_rejects(self.observe(root, read_text=unstable))

    def test_veto_boundary_is_exact(self) -> None:
        cases = (("67999\n", True), ("68000\n", False), ("90000\n", False))
        for raw, passes in cases:
            with self.subTest(raw=raw):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    make_hwmon(root, 0, "k10temp\n", raw)
                    receipt = self.observe(root)
                self.assertEqual(receipt["veto_passed"], passes)
                if passes:
                    temperature.validate_temperature_receipt(receipt, expected_phase="pre_runtime", require_pass=True)
                else:
                    self.assert_rejects(receipt)
                    self.assertEqual(receipt["failure"], "TEMPERATURE_VETO")

    def test_receipt_is_closed_and_mutations_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_hwmon(root, 0, "k10temp\n", "24600\n")
            receipt = self.observe(root)
        self.assertEqual(set(receipt), temperature.RECEIPT_KEYS)
        mutations = {
            "extra": lambda value: value.__setitem__("extra", True),
            "driver": lambda value: value.__setitem__("selected_driver_name", "nouveau"),
            "path": lambda value: value.__setitem__("selected_temperature_path", "/tmp/fake"),
            "raw": lambda value: value.__setitem__("raw_temperature_text", "24700\n"),
            "raw_digest": lambda value: value.__setitem__("raw_temperature_sha256", "0" * 64),
            "raw_value": lambda value: value.__setitem__("raw_millidegrees_c", 24700),
            "scale": lambda value: value.__setitem__("millidegrees_per_c", 1),
            "normalized": lambda value: value.__setitem__("normalized_temperature_c", 25.0),
            "veto": lambda value: value.__setitem__("veto_passed", False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(receipt)
                mutate(changed)
                with self.assertRaises(temperature.TemperatureCustodyError):
                    temperature.validate_temperature_receipt(
                        changed,
                        expected_phase="pre_runtime",
                        require_pass=False,
                    )

    def test_receipt_digest_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_hwmon(root, 0, "k10temp\n", "24600\n")
            receipt = self.observe(root)
        expected = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(temperature.receipt_sha256(receipt), expected)


if __name__ == "__main__":
    unittest.main()
