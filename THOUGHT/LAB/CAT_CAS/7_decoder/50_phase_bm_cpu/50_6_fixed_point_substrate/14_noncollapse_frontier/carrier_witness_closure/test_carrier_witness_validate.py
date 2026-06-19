#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "carrier_witness_validate", HERE / "carrier_witness_validate.py"
)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class CarrierWitnessValidatorTests(unittest.TestCase):
    def test_lockin_recovers_declared_tone(self) -> None:
        tsc_hz = 1_000_000.0
        sample_hz = 4_000.0
        frequency = 125.0
        count = 4000
        origin = 10_000_000
        timestamps = [
            origin + int(index * tsc_hz / sample_hz)
            for index in range(count)
        ]
        values = [
            2.5 + 0.75 * math.cos(
                2.0 * math.pi * frequency * (timestamp - origin) / tsc_hz
            )
            for timestamp in timestamps
        ]
        value_i, value_q, magnitude = validator.lockin(
            timestamps, values, frequency, origin, tsc_hz
        )
        self.assertAlmostEqual(value_i, 0.75, places=3)
        self.assertAlmostEqual(value_q, 0.0, places=3)
        self.assertAlmostEqual(magnitude, 0.75, places=3)

    def test_raw_record_roundtrip(self) -> None:
        records = [(100, 1.25), (200, 1.5), (300, 1.75), (400, 2.0)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw_samples.bin"
            with path.open("wb") as handle:
                for record in records:
                    handle.write(validator.RAW_RECORD.pack(*record))
            with path.open("rb") as handle:
                timestamps, periods = validator.read_raw_window(handle, 0, len(records))
        self.assertEqual(timestamps, [record[0] for record in records])
        self.assertEqual(periods, [record[1] for record in records])

    def test_manifest_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            files = validator.REQUIRED_RUN_FILES - {"run_manifest.json"}
            manifest_files = {}
            for name in files:
                path = run_dir / name
                path.write_bytes((name + "\n").encode())
                manifest_files[name] = {
                    "size": path.stat().st_size,
                    "sha256": validator.sha256_file(path),
                }
            manifest = {"files": manifest_files}
            self.assertEqual(validator.verify_manifest(run_dir, manifest), [])
            (run_dir / "summary.csv").write_text("tampered\n")
            errors = validator.verify_manifest(run_dir, manifest)
            self.assertTrue(any("summary.csv" in error for error in errors))

    def test_required_window_schema_includes_restoration_flag(self) -> None:
        self.assertIn("hash_restored", validator.REQUIRED_WINDOW_COLUMNS)


if __name__ == "__main__":
    unittest.main()
