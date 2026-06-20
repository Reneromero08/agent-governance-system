#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import carrier_witness_finalize as finalizer


class CarrierWitnessFinalizerTests(unittest.TestCase):
    def test_incomplete_run_refuses_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run_1"
            run_dir.mkdir()
            metadata = root / "metadata.json"
            metadata.write_text("{}\n")
            with self.assertRaisesRegex(FileNotFoundError, "incomplete run"):
                finalizer.finalize_run(run_dir, metadata, root / "analyzer.py", "0" * 40)

    def test_exclusive_json_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "value.json"
            finalizer.write_exclusive_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                finalizer.write_exclusive_json(path, {"value": 2})


if __name__ == "__main__":
    unittest.main()
