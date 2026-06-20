#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from make_executor_source_bundle import (
    SOURCE_PATHS,
    add_file_bindings,
    sha256_file,
    verify_bundle,
    verify_file_bindings,
    write_target_manifest,
)


class ExecutorSourceBundleTests(unittest.TestCase):
    def test_campaign_runtime_dependencies_are_transferred(self) -> None:
        names = {path.name for path in SOURCE_PATHS}
        self.assertIn("campaign_plan.py", names)
        self.assertIn("campaign_orders.py", names)
        self.assertIn("make_engineering_smoke_schedule.py", names)
        self.assertIn("test_orchestrator.py", names)

    def test_exact_file_set_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file.txt").write_text("bound", encoding="utf-8")
            bindings = add_file_bindings(root)
            verify_file_bindings(root, bindings)
            (root / "extra.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                verify_file_bindings(root, bindings)

    def test_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root.parent / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            binding = {"../outside.txt": {"size": 7, "sha256": sha256_file(outside)}}
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                verify_file_bindings(root, binding)
            outside.unlink()

    def test_target_manifest_and_transfer_bundle_are_non_authorizing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence.log").write_text("OK\n", encoding="utf-8")
            manifest = write_target_manifest(root, "a" * 40)
            self.assertEqual(manifest["executor_commit"], "a" * 40)
            bundle = {
                "schema_id": "CAT_CAS_PHASE6_EXECUTOR_SOURCE_TRANSFER_V1",
                "executor_commit": "a" * 40,
                "files": {"evidence.log": manifest["files"]["evidence.log"]},
                "acquisition_authorized": False,
                "restoration_authorized": False,
            }
            (root / "target_evidence_manifest.json").unlink()
            path = root / "source_bundle.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            (root / "source_bundle.sha256").write_text(
                f"{sha256_file(path)}  source_bundle.json\n", encoding="utf-8")
            self.assertEqual(verify_bundle(root, "a" * 40)["acquisition_authorized"], False)


if __name__ == "__main__":
    unittest.main()
