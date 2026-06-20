#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from verify_run_manifests import verify


class RunManifestTests(unittest.TestCase):
    def test_malformed_manifest_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            (run / "run_manifest.json").write_text("{", encoding="utf-8")
            self.assertTrue(any("invalid" in error for error in verify(Path(temporary))))

    def test_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (root / "outside").write_text("x", encoding="utf-8")
            manifest = {
                "files": {
                    "../outside": {"size": 1, "sha256": "0" * 64},
                }
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("invalid path" in error for error in verify(root)))

    def test_non_object_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            (run / "run_manifest.json").write_text("[]", encoding="utf-8")
            self.assertTrue(any("JSON object" in error for error in verify(Path(temporary))))

    def test_non_mapping_files_and_binding_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "run_manifest.json").write_text(
                json.dumps({"files": []}), encoding="utf-8"
            )
            self.assertTrue(any("files table" in error for error in verify(root)))
            (run / "run_manifest.json").write_text(
                json.dumps({"files": {"x": []}}), encoding="utf-8"
            )
            self.assertTrue(any("invalid binding" in error for error in verify(root)))


if __name__ == "__main__":
    unittest.main()
