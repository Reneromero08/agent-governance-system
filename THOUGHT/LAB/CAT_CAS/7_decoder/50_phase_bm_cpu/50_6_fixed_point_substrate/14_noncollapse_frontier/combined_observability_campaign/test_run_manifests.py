#!/usr/bin/env python3
from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from verify_run_manifests import verify
class RunManifestTests(unittest.TestCase):
 def test_malformed_manifest_is_reported(self)->None:
  with tempfile.TemporaryDirectory() as t:
   run=Path(t)/"run";run.mkdir();(run/"run_manifest.json").write_text("{")
   self.assertEqual(verify(Path(t)),["run: invalid run_manifest.json"])
 def test_manifest_path_cannot_escape_run(self)->None:
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);run=root/"run";run.mkdir();(root/"outside").write_text("x")
   (run/"run_manifest.json").write_text(json.dumps({"files":{"../outside":{"size":1,"sha256":"bad"}}}))
   self.assertEqual(verify(root),["run: invalid path ../outside"])
if __name__=="__main__":unittest.main()
