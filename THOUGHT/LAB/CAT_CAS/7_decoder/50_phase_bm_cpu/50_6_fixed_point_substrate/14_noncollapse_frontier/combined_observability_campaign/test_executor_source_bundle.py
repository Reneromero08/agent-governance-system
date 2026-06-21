#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from make_executor_source_bundle import (
    SOURCE_PATHS,
    add_file_bindings,
    rename_bundle_with_sidecar,
    sha256_file,
    verify_bundle,
    verify_sha256_sidecar,
    write_sha256_sidecar,
    verify_file_bindings,
    validate_target_validation_evidence,
    write_target_manifest,
)


class ExecutorSourceBundleTests(unittest.TestCase):
    def test_campaign_runtime_dependencies_are_transferred(self) -> None:
        names = {path.name for path in SOURCE_PATHS}
        self.assertIn("campaign_plan.py", names)
        self.assertIn("campaign_orders.py", names)
        self.assertIn("make_engineering_smoke_schedule.py", names)
        self.assertIn("test_orchestrator.py", names)
        self.assertIn("collect_target_engineering_evidence.py", names)

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


    def test_malformed_bundle_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "source_bundle.json"
            path.write_text("[]", encoding="utf-8")
            (root / "source_bundle.sha256").write_text(
                f"{sha256_file(path)}  source_bundle.json\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "JSON object"):
                verify_bundle(root)

    def test_invalid_binding_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "invalid file binding"):
                verify_file_bindings(root, {"x": []})


    def test_validation_report_must_match_raw_twelve_run_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "validation" / "runs"
            runs.mkdir(parents=True)
            records = []
            for index in range(12):
                session_id = f"session_{index:02d}"
                directory = runs / session_id
                directory.mkdir()
                run = {
                    "status": "VALIDATION_ONLY_HARDWARE_NOT_EXECUTED",
                    "hardware_executed": False,
                    "scientific_acquisition_authorized": False,
                    "restoration_authorized": False,
                    "physical_carrier_restoration_claimed": False,
                }
                run_path = directory / "run.json"
                run_path.write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
                manifest = {
                    "files": {
                        "run.json": {
                            "size": run_path.stat().st_size,
                            "sha256": sha256_file(run_path),
                        }
                    }
                }
                manifest_path = directory / "run_manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
                )
                records.append({
                    "session_id": session_id,
                    "runner_exit_code": 0,
                    "hardware_executed": False,
                    "run_manifest_sha256": sha256_file(manifest_path),
                })
            evidence = root / "evidence"
            evidence.mkdir()
            report = {
                "schema_id": "CAT_CAS_PHASE6_VALIDATION_EVIDENCE_V1",
                "runs_root": "validation/runs",
                "sessions_expected": 12,
                "sessions_passed": 12,
                "all_pass": True,
                "hardware_touched": False,
                "records": records,
                "errors": [],
            }
            report_path = evidence / "validation_report.json"
            report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
            validate_target_validation_evidence(root)
            report["records"][0]["run_manifest_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match raw"):
                validate_target_validation_evidence(root)


    def test_renamed_bundle_sidecar_names_the_renamed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "source_bundle.json"
            sidecar = root / "source_bundle.sha256"
            manifest.write_text("{}\n", encoding="utf-8")
            write_sha256_sidecar(manifest, sidecar)

            renamed, renamed_sidecar = rename_bundle_with_sidecar(
                root, "source_bundle.json", "source_transfer_bundle.json"
            )

            self.assertEqual(renamed.name, "source_transfer_bundle.json")
            self.assertEqual(
                renamed_sidecar.read_text(encoding="utf-8"),
                f"{sha256_file(renamed)}  source_transfer_bundle.json\n",
            )
            verify_sha256_sidecar(renamed_sidecar, renamed)

            sha256sum = shutil.which("sha256sum")
            if sha256sum is not None:
                completed = subprocess.run(
                    [sha256sum, "-c", renamed_sidecar.name],
                    cwd=root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_sidecar_rejects_wrong_recorded_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "source_transfer_bundle.json"
            sidecar = root / "source_transfer_bundle.sha256"
            manifest.write_text("{}\n", encoding="utf-8")
            sidecar.write_text(
                f"{sha256_file(manifest)}  source_bundle.json\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "filename mismatch"):
                verify_sha256_sidecar(sidecar, manifest)

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
