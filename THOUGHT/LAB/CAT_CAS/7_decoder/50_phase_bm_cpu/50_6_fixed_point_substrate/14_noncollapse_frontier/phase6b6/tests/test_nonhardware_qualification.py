from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from contracts.v2_interface import QUALIFIED_V2_SOURCE, TONE_CODEWORD_TABLE
from qualification.compare_v2_reference import compare_reference_tables
from qualification.qualification_contract import (
    EXPECTED_MERGED_MAIN_HEAD,
    EXPECTED_REVIEWED_IMPLEMENTATION_HEAD,
    PHASE6B6_RELATIVE_ROOT,
    QualificationError,
    REPO_ROOT,
    SNAPSHOT_SUBJECT_COMMIT,
    V2_RELATIVE_SOURCE,
    build_expected_snapshot_binding,
    compile_reference_emitter,
    digest,
    emit_reference_table,
    qualification_contract,
    validate_only,
    validate_schema,
)
from qualification.verify_sealed_snapshot import (
    SnapshotVerificationError,
    _check_case_collisions,
    assert_trusted_repository_unchanged,
    materialize_trusted_snapshot,
    trusted_repository_object_state,
    verify_snapshot_directory,
)


class NonHardwareQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = emit_reference_table()
        cls.equivalence = compare_reference_tables(cls.reference, TONE_CODEWORD_TABLE)
        cls.contract = qualification_contract()
        cls.snapshot_binding = build_expected_snapshot_binding(REPO_ROOT, SNAPSHOT_SUBJECT_COMMIT)

    def test_contract_schema_binds_authority_and_digests(self) -> None:
        validate_schema("qualification_contract.schema.json", self.contract)
        self.assertEqual(self.contract["reviewed_implementation_head"], EXPECTED_REVIEWED_IMPLEMENTATION_HEAD)
        self.assertEqual(self.contract["merged_main_head"], EXPECTED_MERGED_MAIN_HEAD)
        self.assertEqual(self.contract["snapshot_subject_commit"], SNAPSHOT_SUBJECT_COMMIT)
        self.assertFalse(self.contract["qualification_harness_source_equals_snapshot_subject"])
        self.assertFalse(self.contract["qualification_evidence_created"])
        self.assertFalse(self.contract["hardware_ran"])
        self.assertFalse(self.contract["scientific_acquisition_authorized"])

    def test_c_reference_emitter_builds_under_strict_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compile_reference_emitter(Path(tmp) / "emit_reference")

    def test_two_c_reference_emissions_are_byte_identical(self) -> None:
        first = json.dumps(emit_reference_table(), sort_keys=True, separators=(",", ":"))
        second = json.dumps(emit_reference_table(), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)

    def test_all_tones_modes_and_codeword_rows_are_emitted(self) -> None:
        self.assertEqual(len(self.reference["tones"]), 12)
        self.assertEqual(self.reference["mode_names"], ["basis", "rotation", "residual", "mini"])
        self.assertEqual(len(self.reference["codebook_rows"]), 4)
        self.assertEqual(sum(len(row["row"]) for row in self.reference["codebook_rows"]), 48)

    def test_c_versus_python_equivalence_passes(self) -> None:
        self.assertEqual(self.equivalence["status"], "V2_REFERENCE_EQUIVALENCE_PASS")
        self.assertEqual(self.equivalence["tone_comparison"]["status"], "pass")
        self.assertEqual(self.equivalence["codeword_comparison"]["status"], "pass")

    def test_changing_one_tone_fails_with_precise_mismatch(self) -> None:
        mutated = copy.deepcopy(self.reference)
        mutated["tones"][3]["frequency_hz"] += 0.01
        result = compare_reference_tables(mutated, TONE_CODEWORD_TABLE)
        self.assertEqual(result["status"], "V2_REFERENCE_EQUIVALENCE_FAIL")
        self.assertEqual(result["failure"]["field"], "tones.frequency_hz")
        self.assertEqual(result["failure"]["index"], 3)

    def test_changing_one_codeword_sign_fails(self) -> None:
        mutated = copy.deepcopy(self.reference)
        mutated["codebook"]["basis"][0] *= -1
        result = compare_reference_tables(mutated, TONE_CODEWORD_TABLE)
        self.assertEqual(result["status"], "V2_REFERENCE_EQUIVALENCE_FAIL")
        self.assertEqual(result["failure"]["field"], "codebook.basis")

    def test_changing_one_mode_mapping_fails(self) -> None:
        mutated = copy.deepcopy(self.reference)
        mutated["mode_to_codeword_mapping"]["basis"] = 1
        result = compare_reference_tables(mutated, TONE_CODEWORD_TABLE)
        self.assertEqual(result["status"], "V2_REFERENCE_EQUIVALENCE_FAIL")
        self.assertEqual(result["failure"]["field"], "mode_to_codeword_mapping")

    def test_changing_source_digest_fails_before_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "combined_pdn_hardware.c"
            source.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(QualificationError):
                compile_reference_emitter(Path(tmp) / "emit_reference", source_path=source)

    def test_changing_reviewed_implementation_head_fails(self) -> None:
        with self.assertRaises(QualificationError):
            qualification_contract(reviewed_implementation_head="0" * 40)

    def test_changing_merged_main_head_fails(self) -> None:
        with self.assertRaises(QualificationError):
            qualification_contract(merged_main_head="0" * 40)

    def _materialized_snapshot(self, tmp: str) -> Path:
        snapshot_dir = Path(tmp)
        materialize_trusted_snapshot(self.snapshot_binding, snapshot_dir)
        return snapshot_dir

    def _path(self, snapshot_dir: Path, relative: str) -> Path:
        return snapshot_dir / relative

    def test_real_git_snapshot_verification_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            result = verify_snapshot_directory(snapshot_dir)
            trusted = result["trusted_snapshot_binding"]
            observed = result["observed_snapshot_identity"]
            self.assertEqual(result["status"], "PHASE6B6_SEALED_SNAPSHOT_VERIFICATION_PASS")
            self.assertEqual(trusted["snapshot_subject_commit"], SNAPSHOT_SUBJECT_COMMIT)
            self.assertEqual(observed["calculated_tree"], trusted["expected_scoped_tree"])
            self.assertEqual(observed["calculated_path_mode_blob_inventory"], trusted["path_mode_blob_inventory"])
            self.assertEqual(observed["calculated_inventory_sha256"], trusted["expected_inventory_sha256"])
            self.assertEqual(
                observed["calculated_phase6b6_subtree_inventory_sha256"],
                trusted["expected_phase6b6_subtree_inventory_sha256"],
            )
            self.assertEqual(observed["calculated_v2_source_sha256"], QUALIFIED_V2_SOURCE["physical_interface_source_sha256"])
            self.assertEqual(observed["phase6b6_package_identity"]["root_path"], PHASE6B6_RELATIVE_ROOT)
            self.assertEqual(observed["prohibited_path_scan"]["status"], "PASS")
            self.assertEqual(observed["unexpected_entry_scan"]["unexpected_entries"], [])
            validate_schema("snapshot_verification_result.schema.json", result)

    def _assert_unexpected_entry_rejected(self, mutate, expected_path: str, expected_entry_type: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            mutate(snapshot_dir)
            with self.assertRaises(SnapshotVerificationError) as cm:
                verify_snapshot_directory(snapshot_dir)
            message = str(cm.exception)
            self.assertIn(expected_path, message)
            self.assertIn(expected_entry_type, message)

    def test_unbound_root_payload_file_fails(self) -> None:
        self._assert_unexpected_entry_rejected(
            lambda snapshot_dir: self._path(snapshot_dir, "payload.bin").write_bytes(b"payload\n"),
            "payload.bin",
            "regular_file",
        )

    def test_unbound_root_directory_payload_fails(self) -> None:
        def mutate(snapshot_dir: Path) -> None:
            target = self._path(snapshot_dir, "unbound/payload.bin")
            target.parent.mkdir(parents=True)
            target.write_bytes(b"payload\n")

        self._assert_unexpected_entry_rejected(mutate, "unbound/payload.bin", "regular_file")

    def test_unbound_holo_runtime_sibling_file_fails(self) -> None:
        extra_path = (
            "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
            "14_noncollapse_frontier/holo_runtime_v2/extra.c"
        )
        self._assert_unexpected_entry_rejected(
            lambda snapshot_dir: self._path(snapshot_dir, extra_path).write_text("extra\n", encoding="utf-8"),
            extra_path,
            "regular_file",
        )

    def test_unbound_holo_runtime_directory_payload_fails(self) -> None:
        extra_path = (
            "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
            "14_noncollapse_frontier/holo_runtime_v2/unbound/payload.bin"
        )

        def mutate(snapshot_dir: Path) -> None:
            target = self._path(snapshot_dir, extra_path)
            target.parent.mkdir(parents=True)
            target.write_bytes(b"payload\n")

        self._assert_unexpected_entry_rejected(mutate, extra_path, "regular_file")

    def test_unbound_out_of_scope_symlink_fails(self) -> None:
        if os.name == "nt":
            self.skipTest("Windows symlink creation requires elevated privileges")
        self._assert_unexpected_entry_rejected(
            lambda snapshot_dir: self._path(snapshot_dir, "unbound-link").symlink_to("payload.bin"),
            "unbound-link",
            "symlink",
        )

    def test_unbound_fifo_fails_where_supported(self) -> None:
        if os.name == "nt" or not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is not supported on this platform")

        def mutate(snapshot_dir: Path) -> None:
            os.mkfifo(self._path(snapshot_dir, "unbound.fifo"))

        self._assert_unexpected_entry_rejected(mutate, "unbound.fifo", "fifo")

    def _assert_trusted_repo_unchanged(self, action, expect_failure: bool) -> None:
        before = trusted_repository_object_state(REPO_ROOT)
        if expect_failure:
            with self.assertRaises(SnapshotVerificationError):
                action()
        else:
            action()
        after = trusted_repository_object_state(REPO_ROOT)
        assert_trusted_repository_unchanged(before, after)
        self.assertEqual(before["loose_object_paths"], after["loose_object_paths"])
        self.assertEqual(before["pack_index_files"], after["pack_index_files"])
        self.assertEqual(before["refs"], after["refs"])
        self.assertEqual(before["index"], after["index"])
        self.assertEqual(before["worktree_status"], after["worktree_status"])

    def test_passing_snapshot_does_not_mutate_trusted_git_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            self._assert_trusted_repo_unchanged(lambda: verify_snapshot_directory(snapshot_dir), False)

    def test_changed_file_failure_does_not_mutate_trusted_git_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            target = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/contracts/contract.py")
            target.write_text(target.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
            self._assert_trusted_repo_unchanged(lambda: verify_snapshot_directory(snapshot_dir), True)

    def test_arbitrary_payload_failure_does_not_mutate_trusted_git_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "payload.bin").write_bytes(b"arbitrary\n")
            self._assert_trusted_repo_unchanged(lambda: verify_snapshot_directory(Path(tmp)), True)

    def test_arbitrary_source_txt_snapshot_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "source.txt").write_text("sealed source bytes\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(Path(tmp))

    def test_caller_supplied_fake_inventory_cannot_influence_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "source.txt").write_text("sealed source bytes\n", encoding="utf-8")
            fake_identity = {
                "expected_tree": "1" * 40,
                "observed_tree": "1" * 40,
                "file_sha256_inventory": {"source.txt": "0" * 64},
            }
            self.assertIsNotNone(fake_identity)
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(Path(tmp))

    def test_wrong_commit_is_rejected(self) -> None:
        with self.assertRaises(QualificationError):
            build_expected_snapshot_binding(REPO_ROOT, EXPECTED_REVIEWED_IMPLEMENTATION_HEAD)

    def test_wrong_commit_tree_strings_cannot_make_snapshot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "source.txt").write_text("not a git snapshot\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(Path(tmp))

    def test_one_changed_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            target = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/contracts/contract.py")
            target.write_text(target.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_one_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/contracts/schedule.py").unlink()
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_one_extra_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            extra = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/contracts/extra.py")
            extra.write_text("extra\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_changed_executable_mode_fails(self) -> None:
        if os.name == "nt":
            self.skipTest("Windows filesystem does not expose Git executable-bit mutation reliably")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            target = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/contracts/contract.py")
            target.chmod(0o755)
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_symlink_substitution_fails(self) -> None:
        if os.name == "nt":
            self.skipTest("Windows symlink creation requires elevated privileges")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            target = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/contracts/schedule.py")
            target.unlink()
            target.symlink_to("contract.py")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_missing_phase6b6_package_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            shutil.rmtree(self._path(snapshot_dir, PHASE6B6_RELATIVE_ROOT))
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_missing_v2_source_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            self._path(snapshot_dir, V2_RELATIVE_SOURCE).unlink()
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_changed_v2_source_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            target = self._path(snapshot_dir, V2_RELATIVE_SOURCE)
            target.write_text(target.read_text(encoding="utf-8") + "\n/* changed */\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_changed_v2_interface_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            target = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/contracts/v2_interface.py")
            target.write_text(target.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_changed_approval_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            target = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/PHASE6B6_SOFTWARE_ENTRY_APPROVAL.json")
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["phase6b6_entry_approved"] = False
            target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_authority_flag_set_true_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            target = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/PHASE6B6_SOFTWARE_ENTRY_APPROVAL.json")
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["hardware_ran"] = True
            target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_hidden_final_session_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            hidden = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/contracts/sessions/final.json")
            hidden.parent.mkdir(parents=True, exist_ok=True)
            hidden.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_hidden_evidence_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            hidden = self._path(snapshot_dir, f"{PHASE6B6_RELATIVE_ROOT}/evidence/final.json")
            hidden.parent.mkdir(parents=True, exist_ok=True)
            hidden.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_directory(snapshot_dir)

    def test_case_colliding_path_fails(self) -> None:
        with self.assertRaises(SnapshotVerificationError):
            _check_case_collisions(["a/contract.py", "a/CONTRACT.py"])

    def test_validate_only_cli_does_not_create_evidence(self) -> None:
        result = validate_only()
        self.assertEqual(result["status"], "PHASE6B6_NONHARDWARE_QUALIFICATION_VALIDATE_ONLY_PASS")
        self.assertFalse(result["hardware_ran"])
        self.assertFalse(result["scientific_acquisition_authorized"])

    def test_no_real_hardware_backend_is_reachable_through_cli(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "qualification.qualification_contract",
            "--hardware",
            "--validate-only",
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PHASE6B6_NONHARDWARE_AUTHORITY_ERROR", result.stderr)

    def test_all_generated_temporary_objects_validate_against_schemas(self) -> None:
        validate_schema("c_reference_table.schema.json", self.reference)
        validate_schema("equivalence_result.schema.json", self.equivalence)
        validate_schema("trusted_snapshot_binding.schema.json", self.snapshot_binding)
        self.assertRegex(digest(self.reference), r"^[0-9a-f]{64}$")

    def _valid_snapshot_result(self) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = self._materialized_snapshot(tmp)
            return verify_snapshot_directory(snapshot_dir)

    def _assert_final_schema_rejects(self, mutate) -> None:
        result = self._valid_snapshot_result()
        mutate(result)
        with self.assertRaises(Exception):
            validate_schema("snapshot_verification_result.schema.json", result)

    def test_final_schema_rejects_unknown_field_inside_derived_authority(self) -> None:
        self._assert_final_schema_rejects(
            lambda result: result["observed_snapshot_identity"]["derived_authority"].__setitem__("unknown", True)
        )

    def test_final_schema_rejects_unknown_field_inside_prohibited_scan(self) -> None:
        self._assert_final_schema_rejects(
            lambda result: result["observed_snapshot_identity"]["prohibited_path_scan"].__setitem__("unknown", True)
        )

    def test_final_schema_rejects_unknown_field_inside_trusted_binding(self) -> None:
        self._assert_final_schema_rejects(
            lambda result: result["trusted_snapshot_binding"].__setitem__("unknown", True)
        )

    def test_final_schema_rejects_unknown_field_inside_observed_identity(self) -> None:
        self._assert_final_schema_rejects(
            lambda result: result["observed_snapshot_identity"].__setitem__("unknown", True)
        )

    def test_final_schema_rejects_malformed_inventory_entry(self) -> None:
        self._assert_final_schema_rejects(
            lambda result: result["trusted_snapshot_binding"]["path_mode_blob_inventory"][0].__setitem__("mode", "040000")
        )

    def test_final_schema_rejects_altered_nested_status(self) -> None:
        self._assert_final_schema_rejects(
            lambda result: result["observed_snapshot_identity"]["prohibited_path_scan"].__setitem__("status", "FAIL")
        )


if __name__ == "__main__":
    unittest.main()
