from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from contracts.contract import AUTHORITY
from contracts.v2_interface import QUALIFIED_V2_SOURCE, TONE_CODEWORD_TABLE
from qualification.compare_v2_reference import compare_reference_tables
from qualification.qualification_contract import (
    EXPECTED_MERGED_MAIN_HEAD,
    EXPECTED_REVIEWED_IMPLEMENTATION_HEAD,
    QualificationError,
    compile_reference_emitter,
    digest,
    emit_reference_table,
    qualification_contract,
    qualification_authority_state,
    validate_only,
    validate_schema,
)
from qualification.verify_sealed_snapshot import SnapshotVerificationError, file_sha256, verify_snapshot_identity


class NonHardwareQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = emit_reference_table()
        cls.equivalence = compare_reference_tables(cls.reference, TONE_CODEWORD_TABLE)
        cls.contract = qualification_contract()

    def test_contract_schema_binds_authority_and_digests(self) -> None:
        validate_schema("qualification_contract.schema.json", self.contract)
        self.assertEqual(self.contract["reviewed_implementation_head"], EXPECTED_REVIEWED_IMPLEMENTATION_HEAD)
        self.assertEqual(self.contract["merged_main_head"], EXPECTED_MERGED_MAIN_HEAD)
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

    def _snapshot_identity(self, snapshot_dir: Path) -> dict[str, object]:
        (snapshot_dir / "source.txt").write_text("sealed source bytes\n", encoding="utf-8")
        inventory = {"source.txt": file_sha256(snapshot_dir / "source.txt")}
        return {
            "schema_id": "CAT_CAS_PHASE6B6_SNAPSHOT_IDENTITY_V1",
            "expected_commit": EXPECTED_MERGED_MAIN_HEAD,
            "expected_tree": "1" * 40,
            "observed_tree": "1" * 40,
            "file_sha256_inventory": inventory,
            "qualification_contract_digest": self.contract["qualification_contract_sha256"],
            "phase6b6_source_package_identity": {"root": "phase6b6"},
            "v2_source_identity": {
                "path": QUALIFIED_V2_SOURCE["physical_interface_source_path"],
                "sha256": QUALIFIED_V2_SOURCE["physical_interface_source_sha256"],
            },
            "generated_final_campaign_sessions_present": False,
            "authority": copy.deepcopy(qualification_authority_state()),
        }

    def test_snapshot_identity_passes_for_clean_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp)
            result = verify_snapshot_identity(snapshot_dir, self._snapshot_identity(snapshot_dir))
            self.assertEqual(result["status"], "PHASE6B6_SEALED_SNAPSHOT_VERIFICATION_PASS")

    def test_dirty_or_extra_snapshot_files_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp)
            identity = self._snapshot_identity(snapshot_dir)
            (snapshot_dir / "extra.txt").write_text("unbound\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_identity(snapshot_dir, identity)

    def test_changed_snapshot_bytes_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp)
            identity = self._snapshot_identity(snapshot_dir)
            (snapshot_dir / "source.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(SnapshotVerificationError):
                verify_snapshot_identity(snapshot_dir, identity)

    def test_hardware_authority_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp)
            identity = self._snapshot_identity(snapshot_dir)
            identity["authority"]["hardware_ran"] = True  # type: ignore[index]
            with self.assertRaises(Exception):
                verify_snapshot_identity(snapshot_dir, identity)

    def test_acquisition_authority_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp)
            identity = self._snapshot_identity(snapshot_dir)
            identity["authority"]["scientific_acquisition_authorized"] = True  # type: ignore[index]
            with self.assertRaises(Exception):
                verify_snapshot_identity(snapshot_dir, identity)

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
        self.assertRegex(digest(self.reference), r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
