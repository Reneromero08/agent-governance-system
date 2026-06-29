from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path

from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.contract import (  # noqa: E402
    AUTHORITY,
    NOMINAL_CAMPAIGN_SAMPLE_COUNT,
    ORDER_ARRAYS,
    TOTAL_SLOTS,
    contract_manifest,
    declared_and_executed_order,
    order_family_sequence,
)
from contracts.v2_interface import QUALIFIED_V2_SOURCE, TONE_CODEWORD_TABLE, codebook, tone_hz, verify_v2_table_binding  # noqa: E402
from runtime.explicit_slot_runtime import run_mock  # noqa: E402
from schemas.validate_objects import validate_named  # noqa: E402
from analysis.pipeline import evaluate_sealed, select_on_validation, training_validation_custody  # noqa: E402
from analysis.synthetic import synthetic_custody  # noqa: E402
from contracts.schedule import campaign_schedule, validate_schedule  # noqa: E402


class ScheduleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = campaign_schedule()

    def test_campaign_geometry_and_nominal_samples(self) -> None:
        self.assertEqual(self.schedule["session_count"], 12)
        self.assertEqual(self.schedule["total_slots"], 10368)
        self.assertEqual(TOTAL_SLOTS, 10368)
        self.assertEqual(NOMINAL_CAMPAIGN_SAMPLE_COUNT, 41472000)
        self.assertEqual([session["slot_count"] for session in self.schedule["sessions"]], [864] * 12)
        validate_schedule(self.schedule)

    def test_session_splits_and_routes_are_block_frozen(self) -> None:
        by_split = Counter(session["split"] for session in self.schedule["sessions"])
        self.assertEqual(by_split, {"train": 6, "validation": 2, "test": 4})
        self.assertEqual([session["route"] for session in self.schedule["sessions"][:4]], ["v4s5", "v2s3", "v2s3", "v4s5"])

    def test_order_rotation_including_b5_reverse(self) -> None:
        self.assertEqual(order_family_sequence("b0", "v4s5"), ("FWD", "REV", "RND1", "RND2", "ORDER_LABEL_SHAM"))
        self.assertEqual(order_family_sequence("b1", "v2s3"), ("RND2", "ORDER_LABEL_SHAM", "FWD", "REV", "RND1"))
        self.assertEqual(order_family_sequence("b5", "v4s5"), ("ORDER_LABEL_SHAM", "RND2", "RND1", "REV", "FWD"))

    def test_preamble_prepared_trajectory_tail_counts(self) -> None:
        stages = Counter(slot["stage"] for slot in self.schedule["sessions"][0]["slots"])
        self.assertEqual(stages["preamble"], 96)
        self.assertEqual(stages["prepared_order"], 360)
        self.assertEqual(stages["trajectory"], 384)
        self.assertEqual(stages["tail_drift"], 24)
        preamble_modes = Counter(slot["executed"]["executed_mode"] for slot in self.schedule["sessions"][0]["slots"][:96])
        self.assertEqual(preamble_modes["SENDER_OFF_IDLE"], 48)
        self.assertEqual(preamble_modes["CARRIER_OFF"], 12)
        self.assertEqual(preamble_modes["DECLARATION_SHAM"], 12)
        self.assertEqual(preamble_modes["ANCHOR"], 24)

    def test_order_label_sham_declared_and_executed_are_separate(self) -> None:
        declared_family, declared_order, executed_family, executed_order = declared_and_executed_order("ORDER_LABEL_SHAM", "b0")
        self.assertEqual((declared_family, executed_family), ("RND2", "RND1"))
        self.assertEqual(declared_order, ORDER_ARRAYS["RND2"])
        self.assertEqual(executed_order, ORDER_ARRAYS["RND1"])

    def test_v2_tone_codeword_binding_is_not_order_array(self) -> None:
        self.assertEqual(len(TONE_CODEWORD_TABLE["tones"]), 12)
        self.assertAlmostEqual(TONE_CODEWORD_TABLE["tones"][0]["frequency_hz"], tone_hz(0))
        self.assertEqual(TONE_CODEWORD_TABLE["codebook"], codebook())
        self.assertNotEqual(tuple(TONE_CODEWORD_TABLE["codebook"]["basis"]), ORDER_ARRAYS["FWD"])

    def test_v2_binding_reproduces_qualified_source_identity(self) -> None:
        v2_root = ROOT.parent / "holo_runtime_v2"
        waveform = (v2_root / "waveform_fixture.c").read_text(encoding="utf-8")
        hardware = v2_root / "combined_pdn_hardware.c"
        self.assertIn("log(20.0)", waveform)
        self.assertIn("log(1500.0)", waveform)
        self.assertIn("sin(2.399963 * (index + 1))", waveform)
        self.assertEqual(len([tone_hz(i) for i in range(12)]), 12)
        self.assertEqual(TONE_CODEWORD_TABLE["codebook"]["basis"], codebook()["basis"])
        self.assertEqual(set(TONE_CODEWORD_TABLE["codebook"]), {"basis", "rotation", "residual", "mini"})
        self.assertEqual(TONE_CODEWORD_TABLE["source"]["physical_interface_source_path"], "14_noncollapse_frontier/holo_runtime_v2/combined_pdn_hardware.c")
        self.assertEqual(TONE_CODEWORD_TABLE["source"]["extracted_artifact_sha256"], QUALIFIED_V2_SOURCE["physical_interface_source_sha256"])
        self.assertEqual(TONE_CODEWORD_TABLE["mode_to_codeword_mapping"], {"basis": 0, "rotation": 1, "residual": 2, "mini": 3})
        self.assertEqual(verify_v2_table_binding(hardware)["tone_codeword_table_sha256"], TONE_CODEWORD_TABLE["tone_codeword_table_sha256"])
        self.assertEqual(QUALIFIED_V2_SOURCE["reviewed_source"], "ba48125d15009a044bb869b5716c412b1a8baa1b")
        self.assertEqual(QUALIFIED_V2_SOURCE["source_bundle_sha256"], "bec71b2369587e68a88e9e2b5cb47837a07d5cdef6f13990417e0c0928e85f2f")

    def test_v2_extraction_detects_tone_codeword_and_mapping_mutations(self) -> None:
        source = ROOT.parent / "holo_runtime_v2" / "combined_pdn_hardware.c"
        original = source.read_text(encoding="utf-8")
        mutations = {
            "tone": original.replace("log(1500)", "log(1501)", 1),
            "codeword_bit": original.replace("int weights[4] = {4, 5, 6, 7}", "int weights[4] = {4, 5, 6, 8}", 1),
            "mode_mapping": original.replace('"basis", "rotation", "residual", "mini"', '"basis", "residual", "rotation", "mini"', 1),
        }
        with tempfile.TemporaryDirectory() as temp:
            for label, text in mutations.items():
                with self.subTest(label=label):
                    mutated = Path(temp) / f"{label}.c"
                    mutated.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        verify_v2_table_binding(mutated)

    def test_generated_objects_validate_against_schemas(self) -> None:
        validate_named("scientific_contract.schema.json", contract_manifest())
        validate_named("schedule.schema.json", self.schedule)
        custody = run_mock(self.schedule)
        validate_named("runtime_custody.schema.json", custody)
        custody = synthetic_custody("shared_driven")
        manifest = select_on_validation(training_validation_custody(custody))
        validate_named("analysis_choice.schema.json", manifest)
        result = evaluate_sealed(custody, manifest)
        validate_named("adjudication_result.schema.json", result["adjudication"])

    def test_recursive_custody_schema_rejects_sender_off_codeword(self) -> None:
        custody = run_mock(self.schedule)
        mutated = deepcopy(custody)
        off_row = next(slot for session in mutated["sessions"] for slot in session["slots"] if not slot["u_t"]["drive_on"])
        off_row["u_t"]["codeword_sign"] = 1
        with self.assertRaises(ValidationError):
            validate_named("runtime_custody.schema.json", mutated)

    def test_no_sender_epoch_in_sender_off_slots_and_contiguous_step_epoch(self) -> None:
        first_session = self.schedule["sessions"][0]
        off_slots = [slot for slot in first_session["slots"] if not slot["executed"]["drive_on"]]
        self.assertTrue(off_slots)
        self.assertTrue(all(slot["executed"]["sender_epoch_id"] is None for slot in off_slots))
        self.assertTrue(all(slot["executed"]["executed_codeword_signs"] is None for slot in off_slots))
        self.assertTrue(all(slot["executed"]["executed_v2_mode"] is None for slot in off_slots))
        driven = [slot for slot in first_session["slots"] if slot["executed"]["drive_on"]]
        self.assertTrue(all(len(slot["executed"]["executed_codeword_signs"]) == 12 for slot in driven))
        self.assertTrue(all(slot["executed"]["executed_v2_mode"] == "basis" for slot in driven))
        self.assertTrue(all(slot["executed"]["physical_tone_index"] is None for slot in off_slots))
        step_slots = [slot for slot in first_session["slots"] if slot["packet_id"] == "s0:tone0:step" and slot["executed"]["drive_on"]]
        self.assertEqual(len(step_slots), 4)
        self.assertEqual(len({slot["executed"]["sender_epoch_id"] for slot in step_slots}), 1)
        self.assertEqual([slot["slot_index"] for slot in step_slots], list(range(step_slots[0]["slot_index"], step_slots[0]["slot_index"] + 4)))

    def test_contract_authority_flags_remain_software_only(self) -> None:
        self.assertTrue(AUTHORITY["implementation_authorized"])
        for field in (
            "hardware_ran",
            "authorization_artifact_created",
            "calibration_authorized",
            "scientific_acquisition_authorized",
            "restoration_authorized",
            "target_coupling_authorized",
            "small_wall_authorized",
        ):
            self.assertFalse(AUTHORITY[field])

    def test_deterministic_temp_tree_generation(self) -> None:
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            left_path = Path(left) / "schedule.json"
            right_path = Path(right) / "schedule.json"
            cmd = [sys.executable, "-m", "contracts.schedule", "--out"]
            subprocess.run(cmd + [str(left_path)], cwd=ROOT, check=True, text=True, capture_output=True)
            subprocess.run(cmd + [str(right_path)], cwd=ROOT, check=True, text=True, capture_output=True)
            self.assertEqual(left_path.read_bytes(), right_path.read_bytes())
            payload = json.loads(left_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total_slots"], 10368)


if __name__ == "__main__":
    unittest.main()
