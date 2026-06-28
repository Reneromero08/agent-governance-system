from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

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
from contracts.v2_interface import TONE_CODEWORD_TABLE, codebook, tone_hz  # noqa: E402
from runtime.explicit_slot_runtime import run_mock  # noqa: E402
from schemas.validate_objects import validate_named  # noqa: E402
from analysis.pipeline import evaluate_sealed, select_on_validation  # noqa: E402
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

    def test_generated_objects_validate_against_schemas(self) -> None:
        validate_named("schedule.schema.json", self.schedule)
        custody = run_mock(self.schedule)
        validate_named("runtime_custody.schema.json", custody)
        manifest = select_on_validation(synthetic_custody("shared_driven"))
        validate_named("analysis_choice.schema.json", manifest)
        result = evaluate_sealed(synthetic_custody("shared_driven"), manifest)
        validate_named("adjudication_result.schema.json", result["adjudication"])

    def test_no_sender_epoch_in_sender_off_slots_and_contiguous_step_epoch(self) -> None:
        first_session = self.schedule["sessions"][0]
        off_slots = [slot for slot in first_session["slots"] if not slot["executed"]["drive_on"]]
        self.assertTrue(off_slots)
        self.assertTrue(all(slot["executed"]["sender_epoch_id"] is None for slot in off_slots))
        self.assertTrue(all(slot["executed"]["codeword_bin_permutation"] is None for slot in off_slots))
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
