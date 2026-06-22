from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from receiver_schedule import (
    SENDER_PRIVATE_FIELDS,
    load_receiver_schedule,
    physical_gate_digest,
)

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
import sys
sys.path.insert(0, str(ANALYSIS))
from waveform_reference import intended_v2_gate  # noqa: E402


class ReceiverScheduleTests(unittest.TestCase):
    def test_projection_removes_sender_private_scramble_fields(self) -> None:
        row = {
            "window_index": 0,
            "receiver_codeword_source_index": 4,
            "sender_codeword_source_index": 9,
            "receiver_theta_idx": 2,
            "sender_theta_idx": 7,
            "shared_schedule": False,
            "scramble_key_digest": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "windows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            visible = load_receiver_schedule(path)[0]
        self.assertFalse(SENDER_PRIVATE_FIELDS & visible.keys())
        self.assertEqual(visible["receiver_theta_idx"], 2)

    def test_scramble_produces_distinct_physical_gate_digest(self) -> None:
        timestamps = np.arange(0, 100_000, dtype=np.uint64)
        common = dict(origin_tsc=0, tsc_hz=8_000_000.0,
                      tone_index=5, amplitude_level=2)
        receiver = intended_v2_gate(
            timestamps, phase_index_value=2, **common
        ).astype(np.uint8).tolist()
        sender = intended_v2_gate(
            timestamps, phase_index_value=7, **common
        ).astype(np.uint8).tolist()
        self.assertNotEqual(
            physical_gate_digest(receiver), physical_gate_digest(sender)
        )


if __name__ == "__main__":
    unittest.main()
