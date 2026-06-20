#!/usr/bin/env python3
from __future__ import annotations

import unittest

from campaign_plan import make_plan
from compile_session_schedule import compile_session, validate

SOURCE = "a" * 40


class SessionScheduleTests(unittest.TestCase):
    def test_controls_and_sender_off(self) -> None:
        header, windows = compile_session(make_plan(SOURCE, "b" * 64), "v4s5_seed4")
        self.assertEqual(validate(header, windows), [])
        self.assertEqual(header["partition"], "stress")
        self.assertEqual([row["window_index"] for row in windows], list(range(len(windows))))

        off = [row for row in windows if row["stage"] == "C_PERSISTENCE_OFF"]
        self.assertEqual(len(off), 512)
        self.assertTrue(all(not row["drive_on"] and row["sender_off_required"] for row in off))
        self.assertTrue(all(row["measurement_mode"] == "raw_ring_sender_off" for row in off))

        sham = [row for row in windows if row["family"] == "order_sham"]
        self.assertTrue(sham)
        self.assertTrue(all(row["executed_tone_order"] != row["declared_tone_order"] for row in sham))

        silent = [row for row in windows if row["family"] == "silent"]
        self.assertTrue(silent)
        self.assertTrue(all(not row["drive_on"] for row in silent))

    def test_tampered_sender_off_rejected(self) -> None:
        header, windows = compile_session(make_plan(SOURCE, "b" * 64), "v4s5_seed0")
        target = next(row for row in windows if row["stage"] == "C_PERSISTENCE_OFF")
        target["drive_on"] = True
        self.assertTrue(any("sender-off" in error for error in validate(header, windows)))


if __name__ == "__main__":
    unittest.main()
