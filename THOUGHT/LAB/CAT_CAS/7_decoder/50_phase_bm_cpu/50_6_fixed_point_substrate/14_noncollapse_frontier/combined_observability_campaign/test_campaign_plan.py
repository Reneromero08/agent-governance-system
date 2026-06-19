#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from campaign_orders import frozen_orders
from campaign_plan import make_plan, validate
from generate_campaign_plan import generate, verify

SOURCE = "a" * 40


class CombinedCampaignPlanTests(unittest.TestCase):
    def test_orders_are_distinct_and_balanced(self) -> None:
        orders = frozen_orders()
        self.assertEqual(set(orders), {"FWD", "REV", "RND1", "RND2"})
        for values in orders.values():
            self.assertEqual(set(values), set(range(12)))
        self.assertEqual(len({tuple(values) for values in orders.values()}), 4)

        plan = make_plan(SOURCE, "b" * 64)
        self.assertEqual(validate(plan), [])
        counts = {(name, pos): 0 for name in orders for pos in range(4)}
        for session in plan["sessions"]:
            for pos, name in enumerate(session["order_sequence"]):
                counts[(name, pos)] += 1
        self.assertTrue(all(value == 3 for value in counts.values()))

    def test_explicit_counts_and_partitions(self) -> None:
        plan = make_plan(SOURCE, "b" * 64)
        self.assertEqual(len(plan["sessions"]), 12)
        tone_symbols = sum(len(block["symbols"]) for session in plan["sessions"] for block in session["blocks"]["tone_order"])
        persistence = sum(len(session["blocks"]["persistence"]) for session in plan["sessions"])
        trajectories = sum(len(block["steps"]) for session in plan["sessions"] for block in session["blocks"]["trajectories"])
        self.assertEqual(tone_symbols, 3456)
        self.assertEqual(persistence, 768)
        self.assertEqual(trajectories, 3072)
        for route in ("v4s5", "v2s3"):
            route_sessions = [session for session in plan["sessions"] if session["route"] == route]
            self.assertEqual([session["partition"] for session in route_sessions], ["train", "train", "train", "validation", "stress", "test"])

    def test_no_smuggle_controls(self) -> None:
        plan = make_plan(SOURCE, "b" * 64)
        for session in plan["sessions"]:
            self.assertEqual(len(session["blocks"]["gauge_preamble"]), 8)
            for block in session["blocks"]["tone_order"]:
                families = {row["family"] for row in block["symbols"]}
                self.assertEqual(families, {"real", "wrong", "pseudo", "order_sham", "silent", "scramble"})
                for row in block["symbols"]:
                    if row["family"] == "wrong":
                        self.assertNotEqual(row["actual_mode"], row["declared_mode"])
                    if row["family"] == "order_sham":
                        self.assertNotEqual(row["executed_tone_order"], row["declared_tone_order"])
                    if row["family"] == "pseudo":
                        self.assertNotEqual(row["codeword_bin_permutation"], list(range(12)))
                    if row["family"] == "silent":
                        self.assertFalse(row["drive_on"])
            for event in session["blocks"]["persistence"]:
                self.assertTrue(event["sender_off_required"])
                self.assertFalse(event["periodic_refresh_allowed"])
                self.assertFalse(event["hidden_replay_allowed"])
        self.assertFalse(plan["restoration_authorized"])

    def test_manifest_roundtrip_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ratification = root / "ratification.json"
            ratification.write_text('{"decision":"authorized"}\n', encoding="utf-8")
            output = root / "output"
            manifest = generate(output, SOURCE, ratification)
            self.assertEqual(manifest["summary"]["sessions"], 12)
            self.assertEqual(verify(output), [])
            plan_path = output / "campaign_plan.json"
            plan_path.write_text(plan_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            self.assertTrue(verify(output))


if __name__ == "__main__":
    unittest.main()
