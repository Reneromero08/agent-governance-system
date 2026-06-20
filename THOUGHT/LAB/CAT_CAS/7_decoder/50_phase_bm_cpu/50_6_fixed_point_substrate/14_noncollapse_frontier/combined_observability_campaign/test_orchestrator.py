#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from campaign_plan import make_plan
from run_combined_campaign import runner_command, selected_sessions


def args_with(commit: str | None = "a" * 40, validate: bool = False):
    return argparse.Namespace(
        pin_khz=1600000,
        slot_s=0.5,
        off_window_s=0.5,
        read_hz=4000,
        temp_veto_c=68.0,
        executor_commit=commit,
        runner_validate_only=validate,
    )


class OrchestratorTests(unittest.TestCase):
    def test_route_core_mapping_and_command(self) -> None:
        args = args_with()
        command = runner_command(
            Path("/runner"), Path("/session"), Path("/output"), "v4s5", args
        )
        self.assertEqual(command[command.index("--victim") + 1], "4")
        self.assertEqual(command[command.index("--sender") + 1], "5")
        command = runner_command(
            Path("/runner"), Path("/session"), Path("/output"), "v2s3", args
        )
        self.assertEqual(command[command.index("--victim") + 1], "2")
        self.assertEqual(command[command.index("--sender") + 1], "3")

    def test_hardware_requires_valid_executor_commit(self) -> None:
        for commit in (None, "0" * 40, "A" * 40, "g" * 40, "abc"):
            with self.subTest(commit=commit):
                with self.assertRaisesRegex(ValueError, "executor-commit"):
                    runner_command(
                        Path("/runner"),
                        Path("/session"),
                        Path("/output"),
                        "v4s5",
                        args_with(commit),
                    )

    def test_validation_omits_executor_provenance(self) -> None:
        command = runner_command(
            Path("/runner"),
            Path("/session"),
            Path("/output"),
            "v4s5",
            args_with(None, validate=True),
        )
        self.assertNotIn("--executor-commit", command)
        self.assertIn("--validate-only", command)

    def test_session_selection_preserves_request_order(self) -> None:
        plan = make_plan("a" * 40, "b" * 64)
        chosen = selected_sessions(plan, ["v2s3_seed5", "v4s5_seed4"])
        self.assertEqual(
            [item["session_id"] for item in chosen],
            ["v2s3_seed5", "v4s5_seed4"],
        )
        with self.assertRaises(ValueError):
            selected_sessions(plan, ["missing"])


if __name__ == "__main__":
    unittest.main()
