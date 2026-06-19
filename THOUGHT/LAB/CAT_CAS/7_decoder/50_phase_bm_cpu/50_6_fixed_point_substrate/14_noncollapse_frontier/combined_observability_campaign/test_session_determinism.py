#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from campaign_plan import make_plan
from compile_session_schedule import write_session


class SessionDeterminismTests(unittest.TestCase):
    def test_written_session_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text(json.dumps(make_plan("a" * 40, "b" * 64), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            left = root / "left"
            right = root / "right"
            self.assertEqual(write_session(plan, "v2s3_seed5", left), write_session(plan, "v2s3_seed5", right))
            for name in ("session.json", "windows.jsonl", "session_manifest.json"):
                self.assertEqual((left / name).read_bytes(), (right / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
