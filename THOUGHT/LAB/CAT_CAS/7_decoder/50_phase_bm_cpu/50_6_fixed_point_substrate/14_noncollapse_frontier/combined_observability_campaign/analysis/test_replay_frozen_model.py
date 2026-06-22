from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replay_frozen_model import MODEL_SHA256, replay


class ReplayTests(unittest.TestCase):
    def test_replay_confirms_frozen_outputs_without_selection(self) -> None:
        root = Path(__file__).resolve().parents[9] / "LAW" / "CONTRACTS" / "_runs"
        evidence = root / "phase6_v1_full_raw_adjudication_7c44af0f"
        result = replay(evidence / "frozen_model_before_final_test.json", evidence)
        self.assertEqual(result["frozen_model"]["sha256"], MODEL_SHA256)
        self.assertFalse(result["model_selection_performed"])
        self.assertFalse(result["seed5_retry_performed"])
        self.assertEqual(len(result["recorded_seed5_outputs_confirmed"]), 2)

    def test_digest_mismatch_fails_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / "model.json"
            bad.write_text(json.dumps({"not": "the model"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                replay(bad, Path(temp))


if __name__ == "__main__":
    unittest.main()
