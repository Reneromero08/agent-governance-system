from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_frozen_artifact_binding import MODEL_SHA256, audit_binding


class BindingAuditTests(unittest.TestCase):
    def test_binding_is_explicitly_not_independent_recomputation(self) -> None:
        root = Path(__file__).resolve().parents[9] / "LAW" / "CONTRACTS" / "_runs"
        evidence = root / "phase6_v1_full_raw_adjudication_7c44af0f"
        result = audit_binding(evidence / "frozen_model_before_final_test.json", evidence)
        self.assertEqual(result["frozen_model"]["sha256"], MODEL_SHA256)
        self.assertFalse(result["model_selection_performed"])
        self.assertFalse(result["independent_metric_recomputation_performed"])
        self.assertFalse(result["seed5_retry_performed"])
        self.assertEqual(len(result["recorded_seed5_output_bindings"]), 2)
        self.assertEqual(len(result["recomputation_blockers"]), 4)

    def test_digest_mismatch_fails_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / "model.json"
            bad.write_text(json.dumps({"not": "the model"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                audit_binding(bad, Path(temp))

    def test_cache_digest_mismatch_fails_before_npz_deserialization(self) -> None:
        root = Path(__file__).resolve().parents[9] / "LAW" / "CONTRACTS" / "_runs"
        evidence = root / "phase6_v1_full_raw_adjudication_7c44af0f"
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp)
            (fake / "feature_cache").mkdir()
            (fake / "feature_cache" / "v2s3_seed5.npz").write_bytes(b"not an npz")
            with self.assertRaisesRegex(ValueError, "cache digest mismatch"):
                audit_binding(evidence / "frozen_model_before_final_test.json", fake)


if __name__ == "__main__":
    unittest.main()
