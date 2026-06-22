from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_frozen_artifact_binding import (
    ADJUDICATION_REL,
    CACHE_RELS,
    MODEL_REL,
    PERMANENT_STATEMENTS,
    BindingContract,
    audit_binding,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root: Path) -> BindingContract:
    (root / "feature_cache").mkdir(parents=True)
    model = {
        "seed5_used_for_selection": False,
        "selection_sessions": ["v2s3_seed3", "v4s5_seed3"],
        "selected_operator": {"operator": "O2_BILINEAR"},
        "selected_operator_coefficients": np.zeros((63, 2)).tolist(),
        "stage_b_centroids": np.zeros((32, 14)).tolist(),
        "stage_b_labels": [f"label:{index}" for index in range(32)],
    }
    (root / MODEL_REL).write_text(json.dumps(model), encoding="utf-8")
    ledger = {
        "stage_b_by_heldout_session": {
            session: {"joint_accuracy": .1, "mode_accuracy": .2, "theta_accuracy": .3}
            for session in ("v2s3_seed5", "v4s5_seed5")
        },
        "stage_c_by_heldout_session": {
            session: {"zero_input_gain": .01}
            for session in ("v2s3_seed5", "v4s5_seed5")
        },
        "operator_by_heldout_session": {
            session: {"nrmse": .9}
            for session in ("v2s3_seed5", "v4s5_seed5")
        },
        "stage_b_verdict": "NO_ORDER_RESOLUTION",
        "stage_c_verdict": "DRIVEN_RELATIONAL_TRANSPORT_ONLY",
        "predictive_operator_verdict": "NO_STABLE_PREDICTIVE_OPERATOR",
    }
    adjudication = {
        "schema_id": "CAT_CAS_PHASE6_V1_FULL_RAW_ADJUDICATION_V1",
        "IMPLEMENTATION_RECOVERY_ANALYSIS": ledger,
    }
    (root / ADJUDICATION_REL).write_text(json.dumps(adjudication), encoding="utf-8")
    for relative in CACHE_RELS:
        np.savez(
            root / relative,
            recovery=np.zeros((4, 6, 2)), original=np.zeros((4, 2)),
            gate=np.zeros((4, 2)), raw_sha256=np.array("a" * 64),
        )
    return BindingContract(
        model_sha256=digest(root / MODEL_REL),
        adjudication_sha256=digest(root / ADJUDICATION_REL),
        cache_sha256={relative: digest(root / relative) for relative in CACHE_RELS},
        cache_shapes={"recovery": (4, 6, 2), "original": (4, 2), "gate": (4, 2)},
        cache_nonfinite_counts={"recovery": 0, "original": 0, "gate": 0},
    )


class HermeticBindingAuditTests(unittest.TestCase):
    def test_complete_synthetic_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = audit_binding(root, make_fixture(root))
        self.assertFalse(result["independent_metric_recomputation_performed"])
        self.assertFalse(result["binding_audit_seed5_retry_performed"])
        self.assertFalse(result["historical_pristine_seed5_hygiene_proven"])
        self.assertTrue(result["historical_seed5_reexecution_occurred"])
        self.assertNotIn("seed5_retry_performed", result)
        self.assertEqual(tuple(result["permanent_v1_statement"]), PERMANENT_STATEMENTS)
        self.assertEqual(result["input_class"], "EXTERNALLY_SUPPLIED_EVIDENCE_ARTIFACTS")
        self.assertNotIn(":\\", json.dumps(result))

    def test_missing_external_evidence_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "required external evidence artifact absent"):
                audit_binding(Path(temp))

    def test_model_digest_rejection_precedes_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = make_fixture(root)
            (root / MODEL_REL).write_bytes(b"not json")
            with self.assertRaisesRegex(ValueError, "frozen model digest mismatch"):
                audit_binding(root, contract)

    def test_cache_digest_rejection_precedes_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = make_fixture(root)
            (root / CACHE_RELS[0]).write_bytes(b"not npz")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                audit_binding(root, contract)

    def test_adjudication_digest_rejection_precedes_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = make_fixture(root)
            (root / ADJUDICATION_REL).write_bytes(b"not json")
            with self.assertRaisesRegex(ValueError, "recorded adjudication digest mismatch"):
                audit_binding(root, contract)

    def test_model_schema_and_shape_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = make_fixture(root)
            model = json.loads((root / MODEL_REL).read_text())
            del model["stage_b_labels"]
            (root / MODEL_REL).write_text(json.dumps(model))
            contract = BindingContract(digest(root / MODEL_REL), contract.adjudication_sha256,
                                       contract.cache_sha256, contract.cache_shapes,
                                       contract.cache_nonfinite_counts)
            with self.assertRaisesRegex(ValueError, "schema"):
                audit_binding(root, contract)

    def test_nonfinite_model_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = make_fixture(root)
            model = json.loads((root / MODEL_REL).read_text())
            model["stage_b_centroids"][0][0] = float("nan")
            (root / MODEL_REL).write_text(json.dumps(model))
            contract = BindingContract(digest(root / MODEL_REL), contract.adjudication_sha256,
                                       contract.cache_sha256, contract.cache_shapes,
                                       contract.cache_nonfinite_counts)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                audit_binding(root, contract)


if __name__ == "__main__":
    unittest.main()
