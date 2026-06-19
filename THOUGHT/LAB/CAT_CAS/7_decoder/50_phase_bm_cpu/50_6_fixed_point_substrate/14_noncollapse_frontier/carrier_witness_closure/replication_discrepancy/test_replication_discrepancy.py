#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("replication", HERE / "analyze_replication_discrepancy.py")
replication = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(replication)


def rows():
    result = []
    for trial in range(8):
        for family in ("real", "pseudo", "wrong"):
            mode = trial % 4
            z = np.array(replication.regenerate_codebook()[replication.MODES[mode]], dtype=complex)
            if family == "pseudo":
                z = np.roll(z, 1)
            actual = mode if family != "wrong" else (mode + 1) % 4
            result.append({"family": family, "declared": mode, "actual": actual,
                           "trial": trial, "theta_idx": 0, "hash_restored": 1, "z": z})
    return result


class ReplicationTests(unittest.TestCase):
    def test_codebook_and_schedule_are_deterministic(self):
        code = replication.regenerate_codebook()
        self.assertEqual(code, replication.regenerate_codebook())
        self.assertEqual(replication.regenerate_schedule(4, 48, code),
                         replication.regenerate_schedule(4, 48, code))

    def test_count_and_sparse_denominator_reporting(self):
        rs = rows()
        replication.enrich(rs, replication.regenerate_codebook())
        out = replication.decompose(rs)
        self.assertEqual(sum(out["real_test_rows_per_declared_mode"].values()), 4)
        self.assertIn("combined_denominator", out["pseudo_groups"]["basis"])

    def test_threshold_and_partition_diagnostics_do_not_mutate_verdict(self):
        rs = rows()
        replication.enrich(rs, replication.regenerate_codebook())
        official = replication.decompose(rs)
        alternative = replication.decompose(rs, grouping="declared",
                                             train_selector=lambda r: r["trial"] % 2 == 1,
                                             test_selector=lambda r: r["trial"] % 2 == 0)
        self.assertEqual(official["grouping"], "predicted")
        self.assertEqual(alternative["grouping"], "declared")

    def test_exact_binomial_interval(self):
        low, high = replication.ci_exact(0, 4)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.5)

    def test_manifest_hash_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            replication.dump_json(path, {"x": 1})
            self.assertEqual(len(replication.sha256(path)), 64)

    def test_forbidden_verdict_mutation_label(self):
        self.assertEqual(replication.LABEL, "DIAGNOSTIC_ONLY__NOT_OFFICIAL_VERDICT")
        source = (HERE / "analyze_replication_discrepancy.py").read_text(encoding="utf-8")
        self.assertNotIn('"official_verdict": "PHASE4B_CROSS_CORE_PDN_LOCKIN_WITNESS"', source)

    def test_raw_record_layout(self):
        self.assertEqual(replication.RECORD.size, 16)

    def test_seed4_extraction_key_is_stable(self):
        self.assertEqual("v4s5_matrix_seed4".split("_matrix_seed"), ["v4s5", "4"])


if __name__ == "__main__":
    unittest.main()
