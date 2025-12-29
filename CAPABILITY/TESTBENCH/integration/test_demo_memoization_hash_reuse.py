from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_ROOT = REPO_ROOT / "CAPABILITY" / "TESTBENCH" / "integration" / "_demos" / "memoization_hash_reuse"


def test_phase2_demo_artifacts_are_falsifiable() -> None:
    baseline = DEMO_ROOT / "baseline"
    reuse = DEMO_ROOT / "reuse"

    assert (baseline / "PROOF.json").exists()
    assert (baseline / "LEDGER.jsonl").exists()
    assert (baseline / "DEREF_STATS.json").exists()

    assert (reuse / "PROOF.json").exists()
    assert (reuse / "LEDGER.jsonl").exists()
    assert (reuse / "DEREF_STATS.json").exists()

    # Proof identity: byte-identical.
    assert (baseline / "PROOF.json").read_bytes() == (reuse / "PROOF.json").read_bytes()

    # Memoization hit must be observable.
    assert "memoization:hit" in (reuse / "LEDGER.jsonl").read_text(encoding="utf-8")

    # Hash-first dereference must be measurably smaller (bytes read) in reuse.
    b_deref_stats = json.loads((baseline / "DEREF_STATS.json").open().read())
    r_deref_stats = json.loads((reuse / "DEREF_STATS.json").open().read())

    assert b_deref_stats["deref_count"] > r_deref_stats["deref_count"]
    assert (b_deref_stats["bytes_read_total"] - r_deref_stats["bytes_read_total"]) > 0
