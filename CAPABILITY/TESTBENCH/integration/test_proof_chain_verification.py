#!/usr/bin/env python3
"""
Phase 4.4: Chain Verification Tests (SPECTRUM-03)

Tests for:
- Proof chaining with previous_proof_hash
- Chain verification (verify_chain)
- Chain reconstruction (get_chain_history)

Exit Criteria:
- Chains link correctly via previous_proof_hash
- verify_chain detects broken/tampered chains
- Chain history can be reconstructed from head
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.restore_proof import (
    RestorationProofValidator,
    verify_chain,
    get_chain_history,
    compute_proof_hash,
)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def proof_schema_path() -> Path:
    return REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json"


@pytest.fixture
def make_simple_proof(proof_schema_path: Path):
    """Factory to create simple proofs for testing."""
    validator = RestorationProofValidator(proof_schema_path)

    def _make(
        run_id: str,
        previous_hash: str | None = None,
    ) -> dict:
        state = {
            "domain": {
                f"{run_id}.txt": _sha256_hex(run_id.encode()),
            }
        }
        return validator.generate_proof(
            run_id=run_id,
            catalytic_domains=["domain"],
            pre_state=state,
            post_state=state,
            timestamp="2025-01-01T00:00:00Z",
            previous_proof_hash=previous_hash,
        )

    return _make


# =============================================================================
# Chain Generation Tests
# =============================================================================


class TestChainGeneration:
    """Tests for generating chained proofs."""

    def test_first_proof_has_no_previous(self, make_simple_proof):
        """First proof in chain has no previous_proof_hash."""
        proof = make_simple_proof("run-1")
        assert "previous_proof_hash" not in proof

    def test_second_proof_links_to_first(self, make_simple_proof):
        """Second proof links to first via previous_proof_hash."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2", previous_hash=proof1["proof_hash"])

        assert proof2["previous_proof_hash"] == proof1["proof_hash"]

    def test_chain_of_three(self, make_simple_proof):
        """Chain of three proofs links correctly."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2", previous_hash=proof1["proof_hash"])
        proof3 = make_simple_proof("run-3", previous_hash=proof2["proof_hash"])

        assert "previous_proof_hash" not in proof1
        assert proof2["previous_proof_hash"] == proof1["proof_hash"]
        assert proof3["previous_proof_hash"] == proof2["proof_hash"]


# =============================================================================
# verify_chain Tests
# =============================================================================


class TestVerifyChain:
    """Tests for verify_chain function."""

    def test_empty_chain_rejected(self):
        """Empty chain is rejected."""
        result = verify_chain([])
        assert result["ok"] is False
        assert result["code"] == "CHAIN_EMPTY"

    def test_single_proof_chain(self, make_simple_proof):
        """Single proof chain is valid."""
        proof = make_simple_proof("run-1")
        result = verify_chain([proof])

        assert result["ok"] is True
        assert result["code"] == "CHAIN_VALID"
        assert result["chain_length"] == 1
        assert result["chain_root"] == proof["proof_hash"]
        assert result["chain_head"] == proof["proof_hash"]

    def test_valid_two_proof_chain(self, make_simple_proof):
        """Valid two-proof chain verifies."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2", previous_hash=proof1["proof_hash"])

        result = verify_chain([proof1, proof2])

        assert result["ok"] is True
        assert result["code"] == "CHAIN_VALID"
        assert result["chain_length"] == 2
        assert result["chain_root"] == proof1["proof_hash"]
        assert result["chain_head"] == proof2["proof_hash"]

    def test_valid_three_proof_chain(self, make_simple_proof):
        """Valid three-proof chain verifies."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2", previous_hash=proof1["proof_hash"])
        proof3 = make_simple_proof("run-3", previous_hash=proof2["proof_hash"])

        result = verify_chain([proof1, proof2, proof3])

        assert result["ok"] is True
        assert result["code"] == "CHAIN_VALID"
        assert result["chain_length"] == 3

    def test_first_proof_with_previous_rejected(self, make_simple_proof):
        """First proof with previous_proof_hash is rejected."""
        proof1 = make_simple_proof("run-1", previous_hash="a" * 64)

        result = verify_chain([proof1])

        assert result["ok"] is False
        assert result["code"] == "CHAIN_ROOT_HAS_PREVIOUS"
        assert result["failed_at_index"] == 0

    def test_missing_link_rejected(self, make_simple_proof):
        """Chain with missing link is rejected."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2")  # No link!

        result = verify_chain([proof1, proof2])

        assert result["ok"] is False
        assert result["code"] == "CHAIN_LINK_MISSING"
        assert result["failed_at_index"] == 1

    def test_wrong_link_rejected(self, make_simple_proof):
        """Chain with wrong link is rejected."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2", previous_hash="b" * 64)  # Wrong!

        result = verify_chain([proof1, proof2])

        assert result["ok"] is False
        assert result["code"] == "CHAIN_LINK_MISMATCH"
        assert result["failed_at_index"] == 1

    def test_tampered_proof_rejected(self, make_simple_proof):
        """Tampered proof (modified after hashing) is rejected."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2", previous_hash=proof1["proof_hash"])

        # Tamper with proof1 after it was created
        proof1["run_id"] = "tampered"

        result = verify_chain([proof1, proof2])

        assert result["ok"] is False
        assert result["code"] == "PROOF_HASH_MISMATCH"
        assert result["failed_at_index"] == 0


# =============================================================================
# compute_proof_hash Tests
# =============================================================================


class TestComputeProofHash:
    """Tests for compute_proof_hash function."""

    def test_hash_matches_embedded(self, make_simple_proof):
        """Computed hash matches embedded proof_hash."""
        proof = make_simple_proof("run-1")
        computed = compute_proof_hash(proof)
        assert computed == proof["proof_hash"]

    def test_hash_is_deterministic(self, make_simple_proof):
        """Same proof produces same hash."""
        proof = make_simple_proof("run-1")
        hash1 = compute_proof_hash(proof)
        hash2 = compute_proof_hash(proof)
        assert hash1 == hash2

    def test_different_proofs_different_hashes(self, make_simple_proof):
        """Different proofs produce different hashes."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2")
        assert compute_proof_hash(proof1) != compute_proof_hash(proof2)


# =============================================================================
# get_chain_history Tests
# =============================================================================


class TestGetChainHistory:
    """Tests for get_chain_history function."""

    def test_single_proof_chain(self, make_simple_proof):
        """Single proof returns chain of one."""
        proof = make_simple_proof("run-1")

        def loader(hash: str):
            return None  # No more proofs

        chain = get_chain_history(proof, loader)
        assert len(chain) == 1
        assert chain[0] == proof

    def test_full_chain_reconstruction(self, make_simple_proof):
        """Full chain can be reconstructed from head."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2", previous_hash=proof1["proof_hash"])
        proof3 = make_simple_proof("run-3", previous_hash=proof2["proof_hash"])

        # Create a loader that knows about all proofs
        proof_db = {
            proof1["proof_hash"]: proof1,
            proof2["proof_hash"]: proof2,
            proof3["proof_hash"]: proof3,
        }

        def loader(hash: str):
            return proof_db.get(hash)

        chain = get_chain_history(proof3, loader)

        assert len(chain) == 3
        assert chain[0]["run_id"] == "run-1"
        assert chain[1]["run_id"] == "run-2"
        assert chain[2]["run_id"] == "run-3"

    def test_broken_chain_returns_empty(self, make_simple_proof):
        """Broken chain (missing proof) returns empty list."""
        proof1 = make_simple_proof("run-1")
        proof2 = make_simple_proof("run-2", previous_hash=proof1["proof_hash"])
        proof3 = make_simple_proof("run-3", previous_hash=proof2["proof_hash"])

        # Loader doesn't know about proof1
        proof_db = {
            proof2["proof_hash"]: proof2,
            proof3["proof_hash"]: proof3,
        }

        def loader(hash: str):
            return proof_db.get(hash)

        chain = get_chain_history(proof3, loader)
        assert chain == []  # Broken chain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
