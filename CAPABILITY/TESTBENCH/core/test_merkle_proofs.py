# Phase 1.7.3: Merkle Membership Proofs
"""
Test Merkle membership proofs for partial verification.

These tests verify:
1. Valid proof verification works
2. Tampered proof is rejected
3. Missing sibling is rejected
4. Proof generation is deterministic
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

# Import from merkle module
from CAPABILITY.PRIMITIVES.merkle import (
    MerkleProof,
    build_manifest_root,
    build_manifest_with_proofs,
    verify_membership,
)


def _sha256_hex(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_manifest() -> dict[str, str]:
    """A small manifest with 4 files for testing."""
    return {
        "a/file1.txt": _sha256_hex(b"content of file 1"),
        "a/file2.txt": _sha256_hex(b"content of file 2"),
        "b/file3.txt": _sha256_hex(b"content of file 3"),
        "c/file4.txt": _sha256_hex(b"content of file 4"),
    }


@pytest.fixture
def odd_manifest() -> dict[str, str]:
    """A manifest with 5 files (odd count, tests padding)."""
    return {
        "file1.txt": _sha256_hex(b"one"),
        "file2.txt": _sha256_hex(b"two"),
        "file3.txt": _sha256_hex(b"three"),
        "file4.txt": _sha256_hex(b"four"),
        "file5.txt": _sha256_hex(b"five"),
    }


@pytest.fixture
def large_manifest() -> dict[str, str]:
    """A larger manifest with 100 files for stress testing."""
    return {
        f"dir{i % 10}/file{i:04d}.txt": _sha256_hex(f"content {i}".encode())
        for i in range(100)
    }


# =============================================================================
# Test: Valid Proof Verification
# =============================================================================


def test_valid_proof_verification(small_manifest: dict[str, str]) -> None:
    """
    Valid proofs must verify correctly.

    For each file in the manifest, we should be able to:
    1. Build proofs
    2. Verify membership using only the proof (not the full manifest)
    3. The root computed via proof must match the original root
    """
    root, proofs = build_manifest_with_proofs(small_manifest)

    # Verify root matches the standard build
    expected_root = build_manifest_root(small_manifest)
    assert root == expected_root, "Root from proof builder must match standard root"

    # Verify each file's proof
    for path, bytes_hash in small_manifest.items():
        proof = proofs[path]

        # Verify using MerkleProof object
        assert verify_membership(path, bytes_hash, proof, root), \
            f"Valid proof for {path} must verify"

        # Verify using dict representation (for JSON serialization)
        proof_dict = proof.to_dict()
        assert verify_membership(path, bytes_hash, proof_dict, root), \
            f"Valid proof dict for {path} must verify"


def test_valid_proof_odd_file_count(odd_manifest: dict[str, str]) -> None:
    """
    Odd file counts require padding - proofs must still work.
    """
    root, proofs = build_manifest_with_proofs(odd_manifest)
    expected_root = build_manifest_root(odd_manifest)

    assert root == expected_root, "Root with odd file count must match"

    for path, bytes_hash in odd_manifest.items():
        proof = proofs[path]
        assert verify_membership(path, bytes_hash, proof, root), \
            f"Proof for {path} in odd manifest must verify"


def test_valid_proof_large_manifest(large_manifest: dict[str, str]) -> None:
    """
    100 files - all proofs must verify.
    """
    root, proofs = build_manifest_with_proofs(large_manifest)
    expected_root = build_manifest_root(large_manifest)

    assert root == expected_root
    assert len(proofs) == 100

    # Verify a sample of proofs (all of them for completeness)
    verified = 0
    for path, bytes_hash in large_manifest.items():
        proof = proofs[path]
        if verify_membership(path, bytes_hash, proof, root):
            verified += 1

    assert verified == 100, f"All 100 proofs must verify, got {verified}"


# =============================================================================
# Test: Tampered Proof Rejection
# =============================================================================


def test_tampered_proof_hash_rejected(small_manifest: dict[str, str]) -> None:
    """
    If the file hash is tampered, verification must fail.
    """
    root, proofs = build_manifest_with_proofs(small_manifest)

    path = "a/file1.txt"
    original_hash = small_manifest[path]
    proof = proofs[path]

    # Tamper with the bytes_hash - flip one character
    tampered_hash = original_hash[:-1] + ("0" if original_hash[-1] != "0" else "1")

    # Must reject with tampered hash
    assert not verify_membership(path, tampered_hash, proof, root), \
        "Tampered file hash must be rejected"

    # Original must still work
    assert verify_membership(path, original_hash, proof, root), \
        "Original hash must still verify"


def test_tampered_proof_sibling_rejected(small_manifest: dict[str, str]) -> None:
    """
    If a sibling hash in the proof is tampered, verification must fail.
    """
    root, proofs = build_manifest_with_proofs(small_manifest)

    path = "a/file1.txt"
    bytes_hash = small_manifest[path]
    proof = proofs[path]

    # Make a copy and tamper with a sibling
    proof_dict = proof.to_dict()
    if proof_dict["steps"]:
        original_sibling = proof_dict["steps"][0]["sibling"]
        tampered_sibling = original_sibling[:-1] + ("0" if original_sibling[-1] != "0" else "1")
        proof_dict["steps"][0]["sibling"] = tampered_sibling

        assert not verify_membership(path, bytes_hash, proof_dict, root), \
            "Tampered sibling hash must be rejected"


def test_tampered_proof_wrong_root_rejected(small_manifest: dict[str, str]) -> None:
    """
    If the root is wrong, verification must fail.
    """
    root, proofs = build_manifest_with_proofs(small_manifest)

    path = "a/file1.txt"
    bytes_hash = small_manifest[path]
    proof = proofs[path]

    # Wrong root
    wrong_root = root[:-1] + ("0" if root[-1] != "0" else "1")

    assert not verify_membership(path, bytes_hash, proof, wrong_root), \
        "Wrong root must reject proof"


def test_tampered_proof_wrong_path_rejected(small_manifest: dict[str, str]) -> None:
    """
    If the path in the claim doesn't match the proof's path, reject.
    """
    root, proofs = build_manifest_with_proofs(small_manifest)

    path = "a/file1.txt"
    bytes_hash = small_manifest[path]
    proof = proofs[path]

    # Claim a different path
    wrong_path = "a/file2.txt"

    assert not verify_membership(wrong_path, bytes_hash, proof, root), \
        "Wrong path must reject proof"


# =============================================================================
# Test: Missing Sibling Rejection
# =============================================================================


def test_missing_sibling_rejected(small_manifest: dict[str, str]) -> None:
    """
    If a sibling is missing from the proof chain, verification must fail.
    """
    root, proofs = build_manifest_with_proofs(small_manifest)

    path = "a/file1.txt"
    bytes_hash = small_manifest[path]
    proof = proofs[path]

    # Make a copy with one step removed
    proof_dict = proof.to_dict()
    if len(proof_dict["steps"]) > 1:
        # Remove one step
        proof_dict["steps"] = proof_dict["steps"][:-1]

        assert not verify_membership(path, bytes_hash, proof_dict, root), \
            "Missing sibling must reject proof"


def test_extra_sibling_rejected(small_manifest: dict[str, str]) -> None:
    """
    If there's an extra sibling in the proof chain, verification must fail.
    """
    root, proofs = build_manifest_with_proofs(small_manifest)

    path = "a/file1.txt"
    bytes_hash = small_manifest[path]
    proof = proofs[path]

    # Make a copy with one extra step
    proof_dict = proof.to_dict()
    fake_sibling = _sha256_hex(b"fake sibling")
    proof_dict["steps"].append({"sibling": fake_sibling, "sibling_is_left": False})

    assert not verify_membership(path, bytes_hash, proof_dict, root), \
        "Extra sibling must reject proof"


# =============================================================================
# Test: Deterministic Proof Generation
# =============================================================================


def test_deterministic_proof_generation(small_manifest: dict[str, str]) -> None:
    """
    Same manifest must produce identical proofs every time.
    """
    root1, proofs1 = build_manifest_with_proofs(small_manifest)
    root2, proofs2 = build_manifest_with_proofs(small_manifest)
    root3, proofs3 = build_manifest_with_proofs(small_manifest)

    assert root1 == root2 == root3, "Roots must be identical"

    for path in small_manifest:
        proof1_dict = proofs1[path].to_dict()
        proof2_dict = proofs2[path].to_dict()
        proof3_dict = proofs3[path].to_dict()

        # Compare as canonical JSON for determinism
        json1 = json.dumps(proof1_dict, sort_keys=True)
        json2 = json.dumps(proof2_dict, sort_keys=True)
        json3 = json.dumps(proof3_dict, sort_keys=True)

        assert json1 == json2 == json3, \
            f"Proofs for {path} must be deterministic"


def test_deterministic_with_different_insertion_order() -> None:
    """
    Manifest with same content but different insertion order must produce same proofs.
    """
    # Create two manifests with same content but different insertion order
    manifest1 = {
        "a.txt": _sha256_hex(b"a"),
        "b.txt": _sha256_hex(b"b"),
        "c.txt": _sha256_hex(b"c"),
    }

    manifest2 = {
        "c.txt": _sha256_hex(b"c"),
        "a.txt": _sha256_hex(b"a"),
        "b.txt": _sha256_hex(b"b"),
    }

    root1, proofs1 = build_manifest_with_proofs(manifest1)
    root2, proofs2 = build_manifest_with_proofs(manifest2)

    assert root1 == root2, "Insertion order must not affect root"

    for path in manifest1:
        json1 = json.dumps(proofs1[path].to_dict(), sort_keys=True)
        json2 = json.dumps(proofs2[path].to_dict(), sort_keys=True)
        assert json1 == json2, f"Proofs for {path} must be identical regardless of insertion order"


# =============================================================================
# Test: Serialization Round-Trip
# =============================================================================


def test_proof_serialization_roundtrip(small_manifest: dict[str, str]) -> None:
    """
    Proofs must survive JSON serialization and deserialization.
    """
    root, proofs = build_manifest_with_proofs(small_manifest)

    for path, bytes_hash in small_manifest.items():
        proof = proofs[path]

        # Serialize to dict, then to JSON, then back
        proof_dict = proof.to_dict()
        json_str = json.dumps(proof_dict)
        parsed_dict = json.loads(json_str)
        restored_proof = MerkleProof.from_dict(parsed_dict)

        # Verify the restored proof works
        assert verify_membership(path, bytes_hash, restored_proof, root), \
            f"Restored proof for {path} must verify"


# =============================================================================
# Test: Edge Cases
# =============================================================================


def test_single_file_manifest() -> None:
    """
    Single file manifest - proof should have no siblings.
    """
    manifest = {"only.txt": _sha256_hex(b"only file")}

    root, proofs = build_manifest_with_proofs(manifest)
    expected_root = build_manifest_root(manifest)

    assert root == expected_root

    proof = proofs["only.txt"]
    # Single file: the leaf hash IS the root (after potential self-pairing for padding)
    # With our algorithm, a single leaf gets paired with itself, so we have 1 step
    assert verify_membership("only.txt", manifest["only.txt"], proof, root)


def test_two_file_manifest() -> None:
    """
    Two file manifest - each proof should have exactly 1 sibling.
    """
    manifest = {
        "a.txt": _sha256_hex(b"a"),
        "b.txt": _sha256_hex(b"b"),
    }

    root, proofs = build_manifest_with_proofs(manifest)
    expected_root = build_manifest_root(manifest)

    assert root == expected_root

    for path, bytes_hash in manifest.items():
        proof = proofs[path]
        assert len(proof.steps) == 1, f"Two-file manifest should have 1 step per proof"
        assert verify_membership(path, bytes_hash, proof, root)


def test_empty_manifest_rejected() -> None:
    """
    Empty manifest is forbidden.
    """
    with pytest.raises(ValueError, match="empty manifest is forbidden"):
        build_manifest_with_proofs({})


# =============================================================================
# Test: Cross-Verification with build_manifest_root
# =============================================================================


def test_proof_root_matches_standard_root(large_manifest: dict[str, str]) -> None:
    """
    The root from build_manifest_with_proofs must exactly match build_manifest_root.
    """
    proof_root, _ = build_manifest_with_proofs(large_manifest)
    standard_root = build_manifest_root(large_manifest)

    assert proof_root == standard_root, \
        "Proof builder root must match standard root builder"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
