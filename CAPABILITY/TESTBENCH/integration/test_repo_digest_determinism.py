#!/usr/bin/env python3
"""
Phase 1.5B: Repo Digest + Restore Proof + Purity Scan Tests

Fixture-backed tests for all failure modes:
- Deterministic digest (repeated -> same digest)
- New file outside durable roots -> purity FAIL + restore FAIL
- Modified file outside durable roots -> purity FAIL + restore FAIL
- Tmp residue -> purity FAIL
- All allowed durable-only writes -> purity PASS + restore PASS
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.repo_digest import (
    DigestSpec,
    PurityScan,
    RepoDigest,
    RestoreProof,
    canonical_json_bytes,
    normalize_path,
)


def test_deterministic_digest_repeated():
    """
    Test: Deterministic digest repeated -> same digest.

    Evidence:
    - Running digest twice on identical repo state produces identical digest
    - file_count, exclusions_spec_hash, module_version_hash are all identical
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Create fixture files
        (repo_root / "file1.txt").write_text("content1", encoding="utf-8")
        (repo_root / "file2.txt").write_text("content2", encoding="utf-8")
        (repo_root / "subdir").mkdir()
        (repo_root / "subdir" / "file3.txt").write_text("content3", encoding="utf-8")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[".git", "__pycache__"],
            durable_roots=["outputs"],
            tmp_roots=["_tmp"],
        )

        digest1 = RepoDigest(spec)
        receipt1 = digest1.compute_digest()

        digest2 = RepoDigest(spec)
        receipt2 = digest2.compute_digest()

        # Verify identical digests
        assert receipt1["digest"] == receipt2["digest"], "Digests must be identical for same repo state"
        assert receipt1["file_count"] == receipt2["file_count"]
        assert receipt1["exclusions_spec_hash"] == receipt2["exclusions_spec_hash"]
        assert receipt1["module_version_hash"] == receipt2["module_version_hash"]

        print(f"✓ Deterministic digest: {receipt1['digest']} (repeated)")


def test_new_file_outside_durable_roots_fails():
    """
    Test: New file outside durable roots -> purity FAIL + restore FAIL.

    Evidence:
    - Adding a file outside durable roots changes digest
    - Purity scan detects violation (verdict=FAIL)
    - Restore proof shows FAIL with diff summary (added=[path])
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Initial state
        (repo_root / "original.txt").write_text("original", encoding="utf-8")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=["outputs"],
            tmp_roots=["_tmp"],
        )

        # Pre-digest
        digest_pre = RepoDigest(spec)
        pre_receipt = digest_pre.compute_digest()

        # Mutation: Add new file outside durable roots
        (repo_root / "rogue.txt").write_text("rogue content", encoding="utf-8")

        # Post-digest
        digest_post = RepoDigest(spec)
        post_receipt = digest_post.compute_digest()

        # Verify digests differ
        assert pre_receipt["digest"] != post_receipt["digest"], "Digest must change when file added"

        # Purity scan
        scanner = PurityScan(spec)
        purity_receipt = scanner.scan(pre_receipt, post_receipt)

        assert purity_receipt["verdict"] == "FAIL", "Purity scan must FAIL when file added outside durable roots"

        # Restore proof
        prover = RestoreProof(spec)
        proof = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)

        assert proof["verdict"] == "FAIL", "Restore proof must FAIL"
        assert "diff_summary" in proof, "Diff summary must be present on FAIL"
        assert "rogue.txt" in proof["diff_summary"]["added"], "Rogue file must appear in added list"

        print(f"✓ New file outside durable roots: purity FAIL, restore FAIL, diff shows added=['rogue.txt']")


def test_modified_file_outside_durable_roots_fails():
    """
    Test: Modified file outside durable roots -> purity FAIL + restore FAIL.

    Evidence:
    - Modifying a file outside durable roots changes digest
    - Purity scan detects violation (verdict=FAIL)
    - Restore proof shows FAIL with diff summary (changed=[path])
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Initial state
        (repo_root / "mutable.txt").write_text("original content", encoding="utf-8")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=["outputs"],
            tmp_roots=["_tmp"],
        )

        # Pre-digest
        digest_pre = RepoDigest(spec)
        pre_receipt = digest_pre.compute_digest()

        # Mutation: Modify file
        (repo_root / "mutable.txt").write_text("modified content", encoding="utf-8")

        # Post-digest
        digest_post = RepoDigest(spec)
        post_receipt = digest_post.compute_digest()

        # Verify digests differ
        assert pre_receipt["digest"] != post_receipt["digest"], "Digest must change when file modified"

        # Purity scan
        scanner = PurityScan(spec)
        purity_receipt = scanner.scan(pre_receipt, post_receipt)

        assert purity_receipt["verdict"] == "FAIL", "Purity scan must FAIL when file modified"

        # Restore proof
        prover = RestoreProof(spec)
        proof = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)

        assert proof["verdict"] == "FAIL", "Restore proof must FAIL"
        assert "diff_summary" in proof, "Diff summary must be present on FAIL"
        assert "mutable.txt" in proof["diff_summary"]["changed"], "Modified file must appear in changed list"

        print(f"✓ Modified file outside durable roots: purity FAIL, restore FAIL, diff shows changed=['mutable.txt']")


def test_tmp_residue_fails_purity():
    """
    Test: Tmp residue -> purity FAIL.

    Evidence:
    - Files remaining in tmp roots after run cause purity FAIL
    - tmp_residue list contains all residue paths in canonical order
    - Restore proof shows FAIL
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Initial state
        (repo_root / "clean.txt").write_text("clean", encoding="utf-8")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=["outputs"],
            tmp_roots=["_tmp"],
        )

        # Pre-digest
        digest_pre = RepoDigest(spec)
        pre_receipt = digest_pre.compute_digest()

        # Create tmp root with residue
        tmp_root = repo_root / "_tmp"
        tmp_root.mkdir()
        (tmp_root / "residue1.txt").write_text("residue", encoding="utf-8")
        (tmp_root / "residue2.txt").write_text("residue", encoding="utf-8")

        # Post-digest (tmp roots excluded from digest)
        digest_post = RepoDigest(spec)
        post_receipt = digest_post.compute_digest()

        # Digests should be same (tmp roots excluded)
        assert pre_receipt["digest"] == post_receipt["digest"], "Digest should ignore tmp roots"

        # Purity scan
        scanner = PurityScan(spec)
        purity_receipt = scanner.scan(pre_receipt, post_receipt)

        assert purity_receipt["verdict"] == "FAIL", "Purity scan must FAIL when tmp residue present"
        assert len(purity_receipt["tmp_residue"]) == 2, "Must detect both residue files"
        assert "_tmp/residue1.txt" in purity_receipt["tmp_residue"]
        assert "_tmp/residue2.txt" in purity_receipt["tmp_residue"]

        # Verify canonical ordering
        assert purity_receipt["tmp_residue"] == sorted(purity_receipt["tmp_residue"]), "tmp_residue must be sorted"

        # Restore proof
        prover = RestoreProof(spec)
        proof = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)

        assert proof["verdict"] == "FAIL", "Restore proof must FAIL when purity fails"

        print(f"✓ Tmp residue: purity FAIL, tmp_residue=['_tmp/residue1.txt', '_tmp/residue2.txt']")


def test_durable_only_writes_pass():
    """
    Test: All allowed durable-only writes -> purity PASS + restore PASS.

    Evidence:
    - Writing only to durable roots does not change digest
    - Purity scan passes (verdict=PASS)
    - Restore proof passes (verdict=PASS)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Initial state
        (repo_root / "source.txt").write_text("source", encoding="utf-8")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=["outputs"],
            tmp_roots=["_tmp"],
        )

        # Pre-digest
        digest_pre = RepoDigest(spec)
        pre_receipt = digest_pre.compute_digest()

        # Allowed mutation: Write to durable root
        outputs_dir = repo_root / "outputs"
        outputs_dir.mkdir()
        (outputs_dir / "result1.txt").write_text("result1", encoding="utf-8")
        (outputs_dir / "result2.txt").write_text("result2", encoding="utf-8")

        # Post-digest
        digest_post = RepoDigest(spec)
        post_receipt = digest_post.compute_digest()

        # Digests should be same (durable roots excluded)
        assert pre_receipt["digest"] == post_receipt["digest"], "Digest should ignore durable roots"

        # Purity scan
        scanner = PurityScan(spec)
        purity_receipt = scanner.scan(pre_receipt, post_receipt)

        assert purity_receipt["verdict"] == "PASS", "Purity scan must PASS when only durable roots modified"
        assert len(purity_receipt["tmp_residue"]) == 0, "No tmp residue expected"

        # Restore proof
        prover = RestoreProof(spec)
        proof = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)

        assert proof["verdict"] == "PASS", "Restore proof must PASS"
        assert "diff_summary" not in proof, "No diff summary on PASS"

        print(f"✓ Durable-only writes: purity PASS, restore PASS")


def test_canonical_ordering_paths():
    """
    Test: Canonical ordering of paths in diff summaries.

    Evidence:
    - added, removed, changed lists are sorted
    - Ordering is deterministic across runs
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Initial state
        (repo_root / "a.txt").write_text("a", encoding="utf-8")
        (repo_root / "b.txt").write_text("b", encoding="utf-8")
        (repo_root / "c.txt").write_text("c", encoding="utf-8")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=[],
            tmp_roots=[],
        )

        # Pre-digest
        digest_pre = RepoDigest(spec)
        pre_receipt = digest_pre.compute_digest()

        # Mutations in non-alphabetical order
        (repo_root / "z.txt").write_text("z", encoding="utf-8")  # Added
        (repo_root / "m.txt").write_text("m", encoding="utf-8")  # Added
        (repo_root / "b.txt").write_text("b_modified", encoding="utf-8")  # Changed
        (repo_root / "a.txt").unlink()  # Removed

        # Post-digest
        digest_post = RepoDigest(spec)
        post_receipt = digest_post.compute_digest()

        # Purity scan
        scanner = PurityScan(spec)
        purity_receipt = scanner.scan(pre_receipt, post_receipt)

        # Restore proof
        prover = RestoreProof(spec)
        proof = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)

        # Verify canonical ordering
        assert proof["diff_summary"]["added"] == sorted(proof["diff_summary"]["added"]), "added must be sorted"
        assert proof["diff_summary"]["removed"] == sorted(proof["diff_summary"]["removed"]), "removed must be sorted"
        assert proof["diff_summary"]["changed"] == sorted(proof["diff_summary"]["changed"]), "changed must be sorted"

        # Verify expected content
        assert proof["diff_summary"]["added"] == ["m.txt", "z.txt"]
        assert proof["diff_summary"]["removed"] == ["a.txt"]
        assert proof["diff_summary"]["changed"] == ["b.txt"]

        print(f"✓ Canonical ordering: added={proof['diff_summary']['added']}, removed={proof['diff_summary']['removed']}, changed={proof['diff_summary']['changed']}")


def test_exclusions_are_respected():
    """
    Test: Exclusions are respected in digest computation.

    Evidence:
    - Files under exclusion paths are not included in digest
    - Modifying excluded files does not change digest
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Initial state
        (repo_root / "tracked.txt").write_text("tracked", encoding="utf-8")
        git_dir = repo_root / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config", encoding="utf-8")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[".git"],
            durable_roots=[],
            tmp_roots=[],
        )

        # Pre-digest
        digest_pre = RepoDigest(spec)
        pre_receipt = digest_pre.compute_digest()

        # Modify excluded file
        (git_dir / "config").write_text("modified git config", encoding="utf-8")
        (git_dir / "new_file").write_text("new git file", encoding="utf-8")

        # Post-digest
        digest_post = RepoDigest(spec)
        post_receipt = digest_post.compute_digest()

        # Digests should be identical (exclusions respected)
        assert pre_receipt["digest"] == post_receipt["digest"], "Digest must ignore excluded paths"
        assert pre_receipt["file_count"] == post_receipt["file_count"]

        print(f"✓ Exclusions respected: digest unchanged despite .git modifications")


def test_normalize_path():
    """
    Test: Path normalization produces consistent forward-slash format.

    Evidence:
    - Windows backslashes converted to forward slashes
    - Relative paths normalized consistently
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Test absolute path
        abs_path = repo_root / "subdir" / "file.txt"
        norm = normalize_path(abs_path, repo_root)
        assert norm == "subdir/file.txt", f"Expected 'subdir/file.txt', got '{norm}'"

        # Test relative path
        rel_path = Path("subdir") / "file.txt"
        norm = normalize_path(rel_path, repo_root)
        assert norm == "subdir/file.txt", f"Expected 'subdir/file.txt', got '{norm}'"

        print(f"✓ Path normalization: consistent forward-slash format")


def test_canonical_json_determinism():
    """
    Test: Canonical JSON serialization is deterministic.

    Evidence:
    - Same dict produces same bytes regardless of key insertion order
    - Sorted keys, no whitespace
    """
    dict1 = {"z": 1, "a": 2, "m": 3}
    dict2 = {"a": 2, "m": 3, "z": 1}  # Different insertion order

    bytes1 = canonical_json_bytes(dict1)
    bytes2 = canonical_json_bytes(dict2)

    assert bytes1 == bytes2, "Canonical JSON must be deterministic"
    assert bytes1 == b'{"a":2,"m":3,"z":1}', f"Expected sorted keys, got {bytes1}"

    print(f"✓ Canonical JSON: deterministic serialization")


def test_empty_repo_digest():
    """
    Test: Empty repo produces deterministic digest.

    Evidence:
    - Empty repo (no files) produces valid digest
    - file_count = 0
    - Digest is deterministic
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=[],
            tmp_roots=[],
        )

        digest1 = RepoDigest(spec)
        receipt1 = digest1.compute_digest()

        digest2 = RepoDigest(spec)
        receipt2 = digest2.compute_digest()

        assert receipt1["file_count"] == 0, "Empty repo should have file_count=0"
        assert receipt1["digest"] == receipt2["digest"], "Empty repo digest must be deterministic"

        print(f"✓ Empty repo: digest={receipt1['digest']}, file_count=0")


def test_module_version_hash_in_receipts():
    """
    Test: Module version hash appears in all receipts.

    Evidence:
    - PRE_DIGEST, POST_DIGEST, PURITY_SCAN, RESTORE_PROOF all include module_version_hash
    - Hash is consistent across all receipts
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()
        (repo_root / "file.txt").write_text("content", encoding="utf-8")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=[],
            tmp_roots=[],
        )

        # Generate all receipts
        digest = RepoDigest(spec)
        pre_receipt = digest.compute_digest()
        post_receipt = digest.compute_digest()

        scanner = PurityScan(spec)
        purity_receipt = scanner.scan(pre_receipt, post_receipt)

        prover = RestoreProof(spec)
        proof = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)

        # Verify module_version_hash present
        assert "module_version_hash" in pre_receipt, "PRE_DIGEST must include module_version_hash"
        assert "module_version_hash" in post_receipt, "POST_DIGEST must include module_version_hash"
        assert "scan_module_version_hash" in purity_receipt, "PURITY_SCAN must include scan_module_version_hash"
        assert "proof_module_version_hash" in proof, "RESTORE_PROOF must include proof_module_version_hash"

        # Verify consistency
        assert pre_receipt["module_version_hash"] == post_receipt["module_version_hash"]
        assert pre_receipt["module_version_hash"] == purity_receipt["scan_module_version_hash"]
        assert pre_receipt["module_version_hash"] == proof["proof_module_version_hash"]

        print(f"✓ Module version hash: consistent across all receipts ({pre_receipt['module_version_hash'][:8]}...)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
