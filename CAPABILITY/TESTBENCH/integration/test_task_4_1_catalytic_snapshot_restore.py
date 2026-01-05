#!/usr/bin/env python3
"""
Task 4.1: Catalytic Snapshot & Restore (Z.4.2–Z.4.4)

This test explicitly proves all three requirements:
- 4.1.1: Pre-run snapshot: hash catalytic state before execution (Z.4.2)
- 4.1.2: Post-run restoration: verify byte-identical restoration (Z.4.3)
- 4.1.3: Hard-fail on restoration mismatch (Z.4.4)

Exit Criteria:
- Catalytic domains restore byte-identical (fixture-backed)
- Failure mode is deterministic and fail-closed
"""

import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Adjust path to find catalytic_runtime
CATALYTIC_PATH = REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic"
if str(CATALYTIC_PATH) not in sys.path:
    sys.path.insert(0, str(CATALYTIC_PATH))

from catalytic_runtime import CatalyticRuntime


def _rm(path: Path) -> None:
    """Cleanup helper."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def test_4_1_1_pre_run_snapshot_hashes_catalytic_state():
    """
    Test 4.1.1: Pre-run snapshot captures hash of catalytic state.

    Evidence:
    - PRE_MANIFEST.json contains SHA-256 hashes of all files in catalytic domains
    - Snapshot is taken before command execution
    - Snapshot is deterministic (repeated runs produce same hash for same content)
    """
    test_subdir = "LAW/CONTRACTS/_runs/_tmp/test_4_1_1"
    test_root = REPO_ROOT / test_subdir
    _rm(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    run_id = f"task-4-1-1-{uuid.uuid4().hex[:8]}"
    _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id)

    domain_rel = f"{test_subdir}/domain"
    output_rel = f"{test_subdir}/output.txt"

    domain_abs = REPO_ROOT / domain_rel
    domain_abs.mkdir(parents=True, exist_ok=True)

    # Create some files in the catalytic domain before run
    (domain_abs / "file1.txt").write_text("content1", encoding="utf-8")
    (domain_abs / "file2.txt").write_text("content2", encoding="utf-8")

    try:
        runtime = CatalyticRuntime(
            run_id=run_id,
            catalytic_domains=[domain_rel],
            durable_outputs=[output_rel],
            intent="Test 4.1.1: Pre-run snapshot",
            memoize=False,
        )

        # Execute a simple command that creates output but doesn't touch catalytic domain
        cmd = [sys.executable, "-c", f"from pathlib import Path; Path(r'{REPO_ROOT / output_rel}').write_text('output', encoding='utf-8')"]

        exit_code = runtime.run(cmd)
        assert exit_code == 0, "Run should succeed"

        # Verify PRE_MANIFEST.json exists and contains hashes
        run_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        pre_manifest = run_dir / "PRE_MANIFEST.json"
        assert pre_manifest.exists(), "PRE_MANIFEST.json must exist"

        import json
        pre_data = json.loads(pre_manifest.read_text())

        # Verify domain is captured
        assert domain_rel in pre_data, f"Domain {domain_rel} should be in PRE_MANIFEST"
        domain_snapshot = pre_data[domain_rel]

        # Verify files are hashed
        assert "file1.txt" in domain_snapshot, "file1.txt should be in snapshot"
        assert "file2.txt" in domain_snapshot, "file2.txt should be in snapshot"

        # Verify hashes are SHA-256 (64 hex chars)
        for filename, hash_val in domain_snapshot.items():
            assert len(hash_val) == 64, f"Hash for {filename} should be 64 hex chars (SHA-256)"
            assert all(c in "0123456789abcdef" for c in hash_val), f"Hash for {filename} should be hex"

        print(f"✓ Test 4.1.1 PASSED: Pre-run snapshot captured {len(domain_snapshot)} files with SHA-256 hashes")

    finally:
        _rm(test_root)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id)


def test_4_1_2_post_run_restoration_byte_identical():
    """
    Test 4.1.2: Post-run restoration verifies byte-identical state.

    Evidence:
    - POST_MANIFEST.json matches PRE_MANIFEST.json when catalytic domain is properly restored
    - PROOF.json contains restoration_result.verified = true
    - Restoration is byte-identical (every file hash matches)
    """
    test_subdir = "LAW/CONTRACTS/_runs/_tmp/test_4_1_2"
    test_root = REPO_ROOT / test_subdir
    _rm(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    run_id = f"task-4-1-2-{uuid.uuid4().hex[:8]}"
    _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id)

    domain_rel = f"{test_subdir}/domain"
    output_rel = f"{test_subdir}/output.txt"

    domain_abs = REPO_ROOT / domain_rel
    domain_abs.mkdir(parents=True, exist_ok=True)

    # Create catalytic domain state
    (domain_abs / "temp1.txt").write_text("temp content 1", encoding="utf-8")

    try:
        runtime = CatalyticRuntime(
            run_id=run_id,
            catalytic_domains=[domain_rel],
            durable_outputs=[output_rel],
            intent="Test 4.1.2: Byte-identical restoration",
            memoize=False,
        )

        # Command that:
        # 1. Mutates catalytic domain (adds a file)
        # 2. Restores it back to original state
        # 3. Creates durable output
        cmd = [
            sys.executable,
            "-c",
            f"""
from pathlib import Path
import hashlib

domain = Path(r'{domain_abs}')
output = Path(r'{REPO_ROOT / output_rel}')

# Mutate: add a temporary file to catalytic domain
(domain / 'temp2.txt').write_text('temporary mutation', encoding='utf-8')

# Restore: remove the temporary file (return to original state)
(domain / 'temp2.txt').unlink()

# Create durable output
output.write_text('completed', encoding='utf-8')
""",
        ]

        exit_code = runtime.run(cmd)
        assert exit_code == 0, "Run should succeed with proper restoration"

        run_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id

        # Verify restoration proof
        import json
        proof_path = run_dir / "PROOF.json"
        assert proof_path.exists(), "PROOF.json must exist"

        proof = json.loads(proof_path.read_text())
        assert "restoration_result" in proof, "PROOF must contain restoration_result"
        assert proof["restoration_result"]["verified"] is True, "Restoration must be verified"
        assert proof["restoration_result"]["condition"] == "IDENTICAL", "Restoration must be IDENTICAL"

        # Verify PRE and POST manifests match
        pre_manifest = json.loads((run_dir / "PRE_MANIFEST.json").read_text())
        post_manifest = json.loads((run_dir / "POST_MANIFEST.json").read_text())

        assert pre_manifest == post_manifest, "PRE and POST manifests must match for byte-identical restoration"

        print(f"✓ Test 4.1.2 PASSED: Post-run restoration verified byte-identical")

    finally:
        _rm(test_root)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id)


def test_4_1_3_hard_fail_on_restoration_mismatch():
    """
    Test 4.1.3: Hard-fail on restoration mismatch.

    Evidence:
    - Runtime returns exit code 1 when catalytic domain is not restored
    - PROOF.json contains restoration_result.verified = false
    - Failure is deterministic and fail-closed
    - STATUS.json shows failed state
    """
    test_subdir = "LAW/CONTRACTS/_runs/_tmp/test_4_1_3"
    test_root = REPO_ROOT / test_subdir
    _rm(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    run_id = f"task-4-1-3-{uuid.uuid4().hex[:8]}"
    _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id)

    domain_rel = f"{test_subdir}/domain"
    output_rel = f"{test_subdir}/output.txt"

    domain_abs = REPO_ROOT / domain_rel
    domain_abs.mkdir(parents=True, exist_ok=True)

    # Create initial catalytic domain state
    (domain_abs / "original.txt").write_text("original content", encoding="utf-8")

    try:
        runtime = CatalyticRuntime(
            run_id=run_id,
            catalytic_domains=[domain_rel],
            durable_outputs=[output_rel],
            intent="Test 4.1.3: Hard-fail on mismatch",
            memoize=False,
        )

        # Command that VIOLATES catalytic contract: mutates domain without restoring
        cmd = [
            sys.executable,
            "-c",
            f"""
from pathlib import Path

domain = Path(r'{domain_abs}')
output = Path(r'{REPO_ROOT / output_rel}')

# VIOLATION: Add file to catalytic domain and DON'T remove it
(domain / 'rogue_file.txt').write_text('This should not persist', encoding='utf-8')

# Create output (simulating "successful" work)
output.write_text('output created', encoding='utf-8')
""",
        ]

        exit_code = runtime.run(cmd)

        # Verify HARD FAIL: runtime must return non-zero exit code
        assert exit_code == 1, "Runtime MUST fail-closed (exit 1) when restoration fails"

        run_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id

        # Verify STATUS.json shows failure
        import json
        status_path = run_dir / "STATUS.json"
        assert status_path.exists(), "STATUS.json must exist even on failure"

        status = json.loads(status_path.read_text())
        assert status["status"] == "failed", "Status must be 'failed'"
        assert status["restoration_verified"] is False, "Restoration must be marked as NOT verified"

        # Verify PROOF shows mismatch
        proof_path = run_dir / "PROOF.json"
        assert proof_path.exists(), "PROOF.json must exist even on failure"

        proof = json.loads(proof_path.read_text())
        assert proof["restoration_result"]["verified"] is False, "Proof must show restoration failed"
        assert proof["restoration_result"]["condition"] == "MISMATCH", "Condition must be MISMATCH"

        # Verify RESTORE_DIFF.json captures the violation
        diff_path = run_dir / "RESTORE_DIFF.json"
        assert diff_path.exists(), "RESTORE_DIFF.json must exist"

        diff = json.loads(diff_path.read_text())
        assert domain_rel in diff, "Diff must contain catalytic domain"
        domain_diff = diff[domain_rel]

        # Verify the rogue file is detected
        assert "added" in domain_diff, "Diff must show added files"
        assert "rogue_file.txt" in domain_diff["added"], "Rogue file must be detected in diff"

        print(f"✓ Test 4.1.3 PASSED: Hard-fail on restoration mismatch (exit=1, verified=false, condition=MISMATCH)")

    finally:
        _rm(test_root)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id)


def test_4_1_fixture_backed_determinism():
    """
    Exit Criteria: Catalytic domains restore byte-identical (fixture-backed).

    This test proves determinism: running the same command twice with same inputs
    produces identical restoration proofs.
    """
    test_subdir = "LAW/CONTRACTS/_runs/_tmp/test_4_1_fixture"
    test_root = REPO_ROOT / test_subdir
    _rm(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    domain_rel = f"{test_subdir}/domain"
    output_rel = f"{test_subdir}/output.txt"

    domain_abs = REPO_ROOT / domain_rel
    domain_abs.mkdir(parents=True, exist_ok=True)

    # Fixed fixture content
    (domain_abs / "fixture1.txt").write_text("fixture content 1", encoding="utf-8")
    (domain_abs / "fixture2.txt").write_text("fixture content 2", encoding="utf-8")

    import json
    import hashlib

    def run_and_capture_proof(run_id: str) -> dict:
        """Run command and return proof dict."""
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id)

        runtime = CatalyticRuntime(
            run_id=run_id,
            catalytic_domains=[domain_rel],
            durable_outputs=[output_rel],
            intent="Fixture-backed determinism test",
            memoize=False,
        )

        cmd = [sys.executable, "-c", f"from pathlib import Path; Path(r'{REPO_ROOT / output_rel}').write_text('output', encoding='utf-8')"]

        exit_code = runtime.run(cmd)
        assert exit_code == 0

        run_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        proof = json.loads((run_dir / "PROOF.json").read_text())

        return proof

    try:
        # Run 1
        proof1 = run_and_capture_proof(f"fixture-run-1-{uuid.uuid4().hex[:8]}")

        # Run 2 (same fixture, same command)
        proof2 = run_and_capture_proof(f"fixture-run-2-{uuid.uuid4().hex[:8]}")

        # Verify both runs produced verified restoration
        assert proof1["restoration_result"]["verified"] is True
        assert proof2["restoration_result"]["verified"] is True

        # Verify pre/post hashes are deterministic (same content → same hash)
        # Note: The exact proof structure may vary (timestamps, run_id), but restoration hashes should match
        assert proof1["pre_state_hash"] == proof2["pre_state_hash"], "Pre-state hash must be deterministic"
        assert proof1["post_state_hash"] == proof2["post_state_hash"], "Post-state hash must be deterministic"

        print("✓ Fixture-backed determinism PASSED: Identical fixtures produce identical restoration proofs")

    finally:
        _rm(test_root)
        # Cleanup both run dirs
        for run_dir in (REPO_ROOT / "LAW" / "CONTRACTS" / "_runs").glob("fixture-run-*"):
            _rm(run_dir)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
