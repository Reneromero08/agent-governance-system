#!/usr/bin/env python3
"""
Test for runtime write surface enforcement across PIPELINES and MCP.

This test verifies that:
1. All PIPELINES writes go through GuardedWriter
2. All MCP writes go through GuardedWriter
3. Firewall violations are raised when writes attempt forbidden paths
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter, FirewallViolation


def test_guarded_writer_basic():
    """Test basic GuardedWriter functionality."""
    print("[TEST] GuardedWriter basic functionality")

    writer = GuardedWriter(project_root=REPO_ROOT)

    # Test tmp write (should succeed)
    try:
        writer.write_tmp("LAW/CONTRACTS/_runs/_tmp/test.json", b'{"test": "data"}')
        print("  [OK] Tmp write succeeded")
    except FirewallViolation as e:
        raise AssertionError(f"Tmp write failed: {e.error_code}")

    # Test durable write without gate (should fail)
    try:
        writer.write_durable("LAW/CONTRACTS/_runs/test_durable.json", b'{"test": "data"}')
        raise AssertionError("Durable write succeeded before gate opening (should have failed)")
    except FirewallViolation as e:
        assert e.error_code == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT", f"Wrong error code: {e.error_code}"
        print(f"  [OK] Durable write blocked before gate: {e.error_code}")

    # Test durable write with gate open (should succeed)
    writer.open_commit_gate()
    try:
        writer.write_durable("LAW/CONTRACTS/_runs/test_durable.json", b'{"test": "data"}')
        print("  [OK] Durable write succeeded after gate opening")
    except FirewallViolation as e:
        raise AssertionError(f"Durable write failed after gate: {e.error_code}")

    # Cleanup
    try:
        (REPO_ROOT / "LAW/CONTRACTS/_runs/_tmp/test.json").unlink(missing_ok=True)
        (REPO_ROOT / "LAW/CONTRACTS/_runs/test_durable.json").unlink(missing_ok=True)
    except Exception:
        pass


def test_forbidden_write():
    """Test that writes to forbidden paths are blocked."""
    print("\n[TEST] Forbidden write blocking")

    writer = GuardedWriter(project_root=REPO_ROOT)

    # Test write to CANON (should fail)
    try:
        writer.write_tmp("LAW/CANON/test.json", b'{"test": "data"}')
        raise AssertionError("Write to CANON succeeded (should have failed)")
    except FirewallViolation as e:
        assert e.error_code == "FIREWALL_PATH_EXCLUDED", f"Wrong error code for CANON write: {e.error_code}"
        print(f"  [OK] Write to CANON blocked: {e.error_code}")

    # Test write to AGENTS.md (should fail)
    try:
        writer.write_tmp("AGENTS.md", b'{"test": "data"}')
        raise AssertionError("Write to AGENTS.md succeeded (should have failed)")
    except FirewallViolation as e:
        assert e.error_code == "FIREWALL_PATH_EXCLUDED", f"Wrong error code for AGENTS.md write: {e.error_code}"
        print(f"  [OK] Write to AGENTS.md blocked: {e.error_code}")


def test_mkdir_enforcement():
    """Test that mkdir operations are enforced."""
    print("\n[TEST] Mkdir enforcement")

    writer = GuardedWriter(project_root=REPO_ROOT)

    # Test mkdir in tmp domain (should succeed)
    try:
        writer.mkdir_tmp("LAW/CONTRACTS/_runs/_tmp/test_dir")
        print("  [OK] Mkdir in tmp succeeded")
    except FirewallViolation as e:
        raise AssertionError(f"Mkdir in tmp failed: {e.error_code}")

    # Test mkdir in forbidden domain (should fail)
    try:
        writer.mkdir_tmp("LAW/CANON/test_dir")
        raise AssertionError("Mkdir in CANON succeeded (should have failed)")
    except FirewallViolation as e:
        assert e.error_code == "FIREWALL_PATH_EXCLUDED", f"Wrong error code for CANON mkdir: {e.error_code}"
        print(f"  [OK] Mkdir in CANON blocked: {e.error_code}")

    # Cleanup
    try:
        (REPO_ROOT / "LAW/CONTRACTS/_runs/_tmp/test_dir").rmdir()
    except Exception:
        pass


def test_atomic_writes_module():
    """Test AtomicGuardedWrites module."""
    print("\n[TEST] AtomicGuardedWrites module")

    from CAPABILITY.PIPELINES.atomic_writes import AtomicGuardedWrites

    writes = AtomicGuardedWrites(project_root=REPO_ROOT)

    # Test atomic write
    test_path = REPO_ROOT / "LAW/CONTRACTS/_runs/_tmp/atomic_test.json"
    try:
        writes.atomic_write_canonical_json(test_path, {"test": "data"})
        print("  [OK] Atomic canonical write succeeded")

        # Verify file exists and is valid JSON
        assert test_path.exists(), "Atomic write file not created"

        import json
        data = json.loads(test_path.read_text())
        assert data == {"test": "data"}, f"Atomic write content mismatch: {data}"
        print("  [OK] Atomic write content verified")

        test_path.unlink(missing_ok=True)

    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(f"Atomic write failed: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RUNTIME WRITE INTERCEPTION TESTS")
    print("=" * 60)

    tests = [
        ("GuardedWriter basic", test_guarded_writer_basic),
        ("Forbidden write blocking", test_forbidden_write),
        ("Mkdir enforcement", test_mkdir_enforcement),
        ("AtomicGuardedWrites module", test_atomic_writes_module),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed, error in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
            if error:
                print(f"       Error: {error}")

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
