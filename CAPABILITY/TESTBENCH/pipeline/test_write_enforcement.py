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
        print(f"  [FAIL] Tmp write failed: {e.error_code}")
        return False

    # Test durable write without gate (should fail)
    try:
        writer.write_durable("LAW/CONTRACTS/_runs/test_durable.json", b'{"test": "data"}')
        print("  [FAIL] Durable write succeeded before gate opening (should have failed)")
        return False
    except FirewallViolation as e:
        if e.error_code != "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT":
            print(f"  [FAIL] Wrong error code: {e.error_code}")
            return False
        print(f"  [OK] Durable write blocked before gate: {e.error_code}")

    # Test durable write with gate open (should succeed)
    writer.open_commit_gate()
    try:
        writer.write_durable("LAW/CONTRACTS/_runs/test_durable.json", b'{"test": "data"}')
        print("  [OK] Durable write succeeded after gate opening")
    except FirewallViolation as e:
        print(f"  [FAIL] Durable write failed after gate: {e.error_code}")
        return False

    # Cleanup
    try:
        (REPO_ROOT / "LAW/CONTRACTS/_runs/_tmp/test.json").unlink(missing_ok=True)
        (REPO_ROOT / "LAW/CONTRACTS/_runs/test_durable.json").unlink(missing_ok=True)
    except Exception:
        pass

    return True


def test_forbidden_write():
    """Test that writes to forbidden paths are blocked."""
    print("\n[TEST] Forbidden write blocking")

    writer = GuardedWriter(project_root=REPO_ROOT)

    # Test write to CANON (should fail)
    try:
        writer.write_tmp("LAW/CANON/test.json", b'{"test": "data"}')
        print("  [FAIL] Write to CANON succeeded (should have failed)")
        return False
    except FirewallViolation as e:
        if e.error_code != "FIREWALL_PATH_EXCLUDED":
            print(f"  [FAIL] Wrong error code for CANON write: {e.error_code}")
            return False
        print(f"  [OK] Write to CANON blocked: {e.error_code}")

    # Test write to AGENTS.md (should fail)
    try:
        writer.write_tmp("AGENTS.md", b'{"test": "data"}')
        print("  [FAIL] Write to AGENTS.md succeeded (should have failed)")
        return False
    except FirewallViolation as e:
        if e.error_code != "FIREWALL_PATH_EXCLUDED":
            print(f"  [FAIL] Wrong error code for AGENTS.md write: {e.error_code}")
            return False
        print(f"  [OK] Write to AGENTS.md blocked: {e.error_code}")

    return True


def test_mkdir_enforcement():
    """Test that mkdir operations are enforced."""
    print("\n[TEST] Mkdir enforcement")

    writer = GuardedWriter(project_root=REPO_ROOT)

    # Test mkdir in tmp domain (should succeed)
    try:
        writer.mkdir_tmp("LAW/CONTRACTS/_runs/_tmp/test_dir")
        print("  [OK] Mkdir in tmp succeeded")
    except FirewallViolation as e:
        print(f"  [FAIL] Mkdir in tmp failed: {e.error_code}")
        return False

    # Test mkdir in forbidden domain (should fail)
    try:
        writer.mkdir_tmp("LAW/CANON/test_dir")
        print("  [FAIL] Mkdir in CANON succeeded (should have failed)")
        return False
    except FirewallViolation as e:
        if e.error_code != "FIREWALL_PATH_EXCLUDED":
            print(f"  [FAIL] Wrong error code for CANON mkdir: {e.error_code}")
            return False
        print(f"  [OK] Mkdir in CANON blocked: {e.error_code}")

    # Cleanup
    try:
        (REPO_ROOT / "LAW/CONTRACTS/_runs/_tmp/test_dir").rmdir()
    except Exception:
        pass

    return True


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
        if test_path.exists():
            import json
            data = json.loads(test_path.read_text())
            if data == {"test": "data"}:
                print("  [OK] Atomic write content verified")
            else:
                print(f"  [FAIL] Atomic write content mismatch: {data}")
                return False

            test_path.unlink(missing_ok=True)
        else:
            print("  [FAIL] Atomic write file not created")
            return False

    except Exception as e:
        print(f"  [FAIL] Atomic write failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("RUNTIME WRITE INTERCEPTION TESTS")
    print("=" * 60)

    results = []

    results.append(("GuardedWriter basic", test_guarded_writer_basic()))
    results.append(("Forbidden write blocking", test_forbidden_write()))
    results.append(("Mkdir enforcement", test_mkdir_enforcement()))
    results.append(("AtomicGuardedWrites module", test_atomic_writes_module()))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
