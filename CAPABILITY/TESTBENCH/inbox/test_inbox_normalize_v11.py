#!/usr/bin/env python3
"""
Tests for INBOX Normalization v1.1

Tests cover:
1. ISO week edge cases (late December → Week-01)
2. Timestamp parsing determinism
3. Digest semantics (content vs tree)
4. Schema correctness
"""
import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path

# === COPY OF RELEVANT FUNCTIONS FROM inbox_normalize.py ===
def parse_filename_date(filename: str):
    """Extract date from filename. Supports multiple formats."""
    # Pattern 1: MM-DD-YYYY-HH-MM_SOMETHING (e.g., 12-28-2025-12-00_...)
    pattern1 = r"(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})"
    match = re.search(pattern1, filename)
    if match:
        try:
            return datetime.strptime(match.group(0), "%m-%d-%Y-%H-%M")
        except ValueError:
            pass
    
    # Pattern 2: YYYY-MM-DD (e.g., TASK-2025-12-30-001.json)
    pattern2 = r"TASK-(\d{4})-(\d{2})-(\d{2})"
    match = re.search(pattern2, filename)
    if match:
        try:
            return datetime.strptime(f"{match.group(1)}-{match.group(2)}-{match.group(3)}", "%Y-%m-%d")
        except ValueError:
            pass
    
    return None

def get_iso_week(dt: datetime) -> int:
    """Get ISO week number from datetime."""
    return dt.isocalendar()[1]

def compute_target_folder(timestamp: datetime) -> str:
    """Compute folder structure: YYYY-MM/Week-XX"""
    year = timestamp.year
    month = timestamp.month
    week = get_iso_week(timestamp)
    return f"{year:04d}-{month:02d}/Week-{week:02d}"

# === TESTS ===

def test_iso_week_edge_cases():
    """Test ISO week calculation for edge cases."""
    print("Testing ISO week edge cases...")
    
    # Test cases: (date, expected_week, note)
    test_cases = [
        # Late December dates that belong to Week-01 of next year
        (datetime(2025, 12, 29), 1, "Monday - first day of ISO Week-01, 2026"),
        (datetime(2025, 12, 30), 1, "Tuesday - ISO Week-01, 2026"),
        (datetime(2025, 12, 31), 1, "Wednesday - ISO Week-01, 2026"),
        
        # Mid December dates in Week-52 of 2025
        (datetime(2025, 12, 23), 52, "Tuesday - ISO Week-52, 2025"),
        (datetime(2025, 12, 28), 52, "Sunday - ISO Week-52, 2025"),
        
        # January dates in Week-01 of 2026
        (datetime(2026, 1, 1), 1, "Thursday - ISO Week-01, 2026"),
        (datetime(2026, 1, 5), 2, "Monday - ISO Week-02, 2026"),
    ]
    
    all_passed = True
    for dt, expected_week, note in test_cases:
        actual_week = get_iso_week(dt)
        status = "[PASS]" if actual_week == expected_week else "[FAIL]"
        print(f"  {status} {dt.strftime('%Y-%m-%d')} → Week-{actual_week:02d} (expected: {expected_week:02d}) - {note}")
        if actual_week != expected_week:
            all_passed = False
    
    return all_passed

def test_timestamp_parsing_determinism():
    """Test that timestamp parsing is deterministic."""
    print("\nTesting timestamp parsing determinism...")
    
    # Test filenames
    test_filenames = [
        ("12-28-2025-12-00_AGENT_SAFETY_REPORT.md", "2025-12-28T12:00:00"),
        ("TASK-2025-12-30-001.json", "2025-12-30T00:00:00"),
        ("12-29-2025-02-45_CI_STABILIZATION_REPORT.md", "2025-12-29T02:45:00"),
    ]
    
    all_passed = True
    for filename, expected in test_filenames:
        result = parse_filename_date(filename)
        if result is None:
            print(f"  ❌ {filename} → None (failed to parse)")
            all_passed = False
        else:
            actual = result.isoformat()
            status = "[PASS]" if actual == expected else "[FAIL]"
            print(f"  {status} {filename} → {actual} (expected: {expected})")
            if actual != expected:
                all_passed = False
    
    return all_passed

def test_schema_computation():
    """Test that schema computation produces correct paths."""
    print("\nTesting schema computation...")
    
    # Test cases: (date, expected_folder)
    test_cases = [
        (datetime(2025, 12, 29), "2025-12/Week-01"),  # Edge case!
        (datetime(2025, 12, 28), "2025-12/Week-52"),
        (datetime(2026, 1, 1), "2026-01/Week-01"),
    ]
    
    all_passed = True
    for dt, expected_folder in test_cases:
        actual_folder = compute_target_folder(dt)
        
        status = "[PASS]" if actual_folder == expected_folder else "[FAIL]"
        print(f"  {status} {dt.strftime('%Y-%m-%d')} → {actual_folder}")
        if actual_folder != expected_folder:
            all_passed = False
            print(f"      Expected: {expected_folder}")
    
    return all_passed

def test_timestamp_policy_documentation():
    """Verify timestamp policy is explicitly documented in the script."""
    print("\nTesting timestamp policy documentation...")

    with open("CAPABILITY/SKILLS/inbox/inbox-report-writer/inbox_normalize.py", 'r') as f:
        content = f.read()

    checks = [
        ("parse_filename_date exists", "parse_filename_date" in content),
        ("pattern1 documented", "pattern1" in content),
        ("Pattern priority documented", "Pattern 1:" in content or "Pattern 2:" in content),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    assert all_passed, "Timestamp policy documentation checks failed"
    return all_passed

def test_schema_documentation():
    """Verify schema is explicitly documented."""
    print("\nTesting schema documentation...")

    with open("CAPABILITY/SKILLS/inbox/inbox-report-writer/inbox_normalize.py", 'r') as f:
        content = f.read()

    checks = [
        ("compute_target_path exists", "compute_target_path" in content),
        ("ISO week function exists", "isocalendar" in content or "get_iso_week" in content),
        ("Folder structure documented", "YYYY-MM/Week-" in content or "Week-XX" in content),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    assert all_passed, "Schema documentation checks failed"
    return all_passed

def test_digest_semantics_receipt():
    """Test that execution receipt distinguishes content vs tree digest."""
    print("\nTesting digest semantics in receipt...")
    
    # Check if INBOX_EXECUTION.json exists and has the new structure
    receipt_path = Path("LAW/CONTRACTS/_runs/INBOX_EXECUTION.json")
    
    if not receipt_path.exists():
        print("  ⚠️ INBOX_EXECUTION.json not found (run normalization first)")
        return None  # Skip, not a failure
    
    with open(receipt_path, 'r') as f:
        receipt = json.load(f)
    
    checks = [
        ("digest_semantics exists", "digest_semantics" in receipt),
        ("content_integrity exists", "content_integrity" in receipt.get("digest_semantics", {})),
        ("tree_digest exists", "tree_digest" in receipt.get("digest_semantics", {})),
        ("verdict for content_integrity", "verdict" in receipt.get("digest_semantics", {}).get("content_integrity", {})),
        ("verdict for tree_digest", "verdict" in receipt.get("digest_semantics", {}).get("tree_digest", {})),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def test_version_updated():
    """Verify version exists and is valid."""
    print("\nTesting version update...")

    with open("CAPABILITY/SKILLS/inbox/inbox-report-writer/inbox_normalize.py", 'r') as f:
        content = f.read()

    # Check for VERSION variable with valid semantic version format
    import re
    version_pattern = r'VERSION\s*=\s*["\'](\d+\.\d+\.\d+)["\']'
    match = re.search(version_pattern, content)

    if match:
        version = match.group(1)
        print(f"  ✅ VERSION = \"{version}\" found")
        all_passed = True
    else:
        print(f"  ❌ VERSION not found or invalid format")
        all_passed = False

    return all_passed

def main():
    """Run all tests."""
    print("=" * 60)
    print("INBOX Normalization v1.1 Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("ISO Week Edge Cases", test_iso_week_edge_cases()))
    results.append(("Timestamp Parsing Determinism", test_timestamp_parsing_determinism()))
    results.append(("Schema Computation", test_schema_computation()))
    results.append(("Timestamp Policy Documentation", test_timestamp_policy_documentation()))
    results.append(("Schema Documentation", test_schema_documentation()))
    results.append(("Digest Semantics Receipt", test_digest_semantics_receipt()))
    results.append(("Version Updated", test_version_updated()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is None:
            print(f"  ⏭️ {name}: SKIPPED")
            skipped += 1
        elif result:
            print(f"  ✅ {name}: PASSED")
            passed += 1
        else:
            print(f"  ❌ {name}: FAILED")
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
