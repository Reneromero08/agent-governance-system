#!/usr/bin/env python3
"""
critic.py - Search Protocol Compliance Checker

Implements E.1.3: Critic Protocol Check for validating search log entries
against protocol requirements.
"""

from typing import List, Dict, Any

# Indicators that suggest a conceptual query (should use semantic search, not grep)
CONCEPTUAL_INDICATORS = [
    "how",
    "why",
    "what is",
    "explain",
    "concept",
    "understand",
    "meaning"
]


def check_search_protocol(log_entry: Dict[str, Any]) -> List[str]:
    """
    Check a search log entry for protocol violations.

    Args:
        log_entry: Dict with keys: session_id, timestamp, tool, query, results

    Returns:
        List of violation strings (empty if compliant)
    """
    violations = []

    # Check for missing session_id
    if not log_entry.get("session_id"):
        violations.append("MISSING_SESSION_ID: Every search must have a session_id")

    # Check for keyword search on conceptual queries
    tool = log_entry.get("tool", "")
    query = log_entry.get("query", "").lower()

    if tool == "grep_search":
        for indicator in CONCEPTUAL_INDICATORS:
            if indicator in query:
                violations.append(
                    f"CONCEPTUAL_GREP: Query '{log_entry.get('query', '')}' "
                    f"contains conceptual indicator '{indicator}' - "
                    "consider using semantic search instead of grep_search"
                )
                break  # Only report once per entry

    # Check for empty results
    results = log_entry.get("results", None)
    if results is not None and isinstance(results, list) and len(results) == 0:
        violations.append("EMPTY_RESULTS: Search returned no results - potential issue")

    return violations


def run_tests() -> bool:
    """
    Run self-tests for the search protocol checker.

    Returns:
        True if all tests pass, False otherwise
    """
    all_passed = True

    print("=" * 60)
    print("Search Protocol Checker - Self Tests")
    print("=" * 60)

    # Test 1: Valid log entry (no violations)
    print("\nTest 1: Valid log entry (should have 0 violations)")
    valid_entry = {
        "session_id": "abc123",
        "timestamp": "2026-01-18T10:00:00Z",
        "tool": "grep_search",
        "query": "function_name",
        "results": [{"file": "test.py", "line": 10}]
    }
    violations = check_search_protocol(valid_entry)
    if len(violations) == 0:
        print("  [PASS] No violations found")
    else:
        print(f"  [FAIL] Expected 0 violations, got {len(violations)}: {violations}")
        all_passed = False

    # Test 2: Missing session_id (1 violation)
    print("\nTest 2: Missing session_id (should have 1 violation)")
    missing_session_entry = {
        "timestamp": "2026-01-18T10:00:00Z",
        "tool": "grep_search",
        "query": "function_name",
        "results": [{"file": "test.py", "line": 10}]
    }
    violations = check_search_protocol(missing_session_entry)
    if len(violations) == 1 and "MISSING_SESSION_ID" in violations[0]:
        print("  [PASS] Correctly flagged missing session_id")
    else:
        print(f"  [FAIL] Expected 1 MISSING_SESSION_ID violation, got: {violations}")
        all_passed = False

    # Test 3: Conceptual grep search (1 violation)
    print("\nTest 3: Conceptual grep search (should have 1 violation)")
    conceptual_grep_entry = {
        "session_id": "abc123",
        "timestamp": "2026-01-18T10:00:00Z",
        "tool": "grep_search",
        "query": "how does authentication work",
        "results": [{"file": "auth.py", "line": 5}]
    }
    violations = check_search_protocol(conceptual_grep_entry)
    if len(violations) == 1 and "CONCEPTUAL_GREP" in violations[0]:
        print("  [PASS] Correctly flagged conceptual grep query")
    else:
        print(f"  [FAIL] Expected 1 CONCEPTUAL_GREP violation, got: {violations}")
        all_passed = False

    # Test 4: Empty results (1 violation)
    print("\nTest 4: Empty results (should have 1 violation)")
    empty_results_entry = {
        "session_id": "abc123",
        "timestamp": "2026-01-18T10:00:00Z",
        "tool": "semantic_search",
        "query": "non-existent pattern",
        "results": []
    }
    violations = check_search_protocol(empty_results_entry)
    if len(violations) == 1 and "EMPTY_RESULTS" in violations[0]:
        print("  [PASS] Correctly flagged empty results")
    else:
        print(f"  [FAIL] Expected 1 EMPTY_RESULTS violation, got: {violations}")
        all_passed = False

    # Test 5: Multiple violations
    print("\nTest 5: Multiple violations (should have 3 violations)")
    multi_violation_entry = {
        "timestamp": "2026-01-18T10:00:00Z",
        "tool": "grep_search",
        "query": "explain the concept",
        "results": []
    }
    violations = check_search_protocol(multi_violation_entry)
    if len(violations) == 3:
        print("  [PASS] Correctly found 3 violations")
        for v in violations:
            print(f"    - {v}")
    else:
        print(f"  [FAIL] Expected 3 violations, got {len(violations)}: {violations}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
