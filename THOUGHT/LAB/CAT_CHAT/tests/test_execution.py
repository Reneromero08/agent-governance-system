#!/usr/bin/env python3
"""
Execution Tests (Phase 4.1)
"""

import json
import sys
from pathlib import Path

from catalytic_chat.planner import Planner, post_request_and_plan, verify_plan_stored
from catalytic_chat.message_cassette import MessageCassette

TESTS_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a test fixture from tests/fixtures/."""
    fixture_path = TESTS_DIR / name
    with open(fixture_path, 'r') as f:
        return json.load(f)


def test_steps_created_after_plan_request():
    """Steps are created in cassette after plan request."""
    from catalytic_chat.message_cassette import MessageCassette
    
    request = {
        "run_id": "test_steps_creation",
        "request_id": "req_steps_creation",
        "intent": "Test steps creation",
        "inputs": {"symbols": [], "files": [], "notes": []},
        "budgets": {"max_steps": 1, "max_bytes": 10000000, "max_symbols": 10}
    }
    
    cassette = MessageCassette()
    try:
        message_id, job_id, step_ids = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
        )
        
        conn = cassette._get_conn()
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_steps WHERE job_id = ?
        """, (job_id,))
        count = cursor.fetchone()["count"]
        
        assert count == 0, f"Expected 0 steps for plan with no symbols, got {count}"
        print("[PASS] test_steps_created_after_plan_request")
        
    except Exception as e:
        print(f"[FAIL] test_steps_created_after_plan_request: {e}")
        sys.exit(1)
    finally:
        cassette.close()


def test_plan_request_idempotent_no_duplicate_steps():
    """Re-running plan request does not duplicate steps."""
    from catalytic_chat.message_cassette import MessageCassette
    
    request = {
        "run_id": "test_idempotency",
        "request_id": "req_idempotency",
        "intent": "Test idempotency",
        "inputs": {"symbols": [], "files": [], "notes": []},
        "budgets": {"max_steps": 1, "max_bytes": 10000000, "max_symbols": 10}
    }
    
    cassette = MessageCassette()
    try:
        message_id1, job_id1, step_ids1 = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
        )
        
        message_id2, job_id2, step_ids2 = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
        )
        
        assert message_id1 == message_id2, "Message ID should be identical"
        assert job_id1 == job_id2, "Job ID should be identical"
        
        conn = cassette._get_conn()
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_steps WHERE job_id = ?
        """, (job_id1,))
        count = cursor.fetchone()["count"]
        
        assert count == 0, f"Expected 0 steps after two identical plan requests, got {count}"
        print("[PASS] test_plan_request_idempotent_no_duplicate_steps")
        
    except Exception as e:
        print(f"[FAIL] test_plan_request_idempotent_no_duplicate_steps: {e}")
        sys.exit(1)
    finally:
        cassette.close()


def main():
    """Run all tests."""
    tests = [
        test_steps_created_after_plan_request,
        test_plan_request_idempotent_no_duplicate_steps,
    ]
    
    for test in tests:
        test()
    
    print("\nAll tests passed!")


if __name__ == '__main__':
    main()
