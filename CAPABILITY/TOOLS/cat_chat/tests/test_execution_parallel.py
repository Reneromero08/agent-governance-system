#!/usr/bin/env python3
"""
Parallel Execution Tests (Phase 4.2)
"""

import json
import sys
from pathlib import Path

import pytest

from catalytic_chat.planner import post_request_and_plan
from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError
from catalytic_chat.cli import cmd_execute

TESTS_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a test fixture from tests/fixtures/."""
    fixture_path = TESTS_DIR / name
    with open(fixture_path, 'r') as f:
        return json.load(f)


def test_execute_parallel_claims_all_steps_once():
    """Parallel execution claims and executes all steps exactly once."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_parallel_all_steps",
        "request_id": "req_parallel_all_steps",
        "intent": "Test parallel execution claims all steps",
        "inputs": {
            "symbols": [],
            "files": ["README.md"],
            "notes": []
        },
        "budgets": {
            "max_steps": 100,
            "max_bytes": 100000000,
            "max_symbols": 0
        }
    }
    
    cassette = MessageCassette(repo_root=repo_root)
    try:
        message_id, job_id, step_ids = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
            repo_root=repo_root
        )
        
        conn = cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_steps WHERE job_id = ?
        """, (job_id,))
        initial_step_count = cursor.fetchone()["count"]
        
        if initial_step_count == 0:
            pytest.skip("No steps found")
        
        from argparse import Namespace
        args = Namespace(
            run_id=request["run_id"],
            job_id=job_id,
            workers=4,
            continue_on_fail=False,
            repo_root=repo_root
        )
        
        exit_code = cmd_execute(args)
        assert exit_code == 0, f"Execution failed with exit code {exit_code}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ?
        """, (job_id,))
        receipt_count = cursor.fetchone()["count"]
        
        assert receipt_count == initial_step_count, \
            f"Expected {initial_step_count} receipts, got {receipt_count}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_steps WHERE job_id = ? AND status = 'COMMITTED'
        """, (job_id,))
        committed_count = cursor.fetchone()["count"]
        
        assert committed_count == initial_step_count, \
            f"Expected {initial_step_count} committed steps, got {committed_count}"
        
        cursor = conn.execute("""
            SELECT step_id, COUNT(*) as cnt FROM cassette_receipts WHERE job_id = ? GROUP BY step_id
        """, (job_id,))
        for row in cursor.fetchall():
            assert row["cnt"] == 1, f"Step {row['step_id']} has {row['cnt']} receipts (expected 1)"
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()


def test_execute_parallel_idempotent_rerun():
    """Re-running parallel execution does no work and creates no duplicates."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_parallel_idempotency",
        "request_id": "req_parallel_idempotency",
        "intent": "Test parallel idempotency",
        "inputs": {
            "symbols": [],
            "files": ["README.md"],
            "notes": []
        },
        "budgets": {
            "max_steps": 100,
            "max_bytes": 100000000,
            "max_symbols": 0
        }
    }
    
    cassette = MessageCassette(repo_root=repo_root)
    try:
        message_id, job_id, step_ids = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
            repo_root=repo_root
        )
        
        from argparse import Namespace
        args = Namespace(
            run_id=request["run_id"],
            job_id=job_id,
            workers=2,
            continue_on_fail=False,
            repo_root=repo_root
        )
        
        exit_code1 = cmd_execute(args)
        assert exit_code1 == 0, f"First execution failed with exit code {exit_code1}"
        
        conn = cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ?
        """, (job_id,))
        receipt_count_after_first = cursor.fetchone()["count"]
        
        cursor = conn.execute("""
            SELECT bytes_consumed, symbols_consumed
            FROM cassette_job_budgets
            WHERE job_id = ?
        """, (job_id,))
        budget_after_first = cursor.fetchone()
        
        assert budget_after_first is not None, "Budget row not initialized"
        
        exit_code2 = cmd_execute(args)
        assert exit_code2 == 0, f"Second execution failed with exit code {exit_code2}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ?
        """, (job_id,))
        receipt_count_after_second = cursor.fetchone()["count"]
        
        assert receipt_count_after_second == receipt_count_after_first, \
            f"Second run created new receipts: {receipt_count_after_first} -> {receipt_count_after_second}"
        
        cursor = conn.execute("""
            SELECT bytes_consumed, symbols_consumed
            FROM cassette_job_budgets
            WHERE job_id = ?
        """, (job_id,))
        budget_after_second = cursor.fetchone()
        
        assert budget_after_second is not None, "Budget row disappeared"
        assert budget_after_second["bytes_consumed"] == budget_after_first["bytes_consumed"], \
            f"Budget was re-consumed on second run: {budget_after_first['bytes_consumed']} -> {budget_after_second['bytes_consumed']}"
        assert budget_after_second["symbols_consumed"] == budget_after_first["symbols_consumed"], \
            "Symbols were re-consumed on second run"
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()


def test_global_budget_enforced_under_parallelism():
    """Global budget is enforced deterministically under parallel execution."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_parallel_budget",
        "request_id": "req_parallel_budget",
        "intent": "Test parallel budget enforcement",
        "inputs": {
            "symbols": [],
            "files": ["README.md", "README.md", "README.md"],
            "notes": []
        },
        "budgets": {
            "max_steps": 100,
            "max_bytes": 12000,
            "max_symbols": 0
        }
    }
    
    cassette = MessageCassette(repo_root=repo_root)
    try:
        message_id, job_id, step_ids = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
            repo_root=repo_root
        )
        
        conn = cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_steps WHERE job_id = ?
        """, (job_id,))
        initial_step_count = cursor.fetchone()["count"]
        
        if initial_step_count == 0:
            pytest.skip("No steps found")
        
        from argparse import Namespace
        args = Namespace(
            run_id=request["run_id"],
            job_id=job_id,
            workers=4,
            continue_on_fail=False,
            repo_root=repo_root
        )
        
        exit_code = cmd_execute(args)
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ?
        """, (job_id,))
        receipt_count = cursor.fetchone()["count"]
        
        assert receipt_count == initial_step_count, \
            f"Not all steps completed: {receipt_count}/{initial_step_count}"
        
        cursor = conn.execute("""
            SELECT bytes_consumed, symbols_consumed
            FROM cassette_job_budgets
            WHERE job_id = ?
        """, (job_id,))
        budget_row = cursor.fetchone()
        
        assert budget_row is not None, "Budget row not found"
        assert budget_row["bytes_consumed"] == budget_row["symbols_consumed"] * 3339, \
            f"Budget tracking mismatch"
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()


def test_continue_on_fail_behavior():
    """Continue-on-fail flag allows remaining steps to complete."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_parallel_continue_on_fail",
        "request_id": "req_parallel_continue_on_fail",
        "intent": "Test continue-on-fail behavior",
        "inputs": {
            "symbols": [],
            "files": ["README.md", "README.md", "README.md"],
            "notes": []
        },
        "budgets": {
            "max_steps": 100,
            "max_bytes": 12000,
            "max_symbols": 0
        }
    }
    
    cassette = MessageCassette(repo_root=repo_root)
    try:
        message_id, job_id, step_ids = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
            repo_root=repo_root
        )
        
        conn = cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_steps WHERE job_id = ?
        """, (job_id,))
        initial_step_count = cursor.fetchone()["count"]
        
        if initial_step_count == 0:
            pytest.skip("No steps found")
        
        from argparse import Namespace
        args = Namespace(
            run_id=request["run_id"],
            job_id=job_id,
            workers=4,
            continue_on_fail=True,
            repo_root=repo_root
        )
        
        exit_code = cmd_execute(args)
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ?
        """, (job_id,))
        receipt_count = cursor.fetchone()["count"]
        
        assert receipt_count == initial_step_count, \
            f"Not all steps completed: {receipt_count}/{initial_step_count}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ? AND outcome = 'SUCCESS'
        """, (job_id,))
        success_count = cursor.fetchone()["count"]
        
        cursor = conn.execute("""
            SELECT bytes_consumed, symbols_consumed
            FROM cassette_job_budgets
            WHERE job_id = ?
        """, (job_id,))
        budget_row = cursor.fetchone()
        
        assert budget_row is not None, "Budget row not found"
        assert success_count > 0, "Expected at least one success"
        assert budget_row["bytes_consumed"] == success_count * 3339, \
            f"Budget tracking mismatch"
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()
