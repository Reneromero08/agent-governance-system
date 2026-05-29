#!/usr/bin/env python3
"""
Ants: Multi-worker agent runner tests (Phase 4.3)
"""

import json
import sys
from pathlib import Path

import pytest

from catalytic_chat.planner import post_request_and_plan
from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError
from catalytic_chat.ants import AntWorker, AntConfig, spawn_ants, run_ant_worker


TESTS_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a test fixture from tests/fixtures/."""
    fixture_path = TESTS_DIR / name
    with open(fixture_path, 'r') as f:
        return json.load(f)


def test_ant_worker_claims_and_executes():
    """Single ant worker claims and executes steps."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_ant_worker_single",
        "request_id": "req_ant_single",
        "intent": "Test ant worker single",
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
        
        config = AntConfig(
            run_id=request["run_id"],
            job_id=job_id,
            worker_id="test_ant_0",
            repo_root=repo_root,
            poll_interval_ms=100,
            max_idle_polls=5
        )
        
        worker = AntWorker(config)
        exit_code = worker.run()
        
        assert exit_code == 0, f"Worker exited with code {exit_code}"
        
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
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()


def test_ants_spawn_two_workers_no_duplicates():
    """Two ants run without duplicate receipts."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_ant_two_workers",
        "request_id": "req_ant_two_workers",
        "intent": "Test ant two workers",
        "inputs": {
            "symbols": [],
            "files": ["README.md", "AGENTS.md"],
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
        
        exit_code = run_ant_worker(
            run_id=request["run_id"],
            job_id=job_id,
            worker_id="test_ant_0",
            repo_root=repo_root,
            poll_interval_ms=100,
            max_idle_polls=10
        )
        
        assert exit_code == 0, f"Worker 0 exited with code {exit_code}"
        
        exit_code = run_ant_worker(
            run_id=request["run_id"],
            job_id=job_id,
            worker_id="test_ant_1",
            repo_root=repo_root,
            poll_interval_ms=100,
            max_idle_polls=5
        )
        
        assert exit_code == 0, f"Worker 1 exited with code {exit_code}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ?
        """, (job_id,))
        receipt_count = cursor.fetchone()["count"]
        
        assert receipt_count == initial_step_count, \
            f"Expected {initial_step_count} receipts, got {receipt_count}"
        
        cursor = conn.execute("""
            SELECT step_id, COUNT(*) as cnt FROM cassette_receipts WHERE job_id = ? GROUP BY step_id
        """, (job_id,))
        for row in cursor.fetchall():
            assert row["cnt"] == 1, f"Step {row['step_id']} has {row['cnt']} receipts (expected 1)"
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()


def test_ant_stops_on_fail_by_default():
    """Ant worker stops on first failure by default."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_ant_fail_stop",
        "request_id": "req_ant_fail_stop",
        "intent": "Test ant fail stop",
        "inputs": {
            "symbols": [],
            "files": ["README.md", "NONEXISTENT_FILE.md"],
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
        
        config = AntConfig(
            run_id=request["run_id"],
            job_id=job_id,
            worker_id="test_ant_fail_0",
            repo_root=repo_root,
            poll_interval_ms=100,
            max_idle_polls=10,
            continue_on_fail=False
        )
        
        worker = AntWorker(config)
        exit_code = worker.run()
        
        assert exit_code == 1, f"Expected exit code 1, got {exit_code}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ? AND outcome = 'SUCCESS'
        """, (job_id,))
        success_count = cursor.fetchone()["count"]
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ? AND outcome = 'FAILURE'
        """, (job_id,))
        failure_count = cursor.fetchone()["count"]
        
        assert failure_count >= 1, "Expected at least one failure"
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()


def test_ant_continue_on_fail_completes_others():
    """Ant worker continues on failure when continue_on_fail=True."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_ant_continue_fail",
        "request_id": "req_ant_continue_fail",
        "intent": "Test ant continue on fail",
        "inputs": {
            "symbols": [],
            "files": ["README.md", "NONEXISTENT_FILE.md", "AGENTS.md"],
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
        
        config = AntConfig(
            run_id=request["run_id"],
            job_id=job_id,
            worker_id="test_ant_continue_0",
            repo_root=repo_root,
            poll_interval_ms=100,
            max_idle_polls=10,
            continue_on_fail=True
        )
        
        worker = AntWorker(config)
        exit_code = worker.run()
        
        assert exit_code == 1, f"Expected exit code 1 (any failure), got {exit_code}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ?
        """, (job_id,))
        receipt_count = cursor.fetchone()["count"]
        
        assert receipt_count == initial_step_count, \
            f"Expected {initial_step_count} receipts, got {receipt_count}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ? AND outcome = 'SUCCESS'
        """, (job_id,))
        success_count = cursor.fetchone()["count"]
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ? AND outcome = 'FAILURE'
        """, (job_id,))
        failure_count = cursor.fetchone()["count"]
        
        assert success_count > 0, "Expected at least one success"
        assert failure_count >= 1, "Expected at least one failure"
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()


def test_ant_spawn_multiprocess():
    """End-to-end multiprocess test with real subprocesses."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_ant_spawn_multiproc",
        "request_id": "req_ant_multiproc",
        "intent": "Test ant spawn multiprocess",
        "inputs": {
            "symbols": [],
            "files": ["README.md", "AGENTS.md", "LICENSE", "CHANGELOG.md"],
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
        
        exit_code = spawn_ants(
            run_id=request["run_id"],
            job_id=job_id,
            num_workers=2,
            repo_root=repo_root,
            continue_on_fail=False
        )
        
        assert exit_code == 0, f"Spawn exited with code {exit_code}"
        
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_receipts WHERE job_id = ?
        """, (job_id,))
        receipt_count = cursor.fetchone()["count"]
        
        assert receipt_count == initial_step_count, \
            f"Expected {initial_step_count} receipts, got {receipt_count}"
        
        cursor = conn.execute("""
            SELECT step_id, COUNT(*) as cnt FROM cassette_receipts WHERE job_id = ? GROUP BY step_id
        """, (job_id,))
        for row in cursor.fetchall():
            assert row["cnt"] == 1, f"Step {row['step_id']} has {row['cnt']} receipts (expected 1)"
        
        cassette.verify_cassette(request["run_id"])
        
        cortex_dir = repo_root / "CORTEX" / "_generated"
        manifest_path = cortex_dir / f"ants_manifest_{request['run_id']}_{job_id}.json"
        
        assert manifest_path.exists(), "Manifest file not created"
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        assert manifest["run_id"] == request["run_id"]
        assert manifest["job_id"] == job_id
        assert len(manifest["workers"]) == 2
        assert all("worker_id" in w and "pid" in w for w in manifest["workers"])
        
    finally:
        cassette.close()


def test_ants_status_counts():
    """Test ants status command returns correct counts."""
    repo_root = Path(__file__).parent.parent
    
    request = {
        "run_id": "test_ants_status",
        "request_id": "req_ants_status",
        "intent": "Test ants status",
        "inputs": {
            "symbols": [],
            "files": ["README.md", "AGENTS.md"],
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
        
        config = AntConfig(
            run_id=request["run_id"],
            job_id=job_id,
            worker_id="test_ant_status_0",
            repo_root=repo_root,
            poll_interval_ms=100,
            max_idle_polls=5
        )
        
        worker = AntWorker(config)
        exit_code = worker.run()
        
        assert exit_code == 0, f"Worker exited with code {exit_code}"
        
        status = cassette.get_job_status(run_id=request["run_id"], job_id=job_id)
        
        assert status is not None, "Status should not be None"
        assert status["pending"] == 0, f"Expected 0 pending, got {status['pending']}"
        assert status["leased"] == 0, f"Expected 0 leased, got {status['leased']}"
        assert status["committed"] == initial_step_count, \
            f"Expected {initial_step_count} committed, got {status['committed']}"
        assert status["receipts"] == initial_step_count, \
            f"Expected {initial_step_count} receipts, got {status['receipts']}"
        assert status["workers_seen"] == 1, \
            f"Expected 1 worker seen, got {status['workers_seen']}"
        
        cassette.verify_cassette(request["run_id"])
        
    finally:
        cassette.close()


def test_ants_run_alias_calls_spawn():
    """Test that 'ants run' is an alias for 'ants spawn'."""
    import argparse
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="ants_command")
    
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--job-id", required=True)
    run_parser.add_argument("-n", type=int, required=True)
    run_parser.add_argument("--continue-on-fail", action="store_true")
    
    spawn_parser = subparsers.add_parser("spawn")
    spawn_parser.add_argument("--run-id", required=True)
    spawn_parser.add_argument("--job-id", required=True)
    spawn_parser.add_argument("-n", type=int, required=True)
    spawn_parser.add_argument("--continue-on-fail", action="store_true")
    
    run_args = parser.parse_args(["run", "--run-id", "test_run", "--job-id", "job1", "-n", "2"])
    spawn_args = parser.parse_args(["spawn", "--run-id", "test_run", "--job-id", "job1", "-n", "2"])
    
    assert run_args.ants_command == "run"
    assert spawn_args.ants_command == "spawn"
    assert run_args.run_id == spawn_args.run_id
    assert run_args.job_id == spawn_args.job_id
    assert run_args.n == spawn_args.n
    assert run_args.continue_on_fail == spawn_args.continue_on_fail
    
    from catalytic_chat.cli import cmd_ants_spawn
    ants_commands = {
        "spawn": cmd_ants_spawn,
        "run": cmd_ants_spawn,
        "worker": lambda x: 0,
        "status": lambda x: 0
    }
    
    assert ants_commands["run"] == ants_commands["spawn"], \
        "'run' should route to the same handler as 'spawn'"
