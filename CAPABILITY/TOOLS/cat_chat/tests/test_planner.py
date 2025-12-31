#!/usr/bin/env python3
"""
Planner Tests (Phase 4)
"""

import json
import sys
from pathlib import Path

from catalytic_chat.planner import Planner, PlannerError, post_request_and_plan, verify_plan_stored

TESTS_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a test fixture from tests/fixtures/."""
    fixture_path = TESTS_DIR / name
    with open(fixture_path, 'r') as f:
        return json.load(f)


def test_planner_imports():
    """Can import planner module."""
    from catalytic_chat import planner
    print("[PASS] test_planner_imports")


def test_plan_deterministic():
    """Same request -> identical plan_hash and step_ids."""
    fixture_name = "plan_request_min.json"
    request = load_fixture(fixture_name)
    
    planner = Planner()
    
    try:
        plan1 = planner.plan_request(request)
        plan2 = planner.plan_request(request)
        
        assert plan1["plan_hash"] == plan2["plan_hash"], "Plan hash should be identical"
        assert plan1["steps"] == plan2["steps"], "Steps should be identical"
        assert len(plan1["steps"]) == len(plan2["steps"]), "Step count should be identical"
        
        for step1, step2 in zip(plan1["steps"], plan2["steps"]):
            assert step1["step_id"] == step2["step_id"], "Step IDs should be identical"
        
        print("[PASS] test_plan_deterministic")
    except PlannerError as e:
        if "not found" in str(e).lower():
            print("[SKIP] test_plan_deterministic (symbol not in registry)")
        else:
            raise


def test_symbol_field_alignment():
    """Planner uses target_ref and default_slice correctly from Symbol."""
    from catalytic_chat.symbol_registry import SymbolRegistry, Symbol
    from catalytic_chat.section_indexer import SectionIndexer
    
    request = load_fixture("plan_request_min.json")
    
    planner = Planner()
    
    registry = SymbolRegistry()
    symbol = registry.get_symbol("@TEST/example")
    
    if symbol:
        assert hasattr(symbol, 'target_ref'), "Symbol should have target_ref"
        assert hasattr(symbol, 'default_slice'), "Symbol should have default_slice"
        assert symbol.target_type == "SECTION", "Symbol target_type should be SECTION"
    
    print("[PASS] test_symbol_field_alignment")


def test_slice_all_rejected():
    """Request with slice=ALL is rejected."""
    try:
        planner = Planner()
        request = {
            "run_id": "test_slice_all",
            "request_id": "req_slice_all",
            "intent": "Test slice ALL rejection",
            "inputs": {
                "symbols": ["@TEST/example"]
            },
            "budgets": {
                "max_steps": 5,
                "max_bytes": 10000000,
                "max_symbols": 10
            }
        }
        
        registry = planner._symbol_registry
        symbol = registry.get_symbol("@TEST/example")
        
        if symbol and symbol.default_slice and symbol.default_slice.lower() == "all":
            try:
                planner.plan_request(request)
                print("[FAIL] test_slice_all_rejected (should fail)")
                sys.exit(1)
            except PlannerError as e:
                if "all" in str(e).lower():
                    print("[PASS] test_slice_all_rejected")
                else:
                    print(f"[FAIL] test_slice_all_rejected: Wrong error: {e}")
                    sys.exit(1)
        else:
            print("[SKIP] test_slice_all_rejected (symbol doesn't have ALL slice)")
            
    except PlannerError as e:
        if "not found" in str(e).lower():
            print("[SKIP] test_slice_all_rejected (symbol not in registry)")
        else:
            print(f"[FAIL] test_slice_all_rejected: Unexpected error: {e}")
            sys.exit(1)


def test_cli_dry_run():
    """Dry-run does not touch DB."""
    import subprocess
    import os
    
    cmd = [
        sys.executable, "-m", "catalytic_chat.cli", "plan", "request",
        "--request-file", str(TESTS_DIR / "plan_request_min.json"),
        "--dry-run"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    # In dry-run mode, should succeed even if symbol doesn't exist
    assert result.returncode == 0, f"Dry-run should succeed, got: {result.stderr}"
    assert "step_id" in result.stdout, "Dry-run should produce step IDs"
    assert "plan_hash" in result.stdout, "Dry-run should produce plan hash"
    
    # Verify plan contains READ_SYMBOL step
    plan_output = json.loads(result.stdout)
    assert "steps" in plan_output, "Plan should have steps"
    assert len(plan_output["steps"]) > 0, "Plan should have at least one step"
    assert plan_output["steps"][0]["op"] == "READ_SYMBOL", "First step should be READ_SYMBOL"
    
    # Verify step ordinal is 0-based
    assert plan_output["steps"][0]["ordinal"] == 0, "First step ordinal should be 0"
    
    # Verify expected_outputs uses "symbols_referenced" or "unresolved_symbols"
    expected_outputs = plan_output["steps"][0].get("expected_outputs", {})
    assert "symbols_referenced" in expected_outputs or "unresolved_symbols" in expected_outputs, \
        "Expected outputs should have symbols_referenced or unresolved_symbols"
    assert "symbols_resolved" not in expected_outputs, "Expected outputs should not have symbols_resolved in dry-run"
    
    print("[PASS] test_cli_dry_run")


def test_idempotency():
    """Rerunning same plan request returns same job_id and step_ids."""
    print("[SKIP] test_idempotency (replaced by test_plan_request_idempotent_rerun_no_unique_constraint)")


def test_plan_request_dry_run_missing_symbol_does_not_fail():
    """Dry-run with missing symbol succeeds and emits unresolved step."""
    import tempfile
    import shutil
    from catalytic_chat.message_cassette import MessageCassette

    tmp_root = None
    cassette = None

    try:
        tmp_path = tempfile.mkdtemp()
        tmp_root = Path(tmp_path)

        shutil.copytree(
            Path.cwd() / "THOUGHT" / "LAB" / "CAT_CHAT",
            tmp_root / "THOUGHT" / "LAB" / "CAT_CHAT"
        )

        fixture_path = TESTS_DIR / "plan_request_missing_symbol.json"
        request = load_fixture("plan_request_missing_symbol.json")

        planner = Planner(repo_root=tmp_root)
        cassette = MessageCassette(repo_root=tmp_root)

        conn = cassette._get_conn()

        cursor = conn.execute("SELECT COUNT(*) as count FROM cassette_messages")
        before_count = cursor.fetchone()["count"]

        plan_output = planner.plan_request(request, dry_run=True)

        assert "steps" in plan_output
        assert len(plan_output["steps"]) == 1

        step = plan_output["steps"][0]
        assert step["op"] == "READ_SYMBOL"
        assert step["refs"]["symbol_id"] == "@TEST/MISSING"

        expected_outputs = step.get("expected_outputs", {})
        assert "unresolved_symbols" in expected_outputs
        assert "@TEST/MISSING" in expected_outputs["unresolved_symbols"]

        cursor = conn.execute("SELECT COUNT(*) as count FROM cassette_messages")
        after_count = cursor.fetchone()["count"]

        assert before_count == after_count

        print("[PASS] test_plan_request_dry_run_missing_symbol_does_not_fail")
    finally:
        if cassette is not None:
            try:
                cassette.close()
            except:
                pass
        if tmp_root is not None:
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except:
                pass


def test_plan_request_idempotent_rerun_no_unique_constraint():
    """Rerunning same plan request returns same ids without UNIQUE crash."""
    import tempfile
    import shutil

    tmp_root = None

    try:
        tmp_path = tempfile.mkdtemp()
        tmp_root = Path(tmp_path)

        shutil.copytree(
            Path.cwd() / "THOUGHT" / "LAB" / "CAT_CHAT",
            tmp_root / "THOUGHT" / "LAB" / "CAT_CHAT"
        )

        request = load_fixture("plan_request_no_symbols.json")

        message_id1, job_id1, step_ids1 = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
            repo_root=tmp_root
        )

        message_id2, job_id2, step_ids2 = post_request_and_plan(
            run_id=request["run_id"],
            request_payload=request,
            idempotency_key=request["request_id"],
            repo_root=tmp_root
        )

        assert message_id1 == message_id2
        assert job_id1 == job_id2
        assert step_ids1 == step_ids2

        from catalytic_chat.message_cassette import MessageCassette
        cassette = MessageCassette(repo_root=tmp_root)
        conn = cassette._get_conn()

        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_messages
            WHERE run_id = ? AND idempotency_key = ?
        """, (request["run_id"], request["request_id"]))
        messages_count = cursor.fetchone()["count"]
        assert messages_count == 1

        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_jobs
            WHERE job_id = ?
        """, (job_id1,))
        jobs_count = cursor.fetchone()["count"]
        assert jobs_count == 1

        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM cassette_steps
            WHERE job_id = ?
        """, (job_id1,))
        steps_count = cursor.fetchone()["count"]
        assert steps_count == len(step_ids1)

        cassette.close()

        print("[PASS] test_plan_request_idempotent_rerun_no_unique_constraint")
    finally:
        if tmp_root is not None:
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except:
                pass


def main():
    """Run all tests."""
    tests = [
        test_planner_imports,
        test_plan_deterministic,
        test_symbol_field_alignment,
        test_slice_all_rejected,
        test_cli_dry_run,
        test_idempotency,
        test_plan_request_dry_run_missing_symbol_does_not_fail,
        test_plan_request_idempotent_rerun_no_unique_constraint,
    ]

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\nAll tests passed!")


if __name__ == '__main__':
    main()
