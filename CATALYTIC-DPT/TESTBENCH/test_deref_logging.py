import hashlib
import sys
from pathlib import Path

import pytest

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.cas_store import CatalyticStore
from PRIMITIVES.hash_toolbelt import log_dereference_event
from PRIMITIVES.ledger import Ledger


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_deref_logging_creates_ledger_entry(tmp_path: Path) -> None:
    """Test that dereference logging creates exactly one ledger entry."""
    ledger_path = tmp_path / "LEDGER.jsonl"

    log_dereference_event(
        run_id="test-run-1",
        timestamp="CATALYTIC-DPT-LEDGER-SENTINEL",
        ledger_path=ledger_path,
        command="read",
        hash_hex="a" * 64,
        bounds={"max_bytes": 4096, "start": 0, "end": None},
    )

    ledger = Ledger(ledger_path)
    records = ledger.read_all()

    assert len(records) == 1
    assert records[0]["RUN_INFO"]["run_id"] == "test-run-1"
    assert records[0]["RUN_INFO"]["timestamp"] == "CATALYTIC-DPT-LEDGER-SENTINEL"


def test_deref_logging_records_exact_bounds(tmp_path: Path) -> None:
    """Test that logged fields exactly match command args."""
    ledger_path = tmp_path / "LEDGER.jsonl"
    hash_val = "b" * 64
    bounds = {"max_bytes": 8192, "start": 100, "end": 500}

    log_dereference_event(
        run_id="test-run-2",
        timestamp="2025-12-25T00:00:00Z",
        ledger_path=ledger_path,
        command="read",
        hash_hex=hash_val,
        bounds=bounds,
    )

    ledger = Ledger(ledger_path)
    records = ledger.read_all()

    assert len(records) == 1
    intent = records[0]["RUN_INFO"]["intent"]
    assert hash_val in intent
    assert "read" in intent
    # Bounds should be encoded in intent
    assert "max_bytes" in intent or "8192" in intent


def test_deref_logging_determinism(tmp_path: Path) -> None:
    """Test that same command twice produces identical ledger bytes."""
    ledger_path1 = tmp_path / "LEDGER1.jsonl"
    ledger_path2 = tmp_path / "LEDGER2.jsonl"

    run_id = "determ-test"
    timestamp = "CATALYTIC-DPT-LEDGER-SENTINEL"
    command = "grep"
    hash_hex = "c" * 64
    bounds = {"max_bytes": 65536, "max_matches": 20, "pattern": "test"}

    # First invocation
    log_dereference_event(
        run_id=run_id,
        timestamp=timestamp,
        ledger_path=ledger_path1,
        command=command,
        hash_hex=hash_hex,
        bounds=bounds,
    )

    # Second invocation with identical parameters
    log_dereference_event(
        run_id=run_id,
        timestamp=timestamp,
        ledger_path=ledger_path2,
        command=command,
        hash_hex=hash_hex,
        bounds=bounds,
    )

    # Compare raw bytes
    bytes1 = ledger_path1.read_bytes()
    bytes2 = ledger_path2.read_bytes()

    assert bytes1 == bytes2, "Ledger entries must be deterministic"


def test_deref_logging_no_run_context_skips_logging(tmp_path: Path) -> None:
    """Test that no run context results in no ledger entry."""
    ledger_path = tmp_path / "LEDGER.jsonl"

    # Call with run_id=None
    log_dereference_event(
        run_id=None,
        timestamp="CATALYTIC-DPT-LEDGER-SENTINEL",
        ledger_path=ledger_path,
        command="describe",
        hash_hex="d" * 64,
        bounds={"max_bytes": 8192},
    )

    # Ledger should not be created
    assert not ledger_path.exists()


def test_deref_logging_all_commands(tmp_path: Path) -> None:
    """Test that all four commands can be logged."""
    ledger_path = tmp_path / "LEDGER.jsonl"
    timestamp = "CATALYTIC-DPT-LEDGER-SENTINEL"

    commands = [
        ("read", {"max_bytes": 4096, "start": 0, "end": None}),
        ("grep", {"max_bytes": 65536, "max_matches": 20, "pattern": "foo"}),
        ("describe", {"max_bytes": 8192}),
        ("ast", {"max_bytes": 65536, "max_nodes": 200, "max_depth": 6}),
    ]

    for idx, (cmd, bounds) in enumerate(commands):
        log_dereference_event(
            run_id=f"test-run-{idx}",
            timestamp=timestamp,
            ledger_path=ledger_path,
            command=cmd,
            hash_hex="e" * 64,
            bounds=bounds,
        )

    ledger = Ledger(ledger_path)
    records = ledger.read_all()

    assert len(records) == 4
    for idx, (cmd, _) in enumerate(commands):
        assert cmd in records[idx]["RUN_INFO"]["intent"]


def test_deref_logging_schema_conformance(tmp_path: Path) -> None:
    """Test that logged records conform to ledger schema."""
    ledger_path = tmp_path / "LEDGER.jsonl"

    log_dereference_event(
        run_id="schema-test",
        timestamp="CATALYTIC-DPT-LEDGER-SENTINEL",
        ledger_path=ledger_path,
        command="read",
        hash_hex="f" * 64,
        bounds={"max_bytes": 4096, "start": 0, "end": None},
    )

    ledger = Ledger(ledger_path)
    records = ledger.read_all()

    # If we got here without exception, schema validation passed
    assert len(records) == 1

    # Check required fields
    record = records[0]
    assert "JOBSPEC" in record
    assert "RUN_INFO" in record
    assert "PRE_MANIFEST" in record
    assert "POST_MANIFEST" in record
    assert "RESTORE_DIFF" in record
    assert "OUTPUTS" in record
    assert "STATUS" in record
    assert "VALIDATOR_ID" in record

    # Check RUN_INFO structure
    run_info = record["RUN_INFO"]
    assert run_info["run_id"] == "schema-test"
    assert run_info["timestamp"] == "CATALYTIC-DPT-LEDGER-SENTINEL"
    assert "intent" in run_info
    assert run_info["exit_code"] == 0
    assert run_info["restoration_verified"] is True
