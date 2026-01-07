from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.ledger import Ledger
from CAPABILITY.PRIMITIVES.write_firewall import WriteFirewall


def _record(*, job_id: str, run_id: str) -> dict:
    return {
        "JOBSPEC": {
            "job_id": job_id,
            "phase": 4,
            "task_type": "test_execution",
            "intent": "adversarial ledger test",
            "inputs": {},
            "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": [],
            "determinism": "deterministic",
        },
        "RUN_INFO": {
            "run_id": run_id,
            "timestamp": "CATALYTIC-DPT-LEDGER-SENTINEL",
            "intent": "adversarial",
            "catalytic_domains": [],
            "exit_code": 0,
            "restoration_verified": True,
        },
        "PRE_MANIFEST": {},
        "POST_MANIFEST": {},
        "RESTORE_DIFF": {},
        "OUTPUTS": [],
        "STATUS": {"status": "started", "restoration_verified": True, "exit_code": 0, "validation_passed": True},
        "VALIDATOR_ID": {"validator_semver": "0.1.0", "validator_build_id": "adversarial"},
    }


def test_ledger_rejects_malformed_json_line() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        test_root = Path(tmpdir)

        # Create test directories
        (test_root / "_tmp").mkdir(parents=True, exist_ok=True)
        (test_root / "_durable").mkdir(parents=True, exist_ok=True)

        # Configure firewall for test
        firewall = WriteFirewall(
            tmp_roots=["_tmp"],
            durable_roots=["_durable"],
            project_root=test_root,
        )

        base = test_root / "_tmp" / "ledger_malformed"
        base.mkdir(parents=True, exist_ok=True)
        path = base / "LEDGER.jsonl"

        ledger = Ledger(path)
        ledger.append(_record(job_id="job-1", run_id="run-1"))

        with open(path, "ab") as f:
            f.write(b"{not json}\n")

        assert ledger.verify_append_only() is False
        with pytest.raises(ValueError):
            _ = ledger.read_all()


def test_ledger_rejects_truncated_last_line_mid_byte() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        test_root = Path(tmpdir)

        # Create test directories
        (test_root / "_tmp").mkdir(parents=True, exist_ok=True)
        (test_root / "_durable").mkdir(parents=True, exist_ok=True)

        # Configure firewall for test
        firewall = WriteFirewall(
            tmp_roots=["_tmp"],
            durable_roots=["_durable"],
            project_root=test_root,
        )

        base = test_root / "_tmp" / "ledger_truncate"
        base.mkdir(parents=True, exist_ok=True)
        path = base / "LEDGER.jsonl"

        ledger = Ledger(path)
        ledger.append(_record(job_id="job-1", run_id="run-1"))
        ledger.append(_record(job_id="job-1", run_id="run-2"))

        data = path.read_bytes()
        assert data.endswith(b"\n")

        # Remove the trailing newline to simulate a mid-write crash on the last line.
        path.write_bytes(data[:-1])

        assert ledger.verify_append_only() is False
        with pytest.raises(ValueError):
            _ = ledger.read_all()

