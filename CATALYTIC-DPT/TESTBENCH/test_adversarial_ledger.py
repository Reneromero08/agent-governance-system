from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.ledger import Ledger


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


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
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "ledger_malformed"
    _rm(base)
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
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "ledger_truncate"
    _rm(base)
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

