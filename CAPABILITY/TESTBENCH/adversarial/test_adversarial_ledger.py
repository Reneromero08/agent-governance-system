from pathlib import Path
import sys
import shutil
import pytest
import json

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.ledger import Ledger

def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

def _minimal_jobspec(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "phase": 1,
        "task_type": "primitive_implementation",
        "intent": "ledger test",
        "inputs": {},
        "outputs": {"durable_paths": [], "validation_criteria": {}},
        "catalytic_domains": [],
        "determinism": "deterministic",
    }

def _record(*, job_id: str, run_id: str, status: str = "completed") -> dict:
    return {
        "JOBSPEC": _minimal_jobspec(job_id),
        "RUN_INFO": {
            "run_id": run_id,
            "timestamp": "CATALYTIC-DPT-LEDGER-SENTINEL",
            "intent": "test",
            "catalytic_domains": [],
            "exit_code": 0,
            "restoration_verified": True,
        },
        "PRE_MANIFEST": {},
        "POST_MANIFEST": {},
        "RESTORE_DIFF": {},
        "OUTPUTS": [],
        "STATUS": {"status": status, "restoration_verified": True, "exit_code": 0, "validation_passed": True},
        "VALIDATOR_ID": {"validator_semver": "1.0.0", "validator_build_id": "test"},
    }

def test_ledger_rejects_malformed_json_line(tmp_path: Path):
    path = tmp_path / "LEDGER.jsonl"
    ledger = Ledger(path)
    ledger.append(_record(job_id="job-1", run_id="run-1"))

    with open(path, "ab") as f:
        f.write(b"{not json}\n")

    assert ledger.verify_append_only() is False
    with pytest.raises(ValueError):
        _ = ledger.read_all()

def test_ledger_rejects_truncated_last_line_mid_byte(tmp_path: Path):
    path = tmp_path / "LEDGER.jsonl"
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
