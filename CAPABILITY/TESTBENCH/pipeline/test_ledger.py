import json
import sys
from pathlib import Path

import pytest

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).resolve().parents[2]
# sys.path cleanup

from CAPABILITY.PRIMITIVES.ledger import Ledger


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


def _record(*, job_id: str, run_id: str, status: str = "started") -> dict:
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


def test_append_single_record_one_line(tmp_path: Path) -> None:
    path = tmp_path / "LEDGER.jsonl"
    ledger = Ledger(path)
    ledger.append(_record(job_id="job-1", run_id="run-1"))

    data = path.read_bytes()
    assert data.count(b"\n") == 1
    obj = json.loads(data.decode("utf-8"))
    assert obj["RUN_INFO"]["run_id"] == "run-1"


def test_append_multiple_records_order_preserved(tmp_path: Path) -> None:
    path = tmp_path / "LEDGER.jsonl"
    ledger = Ledger(path)
    ledger.append(_record(job_id="job-1", run_id="run-1"))
    ledger.append(_record(job_id="job-1", run_id="run-2"))

    records = ledger.read_all()
    assert [r["RUN_INFO"]["run_id"] for r in records] == ["run-1", "run-2"]


def test_deterministic_serialization_same_bytes(tmp_path: Path) -> None:
    # Two separate ledger instances should serialize the same record bytes identically.
    r = _record(job_id="job-1", run_id="run-1")
    path1 = tmp_path / "a.jsonl"
    path2 = tmp_path / "b.jsonl"

    Ledger(path1).append(r)
    Ledger(path2).append(r)

    assert path1.read_bytes() == path2.read_bytes()


def test_schema_rejection_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "LEDGER.jsonl"
    ledger = Ledger(path)
    bad = _record(job_id="job-1", run_id="run-1")
    bad.pop("STATUS")
    with pytest.raises(ValueError):
        ledger.append(bad)


def test_append_only_enforcement_detects_truncation(tmp_path: Path) -> None:
    path = tmp_path / "LEDGER.jsonl"
    ledger = Ledger(path)
    ledger.append(_record(job_id="job-1", run_id="run-1"))

    # External truncation attempt.
    path.write_bytes(b"")

    with pytest.raises(RuntimeError):
        ledger.append(_record(job_id="job-1", run_id="run-2"))


def test_corrupt_line_detection(tmp_path: Path) -> None:
    path = tmp_path / "LEDGER.jsonl"
    ledger = Ledger(path)
    ledger.append(_record(job_id="job-1", run_id="run-1"))

    with open(path, "ab") as f:
        f.write(b"{not json}\n")

    assert ledger.verify_append_only() is False
    with pytest.raises(ValueError):
        ledger.read_all()


def test_partial_line_detection(tmp_path: Path) -> None:
    path = tmp_path / "LEDGER.jsonl"
    path.write_bytes(b"{\"a\":1}")
    ledger = Ledger(path)
    assert ledger.verify_append_only() is False
    with pytest.raises(ValueError):
        ledger.read_all()

