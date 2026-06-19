import json
from pathlib import Path

import pytest

import CAPABILITY.TOOLS.utilities.run_planned_suite as planned


def payload():
    return {
        "plan_hash": "a" * 64,
        "suites": [
            {"name": "core", "command": ["python", "-c", "pass"]},
            {"name": "embeddings", "command": ["python", "-c", "pass"]},
        ],
    }


def test_load_plan_requires_nonempty_suite_list(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"suites": []}), encoding="utf-8")
    with pytest.raises(planned.PlannedSuiteError, match="non-empty suites"):
        planned.load_plan(path)


def test_suite_command_returns_exact_frozen_command():
    assert planned.suite_command(payload(), "core") == ["python", "-c", "pass"]


def test_suite_command_rejects_missing_or_duplicate_suite():
    with pytest.raises(planned.PlannedSuiteError, match="found 0"):
        planned.suite_command(payload(), "missing")

    duplicate = payload()
    duplicate["suites"].append(
        {"name": "core", "command": ["python", "-c", "pass"]}
    )
    with pytest.raises(planned.PlannedSuiteError, match="found 2"):
        planned.suite_command(duplicate, "core")


def test_suite_command_rejects_invalid_command():
    bad = {"suites": [{"name": "core", "command": "pytest"}]}
    with pytest.raises(planned.PlannedSuiteError, match="invalid command"):
        planned.suite_command(bad, "core")


def test_run_suite_returns_subprocess_exit_code(monkeypatch):
    class Result:
        returncode = 7

    monkeypatch.setattr(planned.subprocess, "run", lambda *args, **kwargs: Result())
    assert planned.run_suite(payload(), "core") == 7
