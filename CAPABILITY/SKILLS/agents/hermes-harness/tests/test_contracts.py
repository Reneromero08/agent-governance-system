import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hermes_harness.py"


def run_cmd(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], text=True, capture_output=True)


def test_validate_example_audit():
    task_file = ROOT / "examples" / "task.audit.json"
    result = run_cmd("validate", "--task-file", str(task_file))
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(result.stdout)
    assert data["ok"] is True


def test_prompt_contains_context_contract():
    result = run_cmd("prompt", "--task", "Audit repo", "--workspace", ".", "--mode", "audit")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "PARENT ROLE" in out
    assert "MAX CONCURRENT SUBAGENTS" in out
    assert "Subagents know nothing" in out


def test_invalid_mode_rejected():
    result = run_cmd("validate", "--task", "x", "--mode", "bad")
    assert result.returncode != 0
