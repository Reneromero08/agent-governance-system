import json
import sys

import task_runner
from task_runner import run_task


def test_runner_records_last_assistant_text(tmp_path, monkeypatch):
    tasks_dir = tmp_path / "state" / "tasks"
    logs_dir = tmp_path / "state" / "logs"
    task_path = tasks_dir / "task.json"
    stdout_path = logs_dir / "task.jsonl"
    stderr_path = logs_dir / "task.stderr.log"
    task = {
        "task_id": "reviewer-000001",
        "worker_id": "reviewer",
        "session_id": "11111111-1111-1111-1111-111111111111",
        "status": "queued",
        "receipt_path": str(tasks_dir / "receipt.json"),
        "prompt_sha256": task_runner.sha256_text("Review auth"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    worker = {
        "worker_id": "reviewer",
        "session_id": task["session_id"],
        "session_dir": str(tmp_path / "sessions"),
        "workspace": str(tmp_path),
        "name": "reviewer",
        "tools": ["read", "grep", "find", "ls"],
        "provider": "",
        "model": "",
    }
    task_runner.atomic_write(task_path, task)
    spec_path = tasks_dir / "spec.json"
    task_runner.atomic_write(spec_path, {
        "task_path": str(task_path),
        "task_lock_path": str(tasks_dir / "task.lock"),
        "worker": worker,
        "prompt": "Review auth",
        "pi_command": sys.executable,
        "timeout": 10,
        "max_log_bytes": 1024 * 1024,
    })

    class FakeProcess:
        pid = 2468

        def __init__(self, command, **kwargs):
            assert command[command.index("--session-id") + 1] == task["session_id"]
            events = [
                {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "review complete"}]}},
                {"type": "agent_settled"},
            ]
            kwargs["stdout"].write("\n".join(json.dumps(event) for event in events) + "\n")
            kwargs["stdout"].flush()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(task_runner.subprocess, "Popen", FakeProcess)
    assert run_task(spec_path) == 0
    final = json.loads(task_path.read_text(encoding="utf-8"))
    assert final["status"] == "complete"
    assert final["result"] == "review complete"
    assert final["exit_code"] == 0
    assert final["integrity"]["integrity_ok"] is True
    receipt = json.loads((tasks_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["prompt_sha256"] == final["prompt_sha256"]


def test_cancelled_task_cannot_be_overwritten_by_finalizer(tmp_path):
    task_path = tmp_path / "task.json"
    lock_path = tmp_path / "task.lock"
    receipt_path = tmp_path / "receipt.json"
    task_runner.atomic_write(task_path, {
        "task_id": "reviewer-000001",
        "worker_id": "reviewer",
        "session_id": "11111111-1111-1111-1111-111111111111",
        "status": "cancelled",
        "exit_code": 130,
        "receipt_path": str(receipt_path),
    })
    final = task_runner.finalize_task(
        task_path,
        lock_path,
        {"status": "complete", "exit_code": 0},
        {"status": "complete"},
        receipt_path,
    )
    assert final["status"] == "cancelled"
    assert final["exit_code"] == 130
    assert not receipt_path.exists()
