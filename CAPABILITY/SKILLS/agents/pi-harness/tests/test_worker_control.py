from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

import pytest

import worker_control
from worker_control import WorkerController, read_log_tail


def test_worker_identity_is_deterministic_and_persistent(tmp_path):
    ctl = WorkerController(tmp_path / "state")
    worker = ctl.create_worker("reviewer", str(tmp_path))
    reloaded = WorkerController(tmp_path / "state").get_worker("reviewer")
    assert reloaded == worker
    assert worker["session_id"]
    assert worker["tools"] == ["read", "grep", "find", "ls"]


def test_write_worker_requires_scope(tmp_path):
    ctl = WorkerController(tmp_path / "state")
    with pytest.raises(ValueError):
        ctl.create_worker("writer", str(tmp_path), allow_write=True)


def test_write_access_does_not_implicitly_enable_shell(tmp_path):
    ctl = WorkerController(tmp_path / "state")
    worker = ctl.create_worker("writer", str(tmp_path), write_roots=["src"], allow_write=True)
    assert "edit" in worker["tools"]
    assert "write" in worker["tools"]
    assert "bash" not in worker["tools"]


def test_shell_requires_explicit_native_program_allowlist(tmp_path):
    ctl = WorkerController(tmp_path / "state")
    with pytest.raises(ValueError, match="shell-program"):
        ctl.create_worker("shell", str(tmp_path), write_roots=["src"], allow_shell=True)
    worker = ctl.create_worker(
        "governed-shell",
        str(tmp_path),
        write_roots=["src"],
        allow_shell=True,
        shell_programs=[f"python={sys.executable}"],
    )
    assert worker["tools"][-1] == "bash"
    assert worker["shell_programs"] == {"python": str(Path(sys.executable).resolve())}


def test_scopes_are_absolute_and_cannot_escape_workspace(tmp_path):
    ctl = WorkerController(tmp_path / "state")
    worker = ctl.create_worker("reader", str(tmp_path), read_roots=["src"])
    assert worker["read_roots"] == [str((tmp_path / "src").resolve())]
    with pytest.raises(ValueError, match="escapes workspace"):
        ctl.create_worker("escape", str(tmp_path), read_roots=[".."])


def test_identifiers_cannot_traverse_state_directory(tmp_path):
    ctl = WorkerController(tmp_path / "state")
    with pytest.raises(ValueError, match="worker_id"):
        ctl.get_worker("../escape")
    with pytest.raises(ValueError, match="task_id"):
        ctl.get_task("../../escape")


def test_log_reads_are_capped_and_report_truncation(tmp_path):
    log = tmp_path / "worker.log"
    log.write_bytes(b"0123456789")
    text, truncated, size = read_log_tail(log, max_bytes=4)
    assert text == "6789"
    assert truncated is True
    assert size == 10


def test_submit_is_background_and_sequence_is_stable(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 4321

    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(worker_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(worker_control, "process_is_alive", lambda _: True)
    ctl = WorkerController(tmp_path / "state", pi_command=sys.executable)
    ctl.create_worker("reviewer", str(tmp_path))
    task = ctl.submit_task("reviewer", "Review auth")
    assert task["task_id"] == "reviewer-000001"
    assert task["status"] == "queued"
    assert task["pid"] == 4321
    assert calls
    with pytest.raises(ValueError, match="busy"):
        ctl.submit_task("reviewer", "Second task")


def test_concurrent_submission_allocates_only_one_task(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(worker_control.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(worker_control, "process_is_alive", lambda _: True)
    ctl = WorkerController(tmp_path / "state", pi_command=sys.executable)
    ctl.create_worker("reviewer", str(tmp_path))

    def submit():
        try:
            return ctl.submit_task("reviewer", "Review auth")["task_id"]
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: submit(), range(2)))
    assert outcomes.count("reviewer-000001") == 1
    assert sum("busy" in outcome for outcome in outcomes) == 1


def test_launch_failure_is_recorded_and_releases_worker(tmp_path, monkeypatch):
    def fail_launch(*args, **kwargs):
        raise OSError("missing runner")

    monkeypatch.setattr(worker_control.subprocess, "Popen", fail_launch)
    ctl = WorkerController(tmp_path / "state", pi_command=sys.executable)
    ctl.create_worker("reviewer", str(tmp_path))
    task = ctl.submit_task("reviewer", "Review auth")
    assert task["status"] == "failed"
    assert task["exit_code"] == 127
    assert ctl.get_worker("reviewer")["active_task_id"] == ""


def test_next_submit_recovers_dead_background_runner(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(worker_control.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(worker_control, "process_is_alive", lambda _: False)
    ctl = WorkerController(tmp_path / "state", pi_command=sys.executable)
    ctl.create_worker("reviewer", str(tmp_path))
    first = ctl.submit_task("reviewer", "First task")
    second = ctl.submit_task("reviewer", "Second task")
    assert first["task_id"] == "reviewer-000001"
    assert ctl.get_task(first["task_id"])["status"] == "failed"
    assert second["task_id"] == "reviewer-000002"


def test_completed_task_releases_worker(tmp_path):
    ctl = WorkerController(tmp_path / "state")
    worker = ctl.create_worker("reviewer", str(tmp_path))
    task = {
        "task_id": "reviewer-000001",
        "worker_id": "reviewer",
        "session_id": worker["session_id"],
        "status": "complete",
        "stdout_path": str(tmp_path / "out"),
        "stderr_path": str(tmp_path / "err"),
    }
    worker["active_task_id"] = task["task_id"]
    worker_control.atomic_write(ctl._worker_path("reviewer"), worker)
    worker_control.atomic_write(ctl._task_path(task["task_id"]), task)
    ctl.task_status(task["task_id"])
    assert ctl.get_worker("reviewer")["active_task_id"] == ""
