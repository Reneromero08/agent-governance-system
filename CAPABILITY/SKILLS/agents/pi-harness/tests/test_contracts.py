import json
import sys
from pathlib import Path

import pytest

from pi_harness import build_pi_command, build_task_packet, extract_last_assistant_text, inspect_jsonl, resolve_executable


def test_packet_is_scope_locked_and_forbids_git_actions(tmp_path):
    prompt = build_task_packet(
        "Review auth.", str(tmp_path), ["src"], [], ["read", "grep", "find", "ls"]
    )
    assert "STRICT SCOPE LOCK" in prompt
    assert "WRITE_SCOPE: NONE" in prompt
    assert "Do not commit, push, publish, or release" in prompt


def test_write_tools_require_write_scope(tmp_path):
    with pytest.raises(ValueError):
        build_task_packet("Fix auth.", str(tmp_path), ["src"], [], ["read", "edit"])


def test_pi_command_reuses_exact_session_id(tmp_path):
    worker = {
        "worker_id": "reviewer",
        "session_id": "11111111-1111-1111-1111-111111111111",
        "session_dir": str(tmp_path / "sessions"),
        "name": "reviewer",
        "tools": ["read", "grep"],
        "provider": "openai",
        "model": "gpt-5",
    }
    command = build_pi_command(worker, "hello", sys.executable)
    assert command[0] == str(Path(sys.executable).resolve())
    assert command[command.index("--session-id") + 1] == worker["session_id"]
    assert "--no-extensions" in command
    assert command[-1] == "hello"


def test_extracts_last_assistant_message():
    events = [
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "first"}]}},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "final"}]}},
    ]
    raw = "\n".join(json.dumps(event) for event in events)
    assert extract_last_assistant_text(raw) == "final"


def test_integrity_rejects_out_of_scope_writes_and_malformed_json(tmp_path):
    events = [
        {"type": "tool_execution_start", "toolName": "write", "args": {"path": "../escape.txt"}},
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "done"}]}},
        {"type": "agent_settled"},
    ]
    raw = "not-json\n" + "\n".join(json.dumps(event) for event in events)
    audit = inspect_jsonl(raw, str(tmp_path), [str(tmp_path / "allowed")])
    assert audit["integrity_ok"] is False
    assert audit["malformed_jsonl_lines"] == 1
    assert audit["scope_violations"] == [str((tmp_path.parent / "escape.txt").resolve())]


def test_integrity_accepts_settled_scoped_write(tmp_path):
    allowed = tmp_path / "allowed"
    events = [
        {"type": "tool_execution_start", "toolName": "write", "args": {"path": "allowed/file.txt"}},
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "done"}]}},
        {"type": "agent_settled"},
    ]
    audit = inspect_jsonl("\n".join(json.dumps(event) for event in events), str(tmp_path), [str(allowed)])
    assert audit["integrity_ok"] is True
    assert audit["write_paths"] == [str((allowed / "file.txt").resolve())]


def test_integrity_rejects_non_allowlisted_shell_program(tmp_path):
    events = [
        {"type": "tool_execution_start", "toolName": "bash", "args": {"program": "powershell", "args": [], "cwd": "."}},
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "done"}]}},
        {"type": "agent_settled"},
    ]
    audit = inspect_jsonl(
        "\n".join(json.dumps(event) for event in events),
        str(tmp_path),
        [str(tmp_path)],
        shell_programs={"git": "git.exe"},
    )
    assert audit["integrity_ok"] is False
    assert audit["shell_policy_violations"] == ["bash:program-not-allowlisted"]


def test_missing_pi_executable_fails_closed(monkeypatch):
    monkeypatch.setattr("pi_harness.shutil.which", lambda _: None)
    with pytest.raises(ValueError, match="not found on PATH"):
        resolve_executable("definitely-missing-pi")
