import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hermes_harness.py"
VALIDATE = ROOT / "validate.py"


def run_cmd(*args, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True, capture_output=True, timeout=10,
        env={**os.environ, **(env or {})} if env else None,
    )


def run_validate(actual_text, expected_json):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as a:
        a.write(actual_text)
        actual_path = a.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as e:
        json.dump(expected_json, e)
        expected_path = e.name
    return subprocess.run(
        [sys.executable, str(VALIDATE), actual_path, expected_path],
        text=True, capture_output=True, timeout=10,
    )


# ---- basic contract ----

def test_validate_example_audit():
    task_file = ROOT / "examples" / "task.audit.json"
    result = run_cmd("validate", "--task-file", str(task_file))
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(result.stdout)
    assert data["ok"] is True


def test_prompt_contains_task():
    result = run_cmd("prompt", "--task", "Audit repo", "--workspace", ".", "--mode", "audit")
    assert result.returncode == 0, result.stderr
    assert "Audit repo" in result.stdout


def test_invalid_mode_rejected():
    result = run_cmd("validate", "--task", "x", "--mode", "bad")
    assert result.returncode != 0


# ---- persistent_worker mode ----

def test_persistent_worker_mode_valid():
    result = run_cmd("validate", "--task", "Verify", "--mode", "persistent_worker")
    assert result.returncode == 0


def test_persistent_worker_prompt_forbids_delegate():
    result = run_cmd("prompt", "--task", "Do phase 5", "--mode", "persistent_worker", "--conversation", "ccc:worker")
    assert result.returncode == 0
    assert "Do NOT spawn delegate_task" in result.stdout


def test_orchestrator_prompt_allows_delegate():
    result = run_cmd("prompt", "--task", "Audit repo", "--mode", "audit", "--max-workers", "3")
    assert result.returncode == 0
    assert "delegate_task with up to 3 workers" in result.stdout


# ---- golden fixtures ----

def test_golden_fixture_prompt_audit():
    expected = json.loads((ROOT / "fixtures" / "prompt_audit" / "expected.json").read_text())
    task_file = ROOT / "fixtures" / "prompt_audit" / "input.json"
    result = run_cmd("prompt", "--task-file", str(task_file))
    assert result.returncode == 0
    vr = run_validate(result.stdout, expected)
    assert vr.returncode == 0, f"validate failed: {vr.stderr}\nPrompt:\n{result.stdout}"


def test_golden_fixture_prompt_worker():
    expected = json.loads((ROOT / "fixtures" / "prompt_worker" / "expected.json").read_text())
    task_file = ROOT / "fixtures" / "prompt_worker" / "input.json"
    result = run_cmd("prompt", "--task-file", str(task_file))
    assert result.returncode == 0
    vr = run_validate(result.stdout, expected)
    assert vr.returncode == 0, f"validate failed: {vr.stderr}\nPrompt:\n{result.stdout}"


# ---- dry-run ----

def test_dry_run_does_not_call_api():
    result = run_cmd("run", "--task", "Test", "--dry-run")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["endpoint"].endswith("/responses")
    assert "prompt" in data


def test_dry_run_respects_base_url():
    result = run_cmd("run", "--task", "Test", "--dry-run", "--base-url", "http://example.com:9999/v1")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["endpoint"] == "http://example.com:9999/v1/responses"


def test_dry_run_respects_model():
    result = run_cmd("run", "--task", "Test", "--dry-run", "--model", "my-custom-model")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["model"] == "my-custom-model"


def test_dry_run_respects_session_key():
    result = run_cmd("run", "--task", "Test", "--dry-run", "--session-key", "agent:test")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["session_key"] == "agent:test"


# ---- task-file output mapping ----

def test_task_file_output_maps_to_contract():
    task_file = ROOT / "examples" / "task.audit.json"
    result = run_cmd("prompt", "--task-file", str(task_file))
    assert result.returncode == 0
    assert "Markdown report" in result.stdout


# ---- conversation_new validation ----

def test_conversation_new_without_name_rejected():
    result = run_cmd("validate", "--task", "Test", "--conversation-new")
    assert result.returncode != 0, "conversation_new without conversation should fail validate"


def test_dry_run_with_conversation_new_no_name_fails():
    result = run_cmd("run", "--task", "Test", "--conversation-new", "--dry-run")
    assert result.returncode != 0, "dry-run with conversation_new but no conversation should fail"


def test_dry_run_respects_env_base_url():
    result = run_cmd("run", "--task", "Test", "--dry-run", env={"HERMES_API_BASE": "http://env:9999/v1"})
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["endpoint"] == "http://env:9999/v1/responses"


def test_dry_run_respects_env_model():
    result = run_cmd("run", "--task", "Test", "--dry-run", env={"HERMES_MODEL": "env-model"})
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["model"] == "env-model"


# ---- conversation ----

def test_conversation_in_task():
    result = run_cmd("validate", "--task", "Test", "--conversation", "my-conv")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["task"].get("conversation") == "my-conv"


# ---- session_key ----

def test_session_key_in_task():
    result = run_cmd("validate", "--task", "Test", "--session-key", "agent:ags:test")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["task"].get("session_key") == "agent:ags:test"


# ---- X-Hermes-Session-Key header (monkeypatch) ----

def test_session_key_header_sent(monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import urllib.request
    import hermes_harness

    captured_headers = {}

    class FakeResponse:
        def read(self):
            return json.dumps({
                "id": "resp_test", "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}
                ], "usage": {}
            }).encode()

        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_open(req, **kw):
        captured_headers["X-Hermes-Session-Key"] = req.headers.get("X-hermes-session-key", "")
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    result = hermes_harness.call_hermes_responses("test", session_key="agent:ags:test")
    assert captured_headers["X-Hermes-Session-Key"] == "agent:ags:test"
    assert result["text"] == "ok"


# ---- run_task returns raw and response_id (monkeypatch) ----

def test_run_task_returns_raw(monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import hermes_harness

    fake_raw = {"id": "resp_abc", "output": [], "usage": {"input_tokens": 5}}

    def fake_call(**kwargs):
        return {"text": "ok", "response_id": "resp_abc", "usage": {"input_tokens": 5}, "raw": fake_raw}

    monkeypatch.setattr(hermes_harness, "call_hermes_responses", fake_call)
    result = hermes_harness.run_task(task="Test")
    assert result["ok"] is True
    assert result["response_id"] == "resp_abc"
    assert result["usage"]["input_tokens"] == 5
    assert result["raw"] == fake_raw


# ---- max_workers 0 uses worker prompt ----

def test_zero_workers_uses_worker_prompt():
    result = run_cmd("prompt", "--task", "Hello", "--max-workers", "0")
    assert result.returncode == 0
    assert "Do NOT spawn delegate_task" in result.stdout


# ---- validate output fields ----

def test_validate_output_fields():
    task_file = ROOT / "examples" / "task.audit.json"
    result = run_cmd("validate", "--task-file", str(task_file))
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["ok"] is True
    assert data["task"]["mode"] == "audit"


# ---- prompt-level scope locking ----

def test_verify_mode_valid():
    result = run_cmd("validate", "--task", "Harden the thing", "--mode", "persistent_worker_verify")
    assert result.returncode == 0


def test_verify_mode_injects_scope_lock():
    result = run_cmd("prompt", "--task", "Harden results", "--mode", "persistent_worker_verify",
                     "--write-root", "PHASE5_8", "--read-root", "PHASE5_8",
                     "--search-policy", "artifact_only", "--branch-policy", "forbidden")
    assert result.returncode == 0
    out = result.stdout
    assert "STRICT SCOPE LOCK" in out
    assert "WRITE_SCOPE" in out
    assert "READ_SCOPE" in out
    assert "artifact_only" in out
    assert "forbidden" in out
    assert "PHASE5_8" in out


def test_verify_mode_no_scope_lock_in_orchestrator():
    result = run_cmd("prompt", "--task", "Audit repo", "--mode", "audit")
    assert result.returncode == 0
    assert "STRICT SCOPE LOCK" not in result.stdout


def test_search_policy_validation():
    result = run_cmd("validate", "--task", "Test", "--mode", "persistent_worker_verify",
                     "--search-policy", "bad_policy")
    assert result.returncode != 0


def test_branch_policy_validation():
    result = run_cmd("validate", "--task", "Test", "--mode", "persistent_worker_verify",
                     "--branch-policy", "bad_policy")
    assert result.returncode != 0


def test_write_root_in_task():
    result = run_cmd("validate", "--task", "Test", "--mode", "persistent_worker_verify",
                     "--write-root", "foo,bar/baz")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["task"]["write_root"] == ["foo", "bar/baz"]


def test_dry_run_includes_scope():
    result = run_cmd("run", "--task", "Harden", "--mode", "persistent_worker_verify",
                     "--write-root", "PHASE5_8", "--read-root", "PHASE5_8",
                     "--search-policy", "artifact_only", "--dry-run")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "STRICT SCOPE LOCK" in data["prompt"]


def test_scope_lock_forbids_unrelated_issues():
    result = run_cmd("prompt", "--task", "Harden results", "--mode", "persistent_worker_verify",
                     "--write-root", "PHASE5_8", "--read-root", "PHASE5_8")
    assert result.returncode == 0
    out = result.stdout
    assert "unrelated issue outside scope, ignore it" in out


def test_scope_lock_forbids_future_proposals():
    result = run_cmd("prompt", "--task", "Harden results", "--mode", "persistent_worker_verify",
                     "--write-root", "PHASE5_8")
    assert result.returncode == 0
    out = result.stdout
    assert "must not create future-goal proposals" in out


def test_scope_lock_forbids_branches():
    result = run_cmd("prompt", "--task", "Harden results", "--mode", "persistent_worker_verify",
                     "--write-root", "PHASE5_8", "--branch-policy", "forbidden")
    assert result.returncode == 0
    out = result.stdout
    assert "must not create branches" in out.lower()


def test_scope_lock_forbids_external_mutation():
    result = run_cmd("prompt", "--task", "Harden results", "--mode", "persistent_worker_verify",
                     "--write-root", "PHASE5_8")
    assert result.returncode == 0
    out = result.stdout
    assert "must not modify files outside" in out.lower()
