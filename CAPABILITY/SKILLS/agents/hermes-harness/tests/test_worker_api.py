"""Contract tests for the Worker API control plane.

These tests drive the goal loop with a scripted fake caller so they never need
a live Hermes server. They prove:

    - worker registry loads/saves
    - the task packet always carries the STRICT SCOPE LOCK
    - the goal loop stops on GOAL_COMPLETE / GOAL_BLOCKED / max_turns
    - continuation preserves the worker's named conversation
    - native/fake Hermes /goal is never used
    - session_id, conversation, and session_key remain distinct
    - task logs are written
    - an artifact manifest is emitted
    - out-of-scope language is forbidden in the prompt contract
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import worker_control  # noqa: E402
from worker_control import WorkerController  # noqa: E402
from worker_api import WorkerAPI  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class ScriptedCaller:
    """Returns queued outputs and records every call for assertions."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []  # list of dicts: prompt, conversation, session_key, model

    def __call__(self, prompt, conversation="", session_key="", model="", history=None):
        self.calls.append({
            "prompt": prompt, "conversation": conversation,
            "session_key": session_key, "model": model,
            "history_len": len(history) if history is not None else None,
        })
        text = self._outputs.pop(0) if self._outputs else "(no more scripted output)"
        return {"text": text, "response_id": f"resp_{len(self.calls)}", "usage": {"input_tokens": 1}}


def _marker_judge(goal, response):
    """Test judge: done/blocked driven by markers in the scripted output, so the
    existing marker-based scenarios exercise the real judge code path."""
    return {
        "done": "GOAL_COMPLETE: true" in response,
        "blocked": "GOAL_BLOCKED: true" in response,
        "reason": "marker-driven test judge",
    }


def _ctl(tmp_path, outputs, judge=_marker_judge):
    return WorkerController(state_dir=tmp_path / "_state",
                           caller=ScriptedCaller(outputs), judge=judge)


def _make_worker(ctl, **kw):
    defaults = dict(
        worker_id="catcas-auditor",
        role="auditor",
        conversation="ccc:ags:catcas-auditor",
        session_key="agent:ags:catcas",
        workspace="/tmp/fake-repo",
        write_roots=["sandbox/test-scope"],
        read_roots=["sandbox/test-scope"],
    )
    defaults.update(kw)
    return ctl.create_worker(**defaults)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_load_save(tmp_path):
    ctl = _ctl(tmp_path, [])
    w = _make_worker(ctl)
    assert w["worker_id"] == "catcas-auditor"
    # Reload via a fresh controller -> persisted to disk.
    ctl2 = WorkerController(state_dir=tmp_path / "_state")
    loaded = ctl2.get_worker("catcas-auditor")
    assert loaded is not None
    assert loaded["conversation"] == "ccc:ags:catcas-auditor"
    assert "catcas-auditor" in [x["worker_id"] for x in ctl2.list_workers()]


def test_duplicate_worker_rejected(tmp_path):
    ctl = _ctl(tmp_path, [])
    _make_worker(ctl)
    with pytest.raises(ValueError):
        _make_worker(ctl)


def test_default_transports(tmp_path):
    """Persistent reasoning lane (responses) is the default; runs is the opt-in
    execution lane. Persistent worker requires a stable conversation."""
    ctl = _ctl(tmp_path, [])
    w = _make_worker(ctl)
    assert w["persistent_transport"] == "responses"   # canonical memory lane
    assert w["execution_transport"] == "runs"          # opt-in execution lane
    assert w["conversation"]                           # stable conversation present
    assert w["session_key"]                            # memory scope present
    assert w["history"] == []                          # fallback cache, empty


def test_persistent_worker_requires_conversation(tmp_path):
    """Rule 2: every Hermes persistent worker must have a conversation."""
    ctl = _ctl(tmp_path, [])
    w = _make_worker(ctl, conversation="ccc:ags:stable")
    assert w["conversation"] == "ccc:ags:stable"


def test_invalid_transport_rejected(tmp_path):
    ctl = _ctl(tmp_path, [])
    with pytest.raises(ValueError):
        _make_worker(ctl, persistent_transport="carrier-pigeon")
    with pytest.raises(ValueError):
        _make_worker(ctl, worker_id="w2", execution_transport="rockets")


def test_persistent_lane_does_not_thread_client_history(tmp_path):
    """Goal loop default = persistent (responses) lane: continuity is SERVER-side
    via the named conversation. Client-side conversation_history is NOT canonical
    and is NOT replayed."""
    ctl = _ctl(tmp_path, ["turn 1", "turn 2", "done GOAL_COMPLETE: true"])
    _make_worker(ctl)  # default persistent_transport=responses
    ctl.submit_task("catcas-auditor", "Multi-turn.", max_turns=4)
    calls = ctl._caller.calls
    # Every turn: history is None (server-side conversation is the memory).
    assert all(c["history_len"] is None for c in calls)
    # Every turn hit the SAME persistent conversation.
    assert {c["conversation"] for c in calls} == {"ccc:ags:catcas-auditor"}
    # No canonical client-side history accumulated on the worker.
    assert ctl.get_worker("catcas-auditor")["history"] == []


def test_default_lane_is_persistent_not_runs(tmp_path):
    """Req 6: the goal loop does NOT use /v1/runs as the default memory path."""
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "x")
    assert rec["execution_required"] is False
    assert rec["lane"] == "persistent"
    assert rec["persistent_transport"] == "responses"
    assert all(c["history_len"] is None for c in ctl._caller.calls)  # server-side memory


def test_execution_lane_summarized_back_to_persistent(tmp_path):
    """Req 7: an execution-lane (/v1/runs) run is summarized BACK into the
    worker's persistent conversation; runs never replace persistent memory."""
    ctl = _ctl(tmp_path, ["ran the tests, all pass GOAL_COMPLETE: true", "(summary ack)"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "run the tests", execution_required=True, max_turns=2)
    assert rec["execution_required"] is True
    assert rec["lane"] == "execution"
    assert rec.get("execution_summarized_to_conversation") is True
    summary_calls = [c for c in ctl._caller.calls if "[EXECUTION SUMMARY]" in c["prompt"]]
    assert len(summary_calls) == 1
    # The summary went to the SAME persistent conversation + session_key.
    assert summary_calls[0]["conversation"] == "ccc:ags:catcas-auditor"
    assert summary_calls[0]["session_key"] == "agent:ags:catcas"


def test_goal_loop_sends_session_key_every_turn(tmp_path):
    ctl = _ctl(tmp_path, ["a", "b", "c GOAL_COMPLETE: true"])
    _make_worker(ctl, session_key="agent:ags:catcas")
    ctl.submit_task("catcas-auditor", "x", max_turns=4)
    assert {c["session_key"] for c in ctl._caller.calls} == {"agent:ags:catcas"}


def test_distinct_identity_fields(tmp_path):
    """worker_id, conversation, session_key, session_id are DISTINCT."""
    ctl = _ctl(tmp_path, [])
    w = _make_worker(ctl, conversation="conv-A", session_key="memkey-B")
    assert w["conversation"] == "conv-A"
    assert w["session_key"] == "memkey-B"
    assert w["conversation"] != w["session_key"]
    # session_id field exists but is empty by default (only used by session_chat),
    # and is never the conversation.
    assert w.get("session_id", "") == ""
    assert w["conversation"] != w.get("session_id", "")


def test_session_chat_requires_session_id_not_conversation(tmp_path):
    """Rule 4/5: session_id is not a substitute for conversation."""
    ctl = _ctl(tmp_path, [])
    # session_chat persistent transport requires a session_id.
    with pytest.raises(ValueError):
        _make_worker(ctl, persistent_transport="session_chat")
    w = _make_worker(ctl, worker_id="sc", persistent_transport="session_chat",
                     session_id="sess-123", conversation="conv-X")
    assert w["session_id"] == "sess-123"
    assert w["conversation"] == "conv-X"
    assert w["session_id"] != w["conversation"]


# ---------------------------------------------------------------------------
# Scope lock in the packet
# ---------------------------------------------------------------------------

def test_task_packet_includes_scope_lock(tmp_path):
    ctl = _ctl(tmp_path, ["working... GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "Harden the phaseX results.")
    prompt = ctl._caller.calls[0]["prompt"]
    assert "STRICT SCOPE LOCK" in prompt
    assert "WRITE_SCOPE" in prompt
    assert "READ_SCOPE" in prompt
    # Paths render absolute (and OS-native separators), so match the leaf.
    assert "test-scope" in prompt


def test_packet_forbids_out_of_scope_language(tmp_path):
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "Verify integrity.")
    prompt = ctl._caller.calls[0]["prompt"].lower()
    assert "must not modify files outside" in prompt
    assert "must not create branches" in prompt
    assert "must not create future-goal proposals" in prompt
    assert "unrelated issue outside scope, ignore it" in prompt


def test_judge_mode_packet_has_no_marker_contract(tmp_path):
    # judge completion_mode: no GOAL_COMPLETE marker contract; aux judge decides.
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "Do it.", completion_mode="judge",
                    acceptance_criteria="All checks pass.")
    prompt = ctl._caller.calls[0]["prompt"]
    assert "AUTONOMOUS GOAL LOOP" in prompt
    assert "ARTIFACT_MANIFEST" in prompt
    assert "All checks pass." in prompt
    assert "GOAL_COMPLETE: true" not in prompt  # judge decides, no marker


def test_marker_mode_packet_has_marker_contract(tmp_path):
    # use_judge=False: legacy marker contract is present.
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "Do it.", use_judge=False, acceptance_criteria="All checks pass.")
    prompt = ctl._caller.calls[0]["prompt"]
    assert "GOAL LOOP CONTRACT" in prompt
    assert "GOAL_COMPLETE: true" in prompt
    assert "GOAL_BLOCKED: true" in prompt


# ---------------------------------------------------------------------------
# Goal loop stopping conditions
# ---------------------------------------------------------------------------

def test_loop_stops_on_complete(tmp_path):
    ctl = _ctl(tmp_path, ["step 1 done", "step 2 done", "all done GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Multi-step task.", max_turns=6)
    assert rec["status"] == "complete"
    assert len(rec["turns"]) == 3


def test_loop_stops_on_blocked(tmp_path):
    ctl = _ctl(tmp_path, ["cannot find dep GOAL_BLOCKED: true\nreason: missing file"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Blocked task.", max_turns=6)
    assert rec["status"] == "blocked"
    assert len(rec["turns"]) == 1


def test_loop_stops_at_max_turns(tmp_path):
    ctl = _ctl(tmp_path, ["more", "more", "more", "more", "more"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Never finishes.", max_turns=3)
    assert rec["status"] == "budget_exhausted"
    assert len(rec["turns"]) == 3


def test_single_shot_no_goal_loop(tmp_path):
    ctl = _ctl(tmp_path, ["one and done, no marker"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Single shot.", goal_loop=False, max_turns=9)
    assert rec["status"] == "complete"
    assert len(rec["turns"]) == 1


def test_submit_rejects_busy_worker(tmp_path):
    # A worker already running/awaiting must reject a concurrent submit.
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    w = _make_worker(ctl)
    w["status"] = "running"
    ctl.save_worker(w)
    with pytest.raises(ValueError):
        ctl.submit_task("catcas-auditor", "x")
    # awaiting_judgment is also busy.
    w["status"] = "awaiting_judgment"
    ctl.save_worker(w)
    with pytest.raises(ValueError):
        ctl.submit_task("catcas-auditor", "x")


def test_lock_blocks_concurrent_claim(tmp_path):
    # A held (fresh) lock makes a concurrent submit reject atomically.
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    lockp = ctl.workers_dir / "catcas-auditor.lock"
    lockp.write_text("2099-01-01T00:00:00Z", encoding="utf-8")  # far-future = held
    with pytest.raises(ValueError):
        ctl.submit_task("catcas-auditor", "x")
    assert lockp.exists()  # someone else's lock not deleted


def test_lock_steals_stale(tmp_path):
    # A stale lock (crashed holder) is reclaimed so the worker isn't wedged.
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    lockp = ctl.workers_dir / "catcas-auditor.lock"
    lockp.write_text("2000-01-01T00:00:00Z", encoding="utf-8")  # ancient = stale
    rec = ctl.submit_task("catcas-auditor", "x")
    assert rec["status"] == "complete"
    assert not lockp.exists()  # released after the run


def test_lock_true_concurrency(tmp_path):
    # Two real threads submit to the same worker at once: exactly one runs.
    import threading, time

    class SlowCaller:
        def __call__(self, prompt, conversation="", session_key="", model="", history=None):
            time.sleep(0.3)  # hold the worker so the threads genuinely overlap
            return {"text": "GOAL_COMPLETE: true", "usage": {}, "response_id": "r"}

    ctl = WorkerController(state_dir=tmp_path / "_state", caller=SlowCaller(), judge=_marker_judge)
    _make_worker(ctl)
    results = []

    def run():
        try:
            results.append(("ok", ctl.submit_task("catcas-auditor", "x")["status"]))
        except Exception as e:  # noqa: BLE001
            results.append(("err", str(e)))

    t1, t2 = threading.Thread(target=run), threading.Thread(target=run)
    t1.start(); t2.start(); t1.join(); t2.join()
    oks = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    assert len(oks) == 1, results        # exactly one ran
    assert len(errs) == 1, results       # the other was rejected
    assert "busy" in errs[0][1].lower()


def test_loop_handles_backend_error(tmp_path):
    class Boom:
        def __call__(self, *a, **k):  # accepts history kwarg too
            raise RuntimeError("backend down")

    ctl = WorkerController(state_dir=tmp_path / "_state", caller=Boom())
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Will error.")
    assert rec["status"] == "error"
    assert "backend down" in rec.get("error", "")


# ---------------------------------------------------------------------------
# External goal verification (harness-owned completion)
# ---------------------------------------------------------------------------

def _exit_script(tmp_path, name, code):
    p = tmp_path / name
    p.write_text(f"import sys; sys.exit({code})\n", encoding="utf-8")
    return f'"{sys.executable}" "{p}"'


def test_verify_pass_completes(tmp_path):
    ctl = _ctl(tmp_path, ["done GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task(
        "catcas-auditor", "x", max_turns=3,
        verify_command=_exit_script(tmp_path, "ok.py", 0), verify_cwd=str(tmp_path),
    )
    assert rec["status"] == "complete"
    assert len(rec["verifications"]) == 1 and rec["verifications"][0]["passed"] is True
    assert rec["artifact_manifest"]["harness_verified"] is True


def test_verify_fail_rejects_and_exhausts(tmp_path):
    # Agent keeps claiming done; external check keeps failing -> never completes.
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"] * 3)
    _make_worker(ctl)
    rec = ctl.submit_task(
        "catcas-auditor", "x", max_turns=3,
        verify_command=_exit_script(tmp_path, "bad.py", 1), verify_cwd=str(tmp_path),
    )
    assert rec["status"] == "budget_exhausted"
    assert len(rec["verifications"]) == 3
    assert all(v["passed"] is False for v in rec["verifications"])
    assert rec["artifact_manifest"]["harness_verified"] is False
    # The agent was told its GOAL_COMPLETE was rejected.
    assert any("VERIFICATION REJECTED" in c["prompt"] for c in ctl._caller.calls[1:])


def test_verify_reject_then_pass(tmp_path):
    counter = tmp_path / "n.txt"
    script = tmp_path / "vscript.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"c=Path(r'{counter}')\n"
        "n=(int(c.read_text()) if c.exists() else 0)+1\n"
        "c.write_text(str(n))\n"
        "sys.exit(0 if n>=2 else 1)\n", encoding="utf-8")
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"] * 3)
    _make_worker(ctl)
    rec = ctl.submit_task(
        "catcas-auditor", "x", max_turns=4,
        verify_command=f'"{sys.executable}" "{script}"', verify_cwd=str(tmp_path),
    )
    assert rec["status"] == "complete"
    assert len(rec["verifications"]) == 2
    assert rec["verifications"][0]["passed"] is False
    assert rec["verifications"][1]["passed"] is True
    assert rec["artifact_manifest"]["harness_verified"] is True


def test_no_verify_command_is_self_declared(tmp_path):
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "x")
    assert rec["status"] == "complete"
    # No external check ran -> completion is self-declared (harness_verified None).
    assert rec["artifact_manifest"]["harness_verified"] is None


# ---------------------------------------------------------------------------
# Run transport (POST /v1/runs + SSE events + approval) — mocked, no network
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, lines=None, body=None):
        self._lines = lines or []
        self._body = body or b"{}"

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _run_transport_harness(monkeypatch, event_lines):
    """Wire a fake Hermes run API; return (posted_approvals list)."""
    import hermes_run_transport as hrt

    # Make the approval thread run synchronously for deterministic assertions.
    class SyncThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._t, self._a = target, args

        def start(self):
            self._t(*self._a)

    monkeypatch.setattr(hrt.threading, "Thread", SyncThread)
    posted = []

    def fake_urlopen(req, timeout=None):
        url, method = req.full_url, req.get_method()
        if url.endswith("/runs") and method == "POST":
            return _FakeResp(body=json.dumps({"run_id": "run_x", "status": "started"}).encode())
        if url.endswith("/runs/run_x/events"):
            return _FakeResp(lines=event_lines)
        if "/approval" in url and method == "POST":
            posted.append(json.loads(req.data.decode()))
            return _FakeResp(body=json.dumps({"resolved": 1}).encode())
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(hrt.urllib.request, "urlopen", fake_urlopen)
    return posted


def test_call_hermes_run_approval_and_output(monkeypatch):
    import hermes_run_transport as hrt
    lines = [
        b'data: {"event":"approval.request","run_id":"run_x"}\n',
        b'data: {"event":"message.delta","delta":"hello "}\n',
        b'data: {"event":"message.delta","delta":"world"}\n',
        b'data: {"event":"run.completed","output":"hello world","usage":{"total_tokens":5}}\n',
        b": stream closed\n",
    ]
    posted = _run_transport_harness(monkeypatch, lines)
    res = hrt.call_hermes_run("do it", session_key="s")
    assert res["run_id"] == "run_x"
    assert res["approvals"] == 1
    assert any(p.get("choice") == "once" for p in posted)   # auto-approved with "once"
    assert all(p.get("choice") != "always" for p in posted)  # never permanent
    assert res["text"] == "hello world"
    assert res["usage"]["total_tokens"] == 5
    assert res["error"] == ""


def test_call_hermes_run_failure(monkeypatch):
    import hermes_run_transport as hrt
    lines = [
        b'data: {"event":"run.failed","run_id":"run_x","error":"boom"}\n',
        b": stream closed\n",
    ]
    _run_transport_harness(monkeypatch, lines)
    res = hrt.call_hermes_run("do it")
    assert res["error"] == "boom"


# ---------------------------------------------------------------------------
# Judge JSON-parsing robustness (no network)
# ---------------------------------------------------------------------------

def test_judge_json_parsing_variants(monkeypatch):
    import hermes_run_transport as hrt
    cases = [
        ('{"done": true, "reason": "ok"}', True),
        ('```json\n{"done": false, "reason": "nope"}\n```', False),
        ('Verdict: {"done": true, "reason": "complete"} -- hope this helps', True),
        ('{"done": "false", "reason": "string false must NOT complete"}', False),
        ('{"done": "true"}', True),
        ('total garbage, no json at all', False),     # parse fail -> fail-open (done False)
        ('{"reason": "no done key"}', False),          # missing key -> False
        # brace-heavy prose before the verdict (greedy regex used to choke here)
        ('The set {a, b} is handled. Verdict: {"done": true, "reason": "complete"}', True),
        # judge restates the schema object, then the real verdict (pick the one with done)
        ('Schema: {"format": "json"} then {"done": true, "reason": "ok"}', True),
        # done value inside a string must not trip brace matching
        ('{"done": false, "reason": "the marker {GOAL_COMPLETE} was not real"}', False),
    ]
    for content, expected in cases:
        monkeypatch.setattr(hrt, "_post",
                            lambda *a, **k: {"choices": [{"message": {"content": content}}]})
        v = hrt.call_hermes_judge("goal", "resp")
        assert v["done"] == expected, f"{content!r} -> {v}"


def test_judge_surfaces_error_on_network_failure(monkeypatch):
    # call_hermes_judge surfaces an `error` (so the loop can fail fast); it does
    # NOT silently return done=False-with-no-signal.
    import hermes_run_transport as hrt
    def boom(*a, **k):
        raise RuntimeError("deepseek down")
    monkeypatch.setattr(hrt, "_post", boom)
    v = hrt.call_hermes_judge("goal", "resp")
    assert v["done"] is False
    assert v["error"] and "deepseek down" in v["error"]


# ---------------------------------------------------------------------------
# Goal judge loop (Hermes /goal parity: aux judge decides done/continue)
# ---------------------------------------------------------------------------

def test_judge_completes_without_marker(tmp_path):
    # judge mode: agent text has NO marker; the judge decides completion on turn 2.
    calls = {"n": 0}
    def judge(goal, response):
        calls["n"] += 1
        return {"done": calls["n"] >= 2, "blocked": False, "reason": f"turn {calls['n']}", "error": ""}
    ctl = _ctl(tmp_path, ["working...", "more work, looks finished"], judge=judge)
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Do the thing.", completion_mode="judge", max_turns=5)
    assert rec["status"] == "complete"
    assert len(rec["turns"]) == 2
    assert len(rec["judge_verdicts"]) == 2
    assert rec["judge_verdicts"][-1]["done"] is True


def test_judge_continue_exhausts_budget(tmp_path):
    judge = lambda g, r: {"done": False, "blocked": False, "reason": "not yet", "error": ""}
    ctl = _ctl(tmp_path, ["a", "b", "c"], judge=judge)
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Never satisfied.", completion_mode="judge", max_turns=3)
    assert rec["status"] == "budget_exhausted"
    assert len(rec["turns"]) == 3


def test_judge_reason_fed_into_continuation(tmp_path):
    judge = lambda g, r: {"done": False, "blocked": False, "reason": "needs more X", "error": ""}
    ctl = _ctl(tmp_path, ["t1", "t2"], judge=judge)
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "x", completion_mode="judge", max_turns=2)
    # The 2nd prompt carries the judge's reason, and does NOT ask for a marker
    # (judge mode continuation must not contradict judge mode).
    cont = ctl._caller.calls[1]["prompt"]
    assert "needs more X" in cont
    assert "GOAL_COMPLETE: true" not in cont


def test_judge_unavailable_fails_fast(tmp_path):
    # Judge configured but erroring -> EXPLICIT status error, NOT silent
    # fail-open that burns the whole budget.
    def judge(goal, response):
        return {"done": False, "blocked": False, "reason": "", "error": "deepseek down"}
    ctl = _ctl(tmp_path, ["x", "y", "z"], judge=judge)
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "x", completion_mode="judge", max_turns=5)
    assert rec["status"] == "error"
    assert "judge unavailable" in rec["error"]
    assert len(rec["turns"]) == 1  # stopped immediately, did not burn turns


def test_completion_mode_marker_is_default(tmp_path):
    # Default is marker mode: a judge that would say done is NOT consulted; the
    # loop completes on the GOAL_COMPLETE marker.
    judge = lambda g, r: {"done": True, "blocked": False, "reason": "x", "error": ""}
    ctl = _ctl(tmp_path, ["no marker here", "still none GOAL_COMPLETE: true"], judge=judge)
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "x", max_turns=4)  # no completion_mode -> marker
    assert rec["status"] == "complete"
    assert len(rec["turns"]) == 2  # completed on the marker, not the judge
    assert rec["completion_mode"] == "marker"


# ---------------------------------------------------------------------------
# Manager judgment (the dispatcher is the goal judge)
# ---------------------------------------------------------------------------

def test_manager_judgment_pauses(tmp_path):
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "x", judgment_mode="manager", max_turns=3)
    assert rec["status"] == "awaiting_judgment"
    assert ctl.get_worker("catcas-auditor")["status"] == "awaiting_judgment"
    assert len(rec["turns"]) == 1  # worked once, then paused for the manager


def test_manager_accept_completes(tmp_path):
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "x", judgment_mode="manager")
    rec = ctl.judge("catcas-auditor", "accept")
    assert rec["status"] == "complete"
    assert rec["artifact_manifest"]["manager_judgment"]["verdict"] == "accept"


def test_manager_reject_resumes_with_feedback(tmp_path):
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true", "GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "x", judgment_mode="manager", max_turns=3)
    rec = ctl.judge("catcas-auditor", "reject", feedback="add edge cases")
    # Worker resumed, hit GOAL_COMPLETE again -> paused again for the manager.
    assert rec["status"] == "awaiting_judgment"
    assert any("MANAGER REVIEW" in c["prompt"] and "add edge cases" in c["prompt"]
               for c in ctl._caller.calls)
    rec2 = ctl.judge("catcas-auditor", "accept")
    assert rec2["status"] == "complete"
    assert len(rec2["judgments"]) == 2


def test_judge_requires_awaiting_state(tmp_path):
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "x")  # auto mode -> completes, not awaiting
    with pytest.raises(ValueError):
        ctl.judge("catcas-auditor", "accept")


def test_invalid_verdict_rejected(tmp_path):
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "x", judgment_mode="manager")
    with pytest.raises(ValueError):
        ctl.judge("catcas-auditor", "maybe")


def test_api_judge_endpoint(tmp_path):
    api, ctl = _api(tmp_path, ["GOAL_COMPLETE: true"])
    api.dispatch("POST", "/workers", {"worker_id": "w1", "write_roots": ["a"], "read_roots": ["a"]})
    api.dispatch("POST", "/workers/w1/tasks", {"task": "x", "judgment_mode": "manager"})
    state = api.dispatch("GET", "/workers/w1/state", {})[1]
    assert state["status"] == "awaiting_judgment"
    code, body = api.dispatch("POST", "/workers/w1/judge", {"verdict": "accept"})
    assert code == 200 and body["status"] == "complete"


# ---------------------------------------------------------------------------
# Continuation preserves conversation
# ---------------------------------------------------------------------------

def test_continuation_preserves_conversation(tmp_path):
    ctl = _ctl(tmp_path, ["more1", "more2", "more3", "finally GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Long task.", max_turns=2)
    assert rec["status"] == "budget_exhausted"
    convs = {c["conversation"] for c in ctl._caller.calls}
    assert convs == {"ccc:ags:catcas-auditor"}

    cont = ctl.continue_worker("catcas-auditor", max_turns=2)
    assert cont["status"] == "complete"
    # Every call, across submit + continue, used the SAME named conversation.
    convs_all = {c["conversation"] for c in ctl._caller.calls}
    assert convs_all == {"ccc:ags:catcas-auditor"}
    # Turn numbering continued rather than resetting.
    assert len(cont["turns"]) == 4


def test_continuation_requires_prior_task(tmp_path):
    ctl = _ctl(tmp_path, [])
    _make_worker(ctl)
    with pytest.raises(ValueError):
        ctl.continue_worker("catcas-auditor")


# ---------------------------------------------------------------------------
# No fake /goal anywhere
# ---------------------------------------------------------------------------

def test_no_native_goal_in_prompts(tmp_path):
    ctl = _ctl(tmp_path, ["x", "y GOAL_COMPLETE: true"])
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "Audit.", max_turns=4)
    for call in ctl._caller.calls:
        assert "/goal" not in call["prompt"]


def test_no_goal_endpoint_in_source():
    """The control plane source must not reference a /goal dispatch endpoint."""
    for fname in ("worker_control.py", "worker_api.py"):
        src = (SCRIPTS / fname).read_text(encoding="utf-8")
        # Mentioning '/goal' only appears in explanatory text; assert no code
        # builds a goal URL or posts to a goal path.
        assert "/api/goal" not in src
        assert 'POST", "/goal' not in src
        assert "sessions/{session_id}/goal" not in src


def test_docs_distinguish_identity_terms():
    """Req 14: docs distinguish worker_id / conversation / session_key /
    session_id / run_id / task_id."""
    doc = (ROOT / "WORKER_API.md").read_text(encoding="utf-8")
    for term in ("worker_id", "conversation", "session_key", "session_id", "run_id", "task_id"):
        assert term in doc, term


def test_docs_persistent_first_runs_not_memory():
    """Reqs 13/16: docs make /v1/responses the persistent memory lane, demote
    /v1/runs to execution, and state native /goal is not used."""
    doc = (ROOT / "WORKER_API.md").read_text(encoding="utf-8").lower()
    assert "persistent reasoning lane" in doc
    assert "execution lane" in doc
    assert "not the canonical memory" in doc        # runs explicitly demoted
    assert "/goal" in doc and "not used" in doc      # native /goal disclaimed
    # Must NOT claim runs is the persistent memory layer.
    assert "runs is the canonical memory" not in doc
    assert "/v1/runs provides persistent" not in doc


def test_real_default_caller_cannot_reach_network(tmp_path):
    """Safety net: a controller using the REAL default caller must not escape.

    The conftest kill-switch blocks urlopen, so submit_task degrades to status
    'error' instead of silently hitting the live agent and burning tokens.
    """
    ctl = WorkerController(state_dir=tmp_path / "_state")  # no injected caller
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Audit.")
    assert rec["status"] == "error"
    assert "BLOCKED" in rec.get("error", "")


def test_default_caller_uses_responses_transport(monkeypatch, tmp_path):
    """The default backend caller hits /v1/responses, not any goal path."""
    import hermes_harness

    captured = {}

    def fake_responses(prompt, conversation="", session_key="", store=False,
                       base_url="", api_key="", model="", timeout=None):
        captured["called"] = True
        captured["conversation"] = conversation
        return {"text": "GOAL_COMPLETE: true", "response_id": "r1", "usage": {}, "raw": {}}

    # default_caller calls the name imported into worker_control's namespace,
    # so patch it there (not only on hermes_harness).
    monkeypatch.setattr(hermes_harness, "call_hermes_responses", fake_responses)
    monkeypatch.setattr(worker_control, "call_hermes_responses", fake_responses)
    # Rebuild a controller whose default caller closes over the patched fn.
    caller = worker_control.default_caller()
    ctl = WorkerController(state_dir=tmp_path / "_state", caller=caller)
    _make_worker(ctl)
    ctl.submit_task("catcas-auditor", "Audit.")
    assert captured.get("called") is True
    assert captured.get("conversation") == "ccc:ags:catcas-auditor"


# ---------------------------------------------------------------------------
# Logs + manifest
# ---------------------------------------------------------------------------

def test_task_logs_written(tmp_path):
    ctl = _ctl(tmp_path, ["a", "b GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Audit.", max_turns=4)
    log = ctl.get_log(rec["task_id"])
    events = [e["event"] for e in log]
    assert "request" in events
    assert "prompt" in events
    assert "turn_output" in events
    assert "status" in events
    assert "manifest" in events
    assert "final" in events


def test_artifact_manifest_emitted(tmp_path):
    output = (
        "Done.\n"
        "ARTIFACT_MANIFEST:\n"
        "```json\n"
        '{"created_files": ["sandbox/test-scope/new.md"], '
        '"modified_files": ["sandbox/test-scope/roadmap.md"], '
        '"verification": "pytest green"}\n'
        "```\n"
        "GOAL_COMPLETE: true\n"
    )
    ctl = _ctl(tmp_path, [output])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Produce artifacts.")
    man = rec["artifact_manifest"]
    assert man is not None
    assert man["worker_id"] == "catcas-auditor"
    assert man["task_id"] == rec["task_id"]
    assert man["created_files"] == ["sandbox/test-scope/new.md"]
    assert man["modified_files"] == ["sandbox/test-scope/roadmap.md"]
    assert man["write_scope"] == ["sandbox/test-scope"]
    assert man["status"] == "complete"
    assert man["changed_files_source"] == "worker_reported"
    # Worker-level manifest pointer file was written.
    assert Path(ctl.get_worker("catcas-auditor")["artifact_manifest"]).exists()


def test_manifest_without_worker_block(tmp_path):
    ctl = _ctl(tmp_path, ["No manifest block here. GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "No manifest.")
    man = rec["artifact_manifest"]
    assert man["created_files"] == []
    assert man["modified_files"] == []
    assert man["status"] == "complete"


def test_worker_state_after_task(tmp_path):
    ctl = _ctl(tmp_path, ["GOAL_COMPLETE: true"])
    _make_worker(ctl)
    rec = ctl.submit_task("catcas-auditor", "Audit.")
    state = ctl.get_state("catcas-auditor")
    assert state["last_task_id"] == rec["task_id"]
    assert state["last_task"]["status"] == "complete"


# ---------------------------------------------------------------------------
# Write firewall (postflight git-diff scope audit)
# ---------------------------------------------------------------------------

import subprocess as _sp


def _git_repo(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    for args in (["init"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        _sp.run(["git", "-C", str(ws), *args], capture_output=True)
    (ws / "README.md").write_text("init", encoding="utf-8")
    _sp.run(["git", "-C", str(ws), "add", "-A"], capture_output=True)
    _sp.run(["git", "-C", str(ws), "commit", "-m", "init"], capture_output=True)
    return ws


class FileWriter:
    """Caller that actually writes files (simulating agent edits) then reports them."""

    def __init__(self, workspace, files, response):
        self.workspace, self.files, self.response, self.calls = workspace, files, response, []

    def __call__(self, prompt, conversation="", session_key="", model="", history=None):
        self.calls.append({"prompt": prompt})
        for rel in self.files:
            p = Path(self.workspace) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x", encoding="utf-8")
        return {"text": self.response, "usage": {}, "response_id": "r"}


_FW_RESP = (
    "Done.\nGOAL_COMPLETE: true\nARTIFACT_MANIFEST:\n```json\n"
    '{"created_files": ["allowed/in.txt", "forbidden/out.txt"], "modified_files": []}\n```'
)


def test_firewall_reports_agent_escapes(tmp_path):
    ws = _git_repo(tmp_path)
    caller = FileWriter(ws, ["allowed/in.txt", "forbidden/out.txt"], _FW_RESP)
    ctl = WorkerController(state_dir=tmp_path / "_state", caller=caller, judge=_marker_judge)
    ctl.create_worker(worker_id="w", workspace=str(ws), write_roots=["allowed"], read_roots=["allowed"])
    man = ctl.submit_task("w", "do it")["artifact_manifest"]
    assert man["changed_files_source"] == "git_observed"
    # Agent reported both AND both hit disk -> both confirmed.
    assert set(man["agent_confirmed"]) == {"allowed/in.txt", "forbidden/out.txt"}
    assert man["agent_escapes"] == ["forbidden/out.txt"]   # confirmed AND out of scope
    assert man["agent_clean"] is False
    assert man["agent_missing"] == []                      # nothing fabricated
    assert man["agent_unchanged"] == []                    # fresh writes, all changed
    assert (ws / "forbidden/out.txt").exists()             # report-only by default


def test_firewall_busy_tree_isolates_concurrent_edits(tmp_path):
    """Always-dirty tree: an out-of-scope file the agent did NOT report (a
    concurrent edit) lands in unattributed_changes, NOT agent_escapes. The agent
    stays 'clean' and nothing of the user's is reverted."""
    ws = _git_repo(tmp_path)
    # Pre-existing dirty state (baseline) — should be ignored entirely.
    (ws / "preexisting.txt").write_text("dirty before start", encoding="utf-8")
    resp = ('Done.\nGOAL_COMPLETE: true\nARTIFACT_MANIFEST:\n```json\n'
            '{"created_files": ["allowed/in.txt"], "modified_files": []}\n```')
    caller = FileWriter(ws, ["allowed/in.txt", "someones_parallel_work/report.md"], resp)
    ctl = WorkerController(state_dir=tmp_path / "_state", caller=caller, judge=_marker_judge)
    ctl.create_worker(worker_id="w", workspace=str(ws), write_roots=["allowed"], read_roots=["allowed"])
    man = ctl.submit_task("w", "do it", auto_revert=True)["artifact_manifest"]
    assert man["agent_confirmed"] == ["allowed/in.txt"]
    assert man["agent_escapes"] == []                                  # agent breached nothing
    assert man["agent_clean"] is True
    assert "someones_parallel_work/report.md" in man["unattributed_changes"]
    assert "preexisting.txt" not in man["unattributed_changes"]        # baseline ignored
    assert man["scope_reverted"] == []
    assert (ws / "someones_parallel_work/report.md").exists()          # parallel work safe


def test_firewall_revert_never_touches_preexisting_dirty(tmp_path):
    """A tracked file already dirty BEFORE the task (your in-flight edit), out of
    scope, must never be reverted even with auto_revert — it's in the baseline."""
    ws = _git_repo(tmp_path)
    victim = ws / "myfile.txt"
    victim.write_text("MY UNCOMMITTED WORK", encoding="utf-8")  # dirty before task starts
    # Agent (falsely) reports touching the out-of-scope victim.
    resp = ('Done.\nGOAL_COMPLETE: true\nARTIFACT_MANIFEST:\n```json\n'
            '{"created_files": ["allowed/in.txt"], "modified_files": ["myfile.txt"]}\n```')
    caller = FileWriter(ws, ["allowed/in.txt"], resp)
    ctl = WorkerController(state_dir=tmp_path / "_state", caller=caller, judge=_marker_judge)
    ctl.create_worker(worker_id="w", workspace=str(ws), write_roots=["allowed"], read_roots=["allowed"])
    man = ctl.submit_task("w", "do it", auto_revert=True)["artifact_manifest"]
    assert "myfile.txt" not in man["scope_reverted"]          # baseline-protected
    assert victim.read_text(encoding="utf-8") == "MY UNCOMMITTED WORK"  # untouched


def test_firewall_subdir_workspace_attribution(tmp_path):
    """Workspace is a SUBDIR of the repo: git reports repo-root-relative paths,
    but attribution + scope still work via the git prefix."""
    root = _git_repo(tmp_path)
    sub = root / "project"
    sub.mkdir()
    resp = ('Done.\nGOAL_COMPLETE: true\nARTIFACT_MANIFEST:\n```json\n'
            '{"created_files": ["allowed/in.txt", "forbidden/out.txt"], "modified_files": []}\n```')
    caller = FileWriter(sub, ["allowed/in.txt", "forbidden/out.txt"], resp)
    ctl = WorkerController(state_dir=tmp_path / "_state", caller=caller, judge=_marker_judge)
    ctl.create_worker(worker_id="w", workspace=str(sub), write_roots=["allowed"], read_roots=["allowed"])
    man = ctl.submit_task("w", "do it")["artifact_manifest"]
    assert "project/allowed/in.txt" in man["agent_confirmed"]      # in-scope, repo-relative
    assert man["agent_escapes"] == ["project/forbidden/out.txt"]   # out-of-scope detected
    assert man["agent_clean"] is False


def test_firewall_detects_fabrication(tmp_path):
    """Agent claims it created a file but it never hit disk -> agent_missing
    flags the fabrication (the '40 tests passed' failure mode)."""
    ws = _git_repo(tmp_path)
    resp = ('All done!\nGOAL_COMPLETE: true\nARTIFACT_MANIFEST:\n```json\n'
            '{"created_files": ["allowed/ghost.txt"], "modified_files": []}\n```')
    caller = ScriptedCaller([resp])  # does NOT write any file
    ctl = WorkerController(state_dir=tmp_path / "_state", caller=caller, judge=_marker_judge)
    ctl.create_worker(worker_id="w", workspace=str(ws), write_roots=["allowed"], read_roots=["allowed"])
    man = ctl.submit_task("w", "do it")["artifact_manifest"]
    assert man["agent_confirmed"] == []                    # nothing actually changed
    assert "allowed/ghost.txt" in man["agent_missing"]     # claimed but NOT on disk -> fabrication
    assert man["agent_unchanged"] == []
    assert not (ws / "allowed/ghost.txt").exists()


def test_firewall_preexisting_file_is_unchanged_not_fabrication(tmp_path):
    """A claimed file that EXISTS but didn't change this run (e.g. a re-run, or
    identical rewrite) is agent_unchanged, NOT agent_missing/fabrication."""
    ws = _git_repo(tmp_path)
    (ws / "allowed").mkdir()
    (ws / "allowed" / "kept.txt").write_text("already here", encoding="utf-8")  # exists at baseline
    resp = ('Done.\nGOAL_COMPLETE: true\nARTIFACT_MANIFEST:\n```json\n'
            '{"created_files": ["allowed/kept.txt"], "modified_files": []}\n```')
    caller = ScriptedCaller([resp])  # claims it but writes nothing new
    ctl = WorkerController(state_dir=tmp_path / "_state", caller=caller, judge=_marker_judge)
    ctl.create_worker(worker_id="w", workspace=str(ws), write_roots=["allowed"], read_roots=["allowed"])
    man = ctl.submit_task("w", "do it")["artifact_manifest"]
    assert man["agent_missing"] == []                        # NOT fabrication: file exists
    assert "allowed/kept.txt" in man["agent_unchanged"]      # pre-existing / no-op
    assert (ws / "allowed" / "kept.txt").exists()


def test_firewall_auto_revert_only_agent_files(tmp_path):
    ws = _git_repo(tmp_path)
    caller = FileWriter(ws, ["allowed/in.txt", "forbidden/out.txt"], _FW_RESP)
    ctl = WorkerController(state_dir=tmp_path / "_state", caller=caller, judge=_marker_judge)
    ctl.create_worker(worker_id="w", workspace=str(ws), write_roots=["allowed"], read_roots=["allowed"])
    man = ctl.submit_task("w", "do it", auto_revert=True)["artifact_manifest"]
    assert "forbidden/out.txt" in man["scope_reverted"]
    assert not (ws / "forbidden/out.txt").exists()   # out-of-scope reverted
    assert (ws / "allowed/in.txt").exists()           # in-scope kept


def test_firewall_skips_non_git_workspace(tmp_path):
    resp = 'GOAL_COMPLETE: true\nARTIFACT_MANIFEST:\n```json\n{"created_files": ["x"]}\n```'
    ctl = _ctl(tmp_path, [resp])
    ctl.create_worker(worker_id="w", workspace=str(tmp_path / "notgit"), write_roots=["a"], read_roots=["a"])
    man = ctl.submit_task("w", "x")["artifact_manifest"]
    assert man["changed_files_source"] == "worker_reported"
    assert man["agent_confirmed"] is None
    assert man["agent_escapes"] is None


# ---------------------------------------------------------------------------
# API layer (no socket)
# ---------------------------------------------------------------------------

def _api(tmp_path, outputs, api_key=""):
    ctl = _ctl(tmp_path, outputs)
    return WorkerAPI(ctl, api_key=api_key), ctl


def test_api_create_and_get_worker(tmp_path):
    api, _ = _api(tmp_path, [])
    code, body = api.dispatch("POST", "/workers", {
        "worker_id": "w1", "conversation": "conv-1", "session_key": "sk-1",
        "write_roots": ["a/b"], "read_roots": ["a/b"],
    })
    assert code == 201
    assert body["worker_id"] == "w1"
    code, body = api.dispatch("GET", "/workers/w1", {})
    assert code == 200
    assert body["conversation"] == "conv-1"


def test_api_submit_task_runs_loop(tmp_path):
    api, ctl = _api(tmp_path, ["GOAL_COMPLETE: true"])
    api.dispatch("POST", "/workers", {"worker_id": "w1", "write_roots": ["a"], "read_roots": ["a"]})
    code, body = api.dispatch("POST", "/workers/w1/tasks", {"task": "Audit a/."})
    assert code == 200
    assert body["status"] == "complete"
    task_id = body["task_id"]
    code, log = api.dispatch("GET", f"/tasks/{task_id}/log", {})
    assert code == 200
    assert len(log["log"]) > 0


def test_api_continue_endpoint(tmp_path):
    api, ctl = _api(tmp_path, ["more", "done GOAL_COMPLETE: true"])
    api.dispatch("POST", "/workers", {"worker_id": "w1", "write_roots": ["a"], "read_roots": ["a"]})
    api.dispatch("POST", "/workers/w1/tasks", {"task": "x", "max_turns": 1})
    code, body = api.dispatch("POST", "/workers/w1/continue", {"max_turns": 2})
    assert code == 200
    assert body["status"] == "complete"


def test_api_unknown_worker_404(tmp_path):
    api, _ = _api(tmp_path, [])
    code, body = api.dispatch("POST", "/workers/nope/tasks", {"task": "x"})
    assert code == 404


def test_api_busy_worker_returns_400(tmp_path):
    api, ctl = _api(tmp_path, ["GOAL_COMPLETE: true"])
    api.dispatch("POST", "/workers", {"worker_id": "w1", "write_roots": ["a"], "read_roots": ["a"]})
    w = ctl.get_worker("w1"); w["status"] = "running"; ctl.save_worker(w)
    code, body = api.dispatch("POST", "/workers/w1/tasks", {"task": "x"})
    assert code == 400 and "busy" in body["error"].lower()


def test_api_judge_not_awaiting_returns_400(tmp_path):
    api, _ = _api(tmp_path, ["GOAL_COMPLETE: true"])
    api.dispatch("POST", "/workers", {"worker_id": "w1", "write_roots": ["a"], "read_roots": ["a"]})
    api.dispatch("POST", "/workers/w1/tasks", {"task": "x"})  # auto-completes, not awaiting
    code, _body = api.dispatch("POST", "/workers/w1/judge", {"verdict": "accept"})
    assert code == 400


def test_api_auth_on_judge(tmp_path):
    api, _ = _api(tmp_path, [], api_key="secret")
    code, _body = api.dispatch("POST", "/workers/w1/judge", {"verdict": "accept"}, auth_header="")
    assert code == 401


def test_api_auth_enforced(tmp_path):
    api, _ = _api(tmp_path, [], api_key="secret")
    code, _body = api.dispatch("GET", "/workers", {}, auth_header="")
    assert code == 401
    code, _body = api.dispatch("GET", "/workers", {}, auth_header="Bearer secret")
    assert code == 200


def test_api_health_declares_no_native_goal(tmp_path):
    api, _ = _api(tmp_path, [])
    code, body = api.dispatch("GET", "/health", {})
    assert code == 200
    assert body["native_goal"] is False
