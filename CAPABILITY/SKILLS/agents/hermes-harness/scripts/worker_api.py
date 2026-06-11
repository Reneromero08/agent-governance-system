#!/usr/bin/env python3
"""Minimal HTTP Worker API for the Hermes Harness control plane.

Stdlib-only (http.server). No external framework, matching the rest of the
skill which already speaks raw urllib. The API is a thin transport over
WorkerController -- all logic (registry, goal loop, manifests, logs) lives in
worker_control.py and is unit-tested without a socket.

This API IS the harness. Any manager harness (OpenCode or otherwise) is just a
client. It never invokes native Hermes /goal; the goal loop is owned here.

Routes:
    GET  /workers
    POST /workers
    GET  /workers/{worker_id}
    POST /workers/{worker_id}/tasks
    GET  /workers/{worker_id}/tasks/{task_id}
    POST /workers/{worker_id}/continue
    GET  /workers/{worker_id}/state
    GET  /tasks/{task_id}/log
    GET  /health

Auth: optional bearer token via WORKER_API_KEY. If unset, the API is open
(intended for localhost-only control-plane use).
"""

from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from worker_control import WorkerController, DEFAULT_MAX_TURNS  # noqa: E402

# (method, compiled-pattern, handler-name)
Route = Tuple[str, Pattern[str], str]


def _routes() -> List[Route]:
    return [
        ("GET", re.compile(r"^/health$"), "health"),
        ("GET", re.compile(r"^/workers$"), "list_workers"),
        ("POST", re.compile(r"^/workers$"), "create_worker"),
        ("GET", re.compile(r"^/workers/(?P<worker_id>[^/]+)$"), "get_worker"),
        ("GET", re.compile(r"^/workers/(?P<worker_id>[^/]+)/state$"), "get_state"),
        ("POST", re.compile(r"^/workers/(?P<worker_id>[^/]+)/tasks$"), "submit_task"),
        ("GET", re.compile(r"^/workers/(?P<worker_id>[^/]+)/tasks/(?P<task_id>[^/]+)$"), "get_task"),
        ("POST", re.compile(r"^/workers/(?P<worker_id>[^/]+)/continue$"), "continue_worker"),
        ("POST", re.compile(r"^/workers/(?P<worker_id>[^/]+)/judge$"), "judge"),
        ("GET", re.compile(r"^/tasks/(?P<task_id>[^/]+)/log$"), "get_log"),
    ]


class WorkerAPI:
    """Routing core, decoupled from the socket so it can be unit-tested."""

    def __init__(self, controller: WorkerController, api_key: Optional[str] = None) -> None:
        self.ctl = controller
        self.api_key = api_key if api_key is not None else os.environ.get("WORKER_API_KEY", "")
        self.routes = _routes()

    def authorized(self, auth_header: str) -> bool:
        if not self.api_key:
            return True
        return auth_header == f"Bearer {self.api_key}"

    def dispatch(self, method: str, path: str, body: Dict[str, Any], auth_header: str = "") -> Tuple[int, Dict[str, Any]]:
        if not self.authorized(auth_header):
            return 401, {"error": "unauthorized"}
        for m, pat, name in self.routes:
            if m != method:
                continue
            match = pat.match(path)
            if not match:
                continue
            handler: Callable[..., Tuple[int, Dict[str, Any]]] = getattr(self, f"_h_{name}")
            try:
                return handler(body, **match.groupdict())
            except KeyError as exc:
                return 404, {"error": str(exc)}
            except ValueError as exc:
                return 400, {"error": str(exc)}
            except Exception as exc:  # pragma: no cover - defensive
                return 500, {"error": str(exc)}
        return 404, {"error": f"no route for {method} {path}"}

    # -- handlers ----------------------------------------------------------

    def _h_health(self, body):  # noqa: ANN001
        return 200, {"ok": True, "service": "worker-api", "native_goal": False}

    def _h_list_workers(self, body):  # noqa: ANN001
        return 200, {"workers": self.ctl.list_workers()}

    def _h_create_worker(self, body):  # noqa: ANN001
        wid = body.get("worker_id")
        worker = self.ctl.create_worker(
            worker_id=wid,
            role=body.get("role", "specialist"),
            conversation=body.get("conversation", ""),
            session_key=body.get("session_key", ""),
            model=body.get("model", ""),
            workspace=body.get("workspace", "."),
            backend=body.get("backend", "hermes_responses"),
            read_roots=body.get("read_roots"),
            write_roots=body.get("write_roots"),
            deny_write_roots=body.get("deny_write_roots"),
            search_policy=body.get("search_policy", "artifact_only"),
            branch_policy=body.get("branch_policy", "forbidden"),
            persistent_transport=body.get("persistent_transport", "responses"),
            execution_transport=body.get("execution_transport", "runs"),
            session_id=body.get("session_id", ""),
        )
        return 201, worker

    def _h_get_worker(self, body, worker_id):  # noqa: ANN001
        worker = self.ctl.get_worker(worker_id)
        if not worker:
            return 404, {"error": f"unknown worker {worker_id}"}
        return 200, worker

    def _h_get_state(self, body, worker_id):  # noqa: ANN001
        return 200, self.ctl.get_state(worker_id)

    def _h_submit_task(self, body, worker_id):  # noqa: ANN001
        rec = self.ctl.submit_task(
            worker_id,
            body.get("task", ""),
            mode=body.get("mode", "persistent_worker_verify"),
            goal_loop=bool(body.get("goal_loop", True)),
            max_turns=int(body.get("max_turns", DEFAULT_MAX_TURNS)),
            constraints=body.get("constraints", ""),
            output_contract=body.get("output_contract", ""),
            acceptance_criteria=body.get("acceptance_criteria", ""),
            target=body.get("target", ""),
            done_marker=body.get("done_marker", "GOAL_COMPLETE: true"),
            blocked_marker=body.get("blocked_marker", "GOAL_BLOCKED: true"),
            write_roots=body.get("write_roots"),
            read_roots=body.get("read_roots"),
            deny_roots=body.get("deny_write_roots"),
            search_policy=body.get("search_policy"),
            branch_policy=body.get("branch_policy"),
            verify_command=body.get("verify_command", ""),
            verify_timeout=int(body.get("verify_timeout", 120)),
            verify_cwd=body.get("verify_cwd", ""),
            judgment_mode=body.get("judgment_mode", "auto"),
            completion_mode=body.get("completion_mode", "marker"),
            use_judge=body.get("use_judge"),  # deprecated alias; None unless set
            auto_revert=bool(body.get("auto_revert", False)),
            execution_required=bool(body.get("execution_required", False)),
            execution_transport=body.get("execution_transport"),
        )
        rec_out = dict(rec)
        rec_out["session_key_present"] = bool(rec.get("session_key"))
        rec_out["log_path"] = str(self.ctl._log_path(rec["task_id"]))
        return 200, rec_out

    def _h_judge(self, body, worker_id):  # noqa: ANN001
        rec = self.ctl.judge(
            worker_id,
            body.get("verdict", ""),
            feedback=body.get("feedback", ""),
            max_turns=int(body.get("max_turns", DEFAULT_MAX_TURNS)),
        )
        return 200, rec

    def _h_get_task(self, body, worker_id, task_id):  # noqa: ANN001
        rec = self.ctl.get_task(task_id)
        if not rec or rec.get("worker_id") != worker_id:
            return 404, {"error": f"task {task_id} not found for worker {worker_id}"}
        return 200, rec

    def _h_continue_worker(self, body, worker_id):  # noqa: ANN001
        rec = self.ctl.continue_worker(
            worker_id,
            max_turns=int(body.get("max_turns", DEFAULT_MAX_TURNS)),
            nudge=body.get("nudge", ""),
        )
        return 200, rec

    def _h_get_log(self, body, task_id):  # noqa: ANN001
        return 200, {"task_id": task_id, "log": self.ctl.get_log(task_id)}


def make_handler(api: WorkerAPI):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WorkerAPI/0.1"

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if not length:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            if not raw.strip():
                return {}
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {"_raw": parsed}
            except json.JSONDecodeError:
                return {}

        def _handle(self, method: str) -> None:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            body = self._read_body() if method == "POST" else {}
            auth = self.headers.get("Authorization", "")
            code, payload = api.dispatch(method, path, body, auth_header=auth)
            self._send(code, payload)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def log_message(self, fmt, *args):  # silence default stderr logging
            return

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8770, state_dir: Optional[str] = None) -> None:
    controller = WorkerController(state_dir=state_dir)
    api = WorkerAPI(controller)
    handler = make_handler(api)
    httpd = ThreadingHTTPServer((host, port), handler)
    auth = "on" if api.api_key else "off"
    print(f"[worker-api] listening on http://{host}:{port}  (auth: {auth}, native /goal: never)")
    print(f"[worker-api] state dir: {controller.state_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[worker-api] shutting down")
    finally:
        httpd.server_close()


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="HTTP Worker API for the Hermes Harness control plane.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--state-dir", default=None)
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, state_dir=args.state_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
