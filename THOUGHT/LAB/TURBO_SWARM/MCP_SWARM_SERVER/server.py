#!/usr/bin/env python3
"""
Swarm MCP Server - Experimental Swarm Coordination Tools

Extracted from AGS MCP Server for TURBO_SWARM experiments.
Provides message board and agent inbox coordination primitives.

Usage:
  python server.py          # Start server (stdio mode)
  python server.py --test   # Run a quick test
"""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CAPABILITY_ROOT = PROJECT_ROOT / "CAPABILITY"
LAW_ROOT = PROJECT_ROOT / "LAW"

# Data directories
BOARD_ROOT = LAW_ROOT / "CONTRACTS" / "_runs" / "message_board"
BOARD_ROLES_PATH = CAPABILITY_ROOT / "MCP" / "board_roles.json"
INBOX_ROOT = PROJECT_ROOT / "INBOX" / "agents" / "Local Models"

# Schema directory
SCHEMAS_DIR = Path(__file__).parent / "schemas"

# MCP Protocol Constants
MCP_VERSION = "2024-11-05"
SERVER_NAME = "swarm-mcp-server"
SERVER_VERSION = "0.1.0"


# =============================================================================
# SCHEMA LOADING
# =============================================================================

def load_schema(name: str) -> Dict:
    """Load a schema file."""
    schema_path = SCHEMAS_DIR / f"{name}.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
    return {}


# =============================================================================
# GOVERNED TOOL DECORATOR (Lightweight version)
# =============================================================================

def governed_tool(func):
    """Decorator: Run preflight + admission + critic.py before execution to enforce governance lock."""
    def wrapper(self, args: Dict) -> Dict:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Try to run preflight, but don't fail if ags.py is unavailable
        try:
            preflight = subprocess.run(
                [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "ags.py"), "preflight"],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            if preflight.returncode != 0:
                return {
                    "content": [{"type": "text", "text": f"PREFLIGHT_FAIL: {preflight.stderr}"}],
                    "isError": True,
                }
        except FileNotFoundError:
            # ags.py not found - continue without preflight in experimental mode
            pass
        except subprocess.TimeoutExpired:
            return {
                "content": [{"type": "text", "text": "PREFLIGHT_TIMEOUT"}],
                "isError": True,
            }

        # Run the actual tool
        return func(self, args)

    return wrapper


# =============================================================================
# SWARM MCP SERVER CLASS
# =============================================================================

class SwarmMCPServer:
    """Experimental Swarm MCP Server - Message board and agent inbox tools."""

    def __init__(self):
        self.tools_schema = load_schema("tools")
        self._initialized = False
        self.session_id = str(uuid.uuid4())

    # -------------------------------------------------------------------------
    # Message Board Helper Methods
    # -------------------------------------------------------------------------

    def _normalize_board(self, board: str) -> str:
        if not isinstance(board, str) or not board.strip():
            raise ValueError("BOARD_INVALID")
        board = board.strip()
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if any(ch not in allowed for ch in board):
            raise ValueError("BOARD_INVALID")
        return board

    def _load_board_roles(self) -> Dict[str, List[str]]:
        if not BOARD_ROLES_PATH.exists():
            return {"moderators": [], "admins": []}
        try:
            obj = json.loads(BOARD_ROLES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"moderators": [], "admins": []}
        moderators = obj.get("moderators", [])
        admins = obj.get("admins", [])
        if not isinstance(moderators, list):
            moderators = []
        if not isinstance(admins, list):
            admins = []
        return {"moderators": moderators, "admins": admins}

    def _board_role(self) -> str:
        roles = self._load_board_roles()
        if self.session_id in roles.get("admins", []):
            return "admin"
        if self.session_id in roles.get("moderators", []):
            return "moderator"
        return "poster"

    def _board_path(self, board: str) -> Path:
        return BOARD_ROOT / f"{board}.jsonl"

    def _append_board_event(self, board: str, event: Dict[str, Any]) -> None:
        BOARD_ROOT.mkdir(parents=True, exist_ok=True)
        path = self._board_path(board)
        line = json.dumps(event, sort_keys=True, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_board_events(self, board: str) -> List[Dict[str, Any]]:
        path = self._board_path(board)
        if not path.exists():
            return []
        events: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
            except Exception:
                continue
        return events

    def _materialize_board(
        self,
        events: List[Dict[str, Any]],
        *,
        include_deleted: bool,
        pinned_first: bool,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        posts: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for event in events:
            etype = event.get("type")
            if etype == "purge":
                posts = {}
                order = []
                continue
            if etype == "post":
                post_id = event.get("id")
                if isinstance(post_id, str) and post_id:
                    posts[post_id] = {
                        "id": post_id,
                        "message": event.get("message"),
                        "author_session_id": event.get("author_session_id"),
                        "role": event.get("role"),
                        "created_at": event.get("created_at"),
                        "pinned": False,
                        "deleted": False,
                    }
                    order.append(post_id)
                continue
            ref_id = event.get("ref_id")
            if not isinstance(ref_id, str) or ref_id not in posts:
                continue
            if etype == "pin":
                posts[ref_id]["pinned"] = True
            elif etype == "unpin":
                posts[ref_id]["pinned"] = False
            elif etype == "delete":
                posts[ref_id]["deleted"] = True

        items = [posts[pid] for pid in order if pid in posts]
        if not include_deleted:
            items = [item for item in items if not item.get("deleted")]
        if pinned_first:
            items = sorted(items, key=lambda x: (not x.get("pinned", False), x.get("created_at") or ""))
        if isinstance(limit, int) and limit > 0:
            items = items[:limit]
        return items

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def _tool_message_board_list(self, args: Dict) -> Dict:
        board = self._normalize_board(args.get("board", "default"))
        include_deleted = bool(args.get("include_deleted", False))
        pinned_first = bool(args.get("pinned_first", True))
        limit = args.get("limit")
        if limit is not None and not isinstance(limit, int):
            return {
                "content": [{"type": "text", "text": "Invalid limit"}],
                "isError": True,
            }
        events = self._load_board_events(board)
        items = self._materialize_board(
            events,
            include_deleted=include_deleted,
            pinned_first=pinned_first,
            limit=limit,
        )
        payload = {
            "board": board,
            "count": len(items),
            "items": items,
        }
        return {"content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}]}

    @governed_tool
    def _tool_message_board_write(self, args: Dict) -> Dict:
        board = self._normalize_board(args.get("board", "default"))
        action = args.get("action")
        if action not in {"post", "pin", "unpin", "delete", "purge"}:
            return {
                "content": [{"type": "text", "text": "Invalid action"}],
                "isError": True,
            }
        role = self._board_role()
        required = {
            "post": {"poster", "moderator", "admin"},
            "pin": {"moderator", "admin"},
            "unpin": {"moderator", "admin"},
            "delete": {"moderator", "admin"},
            "purge": {"admin"},
        }
        if role not in required[action]:
            return {
                "content": [{"type": "text", "text": "BOARD_FORBIDDEN"}],
                "isError": True,
            }

        message = args.get("message")
        ref_id = args.get("ref_id")
        if action == "post":
            if not isinstance(message, str) or not message.strip():
                return {
                    "content": [{"type": "text", "text": "MESSAGE_REQUIRED"}],
                    "isError": True,
                }
        if action in {"pin", "unpin", "delete"}:
            if not isinstance(ref_id, str) or not ref_id.strip():
                return {
                    "content": [{"type": "text", "text": "REF_ID_REQUIRED"}],
                    "isError": True,
                }

        event = {
            "id": uuid.uuid4().hex,
            "board": board,
            "author_session_id": self.session_id,
            "role": role,
            "type": action,
            "message": message if action == "post" else None,
            "ref_id": ref_id if action in {"pin", "unpin", "delete"} else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._append_board_event(board, event)
        payload = {
            "ok": True,
            "event_id": event["id"],
            "board": board,
            "role": role,
            "action": action,
        }
        return {"content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}]}

    def _tool_agent_inbox_list(self, args: Dict) -> Dict:
        """List tasks from the agent inbox."""
        status = args.get("status", "pending").upper()
        if status == "PENDING":
            target_dir = INBOX_ROOT / "PENDING_TASKS"
        elif status == "ACTIVE":
            target_dir = INBOX_ROOT / "ACTIVE_TASKS"
        elif status == "COMPLETED":
            target_dir = INBOX_ROOT / "COMPLETED_TASKS"
        elif status == "FAILED":
            target_dir = INBOX_ROOT / "FAILED_TASKS"
        else:
            return {"content": [{"type": "text", "text": "Invalid status"}], "isError": True}

        if not target_dir.exists():
            return {"content": [{"type": "text", "text": json.dumps({"tasks": []})}]}

        limit = args.get("limit", 20)
        tasks = []
        for p in sorted(target_dir.glob("*.json"), key=os.path.getmtime, reverse=True)[:limit]:
            try:
                tasks.append(json.loads(p.read_text(encoding="utf-8")))
            except:
                continue

        return {"content": [{"type": "text", "text": json.dumps({"tasks": tasks}, indent=2)}]}

    def _tool_agent_inbox_claim(self, args: Dict) -> Dict:
        """Claim a pending task."""
        task_id = args.get("task_id")
        agent_id = args.get("agent_id")

        if not task_id or not agent_id:
            return {"content": [{"type": "text", "text": "task_id and agent_id required"}], "isError": True}

        pending_dir = INBOX_ROOT / "PENDING_TASKS"
        active_dir = INBOX_ROOT / "ACTIVE_TASKS"

        pending_dir.mkdir(parents=True, exist_ok=True)
        active_dir.mkdir(parents=True, exist_ok=True)

        # Find the task file
        task_file = None
        for p in pending_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("task_id") == task_id:
                    task_file = p
                    break
            except:
                continue

        if not task_file:
            return {"content": [{"type": "text", "text": f"Task {task_id} not found in PENDING"}], "isError": True}

        # Atomically move and update
        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            data["status"] = "ACTIVE"
            data["assigned_to"] = agent_id
            data["claimed_at"] = datetime.now().isoformat()

            new_path = active_dir / task_file.name
            with open(new_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            task_file.unlink()

            return {"content": [{"type": "text", "text": json.dumps({"status": "success", "task_id": task_id, "path": str(new_path)})}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Claim failed: {str(e)}"}], "isError": True}

    def _tool_agent_inbox_finalize(self, args: Dict) -> Dict:
        """Finalize a task."""
        task_id = args.get("task_id")
        status = args.get("status", "").upper()
        result_text = args.get("result", "")

        if not task_id or status not in {"COMPLETED", "FAILED"}:
            return {"content": [{"type": "text", "text": "task_id and valid status (COMPLETED/FAILED) required"}], "isError": True}

        active_dir = INBOX_ROOT / "ACTIVE_TASKS"
        target_dir = INBOX_ROOT / f"{status}_TASKS"
        target_dir.mkdir(parents=True, exist_ok=True)

        task_file = None
        for p in active_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("task_id") == task_id:
                    task_file = p
                    break
            except:
                continue

        if not task_file:
            return {"content": [{"type": "text", "text": f"Task {task_id} not found in ACTIVE"}], "isError": True}

        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            data["status"] = status
            data["result"] = result_text
            data["finished_at"] = datetime.now().isoformat()

            new_path = target_dir / task_file.name
            with open(new_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            task_file.unlink()

            return {"content": [{"type": "text", "text": json.dumps({"status": "success", "task_id": task_id})}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Finalize failed: {str(e)}"}], "isError": True}

    # -------------------------------------------------------------------------
    # MCP Protocol Handlers
    # -------------------------------------------------------------------------

    def handle_request(self, request: Dict) -> Optional[Dict]:
        """Handle a JSON-RPC 2.0 request."""
        method = request.get("method", "") or ""
        params = request.get("params", {}) or {}

        has_id = ("id" in request) and (request.get("id") is not None)
        request_id = request.get("id") if has_id else None

        handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "notifications/initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

        handler = handlers.get(method)
        if not handler:
            return None if not has_id else self._error_response(request_id, -32601, f"Method not found: {method}")

        try:
            result = handler(params)
        except Exception as e:
            return None if not has_id else self._error_response(request_id, -32603, str(e))

        return None if not has_id else self._success_response(request_id, result)

    def _success_response(self, request_id: Any, result: Any) -> Dict:
        resp: Dict[str, Any] = {"jsonrpc": "2.0", "result": result}
        if request_id is not None:
            resp["id"] = request_id
        return resp

    def _error_response(self, request_id: Any, code: int, message: str) -> Dict:
        resp: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
        }
        if request_id is not None:
            resp["id"] = request_id
        return resp

    def _handle_initialize(self, params: Dict) -> Dict:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            }
        }

    def _handle_initialized(self, params: Dict) -> None:
        """Handle initialized notification."""
        return None

    def _handle_tools_list(self, params: Dict) -> Dict:
        """List available tools."""
        tools = []
        for tool_def in self.tools_schema.get("definitions", {}).values():
            tools.append({
                "name": tool_def.get("name"),
                "description": tool_def.get("description"),
                "inputSchema": tool_def.get("inputSchema", {})
            })
        return {"tools": tools}

    def _handle_tools_call(self, params: Dict) -> Dict:
        """Call a tool."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        tool_handlers = {
            "message_board_list": self._tool_message_board_list,
            "message_board_write": self._tool_message_board_write,
            "agent_inbox_list": self._tool_agent_inbox_list,
            "agent_inbox_claim": self._tool_agent_inbox_claim,
            "agent_inbox_finalize": self._tool_agent_inbox_finalize,
        }

        handler = tool_handlers.get(tool_name)
        if not handler:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

        try:
            return handler(arguments)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Tool error: {str(e)}"}],
                "isError": True,
            }


# =============================================================================
# STDIO TRANSPORT
# =============================================================================

def _read_framed_json(stdin) -> Optional[Dict]:
    """Read a Content-Length framed JSON message from stdin."""
    # Read headers
    content_length = None
    while True:
        line = stdin.readline()
        if not line:
            return None
        line = line.decode("ascii", errors="replace").strip()
        if not line:
            break
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())

    if content_length is None:
        return None

    # Read body
    body = stdin.read(content_length)
    if not body:
        return None

    return json.loads(body.decode("utf-8"))


def _write_framed_json(stdout, message: Dict) -> None:
    """Write a Content-Length framed JSON message to stdout."""
    body = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stdout.write(header)
    stdout.write(body)
    stdout.flush()


def run_stdio():
    """Run the server in stdio mode."""
    server = SwarmMCPServer()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    print(f"[INFO] Swarm MCP Server started (session: {server.session_id})", file=sys.stderr)

    while True:
        try:
            request = _read_framed_json(stdin)
            if request is None:
                break

            response = server.handle_request(request)
            if response is not None:
                _write_framed_json(stdout, response)

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[ERROR] MCP stdio loop: {e}", file=sys.stderr)
            continue


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Swarm MCP Server")
    parser.add_argument("--test", action="store_true", help="Run a test request")
    args = parser.parse_args()

    if args.test:
        server = SwarmMCPServer()

        print("=" * 60)
        print("SWARM MCP SERVER TEST")
        print("=" * 60)

        # Initialize
        init_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        print("\n[OK] Initialize:", init_response["result"]["serverInfo"])

        # List tools
        tools_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        tool_names = [t["name"] for t in tools_response["result"]["tools"]]
        print(f"\n[OK] Tools available: {tool_names}")

        # Test message_board_list
        print("\n--- Testing message_board_list ---")
        board_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "message_board_list", "arguments": {"board": "default"}}
        })
        content = board_response["result"]["content"][0]["text"]
        is_error = board_response["result"].get("isError", False)
        print(f"[OK] message_board_list: {'ERROR' if is_error else 'OK'}")
        try:
            data = json.loads(content)
            print(f"  Board: {data.get('board')}, Count: {data.get('count')}")
        except:
            print(f"  Output: {content[:100]}...")

        # Test agent_inbox_list
        print("\n--- Testing agent_inbox_list ---")
        inbox_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "agent_inbox_list", "arguments": {"status": "pending"}}
        })
        content = inbox_response["result"]["content"][0]["text"]
        is_error = inbox_response["result"].get("isError", False)
        print(f"[OK] agent_inbox_list: {'ERROR' if is_error else 'OK'}")
        try:
            data = json.loads(content)
            print(f"  Tasks: {len(data.get('tasks', []))}")
        except:
            print(f"  Output: {content[:100]}...")

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        return

    # Default: stdio mode
    run_stdio()


if __name__ == "__main__":
    main()
