#!/usr/bin/env python3
"""
Terminal Sharing MCP Server (Experimental)

Extracted from CAPABILITY/MCP/server.py on 2026-01-30.
Originally ported from CAT LAB server_CATDPT.py.

Purpose: Bidirectional terminal visibility between human and AI agents.

Status: EXPERIMENTAL - NOT PRODUCTION READY
- Has bugs (module-level functions reference 'self' which doesn't exist)
- Never actually used (storage directory was always empty)
- Needs architecture work for real-time polling/push

To use this in the future:
1. Fix the _ensure_terminals_dir() and _atomic_write_jsonl() calls
2. Implement proper real-time notification mechanism
3. Test with actual terminal sessions
4. Graduate to CAPABILITY/MCP when ready
"""

import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAW_ROOT = PROJECT_ROOT / "LAW"
TERMINALS_DIR = LAW_ROOT / "CONTRACTS" / "_runs" / "terminals"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_terminals_dir():
    """Ensure terminals directory exists."""
    TERMINALS_DIR.mkdir(parents=True, exist_ok=True)


def _terminal_path(terminal_id: str) -> Path:
    """Get path to terminal session file."""
    safe_id = re.sub(r'[^\w\-]', '_', terminal_id)
    return TERMINALS_DIR / f"{safe_id}.jsonl"


def _terminal_meta_path(terminal_id: str) -> Path:
    """Get path to terminal metadata file."""
    safe_id = re.sub(r'[^\w\-]', '_', terminal_id)
    return TERMINALS_DIR / f"{safe_id}.meta.json"


def _atomic_write_jsonl(path: Path, line: str) -> bool:
    """Append a JSON line to file atomically."""
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        return True
    except IOError:
        return False


def _read_jsonl_streaming(path: Path, filter_fn=None, limit: int = 50):
    """Read JSONL file with optional filtering and limit."""
    if not path.exists():
        return
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            try:
                entry = json.loads(line.strip())
                if filter_fn is None or filter_fn(entry):
                    yield entry
                    count += 1
            except json.JSONDecodeError:
                continue


# =============================================================================
# TERMINAL SHARING FUNCTIONS
# =============================================================================

def terminal_register(terminal_id: str, owner: str, cwd: str) -> Dict:
    """Register a terminal for sharing."""
    _ensure_terminals_dir()
    meta_path = _terminal_meta_path(terminal_id)
    session_path = _terminal_path(terminal_id)

    session = {
        "terminal_id": terminal_id,
        "owner": owner,
        "cwd": cwd,
        "created": datetime.now().isoformat(),
        "status": "active",
        "visible_to": ["human", "antigravity", "claude", "gemini", "grok"]
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(session, f, indent=2)
    session_path.touch()

    return {"status": "success", "session": session}


def terminal_log_command(
    terminal_id: str,
    command: str,
    executor: str,
    output: Optional[str] = None,
    exit_code: Optional[int] = None
) -> Dict:
    """Log a command executed in a terminal."""
    _ensure_terminals_dir()
    meta_path = _terminal_meta_path(terminal_id)
    session_path = _terminal_path(terminal_id)

    if not meta_path.exists():
        return {"status": "error", "message": f"Terminal {terminal_id} not registered"}

    entry = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "executor": executor,
        "output": output,
        "exit_code": exit_code
    }

    success = _atomic_write_jsonl(session_path, json.dumps(entry))
    if not success:
        return {"status": "error", "message": "Failed to write command"}

    return {"status": "success", "terminal_id": terminal_id, "command_logged": command}


def terminal_get_output(terminal_id: str, limit: int = 50, since: Optional[str] = None) -> Dict:
    """Retrieve commands and output from a terminal."""
    _ensure_terminals_dir()
    meta_path = _terminal_meta_path(terminal_id)
    session_path = _terminal_path(terminal_id)

    if not meta_path.exists():
        return {"status": "error", "message": f"Terminal {terminal_id} not found"}

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    commands = []
    if session_path.exists():
        def filter_fn(entry):
            return entry.get("timestamp", "") > since if since else True
        commands = list(_read_jsonl_streaming(session_path, filter_fn=filter_fn, limit=limit))

    return {"status": "success", "terminal_id": terminal_id, "owner": meta.get("owner"), "commands": commands}


def terminal_list() -> Dict:
    """List all registered terminals."""
    _ensure_terminals_dir()
    terminals = []
    for meta_file in TERMINALS_DIR.glob("*.meta.json"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                terminals.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue
    return {"status": "success", "count": len(terminals), "terminals": terminals}


# =============================================================================
# MCP SERVER (Minimal - stdio mode)
# =============================================================================

TOOLS_SCHEMA = {
    "terminal_register": {
        "name": "terminal_register",
        "description": "Register a shared terminal for bidirectional visibility between human and AI",
        "inputSchema": {
            "type": "object",
            "properties": {
                "terminal_id": {"type": "string", "description": "Unique identifier for the terminal"},
                "owner": {"type": "string", "description": "Who owns this terminal (e.g., 'human', 'antigravity')"},
                "cwd": {"type": "string", "description": "Current working directory"}
            }
        }
    },
    "terminal_log": {
        "name": "terminal_log",
        "description": "Log a command executed in a shared terminal",
        "inputSchema": {
            "type": "object",
            "properties": {
                "terminal_id": {"type": "string", "description": "Terminal session ID"},
                "command": {"type": "string", "description": "The command that was executed"},
                "executor": {"type": "string", "description": "Who executed it"},
                "output": {"type": "string", "description": "Command output (optional)"},
                "exit_code": {"type": "integer", "description": "Exit code (optional)"}
            },
            "required": ["terminal_id", "command"]
        }
    },
    "terminal_get": {
        "name": "terminal_get",
        "description": "Get commands from a shared terminal",
        "inputSchema": {
            "type": "object",
            "properties": {
                "terminal_id": {"type": "string", "description": "Terminal session ID"},
                "limit": {"type": "integer", "default": 50, "description": "Maximum commands to return"},
                "since": {"type": "string", "description": "ISO timestamp - only return commands after this time"}
            },
            "required": ["terminal_id"]
        }
    },
    "terminal_list": {
        "name": "terminal_list",
        "description": "List all registered shared terminals",
        "inputSchema": {"type": "object", "properties": {}}
    }
}


class TerminalSharingServer:
    """Minimal MCP server for terminal sharing tools."""

    def __init__(self):
        self.session_id = str(uuid.uuid4())

    def handle_request(self, request: Dict) -> Dict:
        """Handle an MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        if method == "initialize":
            return self._make_response(req_id, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "terminal-sharing-server", "version": "0.1.0"},
                "capabilities": {"tools": {}}
            })
        elif method == "tools/list":
            return self._make_response(req_id, {"tools": list(TOOLS_SCHEMA.values())})
        elif method == "tools/call":
            return self._handle_tool_call(req_id, params)
        else:
            return self._make_error(req_id, -32601, f"Unknown method: {method}")

    def _handle_tool_call(self, req_id, params: Dict) -> Dict:
        """Handle a tool call."""
        tool_name = params.get("name")
        args = params.get("arguments", {})

        handlers = {
            "terminal_register": lambda a: terminal_register(
                a.get("terminal_id", f"term-{uuid.uuid4().hex[:8]}"),
                a.get("owner", "human"),
                a.get("cwd", str(PROJECT_ROOT))
            ),
            "terminal_log": lambda a: terminal_log_command(
                a.get("terminal_id"), a.get("command"), a.get("executor", "unknown"),
                a.get("output"), a.get("exit_code")
            ),
            "terminal_get": lambda a: terminal_get_output(
                a.get("terminal_id"), a.get("limit", 50), a.get("since")
            ),
            "terminal_list": lambda a: terminal_list()
        }

        handler = handlers.get(tool_name)
        if not handler:
            return self._make_error(req_id, -32602, f"Unknown tool: {tool_name}")

        try:
            result = handler(args)
            return self._make_response(req_id, {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
            })
        except Exception as e:
            return self._make_response(req_id, {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            })

    def _make_response(self, req_id, result: Dict) -> Dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _make_error(self, req_id, code: int, message: str) -> Dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def main():
    """Run the terminal sharing MCP server in stdio mode."""
    server = TerminalSharingServer()

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick smoke test
        print("Terminal Sharing Server - Smoke Test")
        print("=" * 40)

        # Test register
        result = terminal_register("test-term", "human", "/tmp")
        print(f"Register: {result['status']}")

        # Test list
        result = terminal_list()
        print(f"List: {result['count']} terminals")

        print("=" * 40)
        print("OK")
        return

    # Stdio mode
    print("Terminal Sharing MCP Server (experimental)", file=sys.stderr)
    print("Waiting for JSON-RPC requests on stdin...", file=sys.stderr)

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = server.handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            continue


if __name__ == "__main__":
    main()
