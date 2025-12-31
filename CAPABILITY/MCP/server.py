#!/usr/bin/env python3
"""
AGS MCP Server - Stub Implementation

This is the seam implementation of the AGS MCP server.
It defines the interface but returns "not implemented" for most features.
Full implementation will be added when MCP integration is needed.

Usage:
  python CAPABILITY/MCP/server.py          # Start server (stdio mode)
  python CAPABILITY/MCP/server.py --http   # Start server (HTTP mode, not implemented)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_ROOT = PROJECT_ROOT / "CAPABILITY"
LAW_ROOT = PROJECT_ROOT / "LAW"
NAVIGATION_ROOT = PROJECT_ROOT / "NAVIGATION"

# Load schemas
SCHEMAS_DIR = Path(__file__).parent / "schemas"
# Per ADR: Logs/runs under LAW/CONTRACTS/_runs/
RUNS_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"
LOGS_DIR = LAW_ROOT / "CONTRACTS" / "_runs" / "mcp_logs"
BOARD_ROOT = LAW_ROOT / "CONTRACTS" / "_runs" / "message_board"
BOARD_ROLES_PATH = CAPABILITY_ROOT / "MCP" / "board_roles.json"
INBOX_ROOT = PROJECT_ROOT / "INBOX" / "agents" / "Local Models"


def load_schema(name: str) -> Dict:
    """Load a schema file."""
    schema_path = SCHEMAS_DIR / f"{name}.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text())
    return {}


# MCP Protocol Constants
MCP_VERSION = "2024-11-05"
SERVER_NAME = "ags-mcp-server"
SERVER_VERSION = "0.1.0"


def governed_tool(func):
    """Decorator: Run preflight + admission + critic.py before execution to enforce governance lock."""
    def wrapper(self, args: Dict) -> Dict:
        import os
        import subprocess
        # Mandatory preflight (fail-closed)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        preflight = subprocess.run(
            [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "ags.py"), "preflight"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        if preflight.returncode != 0:
            payload = ((preflight.stdout or "") + "\n" + (preflight.stderr or "")).strip()
            return {
                "content": [{
                    "type": "text",
                    "text": "⛔ PREFLIGHT BLOCKED ⛔\n\nAction blocked. Repository freshness check failed.\n\n" + (payload or "(no details)")
                }],
                "isError": True
            }

        # Mandatory admission control (fail-closed)
        intent_path = env.get("AGS_INTENT_PATH", "").strip()
        if not intent_path:
            return {
                "content": [{
                    "type": "text",
                    "text": "⛔ ADMISSION BLOCKED ⛔\n\nAction blocked. Missing AGS_INTENT_PATH for admission control."
                }],
                "isError": True
            }
        admit = subprocess.run(
            [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "ags.py"), "admit", "--intent", intent_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        if admit.returncode != 0:
            payload = ((admit.stdout or "") + "\n" + (admit.stderr or "")).strip()
            return {
                "content": [{
                    "type": "text",
                    "text": "⛔ ADMISSION BLOCKED ⛔\n\nAction blocked. Admission control rejected intent.\n\n" + (payload or "(no details)")
                }],
                "isError": True
            }
        # Exempt if checking critic itself (avoid infinite loop if critic is broken? No, critic run is separate tool)
        
        # Run critic
        
        res = subprocess.run(
            [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "critic.py")],
            capture_output=True, text=True, encoding="utf-8", errors="ignore", cwd=str(PROJECT_ROOT), env=env
        )
        
        if res.returncode != 0:
            output = (res.stdout + res.stderr).strip()
            return {
                "content": [{
                    "type": "text",
                    "text": f"⛔ GOVERNANCE LOCKDOWN ⛔\n\nAction blocked. The repository has governance violations.\nYou must fix these issues before the Agent can Act.\n\nViolations:\n{output}"
                }],
                "isError": True
            }
        return func(self, args)
    return wrapper


class AGSMCPServer:
    """AGS MCP Server implementation."""

    def __init__(self):
        import uuid
        self.tools_schema = load_schema("tools")
        self.resources_schema = load_schema("resources")
        self._initialized = False
        self.session_id = str(uuid.uuid4())

    def handle_request(self, request: Dict) -> Dict:
        """Handle a JSON-RPC 2.0 request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        # Route to handler
        handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
        }

        handler = handlers.get(method)
        if handler:
            try:
                result = handler(params)
                return self._success_response(request_id, result)
            except Exception as e:
                return self._error_response(request_id, -32603, str(e))
        else:
            return self._error_response(request_id, -32601, f"Method not found: {method}")

    def _success_response(self, request_id: Any, result: Any) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }

    def _error_response(self, request_id: Any, code: int, message: str) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

    def _audit_log(self, tool: str, args: Dict, result_type: str, result_data: Any = None, duration: float = 0.0) -> None:
        """Append a JSON record to the audit log."""
        from datetime import datetime
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_file = LOGS_DIR / "audit.jsonl"
            
            # Truncate large args for logging (e.g. file content)
            safe_args = args.copy()
            if "content" in safe_args and len(str(safe_args["content"])) > 200:
                safe_args["content"] = str(safe_args["content"])[:200] + "...(truncated)"
                
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "tool": tool,
                "arguments": safe_args,
                "status": result_type,
                "duration_ms": round(duration * 1000, 2),
                "result_summary": str(result_data)[:200] if result_data else None
            }
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Audit log failure: {e}", file=sys.stderr)

    def _load_cortex_index(self) -> Dict:
        """Load the cached cortex index (NAVIGATION/CORTEX/cortex.json)."""
        cortex_path = NAVIGATION_ROOT / "CORTEX" / "cortex.json"
        if not cortex_path.exists():
            return {}
        try:
            return json.loads(cortex_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return {}

    def _search_cortex_index(self, query: str, limit: int = 20) -> List[Dict]:
        """Simple substring search over cortex index entities."""
        data = self._load_cortex_index()
        entities = data.get("entities", []) if isinstance(data, dict) else []
        if not query:
            return entities[:limit]
        needle = query.lower()
        results = []
        for entity in entities:
            haystack = " ".join(
                str(entity.get(field, "")) for field in ("path", "title", "summary", "tags")
            ).lower()
            if needle in haystack:
                results.append(entity)
            if len(results) >= limit:
                break
        return results

    def _context_records(self, record_type: str) -> List[Dict]:
        """Collect context records from LAW/CONTEXT/<record_type>."""
        context_dir = LAW_ROOT / "CONTEXT" / record_type
        records = []
        if not context_dir.exists():
            return records
        for path in sorted(context_dir.glob("*.md")):
            title = None
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            records.append({
                "path": str(path.relative_to(PROJECT_ROOT)),
                "title": title or path.stem,
            })
        return records

    def _find_skill_dir(self, skill_name: str) -> Optional[Path]:
        """Locate a skill directory by name under CAPABILITY/SKILLS."""
        if not skill_name:
            return None
        skills_root = CAPABILITY_ROOT / "SKILLS"
        if not skills_root.exists():
            return None
        for root, _, files in os.walk(skills_root):
            if Path(root).name == skill_name and "run.py" in files:
                return Path(root)
        return None

    def _handle_initialize(self, params: Dict) -> Dict:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
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

        # Dispatch to tool handlers
        tool_handlers = {
            # Read tools
            "cortex_query": self._tool_cortex_query,
            "context_search": self._tool_context_search,
            "context_review": self._tool_context_review,
            "canon_read": self._tool_canon_read,
            "codebook_lookup": self._tool_codebook_lookup,
            # Write tools
            "skill_run": self._tool_skill_run,
            "pack_validate": self._tool_pack_validate,
            # Governance tools
            "critic_run": self._tool_critic_run,
            "adr_create": self._tool_adr_create,
            "commit_ceremony": self._tool_commit_ceremony,
            "research_cache": self._tool_research_cache,
            "message_board_list": self._tool_message_board_list,
            "message_board_write": self._tool_message_board_write,
            "agent_inbox_list": self._tool_agent_inbox_list,
            "agent_inbox_claim": self._tool_agent_inbox_claim,
            "agent_inbox_finalize": self._tool_agent_inbox_finalize,
        }

        handler = tool_handlers.get(tool_name)
        if handler:
            import time
            start_time = time.time()
            try:
                result = handler(arguments)
                duration = time.time() - start_time
                is_error = result.get("isError", False)
                self._audit_log(tool_name, arguments, "error" if is_error else "success", result, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                self._audit_log(tool_name, arguments, "crit_error", str(e), duration)
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Internal tool error: {str(e)}"
                    }],
                    "isError": True
                }
        else:
            self._audit_log(tool_name, arguments, "unknown_tool")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Unknown tool: {tool_name}"
                }],
                "isError": True
            }

    def _handle_resources_list(self, params: Dict) -> Dict:
        """List available resources."""
        resources = self.resources_schema.get("resources", [])
        return {"resources": resources}

    def _handle_resources_read(self, params: Dict) -> Dict:
        """Read a resource."""
        uri = params.get("uri", "")

        # Static file map
        uri_map = {
            "ags://canon/contract": LAW_ROOT / "CANON" / "CONTRACT.md",
            "ags://canon/invariants": LAW_ROOT / "CANON" / "INVARIANTS.md",
            "ags://canon/genesis": LAW_ROOT / "CANON" / "GENESIS.md",
            "ags://canon/versioning": LAW_ROOT / "CANON" / "VERSIONING.md",
            "ags://canon/arbitration": LAW_ROOT / "CANON" / "ARBITRATION.md",
            "ags://canon/deprecation": LAW_ROOT / "CANON" / "DEPRECATION.md",
            "ags://canon/migration": LAW_ROOT / "CANON" / "MIGRATION.md",
            "ags://maps/entrypoints": NAVIGATION_ROOT / "maps" / "ENTRYPOINTS.md",
            "ags://agents": PROJECT_ROOT / "AGENTS.md",
        }

        file_path = uri_map.get(uri)
        if file_path and file_path.exists():
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": content
                }]
            }
        
        # Dynamic resources
        elif uri == "ags://context/decisions":
            return self._dynamic_context_resource("decisions")
        elif uri == "ags://context/preferences":
            return self._dynamic_context_resource("preferences")
        elif uri == "ags://context/rejected":
            return self._dynamic_context_resource("rejected")
        elif uri == "ags://context/open":
            return self._dynamic_context_resource("open")
        elif uri == "ags://cortex/index":
            return self._dynamic_cortex_resource()
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    def _dynamic_context_resource(self, record_type: str) -> Dict:
        """Generate dynamic context resource content."""
        context_dir = LAW_ROOT / "CONTEXT" / record_type
        records = []
        if context_dir.exists():
            for path in sorted(context_dir.glob("*.md")):
                title = None
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
                records.append({
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "title": title or path.stem,
                })
        content = json.dumps(records, indent=2)
        return {
            "contents": [{
                "uri": f"ags://context/{record_type}",
                "mimeType": "application/json",
                "text": content
            }]
        }
    
    def _dynamic_cortex_resource(self) -> Dict:
        """Generate dynamic cortex index resource."""
        cortex_path = NAVIGATION_ROOT / "CORTEX" / "cortex.json"
        content = cortex_path.read_text(encoding="utf-8", errors="ignore") if cortex_path.exists() else "{}"
        return {
            "contents": [{
                "uri": "ags://cortex/index",
                "mimeType": "application/json",
                "text": content
            }]
        }

    def _handle_prompts_list(self, params: Dict) -> Dict:
        """List available prompts."""
        return {
            "prompts": [
                {
                    "name": "genesis",
                    "description": "The Genesis Prompt for AGS session bootstrapping"
                },
                {
                    "name": "commit_ceremony",
                    "description": "Checklist for the commit ceremony"
                },
                {
                    "name": "skill_template",
                    "description": "Template for creating a new Skill"
                },
                {
                    "name": "conflict_resolution",
                    "description": "Guide for resolving conflicts in Canon (Arbitration)"
                },
                {
                    "name": "deprecation_workflow",
                    "description": "Checklist for deprecating tokens or features"
                }
            ]
        }

    def _handle_prompts_get(self, params: Dict) -> Dict:
        """Get a specific prompt."""
        prompt_name = params.get("name")

        if prompt_name == "genesis":
            genesis_path = LAW_ROOT / "CANON" / "GENESIS.md"
            if genesis_path.exists():
                content = genesis_path.read_text(encoding="utf-8")
                # Extract the prompt block
                return {
                    "description": "Genesis Prompt for AGS session bootstrapping",
                    "messages": [{
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": content
                        }
                    }]
                }

        if prompt_name == "skill_template":
            skill_md = (CAPABILITY_ROOT / "SKILLS" / "_TEMPLATE" / "SKILL.md").read_text(encoding="utf-8")
            run_py = (CAPABILITY_ROOT / "SKILLS" / "_TEMPLATE" / "run.py").read_text(encoding="utf-8")
            return {
                "description": "Template for creating a new Skill",
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Create a new SKILL following this template:\n\n### SKILL.md\n{skill_md}\n\n### run.py\n{run_py}"
                    }
                }]
            }

        if prompt_name == "conflict_resolution":
            arb_path = LAW_ROOT / "CANON" / "ARBITRATION.md"
            content = arb_path.read_text(encoding="utf-8") if arb_path.exists() else "ARBITRATION.md not found."
            return {
                "description": "Guide for resolving conflicts in Canon",
                "messages": [{
                    "role": "user",
                    "content": { "type": "text", "text": content }
                }]
            }

        if prompt_name == "deprecation_workflow":
            dep_path = LAW_ROOT / "CANON" / "DEPRECATION.md"
            content = dep_path.read_text(encoding="utf-8") if dep_path.exists() else "DEPRECATION.md not found."
            return {
                "description": "Checklist for deprecating tokens or features",
                "messages": [{
                    "role": "user",
                    "content": { "type": "text", "text": content }
                }]
            }

        return {
            "description": f"Prompt '{prompt_name}' not implemented",
            "messages": []
        }

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
        from datetime import datetime, timezone
        import uuid

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
        from datetime import datetime
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
            new_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            task_file.unlink()
            
            return {"content": [{"type": "text", "text": json.dumps({"status": "success", "task_id": task_id, "path": str(new_path)})}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Claim failed: {str(e)}"}], "isError": True}

    def _tool_agent_inbox_finalize(self, args: Dict) -> Dict:
        """Finalize a task."""
        from datetime import datetime
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
            new_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            task_file.unlink()
            
            return {"content": [{"type": "text", "text": json.dumps({"status": "success", "task_id": task_id})}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Finalize failed: {str(e)}"}], "isError": True}

    # Tool implementations
    def _tool_cortex_query(self, args: Dict) -> Dict:
        """Query the cortex using the cached cortex index."""
        try:
            query = args.get("query", "")
            limit = int(args.get("limit", 20))
            results = self._search_cortex_index(query, limit=limit)
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(results, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Cortex query error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_context_search(self, args: Dict) -> Dict:
        """Search context records in LAW/CONTEXT."""
        try:
            query = (args.get("query") or "").lower()
            record_type = args.get("type")
            tag_filters = [str(tag).lower() for tag in args.get("tags", []) or []]
            status_filter = (args.get("status") or "").lower()

            record_types = [record_type] if record_type else ["decisions", "preferences", "rejected", "open"]
            results = []
            for rtype in record_types:
                records = self._context_records(rtype)
                for record in records:
                    path = PROJECT_ROOT / record["path"]
                    content = path.read_text(encoding="utf-8", errors="ignore").lower()
                    if query and query not in content:
                        continue
                    if status_filter and status_filter not in content:
                        continue
                    if tag_filters and not all(tag in content for tag in tag_filters):
                        continue
                    results.append(record)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(results, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Context search error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_context_review(self, args: Dict) -> Dict:
        """Return a stub review summary for context records."""
        try:
            days = args.get("days")
            payload = {
                "checked_days": days,
                "overdue": [],
                "upcoming": [],
            }
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(payload, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Context review error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_canon_read(self, args: Dict) -> Dict:
        """Read a canon file."""
        file_name = args.get("file", "").upper()
        canon_path = LAW_ROOT / "CANON" / f"{file_name}.md"

        if canon_path.exists():
            content = canon_path.read_text(encoding="utf-8", errors="ignore")
            return {
                "content": [{
                    "type": "text",
                    "text": content
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Canon file not found: {file_name}"
                }],
                "isError": True
            }

    @governed_tool
    def _tool_skill_run(self, args: Dict) -> Dict:
        """Execute a skill with the given input."""
        import subprocess
        import tempfile
        
        skill_name = args.get("skill", "")
        skill_input = args.get("input", {})
        
        if not skill_name:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: 'skill' parameter is required"
                }],
                "isError": True
            }
        
        # Validate skill exists
        skill_dir = self._find_skill_dir(skill_name)
        run_script = skill_dir / "run.py" if skill_dir else None

        if not skill_dir:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Skill '{skill_name}' not found. Available skills: {self._list_skills()}"
                }],
                "isError": True
            }
        
        if not run_script or not run_script.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Skill '{skill_name}' has no run.py"
                }],
                "isError": True
            }
        
        try:
            # Create temp files for input/output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
                json.dump(skill_input, f_in)
                input_path = f_in.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_out:
                output_path = f_out.name
            
            # Run the skill
            result = subprocess.run(
                [sys.executable, str(run_script), input_path, output_path],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=60  # 60 second timeout
            )
            
            # Read output
            output_content = Path(output_path).read_text() if Path(output_path).exists() else "{}"
            
            # Clean up
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
            
            if result.returncode == 0:
                return {
                    "content": [{
                        "type": "text",
                        "text": output_content
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Skill failed (exit {result.returncode}):\n{result.stderr}\n\nOutput:\n{output_content}"
                    }],
                    "isError": True
                }
        except subprocess.TimeoutExpired:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: Skill execution timed out (60s limit)"
                }],
                "isError": True
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Skill execution error: {str(e)}"
                }],
                "isError": True
            }
    
    def _list_skills(self) -> str:
        """List available skills."""
        skills_root = CAPABILITY_ROOT / "SKILLS"
        skills = []
        if skills_root.exists():
            for root, _, files in os.walk(skills_root):
                if "run.py" in files and "SKILL.md" in files:
                    rel = Path(root).relative_to(skills_root).as_posix()
                    skills.append(rel)
        return ", ".join(sorted(skills))

    @governed_tool
    def _tool_pack_validate(self, args: Dict) -> Dict:
        """Validate a memory pack."""
        import subprocess
        import tempfile
        
        pack_path = args.get("pack_path", "")
        
        if not pack_path:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: 'pack_path' parameter is required"
                }],
                "isError": True
            }
        
        try:
            # Create temp files for input/output
            skill_input = {"pack_path": pack_path}
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
                json.dump(skill_input, f_in)
                input_path = f_in.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_out:
                output_path = f_out.name
            
            # Run pack-validate skill
            skill_dir = self._find_skill_dir("pack-validate")
            run_script = skill_dir / "run.py" if skill_dir else None
            if not run_script or not run_script.exists():
                return {
                    "content": [{
                        "type": "text",
                        "text": "Error: pack-validate skill not found."
                    }],
                    "isError": True
                }
            result = subprocess.run(
                [sys.executable, str(run_script), input_path, output_path],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=60
            )
            
            # Read output
            output_content = Path(output_path).read_text() if Path(output_path).exists() else "{}"
            
            # Clean up
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
            
            return {
                "content": [{
                    "type": "text",
                    "text": output_content
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Pack validation error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_critic_run(self, args: Dict) -> Dict:
        """Run TOOLS/critic.py to check governance compliance."""
        import subprocess
        
        try:
            result = subprocess.run(
                [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "critic.py")],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "passed": passed,
                        "output": output.strip(),
                        "exit_code": result.returncode
                    }, indent=2)
                }],
                "isError": not passed
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Critic run error: {str(e)}"
                }],
                "isError": True
            }

    @governed_tool
    def _tool_adr_create(self, args: Dict) -> Dict:
        """Create a new ADR with the proper template."""
        import re
        from datetime import datetime
        
        title = args.get("title", "")
        context = args.get("context", "")
        decision = args.get("decision", "")
        status = args.get("status", "proposed")
        
        if not title:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: 'title' parameter is required"
                }],
                "isError": True
            }
        
        # Find next ADR number
        decisions_dir = LAW_ROOT / "CONTEXT" / "decisions"
        existing = list(decisions_dir.glob("ADR-*.md"))
        numbers = []
        for f in existing:
            match = re.match(r"ADR-(\d+)", f.stem)
            if match:
                numbers.append(int(match.group(1)))
        next_num = max(numbers, default=0) + 1
        
        # Generate filename
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:40]
        filename = f"ADR-{next_num:03d}-{slug}.md"
        filepath = decisions_dir / filename
        
        # Generate content
        date = datetime.now().strftime("%Y-%m-%d")
        content = f"""# ADR-{next_num:03d}: {title}

**Date:** {date}
**Status:** {status}
**Tags:** 

## Context

{context if context else "[Describe the context and problem that led to this decision]"}

## Decision

{decision if decision else "[Describe the decision that was made]"}

## Consequences

[Describe the positive and negative consequences of this decision]

## Review

**Review Date:** [Set a date to revisit this decision, e.g., 6 months from now]
"""
        
        try:
            filepath.write_text(content, encoding="utf-8")
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "created": True,
                        "path": str(filepath.relative_to(PROJECT_ROOT)),
                        "adr_number": next_num,
                        "title": title,
                        "message": f"Created {filename}. Please review and fill in the remaining sections."
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"ADR creation error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_commit_ceremony(self, args: Dict) -> Dict:
        """Return the commit ceremony checklist and staged files."""
        import subprocess
        
        try:
            # Run critic
            critic_result = subprocess.run(
                [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "critic.py")],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            critic_passed = critic_result.returncode == 0
            
            # Run contract runner
            runner_result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "LAW" / "CONTRACTS" / "runner.py")],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            fixtures_passed = runner_result.returncode == 0
            
            # Get staged files
            staged_result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            staged_files = [f for f in staged_result.stdout.strip().split("\n") if f]
            
            # Get status
            status_result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            ceremony = {
                "checklist": {
                    "1_failsafe_critic": {
                        "passed": critic_passed,
                        "tool": "TOOLS/critic.py",
                        "output": critic_result.stdout.strip()[-500:] if critic_result.stdout else critic_result.stderr.strip()[-500:]
                    },
                    "2_failsafe_runner": {
                        "passed": fixtures_passed,
                        "tool": "LAW/CONTRACTS/runner.py",
                        "output": runner_result.stdout.strip()[-500:] if runner_result.stdout else runner_result.stderr.strip()[-500:]
                    },
                    "3_files_staged": len(staged_files) > 0,
                    "4_ready_for_commit": critic_passed and fixtures_passed and len(staged_files) > 0
                },
                "staged_files": staged_files,
                "staged_count": len(staged_files),
                "git_status": status_result.stdout.strip(),
                "ceremony_prompt": f"Ready for the Chunked Commit Ceremony? Shall I commit these {len(staged_files)} files?" if (critic_passed and fixtures_passed and len(staged_files) > 0) else "Ceremony cannot proceed - failsafe checks must pass and files must be staged."
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(ceremony, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Commit ceremony error: {str(e)}"
                }],
                "isError": True
            }

    @governed_tool
    def _tool_research_cache(self, args: Dict) -> Dict:
        """Access and manage the research cache via TOOLS/research_cache.py."""
        import subprocess
        
        action = args.get("action")
        url = args.get("url")
        summary = args.get("summary")
        tags = args.get("tags")
        tag_filter = args.get("filter")
        
        if not action:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: 'action' parameter is required"
                }],
                "isError": True
            }
        
        cmd = [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "research_cache.py")]
        
        if action == "lookup":
            if not url:
                return {"content": [{"type": "text", "text": "Error: 'url' required for lookup"}], "isError": True}
            cmd.extend(["--lookup", url])
        elif action == "save":
            if not url or not summary:
                return {"content": [{"type": "text", "text": "Error: 'url' and 'summary' required for save"}], "isError": True}
            cmd.extend(["--save", url, summary])
            if tags:
                cmd.extend(["--tags", tags])
        elif action == "list":
            cmd.append("--list")
            if tag_filter:
                cmd.extend(["--filter", tag_filter])
        elif action == "clear":
            cmd.append("--clear")
        else:
            return {"content": [{"type": "text", "text": f"Error: Invalid action '{action}'"}], "isError": True}
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            if result.returncode == 0:
                return {
                    "content": [{
                        "type": "text",
                        "text": result.stdout if result.stdout else "[OK]"
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Research cache error: {result.stderr}"
                    }],
                    "isError": True
                }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Exception in research cache tool: {str(e)}"
                }],
                "isError": True
            }

    def _tool_codebook_lookup(self, args: Dict) -> Dict:
        """Look up a codebook entry by ID."""
        import subprocess
        
        entry_id = args.get("id", "")
        expand = args.get("expand", False)
        list_all = args.get("list", False)
        
        cmd = [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "codebook_lookup.py")]
        
        if list_all:
            cmd.append("--list")
            cmd.append("--json")
        elif entry_id:
            cmd.append(entry_id)
            if expand:
                cmd.append("--expand")
            else:
                cmd.append("--json")
        else:
            cmd.extend(["--list", "--json"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            if result.returncode == 0:
                return {
                    "content": [{
                        "type": "text",
                        "text": result.stdout
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Codebook lookup failed: {result.stderr or result.stdout}"
                    }],
                    "isError": True
                }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Codebook lookup error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_not_implemented(self, args: Dict) -> Dict:
        """Placeholder for unimplemented tools."""
        return {
            "content": [{
                "type": "text",
                "text": "This tool is staged but not yet implemented. See MCP/MCP_SPEC.md for the implementation roadmap."
            }],
            "isError": False
        }


def run_stdio():
    """Run the server in stdio mode."""
    server = AGSMCPServer()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = server.handle_request(request)
            if response.get("result") is not None or response.get("error") is not None:
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            }
            print(json.dumps(error), flush=True)


def main():
    parser = argparse.ArgumentParser(description="AGS MCP Server")
    parser.add_argument("--http", action="store_true", help="Run in HTTP mode (not implemented)")
    parser.add_argument("--test", action="store_true", help="Run a test request")
    args = parser.parse_args()

    if args.http:
        print("HTTP mode not implemented. Use stdio mode.", file=sys.stderr)
        sys.exit(1)

    if args.test:
        # Test mode: run sample requests for all implemented tools
        server = AGSMCPServer()

        print("="*60)
        print("AGS MCP SERVER TEST")
        print("="*60)

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

        # List resources
        resources_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        })
        print(f"\n[OK] Resources available: {len(resources_response['result']['resources'])} resources")

        # Test cortex_query
        print("\n--- Testing cortex_query ---")
        cortex_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "cortex_query", "arguments": {"query": "packer"}}
        })
        content = cortex_response["result"]["content"][0]["text"]
        is_error = cortex_response["result"].get("isError", False)
        print(f"[OK] cortex_query('packer'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
        if not is_error and content:
            try:
                results = json.loads(content)
                print(f"  Found {len(results)} results")
            except:
                print(f"  Output: {content[:100]}...")

        # Test context_search
        print("\n--- Testing context_search ---")
        context_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "context_search", "arguments": {"type": "decisions"}}
        })
        content = context_response["result"]["content"][0]["text"]
        is_error = context_response["result"].get("isError", False)
        print(f"[OK] context_search(type='decisions'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
        if not is_error and content:
            try:
                results = json.loads(content)
                print(f"  Found {len(results)} records")
            except:
                print(f"  Output: {content[:100]}...")

        # Test context_review
        print("\n--- Testing context_review ---")
        review_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "context_review", "arguments": {"days": 30}}
        })
        content = review_response["result"]["content"][0]["text"]
        is_error = review_response["result"].get("isError", False)
        print(f"[OK] context_review(days=30): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
        if not is_error and content:
            try:
                results = json.loads(content)
                overdue = len(results.get("overdue", []))
                upcoming = len(results.get("upcoming", []))
                print(f"  Overdue: {overdue}, Upcoming: {upcoming}")
            except:
                print(f"  Output: {content[:100]}...")

        # Test canon_read
        print("\n--- Testing canon_read ---")
        canon_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "canon_read", "arguments": {"file": "CONTRACT"}}
        })
        content = canon_response["result"]["content"][0]["text"]
        is_error = canon_response["result"].get("isError", False)
        print(f"[OK] canon_read('CONTRACT'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")

        # Test resource reading
        print("\n--- Testing resources/read ---")
        read_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "resources/read",
            "params": {"uri": "ags://canon/genesis"}
        })
        content = read_response["result"]["contents"][0]["text"]
        print(f"[OK] resources/read('ags://canon/genesis'): {len(content)} chars")

        # Test prompts/get
        print("\n--- Testing prompts/get ---")
        prompt_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "prompts/get",
            "params": {"name": "genesis"}
        })
        messages = prompt_response["result"].get("messages", [])
        print(f"[OK] prompts/get('genesis'): {len(messages)} messages")

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        return

    # Default: stdio mode
    run_stdio()


if __name__ == "__main__":
    main()
