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
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

# Write enforcement
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
# Shared repo-root resolution
from CAPABILITY.PRIMITIVES.paths import repo_root as _repo_root
# Import modular components (primitives and validation)
from .primitives import (
    lock_file as _lock_file,
    unlock_file as _unlock_file,
    atomic_write_jsonl as _atomic_write_jsonl,
    atomic_rewrite_jsonl as _atomic_rewrite_jsonl,
    read_jsonl_streaming as _read_jsonl_streaming,
    validate_task_state_transition as _validate_task_state_transition,
    validate_task_spec as _validate_task_spec,
    compute_hash as _compute_hash,
    validate_against_schema as _validate_against_schema,
    get_validator_build_id,
    clamp_limit,
    VALIDATOR_SEMVER,
    SUPPORTED_VALIDATOR_SEMVERS,
    TASK_STATES,
    MAX_RESULTS_PER_PAGE,
)
from .audit import SessionAuditTracker
from .protocol import _read_message, _write_framed_json
from .validation import (
    is_path_under_root as _is_path_under_root,
    validate_single_path as _validate_single_path,
    check_containment_overlap as _check_containment_overlap,
    validate_jobspec_paths as _validate_jobspec_paths,
    verify_post_run_outputs as _verify_post_run_outputs,
    generate_output_hashes as _generate_output_hashes,
    verify_spectrum02_bundle,
    DURABLE_ROOTS,
    CATALYTIC_ROOTS,
    FORBIDDEN_ROOTS,
    CONTRACTS_DIR,
    SKILLS_DIR,
)

PROJECT_ROOT = _repo_root()
CAPABILITY_ROOT = PROJECT_ROOT / "CAPABILITY"
LAW_ROOT = PROJECT_ROOT / "LAW"
NAVIGATION_ROOT = PROJECT_ROOT / "NAVIGATION"

# Load schemas
SCHEMAS_DIR = Path(__file__).parent / "schemas"
# Per ADR: Logs/runs under LAW/CONTRACTS/_runs/
RUNS_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"
LOGS_DIR = LAW_ROOT / "CONTRACTS" / "_runs" / "mcp_logs"
# Rotate audit.jsonl past this size (bytes); override via env for testing.
AUDIT_LOG_MAX_BYTES = int(os.environ.get("AGS_AUDIT_LOG_MAX_BYTES", str(5 * 1024 * 1024)))

def load_schema(name: str) -> Dict:
    """Load a schema file."""
    schema_path = SCHEMAS_DIR / f"{name}.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
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
            [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "governance" / "critic.py")],
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
        import atexit
        self.tools_schema = load_schema("tools")
        self.resources_schema = load_schema("resources")
        # Tool name -> handler registry; verify_governance.py asserts this
        # stays in sync with schemas/tools.json and the README.
        self.tool_handlers = self._build_tool_registry()
        self._initialized = False
        self.session_id = str(uuid.uuid4())
        # Auto-generate session intent for admission control if not set via env
        if not os.environ.get("AGS_INTENT_PATH", "").strip():
            intent_dir = RUNS_DIR / "intent" / self.session_id
            intent_dir.mkdir(parents=True, exist_ok=True)
            intent_path_file = intent_dir / "intent.json"
            intent_data = {
                "mode": "artifact-only",
                "paths": {
                    "read": [],
                    "write": [str(RUNS_DIR.relative_to(PROJECT_ROOT).as_posix())],
                },
                "allow_repo_write": False,
            }
            import json as _json
            intent_path_file.write_text(_json.dumps(intent_data, indent=2), encoding="utf-8")
            os.environ["AGS_INTENT_PATH"] = str(intent_path_file)
            print(f"[INFO] Auto-generated session intent: {intent_path_file}", file=sys.stderr)
        # Semantic adapter is lazy-initialized on first semantic tool call.
        self.semantic_adapter = None
        self.semantic_available = False
        self._semantic_init_attempted = False
        # Session auditor for ELO file/symbol tracking (E.1.2)
        self.audit_tracker = SessionAuditTracker(
            capability_root=CAPABILITY_ROOT,
            navigation_root=NAVIGATION_ROOT,
            project_root=PROJECT_ROOT,
            canon_resolver=self._resolve_canon_file,
        )
        # Register atexit handler to end session cleanly
        atexit.register(self.audit_tracker.end_session)
        # Write enforcement
        self.writer = GuardedWriter(
            project_root=PROJECT_ROOT,
            tmp_roots=[
                "LAW/CONTRACTS/_runs/_tmp",
                "CAPABILITY/PRIMITIVES/_scratch",
                "NAVIGATION/CORTEX/_generated/_tmp",
            ],
            durable_roots=[
                "LAW/CONTRACTS/_runs",
                "NAVIGATION/CORTEX/_generated",
            ],
            exclusions=[
                "LAW/CANON",
                "AGENTS.md",
                "BUILD",
                ".git",
            ],
        )
        # This writer only handles operational logs (audit.jsonl) under the
        # durable roots; the gate must be open for the lifetime of the server
        # or every audit write is rejected with FIREWALL_TMP_WRITE_WRONG_DOMAIN.
        self.writer.open_commit_gate()






    def _ensure_semantic_adapter(self) -> None:
        """Lazy initialization of the semantic adapter.

        VS Code/Antigravity expect the server to respond quickly to `initialize`.
        Any heavy/optional init must be deferred until a semantic tool is actually called.

        ELO Integration: Passes session_id to adapter for search logging.
        """
        if getattr(self, "_semantic_init_attempted", False):
            return

        self._semantic_init_attempted = True
        try:
            try:
                from .semantic_adapter import SemanticMCPAdapter
            except Exception:
                from semantic_adapter import SemanticMCPAdapter  # type: ignore

            # Pass session_id for ELO search logging
            adapter = SemanticMCPAdapter(session_id=self.session_id)
            init_result = adapter.initialize()
            self.semantic_adapter = adapter
            self.semantic_available = True
            elo_status = "enabled" if init_result.get("elo_available") else "disabled"
            print(f"[INFO] Semantic adapter initialized (ELO: {elo_status})", file=sys.stderr)
        except Exception as e:
            self.semantic_adapter = None
            self.semantic_available = False
            print(f"[INFO] Semantic adapter unavailable: {e}", file=sys.stderr)
    def handle_request(self, request: Dict) -> Optional[Dict]:
        """Handle a JSON-RPC 2.0 request.

        Important: notifications (requests without an `id`, or with `id: null`) must not
        receive a response. Some MCP clients validate strictly and will error if we
        reply with `id: null`.
        """
        method = request.get("method", "") or ""
        params = request.get("params", {}) or {}

        has_id = ("id" in request) and (request.get("id") is not None)
        request_id = request.get("id") if has_id else None

        handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,  # legacy alias
            "notifications/initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
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



    def _rotate_audit_log(self, log_file: Path) -> None:
        """Size-based rotation: audit.jsonl -> audit.jsonl.1 (replacing old .1).

        Failure must never block a tool call; worst case rotation is skipped
        and retried on the next audit write.
        """
        try:
            try:
                size = log_file.stat().st_size
            except OSError:
                return
            if size <= AUDIT_LOG_MAX_BYTES:
                return
            backup = log_file.parent / (log_file.name + ".1")
            if backup.exists():
                # Windows Path.rename cannot overwrite an existing target
                self.writer.unlink(backup)
            self.writer.safe_rename(log_file, backup)
        except Exception as e:
            print(f"Audit log rotation skipped: {e}", file=sys.stderr)

    def _audit_log(self, tool: str, args: Dict, result_type: str, result_data: Any = None, duration: float = 0.0) -> None:
        """Append a JSON record to the audit log."""
        from datetime import datetime
        try:
            self.writer.mkdir_durable(LOGS_DIR, parents=True, exist_ok=True)
            log_file = LOGS_DIR / "audit.jsonl"
            self._rotate_audit_log(log_file)
            
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

    def _resolve_canon_file(self, file_name: str) -> Optional[Path]:
        """Resolve a bare canon name (e.g. CONTRACT) to its file path.

        Canon files live in bucket subdirectories under LAW/CANON (see
        canon.json). Resolution order: canon.json index, then flat path,
        then one-level bucket glob.
        """
        canon_dir = (LAW_ROOT / "CANON").resolve()
        index_path = canon_dir / "canon.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                for bucket in data.get("buckets", {}).values():
                    for rel in bucket.get("files", []):
                        if Path(rel).stem.upper() == file_name:
                            candidate = (canon_dir / rel).resolve()
                            if _is_path_under_root(candidate, canon_dir) and candidate.exists():
                                return candidate
            except (json.JSONDecodeError, OSError):
                pass
        flat = (canon_dir / f"{file_name}.md").resolve()
        if _is_path_under_root(flat, canon_dir) and flat.exists():
            return flat
        for candidate in sorted(canon_dir.glob(f"*/{file_name}.md")):
            candidate = candidate.resolve()
            if _is_path_under_root(candidate, canon_dir):
                return candidate
        return None

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

    def _build_tool_registry(self) -> Dict[str, Any]:
        """Single source of truth mapping tool names to handlers.

        Must stay in sync with schemas/tools.json definitions; enforced by
        the drift checks in verify_governance.py.
        """
        return {
            # Read tools
            "context_search": self._tool_context_search,
            "context_review": self._tool_context_review,
            "canon_read": self._tool_canon_read,
            "codebook_lookup": self._tool_codebook_lookup,
            # Cassette network (unified query interface)
            "cassette_network_query": self._tool_cassette_network_query,
            "skill_discovery": self._tool_skill_discovery,
            "find_related": self._tool_find_related,
            "semantic_stats": self._tool_semantic_stats,
            "memory": self._tool_memory,
            # Write tools
            "skill_run": self._tool_skill_run,
            # Session info tool
            "session_info": self._tool_session_info,
        }

    def _handle_tools_call(self, params: Dict) -> Dict:
        """Call a tool."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        handler = self.tool_handlers.get(tool_name)
        if handler:
            import time
            start_time = time.time()
            try:
                result = handler(arguments)
                duration = time.time() - start_time
                is_error = result.get("isError", False)
                self._audit_log(tool_name, arguments, "error" if is_error else "success", result, duration)
                # E.1.2: Track file/symbol/search access for ELO
                if not is_error:
                    self.audit_tracker.track_tool_access(tool_name, arguments, result)
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

        # Canon resources resolve through canon.json (bucket layout)
        canon_uri_names = {
            "ags://canon/contract": "CONTRACT",
            "ags://canon/invariants": "INVARIANTS",
            "ags://canon/genesis": "GENESIS",
            "ags://canon/versioning": "VERSIONING",
            "ags://canon/arbitration": "ARBITRATION",
            "ags://canon/deprecation": "DEPRECATION",
            "ags://canon/migration": "MIGRATION",
        }

        # Static file map
        uri_map = {
            "ags://maps/entrypoints": NAVIGATION_ROOT / "maps" / "ENTRYPOINTS.md",
            "ags://agents": PROJECT_ROOT / "AGENTS.md",
        }

        if uri in canon_uri_names:
            file_path = self._resolve_canon_file(canon_uri_names[uri])
        else:
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
            genesis_path = self._resolve_canon_file("GENESIS")
            if genesis_path:
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
            arb_path = self._resolve_canon_file("ARBITRATION")
            content = arb_path.read_text(encoding="utf-8") if arb_path else "ARBITRATION.md not found."
            return {
                "description": "Guide for resolving conflicts in Canon",
                "messages": [{
                    "role": "user",
                    "content": { "type": "text", "text": content }
                }]
            }

        if prompt_name == "deprecation_workflow":
            dep_path = self._resolve_canon_file("DEPRECATION")
            content = dep_path.read_text(encoding="utf-8") if dep_path else "DEPRECATION.md not found."
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

    # Tool implementations
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
        """Check context records for overdue/upcoming reviews."""
        try:
            from datetime import datetime
            from CAPABILITY.TOOLS.utilities.review_context import check_reviews

            days = clamp_limit(args.get("days"), default=30, maximum=3650)
            results = check_reviews(warn_days=days)
            now = datetime.now()

            def _serialize(record):
                item = {
                    "id": record.id,
                    "title": record.title,
                    "type": record.record_type,
                    "review": record.review.strftime("%Y-%m-%d") if record.review else None,
                    "path": str(Path(record.path).relative_to(PROJECT_ROOT).as_posix()),
                }
                if record.review:
                    delta = (now - record.review).days
                    item["days_overdue" if delta >= 0 else "days_until"] = abs(delta)
                return item

            payload = {
                "checked_days": days,
                "overdue": [_serialize(r) for r in results["overdue"]],
                "upcoming": [_serialize(r) for r in results["upcoming"]],
                "no_review": [_serialize(r) for r in results["no_review"]],
                "healthy_count": len(results["healthy"]),
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
        # Canon file names are bare identifiers (hyphens allowed, e.g.
        # SPECTRUM-02_RESUME_BUNDLE); anything else (separators, traversal,
        # drive letters) could escape LAW/CANON.
        if not re.fullmatch(r"[A-Z0-9_-]+", file_name):
            return {
                "content": [{
                    "type": "text",
                    "text": f"Invalid canon file name: {args.get('file', '')!r}"
                }],
                "isError": True
            }
        canon_path = self._resolve_canon_file(file_name)

        if canon_path:
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
        
        input_path = None
        output_path = None
        try:
            # Create temp files for input/output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
                json.dump(skill_input, f_in)
                input_path = f_in.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_out:
                output_path = f_out.name

            # No timeout by default (skills handle their own time budgets);
            # operators can bound execution via AGS_SKILL_TIMEOUT (seconds).
            timeout_s = None
            try:
                env_timeout = float(os.environ.get("AGS_SKILL_TIMEOUT", "0"))
                if env_timeout > 0:
                    timeout_s = env_timeout
            except ValueError:
                pass
            result = subprocess.run(
                [sys.executable, str(run_script), input_path, output_path],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=timeout_s,
            )

            # Read output
            output_content = Path(output_path).read_text() if Path(output_path).exists() else "{}"

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
                    "text": "Error: Skill execution timed out (AGS_SKILL_TIMEOUT limit)"
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
        finally:
            if input_path:
                Path(input_path).unlink(missing_ok=True)
            if output_path:
                Path(output_path).unlink(missing_ok=True)
    
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

    def _tool_codebook_lookup(self, args: Dict) -> Dict:
        """Look up a codebook entry by ID with optional stacked filtering."""
        import subprocess

        entry_id = args.get("id", "")
        expand = args.get("expand", False)
        query = args.get("query", "")
        semantic = args.get("semantic", "")
        limit = clamp_limit(args.get("limit"), default=10)
        list_all = args.get("list", False)

        cmd = [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "codebook_lookup.py")]

        if list_all:
            cmd.append("--list")
            cmd.append("--json")
        elif entry_id:
            cmd.append(entry_id)
            if query:
                cmd.extend(["--query", query, "--limit", str(limit), "--json"])
            elif semantic:
                cmd.extend(["--semantic", semantic, "--limit", str(limit), "--json"])
            elif expand:
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
                encoding='utf-8',
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

    def _tool_skill_discovery(self, args: Dict) -> Dict:
        """Find skills by semantic intent query."""
        try:
            from CAPABILITY.PRIMITIVES.skill_index import find_skills_by_intent

            query = args.get("query", "")
            top_k = clamp_limit(args.get("top_k"), default=5)
            threshold = args.get("threshold")

            if not query:
                return {
                    "content": [{"type": "text", "text": "Error: 'query' parameter is required"}],
                    "isError": True
                }

            # Perform skill discovery
            result = find_skills_by_intent(
                query=query,
                top_k=top_k,
                threshold=threshold,
                emit_receipt=True
            )

            # Format results
            if not result.get("results"):
                return {
                    "content": [{
                        "type": "text",
                        "text": f"No skills found matching query: '{query}'"
                    }]
                }

            # Build response text
            lines = [f"Found {len(result['results'])} skills matching '{query}':\n"]

            for i, skill_result in enumerate(result["results"], 1):
                skill_id = skill_result["skill_id"]
                score = skill_result["score"]
                metadata = skill_result.get("metadata", {})

                name = metadata.get("name", skill_id)
                description = metadata.get("description", "No description")
                purpose = metadata.get("purpose", "")

                lines.append(f"{i}. **{name}** (ID: `{skill_id}`, similarity: {score:.3f})")
                lines.append(f"   {description}")
                if purpose and purpose != description:
                    lines.append(f"   Purpose: {purpose[:150]}{'...' if len(purpose) > 150 else ''}")
                lines.append("")

            # Add receipt info
            if "receipt" in result:
                receipt = result["receipt"]
                lines.append(f"\n_Total candidates: {result['total_candidates']}, "
                           f"Model: {receipt.get('model_name', 'unknown')}_")

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(lines)
                }]
            }

        except FileNotFoundError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Skill index not found. Run skill embedding first: {str(e)}"
                }],
                "isError": True
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Skill discovery error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_find_related(self, args: Dict) -> Dict:
        """Find related artifacts by embedding similarity."""
        try:
            from CAPABILITY.PRIMITIVES.cross_ref_index import find_related

            artifact_id = args.get("artifact_id", "")
            top_k = clamp_limit(args.get("top_k"), default=5)
            threshold = args.get("threshold")

            if not artifact_id:
                return {
                    "content": [{"type": "text", "text": "Error: 'artifact_id' parameter is required"}],
                    "isError": True
                }

            # Perform cross-reference query
            result = find_related(
                artifact_id=artifact_id,
                top_k=top_k,
                threshold=threshold,
            )

            # Format results
            if not result.get("related"):
                return {
                    "content": [{
                        "type": "text",
                        "text": f"No related artifacts found for: '{artifact_id}'"
                    }]
                }

            # Build response text
            lines = [f"Found {len(result['related'])} related artifacts for `{artifact_id}`:\n"]

            for i, related_item in enumerate(result["related"], 1):
                rel_id = related_item["artifact_id"]
                rel_type = related_item["artifact_type"]
                rel_path = related_item["artifact_path"]
                similarity = related_item["similarity"]
                metadata = related_item.get("metadata", {})

                lines.append(f"{i}. **{rel_type}**: `{rel_path}` (similarity: {similarity:.3f})")

                # Add type-specific metadata
                if rel_type == "canon":
                    tags = metadata.get("tags")
                    if tags:
                        lines.append(f"   Tags: {tags}")
                elif rel_type == "adr":
                    title = metadata.get("title")
                    status = metadata.get("status")
                    if title:
                        lines.append(f"   Title: {title}")
                    if status:
                        lines.append(f"   Status: {status}")
                elif rel_type == "skill":
                    # Metadata for skills is in JSON, parse if present
                    if isinstance(metadata, dict) and metadata:
                        name = metadata.get("name")
                        if name:
                            lines.append(f"   Name: {name}")

                lines.append("")

            # Add stats
            total_candidates = result.get("total_candidates", 0)
            lines.append(f"\n_Total candidates: {total_candidates}_")

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(lines)
                }]
            }

        except FileNotFoundError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Cross-reference index not found. Run `build_cross_refs()` first: {str(e)}"
                }],
                "isError": True
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Cross-reference query error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_cassette_network_query(self, args: Dict) -> Dict:
        """Query the cassette network."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Cassette network not available"}],
                "isError": True
            }
        
        try:
            return self.semantic_adapter.cassette_network_query(args)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Cassette network error: {str(e)}"}],
                "isError": True
            }
    
    def _tool_semantic_stats(self, args: Dict) -> Dict:
        """Get statistics about semantic embeddings and cassette network."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }
        
        try:
            embedding_stats = self.semantic_adapter.get_embedding_stats()
            network_stats = self.semantic_adapter.get_network_status()
            
            stats = {
                "embeddings": embedding_stats,
                "network": network_stats,
                "semantic_available": True,
                "db_exists": (PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db").exists()
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(stats, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Semantic stats error: {str(e)}"}],
                "isError": True
            }
    
    def _tool_memory(self, args: Dict) -> Dict:
        """Unified memory tool with operation parameter.

        Operations:
        - save: Save a memory to the resident cassette
        - query: Query memories using semantic search
        - recall: Retrieve a full memory by its hash
        - neighbors: Find memories semantically similar to a given memory
        - stats: Get statistics about stored memories
        """
        operation = args.get("operation", "").lower()
        valid_ops = ["save", "query", "recall", "neighbors", "stats"]

        if not operation:
            return {
                "content": [{"type": "text", "text": f"Missing required 'operation' parameter. Must be one of: {', '.join(valid_ops)}"}],
                "isError": True
            }

        if operation not in valid_ops:
            return {
                "content": [{"type": "text", "text": f"Invalid operation '{operation}'. Must be one of: {', '.join(valid_ops)}"}],
                "isError": True
            }

        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }

        # Dispatch to the appropriate adapter method
        op_handlers = {
            "save": self.semantic_adapter.memory_save_tool,
            "query": self.semantic_adapter.memory_query_tool,
            "recall": self.semantic_adapter.memory_recall_tool,
            "neighbors": self.semantic_adapter.semantic_neighbors_tool,
            "stats": self.semantic_adapter.memory_stats_tool,
        }

        try:
            return op_handlers[operation](args)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory {operation} error: {str(e)}"}],
                "isError": True
            }

    def _tool_session_info(self, args: Dict) -> Dict:
        """Get information about the current MCP session including session_id for ADR-021 compliance."""
        from datetime import datetime
        import os
        
        try:
            include_audit_log = args.get("include_audit_log", False)
            limit = clamp_limit(args.get("limit"), default=10)
            
            # Basic session info
            session_info = {
                "session_id": self.session_id,
                "server_name": SERVER_NAME,
                "server_version": SERVER_VERSION,
                "mcp_version": MCP_VERSION,
                "connected_at": datetime.now().isoformat(),
                "project_root": str(PROJECT_ROOT),
                "audit_log_path": str(LOGS_DIR / "audit.jsonl"),
                "adr_021_compliant": True,
                "adr_021_note": "This session_id is automatically logged with all cortex queries and tool calls",
                "semantic_available": self.semantic_available
            }
            
            # Add semantic stats if available
            if self.semantic_available and self.semantic_adapter:
                try:
                    embedding_stats = self.semantic_adapter.get_embedding_stats()
                    session_info["semantic_stats"] = {
                        "embeddings_count": embedding_stats.get("total_embeddings", 0) if isinstance(embedding_stats, dict) else 0,
                        "semantic_ready": True
                    }
                except Exception as e:
                    print(f"[WARNING] Failed to get embedding stats: {e}", file=sys.stderr)
                    session_info["semantic_stats"] = {"semantic_ready": False}
            
            # Add audit log entries if requested
            if include_audit_log:
                audit_entries = []
                log_file = LOGS_DIR / "audit.jsonl"
                if log_file.exists():
                    lines = log_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
                    # Filter for this session_id and get most recent
                    for line in reversed(lines):
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get("session_id") == self.session_id:
                                audit_entries.append(entry)
                                if len(audit_entries) >= limit:
                                    break
                        except Exception as e:
                            print(f"[WARNING] Failed to parse audit log entry: {e}", file=sys.stderr)
                            continue
                session_info["audit_log_entries"] = audit_entries
                session_info["audit_log_count"] = len(audit_entries)
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(session_info, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Session info error: {str(e)}"
                }],
                "isError": True
            }

def run_stdio():
    """Run the server in stdio mode.

    Supports:
    - MCP/LSP Content-Length framing (VS Code, Antigravity, most MCP clients)
    - Legacy newline-delimited JSON (some simpler clients)
    """
    server = AGSMCPServer()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    mode: Optional[str] = None  # "framed" | "jsonl" (auto-detected on first request)

    while True:
        try:
            request, detected = _read_message(stdin, mode)
            if request is None:
                break
            if detected is None:
                break
            if mode is None:
                mode = detected

            if not isinstance(request, dict):
                continue

            response = server.handle_request(request)
            if response is None:
                continue

            if mode == "framed":
                _write_framed_json(stdout, response)
            else:
                stdout.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
                stdout.flush()

        except json.JSONDecodeError:
            if mode == "framed":
                _write_framed_json(stdout, {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}})
            continue
        except EOFError:
            break
        except Exception as e:
            if mode == "framed":
                _write_framed_json(stdout, {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}})
            else:
                print(f"[ERROR] MCP stdio loop: {e}", file=sys.stderr)
            continue


def main():
    parser = argparse.ArgumentParser(description="AGS MCP Server")
    parser.add_argument("--http", action="store_true", help="Run in HTTP mode (not implemented)")
    parser.add_argument("--test", action="store_true", help="Run a test request")
    args = parser.parse_args()

    if args.http:
        print("HTTP mode not implemented. Use stdio mode.", file=sys.stderr)
        sys.exit(1)

    if args.test:
        # Lazy import: selftest imports this module, so resolve it only when
        # the flag is used (the module is fully loaded by then).
        from .selftest import run_selftest
        run_selftest()
        return

    # Default: stdio mode
    run_stdio()


if __name__ == "__main__":
    main()
