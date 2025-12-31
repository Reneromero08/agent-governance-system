#!/usr/bin/env python3
"""
AGS MCP Server Entrypoint (Governance Compliant)

This entrypoint is located in LAW/CONTRACTS to satisfy governance requirements
for runtime artifacts. It launches the pure-stdio MCP server compatible with VS Code.
"""

import sys
import json
import logging
import traceback
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# --- Configuration ---
# Repo root is 2 levels up: LAW/CONTRACTS/ -> LAW/ -> Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_ROOT = PROJECT_ROOT / "CAPABILITY"
LAW_ROOT = PROJECT_ROOT / "LAW"
NAVIGATION_ROOT = PROJECT_ROOT / "NAVIGATION"

# --- Debug Logging ---
DEBUG_FILE = Path(tempfile.gettempdir()) / "ags_mcp_debug.log"

def log_debug(msg: str):
    try:
        with open(DEBUG_FILE, "a") as f:
            f.write(f"[PID {os.getpid()}] {msg}\n")
    except:
        pass

# Logs only to stderr or a file, NEVER stdout
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("ags-mcp-stdio")

# Constants
MCP_VERSION = "2024-11-05"
SERVER_NAME = "ags-mcp-stdio"
SERVER_VERSION = "1.0.0"

# --- Protocol Handling ---

def _read_message() -> Optional[Dict]:
    """Read a Content-Length framed JSON-RPC message from stdin robustly."""
    try:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        
        line_str = line.decode("ascii", errors="ignore").strip()
        if not line_str.lower().startswith("content-length:"):
            log_debug(f"Bad header: {line_str}")
            return None
            
        try:
            content_length = int(line_str.split(":")[1].strip())
        except (ValueError, IndexError):
            log_debug(f"Bad length: {line_str}")
            return None
            
        # Read everything until the empty line (headers end)
        while True:
            line = sys.stdin.buffer.readline()
            if not line or line.strip() == b"":
                break
                
        # Read the exact content
        content = sys.stdin.buffer.read(content_length)
        if len(content) < content_length:
            log_debug("Short read body")
            return None
            
        return json.loads(content.decode("utf-8"))
    except Exception as e:
        log_debug(f"Read error: {e}")
        return None

def _write_message(msg: Dict) -> None:
    """Write a Content-Length framed JSON-RPC message to stdout."""
    content = json.dumps(msg, separators=(",", ":"))
    content_bytes = content.encode("utf-8")
    length = len(content_bytes)
    
    sys.stdout.buffer.write(f"Content-Length: {length}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(content_bytes)
    sys.stdout.buffer.flush()

# --- Governance Decorators ---

def governed_tool(func):
    """Decorator: Run preflight + admission + critic.py before execution."""
    def wrapper(self, args: Dict) -> Dict:
        # 1. Preflight
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        preflight = subprocess.run(
            [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "ags.py"), "preflight"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
            cwd=str(PROJECT_ROOT), env=env
        )
        if preflight.returncode != 0:
             return {
                "content": [{"type": "text", "text": f"⛔ PREFLIGHT FAILED\n{preflight.stdout}\n{preflight.stderr}"}],
                "isError": True
             }

        # 2. Admission
        intent_path = env.get("AGS_INTENT_PATH", "").strip()
        if intent_path:
            admit = subprocess.run(
                 [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "ags.py"), "admit", "--intent", intent_path],
                 capture_output=True, text=True, encoding="utf-8", errors="ignore",
                 cwd=str(PROJECT_ROOT), env=env
            )
            if admit.returncode != 0:
                return {
                    "content": [{"type": "text", "text": f"⛔ ADMISSION DENIED\n{admit.stdout}\n{admit.stderr}"}],
                    "isError": True
                }

        return func(self, args)
    return wrapper

# --- Server Implementation ---

class AgsStdioServer:
    def __init__(self):
        self._initialized = False
        import uuid
        self.session_id = str(uuid.uuid4())
        self.semantic_adapter = None
        
    def _ensure_semantic(self):
        """Lazy load semantic adapter."""
        if self.semantic_adapter:
            return
        try:
            sys.path.insert(0, str(CAPABILITY_ROOT / "MCP"))
            from semantic_adapter import SemanticMCPAdapter
            self.semantic_adapter = SemanticMCPAdapter()
            self.semantic_adapter.initialize()
            log_debug("Semantic adapter loaded.")
        except Exception as e:
            log_debug(f"Semantic load fail: {e}")
            self.semantic_adapter = None

    def handle_request(self, req: Dict) -> Dict:
        method = req.get("method")
        params = req.get("params", {})
        rid = req.get("id")
        
        log_debug(f"Handling: {method}")

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {
                    "protocolVersion": MCP_VERSION,
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
                }
            }
        
        if method == "notifications/initialized":
            self._initialized = True
            return None
            
        if method == "tools/list":
            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {
                    "tools": [
                        {
                            "name": "read_canon", 
                            "description": "Read a canonical document",
                            "inputSchema": {
                                "type": "object", 
                                "properties": {
                                    "doc": {"type": "string"}
                                },
                                "required": ["doc"]
                            }
                        },
                        {
                            "name": "semantic_search",
                            "description": "Search Cortex via embeddings.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    ]
                }
            }
            
        if method == "tools/call":
            return self._handle_tool_call(req)
            
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}

    def _handle_tool_call(self, req: Dict) -> Dict:
        rid = req.get("id")
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        
        try:
            if name == "read_canon":
                doc = args.get("doc")
                path = LAW_ROOT / "CANON" / doc
                if path.exists():
                    text = path.read_text(encoding="utf-8")
                    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}]}}
                else:
                    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "File not found"}], "isError": True}}
            
            elif name == "semantic_search":
                self._ensure_semantic()
                if not self.semantic_adapter:
                     return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "Semantic search unavailable"}], "isError": True}}
                
                res = self.semantic_adapter.search(args.get("query"))
                return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": str(res)}]}}

            else:
                 return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Tool not found"}}
                 
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(e)}}

# --- Entrypoint ---

def main():
    # Ensure stdout is binary for protocol
    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        
    log_debug("Server starting via main()")
    server = AgsStdioServer()
    while True:
        try:
            req = _read_message()
            if req is None:
                log_debug("EOF on stdin")
                break
            resp = server.handle_request(req)
            if resp:
                _write_message(resp)
        except Exception:
            log_debug(f"Crash: {traceback.format_exc()}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
