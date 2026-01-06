"""
MCP Integration for Chat (Phase 3.3.1)

Provides secure, constrained access to MCP tools from the chat runtime.
Enforces strict allowlisting and validation before dispatching to the MCP server.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

class McpAccessError(Exception):
    """Raised when MCP tool access is denied or fails."""
    pass

class ChatToolExecutor:
    """Constrained executor for MCP tools."""
    
    # -------------------------------------------------------------------------
    # CONSTRAINT: Explicit Allowlist
    # Chat may ONLY access these tools. All others are strictly forbidden.
    # This prevents the LLM from accessing dangerous tools (e.g. terminal_bridge)
    # even if the underlying MCP server supports them.
    # -------------------------------------------------------------------------
    ALLOWED_TOOLS: Set[str] = {
        # Read / Query Tools
        "cortex_query",
        "context_search",
        "context_review",
        "canon_read",
        "codebook_lookup",
        "semantic_search",
        "semantic_stats",
        "cassette_network_query",
        
        # Memory / Research Tools
        "research_cache",
        
        # Communication Status Tools (Read-only for now)
        "agent_inbox_list",
        "message_board_list"
    }

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize the executor."""
        if repo_root is None:
            # Attempt to locate repo root relative to this file
            # catalytic_chat -> CAT_CHAT -> LAB -> THOUGHT -> REPO_ROOT (d:\CCC 2.0\AI\agent-governance-system)
            self.repo_root = Path(__file__).resolve().parents[4]
        else:
            self.repo_root = repo_root

        self._server = None

    def _get_server(self):
        """Lazy load the AGS MCP Server."""
        if self._server:
            return self._server
        
        # Ensure we can import CAPABILITY package
        root_str = str(self.repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            
        try:
            from CAPABILITY.MCP.server import AGSMCPServer
            self._server = AGSMCPServer()
            # Initialize server (protocol requirement to be ready)
            self._server.handle_request({"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": "init-1"})
        except ImportError as e:
            raise McpAccessError(f"Failed to import AGS MCP Server: {e}")
        except Exception as e:
            raise McpAccessError(f"Failed to initialize AGS MCP Server: {e}")
        
        return self._server

    def list_tools(self) -> List[Dict[str, Any]]:
        """List allowed tools available to Chat."""
        server = self._get_server()
        
        # 1. Get all tools from server
        req = {
            "jsonrpc": "2.0",
            "id": "list-tools",
            "method": "tools/list",
            "params": {}
        }
        resp = server.handle_request(req)
        
        if not resp or "result" not in resp or "tools" not in resp["result"]:
             raise McpAccessError("Failed to list tools from MCP server")
             
        all_tools = resp["result"]["tools"]
        
        # 2. Filter by ALLOWED_TOOLS
        filtered_tools = [
            t for t in all_tools 
            if t["name"] in self.ALLOWED_TOOLS
        ]
        
        return filtered_tools

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool if allowed and valid."""
        
        # 1. STRICT CONSTRAINT CHECK
        if tool_name not in self.ALLOWED_TOOLS:
            raise McpAccessError(f"Tool access denied: '{tool_name}' is not in the allowed set for Chat.")
            
        server = self._get_server()
        
        # 2. Dispatch to Server
        req = {
            "jsonrpc": "2.0",
            "id": "call-tool",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            resp = server.handle_request(req)
        except Exception as e:
             raise McpAccessError(f"Tool execution internal error: {e}")
             
        # 3. Handle Response
        if not resp:
             raise McpAccessError("No response from MCP server")
             
        if "error" in resp:
            err = resp["error"]
            # Fail closed with clear error
            raise McpAccessError(f"MCP Error {err.get('code')}: {err.get('message')}")
            
        if "result" in resp:
            # Return result (content, isError, etc.)
            return resp["result"]
            
        raise McpAccessError("Malformed response from MCP server")
