#!/usr/bin/env python3
"""
AGS MCP Server - Stub Implementation

This is the seam implementation of the AGS MCP server.
It defines the interface but returns "not implemented" for most features.
Full implementation will be added when MCP integration is needed.

Usage:
  python MCP/server.py          # Start server (stdio mode)
  python MCP/server.py --http   # Start server (HTTP mode, not implemented)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load schemas
SCHEMAS_DIR = Path(__file__).parent / "schemas"


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


class AGSMCPServer:
    """AGS MCP Server implementation."""

    def __init__(self):
        self.tools_schema = load_schema("tools")
        self.resources_schema = load_schema("resources")
        self._initialized = False

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
            "cortex_query": self._tool_cortex_query,
            "context_search": self._tool_context_search,
            "context_review": self._tool_context_review,
            "canon_read": self._tool_canon_read,
            # Write tools return not implemented
            "skill_run": self._tool_not_implemented,
            "pack_validate": self._tool_not_implemented,
        }

        handler = tool_handlers.get(tool_name)
        if handler:
            return handler(arguments)
        else:
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

        # Map URI to file
        uri_map = {
            "ags://canon/contract": PROJECT_ROOT / "CANON" / "CONTRACT.md",
            "ags://canon/invariants": PROJECT_ROOT / "CANON" / "INVARIANTS.md",
            "ags://canon/genesis": PROJECT_ROOT / "CANON" / "GENESIS.md",
            "ags://canon/versioning": PROJECT_ROOT / "CANON" / "VERSIONING.md",
            "ags://canon/arbitration": PROJECT_ROOT / "CANON" / "ARBITRATION.md",
            "ags://canon/deprecation": PROJECT_ROOT / "CANON" / "DEPRECATION.md",
            "ags://canon/migration": PROJECT_ROOT / "CANON" / "MIGRATION.md",
            "ags://maps/entrypoints": PROJECT_ROOT / "MAPS" / "ENTRYPOINTS.md",
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
        elif uri.startswith("ags://context/") or uri.startswith("ags://cortex/"):
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps({"status": "not_implemented", "message": "Dynamic resources pending implementation"})
                }]
            }
        else:
            raise ValueError(f"Unknown resource: {uri}")

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
                    "name": "adr_template",
                    "description": "Template for creating Architecture Decision Records"
                }
            ]
        }

    def _handle_prompts_get(self, params: Dict) -> Dict:
        """Get a specific prompt."""
        prompt_name = params.get("name")

        if prompt_name == "genesis":
            genesis_path = PROJECT_ROOT / "CANON" / "GENESIS.md"
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

        return {
            "description": f"Prompt '{prompt_name}' not implemented",
            "messages": []
        }

    # Tool implementations
    def _tool_cortex_query(self, args: Dict) -> Dict:
        """Query the cortex."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "CORTEX"))
            from query import search_index
            results = search_index(args.get("query", ""))
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(results, indent=2)
                }]
            }
        except ImportError:
            return self._tool_not_implemented(args)

    def _tool_context_search(self, args: Dict) -> Dict:
        """Search context records."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "CONTEXT"))
            # Import would need adjustment for actual module name
            return self._tool_not_implemented(args)
        except Exception:
            return self._tool_not_implemented(args)

    def _tool_context_review(self, args: Dict) -> Dict:
        """Check for overdue reviews."""
        return self._tool_not_implemented(args)

    def _tool_canon_read(self, args: Dict) -> Dict:
        """Read a canon file."""
        file_name = args.get("file", "").upper()
        canon_path = PROJECT_ROOT / "CANON" / f"{file_name}.md"

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
        # Test mode: run a sample request
        server = AGSMCPServer()

        # Initialize
        init_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        print("Initialize response:", json.dumps(init_response, indent=2))

        # List tools
        tools_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        print("\nTools:", json.dumps(tools_response, indent=2))

        # List resources
        resources_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        })
        print("\nResources:", json.dumps(resources_response, indent=2))

        # Read a resource
        read_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": "ags://canon/genesis"}
        })
        print("\nGenesis content length:", len(read_response.get("result", {}).get("contents", [{}])[0].get("text", "")))

        return

    # Default: stdio mode
    run_stdio()


if __name__ == "__main__":
    main()
