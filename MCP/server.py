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
            # Read tools
            "cortex_query": self._tool_cortex_query,
            "context_search": self._tool_context_search,
            "context_review": self._tool_context_review,
            "canon_read": self._tool_canon_read,
            # Write tools
            "skill_run": self._tool_skill_run,
            "pack_validate": self._tool_pack_validate,
            # Governance tools
            "critic_run": self._tool_critic_run,
            "adr_create": self._tool_adr_create,
            "commit_ceremony": self._tool_commit_ceremony,
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

        # Static file map
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
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "CONTEXT" / "query-context.py"),
             "--type", record_type, "--json"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        content = result.stdout if result.returncode == 0 else "[]"
        return {
            "contents": [{
                "uri": f"ags://context/{record_type}",
                "mimeType": "application/json",
                "text": content
            }]
        }
    
    def _dynamic_cortex_resource(self) -> Dict:
        """Generate dynamic cortex index resource."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "CORTEX" / "query.py"), "--json"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        content = result.stdout if result.returncode == 0 else "{}"
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
        """Query the cortex using CORTEX/query.py."""
        try:
            import subprocess
            query = args.get("query", "")
            entity_type = args.get("type", "")
            
            cmd = [sys.executable, str(PROJECT_ROOT / "CORTEX" / "query.py")]
            
            # Build command based on arguments
            if query:
                cmd.extend(["--find", query])
            if entity_type and entity_type != "all":
                cmd.extend(["--type", entity_type])
            
            # If no specific query, list all
            if not query and not entity_type:
                cmd.append("--list")
            
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
                        "text": result.stdout if result.stdout else "No results found."
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Cortex query failed: {result.stderr}"
                    }],
                    "isError": True
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
        """Search context records using CONTEXT/query-context.py."""
        try:
            import subprocess
            cmd = [sys.executable, str(PROJECT_ROOT / "CONTEXT" / "query-context.py")]
            
            # Add arguments
            if args.get("query"):
                cmd.append(args["query"])
            if args.get("tags"):
                for tag in args["tags"]:
                    cmd.extend(["--tag", tag])
            if args.get("status"):
                cmd.extend(["--status", args["status"]])
            if args.get("type"):
                cmd.extend(["--type", args["type"]])
            
            # Always use JSON output
            cmd.append("--json")
            
            # If no filters, list all
            if len(cmd) == 3:  # Only script + --json
                cmd.insert(2, "--list")
            
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
                        "text": result.stdout if result.stdout else "No records found."
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Context search failed: {result.stderr}"
                    }],
                    "isError": True
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
        """Check for overdue reviews using CONTEXT/review-context.py."""
        try:
            import subprocess
            cmd = [sys.executable, str(PROJECT_ROOT / "CONTEXT" / "review-context.py")]
            
            # Add arguments
            if args.get("days"):
                cmd.extend(["--due", str(args["days"])])
            
            # Always use JSON output
            cmd.append("--json")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            if result.returncode == 0 or result.returncode == 1:  # 1 = has overdue
                return {
                    "content": [{
                        "type": "text",
                        "text": result.stdout if result.stdout else "No review data."
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Context review failed: {result.stderr}"
                    }],
                    "isError": True
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
        skill_dir = PROJECT_ROOT / "SKILLS" / skill_name
        run_script = skill_dir / "run.py"
        
        if not skill_dir.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Skill '{skill_name}' not found. Available skills: {self._list_skills()}"
                }],
                "isError": True
            }
        
        if not run_script.exists():
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
        skills_dir = PROJECT_ROOT / "SKILLS"
        skills = []
        for d in skills_dir.iterdir():
            if d.is_dir() and not d.name.startswith("_") and (d / "run.py").exists():
                skills.append(d.name)
        return ", ".join(skills)

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
            run_script = PROJECT_ROOT / "SKILLS" / "pack-validate" / "run.py"
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
                [sys.executable, str(PROJECT_ROOT / "TOOLS" / "critic.py")],
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
        decisions_dir = PROJECT_ROOT / "CONTEXT" / "decisions"
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
                [sys.executable, str(PROJECT_ROOT / "TOOLS" / "critic.py")],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            critic_passed = critic_result.returncode == 0
            
            # Run contract runner
            runner_result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "CONTRACTS" / "runner.py")],
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
                        "tool": "CONTRACTS/runner.py",
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
        print("\n✓ Initialize:", init_response["result"]["serverInfo"])

        # List tools
        tools_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        tool_names = [t["name"] for t in tools_response["result"]["tools"]]
        print(f"\n✓ Tools available: {tool_names}")

        # List resources
        resources_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        })
        print(f"\n✓ Resources available: {len(resources_response['result']['resources'])} resources")

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
        print(f"✓ cortex_query('packer'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
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
        print(f"✓ context_search(type='decisions'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
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
        print(f"✓ context_review(days=30): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
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
        print(f"✓ canon_read('CONTRACT'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")

        # Test resource reading
        print("\n--- Testing resources/read ---")
        read_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "resources/read",
            "params": {"uri": "ags://canon/genesis"}
        })
        content = read_response["result"]["contents"][0]["text"]
        print(f"✓ resources/read('ags://canon/genesis'): {len(content)} chars")

        # Test prompts/get
        print("\n--- Testing prompts/get ---")
        prompt_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "prompts/get",
            "params": {"name": "genesis"}
        })
        messages = prompt_response["result"].get("messages", [])
        print(f"✓ prompts/get('genesis'): {len(messages)} messages")

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        return

    # Default: stdio mode
    run_stdio()


if __name__ == "__main__":
    main()
