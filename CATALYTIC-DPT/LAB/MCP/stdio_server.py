#!/usr/bin/env python3
import sys
import json
import logging
import traceback
from pathlib import Path

# Add current directory to path so we can import server
sys.path.append(str(Path(__file__).parent.parent))

from MCP.server import MCPTerminalServer

# Configure logging to stderr so it doesn't interfere with stdout JSON-RPC
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp-stdio-server")

class StdIoServer:
    def __init__(self):
        self.server = MCPTerminalServer()
        self.tools = [
            {
                "name": "register_terminal",
                "description": "Register a terminal for sharing and monitoring",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "terminal_id": {"type": "string"},
                        "owner": {"type": "string"},
                        "cwd": {"type": "string"}
                    },
                    "required": ["terminal_id", "owner", "cwd"]
                }
            },
            {
                "name": "log_terminal_command",
                "description": "Log a command executed in a terminal",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "terminal_id": {"type": "string"},
                        "command": {"type": "string"},
                        "output": {"type": "string"},
                        "exit_code": {"type": "integer"}
                    },
                    "required": ["terminal_id", "command"]
                }
            },
            {
                "name": "dispatch_task",
                "description": "Dispatch a task to another agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "to_agent": {"type": "string"},
                        "task_spec": {"type": "object"}
                    },
                    "required": ["to_agent", "task_spec"]
                }
            },
            {
                "name": "get_pending_tasks",
                "description": "Get pending tasks for an agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"}
                    },
                    "required": ["agent_id"]
                }
            },
            {
                "name": "report_result",
                "description": "Report result of a task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "from_agent": {"type": "string"},
                        "status": {"type": "string"},
                        "result": {"type": "object"},
                        "errors": {"type": "array"}
                    },
                    "required": ["task_id", "from_agent", "status", "result"]
                }
            },
            {
                "name": "send_directive",
                "description": "Send a high-level directive to an agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "from_agent": {"type": "string"},
                        "to_agent": {"type": "string"},
                        "directive": {"type": "string"},
                        "metadata": {"type": "object"}
                    },
                    "required": ["from_agent", "to_agent", "directive"]
                }
            },
            {
                "name": "get_directives",
                "description": "Get pending directives for an agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"}
                    },
                    "required": ["agent_id"]
                }
            },
            {
                "name": "run_shell_command",
                "description": "Execute a shell command in the current environment",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "skill_run",
                "description": "Execute a CATALYTIC-DPT skill with input/output files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string", "description": "Skill name (e.g., 'ant-worker', 'governor')"},
                        "input": {"type": "object", "description": "Input parameters for the skill"}
                    },
                    "required": ["skill", "input"]
                }
            }
        ]

    def run(self):
        """Run the stdio loop."""
        for line in sys.stdin:
            try:
                line = line.strip()
                if not line:
                    continue
                
                request = json.loads(line)
                response = self.handle_request(request)
                
                if response:
                    print(json.dumps(response), flush=True)
                    
            except Exception as e:
                logger.error(f"Error processing line: {e}")
                logger.error(traceback.format_exc())

    def handle_request(self, request):
        method = request.get("method")
        params = request.get("params", {})
        msg_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05", # Approximate MCP version
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "Catalytic-MCP",
                        "version": "1.0.0"
                    }
                }
            }
            
        elif method == "notifications/initialized":
            # No response needed
            return None
            
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": self.tools
                }
            }
            
        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            
            try:
                result = self.call_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
        
        # Default/Ping
        elif method == "ping":
             return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {}
            }
            
        return None

    def call_tool(self, name, args):
        logger.info(f"Calling tool: {name} with args: {args}")
        
        if name == "register_terminal":
            return self.server.register_terminal(
                args.get("terminal_id"), 
                args.get("owner"), 
                args.get("cwd")
            )
        elif name == "log_terminal_command":
            return self.server.log_terminal_command(
                args.get("terminal_id"),
                args.get("command"),
                args.get("user", "unknown"),
                args.get("output", ""),
                args.get("exit_code", 0)
            )
        elif name == "dispatch_task":
            task_spec = args.get("task_spec", {})
            task_id = task_spec.get("task_id", f"task-{hash(str(task_spec)) % 100000}")
            return self.server.dispatch_task(
                task_id=task_id,
                task_spec=task_spec,
                from_agent=args.get("from_agent", "Claude"),
                to_agent=args.get("to_agent"),
                priority=args.get("priority", 5)
            )
        elif name == "get_pending_tasks":
            return self.server.get_pending_tasks(args.get("agent_id"))
        elif name == "report_result":
            return self.server.report_result(
                args.get("task_id"),
                args.get("from_agent"),
                args.get("status"),
                args.get("result"),
                args.get("errors", [])
            )
        elif name == "send_directive":
            return self.server.send_directive(
                from_level=args.get("from_agent", "Claude"),
                to_agent=args.get("to_agent"),
                directive=args.get("directive"),
                context=args.get("metadata", {})
            )
        elif name == "get_directives":
            return self.server.get_directives(args.get("agent_id"))
        elif name == "run_shell_command":
            import subprocess
            cmd = args.get("command")
            cwd = args.get("cwd")
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True
                )
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode
                }
            except Exception as e:
                return {
                    "stdout": "",
                    "stderr": str(e),
                    "exit_code": -1
                }
        elif name == "skill_run":
            import subprocess
            import tempfile

            skill_name = args.get("skill", "")
            skill_input = args.get("input", {})

            # Build skill path
            skill_dir = Path(__file__).parent.parent / "SKILLS" / skill_name
            skill_runner = skill_dir / "run.py"

            if not skill_runner.exists():
                return {
                    "status": "error",
                    "message": f"Skill '{skill_name}' not found at {skill_runner}"
                }

            # Create temp input/output files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
                json.dump(skill_input, input_file, indent=2)
                input_path = input_file.name

            output_path = input_path.replace('.json', '_output.json')

            try:
                # Execute skill
                result = subprocess.run(
                    [sys.executable, str(skill_runner), input_path, output_path],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                # Read output
                if Path(output_path).exists():
                    with open(output_path) as f:
                        output_data = json.load(f)
                else:
                    output_data = {
                        "status": "error",
                        "message": "No output file generated"
                    }

                # Cleanup
                Path(input_path).unlink(missing_ok=True)
                Path(output_path).unlink(missing_ok=True)

                return {
                    "status": "success",
                    "skill": skill_name,
                    "output": output_data,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }

            except subprocess.TimeoutExpired:
                return {
                    "status": "timeout",
                    "skill": skill_name,
                    "message": "Skill execution timed out after 120s"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "skill": skill_name,
                    "message": str(e)
                }

        raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    server = StdIoServer()
    server.run()
