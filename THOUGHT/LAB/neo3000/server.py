import http.server
import json
import socketserver
import urllib.parse
from pathlib import Path
import os
import sys
import subprocess
import time

# Ensure CORTEX is in path for imports
REPO_ROOT = Path(__file__).resolve().parents[3] # Up from CAPABILITY/TOOLS/neo3000
sys.path.append(str(REPO_ROOT))

CORTEX_PATH = REPO_ROOT / "NAVIGATION" / "CORTEX" / "semantic"
sys.path.append(str(CORTEX_PATH))

try:
    import query
except ImportError:
    # Fallback or error
    print("WARNING: Could not import 'query' from CORTEX. Graph features will be disabled.")

# Swarm orchestrator path
SWARM_SKILL_DIR = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM"
SWARM_RUNS_DIR = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "swarm_runs"

PORT = 8000
STATIC_DIR = Path(__file__).resolve().parent / "static"

# State Store (In-memory for prototype)
LATEST_CONTENT = {
    "target": "NONE",
    "text": "NO ACTIVE NEURAL LINK FOUND. AWAITING DISTILLATION...",
    "timestamp": 0
}
PENDING_COMMANDS = []

# Swarm Observability Helpers
def is_process_alive(pid: int) -> bool:
    """Check if process with given PID is alive (Windows-compatible)."""
    if not pid:
        return False
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False

def get_swarm_runs():
    """Get all swarm runs with their metadata."""
    if not SWARM_RUNS_DIR.exists():
        return []

    runs = []
    for run_dir in SWARM_RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        registry_file = run_dir / "registry.json"
        if not registry_file.exists():
            continue

        try:
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry = json.load(f)

            # Count agent statuses
            total = len(registry.get("agents", []))
            running = 0
            exited = 0
            failed = 0

            for agent in registry.get("agents", []):
                pid = agent.get("pid")
                exit_code = agent.get("exit_code")

                if exit_code is not None:
                    if exit_code == 0:
                        exited += 1
                    else:
                        failed += 1
                elif is_process_alive(pid):
                    running += 1
                else:
                    exited += 1

            runs.append({
                "run_id": registry.get("run_id"),
                "started": registry.get("started_at"),
                "total_agents": total,
                "running": running,
                "exited": exited,
                "failed": failed,
                "path": str(run_dir)
            })
        except Exception:
            pass

    # Sort by start time, most recent first
    runs.sort(key=lambda r: r.get("started", ""), reverse=True)
    return runs

def get_swarm_agents(run_id: str):
    """Get detailed agent information for a specific run."""
    run_dir = SWARM_RUNS_DIR / run_id
    registry_file = run_dir / "registry.json"

    if not registry_file.exists():
        return None

    try:
        with open(registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)

        agents = []
        for agent in registry.get("agents", []):
            pid = agent.get("pid")
            exit_code = agent.get("exit_code")

            # Determine status
            if exit_code is not None:
                status = "failed" if exit_code != 0 else "exited"
            elif is_process_alive(pid):
                status = "running"
            else:
                status = "exited"

            # Get last log line
            log_file = run_dir / f"{agent['agent_id']}.log"
            last_log = None
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()
                        if lines:
                            last_log = lines[-1].strip()
                except Exception:
                    pass

            agents.append({
                "agent_id": agent.get("agent_id"),
                "pid": pid,
                "status": status,
                "exit_code": exit_code,
                "started": agent.get("started_at"),
                "last_log": last_log
            })

        return {
            "run_id": registry.get("run_id"),
            "started": registry.get("started_at"),
            "agents": agents
        }
    except Exception:
        return None

def get_agent_logs(run_id: str, agent_id: str, lines: int = 50):
    """Get recent log lines for an agent."""
    run_dir = SWARM_RUNS_DIR / run_id
    log_file = run_dir / f"{agent_id}.log"

    if not log_file.exists():
        return []

    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            return [line.rstrip() for line in all_lines[-lines:]]
    except Exception:
        return []

class Neo3000Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # API Handlers
        if path.startswith("/api/"):
            return self.handle_api(path, parsed_url.query)
        
        # Default to standard file serving
        return super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.startswith("/api/"):
            return self.handle_api(path, parsed_url.query)
        
        self.send_error(404, "File not found")

    def handle_api(self, path, query_str):
        params = urllib.parse.parse_qs(query_str)
        response_data = {"ok": False, "error": "Unknown endpoint"}
        status_code = 404

        try:
            if path == "/api/status":
                response_data = {
                    "ok": True, 
                    "system": "NEO3000", 
                    "version": "1.0.0", 
                    "status": "OPERATIONAL",
                    "cortex_version": query.get_metadata("cortex_version")
                }
                status_code = 200

            elif path == "/api/search":
                q = params.get("q", [""])[0]
                results = query.find_entities_containing_path(q)
                response_data = {"ok": True, "results": results}
                status_code = 200

            elif path == "/api/entity":
                eid = params.get("id", [""])[0]
                result = query.get_entity_by_id(eid)
                if result:
                    response_data = {"ok": True, "entity": result}
                    status_code = 200
                else:
                    response_data = {"ok": False, "error": "Entity not found"}
                    status_code = 404

            elif path == "/api/bridge/pending":
                # Browser calls this to see if the AI wants it to do something
                if PENDING_COMMANDS:
                    response_data = {"ok": True, "command": PENDING_COMMANDS.pop(0)}
                else:
                    response_data = {"ok": True, "command": None}
                status_code = 200

            elif path == "/api/bridge/uplink":
                # AI calls this to send a command to the browser
                if self.command == "POST":
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    PENDING_COMMANDS.append(data)
                    response_data = {"ok": True}
                    status_code = 200

            elif path == "/api/bridge/content":
                if self.command == "GET":
                    response_data = {"ok": True, "content": LATEST_CONTENT}
                    status_code = 200
                elif self.command == "POST":
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    LATEST_CONTENT["target"] = data.get("target", "UNKNOWN")
                    LATEST_CONTENT["text"] = data.get("text", "...")
                    LATEST_CONTENT["timestamp"] = data.get("timestamp", 0)
                    
                    response_data = {"ok": True}
                    status_code = 200

            elif path == "/api/constellation":
                # Generate a graph of the repository structure
                data = query.export_to_json()
                entities = data.get("entities", [])
                nodes = []
                edges = []
                folders = set()
                
                for item in entities:
                    # File node
                    nodes.append({
                        "id": item["id"],
                        "label": item["title"],
                        "group": "page",
                        "path": item["paths"]["source"]
                    })
                    
                    # Connection to folder
                    p = Path(item["paths"]["source"])
                    parts = list(p.parts)[:-1]
                    if parts:
                        parent_path = "/".join(parts)
                        if parent_path not in folders:
                            nodes.append({
                                "id": f"dir:{parent_path}",
                                "label": parts[-1],
                                "group": "folder"
                            })
                            folders.add(parent_path)
                            
                            # Connect folder to its parent if exists
                            if len(parts) > 1:
                                p_parent = "/".join(parts[:-1])
                                edges.append({"from": f"dir:{p_parent}", "to": f"dir:{parent_path}"})
                        
                        edges.append({"from": f"dir:{parent_path}", "to": item["id"]})
                    else:
                        # Root files connect to root
                        if "dir:root" not in folders:
                            nodes.append({"id": "dir:root", "label": "ROOT", "group": "folder"})
                            folders.add("dir:root")
                        edges.append({"from": "dir:root", "to": item["id"]})

                response_data = {"ok": True, "nodes": nodes, "edges": edges}
                status_code = 200

            elif path == "/api/bridge/simulate":
                # Mock a ChatGPT session for the user to see
                LATEST_CONTENT["target"] = "chatgpt.com"
                LATEST_CONTENT["text"] = "### NEURAL DISTILLATION IN PROGRESS...\n\n**Operator:** Explain the Phase 7.3 Swarm Elision.\n\n**ChatGPT:** Swarm elision is a deterministic optimization where a redundant swarm execution is skipped if its specification and component hashes match a verified proof in the top-level chain. This ensures byte-identical results without unnecessary compute cycles."
                LATEST_CONTENT["timestamp"] = 0
                response_data = {"ok": True}
                status_code = 200

            elif path == "/api/bridge/result":
                # AI calls this to read what the browser captured
                response_data = {"ok": True, "content": LATEST_CONTENT}
                status_code = 200

            elif path == "/api/swarm/runs":
                # List all swarm runs
                runs = get_swarm_runs()
                response_data = {"ok": True, "runs": runs}
                status_code = 200

            elif path == "/api/swarm/agents":
                # Get agents for a specific run
                run_id = params.get("run_id", [""])[0]
                if not run_id:
                    response_data = {"ok": False, "error": "Missing run_id parameter"}
                    status_code = 400
                else:
                    agents_data = get_swarm_agents(run_id)
                    if agents_data:
                        response_data = {"ok": True, "data": agents_data}
                        status_code = 200
                    else:
                        response_data = {"ok": False, "error": "Run not found"}
                        status_code = 404

            elif path == "/api/swarm/logs":
                # Get logs for a specific agent
                run_id = params.get("run_id", [""])[0]
                agent_id = params.get("agent_id", [""])[0]
                lines_param = params.get("lines", ["50"])[0]

                if not run_id or not agent_id:
                    response_data = {"ok": False, "error": "Missing run_id or agent_id parameter"}
                    status_code = 400
                else:
                    try:
                        lines = int(lines_param)
                    except ValueError:
                        lines = 50

                    logs = get_agent_logs(run_id, agent_id, lines)
                    response_data = {"ok": True, "logs": logs}
                    status_code = 200

        except Exception as e:
            response_data = {"ok": False, "error": str(e)}
            status_code = 500

        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode("utf-8"))

def run():
    print(f"--- NEO3000 KERNEL BOOTING ---")
    print(f"Local access: http://localhost:{PORT}")
    print(f"Static root: {STATIC_DIR}")
    
    with socketserver.TCPServer(("", PORT), Neo3000Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n--- SHUTTING DOWN NEO3000 ---")
            httpd.shutdown()

if __name__ == "__main__":
    run()
