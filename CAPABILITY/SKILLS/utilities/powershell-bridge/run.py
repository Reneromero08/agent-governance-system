#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any, Dict
import sys # re-import to be safe
try:
    # Add repo root to path for imports
    PROJECT_ROOT_GUESS = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT_GUESS) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_GUESS))
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any], writer: Any = None) -> None:
    if not writer:
        raise RuntimeError("GuardedWriter required for write_json")
    try:
         # Assuming path is relative to project root or absolute within project
         # If absolute, make relative
         try:
             rel_path = str(path.resolve().relative_to(writer.project_root))
         except ValueError:
             rel_path = str(path)
         
         writer.write_auto(rel_path, json.dumps(payload, indent=2, sort_keys=True))
    except Exception as e:
         print(f"Write failed: {e}")
         raise


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = load_json(input_path)

    root = Path(payload.get("repo_root")) if payload.get("repo_root") else repo_root()
    bridge_ps1 = root / "CAPABILITY" / "MCP" / "powershell_bridge.ps1"
    bridge_cfg = root / "CAPABILITY" / "MCP" / "powershell_bridge_config.json"

    output = {
        "ok": bridge_ps1.exists() and bridge_cfg.exists(),
        "paths": {
            "bridge_script": bridge_ps1.relative_to(root).as_posix(),
            "bridge_config": bridge_cfg.relative_to(root).as_posix(),
            "log_dir": "LAW/CONTRACTS/_runs/mcp_logs",
        },
        "instructions": [
            "Update CAPABILITY/MCP/powershell_bridge_config.json token.",
            "Start: powershell -ExecutionPolicy Bypass -File CAPABILITY\\MCP\\powershell_bridge.ps1",
            "Test: POST http://127.0.0.1:8765/run with X-Bridge-Token header.",
        ],
    }

    # Init writer
    try:
         from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    except ImportError:
         print("GuardedWriter import failed")
         return 1

    writer = GuardedWriter(root, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
    write_json(output_path, output, writer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
