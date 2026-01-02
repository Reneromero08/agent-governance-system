#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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

    write_json(output_path, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
