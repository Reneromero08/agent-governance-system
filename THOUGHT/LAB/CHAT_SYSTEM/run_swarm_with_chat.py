#!/usr/bin/env python3
"""
Run Swarm with Catalytic Chat Integration.

Wraps SwarmRuntime to log all events to chat system.
"""

import sys
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict

# Explicitly set repo root
repo_root = Path("D:/CCC 2.0/AI/agent-governance-system")
catalytic_path = repo_root / "CATALYTIC-DPT"

# Add CATALYTIC-DPT to sys.path for internal imports
sys.path.insert(0, str(catalytic_path))

# Load swarm_runtime from absolute path
swarm_runtime_path = catalytic_path / "PIPELINES" / "swarm_runtime.py"
spec = importlib.util.spec_from_file_location("swarm_runtime", str(swarm_runtime_path))
swarm_runtime_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swarm_runtime_module)
SwarmRuntime = swarm_runtime_module.SwarmRuntime

# Add chat system path
chat_system_path = catalytic_path / "LAB" / "CHAT_SYSTEM"
sys.path.insert(0, str(chat_system_path))
os.chdir(str(chat_system_path))

from swarm_chat_logger import SwarmChatLogger


class ChatSwarmRuntime(SwarmRuntime):
    """SwarmRuntime with chat system logging."""

    def __init__(self, *, project_root: Path, runs_root: Path, session_id: str):
        """Initialize with chat logging.

        Args:
            project_root: Repository root
            runs_root: Runs directory
            session_id: Chat session ID for this swarm run
        """
        super().__init__(project_root=project_root, runs_root=runs_root)
        self.chat_logger = SwarmChatLogger(session_id=session_id, repo_root=project_root)
        self.session_id = session_id

    def run(self, *, swarm_id: str, spec_path: Path) -> Dict[str, Any]:
        """Run swarm with chat logging.

        Args:
            swarm_id: Swarm identifier
            spec_path: Path to swarm spec JSON

        Returns:
            Execution result dictionary
        """
        import json

        with open(spec_path, "r") as f:
            spec = json.load(f)

        self.chat_logger.log_swarm_start(swarm_id, spec)

        try:
            result = super().run(swarm_id=swarm_id, spec_path=spec_path)

            summary = {
                "total_nodes": len(spec.get("nodes", [])),
                "nodes_executed": result.get("nodes", 0),
                "elided": result.get("elided", False),
                "status": "success" if result.get("ok") else "failed"
            }

            self.chat_logger.log_swarm_complete(swarm_id, summary)

            return result

        except Exception as e:
            error_summary = {
                "total_nodes": len(spec.get("nodes", [])),
                "status": "error",
                "error": str(e)
            }
            self.chat_logger.log_swarm_complete(swarm_id, error_summary)
            raise


if __name__ == "__main__":
    print("Running Swarm with Catalytic Chat Integration...")

    repo_root = Path(__file__).parent.parent.parent.parent
    runs_root = repo_root / "CATALYTIC-DPT" / "_runs"

    # Example: Create a simple swarm spec
    swarm_spec = {
        "swarm_version": "1.0.0",
        "swarm_id": "test-chat-swarm",
        "nodes": [
            {
                "node_id": "node-1",
                "pipeline_id": "test-pipeline-1",
                "dependencies": []
            },
            {
                "node_id": "node-2",
                "pipeline_id": "test-pipeline-2",
                "dependencies": ["node-1"]
            }
        ]
    }

    spec_path = repo_root / "MEMORY" / "LLM_PACKER" / "_packs" / "chat" / "test_swarm_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    spec_path.write_text(json.dumps(swarm_spec, indent=2))

    print(f"Swarm spec written to: {spec_path}")

    print("\nInitializing ChatSwarmRuntime...")
    runtime = ChatSwarmRuntime(
        project_root=repo_root,
        runs_root=runs_root,
        session_id="swarm-session-test-001"
    )

    print("Running swarm...")
    result = runtime.run(
        swarm_id="test-chat-swarm",
        spec_path=spec_path
    )

    print(f"\nSwarm result: {result}")

    print("\nChat session history:")
    for msg in runtime.chat_logger.get_history():
        role_emoji = "[SYSTEM]" if msg.role == "system" else "[GOV]" if msg.role == "governor" else f"[{msg.role.upper()}]"
        content_clean = msg.content.replace("🚀", "").replace("▶️", "").replace("✅", "").replace("❌", "").replace("🔧", "").replace("🏁", "")
        print(f"{role_emoji} {content_clean[:80]}")

    print(f"\nChat session: {runtime.session_id}")
    print(f"Total events logged: {len(runtime.chat_logger.get_history())}")
    print(f"Data stored in: MEMORY/LLM_PACKER/_packs/chat/")
