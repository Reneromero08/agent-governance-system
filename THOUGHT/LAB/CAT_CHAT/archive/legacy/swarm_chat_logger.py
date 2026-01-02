#!/usr/bin/env python3
"""
Swarm Integration with Catalytic Chat System.

Logs swarm execution events to the chat system for observability.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add CHAT_SYSTEM to path
chat_system_path = Path(__file__).parent.parent.parent.parent.parent / "CATALYTIC-DPT" / "LAB" / "CHAT_SYSTEM"
sys.path.insert(0, str(chat_system_path))

from chat_db import ChatDB
from message_writer import MessageWriter


class SwarmChatLogger:
    """Logs swarm activities to catalytic chat system."""

    def __init__(self, session_id: str, repo_root: Optional[Path] = None):
        """Initialize chat logger for swarm session.

        Args:
            session_id: Swarm run identifier
            repo_root: Repository root (defaults to this file's repo root)
        """
        if repo_root is None:
            repo_root = Path(__file__).parent.parent.parent.parent.parent

        chat_data_dir = repo_root / "MEMORY" / "LLM_PACKER" / "_packs" / "chat"
        chat_data_dir.mkdir(parents=True, exist_ok=True)

        db_path = chat_data_dir / "chat.db"
        db = ChatDB(db_path=db_path)
        db.init_db()

        self.writer = MessageWriter(db=db, claude_dir=chat_data_dir)
        self.session_id = session_id
        self.last_message_uuid: Optional[str] = None

    def log_event(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a swarm event.

        Args:
            role: Event type (governor, ant, system, user)
            content: Event description
            metadata: Optional event metadata
        """
        uuid = self.writer.write_message(
            session_id=self.session_id,
            role=role,
            content=content,
            parent_uuid=self.last_message_uuid,
            metadata=metadata or {}
        )
        self.last_message_uuid = uuid
        return uuid

    def log_swarm_start(self, swarm_id: str, spec: Dict[str, Any]):
        """Log swarm initialization."""
        return self.log_event(
            role="system",
            content=f"🚀 Swarm started: {swarm_id}",
            metadata={
                "event_type": "swarm_start",
                "swarm_id": swarm_id,
                "node_count": len(spec.get("nodes", [])),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

    def log_pipeline_start(self, node_id: str, pipeline_id: str):
        """Log pipeline execution start."""
        return self.log_event(
            role="governor",
            content=f"▶️ Starting pipeline: {pipeline_id} (node: {node_id})",
            metadata={
                "event_type": "pipeline_start",
                "node_id": node_id,
                "pipeline_id": pipeline_id
            }
        )

    def log_pipeline_complete(self, node_id: str, pipeline_id: str, result: Dict[str, Any]):
        """Log pipeline execution completion."""
        return self.log_event(
            role="governor",
            content=f"✅ Pipeline complete: {pipeline_id} (node: {node_id})",
            metadata={
                "event_type": "pipeline_complete",
                "node_id": node_id,
                "pipeline_id": pipeline_id,
                "result": result
            }
        )

    def log_pipeline_fail(self, node_id: str, pipeline_id: str, error: str):
        """Log pipeline execution failure."""
        return self.log_event(
            role="governor",
            content=f"❌ Pipeline failed: {pipeline_id} (node: {node_id}) - {error}",
            metadata={
                "event_type": "pipeline_fail",
                "node_id": node_id,
                "pipeline_id": pipeline_id,
                "error": error
            }
        )

    def log_agent_action(self, agent_id: str, action: str, details: str):
        """Log agent action."""
        return self.log_event(
            role=agent_id,
            content=f"🔧 {action}: {details}",
            metadata={
                "event_type": "agent_action",
                "agent_id": agent_id,
                "action": action
            }
        )

    def log_swarm_complete(self, swarm_id: str, summary: Dict[str, Any]):
        """Log swarm completion."""
        return self.log_event(
            role="system",
            content=f"🏁 Swarm completed: {swarm_id}\nSummary: {summary}",
            metadata={
                "event_type": "swarm_complete",
                "swarm_id": swarm_id,
                "summary": summary
            }
        )

    def get_history(self):
        """Retrieve swarm execution history."""
        db = self.writer.db
        return db.get_session_messages(self.session_id)


if __name__ == "__main__":
    print("Testing Swarm Chat Logger...")

    logger = SwarmChatLogger(session_id="test-swarm-run-001")

    logger.log_swarm_start("test-swarm", {
        "nodes": [
            {"node_id": "node-1", "pipeline_id": "pipeline-1"},
            {"node_id": "node-2", "pipeline_id": "pipeline-2"}
        ]
    })

    logger.log_pipeline_start("node-1", "pipeline-1")
    logger.log_agent_action("ant-1", "refactor", "Updated function signatures")
    logger.log_pipeline_complete("node-1", "pipeline-1", {"status": "success", "files_changed": 3})

    logger.log_pipeline_start("node-2", "pipeline-2")
    logger.log_agent_action("ant-2", "test", "Running unit tests")
    logger.log_pipeline_complete("node-2", "pipeline-2", {"status": "success", "tests_passed": 42})

    logger.log_swarm_complete("test-swarm", {
        "total_nodes": 2,
        "success_count": 2,
        "failure_count": 0,
        "duration_seconds": 120
    })

    print("\nSwarm events logged!")
    print(f"Session: {logger.session_id}")
    print(f"Messages logged: {len(logger.get_history())}")
    print(f"Data location: MEMORY/LLM_PACKER/_packs/chat/")

    print("\nSample history:")
    for msg in logger.get_history()[-3:]:
        content_preview = msg.content[:60].replace("🚀", "[START]").replace("▶️", "[START]").replace("✅", "[OK]").replace("❌", "[FAIL]").replace("🔧", "[ACT]").replace("🏁", "[DONE]")
        print(f"  [{msg.role}]: {content_preview}...")
