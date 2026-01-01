#!/usr/bin/env python3
"""
Catalytic Chat Message Writer

Implements triple-write strategy: DB (primary) + JSONL (mechanical) + MD (human).
All writes are atomic - all three must succeed or none.

Part of ADR-031: Catalytic Chat Triple-Write Architecture.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid as uuid_lib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from chat_db import ChatDB, ChatMessage, MessageChunk, MessageVector
from embedding_engine import ChatEmbeddingEngine


class MessageWriter:
    """Writes messages using triple-write strategy."""

    def __init__(
        self,
        db: Optional[ChatDB] = None,
        embedding_engine: Optional[ChatEmbeddingEngine] = None,
        claude_dir: Optional[Path] = None
    ):
        """Initialize message writer.

        Args:
            db: ChatDB instance (creates new one if None)
            embedding_engine: EmbeddingEngine instance (creates new if None)
            claude_dir: Claude config directory (~/.claude by default)
        """
        self.db = db or ChatDB()
        self.embedding_engine = embedding_engine or ChatEmbeddingEngine()
        self.claude_dir = claude_dir or Path.home() / ".claude"

        self.chunk_size_tokens = 500

    def write_message(
        self,
        session_id: str,
        role: str,
        content: str,
        parent_uuid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_sidechain: bool = False,
        is_meta: bool = False,
        cwd: Optional[str] = None
    ) -> str:
        """Write message with triple-write strategy.

        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            parent_uuid: Parent message UUID
            metadata: Optional metadata dict
            is_sidechain: Whether this is a sidechain message
            is_meta: Whether this is a meta message
            cwd: Current working directory

        Returns:
            Message UUID
        """
        msg_uuid = str(uuid_lib.uuid4())
        content_hash = self.db.compute_content_hash(content)
        timestamp = datetime.utcnow().isoformat() + "Z"

        message = ChatMessage(
            session_id=session_id,
            uuid=msg_uuid,
            parent_uuid=parent_uuid,
            role=role,
            content=content,
            content_hash=content_hash,
            timestamp=timestamp,
            metadata=metadata,
            is_sidechain=is_sidechain,
            is_meta=is_meta,
            cwd=cwd
        )

        with self.db.get_connection() as conn:
            msg_id = self.db.insert_message(message, conn)

            chunks = self._chunk_message(content, msg_id, content_hash)
            for chunk in chunks:
                self.db.insert_chunk(chunk, conn)

            self._generate_embeddings(chunks, conn)

            jsonl_path = self._get_jsonl_path(session_id)
            md_path = self._get_md_path(session_id)

            self._write_jsonl_export(session_id, jsonl_path, conn)
            self._write_md_export(session_id, md_path, conn)

        return msg_uuid

    def _chunk_message(
        self,
        content: str,
        message_id: int,
        content_hash: str
    ) -> List[MessageChunk]:
        """Split message into chunks for embedding.

        Args:
            content: Message content
            message_id: Message database ID
            content_hash: Content hash

        Returns:
            List of MessageChunk instances
        """
        words = content.split()

        if len(words) <= self.chunk_size_tokens:
            return [
                MessageChunk(
                    message_id=message_id,
                    chunk_index=0,
                    chunk_hash=hashlib.sha256(content.encode()).hexdigest(),
                    content=content,
                    token_count=len(words)
                )
            ]

        chunks = []
        chunk_index = 0
        chunk_words = []

        for word in words:
            chunk_words.append(word)

            if len(chunk_words) >= self.chunk_size_tokens:
                chunk_content = " ".join(chunk_words)
                chunks.append(MessageChunk(
                    message_id=message_id,
                    chunk_index=chunk_index,
                    chunk_hash=hashlib.sha256(
                        chunk_content.encode()
                    ).hexdigest(),
                    content=chunk_content,
                    token_count=len(chunk_words)
                ))
                chunk_index += 1
                chunk_words = []

        if chunk_words:
            chunk_content = " ".join(chunk_words)
            chunks.append(MessageChunk(
                message_id=message_id,
                chunk_index=chunk_index,
                chunk_hash=hashlib.sha256(
                    chunk_content.encode()
                ).hexdigest(),
                content=chunk_content,
                token_count=len(chunk_words)
            ))

        return chunks

    def _generate_embeddings(
        self,
        chunks: List[MessageChunk],
        conn
    ) -> None:
        """Generate and store vector embeddings for chunks.

        Args:
            chunks: List of MessageChunk instances
            conn: Database connection
        """
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_engine.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            vector = MessageVector(
                chunk_hash=chunk.chunk_hash,
                embedding=self.embedding_engine.serialize(embedding),
                model_id=self.embedding_engine.MODEL_ID,
                dimensions=self.embedding_engine.DIMENSIONS,
                created_at=datetime.utcnow().isoformat() + "Z"
            )
            self.db.insert_vector(vector, conn)

    def _get_jsonl_path(self, session_id: str) -> Path:
        """Get JSONL export path for session.

        Args:
            session_id: Session identifier

        Returns:
            Path to JSONL file
        """
        projects_dir = self.claude_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)
        return projects_dir / f"{session_id}.jsonl"

    def _get_md_path(self, session_id: str) -> Path:
        """Get Markdown export path for session.

        Args:
            session_id: Session identifier

        Returns:
            Path to Markdown file
        """
        projects_dir = self.claude_dir / "projects"
        return projects_dir / f"{session_id}.md"

    def _write_jsonl_export(
        self,
        session_id: str,
        output_path: Path,
        conn
    ) -> None:
        """Generate JSONL export from database.

        Args:
            session_id: Session identifier
            output_path: Path to write JSONL file
            conn: Database connection
        """
        messages = self.db.get_session_messages(session_id, conn=conn)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", encoding="utf-8") as f:
            for msg in messages:
                record = {
                    "uuid": msg.uuid,
                    "parentUuid": msg.parent_uuid,
                    "sessionId": session_id,
                    "type": msg.role,
                    "message": {
                        "role": msg.role,
                        "content": msg.content,
                        "usage": msg.metadata.get("usage") if msg.metadata else None
                    },
                    "isSidechain": msg.is_sidechain,
                    "isMeta": msg.is_meta,
                    "timestamp": msg.timestamp,
                    "cwd": msg.cwd
                }
                f.write(json.dumps(record) + "\n")

    def _write_md_export(
        self,
        session_id: str,
        output_path: Path,
        conn
    ) -> None:
        """Generate Markdown export from database.

        Args:
            session_id: Session identifier
            output_path: Path to write Markdown file
            conn: Database connection
        """
        messages = self.db.get_session_messages(session_id, conn=conn)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"# Session: {session_id}\n\n")

            for msg in messages:
                role_emoji = "👤" if msg.role == "user" else "🤖"
                f.write(f"## {role_emoji} {msg.role.title()} ({msg.timestamp})\n\n")
                f.write(f"**UUID:** `{msg.uuid}`\n\n")

                if msg.parent_uuid:
                    f.write(f"**Parent:** `{msg.parent_uuid}`\n\n")

                if msg.cwd:
                    f.write(f"**Working Dir:** `{msg.cwd}`\n\n")

                f.write("```\n")
                f.write(msg.content)
                f.write("\n```\n\n")
                f.write("---\n\n")


if __name__ == "__main__":
    import tempfile

    print("Testing MessageWriter...")

    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir = Path(tmpdir) / ".claude"
        claude_dir.mkdir()

        db_path = claude_dir / "chat.db"
        db = ChatDB(db_path)
        db.init_db()

        writer = MessageWriter(db=db, claude_dir=claude_dir)

        uuid1 = writer.write_message(
            session_id="test-session",
            role="user",
            content="Hello, this is a test message."
        )
        print(f"Wrote message UUID: {uuid1}")

        uuid2 = writer.write_message(
            session_id="test-session",
            role="assistant",
            content="This is a response from the assistant.",
            parent_uuid=uuid1
        )
        print(f"Wrote message UUID: {uuid2}")

        jsonl_path = writer._get_jsonl_path("test-session")
        md_path = writer._get_md_path("test-session")

        print(f"JSONL export: {jsonl_path}")
        print(f"MD export: {md_path}")

        assert jsonl_path.exists(), "JSONL file not created"
        assert md_path.exists(), "MD file not created"

        with open(jsonl_path, "r") as f:
            jsonl_content = f.read()
            print(f"JSONL content:\n{jsonl_content[:200]}...")
            assert "test message" in jsonl_content

        with open(md_path, "r") as f:
            md_content = f.read()
            print(f"MD content:\n{md_content[:200]}...")
            assert "test message" in md_content

    print("\nAll tests passed!")
