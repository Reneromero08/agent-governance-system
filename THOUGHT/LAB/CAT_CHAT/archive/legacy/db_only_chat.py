#!/usr/bin/env python3
"""
DB-Only Chat Interface.

Chat reads/writes ONLY in SQLite database.
Exports (JSONL/MD) are generated on-demand, never auto-written.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# Explicitly set paths
repo_root = Path("D:/CCC 2.0/AI/agent-governance-system")
chat_system_path = repo_root / "CATALYTIC-DPT" / "LAB" / "CHAT_SYSTEM"
sys.path.insert(0, str(chat_system_path))
os.chdir(str(chat_system_path))

from chat_db import ChatDB
from embedding_engine import ChatEmbeddingEngine


class DBOnlyChat:
    """DB-only chat interface. No automatic exports."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize DB-only chat.

        Args:
            db_path: Path to database (defaults to repo location)
        """
        if db_path is None:
            # Use local database in CHAT_SYSTEM directory
            db_path = Path(__file__).parent / "chat.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = ChatDB(db_path=db_path)
        self.db.init_db()
        self.engine = ChatEmbeddingEngine()

    def write_message(
        self,
        session_id: str,
        role: str,
        content: str,
        parent_uuid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Write message to DB ONLY.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system, governor, ant-*)
            content: Message content
            parent_uuid: Parent message UUID
            metadata: Optional metadata dict

        Returns:
            Message UUID
        """
        import hashlib
        import uuid as uuid_lib
        from datetime import datetime
        from chat_db import ChatMessage, MessageChunk, MessageVector

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
            metadata=metadata or {}
        )

        with self.db.get_connection() as conn:
            msg_id = self.db.insert_message(message, conn)

            # Chunk the message
            words = content.split()
            chunk_size = 500

            chunks = []
            chunk_words = []
            chunk_index = 0

            for word in words:
                chunk_words.append(word)
                if len(chunk_words) >= chunk_size:
                    chunk_content = " ".join(chunk_words)
                    chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
                    chunk = MessageChunk(
                        message_id=msg_id,
                        chunk_index=chunk_index,
                        chunk_hash=chunk_hash,
                        content=chunk_content,
                        token_count=len(chunk_words)
                    )
                    self.db.insert_chunk(chunk, conn)
                    chunks.append(chunk)

                    chunk_index += 1
                    chunk_words = []

            if chunk_words:
                chunk_content = " ".join(chunk_words)
                chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
                chunk = MessageChunk(
                    message_id=msg_id,
                    chunk_index=chunk_index,
                    chunk_hash=chunk_hash,
                    content=chunk_content,
                    token_count=len(chunk_words)
                )
                self.db.insert_chunk(chunk, conn)
                chunks.append(chunk)

            # Generate embeddings for chunks
            if chunks:
                texts = [chunk.content for chunk in chunks]
                embeddings = self.engine.embed_batch(texts)

                for chunk, embedding in zip(chunks, embeddings):
                    vector = MessageVector(
                        chunk_hash=chunk.chunk_hash,
                        embedding=self.engine.serialize(embedding),
                        model_id=self.engine.MODEL_ID,
                        dimensions=self.engine.DIMENSIONS,
                        created_at=datetime.utcnow().isoformat() + "Z"
                    )
                    self.db.insert_vector(vector, conn)

        return msg_uuid

    def read_message(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Read message from DB ONLY.

        Args:
            uuid: Message UUID

        Returns:
            Message dict or None
        """
        from chat_db import ChatMessage

        msg = self.db.get_message_by_uuid(uuid)
        if msg is None:
            return None

        return {
            "uuid": msg.uuid,
            "session_id": msg.session_id,
            "parent_uuid": msg.parent_uuid,
            "role": msg.role,
            "content": msg.content,
            "content_hash": msg.content_hash,
            "timestamp": msg.timestamp,
            "metadata": msg.metadata,
            "is_sidechain": msg.is_sidechain,
            "is_meta": msg.is_meta,
            "cwd": msg.cwd
        }

    def read_session(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read session from DB ONLY.

        Args:
            session_id: Session identifier
            limit: Optional message limit

        Returns:
            List of message dicts
        """
        messages = self.db.get_session_messages(session_id, limit=limit)
        return [
            {
                "uuid": msg.uuid,
                "session_id": msg.session_id,
                "parent_uuid": msg.parent_uuid,
                "role": msg.role,
                "content": msg.content,
                "content_hash": msg.content_hash,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata,
                "is_sidechain": msg.is_sidechain,
                "is_meta": msg.is_meta,
                "cwd": msg.cwd
            }
            for msg in messages
        ]

    def search_semantic(
        self,
        query: str,
        session_id: Optional[str] = None,
        threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search chat using vector similarity (DB ONLY).

        Args:
            query: Search query
            session_id: Optional session filter
            threshold: Similarity threshold (0-1)
            limit: Max results

        Returns:
            List of (message, chunk, similarity) tuples
        """
        query_emb = self.engine.embed(query)

        results = []

        if session_id:
            messages = self.db.get_session_messages(session_id)
        else:
            # For now, only support session-scoped search
            # Would need to implement get_all_messages() for global search
            messages = []

        for msg in messages:
            chunks = self.db.get_message_chunks(msg.message_id)

            for chunk in chunks:
                vectors = self.db.get_chunk_vectors([chunk.chunk_hash])

                if vectors:
                    chunk_emb = self.engine.deserialize(vectors[0].embedding)
                    sim = self.engine.cosine_similarity(query_emb, chunk_emb)

                    if sim >= threshold:
                        results.append({
                            "message_uuid": msg.uuid,
                            "message_role": msg.role,
                            "chunk_content": chunk.content,
                            "chunk_index": chunk.chunk_index,
                            "similarity": sim,
                            "timestamp": msg.timestamp
                        })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:limit]

    def export_jsonl(self, session_id: str, output_path: Optional[Path] = None) -> Path:
        """Export session to JSONL (ON DEMAND ONLY).

        This is the ONLY time we write to external files.

        Args:
            session_id: Session identifier
            output_path: Optional output path (defaults to projects/ dir)

        Returns:
            Path to written JSONL file
        """
        import json

        if output_path is None:
            # Use local projects directory in CHAT_SYSTEM
            projects_dir = Path(__file__).parent / "projects"
            projects_dir.mkdir(parents=True, exist_ok=True)
            output_path = projects_dir / f"{session_id}.jsonl"

        messages = self.db.get_session_messages(session_id)

        with open(output_path, "w", encoding="utf-8") as f:
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

        return output_path

    def export_md(self, session_id: str, output_path: Optional[Path] = None) -> Path:
        """Export session to Markdown (ON DEMAND ONLY).

        This is the ONLY time we write to external files.

        Args:
            session_id: Session identifier
            output_path: Optional output path (defaults to projects/ dir)

        Returns:
            Path to written MD file
        """
        if output_path is None:
            # Use local projects directory in CHAT_SYSTEM
            projects_dir = Path(__file__).parent / "projects"
            projects_dir.mkdir(parents=True, exist_ok=True)
            output_path = projects_dir / f"{session_id}.md"

        messages = self.db.get_session_messages(session_id)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Session: {session_id}\n\n")

            for msg in messages:
                role_emoji = "👤" if msg.role == "user" else "🤖" if msg.role == "assistant" else "⚙️"
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

        return output_path


if __name__ == "__main__":
    print("Testing DB-Only Chat Interface...")

    chat = DBOnlyChat()

    # Write messages to DB only
    print("\n1. Writing messages to DB only...")
    uuid1 = chat.write_message(
        session_id="db-only-session",
        role="user",
        content="How do I use this system?"
    )
    print(f"   Wrote: {uuid1}")

    uuid2 = chat.write_message(
        session_id="db-only-session",
        role="assistant",
        content="Use write_message() to save to DB, read_message() to retrieve.",
        parent_uuid=uuid1
    )
    print(f"   Wrote: {uuid2}")

    # Read from DB only
    print("\n2. Reading messages from DB only...")
    messages = chat.read_session("db-only-session")
    print(f"   Found {len(messages)} messages in session")
    for msg in messages:
        print(f"   [{msg['role']}]: {msg['content'][:60]}...")

    # Search using vectors in DB
    print("\n3. Searching DB using vector similarity...")
    results = chat.search_semantic(
        query="how to use the system",
        session_id="db-only-session",
        threshold=0.5
    )
    print(f"   Found {len(results)} similar chunks")
    for result in results[:3]:
        print(f"   [sim={result['similarity']:.2f}] {result['chunk_content'][:50]}...")

    # Export on demand (this is the ONLY time we write to files)
    print("\n4. Exporting on demand (ONLY time we write to files)...")
    jsonl_path = chat.export_jsonl("db-only-session")
    print(f"   JSONL exported: {jsonl_path}")

    md_path = chat.export_md("db-only-session")
    print(f"   MD exported: {md_path}")

    print("\n[OK] DB-Only chat interface working!")
    print("   - All storage in DB")
    print("   - All retrieval from DB")
    print("   - Exports only on demand")
