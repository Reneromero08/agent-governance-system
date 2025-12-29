#!/usr/bin/env python3
"""
Catalytic Chat Database

Manages SQLite database for chat message storage with hash-based indexing
and vector embeddings for semantic search.

Part of ADR-031: Catalytic Chat Triple-Write Architecture.
"""

import sqlite3
import hashlib
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class ChatMessage:
    """Chat message data model."""
    message_id: Optional[int] = None
    session_id: str = ""
    uuid: str = ""
    parent_uuid: Optional[str] = None
    role: str = ""
    content: str = ""
    content_hash: str = ""
    timestamp: str = ""
    metadata: Optional[Dict[str, Any]] = None
    is_sidechain: bool = False
    is_meta: bool = False
    cwd: Optional[str] = None


@dataclass
class MessageChunk:
    """Chunk of a long message for embedding."""
    chunk_id: Optional[int] = None
    message_id: int = 0
    chunk_index: int = 0
    chunk_hash: str = ""
    content: str = ""
    token_count: int = 0


@dataclass
class MessageVector:
    """Vector embedding for a message chunk."""
    chunk_hash: str = ""
    embedding: Optional[bytes] = None
    model_id: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    created_at: str = ""


class ChatDB:
    """Manages chat database connection and operations."""

    CURRENT_VERSION = 1
    CHUNK_SIZE_TOKENS = 500

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize chat database.

        Args:
            db_path: Path to database file. Defaults to ~/.claude/chat.db
        """
        if db_path is None:
            claude_dir = Path.home() / ".claude"
            claude_dir.mkdir(parents=True, exist_ok=True)
            db_path = claude_dir / "chat.db"

        self.db_path = db_path
        self._conn = None

    @contextmanager
    def get_connection(self):
        """Get database connection with context management.

        Yields:
            sqlite3 connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize database with schema."""
        with self.get_connection() as conn:
            self._create_schema(conn)
            self._run_migrations(conn)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database tables."""

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                uuid TEXT NOT NULL UNIQUE,
                parent_uuid TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata JSON,
                is_sidechain INTEGER DEFAULT 0,
                is_meta INTEGER DEFAULT 0,
                cwd TEXT,
                FOREIGN KEY (parent_uuid) REFERENCES chat_messages(uuid)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_hash TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                FOREIGN KEY (message_id) REFERENCES chat_messages(message_id),
                UNIQUE(message_id, chunk_index)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_vectors (
                chunk_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_id TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
                dimensions INTEGER NOT NULL DEFAULT 384,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_hash) REFERENCES message_chunks(chunk_hash)
            )
        """)

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                content,
                chunk_id UNINDEXED,
                role,
                tokenize='porter unicode61'
            )
        """)

        self._create_indexes(conn)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create performance indexes."""
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON chat_messages(session_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
            ON chat_messages(timestamp)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_content_hash
            ON chat_messages(content_hash)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_message
            ON message_chunks(message_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_created
            ON message_vectors(created_at)
        """)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run database migrations."""
        cursor = conn.execute(
            "SELECT value FROM chat_metadata WHERE key = 'db_version'"
        )
        row = cursor.fetchone()

        current_version = int(row["value"]) if row else 0

        if current_version < self.CURRENT_VERSION:
            self._migrate(conn, current_version, self.CURRENT_VERSION)

    def _migrate(
        self,
        conn: sqlite3.Connection,
        from_version: int,
        to_version: int
    ) -> None:
        """Migrate database between versions."""
        for version in range(from_version + 1, to_version + 1):
            if version == 1:
                conn.execute("""
                    INSERT INTO chat_metadata (key, value)
                    VALUES ('db_version', '1')
                """)

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA-256 hash of message content.

        Args:
            content: Message content string

        Returns:
            64-character hex string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def insert_message(
        self,
        message: ChatMessage,
        conn: Optional[sqlite3.Connection] = None
    ) -> int:
        """Insert a chat message.

        Args:
            message: ChatMessage instance
            conn: Optional existing connection (for transactions)

        Returns:
            message_id of inserted message
        """
        metadata_json = json.dumps(message.metadata) if message.metadata else None
        is_sidechain = 1 if message.is_sidechain else 0
        is_meta = 1 if message.is_meta else 0

        sql = """
            INSERT INTO chat_messages
                (session_id, uuid, parent_uuid, role, content, content_hash,
                 timestamp, metadata, is_sidechain, is_meta, cwd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        if conn:
            cursor = conn.execute(sql, (
                message.session_id,
                message.uuid,
                message.parent_uuid,
                message.role,
                message.content,
                message.content_hash,
                message.timestamp,
                metadata_json,
                is_sidechain,
                is_meta,
                message.cwd
            ))
            last_id = cursor.lastrowid
            if last_id is None:
                raise RuntimeError("Failed to insert message: lastrowid is None")
            return last_id
        else:
            with self.get_connection() as c:
                cursor = c.execute(sql, (
                    message.session_id,
                    message.uuid,
                    message.parent_uuid,
                    message.role,
                    message.content,
                    message.content_hash,
                    message.timestamp,
                    metadata_json,
                    is_sidechain,
                    is_meta,
                    message.cwd
                ))
                last_id = cursor.lastrowid
                if last_id is None:
                    raise RuntimeError("Failed to insert message: lastrowid is None")
                return last_id

    def get_message_by_uuid(
        self,
        uuid: str,
        conn: Optional[sqlite3.Connection] = None
    ) -> Optional[ChatMessage]:
        """Retrieve message by UUID.

        Args:
            uuid: Message UUID
            conn: Optional existing connection

        Returns:
            ChatMessage or None if not found
        """
        sql = """
            SELECT * FROM chat_messages WHERE uuid = ?
        """

        if conn:
            row = conn.execute(sql, (uuid,)).fetchone()
        else:
            with self.get_connection() as c:
                row = c.execute(sql, (uuid,)).fetchone()

        if not row:
            return None

        return self._row_to_message(row)

    def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        conn: Optional[sqlite3.Connection] = None
    ) -> List[ChatMessage]:
        """Retrieve all messages for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            conn: Optional existing connection

        Returns:
            List of ChatMessage instances
        """
        sql = """
            SELECT * FROM chat_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """

        if limit:
            sql += f" LIMIT {limit}"

        if conn:
            rows = conn.execute(sql, (session_id,)).fetchall()
        else:
            with self.get_connection() as c:
                rows = c.execute(sql, (session_id,)).fetchall()

        return [self._row_to_message(row) for row in rows]

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ChatMessage:
        """Convert database row to ChatMessage."""
        metadata = None
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        return ChatMessage(
            message_id=row["message_id"],
            session_id=row["session_id"],
            uuid=row["uuid"],
            parent_uuid=row["parent_uuid"],
            role=row["role"],
            content=row["content"],
            content_hash=row["content_hash"],
            timestamp=row["timestamp"],
            metadata=metadata,
            is_sidechain=bool(row["is_sidechain"]),
            is_meta=bool(row["is_meta"]),
            cwd=row["cwd"]
        )

    def insert_chunk(
        self,
        chunk: MessageChunk,
        conn: Optional[sqlite3.Connection] = None
    ) -> int:
        """Insert a message chunk.

        Args:
            chunk: MessageChunk instance
            conn: Optional existing connection

        Returns:
            chunk_id of inserted chunk
        """
        sql = """
            INSERT INTO message_chunks
                (message_id, chunk_index, chunk_hash, content, token_count)
            VALUES (?, ?, ?, ?, ?)
        """

        if conn:
            cursor = conn.execute(sql, (
                chunk.message_id,
                chunk.chunk_index,
                chunk.chunk_hash,
                chunk.content,
                chunk.token_count
            ))
            last_id = cursor.lastrowid
            if last_id is None:
                raise RuntimeError("Failed to insert chunk: lastrowid is None")
            return last_id
        else:
            with self.get_connection() as c:
                cursor = c.execute(sql, (
                    chunk.message_id,
                    chunk.chunk_index,
                    chunk.chunk_hash,
                    chunk.content,
                    chunk.token_count
                ))
                last_id = cursor.lastrowid
                if last_id is None:
                    raise RuntimeError("Failed to insert chunk: lastrowid is None")
                return last_id

    def insert_vector(
        self,
        vector: MessageVector,
        conn: Optional[sqlite3.Connection] = None
    ) -> None:
        """Insert a vector embedding.

        Args:
            vector: MessageVector instance
            conn: Optional existing connection
        """
        sql = """
            INSERT OR REPLACE INTO message_vectors
                (chunk_hash, embedding, model_id, dimensions, created_at)
            VALUES (?, ?, ?, ?, ?)
        """

        if conn:
            conn.execute(sql, (
                vector.chunk_hash,
                vector.embedding,
                vector.model_id,
                vector.dimensions,
                vector.created_at
            ))
        else:
            with self.get_connection() as c:
                c.execute(sql, (
                    vector.chunk_hash,
                    vector.embedding,
                    vector.model_id,
                    vector.dimensions,
                    vector.created_at
                ))

    def get_message_chunks(
        self,
        message_id: int,
        conn: Optional[sqlite3.Connection] = None
    ) -> List[MessageChunk]:
        """Retrieve chunks for a message.

        Args:
            message_id: Message ID
            conn: Optional existing connection

        Returns:
            List of MessageChunk instances
        """
        sql = """
            SELECT * FROM message_chunks
            WHERE message_id = ?
            ORDER BY chunk_index ASC
        """

        if conn:
            rows = conn.execute(sql, (message_id,)).fetchall()
        else:
            with self.get_connection() as c:
                rows = c.execute(sql, (message_id,)).fetchall()

        return [
            MessageChunk(
                chunk_id=row["chunk_id"],
                message_id=row["message_id"],
                chunk_index=row["chunk_index"],
                chunk_hash=row["chunk_hash"],
                content=row["content"],
                token_count=row["token_count"]
            )
            for row in rows
        ]

    def get_chunk_vectors(
        self,
        chunk_hashes: List[str],
        conn: Optional[sqlite3.Connection] = None
    ) -> List[MessageVector]:
        """Retrieve vectors for multiple chunks.

        Args:
            chunk_hashes: List of chunk hashes
            conn: Optional existing connection

        Returns:
            List of MessageVector instances
        """
        if not chunk_hashes:
            return []

        placeholders = ",".join("?" * len(chunk_hashes))
        sql = f"""
            SELECT * FROM message_vectors
            WHERE chunk_hash IN ({placeholders})
        """

        if conn:
            rows = conn.execute(sql, chunk_hashes).fetchall()
        else:
            with self.get_connection() as c:
                rows = c.execute(sql, chunk_hashes).fetchall()

        return [
            MessageVector(
                chunk_hash=row["chunk_hash"],
                embedding=row["embedding"],
                model_id=row["model_id"],
                dimensions=row["dimensions"],
                created_at=row["created_at"]
            )
            for row in rows
        ]


if __name__ == "__main__":
    import tempfile
    import os

    print("Testing ChatDB...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chat.db"
        db = ChatDB(db_path)

        db.init_db()
        print("Database initialized")

        message = ChatMessage(
            session_id="test-session-123",
            uuid="msg-uuid-001",
            parent_uuid=None,
            role="user",
            content="Hello, this is a test message.",
            content_hash=db.compute_content_hash("Hello, this is a test message."),
            timestamp="2025-12-29T12:00:00Z",
            metadata={"test_key": "test_value"},
            cwd="/test/dir"
        )

        msg_id = db.insert_message(message)
        print(f"Inserted message with ID: {msg_id}")

        retrieved = db.get_message_by_uuid("msg-uuid-001")
        assert retrieved is not None, "Message not found"
        print(f"Retrieved message UUID: {retrieved.uuid}")
        print(f"Retrieved content: {retrieved.content}")
        assert retrieved.content == message.content

        messages = db.get_session_messages("test-session-123")
        print(f"Session has {len(messages)} message(s)")
        assert len(messages) == 1

    print("\nAll tests passed!")
