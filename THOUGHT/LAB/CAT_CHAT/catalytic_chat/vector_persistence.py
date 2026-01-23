#!/usr/bin/env python3
"""
Vector Persistence Module (Phase J.0)

Handles persistent storage of embeddings in SQLite for fast E-score computation.
Embeddings stored as BLOB (1536 bytes = 384 float32 values).

Part of Phase J.0: Vector Persistence for E-score optimization.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# Constants for embedding dimensions
EMBEDDING_DIM = 384
EMBEDDING_BYTES = EMBEDDING_DIM * 4  # float32 = 4 bytes (1536 bytes total)
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class VectorPersistenceError(Exception):
    """Exception for vector persistence operations."""
    pass


@dataclass
class PersistedVector:
    """Vector with metadata for retrieval."""
    event_id: str
    session_id: str
    content_hash: str
    embedding: np.ndarray
    sequence_num: int


def validate_embedding(blob: bytes) -> bool:
    """Validate BLOB is correctly formed embedding.

    Checks:
    - Size matches expected bytes (1536 = 384 * 4)
    - No NaN values
    - No Inf values
    - Reasonable norm (not all zeros, not extremely large)

    Args:
        blob: Raw bytes from database

    Returns:
        True if valid embedding, False otherwise
    """
    # Check size
    if len(blob) != EMBEDDING_BYTES:
        return False

    try:
        embedding = np.frombuffer(blob, dtype=np.float32)
    except ValueError:
        return False

    # Check for NaN
    if np.any(np.isnan(embedding)):
        return False

    # Check for Inf
    if np.any(np.isinf(embedding)):
        return False

    # Check reasonable norm (typical embeddings have norm ~1.0)
    norm = np.linalg.norm(embedding)
    if norm == 0.0:
        return False  # All zeros is invalid
    if norm > 100.0:
        return False  # Unreasonably large

    return True


class VectorPersistence:
    """Manages vector persistence in session_event_embeddings table.

    Provides SQLite-backed storage for embeddings with:
    - Efficient BLOB storage (1536 bytes per vector)
    - Content hash lookup for deduplication
    - Session-based batch loading for E-score computation
    - Backfill detection for migration support

    Usage:
        with VectorPersistence(db_path) as vp:
            vp.store_embedding(event_id, session_id, content_hash, embedding)
            vectors = vp.load_vectors(session_id)
    """

    def __init__(self, db_path: Path):
        """Initialize vector persistence.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection with standard settings."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    def ensure_schema(self) -> bool:
        """Create embeddings table if not exists.

        Creates:
        - session_event_embeddings table with BLOB storage
        - Indexes on session_id and content_hash

        Returns:
            True if table was created, False if already existed
        """
        conn = self._get_conn()

        # Check if table already exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='session_event_embeddings'
        """)
        existed = cursor.fetchone() is not None

        # Create table (idempotent)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_event_embeddings (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (event_id) REFERENCES session_events(event_id)
            )
        """)

        # Create indexes for efficient lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_session_id
            ON session_event_embeddings(session_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_content_hash
            ON session_event_embeddings(content_hash)
        """)

        conn.commit()

        return not existed

    def store_embedding(
        self,
        event_id: str,
        session_id: str,
        content_hash: str,
        embedding: np.ndarray,
        model: str = DEFAULT_MODEL
    ) -> None:
        """Store single embedding.

        Uses INSERT OR REPLACE for idempotent storage.

        Args:
            event_id: Primary key linking to session_events
            session_id: Session identifier for batch loading
            content_hash: Hash of content for deduplication lookup
            embedding: numpy array of shape (384,) with float32 dtype
            model: Embedding model identifier

        Raises:
            VectorPersistenceError: If embedding shape is invalid
        """
        if embedding.shape != (EMBEDDING_DIM,):
            raise VectorPersistenceError(
                f"Expected embedding shape ({EMBEDDING_DIM},), got {embedding.shape}"
            )

        blob = embedding.astype(np.float32).tobytes()
        created_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO session_event_embeddings
            (event_id, session_id, content_hash, embedding, embedding_model, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (event_id, session_id, content_hash, blob, model, created_at))
        conn.commit()

    def store_embeddings_batch(
        self,
        items: List[Tuple[str, str, str, np.ndarray]],
        model: str = DEFAULT_MODEL
    ) -> int:
        """Batch insert for migration efficiency.

        Args:
            items: List of (event_id, session_id, content_hash, embedding) tuples
            model: Embedding model identifier

        Returns:
            Count of embeddings inserted

        Raises:
            VectorPersistenceError: If any embedding has invalid shape
        """
        if not items:
            return 0

        conn = self._get_conn()
        created_at = datetime.now(timezone.utc).isoformat()

        rows = []
        for event_id, session_id, content_hash, embedding in items:
            if embedding.shape != (EMBEDDING_DIM,):
                raise VectorPersistenceError(
                    f"Invalid embedding shape for event {event_id}: "
                    f"expected ({EMBEDDING_DIM},), got {embedding.shape}"
                )
            blob = embedding.astype(np.float32).tobytes()
            rows.append((event_id, session_id, content_hash, blob, model, created_at))

        conn.executemany("""
            INSERT OR REPLACE INTO session_event_embeddings
            (event_id, session_id, content_hash, embedding, embedding_model, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()

        return len(rows)

    def load_vectors(self, session_id: str) -> List[PersistedVector]:
        """Load all embeddings for a session (J.0.3).

        Retrieves embeddings with sequence numbers for ordered E-score computation.

        Args:
            session_id: Session to load vectors for

        Returns:
            List of PersistedVector objects ordered by sequence_num
        """
        conn = self._get_conn()

        # Join with session_events to get sequence_num for ordering
        cursor = conn.execute("""
            SELECT
                e.event_id,
                e.session_id,
                e.content_hash,
                e.embedding,
                COALESCE(se.sequence_num, 0) as sequence_num
            FROM session_event_embeddings e
            LEFT JOIN session_events se ON e.event_id = se.event_id
            WHERE e.session_id = ?
            ORDER BY sequence_num ASC
        """, (session_id,))

        vectors = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            vectors.append(PersistedVector(
                event_id=row["event_id"],
                session_id=row["session_id"],
                content_hash=row["content_hash"],
                embedding=embedding,
                sequence_num=row["sequence_num"]
            ))

        return vectors

    def get_embedding_by_hash(self, content_hash: str) -> Optional[np.ndarray]:
        """Lookup single embedding by content hash.

        Useful for deduplication - same content should have same embedding.

        Args:
            content_hash: Hash of the content

        Returns:
            Embedding array if found, None otherwise
        """
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT embedding FROM session_event_embeddings
            WHERE content_hash = ?
            LIMIT 1
        """, (content_hash,))

        row = cursor.fetchone()
        if row is None:
            return None

        return np.frombuffer(row["embedding"], dtype=np.float32)

    def get_embedding_by_event_id(self, event_id: str) -> Optional[np.ndarray]:
        """Lookup single embedding by event ID.

        Args:
            event_id: Event identifier

        Returns:
            Embedding array if found, None otherwise
        """
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT embedding FROM session_event_embeddings
            WHERE event_id = ?
        """, (event_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        return np.frombuffer(row["embedding"], dtype=np.float32)

    def get_vectors_needing_backfill(self, session_id: str) -> List[Dict[str, Any]]:
        """Find turns without embeddings for migration (J.0.4).

        Identifies session_events that don't have corresponding embeddings,
        allowing incremental backfill of existing data.

        Args:
            session_id: Session to check for missing embeddings

        Returns:
            List of dicts with event_id, content_hash, sequence_num, payload_json
        """
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT
                se.event_id,
                se.content_hash,
                se.sequence_num,
                se.payload_json
            FROM session_events se
            LEFT JOIN session_event_embeddings see ON se.event_id = see.event_id
            WHERE se.session_id = ? AND see.event_id IS NULL
            ORDER BY se.sequence_num ASC
        """, (session_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "event_id": row["event_id"],
                "content_hash": row["content_hash"],
                "sequence_num": row["sequence_num"],
                "payload_json": row["payload_json"]
            })

        return results

    def count_embeddings(self, session_id: Optional[str] = None) -> int:
        """Count stored embeddings.

        Args:
            session_id: Optional session filter. If None, counts all embeddings.

        Returns:
            Count of embeddings
        """
        conn = self._get_conn()

        if session_id is None:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM session_event_embeddings"
            )
        else:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM session_event_embeddings WHERE session_id = ?",
                (session_id,)
            )

        return cursor.fetchone()[0]

    def delete_session_embeddings(self, session_id: str) -> int:
        """Delete all embeddings for a session.

        Args:
            session_id: Session to delete embeddings for

        Returns:
            Count of embeddings deleted
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM session_event_embeddings WHERE session_id = ?",
            (session_id,)
        )
        conn.commit()
        return cursor.rowcount

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "VectorPersistence":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close connection."""
        self.close()


def get_vector_persistence(repo_root: Optional[Path] = None) -> VectorPersistence:
    """Factory function to get VectorPersistence instance.

    Uses canonical path helpers for database location.

    Args:
        repo_root: Repository root path. Defaults to current working directory.

    Returns:
        Configured VectorPersistence instance with schema ensured
    """
    from .paths import get_cat_chat_db

    db_path = get_cat_chat_db(repo_root)
    vp = VectorPersistence(db_path)
    vp.ensure_schema()
    return vp


if __name__ == "__main__":
    import tempfile

    print("Testing VectorPersistence...")

    # Create temp database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Test basic operations
        with VectorPersistence(db_path) as vp:
            # Create mock session_events table for foreign key constraint
            conn = vp._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sequence_num INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
            """)
            # Insert mock events for testing
            conn.executemany(
                "INSERT INTO session_events VALUES (?, ?, ?, ?, ?)",
                [
                    ("evt_001", "sess_001", 1, "hash_001", "{}"),
                    ("evt_002", "sess_001", 2, "hash_002", "{}"),
                    ("evt_003", "sess_001", 3, "hash_003", "{}"),
                    ("evt_004", "sess_002", 1, "hash_004", "{}"),
                    ("evt_005", "sess_001", 4, "hash_005", "{}"),  # For backfill test
                ]
            )
            conn.commit()

            # Ensure schema
            created = vp.ensure_schema()
            print(f"Schema created: {created}")
            assert created is True

            # Second call should return False
            created = vp.ensure_schema()
            print(f"Schema already existed: {not created}")
            assert created is False

            # Test single store
            embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vp.store_embedding(
                event_id="evt_001",
                session_id="sess_001",
                content_hash="hash_001",
                embedding=embedding
            )
            print("Single embedding stored")

            # Test retrieval by hash
            retrieved = vp.get_embedding_by_hash("hash_001")
            assert retrieved is not None
            assert np.allclose(embedding, retrieved)
            print("Embedding retrieved by hash")

            # Test retrieval by event_id
            retrieved = vp.get_embedding_by_event_id("evt_001")
            assert retrieved is not None
            assert np.allclose(embedding, retrieved)
            print("Embedding retrieved by event_id")

            # Test batch store
            batch_items = [
                ("evt_002", "sess_001", "hash_002", np.random.randn(EMBEDDING_DIM).astype(np.float32)),
                ("evt_003", "sess_001", "hash_003", np.random.randn(EMBEDDING_DIM).astype(np.float32)),
                ("evt_004", "sess_002", "hash_004", np.random.randn(EMBEDDING_DIM).astype(np.float32)),
            ]
            count = vp.store_embeddings_batch(batch_items)
            assert count == 3
            print(f"Batch stored: {count} embeddings")

            # Test count
            total = vp.count_embeddings()
            sess1_count = vp.count_embeddings("sess_001")
            print(f"Total embeddings: {total}, Session 1: {sess1_count}")
            assert total == 4
            assert sess1_count == 3

            # Test load vectors with proper sequence ordering
            vectors = vp.load_vectors("sess_001")
            print(f"Loaded {len(vectors)} vectors for session 1")
            assert len(vectors) == 3
            # Verify ordering by sequence_num
            assert vectors[0].sequence_num == 1
            assert vectors[1].sequence_num == 2
            assert vectors[2].sequence_num == 3
            print("Vectors ordered correctly by sequence_num")

            # Test backfill detection (evt_005 has no embedding)
            needs_backfill = vp.get_vectors_needing_backfill("sess_001")
            print(f"Events needing backfill: {len(needs_backfill)}")
            assert len(needs_backfill) == 1
            assert needs_backfill[0]["event_id"] == "evt_005"
            print("Backfill detection working")

            # Test delete
            deleted = vp.delete_session_embeddings("sess_002")
            assert deleted == 1
            print(f"Deleted {deleted} embeddings")

            # Test validation
            valid_blob = embedding.tobytes()
            assert validate_embedding(valid_blob) is True
            print("Valid embedding validated")

            invalid_blob = b"too short"
            assert validate_embedding(invalid_blob) is False
            print("Invalid blob rejected")

            nan_embedding = np.full(EMBEDDING_DIM, np.nan, dtype=np.float32)
            assert validate_embedding(nan_embedding.tobytes()) is False
            print("NaN embedding rejected")

            zero_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            assert validate_embedding(zero_embedding.tobytes()) is False
            print("Zero embedding rejected")

    print("\nAll tests passed!")
