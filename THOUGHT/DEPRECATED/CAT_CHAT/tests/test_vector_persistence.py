#!/usr/bin/env python3
"""
Unit Tests for Vector Persistence (Phase J.0)

Tests for:
- Serialization/deserialization roundtrip
- BLOB size validation
- Embedding validation (NaN, Inf, size)
- Store and load operations
- Batch operations
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import time

from catalytic_chat.vector_persistence import (
    VectorPersistence,
    VectorPersistenceError,
    PersistedVector,
    validate_embedding,
    EMBEDDING_DIM,
    EMBEDDING_BYTES,
)
from catalytic_chat.embedding_engine import ChatEmbeddingEngine


class TestSerialization:
    """Test embedding serialization/deserialization."""

    def test_roundtrip_preserves_values(self):
        """Serialize then deserialize returns identical array."""
        engine = ChatEmbeddingEngine()
        original = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        blob = engine.serialize(original)
        recovered = engine.deserialize(blob)
        np.testing.assert_array_equal(original, recovered)

    def test_blob_size_is_correct(self):
        """BLOB is exactly 384 * 4 = 1536 bytes."""
        engine = ChatEmbeddingEngine()
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        blob = engine.serialize(vec)
        assert len(blob) == EMBEDDING_BYTES

    def test_wrong_dimension_raises(self):
        """Vector with wrong dimension raises ValueError."""
        engine = ChatEmbeddingEngine()
        bad_vec = np.random.randn(128).astype(np.float32)
        with pytest.raises(ValueError):
            engine.serialize(bad_vec)


class TestValidation:
    """Test embedding validation."""

    def test_valid_embedding_passes(self):
        """Properly formed embedding validates."""
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize
        blob = vec.tobytes()
        assert validate_embedding(blob) is True

    def test_corrupted_size_fails(self):
        """Wrong size blob fails validation."""
        bad_blob = b"too short"
        assert validate_embedding(bad_blob) is False

    def test_nan_values_fail(self):
        """NaN values fail validation."""
        vec = np.full(EMBEDDING_DIM, np.nan, dtype=np.float32)
        blob = vec.tobytes()
        assert validate_embedding(blob) is False

    def test_inf_values_fail(self):
        """Infinite values fail validation."""
        vec = np.full(EMBEDDING_DIM, np.inf, dtype=np.float32)
        blob = vec.tobytes()
        assert validate_embedding(blob) is False

    def test_zero_values_fail(self):
        """All-zero embedding fails validation (norm == 0)."""
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        blob = vec.tobytes()
        assert validate_embedding(blob) is False

    def test_unreasonably_large_norm_fails(self):
        """Embedding with extremely large values fails validation."""
        vec = np.full(EMBEDDING_DIM, 100.0, dtype=np.float32)
        blob = vec.tobytes()
        assert validate_embedding(blob) is False


def _setup_test_database(conn):
    """Setup test database by disabling foreign key checks and creating mock tables.

    The VectorPersistence schema has a FOREIGN KEY reference to session_events.
    For unit tests, we disable FK enforcement to isolate embedding storage testing.
    We also create the referenced table structure in case queries join against it.
    """
    # Disable foreign key enforcement for isolated unit tests
    conn.execute("PRAGMA foreign_keys = OFF")

    # Create mock session_events table for any LEFT JOIN operations
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            sequence_num INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
    """)
    conn.commit()


class TestPersistence:
    """Test database persistence operations."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create VectorPersistence with temp database and mock session_events table."""
        db_path = tmp_path / "test.db"
        p = VectorPersistence(db_path)
        # Setup test database (disable FK, create mock tables)
        _setup_test_database(p._get_conn())
        p.ensure_schema()
        return p

    def test_store_and_load_roundtrip(self, persistence):
        """Store embeddings and load them back."""
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        event_id = "evt_test_1"
        session_id = "session_test"
        content_hash = "abc123"

        persistence.store_embedding(event_id, session_id, content_hash, vec)

        loaded = persistence.load_vectors(session_id)

        assert len(loaded) == 1
        assert loaded[0].event_id == event_id
        assert loaded[0].content_hash == content_hash
        np.testing.assert_array_almost_equal(loaded[0].embedding, vec)

    def test_load_empty_session_returns_empty_list(self, persistence):
        """Loading nonexistent session returns empty list."""
        loaded = persistence.load_vectors("nonexistent")
        assert loaded == []

    def test_get_embedding_by_hash(self, persistence):
        """Lookup by content hash works."""
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        persistence.store_embedding("evt_1", "session_1", "hash_abc", vec)

        result = persistence.get_embedding_by_hash("hash_abc")

        assert result is not None
        np.testing.assert_array_almost_equal(result, vec)

    def test_get_embedding_by_hash_missing(self, persistence):
        """Missing hash returns None."""
        result = persistence.get_embedding_by_hash("nonexistent_hash")
        assert result is None

    def test_get_embedding_by_event_id(self, persistence):
        """Lookup by event ID works."""
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        persistence.store_embedding("evt_lookup_1", "session_1", "hash_xyz", vec)

        result = persistence.get_embedding_by_event_id("evt_lookup_1")

        assert result is not None
        np.testing.assert_array_almost_equal(result, vec)

    def test_get_embedding_by_event_id_missing(self, persistence):
        """Missing event ID returns None."""
        result = persistence.get_embedding_by_event_id("nonexistent_event")
        assert result is None

    def test_batch_insert(self, persistence):
        """Batch insert multiple embeddings."""
        embeddings = [
            (f"evt_{i}", "session_batch", f"hash_{i}",
             np.random.randn(EMBEDDING_DIM).astype(np.float32))
            for i in range(10)
        ]

        count = persistence.store_embeddings_batch(embeddings)

        assert count == 10
        loaded = persistence.load_vectors("session_batch")
        assert len(loaded) == 10

    def test_batch_insert_empty_list(self, persistence):
        """Batch insert with empty list returns 0."""
        count = persistence.store_embeddings_batch([])
        assert count == 0

    def test_batch_insert_performance(self, persistence):
        """Batch insert 1000 vectors in reasonable time."""
        embeddings = [
            (f"evt_{i}", "session_perf", f"hash_{i}",
             np.random.randn(EMBEDDING_DIM).astype(np.float32))
            for i in range(1000)
        ]

        start = time.time()
        count = persistence.store_embeddings_batch(embeddings)
        elapsed = time.time() - start

        assert count == 1000
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_insert_or_replace_idempotent(self, persistence):
        """Inserting same event_id replaces existing."""
        vec1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        persistence.store_embedding("evt_1", "session_1", "hash_1", vec1)
        persistence.store_embedding("evt_1", "session_1", "hash_1", vec2)

        loaded = persistence.load_vectors("session_1")
        assert len(loaded) == 1
        np.testing.assert_array_almost_equal(loaded[0].embedding, vec2)

    def test_context_manager(self, tmp_path):
        """Context manager properly closes connection."""
        db_path = tmp_path / "test_cm.db"

        with VectorPersistence(db_path) as p:
            _setup_test_database(p._get_conn())
            p.ensure_schema()
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            p.store_embedding("evt_1", "session_1", "hash_1", vec)

        # Should be able to open again
        with VectorPersistence(db_path) as p2:
            _setup_test_database(p2._get_conn())
            loaded = p2.load_vectors("session_1")
            assert len(loaded) == 1

    def test_store_wrong_shape_raises(self, persistence):
        """Storing embedding with wrong shape raises VectorPersistenceError."""
        bad_vec = np.random.randn(128).astype(np.float32)

        with pytest.raises(VectorPersistenceError):
            persistence.store_embedding("evt_bad", "session_1", "hash_bad", bad_vec)

    def test_batch_insert_wrong_shape_raises(self, persistence):
        """Batch insert with wrong shape raises VectorPersistenceError."""
        embeddings = [
            ("evt_1", "session_1", "hash_1",
             np.random.randn(128).astype(np.float32))  # Wrong dimension
        ]

        with pytest.raises(VectorPersistenceError):
            persistence.store_embeddings_batch(embeddings)

    def test_count_embeddings_all(self, persistence):
        """Count all embeddings in database."""
        for i in range(5):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            persistence.store_embedding(f"evt_{i}", f"session_{i % 2}", f"hash_{i}", vec)

        total = persistence.count_embeddings()
        assert total == 5

    def test_count_embeddings_by_session(self, persistence):
        """Count embeddings for specific session."""
        for i in range(5):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            persistence.store_embedding(f"evt_{i}", f"session_{i % 2}", f"hash_{i}", vec)

        session_0_count = persistence.count_embeddings("session_0")
        session_1_count = persistence.count_embeddings("session_1")

        assert session_0_count == 3  # i=0,2,4
        assert session_1_count == 2  # i=1,3

    def test_delete_session_embeddings(self, persistence):
        """Delete all embeddings for a session."""
        for i in range(5):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            persistence.store_embedding(f"evt_{i}", f"session_{i % 2}", f"hash_{i}", vec)

        deleted = persistence.delete_session_embeddings("session_0")

        assert deleted == 3
        assert persistence.count_embeddings("session_0") == 0
        assert persistence.count_embeddings("session_1") == 2


class TestSchemaCreation:
    """Test schema creation."""

    def test_ensure_schema_creates_table(self, tmp_path):
        """ensure_schema creates the embeddings table."""
        import sqlite3

        db_path = tmp_path / "test_schema.db"
        p = VectorPersistence(db_path)
        p.ensure_schema()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='session_event_embeddings'"
        )
        assert cursor.fetchone() is not None
        conn.close()
        p.close()

    def test_ensure_schema_idempotent(self, tmp_path):
        """Calling ensure_schema multiple times is safe."""
        db_path = tmp_path / "test_idempotent.db"
        p = VectorPersistence(db_path)
        # Setup test database (disable FK, create mock tables)
        _setup_test_database(p._get_conn())

        # Call multiple times
        p.ensure_schema()
        p.ensure_schema()
        p.ensure_schema()

        # Should still work
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        p.store_embedding("evt_1", "session_1", "hash_1", vec)
        loaded = p.load_vectors("session_1")
        assert len(loaded) == 1
        p.close()

    def test_ensure_schema_returns_creation_status(self, tmp_path):
        """ensure_schema returns True on first call, False on subsequent calls."""
        db_path = tmp_path / "test_creation_status.db"
        p = VectorPersistence(db_path)

        # First call should return True (created)
        first_result = p.ensure_schema()
        assert first_result is True

        # Second call should return False (already existed)
        second_result = p.ensure_schema()
        assert second_result is False

        p.close()

    def test_ensure_schema_creates_indexes(self, tmp_path):
        """ensure_schema creates necessary indexes."""
        import sqlite3

        db_path = tmp_path / "test_indexes.db"
        p = VectorPersistence(db_path)
        p.ensure_schema()

        conn = sqlite3.connect(str(db_path))

        # Check for session_id index
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_embeddings_session_id'"
        )
        assert cursor.fetchone() is not None

        # Check for content_hash index
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_embeddings_content_hash'"
        )
        assert cursor.fetchone() is not None

        conn.close()
        p.close()


class TestPersistedVector:
    """Test PersistedVector dataclass."""

    def test_persisted_vector_fields(self):
        """PersistedVector has expected fields."""
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        pv = PersistedVector(
            event_id="evt_1",
            session_id="session_1",
            content_hash="hash_1",
            embedding=vec,
            sequence_num=42
        )

        assert pv.event_id == "evt_1"
        assert pv.session_id == "session_1"
        assert pv.content_hash == "hash_1"
        np.testing.assert_array_equal(pv.embedding, vec)
        assert pv.sequence_num == 42
