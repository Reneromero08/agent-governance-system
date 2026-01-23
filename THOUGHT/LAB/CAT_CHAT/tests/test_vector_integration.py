"""
Integration Tests for Vector Persistence (Phase J.0)

Tests the integration between:
- TurnCompressor + VectorPersistence
- GeometricContextAssembler cache usage
- AutoContextManager session resume
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sqlite3

from catalytic_chat.vector_persistence import (
    VectorPersistence,
    EMBEDDING_DIM,
)


class TestTurnCompressorIntegration:
    """Test TurnCompressor integration with VectorPersistence."""

    @pytest.fixture
    def setup_db(self, tmp_path):
        """Create database with required schema."""
        db_path = tmp_path / "test.db"

        # Create session_events table first (for FK constraint)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT,
                prev_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        persistence = VectorPersistence(db_path)
        persistence.ensure_schema()

        return db_path, persistence

    def test_compressor_stores_embedding(self, setup_db):
        """TurnCompressor stores embedding when configured."""
        from catalytic_chat.turn_compressor import TurnCompressor, TurnContent

        db_path, persistence = setup_db

        # Mock embedding function
        mock_embed = Mock(return_value=np.random.randn(EMBEDDING_DIM).astype(np.float32))

        compressor = TurnCompressor(
            db_path=db_path,
            session_id="test_session",
            embed_fn=mock_embed,
            vector_persistence=persistence
        )

        # Create a turn
        turn = TurnContent(
            turn_id="turn_1",
            user_query="What is catalytic computing?",
            assistant_response="It is a paradigm for bounded computation.",
            timestamp="2026-01-23T00:00:00Z"
        )

        # Compress the turn
        result = compressor.compress_turn(turn)

        # Verify embedding was stored
        loaded = persistence.load_vectors("test_session")
        assert len(loaded) >= 1
        mock_embed.assert_called_once()

    def test_compressor_works_without_embedding(self, setup_db):
        """TurnCompressor works without embedding configuration."""
        from catalytic_chat.turn_compressor import TurnCompressor, TurnContent

        db_path, persistence = setup_db

        # No embed_fn provided
        compressor = TurnCompressor(
            db_path=db_path,
            session_id="test_session_no_embed"
        )

        turn = TurnContent(
            turn_id="turn_1",
            user_query="Hello",
            assistant_response="Hi there",
            timestamp="2026-01-23T00:00:00Z"
        )

        # Should not raise
        result = compressor.compress_turn(turn)
        assert result is not None

    def test_compressor_stores_multiple_turns(self, setup_db):
        """TurnCompressor stores embeddings for multiple turns."""
        from catalytic_chat.turn_compressor import TurnCompressor, TurnContent

        db_path, persistence = setup_db

        # Deterministic embedding function
        call_count = [0]
        def mock_embed(text):
            call_count[0] += 1
            vec = np.random.RandomState(call_count[0]).randn(EMBEDDING_DIM).astype(np.float32)
            return vec

        compressor = TurnCompressor(
            db_path=db_path,
            session_id="multi_turn_session",
            embed_fn=mock_embed,
            vector_persistence=persistence
        )

        # Create multiple turns
        for i in range(3):
            turn = TurnContent(
                turn_id=f"turn_{i}",
                user_query=f"Query {i}",
                assistant_response=f"Response {i}",
                timestamp=f"2026-01-23T0{i}:00:00Z"
            )
            compressor.compress_turn(turn)

        # Verify all embeddings were stored
        loaded = persistence.load_vectors("multi_turn_session")
        assert len(loaded) == 3


class TestGeometricAssemblerIntegration:
    """Test GeometricContextAssembler cache integration."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create VectorPersistence with temp database."""
        db_path = tmp_path / "test.db"

        # Create required session_events table for FK constraint
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT,
                prev_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL
            )
        """)
        # Insert mock events for FK constraint
        conn.execute("""
            INSERT INTO session_events
            (event_id, session_id, event_type, sequence_num, timestamp, payload_json, content_hash, prev_hash, chain_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("evt_1", "session_1", "test", 0, "2026-01-23T00:00:00Z", "{}", "hash_abc", "0"*64, "0"*64))
        conn.commit()
        conn.close()

        p = VectorPersistence(db_path)
        p.ensure_schema()
        return p

    def test_assembler_uses_preloaded_cache(self, persistence):
        """Assembler uses preloaded vectors from cache."""
        from catalytic_chat.geometric_context_assembler import GeometricContextAssembler

        # Pre-populate vectors
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        persistence.store_embedding("evt_1", "session_1", "hash_abc", vec)

        # Create assembler with persistence
        assembler = GeometricContextAssembler(
            vector_persistence=persistence
        )

        # Preload vectors
        cache = {v.content_hash: v.embedding for v in persistence.load_vectors("session_1")}
        assembler.preload_vectors(cache)

        # Check stats
        assert assembler._geo_stats.get('preloaded', 0) == 1

    def test_assembler_works_without_persistence(self):
        """Assembler works without vector_persistence."""
        from catalytic_chat.geometric_context_assembler import GeometricContextAssembler

        # No persistence provided
        assembler = GeometricContextAssembler()

        # Should initialize without error
        assert assembler._vector_persistence is None
        assert assembler._vector_cache == {}

    def test_assembler_preload_multiple_vectors(self, persistence, tmp_path):
        """Assembler correctly preloads multiple vectors."""
        from catalytic_chat.geometric_context_assembler import GeometricContextAssembler

        # Add more mock events
        conn = sqlite3.connect(str(persistence.db_path))
        for i in range(2, 5):
            conn.execute("""
                INSERT INTO session_events
                (event_id, session_id, event_type, sequence_num, timestamp, payload_json, content_hash, prev_hash, chain_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (f"evt_{i}", "session_1", "test", i, "2026-01-23T00:00:00Z", "{}", f"hash_{i}", "0"*64, "0"*64))
        conn.commit()
        conn.close()

        # Store multiple vectors
        for i in range(1, 5):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            persistence.store_embedding(f"evt_{i}", "session_1", f"hash_{i}", vec)

        # Create assembler and preload
        assembler = GeometricContextAssembler(vector_persistence=persistence)
        loaded = persistence.load_vectors("session_1")
        cache = {v.content_hash: v.embedding for v in loaded}
        assembler.preload_vectors(cache)

        assert assembler._geo_stats['preloaded'] == 4
        assert len(assembler._vector_cache) == 4


class TestAutoContextManagerIntegration:
    """Test AutoContextManager session vector loading."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create VectorPersistence with temp database."""
        db_path = tmp_path / "test.db"

        # Create required session_events table
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT,
                prev_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL
            )
        """)
        # Insert mock events for FK constraint
        conn.executemany("""
            INSERT INTO session_events
            (event_id, session_id, event_type, sequence_num, timestamp, payload_json, content_hash, prev_hash, chain_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            ("evt_1", "session_test", "test", 0, "2026-01-23T00:00:00Z", "{}", "hash_1", "0"*64, "0"*64),
            ("evt_2", "session_test", "test", 1, "2026-01-23T00:00:00Z", "{}", "hash_2", "0"*64, "0"*64),
        ])
        conn.commit()
        conn.close()

        p = VectorPersistence(db_path)
        p.ensure_schema()
        return p

    def test_load_session_vectors_returns_dict(self, persistence, tmp_path):
        """load_session_vectors returns content_hash -> embedding dict."""
        from catalytic_chat.auto_context_manager import AutoContextManager
        from catalytic_chat.adaptive_budget import AdaptiveBudget

        # Store some vectors
        vec1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        persistence.store_embedding("evt_1", "session_test", "hash_1", vec1)
        persistence.store_embedding("evt_2", "session_test", "hash_2", vec2)

        # Create budget
        budget = AdaptiveBudget(context_window=4096)

        # Create manager
        manager = AutoContextManager(
            session_id="session_test",
            db_path=tmp_path / "test.db",
            budget=budget,
            vector_persistence=persistence
        )

        # Load vectors
        vectors = manager.load_session_vectors()

        assert len(vectors) == 2
        assert "hash_1" in vectors
        assert "hash_2" in vectors
        np.testing.assert_array_almost_equal(vectors["hash_1"], vec1)

    def test_load_session_vectors_without_persistence(self, tmp_path):
        """load_session_vectors returns empty dict without persistence."""
        from catalytic_chat.auto_context_manager import AutoContextManager
        from catalytic_chat.adaptive_budget import AdaptiveBudget
        from catalytic_chat.session_capsule import SessionCapsule

        # Create session first
        db_path = tmp_path / "test.db"
        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()
        capsule.close()

        # Create budget
        budget = AdaptiveBudget(context_window=4096)

        manager = AutoContextManager(
            session_id=session_id,
            db_path=db_path,
            budget=budget,
            vector_persistence=None
        )

        vectors = manager.load_session_vectors()

        assert vectors == {}

    def test_load_session_vectors_empty_session(self, tmp_path):
        """load_session_vectors returns empty dict for session with no vectors."""
        from catalytic_chat.auto_context_manager import AutoContextManager
        from catalytic_chat.adaptive_budget import AdaptiveBudget
        from catalytic_chat.session_capsule import SessionCapsule

        db_path = tmp_path / "empty.db"

        # Create session
        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()
        capsule.close()

        # Create persistence but don't store any vectors
        persistence = VectorPersistence(db_path)
        persistence.ensure_schema()

        budget = AdaptiveBudget(context_window=4096)

        manager = AutoContextManager(
            session_id=session_id,
            db_path=db_path,
            budget=budget,
            vector_persistence=persistence
        )

        vectors = manager.load_session_vectors()
        assert vectors == {}


class TestEndToEndFlow:
    """Test complete flow from compression to resume."""

    @pytest.fixture
    def setup_db(self, tmp_path):
        """Create database with full schema."""
        db_path = tmp_path / "test_e2e.db"

        # Create session_events table
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT,
                prev_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        persistence = VectorPersistence(db_path)
        persistence.ensure_schema()

        return db_path, persistence

    def test_compress_then_resume_flow(self, setup_db):
        """Test complete flow: compress turn -> close -> resume -> load vectors."""
        from catalytic_chat.turn_compressor import TurnCompressor, TurnContent

        db_path, persistence = setup_db

        # Mock embedding function with deterministic output
        test_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        mock_embed = Mock(return_value=test_vec.copy())

        # Session 1: Compress turns
        compressor = TurnCompressor(
            db_path=db_path,
            session_id="e2e_session",
            embed_fn=mock_embed,
            vector_persistence=persistence
        )

        turn = TurnContent(
            turn_id="turn_1",
            user_query="Test query",
            assistant_response="Test response",
            timestamp="2026-01-23T00:00:00Z"
        )

        compressor.compress_turn(turn)

        # Close persistence
        persistence.close()

        # Session 2: Resume and load vectors
        persistence2 = VectorPersistence(db_path)
        loaded = persistence2.load_vectors("e2e_session")

        assert len(loaded) >= 1
        # Verify embedding matches what was stored
        np.testing.assert_array_almost_equal(loaded[0].embedding, test_vec)

        persistence2.close()

    def test_multiple_session_isolation(self, setup_db):
        """Vectors from different sessions are isolated."""
        from catalytic_chat.turn_compressor import TurnCompressor, TurnContent

        db_path, persistence = setup_db

        # Create turns in two different sessions
        sessions = ["session_a", "session_b"]
        embeddings = {}

        for session_id in sessions:
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            embeddings[session_id] = vec.copy()
            mock_embed = Mock(return_value=vec)

            compressor = TurnCompressor(
                db_path=db_path,
                session_id=session_id,
                embed_fn=mock_embed,
                vector_persistence=persistence
            )

            turn = TurnContent(
                turn_id=f"turn_{session_id}",
                user_query=f"Query for {session_id}",
                assistant_response=f"Response for {session_id}",
                timestamp="2026-01-23T00:00:00Z"
            )

            compressor.compress_turn(turn)

        # Verify each session gets only its own vectors
        for session_id in sessions:
            loaded = persistence.load_vectors(session_id)
            assert len(loaded) == 1
            np.testing.assert_array_almost_equal(
                loaded[0].embedding,
                embeddings[session_id]
            )

    def test_vector_persistence_survives_reconnection(self, setup_db):
        """Vectors persist across database reconnections."""
        from catalytic_chat.turn_compressor import TurnCompressor, TurnContent

        db_path, persistence = setup_db

        # Store vector
        test_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        mock_embed = Mock(return_value=test_vec.copy())

        compressor = TurnCompressor(
            db_path=db_path,
            session_id="persist_session",
            embed_fn=mock_embed,
            vector_persistence=persistence
        )

        turn = TurnContent(
            turn_id="turn_persist",
            user_query="Persist test",
            assistant_response="Persist response",
            timestamp="2026-01-23T00:00:00Z"
        )

        compressor.compress_turn(turn)
        persistence.close()

        # Reconnect multiple times
        for i in range(3):
            persistence_new = VectorPersistence(db_path)
            loaded = persistence_new.load_vectors("persist_session")
            assert len(loaded) >= 1
            np.testing.assert_array_almost_equal(loaded[0].embedding, test_vec)
            persistence_new.close()


class TestVectorPersistenceEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def setup_db(self, tmp_path):
        """Create database with required schema."""
        db_path = tmp_path / "edge_case.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT,
                prev_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        persistence = VectorPersistence(db_path)
        persistence.ensure_schema()

        return db_path, persistence

    def test_embedding_dimension_validation(self, setup_db):
        """VectorPersistence rejects wrong dimension embeddings."""
        from catalytic_chat.vector_persistence import VectorPersistenceError

        db_path, persistence = setup_db

        # Create session event for FK
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            INSERT INTO session_events
            (event_id, session_id, event_type, sequence_num, timestamp, payload_json, content_hash, prev_hash, chain_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("evt_wrong_dim", "session_1", "test", 0, "2026-01-23T00:00:00Z", "{}", "hash_wrong", "0"*64, "0"*64))
        conn.commit()
        conn.close()

        # Wrong dimension embedding
        wrong_vec = np.random.randn(128).astype(np.float32)  # Wrong size

        with pytest.raises(VectorPersistenceError):
            persistence.store_embedding("evt_wrong_dim", "session_1", "hash_wrong", wrong_vec)

    def test_duplicate_event_id_replaces(self, setup_db):
        """Storing same event_id replaces the embedding."""
        db_path, persistence = setup_db

        # Create session event for FK
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            INSERT INTO session_events
            (event_id, session_id, event_type, sequence_num, timestamp, payload_json, content_hash, prev_hash, chain_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("evt_dup", "session_1", "test", 0, "2026-01-23T00:00:00Z", "{}", "hash_dup", "0"*64, "0"*64))
        conn.commit()
        conn.close()

        # Store first embedding
        vec1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        persistence.store_embedding("evt_dup", "session_1", "hash_dup", vec1)

        # Store second embedding with same event_id
        vec2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        persistence.store_embedding("evt_dup", "session_1", "hash_dup", vec2)

        # Should have only one embedding, the second one
        loaded = persistence.load_vectors("session_1")
        assert len(loaded) == 1
        np.testing.assert_array_almost_equal(loaded[0].embedding, vec2)

    def test_get_embedding_by_hash_not_found(self, setup_db):
        """get_embedding_by_hash returns None for unknown hash."""
        db_path, persistence = setup_db

        result = persistence.get_embedding_by_hash("nonexistent_hash")
        assert result is None

    def test_get_embedding_by_event_id_not_found(self, setup_db):
        """get_embedding_by_event_id returns None for unknown event."""
        db_path, persistence = setup_db

        result = persistence.get_embedding_by_event_id("nonexistent_event")
        assert result is None

    def test_count_embeddings_by_session(self, setup_db):
        """count_embeddings correctly filters by session."""
        db_path, persistence = setup_db

        # Create events for two sessions
        conn = sqlite3.connect(str(db_path))
        for i, session in enumerate(["sess_a", "sess_b", "sess_a"]):
            conn.execute("""
                INSERT INTO session_events
                (event_id, session_id, event_type, sequence_num, timestamp, payload_json, content_hash, prev_hash, chain_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (f"evt_{i}", session, "test", i, "2026-01-23T00:00:00Z", "{}", f"hash_{i}", "0"*64, "0"*64))
        conn.commit()
        conn.close()

        # Store embeddings
        for i, session in enumerate(["sess_a", "sess_b", "sess_a"]):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            persistence.store_embedding(f"evt_{i}", session, f"hash_{i}", vec)

        assert persistence.count_embeddings() == 3
        assert persistence.count_embeddings("sess_a") == 2
        assert persistence.count_embeddings("sess_b") == 1
        assert persistence.count_embeddings("sess_c") == 0

    def test_delete_session_embeddings(self, setup_db):
        """delete_session_embeddings removes only target session vectors."""
        db_path, persistence = setup_db

        # Create events for two sessions
        conn = sqlite3.connect(str(db_path))
        for i, session in enumerate(["sess_del", "sess_keep"]):
            conn.execute("""
                INSERT INTO session_events
                (event_id, session_id, event_type, sequence_num, timestamp, payload_json, content_hash, prev_hash, chain_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (f"evt_{i}", session, "test", i, "2026-01-23T00:00:00Z", "{}", f"hash_{i}", "0"*64, "0"*64))
        conn.commit()
        conn.close()

        # Store embeddings
        for i, session in enumerate(["sess_del", "sess_keep"]):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            persistence.store_embedding(f"evt_{i}", session, f"hash_{i}", vec)

        # Delete one session
        deleted = persistence.delete_session_embeddings("sess_del")
        assert deleted == 1

        # Verify counts
        assert persistence.count_embeddings("sess_del") == 0
        assert persistence.count_embeddings("sess_keep") == 1
