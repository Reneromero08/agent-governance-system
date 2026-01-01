#!/usr/bin/env python3
"""
Comprehensive tests for Catalytic Chat System.

Tests for chat_db, embedding_engine, and message_writer modules.
Part of ADR-031: Catalytic Chat Triple-Write Architecture.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import hashlib
import uuid as uuid_lib

import sys
sys.path.insert(0, str(Path(__file__).parent))

from chat_db import ChatDB, ChatMessage, MessageChunk, MessageVector
from embedding_engine import ChatEmbeddingEngine
from message_writer import MessageWriter


class TestChatDB:
    """Test ChatDB functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_chat.db"
            db = ChatDB(db_path)
            db.init_db()
            yield db

    def test_database_initialization(self, temp_db):
        """Test that database initializes correctly."""
        assert temp_db.db_path.exists()
        assert temp_db.db_path.is_file()

    def test_insert_message(self, temp_db):
        """Test inserting a message."""
        message = ChatMessage(
            session_id="test-session",
            uuid="msg-001",
            parent_uuid=None,
            role="user",
            content="Test message",
            content_hash=ChatDB.compute_content_hash("Test message"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        msg_id = temp_db.insert_message(message)
        assert msg_id > 0

    def test_retrieve_message_by_uuid(self, temp_db):
        """Test retrieving a message by UUID."""
        message = ChatMessage(
            session_id="test-session",
            uuid="msg-002",
            parent_uuid=None,
            role="user",
            content="Test message for retrieval",
            content_hash=ChatDB.compute_content_hash("Test message for retrieval"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        temp_db.insert_message(message)

        retrieved = temp_db.get_message_by_uuid("msg-002")
        assert retrieved is not None
        assert retrieved.uuid == "msg-002"
        assert retrieved.content == "Test message for retrieval"
        assert retrieved.role == "user"

    def test_get_nonexistent_message(self, temp_db):
        """Test retrieving a non-existent message."""
        result = temp_db.get_message_by_uuid("non-existent-uuid")
        assert result is None

    def test_get_session_messages(self, temp_db):
        """Test retrieving all messages for a session."""
        msg1 = ChatMessage(
            session_id="session-a",
            uuid="msg-003",
            parent_uuid=None,
            role="user",
            content="First message",
            content_hash=ChatDB.compute_content_hash("First message"),
            timestamp="2025-12-29T10:00:00Z"
        )
        msg2 = ChatMessage(
            session_id="session-a",
            uuid="msg-004",
            parent_uuid="msg-003",
            role="assistant",
            content="Second message",
            content_hash=ChatDB.compute_content_hash("Second message"),
            timestamp="2025-12-29T10:05:00Z"
        )
        msg3 = ChatMessage(
            session_id="session-b",
            uuid="msg-005",
            parent_uuid=None,
            role="user",
            content="Different session",
            content_hash=ChatDB.compute_content_hash("Different session"),
            timestamp="2025-12-29T11:00:00Z"
        )

        temp_db.insert_message(msg1)
        temp_db.insert_message(msg2)
        temp_db.insert_message(msg3)

        messages_a = temp_db.get_session_messages("session-a")
        assert len(messages_a) == 2
        assert messages_a[0].uuid == "msg-003"
        assert messages_a[1].uuid == "msg-004"

        messages_b = temp_db.get_session_messages("session-b")
        assert len(messages_b) == 1
        assert messages_b[0].uuid == "msg-005"

    def test_session_limit(self, temp_db):
        """Test limiting session messages."""
        for i in range(5):
            msg = ChatMessage(
                session_id="limited-session",
                uuid=f"msg-{i}",
                parent_uuid=None,
                role="user",
                content=f"Message {i}",
                content_hash=ChatDB.compute_content_hash(f"Message {i}"),
                timestamp=f"2025-12-29T1{i}:00:00Z"
            )
            temp_db.insert_message(msg)

        messages = temp_db.get_session_messages("limited-session", limit=3)
        assert len(messages) == 3

    def test_content_hash(self):
        """Test content hash computation."""
        hash1 = ChatDB.compute_content_hash("test content")
        hash2 = ChatDB.compute_content_hash("test content")
        hash3 = ChatDB.compute_content_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64

    def test_message_metadata(self, temp_db):
        """Test message with metadata."""
        metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        message = ChatMessage(
            session_id="test-session",
            uuid="msg-006",
            parent_uuid=None,
            role="user",
            content="Message with metadata",
            content_hash=ChatDB.compute_content_hash("Message with metadata"),
            timestamp=datetime.utcnow().isoformat() + "Z",
            metadata=metadata
        )
        temp_db.insert_message(message)

        retrieved = temp_db.get_message_by_uuid("msg-006")
        assert retrieved is not None
        assert retrieved.metadata == metadata

    def test_insert_chunk(self, temp_db):
        """Test inserting message chunks."""
        message = ChatMessage(
            session_id="test-session",
            uuid="msg-007",
            parent_uuid=None,
            role="user",
            content="Test message",
            content_hash=ChatDB.compute_content_hash("Test message"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        msg_id = temp_db.insert_message(message)

        chunk = MessageChunk(
            message_id=msg_id,
            chunk_index=0,
            chunk_hash=hashlib.sha256("chunk content".encode()).hexdigest(),
            content="chunk content",
            token_count=2
        )
        chunk_id = temp_db.insert_chunk(chunk)
        assert chunk_id > 0

    def test_get_message_chunks(self, temp_db):
        """Test retrieving chunks for a message."""
        message = ChatMessage(
            session_id="test-session",
            uuid="msg-008",
            parent_uuid=None,
            role="user",
            content="Test message",
            content_hash=ChatDB.compute_content_hash("Test message"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        msg_id = temp_db.insert_message(message)

        chunks = [
            MessageChunk(
                message_id=msg_id,
                chunk_index=i,
                chunk_hash=hashlib.sha256(f"chunk {i}".encode()).hexdigest(),
                content=f"chunk {i}",
                token_count=2
            )
            for i in range(3)
        ]

        for chunk in chunks:
            temp_db.insert_chunk(chunk)

        retrieved_chunks = temp_db.get_message_chunks(msg_id)
        assert len(retrieved_chunks) == 3
        assert retrieved_chunks[0].chunk_index == 0
        assert retrieved_chunks[1].chunk_index == 1
        assert retrieved_chunks[2].chunk_index == 2

    def test_insert_vector(self, temp_db):
        """Test inserting vector embedding."""
        message = ChatMessage(
            session_id="test-session",
            uuid="msg-vector-1",
            parent_uuid=None,
            role="user",
            content="Test message for vector",
            content_hash=ChatDB.compute_content_hash("Test message for vector"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        msg_id = temp_db.insert_message(message)

        chunk_hash = hashlib.sha256("test chunk".encode()).hexdigest()
        chunk = MessageChunk(
            message_id=msg_id,
            chunk_index=0,
            chunk_hash=chunk_hash,
            content="test chunk",
            token_count=2
        )
        temp_db.insert_chunk(chunk)

        embedding = np.random.randn(384).astype(np.float32)
        vector = MessageVector(
            chunk_hash=chunk_hash,
            embedding=embedding.tobytes(),
            model_id="all-MiniLM-L6-v2",
            dimensions=384,
            created_at=datetime.utcnow().isoformat() + "Z"
        )

        temp_db.insert_vector(vector)

    def test_get_chunk_vectors(self, temp_db):
        """Test retrieving multiple vectors."""
        chunk_hashes = []
        for i in range(3):
            message = ChatMessage(
                session_id="test-session",
                uuid=f"msg-vector-{i}",
                parent_uuid=None,
                role="user",
                content=f"Test message for vector {i}",
                content_hash=ChatDB.compute_content_hash(f"Test message for vector {i}"),
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            msg_id = temp_db.insert_message(message)

            chunk_hash = hashlib.sha256(f"chunk {i}".encode()).hexdigest()
            chunk_hashes.append(chunk_hash)

            chunk = MessageChunk(
                message_id=msg_id,
                chunk_index=0,
                chunk_hash=chunk_hash,
                content=f"chunk {i}",
                token_count=2
            )
            temp_db.insert_chunk(chunk)

            embedding = np.random.randn(384).astype(np.float32)
            vector = MessageVector(
                chunk_hash=chunk_hash,
                embedding=embedding.tobytes(),
                model_id="all-MiniLM-L6-v2",
                dimensions=384,
                created_at=datetime.utcnow().isoformat() + "Z"
            )
            temp_db.insert_vector(vector)

        vectors = temp_db.get_chunk_vectors(chunk_hashes)
        assert len(vectors) == 3

    def test_get_empty_chunk_vectors(self, temp_db):
        """Test retrieving vectors with empty list."""
        vectors = temp_db.get_chunk_vectors([])
        assert vectors == []

    def test_message_flags(self, temp_db):
        """Test message sidechain and meta flags."""
        message = ChatMessage(
            session_id="test-session",
            uuid="msg-009",
            parent_uuid=None,
            role="user",
            content="Test message",
            content_hash=ChatDB.compute_content_hash("Test message"),
            timestamp=datetime.utcnow().isoformat() + "Z",
            is_sidechain=True,
            is_meta=True,
            cwd="/test/dir"
        )
        temp_db.insert_message(message)

        retrieved = temp_db.get_message_by_uuid("msg-009")
        assert retrieved.is_sidechain is True
        assert retrieved.is_meta is True
        assert retrieved.cwd == "/test/dir"

    def test_unique_uuid_constraint(self, temp_db):
        """Test that UUID must be unique."""
        message1 = ChatMessage(
            session_id="test-session",
            uuid="msg-010",
            parent_uuid=None,
            role="user",
            content="First message",
            content_hash=ChatDB.compute_content_hash("First message"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        message2 = ChatMessage(
            session_id="test-session",
            uuid="msg-010",
            parent_uuid=None,
            role="assistant",
            content="Second message",
            content_hash=ChatDB.compute_content_hash("Second message"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

        temp_db.insert_message(message1)
        with pytest.raises(Exception):
            temp_db.insert_message(message2)


class TestEmbeddingEngine:
    """Test ChatEmbeddingEngine functionality."""

    @pytest.fixture
    def engine(self):
        """Create embedding engine instance."""
        return ChatEmbeddingEngine()

    def test_single_embedding(self, engine):
        """Test generating a single embedding."""
        text = "This is a test message for embedding."
        embedding = engine.embed(text)

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_empty_embedding(self, engine):
        """Test embedding empty text."""
        embedding = engine.embed("")
        assert np.allclose(embedding, np.zeros(384, dtype=np.float32))

    def test_batch_embeddings(self, engine):
        """Test generating batch embeddings."""
        texts = ["First message", "Second message", "Third message"]
        embeddings = engine.embed_batch(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_empty_batch_embeddings(self, engine):
        """Test embedding empty batch."""
        embeddings = engine.embed_batch([])
        assert embeddings.shape == (0, 384)

    def test_batch_with_empty_texts(self, engine):
        """Test batch with some empty texts."""
        texts = ["First message", "", "Third message"]
        embeddings = engine.embed_batch(texts)

        assert embeddings.shape == (3, 384)
        assert np.allclose(embeddings[1], np.zeros(384, dtype=np.float32))

    def test_cosine_similarity(self, engine):
        """Test cosine similarity calculation."""
        text1 = "This is about programming."
        text2 = "This is about coding."
        text3 = "This is about cooking."

        emb1 = engine.embed(text1)
        emb2 = engine.embed(text2)
        emb3 = engine.embed(text3)

        sim_12 = engine.cosine_similarity(emb1, emb2)
        sim_13 = engine.cosine_similarity(emb1, emb3)

        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1
        assert sim_12 > sim_13

    def test_self_similarity(self, engine):
        """Test similarity of embedding with itself."""
        text = "Test message"
        embedding = engine.embed(text)
        sim = engine.cosine_similarity(embedding, embedding)

        assert sim > 0.999

    def test_zero_vector_similarity(self, engine):
        """Test similarity with zero vector."""
        text = "Test message"
        embedding = engine.embed(text)
        zero_vector = np.zeros(384, dtype=np.float32)
        sim = engine.cosine_similarity(embedding, zero_vector)

        assert sim == 0.0

    def test_batch_similarity(self, engine):
        """Test batch similarity calculation."""
        query_text = "programming in python"
        candidate_texts = [
            "python code examples",
            "cooking recipes",
            "javascript development",
            "data science with python"
        ]

        query = engine.embed(query_text)
        candidates = engine.embed_batch(candidate_texts)
        similarities = engine.batch_similarity(query, candidates)

        assert len(similarities) == 4
        assert all(0 <= s <= 1 for s in similarities)

    def test_serialize_embedding(self, engine):
        """Test embedding serialization."""
        embedding = np.random.randn(384).astype(np.float32)
        blob = engine.serialize(embedding)

        assert len(blob) == 384 * 4

    def test_serialize_wrong_shape(self, engine):
        """Test serializing embedding with wrong shape."""
        wrong_embedding = np.random.randn(100).astype(np.float32)
        with pytest.raises(ValueError):
            engine.serialize(wrong_embedding)

    def test_deserialize_embedding(self, engine):
        """Test embedding deserialization."""
        original = np.random.randn(384).astype(np.float32)
        blob = engine.serialize(original)
        restored = engine.deserialize(blob)

        assert np.allclose(original, restored)

    def test_deserialize_wrong_size(self, engine):
        """Test deserializing blob with wrong size."""
        wrong_blob = b"wrong size"
        with pytest.raises(ValueError):
            engine.deserialize(wrong_blob)

    def test_deserialize_batch(self, engine):
        """Test batch deserialization."""
        embeddings = np.random.randn(3, 384).astype(np.float32)
        blobs = [engine.serialize(emb) for emb in embeddings]
        restored = engine.deserialize_batch(blobs)

        assert restored.shape == (3, 384)
        assert np.allclose(embeddings, restored)

    def test_deserialize_empty_batch(self, engine):
        """Test deserializing empty batch."""
        restored = engine.deserialize_batch([])
        assert restored.shape == (0, 384)


class TestMessageWriter:
    """Test MessageWriter functionality."""

    @pytest.fixture
    def temp_writer(self):
        """Create temporary writer for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_dir = Path(tmpdir) / ".claude"
            claude_dir.mkdir()

            db_path = claude_dir / "chat.db"
            db = ChatDB(db_path)
            db.init_db()

            writer = MessageWriter(db=db, claude_dir=claude_dir)
            yield writer, db, claude_dir

    def test_write_simple_message(self, temp_writer):
        """Test writing a simple message."""
        writer, db, claude_dir = temp_writer

        uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content="Hello, world!"
        )

        assert uuid is not None

        message = db.get_message_by_uuid(uuid)
        assert message is not None
        assert message.content == "Hello, world!"
        assert message.role == "user"

    def test_write_message_with_parent(self, temp_writer):
        """Test writing a message with parent."""
        writer, db, claude_dir = temp_writer

        parent_uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content="Hello"
        )

        child_uuid = writer.write_message(
            session_id="test-session",
            role="assistant",
            content="Hi there!",
            parent_uuid=parent_uuid
        )

        child = db.get_message_by_uuid(child_uuid)
        assert child.parent_uuid == parent_uuid

    def test_write_message_with_metadata(self, temp_writer):
        """Test writing a message with metadata."""
        writer, db, claude_dir = temp_writer

        metadata = {"usage": {"prompt_tokens": 100}}
        uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content="Test",
            metadata=metadata
        )

        message = db.get_message_by_uuid(uuid)
        assert message.metadata == metadata

    def test_write_long_message_chunking(self, temp_writer):
        """Test that long messages are chunked."""
        writer, db, claude_dir = temp_writer

        long_content = " ".join([f"word {i}" for i in range(600)])
        uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content=long_content
        )

        message = db.get_message_by_uuid(uuid)
        chunks = db.get_message_chunks(message.message_id)

        assert len(chunks) > 1

    def test_write_short_message_single_chunk(self, temp_writer):
        """Test that short messages have single chunk."""
        writer, db, claude_dir = temp_writer

        short_content = "Short message"
        uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content=short_content
        )

        message = db.get_message_by_uuid(uuid)
        chunks = db.get_message_chunks(message.message_id)

        assert len(chunks) == 1

    def test_jsonl_export_created(self, temp_writer):
        """Test that JSONL export is created."""
        writer, db, claude_dir = temp_writer

        writer.write_message(
            session_id="test-session",
            role="user",
            content="Test message"
        )

        jsonl_path = writer._get_jsonl_path("test-session")
        assert jsonl_path.exists()

    def test_md_export_created(self, temp_writer):
        """Test that Markdown export is created."""
        writer, db, claude_dir = temp_writer

        writer.write_message(
            session_id="test-session",
            role="user",
            content="Test message"
        )

        md_path = writer._get_md_path("test-session")
        assert md_path.exists()

    def test_jsonl_export_content(self, temp_writer):
        """Test JSONL export content."""
        writer, db, claude_dir = temp_writer

        uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content="Test message"
        )

        jsonl_path = writer._get_jsonl_path("test-session")
        with open(jsonl_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["uuid"] == uuid
        assert record["type"] == "user"
        assert record["message"]["content"] == "Test message"

    def test_md_export_content(self, temp_writer):
        """Test Markdown export content."""
        writer, db, claude_dir = temp_writer

        writer.write_message(
            session_id="test-session",
            role="user",
            content="Test message"
        )

        md_path = writer._get_md_path("test-session")
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# Session: test-session" in content
        assert "User" in content
        assert "Test message" in content

    def test_multiple_messages_exports(self, temp_writer):
        """Test exports with multiple messages."""
        writer, db, claude_dir = temp_writer

        for i in range(3):
            writer.write_message(
                session_id="test-session",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )

        jsonl_path = writer._get_jsonl_path("test-session")
        with open(jsonl_path, "r") as f:
            lines = f.readlines()

        assert len(lines) > 0

        md_path = writer._get_md_path("test-session")
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Message 0" in content
        assert "Message 1" in content
        assert "Message 2" in content

    def test_message_flags(self, temp_writer):
        """Test writing message with flags."""
        writer, db, claude_dir = temp_writer

        uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content="Test message",
            is_sidechain=True,
            is_meta=True,
            cwd="/test/dir"
        )

        message = db.get_message_by_uuid(uuid)
        assert message.is_sidechain is True
        assert message.is_meta is True
        assert message.cwd == "/test/dir"

    def test_chunk_hashes(self, temp_writer):
        """Test that chunks have correct hashes."""
        writer, db, claude_dir = temp_writer

        content = "Test message for chunking"
        uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content=content
        )

        message = db.get_message_by_uuid(uuid)
        chunks = db.get_message_chunks(message.message_id)

        for chunk in chunks:
            expected_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
            assert chunk.chunk_hash == expected_hash

    def test_embeddings_created(self, temp_writer):
        """Test that embeddings are created for chunks."""
        writer, db, claude_dir = temp_writer

        content = "Test message for embeddings"
        uuid = writer.write_message(
            session_id="test-session",
            role="user",
            content=content
        )

        message = db.get_message_by_uuid(uuid)
        chunks = db.get_message_chunks(message.message_id)
        chunk_hashes = [chunk.chunk_hash for chunk in chunks]

        vectors = db.get_chunk_vectors(chunk_hashes)

        assert len(vectors) == len(chunks)

    def test_append_to_existing_exports(self, temp_writer):
        """Test that new messages are appended to existing exports."""
        writer, db, claude_dir = temp_writer

        writer.write_message(
            session_id="test-session",
            role="user",
            content="First message"
        )

        writer.write_message(
            session_id="test-session",
            role="assistant",
            content="Second message"
        )

        jsonl_path = writer._get_jsonl_path("test-session")
        with open(jsonl_path, "r") as f:
            content = f.read()

        assert "First message" in content
        assert "Second message" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
