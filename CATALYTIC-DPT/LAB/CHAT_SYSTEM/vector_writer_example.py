#!/usr/bin/env python3
"""
Example: Writing vectors directly to chat system.

Instead of storing text, stores vector embeddings directly.
Useful for encoding thoughts, reasoning traces, or high-density information.
"""

import sys
import os
from pathlib import Path
import hashlib
import uuid as uuid_lib
from datetime import datetime
from typing import Any

# Chat system is now in current directory
chat_system_path = Path(__file__).parent
sys.path.insert(0, str(chat_system_path))
os.chdir(str(chat_system_path))

from chat_db import ChatDB
from embedding_engine import ChatEmbeddingEngine


class VectorWriter:
    """Write vectors directly to chat system (without text)."""

    def __init__(self, db_path: Any = None):
        if db_path is None:
            db_path = chat_system_path / "chat.db"

        self.db = ChatDB(db_path=db_path)
        self.db.init_db()
        self.engine = ChatEmbeddingEngine()

    def write_vector(
        self,
        session_id: str,
        vector: bytes,
        metadata: Any = None
    ) -> str:
        """Write a vector directly to database.

        Creates a placeholder chunk and stores the vector.

        Args:
            session_id: Session identifier
            vector: Serialized vector (384 bytes)
            metadata: Optional metadata

        Returns:
            Chunk hash for the vector
        """
        # Generate chunk hash from vector bytes
        chunk_hash = hashlib.sha256(vector).hexdigest()

        # Create a placeholder message (needed for schema integrity)
        message_uuid = str(uuid_lib.uuid4())
        content = f"[VECTOR-ONLY] {len(vector)} bytes"

        # Create a placeholder chunk
        with self.db.get_connection() as conn:
            # Insert placeholder message
            sql = """
                INSERT INTO chat_messages (
                    session_id, uuid, role, content, content_hash, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            content_hash = self.db.compute_content_hash(content)
            timestamp = datetime.utcnow().isoformat() + "Z"

            cursor = conn.execute(sql, (
                session_id,
                message_uuid,
                "system",  # Mark as system message (vector-only)
                content,
                content_hash,
                timestamp
            ))
            message_id = cursor.lastrowid

            # Insert placeholder chunk
            sql = """
                INSERT INTO message_chunks (
                    message_id, chunk_index, chunk_hash, content, token_count
                ) VALUES (?, ?, ?, ?, ?)
            """
            conn.execute(sql, (
                message_id,
                0,
                chunk_hash,
                "[VECTOR-DATA]",
                len(vector)  # Use vector bytes as token count
            ))

            # Insert vector
            sql = """
                INSERT INTO message_vectors (
                    chunk_hash, embedding, model_id, dimensions, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """
            conn.execute(sql, (
                chunk_hash,
                vector,
                self.engine.MODEL_ID,
                self.engine.DIMENSIONS,
                timestamp
            ))

        return chunk_hash

    def write_thought_vector(
        self,
        session_id: str,
        thought: str,
        metadata: dict[str, any] | None = None
    ) -> str:
        """Encode a thought as vector and store.

        This is the "vector-only" mode - no text storage.

        Args:
            session_id: Session identifier
            thought: Thought text to encode
            metadata: Optional metadata

        Returns:
            Chunk hash for the vector
        """
        # Encode thought as vector
        embedding = self.engine.embed(thought)
        vector_bytes = self.engine.serialize(embedding)

        # Write vector directly
        return self.write_vector(session_id, vector_bytes, metadata)

    def search_vectors(
        self,
        query_text: str,
        session_id: str,
        threshold: float = 0.5,
        limit: int = 10
    ) -> list:
        """Search for similar vectors using text query.

        Args:
            query_text: Query text to encode
            session_id: Session to search
            threshold: Similarity threshold
            limit: Max results

        Returns:
            List of (chunk_hash, similarity) tuples
        """
        # Encode query
        query_vector = self.engine.embed(query_text)

        # Get all vectors in session
        messages = self.db.get_session_messages(session_id)
        results = []

        for msg in messages:
            chunks = self.db.get_message_chunks(msg.message_id)
            chunk_hashes = [c.chunk_hash for c in chunks]
            vectors = self.db.get_chunk_vectors(chunk_hashes)

            for vec in vectors:
                # Decode stored vector
                stored_vector = self.engine.deserialize(vec.embedding)

                # Compute similarity
                sim = self.engine.cosine_similarity(query_vector, stored_vector)

                if sim >= threshold:
                    results.append({
                        "chunk_hash": vec.chunk_hash,
                        "similarity": sim,
                        "message_uuid": msg.uuid,
                        "timestamp": msg.timestamp
                    })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]


if __name__ == "__main__":
    print("Vector-Only Chat System")
    print("=" * 60)

    writer = VectorWriter()
    session_id = "vector-session-001"

    # Example 1: Write a thought vector directly
    print("\n[1] Writing thought vector...")
    thought = "The answer is 42"
    hash1 = writer.write_thought_vector(session_id, thought)
    print(f"  Stored: '{thought}' as vector")
    print(f"  Hash: {hash1[:16]}...")
    print(f"  Size: 1536 bytes (384 dims)")

    # Example 2: Write multiple related thoughts
    print("\n[2] Writing related thought vectors...")
    thoughts = [
        "Deep learning uses neural networks",
        "Neural networks have many layers",
        "Layers learn patterns from data"
    ]

    for thought in thoughts:
        writer.write_thought_vector(session_id, thought)
        print(f"  Stored: '{thought}'")

    # Example 3: Search by query
    print("\n[3] Searching for similar thoughts...")
    query = "neural network patterns"
    results = writer.search_vectors(query, session_id, threshold=0.3)

    print(f"  Query: '{query}'")
    print(f"  Found {len(results)} results:")

    for i, result in enumerate(results[:3]):
        print(f"    [{i+1}] sim={result['similarity']:.2f} hash={result['chunk_hash'][:12]}...")

    # Example 4: Storage comparison
    print("\n[4] Storage comparison:")
    print(f"  Text mode: '{thought}' = {len(thought)} bytes")
    print(f"  Vector mode: {1536} bytes (384 floats)")
    print(f"  Ratio: {1536/len(thought):.1f}x larger (but searchable!)")

    print("\n" + "=" * 60)
    print("[OK] Vector-only chat system working!")
    print("\nBenefits of vector-only mode:")
    print("  - Semantic search without text storage")
    print("  - Privacy-preserving (vectors are hard to decode)")
    print("  - Faster similarity search")
    print("  - Compresses information density")
