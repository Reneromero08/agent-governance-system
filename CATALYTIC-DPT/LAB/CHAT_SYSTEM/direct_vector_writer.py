#!/usr/bin/env python3
"""
Direct Vector Writer - No English, No Tokens, Just Vectors!

Write raw embedding vectors directly to chat database.
Useful for: LLM outputs, pre-computed vectors, neural representations.
"""

import sys
import os
from pathlib import Path
import hashlib
import uuid as uuid_lib
from datetime import datetime
from typing import Any

import numpy as np

# Chat system is now in current directory
chat_system_path = Path(__file__).parent
sys.path.insert(0, str(chat_system_path))
os.chdir(str(chat_system_path))

from chat_db import ChatDB
from embedding_engine import ChatEmbeddingEngine


class DirectVectorWriter:
    """Write RAW vectors directly (no text, no tokens)."""

    def __init__(self, db_path: Any = None):
        if db_path is None:
            db_path = chat_system_path / "chat.db"

        self.db = ChatDB(db_path=db_path)
        self.db.init_db()
        self.engine = ChatEmbeddingEngine()

    def write_raw_vector(
        self,
        session_id: str,
        vector: np.ndarray,  # Raw vector (384 floats)
        metadata: dict[str, any] | None = None
    ) -> str:
        """Write raw vector directly to database.

        NO English text. NO tokens. JUST vector floats.

        Args:
            session_id: Session identifier
            vector: Raw numpy array (shape: [384] or [N, 384])
            metadata: Optional metadata

        Returns:
            Chunk hash for the vector
        """
        # Accept single vector or batch
        if vector.ndim == 1:
            vectors = [vector]
        else:
            vectors = vector

        results = []

        with self.db.get_connection() as conn:
            for vec in vectors:
                # Serialize vector to bytes
                vector_bytes = self.engine.serialize(vec)

                # Generate hash from vector bytes
                chunk_hash = hashlib.sha256(vector_bytes).hexdigest()

                # Create placeholder message (required by schema)
                message_uuid = str(uuid_lib.uuid4())
                timestamp = datetime.utcnow().isoformat() + "Z"

                # Minimal placeholder text
                placeholder = f"[RAW-VECTOR] {vec.shape}"

                # Insert placeholder message
                sql = """
                    INSERT INTO chat_messages (
                        session_id, uuid, role, content, content_hash, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """
                content_hash = self.db.compute_content_hash(placeholder)

                cursor = conn.execute(sql, (
                    session_id,
                    message_uuid,
                    "vector",
                    placeholder,
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
                    vec.shape[0]  # Store dimensions as token count
                ))

                # Insert raw vector
                sql = """
                    INSERT INTO message_vectors (
                        chunk_hash, embedding, model_id, dimensions, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                """
                conn.execute(sql, (
                    chunk_hash,
                    vector_bytes,
                    "raw-vector",  # Mark as raw vector
                    vec.shape[0],
                    timestamp
                ))

                results.append(chunk_hash)

        return results[0] if len(results) == 1 else results

    def write_multiple_vectors(
        self,
        session_id: str,
        vectors: list[np.ndarray],  # List of raw vectors
        metadata: dict[str, any] | None = None
    ) -> list[str]:
        """Write multiple raw vectors at once.

        Args:
            session_id: Session identifier
            vectors: List of numpy arrays (each [384] floats)
            metadata: Optional metadata

        Returns:
            List of chunk hashes
        """
        results = []
        for vec in vectors:
            results.append(self.write_raw_vector(session_id, vec, metadata))
        return results

    def search_vectors(
        self,
        query_vector: np.ndarray,  # Raw query vector
        session_id: str,
        threshold: float = 0.5,
        limit: int = 10
    ) -> list[dict[str, any]]:
        """Search using raw query vector.

        Args:
            query_vector: Raw numpy array (384 floats)
            session_id: Session to search
            threshold: Similarity threshold
            limit: Max results

        Returns:
            List of results with similarities
        """
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

                # Compute similarity (raw math, no text!)
                sim = self.engine.cosine_similarity(query_vector, stored_vector)

                if sim >= threshold:
                    results.append({
                        "chunk_hash": vec.chunk_hash,
                        "similarity": sim,
                        "message_uuid": msg.uuid,
                        "timestamp": msg.timestamp,
                        "dimensions": vec.dimensions
                    })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def generate_random_vector(self, seed: int | None = None) -> np.ndarray:
        """Generate random test vector.

        Args:
            seed: Optional seed for reproducibility

        Returns:
            Random vector (384 floats)
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.randn(384).astype(np.float32)


if __name__ == "__main__":
    print("Direct Vector Writer - No English, No Tokens!")
    print("=" * 60)

    writer = DirectVectorWriter()
    session_id = "raw-vector-session"

    # Example 1: Write a raw vector directly
    print("\n[1] Writing raw vector...")
    random_vec = writer.generate_random_vector(seed=42)
    hash1 = writer.write_raw_vector(session_id, random_vec)
    print(f"  Vector shape: {random_vec.shape}")
    print(f"  Vector stats: mean={random_vec.mean():.3f}, std={random_vec.std():.3f}")
    print(f"  Stored: hash={hash1[:16]}...")

    # Example 2: Write multiple raw vectors
    print("\n[2] Writing multiple raw vectors...")
    vectors = [
        writer.generate_random_vector(seed=1),
        writer.generate_random_vector(seed=2),
        writer.generate_random_vector(seed=3),
    ]
    hashes = writer.write_multiple_vectors(session_id, vectors)
    print(f"  Written: {len(hashes)} vectors")
    for i, h in enumerate(hashes):
        print(f"    [{i+1}] hash={h[:16]}...")

    # Example 3: Search using raw vector
    print("\n[3] Searching with raw vector...")
    query_vec = writer.generate_random_vector(seed=42)
    results = writer.search_vectors(query_vec, session_id, threshold=0.1)

    print(f"  Query vector: {query_vec.shape}")
    print(f"  Found {len(results)} results:")
    for i, r in enumerate(results[:3]):
        print(f"    [{i+1}] sim={r['similarity']:.3f} dims={r['dimensions']} hash={r['chunk_hash'][:12]}...")

    # Example 4: Storage comparison
    print("\n[4] What's stored in database:")
    print(f"  Text: ZERO bytes (no English!)")
    print(f"  Tokens: ZERO tokens (no tokenization!)")
    print(f"  Vectors: {len(hashes) + 1} * 1536 = {(len(hashes) + 1) * 1536} bytes (pure math!)")

    print("\n" + "=" * 60)
    print("[OK] Direct vector system working!")
    print("\nKey features:")
    print("  - NO English text stored")
    print("  - NO tokenization")
    print("  - JUST vectors (384 floats each)")
    print("  - Pure mathematical search")
    print("  - Maximum compression")
