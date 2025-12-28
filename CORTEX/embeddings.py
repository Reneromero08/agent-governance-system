#!/usr/bin/env python3
"""
CORTEX Embedding Engine

Generates and manages vector embeddings for semantic search.
Part of the Semantic Core architecture (ADR-030).

Features:
- Sentence transformer embeddings (all-MiniLM-L6-v2)
- Batch processing for efficiency
- Cosine similarity computation
- SQLite BLOB serialization
- Lazy model loading
"""

import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import struct


class EmbeddingEngine:
    """Generate and manage embeddings for CORTEX sections."""

    MODEL_ID = "all-MiniLM-L6-v2"
    DIMENSIONS = 384

    def __init__(self, model_id: Optional[str] = None):
        """Initialize embedding engine.

        Args:
            model_id: Override default model (for testing/migration)
        """
        self.model_id = model_id or self.MODEL_ID
        self._model = None  # Lazy load

    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_id)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            numpy array of shape (384,) with float32 dtype
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.DIMENSIONS, dtype=np.float32)

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            numpy array of shape (len(texts), 384) with float32 dtype
        """
        if not texts:
            return np.array([]).reshape(0, self.DIMENSIONS).astype(np.float32)

        # Filter empty texts and track indices
        non_empty_texts = []
        non_empty_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        # Initialize result with zeros
        result = np.zeros((len(texts), self.DIMENSIONS), dtype=np.float32)

        if non_empty_texts:
            # Encode non-empty texts
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Place embeddings at original indices
            for idx, embedding in zip(non_empty_indices, embeddings):
                result[idx] = embedding.astype(np.float32)

        return result

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Similarity score in range [-1, 1] (typically [0, 1] for this model)
        """
        # Normalize to avoid numerical issues
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def batch_similarity(
        self,
        query: np.ndarray,
        candidates: np.ndarray
    ) -> np.ndarray:
        """Compute similarities between query and multiple candidates.

        Args:
            query: Query embedding of shape (384,)
            candidates: Candidate embeddings of shape (N, 384)

        Returns:
            Array of similarity scores of shape (N,)
        """
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        # Normalize candidates
        candidate_norms = candidates / (
            np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8
        )

        # Compute dot products (cosine similarity with normalized vectors)
        similarities = np.dot(candidate_norms, query_norm)

        return similarities

    def serialize(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding for SQLite storage.

        Format: 384 float32 values packed as little-endian bytes

        Args:
            embedding: numpy array of shape (384,)

        Returns:
            bytes representation (1536 bytes = 384 * 4)
        """
        if embedding.shape != (self.DIMENSIONS,):
            raise ValueError(
                f"Expected embedding shape ({self.DIMENSIONS},), "
                f"got {embedding.shape}"
            )

        return embedding.astype(np.float32).tobytes()

    def deserialize(self, blob: bytes) -> np.ndarray:
        """Deserialize embedding from SQLite storage.

        Args:
            blob: bytes representation from database

        Returns:
            numpy array of shape (384,)
        """
        expected_size = self.DIMENSIONS * 4  # 4 bytes per float32
        if len(blob) != expected_size:
            raise ValueError(
                f"Expected {expected_size} bytes, got {len(blob)}"
            )

        return np.frombuffer(blob, dtype=np.float32)

    def deserialize_batch(self, blobs: List[bytes]) -> np.ndarray:
        """Deserialize multiple embeddings efficiently.

        Args:
            blobs: List of bytes representations

        Returns:
            numpy array of shape (len(blobs), 384)
        """
        if not blobs:
            return np.array([]).reshape(0, self.DIMENSIONS).astype(np.float32)

        result = np.zeros((len(blobs), self.DIMENSIONS), dtype=np.float32)
        for i, blob in enumerate(blobs):
            result[i] = self.deserialize(blob)

        return result


def get_embedding_engine(model_id: Optional[str] = None) -> EmbeddingEngine:
    """Factory function to get embedding engine instance.

    Args:
        model_id: Optional model override

    Returns:
        Configured EmbeddingEngine instance
    """
    return EmbeddingEngine(model_id=model_id)


if __name__ == "__main__":
    # Self-test
    print("Testing EmbeddingEngine...")

    engine = EmbeddingEngine()

    # Test single embedding
    text = "This is a test sentence for embedding generation."
    embedding = engine.embed(text)
    print(f"Single embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32

    # Test batch embedding
    texts = [
        "First sentence",
        "Second sentence",
        "Third sentence with more words"
    ]
    embeddings = engine.embed_batch(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (3, 384)

    # Test similarity
    sim = engine.cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first two: {sim:.4f}")
    assert 0 <= sim <= 1

    # Test serialization
    blob = engine.serialize(embedding)
    print(f"Serialized size: {len(blob)} bytes")
    assert len(blob) == 384 * 4

    # Test deserialization
    restored = engine.deserialize(blob)
    print(f"Restored shape: {restored.shape}")
    assert np.allclose(embedding, restored)

    # Test batch similarity
    query = embeddings[0]
    similarities = engine.batch_similarity(query, embeddings)
    print(f"Batch similarities: {similarities}")
    assert len(similarities) == 3
    assert similarities[0] > 0.99  # Self-similarity should be ~1.0

    print("\nAll tests passed!")
