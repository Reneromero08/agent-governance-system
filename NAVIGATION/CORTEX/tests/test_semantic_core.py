#!/usr/bin/env python3
"""
Test suite for Semantic Core (Phase 1: Vector Foundation)

Tests:
1. EmbeddingEngine functionality
2. Vector serialization/deserialization
3. Similarity computation
4. Database schema creation
5. Vector indexing
6. Semantic search

Run: python test_semantic_core.py
"""

import sqlite3
import tempfile
import shutil
from pathlib import Path
import sys

# Add Root to path for imports
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
except ImportError:
    np = None

# Test status tracking
tests_passed = 0
tests_failed = 0

def cortex_test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global tests_passed, tests_failed
            if np is None:
                print(f"  [SKIP] {name} (numpy missing)")
                return
            try:
                print(f"\n[TEST] {name}")
                func()
                print(f"  [PASS] {name}")
                tests_passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {name}: {e}")
                tests_failed += 1
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                tests_failed += 1
        return wrapper
    return decorator


@cortex_test("EmbeddingEngine initialization")
def test_embedding_init():
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    assert engine.MODEL_ID == "all-MiniLM-L6-v2"
    assert engine.DIMENSIONS == 384
    print(f"    Model: {engine.MODEL_ID}, Dimensions: {engine.DIMENSIONS}")


@cortex_test("Single text embedding")
def test_single_embedding():
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    text = "This is a test sentence for semantic embedding."

    embedding = engine.embed(text)

    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
    assert not np.all(embedding == 0), "Embedding should not be all zeros"

    print(f"    Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"    Sample values: {embedding[:5]}")


@cortex_test("Batch embedding")
def test_batch_embedding():
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    texts = [
        "First sentence about programming",
        "Second sentence about cooking",
        "Third sentence about programming languages"
    ]

    embeddings = engine.embed_batch(texts)

    assert embeddings.shape == (3, 384), f"Expected shape (3, 384), got {embeddings.shape}"
    assert embeddings.dtype == np.float32

    # Check that similar sentences have higher similarity
    sim_1_3 = engine.cosine_similarity(embeddings[0], embeddings[2])
    sim_1_2 = engine.cosine_similarity(embeddings[0], embeddings[1])

    print(f"    Similarity (programming vs programming): {sim_1_3:.3f}")
    print(f"    Similarity (programming vs cooking): {sim_1_2:.3f}")

    # Programming sentences should be more similar to each other
    assert sim_1_3 > sim_1_2, "Related sentences should have higher similarity"


@cortex_test("Embedding serialization")
def test_serialization():
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    original = engine.embed("Test text for serialization")

    # Serialize
    blob = engine.serialize(original)
    assert isinstance(blob, bytes)
    assert len(blob) == 384 * 4, f"Expected 1536 bytes, got {len(blob)}"

    # Deserialize
    restored = engine.deserialize(blob)
    assert restored.shape == original.shape
    assert np.allclose(original, restored), "Restored embedding should match original"

    print(f"    Serialized size: {len(blob)} bytes")
    print(f"    Match after deserialization: {np.allclose(original, restored)}")


@cortex_test("Cosine similarity")
def test_cosine_similarity():
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()

    text1 = "The cat sat on the mat"
    text2 = "The cat sat on the mat"
    text3 = "Quantum physics is complex"

    emb1 = engine.embed(text1)
    emb2 = engine.embed(text2)
    emb3 = engine.embed(text3)

    # Identical texts should have similarity ~1.0
    sim_identical = engine.cosine_similarity(emb1, emb2)
    assert sim_identical > 0.99, f"Identical texts should have similarity > 0.99, got {sim_identical}"

    # Unrelated texts should have lower similarity
    sim_unrelated = engine.cosine_similarity(emb1, emb3)
    assert sim_unrelated < sim_identical

    print(f"    Identical text similarity: {sim_identical:.4f}")
    print(f"    Unrelated text similarity: {sim_unrelated:.4f}")


@cortex_test("Batch similarity computation")
def test_batch_similarity():
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()

    query = engine.embed("programming language")
    candidates = engine.embed_batch([
        "Python is a programming language",
        "Cooking pasta requires boiling water",
        "JavaScript is used for web programming"
    ])

    similarities = engine.batch_similarity(query, candidates)

    assert len(similarities) == 3
    assert similarities[0] > similarities[1], "Programming related should be more similar"
    assert similarities[2] > similarities[1], "Programming related should be more similar"

    print(f"    Similarities: {similarities}")


@cortex_test("Vector schema creation")
def test_schema_creation():
    from NAVIGATION.CORTEX.semantic.vector_indexer import VectorIndexer

    # Create temporary database
    temp_dir = Path(tempfile.mkdtemp())
    try:
        db_path = temp_dir / "test.db"

        indexer = VectorIndexer(db_path=db_path)

        # Check tables exist
        cursor = indexer.conn.execute("""
            SELECT name FROM sqlite_master WHERE type='table'
            AND name IN ('section_vectors', 'embedding_metadata')
        """)
        tables = [row[0] for row in cursor.fetchall()]

        assert 'section_vectors' in tables, "section_vectors table should exist"
        assert 'embedding_metadata' in tables, "embedding_metadata table should exist"

        # Check metadata
        cursor = indexer.conn.execute("SELECT * FROM embedding_metadata")
        metadata = cursor.fetchone()
        assert metadata is not None, "Metadata should be initialized"

        print(f"    Tables created: {tables}")
        print(f"    Default model: {metadata[1]}")

        indexer.close()

    finally:
        shutil.rmtree(temp_dir)


@cortex_test("Vector indexing")
def test_vector_indexing():
    from NAVIGATION.CORTEX.semantic.vector_indexer import VectorIndexer
    import hashlib

    temp_dir = Path(tempfile.mkdtemp())
    try:
        db_path = temp_dir / "test.db"

        # Create database and add test section
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sections (
                hash TEXT PRIMARY KEY,
                content TEXT,
                file_path TEXT,
                section_name TEXT
            )
        """)

        test_content = "This is a test section about programming"
        test_hash = hashlib.sha256(test_content.encode()).hexdigest()

        conn.execute("""
            INSERT INTO sections (hash, content, file_path, section_name)
            VALUES (?, ?, ?, ?)
        """, (test_hash, test_content, "test.py", "test_section"))
        conn.commit()
        conn.close()

        # Index it
        indexer = VectorIndexer(db_path=db_path)
        success = indexer.index_section(test_hash, test_content)

        assert success, "Indexing should succeed"

        # Verify embedding exists
        cursor = indexer.conn.execute("""
            SELECT hash, embedding FROM section_vectors WHERE hash = ?
        """, (test_hash,))
        row = cursor.fetchone()

        assert row is not None, "Embedding should be stored"
        assert len(row['embedding']) == 384 * 4, "Embedding should be correct size"

        print(f"    Indexed hash: {test_hash[:16]}...")
        print(f"    Embedding size: {len(row['embedding'])} bytes")

        indexer.close()

    finally:
        shutil.rmtree(temp_dir)


@cortex_test("Semantic search")
def test_semantic_search():
    from NAVIGATION.CORTEX.semantic.vector_indexer import VectorIndexer
    from NAVIGATION.CORTEX.semantic.semantic_search import SemanticSearch
    import hashlib

    temp_dir = Path(tempfile.mkdtemp())
    try:
        db_path = temp_dir / "test.db"

        # Create database with test sections
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sections (
                hash TEXT PRIMARY KEY,
                content TEXT,
                file_path TEXT,
                section_name TEXT,
                line_range TEXT
            )
        """)

        test_sections = [
            ("Python is a high-level programming language", "python.md", "intro"),
            ("Cooking pasta requires boiling water", "cooking.md", "pasta"),
            ("JavaScript is used for web development", "javascript.md", "intro"),
        ]

        for content, file_path, section_name in test_sections:
            hash_val = hashlib.sha256(content.encode()).hexdigest()
            conn.execute("""
                INSERT INTO sections (hash, content, file_path, section_name)
                VALUES (?, ?, ?, ?)
            """, (hash_val, content, file_path, section_name))

        conn.commit()
        conn.close()

        # Index all sections
        indexer = VectorIndexer(db_path=db_path)
        stats = indexer.index_all(verbose=False)
        assert stats['indexed'] == 3, f"Should index 3 sections, got {stats['indexed']}"
        indexer.close()

        # Search
        searcher = SemanticSearch(db_path=db_path)
        results = searcher.search("programming languages", top_k=3)

        assert len(results) > 0, "Should return results"
        assert results[0].similarity > 0.5, "Top result should have high similarity"

        # Programming-related should be top results
        top_files = [r.file_path for r in results[:2]]
        assert "python.md" in top_files or "javascript.md" in top_files

        print(f"    Found {len(results)} results")
        print(f"    Top result: {results[0].section_name} (similarity: {results[0].similarity:.3f})")

        searcher.close()

    finally:
        shutil.rmtree(temp_dir)


@cortex_test("Empty text handling")
def test_empty_text():
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()

    # Empty string
    emb1 = engine.embed("")
    assert np.all(emb1 == 0), "Empty text should produce zero vector"

    # Whitespace only
    emb2 = engine.embed("   ")
    assert np.all(emb2 == 0), "Whitespace should produce zero vector"

    # Batch with empty
    embeddings = engine.embed_batch(["valid text", "", "  ", "another valid"])
    assert embeddings.shape == (4, 384)
    assert not np.all(embeddings[0] == 0), "Valid text should have non-zero embedding"
    assert np.all(embeddings[1] == 0), "Empty should be zero"
    assert np.all(embeddings[2] == 0), "Whitespace should be zero"

    print(f"    Empty text handled correctly")


def main():
    print("="*60)
    print("CORTEX Semantic Core - Phase 1 Test Suite")
    print("="*60)

    # Run all tests
    test_embedding_init()
    test_single_embedding()
    test_batch_embedding()
    test_serialization()
    test_cosine_similarity()
    test_batch_similarity()
    test_schema_creation()
    test_vector_indexing()
    test_semantic_search()
    test_empty_text()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print(f"Total:  {tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print(f"\n[FAIL] {tests_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
