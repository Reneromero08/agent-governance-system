#!/usr/bin/env python3
"""
Retrieve context from test sandbox database.

Minimal version of cortex_geometric for isolated testing.
Uses E-gating (Born rule) for relevance filtering.
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine


def retrieve(query: str, k: int = 5, threshold: float = 0.3, domain: str = None) -> List[str]:
    """
    Retrieve context from test sandbox database.

    Args:
        query: Search query
        k: Number of results
        threshold: E threshold for filtering (Born rule)
        domain: Optional domain filter (math, code, logic, chemistry)

    Returns:
        List of context strings
    """
    # Find database - try multiple locations
    possible_paths = [
        Path(__file__).parent / "test_sandbox.db",
        Path("test_sandbox.db"),
        Path().absolute() / "test_sandbox.db"
    ]

    db_path = None
    for path in possible_paths:
        if path.exists():
            db_path = path
            break

    if db_path is None:
        print(f"[ERROR] Test DB not found. Tried:")
        for p in possible_paths:
            print(f"  - {p.absolute()} (exists: {p.exists()})")
        return []

    engine = EmbeddingEngine()
    query_vec = engine.embed(query)
    query_vec = query_vec / np.linalg.norm(query_vec)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Get all vectors
    cur.execute("SELECT doc_id, vector_blob, Df FROM geometric_index")
    rows = cur.fetchall()

    results = []
    for doc_id, vector_blob, Df in rows:
        doc_vec = np.frombuffer(vector_blob, dtype=np.float32)
        doc_vec = doc_vec / np.linalg.norm(doc_vec)

        # E = <psi|phi> (Born rule, Q44 validated)
        E = float(np.dot(query_vec, doc_vec))

        if E >= threshold:
            # Get content from chunks
            cur.execute("SELECT content, domain FROM chunks WHERE chunk_id LIKE ?", (f"%{doc_id[:8]}%",))
            content_row = cur.fetchone()
            if content_row:
                content, doc_domain = content_row

                # Apply domain filter if specified
                if domain is None or doc_domain == domain:
                    results.append((E, content))

    conn.close()

    # Sort by E descending
    results.sort(key=lambda x: x[0], reverse=True)

    # Return top k contents
    return [content for E, content in results[:k]]


def retrieve_with_scores(query: str, k: int = 5, threshold: float = 0.3, domain: str = None) -> List[Tuple[float, float, str]]:
    """
    Retrieve with E scores and Df values.

    Returns:
        List of (E, Df, content) tuples
    """
    # Find database - try multiple locations
    possible_paths = [
        Path(__file__).parent / "test_sandbox.db",
        Path("test_sandbox.db"),
        Path().absolute() / "test_sandbox.db"
    ]

    db_path = None
    for path in possible_paths:
        if path.exists():
            db_path = path
            break

    if db_path is None:
        return []

    engine = EmbeddingEngine()
    query_vec = engine.embed(query)
    query_vec = query_vec / np.linalg.norm(query_vec)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT doc_id, vector_blob, Df FROM geometric_index")
    rows = cur.fetchall()

    # Simpler approach: match by content_preview
    doc_to_content = {}
    for doc_id, vector_blob, Df in rows:
        cur.execute("SELECT content_preview FROM geometric_index WHERE doc_id = ?", (doc_id,))
        preview = cur.fetchone()[0]

        # Find matching chunk by preview
        cur.execute("SELECT content, domain FROM chunks WHERE content LIKE ?", (preview[:100] + "%",))
        match = cur.fetchone()
        if match:
            doc_to_content[doc_id] = (match[0], match[1], Df)

    results = []
    for doc_id, vector_blob, Df in rows:
        doc_vec = np.frombuffer(vector_blob, dtype=np.float32)
        doc_vec = doc_vec / np.linalg.norm(doc_vec)

        # E = <psi|phi> (Born rule)
        E = float(np.dot(query_vec, doc_vec))

        if E >= threshold:
            if doc_id in doc_to_content:
                content, doc_domain, df_val = doc_to_content[doc_id]

                # Apply domain filter if specified
                if domain is None or doc_domain == domain:
                    results.append((E, Df, content))

    conn.close()

    # Sort by E descending
    results.sort(key=lambda x: x[0], reverse=True)

    return results[:k]


if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python retrieve.py <query>")
        print('Example: python retrieve.py "how to solve quadratic equations"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n")

    results = retrieve_with_scores(query, k=3)

    if not results:
        print("No results found. Did you build the database?")
        print("Run: python test_sandbox/build_test_db.py")
    else:
        print(f"Retrieved {len(results)} documents:\n")
        for i, (E, Df, content) in enumerate(results, 1):
            preview = content[:150].replace('\n', ' ')
            print(f"[{i}] E={E:.3f}, Df={Df:.1f}")
            print(f"    {preview}...\n")
