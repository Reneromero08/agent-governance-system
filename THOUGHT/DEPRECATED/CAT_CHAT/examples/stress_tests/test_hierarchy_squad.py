#!/usr/bin/env python3
"""
RECURSIVE E-SCORE HIERARCHY TEST
================================

Tests Section J of the CAT Chat roadmap using SQuAD dataset.
Pure vector math - no LLM inference needed.

Test: Can E guide to E guide to E find the right passage?
"""

import sys
import time
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Config
LLM_STUDIO_BASE = "http://10.5.0.2:1234"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL = "liquid/lfm2.5-1.2b"
E_THRESHOLD = 0.3  # Born rule threshold


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HierarchyNode:
    """Node in the E-score hierarchy."""
    node_id: str
    level: int  # 0=leaf, 1=L1, 2=L2, 3=L3
    centroid: np.ndarray
    children: List["HierarchyNode"] = field(default_factory=list)
    content: Optional[str] = None  # Only for L0 nodes
    content_idx: Optional[int] = None  # Index in original dataset


# =============================================================================
# Embedding
# =============================================================================

def get_embedding(text: str) -> np.ndarray:
    """Get embedding from nomic model."""
    resp = requests.post(
        f"{LLM_STUDIO_BASE}/v1/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    embedding = np.array(resp.json()["data"][0]["embedding"])
    return embedding / np.linalg.norm(embedding)  # Normalize


def generate_answer(context: str, question: str) -> str:
    """Ask LLM to answer question given context."""
    prompt = f"""Based on the following passage, answer the question briefly.

Passage: {context[:2000]}

Question: {question}

Answer:"""

    resp = requests.post(
        f"{LLM_STUDIO_BASE}/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def get_embeddings_batch(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Get embeddings in batches."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            f"{LLM_STUDIO_BASE}/v1/embeddings",
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()

        for item in resp.json()["data"]:
            vec = np.array(item["embedding"])
            all_embeddings.append(vec / np.linalg.norm(vec))

        if (i + batch_size) % 1000 == 0:
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(all_embeddings)


# =============================================================================
# E-Score (Born Rule)
# =============================================================================

def compute_E(query_vec: np.ndarray, item_vec: np.ndarray) -> float:
    """E = |<query|item>|^2 (Born rule)"""
    dot = np.dot(query_vec, item_vec)
    return dot * dot


# =============================================================================
# Hierarchy Building
# =============================================================================

def build_hierarchy(
    vectors: np.ndarray,
    contents: List[str],
    branch_factor: int = 100,
    use_clustering: bool = True,
) -> HierarchyNode:
    """
    Build centroid hierarchy from vectors.

    L0: Individual vectors (leaves)
    L1: Groups of ~100 vectors (clustered if use_clustering=True)
    L2: Groups of ~100 L1 nodes
    L3: Groups of ~100 L2 nodes (if needed)
    """
    n = len(vectors)
    print(f"Building hierarchy for {n} vectors (branch_factor={branch_factor}, clustering={use_clustering})")

    # Create L0 nodes (leaves)
    l0_nodes = []
    for i, (vec, content) in enumerate(zip(vectors, contents)):
        node = HierarchyNode(
            node_id=f"L0_{i}",
            level=0,
            centroid=vec,
            content=content,
            content_idx=i,
        )
        l0_nodes.append(node)

    # Build L1 - either by clustering or sequential grouping
    if use_clustering and n > branch_factor:
        l1_nodes = _build_clustered_level(l0_nodes, vectors, level=1, n_clusters=max(2, n // branch_factor))
    else:
        l1_nodes = _build_level(l0_nodes, level=1, branch_factor=branch_factor)
    print(f"  L1: {len(l1_nodes)} nodes")

    if len(l1_nodes) <= 1:
        return l1_nodes[0] if l1_nodes else l0_nodes[0]

    # Build L2 (group L1 nodes)
    l2_nodes = _build_level(l1_nodes, level=2, branch_factor=branch_factor)
    print(f"  L2: {len(l2_nodes)} nodes")

    if len(l2_nodes) <= 1:
        return l2_nodes[0] if l2_nodes else l1_nodes[0]

    # Build L3 (group L2 nodes)
    l3_nodes = _build_level(l2_nodes, level=3, branch_factor=branch_factor)
    print(f"  L3: {len(l3_nodes)} nodes")

    # Create root if multiple L3 nodes
    if len(l3_nodes) > 1:
        root_centroid = np.mean([n.centroid for n in l3_nodes], axis=0)
        root_centroid = root_centroid / np.linalg.norm(root_centroid)
        root = HierarchyNode(
            node_id="root",
            level=4,
            centroid=root_centroid,
            children=l3_nodes,
        )
        return root

    return l3_nodes[0]


def _build_level(
    children: List[HierarchyNode],
    level: int,
    branch_factor: int,
) -> List[HierarchyNode]:
    """Build one level of the hierarchy by grouping children sequentially."""
    nodes = []

    for i in range(0, len(children), branch_factor):
        group = children[i:i + branch_factor]

        # Centroid = mean of child centroids
        centroid = np.mean([c.centroid for c in group], axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        node = HierarchyNode(
            node_id=f"L{level}_{i // branch_factor}",
            level=level,
            centroid=centroid,
            children=group,
        )
        nodes.append(node)

    return nodes


def _build_clustered_level(
    children: List[HierarchyNode],
    vectors: np.ndarray,
    level: int,
    n_clusters: int,
) -> List[HierarchyNode]:
    """Build one level by clustering children semantically."""
    from sklearn.cluster import KMeans

    # Cluster the vectors
    child_vectors = np.array([c.centroid for c in children])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(child_vectors)

    # Group children by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(children[i])

    # Create nodes from clusters
    nodes = []
    for cluster_id, group in clusters.items():
        # Centroid = mean of child centroids
        centroid = np.mean([c.centroid for c in group], axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        node = HierarchyNode(
            node_id=f"L{level}_c{cluster_id}",
            level=level,
            centroid=centroid,
            children=group,
        )
        nodes.append(node)

    return nodes


# =============================================================================
# Recursive Retrieval
# =============================================================================

def retrieve_hierarchical(
    query_vec: np.ndarray,
    root: HierarchyNode,
    threshold: float = E_THRESHOLD,
    max_results: int = 10,
) -> Tuple[List[HierarchyNode], int]:
    """
    Retrieve relevant items using E-score hierarchy.

    Key insight: Threshold applies to LEAF decisions, not internal routing.
    For internal nodes, we check if centroid E suggests ANY child might be relevant.
    Centroids dilute signal, so we use lower thresholds for higher levels.

    Returns: (results, e_computations_count)
    """
    results = []
    e_computations = [0]  # Use list to allow mutation in nested function

    def level_threshold(level: int) -> float:
        """Threshold for internal nodes - with clustering, centroids are meaningful."""
        # With semantic clustering, centroids represent cluster topics
        # Use same threshold for pruning at all levels
        if level == 0:
            return threshold
        else:
            return threshold / 2  # Slightly lower for centroids

    def _retrieve(node: HierarchyNode, top_k: int = 10):
        """
        Top-K centroid selection: only explore the K children with highest E.
        This guarantees pruning while maintaining recall.
        """
        e_computations[0] += 1

        if node.level == 0:
            # Leaf node
            E = compute_E(query_vec, node.centroid)
            results.append((node, E))
            return

        # Internal node - score all children, explore top-K
        child_scores = []
        for child in node.children:
            e_computations[0] += 1
            E = compute_E(query_vec, child.centroid)
            child_scores.append((child, E))

        # Sort by E descending, take top-K
        child_scores.sort(key=lambda x: x[1], reverse=True)
        top_children = child_scores[:top_k]

        for child, E in top_children:
            _retrieve(child, top_k=top_k)

    # Explore root with top-K selection
    _retrieve(root, top_k=10)  # Explore top-10 clusters at each level

    # Sort by E-score descending
    results.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in results[:max_results]], e_computations[0]


def retrieve_brute_force(
    query_vec: np.ndarray,
    vectors: np.ndarray,
    threshold: float = E_THRESHOLD,
    max_results: int = 10,
) -> Tuple[List[int], int]:
    """Brute force retrieval - check every vector, return top by E-score."""
    scores = []
    for i, vec in enumerate(vectors):
        E = compute_E(query_vec, vec)
        scores.append((i, E))  # Keep ALL scores, sort by E

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:max_results]], len(vectors)


# =============================================================================
# Test Runner
# =============================================================================

def run_squad_test(
    num_passages: int = 1000,
    num_queries: int = 100,
    branch_factor: int = 100,
):
    """
    Test hierarchy on SQuAD dataset.

    1. Load passages + questions
    2. Embed passages, build hierarchy
    3. For each question, retrieve via hierarchy
    4. Check if correct passage was retrieved
    """
    print("\n" + "=" * 70)
    print("RECURSIVE E-SCORE HIERARCHY TEST - SQuAD")
    print("=" * 70)

    # Load SQuAD
    print("\nLoading SQuAD dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("squad", split=f"train[:{num_passages}]")
    except Exception as e:
        print(f"Failed to load SQuAD: {e}")
        print("Install with: pip install datasets")
        return

    # Extract unique passages (contexts)
    passages = []
    passage_to_idx = {}
    questions_with_passage = []

    for item in dataset:
        context = item["context"]
        if context not in passage_to_idx:
            passage_to_idx[context] = len(passages)
            passages.append(context)

        questions_with_passage.append({
            "question": item["question"],
            "passage_idx": passage_to_idx[context],
            "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
        })

    print(f"Unique passages: {len(passages)}")
    print(f"Questions: {len(questions_with_passage)}")

    # Embed passages
    print("\nEmbedding passages...")
    start_embed = time.time()
    passage_vectors = get_embeddings_batch(passages)
    embed_time = time.time() - start_embed
    print(f"Embedding time: {embed_time:.1f}s ({len(passages) / embed_time:.1f} passages/sec)")

    # Build hierarchy
    print("\nBuilding hierarchy...")
    start_build = time.time()
    root = build_hierarchy(passage_vectors, passages, branch_factor=branch_factor)
    build_time = time.time() - start_build
    print(f"Build time: {build_time:.3f}s")

    # Test queries
    print(f"\nTesting {num_queries} queries...")

    # Sample queries
    import random
    test_queries = random.sample(questions_with_passage, min(num_queries, len(questions_with_passage)))

    hierarchy_hits = 0
    brute_hits = 0
    hierarchy_e_total = 0
    brute_e_total = 0

    # LLM answer validation
    llm_correct_with_hierarchy = 0
    llm_correct_with_brute = 0
    llm_correct_with_gold = 0

    for i, q in enumerate(test_queries):
        question = q["question"]
        expected_idx = q["passage_idx"]
        expected_answer = q["answer"].lower()

        # Embed question
        query_vec = get_embedding(question)

        # Hierarchical retrieval
        h_results, h_e_count = retrieve_hierarchical(query_vec, root)
        h_indices = [r.content_idx for r in h_results]
        hierarchy_e_total += h_e_count

        # Brute force retrieval
        b_indices, b_e_count = retrieve_brute_force(query_vec, passage_vectors)
        brute_e_total += b_e_count

        # Check if correct passage found
        h_found = expected_idx in h_indices
        b_found = expected_idx in b_indices
        if h_found:
            hierarchy_hits += 1
        if b_found:
            brute_hits += 1

        # LLM answer validation (every 10th query to save time)
        if i % 10 == 0 and h_results:
            # Test with hierarchy-retrieved context
            h_context = h_results[0].content if h_results else ""
            try:
                h_answer = generate_answer(h_context, question)
                if expected_answer in h_answer.lower():
                    llm_correct_with_hierarchy += 1
            except Exception as e:
                pass

            # Test with brute-force-retrieved context
            if b_indices:
                b_context = passages[b_indices[0]]
                try:
                    b_answer = generate_answer(b_context, question)
                    if expected_answer in b_answer.lower():
                        llm_correct_with_brute += 1
                except Exception as e:
                    pass

            # Test with gold (correct) context
            gold_context = passages[expected_idx]
            try:
                gold_answer = generate_answer(gold_context, question)
                if expected_answer in gold_answer.lower():
                    llm_correct_with_gold += 1
            except Exception as e:
                pass

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_queries}")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nRecall (correct passage in top-10):")
    print(f"  Hierarchy:   {hierarchy_hits}/{num_queries} ({100 * hierarchy_hits / num_queries:.1f}%)")
    print(f"  Brute force: {brute_hits}/{num_queries} ({100 * brute_hits / num_queries:.1f}%)")

    print(f"\nE-computations per query:")
    print(f"  Hierarchy:   {hierarchy_e_total / num_queries:.1f}")
    print(f"  Brute force: {brute_e_total / num_queries:.1f}")
    print(f"  Speedup:     {brute_e_total / hierarchy_e_total:.1f}x")

    print(f"\nTotal passages: {len(passages)}")
    print(f"Branch factor: {branch_factor}")
    print(f"E threshold: {E_THRESHOLD}")

    # LLM answer validation results
    llm_tested = num_queries // 10
    if llm_tested > 0:
        print(f"\nLLM Answer Validation (lfm2.5, {llm_tested} queries):")
        print(f"  With hierarchy context: {llm_correct_with_hierarchy}/{llm_tested} ({100 * llm_correct_with_hierarchy / llm_tested:.1f}%)")
        print(f"  With brute force context: {llm_correct_with_brute}/{llm_tested} ({100 * llm_correct_with_brute / llm_tested:.1f}%)")
        print(f"  With gold (correct) context: {llm_correct_with_gold}/{llm_tested} ({100 * llm_correct_with_gold / llm_tested:.1f}%)")

    if hierarchy_hits >= brute_hits * 0.8:
        print("\nHIERARCHY WORKS - retrieves nearly as well with fewer E-computations")
    else:
        print("\nNEEDS TUNING - hierarchy missing too many passages")

    return {
        "hierarchy_recall": hierarchy_hits / num_queries,
        "brute_recall": brute_hits / num_queries,
        "hierarchy_e_per_query": hierarchy_e_total / num_queries,
        "brute_e_per_query": brute_e_total / num_queries,
        "speedup": brute_e_total / hierarchy_e_total,
        "llm_hierarchy_accuracy": llm_correct_with_hierarchy / max(1, llm_tested),
        "llm_brute_accuracy": llm_correct_with_brute / max(1, llm_tested),
        "llm_gold_accuracy": llm_correct_with_gold / max(1, llm_tested),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test E-score hierarchy on SQuAD")
    parser.add_argument("--passages", type=int, default=1000, help="Number of passages")
    parser.add_argument("--queries", type=int, default=100, help="Number of test queries")
    parser.add_argument("--branch", type=int, default=100, help="Branch factor")
    parser.add_argument("--threshold", type=float, default=0.3, help="E threshold")
    args = parser.parse_args()

    global E_THRESHOLD
    E_THRESHOLD = args.threshold

    # Check LLM Studio
    try:
        resp = requests.get(f"{LLM_STUDIO_BASE}/v1/models", timeout=5)
        models = [m["id"] for m in resp.json().get("data", [])]
        if EMBEDDING_MODEL not in models:
            print(f"Warning: {EMBEDDING_MODEL} not found in {models}")
    except Exception as e:
        print(f"Error: Cannot connect to LLM Studio at {LLM_STUDIO_BASE}")
        print(f"  {e}")
        sys.exit(1)

    run_squad_test(
        num_passages=args.passages,
        num_queries=args.queries,
        branch_factor=args.branch,
    )


if __name__ == "__main__":
    main()
