"""
Test Iso-Temporal Protocol on FERAL_RESIDENT data.

Tests the hypothesis: Does tracking "rotation signature" (processing context)
improve retrieval over pure E-score?

Data source: FERAL_RESIDENT/data/feral_eternal.db
- 849 paper sections from 23 arxiv papers
- Each paper's sections were processed sequentially
- "Context" = centroid of neighboring sections from same paper

Key insight: Papers have semantic structure. Sections near each other share context.
If you're asking about "weight sharing", and that section was processed right after
"quantization", then the context vector captures that relationship.
"""
import sys
import sqlite3
import json
import re
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Config
LLM_STUDIO_BASE = "http://10.5.0.2:1234"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
DB_PATH = Path(__file__).parent.parent.parent / "FERAL_RESIDENT" / "data" / "db" / "feral_eternal.db"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PaperSection:
    """A section from a paper with its vector and context."""
    section_id: str
    paper_id: str
    content: str
    vector: np.ndarray
    sequence_idx: int  # Position in paper (0, 1, 2, ...)
    context_vector: Optional[np.ndarray] = None  # Centroid of neighbors
    rotation_frame: Optional[np.ndarray] = None  # Full rotation matrix (orthonormal basis)


# =============================================================================
# Core Functions
# =============================================================================

def compute_E(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute E-score (Born rule): E = |<v1|v2>|^2"""
    dot = np.dot(v1, v2)
    return dot * dot


def get_embedding(text: str) -> np.ndarray:
    """Get embedding from LLM Studio."""
    resp = requests.post(
        f"{LLM_STUDIO_BASE}/v1/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text[:8000]},
        timeout=30,
    )
    resp.raise_for_status()
    embedding = np.array(resp.json()["data"][0]["embedding"])
    return embedding / np.linalg.norm(embedding)


def load_feral_data() -> Tuple[List[PaperSection], Dict[str, List[PaperSection]]]:
    """Load vectors and content from feral_eternal.db using temporal links."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Use temporal links if available, otherwise fall back to receipts
    cursor.execute("PRAGMA table_info(vectors)")
    columns = [col[1] for col in cursor.fetchall()]
    has_temporal = "sequence_id" in columns

    if has_temporal:
        # Use new temporal structure
        cursor.execute('''
            SELECT v.vector_id, v.vec_blob, v.sequence_id, v.sequence_idx,
                   v.context_vec_blob, r.metadata
            FROM vectors v
            LEFT JOIN receipts r ON SUBSTR(v.vec_sha256, 1, 16) = r.output_hash
            WHERE v.sequence_id IS NOT NULL AND v.sequence_id LIKE 'paper_%'
            ORDER BY v.sequence_id, v.sequence_idx
        ''')
    else:
        # Fall back to parsing receipts
        cursor.execute('''
            SELECT v.vector_id, v.vec_blob, r.metadata, r.created_at
            FROM vectors v
            JOIN receipts r ON SUBSTR(v.vec_sha256, 1, 16) = r.output_hash
            WHERE r.operation = 'paper_load'
            ORDER BY r.created_at
        ''')

    sections = []
    papers = {}

    for row in cursor.fetchall():
        if has_temporal:
            vector_id, vec_blob, sequence_id, seq_idx, context_blob, meta_json = row
            paper_id = sequence_id.replace("paper_", "") if sequence_id else "unknown"
            meta = json.loads(meta_json) if meta_json else {}
            content = meta.get("content", meta.get("text_preview", ""))
        else:
            vector_id, vec_blob, meta_json, _ = row
            meta = json.loads(meta_json) if meta_json else {}
            paper_id = meta.get("paper_id", "unknown")
            content = meta.get("content", meta.get("text_preview", ""))
            context_blob = None
            seq_idx = len(papers.get(paper_id, []))

        vec = np.frombuffer(vec_blob, dtype=np.float32)

        if paper_id not in papers:
            papers[paper_id] = []

        section = PaperSection(
            section_id=vector_id,
            paper_id=paper_id,
            content=content,
            vector=vec,
            sequence_idx=seq_idx,
        )

        # Use precomputed context if available
        if has_temporal and context_blob:
            section.context_vector = np.frombuffer(context_blob, dtype=np.float32)

        sections.append(section)
        papers[paper_id].append(section)

    conn.close()

    print(f"Loaded {len(sections)} sections from {len(papers)} papers")
    if has_temporal:
        print(f"  (Using precomputed temporal links)")
    return sections, papers


def build_context_vectors(papers: Dict[str, List[PaperSection]], k: int = 3, causal: bool = True):
    """
    Build context vectors for each section.

    Context = centroid of K neighboring sections in same paper.
    This is the "rotation signature" - the semantic neighborhood.

    If causal=True (default), only use PREVIOUS sections (realistic).
    If causal=False, use both before and after (unrealistic but for comparison).
    """
    for paper_id, sections in papers.items():
        for i, section in enumerate(sections):
            if causal:
                # Only previous sections (realistic - can't know future)
                start = max(0, i - k)
                neighbors = [sections[j].vector for j in range(start, i)]
            else:
                # Both sides (unrealistic - includes future)
                start = max(0, i - k)
                end = min(len(sections), i + k + 1)
                neighbors = [s.vector for j, s in enumerate(sections[start:end]) if j + start != i]

            if neighbors:
                context = np.mean(neighbors, axis=0)
                context = context / np.linalg.norm(context)
                section.context_vector = context
            else:
                section.context_vector = section.vector  # Fallback to self


def build_rotation_frames(papers: Dict[str, List[PaperSection]], k: int = 3):
    """
    Build full rotation frames for each section.

    This is the complete "Iso-Temporal Protocol":
    - Take K previous vectors
    - Gram-Schmidt orthonormalize to get local coordinate frame
    - This frame IS the "angle" you were at when learning this item

    The frame captures not just WHERE but the full ORIENTATION.
    """
    for paper_id, sections in papers.items():
        for i, section in enumerate(sections):
            # Get previous k vectors (the context leading up to this point)
            start = max(0, i - k)
            prev_vectors = [sections[j].vector for j in range(start, i)]

            if len(prev_vectors) >= 2:
                # Stack and orthonormalize via QR decomposition
                # This gives us the local coordinate frame
                V = np.stack(prev_vectors, axis=1)  # shape: (384, k)
                Q, R = np.linalg.qr(V)  # Q is orthonormal basis
                section.rotation_frame = Q  # shape: (384, min(k, 384))
            elif len(prev_vectors) == 1:
                # Single vector - use it as the "frame"
                v = prev_vectors[0].reshape(-1, 1)
                section.rotation_frame = v / np.linalg.norm(v)
            else:
                # No previous context - use identity projection
                section.rotation_frame = None


def compute_frame_similarity(query_vec: np.ndarray, frame: np.ndarray) -> float:
    """
    Compute how aligned the query is with the rotation frame.

    This is the "inverse rotation" check:
    If query projects strongly onto the frame, we were "facing the same direction."

    Returns: sum of squared projections onto frame basis vectors (like E-score but for a subspace)
    """
    if frame is None:
        return 0.0

    # Project query onto each basis vector in the frame
    # Total "alignment" = sum of |<query|basis_i>|^2
    projections = frame.T @ query_vec  # shape: (k,)
    alignment = np.sum(projections ** 2)

    return alignment


def generate_test_queries(
    sections: List[PaperSection],
    papers: Dict[str, List[PaperSection]],
    n_queries: int = 100,
) -> List[Dict]:
    """
    Generate test queries for RELATED section retrieval.

    Test: Given section N+1 as query, can we find section N?
    This tests whether context (shared paper neighborhood) helps.

    Pure E-score: relies only on semantic similarity
    Context-aware: boosts items that share processing context
    """
    queries = []

    # For each paper, create query pairs (section i+1 -> find section i)
    for paper_id, paper_sections in papers.items():
        if len(paper_sections) < 3:
            continue

        for i in range(1, len(paper_sections) - 1):
            query_section = paper_sections[i + 1]  # Use section i+1 as query
            target_section = paper_sections[i]     # Target is section i

            # Extract text for display
            content = query_section.content
            content = re.sub(r'@Paper-\d+\.\d+', '', content).strip()
            lines = content.split('\n')
            query_text = lines[0] if lines else content[:100]
            query_text = re.sub(r'[#*_`]', '', query_text).strip()

            queries.append({
                'query': query_text[:200],
                'query_vec': query_section.vector,
                'ground_truth_id': target_section.section_id,
                'paper_id': paper_id,
                'query_seq': query_section.sequence_idx,
                'target_seq': target_section.sequence_idx,
            })

    # Sample if too many
    if len(queries) > n_queries:
        queries = random.sample(queries, n_queries)

    return queries


# =============================================================================
# Retrieval Methods
# =============================================================================

def retrieve_pure_e(
    query_vec: np.ndarray,
    sections: List[PaperSection],
    max_results: int = 10,
) -> List[Tuple[PaperSection, float]]:
    """Pure E-score retrieval."""
    scores = []
    for section in sections:
        E = compute_E(query_vec, section.vector)
        scores.append((section, E))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:max_results]


def retrieve_context_aware(
    query_vec: np.ndarray,
    sections: List[PaperSection],
    lambda_: float = 0.3,
    max_results: int = 10,
) -> List[Tuple[PaperSection, float]]:
    """
    Context-aware retrieval.

    Score = E(query, item) + lambda * E(query, item_context)

    The idea: if you're in a similar "orientation" (context) to when
    the item was processed, boost its score.
    """
    scores = []
    for section in sections:
        E_item = compute_E(query_vec, section.vector)
        E_context = compute_E(query_vec, section.context_vector) if section.context_vector is not None else 0
        combined = E_item + lambda_ * E_context
        scores.append((section, combined, E_item, E_context))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [(s[0], s[1]) for s in scores[:max_results]]


def retrieve_context_only(
    query_vec: np.ndarray,
    sections: List[PaperSection],
    max_results: int = 10,
) -> List[Tuple[PaperSection, float]]:
    """Retrieve by context similarity only (ablation)."""
    scores = []
    for section in sections:
        if section.context_vector is not None:
            E = compute_E(query_vec, section.context_vector)
            scores.append((section, E))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:max_results]


def retrieve_rotation_frame(
    query_vec: np.ndarray,
    sections: List[PaperSection],
    lambda_: float = 0.5,
    max_results: int = 10,
) -> List[Tuple[PaperSection, float]]:
    """
    Full Iso-Temporal retrieval using rotation frames.

    Score = E(query, item) + lambda * frame_alignment(query, item_frame)

    The frame alignment measures: "Was this item learned while facing
    the same direction I'm currently facing?"
    """
    scores = []
    for section in sections:
        E_item = compute_E(query_vec, section.vector)
        frame_sim = compute_frame_similarity(query_vec, section.rotation_frame)
        combined = E_item + lambda_ * frame_sim
        scores.append((section, combined, E_item, frame_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [(s[0], s[1]) for s in scores[:max_results]]


def retrieve_frame_only(
    query_vec: np.ndarray,
    sections: List[PaperSection],
    max_results: int = 10,
) -> List[Tuple[PaperSection, float]]:
    """Retrieve by rotation frame alignment only."""
    scores = []
    for section in sections:
        if section.rotation_frame is not None:
            frame_sim = compute_frame_similarity(query_vec, section.rotation_frame)
            scores.append((section, frame_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:max_results]


# =============================================================================
# Test Runner
# =============================================================================

def run_test(
    sections: List[PaperSection],
    queries: List[Dict],
    lambdas: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    test_rotation: bool = False,
) -> Dict:
    """Run retrieval test comparing methods."""
    results = {l: {'hits': 0, 'mrr': 0.0} for l in lambdas}
    results['context_only'] = {'hits': 0, 'mrr': 0.0}

    if test_rotation:
        results['frame_0.5'] = {'hits': 0, 'mrr': 0.0}
        results['frame_only'] = {'hits': 0, 'mrr': 0.0}

    for i, q in enumerate(queries):
        target_id = q['ground_truth_id']
        query_vec = q['query_vec']  # Use stored vector

        # Test each lambda (context centroid method)
        for lambda_ in lambdas:
            results_list = retrieve_context_aware(query_vec, sections, lambda_=lambda_)
            retrieved_ids = [s.section_id for s, _ in results_list]

            if target_id in retrieved_ids:
                results[lambda_]['hits'] += 1
                rank = retrieved_ids.index(target_id) + 1
                results[lambda_]['mrr'] += 1.0 / rank

        # Context-only (ablation)
        ctx_results = retrieve_context_only(query_vec, sections)
        ctx_ids = [s.section_id for s, _ in ctx_results]
        if target_id in ctx_ids:
            results['context_only']['hits'] += 1
            rank = ctx_ids.index(target_id) + 1
            results['context_only']['mrr'] += 1.0 / rank

        # Rotation frame methods
        if test_rotation:
            # Frame-aware retrieval
            frame_results = retrieve_rotation_frame(query_vec, sections, lambda_=0.5)
            frame_ids = [s.section_id for s, _ in frame_results]
            if target_id in frame_ids:
                results['frame_0.5']['hits'] += 1
                rank = frame_ids.index(target_id) + 1
                results['frame_0.5']['mrr'] += 1.0 / rank

            # Frame-only retrieval
            frame_only_results = retrieve_frame_only(query_vec, sections)
            frame_only_ids = [s.section_id for s, _ in frame_only_results]
            if target_id in frame_only_ids:
                results['frame_only']['hits'] += 1
                rank = frame_only_ids.index(target_id) + 1
                results['frame_only']['mrr'] += 1.0 / rank

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(queries)}")

    # Normalize
    n = len(queries)
    for key in results:
        results[key]['recall'] = results[key]['hits'] / n
        results[key]['mrr'] = results[key]['mrr'] / n

    return results


# =============================================================================
# SYNTHETIC CONVERSATION TEST
# =============================================================================

@dataclass
class ConversationTurn:
    """A turn in a synthetic conversation."""
    turn_id: str
    conversation_id: str
    vector: np.ndarray
    sequence_idx: int
    source_section: PaperSection  # The paper section this was derived from
    context_vector: Optional[np.ndarray] = None
    rotation_frame: Optional[np.ndarray] = None


def create_synthetic_conversations(
    papers: Dict[str, List[PaperSection]],
    n_conversations: int = 50,
    turns_per_convo: int = 5,
    seed: int = 42,
) -> Tuple[List[ConversationTurn], Dict[str, List[ConversationTurn]]]:
    """
    Create synthetic conversations from paper sections.

    Strategy: Each conversation uses sections from the SAME paper,
    but we shuffle the pool so that multiple papers contribute sections
    that might be semantically similar across conversations.

    This tests: Can context disambiguate when multiple items have similar E-scores?
    If section A and section B both discuss "attention mechanisms" but A came after
    "transformer architecture" and B came after "RNN limitations", context should help.
    """
    rng = random.Random(seed)

    # Collect all papers with enough sections
    eligible_papers = [(pid, secs) for pid, secs in papers.items() if len(secs) >= turns_per_convo]

    if len(eligible_papers) < n_conversations:
        # Reuse papers if needed
        eligible_papers = eligible_papers * (n_conversations // len(eligible_papers) + 1)

    rng.shuffle(eligible_papers)

    all_turns = []
    conversations = {}

    for i in range(n_conversations):
        paper_id, sections = eligible_papers[i]
        conv_id = f"conv_{i:03d}"

        # Select turns_per_convo consecutive sections from this paper
        start_idx = rng.randint(0, len(sections) - turns_per_convo)
        conv_sections = sections[start_idx:start_idx + turns_per_convo]

        conv_turns = []
        for j, section in enumerate(conv_sections):
            turn = ConversationTurn(
                turn_id=f"{conv_id}_t{j}",
                conversation_id=conv_id,
                vector=section.vector.copy(),
                sequence_idx=j,
                source_section=section,
            )
            conv_turns.append(turn)
            all_turns.append(turn)

        conversations[conv_id] = conv_turns

    return all_turns, conversations


def create_causal_conversations(
    papers: Dict[str, List[PaperSection]],
    n_conversations: int = 50,
    turns_per_convo: int = 5,
    context_influence: float = 0.7,  # How much previous context affects each turn
    seed: int = 42,
) -> Tuple[List[ConversationTurn], Dict[str, List[ConversationTurn]]]:
    """
    Create TRULY CAUSAL synthetic conversations.

    Key difference: Each turn vector IS INFLUENCED BY previous turns.
    This simulates how an LLM generates responses based on context.

    Turn T = context_influence * centroid(T-1, T-2, ...) + (1-context_influence) * topic_seed + noise

    This creates the property we're testing: the "angle" you were facing
    when learning T is predictive of T's semantic content.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Get all unique paper vectors as "topic seeds"
    all_sections = []
    for sections in papers.values():
        all_sections.extend(sections)

    all_turns = []
    conversations = {}

    for i in range(n_conversations):
        conv_id = f"conv_{i:03d}"
        conv_turns = []

        # Pick a random topic seed for this conversation
        topic_seed = rng.choice(all_sections).vector.copy()

        for j in range(turns_per_convo):
            if j == 0:
                # First turn: just topic + noise
                vec = topic_seed.copy()
                noise = np_rng.randn(len(vec)) * 0.1
                vec = vec + noise
            else:
                # Subsequent turns: influenced by context
                # Context = mean of previous k turns (k = min(3, j))
                k = min(3, j)
                prev_vecs = [conv_turns[-idx-1].vector for idx in range(k)]
                context_vec = np.mean(prev_vecs, axis=0)

                # Turn = blend of context and topic
                vec = context_influence * context_vec + (1 - context_influence) * topic_seed

                # Add noise to make it non-trivial
                noise = np_rng.randn(len(vec)) * 0.15
                vec = vec + noise

            # Normalize
            vec = vec / np.linalg.norm(vec)

            turn = ConversationTurn(
                turn_id=f"{conv_id}_t{j}",
                conversation_id=conv_id,
                vector=vec,
                sequence_idx=j,
                source_section=None,
            )
            conv_turns.append(turn)
            all_turns.append(turn)

        conversations[conv_id] = conv_turns

    return all_turns, conversations


def build_conversation_contexts(
    conversations: Dict[str, List[ConversationTurn]],
    k: int = 3,
):
    """Build causal context vectors for conversation turns."""
    for conv_id, turns in conversations.items():
        for i, turn in enumerate(turns):
            # Only previous turns (causal)
            start = max(0, i - k)
            prev_vecs = [turns[j].vector for j in range(start, i)]

            if prev_vecs:
                context = np.mean(prev_vecs, axis=0)
                context = context / np.linalg.norm(context)
                turn.context_vector = context
            else:
                turn.context_vector = turn.vector


def build_conversation_frames(
    conversations: Dict[str, List[ConversationTurn]],
    k: int = 3,
):
    """Build rotation frames for conversation turns."""
    for conv_id, turns in conversations.items():
        for i, turn in enumerate(turns):
            start = max(0, i - k)
            prev_vecs = [turns[j].vector for j in range(start, i)]

            if len(prev_vecs) >= 2:
                V = np.stack(prev_vecs, axis=1)
                Q, R = np.linalg.qr(V)
                turn.rotation_frame = Q
            elif len(prev_vecs) == 1:
                v = prev_vecs[0].reshape(-1, 1)
                turn.rotation_frame = v / np.linalg.norm(v)
            else:
                turn.rotation_frame = None


def generate_conversation_queries(
    conversations: Dict[str, List[ConversationTurn]],
    n_queries: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate queries for conversation retrieval.

    Test: Given turn T, find turn T-1 from the same conversation.
    This is harder than paper sections because:
    - Multiple conversations may have similar topics
    - Pure E-score might confuse semantically similar turns across conversations
    - But context (the path you took to get to T) should disambiguate
    """
    rng = random.Random(seed)
    queries = []

    for conv_id, turns in conversations.items():
        for i in range(1, len(turns)):
            query_turn = turns[i]
            target_turn = turns[i - 1]

            queries.append({
                'query_vec': query_turn.vector,
                'query_context': query_turn.context_vector,
                'query_frame': query_turn.rotation_frame,
                'ground_truth_id': target_turn.turn_id,
                'conversation_id': conv_id,
                'query_seq': query_turn.sequence_idx,
                'target_seq': target_turn.sequence_idx,
            })

    if len(queries) > n_queries:
        queries = rng.sample(queries, n_queries)

    return queries


def run_conversation_test(
    all_turns: List[ConversationTurn],
    queries: List[Dict],
    test_rotation: bool = True,
) -> Dict:
    """Run retrieval test on synthetic conversations."""

    # Methods to test
    methods = {
        'pure_E': {'hits': 0, 'mrr': 0.0},
        'context_E0.3': {'hits': 0, 'mrr': 0.0},
        'context_E0.5': {'hits': 0, 'mrr': 0.0},
        'context_only': {'hits': 0, 'mrr': 0.0},
    }

    if test_rotation:
        methods['frame_E0.5'] = {'hits': 0, 'mrr': 0.0}
        methods['frame_only'] = {'hits': 0, 'mrr': 0.0}

    for i, q in enumerate(queries):
        query_vec = q['query_vec']
        target_id = q['ground_truth_id']

        # Score all turns by different methods
        scores = {}
        for turn in all_turns:
            E_item = compute_E(query_vec, turn.vector)
            E_context = compute_E(query_vec, turn.context_vector) if turn.context_vector is not None else 0
            frame_sim = compute_frame_similarity(query_vec, turn.rotation_frame) if test_rotation else 0

            scores[turn.turn_id] = {
                'pure_E': E_item,
                'context_E0.3': E_item + 0.3 * E_context,
                'context_E0.5': E_item + 0.5 * E_context,
                'context_only': E_context,
                'frame_E0.5': E_item + 0.5 * frame_sim if test_rotation else 0,
                'frame_only': frame_sim if test_rotation else 0,
            }

        # Rank by each method
        for method in methods:
            ranked = sorted(scores.items(), key=lambda x: x[1][method], reverse=True)
            top_10_ids = [turn_id for turn_id, _ in ranked[:10]]

            if target_id in top_10_ids:
                methods[method]['hits'] += 1
                rank = top_10_ids.index(target_id) + 1
                methods[method]['mrr'] += 1.0 / rank

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(queries)}")

    # Normalize
    n = len(queries)
    for method in methods:
        methods[method]['recall'] = methods[method]['hits'] / n
        methods[method]['mrr'] = methods[method]['mrr'] / n

    return methods


def run_synthetic_conversation_test(
    papers: Dict[str, List[PaperSection]],
    n_conversations: int = 50,
    turns_per_convo: int = 5,
    n_queries: int = 100,
    context_k: int = 3,
    seed: int = 42,
    causal: bool = False,
    context_influence: float = 0.7,
):
    """Main synthetic conversation test."""
    print("\n" + "=" * 70)
    print("SYNTHETIC CONVERSATION TEST" + (" (CAUSAL)" if causal else " (NON-CAUSAL)"))
    print("=" * 70)
    print(f"Conversations: {n_conversations}")
    print(f"Turns per conversation: {turns_per_convo}")
    print(f"Context window: {context_k}")
    if causal:
        print(f"Context influence: {context_influence:.0%}")
    print("=" * 70)

    # Create conversations
    if causal:
        print("\nCreating CAUSAL synthetic conversations...")
        print("  (Each turn vector is influenced by previous context)")
        all_turns, conversations = create_causal_conversations(
            papers, n_conversations, turns_per_convo, context_influence, seed
        )
    else:
        print("\nCreating synthetic conversations from paper sections...")
        all_turns, conversations = create_synthetic_conversations(
            papers, n_conversations, turns_per_convo, seed
        )
    print(f"  Created {len(conversations)} conversations with {len(all_turns)} total turns")

    # Build contexts
    print(f"\nBuilding causal context vectors (k={context_k})...")
    build_conversation_contexts(conversations, k=context_k)

    print(f"Building rotation frames (k={context_k})...")
    build_conversation_frames(conversations, k=context_k)

    # Generate queries
    print(f"\nGenerating {n_queries} test queries...")
    queries = generate_conversation_queries(conversations, n_queries, seed)
    print(f"  Generated {len(queries)} queries")

    # Run test
    print("\nRunning retrieval test...")
    results = run_conversation_test(all_turns, queries)

    # Report
    print("\n" + "=" * 70)
    print("SYNTHETIC CONVERSATION RESULTS")
    print("=" * 70)

    print("\nRecall@10 by method:")
    print("-" * 50)
    for method, r in sorted(results.items(), key=lambda x: x[1]['recall'], reverse=True):
        print(f"  {method:15s}: Recall={r['recall']:.1%}  MRR={r['mrr']:.3f}")

    # Analysis
    print("\n" + "-" * 50)
    pure_recall = results['pure_E']['recall']
    best_ctx = max(['context_E0.3', 'context_E0.5'], key=lambda m: results[m]['recall'])
    best_recall = results[best_ctx]['recall']

    if best_recall > pure_recall:
        improvement = (best_recall - pure_recall) / max(pure_recall, 0.001) * 100
        print(f"CONTEXT HELPS: +{improvement:.1f}% recall ({best_ctx})")
    else:
        print(f"CONTEXT DOES NOT HELP: Pure E-score is sufficient")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Iso-Temporal Protocol on FERAL data")
    parser.add_argument("--queries", type=int, default=100, help="Number of test queries")
    parser.add_argument("--context-k", type=int, default=3, help="Context window size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rotation", action="store_true", help="Test full rotation frame method")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic conversation test")
    parser.add_argument("--causal", action="store_true", help="Use causal turn generation (turns depend on context)")
    parser.add_argument("--context-influence", type=float, default=0.7, help="How much context influences each turn (0-1)")
    parser.add_argument("--n-convos", type=int, default=50, help="Number of synthetic conversations")
    parser.add_argument("--turns", type=int, default=5, help="Turns per conversation")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data first (needed for both tests)
    print("Loading FERAL_RESIDENT data...")
    sections, papers = load_feral_data()

    # Run synthetic conversation test if requested
    if args.synthetic:
        run_synthetic_conversation_test(
            papers,
            n_conversations=args.n_convos,
            turns_per_convo=args.turns,
            n_queries=args.queries,
            context_k=args.context_k,
            seed=args.seed,
            causal=args.causal,
            context_influence=args.context_influence,
        )
        return

    print("=" * 70)
    print("ISO-TEMPORAL PROTOCOL TEST - FERAL_RESIDENT DATA")
    print("=" * 70)
    print(f"Database: {DB_PATH}")
    print(f"Context window: {args.context_k} neighbors")
    print(f"Test queries: {args.queries}")
    print(f"Rotation frames: {args.rotation}")
    print("=" * 70)

    # Build context vectors (centroid method)
    print(f"\nBuilding context vectors (k={args.context_k})...")
    build_context_vectors(papers, k=args.context_k)

    # Build rotation frames (full rotation method)
    if args.rotation:
        print(f"\nBuilding rotation frames (k={args.context_k})...")
        build_rotation_frames(papers, k=args.context_k)

    # Generate test queries
    print(f"\nGenerating {args.queries} test queries...")
    print("  Test: Given section N+1, can we find section N? (adjacent section retrieval)")
    queries = generate_test_queries(sections, papers, n_queries=args.queries)
    print(f"  Generated {len(queries)} valid queries")

    # Run test
    print("\nRunning retrieval test...")
    results = run_test(sections, queries, test_rotation=args.rotation)

    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nRecall@10 by lambda (E_item + lambda * E_context):")
    print("-" * 50)
    for lambda_ in sorted([k for k in results.keys() if isinstance(k, float)]):
        r = results[lambda_]
        print(f"  lambda={lambda_:.1f}: Recall={r['recall']:.1%}  MRR={r['mrr']:.3f}")

    print(f"\n  context_only: Recall={results['context_only']['recall']:.1%}  MRR={results['context_only']['mrr']:.3f}")

    if args.rotation:
        print("\nRotation Frame Methods:")
        print("-" * 50)
        print(f"  frame+E (lambda=0.5): Recall={results['frame_0.5']['recall']:.1%}  MRR={results['frame_0.5']['mrr']:.3f}")
        print(f"  frame_only:           Recall={results['frame_only']['recall']:.1%}  MRR={results['frame_only']['mrr']:.3f}")

    # Analysis
    print("\n" + "-" * 50)
    pure_recall = results[0.0]['recall']
    best_lambda = max([k for k in results.keys() if isinstance(k, float)],
                      key=lambda l: results[l]['recall'])
    best_recall = results[best_lambda]['recall']

    if best_recall > pure_recall:
        improvement = (best_recall - pure_recall) / pure_recall * 100
        print(f"CONTEXT HELPS: +{improvement:.1f}% recall at lambda={best_lambda}")
        print("Iso-Temporal Protocol VALIDATED")
    else:
        print("CONTEXT DOES NOT HELP: Pure E-score is sufficient")
        print("Iso-Temporal Protocol NOT VALIDATED (for this dataset)")

    if args.rotation:
        ctx_only = results['context_only']['recall']
        frame_only = results['frame_only']['recall']
        if frame_only > ctx_only:
            print(f"\nROTATION FRAME BEATS CENTROID: {frame_only:.1%} vs {ctx_only:.1%}")
        elif frame_only == ctx_only:
            print(f"\nROTATION FRAME EQUALS CENTROID: {frame_only:.1%}")
        else:
            print(f"\nCENTROID BEATS ROTATION FRAME: {ctx_only:.1%} vs {frame_only:.1%}")

    return results


if __name__ == "__main__":
    main()
