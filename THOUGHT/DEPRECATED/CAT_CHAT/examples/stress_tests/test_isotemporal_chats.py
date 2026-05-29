"""
Test Iso-Temporal Protocol on REAL chat conversations.

Uses actual LFM chat sessions from test_chats/ to validate that temporal
context (rotation signature) improves retrieval of relevant conversation history.

Test scenario:
- Load real chat sessions from session capsule DBs
- For each turn T, try to retrieve previous relevant turns
- Compare: Pure E-score vs Context-aware E-score vs Rotation frame
- Validate that "where you were when you learned it" helps retrieval
"""

import sqlite3
import json
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sys
import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Config
LLM_STUDIO_BASE = "http://10.5.0.2:1234"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHAT_DB_DIR = Path(__file__).parent / "test_chats"

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ChatTurn:
    """A single turn in a chat conversation."""
    turn_id: str
    session_id: str
    sequence_num: int
    role: str  # 'user' or 'assistant'
    content: str
    vector: np.ndarray
    timestamp: str
    # Temporal fields
    context_vector: Optional[np.ndarray] = None
    rotation_frame: Optional[np.ndarray] = None
    prev_turn_id: Optional[str] = None
    next_turn_id: Optional[str] = None


# =============================================================================
# Data Loading
# =============================================================================

def get_embedding(text: str) -> np.ndarray:
    """Get embedding from LLM Studio."""
    resp = requests.post(
        f"{LLM_STUDIO_BASE}/v1/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    embedding = np.array(resp.json()["data"][0]["embedding"])
    return embedding / np.linalg.norm(embedding)


def load_chat_sessions(db_path: Path) -> Tuple[List[ChatTurn], Dict[str, List[ChatTurn]]]:
    """
    Load chat turns from a session capsule database.

    Returns:
        - all_turns: Flat list of all turns across all sessions
        - sessions: Dict mapping session_id to list of turns
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get all stored turns (from Auto-Controlled Context Loop)
    cursor.execute('''
        SELECT session_id, event_id, event_type, sequence_num, timestamp, payload_json
        FROM session_events
        WHERE event_type = 'turn_stored'
        ORDER BY session_id, sequence_num
    ''')

    all_turns = []
    sessions = {}

    print(f"Loading turns from {db_path.name}...")

    for row in cursor.fetchall():
        session_id, event_id, event_type, seq_num, timestamp, payload_json = row
        payload = json.loads(payload_json)

        # Extract turn info from payload
        turn_id = payload.get('turn_id', event_id)
        summary = payload.get('summary', '')

        if not summary or len(summary.strip()) == 0:
            continue

        # Parse summary to extract role (usually starts with "User:" or "Assistant:")
        role = 'unknown'
        content = summary
        if summary.startswith('User:'):
            role = 'user'
            content = summary[5:].strip()
        elif summary.startswith('Assistant:'):
            role = 'assistant'
            content = summary[10:].strip()

        # Get embedding
        try:
            vec = get_embedding(content)
        except Exception as e:
            print(f"  Warning: Failed to embed turn {turn_id}: {e}")
            continue

        turn = ChatTurn(
            turn_id=turn_id,
            session_id=session_id,
            sequence_num=seq_num,
            role=role,
            content=content,
            vector=vec,
            timestamp=timestamp,
        )

        all_turns.append(turn)

        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(turn)

    conn.close()

    # Link prev/next turns within each session
    for session_id, turns in sessions.items():
        turns.sort(key=lambda t: t.sequence_num)
        for i, turn in enumerate(turns):
            if i > 0:
                turn.prev_turn_id = turns[i-1].turn_id
            if i < len(turns) - 1:
                turn.next_turn_id = turns[i+1].turn_id

    print(f"  Loaded {len(all_turns)} turns from {len(sessions)} sessions")

    return all_turns, sessions


def load_all_chats() -> Tuple[List[ChatTurn], Dict[str, List[ChatTurn]]]:
    """Load all chat sessions from test_chats directory."""
    all_turns = []
    all_sessions = {}

    for db_file in CHAT_DB_DIR.glob("*.db"):
        turns, sessions = load_chat_sessions(db_file)
        all_turns.extend(turns)
        all_sessions.update(sessions)

    print(f"\nTotal: {len(all_turns)} turns from {len(all_sessions)} sessions")
    return all_turns, all_sessions


# =============================================================================
# Temporal Context Building
# =============================================================================

def build_context_vectors(sessions: Dict[str, List[ChatTurn]], k: int = 3):
    """Build causal context vectors for each turn (centroid of previous k)."""
    for session_id, turns in sessions.items():
        for i, turn in enumerate(turns):
            # Only previous turns (causal)
            start = max(0, i - k)
            prev_vecs = [turns[j].vector for j in range(start, i)]

            if prev_vecs:
                context = np.mean(prev_vecs, axis=0)
                context = context / np.linalg.norm(context)
                turn.context_vector = context
            else:
                # First turn: use itself as context
                turn.context_vector = turn.vector


def build_rotation_frames(sessions: Dict[str, List[ChatTurn]], k: int = 3):
    """Build rotation frames for each turn (QR decomposition of previous k)."""
    for session_id, turns in sessions.items():
        for i, turn in enumerate(turns):
            start = max(0, i - k)
            prev_vecs = [turns[j].vector for j in range(start, i)]

            if len(prev_vecs) >= 2:
                # Full QR decomposition
                V = np.stack(prev_vecs, axis=1)
                Q, R = np.linalg.qr(V)
                turn.rotation_frame = Q
            elif len(prev_vecs) == 1:
                # Single vector: normalize
                v = prev_vecs[0].reshape(-1, 1)
                turn.rotation_frame = v / np.linalg.norm(v)
            else:
                # No previous context
                turn.rotation_frame = None


# =============================================================================
# Retrieval Test
# =============================================================================

def compute_E(v1: np.ndarray, v2: np.ndarray) -> float:
    """Born rule: E = |<v1|v2>|^2"""
    dot = np.dot(v1, v2)
    return dot * dot


def compute_frame_similarity(query_vec: np.ndarray, frame: Optional[np.ndarray]) -> float:
    """Project query onto rotation frame and compute overlap."""
    if frame is None:
        return 0.0

    # Project query onto frame subspace
    proj = frame @ (frame.T @ query_vec)
    # E-score between query and its projection
    return compute_E(query_vec, proj)


def generate_retrieval_queries(
    sessions: Dict[str, List[ChatTurn]],
    n_queries: int = 100,
    min_turn_idx: int = 3,  # Only query turns with at least 3 previous turns
    seed: int = 42,
) -> List[Dict]:
    """
    Generate retrieval queries from chat turns.

    Test: Given turn T, find the PREVIOUS turn T-1 from same session.
    This tests if temporal context helps disambiguate when multiple
    sessions have similar topics.
    """
    rng = random.Random(seed)
    queries = []

    for session_id, turns in sessions.items():
        if len(turns) < min_turn_idx + 1:
            continue

        # Only query turns that have sufficient context
        for i in range(min_turn_idx, len(turns)):
            query_turn = turns[i]
            target_turn = turns[i - 1]  # Previous turn

            queries.append({
                'query_vec': query_turn.vector,
                'query_context': query_turn.context_vector,
                'query_frame': query_turn.rotation_frame,
                'ground_truth_id': target_turn.turn_id,
                'session_id': session_id,
                'query_seq': query_turn.sequence_num,
                'target_seq': target_turn.sequence_num,
                'query_content': query_turn.content[:100],  # For debugging
                'target_content': target_turn.content[:100],
            })

    if len(queries) > n_queries:
        queries = rng.sample(queries, n_queries)

    return queries


def run_retrieval_test(
    all_turns: List[ChatTurn],
    queries: List[Dict],
    test_rotation: bool = True,
) -> Dict:
    """
    Run retrieval test.

    For each query, rank ALL turns by different scoring methods
    and check if ground truth appears in top-10.
    """
    methods = {
        'pure_E': {'hits': 0, 'mrr': 0.0},
        'context_E0.2': {'hits': 0, 'mrr': 0.0},
        'context_E0.5': {'hits': 0, 'mrr': 0.0},
        'context_only': {'hits': 0, 'mrr': 0.0},
    }

    if test_rotation:
        methods['frame_E0.5'] = {'hits': 0, 'mrr': 0.0}
        methods['frame_only'] = {'hits': 0, 'mrr': 0.0}

    for i, q in enumerate(queries):
        query_vec = q['query_vec']
        target_id = q['ground_truth_id']

        # Score all turns
        scores = {}
        for turn in all_turns:
            E_item = compute_E(query_vec, turn.vector)
            E_context = compute_E(query_vec, turn.context_vector) if turn.context_vector is not None else 0
            frame_sim = compute_frame_similarity(query_vec, turn.rotation_frame) if test_rotation else 0

            scores[turn.turn_id] = {
                'pure_E': E_item,
                'context_E0.2': E_item + 0.2 * E_context,
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


# =============================================================================
# Main Test
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Iso-Temporal Protocol on real chat sessions")
    parser.add_argument("--db", type=str, help="Specific DB file to test (default: all in test_chats/)")
    parser.add_argument("--queries", type=int, default=100, help="Number of test queries")
    parser.add_argument("--context-k", type=int, default=3, help="Context window size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rotation", action="store_true", help="Test rotation frame method")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("ISO-TEMPORAL PROTOCOL TEST - REAL CHAT SESSIONS")
    print("=" * 70)
    print(f"Context window: {args.context_k} turns")
    print(f"Test queries: {args.queries}")
    print(f"Rotation frames: {args.rotation}")
    print("=" * 70)

    # Load chat sessions
    if args.db:
        db_path = Path(args.db)
        all_turns, sessions = load_chat_sessions(db_path)
    else:
        all_turns, sessions = load_all_chats()

    if len(all_turns) == 0:
        print("ERROR: No turns loaded!")
        return

    # Build temporal context
    print(f"\nBuilding context vectors (k={args.context_k})...")
    build_context_vectors(sessions, k=args.context_k)

    if args.rotation:
        print(f"Building rotation frames (k={args.context_k})...")
        build_rotation_frames(sessions, k=args.context_k)

    # Generate queries
    print(f"\nGenerating {args.queries} retrieval queries...")
    queries = generate_retrieval_queries(sessions, n_queries=args.queries, seed=args.seed)
    print(f"  Generated {len(queries)} queries")

    if len(queries) == 0:
        print("ERROR: No queries generated!")
        return

    # Run test
    print("\nRunning retrieval test...")
    results = run_retrieval_test(all_turns, queries, test_rotation=args.rotation)

    # Report
    print("\n" + "=" * 70)
    print("RESULTS: REAL CHAT SESSION RETRIEVAL")
    print("=" * 70)

    print("\nRecall@10 by method:")
    print("-" * 50)
    for method, r in sorted(results.items(), key=lambda x: x[1]['recall'], reverse=True):
        print(f"  {method:15s}: Recall={r['recall']:.1%}  MRR={r['mrr']:.3f}")

    # Analysis
    print("\n" + "-" * 50)
    pure_recall = results['pure_E']['recall']
    best_ctx = max(['context_E0.2', 'context_E0.5'], key=lambda m: results[m]['recall'])
    best_recall = results[best_ctx]['recall']

    if best_recall > pure_recall:
        improvement = (best_recall - pure_recall) / max(pure_recall, 0.001) * 100
        print(f"TEMPORAL CONTEXT HELPS: +{improvement:.1f}% recall ({best_ctx})")
    else:
        print(f"TEMPORAL CONTEXT DOES NOT HELP: Pure E-score sufficient")

    if args.rotation:
        frame_recall = results.get('frame_E0.5', {}).get('recall', 0)
        if frame_recall > best_recall:
            improvement = (frame_recall - best_recall) / max(best_recall, 0.001) * 100
            print(f"ROTATION FRAME WINS: +{improvement:.1f}% over best context")
        elif frame_recall > pure_recall:
            improvement = (frame_recall - pure_recall) / max(pure_recall, 0.001) * 100
            print(f"ROTATION FRAME HELPS: +{improvement:.1f}% over pure E-score")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
